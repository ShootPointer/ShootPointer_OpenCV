import logging
import time
import shutil
import json
import asyncio
import hashlib 
from pathlib import Path
from typing import Optional

from redis.asyncio import Redis, ConnectionError as RedisConnectionError

from app.core.config import settings
from app.core.redis_client import get_redis_client
from app.schemas.redis import AITaskPayload, UploadStatus

logger = logging.getLogger(__name__)

# Redis 상태 보고 키
def get_status_key(job_id: str) -> str:
    """Redis에서 작업 상태를 저장하는 키를 반환합니다."""
    return f"job:{job_id}:status"

# ─────────────────────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────────────────────

def calculate_file_checksum(file_path: Path) -> str:
    """주어진 파일 경로의 SHA256 체크섬을 계산합니다."""
    hasher = hashlib.sha256()
    try:
        # 파일을 바이너리 모드로 읽고, 청크 단위로 해시 업데이트
        with file_path.open('rb') as f:
            while chunk := f.read(8192): # 8KB 청크
                hasher.update(chunk)
        # 'sha256:' 접두사 포함하여 반환
        return f"sha256:{hasher.hexdigest()}"
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {e}")
        return "sha256:error" # 오류 발생 시 오류 값 반환

# ─────────────────────────────────────────────────────────────
# 진행률 보고 로직
# ─────────────────────────────────────────────────────────────

async def report_progress_to_spring(job_id: str, status: str, progress: float):
    """
    Redis를 통해 Spring 서버에 작업 진행 상태를 업데이트합니다. (0%~99% 중간 보고용)
    """
    try:
        redis: Redis = get_redis_client()
    except ConnectionError as e:
        logger.error(f"Redis not available, cannot report status for Job {job_id}: {e}")
        return

    try:
        # 상태 JSON (Spring 서버와의 약속된 포맷)
        status_data = {
            "jobId": job_id,
            "status": status,
            "progress": f"{progress:.2f}",
            "timestamp": int(time.time())
        }
        
        await redis.set(get_status_key(job_id), json.dumps(status_data), ex=3600) 
        logger.info(f"Job {job_id} status updated: {status} ({progress:.2f}%)")
    except RedisConnectionError as e:
        logger.error(f"Redis connection dropped or operation failed for Job {job_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to report status to Redis for Job {job_id}: {e}")

async def report_final_completion_to_spring(
    job_id: str, 
    final_file_path: Path, 
    checksum: str
):
    """
    최종 완료된 원본 영상 정보를 Spring 서버에 약속된 JSON 형식으로 Redis를 통해 보고합니다.
    """
    try:
        redis: Redis = get_redis_client()
    except ConnectionError as e:
        logger.error(f"Redis not available, cannot report final status for Job {job_id}: {e}")
        return
    
    # TODO: durationSec, memberId는 실제 토큰 데이터나 미디어 분석 라이브러리(ffprobe)에서 가져와야 합니다.
    try:
        file_size_bytes = final_file_path.stat().st_size
        source_url = f"{settings.EXTERNAL_BASE_URL}/{final_file_path.name}"
        member_id = settings.MEMBER_ID 
        duration_sec = 0.0 # 임시 값

    except Exception as e:
        logger.error(f"Failed to get file stats for final notification: {e}")
        file_size_bytes = 0
        source_url = "error_url"
        
    # 최종 완료 JSON 페이로드 구성
    payload = {
        "status": 200,
        "success": True,
        "data": {
            "type": "UPLOAD_COMPLETE",
            "memberId": member_id, 
            "jobId": job_id,
            "sizeBytes": file_size_bytes,
            "sourceUrl": source_url,
            "checksum": checksum,
            "durationSec": duration_sec
        },
        "message": "Original video successfully merged and stored.",
        "timestamp": int(time.time() * 1000)
    }

    try:
        await redis.set(get_status_key(job_id), json.dumps(payload), ex=3600) 
        logger.info(f"Final completion JSON reported to Redis for Job {job_id}.")
    except Exception as e:
        logger.error(f"Failed to report final completion JSON to Redis for Job {job_id}: {e}")

# ─────────────────────────────────────────────────────────────
# AI Worker 연동 로직 (Redis Queue)
# ─────────────────────────────────────────────────────────────

async def _trigger_ai_worker(job_id: str, final_file_path: Path):
    """
    AI Worker가 처리할 작업 요청을 Redis List(Queue)에 푸시하고, 
    Spring 서버에 AI 작업 시작 대기 상태를 보고합니다.
    """
    try:
        redis: Redis = get_redis_client()
    except ConnectionError:
        logger.error(f"Redis not available, cannot queue AI task for Job {job_id}.")
        return

    # 1. AI Worker에게 전달할 페이로드 구성
    ai_payload = AITaskPayload(
        jobId=job_id,
        memberId=settings.MEMBER_ID, # TODO: 실제 Member ID 사용
        originalFilePath=str(final_file_path.resolve()) # 절대 경로를 문자열로 전달
    )

    try:
        # 2. Redis List에 PUSH (큐에 넣기)
        push_count = await redis.rpush(
            settings.AI_QUEUE_NAME, 
            ai_payload.model_dump_json() # Pydantic 모델을 JSON 문자열로 직렬화
        )
        logger.info(f"AI Task for Job {job_id} successfully pushed to queue '{settings.AI_QUEUE_NAME}'. Queue size: {push_count}")
        
        # 3. Spring 서버에 AI 작업 시작 대기 상태 보고
        await report_progress_to_spring(job_id, UploadStatus.AI_START_PENDING.value, 100.0)

    except Exception as e:
        logger.error(f"Failed to queue AI task for Job {job_id}: {e}")
        # AI 큐에 넣는 것까지 실패했다면, Spring에게도 실패를 알려야 합니다.
        await report_progress_to_spring(job_id, UploadStatus.ERROR.value, 0.0)

# ─────────────────────────────────────────────────────────────
# 핵심 비즈니스 로직: 병합 및 정리 (백그라운드 태스크)
# ─────────────────────────────────────────────────────────────

async def merge_chunks_and_cleanup(
    job_id: str, 
    file_name: str, 
    total_parts: int, 
    chunk_dir: Path
):
    """
    백그라운드에서 실행되는 메인 병합 작업 로직입니다. 
    """
    final_path: Optional[Path] = None
    calculated_checksum: Optional[str] = None
    
    # 1. 상태 보고: 작업 시작 (0%로 초기화)
    await report_progress_to_spring(job_id, "IN_PROGRESS", 0.0)
    
    try:
        # 2. 청크 파일 목록 정렬 및 최종 검증
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        if len(chunk_files) != total_parts:
             raise Exception("Integrity check failure during merge.")
        
        # 3. 병합 작업 실행
        final_path = Path(settings.TEMP_ROOT) / f"{job_id}_{file_name}"
        logger.info(f"Starting merge of {total_parts} chunks into {final_path}")
        
        # 실제 병합 및 진행률 보고 로직 (90%까지)
        with final_path.open("wb") as outfile:
            for i, chunk_file in enumerate(chunk_files):
                with chunk_file.open("rb") as infile:
                    # 실제 파일 복사 (병합)
                    shutil.copyfileobj(infile, outfile) 
                
                # 진행률 보고 (병합 진행도)
                merge_progress = (i + 1) / total_parts * 90.0
                await report_progress_to_spring(job_id, "IN_PROGRESS", merge_progress)
                
        logger.info(f"Merge completed. Starting checksum calculation.")

        # 4. 체크섬 계산 및 95% 보고
        calculated_checksum = calculate_file_checksum(final_path)
        await report_progress_to_spring(job_id, "IN_PROGRESS", 95.0)
        
        # 5. Spring 서버에 최종 완료 JSON 보고 (Upload 완료)
        await report_final_completion_to_spring(
            job_id, 
            final_path, 
            calculated_checksum
        )

        # 6. AI Worker Queue에 작업 푸시 및 상태 보고
        await _trigger_ai_worker(job_id, final_path)
        
    except Exception as e:
        logger.error(f"Critical error during merge/cleanup/trigger for Job {job_id}: {e}", exc_info=True)
        # 7. 상태 보고: 작업 실패
        await report_progress_to_spring(job_id, "FAILED", 0.0)
        
    finally:
        # 8. 임시 청크 폴더 삭제
        if chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)
            logger.info(f"Cleanup complete for Job {job_id}: removed {chunk_dir}")