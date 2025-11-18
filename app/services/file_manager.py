# app/services/file_manager.py
import os
import logging
import time
import shutil
import json
import asyncio
import hashlib
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, Any

from redis.asyncio import Redis, ConnectionError as RedisConnectionError

from app.core.config import settings
from app.core.redis_client import get_redis_client
from app.schemas.redis import AITaskPayload, UploadStatus

logger = logging.getLogger(__name__)

# Redis 상태 보고 키
def get_status_key(job_id: str) -> str:
    """Redis에서 작업 상태를 저장하는 키를 반환합니다."""
    return f"job:{job_id}:status"

def get_meta_key(job_id: str) -> str:
    """작업 메타데이터(멤버, 하이라이트키, 원본경로)를 저장하는 키."""
    return f"job:{job_id}:meta"

# ─────────────────────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────────────────────

def calculate_file_checksum(file_path: Path) -> str:
    """주어진 파일 경로의 SHA256 체크섬을 계산합니다."""
    hasher = hashlib.sha256()
    try:
        with file_path.open('rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB 청크
                if not chunk:
                    break
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"
    except Exception as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {e}")
        return "sha256:error"

# ─────────────────────────────────────────────────────────────
# 진행률/완료 보고 로직
# ─────────────────────────────────────────────────────────────

async def report_progress_to_spring(job_id: str, status: str, progress: float):
    """
    Redis를 통해 Spring 서버에 작업 진행 상태를 업데이트합니다. (0%~99% 중간 보고용)
    - key: job:{jobId}:status (스냅샷)
    - channel: {REDIS_UPLOAD_PROGRESS_CHANNEL}:{jobId} (Pub/Sub)
    """
    try:
        redis: Redis = get_redis_client()
    except RedisConnectionError as e:
        logger.error(f"Redis not available, cannot report status for Job {job_id}: {e}")
        return
    except Exception as e:
        logger.error(f"Unknown error getting Redis client for Job {job_id}: {e}")
        return

    # 공통 페이로드 (SET / PUBLISH 둘 다 이걸 사용)
    status_data = {
        "jobId": job_id,
        "status": status,
        "progress": f"{progress:.2f}",
        "timestamp": int(time.time()),
    }

    # 1) 스냅샷 저장 (기존 동작 유지)
    try:
        await redis.set(get_status_key(job_id), json.dumps(status_data), ex=3600)
        logger.info(f"Job {job_id} status updated: {status} ({progress:.2f}%)")
    except RedisConnectionError as e:
        logger.error(f"Redis connection dropped or operation failed for Job {job_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to report status to Redis for Job {job_id}: {e}")

    # 2) Pub/Sub 채널로도 발행 (백엔드가 SUBSCRIBE 하는 용도)
    try:
        channel = f"{settings.REDIS_UPLOAD_PROGRESS_CHANNEL}:{job_id}"
        await redis.publish(channel, json.dumps(status_data))
        logger.info(
            f"Job {job_id} progress PUBLISHED to {channel}: "
            f"{status} ({progress:.2f}%)"
        )
    except Exception as e:
        logger.error(f"Failed to publish progress for Job {job_id} to channel: {e}")

async def report_final_completion_to_spring(
    job_id: str,
    final_file_path: Path,
    checksum: str,
    member_id_override: Optional[str] = None,
):
    """
    최종 완료된 원본 영상 정보를 Spring 서버에 약속된 JSON 형식으로 Redis를 통해 보고합니다.
    - key: job:{jobId}:status (스냅샷)
    - channel: {REDIS_UPLOAD_PROGRESS_CHANNEL}:{jobId} (Pub/Sub)
    """
    try:
        redis: Redis = get_redis_client()
    except RedisConnectionError as e:
        logger.error(f"Redis not available, cannot report final status for Job {job_id}: {e}")
        return
    except Exception as e:
        logger.error(f"Unknown error getting Redis client for Job {job_id}: {e}")
        return

    try:
        file_size_bytes = final_file_path.stat().st_size

        # SAVE_ROOT 기준 상대 경로를 외부 URL에 붙여서 노출
        try:
            rel = final_file_path.relative_to(Path(settings.SAVE_ROOT))
            rel_str = str(rel).replace("\\", "/")
            source_url = f"{settings.EXTERNAL_BASE_URL}/{rel_str}"
        except Exception:
            # 상대 경로 계산 실패 시 파일명만 노출(폴백)
            source_url = f"{settings.EXTERNAL_BASE_URL}/{final_file_path.name}"

        member_id = member_id_override or settings.MEMBER_ID
        duration_sec = 0.0  # TODO: 실제 분석 값으로 대체(ffprobe 등)

    except Exception as e:
        logger.error(f"Failed to get file stats for final notification: {e}")
        file_size_bytes = 0
        source_url = "error_url"
        member_id = member_id_override or settings.MEMBER_ID
        duration_sec = 0.0

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
            "durationSec": duration_sec,
        },
        "message": "Original video successfully merged and stored.",
        "timestamp": int(time.time() * 1000),
    }

    # 1) 스냅샷 저장 (기존 동작 유지)
    try:
        await redis.set(get_status_key(job_id), json.dumps(payload), ex=3600)
        logger.info(f"Final completion JSON reported to Redis for Job {job_id}.")
    except Exception as e:
        logger.error(f"Failed to report final completion JSON to Redis for Job {job_id}: {e}")

    # 2) Pub/Sub 채널로도 발행 (백엔드에서 SUBSCRIBE로 받는 용도)
    try:
        channel = f"{settings.REDIS_UPLOAD_PROGRESS_CHANNEL}:{job_id}"
        await redis.publish(channel, json.dumps(payload))
        logger.info(
            f"Final completion JSON PUBLISHED to channel {channel} for Job {job_id}."
        )
    except Exception as e:
        logger.error(
            f"Failed to publish final completion JSON to channel for Job {job_id}: {e}"
        )

# ─────────────────────────────────────────────────────────────
# 메타 저장 유틸
# ─────────────────────────────────────────────────────────────

async def _save_job_meta(job_id: str, meta: Dict[str, Any]) -> None:
    """AI 워커/백엔드가 조회할 수 있도록 작업 메타를 Redis에 저장."""
    try:
        redis: Redis = get_redis_client()
    except Exception as e:
        logger.warning(f"Redis unavailable; meta not saved for {job_id}: {e}")
        return
    try:
        await redis.set(get_meta_key(job_id), json.dumps(meta), ex=86400)  # 1 day
        logger.info(f"Saved job meta for {job_id}: {meta}")
    except Exception as e:
        logger.warning(f"Failed to save job meta for {job_id}: {e}")

# ─────────────────────────────────────────────────────────────
# AI Worker 연동 로직 (Redis Queue)
# ─────────────────────────────────────────────────────────────

async def _trigger_ai_worker(job_id: str, final_file_path: Path, member_id: Optional[str], highlight_identifier: str):
    """
    AI Worker가 처리할 작업 요청을 Redis List(Queue)에 푸시하고,
    Spring 서버에 AI 작업 시작 대기 상태를 보고합니다.
    """
    try:
        redis: Redis = get_redis_client()
    except RedisConnectionError:
        logger.error(f"Redis not available, cannot queue AI task for Job {job_id}.")
        return
    except Exception as e:
        logger.error(f"Unknown error getting Redis client for Job {job_id}: {e}")
        return

    # 1. AI Worker에게 전달할 페이로드 구성
    ai_payload = AITaskPayload(
        jobId=job_id,
        memberId=(member_id or settings.MEMBER_ID),
        # originalFilePath=str(final_file_path.resolve())  # ⬅️ 이전(컨테이너 내부 절대 경로 위험)
        originalFilePath=str(final_file_path)  # ⬅️ 공유 볼륨 경로 그대로 전달
    )

    # extra 필드(워커 모델에 없어도 무시되도록 JSON에만 포함)
    payload_dict = json.loads(ai_payload.model_dump_json())
    payload_dict["highlightKey"] = highlight_identifier  # 추가 메타

    try:
        push_count = await redis.rpush(
            getattr(settings, "REDIS_QUEUE_NAME", "opencv-ai-job-queue"),
            json.dumps(payload_dict)
        )
        logger.info(
            f"AI Task for Job {job_id} pushed to queue '{getattr(settings, 'REDIS_QUEUE_NAME', 'opencv-ai-job-queue')}'. "
            f"Queue size: {push_count}"
        )

        # AI 시작 대기 상태: 여기서는 그대로 100% (업로드 단계는 이미 끝난 상태)
        await report_progress_to_spring(job_id, UploadStatus.AI_START_PENDING.value, 100.0)

    except Exception as e:
        logger.error(f"Failed to queue AI task for Job {job_id}: {e}")
        await report_progress_to_spring(job_id, UploadStatus.ERROR.value, 0.0)

# ─────────────────────────────────────────────────────────────
# 핵심 비즈니스 로직: 병합 및 정리 (백그라운드 태스크)
# ─────────────────────────────────────────────────────────────

async def merge_chunks_and_cleanup(
    job_id: str,
    file_name: str,
    total_parts: int,
    chunk_dir: Path,
    member_id: Optional[str] = None,   # ← A-2: 선택 인자 추가 (기존 호출과 호환)
):
    """
    백그라운드에서 실행되는 메인 병합 작업 로직입니다.
    - 청크를 임시 경로에 병합한 뒤
    - 공유 볼륨(SAVE_ROOT/{job_id}/{file_name})으로 이동
    - 완료 정보 Redis 보고 및 AI Worker 큐 트리거

    🔹 청크 업로드 구간에서 이미 0~90% 진행률을 보고하고 있으므로,
       이 함수에서는:
       - 병합 완료 시 99% (PROCESSING)
       - 최종 저장 완료 시 100% (UPLOAD_COMPLETE)
       만 추가로 보고합니다.
    """
    # 최종 저장 디렉토리(SAVE_ROOT/{job_id})
    final_save_dir = Path(settings.SAVE_ROOT) / job_id
    final_save_dir.mkdir(parents=True, exist_ok=True)

    # 최종 저장 경로(SAVE_ROOT/{job_id}/{file_name})
    final_save_path = final_save_dir / file_name

    # 임시 병합 경로(TEMP_ROOT/temp_{job_id}_{file_name})
    temp_merge_path = Path(settings.TEMP_ROOT) / f"temp_{job_id}_{file_name}"

    final_path: Optional[Path] = None
    calculated_checksum: Optional[str] = None

    # ⚠️ 여기서는 더 이상 0% 초기화 호출을 하지 않음
    #    (청크 업로드 단계에서 이미 0~90%를 보고하고 있음)

    try:
        # 2) 청크 파일 목록 정렬 및 검증
        #   - 업로드 라우터에서 '{job_id}_{file_name}.{part_index}' 형태로 저장했다고 가정
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        if len(chunk_files) != total_parts:
            raise Exception(
                f"Integrity check failure during merge. expected={total_parts}, actual={len(chunk_files)}"
            )

        logger.info(f"Starting merge of {total_parts} chunks into {temp_merge_path}")

        # 3) 임시 경로에 병합
        #    병합 중 세부 진행률은 별도로 보내지 않고,
        #    전체 병합이 끝난 시점에 99%로 일괄 보고
        with temp_merge_path.open("wb") as outfile:
            for chunk_file in chunk_files:
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        logger.info("Merge completed at temp location. Moving to final save dir and calculating checksum.")

        # 3.2) 병합 완료 시점: 99% 보고 (PROCESSING 상태)
        try:
            await report_progress_to_spring(
                job_id,
                UploadStatus.PROCESSING.value,  # "PROCESSING"
                99.0,
            )
        except Exception as e:
            logger.error(f"Failed to report 99% PROCESSING for Job {job_id}: {e}")

        # 3.5) 최종 경로로 이동 (SAVE_ROOT/{jobId}/{file_name})
        shutil.move(str(temp_merge_path), str(final_save_path))
        final_path = final_save_path

        # 4) 원본이 최종 SAVE_ROOT 경로에 저장된 시점: 100% 보고 (UPLOAD_COMPLETE 상태)
        try:
            await report_progress_to_spring(
                job_id,
                UploadStatus.UPLOAD_COMPLETE.value,  # "UPLOAD_COMPLETE"
                100.0,
            )
        except Exception as e:
            logger.error(f"Failed to report 100% UPLOAD_COMPLETE for Job {job_id}: {e}")

        # 4.5) 체크섬 계산 (진행률에는 영향 없음)
        calculated_checksum = calculate_file_checksum(final_path)

        # 4.6) 하이라이트 식별자 생성 + 메타 저장
        highlight_identifier = str(uuid4())
        await _save_job_meta(
            job_id,
            {
                "memberId": (member_id or settings.MEMBER_ID),
                "highlightIdentifier": highlight_identifier,
                "originalFilePath": str(final_path),
            },
        )

        # 5) 최종 완료 JSON 보고 (멤버 반영)
        await report_final_completion_to_spring(
            job_id,
            final_path,
            calculated_checksum,
            member_id_override=member_id,
        )

        # 6) AI Worker 큐 푸시 및 상태 보고 (하이라이트키 포함)
        await _trigger_ai_worker(job_id, final_path, member_id, highlight_identifier)

    except Exception as e:
        logger.error(
            f"Critical error during merge/cleanup/trigger for Job {job_id}: {e}",
            exc_info=True
        )
        # 에러 시 기존 문자열 상태 유지 ("FAILED") — 백엔드가 이 값에 맞춰있을 수 있어서 건드리지 않음
        await report_progress_to_spring(job_id, "FAILED", 0.0)

    finally:
        # 7) 임시 청크 폴더 삭제
        try:
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)
                logger.info(f"Cleanup complete for Job {job_id}: removed chunk dir {chunk_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove chunk dir {chunk_dir} for Job {job_id}: {e}")

        # 8) 남아있을 수 있는 임시 병합 파일 제거(이동 실패 등의 경우)
        try:
            if temp_merge_path.exists():
                os.remove(temp_merge_path)
                logger.info(f"Cleanup complete for Job {job_id}: removed temp file {temp_merge_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_merge_path} for Job {job_id}: {e}")
