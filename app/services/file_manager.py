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
        with file_path.open("rb") as f:
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
# 진행률/완료 보고 로직 (Spring ProgressData 규격)
# ─────────────────────────────────────────────────────────────

async def report_progress_to_spring(
    job_id: str,
    progress_type: str,
    progress: Optional[float] = None,
    *,
    member_id: Optional[str] = None,
    total_bytes: Optional[int] = None,      # ← 인자는 유지하지만
    received_bytes: Optional[int] = None,   #    아래 payload 에서는 더 이상 사용하지 않음
    size_bytes: Optional[int] = None,
    checksum: Optional[str] = None,
    duration_sec: Optional[float] = None,
    stage: Optional[str] = None,
    current_clip: Optional[int] = None,
    total_clips: Optional[int] = None,
) -> None:
    """
    Spring ProgressData 최종 규격에 맞춘 공통 보고 함수.

    👉 Redis/PubSub 로 나가는 JSON 형식은 타입에 따라 아래와 같이 제한됨.

    1) 청크 업로드 중 (UPLOADING)
    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990000000,
      "type": "UPLOADING",
      "progress": 32.5,
      "jobId": "job1",
      "memberId": "xxxx"
    }

    2) 업로드/병합 완료 (UPLOAD_COMPLETE)
    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990001000,
      "type": "UPLOAD_COMPLETE",
      "jobId": "job1",
      "memberId": "xxxx"
    }

    3) AI 처리 중 (PROCESSING)
    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990002000,
      "type": "PROCESSING",
      "progress": 32.5,
      "jobId": "job1",
      "memberId": "xxxx"
    }

    4) AI 처리 완료 (COMPLETE)
    {
      "status": 200,
      "success": true,
      "timeStamp": 1731990003000,
      "type": "COMPLETE",
      "jobId": "job1",
      "memberId": "xxxx"
    }

    🔹 그 외의 필드(totalBytes, sizeBytes, checksum, stage, currentClip, totalClips 등)는
       백엔드 스펙에 맞추기 위해 전송하지 않는다.
    """
    try:
        redis: Redis = get_redis_client()
    except RedisConnectionError as e:
        logger.error(f"Redis not available, cannot report status for Job {job_id}: {e}")
        return
    except Exception as e:
        logger.error(f"Unknown error getting Redis client for Job {job_id}: {e}")
        return

    # 타입 문자열 정규화
    normalized_type = str(progress_type)

    # 공통 필드 (항상 6개 고정)
    payload: Dict[str, Any] = {
        "status": 200,
        "success": True,
        "timeStamp": int(time.time() * 1000),          # ms 단위
        "type": normalized_type,
        "jobId": str(job_id),
        "memberId": str(member_id or settings.MEMBER_ID),
    }

    # 🔹 progress 는 UPLOADING / PROCESSING 일 때만 포함
    if normalized_type in ("UPLOADING", "PROCESSING") and progress is not None:
        payload["progress"] = float(progress)

    # (나머지 totalBytes, sizeBytes, checksum, stage 등은
    #  인자로만 받고 payload 에는 넣지 않는다.)

    # 1) 스냅샷 저장
    try:
        await redis.set(get_status_key(job_id), json.dumps(payload), ex=3600)
        logger.info(
            f"Job {job_id} status updated: type={normalized_type}, "
            f"progress={payload.get('progress')}"
        )
    except RedisConnectionError as e:
        logger.error(f"Redis connection dropped or operation failed for Job {job_id}: {e}")
    except Exception as e:
        logger.error(f"Failed to report status to Redis for Job {job_id}: {e}")

    # 2) Pub/Sub 발행
    try:
        channel = f"{settings.REDIS_UPLOAD_PROGRESS_CHANNEL}:{job_id}"
        await redis.publish(channel, json.dumps(payload))
        logger.info(
            f"Job {job_id} progress PUBLISHED to {channel}: "
            f"type={normalized_type}, progress={payload.get('progress')}"
        )
    except Exception as e:
        logger.error(f"Failed to publish progress for Job {job_id} to channel: {e}")


async def report_final_completion_to_spring(
    job_id: str,
    final_file_path: Path,
    checksum: str,
    member_id_override: Optional[str] = None,
) -> None:
    """
    UPLOAD_COMPLETE 단계(원본 영상 병합 + SAVE_ROOT에 최종 저장 완료)에 대한
    최종 완료 보고.

    👉 최종적으로는 type="UPLOAD_COMPLETE", progress 없이
       {status, success, timeStamp, type, jobId, memberId} 만 전송된다.
    """
    try:
        # 지금은 사이즈/길이/체크섬을 Redis payload 로 보내지 않는다 (스펙 최소화)
        _ = final_file_path.stat().st_size  # 필요하면 로컬 로그 정도로만 활용 가능
        member_id = member_id_override or settings.MEMBER_ID
    except Exception as e:
        logger.error(f"Failed to get file stats for final notification: {e}")
        member_id = member_id_override or settings.MEMBER_ID

    # UploadStatus 에 UPLOAD_COMPLETE 가 있으면 그 값을 사용, 없으면 문자열 사용
    try:
        progress_type = UploadStatus.UPLOAD_COMPLETE.value  # type: ignore[attr-defined]
    except Exception:
        progress_type = "UPLOAD_COMPLETE"

    try:
        # 🔹 UPLOAD_COMPLETE 에서는 progress 를 포함하지 않음
        await report_progress_to_spring(
            job_id,
            progress_type,
            None,
            member_id=member_id,
        )
        logger.info(f"Final completion reported to Redis for Job {job_id}.")
    except Exception as e:
        logger.error(f"Failed to report final completion to Redis for Job {job_id}: {e}")


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

async def _trigger_ai_worker(
    job_id: str,
    final_file_path: Path,
    member_id: Optional[str],
    highlight_identifier: str,
) -> None:
    """
    AI Worker가 처리할 작업 요청을 Redis List(Queue)에 푸시하고,
    Spring 서버에 '하이라이트 생성 대기(QUEUE에 쌓인 상태)' 진행 상태를 보고.
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
        originalFilePath=str(final_file_path),  # ⬅️ 공유 볼륨 경로 그대로 전달
    )

    # extra 필드(워커 모델에 없어도 무시되도록 JSON에만 포함)
    payload_dict = json.loads(ai_payload.model_dump_json())
    payload_dict["highlightKey"] = highlight_identifier  # 추가 메타

    try:
        queue_name = getattr(settings, "REDIS_QUEUE_NAME", "opencv-ai-job-queue")
        push_count = await redis.rpush(queue_name, json.dumps(payload_dict))
        logger.info(
            f"AI Task for Job {job_id} pushed to queue '{queue_name}'. "
            f"Queue size: {push_count}"
        )

        # 🔹 하이라이트 생성 단계 시작 전, type="PROCESSING", progress=0 으로 한 번 보고
        try:
            try:
                processing_type = UploadStatus.PROCESSING.value  # type: ignore[attr-defined]
            except Exception:
                processing_type = "PROCESSING"

            await report_progress_to_spring(
                job_id,
                processing_type,
                0.0,
                member_id=(member_id or settings.MEMBER_ID),
            )
        except Exception as e:
            logger.error(f"Failed to report PROCESSING(QUEUED) for Job {job_id}: {e}")

    except Exception as e:
        logger.error(f"Failed to queue AI task for Job {job_id}: {e}")
        # 에러 시에는 기존 ERROR/FAILED 플로우와 호환되도록 문자열 사용
        try:
            error_type = UploadStatus.ERROR.value  # type: ignore[attr-defined]
        except Exception:
            error_type = "FAILED"
        await report_progress_to_spring(job_id, error_type, None)


# ─────────────────────────────────────────────────────────────
# 핵심 비즈니스 로직: 병합 및 정리 (백그라운드 태스크)
# ─────────────────────────────────────────────────────────────

async def merge_chunks_and_cleanup(
    job_id: str,
    file_name: str,
    total_parts: int,
    chunk_dir: Path,
    member_id: Optional[str] = None,  # ← 기존 호출과 호환
) -> None:
    """
    백그라운드에서 실행되는 메인 병합 작업 로직입니다.
    - 청크를 임시 경로에 병합한 뒤
    - 공유 볼륨(SAVE_ROOT/{job_id}/{file_name})으로 이동
    - 완료 정보 Redis 보고 및 AI Worker 큐 트리거

    🔹 청크 업로드 구간에서 이미 0~90% 진행률을 보고하고 있으므로,
       이 함수에서는:
       - 병합 완료 시 PROCESSING 99% (여기서도 type=PROCESSING, progress 포함)
       - 최종 저장 완료 시 UPLOAD_COMPLETE (progress 없이)
       만 추가로 보고.
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

    try:
        # 2) 청크 파일 목록 정렬 및 검증
        chunk_files = sorted(chunk_dir.glob(f"{job_id}_{file_name}.*"))
        if len(chunk_files) != total_parts:
            raise Exception(
                f"Integrity check failure during merge. expected={total_parts}, actual={len(chunk_files)}"
            )

        logger.info(f"Starting merge of {total_parts} chunks into {temp_merge_path}")

        # 3) 임시 경로에 병합
        with temp_merge_path.open("wb") as outfile:
            for chunk_file in chunk_files:
                with chunk_file.open("rb") as infile:
                    shutil.copyfileobj(infile, outfile)

        logger.info("Merge completed at temp location. Moving to final save dir and calculating checksum.")

        # 3.2) 병합 완료 시점: PROCESSING 99%
        try:
            try:
                processing_type = UploadStatus.PROCESSING.value  # type: ignore[attr-defined]
            except Exception:
                processing_type = "PROCESSING"

            await report_progress_to_spring(
                job_id,
                processing_type,
                99.0,
                member_id=(member_id or settings.MEMBER_ID),
            )
        except Exception as e:
            logger.error(f"Failed to report 99% PROCESSING for Job {job_id}: {e}")

        # 3.5) 최종 경로로 이동 (SAVE_ROOT/{jobId}/{file_name})
        shutil.move(str(temp_merge_path), str(final_save_path))
        final_path = final_save_path

        # 4) 체크섬 계산 (현재는 Redis payload 에 사용하지 않지만, 필요 시 로그/검증용)
        calculated_checksum = calculate_file_checksum(final_path)

        # 5) 최종 완료 JSON 보고 (type=UPLOAD_COMPLETE, progress 없음)
        await report_final_completion_to_spring(
            job_id,
            final_path,
            calculated_checksum,
            member_id_override=member_id,
        )

        # 6) AI Worker 큐 푸시 및 상태 보고 (하이라이트키 포함)
        await _trigger_ai_worker(job_id, final_path, member_id, highlight_identifier=str(uuid4()))

    except Exception as e:
        logger.error(
            f"Critical error during merge/cleanup/trigger for Job {job_id}: {e}",
            exc_info=True,
        )
        # 에러 시 기존 문자열 상태 유지 ("FAILED")
        await report_progress_to_spring(job_id, "FAILED", None)
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
