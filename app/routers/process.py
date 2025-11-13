# app/routers/process.py
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, List

from fastapi import APIRouter, BackgroundTasks, Body, Query
from fastapi.responses import JSONResponse

from app.core.redis_client import get_redis_client
from app.schemas.redis import AITaskPayload, UploadStatus, ProgressMessage
from app.core.config import settings
from app.routers.player import JERSEY_CACHE

router = APIRouter(prefix="/api", tags=["process"])
logger = logging.getLogger(__name__)

# ─────────────────────────
# Job 상태 메모리 저장소 (디버깅용)
# ─────────────────────────
class JobState:
    """처리 Job의 상태를 임시로 저장하는 인메모리 모델"""
    def __init__(self, upload_id: str, member_id: str, highlight_key: Optional[str] = None):
        self.uploadId = upload_id
        self.memberId = member_id
        self.highlightKey = highlight_key
        self.status: str = "queued"      # queued|running|done|error
        self.progress: float = 0.0
        self.message: str = ""
        self.resultUrl: Optional[str] = None
        self.error: Optional[str] = None
        self.requestId: str = uuid.uuid4().hex[:8]

JOBS: Dict[str, JobState] = {}

# ─────────────────────────
# 내부: AI Worker 큐잉
# ─────────────────────────
async def _queue_ai_worker_job(job_payload: AITaskPayload, queue_name: str = settings.REDIS_QUEUE_NAME):
    """비동기 Redis 클라이언트를 사용하여 AI Worker 큐(List)에 작업을 푸시"""
    redis_async = get_redis_client()
    message = job_payload.model_dump_json()
    await redis_async.rpush(queue_name, message)
    logger.info(f"[queue] PUSHED job to {queue_name}: {job_payload.jobId}")

# ─────────────────────────
# 내부: Spring용 진행률 Pub/Sub (비동기)
# ─────────────────────────
async def _publish_progress_async(
    job_id: str,
    status: UploadStatus,
    message: str = "",
    current: int = 0,
    total: int = 1,
) -> None:
    """간단 Pub/Sub 발행"""
    try:
        redis_async = get_redis_client()
        progress_msg = ProgressMessage(
            jobId=job_id,
            status=status,
            message=message,
            current=current,
            total=total,
        )
        channel = f"{settings.REDIS_UPLOAD_PROGRESS_CHANNEL}:{job_id}"
        await redis_async.publish(channel, progress_msg.model_dump_json())
        logger.debug(f"[pubsub] {channel}: {status.value} ({current}/{total})")
    except Exception as e:
        logger.error(f"[pubsub] Failed to publish progress for job {job_id}: {e}")

# ─────────────────────────
# 유틸: 병합본 실제 경로 찾기
#   - 저장 형식: SAVE_ROOT/{uploadId}/{fileName}
#   - 업로더가 보낸 fileName을 모르므로 디렉터리 내 '영상 파일' 1개를 탐색
# ─────────────────────────
_VIDEO_EXTS: List[str] = [".mp4", ".mkv", ".mov", ".m4v"]

def _find_merged_original(upload_id: str) -> Optional[Path]:
    base = Path(settings.SAVE_ROOT) / upload_id
    if not base.exists():
        return None
    # 가장 최근 파일 우선
    candidates = sorted(
        [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None

# ─────────────────────────
# API: Job 시작 (큐잉 + Pub/Sub 알림)
# ─────────────────────────
@router.post("/process/{uploadId}", summary="Start processing (Queue Job for AI Worker)")
async def start_process(
    uploadId: str,
    background: BackgroundTasks,
    memberId: str = Query(..., description="멤버 식별자 (X-Member-Id와 동일한 의미)"),
    highlightKey: Optional[str] = Query(
        None,
        description="원본-하이라이트 그룹 구분용 키(폴더/식별자). 없으면 'default'로 처리.",
    ),
    maxClips: int = Query(settings.MAX_CLIPS, description="최대 클립 수"),
    body: dict = Body(default={}, description="옵션: 콜백 URL {progressCallbackUrl, completedCallbackUrl}"),
):
    """
    업로드 완료 후 호출되어 AI Worker에 Job을 큐잉.
    - 등번호는 /api/send-img 로 캐시에 저장된 값을 사용
    - 병합본 경로: {SAVE_ROOT}/{uploadId}/{fileName} 형태로 탐색
    """
    # 1) 등번호 확인 (요구사항 유지)
    if memberId not in JERSEY_CACHE:
        return JSONResponse(
            status_code=400,
            content={"error": "jersey_missing", "detail": "jersey number not found in cache. Call /api/send-img first."}
        )
    jersey_number = JERSEY_CACHE[memberId]

    # 2) Job 상태 임시 저장
    job = JobState(uploadId, memberId, highlight_key=highlightKey)
    JOBS[uploadId] = job

    # 3) 병합본 실제 파일 경로 찾기
    merged = _find_merged_original(uploadId)
    if not merged:
        logger.error(f"[process] merged original not found under {settings.SAVE_ROOT}/{uploadId}")
        return JSONResponse(
            status_code=404,
            content={"error": "original_not_found", "detail": f"merged original not found: {settings.SAVE_ROOT}/{uploadId}"}
        )
    original_file_path = str(merged)

    # 4) AI Worker 페이로드 (요청대로 '3필드만')
    job_payload = AITaskPayload(
        jobId=uploadId,
        memberId=memberId,
        originalFilePath=original_file_path,
        highlightKey=highlightKey,   # ← 추가!
    )

    logger.info(
        f"[process] Queueing job: uploadId={uploadId}, memberId={memberId}, jersey={jersey_number}, "
        f"highlightKey={highlightKey}, maxClips={maxClips}, original={original_file_path}"
    )

    # 5) Redis Queue에 Job 넣기
    try:
        await _queue_ai_worker_job(job_payload)
    except ConnectionError as e:
        logger.error(f"[process] Redis Connection Error (Queueing): {e}")
        JOBS[uploadId].status = "error"
        JOBS[uploadId].error = "Redis connection failed for queueing."
        return JSONResponse(
            status_code=503,
            content={"error": "queue_unreachable", "detail": "Cannot connect to Redis queue service."}
        )

    # 6) Pub/Sub 알림 (요구사항 유지)
    await _publish_progress_async(
        job_id=uploadId,
        status=UploadStatus.AI_START_PENDING,
        message=f"Job queued successfully. Jersey: {jersey_number}",
        current=0,
        total=1,
    )

    # 7) 202 Accepted
    response_payload = {
        "status": 202,
        "success": True,
        "message": "Job successfully queued for AI Worker and Spring notified.",
        "uploadId": uploadId,
        "memberId": memberId,
        "jerseyNumber": jersey_number,
        "highlightKey": highlightKey,
        "maxClips": maxClips,
        "originalFilePath": original_file_path,
    }
    return JSONResponse(status_code=202, content=response_payload)

# ─────────────────────────
# 상태 조회 (디버깅용)
# ─────────────────────────
@router.get("/process/{uploadId}/status", summary="Get processing status (from in-memory or Redis/DB)")
async def get_status(uploadId: str):
    job = JOBS.get(uploadId)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "job not found in local memory."}
        )
    return {
        "status": job.status,
        "progress": round(job.progress, 1),
        "message": job.message,
        "uploadId": job.uploadId,
        "memberId": job.memberId,
        "requestId": job.requestId,
        "error": job.error,
        "resultUrl": job.resultUrl,
        "highlightKey": job.highlightKey,
    }

# ─────────────────────────
# 결과 다운로드(DEPRECATED)
# ─────────────────────────
@router.get("/process/{uploadId}/result", summary="Download ZIP if ready - DEPRECATED")
async def get_result(uploadId: str):
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "Download endpoint is deprecated. Use the static file URL provided in the completion callback."
        }
    )
