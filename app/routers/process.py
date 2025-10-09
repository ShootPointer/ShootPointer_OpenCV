# app/routers/process.py
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, List

import httpx
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi import Query, Body
from fastapi.responses import JSONResponse, FileResponse

from app.core.config import settings
from app.services.pipeline import generate_highlight_clips
from app.services.jersey import detect_player_segments
from app.services.ffmpeg import get_duration
from app.services.streaming import build_zip_spooled
# ⛳ send-img에서 쓰던 캐시 재사용 (등번호 기억)
from app.routers.player import JERSEY_CACHE

router = APIRouter(prefix="/api", tags=["process"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("/tmp/presigned_uploads")
RESULT_DIR = Path("/tmp/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────
# 간단 Job 상태 메모리 저장소
# ─────────────────────────
class JobState:
    def __init__(self, upload_id: str, member_id: str):
        self.uploadId = upload_id
        self.memberId = member_id
        self.status: str = "queued"       # queued|running|done|error
        self.progress: float = 0.0        # 0.0~100.0
        self.message: str = ""
        self.resultZip: Optional[Path] = None
        self.videoPath: Optional[Path] = None
        self.error: Optional[str] = None
        self.requestId: str = uuid.uuid4().hex[:8]

JOBS: Dict[str, JobState] = {}

# ─────────────────────────
# 유틸
# ─────────────────────────
def _find_uploaded(member_id: str, upload_id: str) -> Optional[Path]:
    # presigned_upload.py가 저장한 규칙: {memberId}_{uploadId}.mp4
    cand = UPLOAD_DIR / f"{member_id}_{upload_id}.mp4"
    return cand if cand.exists() else None

async def _safe_post_json(url: str, payload: dict, timeout: float = 10.0):
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as cli:
            r = await cli.post(url, json=payload)
            return r.status_code, (r.text[:200] if r.text else "")
    except Exception as e:
        return -1, str(e)

async def _notify(job: JobState, progress_url: Optional[str], completed_url: Optional[str]):
    # 진행률 또는 완료 통지(둘 중 필요한 쪽으로만 호출)
    payload = {
        "uploadId": job.uploadId,
        "memberId": job.memberId,
        "status": job.status,
        "progress": round(job.progress, 1),
        "message": job.message,
        "requestId": job.requestId,
    }
    if job.status == "done":
        payload["downloadUrl"] = f"/api/process/{job.uploadId}/result"  # OpenCV 서버의 다운로드 엔드포인트
    url = completed_url if job.status == "done" and completed_url else progress_url
    if not url:
        return
    code, note = await _safe_post_json(url, payload)
    logger.info(f"[process.notify] [{job.requestId}] -> {url} ({code}) {note}")

def _set_progress(job: JobState, p: float, msg: str):
    job.progress = max(0.0, min(100.0, p))
    job.message = msg
    logger.info(f"[process] [{job.requestId}] {job.uploadId} {job.progress:.1f}% - {msg}")

# ─────────────────────────
# 처리 본체
# ─────────────────────────
async def _run_job(job: JobState, progress_url: Optional[str], completed_url: Optional[str], max_clips: int):
    try:
        job.status = "running"
        _set_progress(job, 1.0, "locating-upload")

        src = _find_uploaded(job.memberId, job.uploadId)
        if not src:
            raise FileNotFoundError("uploaded video not found")
        job.videoPath = src
        total = get_duration(src)

        # 등번호 찾기 (send-img 캐시에 저장해둔 값)
        if job.memberId not in JERSEY_CACHE:
            raise RuntimeError("jersey number not found in cache (call /api/send-img first)")
        jersey_number = JERSEY_CACHE[job.memberId]

        # 세그먼트 검출
        _set_progress(job, 20.0, "detecting-player-segments")
        segs = detect_player_segments(src, jersey_number)
        _set_progress(job, 40.0, f"segments={len(segs)}")

        # 이벤트(후보) 만들기: segs의 중앙값들(너희 로직에 맞게 바꿔도 됨)
        events: List[float] = []
        for s, e in segs:
            mid = (s + e) / 2.0
            if 0.0 <= mid <= total:
                events.append(mid)
        if not events:
            # 세그먼트가 없다면, 전체 중간 하나라도 생성(빈 결과 방지)
            events = [max(0.2, total * 0.5)]

        # 클립 생성
        _set_progress(job, 65.0, "cutting-clips")
        results = generate_highlight_clips(
            src,
            jersey_number,
            events,
            pre=settings.DEFAULT_PRE,
            post=settings.DEFAULT_POST,
            max_clips=max_clips,
        )
        clip_paths = [p for _, p in results]

        # ZIP 작성
        _set_progress(job, 85.0, "zipping")
        spooled = build_zip_spooled(clip_paths, arc_prefix=f"#{jersey_number}_")
        out_zip = RESULT_DIR / f"{job.memberId}_{job.uploadId}.zip"
        with out_zip.open("wb") as f:
            for chunk in iter(lambda: spooled.read(64 * 1024), b""):
                if not chunk: break
                f.write(chunk)
        spooled.close()

        job.resultZip = out_zip
        job.status = "done"
        _set_progress(job, 100.0, "completed")
        await _notify(job, progress_url, completed_url)

        # (선택) 원본 정리: 운영정책에 맞게 주석 해제
        # try: src.unlink(missing_ok=True)
        # except: pass

    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.message = f"error: {e}"
        logger.exception(f"[process] [{job.requestId}] job failed: {e}")
        await _notify(job, progress_url, completed_url)

# ─────────────────────────
# API
# ─────────────────────────
@router.post("/process/{uploadId}", summary="Start processing an uploaded video")
async def start_process(
    request: Request,
    background: BackgroundTasks,
    uploadId: str,
    memberId: str = Query(..., description="멤버 식별자"),
    maxClips: int = Query(settings.MAX_CLIPS, description="최대 클립 수"),
    body: dict = Body(
        default={},
        description="옵션: {progressCallbackUrl, completedCallbackUrl} (Spring가 받을 엔드포인트)"
    ),
):
    """
    - presigned 업로드가 끝난 뒤 Spring이 호출
    - send-img를 통해 등번호가 캐시에 있어야 함
    - 비동기로 처리 시작 → 즉시 202 응답
    """
    req_id = uuid.uuid4().hex[:8]
    job = JobState(uploadId, memberId)
    job.requestId = req_id
    JOBS[uploadId] = job

    progress_url = (body or {}).get("progressCallbackUrl")
    completed_url = (body or {}).get("completedCallbackUrl")

    logger.info(f"[process] [{req_id}] start uploadId={uploadId} memberId={memberId} "
                f"progressUrl={progress_url} completedUrl={completed_url}")

    # 바로 진행률 0% 알림 한번
    await _notify(job, progress_url, None)

    # 백그라운드 실행
    background.add_task(_run_job, job, progress_url, completed_url, maxClips)
    return JSONResponse(status_code=202, content={
        "status": 202,
        "success": True,
        "uploadId": uploadId,
        "memberId": memberId,
        "requestId": req_id
    })

@router.get("/process/{uploadId}/status", summary="Get processing status")
async def get_status(uploadId: str):
    job = JOBS.get(uploadId)
    if not job:
        return JSONResponse(status_code=404, content={"status": "error", "message": "job not found"})
    return {
        "status": job.status,
        "progress": round(job.progress, 1),
        "message": job.message,
        "uploadId": job.uploadId,
        "memberId": job.memberId,
        "requestId": job.requestId,
        "error": job.error,
        "hasResult": bool(job.resultZip and job.resultZip.exists()),
        "downloadUrl": (f"/api/process/{uploadId}/result" if job.resultZip and job.resultZip.exists() else None),
    }

@router.get("/process/{uploadId}/result", summary="Download ZIP if ready")
async def get_result(uploadId: str):
    job = JOBS.get(uploadId)
    if not job or job.status != "done" or not job.resultZip or not job.resultZip.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "result not ready"})
    return FileResponse(path=job.resultZip, filename=job.resultZip.name, media_type="application/zip")
