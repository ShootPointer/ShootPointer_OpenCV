# app/routers/frames.py
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from starlette import status

from app.core.config import settings
from app.services.frames import sample_frames_with_timestamps

router = APIRouter(prefix="/api/frames", tags=["frames"])
logger = logging.getLogger(__name__)

TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────── helpers ─────────────────────────
def _getMemberId(req: Request) -> str:
    return req.headers.get("X-Member-Id") or "__GLOBAL__"

def _isAllowedVideoType(contentType: Optional[str]) -> bool:
    return (contentType or "").lower() in {
        "video/mp4",
        "video/quicktime",       # .mov
        "video/x-matroska",      # .mkv
        "video/webm",
    }

def _safeSuffix(name: Optional[str]) -> str:
    suffix = (Path(name or "").suffix or "").lower()
    if not suffix or len(suffix) > 5:
        suffix = ".mp4"
    return suffix

def _saveUpload(video: UploadFile) -> Path:
    """
    업로드 파일을 /tmp/uploads 에 안전하게 저장 (1MB 청크 저장: 메모리 과점유 방지)
    """
    suffix = _safeSuffix(video.filename)
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with tmp.open("wb") as f:
        while True:
            chunk = video.file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            f.write(chunk)
            size += len(chunk)
    logger.info(f"[frames.upload] saved -> {tmp.name} ({size} bytes)")
    return tmp

def _clampFloat(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _clampInt(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

# ───────────────────────── endpoint ─────────────────────────
@router.post("/probe", summary="Sample thumbnails with timestamps")
async def framesProbe(
    request: Request,
    # 요청 필드명은 프런트 호환 위해 'video' 유지
    video: UploadFile = File(..., description="풀경기 영상(mp4 등)"),
    fps: float = Form(0.5, description="초당 추출 프레임 수. 예: 0.5면 2초에 1장"),
    maxFrames: int = Form(40, description="최대 프레임 수 (상한)"),
    scaleWidth: int = Form(640, description="썸네일 가로 최대 폭"),
    # 선택: ffmpeg/opencv 중 택1 (지정 없으면 ffmpeg)
    engine: Literal["ffmpeg", "opencv"] = Form("ffmpeg"),
):
    """
    영상에서 일정 간격(fps)으로 썸네일을 추출하고, 각 프레임의 실제 타임스탬프(초)를 함께 반환.
    성공: {status:200, success:true, count, frames, memberId}
    실패: {status:'error', message, memberId}
    """
    memberId = _getMemberId(request)

    # 1) 타입/용량 검증
    if not _isAllowedVideoType(video.content_type):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"status": "error", "message": f"unsupportedVideoType: {video.content_type}", "memberId": memberId},
        )
    if settings.MAX_UPLOAD_BYTES and settings.MAX_UPLOAD_BYTES > 0:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "status": "error",
                    "message": f"payloadTooLarge: {cl} > {settings.MAX_UPLOAD_BYTES}",
                    "memberId": memberId,
                },
            )

    # 2) 파라미터 정제(이상값 방어)
    try:
        fps = float(fps)
        maxFrames = int(maxFrames)
        scaleWidth = int(scaleWidth)
    except Exception:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": "invalidFormValues", "memberId": memberId},
        )
    # fps: (0, 30] 범위로 클램프
    fps = _clampFloat(fps, 0.05, 30.0)
    # maxFrames: [1, 200] 범위
    maxFrames = _clampInt(maxFrames, 1, 200)
    # scaleWidth: [160, 1920] 범위
    scaleWidth = _clampInt(scaleWidth, 160, 1920)

    # 3) 저장 → 추출 → 정리
    tmpIn = _saveUpload(video)
    try:
        frames = sample_frames_with_timestamps(
            tmpIn, fps=fps, max_frames=maxFrames, scale_width=scaleWidth, engine=engine
        )
        # frames 구조 예: [{ "t": float, "image_path": str }, ...]
        return {
            "status": 200,
            "success": True,
            "count": len(frames),
            "frames": frames,
            "memberId": memberId,
        }
    except Exception as e:
        logger.exception(f"[/frames/probe] failed: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e), "memberId": memberId},
        )
    finally:
        try:
            tmpIn.unlink(missing_ok=True)
        except Exception:
            pass
