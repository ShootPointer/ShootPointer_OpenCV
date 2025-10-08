# app/routers/player.py
from __future__ import annotations

import logging
import shutil
import uuid
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from fastapi import (
    APIRouter, UploadFile, File, Form, Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from starlette import status

from app.core.config import settings
from app.services.jersey import (
    ocr_jersey_image_bytes,
    detect_player_segments,
)
from app.services.pipeline import generate_highlight_clips
from app.services.streaming import build_zip_spooled, stream_zip_and_cleanup

router = APIRouter(prefix="/api", tags=["player"])
logger = logging.getLogger(__name__)

# 업로드 임시 폴더
TMP_DIR = Path("/tmp/uploads")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 간단 캐시: memberId -> 등번호
JERSEY_CACHE: Dict[str, int] = {}

# ───────────────────────── helpers ─────────────────────────
def _save_upload(video: UploadFile) -> Path:
    suffix = Path(video.filename or "").suffix or ".mp4"
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        size = tmp.stat().st_size
    except Exception:
        size = -1
    logger.info(f"[upload] saved -> {tmp.name} ({size} bytes)")
    return tmp

def _in_any_segment(t: float, segments: List[Tuple[float, float]]) -> bool:
    for s, e in segments:
        if s <= t <= e:
            return True
    return False

def _parse_timestamps(raw: str) -> List[float]:
    out: List[float] = []
    for part in (raw or "").replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = float(p)
            if v >= 0:
                out.append(v)
        except Exception:
            pass
    return sorted(set(out))

def _get_member_id(req: Request) -> str:
    return req.headers.get("X-Member-Id") or "__GLOBAL__"

def _digits_only(s: str) -> str:
    """문자열에서 숫자만 추출(공백/하이픈 등 제거)."""
    return "".join(re.findall(r"\d+", s or ""))

# ───────────────────── 1) 이미지 수신 & OCR ─────────────────────
@router.post("/send-img", summary="Ocr From Image")
async def send_img(
    request: Request,
    image: UploadFile = File(..., alias="image", description="등번호가 보이는 이미지"),
    back_number: str | None = Form(None, alias="backNumber", description="(선택) 기대 등번호(문자열 허용)"),
):
    """
    이미지에서 등번호를 OCR로 추출해 캐시에 저장.
    - 헤더 `X-Member-Id` 로 사용자 구분(없으면 __GLOBAL__).
    - backNumber는 '문자열'로 받고 내부에서 숫자만 추출해 비교.
    - 성공: 200 {detectedNumber, confidence, expectedNumber, match, memberId}
    - 유효성/처리 실패: 422 {status:'error', step, message, memberId}
    """
    step = "start"
    member_id = _get_member_id(request)
    try:
        # 입력 로깅(민감정보 제외)
        logger.info(f"[/send-img] member={member_id}, filename={image.filename}, "
                    f"content_type={image.content_type}, backNumber_raw={back_number!r}")

        step = "read_image"
        img_bytes = await image.read()
        if not img_bytes:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "빈 이미지입니다.", "memberId": member_id},
            )

        step = "ocr"
        digits, conf = ocr_jersey_image_bytes(img_bytes)
        logger.info(f"[/send-img] OCR -> digits={digits!r}, conf={conf:.2f}, member={member_id}")
        if not digits:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "숫자 인식 실패", "memberId": member_id},
            )

        step = "cache"
        detected = int(digits)
        JERSEY_CACHE[member_id] = detected

        # 기대값 비교(문자열 허용)
        expected_clean = _digits_only(back_number) if back_number is not None else None
        match: Optional[bool] = None
        if expected_clean:
            try:
                match = (int(expected_clean) == detected)
            except Exception:
                # 기대값이 숫자로 변환 불가 → 비교 생략
                match = None

        step = "done"
        return {
            "status": "ok",
            "detectedNumber": detected,
            "confidence": conf,
            "expectedNumber": (int(expected_clean) if expected_clean else None),
            "match": match,
            "memberId": member_id,
        }

    except Exception as e:
        logger.exception(f"[/send-img] failed at step={step}: {e}")
        # 내부 오류는 400으로, 처리/유효성은 위에서 422로 반환
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "step": step, "message": str(e), "memberId": member_id},
        )

# ─────────────── 2) 세그먼트 검출 (번호는 캐시에서) ───────────────
@router.post("/segments", summary="Detect Segments")
async def segments(
    request: Request,
    videoFile: UploadFile = File(..., alias="videoFile", description="풀경기 영상(mp4 등)"),
):
    member_id = _get_member_id(request)
    if member_id not in JERSEY_CACHE:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "등번호 정보가 없습니다. 먼저 /api/send-img 로 이미지를 보내주세요.",
                     "memberId": member_id},
        )

    jersey_number = JERSEY_CACHE[member_id]
    tmp_in = _save_upload(videoFile)

    try:
        segs = detect_player_segments(tmp_in, jersey_number)
        logger.info(f"[/segments] member={member_id} jersey={jersey_number} segs={len(segs)}")
        return {"segments": segs, "jerseyNumber": jersey_number, "memberId": member_id}
    except Exception as e:
        logger.exception(f"[/segments] failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e), "memberId": member_id})
    finally:
        try:
            tmp_in.unlink(missing_ok=True)
        except Exception:
            pass

# ─────────────── 3) 하이라이트 ZIP 생성 (번호는 캐시에서) ───────────────
@router.post("/highlight", summary="Generate Highlights Zip")
async def highlight(
    request: Request,
    videoFile: UploadFile = File(..., alias="videoFile", description="풀경기 영상(mp4 등)"),
    timestamps: str = Form(..., alias="timestamps", description="쉼표구분 초 리스트, 예: 30,75,120"),
    pre: float = Form(default=settings.DEFAULT_PRE, alias="pre"),
    post: float = Form(default=settings.DEFAULT_POST, alias="post"),
    maxClips: int = Form(default=settings.MAX_CLIPS, alias="maxClips"),
):
    member_id = _get_member_id(request)
    if member_id not in JERSEY_CACHE:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "등번호 정보가 없습니다. 먼저 /api/send-img 로 이미지를 보내주세요.",
                     "memberId": member_id},
        )

    jersey_number = JERSEY_CACHE[member_id]
    tmp_in = _save_upload(videoFile)
    clip_paths: List[Path] = []
    tmp_to_cleanup: List[Path] = [tmp_in]

    try:
        segments = detect_player_segments(tmp_in, jersey_number)

        events = _parse_timestamps(timestamps)
        events = [t for t in events if _in_any_segment(t, segments)] if segments else events
        if not events:
            raise RuntimeError("선수 구간과 겹치는 이벤트가 없습니다. (timestamps 확인)")

        if maxClips and maxClips > 0:
            events = events[:maxClips]

        logger.info(f"[/highlight] member={member_id} jersey={jersey_number} events={events}")

        results = generate_highlight_clips(
            tmp_in, jersey_number, events, pre=pre, post=post, max_clips=maxClips
        )
        clip_paths = [Path(p) for _, p in results]

        spooled = build_zip_spooled(clip_paths, arc_prefix=f"#{jersey_number}_")
        return stream_zip_and_cleanup(spooled, clip_paths, tmp_to_cleanup)

    except Exception as e:
        logger.exception(f"[/highlight] failed: {e}")
        for p in clip_paths:
            try: Path(p).unlink(missing_ok=True)
            except Exception: pass
        try: Path(tmp_in).unlink(missing_ok=True)
        except Exception: pass
        return JSONResponse(status_code=400, content={"error": str(e), "memberId": member_id})
