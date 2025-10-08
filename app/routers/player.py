# app/routers/player.py
from __future__ import annotations

import logging
import shutil
import uuid
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from fastapi import (
    APIRouter, UploadFile, File, Form, Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from starlette import status

import cv2
import numpy as np

from app.core.config import settings
from app.services.jersey import (
    ocr_jersey_image_bytes,
    detect_player_segments,
    # 힌트 기반 랭킹 버전이 jersey.py에 있다면 사용
    # (없으면 아래 try/except NameError 폴백)
    # ocr_jersey_image_bytes_with_hint,
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
    tmp = TMP_DIR / f"{uuid.uuid4().hex}{suffix}."
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

def _digits_only(s: Optional[str]) -> str:
    """문자열에서 숫자만 추출(공백/하이픈 등 제거)."""
    return "".join(re.findall(r"\d+", s or ""))

def _is_allowed_image_type(content_type: str | None) -> bool:
    return (content_type or "").lower() in {
        "image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"
    }

# ───────────────────── 1) 이미지 수신 & OCR ─────────────────────
@router.post("/send-img", summary="Ocr From Image (camelCase only)")
async def send_img(
    request: Request,
    image: UploadFile = File(..., alias="image", description="등번호가 보이는 이미지"),
    # ✅ 카멜식만 허용
    backNumber: str | None = Form(None, alias="backNumber", description="(선택) 기대 등번호(문자열 허용)"),
):
    """
    이미지에서 등번호를 OCR로 추출해 캐시에 저장.
    - 헤더 `X-Member-Id` 로 사용자 구분(없으면 __GLOBAL__).
    - backNumber는 '문자열'로 받고 내부에서 숫자만 추출해 비교/힌트로 사용.
    - 성공: 200 {detectedNumber, confidence, expectedNumber, match, memberId, tookMs}
    - 유효성/처리 실패: 4xx {status:'error', step, message, memberId}
    """
    t0 = time.perf_counter()
    step = "start"
    member_id = _get_member_id(request)

    # 입력 로깅(민감정보 제외) — camelCase만
    logger.info(
        f"[/send-img] member={member_id}, filename={image.filename}, "
        f"content_type={image.content_type}, backNumber_raw={backNumber!r}"
    )

    try:
        # ── 1) 컨텐츠 타입/사이즈 검증
        step = "validate_input"
        if not _is_allowed_image_type(image.content_type):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step,
                         "message": f"지원하지 않는 이미지 타입입니다: {image.content_type}",
                         "memberId": member_id},
            )

        # 파일 읽기
        step = "read_image"
        img_bytes = await image.read()
        if not img_bytes:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "빈 이미지입니다.", "memberId": member_id},
            )
        if settings.MAX_UPLOAD_BYTES and len(img_bytes) > settings.MAX_UPLOAD_BYTES:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"status": "error", "step": step,
                         "message": f"이미지 크기가 제한({settings.MAX_UPLOAD_BYTES} bytes)를 초과했습니다.",
                         "memberId": member_id},
            )

        # ── 2) 디코드 & 기본 정보 로깅
        step = "decode_image"
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("이미지 디코드 실패")
            H, W = bgr.shape[:2]
            logger.info(f"[/send-img] decoded image size = {W}x{H}")
        except Exception as e:
            logger.exception(f"[/send-img] decode failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "이미지 디코드 실패", "memberId": member_id},
            )

        # ✅ 2-1) 너무 크면 리사이즈 후, PNG로 재인코딩하여 OCR 입력 바이트 교체
        #   - 짧은 변 기준 1000px로 축소 (속도↑, 품질 손실 최소화)
        step = "resize_if_needed"
        try:
            shorter = min(W, H)
            if shorter > 1000:
                scale = 1000.0 / float(shorter)
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                H, W = bgr.shape[:2]
                # PNG로 재인코딩하여 img_bytes 갱신 (무손실/안정적)
                ok, enc = cv2.imencode(".png", bgr)
                if not ok:
                    raise ValueError("리사이즈 후 인코딩 실패")
                img_bytes = enc.tobytes()
                logger.info(f"[/send-img] resized for OCR = {W}x{H}")
        except Exception as e:
            # 리사이즈 실패 시 원본 바이트로 진행(경고만)
            logger.warning(f"[/send-img] resize skipped due to error: {e}")

        # DEBUG 모드면 OCR 주요 세팅 로그
        if settings.DEBUG_OCR:
            try:
                logger.debug(
                    f"[/send-img] DEBUG_OCR on — "
                    f"OEM={settings.JERSEY_TESSERACT_OEM}, "
                    f"PSMs={getattr(settings, 'OCR_PSMS', None)}, "
                    f"SCALES={getattr(settings, 'OCR_SCALES', None)}, "
                    f"INVERT={getattr(settings, 'OCR_TRY_INVERT', None)}, "
                    f"TIMEOUT={settings.OCR_TIMEOUT_SEC}s"
                )
            except Exception:
                pass

        # ── 3) OCR
        step = "ocr"
        expected_digits = _digits_only(backNumber) if backNumber else ""
        try:
            # 힌트 기반 함수가 있으면 사용, 없으면 기본 함수 사용
            from app.services.jersey import ocr_jersey_image_bytes_with_hint  # type: ignore
            if expected_digits:
                digits, conf = ocr_jersey_image_bytes_with_hint(img_bytes, expected_digits)
            else:
                digits, conf = ocr_jersey_image_bytes(img_bytes)
        except Exception as e:
            # 함수가 없거나 내부에서 오류가 난 경우 기본으로 폴백
            logger.warning(f"[/send-img] hint OCR unavailable or failed: {e}")
            digits, conf = ocr_jersey_image_bytes(img_bytes)

        logger.info(f"[/send-img] OCR -> digits={digits!r}, conf={conf:.2f}, member={member_id}")
        if not digits:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "숫자 인식 실패", "memberId": member_id},
            )

        # ── 4) 캐시 저장 및 기대값 비교
        step = "cache"
        try:
            detected = int(digits)
        except Exception:
            detected = int(_digits_only(digits) or "0")
        JERSEY_CACHE[member_id] = detected

        match: Optional[bool] = None
        expected_num: Optional[int] = None
        if expected_digits:
            try:
                expected_num = int(expected_digits)
                match = (expected_num == detected)
            except Exception:
                match = None

        took_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        # ── 5) 정상 응답
        step = "done"
        return {
            "status": "ok",
            "detectedNumber": detected,
            "confidence": conf,
            "expectedNumber": expected_num,
            "match": match,
            "memberId": member_id,
            "tookMs": took_ms,
        }

    except Exception as e:
        logger.exception(f"[/send-img] failed at step={step}: {e}")
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
        segs = detect_player_segments(tmp_in, jersey_number)

        events = _parse_timestamps(timestamps)
        events = [t for t in events if _in_any_segment(t, segs)] if segs else events
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
