# app/routers/player.py
from __future__ import annotations

import logging
import json
import re
import time
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from starlette import status

from app.core.config import settings
from app.services.jersey import (
    ocr_jersey_image_bytes,
    ocr_jersey_image_bytes_with_hint,
)

router = APIRouter(prefix="/api", tags=["player"])
logger = logging.getLogger(__name__)

# 단순 캐시: memberId -> 등번호
JERSEY_CACHE: Dict[str, int] = {}

# ───────────────────────── helpers ─────────────────────────
def _get_member_id(req: Request) -> str:
    return req.headers.get("X-Member-Id") or "__GLOBAL__"

def _digits_only(s: Optional[str]) -> str:
    return "".join(re.findall(r"\d+", s or ""))

def _is_allowed_image_type(content_type: str | None) -> bool:
    return (content_type or "").lower() in {"image/jpeg", "image/jpg"}

# ───────────────────── 1) 이미지 수신 & OCR ─────────────────────
@router.post("/send-img", summary="Ocr From Image (camelCase only)")
async def send_img(
    request: Request,
    image: UploadFile = File(..., alias="image", description="등번호가 보이는 이미지(JPEG)"),
    backNumber: str | None = Form(None, alias="backNumber"),
    backNumberRequestDto: str | None = Form(None, alias="backNumberRequestDto"),
):
    """
    이미지에서 등번호를 OCR로 추출해 캐시에 저장.
    - 헤더 X-Member-Id 로 사용자 구분(없으면 __GLOBAL__).
    - 힌트(backNumber/Dto)는 숫자만 추출해 비교/가중치로 사용.
    """
    t0 = time.perf_counter()
    step = "start"
    member_id = _get_member_id(request)

    logger.info(
        f"[/send-img] member={member_id}, filename={image.filename}, "
        f"content_type={image.content_type}, backNumber_raw={backNumber!r}, "
        f"backNumberRequestDto_raw={backNumberRequestDto!r}"
    )

    try:
        # 타입/크기 검사
        step = "validate_input"
        if not _is_allowed_image_type(image.content_type):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "JPEG만 지원합니다.", "memberId": member_id},
            )

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

        # 디코드 및 필요 시 리사이즈
        step = "decode_image"
        try:
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("이미지 디코드 실패")
            H, W = bgr.shape[:2]
            logger.info(f"[/send-img] decoded {W}x{H}")
        except Exception as e:
            logger.exception(f"[/send-img] decode failed: {e}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": step, "message": "이미지 디코드 실패", "memberId": member_id},
            )

        step = "resize_if_needed"
        try:
            shorter = min(W, H)
            if shorter > 1000:
                scale = 1000.0 / float(shorter)
                bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                H, W = bgr.shape[:2]
                ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    img_bytes = enc.tobytes()
                logger.info(f"[/send-img] resized to {W}x{H}")
        except Exception as e:
            logger.warning(f"[/send-img] resize skipped: {e}")

        if settings.DEBUG_OCR:
            logger.debug(
                f"[/send-img] DEBUG_OCR OEM={settings.JERSEY_TESSERACT_OEM}, "
                f"PSMs={getattr(settings, 'OCR_PSMS', None)}, "
                f"SCALES={getattr(settings, 'OCR_SCALES', None)}, "
                f"INVERT={getattr(settings, 'OCR_TRY_INVERT', None)}, "
                f"TIMEOUT={settings.OCR_TIMEOUT_SEC}s"
            )

        # 힌트 파싱
        step = "ocr"
        expected_digits = ""
        if backNumberRequestDto:
            try:
                parsed = json.loads(backNumberRequestDto)
                if isinstance(parsed, dict) and parsed.get("backNumber") is not None:
                    expected_digits = _digits_only(str(parsed["backNumber"]))
            except Exception as e:
                logger.warning(f"[/send-img] backNumberRequestDto parse failed: {e}")
        if not expected_digits and backNumber:
            expected_digits = _digits_only(backNumber)

        # OCR 실행
        if expected_digits:
            digits, conf = ocr_jersey_image_bytes_with_hint(img_bytes, expected_digits)
        else:
            digits, conf = ocr_jersey_image_bytes(img_bytes)

        logger.info(f"[/send-img] OCR -> digits={digits!r}, conf={conf:.2f}, member={member_id}")

        # 임계값 검증(기본 0.5)
        conf_threshold = getattr(settings, "JERSEY_OCR_CONFIDENCE_THRESHOLD", 0.5)
        if conf < conf_threshold or not digits:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": "ocr_confidence_check",
                         "message": f"인식 정확도({conf:.2f})가 낮습니다(임계값 {conf_threshold:.2f}).",
                         "memberId": member_id},
            )

        # 캐시 저장 + 기대값 비교
        try:
            detected = int(digits)
        except Exception:
            detected = int(_digits_only(digits) or "0")
        if detected == 0:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"status": "error", "step": "cache",
                         "message": "유효하지 않은 등번호(0)입니다.", "memberId": member_id},
            )
        JERSEY_CACHE[member_id] = detected

        match: Optional[bool] = None
        if expected_digits:
            try:
                match = (int(expected_digits) == detected)
            except Exception:
                match = None

        # 성공
        dur_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": 200,
            "success": True,
            "backNumber": detected,
            "confidence": conf,
            "match": match,
            "elapsedMs": round(dur_ms, 1),
        }

    except Exception as e:
        logger.exception(f"[/send-img] failed at step={step}: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "step": step, "message": str(e)})
