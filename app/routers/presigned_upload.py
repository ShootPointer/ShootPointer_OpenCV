# app/routers/presigned_upload.py
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.crypto import hmac_compare

router = APIRouter(prefix="/api", tags=["presigned-upload"])
logger = logging.getLogger(__name__)

# 업로드 보관 폴더
UPLOAD_DIR = Path("/tmp/presigned_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _now_epoch() -> int:
    return int(time.time())


def _limit_check(content_length: Optional[int]) -> None:
    lim = int(getattr(settings, "MAX_UPLOAD_BYTES", 0) or 0)
    if lim and content_length and content_length > lim:
        # 여기서는 예외를 던지지 않고 상위에서 메시지 구성
        raise ValueError(f"payload too large: {content_length} > {lim}")


def _client_ip(req: Request) -> str:
    # 프록시 환경 고려(X-Forwarded-For)
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "-"


@router.put("/upload", summary="Upload via Presigned URL (HMAC-SHA256)")
async def presigned_upload(
    request: Request,
    expires: int = Query(..., description="만료 epoch seconds"),
    memberId: str = Query(..., description="멤버 식별자"),
    signature: str = Query(..., description="HMAC-SHA256 서명"),
):
    """
    Spring이 생성한 Presigned URL을 검증하고 업로드 본문을 저장한다.

    - message = "expires:memberId"
    - signature = HMAC-SHA256(secretKey, message) (hex/base64/base64url 허용)
    - expires 유효성(현재시간보다 과거면 만료) 검사
    - Content-Length 제한 검사(설정 시)
    - 성공시: 200 { status:200, success:true, uploadId, path, ... }
    - 실패시: 4xx/5xx { status:"error", step, message, requestId, memberId }
    """
    req_id = uuid.uuid4().hex[:8]  # 짧은 상관ID
    step = "start"

    # 요청 메타 로깅(트래픽 파악용)
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "-")
    cl_header = request.headers.get("content-length")
    logger.info(
        f"[presigned] [{req_id}] -> PUT /api/upload from {ip} "
        f"(ua={ua}) memberId={memberId!r} expires={expires} content-length={cl_header}"
    )

    try:
        # 0) 서버 준비 상태 확인
        step = "check_secret"
        if not settings.BACKEND_SECRET:
            msg = "server missing BACKEND_SECRET"
            logger.error(f"[presigned] [{req_id}] {step} -> {msg}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "step": step, "message": msg, "requestId": req_id, "memberId": memberId},
            )

        # 1) 만료 검사
        step = "expires_check"
        now = _now_epoch()
        if expires < now:
            msg = f"presigned url expired (now={now}, expires={expires})"
            logger.warning(f"[presigned] [{req_id}] {step} -> {msg}")
            return JSONResponse(
                status_code=401,
                content={"status": "error", "step": step, "message": msg, "requestId": req_id, "memberId": memberId},
            )

        # 2) 시그니처 검사
        step = "signature_check"
        message = f"{expires}:{memberId}"
        if not hmac_compare(settings.BACKEND_SECRET, message, signature):
            msg = "invalid signature"
            logger.warning(
                f"[presigned] [{req_id}] {step} -> {msg} "
                f"(message='{message}', sig_prefix='{signature[:8]}...')"
            )
            return JSONResponse(
                status_code=401,
                content={"status": "error", "step": step, "message": msg, "requestId": req_id, "memberId": memberId},
            )

        # 3) 용량 제한(있다면) 선점검
        step = "length_check"
        try:
            content_length = int(cl_header) if (cl_header and cl_header.isdigit()) else None
            _limit_check(content_length)
        except ValueError as ve:
            msg = str(ve)
            logger.warning(f"[presigned] [{req_id}] {step} -> {msg}")
            return JSONResponse(
                status_code=413,
                content={"status": "error", "step": step, "message": msg, "requestId": req_id, "memberId": memberId},
            )

        # 4) 본문 스트리밍 저장
        step = "save_body"
        upload_id = uuid.uuid4().hex
        out_path = UPLOAD_DIR / f"{memberId}_{upload_id}.mp4"

        size = 0
        lim = int(getattr(settings, "MAX_UPLOAD_BYTES", 0) or 0)
        with out_path.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                f.write(chunk)
                size += len(chunk)
                # 실시간 제한 초과 차단
                if lim and size > lim:
                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    msg = f"payload too large while streaming: {size} > {lim}"
                    logger.warning(f"[presigned] [{req_id}] {step} -> {msg}")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "status": "error",
                            "step": step,
                            "message": msg,
                            "requestId": req_id,
                            "memberId": memberId,
                        },
                    )

        # 5) 완료 로깅 + 응답
        step = "done"
        logger.info(
            f"[presigned] [{req_id}] upload ok member={memberId} size={size} file={out_path.name}"
        )
        return {
            "status": 200,
            "success": True,
            "uploadId": upload_id,
            "path": str(out_path),
            "memberId": memberId,
            "size": size,
            "expires": expires,
            "requestId": req_id,
        }

    except Exception as e:
        # 예기치 못한 오류
        logger.exception(f"[presigned] [{req_id}] failed at step={step}: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "step": step,
                "message": str(e),
                "requestId": req_id,
                "memberId": memberId,
            },
        )
