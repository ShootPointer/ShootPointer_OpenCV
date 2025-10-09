# app/main.py
from __future__ import annotations

import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import upload, highlight, player, frames
from app.routers import presigned_upload, process

# ─────────────────────────────────────────────────────────────
# 로깅 초기화
# ─────────────────────────────────────────────────────────────
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="Basket Highlight AI", version="0.6.0")


# ─────────────────────────────────────────────────────────────
# 미들웨어: 요청/응답 로깅
#   - 큰 파일 업로드 환경이라 body 로깅은 기본 OFF
# ─────────────────────────────────────────────────────────────
class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        try:
            if settings.REQUEST_LOG_BODY:
                try:
                    body = await request.body()
                    logger.info(f"{request.method} {request.url.path} body={body[:300]}...")
                except Exception:
                    logger.info(f"{request.method} {request.url.path} (body read skipped)")
            else:
                logger.info(f"{request.method} {request.url.path}")

            response = await call_next(request)
            dur_ms = (time.time() - start) * 1000
            logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({dur_ms:.1f}ms)")
            return response
        except Exception as e:
            dur_ms = (time.time() - start) * 1000
            logger.exception(f"Unhandled error on {request.method} {request.url.path} ({dur_ms:.1f}ms): {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "internal_server_error", "detail": str(e)},
            )


# ─────────────────────────────────────────────────────────────
# 미들웨어: 업로드 용량 제한 (Content-Length 기반)
#   - .env의 MAX_UPLOAD_BYTES 초과 시 413 반환
# ─────────────────────────────────────────────────────────────
class UploadLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            if settings.MAX_UPLOAD_BYTES and settings.MAX_UPLOAD_BYTES > 0:
                cl = request.headers.get("content-length")
                if cl and cl.isdigit():
                    if int(cl) > settings.MAX_UPLOAD_BYTES:
                        logger.warning(f"413 payload too large: {cl} bytes > limit {settings.MAX_UPLOAD_BYTES}")
                        return JSONResponse(
                            status_code=413,
                            content={"error": "payload_too_large", "detail": "upload exceeds MAX_UPLOAD_BYTES"},
                        )
        except Exception as e:
            # 제한 체크 자체 실패 시엔 그냥 통과(로그만 남김)
            logger.warning(f"Upload limit check failed: {e}")

        return await call_next(request)


app.add_middleware(RequestLogMiddleware)
app.add_middleware(UploadLimitMiddleware)


# ─────────────────────────────────────────────────────────────
# 전역 예외 핸들러
# ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error {request.url.path}: {exc.errors()}")
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors()})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP error {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "detail": exc.detail})


# ─────────────────────────────────────────────────────────────
# 헬스체크
# ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# 라우터 등록
# ─────────────────────────────────────────────────────────────
app.include_router(upload.router)
app.include_router(highlight.router)
app.include_router(player.router)
app.include_router(frames.router)
app.include_router(presigned_upload.router)
app.include_router(process.router)