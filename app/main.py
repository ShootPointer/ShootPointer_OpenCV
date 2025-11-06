# app/main.py
from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import upload, highlight, player, frames
from app.routers import presigned_upload, process
from app.routers import ai_demo  # ✅ 추가: 데모 하이라이트 라우터

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
# CORS (settings.ALLOW_ORIGINS 사용)
# ─────────────────────────────────────────────────────────────
allow_origins = [o.strip() for o in settings.ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            # 제한 체크 실패 시 통과(로그만)
            logger.warning(f"Upload limit check failed: {e}")

        return await call_next(request)

app.add_middleware(RequestLogMiddleware)
app.add_middleware(UploadLimitMiddleware)

# ─────────────────────────────────────────────────────────────
# 정적 파일 서빙 마운트
#   - STATIC_BASE_URL의 path를 FastAPI에 매핑
#   - 예: STATIC_BASE_URL = "http://tkv0011.ddns.net:8000/static/highlights"
#       -> path "/static/highlights" 를 SAVE_ROOT에 연결
# ─────────────────────────────────────────────────────────────
parsed = urlparse(settings.STATIC_BASE_URL)
static_path = parsed.path or "/static/highlights"

app.mount(
    static_path,
    StaticFiles(directory=settings.SAVE_ROOT, html=False),
    name="static-highlights",
)

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

# ✅ 데모 하이라이트(자동식별 → 지정 구간 컷 → 메타데이터/집계) 라우터 등록
app.include_router(ai_demo.router, prefix="")
