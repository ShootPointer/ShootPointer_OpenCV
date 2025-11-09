from __future__ import annotations

import logging
import time
import os
import shutil
import subprocess
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
# ⬇️ 수정: upload를 제거하고 모든 라우터를 한 줄로 정리
from app.routers import highlight, player, frames, presigned_upload, process, ai_demo
# ⬆️ 수정

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
# 유틸: 실행파일 버전 확인/존재 확인
# ─────────────────────────────────────────────────────────────
def _which(cmd: str) -> str | None:
    return shutil.which(cmd)

def _run_out(args: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=5).decode("utf-8", errors="ignore").strip()
        return True, out
    except Exception as e:
        return False, str(e)

def _check_ff_binaries() -> dict:
    ok_ffmpeg = _which("ffmpeg") is not None
    ok_ffprobe = _which("ffprobe") is not None
    ffmpeg_ver = _run_out(["ffmpeg", "-version"])[1] if ok_ffmpeg else "not found"
    ffprobe_ver = _run_out(["ffprobe", "-version"])[1] if ok_ffprobe else "not found"
    return {
        "ffmpeg_found": ok_ffmpeg,
        "ffprobe_found": ok_ffprobe,
        "ffmpeg_version": ffmpeg_ver.splitlines()[0] if isinstance(ffmpeg_ver, str) else ffmpeg_ver,
        "ffprobe_version": ffprobe_ver.splitlines()[0] if isinstance(ffprobe_ver, str) else ffprobe_ver,
    }

def _check_font() -> dict:
    font = os.getenv("DRAW_FONTFILE") or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    exists = os.path.exists(font)
    return {"font_path": font, "exists": exists}

def _check_save_root() -> dict:
    root = settings.SAVE_ROOT
    ok_path = True
    ok_write = True
    msg = "ok"
    try:
        os.makedirs(root, exist_ok=True)
        test_path = os.path.join(root, ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        ok_write = False
        msg = str(e)
    if not os.path.isdir(root):
        ok_path = False
        msg = f"not a directory: {root}"
    return {"save_root": root, "dir_ok": ok_path, "writable": ok_write, "detail": msg}

def _check_static_mount() -> dict:
    parsed = urlparse(settings.STATIC_BASE_URL)
    return {
        "STATIC_BASE_URL": settings.STATIC_BASE_URL,
        "mount_path": parsed.path or "/static/highlights"
    }

# ─────────────────────────────────────────────────────────────
# 정적 파일 서빙 마운트
#   - STATIC_BASE_URL의 path를 FastAPI에 매핑
#   - 예: STATIC_BASE_URL = "http://tkv0011.ddns.net:8000/static/highlights"
#     -> path "/static/highlights" 를 SAVE_ROOT에 연결
# ─────────────────────────────────────────────────────────────
parsed = urlparse(settings.STATIC_BASE_URL)
static_path = parsed.path or "/static/highlights"

app.mount(
    static_path,
    StaticFiles(directory=settings.SAVE_ROOT, html=False),
    name="static-highlights",
)

# ─────────────────────────────────────────────────────────────
# 스타트업 셀프체크: 환경 차이로 인한 실패 예방
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup_selfcheck():
    env_summary = {
        "LOG_LEVEL": settings.LOG_LEVEL,
        "SAVE_ROOT": settings.SAVE_ROOT,
        "STATIC_BASE_URL": settings.STATIC_BASE_URL,
        "ALLOW_ORIGINS": settings.ALLOW_ORIGINS,
        "MAX_UPLOAD_BYTES": settings.MAX_UPLOAD_BYTES,
        "FFMPEG_TIMEOUT_SEC": settings.FFMPEG_TIMEOUT_SEC,
    }
    logger.info(f"[startup] ENV: {env_summary}")

    ff = _check_ff_binaries()
    font = _check_font()
    sr = _check_save_root()
    st = _check_static_mount()

    logger.info(f"[startup] ffmpeg/ffprobe: {ff}")
    logger.info(f"[startup] font: {font}")
    logger.info(f"[startup] static mount: {st}")
    if not sr["dir_ok"] or not sr["writable"]:
        logger.error(f"[startup] SAVE_ROOT not writable: {sr}")
    else:
        logger.info(f"[startup] SAVE_ROOT OK: {sr}")

# (선택) 수동 점검용 엔드포인트
@app.get("/debug/selfcheck")
def debug_selfcheck():
    return {
        "env": {
            "LOG_LEVEL": settings.LOG_LEVEL,
            "SAVE_ROOT": settings.SAVE_ROOT,
            "STATIC_BASE_URL": settings.STATIC_BASE_URL,
            "ALLOW_ORIGINS": settings.ALLOW_ORIGINS,
            "MAX_UPLOAD_BYTES": settings.MAX_UPLOAD_BYTES,
            "FFMPEG_TIMEOUT_SEC": settings.FFMPEG_TIMEOUT_SEC,
        },
        "ffmpeg": _check_ff_binaries(),
        "font": _check_font(),
        "save_root": _check_save_root(),
        "static_mount": _check_static_mount(),
        "status": "ok",
    }

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
# app.include_router(upload.router) # ⬅️ 제거됨
app.include_router(highlight.router)
app.include_router(player.router)
app.include_router(frames.router)
app.include_router(presigned_upload.router)
app.include_router(process.router)

# ✅ 데모 하이라이트(자동식별 → 지정 구간 컷 → 메타데이터/집계) 라우터 등록
app.include_router(ai_demo.router, prefix="")