from __future__ import annotations

import logging
import time
import os
import shutil
import subprocess
from urllib.parse import urlparse
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware


# --- Redis imports ---
from app.core.redis_client import init_redis, close_redis, get_redis_client

from app.core.config import settings
from app.core.logging import setup_logging
# 필요한 라우터만 포함: player(OCR), presigned_upload(청크 업로드), process(병합 시작)
from app.routers import player, presigned_upload, process

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
# 미들웨어: 요청/응답 로깅 (대용량 업로드라 body 로깅 기본 OFF)
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
# ─────────────────────────────────────────────────────────────
class UploadLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            if settings.MAX_UPLOAD_BYTES and settings.MAX_UPLOAD_BYTES > 0:
                cl = request.headers.get("content-length")
                if cl and cl.isdigit():
                    if int(cl) > settings.MAX_UPLOAD_BYTES:
                        logger.warning(
                            f"413 payload too large: {cl} bytes > limit {settings.MAX_UPLOAD_BYTES}"
                        )
                        return JSONResponse(
                            status_code=413,
                            content={"error": "payload_too_large", "detail": "upload exceeds MAX_UPLOAD_BYTES"},
                        )
        except Exception as e:
            logger.warning(f"Upload limit check failed: {e}")

        return await call_next(request)


app.add_middleware(RequestLogMiddleware)
app.add_middleware(UploadLimitMiddleware)


# ─────────────────────────────────────────────────────────────
# 유틸: 실행파일 버전/존재 확인 (FFmpeg, FFprobe)
# ─────────────────────────────────────────────────────────────
def _which(cmd: str) -> Optional[str]:
    """시스템 경로에서 실행 파일 존재 여부를 확인합니다."""
    return shutil.which(cmd)


def _run_out(args: list[str]) -> tuple[bool, str]:
    """외부 명령어를 실행하고 결과를 반환합니다."""
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=5).decode(
            "utf-8", errors="ignore"
        ).strip()
        return True, out
    except Exception as e:
        return False, str(e)


def _check_ff_binaries() -> dict:
    """ffmpeg, ffprobe 존재 여부 및 버전 정보를 확인합니다."""
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


def _check_save_root() -> dict:
    """최종 저장 경로의 존재 여부 및 쓰기 가능 여부를 확인합니다."""
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
    """정적 파일 서빙 URL 및 마운트 경로를 확인합니다."""
    parsed = urlparse(settings.STATIC_BASE_URL)
    return {
        "STATIC_BASE_URL": settings.STATIC_BASE_URL,
        "mount_path": parsed.path or "/static/highlights",
    }


# ─────────────────────────────────────────────────────────────
# 정적 파일 서빙 마운트
# ─────────────────────────────────────────────────────────────
parsed = urlparse(settings.STATIC_BASE_URL)
static_path = parsed.path or "/static/highlights"

app.mount(
    static_path,
    StaticFiles(directory=settings.SAVE_ROOT, html=False),
    name="static-highlights",
)
app.mount(
    "/highlight",
    StaticFiles(directory="/home/videos/highlight", html=False),
    name="highlight-files",
)

# ─────────────────────────────────────────────────────────────
# 스타트업: 셀프체크 및 Redis 연결 초기화
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup_selfcheck():
    # Redis 연결
    logger.info("[startup] Initializing Redis connection...")
    try:
        await init_redis()
        logger.info("[startup] Redis connection established successfully.")
    except Exception as e:
        logger.error(f"[startup] FATAL ERROR: Failed to connect to Redis. Functionality limited. Error: {e}")

    # 환경 요약 로그
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
    sr = _check_save_root()
    st = _check_static_mount()

    logger.info(f"[startup] ffmpeg/ffprobe: {ff}")
    logger.info(f"[startup] static mount: {st}")
    if not sr["dir_ok"] or not sr["writable"]:
        logger.error(f"[startup] SAVE_ROOT not writable: {sr}")
    else:
        logger.info(f"[startup] SAVE_ROOT OK: {sr}")


# ─────────────────────────────────────────────────────────────
# 셧다운: Redis 연결 종료
# ─────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def _shutdown_cleanup():
    logger.info("[shutdown] Closing Redis connection...")
    await close_redis()
    logger.info("[shutdown] Redis connection closed.")


# ─────────────────────────────────────────────────────────────
# 디버그 셀프체크
# ─────────────────────────────────────────────────────────────
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

    # Pydantic Validation Errors를 JSON 직렬화 가능하게 정리
    errors_list: list[dict[str, Any]] = []
    for error in exc.errors():
        errors_list.append({
            "loc": [str(loc) for loc in error.get("loc", [])],
            "msg": str(error.get("msg", "Validation failed")),
            "type": str(error.get("type", "unknown_type")),
            "input": str(error.get("input")),
        })

    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": errors_list})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(f"HTTP error {request.url.path}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": "http_error", "detail": exc.detail})


# ─────────────────────────────────────────────────────────────
# 헬스체크 (Redis 상태 포함)
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    redis_status = "ERROR - Redis client not initialized"
    try:
        redis_client = get_redis_client()
        if redis_client:
            # Redis가 비동기 클라이언트임을 가정하고 ping 사용
            await redis_client.ping()
            redis_status = "OK"
        else:
            redis_status = "ERROR - Redis client not initialized"
    except Exception as e:
        redis_status = f"ERROR - Redis unreachable ({e})"

    return {"status": "ok", "redis_connection": redis_status}


# ─────────────────────────────────────────────────────────────
# 라우터 등록 (게이트웨이/업로드/OCR 관련 기능만 유지)
# ─────────────────────────────────────────────────────────────
app.include_router(player.router)        # 등번호 OCR (단일 이미지)
app.include_router(presigned_upload.router) # 청크 업로드
app.include_router(process.router)       # 병합 시작
# 불필요한 라우터 제거: highlight, frames, ai_demo