# app/core/config.py
from __future__ import annotations

import os
from typing import List

from pydantic import Field

# BaseSettings 호환성: v2(pydantic-settings) 우선, 없으면 v1로 폴백
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # Pydantic v2 권장
    _HAS_V2_SETTINGS = True
except Exception:  # pragma: no cover
    from pydantic import BaseSettings  # Pydantic v1 폴백
    SettingsConfigDict = None
    _HAS_V2_SETTINGS = False


def _getenv_bool(key: str, default: bool = False) -> bool:
    """환경변수 불리언 파싱: '1','true','yes','y','on' → True (대소문자 무시)"""
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _getenv_floats(key: str, default: str) -> List[float]:
    """콤마 구분 float 리스트 파싱"""
    raw = os.getenv(key, default)
    out: List[float] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            pass
    return out


def _getenv_ints(key: str, default: str) -> List[int]:
    """콤마 구분 int 리스트 파싱"""
    raw = os.getenv(key, default)
    out: List[int] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            pass
    return out


class Settings(BaseSettings):
    # v2(pydantic-settings)면 model_config로 .env 지정, 아니면 v1 Config로 폴백
    if _HAS_V2_SETTINGS:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"

    # ── 파일/정적 URL ─────────────────────────────────────────────
    SAVE_ROOT: str = Field("/data/highlights", env="SAVE_ROOT")
    STATIC_BASE_URL: str = Field("http://localhost:8000/static/highlights", env="STATIC_BASE_URL")

    # ── 등번호/OCR 기본 파라미터 ─────────────────────────────────
    JERSEY_NUM_CONF: float = Field(0.5, env="JERSEY_NUM_CONF")

    # ── Tesseract 세부 튜닝 ─────────────────────────────────────
    JERSEY_TESSERACT_PSM: str = Field("7", env="JERSEY_TESSERACT_PSM")
    JERSEY_TESSERACT_OEM: str = Field("3", env="JERSEY_TESSERACT_OEM")
    OCR_PSMS: List[int] = _getenv_ints("OCR_PSMS", os.getenv("JERSEY_TESSERACT_PSM", "7"))
    OCR_SCALES: List[float] = _getenv_floats("OCR_SCALES", "1.5,2.0,3.0")
    OCR_TRY_INVERT: bool = _getenv_bool("OCR_TRY_INVERT", True)
    DEBUG_OCR: bool = _getenv_bool("DEBUG_OCR", True)
    OCR_TIMEOUT_SEC: int = Field(30, env="OCR_TIMEOUT_SEC")
    OCR_MAX_COMBOS: int = Field(120, env="OCR_MAX_COMBOS")
    OCR_MAX_SEC: float = Field(8.0, env="OCR_MAX_SEC")
    JERSEY_OCR_CONFIDENCE_THRESHOLD: float = Field(0.5, env="JERSEY_OCR_CONFIDENCE_THRESHOLD")

    # ── 실행/로깅/안정화 ───────────────────────────────────────
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    REQUEST_LOG_BODY: bool = Field(False, env="REQUEST_LOG_BODY")
    FFMPEG_TIMEOUT_SEC: int = Field(120, env="FFMPEG_TIMEOUT_SEC")
    MAX_UPLOAD_BYTES: int = Field(209_715_200, env="MAX_UPLOAD_BYTES")
    MAX_CLIPS: int = Field(10, env="MAX_CLIPS")

    # ── CORS ───────────────────────────────────────────────────
    ALLOW_ORIGINS: str = Field("*", env="ALLOW_ORIGINS")

    # ── 백엔드 콜백/보안 ───────────────────────────────────────
    BACKEND_SECRET: str = Field("", env="BACKEND_SECRET")
    CALLBACK_CONNECT_TIMEOUT: float = Field(30.0, env="CALLBACK_CONNECT_TIMEOUT")
    CALLBACK_READ_TIMEOUT: float = Field(60.0, env="CALLBACK_READ_TIMEOUT")
    CALLBACK_WRITE_TIMEOUT: float = Field(60.0, env="CALLBACK_WRITE_TIMEOUT")

    # ── 업로드/청크/암복호화 키 ─────────────────────────────────
    AES_GCM_SECRET: str = Field("", env="AES_GCM_SECRET")
    PRE_SIGNED_SECRET: str = Field("", env="PRE_SIGNED_SECRET")

    # ── 업로드 디렉터리/임시경로/외부접근 URL ─────────────────────
    PRESIGNED_EXPIRES_MIN: int = Field(30, env="PRESIGNED_EXPIRES_MIN")
    UPLOAD_DIR: str = Field("/tmp/uploads", env="UPLOAD_DIR")
    UPLOAD_CHUNK_MAX_MB: int = Field(0, env="UPLOAD_CHUNK_MAX_MB")
    TEMP_ROOT: str = Field(default_factory=lambda: os.getenv("TEMP_ROOT", os.getenv("UPLOAD_DIR", "/tmp/uploads")))
    CHUNK_STORAGE_ROOT: str = Field(
        default_factory=lambda: os.getenv(
            "CHUNK_STORAGE_ROOT",
            os.path.join(os.getenv("TEMP_ROOT", os.getenv("UPLOAD_DIR", "/tmp/uploads")), "chunks"),
        )
    )
    EXTERNAL_BASE_URL: str = Field("http://fastapi-host:8000/files", env="EXTERNAL_BASE_URL")

    # ── Redis ─────────────────────────────────────────────────
    MEMBER_ID: str = Field("default_user_id", env="MEMBER_ID")
    REDIS_URL: str = Field("redis://127.0.0.1:6379/0", env="REDIS_URL")

    # 1) FastAPI -> AI Worker 작업큐(List)
    REDIS_QUEUE_NAME: str = Field("opencv-ai-job-queue", env="REDIS_AI_JOB_QUEUE")
    # 2) 업로드 진행률 Pub/Sub
    REDIS_UPLOAD_PROGRESS_CHANNEL: str = Field("opencv-progress-upload", env="REDIS_UPLOAD_PROGRESS_CHANNEL")
    # 3) 하이라이트 진행률 Pub/Sub
    REDIS_HIGHLIGHT_PROGRESS_CHANNEL: str = Field("opencv-progress-highlight", env="REDIS_HIGHLIGHT_PROGRESS_CHANNEL")

    PROGRESS_PUBLISH_INTERVAL_SEC: float = Field(5.0, env="PROGRESS_PUBLISH_INTERVAL_SEC")
    PROGRESS_INTERVAL_SEC: float = Field(5.0, env="PROGRESS_INTERVAL_SEC")

    # ── 결과 보고(KV/TTL/콜백) ─────────────────────────────────
    PUBLISH_RESULT_AS_KV: bool = Field(True, env="PUBLISH_RESULT_AS_KV")
    RESULT_KEY_PREFIX: str = Field("highlight-", env="RESULT_KEY_PREFIX")
    RESULT_TTL_SECONDS: int = Field(86_400, env="RESULT_TTL_SECONDS")
    RESULT_CALLBACK_BASE_URL: str = Field("", env="RESULT_CALLBACK_BASE_URL")
    RESULT_CALLBACK_PATH_TEMPLATE: str = Field("{jobId}", env="RESULT_CALLBACK_PATH_TEMPLATE")


settings = Settings()
