# app/core/config.py
from __future__ import annotations

import os
from typing import List

from pydantic import Field

# BaseSettings 호환성: v2(pydantic-settings) 우선, 없으면 v1로 폴백
try:
    from pydantic_settings import BaseSettings  # Pydantic v2 권장
except Exception:  # pragma: no cover
    from pydantic import BaseSettings  # Pydantic v1 폴백


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
    # ── 파일/정적 URL ─────────────────────────────────────────────
    # 컨테이너 내부 저장 루트 (최종 병합 파일 저장 경로)
    SAVE_ROOT: str = Field("/data/highlights", env="SAVE_ROOT")
    # 정적 파일을 외부에서 접근할 기본 URL (예: Nginx/Static 라우팅)
    STATIC_BASE_URL: str = Field("http://localhost:8000/static/highlights", env="STATIC_BASE_URL")

    # ── 등번호/OCR 기본 파라미터 ─────────────────────────────────
    # 숫자 신뢰도 하한(0~1)
    JERSEY_NUM_CONF: float = Field(0.5, env="JERSEY_NUM_CONF")

    # ── Tesseract 세부 튜닝 ─────────────────────────────────────
    # 단일 PSM(레거시 호환). 예: "7" (single line)
    JERSEY_TESSERACT_PSM: str = Field("7", env="JERSEY_TESSERACT_PSM")
    # OCR 엔진 모드
    JERSEY_TESSERACT_OEM: str = Field("3", env="JERSEY_TESSERACT_OEM")
    # 여러 PSM을 콤마로 지정해 순차 시도. (지정 없으면 위 단일 PSM 사용)
    # 환경변수 문자열을 직접 파싱 (예: "10,8,7")
    OCR_PSMS: List[int] = _getenv_ints("OCR_PSMS", os.getenv("JERSEY_TESSERACT_PSM", "7"))
    # 스케일 업 비율(예: "1.5,2.0,3.0")
    OCR_SCALES: List[float] = _getenv_floats("OCR_SCALES", "1.5,2.0,3.0")
    # 반전 시도 여부 (흰 글자/검은 유니폼 등)
    OCR_TRY_INVERT: bool = _getenv_bool("OCR_TRY_INVERT", True)
    # 디버그 로깅
    DEBUG_OCR: bool = _getenv_bool("DEBUG_OCR", True)
    # Tesseract 호출 타임아웃(초)
    OCR_TIMEOUT_SEC: int = Field(30, env="OCR_TIMEOUT_SEC")
    # 조합/시간 예산
    OCR_MAX_COMBOS: int = Field(120, env="OCR_MAX_COMBOS")
    OCR_MAX_SEC: float = Field(8.0, env="OCR_MAX_SEC")
    JERSEY_OCR_CONFIDENCE_THRESHOLD: float = Field(0.5, env="JERSEY_OCR_CONFIDENCE_THRESHOLD")

    # ── 실행/로깅/안정화 ───────────────────────────────────────
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")  # DEBUG/INFO/WARNING/ERROR
    REQUEST_LOG_BODY: bool = Field(False, env="REQUEST_LOG_BODY")
    FFMPEG_TIMEOUT_SEC: int = Field(120, env="FFMPEG_TIMEOUT_SEC")
    # 업로드 최대 크기(바이트). 기본 200MB
    MAX_UPLOAD_BYTES: int = Field(209_715_200, env="MAX_UPLOAD_BYTES")
    # 하이라이트 최대 클립 수
    MAX_CLIPS: int = Field(10, env="MAX_CLIPS")

    # ── CORS ───────────────────────────────────────────────────
    ALLOW_ORIGINS: str = Field("*", env="ALLOW_ORIGINS")

    # ── 백엔드 콜백/보안 ───────────────────────────────────────
    BACKEND_SECRET: str = Field("", env="BACKEND_SECRET")
    CALLBACK_CONNECT_TIMEOUT: float = Field(30.0, env="CALLBACK_CONNECT_TIMEOUT")
    CALLBACK_READ_TIMEOUT: float = Field(60.0, env="CALLBACK_READ_TIMEOUT")
    CALLBACK_WRITE_TIMEOUT: float = Field(60.0, env="CALLBACK_WRITE_TIMEOUT")

    # ── 업로드/청크/암복호화 키 ─────────────────────────────────
    AES_GCM_SECRET: str = Field(
        "36b82ad3a534b95842e1bf29e538352481e4e265815042b12f4660852d89701a",
        env="AES_GCM_SECRET"
    )
    PRE_SIGNED_SECRET: str = Field("", env="PRE_SIGNED_SECRET")

    # ── 업로드 디렉터리/임시경로/외부접근 URL ─────────────────────
    PRESIGNED_EXPIRES_MIN: int = Field(30, env="PRESIGNED_EXPIRES_MIN")
    UPLOAD_DIR: str = Field("/tmp/uploads", env="UPLOAD_DIR")
    UPLOAD_CHUNK_MAX_MB: int = Field(0, env="UPLOAD_CHUNK_MAX_MB")  # 0이면 제한 없음(서버 정책으로 제어)
    TEMP_ROOT: str = Field(default_factory=lambda: os.getenv("TEMP_ROOT", os.getenv("UPLOAD_DIR", "/tmp/uploads")))
    CHUNK_STORAGE_ROOT: str = Field(
        default_factory=lambda: os.getenv(
            "CHUNK_STORAGE_ROOT",
            os.path.join(os.getenv("TEMP_ROOT", os.getenv("UPLOAD_DIR", "/tmp/uploads")), "chunks"),
        )
    )
    # FastAPI 컨테이너 외부에서 최종 병합 파일 접근 기본 URL (필요 시 Nginx 경유)
    EXTERNAL_BASE_URL: str = Field("http://fastapi-host:8000/files", env="EXTERNAL_BASE_URL")

    # ── Redis ─────────────────────────────────────────────────
    MEMBER_ID: str = Field("default_user_id", env="MEMBER_ID")
    REDIS_URL: str = Field("redis://:rlaehdus00!!@host.docker.internal:6379/0", env="REDIS_URL")

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

    class Config:
        # v2(pydantic-settings)에서도 인식됨
        env_file = ".env"
        # 알 수 없는 환경변수 필드는 무시
        extra = "ignore"


settings = Settings()
