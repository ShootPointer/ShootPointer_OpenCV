# app/core/config.py
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional
import os

def _getenv_bool(key: str, default: bool = False) -> bool:
    """환경변수 불리언 파싱: '1','true','yes','on' → True (대소문자 무시)"""
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

class Settings(BaseModel):
    # ── 기본 컷 길이/최대 클립 수 ──────────────────────────────
    DEFAULT_PRE: float = float(os.getenv("DEFAULT_PRE", "5.0"))
    DEFAULT_POST: float = float(os.getenv("DEFAULT_POST", "5.0"))
    MAX_CLIPS: int = int(os.getenv("MAX_CLIPS", "10"))

    # ── 자동 후보(오디오 기반) ────────────────────────────────
    SILENCE_THRESHOLD_DB: float = float(os.getenv("SILENCE_THRESHOLD_DB", "-28"))
    SILENCE_MIN_DUR: float = float(os.getenv("SILENCE_MIN_DUR", "0.30"))
    AUTO_TOPK: int = int(os.getenv("AUTO_TOPK", "5"))

    # ── 등번호 감지 파라미터(OpenCV + Tesseract) ─────────────
    JERSEY_SAMPLE_FPS: float = float(os.getenv("JERSEY_SAMPLE_FPS", "1.5"))   # 초당 샘플 프레임
    JERSEY_MIN_SEG_DUR: float = float(os.getenv("JERSEY_MIN_SEG_DUR", "1.2")) # 최소 구간 길이(초)
    JERSEY_MERGE_GAP: float = float(os.getenv("JERSEY_MERGE_GAP", "2.0"))     # 인접 구간 병합 간격(초)
    JERSEY_TESSERACT_PSM: str = os.getenv("JERSEY_TESSERACT_PSM", "7")        # 한 줄 숫자 가정
    JERSEY_TESSERACT_OEM: str = os.getenv("JERSEY_TESSERACT_OEM", "3")        # LSTM 기본
    JERSEY_NUM_CONF: float = float(os.getenv("JERSEY_NUM_CONF", "0.5"))       # 숫자 신뢰도 하한(0~1)

    # ── 실행/로깅/안정화 ─────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")                           # DEBUG/INFO/WARNING/ERROR
    REQUEST_LOG_BODY: bool = _getenv_bool("REQUEST_LOG_BODY", False)          # 대용량 업로드 고려 시 기본 False

    # 서브프로세스(FFmpeg 등) 타임아웃(초)
    FFMPEG_TIMEOUT_SEC: int = int(os.getenv("FFMPEG_TIMEOUT_SEC", "120"))
    OCR_TIMEOUT_SEC: int = int(os.getenv("OCR_TIMEOUT_SEC", "30"))

    # 업로드 최대 크기(바이트). 0 또는 미설정이면 제한 없음
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", "209715200"))   # 기본 200MB

settings = Settings()

