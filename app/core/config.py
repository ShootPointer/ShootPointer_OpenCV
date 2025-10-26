# app/core/config.py
from __future__ import annotations
import os
from pydantic import BaseModel

def _getenv_bool(key: str, default: bool = False) -> bool:
    """환경변수 불리언 파싱: '1','true','yes','y','on' → True (대소문자 무시)"""
    val = os.getenv(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _getenv_floats(key: str, default: str) -> list[float]:
    """콤마 구분 float 리스트 파싱"""
    raw = os.getenv(key, default)
    out: list[float] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(float(s))
        except Exception:
            pass
    return out

def _getenv_ints(key: str, default: str) -> list[int]:
    """콤마 구분 int 리스트 파싱"""
    raw = os.getenv(key, default)
    out: list[int] = []
    for s in raw.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            pass
    return out

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
    JERSEY_NUM_CONF: float = float(os.getenv("JERSEY_NUM_CONF", "0.5"))       # 숫자 신뢰도 하한(0~1)

    # ── OCR / Tesseract ───────────────────────────────────────
    # 단일 PSM(레거시 호환). 예: "7" (한 줄)
    JERSEY_TESSERACT_PSM: str = os.getenv("JERSEY_TESSERACT_PSM", "7")
    # OEM 그대로 유지
    JERSEY_TESSERACT_OEM: str = os.getenv("JERSEY_TESSERACT_OEM", "3")
    # 여러 PSM을 콤마로 지정해 순차 시도. 지정 없으면 위 단일 PSM으로 fallback.
    OCR_PSMS: list[int] = _getenv_ints("OCR_PSMS", JERSEY_TESSERACT_PSM)
    # 스케일 업 비율들(콤마 구분). 숫자가 작거나 흐릴 때 확대 시도.
    OCR_SCALES: list[float] = _getenv_floats("OCR_SCALES", "1.5,2.0,3.0")
    # 반전(흰 글자/검은 유니폼 등) 시도
    OCR_TRY_INVERT: bool = _getenv_bool("OCR_TRY_INVERT", True)
    # 디버그(실패 원인/후보 로그)
    DEBUG_OCR: bool = _getenv_bool("DEBUG_OCR", True)
    # Tesseract 호출 타임아웃(초)
    OCR_TIMEOUT_SEC: int = int(os.getenv("OCR_TIMEOUT_SEC", "30"))
    # OCR 조합/시간 예산
    OCR_MAX_COMBOS: int = int(os.getenv("OCR_MAX_COMBOS", "120"))
    OCR_MAX_SEC: float = float(os.getenv("OCR_MAX_SEC", "8.0"))

    # ── 실행/로깅/안정화 ─────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")                           # DEBUG/INFO/WARNING/ERROR
    REQUEST_LOG_BODY: bool = _getenv_bool("REQUEST_LOG_BODY", False)          # 대용량 업로드 고려 시 기본 False
    # 서브프로세스(FFmpeg 등) 타임아웃(초)
    FFMPEG_TIMEOUT_SEC: int = int(os.getenv("FFMPEG_TIMEOUT_SEC", "120"))
    # 업로드 최대 크기(바이트). 0 또는 미설정이면 제한 없음 (기본 200MB)
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", "209715200"))

    # ── FFmpeg 하드웨어 디코드(옵션) ─────────────────────────
    # ✅ 두 이름 모두 지원: FF_HWACCEL(신규) 또는 FFMPEG_HWACCEL(기존)
    FF_HWACCEL: str = (os.getenv("FF_HWACCEL", "") or os.getenv("FFMPEG_HWACCEL", "")).strip().lower()
    FF_HWACCEL_DEVICE: str = (os.getenv("FF_HWACCEL_DEVICE", "") or os.getenv("FFMPEG_HWACCEL_DEVICE", "")).strip()

    # ── OpenCV CUDA 사용 옵션(옵션) ──────────────────────────
    # OpenCV가 CUDA로 빌드되어 있고 GPU가 있으면 True로 활성화, 아니면 자동 폴백
    USE_CUDA: bool = _getenv_bool("USE_CUDA", True)
    # 너무 큰 이미지는 GPU로 보내기 전 선리사이즈(긴 변 기준)
    CUDA_RESIZE_MAX: int = int(os.getenv("CUDA_RESIZE_MAX", "1920"))

    # ── CORS(옵션) ───────────────────────────────────────────
    # 여러 출처는 콤마로 구분
    ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")

    # ── 백엔드 콜백/전달(저장 없이 즉시 업로드) ───────────────
    BACKEND_SECRET: str = os.getenv("BACKEND_SECRET", "")                     # 백엔드-서버 공유 시크릿(없으면 검증 생략)
    CALLBACK_CONNECT_TIMEOUT: float = float(os.getenv("CALLBACK_CONNECT_TIMEOUT", "30"))
    CALLBACK_READ_TIMEOUT: float = float(os.getenv("CALLBACK_READ_TIMEOUT", "60"))
    CALLBACK_WRITE_TIMEOUT: float = float(os.getenv("CALLBACK_WRITE_TIMEOUT", "60"))

    # ─────────────────────────────────────────────────────────
    # ⬇⬇⬇ 이번 단계에 필요한 추가 설정만 최소로 보강 ⬇⬇⬇
    # ─────────────────────────────────────────────────────────
    # 파일 저장 루트 (컨테이너 내부 경로) - 예: /data/highlights
    SAVE_ROOT: str = os.getenv("SAVE_ROOT", "/data/highlights")
    # 외부에서 접근할 정적 URL 베이스 - 예: https://your-domain/static/highlights
    STATIC_BASE_URL: str = os.getenv("STATIC_BASE_URL", "https://your-domain/static/highlights")
    # Redis 연결 URL - 예: redis://redis:6379/0
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # 결과를 Redis Key/Value로 저장할지 여부 (백엔드가 GET으로 즉시 조회 가능)
    PUBLISH_RESULT_AS_KV: bool = _getenv_bool("PUBLISH_RESULT_AS_KV", True)
    # 결과 Key 프리픽스: highlight-{jobId}
    RESULT_KEY_PREFIX: str = os.getenv("RESULT_KEY_PREFIX", "highlight-")
    # 결과 Key TTL(초). 0 이하면 만료 없음. 운영에선 1일(86400) 권장.
    RESULT_TTL_SECONDS: int = int(os.getenv("RESULT_TTL_SECONDS", "86400"))

settings = Settings()
