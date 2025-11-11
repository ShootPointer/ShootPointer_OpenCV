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
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")                       # DEBUG/INFO/WARNING/ERROR
    REQUEST_LOG_BODY: bool = _getenv_bool("REQUEST_LOG_BODY", False)       # 대용량 업로드 대비 False 권장
    # 서브프로세스(FFmpeg 등) 타임아웃(초)
    FFMPEG_TIMEOUT_SEC: int = int(os.getenv("FFMPEG_TIMEOUT_SEC", "120"))
    # 업로드 최대 크기(바이트). 0 또는 미설정이면 제한 없음 (기본 200MB)
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", "209715200"))

    # ── FFmpeg 하드웨어 디코드(옵션) ─────────────────────────
    # ✅ 두 이름 모두 지원: FF_HWACCEL(신규) 또는 FFMPEG_HWACCEL(기존)
    FF_HWACCEL: str = (os.getenv("FF_HWACCEL", "") or os.getenv("FFMPEG_HWACCEL", "")).strip().lower()
    FF_HWACCEL_DEVICE: str = (os.getenv("FF_HWACCEL_DEVICE", "") or os.getenv("FFMPEG_HWACCEL_DEVICE", "")).strip()
    # 컷 실패 시 재인코드 여부 / 스레드 수 / 재인코드 파라미터
    FFMPEG_CUT_FALLBACK_REENCODE: bool = _getenv_bool("FFMPEG_CUT_FALLBACK_REENCODE", False)
    FFMPEG_THREADS: int = int(os.getenv("FFMPEG_THREADS", "0") or "0")
    FFMPEG_REENC_VCODEC: str = os.getenv("FFMPEG_REENC_VCODEC", "libx264")
    FFMPEG_REENC_PRESET: str = os.getenv("FFMPEG_REENC_PRESET", "veryfast")
    FFMPEG_REENC_CRF: int = int(os.getenv("FFMPEG_REENC_CRF", "23") or "23")
    FFMPEG_REENC_ACODEC: str = os.getenv("FFMPEG_REENC_ACODEC", "aac")

    # ── OpenCV CUDA 사용 옵션(옵션) ──────────────────────────
    # OpenCV가 CUDA로 빌드되어 있고 GPU가 있으면 True로 활성화, 아니면 자동 폴백
    USE_CUDA: bool = _getenv_bool("USE_CUDA", True)
    # 너무 큰 이미지는 GPU로 보내기 전 선리사이즈(긴 변 기준)
    CUDA_RESIZE_MAX: int = int(os.getenv("CUDA_RESIZE_MAX", "1920"))

    # ── CORS(옵션) ───────────────────────────────────────────
    # 여러 출처는 콤마로 구분
    ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")

    # ── 백엔드 콜백/전달(저장 없이 즉시 업로드) ───────────────
    BACKEND_SECRET: str = os.getenv("BACKEND_SECRET", "")   # 백엔드-서버 공유 시크릿(없으면 검증 생략)
    CALLBACK_CONNECT_TIMEOUT: float = float(os.getenv("CALLBACK_CONNECT_TIMEOUT", "30"))
    CALLBACK_READ_TIMEOUT: float = float(os.getenv("CALLBACK_READ_TIMEOUT", "60"))
    CALLBACK_WRITE_TIMEOUT: float = float(os.getenv("CALLBACK_WRITE_TIMEOUT", "60"))
    
    # ⬇⬇⬇ 청크 업로드 및 암호화/복호화에 필요한 키 ⬇⬇⬇
    # crypto.py에서 사용할 AES-GCM 복호화 키
    AES_GCM_SECRET: str = os.getenv("AES_GCM_SECRET", "") 
    # 기존 PRE_SIGNED_SECRET 변수는 그대로 유지 (다른 곳에서 사용될 수 있으므로)
    PRE_SIGNED_SECRET: str = os.getenv("PRE_SIGNED_SECRET", "")
    # ⬆⬆⬆ 청크 업로드 및 암호화/복호화에 필요한 키 ⬆⬆⬆

    PRESIGNED_EXPIRES_MIN: int = int(os.getenv("PRESIGNED_EXPIRES_MIN", "30") or "30")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    UPLOAD_CHUNK_MAX_MB: int = int(os.getenv("UPLOAD_CHUNK_MAX_MB", "0") or "0")
    # ↓ 추가 (TEMP_ROOT가 없어서 크래시 나니, UPLOAD_DIR을 기본으로 물려주기)
    TEMP_ROOT: str = os.getenv("TEMP_ROOT", UPLOAD_DIR)

    # ── FastAPI 업로드/Redis 연동 (FastAPI 필수 변수 추가) ────────
    # FastAPI가 청크를 임시 저장할 폴더 (TEMP_ROOT 아래의 'chunks' 폴더)
    CHUNK_STORAGE_ROOT: str = os.getenv("CHUNK_STORAGE_ROOT", os.path.join(TEMP_ROOT, "chunks"))
    # FastAPI가 Spring 서버에 보고할 때 사용할 사용자 ID (임시)
    MEMBER_ID: str = os.getenv("MEMBER_ID", "default_user_id") 
    # FastAPI 컨테이너 외부에서 최종 병합 파일에 접근할 수 있는 기본 URL
    EXTERNAL_BASE_URL: str = os.getenv("EXTERNAL_BASE_URL", "http://fastapi-host:8000/files")
    # ─────────────────────────────────────────────────────────────

    # 기본 Plan registry 모듈 경로.
    PLAN_REGISTRY_PY: str = os.getenv("PLAN_REGISTRY_PY", "app.data.registry")

    # ─────────────────────────────────────────────────────────
    # ⬇⬇⬇ Redis 및 파일 저장 경로 설정 ⬇⬇⬇
    # ─────────────────────────────────────────────────────────
    # [1. REDIS_URL 설정]: 홈 서버의 Redis에 접속하기 위해 127.0.0.1 (호스트) 사용
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0") 
    
    # [2. AI Worker 전용 큐 (List) 이름]: FastAPI -> AI Worker 명령 전달용 (List 자료구조)
    REDIS_AI_JOB_QUEUE: str = os.getenv("REDIS_AI_JOB_QUEUE", "opencv-ai-job-queue")
    
    # [3. 채널 이름 재정의 및 정리] (Pub/Sub 채널)
    # Spring 서버로 업로드 진행률을 보낼 채널 
    REDIS_UPLOAD_PROGRESS_CHANNEL: str = os.getenv(
        "REDIS_UPLOAD_PROGRESS_CHANNEL", "opencv-progress-upload"
    )
    # Spring 서버로 하이라이트 진행률을 보낼 채널 
    REDIS_HIGHLIGHT_PROGRESS_CHANNEL: str = os.getenv(
        "REDIS_HIGHLIGHT_PROGRESS_CHANNEL", "opencv-progress-highlight"
    )

    PROGRESS_PUBLISH_INTERVAL_SEC: float = float(os.getenv("PROGRESS_PUBLISH_INTERVAL_SEC", "5") or "5")
    PROGRESS_INTERVAL_SEC: float = float(os.getenv("PROGRESS_INTERVAL_SEC", "5") or "5")

    # 파일 저장 루트 (컨테이너 내부 경로) - job_id 폴더가 이 아래에 생성됩니다.
    SAVE_ROOT: str = os.getenv("SAVE_ROOT", "/data/highlights")
    # 외부에서 접근할 정적 URL 베이스 - 예: https://your-domain/static/highlights
    STATIC_BASE_URL: str = os.getenv("STATIC_BASE_URL", "https://your-domain/static/highlights")
    # ─────────────────────────────────────────────────────────
    # ⬆⬆⬆ Redis 및 파일 저장 경로 설정 ⬆⬆⬆
    # ─────────────────────────────────────────────────────────
    
    # 결과를 Redis Key/Value로 저장할지 여부 (백엔드가 GET으로 즉시 조회 가능)
    PUBLISH_RESULT_AS_KV: bool = _getenv_bool("PUBLISH_RESULT_AS_KV", True)
    # 결과 Key 프리픽스: highlight-{jobId}
    RESULT_KEY_PREFIX: str = os.getenv("RESULT_KEY_PREFIX", "highlight-")
    # 결과 Key TTL(초). 0 이하면 만료 없음. 운영에선 1일(86400) 권장.
    RESULT_TTL_SECONDS: int = int(os.getenv("RESULT_TTL_SECONDS", "86400"))
    # 최소 발행 스코어(필요 시 사용)
    PUBLISH_THRESHOLD: int = int(os.getenv("PUBLISH_THRESHOLD", "0") or "0")
    
    # 하이라이트 완료 콜백(HTTP)
    RESULT_CALLBACK_BASE_URL: str = os.getenv("RESULT_CALLBACK_BASE_URL", "")
    RESULT_CALLBACK_PATH_TEMPLATE: str = os.getenv(
        "RESULT_CALLBACK_PATH_TEMPLATE", "{jobId}"
    )
    # ── (신규) FFmpeg 오버레이/출력 튜닝 ───────────────────────
    # Shorts 세로 해상도 (bh_edit에서 drawtext/scale용) — 코드 기본값은 1080x1920
    SHORTS_WIDTH: int = int(os.getenv("SHORTS_WIDTH", "1080"))
    SHORTS_HEIGHT: int = int(os.getenv("SHORTS_HEIGHT", "1920"))
    # Windows 등 폰트 경로 지정(미지정 시 시스템 기본 폰트 사용 시도)
    DRAW_FONTFILE: str = os.getenv("DRAW_FONTFILE", "")
    DRAW_FONTSIZE: int = int(os.getenv("DRAW_FONTSIZE", "48"))
    # FFmpeg 인코딩 품질/속도
    FFMPEG_PRESET: str = os.getenv("FFMPEG_PRESET", "veryfast")
    FFMPEG_CRF: int = int(os.getenv("FFMPEG_CRF", "20"))
    FFMPEG_ABR: str = os.getenv("FFMPEG_ABR", "128k")

    # ── (신규) 배치/탐지 튜닝(코드 수정 없이 환경으로 미세조정) ─────────
    # 다운스케일 기준(긴 변 / SCALE_BASE). 기본 960 → 값 낮추면 속도↑, 높이면 정확도↑
    SCALE_BASE: int = int(os.getenv("SCALE_BASE", "960"))
    # 공 HSV 범위(오렌지 계열). "Hmin,Smin,Vmin" / "Hmax,Smax,Vmax"
    BALL_HSV_MIN: str = os.getenv("BALL_HSV_MIN", "5,60,60")
    BALL_HSV_MAX: str = os.getenv("BALL_HSV_MAX", "25,255,255")
    # HoughLinesP 파라미터(코트 라인)
    HOUGH_MIN_LINE_RATIO: float = float(os.getenv("HOUGH_MIN_LINE_RATIO", "0.25"))   # minLineLength = ratio * width
    HOUGH_MAX_GAP: int = int(os.getenv("HOUGH_MAX_GAP", "20"))
    HOUGH_THRESH: int = int(os.getenv("HOUGH_THRESH", "120"))
    # Homography(RANSAC) 허용 오차
    HOMO_RANSAC_REPROJ_THRESH: float = float(os.getenv("HOMO_RANSAC_REPROJ_THRESH", "3.0"))

    # ── Presigned 업로드 완료 후 AI 데모 자동 실행 ─────────────
    AUTO_RUN_AI_DEMO: bool = _getenv_bool("AUTO_RUN_AI_DEMO", True)
    AUTO_AI_DEMO_OVERLAY_TAG: str = os.getenv("AUTO_AI_DEMO_OVERLAY_TAG", "AI-Selector")
    AUTO_AI_DEMO_MERGE_OUTPUT: bool = _getenv_bool("AUTO_AI_DEMO_MERGE_OUTPUT", True)

settings = Settings()