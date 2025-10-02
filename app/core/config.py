from pydantic import BaseModel
import os

class Settings(BaseModel):
    DEFAULT_PRE: float = float(os.getenv("DEFAULT_PRE", "5.0"))
    DEFAULT_POST: float = float(os.getenv("DEFAULT_POST", "5.0"))
    MAX_CLIPS: int = int(os.getenv("MAX_CLIPS", "10"))

    # 자동 후보(오디오)
    SILENCE_THRESHOLD_DB: float = float(os.getenv("SILENCE_THRESHOLD_DB", "-28"))
    SILENCE_MIN_DUR: float = float(os.getenv("SILENCE_MIN_DUR", "0.30"))
    AUTO_TOPK: int = int(os.getenv("AUTO_TOPK", "5"))

    # 등번호 감지 파라미터
    JERSEY_SAMPLE_FPS: float = float(os.getenv("JERSEY_SAMPLE_FPS", "1.5"))  # 초당 1.5프레임 샘플
    JERSEY_MIN_SEG_DUR: float = float(os.getenv("JERSEY_MIN_SEG_DUR", "1.2"))  # 최소 구간 길이(초)
    JERSEY_MERGE_GAP: float = float(os.getenv("JERSEY_MERGE_GAP", "2.0"))      # 인접 구간 병합 간격(초)
    JERSEY_TESSERACT_PSM: str = os.getenv("JERSEY_TESSERACT_PSM", "7")         # 한 줄 숫자 가정
    JERSEY_TESSERACT_OEM: str = os.getenv("JERSEY_TESSERACT_OEM", "3")         # LSTM 기본
    JERSEY_NUM_CONF: float = float(os.getenv("JERSEY_NUM_CONF", "0.5"))        # 숫자 신뢰도 하한(0~1)

settings = Settings()
