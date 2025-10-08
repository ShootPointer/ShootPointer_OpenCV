# app/schemas/player.py
from typing import List, Tuple, Optional
from app.core.schema import CamelModel

class OcrResponse(CamelModel):
    status: str = "ok"
    detected_number: int
    confidence: float
    expected_number: Optional[int] = None
    match: Optional[bool] = None
    member_id: Optional[str] = None

class ErrorResponse(CamelModel):
    status: str = "error"
    message: str
    member_id: Optional[str] = None

class SegmentsResponse(CamelModel):
    segments: List[Tuple[float, float]]
    jersey_number: int

class HighlightAcceptResponse(CamelModel):
    status: str
    clips: int
    member_id: Optional[str] = None

class JobStatus(CamelModel):
    status: str
    # 필요 시 더 추가
