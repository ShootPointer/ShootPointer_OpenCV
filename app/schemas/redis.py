from enum import Enum

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────
# Redis 통신 상태 (Upload / Merge / AI Processing)
# ─────────────────────────────────────────────────────────────

class UploadStatus(str, Enum):
    """업로드 및 병합 상태"""
    # 초기 청크 업로드 중 (Chunk 엔드포인트에서 주기적으로 발행)
    UPLOADING = "UPLOADING"
    # 청크가 모두 모여 병합/처리 대기 중 (Complete 엔드포인트 응답 후)
    PROCESSING = "PROCESSING"
    # 원본 파일 저장 및 정리 완료 (Redis JSON 보고로 대체됨)
    UPLOAD_COMPLETE = "UPLOAD_COMPLETE" 
    # AI 작업 시작 대기 중 (AI Queue에 넣었음을 알림)
    AI_START_PENDING = "AI_START_PENDING" 
    # 오류 발생
    ERROR = "ERROR"

class HighlightStatus(str, Enum):
    """AI 하이라이트 처리 상태 (AI Worker가 발행할 상태)"""
    # 하이라이트 생성 중
    PROCESSING = "PROCESSING"
    # 최종 하이라이트 영상 생성 및 저장 완료
    COMPLETE = "COMPLETE"
    # 오류 발생
    ERROR = "ERROR"

# ─────────────────────────────────────────────────────────────
# Redis Pub/Sub 메시지 모델
# ─────────────────────────────────────────────────────────────

class ProgressMessage(BaseModel):
    """
    Redis Pub/Sub을 통해 전송되는 진행률 메시지 표준 형식
    """
    jobId: str = Field(..., description="작업 고유 ID")
    # 상태는 업로드 또는 하이라이트 중 하나
    status: UploadStatus | HighlightStatus = Field(..., description="현재 작업 상태")
    message: str = Field(default="", description="사용자에게 보여줄 상세 메시지")
    current: int = Field(default=0, description="현재 진행도 카운트")
    total: int = Field(default=1, description="전체 진행도 카운트")
    resultUrl: str = Field(default="", description="최종 결과 파일 접근 URL (완료 시)")


class AITaskPayload(BaseModel):
    """
    AI Worker Queue (Redis List)에 푸시될 작업 요청 페이로드
    """
    jobId: str = Field(..., description="작업 고유 ID")
    memberId: str = Field(..., description="요청 사용자 ID")
    originalFilePath: str = Field(..., description="AI Worker가 접근해야 하는 원본 파일의 임시 저장 경로/URL")
    # TODO: AI 작업의 종류, 파라미터 등 필요한 메타데이터 추가