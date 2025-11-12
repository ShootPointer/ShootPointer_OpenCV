#app/schemas/presigned.py
from datetime import datetime
from pydantic import BaseModel, field_validator
from typing import ClassVar

class PresignedChunkMetadata(BaseModel):
    # 이 클래스는 복호화된 평문 문자열을 파싱한 결과를 담습니다.
    expires: datetime
    member_id: str
    job_id: str
    file_name: str
    
    # 클래스 변수: 복호화된 평문 형식
    RAW_STRING_FORMAT: ClassVar[str] = "expires:memberId:jobId:fileName"

    @classmethod
    def from_raw_string(cls, raw_string: str) -> 'PresignedChunkMetadata':
        """콜론으로 구분된 평문 문자열에서 메타데이터 객체를 생성합니다."""
        try:
            parts = raw_string.split(":")
            if len(parts) != 4:
                raise ValueError("Raw string must contain exactly 4 colon-separated parts.")
            
            # parts[0]: expires (ISO 8601 형식 문자열로 가정)
            expires_dt = datetime.fromisoformat(parts[0]) 
            
            # parts[1]: memberId
            member_id = parts[1]
            
            # parts[2]: jobId
            job_id = parts[2]
            
            # parts[3]: fileName
            file_name = parts[3]
            
            return cls(
                expires=expires_dt,
                member_id=member_id,
                job_id=job_id,
                file_name=file_name
            )
        except Exception as e:
            # 파싱 실패 시 명확한 에러 메시지를 반환
            raise ValueError(f"Failed to parse presigned metadata string: {e}")

# Complete 엔드포인트도 동일한 토큰 구조를 사용한다고 가정하고 재정의
PresignedCompleteMetadata = PresignedChunkMetadata