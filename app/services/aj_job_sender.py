#app/services/ai_job_sender.py
import asyncio
import json
import os
import logging
from typing import Optional

from pydantic import BaseModel, Field
from redis.asyncio import Redis, ConnectionError as RedisConnectionError
from app.core.config import settings

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
# 1. 설정 및 데이터 스키마 정의
# ─────────────────────────────────────────────────────────────

class Settings(BaseModel):
    """Redis 접속 정보 및 Queue 이름을 정의합니다."""
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://host.docker.internal:6379/0") 
    # AI 워커가 모니터링할 Queue 이름
    AI_QUEUE_NAME: str = os.getenv("REDIS_AI_JOB_QUEUE", "opencv-ai-job-queue") 
    
settings = Settings()

class AITaskPayload(BaseModel):
    """AI Worker에게 전달할 작업의 데이터 구조 (계약)"""
    jobId: str = Field(..., description="작업 고유 ID")
    memberId: str = Field(..., description="요청 사용자 ID")
    originalFilePath: str = Field(..., description="원본 파일의 서버 내부 경로")

# ─────────────────────────────────────────────────────────────
# 2. Redis 클라이언트 관리
# ─────────────────────────────────────────────────────────────

_redis_client: Optional[Redis] = None

def get_redis_client() -> Redis:
    """Redis 클라이언트를 싱글톤으로 반환합니다."""
    global _redis_client
    if _redis_client is None:
        try:
            # decode_responses=True는 Redis에서 가져오는 데이터(바이트)를 문자열로 자동 디코딩합니다.
            _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            logger.info("Redis client successfully initialized.")
        except Exception:
            raise RedisConnectionError("Redis connection failed during initialization.")
    return _redis_client

# ─────────────────────────────────────────────────────────────
# 3. 작업 전송 (Enqueue) 로직
# ─────────────────────────────────────────────────────────────

async def send_ai_job(job_id: str, member_id: str, file_path: str) -> bool:
    """
    AI 워커에게 처리 요청을 보내기 위해 Redis Queue에 작업을 추가합니다.
    """
    try:
        redis = get_redis_client()
        
        # Pydantic 모델을 사용하여 페이로드 생성 및 유효성 검사
        payload = AITaskPayload(
            jobId=job_id,
            memberId=member_id,
            originalFilePath=file_path
        )
        
        # JSON 문자열로 직렬화
        json_payload = payload.model_dump_json()
        
        # Redis List의 오른쪽(Right)에 페이로드 추가 (RPUSH)
        # AI 워커(소비자)는 BLPOP으로 List의 왼쪽(Left)에서 작업을 가져갑니다.
        await redis.rpush(settings.AI_QUEUE_NAME, json_payload)
        
        logger.info(f"Successfully enqueued job {job_id} to queue: {settings.AI_QUEUE_NAME}")
        logger.debug(f"Payload: {json_payload}")
        
        return True
        
    except RedisConnectionError as e:
        logger.error(f"Failed to connect to Redis while sending job: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while enqueuing job: {e}")
        return False

# ─────────────────────────────────────────────────────────────
# 4. 사용 예시 (테스트 목적)
# ─────────────────────────────────────────────────────────────

async def main_test():
    """모듈 테스트를 위한 비동기 실행 함수"""
    test_job_id = "test-job-a1b2c3d4"
    test_member_id = "user-123"
    test_file_path = "/uploads/raw/2025/video_segment_456.mp4"
    
    print("-" * 50)
    print(f"Attempting to send test job: {test_job_id}")
    
    success = await send_ai_job(test_job_id, test_member_id, test_file_path)
    
    if success:
        print("\n✅ Job Enqueue Success.")
        print("   -> Redis Queue에 작업이 성공적으로 추가되었습니다.")
        # Redis에서 방금 추가한 작업을 확인해보는 테스트 로직 (선택 사항)
        # redis = get_redis_client()
        # item = await redis.lpop(settings.AI_QUEUE_NAME)
        # print(f"   -> 큐에서 확인된 항목 (테스트): {item}")
    else:
        print("\n❌ Job Enqueue Failed. Check Redis connection.")
    
    # 연결 종료
    await get_redis_client().close()

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트 로직이 실행됩니다.
    # 참고: Redis 서버가 127.0.0.1:6379에서 실행 중이어야 합니다.
    asyncio.run(main_test())