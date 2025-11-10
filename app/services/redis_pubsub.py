import logging
from typing import Optional
from redis import Redis
from app.core.config import settings
from app.schemas.redis import ProgressMessage, UploadStatus

logger = logging.getLogger(__name__)

# Redis 클라이언트 싱글톤 인스턴스
_redis_client: Optional[Redis] = None

def init_redis_client() -> Redis:
    """
    Redis 연결을 초기화하고 싱글톤 인스턴스를 반환합니다. 
    (FastAPI의 startup 이벤트에서 호출되도록 설계)
    """
    global _redis_client
    
    if _redis_client is None:
        try:
            # 설정 파일에서 REDIS_URL을 사용하여 연결. (호스트 IP:6379로 설정 필요)
            _redis_client = Redis.from_url(
                settings.REDIS_URL, 
                decode_responses=False # 바이너리 데이터 처리를 위해 False 유지
            )
            _redis_client.ping() # 연결 테스트
            logger.info("Successfully connected to Redis at %s", settings.REDIS_URL)
            return _redis_client
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {settings.REDIS_URL}: {e}")
            _redis_client = None
            # 연결 실패 시 런타임 오류 발생. 다른 서비스들이 안전하게 Optional로 처리할 수 있도록 함.
            raise RuntimeError(f"Cannot connect to Redis: {e}")
            
    return _redis_client

def get_redis_client() -> Optional[Redis]:
    """
    Redis 클라이언트 인스턴스를 가져옵니다. 초기화되지 않았으면 None 반환.
    """
    return _redis_client

def publish_progress(
    job_id: str, 
    status: UploadStatus, 
    message: str = "", 
    current: int = 0, 
    total: int = 1
) -> None:
    """
    업로드/처리 진행률 메시지를 Redis Pub/Sub 채널에 발행합니다.
    """
    client = get_redis_client()
    if client is None:
        logger.warning(f"Redis client is not available for publishing progress for job {job_id}.")
        return

    progress_msg = ProgressMessage(
        jobId=job_id,
        status=status,
        message=message,
        current=current,
        total=total
    )
    
    channel = f"{settings.REDIS_UPLOAD_PROGRESS_CHANNEL}:{job_id}"
    try:
        # Pydantic 모델을 JSON 문자열로 변환하여 발행
        client.publish(channel, progress_msg.model_dump_json())
        logger.debug(f"Published progress for {job_id}: {status.value} (Current: {current}/{total})")
    except Exception as e:
        logger.error(f"Failed to publish progress to Redis for job {job_id}: {e}")