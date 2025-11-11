#app/core/redis_client.py
from typing import Optional
import logging

# redis-py의 비동기 버전(asyncio)을 임포트합니다.
import redis.asyncio as redis
from redis.asyncio import Redis

# 환경 설정 임포트 (이전 단계에서 수정한 설정 사용)
from app.core.config import settings

# 로거 설정
logger = logging.getLogger(__name__)

# Redis 연결 인스턴스를 저장할 전역 변수 (선택적 타입 힌트 사용)
_redis_client: Optional[Redis] = None


def get_redis_client() -> Redis:
    """
    애플리케이션 전반에서 사용할 Redis 클라이언트 인스턴스를 반환합니다.
    애플리케이션 시작 시 `init_redis`를 통해 반드시 초기화되어 있어야 합니다.
    """
    if _redis_client is None:
        # 초기화 전에 클라이언트를 호출하면 예외 발생
        raise ConnectionError("Redis client is not initialized. Call init_redis() first.")
    return _redis_client


async def init_redis():
    """
    Redis 서버에 비동기적으로 연결하고 전역 클라이언트 인스턴스를 초기화합니다.
    FastAPI의 'startup' 이벤트 핸들러에서 호출될 예정입니다.
    """
    global _redis_client
    
    # 이미 초기화된 경우 다시 연결하지 않습니다.
    if _redis_client is not None:
        logger.warning("Redis client is already initialized.")
        return

    logger.info("Attempting to connect to Redis at %s", settings.REDIS_URL)

    try:
        # settings.REDIS_URL을 사용하여 Redis 연결 인스턴스 생성
        # decode_responses=True로 설정하여 get/lpop 등의 명령 결과가 자동으로 문자열로 디코딩되도록 합니다.
        _redis_client = redis.from_url(
            url=settings.REDIS_URL, 
            encoding="utf-8", 
            decode_responses=True,
            socket_timeout=5,  # 연결 타임아웃
            health_check_interval=30 # 30초마다 헬스 체크
        )
        
        # 실제 연결 테스트 (ping)
        await _redis_client.ping()
        logger.info("Successfully connected to Redis!")

    except redis.exceptions.ConnectionError as e:
        logger.error("Failed to connect to Redis at %s: %s", settings.REDIS_URL, e)
        # 연결 실패 시 애플리케이션 시작을 중단하거나 적절히 처리할 수 있습니다.
        # 여기서는 ConnectionError를 다시 발생시켜 FastAPI가 시작에 실패하도록 합니다.
        raise ConnectionError(f"Could not connect to Redis: {e}") from e
    except Exception as e:
        logger.error("An unexpected error occurred during Redis initialization: %s", e)
        raise


async def close_redis():
    """
    Redis 클라이언트 연결을 안전하게 닫습니다.
    FastAPI의 'shutdown' 이벤트 핸들러에서 호출될 예정입니다.
    """
    global _redis_client
    if _redis_client:
        logger.info("Closing Redis connection.")
        # 비동기 연결 종료
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed.")
    else:
        logger.warning("Redis client was not active when shutdown was called.")