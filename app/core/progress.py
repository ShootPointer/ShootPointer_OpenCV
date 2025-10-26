# app/core/progress.py
from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, Optional

from redis import asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

class PROGRESS_TYPE:
    UPLOAD_START = "UPLOAD_START"
    UPLOAD_PROGRESS = "UPLOAD_PROGRESS"
    PROCESSING_START = "PROCESSING_START"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"

class ProgressBus:
    _pool: Optional[aioredis.Redis] = None

    @classmethod
    async def _conn(cls) -> aioredis.Redis:
        if cls._pool is None:
            cls._pool = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        return cls._pool

    @classmethod
    async def publish_kv(cls, job_id: str, value: Dict[str, Any]) -> None:
        """
        백엔드 요구 포맷: key=jobId, value=JSON
        메시지 예) {"key":"<jobId>","value":{...}}
        """
        payload = {"key": job_id, "value": value}
        try:
            r = await cls._conn()
            await r.publish(settings.REDIS_UPLOAD_CHANNEL, json.dumps(payload))
        except Exception as e:
            logger.warning(f"[redis.publish] failed jobId={job_id}: {e}")
