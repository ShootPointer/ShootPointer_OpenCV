# app/services/pubsub.py
from __future__ import annotations
import json
import time
import logging
from typing import Optional, Dict, Any, List
import redis.asyncio as aioredis
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

class RedisPublisher:
    def __init__(self, url: str):
        self._url = url
        self._client: Optional[aioredis.Redis] = None

    async def _get(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def publish(self, channel: str, payload: Dict[str, Any]) -> None:
        """
        PUB/SUB 채널로 메시지 발행
        """
        try:
            r = await self._get()
            await r.publish(channel, json.dumps(payload, ensure_ascii=False))
            logger.info(f"[redis.pub] {channel} -> {payload.get('type') or payload.get('data', {}).get('type')}")
        except Exception as e:
            logger.exception(f"[redis.pub] publish failed: {e}")

    async def set(self, key: str, value: Dict[str, Any], ttl_seconds: int = 0) -> None:
        """
        Key/Value 저장 (선택 TTL)
        """
        try:
            r = await self._get()
            await r.set(key, json.dumps(value, ensure_ascii=False))
            if ttl_seconds and ttl_seconds > 0:
                await r.expire(key, ttl_seconds)
            logger.info(f"[redis.kv] SET {key} (ttl={ttl_seconds})")
        except Exception as e:
            logger.exception(f"[redis.kv] set failed: {e}")

publisher = RedisPublisher(settings.REDIS_URL)

def _timestamp_ms() -> int:
    return int(time.time() * 1000)

def _base_payload(
    *,
    status: int,
    success: bool,
    data: Dict[str, Any],
    message: str = "",
) -> Dict[str, Any]:
    return {
        "status": int(status),
        "success": bool(success),
        "data": data,
        "message": message,
        "timestamp": _timestamp_ms(),
    }

# ─────────────────────────────────────────────────────────────
# 진행률 메시지(신규 포멧)
# ─────────────────────────────────────────────────────────────
def progress_payload(
    *,
    type_: str,
    member_id: str,
    job_id: str,
    progress: Optional[float] = None,    
    message: Optional[str] = None,
    total_bytes: Optional[int] = None,
    received_bytes: Optional[int] = None,
    stage: Optional[str] = None,
    current_clip: Optional[int] = None,
    total_clips: Optional[int] = None,    
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "type": type_,
        "memberId": member_id,
        "jobId": job_id,
    }
    if progress is not None:
        data["progress"] = round(float(progress), 4)
    if total_bytes is not None:
        data["totalBytes"] = int(total_bytes)
    if received_bytes is not None:
        data["receivedBytes"] = int(received_bytes)
    if stage:
        data["stage"] = stage
    if current_clip is not None:
        data["currentClip"] = int(current_clip)
    if total_clips is not None:
        data["totalClips"] = int(total_clips)

    return _base_payload(
        status=200,
        success=True,
        data=data,
        message=message or "",
    )

def _default_progress_channel(type_: str) -> str:
    if type_ in {"UPLOADING", "UPLOAD_COMPLETE"}:
        return settings.REDIS_UPLOAD_PROGRESS_CHANNEL
    return settings.REDIS_HIGHLIGHT_PROGRESS_CHANNEL

async def publish_progress(
    member_id: str,
    job_id: str,
    progress: Optional[float],
    message: str = "",
    *,
    type_: str = "PROCESSING",
    stage: Optional[str] = None,
    total_bytes: Optional[int] = None,
    received_bytes: Optional[int] = None,
    current_clip: Optional[int] = None,
    total_clips: Optional[int] = None,
    channel: Optional[str] = None,
) -> None:
    """요구 포맷으로 진행률 발행(신규 채널 + 기존 호환)."""

    payload = progress_payload(
        type_=type_,
        member_id=member_id,
        job_id=job_id,
        progress=progress,
        message=message,
        total_bytes=total_bytes,
        received_bytes=received_bytes,
        stage=stage,
        current_clip=current_clip,
        total_clips=total_clips,
    )

    channels = [channel or _default_progress_channel(type_)]
    # 레거시 호환: job별 채널 유지
    channels.append(f"progress:{member_id}:{job_id}")

    for ch in channels:
        await publisher.publish(ch, payload)

# ─────────────────────────────────────────────────────────────
# 결과 메시지(KV + PUB/SUB)
# 백엔드 요구 포맷(Value):
# {
#   "status": 200,
#   "success": true,
#   "data": {
#     "type": "COMPLETE",
#     "memberId": "abc123",
#     "urls": [...],
#     "count": 3
#   },
#   "message": "highlight clips ready"
# }
# ─────────────────────────────────────────────────────────────
def _kv_result_payload(
    *,
    status: int,
    success: bool,
    type_: str,
    member_id: str,
    job_id: str,
    urls: List[str],
    message: str
) -> Dict[str, Any]:
    return {
        "status": int(status),
        "success": bool(success),
        "data": {
            "type": type_,
            "memberId": member_id,
            "jobId": job_id,
            "urls": urls,
            "count": len(urls),
        },
        "message": message,
        "timestamp": _timestamp_ms(),
    }

async def _save_result_kv(job_id: str, payload: Dict[str, Any]) -> None:
    """
    highlight-{jobId} 키에 Value(JSON) 저장 (+ TTL)
    """
    key = f"{settings.RESULT_KEY_PREFIX}{job_id}"
    await publisher.set(key, payload, ttl_seconds=settings.RESULT_TTL_SECONDS)

async def _post_callback(job_id: str, payload: Dict[str, Any]) -> None:
    base = settings.RESULT_CALLBACK_BASE_URL.strip()
    if not base:
        return

    path = settings.RESULT_CALLBACK_PATH_TEMPLATE.format(jobId=job_id, job_id=job_id)
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"

    timeout = httpx.Timeout(
        connect=settings.CALLBACK_CONNECT_TIMEOUT,
        read=settings.CALLBACK_READ_TIMEOUT,
        write=settings.CALLBACK_WRITE_TIMEOUT,
        pool=None,
    )

    headers: Dict[str, str] = {}
    if settings.BACKEND_SECRET:
        headers["x-backend-secret"] = settings.BACKEND_SECRET

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            logger.info(f"[callback] POST {url} -> {resp.status_code}")
    except Exception as e:
        logger.exception(f"[callback] failed POST {url}: {e}")

async def publish_result(member_id: str, job_id: str, urls: List[str], final: bool = True) -> None:
    """
    - result:{memberId}:{jobId} 채널로 결과 발행
    - highlight-{jobId} 키로 KV 저장(옵션)
    """
    type_ = "COMPLETE" if final else "PARTIAL_COMPLETE"
    msg = "highlight clips ready" if final else f"{len(urls)} clips ready (partial)"
    payload = _kv_result_payload(
        status=200,
        success=True,
        type_=type_,
        member_id=member_id,
        job_id=job_id,
        urls=urls,
        message=msg,
    )

    # PUB/SUB (구독자 실시간 반영용)
    await publisher.publish(settings.REDIS_HIGHLIGHT_PROGRESS_CHANNEL, payload)
    await publisher.publish(f"result:{member_id}:{job_id}", payload)

    # KV 저장 (백엔드가 키로 즉시 조회)
    if settings.PUBLISH_RESULT_AS_KV:
        await _save_result_kv(job_id, payload)

    # 백엔드 HTTP 콜백(요청 시)
    await _post_callback(job_id, payload)

async def publish_error(member_id: str, job_id: str, message: str, status: int = 500) -> None:
    """
    오류도 동일한 스키마 유지 (data.type = ERROR, urls=[], count=0)
    """
    payload = {
        "status": int(status),
        "success": False,
        "data": {
            "type": "ERROR",
            "memberId": member_id,
            "jobId": job_id,
            "urls": [],
            "count": 0,
        },
        "message": message,
        "timestamp": _timestamp_ms(),        
    }

    await publisher.publish(settings.REDIS_HIGHLIGHT_PROGRESS_CHANNEL, payload)
    await publisher.publish(f"result:{member_id}:{job_id}", payload)

    if settings.PUBLISH_RESULT_AS_KV:
        await _save_result_kv(job_id, payload)

    await _post_callback(job_id, payload)

# ─────────────────────────────────────────────────────────────
# (호환 목적) 기존 호출부가 사용 중일 수 있는 헬퍼
# complete_payload: 결과 값을 생성만(발행/저장은 안 함)
# ─────────────────────────────────────────────────────────────
def complete_payload(member_id: str, job_id: str, urls: List[str], message: str = "highlight clips ready") -> Dict[str, Any]:
    """
    호환용: 결과 JSON만 생성 (KV 스키마에 맞춰 반환)
    """
    return _kv_result_payload(
        status=200,
        success=True,
        type_="COMPLETE",
        member_id=member_id,
        job_id=job_id,
        urls=urls,
        message=message,
    )
