# app/services/callback.py
from __future__ import annotations
from tempfile import SpooledTemporaryFile
from typing import Optional, Dict, Any

import json
import httpx

from app.core.config import settings

async def post_zip_to_callback(
    callback_url: str,
    zip_spooled: SpooledTemporaryFile,
    meta: Optional[Dict[str, Any]] = None,
    bearer_or_token: Optional[str] = None,
):
    """
    콜백 URL로 ZIP을 multipart/form-data (file 필드)로 업로드.
    - meta: 추가 폼필드(문자열/숫자/리스트/딕셔너리 자동 문자열화)
    - bearer_or_token: Authorization 헤더에 Bearer 혹은 커스텀 토큰으로 전달하고 싶을 때
    """
    # spooled 파일 포인터 보장
    try:
        zip_spooled.seek(0)
    except Exception:
        pass

    headers = {}
    # 서버 간 인증이 필요하면 헤더로 전달(백엔드와 합의 필요)
    if bearer_or_token:
        headers["Authorization"] = f"Bearer {bearer_or_token}"

    timeout = httpx.Timeout(
        connect=settings.CALLBACK_CONNECT_TIMEOUT,
        read=settings.CALLBACK_READ_TIMEOUT,
        write=settings.CALLBACK_WRITE_TIMEOUT,
        pool=None,
    )

    # 폼 데이터 구성
    data = {}
    if meta:
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                data[k] = json.dumps(v, ensure_ascii=False)
            else:
                data[k] = str(v)

    files = {"file": ("player_highlights.zip", zip_spooled, "application/zip")}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        resp = await client.post(callback_url, data=data, files=files)
        resp.raise_for_status()
        return resp.status_code
