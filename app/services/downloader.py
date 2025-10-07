# app/services/downloader.py
from __future__ import annotations
from pathlib import Path
import tempfile
import httpx

async def download_to_temp(url: str, suffix: str = ".mp4") -> Path:
    """
    원본을 영구 저장하지 않고, 처리용으로만 임시파일에 받아서 반환.
    호출자가 사용 후 삭제해야 한다.
    """
    tmp = Path(tempfile.mkstemp(prefix="vid_", suffix=suffix)[1])
    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
    return tmp
def sync(url: str, suffix: str = ".mp4"):
    import anyio
    return anyio.run(lambda: download_to_temp(url, suffix=suffix))