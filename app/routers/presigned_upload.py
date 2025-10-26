# app/routers/presigned_upload.py
from __future__ import annotations

import json
import time
import uuid
import tempfile
from pathlib import Path
from typing import Optional

import anyio
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.crypto import hmac_sha256_hex
from app.core.progress import ProgressBus, PROGRESS_TYPE
from app.core.logging import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# 업로드 바이트를 임시 파일로 받아둔 뒤, 처리가 끝나면 즉시 삭제한다.
# 영구 저장/DB 업로드는 하지 않는다.

def _now_ms() -> int:
    return int(time.time() * 1000)

def _ok(payload: dict) -> dict:
    # 백엔드 요구 응답 포맷: {status:200, suceess:true, type:UPLOAD_SUCCESS}
    # 오타("suceess")가 명시라면 그대로 맞춰줌. success도 같이 내려줌(하위 호환)
    return {
        "status": 200,
        "suceess": True,   # <- 요청대로 철자 그대로
        "success": True,   # <- 하위 호환
        **payload,
    }

@router.put("/upload", summary="Pre-signed 업로드 수신 (서명 검증 + 진행률 PUB)")
async def upload_video(
    request: Request,
    expires: int = Query(..., description="만료 시각(ms since epoch)"),
    memberId: str = Query(..., description="멤버 식별자"),
    jobId: str = Query(..., description="업로드/처리 작업 ID"),
    signature: str = Query(..., description="HMAC-SHA256 HEX(signature)"),
    fileName: str = Query(..., description="원본 파일명"),
):
    step = "parse_query"
    t0 = time.perf_counter()
    tmp_path: Optional[Path] = None
    recv_bytes = 0
    total_bytes = int(request.headers.get("content-length") or 0)

    try:
        # 1) 형식/만료/서명 검증
        # message = f"{expiresMs}:{memberId}:{jobId}:{fileName}"
        step = "verify_signature"
        if expires <= 0:
            return JSONResponse(status_code=422, content={"status": "error", "step": step, "message": "expires(ms) required"})
        now = _now_ms()
        if now > expires:
            return JSONResponse(status_code=401, content={"status": "error", "step": step, "message": "url expired"})

        msg = f"{expires}:{memberId}:{jobId}:{fileName}"
        expected_sig = hmac_sha256_hex(settings.BACKEND_SECRET, msg)
        if not signature or signature.lower() != expected_sig.lower():
            logger.warning(f"[upload] signature mismatch jobId={jobId} expected={expected_sig} got={signature}")
            return JSONResponse(status_code=401, content={"status": "error", "step": step, "message": "invalid signature"})

        # 2) UPLOAD_START publish
        step = "publish_start"
        await ProgressBus.publish_kv(
            job_id=jobId,
            value={
                "type": PROGRESS_TYPE.UPLOAD_START,
                "progress": 0.0,
                "totalBytes": total_bytes or None,
                "receivedBytes": 0,
                "timestampMs": _now_ms(),
            },
        )

        # 3) 스트리밍 수신(임시 파일). 5초 단위 진행률 PUB
        step = "stream_and_save"
        # 임시 파일 생성 (자동 삭제 위해 나중에 unlink)
        fd, path_str = tempfile.mkstemp(prefix="upload_", suffix=Path(fileName).suffix or ".mp4")
        tmp_path = Path(path_str)

        last_pub = time.perf_counter()
        interval = float(settings.PROGRESS_INTERVAL_SEC or 5.0)

        async with await anyio.open_file(tmp_path, "wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                recv_bytes += len(chunk)
                await f.write(chunk)

                now_t = time.perf_counter()
                if now_t - last_pub >= interval:
                    last_pub = now_t
                    prog = (recv_bytes / total_bytes) if total_bytes > 0 else None
                    await ProgressBus.publish_kv(
                        job_id=jobId,
                        value={
                            "type": PROGRESS_TYPE.UPLOAD_PROGRESS,
                            "progress": prog,
                            "totalBytes": total_bytes or None,
                            "receivedBytes": recv_bytes,
                            "timestampMs": _now_ms(),
                        },
                    )

        # 끝난 직후 한 번 더 진행률 PUB(100% 보정)
        await ProgressBus.publish_kv(
            job_id=jobId,
            value={
                "type": PROGRESS_TYPE.UPLOAD_PROGRESS,
                "progress": 1.0 if total_bytes > 0 else None,
                "totalBytes": total_bytes or None,
                "receivedBytes": recv_bytes,
                "timestampMs": _now_ms(),
            },
        )

        # 4) 업로드 성공 PUB(스펙엔 COMPLETE만 있지만, 업로드 쪽 완료 알림도 필요하면 여기서 한 번 더)
        step = "publish_completed"
        # 필요 시 Spring에서 업로드 완료를 감지 후 PROCESSING_START를 트리거
        # 여기서는 업로드 완료를 알리진 않고, HTTP 응답에서 성공만 돌려줌.

        took_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        # 5) HTTP 응답(백엔드 요구 포맷)
        #   {
        #     "status":200,
        #     "suceess":true,
        #     "type": "UPLOAD_SUCCESS"
        #   }
        return _ok({
            "type": "UPLOAD_SUCCESS",
            "jobId": jobId,
            "receivedBytes": recv_bytes,
            "totalBytes": total_bytes or None,
            "tookMs": took_ms,
        })

    except Exception as e:
        logger.exception(f"[/api/upload] failed at step={step}: {e}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "step": step, "message": str(e)},
        )
    finally:
        # 원본은 저장하지 않으므로 즉시 삭제
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
