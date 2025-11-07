# app/routers/presigned_upload.py
from __future__ import annotations

import asyncio
import binascii
import hashlib
import json
import shutil
import tempfile
import time
import uuid
from base64 import b64decode
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import anyio
from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    Header,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.crypto import (
    hmac_sha256_hex,               # (레거시 단일 업로드) HMAC-SHA256 HEX
    verify_chunk_signature_b64url, # presigned 청크 PUT 검증(Base64URL)
    verify_complete_signature_b64url,  # presigned 완료 POST 검증(Base64URL)
    verify_highlight_token,        # highlightKey 토큰 복호화/검증
)
from app.core.logging import logging
from app.core.progress import ProgressBus, PROGRESS_TYPE
from app.services.ai_demo_runner import (
    AIHighlightError,
    run_ai_demo_from_path,
)
from app.services.ffmpeg import get_duration   # 병합 후 길이 메타 계산
from app.services.pubsub import (
    publish_error,
    publish_progress,
    publish_result,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


def _ok(payload: dict) -> dict:
    # 백엔드 요구 응답 포맷: {status:200, suceess:true, ...}
    return {
        "status": 200,
        "suceess": True,  # 요청된 철자 유지
        "success": True,  # 하위 호환
        **payload,
    }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _ext_ok(name: str) -> bool:
    # 🔒 허용 확장자만 통과 (필요시 .env로 빼도 됨)
    allowed = {".mp4", ".mov", ".mkv", ".m4v"}
    return Path(name).suffix.lower() in allowed


def _now_local_str() -> str:
    """LocalDateTime ISO8601 (초) e.g., 2025-11-06T17:24:51"""
    return datetime.now().isoformat(timespec="seconds")


def _now_local_path() -> str:
    """경로 안전(LocalDateTime) e.g., 2025-11-06T17-24-51"""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


# 업로드 청크 최대 바이트 (없으면 무제한)
_CHUNK_MAX_BYTES = int(getattr(settings, "UPLOAD_CHUNK_MAX_MB", 0) or 0) * 1024 * 1024

# 청크 저장 베이스 디렉토리
UPLOAD_DIR = Path(getattr(settings, "UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 에러 / 헬퍼 (Codex 패치 반영)
# ─────────────────────────────────────────────────────────────

class ChunkUploadError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


async def _iter_upload_file(upload: UploadFile, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(upload.file.read, chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        await upload.close()


async def _iter_bytes_once(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def _store_chunk(
    *,
    upload_id: str,
    part_number: int,
    expires: str,
    signature: str,
    member_id: str,
    highlight_token: str,
    chunk_iter: AsyncIterator[bytes],
    declared_size: Optional[int],
    content_sha256: Optional[str],
) -> Tuple[str, int]:
    """
    presigned 정보와 토큰을 검증하고, 청크를
    /tmp/uploads/<uploadId>/part-000001 형태로 저장.
    return: (highlight_key, received_size)
    """
    # highlightKey 토큰 검증
    tok_ok, highlight_key = verify_highlight_token(highlight_token)
    if not tok_ok:
        raise ChunkUploadError(401, highlight_key)

    # presigned 서명/만료 검증
    ok, reason = verify_chunk_signature_b64url(
        expires_ms=expires,
        member_id=member_id,
        job_id=highlight_key,
        upload_id=upload_id,
        part_number=part_number,
        signature_b64url=signature,
    )
    if not ok:
        raise ChunkUploadError(401, reason)

    base = UPLOAD_DIR / Path(upload_id).name  # path traversal 방지
    _ensure_dir(base)
    dst = base / f"part-{int(part_number):06d}"

    # Content-Length 기반 선제 체크
    if _CHUNK_MAX_BYTES and declared_size and declared_size > _CHUNK_MAX_BYTES:
        raise ChunkUploadError(413, "chunk too large")

    size = 0
    sha256 = hashlib.sha256()

    async with await anyio.open_file(dst, "wb") as f:
        async for chunk in chunk_iter:
            if not chunk:
                continue
            size += len(chunk)

            if _CHUNK_MAX_BYTES and size > _CHUNK_MAX_BYTES:
                try:
                    dst.unlink(missing_ok=True)
                except Exception:
                    pass
                raise ChunkUploadError(413, "chunk too large")

            sha256.update(chunk)
            await f.write(chunk)

    # 무결성 검증 (옵션)
    if content_sha256:
        got = sha256.hexdigest()
        header_val = content_sha256.lower()
        if header_val.startswith("sha256:"):
            header_val = header_val.split(":", 1)[1]
        if header_val != got:
            try:
                dst.unlink(missing_ok=True)
            except Exception:
                pass
            raise ChunkUploadError(422, "chunk checksum mismatch")

    logger.info(f"[chunk] {upload_id} part={part_number} bytes={size}")

    # 업로드 진행률 (part 단위) PUB
    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,
            value={
                "type": PROGRESS_TYPE.UPLOAD_PROGRESS,
                "progress": None,
                "receivedBytes": size,
                "partNumber": int(part_number),
                "timestampMs": _now_ms(),
            },
        )
    except Exception as e:
        logger.debug(f"[chunk] progress publish skip: {e}")

    try:
        await publish_progress(
            member_id,
            highlight_key,
            None,
            "chunk stored",
            type_="UPLOADING",
            stage="uploading",
            received_bytes=size,
        )
    except Exception as e:
        logger.debug(f"[chunk] progress publish skip(pubsub): {e}")

    return highlight_key, size


def _parse_presigned_payload(presigned_raw: str, chunk_index: int) -> Dict[str, Any]:
    """
    multipart(form-data)로 들어온 presigned JSON + url 쿼리에서
    uploadId, partNumber, expires, signature, memberId, highlightToken, contentSha256 추출.
    """
    try:
        payload = json.loads(presigned_raw)
    except Exception:
        raise ChunkUploadError(400, "invalid presigned payload")

    headers = payload.get("headers") or {}

    upload_id = payload.get("uploadId") or payload.get("upload_id")
    part_number = payload.get("partNumber") or payload.get("part_number")
    expires = payload.get("expires") or payload.get("expiresMs") or payload.get("expires_ms")
    signature = payload.get("signature") or payload.get("sig")
    member_id = (
        payload.get("memberId")
        or payload.get("member_id")
        or headers.get("x-member-id")
        or headers.get("X-Member-Id")
    )
    highlight_token = (
        payload.get("highlightKey")
        or payload.get("highlightToken")
        or headers.get("x-highlight-key")
        or headers.get("X-Highlight-Key")
    )
    content_sha256 = (
        payload.get("contentSha256")
        or payload.get("content_sha256")
        or headers.get("x-content-sha256")
        or headers.get("X-Content-Sha256")
    )

    url = payload.get("url")
    if url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        upload_id = upload_id or (qs.get("uploadId") or qs.get("upload_id") or [None])[0]
        expires = expires or (qs.get("expires") or qs.get("expiresMs") or qs.get("expires_ms") or [None])[0]
        signature = signature or (qs.get("signature") or qs.get("sig") or [None])[0]
        member_id = member_id or (qs.get("memberId") or qs.get("member_id") or [None])[0]
        part_number = part_number or (qs.get("partNumber") or qs.get("part_number") or [None])[0]

    if part_number is None:
        part_number = chunk_index + 1

    try:
        part_number_int = int(part_number)
    except Exception:
        raise ChunkUploadError(422, "invalid partNumber")

    if not upload_id or not expires or not signature or not member_id or not highlight_token:
        raise ChunkUploadError(400, "presigned payload missing required fields")

    return {
        "uploadId": str(upload_id),
        "partNumber": part_number_int,
        "expires": str(expires),
        "signature": str(signature),
        "memberId": str(member_id),
        "highlightToken": str(highlight_token),
        "contentSha256": str(content_sha256) if content_sha256 else None,
    }


async def _publish_processing_event(
    highlight_key: str,
    *,
    member_id: str,
    type_: str,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    value: Dict[str, Any] = {
        "type": type_,
        "timestampMs": _now_ms(),
        "memberId": member_id,
    }
    if progress is not None:
        value["progress"] = float(progress)
    if message:
        value["message"] = message
    if extra:
        value.update(extra)

    try:
        await ProgressBus.publish_kv(job_id=highlight_key, value=value)
    except Exception as e:
        logger.debug(f"[ai-demo] progress publish skip: {e}")


async def _auto_ai_demo_after_complete(member_id: str, highlight_key: str, source_path: Path) -> None:
    """
    Presigned 업로드 완료 후, 자동으로 AI-Demo(하이라이트 생성)를 백그라운드에서 수행.
    """
    stage_meta = {"stage": "ai-demo", "sourceFolder": source_path.parent.name}

    await _publish_processing_event(
        highlight_key,
        member_id=member_id,
        type_=PROGRESS_TYPE.PROCESSING_START,
        progress=0.0,
        message="ai-demo queued",
        extra=stage_meta,
    )

    try:
        await publish_progress(
            member_id,
            highlight_key,
            0.05,
            "ai-demo queued",
            type_="PROCESSING",
            stage="ai-demo",
        )    
    except Exception:
        pass

    try:
        # run_ai_demo_from_path 는 동기 함수이므로 to_thread 사용
        result = await anyio.to_thread.run_sync(
            run_ai_demo_from_path,
            source_path,
            member_id,
            highlight_key,
            source_path.stem,
            getattr(settings, "AUTO_AI_DEMO_OVERLAY_TAG", "AI-Selector"),
            getattr(settings, "AUTO_AI_DEMO_MERGE_OUTPUT", True),
        )

        summary_url = result.summary_url
        clip_count = len(result.public_urls)

        await _publish_processing_event(
            highlight_key,
            member_id=member_id,
            type_=PROGRESS_TYPE.PROCESSING,
            progress=0.6,
            message="ai-demo clips saved",
            extra={**stage_meta, "summaryUrl": summary_url, "clipCount": clip_count},
        )

        try:
            await publish_progress(
                member_id,
                highlight_key,
                0.6,
                "ai-demo clips saved",
                type_="PROCESSING",
                stage="ai-demo",
                total_clips=clip_count,
                current_clip=clip_count,
            )        
        except Exception:
            pass

        await publish_result(member_id, highlight_key, result.public_urls, final=True)

        await _publish_processing_event(
            highlight_key,
            member_id=member_id,
            type_=PROGRESS_TYPE.COMPLETE,
            progress=1.0,
            message="ai-demo completed",
            extra={**stage_meta, "summaryUrl": summary_url, "clipCount": clip_count},
        )

    except AIHighlightError as e:
        msg = str(e)
        logger.info(f"[ai-demo] highlightKey={highlight_key} skipped: {msg}")
        await publish_error(
            member_id,
            highlight_key,
            msg,
            status=404 if getattr(e, "status_code", 400) == 404 else 400,
        )
        await _publish_processing_event(
            highlight_key,
            member_id=member_id,
            type_=PROGRESS_TYPE.COMPLETE,
            progress=0.0,
            message=msg,
            extra={**stage_meta, "error": msg},
        )

    except Exception as e:
        msg = f"ai-demo failed: {e}"
        logger.exception(f"[ai-demo] highlightKey={highlight_key} failed: {e}")
        await publish_error(member_id, highlight_key, msg)
        await _publish_processing_event(
            highlight_key,
            member_id=member_id,
            type_=PROGRESS_TYPE.COMPLETE,
            progress=0.0,
            message=msg,
            extra={**stage_meta, "error": str(e)},
        )

# ─────────────────────────────────────────────────────────────
# (A) 단일 업로드: PUT /api/upload  (레거시 – 유지)
# ─────────────────────────────────────────────────────────────

@router.put("/upload", summary="Pre-signed 업로드 수신 (서명 검증 + 진행률 PUB)")
async def upload_video(
    request: Request,
    expires: int = Query(..., description="만료 시각(ms since epoch)"),
    memberId: str = Query(..., description="멤버 식별자"),
    jobId: str = Query(..., description="업로드/처리 작업 ID(레거시, highlightKey와 동일 개념)"),
    signature: str = Query(..., description="HMAC-SHA256 HEX(signature)"),
    fileName: str = Query(..., description="원본 파일명"),
):
    step = "parse_query"
    t0 = time.perf_counter()
    tmp_path: Optional[Path] = None
    recv_bytes = 0
    total_bytes = int(request.headers.get("content-length") or 0)

    try:
        if not _ext_ok(fileName):
            return JSONResponse(
                status_code=415,
                content={"status": "error", "message": "unsupported media type"},
            )

        # 1) 만료/서명 검증
        step = "verify_signature"
        if expires <= 0:
            return JSONResponse(
                status_code=422,
                content={"status": "error", "step": step, "message": "expires(ms) required"},
            )

        now = _now_ms()
        if now > expires:
            return JSONResponse(
                status_code=401,
                content={"status": "error", "step": step, "message": "url expired"},
            )

        msg = f"{expires}:{memberId}:{jobId}:{fileName}"
        expected_sig = hmac_sha256_hex(settings.BACKEND_SECRET, msg)
        if not signature or signature.lower() != expected_sig.lower():
            logger.warning(
                f"[upload] signature mismatch jobId={jobId} expected={expected_sig} got={signature}"
            )
            return JSONResponse(
                status_code=401,
                content={"status": "error", "step": step, "message": "invalid signature"},
            )

        # 2) UPLOAD_START publish
        step = "publish_start"
        try:
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
        except Exception as e:
            logger.warning(f"[upload] progress publish failed: {e}")

        try:
            await publish_progress(
                memberId,
                jobId,
                0.0,
                "original video uploading",
                type_="UPLOADING",
                total_bytes=total_bytes or None,
                received_bytes=0,
                stage="uploading",
            )
        except Exception as e:
            logger.debug(f"[upload] progress publish skip(pubsub): {e}")
        
        # 3) 스트리밍 수신(임시 파일). 5초 단위 진행률 PUB
        step = "stream_and_save"
        fd, path_str = tempfile.mkstemp(
            prefix="upload_", suffix=Path(fileName).suffix or ".mp4"
        )
        tmp_path = Path(path_str)
        last_pub = time.perf_counter()
        interval = float(getattr(settings, "PROGRESS_INTERVAL_SEC", 5.0) or 5.0)

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
                    try:
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
                    except Exception as e:
                        logger.debug(f"[upload] progress publish skip: {e}")

                    try:
                        await publish_progress(
                            memberId,
                            jobId,
                            prog,
                            "original video uploading",
                            type_="UPLOADING",
                            total_bytes=total_bytes or None,
                            received_bytes=recv_bytes,
                            stage="uploading",
                        )
                    except Exception as e:
                        logger.debug(f"[upload] progress publish skip(pubsub): {e}")

        # 종료 시 100% 보정
        try:
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
        except Exception as e:
            logger.debug(f"[upload] final progress publish skip: {e}")

        try:
            await publish_progress(
                memberId,
                jobId,
                1.0 if total_bytes > 0 else None,
                "original video uploading",
                type_="UPLOADING",
                total_bytes=total_bytes or None,
                received_bytes=recv_bytes,
                stage="uploading",
            )
        except Exception as e:
            logger.debug(f"[upload] final progress publish skip(pubsub): {e}")

        took_ms = round((time.perf_counter() - t0) * 1000.0, 1)

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

# ─────────────────────────────────────────────────────────────
# (B) 청크 업로드: PUT /api/presigned/chunk (바이너리)
# ─────────────────────────────────────────────────────────────

@router.put("/presigned/chunk", summary="Presigned 청크 업로드 (Base64URL 서명)")
async def put_chunk(
    request: Request,
    uploadId: str,
    partNumber: int,
    expires: str,
    signature: str,
    x_member_id: str = Header(..., alias="x-member-id"),
    x_highlight_token: str = Header(..., alias="x-highlight-key"),
    x_content_sha256: str | None = Header(None, alias="x-content-sha256"),
):
    # partNumber 검증
    if not isinstance(partNumber, int) or partNumber < 1:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": "invalid partNumber"},
        )

    cl = request.headers.get("content-length")
    declared_size = int(cl) if cl and cl.isdigit() else None

    try:
        highlight_key, size = await _store_chunk(
            upload_id=uploadId,
            part_number=partNumber,
            expires=expires,
            signature=signature,
            member_id=x_member_id,
            highlight_token=x_highlight_token,
            chunk_iter=request.stream(),
            declared_size=declared_size,
            content_sha256=x_content_sha256,
        )
    except ChunkUploadError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "error", "message": e.message},
        )

    return _ok({
        "type": "UPLOADING",
        "uploadId": uploadId,
        "highlightKey": highlight_key,
        "partNumber": int(partNumber),
        "receivedBytes": size,
        "message": "chunk stored",
        "timestamp": _now_ms(),
    })

# ─────────────────────────────────────────────────────────────
# (B-2) 청크 업로드: POST /api/presigned/chunk (multipart + base64)
# ─────────────────────────────────────────────────────────────

@router.post("/presigned/chunk", summary="Presigned 청크 업로드 (multipart 지원)")
async def post_chunk_form(
    file: UploadFile | None = File(default=None),
    presigned: str = Form(...),
    chunkIndex: int = Form(...),
    chunkData: str | None = Form(
        default=None,
        description="base64 인코딩된 청크 데이터(선택)",
    ),
):
    # chunkIndex 검증
    try:
        chunk_idx = int(chunkIndex)
    except Exception:
        if file is not None:
            await file.close()
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": "invalid chunkIndex"},
        )

    # presigned payload 파싱
    try:
        parsed = _parse_presigned_payload(presigned, chunk_idx)
    except ChunkUploadError as e:
        if file is not None:
            await file.close()
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "error", "message": e.message},
        )

    if file is None and not chunkData:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "chunk payload missing"},
        )

    # 청크 데이터 소스 결정
    chunk_iter: AsyncIterator[bytes]
    declared_size: Optional[int] = None

    if chunkData:
        data = chunkData.strip()
        if "," in data:
            data = data.split(",", 1)[1]
        try:
            raw = b64decode(data)
        except (binascii.Error, ValueError):
            if file is not None:
                await file.close()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "invalid base64 chunkData"},
            )

        chunk_iter = _iter_bytes_once(raw)
        declared_size = len(raw)

        if file is not None:
            await file.close()
            file = None
    else:
        # file 업로드 사용
        assert file is not None
        chunk_iter = _iter_upload_file(file)

    try:
        highlight_key, size = await _store_chunk(
            upload_id=parsed["uploadId"],
            part_number=parsed["partNumber"],
            expires=parsed["expires"],
            signature=parsed["signature"],
            member_id=parsed["memberId"],
            highlight_token=parsed["highlightToken"],
            chunk_iter=chunk_iter,
            declared_size=declared_size,
            content_sha256=parsed.get("contentSha256"),
        )
    except ChunkUploadError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "error", "message": e.message},
        )

    return _ok({
        "type": "UPLOADING",
        "uploadId": parsed["uploadId"],
        "highlightKey": highlight_key,
        "partNumber": parsed["partNumber"],
        "receivedBytes": size,
        "message": "chunk stored",
        "timestamp": _now_ms(),
    })

# ─────────────────────────────────────────────────────────────
# (C) 병합 완료: POST /api/presigned/complete
# ─────────────────────────────────────────────────────────────

def _merge_parts(upload_id: str, member_id: str, highlight_key: str, file_name: str) -> Tuple[Path, int, str]:
    """
    청크를 병합해 최종 원본 파일로 저장.
    return: (dst_path, total_size_bytes, ldt_path)
    """
    base = UPLOAD_DIR / Path(upload_id).name
    if not base.exists():
        raise FileNotFoundError("no parts for uploadId")

    parts = sorted(base.glob("part-*"))
    if not parts:
        raise FileNotFoundError("no parts found")

    # 연속성 체크: part-000001부터 순서대로 존재하는지 확인
    for i, p in enumerate(parts, start=1):
        expect = f"part-{i:06d}"
        if p.name != expect:
            raise FileNotFoundError(f"missing part {expect}")

    if not _ext_ok(file_name):
        raise ValueError("unsupported media type")

    ext = Path(file_name).suffix or ".mp4"

    # LocalDateTime 서브폴더
    ldt_path = _now_local_path()
    dst_dir = Path(settings.SAVE_ROOT) / member_id / highlight_key / ldt_path
    _ensure_dir(dst_dir)

    dst_path = dst_dir / f"original_{uuid.uuid4().hex[:8]}{ext}"

    total = 0
    with dst_path.open("wb") as w:
        for p in parts:
            s = p.stat().st_size
            with p.open("rb") as r:
                shutil.copyfileobj(r, w)
            total += s

    # 파트 정리
    try:
        shutil.rmtree(base, ignore_errors=True)
    except Exception:
        pass

    return dst_path, total, ldt_path


@router.post("/presigned/complete", summary="Presigned 업로드 병합 완료 → sourceUrl 반환")
async def post_complete(
    x_upload_id: str = Header(..., alias="x-upload-id"),
    x_sig: str = Header(..., alias="x-sig"),
    x_expires: str = Header(..., alias="x-expires"),
    payload: dict = Body(...),
):
    member_id = str(payload.get("memberId", "")).strip()
    highlight_token = str(payload.get("highlightKey", "")).strip()
    file_name = str(payload.get("fileName", "")).strip()
    total_bytes_decl: int | None = payload.get("totalBytes")

    if not (member_id and highlight_token and file_name):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "memberId/highlightKey/fileName required"},
        )

    tok_ok, highlight_key = verify_highlight_token(highlight_token)
    if not tok_ok:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": highlight_key},
        )

    # 1) 완료 서명 검증
    ok, reason = verify_complete_signature_b64url(
        expires_ms=x_expires,
        member_id=member_id,
        job_id=highlight_key,
        upload_id=x_upload_id,
        signature_b64url=x_sig,
    )
    if not ok:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": reason},
        )

    # 2) 병합
    try:
        out_path, total_bytes_actual, ldt_path = _merge_parts(
            x_upload_id, member_id, highlight_key, file_name
        )
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)},
        )

    # 3) 총 바이트 검증(선택)
    if isinstance(total_bytes_decl, int) and total_bytes_decl > 0:
        if abs(total_bytes_decl - total_bytes_actual) > 0:
            logger.warning(
                f"[complete] total bytes mismatch: decl={total_bytes_decl} actual={total_bytes_actual}"
            )

    # 4) 공개 URL + 길이
    public_url = (
        f"{settings.STATIC_BASE_URL.rstrip('/')}/"
        f"{member_id}/{highlight_key}/{ldt_path}/{out_path.name}"
    )
    try:
        duration = get_duration(out_path)
    except Exception as e:
        logger.debug(f"[complete] get_duration failed: {e}")
        duration = 0.0

    # 업로드 완료 신호
    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,
            value={
                "type": PROGRESS_TYPE.UPLOAD_COMPLETE,
                "progress": 1.0,
                "sizeBytes": int(total_bytes_actual),
                "sourceUrl": public_url,
                "folder": ldt_path,
                "timestampMs": _now_ms(),
            },
        )
    except Exception as e:
        logger.debug(f"[complete] progress publish skip: {e}")

    try:
        await publish_progress(
            member_id,
            highlight_key,
            None,
            "original video received",
            type_="UPLOAD_COMPLETE",
            total_bytes=int(total_bytes_actual),
            stage="uploading",
        )
    except Exception as e:
        logger.debug(f"[complete] progress publish skip(pubsub): {e}")

    # 5) (옵션) Presigned 업로드 완료 후 AI-DEMO 자동 실행
    if getattr(settings, "AUTO_RUN_AI_DEMO", False):
        try:
            asyncio.create_task(
                _auto_ai_demo_after_complete(member_id, highlight_key, out_path)
            )
        except RuntimeError as e:
            logger.warning(f"[complete] auto ai-demo scheduling failed: {e}")

    # 6) 응답
    return _ok({
        "type": "UPLOAD_COMPLETE",
        "memberId": member_id,
        "highlightKey": highlight_key,
        "uploadId": x_upload_id,
        "sizeBytes": int(total_bytes_actual),
        "sourceUrl": public_url,
        "durationSec": round(float(duration), 3),
        "folder": ldt_path,
        "localDateTime": _now_local_str(),
        "message": "original video received",
        "timestamp": _now_ms(),
    })
