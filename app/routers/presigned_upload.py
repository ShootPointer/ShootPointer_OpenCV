# app/routers/presigned_upload.py
from __future__ import annotations

import json
import time
import uuid
import tempfile
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import anyio
from fastapi import APIRouter, Request, Query, Header, Body
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logging
from app.core.progress import ProgressBus, PROGRESS_TYPE
from app.core.crypto import (
    hmac_sha256_hex,                          # 레거시 단일 업로드 서명
    verify_chunk_signature_b64url,            # 청크 PUT 검증(B64URL)
    verify_complete_signature_b64url,         # 완료 POST 검증(B64URL)
    verify_highlight_token,                   # ✅ NEW: 하이라이트 토큰 검증
)

from app.services.ffmpeg import get_duration  # 병합 후 길이 메타 계산

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────
def _now_ms() -> int:
    return int(time.time() * 1000)

def _ldt_folder() -> str:
    # LocalDateTime 폴더명: 2025-11-06T17-24-51
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

def _ok(payload: dict) -> dict:
    return {
        "status": 200,
        "suceess": True,  # 요청된 철자 유지
        "success": True,  # 하위 호환
        **payload,
    }

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _ext_ok(name: str) -> bool:
    allowed = {".mp4", ".mov", ".mkv", ".m4v"}
    return Path(name).suffix.lower() in allowed

# 업로드 청크 최대 바이트 (없으면 무제한)
_CHUNK_MAX_BYTES = int(getattr(settings, "UPLOAD_CHUNK_MAX_MB", 0) or 0) * 1024 * 1024

# ─────────────────────────────────────────────────────────────
# (A) 단일 업로드: PUT /api/upload  — 레거시 유지
# ─────────────────────────────────────────────────────────────
@router.put("/upload", summary="Pre-signed 업로드 수신 (서명 검증 + 진행률 PUB) [legacy]")
async def upload_video(
    request: Request,
    expires: int = Query(..., description="만료 시각(ms since epoch)"),
    memberId: str = Query(..., description="멤버 식별자"),
    jobId: str = Query(..., description="업로드/처리 작업 ID"),
    signature: str = Query(..., description="HMAC-SHA256 HEX(signature)"),
    fileName: str = Query(..., description="원본 파일명"),
):
    """
    ⚠️ 레거시: 병합 저장을 하지 않고 임시파일을 지움. 새 흐름(청크 업로드)을 권장.
    """
    step = "parse_query"
    t0 = time.perf_counter()
    tmp_path: Optional[Path] = None
    recv_bytes = 0
    total_bytes = int(request.headers.get("content-length") or 0)

    try:
        if not _ext_ok(fileName):
            return JSONResponse(status_code=415, content={"status": "error", "message": "unsupported media type"})

        # 1) 만료/서명 검증
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

        # 2) 진행률 PUB (생략 가능)
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
            logger.debug(f"[upload] progress publish skip: {e}")

        # 3) 스트리밍 수신 → 임시파일 (병합/저장 안함)
        step = "stream_and_save"
        fd, path_str = tempfile.mkstemp(prefix="upload_", suffix=Path(fileName).suffix or ".mp4")
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
                    except Exception:
                        pass

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
        except Exception:
            pass

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
        return JSONResponse(status_code=400, content={"status": "error", "step": step, "message": str(e)})
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# (B) 청크 업로드
#     헤더:
#       - x-highlight-token: HMAC-signed token (권장, NEW)
#       - (fallback) x-member-id, x-job-id  ← job_id에 highlightKey를 넣어도 동작
# ─────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(getattr(settings, "UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _resolve_identity_from_headers(x_highlight_token: str | None,
                                   x_member_id_legacy: str | None,
                                   x_job_id_legacy: str | None) -> Tuple[bool, str, str, str]:
    """
    return: (ok, reason, member_id, highlight_key)
    우선순위:
      1) x-highlight-token → verify_highlight_token
      2) 레거시(x-member-id, x-job-id) → job_id를 highlightKey로 간주
    """
    if x_highlight_token:
        ok, reason, payload = verify_highlight_token(x_highlight_token)
        if not ok:
            return False, f"highlight token: {reason}", "", ""
        return True, "ok", str(payload["memberId"]), str(payload["highlightKey"])
    # fallback
    if x_member_id_legacy and x_job_id_legacy:
        return True, "legacy", x_member_id_legacy, x_job_id_legacy
    return False, "identity headers missing", "", ""

@router.put("/presigned/chunk", summary="Presigned 청크 업로드 (Base64URL signature + HighlightToken)")
async def put_chunk(
    request: Request,
    uploadId: str,
    partNumber: int,
    expires: str,
    signature: str,
    x_highlight_token: str | None = Header(None, alias="x-highlight-token"),
    x_member_id: str | None = Header(None, alias="x-member-id"),  # legacy
    x_job_id: str | None = Header(None, alias="x-job-id"),        # legacy(여기에 highlightKey가 들어와도 허용)
    x_content_sha256: str | None = Header(None, alias="x-content-sha256"),
):
    if not isinstance(partNumber, int) or partNumber < 1:
        return JSONResponse(status_code=422, content={"status": "error", "message": "invalid partNumber"})

    # 0) 토큰/레거시 헤더에서 memberId, highlightKey 얻기
    ok, reason, member_id, highlight_key = _resolve_identity_from_headers(
        x_highlight_token, x_member_id, x_job_id
    )
    if not ok:
        return JSONResponse(status_code=401, content={"status": "error", "message": reason})

    # 1) 청크 서명/만료 검증 (기존 인터페이스의 job_id 자리에 highlightKey를 전달)
    ok, reason = verify_chunk_signature_b64url(
        expires_ms=expires,
        member_id=member_id,
        job_id=highlight_key,          # ← 기존 verify 함수와 호환 위해 job_id에 전달
        upload_id=uploadId,
        part_number=partNumber,
        signature_b64url=signature,
    )
    if not ok:
        return JSONResponse(status_code=401, content={"status": "error", "message": reason})

    # 2) 청크 저장
    base = UPLOAD_DIR / Path(uploadId).name
    _ensure_dir(base)
    dst = base / f"part-{int(partNumber):06d}"

    cl = request.headers.get("content-length")
    if _CHUNK_MAX_BYTES and cl and cl.isdigit():
        if int(cl) > _CHUNK_MAX_BYTES:
            return JSONResponse(status_code=413, content={"status": "error", "message": "chunk too large"})

    size = 0
    sha256 = hashlib.sha256()
    async with await anyio.open_file(dst, "wb") as f:
        async for chunk in request.stream():
            if not chunk:
                continue
            size += len(chunk)
            if _CHUNK_MAX_BYTES and size > _CHUNK_MAX_BYTES:
                try: dst.unlink(missing_ok=True)
                except: pass
                return JSONResponse(status_code=413, content={"status":"error","message":"chunk too large"})
            sha256.update(chunk)
            await f.write(chunk)

    if x_content_sha256:
        got = sha256.hexdigest()
        xh = x_content_sha256.split(":", 1)[1].lower() if x_content_sha256.lower().startswith("sha256:") else x_content_sha256.lower()
        if xh != got:
            try: dst.unlink(missing_ok=True)
            except: pass
            return JSONResponse(status_code=422, content={"status":"error","message":"chunk checksum mismatch"})

    logger.info(f"[chunk] {uploadId} part={partNumber} bytes={size} member={member_id} hk={highlight_key}")

    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,   # 진행표시는 highlightKey 기준
            value={
                "type": PROGRESS_TYPE.UPLOAD_PROGRESS,
                "progress": None,
                "receivedBytes": size,
                "partNumber": int(partNumber),
                "timestampMs": _now_ms(),
            },
        )
    except Exception:
        pass

    return _ok({
        "type": "UPLOADING",
        "uploadId": uploadId,
        "partNumber": int(partNumber),
        "receivedBytes": size,
        "message": "chunk stored",
        "timestamp": _now_ms(),
    })


# ─────────────────────────────────────────────────────────────
# (C) 병합 완료
#     헤더:
#       x-upload-id, x-sig, x-expires
#       x-highlight-token (권장) / (폴백) 없음
#     바디: { fileName, totalBytes? }  + 토큰에서 memberId/highlightKey 복원
# ─────────────────────────────────────────────────────────────
def _merge_parts(upload_id: str, member_id: str, highlight_key: str, file_name: str) -> Tuple[Path, str, int]:
    base = UPLOAD_DIR / Path(upload_id).name
    if not base.exists():
        raise FileNotFoundError("no parts for uploadId")

    parts = sorted(base.glob("part-*"))
    if not parts:
        raise FileNotFoundError("no parts found")

    for i, p in enumerate(parts, start=1):
        expect = f"part-{i:06d}"
        if p.name != expect:
            raise FileNotFoundError(f"missing part {expect}")

    if not _ext_ok(file_name):
        raise ValueError("unsupported media type")

    ext = Path(file_name).suffix or ".mp4"
    ldt = _ldt_folder()
    out_dir = Path(settings.SAVE_ROOT) / member_id / highlight_key / ldt
    _ensure_dir(out_dir)
    out_path = out_dir / f"original_{uuid.uuid4().hex[:8]}{ext}"

    total = 0
    with out_path.open("wb") as w:
        for p in parts:
            s = p.stat().st_size
            with p.open("rb") as r:
                shutil.copyfileobj(r, w)
            total += s

    try:
        shutil.rmtree(base, ignore_errors=True)
    except Exception:
        pass

    return out_path, ldt, total

@router.post("/presigned/complete", summary="Presigned 업로드 병합 완료 → sourceUrl 반환")
async def post_complete(
    x_upload_id: str = Header(..., alias="x-upload-id"),
    x_sig: str = Header(..., alias="x-sig"),
    x_expires: str = Header(..., alias="x-expires"),
    x_highlight_token: str | None = Header(None, alias="x-highlight-token"),
    payload: dict = Body(...),
):
    """
    새 모델:
      - memberId/highlightKey는 x-highlight-token에서 복원
      - 레거시 바디 필드(memberId/jobId) 의존 제거(있어도 무시)
    """
    file_name = str(payload.get("fileName", "")).strip()
    total_bytes_decl: int | None = payload.get("totalBytes")

    if not file_name:
        return JSONResponse(status_code=400, content={"status":"error","message":"fileName required"})

    # 0) 토큰 검증
    ok, reason, p = verify_highlight_token(x_highlight_token or "")
    if not ok:
        return JSONResponse(status_code=401, content={"status":"error","message":f"highlight token: {reason}"})
    member_id = str(p["memberId"])
    highlight_key = str(p["highlightKey"])

    # 1) 완료 서명 검증 (기존 인터페이스의 job_id 자리에 highlightKey 전달)
    ok, reason = verify_complete_signature_b64url(
        expires_ms=x_expires,
        member_id=member_id,
        job_id=highlight_key,     # ← 호환성
        upload_id=x_upload_id,
        signature_b64url=x_sig,
    )
    if not ok:
        return JSONResponse(status_code=401, content={"status":"error","message":reason})

    # 2) 병합
    try:
        out_path, ldt_folder, total_bytes_actual = _merge_parts(x_upload_id, member_id, highlight_key, file_name)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse(status_code=422, content={"status":"error","message":str(e)})

    if isinstance(total_bytes_decl, int) and total_bytes_decl > 0:
        if abs(total_bytes_decl - total_bytes_actual) > 0:
            logger.warning(f"[complete] total bytes mismatch: decl={total_bytes_decl} actual={total_bytes_actual}")

    # 3) 공개 URL + 길이
    public_url = f"{settings.STATIC_BASE_URL.rstrip('/')}/{member_id}/{highlight_key}/{ldt_folder}/{out_path.name}"
    try:
        duration = get_duration(out_path)
    except Exception as e:
        logger.debug(f"[complete] get_duration failed: {e}")
        duration = 0.0

    # 진행 알림(옵션) — highlightKey 기준
    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,
            value={
                "type": PROGRESS_TYPE.UPLOAD_COMPLETE,
                "progress": 1.0,
                "sizeBytes": total_bytes_actual,
                "sourceUrl": public_url,
                "ldtFolder": ldt_folder,
                "timestampMs": _now_ms(),
            },
        )
    except Exception:
        pass

    return _ok({
        "type": "UPLOAD_COMPLETE",
        "memberId": member_id,
        "highlightKey": highlight_key,
        "ldtFolder": ldt_folder,
        "uploadId": x_upload_id,
        "sizeBytes": int(total_bytes_actual),
        "sourceUrl": public_url,
        "durationSec": round(float(duration), 3),
        "message": "original video received",
        "timestamp": _now_ms(),
    })
