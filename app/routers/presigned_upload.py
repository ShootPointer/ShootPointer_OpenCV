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
    hmac_sha256_hex,                          # (레거시 단일 업로드) HMAC-SHA256 HEX
    verify_chunk_signature_b64url,            # presigned 청크 PUT 검증(Base64URL)
    verify_complete_signature_b64url,         # presigned 완료 POST 검증(Base64URL)
    verify_highlight_token,                   # highlightKey 토큰 복호화/검증
)
from app.services.ffmpeg import get_duration  # 병합 후 길이 메타 계산

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


# ─────────────────────────────────────────────────────────────
# (A) 단일 업로드: PUT /api/upload  (레거시 – 유지)
#     NOTE: 기존 클라이언트 호환을 위해 jobId를 그대로 받되,
#           내부적으로는 highlightKey와 동일 개념으로 취급해도 무방.
#           이 엔드포인트는 임시 파일만 받고 최종 저장은 하지 않음.
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

        # 3) 스트리밍 수신(임시 파일). 5초 단위 진행률 PUB
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
                    except Exception as e:
                        logger.debug(f"[upload] progress publish skip: {e}")

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

        took_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return _ok({
            "type": "UPLOAD_SUCCESS",
            "jobId": jobId,  # 레거시 호환
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
        # 원본은 저장하지 않으므로 즉시 삭제 (단일 업로드 엔드포인트의 정책)
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# (B) 청크 업로드: PUT /api/presigned/chunk
#     쿼리: uploadId, partNumber, expires, signature(base64url)
#     헤더: x-member-id, x-highlight-key, (옵션) x-content-sha256
#     본문: 바이너리(이 파트 내용)
# ─────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(getattr(settings, "UPLOAD_DIR", "/tmp/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    # 🔒 partNumber 검증
    if not isinstance(partNumber, int) or partNumber < 1:
        return JSONResponse(status_code=422, content={"status": "error", "message": "invalid partNumber"})

    # 1) highlightKey 토큰 복호화 → 실제 highlightKey 획득
    tok_ok, highlight_key = verify_highlight_token(x_highlight_token)
    if not tok_ok:
        return JSONResponse(status_code=401, content={"status": "error", "message": highlight_key})

    # 2) 서명/만료 검증
    ok, reason = verify_chunk_signature_b64url(
        expires_ms=expires,
        member_id=x_member_id,
        job_id=highlight_key,     # 내부 검증 함수는 job_id 파라미터명을 쓰지만 값은 highlightKey
        upload_id=uploadId,
        part_number=partNumber,
        signature_b64url=signature,
    )
    if not ok:
        return JSONResponse(status_code=401, content={"status": "error", "message": reason})

    # 3) 청크 저장 (/tmp/uploads/<uploadId>/part-000001)
    base = UPLOAD_DIR / Path(uploadId).name   # 🔒 path traversal 방지
    _ensure_dir(base)
    dst = base / f"part-{int(partNumber):06d}"

    # 📏 content-length가 있다면 선제 한도 체크
    cl = request.headers.get("content-length")
    if _CHUNK_MAX_BYTES and cl and cl.isdigit():
        if int(cl) > _CHUNK_MAX_BYTES:
            return JSONResponse(status_code=413, content={"status":"error","message":"chunk too large"})

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

    # 🔒 (선택) 청크 무결성: 헤더 제공 시 비교
    if x_content_sha256:
        got = sha256.hexdigest()
        if x_content_sha256.lower().startswith("sha256:"):
            xh = x_content_sha256.split(":", 1)[1].lower()
        else:
            xh = x_content_sha256.lower()
        if xh != got:
            try: dst.unlink(missing_ok=True)
            except: pass
            return JSONResponse(status_code=422, content={"status":"error","message":"chunk checksum mismatch"})

    logger.info(f"[chunk] {uploadId} part={partNumber} bytes={size}")

    # (옵션) 업로드 진행률 PUB — part 단위로 간단 표기
    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,   # 진행률 키 = highlightKey
            value={
                "type": PROGRESS_TYPE.UPLOAD_PROGRESS,
                "progress": None,  # 총합을 모르면 None
                "receivedBytes": size,
                "partNumber": int(partNumber),
                "timestampMs": _now_ms(),
            },
        )
    except Exception as e:
        logger.debug(f"[chunk] progress publish skip: {e}")

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
# (C) 병합 완료: POST /api/presigned/complete
#     헤더: x-upload-id, x-sig(Base64URL), x-expires(ms)
#     바디: { memberId, highlightKey, fileName, totalBytes, parts:[...] }
#     처리: /tmp/uploads/<uploadId>/part-*  →  /data/highlights/<member>/<highlightKey>/<LocalDateTime>/original_*.mp4
#           + STATIC_BASE_URL 로 sourceUrl 구성, durationSec 포함
# ─────────────────────────────────────────────────────────────
def _merge_parts(upload_id: str, member_id: str, highlight_key: str, file_name: str) -> Tuple[Path, int, str]:
    """
    청크를 병합해 최종 원본 파일로 저장.
    return: (dst_path, total_size_bytes, ldt_path)
    """
    base = UPLOAD_DIR / Path(upload_id).name    # 🔒 path traversal 방지
    if not base.exists():
        raise FileNotFoundError("no parts for uploadId")

    parts = sorted(base.glob("part-*"))
    if not parts:
        raise FileNotFoundError("no parts found")

    # 🧾 연속성 체크: part-000001부터 차례대로 있는지(누락 감지)
    for i, p in enumerate(parts, start=1):
        expect = f"part-{i:06d}"
        if p.name != expect:
            raise FileNotFoundError(f"missing part {expect}")

    # 🔒 파일명 확장자 체크
    if not _ext_ok(file_name):
        raise ValueError("unsupported media type")

    ext = Path(file_name).suffix or ".mp4"

    # LocalDateTime 서브폴더
    ldt_path = _now_local_path()  # 경로용 (YYYY-MM-DDTHH-MM-SS)
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
    member_id     = str(payload.get("memberId", "")).strip()
    highlight_token = str(payload.get("highlightKey", "")).strip()
    file_name     = str(payload.get("fileName", "")).strip()
    total_bytes_decl: int | None = payload.get("totalBytes")

    if not (member_id and highlight_token and file_name):
        return JSONResponse(status_code=400, content={"status":"error","message":"memberId/highlightKey/fileName required"})

    tok_ok, highlight_key = verify_highlight_token(highlight_token)
    if not tok_ok:
        return JSONResponse(status_code=401, content={"status":"error","message":highlight_key})

    # 1) 완료 서명 검증
    ok, reason = verify_complete_signature_b64url(
        expires_ms=x_expires,
        member_id=member_id,
        job_id=highlight_key,        # 내부 파라미터명은 job_id지만 값은 highlightKey
        upload_id=x_upload_id,
        signature_b64url=x_sig,
    )
    if not ok:
        return JSONResponse(status_code=401, content={"status":"error","message":reason})

    # 2) 병합
    try:
        out_path, total_bytes_actual, ldt_path = _merge_parts(x_upload_id, member_id, highlight_key, file_name)
    except (FileNotFoundError, ValueError) as e:
        return JSONResponse(status_code=422, content={"status":"error","message":str(e)})

    # 🧾 총 바이트 교차검증(선택)
    if isinstance(total_bytes_decl, int) and total_bytes_decl > 0:
        if abs(total_bytes_decl - total_bytes_actual) > 0:
            logger.warning(f"[complete] total bytes mismatch: decl={total_bytes_decl} actual={total_bytes_actual}")

    # 3) 공개 URL + 길이
    public_url = f"{settings.STATIC_BASE_URL.rstrip('/')}/{member_id}/{highlight_key}/{ldt_path}/{out_path.name}"
    try:
        duration = get_duration(out_path)
    except Exception as e:
        logger.debug(f"[complete] get_duration failed: {e}")
        duration = 0.0

    # 완료 신호(PUB) — key는 highlightKey 사용
    try:
        await ProgressBus.publish_kv(
            job_id=highlight_key,
            value={
                "type": PROGRESS_TYPE.UPLOAD_COMPLETE,
                "progress": 1.0,
                "sizeBytes": int(total_bytes_actual),
                "sourceUrl": public_url,
                "folder": ldt_path,  # LocalDateTime 폴더명 (YYYY-MM-DDTHH-MM-SS)
                "timestampMs": _now_ms(),
            },
        )
    except Exception as e:
        logger.debug(f"[complete] progress publish skip: {e}")

    # 4) 응답
    return _ok({
        "type": "UPLOAD_COMPLETE",
        "memberId": member_id,
        "highlightKey": highlight_key,
        "uploadId": x_upload_id,
        "sizeBytes": int(total_bytes_actual),
        "sourceUrl": public_url,
        "durationSec": round(float(duration), 3),
        "folder": ldt_path,                      # ✅ 어디 폴더에 저장됐는지 바로 반환
        "localDateTime": _now_local_str(),       # ✅ 사람이 읽는 LDT (로컬)
        "message": "original video received",
        "timestamp": _now_ms(),
    })
