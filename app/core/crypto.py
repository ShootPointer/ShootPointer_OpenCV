# app/core/crypto.py
from __future__ import annotations

import base64
import hmac
import hashlib
import time
import logging
import json
from urllib.parse import unquote_plus
from typing import Tuple, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 공통: 상수시간 비교 & 시크릿 헬퍼
# ─────────────────────────────────────────────────────────────
def consteq(a: str, b: str) -> bool:
    """상수시간 비교 (문자열 그대로 비교)"""
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False

def _get_presigned_secret() -> str:
    """
    Presigned/Token 서명을 위한 시크릿(문자열) 선택 규칙:
    1) settings.PRE_SIGNED_SECRET 우선
    2) 없으면 settings.BACKEND_SECRET (레거시 호환)
    """
    sec = getattr(settings, "PRE_SIGNED_SECRET", "") or getattr(settings, "BACKEND_SECRET", "")
    if not sec:
        logger.warning("[crypto] PRE_SIGNED_SECRET/BACKEND_SECRET both empty")
    return sec

def _secret_to_bytes(auto: str) -> bytes:
    """
    시크릿이 평문일 수도, base64url 인코딩일 수도 있음.
    - 우선 base64url 디코딩을 시도
    - 실패하면 utf-8 바이트로 사용
    """
    if not auto:
        return b""
    s = auto
    # base64url 패딩 보정
    try:
        pad = 4 - (len(s) % 4)
        if pad and pad < 4:
            s_padded = s + "=" * pad
        else:
            s_padded = s
        return base64.urlsafe_b64decode(s_padded.encode("ascii"))
    except Exception:
        return auto.encode("utf-8")

# ─────────────────────────────────────────────────────────────
# Base64URL helper (no padding)
# ─────────────────────────────────────────────────────────────
def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

def _b64url_decode(s: str) -> bytes:
    pad = 4 - (len(s) % 4)
    if pad and pad < 4:
        s = s + "=" * pad
    return base64.urlsafe_b64decode(s.encode("ascii"))

# ─────────────────────────────────────────────────────────────
# HEX 포맷 HMAC (레거시)
# ─────────────────────────────────────────────────────────────
def hmac_sha256_hex(secret: str, message: str) -> str:
    key = secret.encode("utf-8") if not isinstance(secret, (bytes, bytearray)) else secret
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()

# ─────────────────────────────────────────────────────────────
# Base64URL 포맷 HMAC (권장)
# ─────────────────────────────────────────────────────────────
def hmac_sha256_b64url(secret: str, message: str) -> str:
    key = secret.encode("utf-8") if not isinstance(secret, (bytes, bytearray)) else secret
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(mac)

# ─────────────────────────────────────────────────────────────
# 레거시 presigned 검증 (ms 만료 + memberId)
# ─────────────────────────────────────────────────────────────
def verify_presigned_ms(expires_ms: str, member_id: str, signature_hex: str) -> tuple[bool, str]:
    """
    presigned URL 검증 (만료시간: epoch milliseconds)
    반환: (ok, reason)
    """
    if not expires_ms or not member_id or not signature_hex:
        return False, "missing_parameter"

    try:
        exp_ms = int(expires_ms)
    except ValueError:
        return False, "invalid_expires"

    now_ms = int(time.time() * 1000)
    if now_ms > exp_ms:
        return False, "expired"

    mid = unquote_plus(member_id)
    msg = f"{expires_ms}:{mid}"

    secret = getattr(settings, "BACKEND_SECRET", "")
    expected = hmac_sha256_hex(secret, msg)
    if not consteq(expected.lower(), signature_hex.lower()):
        logger.warning(
            "presign signature mismatch",
            extra={"memberId": mid, "expiresMs": expires_ms, "sigRecv": signature_hex, "sigExp": expected}
        )
        return False, "invalid_signature"

    return True, "ok"

# ─────────────────────────────────────────────────────────────
# Presigned 청크/완료 서명 (Base64URL)
#   - 청크 PUT:  msg = "expires:memberId:jobId:uploadId:partNumber"
#   - 완료 POST: msg = "expires:memberId:jobId:uploadId"
#   - 여기서 jobId 자리에 highlightKey를 넘겨도 호환되게 설계됨
# ─────────────────────────────────────────────────────────────
def make_chunk_signature_b64url(expires_ms: str, member_id: str, job_id: str, upload_id: str, part_number: str | int) -> str:
    msg = f"{expires_ms}:{member_id}:{job_id}:{upload_id}:{part_number}"
    return hmac_sha256_b64url(_get_presigned_secret(), msg)

def make_complete_signature_b64url(expires_ms: str, member_id: str, job_id: str, upload_id: str) -> str:
    msg = f"{expires_ms}:{member_id}:{job_id}:{upload_id}"
    return hmac_sha256_b64url(_get_presigned_secret(), msg)

def verify_chunk_signature_b64url(
    expires_ms: str,
    member_id: str,
    job_id: str,
    upload_id: str,
    part_number: str | int,
    signature_b64url: str,
) -> tuple[bool, str]:
    try:
        exp = int(expires_ms)
    except Exception:
        return False, "invalid_expires"
    if int(time.time() * 1000) > exp:
        return False, "expired"

    expected = make_chunk_signature_b64url(expires_ms, member_id, job_id, upload_id, part_number)
    if not consteq(expected, signature_b64url):
        return False, "invalid_signature"
    return True, "ok"

def verify_complete_signature_b64url(
    expires_ms: str,
    member_id: str,
    job_id: str,
    upload_id: str,
    signature_b64url: str,
) -> tuple[bool, str]:
    try:
        exp = int(expires_ms)
    except Exception:
        return False, "invalid_expires"
    if int(time.time() * 1000) > exp:
        return False, "expired"

    expected = make_complete_signature_b64url(expires_ms, member_id, job_id, upload_id)
    if not consteq(expected, signature_b64url):
        return False, "invalid_signature"
    return True, "ok"

# ─────────────────────────────────────────────────────────────
# NEW: Highlight Token (JWT-like HS256)
# header.payload.signature (Base64URL, no padding)
# payload: memberId, highlightKey, exp (Unix seconds)
# ─────────────────────────────────────────────────────────────
_HEADER = {"alg": "HS256", "typ": "HLK"}

def sign_highlight_token(member_id: str, highlight_key: str, ttl_seconds: int) -> str:
    """
    HMAC-SHA256 서명 토큰 생성: header.payload.signature
    exp = now + ttl_seconds (초)
    시크릿은 PRE_SIGNED_SECRET(있으면) 또는 BACKEND_SECRET 사용.
    """
    now = int(time.time())
    payload = {
        "memberId": member_id,
        "highlightKey": highlight_key,
        "exp": now + int(ttl_seconds),
    }
    h_b = _b64url_encode(json.dumps(_HEADER, separators=(",", ":")).encode("utf-8"))
    p_b = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    msg = f"{h_b}.{p_b}".encode("ascii")

    key_b = _secret_to_bytes(_get_presigned_secret())
    sig = hmac.new(key_b, msg, hashlib.sha256).digest()
    s_b = _b64url_encode(sig)
    return f"{h_b}.{p_b}.{s_b}"

def verify_highlight_token(token: str) -> Tuple[bool, str, Dict[str, Any] | None]:
    """
    토큰 검증 + payload 반환
    return: (ok, reason, payload|None)
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False, "malformed token", None
        h_b, p_b, s_b = parts

        header = json.loads(_b64url_decode(h_b))
        if header.get("alg") != "HS256":
            return False, "unsupported alg", None

        msg = f"{h_b}.{p_b}".encode("ascii")
        key_b = _secret_to_bytes(_get_presigned_secret())
        expect = _b64url_encode(hmac.new(key_b, msg, hashlib.sha256).digest())
        if not consteq(expect, s_b):
            return False, "invalid signature", None

        payload = json.loads(_b64url_decode(p_b))
        member_id = str(payload.get("memberId") or "").strip()
        highlight_key = str(payload.get("highlightKey") or "").strip()
        exp = int(payload.get("exp") or 0)

        if not member_id or not highlight_key or exp <= 0:
            return False, "invalid payload", None

        now = int(time.time())
        if now > exp:
            return False, "token expired", None

        return True, "ok", payload
    except Exception as e:
        return False, str(e), None
