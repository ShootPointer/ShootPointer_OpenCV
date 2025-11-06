# app/core/crypto.py
from __future__ import annotations

import base64
import json
import os
import hmac
import hashlib
import time
import logging
from urllib.parse import unquote_plus

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 공통: 상수시간 비교 & 시크릿 선택
# ─────────────────────────────────────────────────────────────
def consteq(a: str, b: str) -> bool:
    """상수시간 비교 (b64url/hex 여부와 무관하게 '문자열 그대로' 비교)"""
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False

def _get_presigned_secret() -> str:
    """
    Presigned 업로드용 시크릿 선택 규칙:
    1) settings.PRE_SIGNED_SECRET 있으면 우선 사용
    2) 없으면 settings.BACKEND_SECRET (레거시 호환)
    """
    sec = getattr(settings, "PRE_SIGNED_SECRET", "") or getattr(settings, "BACKEND_SECRET", "")
    if not sec:
        logger.warning("[crypto] PRE_SIGNED_SECRET/BACKEND_SECRET both empty")
    return sec

# ─────────────────────────────────────────────────────────────
# HEX 포맷 HMAC (레거시: 백엔드 서명 검증에 사용)
# ─────────────────────────────────────────────────────────────
def hmac_sha256_hex(secret: str, message: str) -> str:
    key = secret.encode("utf-8") if not isinstance(secret, (bytes, bytearray)) else secret
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()

# ─────────────────────────────────────────────────────────────
# Base64URL 포맷 HMAC (Presigned 청크 업로드에 권장)
# ─────────────────────────────────────────────────────────────
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def _b64url_decode(data: str) -> bytes:
    data = data.strip()
    pad = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + pad)

def hmac_sha256_b64url(secret: str, message: str) -> str:
    key = secret.encode("utf-8") if not isinstance(secret, (bytes, bytearray)) else secret
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    return _b64url(mac)

# ─────────────────────────────────────────────────────────────
# AES-GCM 기반 highlightKey 토큰 (nonce.ciphertext.tag)
# ─────────────────────────────────────────────────────────────
def _derive_aes_key(secret: str) -> bytes:
    if isinstance(secret, (bytes, bytearray)):
        secret_bytes = bytes(secret)
    else:
        secret_bytes = secret.encode("utf-8")
    if not secret_bytes:
        raise ValueError("secret required")
    # AESGCM 키 길이 = 16/24/32 bytes → SHA256 파생으로 32바이트 고정
    return hashlib.sha256(secret_bytes).digest()

def encrypt_highlight_key(highlight_key: str, expires_ms: int | float) -> str:
    """highlightKey 및 만료(ms)를 AES-GCM으로 암호화해 토큰 생성."""
    highlight_key = str(highlight_key or "").strip()
    if not highlight_key:
        raise ValueError("highlight_key required")
    try:
        exp_ms = int(expires_ms)
    except Exception as exc:
        raise ValueError("expires_ms must be int") from exc
    if exp_ms <= 0:
        raise ValueError("expires_ms must be positive")

    payload = json.dumps({"highlightKey": highlight_key, "expiresMs": exp_ms}, separators=(",", ":")).encode("utf-8")

    secret = _get_presigned_secret()
    if not secret:
        raise ValueError("presigned secret empty")

    key = _derive_aes_key(secret)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    cipher = aes.encrypt(nonce, payload, None)
    # cryptography AESGCM.encrypt → ciphertext + tag(16B)
    ciphertext, tag = cipher[:-16], cipher[-16:]
    return ".".join((_b64url(nonce), _b64url(ciphertext), _b64url(tag)))

def issue_highlight_token(highlight_key: str, ttl_seconds: int | float) -> str:
    """TTL(초)을 받아 highlightKey 토큰을 발급한다."""
    try:
        ttl_seconds = float(ttl_seconds)
    except Exception as exc:
        raise ValueError("ttl_seconds must be numeric") from exc

    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    ttl_ms = int(ttl_seconds * 1000)
    now_ms = int(time.time() * 1000)
    expires_ms = now_ms + ttl_ms

    return encrypt_highlight_key(highlight_key, expires_ms)

def decrypt_highlight_key(token: str) -> tuple[bool, str]:
    """AES-GCM 토큰을 복호화해 highlightKey를 반환."""
    token = str(token or "").strip()
    if not token:
        return False, "missing_token"

    secret = _get_presigned_secret()
    if not secret:
        return False, "invalid_token"

    parts = token.split(".")
    if len(parts) != 3:
        return False, "invalid_token"

    try:
        nonce = _b64url_decode(parts[0])
        ciphertext = _b64url_decode(parts[1])
        tag = _b64url_decode(parts[2])
    except Exception:
        return False, "invalid_token"

    if len(nonce) != 12:
        return False, "invalid_token"

    try:
        key = _derive_aes_key(secret)
        aes = AESGCM(key)
        plaintext = aes.decrypt(nonce, ciphertext + tag, None)
    except Exception:
        return False, "invalid_token"

    try:
        data = json.loads(plaintext.decode("utf-8"))
    except Exception:
        return False, "invalid_token"

    highlight_key = str(data.get("highlightKey", "")).strip()
    if not highlight_key:
        return False, "invalid_token"

    try:
        expires_ms = int(data.get("expiresMs", 0))
    except Exception:
        return False, "invalid_token"

    if expires_ms <= 0:
        return False, "invalid_token"

    now_ms = int(time.time() * 1000)
    if now_ms > expires_ms:
        return False, "expired"

    return True, highlight_key


def verify_highlight_token(token: str) -> tuple[bool, str]:
    """decrypt_highlight_key 래퍼 — 기존 verify_* 패턴과 이름을 맞춘다."""
    return decrypt_highlight_key(token)

# ─────────────────────────────────────────────────────────────
# 레거시: 만료(ms) + memberId 기반 presigned 검증
#   - 메시지 포맷: "expiresMs:memberId"
#   - 서명 포맷: hex
# ─────────────────────────────────────────────────────────────
def verify_presigned_ms(expires_ms: str, member_id: str, signature_hex: str) -> tuple[bool, str]:
    """
    presigned URL 검증 (만료시간: epoch milliseconds)
    반환: (ok, reason)  # ok가 False면 reason에 사유 문자열
    """
    if not expires_ms or not member_id or not signature_hex:
        return False, "missing_parameter"

    # 1) 만료 파싱 (ms 길이의 정수)
    try:
        exp_ms = int(expires_ms)  # long
    except ValueError:
        return False, "invalid_expires"

    # 2) 만료 체크 (현재 ms와 비교)
    now_ms = int(time.time() * 1000)
    if now_ms > exp_ms:
        return False, "expired"

    # 3) 메시지 구성 (백엔드와 동일 포맷)
    mid = unquote_plus(member_id)
    msg = f"{expires_ms}:{mid}"

    # 4) 시그니처 상수시간 비교 (hex)
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
# Presigned 청크 업로드용 시그니처 빌더/검증자 (Base64URL)
#   - 청크 PUT:  msg = "expires:memberId:jobId:uploadId:partNumber"
#   - 완료 POST: msg = "expires:memberId:jobId:uploadId"
#   - 서명 포맷: base64url( HMAC-SHA256(msg) )
# ─────────────────────────────────────────────────────────────
def make_chunk_signature_b64url(expires_ms: str, member_id: str, job_id: str, upload_id: str, part_number: str | int) -> str:
    """
    청크 업로드용 기대 서명 생성 (Base64URL).
    """
    msg = f"{expires_ms}:{member_id}:{job_id}:{upload_id}:{part_number}"
    return hmac_sha256_b64url(_get_presigned_secret(), msg)

def make_complete_signature_b64url(expires_ms: str, member_id: str, job_id: str, upload_id: str) -> str:
    """
    병합 완료용 기대 서명 생성 (Base64URL).
    """
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
    """
    청크 PUT 서명 검증 (Base64URL).
    """
    # 만료 검사
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
    """
    병합 완료 POST 서명 검증 (Base64URL).
    """
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
