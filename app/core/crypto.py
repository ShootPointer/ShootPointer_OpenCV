# app/core/crypto.py
from __future__ import annotations

import base64
import binascii
import time
import logging
import re
from typing import NamedTuple

# PyCryptodome 사용 (pip install pycryptodome 필요)
from Crypto.Cipher import AES
from app.core.config import settings

logger = logging.getLogger(__name__)

# GCM 고정 태그 길이
TAG_LENGTH = 16
# 표준 GCM Nonce/IV 길이(권장 12바이트)
IV_LENGTH = 12

# ─────────────────────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────────────────────

class DecryptedToken(NamedTuple):
    expires: int
    memberId: str
    jobId: str
    fileName: str

# ─────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────

def _b64url_decode(data: str) -> bytes:
    """Base64URL 디코딩 (패딩 자동 보정)"""
    s = (data or "").strip()
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)

def _get_aes_key() -> bytes:
    """
    settings.AES_GCM_SECRET를 32바이트 키로 변환.
    - 64글자 hex면 hex 디코드
    - 아니면 base64/base64url로 디코드
    """
    raw = (getattr(settings, "AES_GCM_SECRET", "") or "").strip()
    if not raw:
        logger.error("AES_GCM_SECRET is missing in settings. AUTHENTICATION WILL FAIL.")
        raise RuntimeError("Missing AES_GCM_SECRET configuration.")

    # 64 hex → 32 bytes
    if re.fullmatch(r"[0-9a-fA-F]{64}", raw):
        try:
            key_bytes = bytes.fromhex(raw)
            if len(key_bytes) != 32:
                logger.error("Hex AES_GCM_SECRET does not decode to 32 bytes.")
                raise ValueError("Invalid AES key length.")
            return key_bytes
        except Exception as e:
            logger.error(f"Failed to hex-decode AES_GCM_SECRET: {e}")
            raise ValueError("Invalid hex for AES_GCM_SECRET.")

    # Base64 / URL-safe Base64
    try:
        pad = "=" * ((4 - (len(raw) % 4)) % 4)
        try:
            key_bytes = base64.urlsafe_b64decode(raw + pad)
        except binascii.Error:
            key_bytes = base64.b64decode(raw + pad)

        if len(key_bytes) != 32:
            logger.error("AES_GCM_SECRET is not 32 bytes (256 bits) long.")
            raise ValueError("Invalid AES key length.")
        return key_bytes
    except Exception as e:
        logger.error(f"Failed to decode AES_GCM_SECRET: {e}")
        raise ValueError("Invalid format for AES_GCM_SECRET.")

# ─────────────────────────────────────────────────────────────
# 핵심 암호화 클래스
# ─────────────────────────────────────────────────────────────

class AESGCMCrypto:
    """
    AES-GCM 복호화 및 검증 서비스.
    """
    def __init__(self):
        self._key = _get_aes_key()

    def decrypt_token(self, presigned_token: str) -> DecryptedToken:
        """
        presigned_token(Base64URL) → [IV(12)] [Ciphertext] [Tag(16)] 분리 후
        AES-GCM decrypt_and_verify로 복호화/검증.
        평문 포맷: "expires:memberId:jobId:fileName"
        """
        # 1) 토큰 Base64URL 디코딩
        try:
            combined = _b64url_decode(presigned_token)
        except Exception:
            raise ValueError("Invalid Base64URL encoding of the presigned token")

        # 2) 길이 검증 및 분리
        if len(combined) < IV_LENGTH + TAG_LENGTH + 1:
            # 최소 한 바이트 이상의 ciphertext가 있어야 함
            raise ValueError("Token too short to contain IV, Ciphertext, and Tag.")

        iv = combined[:IV_LENGTH]
        tag = combined[-TAG_LENGTH:]
        cipher_bytes = combined[IV_LENGTH:-TAG_LENGTH]

        if len(iv) != IV_LENGTH:
            raise ValueError("Invalid AES-GCM Nonce/IV length (must be 12 bytes)")

        # 3) 복호화 + 무결성 검증
        try:
            cipher = AES.new(self._key, AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(cipher_bytes, tag)
            message = plaintext.decode("utf-8")
        except Exception as e:
            logger.warning(f"AES-GCM verification failed (Integrity check error): {e}")
            raise ValueError("Presigned signature verification failed (Integrity check error)")

        # 4) 파싱
        try:
            parts = message.split(":", 3)
            if len(parts) != 4:
                logger.error(f"Invalid message format: {message}")
                raise ValueError("Invalid message format in presigned token")
            expires_str, member_id, job_id, file_name = parts
            expires = int(expires_str)
        except Exception:
            raise ValueError("Invalid message content or format in presigned token")

        # 5) 만료 확인
        if expires < int(time.time()):
            raise ValueError("Presigned expired")

        return DecryptedToken(
            expires=expires,
            memberId=member_id,
            jobId=job_id,
            fileName=file_name,
        )

# ─────────────────────────────────────────────────────────────
# FastAPI 의존성
# ─────────────────────────────────────────────────────────────

def get_crypto_service() -> AESGCMCrypto:
    """FastAPI Depends()에서 사용할 암호화 서비스 인스턴스."""
    return AESGCMCrypto()
