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

# 로거 설정: 복호화 과정을 상세히 추적합니다.
logger = logging.getLogger("AESGCMCrypto") 
logger.setLevel(logging.INFO) # INFO 레벨 이상으로 설정하여 상세 로그를 확인합니다.

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
    """Base64URL 디코딩 (Python 표준 라이브러리를 통해 패딩 자동 처리)"""
    s = (data or "").strip()
    logger.debug(f"Attempting Base64URL decode for input length: {len(s)}")
    try:
        # Python의 urlsafe_b64decode는 패딩 없는 경우에도 견고함
        decoded = base64.urlsafe_b64decode(s)
        logger.debug(f"Base64URL decoded successfully. Result length: {len(decoded)} bytes.")
        return decoded
    except Exception as e:
        logger.error(f"Base64URL decode failed. Input: '{s[:30]}...'. Error: {e}")
        raise ValueError("Invalid Base64URL encoding")

def _get_aes_key() -> bytes:
    """
    settings.AES_GCM_SECRET를 32바이트 키로 변환하고 과정을 로깅합니다.
    """
    raw = (getattr(settings, "AES_GCM_SECRET", "") or "").strip()
    if not raw:
        logger.critical("AES_GCM_SECRET is missing in settings. AUTHENTICATION WILL FAIL.")
        raise RuntimeError("Missing AES_GCM_SECRET configuration.")
    
    logger.info(f"Key material received (length: {len(raw)}). Attempting to derive 32-byte key.")

    # 1. 64 hex → 32 bytes 시도
    if re.fullmatch(r"[0-9a-fA-F]{64}", raw):
        try:
            key_bytes = bytes.fromhex(raw)
            if len(key_bytes) == 32:
                logger.info("Key successfully decoded from 64-char HEX to 32 bytes (256 bits).")
                return key_bytes
            
            # 길이 불일치
            logger.error(f"Hex key length mismatch. Decoded to {len(key_bytes)} bytes, expected 32.")
            raise ValueError("Invalid AES key length from hex.")
        except Exception:
            # hex 디코딩 자체에 실패하면 다음 Base64 시도로 넘어갑니다.
            logger.warning("Hex decoding failed. Trying Base64/Base64URL.")
            pass # continue to next decoding attempt

    # 2. Base64 / URL-safe Base64 시도
    try:
        pad = "=" * ((4 - (len(raw) % 4)) % 4)
        
        # Base64URL 디코딩 시도
        try:
            key_bytes = base64.urlsafe_b64decode(raw + pad)
            decode_method = "Base64URL"
        except binascii.Error:
            # 일반 Base64 디코딩 시도
            key_bytes = base64.b64decode(raw + pad)
            decode_method = "Base64"

        if len(key_bytes) != 32:
            logger.error(f"{decode_method} key length mismatch. Decoded to {len(key_bytes)} bytes, expected 32.")
            raise ValueError("Invalid AES key length from Base64.")
        
        logger.info(f"Key successfully decoded from {decode_method} to 32 bytes (256 bits).")
        return key_bytes
    except Exception as e:
        logger.critical(f"Key derivation FAILED. Invalid format or length. Error: {e}")
        raise ValueError("Invalid format or length for AES_GCM_SECRET.")

# ─────────────────────────────────────────────────────────────
# 핵심 암호화 클래스
# ─────────────────────────────────────────────────────────────

class AESGCMCrypto:
    """
    AES-GCM 복호화 및 검증 서비스.
    """
    def __init__(self):
        self._key = _get_aes_key()
        logger.info("AESGCMCrypto service initialized with key.")

    def decrypt_token(self, presigned_token: str) -> DecryptedToken:
        """
        presigned_token(Base64URL) 복호화 및 검증.
        """
        logger.info(f"Starting token decryption. Token length: {len(presigned_token)}")
        
        # 1) 토큰 Base64URL 디코딩
        try:
            combined = _b64url_decode(presigned_token)
        except ValueError:
            raise ValueError("Invalid Base64URL encoding of the presigned token")

        # 2) 길이 검증 및 분리
        combined_len = len(combined)
        min_len = IV_LENGTH + TAG_LENGTH + 1
        
        if combined_len < min_len:
            logger.error(f"Token size check failed: combined length {combined_len} bytes. Min required: {min_len}.")
            raise ValueError("Token too short to contain IV, Ciphertext, and Tag.")

        iv = combined[:IV_LENGTH]
        tag = combined[-TAG_LENGTH:]
        cipher_bytes = combined[IV_LENGTH:-TAG_LENGTH]
        
        logger.info(
            f"Token split successful. IV: {len(iv)} bytes, Ciphertext: {len(cipher_bytes)} bytes, Tag: {len(tag)} bytes."
        )

        if len(iv) != IV_LENGTH:
             logger.error(f"IV length check failed: {len(iv)} bytes, expected {IV_LENGTH}.")
             raise ValueError("Invalid AES-GCM Nonce/IV length (must be 12 bytes)")

        # 3) 복호화 + 무결성 검증
        try:
            logger.info("Attempting AES-GCM decrypt_and_verify...")
            cipher = AES.new(self._key, AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(cipher_bytes, tag)
            message = plaintext.decode("utf-8")
            logger.info("Decryption and verification successful.")
        except Exception as e:
            logger.error(
                f"AES-GCM verification FAILED. Key/Token mismatch suspected. IV/Tag integrity error: {e}"
            )
            # 복호화 실패 시, 보안상의 이유로 상세한 정보를 외부에 노출하지 않습니다.
            raise ValueError("Presigned signature verification failed (Integrity check error).")

        # 4) 파싱
        try:
            parts = message.split(":", 3)
            if len(parts) != 4:
                logger.error(f"Message parsing FAILED. Expected 4 parts, got {len(parts)}.")
                logger.error(f"Raw message: '{message}'. Parts: {parts}")
                raise ValueError("Invalid message format in presigned token")
            
            expires_str, member_id, job_id, file_name = parts
            expires = int(expires_str)
            logger.info(f"Message parsed successfully. Expiry: {expires}, Member: {member_id}, Job: {job_id}.")
        except Exception as e:
            logger.error(f"Message content parsing FAILED. Error: {e}")
            raise ValueError("Invalid message content or format in presigned token")

        # 5) 만료 확인
        current_time = int(time.time())
        if expires < current_time:
            logger.error(f"Token expired. Expiry time: {expires}, Current time: {current_time}.")
            raise ValueError("Presigned expired")
            
        logger.info("Token validation successful and not expired.")

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