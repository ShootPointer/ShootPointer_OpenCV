# app/core/crypto.py
from __future__ import annotations
import hmac, hashlib
import time
import logging
from urllib.parse import unquote_plus
from app.core.config import settings

logger = logging.getLogger(__name__)

def hmac_sha256_hex(secret: str, message: str) -> str:
    key = secret.encode("utf-8") if not isinstance(secret, (bytes, bytearray)) else secret
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()

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

    # 4) 시그니처 상수시간 비교
    expected = hmac_sha256_hex(settings.BACKEND_SECRET, msg)
    if not hmac.compare_digest(expected, signature_hex.lower()):
        logger.warning(
            "presign signature mismatch",
            extra={"memberId": mid, "expiresMs": expires_ms, "sigRecv": signature_hex, "sigExp": expected}
        )
        return False, "invalid_signature"

    return True, "ok"
