# app/core/crypto.py
from __future__ import annotations
import base64
import hashlib
import hmac
from typing import Tuple

def _maybe_b64decode(s: str) -> bytes:
    """
    secretKey가 base64일 수도, 그냥 평문일 수도 있다고 해서
    둘 다 허용: base64가 아니면 UTF-8바이트로 사용.
    """
    try:
        # urlsafe/base64 모두 허용 (패딩 없어도 시도)
        pad = '=' * (-len(s) % 4)
        return base64.urlsafe_b64decode((s + pad).encode("utf-8"))
    except Exception:
        return s.encode("utf-8")

def hmac_sha256(secret: str, message: str) -> Tuple[str, str, str]:
    """
    HMAC-SHA256(비밀키=secret, 메시지=message)을 계산.
    반환: (hex, base64, base64url)
    """
    key = _maybe_b64decode(secret)
    mac = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    hex_ = mac.hex()
    b64 = base64.b64encode(mac).decode("utf-8")            # 표준 base64(+/), 패딩 포함
    b64url = base64.urlsafe_b64encode(mac).decode("utf-8") # URL-safe(-_), 패딩 포함
    return hex_, b64, b64url

def hmac_compare(secret: str, message: str, given_sig: str) -> bool:
    """
    주어진 signature(given_sig)가 hex / base64 / base64url(패딩 유무 상관없음)
    어느 형식이든 매칭되면 True.
    """
    hex_, b64, b64url = hmac_sha256(secret, message)

    # 패딩 제거/유지 양쪽 비교
    cands = {
        hex_,
        b64, b64.rstrip("="),
        b64url, b64url.rstrip("="),
    }
    # 대소문자/공백 방어
    normalized = (given_sig or "").strip()
    if normalized.lower() in {h.lower() for h in cands}:
        return True
    return False
