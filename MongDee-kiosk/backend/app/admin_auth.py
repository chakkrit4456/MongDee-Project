"""Admin PIN that must accompany every request that deletes data.

Enforced on the server (an `X-Admin-Pin` header), not just in the UI, because
the API is otherwise open to anyone at the kiosk. The PIN is stored salted and
hashed (scrypt) in the settings table. Repeated wrong guesses lock the check
for a minute so a keyboard at the booth can't be used to brute-force it.
"""
import hashlib
import hmac
import os
import re
import time
from collections import deque
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app import db

PIN_KEY = "admin_pin"
PIN_PATTERN = re.compile(r"^\d{4,12}$")
MAX_FAILURES = 5
FAILURE_WINDOW_S = 300.0
LOCK_S = 60.0

_failures: deque = deque()
_locked_until = 0.0


def _hash(pin: str, salt: bytes) -> bytes:
    return hashlib.scrypt(pin.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)


def pin_is_set() -> bool:
    return db.get_setting(PIN_KEY) is not None


def set_pin(pin: str) -> None:
    if not PIN_PATTERN.match(pin or ""):
        raise HTTPException(400, {"code": "pin_format", "message": "รหัสต้องเป็นตัวเลข 4–12 หลัก"})
    salt = os.urandom(16)
    db.set_setting(PIN_KEY, f"scrypt${salt.hex()}${_hash(pin, salt).hex()}")


def clear_pin() -> None:
    db.delete_setting(PIN_KEY)


def _matches(pin: Optional[str]) -> bool:
    stored = db.get_setting(PIN_KEY)
    if not stored or not pin:
        return False
    try:
        _, salt_hex, hash_hex = stored.split("$")
        return hmac.compare_digest(_hash(pin, bytes.fromhex(salt_hex)), bytes.fromhex(hash_hex))
    except ValueError:
        return False


def reset_lockout() -> None:
    global _locked_until
    _failures.clear()
    _locked_until = 0.0


def check_pin(pin: Optional[str], now: Optional[float] = None) -> None:
    """Returns silently for the right PIN; raises 429 while locked, 401 when wrong."""
    global _locked_until
    now = time.monotonic() if now is None else now
    if now < _locked_until:
        wait = int(_locked_until - now) + 1
        raise HTTPException(429, {"code": "pin_locked", "message": f"ใส่รหัสผิดหลายครั้ง กรุณารอ {wait} วินาที", "retry_after": wait})
    if _matches(pin):
        _failures.clear()
        return
    _failures.append(now)
    while _failures and now - _failures[0] > FAILURE_WINDOW_S:
        _failures.popleft()
    if len(_failures) >= MAX_FAILURES:
        _locked_until = now + LOCK_S
        _failures.clear()
    raise HTTPException(401, {"code": "pin_invalid", "message": "รหัสไม่ถูกต้อง"})


def require_pin(x_admin_pin: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency for routes that delete data."""
    if not pin_is_set():
        raise HTTPException(403, {"code": "pin_not_set", "message": "ยังไม่ได้ตั้งรหัสผ่านแอดมิน"})
    check_pin(x_admin_pin)


router = APIRouter(prefix="/api/admin", tags=["admin"])


class PinBody(BaseModel):
    new_pin: str
    current_pin: Optional[str] = None


@router.get("/pin")
def pin_status():
    return {"is_set": pin_is_set()}


@router.post("/pin")
def change_pin(body: PinBody):
    """First call sets the PIN; afterwards the current PIN is required to change it."""
    if pin_is_set():
        check_pin(body.current_pin)
    set_pin(body.new_pin)
    return {"is_set": True}
