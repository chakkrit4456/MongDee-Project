"""In-memory store of captured-but-not-yet-confirmed product frames.

Shared by two flows: the enrollment page's own capture button, and the
kiosk recognition loop, which stashes a burst here whenever it can't
identify a product — so if staff choose to add it as new, they don't have
to recapture it, they just pick up the session id that's already loaded.
"""
from __future__ import annotations

import base64
import time
import uuid
from typing import Optional

import cv2
import numpy as np

SESSION_TTL_S = 300
_sessions: dict[str, dict] = {}


def _purge_expired() -> None:
    now = time.time()
    for sid in [sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL_S]:
        _sessions.pop(sid, None)


def create_session() -> str:
    _purge_expired()
    session_id = uuid.uuid4().hex
    _sessions[session_id] = {"frames": {}, "embeddings": {}, "created_at": time.time()}
    return session_id


def add_batch(session_id: str, frames: list[np.ndarray], embeddings: list[np.ndarray]) -> list[dict]:
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(session_id)
    session["created_at"] = time.time()  # touch to extend TTL while actively used

    added = []
    for frame, emb in zip(frames, embeddings):
        frame_id = uuid.uuid4().hex
        session["frames"][frame_id] = frame
        session["embeddings"][frame_id] = emb
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        thumb_url = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else ""
        added.append({"frame_id": frame_id, "thumbnail_data_url": thumb_url})
    return added


def get_session(session_id: str) -> Optional[dict]:
    _purge_expired()
    return _sessions.get(session_id)


def pop_session(session_id: str) -> Optional[dict]:
    _purge_expired()
    return _sessions.pop(session_id, None)


def discard_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
