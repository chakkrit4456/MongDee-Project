"""Webcam open/release helpers for the standalone detect.py demo.

Reuses core.vision's already-hardened `_open_capture` (tries
DirectShow/MSMF/V4L2 backends and several MJPEG/resolution profiles, and
requires one real frame before accepting a handle — see that module's own
docstring for why) instead of re-implementing camera-quirk handling here,
so detect.py opens webcams exactly as reliably as the main booth app does.
"""

from __future__ import annotations

import cv2

from core.vision import _open_capture


class Camera:
    """One webcam. `label` (e.g. "Camera 0") tags this device's own log
    lines/on-screen status so a two-camera run can tell them apart."""

    def __init__(self, device, label: str):
        self.device = device
        self.label = label
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        cap = _open_capture(self.device, label=self.label)
        if not cap.isOpened():
            cap.release()
            self._cap = None
            return False
        self._cap = cap
        return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
