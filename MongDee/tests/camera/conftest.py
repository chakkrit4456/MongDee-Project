"""Shared test helpers for the camera package's test suite."""

from __future__ import annotations

import cv2
import numpy as np


class FakeVideoCapture:
    """Stand-in for cv2.VideoCapture, used by the USB/RTSP/HLS adapter
    tests so they exercise the adapters' real open/read/release/reconnect
    logic without needing a real camera or network stream."""

    def __init__(self, source, backend=None, *, opens=True, frames=None, width=640, height=480):
        self.source = source
        self.backend = backend
        self._opens = opens
        if frames is None:
            frames = [np.zeros((height, width, 3), dtype=np.uint8)]
        self._frames = list(frames)
        self._read_index = 0
        self._props: dict[int, float] = {}
        self.released = False

    def isOpened(self):
        return self._opens

    def set(self, prop, value):
        self._props[prop] = value
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frames[0].shape[1] if self._frames else 0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frames[0].shape[0] if self._frames else 0
        return self._props.get(prop, 0)

    def read(self):
        if self._read_index >= len(self._frames):
            return False, None
        frame = self._frames[self._read_index]
        self._read_index += 1
        return True, frame

    def release(self):
        self.released = True
