"""USB webcam adapter.

Config: protocol="usb", device_index=<int> (0 = first webcam, matching
cv2's own numbering — see camera/README.md for how to find the index for
a specific USB camera when more than one is plugged in).
"""

from __future__ import annotations

import sys

import cv2

from camera.base import CameraConfig, CameraConnectionError, CameraSource


class USBCamera(CameraSource):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        # DirectShow avoids a multi-second MSMF enumeration stall on Windows;
        # CAP_ANY lets OpenCV pick the right backend elsewhere (Video4Linux2, AVFoundation).
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.config.device_index, backend)
        # Negotiate compressed USB transport before the first frame.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.request_width or 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.request_height or 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.config.request_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.request_width)
        if self.config.request_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.request_height)
        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(
                f"USB camera {self.config.id!r}: device index {self.config.device_index} did not open "
                f"(unplugged, in use by another process, or index out of range)"
            )
        self._cap = cap

    def read(self):
        if self._cap is None:
            raise CameraConnectionError(f"USB camera {self.config.id!r}: read() called before open()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraConnectionError(f"USB camera {self.config.id!r}: read() failed (device disconnected?)")
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def native_resolution(self):
        if self._cap is None:
            return None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if w and h else None
