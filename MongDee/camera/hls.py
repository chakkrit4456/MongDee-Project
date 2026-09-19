"""HLS adapter — relies on OpenCV's own bundled FFmpeg to demux .m3u8
playlists directly. opencv-python ships FFmpeg support even on a machine
with no system `ffmpeg` binary installed (verified against the venv used
for this project: `cv2.getBuildInformation()` reports
"FFMPEG: YES (prebuilt binaries)" even though `ffmpeg` is not on PATH).

URL format: any http(s):// URL ending in .m3u8 (master or media playlist).
"""

from __future__ import annotations

import cv2

from camera.base import CameraConfig, CameraConnectionError, CameraSource, enable_hw_video_acceleration, redact_url


class HLSCamera(CameraSource):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        if not config.url:
            raise ValueError(f"HLS camera {config.id!r}: config.url is required")
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.config.url, cv2.CAP_FFMPEG)
        if self.config.hw_accel:
            enable_hw_video_acceleration(cap)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.config.connect_timeout_sec * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.config.read_timeout_sec * 1000))
        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(
                f"HLS camera {self.config.id!r}: could not open playlist at {redact_url(self.config.url)}"
            )
        self._cap = cap

    def read(self):
        if self._cap is None:
            raise CameraConnectionError(f"HLS camera {self.config.id!r}: read() called before open()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraConnectionError(
                f"HLS camera {self.config.id!r}: read() failed at {redact_url(self.config.url)}"
            )
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
