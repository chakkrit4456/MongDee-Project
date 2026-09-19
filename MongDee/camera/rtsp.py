"""RTSP adapter — also used internally by ONVIFCamera once it has resolved
a concrete stream URI (see camera/onvif.py).

URL format: rtsp://[username:password@]host[:port]/path
e.g. rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101
"""

from __future__ import annotations

import cv2

from camera.base import CameraConfig, CameraConnectionError, CameraSource, enable_hw_video_acceleration, redact_url


class RTSPCamera(CameraSource):
    def __init__(self, config: CameraConfig, url: str | None = None):
        super().__init__(config)
        # `url` override lets ONVIFCamera hand this class a resolved stream
        # URI without needing to mutate the (frozen) CameraConfig.
        self._url = url if url is not None else config.url
        if not self._url:
            raise ValueError(f"RTSP camera {config.id!r}: config.url is required")
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        if self.config.hw_accel:
            enable_hw_video_acceleration(cap)
        # Not every OpenCV/FFmpeg build honours these two, but when it does
        # they stop a dead RTSP session from hanging read() forever
        # (Master Prompt section 32: "timeout handling").
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.config.connect_timeout_sec * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.config.read_timeout_sec * 1000))
        # Keep OpenCV's own internal buffer at 1 frame, so a slow consumer
        # gets a recent frame instead of opencv silently queuing stale ones.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(
                f"RTSP camera {self.config.id!r}: could not open stream at {redact_url(self._url)}"
            )
        self._cap = cap

    def read(self):
        if self._cap is None:
            raise CameraConnectionError(f"RTSP camera {self.config.id!r}: read() called before open()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraConnectionError(
                f"RTSP camera {self.config.id!r}: read() failed at {redact_url(self._url)}"
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
