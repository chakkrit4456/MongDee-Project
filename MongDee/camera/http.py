"""HTTP camera adapter — two modes:

  mjpeg      persistent multipart/x-mixed-replace push stream
             e.g. http://192.168.1.50/video  (a common IP-camera "MJPEG stream" URL)
  snapshot   poll a single-still-image URL on every read()
             e.g. http://192.168.1.50/snapshot.jpg

Gateway-level FPS throttling (CameraConfig.processing_fps, applied in
camera/gateway.py) controls how often read() is actually called in both
modes, so there is no separate polling-interval knob here.
"""

from __future__ import annotations

import cv2
import numpy as np
import requests

from camera.base import CameraConfig, CameraConnectionError, CameraSource, redact_url

_BOUNDARY_SCAN_CHUNK = 4096
_MAX_PART_BYTES = 8 * 1024 * 1024  # refuse to buffer more than 8MB looking for one JPEG frame


def _parse_boundary(content_type: str) -> bytes | None:
    parts = [p.strip() for p in content_type.split(";")]
    if not parts or "multipart" not in parts[0].lower():
        return None
    for p in parts[1:]:
        if p.lower().startswith("boundary="):
            boundary = p.split("=", 1)[1].strip().strip('"')
            return boundary.encode("ascii", errors="ignore")
    return None


def _parse_content_length(header_block: bytes) -> int | None:
    """Pull Content-Length out of one multipart part's header block (the
    bytes between a boundary line and the blank line that ends it)."""
    for line in header_block.split(b"\r\n"):
        name, sep, value = line.partition(b":")
        if sep and name.strip().lower() == b"content-length":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


class HTTPCamera(CameraSource):
    def __init__(self, config: CameraConfig):
        super().__init__(config)
        if not config.url:
            raise ValueError(f"HTTP camera {config.id!r}: config.url is required")
        if config.http_mode not in ("mjpeg", "snapshot"):
            raise ValueError(
                f"HTTP camera {config.id!r}: http_mode must be 'mjpeg' or 'snapshot', got {config.http_mode!r}"
            )
        self._session: requests.Session | None = None
        self._resp: requests.Response | None = None
        self._byte_iter = None
        self._buf = b""
        self._boundary: bytes | None = None
        self._last_shape: tuple[int, int] | None = None

    def _auth(self):
        return (self.config.username, self.config.password) if self.config.username else None

    def open(self) -> None:
        session = requests.Session()
        if self.config.http_mode == "snapshot":
            try:
                resp = session.get(self.config.url, auth=self._auth(), timeout=self.config.connect_timeout_sec)
                resp.raise_for_status()
            except requests.RequestException as exc:
                session.close()
                raise CameraConnectionError(
                    f"HTTP camera {self.config.id!r}: snapshot GET failed for {redact_url(self.config.url)}: {exc}"
                ) from exc
            self._session = session
            return

        try:
            resp = session.get(
                self.config.url, auth=self._auth(), stream=True, timeout=self.config.connect_timeout_sec
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            session.close()
            raise CameraConnectionError(
                f"HTTP camera {self.config.id!r}: MJPEG connect failed for {redact_url(self.config.url)}: {exc}"
            ) from exc
        boundary = _parse_boundary(resp.headers.get("Content-Type", ""))
        if boundary is None:
            resp.close()
            session.close()
            raise CameraConnectionError(
                f"HTTP camera {self.config.id!r}: expected multipart/x-mixed-replace with a boundary, "
                f"got Content-Type={resp.headers.get('Content-Type')!r} from {redact_url(self.config.url)}"
            )
        self._session = session
        self._resp = resp
        self._byte_iter = resp.iter_content(chunk_size=_BOUNDARY_SCAN_CHUNK)
        self._buf = b""
        self._boundary = boundary

    def read(self) -> np.ndarray:
        if self._session is None:
            raise CameraConnectionError(f"HTTP camera {self.config.id!r}: read() called before open()")
        frame = self._read_snapshot() if self.config.http_mode == "snapshot" else self._read_mjpeg_frame()
        self._last_shape = (frame.shape[1], frame.shape[0])
        return frame

    def _read_snapshot(self) -> np.ndarray:
        try:
            resp = self._session.get(self.config.url, auth=self._auth(), timeout=self.config.read_timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise CameraConnectionError(f"HTTP camera {self.config.id!r}: snapshot GET failed: {exc}") from exc
        frame = cv2.imdecode(np.frombuffer(resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise CameraConnectionError(
                f"HTTP camera {self.config.id!r}: snapshot response was not a decodable image "
                f"({len(resp.content)} bytes)"
            )
        return frame

    def _read_mjpeg_frame(self) -> np.ndarray:
        """Parse one part out of the multipart/x-mixed-replace stream.

        Prefers each part's own Content-Length header (RFC 2046 §5.1.1) to
        know exactly how many body bytes to take — this is what lets the
        very last frame before a stream closes be read correctly, since it
        does not require ever seeing a following boundary. Falls back to
        scanning for the next boundary only when a camera omits
        Content-Length (some cheap ones do).
        """
        boundary = b"--" + self._boundary
        try:
            while True:
                marker = self._buf.find(boundary)
                if marker == -1:
                    self._fill_buf()
                    continue

                header_start = marker + len(boundary)
                header_end = self._buf.find(b"\r\n\r\n", header_start)
                if header_end == -1:
                    if len(self._buf) - marker > _MAX_PART_BYTES:
                        raise CameraConnectionError(
                            f"HTTP camera {self.config.id!r}: multipart headers exceeded "
                            f"{_MAX_PART_BYTES} bytes without a terminating blank line (malformed stream?)"
                        )
                    self._fill_buf()
                    continue

                body_start = header_end + 4
                content_length = _parse_content_length(self._buf[header_start:header_end])

                if content_length is not None:
                    body_end = body_start + content_length
                    while len(self._buf) < body_end:
                        self._fill_buf()
                    jpeg_bytes = self._buf[body_start:body_end]
                    self._buf = self._buf[body_end:]
                else:
                    next_boundary = self._buf.find(boundary, body_start)
                    while next_boundary == -1:
                        if len(self._buf) - body_start > _MAX_PART_BYTES:
                            raise CameraConnectionError(
                                f"HTTP camera {self.config.id!r}: no JPEG frame boundary found within "
                                f"{_MAX_PART_BYTES} bytes (malformed stream, and no Content-Length to fall back on)"
                            )
                        self._fill_buf()
                        next_boundary = self._buf.find(boundary, body_start)
                    jpeg_bytes = self._buf[body_start:next_boundary].rstrip(b"\r\n")
                    self._buf = self._buf[next_boundary:]

                if not jpeg_bytes:
                    continue  # empty part between two boundaries — keep scanning
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
                # decodable-looking part but not a valid JPEG — skip it, keep scanning
        except (StopIteration, requests.RequestException) as exc:
            raise CameraConnectionError(f"HTTP camera {self.config.id!r}: MJPEG stream ended/failed: {exc}") from exc

    def _fill_buf(self) -> None:
        self._buf += next(self._byte_iter)

    def release(self) -> None:
        if self._resp is not None:
            self._resp.close()
            self._resp = None
        if self._session is not None:
            self._session.close()
            self._session = None
        self._byte_iter = None
        self._buf = b""

    def native_resolution(self):
        return self._last_shape

    def is_pull_model(self) -> bool:
        # snapshot mode: every read() is its own GET, nothing to drain.
        # mjpeg mode: a persistent decode of a push stream, same as RTSP/USB.
        return self.config.http_mode == "snapshot"
