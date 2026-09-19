"""Exercises HTTPCamera against real local HTTP servers (not mocks) so the
multipart/x-mixed-replace parsing and snapshot polling are verified against
actual bytes on the wire, not a stand-in for them.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from camera.base import CameraConfig, CameraConnectionError
from camera.gateway import CameraGateway
from camera.http import HTTPCamera, _parse_boundary


def _jpeg_bytes(color, size=(32, 32)) -> bytes:
    img = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


class _MJPEGHandler(BaseHTTPRequestHandler):
    frames: list[bytes] = []

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            for jpg in self.frames:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args):  # keep test output quiet
        pass


class _SnapshotHandler(BaseHTTPRequestHandler):
    jpeg: bytes = b""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(self.jpeg)))
        self.end_headers()
        self.wfile.write(self.jpeg)

    def log_message(self, fmt, *args):
        pass


def _start_server(handler_cls) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def mjpeg_server():
    frames = [_jpeg_bytes((255, 0, 0)), _jpeg_bytes((0, 255, 0)), _jpeg_bytes((0, 0, 255))]
    handler_cls = type("BoundHandler", (_MJPEGHandler,), {"frames": frames})
    server = _start_server(handler_cls)
    yield server, frames
    server.shutdown()
    server.server_close()


@pytest.fixture
def snapshot_server():
    jpeg = _jpeg_bytes((10, 20, 30))
    handler_cls = type("BoundHandler", (_SnapshotHandler,), {"jpeg": jpeg})
    server = _start_server(handler_cls)
    yield server, jpeg
    server.shutdown()
    server.server_close()


def _config(url, **overrides):
    base = {"id": "CAM04", "protocol": "http", "url": url}
    base.update(overrides)
    return CameraConfig.from_dict(base)


def test_parse_boundary():
    assert _parse_boundary("multipart/x-mixed-replace; boundary=frame") == b"frame"
    assert _parse_boundary('multipart/x-mixed-replace; boundary="frame"') == b"frame"
    assert _parse_boundary("image/jpeg") is None


def test_mjpeg_stream_end_to_end(mjpeg_server):
    server, frames = mjpeg_server
    port = server.server_address[1]
    cam = HTTPCamera(_config(f"http://127.0.0.1:{port}/video", http_mode="mjpeg"))
    cam.open()
    try:
        received = [cam.read() for _ in range(len(frames))]
        assert len(received) == 3
        for frame in received:
            assert frame is not None and frame.shape[2] == 3
        assert cam.native_resolution() == (32, 32)
    finally:
        cam.release()


def test_mjpeg_stream_end_raises_connection_error(mjpeg_server):
    server, frames = mjpeg_server
    port = server.server_address[1]
    cam = HTTPCamera(_config(f"http://127.0.0.1:{port}/video", http_mode="mjpeg"))
    cam.open()
    for _ in range(len(frames)):
        cam.read()
    with pytest.raises(CameraConnectionError, match="ended/failed"):
        cam.read()
    cam.release()


def test_snapshot_mode_end_to_end(snapshot_server):
    server, jpeg = snapshot_server
    port = server.server_address[1]
    cam = HTTPCamera(_config(f"http://127.0.0.1:{port}/snap.jpg", http_mode="snapshot"))
    cam.open()
    try:
        frame1 = cam.read()
        frame2 = cam.read()  # polling mode: must succeed on repeated calls
        assert frame1 is not None and frame2 is not None
        assert cam.native_resolution() == (32, 32)
    finally:
        cam.release()


def test_open_rejects_non_multipart_content_type(snapshot_server):
    server, jpeg = snapshot_server
    port = server.server_address[1]
    cam = HTTPCamera(_config(f"http://127.0.0.1:{port}/x", http_mode="mjpeg"))
    with pytest.raises(CameraConnectionError, match="multipart/x-mixed-replace"):
        cam.open()


def test_connection_refused_raises():
    cam = HTTPCamera(_config("http://127.0.0.1:1/video", http_mode="snapshot", connect_timeout_sec=2.0))
    with pytest.raises(CameraConnectionError):
        cam.open()


def test_missing_url_raises_at_construction():
    with pytest.raises(ValueError, match="config.url is required"):
        HTTPCamera(CameraConfig.from_dict({"id": "CAM04", "protocol": "http"}))


def test_invalid_http_mode_rejected():
    with pytest.raises(ValueError, match="http_mode must be"):
        HTTPCamera(_config("http://127.0.0.1:9/video", http_mode="carrier-pigeon"))


class _CountingSnapshotHandler(_SnapshotHandler):
    """Same as _SnapshotHandler but records how many GETs it served, so a
    test can prove the gateway's FPS throttle actually limits *requests*
    for a pull-model source instead of firing them as fast as possible and
    discarding the results (see camera/gateway.py CameraSource.is_pull_model)."""

    counter_lock = threading.Lock()
    count = 0

    def do_GET(self):
        with self.counter_lock:
            type(self).count += 1
        super().do_GET()


def test_gateway_snapshot_polling_respects_fps_target():
    jpeg = _jpeg_bytes((1, 2, 3))
    handler_cls = type("BoundCountingHandler", (_CountingSnapshotHandler,), {"jpeg": jpeg, "count": 0})
    server = _start_server(handler_cls)
    try:
        port = server.server_address[1]
        cfg = _config(f"http://127.0.0.1:{port}/snap.jpg", http_mode="snapshot", processing_fps=5.0)
        gateway = CameraGateway()
        gateway.add_camera(cfg)
        try:
            time.sleep(0.6)
        finally:
            gateway.stop_all()
        # 5 fps for 0.6s ~= 3 requests. A source that busy-polled as fast as
        # possible (the bug this test guards against) would rack up hundreds.
        assert 1 <= handler_cls.count <= 12
    finally:
        server.shutdown()
        server.server_close()
