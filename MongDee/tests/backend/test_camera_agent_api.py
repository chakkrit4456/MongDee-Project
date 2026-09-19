"""Tests for the Camera Agent-facing endpoints added to backend/api/app.py
(register / frame ingest / live stream) — see backend/remote_frames.py and
camera_agent/client.py. MongDee_Cloud_Vercel_Remote_AI_Server_Master_
Prompt.md sections 4, 29, 32.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.database.db import Database
from backend.remote_frames import CompositeFrameSource, RemoteFrameReceiver


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "api.db")
    yield d
    d.close()


def _jpeg_bytes(fill=5, size=(20, 20)):
    frame = np.full((size[1], size[0], 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def test_register_and_frame_without_remote_frames_returns_503(db):
    client = TestClient(create_app(db))
    assert client.post("/api/cameras/register", json={"cameraId": "CAM-1"}).status_code == 503


def test_register_camera_creates_db_row_and_receiver_entry(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver))
    r = client.post("/api/cameras/register", json={
        "cameraId": "cam-01", "name": "Entrance", "resolution": "1280x720", "fps": 25, "codec": "H264",
    })
    assert r.status_code == 200
    assert r.json() == {"cameraId": "cam-01", "status": "connecting"}
    assert "cam-01" in receiver.camera_ids()

    cams = db.cameras()
    assert cams[0]["id"] == "cam-01"
    assert cams[0]["protocol"] == "remote_agent"
    assert cams[0]["resolution"] == "1280x720"


def test_register_camera_requires_camera_id(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver))
    assert client.post("/api/cameras/register", json={}).status_code == 400


def test_register_and_operator_role_enforced(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver, keys={"vk": "viewer", "ok": "operator"}))
    # viewer cannot register a camera (would let a read-only key inject video)
    assert client.post("/api/cameras/register", json={"cameraId": "cam-01"}, headers={"X-API-Key": "vk"}).status_code == 403
    r = client.post("/api/cameras/register", json={"cameraId": "cam-01"}, headers={"X-API-Key": "ok"})
    assert r.status_code == 200


def test_frame_ingest_requires_registration_first(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver))
    r = client.post(
        "/api/cameras/cam-01/frame",
        files={"frame": ("f.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 404


def test_frame_ingest_accepts_registered_camera_and_updates_receiver(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver))
    client.post("/api/cameras/register", json={"cameraId": "cam-01"})

    r = client.post(
        "/api/cameras/cam-01/frame",
        files={"frame": ("f.jpg", _jpeg_bytes(fill=9), "image/jpeg")},
        data={"timestamp": "123.0", "sequence": "1"},
    )
    assert r.status_code == 200
    assert r.json() == {"accepted": True}

    frame = receiver.latest_frame("cam-01")
    assert frame is not None
    assert frame.timestamp == 123.0


def test_frame_ingest_rejects_undecodable_payload(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, remote_frames=receiver))
    client.post("/api/cameras/register", json={"cameraId": "cam-01"})
    r = client.post(
        "/api/cameras/cam-01/frame",
        files={"frame": ("f.jpg", b"not a real jpeg", "image/jpeg")},
    )
    assert r.status_code == 400


def test_stream_without_frame_source_returns_503(db):
    client = TestClient(create_app(db))
    assert client.get("/api/cameras/cam-01/stream").status_code == 503


def test_stream_unknown_camera_returns_404(db):
    receiver = RemoteFrameReceiver()
    client = TestClient(create_app(db, frame_source=receiver))
    assert client.get("/api/cameras/cam-01/stream").status_code == 404


def test_mjpeg_chunks_generator_serves_latest_frame_as_jpeg():
    # The route's generator runs forever by design (a live MJPEG feed never
    # ends on its own) — driving it through a real HTTP round-trip via
    # TestClient hangs (there is no client disconnect to cancel it), so
    # this drives the generator function directly instead, bounded to one
    # iteration, via asyncio.run (no pytest-asyncio dependency needed).
    from backend.api.app import _mjpeg_chunks

    async def _run():
        receiver = RemoteFrameReceiver()
        receiver.register("cam-01")
        receiver.ingest_frame("cam-01", np.full((10, 10, 3), 4, dtype=np.uint8), 100.0)

        gen = _mjpeg_chunks(receiver, "cam-01", stream_fps=1000)  # fast, so the test doesn't wait on real time
        try:
            return await gen.__anext__()
        finally:
            await gen.aclose()

    chunk = asyncio.run(_run())
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk


def test_mjpeg_chunks_generator_skips_iterations_with_no_frame_yet():
    from backend.api.app import _mjpeg_chunks

    async def _run():
        receiver = RemoteFrameReceiver()
        receiver.register("cam-01")  # registered, but no frame ingested yet

        gen = _mjpeg_chunks(receiver, "cam-01", stream_fps=1000)
        task = asyncio.ensure_future(gen.__anext__())
        try:
            done, _ = await asyncio.wait({task}, timeout=0.05)
            assert not done, "must not yield a chunk before any frame has arrived"
        finally:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            await gen.aclose()

    asyncio.run(_run())


def test_health_bare_path_alias(db):
    client = TestClient(create_app(db))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/health").json()["status"] == "ok"


def test_metrics_endpoint_reports_hardware_and_remote_cameras(db):
    from core.performance import detect_hardware

    receiver = RemoteFrameReceiver()
    receiver.register("cam-01")
    hw = detect_hardware()
    client = TestClient(create_app(db, remote_frames=receiver, hardware_profile=hw))
    payload = client.get("/metrics").json()
    assert payload["hardware"]["tier"] in ("LOW", "MEDIUM", "HIGH")
    assert "cam-01" in payload["remote_cameras"]


def test_cors_headers_present_on_response(db):
    client = TestClient(create_app(db, cors_origins=["https://dashboard.example.com"]))
    r = client.get("/api/health", headers={"Origin": "https://dashboard.example.com"})
    assert r.headers.get("access-control-allow-origin") == "https://dashboard.example.com"


def test_stream_route_404_check_uses_composite_camera_ids(db):
    # The route's "camera not found" check works against whatever
    # frame_source it was given, local+remote merge included — verified
    # here via routing only; the infinite generator body is covered by the
    # _mjpeg_chunks tests above, not a live HTTP round-trip.
    class _FakeLocal:
        def camera_ids(self):
            return ["LOCAL-1"]

        def latest_frame(self, camera_id):
            return None

    remote = RemoteFrameReceiver()
    remote.register("REMOTE-1")
    composite = CompositeFrameSource(_FakeLocal(), remote)
    client = TestClient(create_app(db, frame_source=composite))

    assert client.get("/api/cameras/UNKNOWN-1/stream").status_code == 404
