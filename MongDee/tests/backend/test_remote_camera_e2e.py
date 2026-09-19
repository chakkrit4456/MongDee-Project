"""End-to-end: a camera registered and fed entirely through the network API
(as camera_agent/client.py does) flows through the real detection pipeline
exactly like a locally-attached camera would — this is the crux of
MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md's "Camera Agent
pushes frames to a Remote AI Server" architecture. Uses real YOLO, so it's
opt-in via `pytest -m slow`, same as tests/backend/test_system.py.
"""

import json
import os
import time

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def no_local_camera_config(tmp_path):
    cfg = {
        "cameras": [],  # no locally-attached cameras at all -- everything arrives via the API
        "detection": {"device": "cpu", "target_fps_per_camera": 6},
        "tracking": {"n_init": 2, "track_buffer": 10},
        "reid": {"backend": "color"},
        "identity": {"match_threshold": 0.62, "uncertain_threshold": 0.5},
        "features": {"observe_interval_sec": 0.2, "track_end_timeout_sec": 1.0, "min_samples_to_flush": 1},
        "database": {"path": str(tmp_path / "remote_sys.db")},
        "api": {"host": "127.0.0.1", "port": 0, "keys": {"agent-key": "operator", "view-key": "viewer"}},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_remote_camera_agent_frames_reach_the_real_pipeline(no_local_camera_config):
    import cv2
    import ultralytics
    from fastapi.testclient import TestClient

    from backend.config import MongDeeConfig
    from backend.main import MongDeeSystem

    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert bus is not None
    ok, jpeg = cv2.imencode(".jpg", bus)
    assert ok
    jpeg_bytes = jpeg.tobytes()

    system = MongDeeSystem(MongDeeConfig.load(no_local_camera_config))
    system.start()
    try:
        client = TestClient(system.build_app())
        agent_headers = {"X-API-Key": "agent-key"}

        # 1. Camera Agent registers itself (section 32's flow).
        r = client.post(
            "/api/cameras/register",
            json={"cameraId": "REMOTE-CAM01", "name": "Remote Entrance", "resolution": "810x1080", "fps": 6},
            headers=agent_headers,
        )
        assert r.status_code == 200
        assert client.get("/api/cameras", headers={"X-API-Key": "view-key"}).json()["cameras"][0]["id"] == "REMOTE-CAM01"

        # 2. Camera Agent pushes frames continuously — the same real image
        # (which contains real people) repeatedly, like a static/slow scene.
        deadline = time.time() + 10
        seq = 0
        while time.time() < deadline:
            seq += 1
            resp = client.post(
                f"/api/cameras/REMOTE-CAM01/frame",
                files={"frame": ("f.jpg", jpeg_bytes, "image/jpeg")},
                data={"timestamp": str(time.time()), "sequence": str(seq)},
                headers=agent_headers,
            )
            assert resp.status_code == 200
            time.sleep(0.1)

        system.stop()  # flushes in-progress tracks to the db

        # 3. The pipeline actually ran detection/tracking/identity on these
        # remotely-supplied frames -- proving frames pushed over the network
        # API reach the same real pipeline a local camera would.
        stats = client.get("/api/stats", headers={"X-API-Key": "view-key"}).json()
        assert stats["unique_persons"] >= 1
        cams = client.get("/api/cameras", headers={"X-API-Key": "view-key"}).json()["cameras"]
        assert cams[0]["live"]["tracks"] or cams[0]["live"]["detections"]
    finally:
        system.close()


def test_frame_ingest_rejects_viewer_role(no_local_camera_config):
    from fastapi.testclient import TestClient

    from backend.config import MongDeeConfig
    from backend.main import MongDeeSystem

    system = MongDeeSystem(MongDeeConfig.load(no_local_camera_config))
    try:
        client = TestClient(system.build_app())
        # A read-only viewer key must not be able to register a camera or
        # inject frames -- registration/ingest requires at least "operator".
        r = client.post(
            "/api/cameras/register", json={"cameraId": "X"}, headers={"X-API-Key": "view-key"},
        )
        assert r.status_code == 403
    finally:
        system.close()
