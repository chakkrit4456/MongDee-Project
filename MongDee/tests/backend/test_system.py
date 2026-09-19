"""End-to-end: the whole MongDee vision system (cameras -> detection ->
tracking -> reid -> identity -> database -> API) driven by synthetic
cameras, with real YOLO. Opt-in via `pytest -m slow`.
"""

import json
import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def system_config(tmp_path):
    cfg = {
        "cameras": [
            {"id": "CAM01", "protocol": "usb", "processing_fps": 6, "location": "a"},
            {"id": "CAM02", "protocol": "usb", "processing_fps": 6, "location": "b"},
        ],
        "detection": {"device": "cpu", "target_fps_per_camera": 6},
        "tracking": {"n_init": 2, "track_buffer": 10},
        "reid": {"backend": "color"},
        "identity": {"match_threshold": 0.62, "uncertain_threshold": 0.5},
        "features": {"observe_interval_sec": 0.2, "track_end_timeout_sec": 1.0, "min_samples_to_flush": 1},
        "database": {"path": str(tmp_path / "sys.db")},
        "api": {"host": "127.0.0.1", "port": 0},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_full_system_end_to_end(system_config, monkeypatch):
    import cv2
    import ultralytics

    import camera.gateway as gateway_mod
    from camera.base import CameraSource
    from backend.config import MongDeeConfig
    from backend.main import MongDeeSystem
    from fastapi.testclient import TestClient

    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert bus is not None

    class StaticCam(CameraSource):
        def open(self):
            pass

        def read(self):
            time.sleep(0.04)
            return bus.copy()

        def release(self):
            pass

    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: StaticCam(cfg))

    system = MongDeeSystem(MongDeeConfig.load(system_config))
    system.start()
    try:
        time.sleep(10)
        system.stop()  # flushes in-progress tracks to the db

        client = TestClient(system.build_app())
        stats = client.get("/api/stats").json()
        assert stats["unique_persons"] >= 1
        persons = client.get("/api/persons").json()["persons"]
        assert persons
        events = client.get("/api/events").json()["events"]
        assert any(e["event_type"] in ("NEW_PERSON", "PERSON_REIDENTIFIED") for e in events)
        cams = client.get("/api/cameras").json()["cameras"]
        assert {c["id"] for c in cams} == {"CAM01", "CAM02"}
    finally:
        system.close()


def test_full_system_with_spatial_and_product(tmp_path, monkeypatch):
    import cv2
    import ultralytics

    import camera.gateway as gateway_mod
    from camera.base import CameraSource
    from backend.config import MongDeeConfig
    from backend.main import MongDeeSystem
    from fastapi.testclient import TestClient
    from vision.spatial.calibration import CalibrationStore, CameraCalibration

    bus = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    h, w = bus.shape[:2]

    calib = CalibrationStore()
    calib.set(CameraCalibration.from_points(
        "CAM01", [(0, 0), (w, 0), (w, h), (0, h)], [(0, 0), (10, 0), (10, 6), (0, 6)]))
    calib.save(tmp_path / "calib.json")

    layout = {
        "booth_id": "BOOTH-01", "width": 10, "length": 6, "version": 1,
        "objects": [{"id": "ZONE-A", "object_type": "product_zone", "x": 5, "y": 3, "width": 6, "height": 6}],
    }
    (tmp_path / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (tmp_path / "gi.json").write_text(json.dumps({"products": [
        {"id": "GI-001", "name": "Local Honey", "class_name": "bottle"}]}), encoding="utf-8")

    cfg = {
        "cameras": [{"id": "CAM01", "protocol": "usb", "processing_fps": 6}],
        "detection": {"device": "cpu", "target_fps_per_camera": 6},
        "tracking": {"n_init": 2, "track_buffer": 10},
        "reid": {"backend": "color"},
        "identity": {"match_threshold": 0.6, "uncertain_threshold": 0.5},
        "features": {"observe_interval_sec": 0.2, "eager_submit_samples": 2, "track_end_timeout_sec": 1.0},
        "interest": {"near_distance_m": 8.0, "min_look_duration_sec": 0.5, "min_dwell_duration_sec": 0.5},
        "database": {"path": str(tmp_path / "sys2.db")},
        "spatial": {"enabled": True, "booth_id": "BOOTH-01",
                    "layout_path": str(tmp_path / "layout.json"), "calibration_path": str(tmp_path / "calib.json")},
        "product_pipeline": {"enabled": True, "gi_database_path": str(tmp_path / "gi.json")},
        "api": {"host": "127.0.0.1", "port": 0},
    }
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    class StaticCam(CameraSource):
        def open(self): pass
        def read(self):
            time.sleep(0.04)
            return bus.copy()
        def release(self): pass

    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda c: StaticCam(c))

    system = MongDeeSystem(MongDeeConfig.load(tmp_path / "cfg.json"))
    system.start()
    try:
        time.sleep(12)
        system.stop()

        client = TestClient(system.build_app())
        # spatial: at least one world position recorded
        assert system.db.query_one("SELECT COUNT(*) AS n FROM positions")["n"] >= 1
        # booth layout persisted + served
        assert client.get("/api/booths/BOOTH-01/layout").json()["version"] == 1
        # live map has positions
        m = client.get("/api/map").json()
        assert isinstance(m["positions"], list)
        # heatmap available
        assert client.get("/api/analytics/heatmap?kind=traffic").status_code == 200
        # product db seeded
        assert client.get("/api/products").json()["products"][0]["id"] == "GI-001"
    finally:
        system.close()
