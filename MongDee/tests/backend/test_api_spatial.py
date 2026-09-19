import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.database.db import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "sp.db")
    d.upsert_product({"id": "GI-001", "name": "Thung Kula Rice", "class_name": "rice_bag", "province": "Roi Et"})
    d.insert_product_detection({"product_id": "GI-001", "camera_id": "CAM01", "local_track_id": 1,
                                "status": "KNOWN_PRODUCT", "zone_id": "ZONE-A", "bbox": [0, 0, 10, 10],
                                "confidence": 0.9, "timestamp": 1.0})
    d.insert_position({"global_person_id": "PERSON-0001", "camera_id": "CAM01", "x": 3.2, "y": 2.1,
                       "zone_id": "ZONE-A", "confidence": 0.9, "timestamp": 1.0})
    d.insert_interest_event({"event_type": "CUSTOMER_INTEREST_ENDED", "global_id": "PERSON-0001",
                             "target_id": "ZONE-A", "status": "HIGH_INTEREST", "score": 0.8,
                             "look_duration": 5.4, "dwell_duration": 6.0, "signals": ["distance", "stop"], "timestamp": 2.0})
    yield d
    d.close()


def test_designer_page_served(db):
    r = TestClient(create_app(db)).get("/designer")
    assert r.status_code == 200 and "Booth Designer" in r.text


def test_layout_crud_and_versioning(db):
    client = TestClient(create_app(db))
    layout_v1 = {"width": 10, "length": 6, "version": 1, "objects": [
        {"id": "ZONE-A", "object_type": "product_zone", "x": 3, "y": 2, "width": 2, "height": 2},
        {"id": "CAM01", "object_type": "camera", "x": 0, "y": 0, "metadata": {"camera_id": "CAM01"}},
    ]}
    r = client.post("/api/booths/BOOTH-01/layout", json=layout_v1)
    assert r.status_code == 200 and r.json()["version"] == 1

    layout_v2 = {**layout_v1, "version": 2}
    layout_v2["objects"].append({"id": "SHELF-A", "object_type": "shelf", "x": 5, "y": 3, "width": 1.5, "height": 0.5})
    client.post("/api/booths/BOOTH-01/layout", json=layout_v2)

    active = client.get("/api/booths/BOOTH-01/layout").json()
    assert active["version"] == 2
    assert len(active["objects"]) == 3

    versions = client.get("/api/booths/BOOTH-01/layout/versions").json()["versions"]
    assert {v["version"] for v in versions} == {1, 2}

    client.post("/api/booths/BOOTH-01/layout/1/activate")
    assert client.get("/api/booths/BOOTH-01/layout").json()["version"] == 1

    assert client.get("/api/booths").json()["booths"][0]["id"] == "BOOTH-01"


def test_layout_validation_rejects_bad_object(db):
    client = TestClient(create_app(db))
    bad = {"width": 10, "length": 6, "objects": [{"id": "X", "object_type": "teleporter", "x": 0, "y": 0}]}
    r = client.post("/api/booths/BOOTH-01/layout", json=bad)
    assert r.status_code == 400
    assert "teleporter" in r.json()["detail"]


def test_product_endpoints(db):
    client = TestClient(create_app(db))
    products = client.get("/api/products").json()["products"]
    assert products[0]["id"] == "GI-001"
    detail = client.get("/api/products/GI-001").json()
    assert detail["province"] == "Roi Et"
    assert detail["analytics"]["total_detections"] == 1
    assert detail["analytics"]["by_status"] == {"KNOWN_PRODUCT": 1}
    assert client.get("/api/products/NOPE").status_code == 404


def test_movement_and_interest_endpoints(db):
    client = TestClient(create_app(db))
    path = client.get("/api/persons/PERSON-0001/movement").json()["path"]
    assert path[0]["x"] == 3.2

    interests = client.get("/api/persons/PERSON-0001/interests").json()["interests"]
    assert interests[0]["status"] == "HIGH_INTEREST"
    assert interests[0]["signals"] == ["distance", "stop"]

    za = client.get("/api/zones/ZONE-A/analytics").json()
    assert za["unique_visitors"] == 1
    assert za["interest_by_status"] == {"HIGH_INTEREST": 1}


def test_live_map_and_heatmap_without_analytics(db):
    client = TestClient(create_app(db))
    assert client.get("/api/map").json() == {"positions": [], "interest": []}
    assert client.get("/api/analytics/heatmap").status_code == 503


def test_live_map_with_analytics(db):
    from vision.booth_analytics import BoothAnalytics
    from vision.spatial.booth import BoothLayout, BoothObject
    from vision.spatial.calibration import CalibrationStore, CameraCalibration

    store = CalibrationStore()
    store.set(CameraCalibration.from_points(
        "CAM01", [(0, 0), (1000, 0), (1000, 1000), (0, 1000)], [(0, 0), (10, 0), (10, 6), (0, 6)]))
    layout = BoothLayout("BOOTH-01", 10, 6)
    layout.add(BoothObject("ZONE-A", "product_zone", x=5, y=3, width=3, height=3))
    ba = BoothAnalytics(layout, store)
    ba.observe_person("CAM01", "PERSON-0001", [400, 200, 600, 600], timestamp=0.0)

    client = TestClient(create_app(db, analytics=ba))
    m = client.get("/api/map").json()
    assert m["positions"][0]["global_id"] == "PERSON-0001"
    hm = client.get("/api/analytics/heatmap?kind=traffic").json()
    assert hm["kind"] == "traffic" and "grid" in hm
    assert client.get("/api/analytics/heatmap?kind=bogus").status_code == 400
