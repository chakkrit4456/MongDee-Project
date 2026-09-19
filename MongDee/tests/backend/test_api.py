import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import EventHub, LiveState, create_app
from backend.database.db import Database
from vision.events import PERSON_NEW, PersonEvent


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "api.db")
    d.upsert_camera({"id": "CAM01", "name": "Entrance", "protocol": "usb", "status": "online", "fps": 6})
    d.upsert_global_person({
        "id": "PERSON-0001", "first_seen": 1.0, "last_seen": 9.0,
        "estimated_gender": "male", "cameras_seen": ["CAM01", "CAM02"], "shirt_color": "blue",
    })
    d.insert_event({"event_type": "NEW_PERSON", "timestamp": 1.0, "camera_id": "CAM01", "global_id": "PERSON-0001", "local_track_id": 3})
    yield d
    d.close()


def test_dashboard_page_served(db):
    client = TestClient(create_app(db))
    r = client.get("/")
    assert r.status_code == 200
    assert "MongDee Vision" in r.text
    assert "text/html" in r.headers["content-type"]


def test_designer_page_served(db):
    client = TestClient(create_app(db))
    r = client.get("/designer")
    assert r.status_code == 200
    assert "Booth Designer" in r.text


def test_dashboard_config_js_served_with_javascript_content_type(db):
    client = TestClient(create_app(db))
    r = client.get("/config.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
    assert "MONGDEE_API_BASE" in r.text


def test_health_and_read_endpoints(db):
    client = TestClient(create_app(db))
    assert client.get("/api/health").json()["status"] == "ok"

    cams = client.get("/api/cameras").json()["cameras"]
    assert cams[0]["id"] == "CAM01"

    persons = client.get("/api/persons").json()["persons"]
    assert persons[0]["id"] == "PERSON-0001"
    assert persons[0]["cameras_seen"] == ["CAM01", "CAM02"]

    person = client.get("/api/persons/PERSON-0001").json()
    assert person["shirt_color"] == "blue"
    assert "tracks" in person

    assert client.get("/api/persons/NOPE").status_code == 404

    events = client.get("/api/events").json()["events"]
    assert events[0]["event_type"] == "NEW_PERSON"

    stats = client.get("/api/stats").json()
    assert stats["unique_persons"] == 1
    assert stats["camera_person_counts"] == {"CAM01": 1, "CAM02": 1}


def test_api_key_enforced(db):
    client = TestClient(create_app(db, api_key="secret123"))
    assert client.get("/api/persons").status_code == 401
    assert client.get("/api/persons", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.get("/api/persons", headers={"X-API-Key": "secret123"})
    assert ok.status_code == 200
    # health is public
    assert client.get("/api/health").status_code == 200


def test_api_key_accepted_via_query_param_for_img_tag_use(db):
    # A plain <img>/<video> tag (the dashboard's live camera stream) can't
    # set a custom header, so ?api_key= must work too.
    client = TestClient(create_app(db, api_key="secret123"))
    assert client.get("/api/persons?api_key=wrong").status_code == 401
    assert client.get("/api/persons?api_key=secret123").status_code == 200


def test_role_based_access(db):
    client = TestClient(create_app(db, keys={"vk": "viewer", "ak": "admin"}))
    layout = {"width": 10, "length": 6, "objects": []}
    # viewer can read
    assert client.get("/api/persons", headers={"X-API-Key": "vk"}).status_code == 200
    # viewer cannot save a layout
    assert client.post("/api/booths/B1/layout", json=layout, headers={"X-API-Key": "vk"}).status_code == 403
    # admin can
    assert client.post("/api/booths/B1/layout", json=layout, headers={"X-API-Key": "ak"}).status_code == 200
    # admin action is audited
    audit = client.get("/api/audit", headers={"X-API-Key": "ak"}).json()["entries"]
    assert any("layout_saved" in e["message"] for e in audit)
    # viewer cannot see the audit log
    assert client.get("/api/audit", headers={"X-API-Key": "vk"}).status_code == 403


def test_live_state_in_responses(db):
    live = LiveState()
    live.update_camera_status("CAM01", "online", "connected")
    live.update_camera_counts("CAM01", detections=3, tracks=2, fps=4.1, latency_ms=120.0)
    live.set_unique_count(7)
    client = TestClient(create_app(db, live=live))
    cams = client.get("/api/cameras").json()["cameras"]
    assert cams[0]["live"]["tracks"] == 2
    assert client.get("/api/stats").json()["live"]["unique_persons"] == 7


def test_ws_events_stream(db):
    hub = EventHub()
    app = create_app(db, event_hub=hub)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as ws:
            hub.publish(PersonEvent.now(event_type=PERSON_NEW, camera_id="CAM01", global_id="PERSON-0002", local_track_id=5))
            msg = ws.receive_json()
            assert msg["event_type"] == "NEW_PERSON"
            assert msg["global_id"] == "PERSON-0002"


def test_ws_events_requires_key_when_set(db):
    hub = EventHub()
    app = create_app(db, event_hub=hub, api_key="k")
    with TestClient(app) as client:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/events"):
                pass
        with client.websocket_connect("/ws/events?api_key=k"):
            pass
