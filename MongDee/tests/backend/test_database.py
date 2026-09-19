import json

import numpy as np
import pytest

from backend.database.db import Database


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


def test_migrations_applied(db):
    assert db.schema_version == 2
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"cameras", "global_persons", "camera_tracks", "detections", "events", "camera_transitions"} <= tables


def test_migrations_are_idempotent(tmp_path):
    p = tmp_path / "x.db"
    Database(p).close()
    db2 = Database(p)  # opening again must not re-run migration 1
    assert db2.schema_version == 2
    db2.close()


def test_upsert_camera_and_status(db):
    db.upsert_camera({"id": "CAM01", "name": "Entrance", "protocol": "rtsp", "location": "lobby", "status": "unknown", "fps": 8})
    db.set_camera_status("CAM01", "online")
    cams = db.cameras()
    assert len(cams) == 1
    assert cams[0]["status"] == "online"
    # upsert again updates, doesn't duplicate
    db.upsert_camera({"id": "CAM01", "name": "Entrance 2", "protocol": "rtsp"})
    assert len(db.cameras()) == 1
    assert db.cameras()[0]["name"] == "Entrance 2"


def test_upsert_global_person_merges_seen_times(db):
    db.upsert_global_person({"id": "PERSON-0001", "first_seen": 100.0, "last_seen": 110.0, "cameras_seen": ["CAM01"]})
    db.upsert_global_person({"id": "PERSON-0001", "first_seen": 90.0, "last_seen": 130.0, "cameras_seen": ["CAM01", "CAM02"]})
    p = db.global_person("PERSON-0001")
    assert p["first_seen"] == 90.0
    assert p["last_seen"] == 130.0
    assert p["cameras_seen"] == ["CAM01", "CAM02"]


def test_insert_and_query_events(db):
    db.insert_event({"event_type": "NEW_PERSON", "timestamp": 1.0, "camera_id": "CAM01", "global_id": "PERSON-0001", "local_track_id": 5, "payload": {"a": 1}})
    db.insert_event({"event_type": "PERSON_REIDENTIFIED", "timestamp": 2.0, "camera_id": "CAM02", "global_id": "PERSON-0001", "local_track_id": 7})
    all_events = db.events()
    assert len(all_events) == 2
    assert all_events[0]["event_type"] == "PERSON_REIDENTIFIED"  # newest first
    assert all_events[1]["payload"] == {"a": 1}
    only_new = db.events(event_type="NEW_PERSON")
    assert len(only_new) == 1


def test_stats_aggregation(db):
    db.upsert_global_person({"id": "PERSON-0001", "estimated_gender": "male", "estimated_age_group": "adult", "cameras_seen": ["CAM01"]})
    db.upsert_global_person({"id": "PERSON-0002", "estimated_gender": "unknown", "estimated_age_group": "adult", "cameras_seen": ["CAM01", "CAM02"]})
    db.upsert_camera({"id": "CAM01", "status": "online"})
    db.upsert_camera({"id": "CAM02", "status": "offline"})
    stats = db.stats()
    assert stats["unique_persons"] == 2
    assert stats["by_gender"] == {"male": 1, "unknown": 1}
    assert stats["by_age_group"] == {"adult": 2}
    assert stats["camera_status"] == {"online": 1, "offline": 1}
    assert db.camera_person_counts() == {"CAM01": 2, "CAM02": 1}


def test_insert_embedding_roundtrip(db):
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    db.insert_embedding("PERSON-0001", "CAM01", 3, vec.tobytes(), 3, 0.8)
    row = db.query_one("SELECT * FROM reid_embeddings")
    restored = np.frombuffer(row["vector"], dtype=np.float32)
    assert np.allclose(restored, vec)
    assert row["dim"] == 3


def test_insert_detections_batch(db):
    db.insert_detections([
        {"camera_id": "CAM01", "local_track_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 20, "confidence": 0.9, "timestamp": 1.0},
        {"camera_id": "CAM01", "local_track_id": 2, "x1": 5, "y1": 5, "x2": 15, "y2": 25, "confidence": 0.8, "timestamp": 1.0},
    ])
    assert db.query_one("SELECT COUNT(*) AS n FROM detections")["n"] == 2
