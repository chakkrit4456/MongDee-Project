import numpy as np
import pytest

from backend.database.db import Database
from backend.database.sink import DatabaseSink
from vision.attributes.extractor import AttributeValue, PersonAttributes
from vision.events import PERSON_CAMERA_TRANSITION, PERSON_NEW, PERSON_REIDENTIFIED, PersonEvent
from vision.identity.global_person import GlobalPerson, TrackSummary


class _FakeIdentity:
    def __init__(self):
        attrs = PersonAttributes(shirt_color=AttributeValue("blue", 0.8))
        self._p = GlobalPerson(
            "PERSON-0001",
            TrackSummary("CAM01", 1, 0.0, 10.0, np.array([1.0, 0.0, 0.0]), 3, attrs, 0.7),
        )
        self._p.merge(TrackSummary("CAM02", 7, 12.0, 20.0, np.array([0.9, 0.1, 0.0]), 2, attrs, 0.7))

    def get(self, gid):
        return self._p if gid == "PERSON-0001" else None


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "sink.db")
    yield d
    d.close()


def test_sink_persists_new_person_event(db):
    sink = DatabaseSink(db, identity_manager=_FakeIdentity())
    sink.on_person_event(
        PersonEvent.now(event_type=PERSON_NEW, camera_id="CAM01", global_id="PERSON-0001", local_track_id=1, match_score=0.0)
    )
    persons = db.global_persons()
    assert len(persons) == 1
    assert persons[0]["id"] == "PERSON-0001"
    assert persons[0]["shirt_color"] == "blue"
    assert set(persons[0]["cameras_seen"]) == {"CAM01", "CAM02"}
    assert len(db.events()) == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM camera_tracks")["n"] == 1
    assert db.query_one("SELECT COUNT(*) AS n FROM reid_embeddings")["n"] == 1


def test_sink_records_camera_transition(db):
    sink = DatabaseSink(db, identity_manager=_FakeIdentity())
    sink.on_person_event(
        PersonEvent.now(
            event_type=PERSON_CAMERA_TRANSITION, camera_id="CAM02", global_id="PERSON-0001",
            local_track_id=7, match_score=0.9, payload={"from_camera": "CAM01", "to_camera": "CAM02", "transit_seconds": 4.2},
        )
    )
    row = db.query_one("SELECT * FROM camera_transitions")
    assert row["from_camera"] == "CAM01"
    assert row["to_camera"] == "CAM02"
    assert row["transit_seconds"] == 4.2


def test_sink_swallows_db_errors(db):
    db.close()  # force every write to fail
    sink = DatabaseSink(db, identity_manager=_FakeIdentity())
    # must not raise into the pipeline thread
    sink.on_person_event(PersonEvent.now(event_type=PERSON_NEW, camera_id="CAM01", global_id="PERSON-0001", local_track_id=1))
    sink.on_camera_status("CAM01", "online")


def test_sink_detection_logging_is_sampled(db):
    from camera.base import Frame
    from vision.detection.detector import Detection
    from vision.pipeline import PipelineResult

    sink = DatabaseSink(db, log_detections=True, detection_sample_interval_sec=10.0)
    frame = Frame("CAM01", np.zeros((4, 4, 3), np.uint8), 0.0, 0)
    det = Detection([0, 0, 10, 20], 0.9, "CAM01", 1.0)
    sink.on_result(PipelineResult("CAM01", frame, [det], []))
    sink.on_result(PipelineResult("CAM01", frame, [det], []))  # within sample interval -> skipped
    assert db.query_one("SELECT COUNT(*) AS n FROM detections")["n"] == 1
