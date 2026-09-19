"""Integration tests for web/booth_manager.py's FairFace attribute wiring —
spec: MongDee_Master_Prompt_FairFace_Age_Gender.md.

Same "bare" BoothManager pattern as test_booth_manager_tripwire.py (real
constructor loads YOLO/cameras/a real FairFace checkpoint, none of which
_on_person_tracks/_update_reid/_update_attributes touch) — only this time
also wiring a fake reid_embedder (Multi-Camera Person Re-ID is this
feature's hard prerequisite per the spec's own architecture: Re-ID -> Global
Person ID -> Attribute Analyzer) and a fake attribute_backend so no real
CNNs load in these tests.
"""

from __future__ import annotations

import json
import threading

import numpy as np

from core import database as db
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.reid import GlobalIdentityRegistry, ReIDSampler
from core.tripwire import TripwireCounter, TripwireLine
from web.booth_manager import BoothManager


class _FakeEmbedder:
    """Deterministic fake Re-ID embedder — same fixed vector every call, so
    every track resolves to the same Global Person without needing a real
    MobileNetV3 model."""

    def embed(self, crop):
        return np.array([1.0, 0.0, 0.0, 0.0])


class _FakeAttributeBackend:
    def __init__(self, gender="male", gender_conf=0.9, age_group="20-29", age_conf=0.8):
        self.gender, self.gender_conf = gender, gender_conf
        self.age_group, self.age_conf = age_group, age_conf

    def predict_gender(self, crop):
        return self.gender, self.gender_conf

    def predict_age_group(self, crop):
        return self.age_group, self.age_conf


def _bare_booth_manager(db_path, attribute_backend=None):
    bm = object.__new__(BoothManager)
    bm.db_path = db_path
    bm.booth_id = "BOOTH-1"
    bm.event_id = "EVT-1"
    bm.camera_ids = ["CAM-1"]
    bm._lock = threading.Lock()
    bm._person_tracks = {"CAM-1": []}
    bm._tripwire_counters = {}
    bm.workers = {}
    bm.recent_alerts = []
    bm.session_manager = BoothSessionManager()
    bm.reid_registry = GlobalIdentityRegistry()
    bm._reid_sampler = ReIDSampler(sample_interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    bm.reid_embedder = _FakeEmbedder()
    bm.attribute_backend = attribute_backend
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    bm._attribute_sampler = AttributeSampler(interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    return bm


def _track(track_id, x1=100, y1=100, x2=200, y2=400, first_seen=0.0):
    return {"track_id": track_id, "bbox": [x1, y1, x2, y2], "first_seen": first_seen, "last_seen": first_seen,
            "category": "unknown"}


def _seed_booth(db_path):
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")


def _fake_frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def test_no_attribute_backend_is_a_no_op(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, attribute_backend=None)

    bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())

    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 0
    assert breakdown["gender"]["UNKNOWN"] == 0  # no row written at all, not even an UNKNOWN one


def test_attribute_analysis_writes_person_attributes_row(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    backend = _FakeAttributeBackend(gender="male", gender_conf=0.9, age_group="20-29", age_conf=0.8)
    bm = _bare_booth_manager(db_path, attribute_backend=backend)

    # One frame is never enough to label a person (see ATTRIBUTE_GENDER_MIN_SAMPLES)...
    bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())
    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 0
    # ...a few agreeing frames are.
    for _ in range(2):
        bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())

    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 1
    assert breakdown["age_category"]["CHILD"] == 0   # age / child is no longer reported
    assert breakdown["age_category"]["ADULT"] == 0


def test_low_confidence_prediction_stays_unknown(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    backend = _FakeAttributeBackend(gender="male", gender_conf=0.1, age_group="20-29", age_conf=0.1)
    bm = _bare_booth_manager(db_path, attribute_backend=backend)

    bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())

    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 0
    assert breakdown["gender"]["UNKNOWN"] == 1


def test_same_global_person_across_two_cameras_shares_one_attribute_row(tmp_path):
    """Spec section 18: attribute analysis is keyed by Global Person ID, so
    the same physical person appearing on two cameras (same fake embedding
    here => same Global Person) refines one row, not two."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, attribute_backend=_FakeAttributeBackend())
    bm.camera_ids = ["CAM-1", "CAM-2"]

    bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-2", [_track(9)], [], 640, 480, frame=_fake_frame())

    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 1  # still one distinct Global Person


def test_tripwire_crossing_carries_attribute_snapshot_when_available(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)
    bm = _bare_booth_manager(db_path, attribute_backend=_FakeAttributeBackend())
    bm._load_tripwire("CAM-1")

    # Establish Re-ID + a confident attribute profile for track 1 first.
    for _ in range(3):
        bm._on_person_tracks("CAM-1", [_track(1, x1=int(0.6 * 640) - 50, x2=int(0.6 * 640) + 50)],
                              [], 640, 480, frame=_fake_frame())
    # Now cross the tripwire, confirmed over two frames (core.tripwire's
    # TRIPWIRE_CONFIRM_FRAMES).
    for _ in range(2):
        bm._on_person_tracks("CAM-1", [_track(1, x1=int(0.3 * 640) - 50, x2=int(0.3 * 640) + 50)],
                              [], 640, 480, frame=_fake_frame())

    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(crossings) == 1
    snapshot = json.loads(crossings[0]["attribute_snapshot_json"])
    assert snapshot["gender"] == "MALE"
    assert snapshot["age_category"] == "UNKNOWN"
    assert "global_person_id" in snapshot


def test_tripwire_crossing_snapshot_is_none_without_attribute_backend(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)
    bm = _bare_booth_manager(db_path, attribute_backend=None)
    bm._load_tripwire("CAM-1")

    bm._on_person_tracks("CAM-1", [_track(1, x1=int(0.6 * 640) - 50, x2=int(0.6 * 640) + 50)],
                          [], 640, 480, frame=_fake_frame())
    for _ in range(2):
        bm._on_person_tracks("CAM-1", [_track(1, x1=int(0.3 * 640) - 50, x2=int(0.3 * 640) + 50)],
                              [], 640, 480, frame=_fake_frame())

    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(crossings) == 1
    assert crossings[0]["attribute_snapshot_json"] is None


def test_local_track_id_never_used_as_global_person_id(tmp_path):
    """Spec section 18: Local Track ID != Global Person ID. Track ID here is
    the plain int 1; the written global_id must be core.reid's own
    'P000001'-style identity, never the raw local track id."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, attribute_backend=_FakeAttributeBackend())

    bm._on_person_tracks("CAM-1", [_track(1)], [], 640, 480, frame=_fake_frame())

    global_id = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert global_id is not None
    assert global_id != "1" and global_id != 1
    assert global_id.startswith("P")
