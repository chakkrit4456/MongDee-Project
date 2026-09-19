"""Integration tests for web/booth_manager.py's Booth Session / Dwell Time
wiring — spec: MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md
sections 29 ("Tripwire กับ Global Identity"), 30 ("Double Count
Prevention") and the broader booth-session/dwell-time requirement (a
continuous ENTRY->EXIT visit per identity must survive a camera switch).

Same "bare" BoothManager pattern as test_booth_manager_tripwire.py /
test_booth_manager_attributes.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from core import database as db
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.reid import GlobalIdentityRegistry, ReIDSampler
from core.tripwire import CROSSING_COOLDOWN_SEC
from web.booth_manager import BoothManager


class _FakeEmbedder:
    """Deterministic fake Re-ID embedder — same fixed vector every call, so
    every track resolves to the same Global Person without needing a real
    MobileNetV3 model (same fake used by test_booth_manager_attributes.py)."""

    def embed(self, crop):
        return np.array([1.0, 0.0, 0.0, 0.0])


def _bare_booth_manager(db_path, camera_ids=("CAM-1",), with_reid=False):
    bm = object.__new__(BoothManager)
    bm.db_path = db_path
    bm.booth_id = "BOOTH-1"
    bm.event_id = "EVT-1"
    bm.camera_ids = list(camera_ids)
    bm._lock = threading.Lock()
    bm._person_tracks = {cid: [] for cid in camera_ids}
    bm._tripwire_counters = {}
    bm.workers = {}
    bm.recent_alerts = []
    bm.session_manager = BoothSessionManager()
    bm.reid_registry = GlobalIdentityRegistry()
    bm._reid_sampler = ReIDSampler(sample_interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    bm.reid_embedder = _FakeEmbedder() if with_reid else None
    bm.attribute_backend = None
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    bm._attribute_sampler = AttributeSampler(interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    return bm


def _track(track_id, nx, ny=0.5, frame_w=640, frame_h=480, first_seen=0.0):
    px, py = nx * frame_w, ny * frame_h
    return {"track_id": track_id, "bbox": [px - 10, py - 40, px + 10, py],
            "first_seen": first_seen, "last_seen": first_seen, "category": "unknown"}


def _fake_frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def _seed_booth(db_path):
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")


def _configure_tripwire(bm, camera_id, inside_side="A"):
    db.upsert_tripwire(bm.db_path, f"TW-{camera_id}", bm.booth_id, camera_id,
                        0.5, 0.0, 0.5, 1.0, inside_side, True)
    bm._load_tripwire(camera_id)


def test_entry_then_exit_opens_and_closes_a_booth_session_without_reid(tmp_path):
    """Re-ID disabled (reid_embedder=None) — dwell tracking still works via
    the per-camera fallback key, just without cross-camera merging."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, with_reid=False)
    _configure_tripwire(bm, "CAM-1")

    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)  # baseline: outside
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)  # candidate crossing
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)  # confirmed IN

    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert len(sessions) == 1
    assert sessions[0]["status"] == "active"
    assert sessions[0]["global_person_id"] is None  # honest: Re-ID never resolved this
    assert bm.session_manager.active_count() == 1

    # TripwireCounter debounces further crossings from the same track for
    # CROSSING_COOLDOWN_SEC after the IN above — wait it out so the OUT
    # below is a genuine second crossing, not a suppressed duplicate.
    time.sleep(CROSSING_COOLDOWN_SEC + 0.1)
    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)  # candidate exit
    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)  # confirmed OUT

    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert sessions[0]["status"] == "closed"
    assert sessions[0]["dwell_seconds"] is not None and sessions[0]["dwell_seconds"] >= 0
    assert bm.session_manager.active_count() == 0


def test_crossing_stores_resolved_global_person_id_when_reid_is_enabled(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, with_reid=True)
    _configure_tripwire(bm, "CAM-1")

    # First reading resolves Re-ID (fake embedder -> same global id every
    # time) *and* establishes the tripwire baseline side; no crossing yet.
    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480, frame=_fake_frame())
    global_id = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert global_id is not None

    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())

    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert crossings[0]["direction"] == "in"
    assert crossings[0]["global_person_id"] == global_id

    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert sessions[0]["global_person_id"] == global_id
    assert sessions[0]["person_key"] == global_id


def test_duplicate_entry_across_two_cameras_does_not_open_a_second_session(tmp_path):
    """Spec section 30: CAM-A P IN then CAM-B P IN (same Global Person) must
    not double the active-session count."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, camera_ids=("CAM-1", "CAM-2"), with_reid=True)
    _configure_tripwire(bm, "CAM-1")
    _configure_tripwire(bm, "CAM-2")

    # CAM-1: resolve Re-ID, then cross IN.
    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())
    assert bm.session_manager.active_count() == 1

    # CAM-2: a *different* local track_id, but the fake embedder resolves it
    # to the same Global Person -> its own IN crossing must not open a
    # second session.
    bm._on_person_tracks("CAM-2", [_track(9, 0.7)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-2", [_track(9, 0.3)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-2", [_track(9, 0.3)], [], 640, 480, frame=_fake_frame())

    assert bm.session_manager.active_count() == 1
    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert len(sessions) == 1
    assert sorted(sessions[0]["cameras_seen"]) == ["CAM-1", "CAM-2"]
    # both cameras' own crossings are still individually observable
    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(crossings) == 2


def test_camera_switch_mid_visit_is_recorded_without_a_second_tripwire(tmp_path):
    """Spec section 21/29: Re-ID observing the same Global Person at a
    second camera (no tripwire there at all) must still show up in the open
    session's cameras_seen."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, camera_ids=("CAM-1", "CAM-2"), with_reid=True)
    _configure_tripwire(bm, "CAM-1")  # CAM-2 has no tripwire configured

    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480, frame=_fake_frame())
    assert bm.session_manager.active_count() == 1

    bm._on_person_tracks("CAM-2", [_track(9, 0.5)], [], 640, 480, frame=_fake_frame())

    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert sorted(sessions[0]["cameras_seen"]) == ["CAM-1", "CAM-2"]
    assert sessions[0]["status"] == "active"  # still one open visit, not closed by the sighting


def test_get_booth_session_status_reports_live_active_sessions(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    bm = _bare_booth_manager(db_path, with_reid=False)
    _configure_tripwire(bm, "CAM-1")

    assert bm.get_booth_session_status() == {"active_count": 0, "sessions": []}

    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)

    status = bm.get_booth_session_status()
    assert status["active_count"] == 1
    assert status["sessions"][0]["status"] == "active"
