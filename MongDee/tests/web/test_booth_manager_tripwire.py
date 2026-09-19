"""Integration tests for web/booth_manager.py's Virtual Tripwire wiring —
does a real crossing get detected, logged to the database with the right
booth_id/event_id, and pushed to the CameraWorker as a live overlay?

BoothManager's real constructor loads a YOLO model, a recognizer, camera
devices, etc. — none of which _on_person_tracks/_update_tripwire/
_load_tripwire touch — so these tests build a "bare" instance with
__init__ skipped and only the handful of attributes those methods actually
read/write set by hand. This is the same reason web/ has no other unit
tests today (see BUILD.md / this project's own test layout): the rest of
BoothManager is exercised live, not through a mocked constructor. The
tripwire wiring is new and easy to get subtly wrong (booth_id/event_id
mixups, forgetting to push the overlay, etc.), so it earns a direct test
despite that precedent.
"""

from __future__ import annotations

import threading
import time

from core import database as db
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.reid import GlobalIdentityRegistry, ReIDSampler
from core.vision import CameraWorker
from web.booth_manager import BoothManager


def _bare_booth_manager(db_path):
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
    bm._reid_sampler = ReIDSampler()
    bm.reid_embedder = None  # no CNN load in tests — _update_reid() is a no-op
    bm.attribute_backend = None  # no FairFace load in tests — _update_attributes() is a no-op
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    bm._attribute_sampler = AttributeSampler()
    return bm


def _track(track_id, nx, ny=0.5, frame_w=640, frame_h=480):
    px, py = nx * frame_w, ny * frame_h
    return {"track_id": track_id, "bbox": [px - 10, py - 40, px + 10, py],
            "first_seen": 0.0, "last_seen": 0.0, "category": "unknown"}


def _seed_booth(db_path):
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")


def test_load_tripwire_reads_saved_config(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)

    bm = _bare_booth_manager(db_path)
    bm._load_tripwire("CAM-1")
    counter = bm._tripwire_counters["CAM-1"]
    assert counter.get_line().inside_side == "A"


def test_crossing_is_logged_to_database_and_pushes_alert(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)

    bm = _bare_booth_manager(db_path)
    bm._load_tripwire("CAM-1")

    # First reading establishes SIDE_B (outside); no crossing yet.
    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)
    assert db.query_tripwire_stats(db_path, event_id="EVT-1")["total_in"] == 0

    # Crosses to SIDE_A (inside) -> one IN event, logged under this booth's
    # own booth_id/event_id (not hard-coded/mixed up). core.tripwire requires
    # TRIPWIRE_CONFIRM_FRAMES (2) consecutive readings on the new side before
    # it treats the flip as a real crossing rather than a single noisy frame.
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)
    stats = db.query_tripwire_stats(db_path, event_id="EVT-1")
    assert stats == {"total_in": 1, "total_out": 0, "current_inside": 1, "peak_inside": 1}

    crossings = db.query_tripwire_crossings(db_path, event_id="EVT-1")
    assert len(crossings) == 1
    assert crossings[0]["camera_id"] == "CAM-1"
    assert crossings[0]["track_id"] == 1
    assert crossings[0]["direction"] == "in"
    assert crossings[0]["booth_id"] == "BOOTH-1"
    assert crossings[0]["event_id"] == "EVT-1"

    assert len(bm.recent_alerts) == 1
    assert bm.recent_alerts[0]["type"] == "tripwire_in"


def test_no_configured_line_logs_nothing(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    # deliberately never call upsert_tripwire

    bm = _bare_booth_manager(db_path)
    bm._load_tripwire("CAM-1")

    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)
    assert db.query_tripwire_crossings(db_path, event_id="EVT-1") == []


def test_overlay_pushed_to_camera_worker_on_load_and_on_crossing(tmp_path):
    """Confirms _load_tripwire/_update_tripwire actually call
    CameraWorker.set_tripwire_overlay() with the line and fresh counts —
    the mechanism that keeps the line + IN/OUT numbers visible on the live
    video (core/vision.py's _draw_tripwire) independent of AI pass rate."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)

    bm = _bare_booth_manager(db_path)

    # A CameraWorker that's never had .start()/run() called touches no
    # camera I/O and starts no thread — only its plain
    # set_tripwire_overlay() attribute setter is exercised here.
    worker = CameraWorker(camera_id="CAM-1", device=0, model=type("M", (), {"names": {0: "person"}})(),
                           allowed_classes=[])
    bm.workers = {"CAM-1": worker}
    bm._load_tripwire("CAM-1")
    assert worker._tripwire_overlay is not None  # line pushed immediately on load
    assert worker._tripwire_overlay["count_in"] == 0

    bm._on_person_tracks("CAM-1", [_track(1, 0.7)], [], 640, 480)
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)  # candidate crossing, frame 1/2
    bm._on_person_tracks("CAM-1", [_track(1, 0.3)], [], 640, 480)  # confirmed crossing IN, frame 2/2
    assert worker._tripwire_overlay["count_in"] == 1
    assert worker._tripwire_overlay["count_out"] == 0


def test_remove_tripwire_clears_overlay_and_line(tmp_path):
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.upsert_tripwire(db_path, "TW-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)

    bm = _bare_booth_manager(db_path)
    worker = CameraWorker(camera_id="CAM-1", device=0, model=type("M", (), {"names": {0: "person"}})(),
                           allowed_classes=[])
    bm.workers = {"CAM-1": worker}
    bm._load_tripwire("CAM-1")
    assert worker._tripwire_overlay is not None

    bm.remove_tripwire("CAM-1")
    assert worker._tripwire_overlay is None
    assert bm._tripwire_counters["CAM-1"].get_line() is None
    assert db.get_tripwire(db_path, "BOOTH-1", "CAM-1") is None


def test_activate_booth_reloads_tripwire_for_new_booth_scope(tmp_path):
    """A tripwire is scoped per (booth_id, camera_id) — switching which
    registry booth this process reports as must reload each camera's line
    from *that* booth's own saved config, not keep showing/counting
    against whichever booth was active a moment ago (see
    BoothManager.activate_booth).

    activate_booth() also persists the newly-active booth id to a
    booth_settings.json next to db_path (booth_settings_path_for) — since
    db_path here is already inside tmp_path, that side effect is
    automatically isolated too, with nothing to monkeypatch."""
    db_path = tmp_path / "test.db"
    _seed_booth(db_path)
    db.create_booth(db_path, "BOOTH-2", "Booth 2", "EVT-1")
    db.upsert_tripwire(db_path, "TW-B1-CAM-1", "BOOTH-1", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)
    db.upsert_tripwire(db_path, "TW-B2-CAM-1", "BOOTH-2", "CAM-1", 0.2, 0.0, 0.2, 1.0, "B", True)

    bm = _bare_booth_manager(db_path)
    bm._load_tripwire("CAM-1")
    assert bm._tripwire_counters["CAM-1"].get_line().id == "TW-B1-CAM-1"

    bm.booth_name = "Booth 1"
    bm.activate_booth("BOOTH-2")
    assert bm.booth_id == "BOOTH-2"
    assert bm._tripwire_counters["CAM-1"].get_line().id == "TW-B2-CAM-1"
    assert bm._tripwire_counters["CAM-1"].get_line().inside_side == "B"
