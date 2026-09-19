"""Integration test for web/booth_manager.py's Gender/Age Attribution on
Product Interest (hold) events — spec: MongDee person-gender/age master
prompt sections 22/26/27/34 ("Interest Event Schema" / "Gender Attribution"
/ "Age Attribution" / "Unique Person vs Interest Events").

Same "bare" BoothManager pattern as test_booth_manager_tripwire.py /
test_booth_manager_attributes.py.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from core import database as db
from core.aggregator import DetectionAggregator
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.interest_tracker import INTEREST_CONFIRMATION_SECONDS, ProductInterestTracker
from core.reid import GlobalIdentityRegistry, ReIDSampler
from web.booth_manager import BoothManager


def _bare_booth_manager(db_path):
    bm = object.__new__(BoothManager)
    bm.db_path = db_path
    bm.booth_id = "BOOTH-1"
    bm.event_id = "EVT-1"
    bm.camera_ids = ["CAM-1"]
    bm._lock = threading.Lock()
    bm._person_tracks = {"CAM-1": []}
    bm._product_detections = {"CAM-1": []}
    bm._tripwire_counters = {}
    bm.workers = {}
    bm.recent_alerts = []
    # WATER must be a live catalog product: a class that is NOT in the catalog is a "ghost"
    # (deleted product) and BoothManager now ignores it - see test_ghost_product_* below.
    bm.catalog = {"WATER": {"name": "WATER"}}
    bm.aggregator = DetectionAggregator()
    bm.interest_tracker = ProductInterestTracker()
    bm._open_interest_rows = {}
    bm.session_manager = BoothSessionManager()
    bm.reid_registry = GlobalIdentityRegistry()
    bm._reid_sampler = ReIDSampler()
    bm.reid_embedder = None
    bm.attribute_backend = None
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    bm._attribute_sampler = AttributeSampler()
    return bm


def _fire_one_hold_event(bm, holder_track_id=7):
    """Drives ProductInterestTracker through a full pick-up -> release cycle
    for camera CAM-1 (see core/interest_tracker.py's state machine): a
    product resting at (100,100)-(140,180), then picked up by a person track
    that overlaps its new position, held for >= INTEREST_CONFIRMATION_
    SECONDS (so it actually confirms — see core/interest_tracker.py), then
    released with no holder nearby. Returns nothing — the events land in
    product_hold_events via BoothManager._on_detections's confirm/finalize
    wiring."""
    resting = [100.0, 100.0, 140.0, 180.0]
    moved = [200.0, 100.0, 240.0, 180.0]
    holder_bbox = [180.0, 90.0, 260.0, 190.0]  # covers moved product's centroid (220,140)

    bm._on_detections("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": resting}])

    bm._person_tracks["CAM-1"] = [{"track_id": holder_track_id, "bbox": holder_bbox, "first_seen": 0.0}]
    bm._on_detections("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": moved}])  # hold begins

    time.sleep(INTEREST_CONFIRMATION_SECONDS + 0.1)
    # still holding -> this tick is what fires the "confirmed" event live,
    # before any release (spec section 20: confirm at 2s, not at release)
    bm._on_detections("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": moved}])

    bm._person_tracks["CAM-1"] = []
    released = [202.0, 102.0, 242.0, 182.0]  # close to `moved` -> reads as "not moved"
    bm._on_detections("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": released}])  # finalize


def test_hold_event_without_reid_has_no_attribution(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    bm = _bare_booth_manager(db_path)

    _fire_one_hold_event(bm)

    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert len(rows) == 1
    assert rows[0]["holder_track_id"] == 7
    assert rows[0]["global_person_id"] is None
    assert rows[0]["gender"] is None
    assert rows[0]["age_category"] is None


def test_hold_event_snapshots_resolved_global_person_and_attributes(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    bm = _bare_booth_manager(db_path)

    # Seed Re-ID + an "ok" attribute profile for local track 7 on CAM-1,
    # standing in for what _update_reid/_update_attributes would normally
    # populate from real camera frames.
    global_id, _is_new = bm.reid_registry.resolve("CAM-1", 7, np.array([1.0, 0.0, 0.0, 0.0]), time.time())
    for _ in range(3):
        bm.attribute_smoother.add_sample(global_id, "female", 0.9, "20-29", 0.8, time.time())

    _fire_one_hold_event(bm, holder_track_id=7)

    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert len(rows) == 1
    assert rows[0]["global_person_id"] == global_id
    assert rows[0]["gender"] == "FEMALE"
    assert rows[0]["age_category"] == "UNKNOWN"   # age is no longer analysed

    overview = db.query_interest_overview(db_path, booth_id="BOOTH-1")
    assert overview == {"total_interest_events": 1, "unique_interested_persons": 1}


def test_global_person_id_recorded_even_without_an_ok_attribute_profile(tmp_path):
    """Spec section 34: unique_persons must count a resolved identity even
    when no gender/age profile exists yet for it (attribute_backend unset,
    or AttributeSampler hasn't cleared this person yet) — global_person_id
    is never gated on attributes being 'ok'."""
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    bm = _bare_booth_manager(db_path)

    global_id, _is_new = bm.reid_registry.resolve("CAM-1", 7, np.array([1.0, 0.0, 0.0, 0.0]), time.time())
    # deliberately no attribute_smoother.add_sample call

    _fire_one_hold_event(bm, holder_track_id=7)

    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert rows[0]["global_person_id"] == global_id
    assert rows[0]["gender"] is None  # honest: no attribute data, not a guess


def test_ghost_product_detection_is_ignored_end_to_end(tmp_path):
    """A detection whose product is not in the catalog (deleted, stale gallery, stale worker)
    must not reach the snapshot, the aggregator or the interest tracker, and must never write
    an interest row."""
    db_path = tmp_path / "t.db"
    db.init_db(db_path) if hasattr(db, "init_db") else None
    bm = _bare_booth_manager(db_path)
    bm.catalog = {}                                     # the product was deleted
    _fire_one_hold_event(bm)
    assert bm._product_detections["CAM-1"] == []
    assert bm.interest_tracker.live_states() == []
    assert bm._open_interest_rows == {}
    assert bm.get_live_analytics()["products"] == []
