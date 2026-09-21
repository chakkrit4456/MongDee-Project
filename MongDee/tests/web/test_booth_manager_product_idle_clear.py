"""Regression test for a stale-detection bug found while fixing MongDee's
ghost-product-detection issue: core.aggregator.DetectionAggregator times out
its own `current_product` after IDLE_RESET_SEC of no sightings (see
_decide()), but only ever *notifies* BoothManager of a NEW confirmed
product via the dict update() returns -- going idle produces no event at
all. web/booth_manager.py's _on_detections used to only ever set
self.current_product, never clear it, so /api/state (and therefore the
booth UI) kept showing the last product seen indefinitely, even long after
it left every camera. See web/booth_manager.py's _on_detections /
_clear_current_product and web/static/product_view.js's pollState.

Same "bare" BoothManager construction pattern as
test_booth_manager_interest_attribution.py.
"""

from __future__ import annotations

import threading

from core import database as db
from core.aggregator import DetectionAggregator
import core.aggregator as aggregator_module
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.interest_tracker import ProductInterestTracker
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
    bm.catalog = {"WATER": {"name": "WATER", "tagline": "", "price": "", "description": "", "faq": []}}
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
    bm.current_product = None
    bm.product_seq = 0
    return bm


def test_current_product_clears_once_the_product_leaves_every_camera(tmp_path, monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(aggregator_module.time, "monotonic", lambda: fake_now[0])

    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    bm = _bare_booth_manager(db_path)

    box = [10.0, 10.0, 60.0, 60.0]
    det = [{"class_name": "WATER", "conf": 0.9, "bbox": box}]
    # Repeated sightings close enough together (well under CONCURRENT_WINDOW_SEC
    # each) to build up SINGLE_CAMERA_STABLE_SEC of continuous evidence -- what a
    # real product sitting in frame across many AI passes looks like.
    for _ in range(6):
        bm._on_detections("CAM-1", det)
        fake_now[0] += 0.3

    assert bm.current_product is not None
    assert bm.current_product["key"] == "WATER"
    seq_when_confirmed = bm.product_seq

    # The product is now gone from every camera. Once IDLE_RESET_SEC has
    # passed with nothing seen, the very next _on_detections call (even
    # reporting nothing) must clear the stale product instead of leaving it
    # displayed forever.
    fake_now[0] += aggregator_module.IDLE_RESET_SEC + 0.1
    bm._on_detections("CAM-1", [])

    assert bm.current_product is None
    # Going idle is not itself a new "product recognized" event.
    assert bm.product_seq == seq_when_confirmed
