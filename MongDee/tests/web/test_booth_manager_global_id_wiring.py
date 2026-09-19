"""Confirms BoothManager._make_worker actually wires its real
GlobalIdentityRegistry into each CameraWorker it creates (spec: MongDee AI
Vision master prompt section 19 -- the live box overlay's global_id lookup
is worthless if this wiring is wrong, even though
core/test_camera_worker_global_id_overlay.py already proves the labeling
logic itself is correct in isolation).
"""

from __future__ import annotations

import threading
import time

import web.booth_manager as booth_manager_module
from core.attributes import GlobalPersonAttributeSmoother
from core.reid import GlobalIdentityRegistry, GlobalPerson
from web.booth_manager import BoothManager


def _bare_booth_manager():
    bm = object.__new__(BoothManager)
    bm.model = object()
    bm.catalog = type("FakeCatalog", (), {"product_keys": lambda self: []})()
    bm.recognizer = None
    bm.model_device = "cpu"
    bm.gender_age_backend = None
    bm.performance_monitor = None
    bm.ai_worker = None
    bm._on_frame = lambda *a: None
    bm._on_detections = lambda *a: None
    bm._on_status = lambda *a: None
    bm._on_person_tracks = lambda *a: None
    # require_confirmation=True matches the real BoothManager.__init__ wiring
    # (see web/booth_manager.py) -- this test is about validating that exact
    # production wiring, so it must reproduce that exact setting rather than
    # the registry's more permissive default.
    bm.reid_registry = GlobalIdentityRegistry(require_confirmation=True)
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    return bm


def test_make_worker_passes_the_real_reid_registry_lookup(monkeypatch):
    captured_kwargs = {}

    def fake_camera_worker(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(booth_manager_module, "CameraWorker", fake_camera_worker)
    bm = _bare_booth_manager()

    bm._make_worker("CAM-1", 0)

    assert captured_kwargs["global_id_resolver"] == bm.reid_registry.get_confirmed_global_id_for
    # And it's actually connected to the real registry, not a stand-in --
    # resolving a never-seen (camera_id, track_id) pair is None (no crash),
    # and once Re-ID maps AND CONFIRMS one, the same resolver reflects it.
    # A mapped-but-not-yet-confirmed track (still provisional/UNKNOWN) must
    # not resolve to an ID at all -- see core.reid's
    # get_confirmed_global_id_for docstring on why the on-screen label
    # deliberately uses the confirmed-only accessor, not get_global_id_for.
    resolver = captured_kwargs["global_id_resolver"]
    assert resolver("CAM-1", 999) is None
    bm.reid_registry._local_to_global[("CAM-1", 999)] = "P000001"
    bm.reid_registry._people["P000001"] = GlobalPerson(
        global_id="P000001", first_seen=0.0, last_seen=0.0, confirmed=False)
    assert resolver("CAM-1", 999) is None   # mapped, but not yet a confirmed visitor
    bm.reid_registry._people["P000001"].confirmed = True
    assert resolver("CAM-1", 999) == "P000001"


def test_make_worker_passes_the_real_attribute_smoother_lookup(monkeypatch):
    # Same wiring guarantee as the reid_registry test above, for
    # global_attribute_resolver -- this is what makes two cameras agree on
    # the same Global Person's gender/age instead of guessing independently.
    captured_kwargs = {}

    def fake_camera_worker(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(booth_manager_module, "CameraWorker", fake_camera_worker)
    bm = _bare_booth_manager()

    bm._make_worker("CAM-1", 0)

    assert captured_kwargs["global_attribute_resolver"] == bm.attribute_smoother.get
    resolver = captured_kwargs["global_attribute_resolver"]
    # Never-seen global_id -> "unknown" status, not a crash.
    assert resolver("P000001").status == "unknown"
    # And it's the real, shared smoother: a sample added on one "camera"
    # (i.e. through the same bm.attribute_smoother instance) is immediately
    # visible through this resolver, exactly like it would be to a second
    # camera's CameraWorker in the real multi-camera wiring.
    # Real wall-clock time, not 0.0 -- resolver("P000001") below calls
    # get() with no `now` (the real production call pattern), which
    # defaults to time.time(); an epoch-0 sample would look ~decades stale
    # against that and be (correctly) expired by
    # ATTRIBUTE_STALE_GRACE_SEC before this test ever reads it back.
    for _ in range(3):
        bm.attribute_smoother.add_sample("P000001", "female", 0.9, "20-29", 0.8, now=time.time())
    assert resolver("P000001").gender == "FEMALE"
