"""A visitor is only counted (written to `global_persons`, the row behind
db.query_unique_people_count / the Dashboard's Unique People total) once their identity is
confirmed AND — when this booth runs gender classification at all — their gender has reached
confident, stable evidence. An UNKNOWN person (no readable face yet, or evidence too weak) must
never be counted as a customer, and must never be double-counted or given a new id once their
gender does become confident: the same global_id starts counting in place.

Booths with no gender backend configured have nothing to wait for and keep counting by identity
confirmation alone (core.reid.GlobalIdentityRegistry.is_counted) — this file also locks in that
fallback so the gate added in web.booth_manager.BoothManager._maybe_count_customer never breaks
booths that don't classify gender at all."""
from __future__ import annotations

import threading

import numpy as np

from core import database as db
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.booth_sessions import BoothSessionManager
from core.reid import GlobalIdentityRegistry, ReIDSampler
from web import booth_manager as booth_manager_module
from web.booth_manager import BoothManager

DIM = 32
LOOK = np.random.default_rng(7).normal(size=DIM)
LOOK = LOOK / np.linalg.norm(LOOK)


class _Embedder:
    def embed(self, crop):
        return LOOK


class _UnknownBackend:
    """Never produces a readable face — every observation stays UNKNOWN, forever."""

    def predict_gender(self, crop):
        return "unknown", 0.0

    def predict_age_group(self, crop):
        return "unknown", 0.0


class _ConfidentMaleBackend:
    def predict_gender(self, crop):
        return "male", 0.95

    def predict_age_group(self, crop):
        return "unknown", 0.0


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _manager(tmp_path, monkeypatch, backend):
    clock = _Clock()
    monkeypatch.setattr(booth_manager_module.time, "time", clock)
    db_path = tmp_path / "t.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    bm = object.__new__(BoothManager)
    bm.db_path, bm.booth_id, bm.event_id = db_path, "BOOTH-1", "EVT-1"
    bm.camera_ids = ["CAM-1"]
    bm._lock = threading.Lock()
    bm._person_tracks = {"CAM-1": []}
    bm._tripwire_counters = {}
    bm.workers = {}
    bm.recent_alerts = []
    bm.session_manager = BoothSessionManager()
    bm.reid_registry = GlobalIdentityRegistry(require_confirmation=True)
    bm._reid_sampler = ReIDSampler(sample_interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    bm.reid_embedder = _Embedder()
    bm.attribute_backend = backend
    bm.attribute_smoother = GlobalPersonAttributeSmoother()
    bm._attribute_sampler = AttributeSampler(interval_sec=0.0, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    return bm, clock, db_path


def _track(tid, first_seen=0.0):
    return {"track_id": tid, "bbox": [100, 100, 200, 400], "first_seen": first_seen,
            "last_seen": first_seen, "category": "unknown"}


def _frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def _see(bm, clock, tracks, evicted=(), dt=0.0):
    clock.t += dt
    bm._on_person_tracks("CAM-1", tracks, list(evicted), 640, 480, frame=_frame())


def test_unknown_gender_person_is_not_counted_even_once_identity_is_confirmed(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=_UnknownBackend())
    _see(bm, clock, [_track(1, first_seen=clock.t)])
    _see(bm, clock, [_track(1, first_seen=clock.t - 1.0)], dt=1.0)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert gid is not None
    assert bm.reid_registry.is_counted(gid)          # the identity itself is confirmed (real, not a blink) ...
    assert bm.attribute_smoother.get(gid, clock.t).status == "unknown"
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 0   # ... but never counted as a customer
    assert bm.attribute_smoother.get(gid, clock.t).gender == "UNKNOWN"


def test_person_is_counted_the_moment_gender_becomes_confident_same_id(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=_ConfidentMaleBackend())
    _see(bm, clock, [_track(1, first_seen=clock.t)])
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 0   # not yet: only 1 observation so far
    _see(bm, clock, [_track(1, first_seen=clock.t - 1.0)], dt=1.0)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert bm.attribute_smoother.get(gid, clock.t).status == "ok"
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
    row = db.query_unique_people_count(db_path, booth_id="BOOTH-1")
    assert row == 1   # counted exactly once, under the SAME global_id — no re-identification


def test_no_gender_backend_counts_by_identity_confirmation_alone(tmp_path, monkeypatch):
    """A booth running no gender classifier has nothing to wait for: unchanged behaviour."""
    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=None)
    _see(bm, clock, [_track(1, first_seen=clock.t)])
    _see(bm, clock, [_track(1, first_seen=clock.t - 1.0)], dt=1.0)
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
