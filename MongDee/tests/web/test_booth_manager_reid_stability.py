"""Booth-level Re-ID stability: flickering boxes / churned tracks must not become new visitors, a one-blink
false detection must not be counted, merges must reach the database, and far-away people get a gender from
whole-body evidence."""
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


def _unit(v):
    return v / np.linalg.norm(v)


LOOK = _unit(np.random.default_rng(1).normal(size=DIM))


def look(cos, seed):
    o = _unit(np.random.default_rng(seed).normal(size=DIM))
    o = _unit(o - LOOK * float(o @ LOOK))
    return _unit(cos * LOOK + np.sqrt(1 - cos ** 2) * o)


class _Embedder:
    def __init__(self):
        self.next = LOOK

    def embed(self, crop):
        return self.next


class _NoFaceBackend:                       # a far-away person: the face model can read nothing
    def predict_gender(self, crop):
        return "unknown", 0.0

    def predict_age_group(self, crop):
        return "unknown", 0.0


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _FakeLearner:
    class model:                            # noqa: N801 - mimics learner.model.predict
        @staticmethod
        def predict(features):
            return 0.92, 0.9

        @staticmethod
        def stats():
            return {}

    def observe(self, *a, **k):
        pass

    def merge(self, *a, **k):
        pass


def _manager(tmp_path, monkeypatch, backend=None, learner=None):
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
    if learner is not None:
        bm.body_learner = learner
    return bm, clock, db_path


def _track(tid, dx=0, first_seen=0.0):
    return {"track_id": tid, "bbox": [100 + dx, 100, 200 + dx, 400], "first_seen": first_seen,
            "last_seen": first_seen, "category": "unknown"}


def _frame():
    return np.full((480, 640, 3), 128, dtype=np.uint8)


def _see(bm, clock, tracks, evicted=(), dt=0.0):
    clock.t += dt
    bm._on_person_tracks("CAM-1", tracks, list(evicted), 640, 480, frame=_frame())


def test_a_one_blink_false_detection_is_never_counted_as_a_visitor(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch)
    _see(bm, clock, [_track(1, first_seen=clock.t)])
    assert bm.reid_registry.unique_people_count() == 0
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 0
    _see(bm, clock, [_track(1, first_seen=clock.t - 1.0)], dt=1.0)         # still there a second later: a real person
    assert bm.reid_registry.unique_people_count() == 1
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1


def test_a_flicker_that_churns_the_track_id_keeps_one_visitor(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch)
    _see(bm, clock, [_track(1, first_seen=clock.t)])
    _see(bm, clock, [_track(1, first_seen=clock.t - 1.0)], dt=1.0)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    _see(bm, clock, [], evicted=[{"track_id": 1, "first_seen": 0.0, "last_seen": 1.0, "category": "unknown"}], dt=1.0)
    bm.reid_embedder.next = look(0.35, 5)               # reborn a moment later, from another angle
    _see(bm, clock, [_track(2, dx=12, first_seen=clock.t)], dt=1.0)
    _see(bm, clock, [_track(2, dx=14, first_seen=clock.t - 1.0)], dt=1.0)
    assert bm.reid_registry.get_global_id_for("CAM-1", 2) == gid
    assert bm.reid_registry.unique_people_count() == 1
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
    assert bm.reid_registry.stats()["stitched"] >= 1


def test_registry_merges_reach_the_database_and_the_gender_evidence(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch, learner=_FakeLearner())
    bm.body_learner = _FakeLearner()
    db.upsert_global_person(db_path, "BOOTH-1", "EVT-1", "P000001", 1.0, 5.0, 1)
    db.upsert_global_person(db_path, "BOOTH-1", "EVT-1", "P000002", 2.0, 6.0, 1)
    bm.attribute_smoother.add_sample("P000002", "female", 0.9, now=clock.t)
    bm.attribute_smoother.add_sample("P000002", "female", 0.9, now=clock.t)
    bm.reid_registry._merge_log.append(("P000002", "P000001"))          # what the registry reports after merging
    bm._apply_reid_merges()
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
    assert bm.attribute_smoother.get("P000001").gender == "FEMALE"


def test_a_far_person_with_no_readable_face_gets_a_gender_from_whole_body_evidence(tmp_path, monkeypatch):
    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=_NoFaceBackend(), learner=_FakeLearner())
    for i in range(8):
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=1.0)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    result = bm.attribute_smoother.get(gid)
    assert result.gender == "MALE" and result.source == "body"
    rows = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert rows["gender"]["MALE"] == 1


def test_without_a_trained_body_model_a_far_person_stays_unknown(tmp_path, monkeypatch):
    class _Untrained(_FakeLearner):
        class model:                        # noqa: N801
            @staticmethod
            def predict(features):
                return None

    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=_NoFaceBackend(), learner=_Untrained())
    for i in range(8):
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=1.0)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert bm.attribute_smoother.get(gid).gender == "UNKNOWN"
