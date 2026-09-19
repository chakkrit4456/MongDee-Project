"""Booth page must count PEOPLE, not tracks: one Global Person ID seen by two cameras is one person (field bug
2026-09-19: 'the same ID shows as 2 people'). Also: face identity drives Re-ID across cameras."""
from __future__ import annotations

import numpy as np

from tests.web.test_booth_manager_reid_stability import _manager, _track


class _FaceByBrightness:
    """Fake FaceIdentity: the crop's brightness encodes WHOSE face it is (bright = person 0, dark = person 1);
    `same` forces one face for everyone."""

    def __init__(self, same=False):
        self.same = same
        rng = np.random.default_rng(1)
        self.faces = [v / np.linalg.norm(v) for v in rng.normal(size=(2, 128))]

    def embed(self, crop):
        return self.faces[0 if self.same or float(np.mean(crop)) > 150 else 1]


class _Backend:
    def predict_gender_detail(self, crop):
        return {"gender": "female" if float(np.mean(crop)) > 150 else "male", "confidence": 0.95, "face_px": 90.0}

    def predict_gender(self, crop):
        d = self.predict_gender_detail(crop)
        return d["gender"], d["confidence"]

    def predict_age_group(self, crop):
        return "unknown", 0.0


def _setup(tmp_path, monkeypatch, same_face):
    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=_Backend())
    bm.camera_ids = ["CAM-1", "CAM-2"]
    bm._person_tracks = {"CAM-1": [], "CAM-2": []}
    bm._product_detections = {}
    bm.face_identity = _FaceByBrightness(same=same_face)
    for _ in range(14):
        clock.t += 0.4
        bright = np.full((480, 640, 3), 200, dtype=np.uint8)
        dark = np.full((480, 640, 3), 60, dtype=np.uint8)
        t1, t2 = _track(1, first_seen=clock.t - 20), _track(1, first_seen=clock.t - 20)
        t1["category"], t2["category"] = "female", "unknown"
        bm._on_person_tracks("CAM-1", [t1], [], 640, 480, frame=bright)
        bm._on_person_tracks("CAM-2", [t2], [], 640, 480, frame=dark)
    return bm, clock


def test_one_id_seen_by_two_cameras_is_counted_once(tmp_path, monkeypatch):
    bm, _ = _setup(tmp_path, monkeypatch, same_face=True)
    assert bm.reid_registry.get_global_id_for("CAM-1", 1) == bm.reid_registry.get_global_id_for("CAM-2", 1)
    assert bm._distinct_people_now() == 1                      # was 2 (one per camera)


def test_two_different_faces_are_two_ids_and_two_people(tmp_path, monkeypatch):
    bm, _ = _setup(tmp_path, monkeypatch, same_face=False)
    assert bm.reid_registry.get_global_id_for("CAM-1", 1) != bm.reid_registry.get_global_id_for("CAM-2", 1)
    assert bm._distinct_people_now() == 2
    assert bm.reid_registry.stats()["face_matches"] == 0


def test_camera_snapshot_counts_one_person_per_id(tmp_path, monkeypatch):
    bm, _ = _setup(tmp_path, monkeypatch, same_face=True)
    # two tracks of the same camera mapped to the same identity (a flickering box) are one person
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    bm.reid_registry._local_to_global[("CAM-1", 2)] = gid
    bm._person_tracks["CAM-1"] = [{"track_id": 1, "bbox": [0, 0, 1, 1], "category": "female"},
                                  {"track_id": 2, "bbox": [0, 0, 1, 1], "category": "unknown"}]
    snap = bm.get_camera_snapshot("CAM-1")
    assert snap["people_total"] == 1
    assert snap["male"] + snap["female"] + snap["unknown"] == 1        # one person, one category


def test_one_id_shows_one_gender_on_every_camera(tmp_path, monkeypatch):
    bm, clock = _setup(tmp_path, monkeypatch, same_face=True)
    a = bm._resolved_gender_for_track("CAM-1", 1)
    b = bm._resolved_gender_for_track("CAM-2", 1)
    assert a is not None and a[0] == b[0]                          # never female here and male there


def test_a_person_without_decided_gender_still_gets_a_best_guess(tmp_path, monkeypatch):
    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=_Backend())
    bm.reid_registry.observe_frame("CAM-1", [1], clock.t)
    gid, _new = bm.reid_registry.resolve("CAM-1", 1, np.ones(8, np.float32), clock.t, bbox=[0, 0, 10, 20])
    assert bm._resolved_gender_for_track("CAM-1", 1) is None                # nothing observed yet: pending
    bm.attribute_smoother.add_sample(gid, "female", 0.9, now=clock.t)       # ONE face sample: below the "decided" bar
    assert bm.attribute_smoother.get(gid, clock.t).status != "ok"
    guess = bm._resolved_gender_for_track("CAM-1", 1)
    assert guess is not None and guess[0] == "female"                       # ... but it is not UNKNOWN any more
