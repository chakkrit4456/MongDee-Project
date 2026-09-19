"""Field bug (2026-09-19): a woman on CAM-1 and a man on CAM-2 - different gender, clothes, hair - were shown with the
SAME global id and the man got the woman's 'FEMALE' label. Whatever the appearance embedding says, a track whose own
face says 'male' must be split off an identity already decided 'female'."""
from __future__ import annotations

import numpy as np

from tests.web.test_booth_manager_reid_stability import _manager, _track, _frame


class _ByFrameBackend:
    """Reads 'gender' from the brightness of the crop: bright frame = woman's camera, dark = man's."""

    def predict_gender_detail(self, crop):
        woman = float(np.mean(crop)) > 150
        return {"gender": "female" if woman else "male", "confidence": 0.95, "face_px": 90.0}

    def predict_gender(self, crop):
        d = self.predict_gender_detail(crop)
        return d["gender"], d["confidence"]

    def predict_age_group(self, crop):
        return "unknown", 0.0


def _step(bm, clock, dt=0.4):
    clock.t += dt
    bright = np.full((480, 640, 3), 200, dtype=np.uint8)
    dark = np.full((480, 640, 3), 60, dtype=np.uint8)
    bm._on_person_tracks("CAM-1", [_track(1, first_seen=clock.t - 20)], [], 640, 480, frame=bright)
    bm._on_person_tracks("CAM-2", [_track(1, first_seen=clock.t - 20)], [], 640, 480, frame=dark)


def test_a_man_is_not_kept_under_a_womans_id_even_when_the_embeddings_are_identical(tmp_path, monkeypatch):
    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=_ByFrameBackend())
    bm.camera_ids = ["CAM-1", "CAM-2"]
    bm._person_tracks = {"CAM-1": [], "CAM-2": []}
    for _ in range(14):                                     # the embedder returns the SAME vector for both: worst case
        _step(bm, clock)
    woman = bm.reid_registry.get_global_id_for("CAM-1", 1)
    man = bm.reid_registry.get_global_id_for("CAM-2", 1)
    assert woman != man
    assert bm.attribute_smoother.get(woman, clock.t).gender == "FEMALE"
    assert bm.attribute_smoother.get(man, clock.t).gender == "MALE"
    assert bm.reid_registry.stats()["split_wrong_identity"] >= 1


def test_two_cameras_and_one_woman_keep_one_id(tmp_path, monkeypatch):
    class _Woman(_ByFrameBackend):
        def predict_gender_detail(self, crop):
            return {"gender": "female", "confidence": 0.95, "face_px": 90.0}

    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=_Woman())
    bm.camera_ids = ["CAM-1", "CAM-2"]
    bm._person_tracks = {"CAM-1": [], "CAM-2": []}
    for _ in range(14):
        _step(bm, clock)
    assert bm.reid_registry.get_global_id_for("CAM-1", 1) == bm.reid_registry.get_global_id_for("CAM-2", 1)
    assert bm.reid_registry.stats()["split_wrong_identity"] == 0
