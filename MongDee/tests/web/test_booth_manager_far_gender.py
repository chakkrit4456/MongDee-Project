"""A far-away person who already has an identity but no gender is analysed on its own faster clock, small faces
are passed on with their size, and a person that is already labelled costs no extra analysis."""
from __future__ import annotations

from tests.web.test_booth_manager_reid_stability import _manager, _see, _track


class _DetailBackend:
    """A face model that only ever sees small faces (24 px) and is right 100% of the time."""

    def __init__(self, gender="female", conf=0.9, face_px=24.0):
        self.calls = 0
        self._d = {"gender": gender, "confidence": conf, "face_px": face_px}

    def predict_gender_detail(self, crop):
        self.calls += 1
        return dict(self._d)

    def predict_gender(self, crop):                     # must not be used when the detail call exists
        raise AssertionError("legacy call used")

    def predict_age_group(self, crop):
        return "unknown", 0.0


def test_small_faces_are_reported_with_their_size_and_label_the_person_after_a_few_frames(tmp_path, monkeypatch):
    backend = _DetailBackend()
    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=backend)
    for i in range(12):
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=0.4)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    result = bm.attribute_smoother.get(gid, clock.t)
    assert result.gender == "FEMALE" and result.source == "face"
    # 0.9-confidence 24 px faces are worth ~1/2.4 of a full face: the label needed more than 2 frames
    assert backend.calls >= 3


def test_an_unlabelled_known_person_is_analysed_between_reid_turns(tmp_path, monkeypatch):
    class _Never(_DetailBackend):
        def predict_gender_detail(self, crop):
            self.calls += 1
            return None                                  # no usable face, ever

    backend = _Never()
    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=backend)
    bm._reid_sampler._sample_interval_sec = 1.5          # Re-ID turns are rare ...
    bm._attribute_sampler._interval_sec = 1.5
    for i in range(12):                                  # ... one AI pass every 0.4 s
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=0.4)
    assert backend.calls >= 9                            # ... yet the unknown person is sampled ~every pass


def test_a_labelled_person_is_not_analysed_again_between_reid_turns(tmp_path, monkeypatch):
    backend = _DetailBackend(face_px=90.0)
    bm, clock, _ = _manager(tmp_path, monkeypatch, backend=backend)
    bm._reid_sampler._sample_interval_sec = 1.5
    bm._attribute_sampler._interval_sec = 1.5
    for i in range(4):
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=0.4)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    assert bm.attribute_smoother.get(gid, clock.t).gender == "FEMALE"
    calls = backend.calls
    for i in range(6):
        _see(bm, clock, [_track(1, first_seen=clock.t - 10)], dt=0.2)
    assert backend.calls - calls <= 1                    # only the ordinary 1.5 s clock, never the urgent one


# --------------------------------------------------------------------------------------- CLIP path
def test_with_clip_a_far_person_with_no_readable_face_gets_a_body_label(tmp_path, monkeypatch):
    import numpy as np

    from core.body_gender import BodyGenderLearner
    from core.clip_gender import ClipBodyGenderModel
    from tests.web.test_booth_manager_reid_stability import _NoFaceBackend

    dim = 16
    rng = np.random.default_rng(0)
    direction = rng.normal(size=dim)
    direction /= np.linalg.norm(direction)

    def feats(sign):
        emb = rng.normal(size=dim) * 0.4 + sign * direction
        return np.concatenate([[sign * 3.0], emb / np.linalg.norm(emb)]).astype(np.float32)

    model = ClipBodyGenderModel(dim)
    learner = BodyGenderLearner(model)
    for i in range(20):                                   # people the face model already decided, earlier
        sign = 1 if i % 2 == 0 else -1
        for _ in range(3):
            learner.observe(f"old{i}", feats(sign), "male" if sign > 0 else "female")
    assert model.ready()

    class _Clip:
        embedding_dim = dim

        def features(self, crop):
            return feats(1)                               # this far-away person looks male to CLIP

    bm, clock, db_path = _manager(tmp_path, monkeypatch, backend=_NoFaceBackend(), learner=learner)
    bm._clip_backend = _Clip()
    for i in range(10):
        _see(bm, clock, [_track(1, first_seen=clock.t - i)], dt=0.4)
    gid = bm.reid_registry.get_global_id_for("CAM-1", 1)
    result = bm.attribute_smoother.get(gid, clock.t)
    assert result.gender == "MALE" and result.source == "body"
