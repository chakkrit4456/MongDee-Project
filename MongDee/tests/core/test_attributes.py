"""Unit tests for core/attributes.py — spec section 19 (Testing) of
MongDee_Master_Prompt_FairFace_Age_Gender.md.

FairFaceBackend is tested against a randomly-initialized ResNet34 with the
correct 18-wide fc head (race7+gender2+age9) saved to a tmp checkpoint —
this verifies the *parsing/plumbing* (correct logit slicing, softmax,
label mapping, face-detection gating, device fallback) is correct, not real
prediction accuracy, since no real FairFace weights are available in this
environment (see module docstring: this repo never bundles/auto-downloads
third-party model weights). Real-accuracy validation against actual
photos/faces is listed as a remaining limitation in the final report.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision.models import resnet34

from core.attributes import (
    AGE_GROUP_LABELS,
    ATTRIBUTE_AGE_MIN_CONFIDENCE,
    ATTRIBUTE_GENDER_MIN_CONFIDENCE,
    ATTRIBUTE_MIN_FACE_SIZE_PX,
    FC_OUT_FEATURES,
    GENDER_LABELS,
    AttributeSampler,
    FairFaceBackend,
    GlobalPersonAttributeSmoother,
    YuNetFaceDetector,
    face_quality_ok,
    to_age_category,
)


class _FakeYuNetHandle:
    """Stands in for the cv2.FaceDetectorYN object cv2.FaceDetectorYN_create
    returns — lets YuNetFaceDetector's own bbox/score-column/filtering logic
    be tested without a real ONNX model file (none is available in this
    environment; see module docstring)."""

    def __init__(self, faces_by_size):
        self._faces_by_size = faces_by_size
        self.last_input_size = None

    def setInputSize(self, size):
        self.last_input_size = size

    def detect(self, image):
        return 1, self._faces_by_size


class _FakeFaceDetector:
    """Stands in for YuNetFaceDetector in tests — no real ONNX weights are
    available in this environment (see module docstring: this repo never
    bundles/auto-downloads third-party model files). Lets a face bbox
    (or "no face") be pinned deterministically per test."""

    def __init__(self, result=None):
        self.result = result  # None, or (bbox, confidence)

    def detect(self, person_crop_bgr):
        return self.result


# ------------------------------------------------------------- age category

def test_age_is_no_longer_categorised_child_detection_was_removed():
    for group in ["0-2", "3-9", "10-19", "20-29", "30-39", "70+", "unknown", "garbage"]:
        assert to_age_category(group) == "UNKNOWN"


# --------------------------------------------------------------- face crop

def _make_face_row(x, y, w, h, score):
    row = np.zeros(15)
    row[0], row[1], row[2], row[3], row[14] = x, y, w, h, score
    return row


def test_yunet_detector_no_faces_returns_none(monkeypatch):
    import cv2
    fake_handle = _FakeYuNetHandle(faces_by_size=None)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **kw: fake_handle)
    detector = YuNetFaceDetector(onnx_model_path="fake.onnx")
    assert detector.detect(np.zeros((200, 200, 3), dtype=np.uint8)) is None


def test_yunet_detector_empty_crop_returns_none(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **kw: _FakeYuNetHandle(None))
    detector = YuNetFaceDetector(onnx_model_path="fake.onnx")
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_yunet_detector_picks_highest_score_face(monkeypatch):
    import cv2
    # Both faces sized comfortably above ATTRIBUTE_MIN_FACE_SIZE_PX so this
    # stays a pure score-selection test, not a size-gate test (see
    # test_yunet_detector_rejects_faces_below_min_size /
    # test_yunet_detector_rejects_the_real_observed_undersized_face for that).
    faces = np.array([
        _make_face_row(10, 10, 80, 80, 0.7),
        _make_face_row(100, 100, 90, 90, 0.95),  # highest score — must win
    ])
    fake_handle = _FakeYuNetHandle(faces_by_size=faces)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **kw: fake_handle)
    detector = YuNetFaceDetector(onnx_model_path="fake.onnx")
    result = detector.detect(np.zeros((300, 300, 3), dtype=np.uint8))
    assert result is not None
    bbox, score = result
    assert bbox == [100.0, 100.0, 190.0, 190.0]
    assert score == pytest.approx(0.95)


def test_yunet_detector_rejects_faces_below_min_size(monkeypatch):
    import cv2
    tiny_face = np.array([_make_face_row(0, 0, ATTRIBUTE_MIN_FACE_SIZE_PX - 1,
                                          ATTRIBUTE_MIN_FACE_SIZE_PX - 1, 0.99)])
    fake_handle = _FakeYuNetHandle(faces_by_size=tiny_face)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **kw: fake_handle)
    detector = YuNetFaceDetector(onnx_model_path="fake.onnx", min_face_size_px=ATTRIBUTE_MIN_FACE_SIZE_PX)
    assert detector.detect(np.zeros((300, 300, 3), dtype=np.uint8)) is None


def test_yunet_detector_rejects_the_real_observed_undersized_face(monkeypatch):
    # Real case caught in a live two-camera test: a genuine 58x68px face crop (CAM-2, 320x240 feed, non-frontal
    # angle) made FairFace confidently (69%) predict the wrong gender. Such faces are no longer thrown away
    # outright (that left every far person UNKNOWN); instead a face below full-trust size needs >= 0.80
    # confidence and counts for less. This exact shape must still never label a person on its own.
    import cv2
    observed_face = np.array([_make_face_row(154, 165, 58, 69, 0.75)])
    fake_handle = _FakeYuNetHandle(faces_by_size=observed_face)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **kw: fake_handle)
    strict = YuNetFaceDetector(onnx_model_path="fake.onnx", min_face_size_px=ATTRIBUTE_MIN_FACE_SIZE_PX)
    assert strict.detect(np.zeros((240, 320, 3), dtype=np.uint8)) is None      # the legacy strict floor still works
    sm = GlobalPersonAttributeSmoother()
    for i in range(6):
        result = sm.add_sample("P1", "female", 0.69, now=float(i), face_px=58.0)
    assert result.gender == "UNKNOWN"


def test_face_quality_rejects_flat_dark_image():
    dark = np.full((100, 100, 3), 5, dtype=np.uint8)  # uniform (zero blur variance) and too dark
    assert face_quality_ok(dark) is False


def test_face_quality_rejects_blown_out_image():
    bright = np.full((100, 100, 3), 250, dtype=np.uint8)
    assert face_quality_ok(bright) is False


def test_face_quality_rejects_empty_crop():
    assert face_quality_ok(np.zeros((0, 0, 3), dtype=np.uint8)) is False


def test_face_quality_accepts_textured_midtone_image():
    rng = np.random.default_rng(1)
    textured = rng.integers(80, 180, (100, 100, 3), dtype=np.uint8)
    assert face_quality_ok(textured) is True


# ------------------------------------------------------------- FairFace CNN

def _fake_checkpoint(tmp_path):
    model = resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, FC_OUT_FEATURES)
    path = tmp_path / "fake_fairface.pt"
    torch.save(model.state_dict(), path)
    return path


def test_backend_loads_matching_checkpoint_shape(tmp_path):
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(), device="cpu")
    assert backend.device == torch.device("cpu")


def test_backend_rejects_mismatched_checkpoint_shape(tmp_path):
    model = resnet34()
    model.fc = torch.nn.Linear(model.fc.in_features, 5)  # wrong width — must fail loudly, not misalign labels
    path = tmp_path / "wrong_shape.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(RuntimeError):
        FairFaceBackend(checkpoint_path=path, face_detector=_FakeFaceDetector(), device="cpu")


def test_no_face_detected_returns_unknown(tmp_path):
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(result=None), device="cpu")
    blank = np.zeros((200, 200, 3), dtype=np.uint8)  # detector configured to find no face
    gender, gender_conf = backend.predict_gender(blank)
    age, age_conf = backend.predict_age_group(blank)
    assert (gender, gender_conf) == ("unknown", 0.0)
    assert (age, age_conf) == ("unknown", 0.0)


def test_empty_crop_returns_unknown_without_crashing(tmp_path):
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(), device="cpu")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert backend.predict_gender(empty) == ("unknown", 0.0)
    assert backend.predict_age_group(empty) == ("unknown", 0.0)


def test_valid_face_crop_produces_a_prediction(tmp_path):
    """"valid face crop" (spec section 19's required Attribute test list):
    once a face IS detected (here, a fixed bbox from the fake detector — no
    real ONNX weights available, see module docstring), a label must come
    back instead of "unknown", exercising the full predict_gender/
    predict_age_group -> _face_crop_or_none -> _predict_face path.

    person_crop must actually clear face_quality_ok's blur/brightness gate
    (see test_face_quality_accepts_textured_midtone_image's identical
    construction) now that _face_crop_or_none enforces it — a flat
    np.full(...) crop has zero blur variance and would be (correctly)
    rejected as low-quality before ever reaching the model."""
    fake_face_bbox = [20.0, 20.0, 80.0, 80.0]
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(result=(fake_face_bbox, 0.9)),
                               device="cpu")
    rng = np.random.default_rng(2)
    person_crop = rng.integers(80, 180, (150, 100, 3), dtype=np.uint8)
    gender, gender_conf = backend.predict_gender(person_crop)
    age, age_conf = backend.predict_age_group(person_crop)
    assert gender in GENDER_LABELS
    assert age in AGE_GROUP_LABELS
    assert 0.0 < gender_conf <= 1.0
    assert 0.0 < age_conf <= 1.0


def test_predict_face_parsing_slices_logits_correctly(tmp_path, monkeypatch):
    """Bypasses face detection (no real face image available in this
    environment) to directly verify the logit-slicing/label-mapping logic:
    feeding a known logit vector must produce the exact expected label."""
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(), device="cpu")

    fake_logits = np.full(FC_OUT_FEATURES, -10.0)
    fake_logits[7 + 1] = 10.0   # gender index 1 -> "female"
    fake_logits[7 + 2 + 3] = 10.0  # age index 3 -> "20-29"
    monkeypatch.setattr(backend, "_forward", lambda face: fake_logits)

    fake_face = np.full((64, 64, 3), 128, dtype=np.uint8)
    gender, gender_conf, age, age_conf = backend._predict_face(fake_face)
    assert gender == "female"
    assert age == "20-29"
    assert gender_conf > 0.99
    assert age_conf > 0.99


# --------------------------------------------------------- temporal smoothing

def _feed(smoother, gid, gender, conf, n, start=0.0):
    result = None
    for i in range(n):
        result = smoother.add_sample(gid, gender, conf, "unknown", 0.0, now=start + i)
    return result


def test_smoother_low_confidence_samples_never_label():
    smoother = GlobalPersonAttributeSmoother()
    result = _feed(smoother, "P1", "male", ATTRIBUTE_GENDER_MIN_CONFIDENCE - 0.01, 50)
    assert result.gender == "UNKNOWN"
    assert result.status == "unknown"


def test_smoother_never_labels_from_a_single_frame_however_confident():
    smoother = GlobalPersonAttributeSmoother()
    result = smoother.add_sample("P1", "male", 0.99, "20-29", 0.9, now=100.0)
    assert result.gender == "UNKNOWN"
    assert result.status == "unknown"


def test_smoother_labels_after_a_few_agreeing_confident_frames_and_never_reports_age():
    smoother = GlobalPersonAttributeSmoother()
    result = _feed(smoother, "P1", "male", 0.9, 3)
    assert result.gender == "MALE"
    assert result.status == "ok"
    assert result.age_group == "UNKNOWN" and result.age_category == "UNKNOWN"
    assert result.gender_confidence == pytest.approx(0.9)


def test_smoother_a_lone_wrong_frame_cannot_flip_an_established_label():
    smoother = GlobalPersonAttributeSmoother()
    _feed(smoother, "P1", "male", 0.9, 5)
    result = smoother.add_sample("P1", "female", 0.95, "unknown", 0.0, now=10.0)
    assert result.gender == "MALE"
    result = smoother.add_sample("P1", "male", 0.9, "unknown", 0.0, now=11.0)
    assert result.gender == "MALE"


def test_smoother_sustained_opposite_evidence_does_change_the_label_without_flicker():
    smoother = GlobalPersonAttributeSmoother()
    _feed(smoother, "P1", "male", 0.9, 5)
    labels = []
    for i in range(30):
        labels.append(smoother.add_sample("P1", "female", 0.9, "unknown", 0.0, now=10.0 + i).gender)
    assert labels[-1] == "FEMALE"
    # Once it has left MALE it never bounces back and forth.
    changes = sum(1 for x, y in zip(labels, labels[1:]) if x != y)
    assert changes <= 2      # MALE -> UNKNOWN -> FEMALE at most


def test_smoother_conflicting_frames_stay_unknown_instead_of_guessing():
    smoother = GlobalPersonAttributeSmoother()
    labels = []
    for i in range(20):
        g = "male" if i % 2 == 0 else "female"
        labels.append(smoother.add_sample("P1", g, 0.85, "unknown", 0.0, now=float(i)).gender)
    assert labels[-1] == "UNKNOWN"


def test_smoother_one_over_confident_frame_is_capped():
    # FairFace is over-confident on tiny faces: a single 99.9% frame must not outweigh
    # several honest ones from the other side.
    smoother = GlobalPersonAttributeSmoother()
    smoother.add_sample("P1", "male", 0.999, "unknown", 0.0, now=1.0)
    result = _feed(smoother, "P1", "female", 0.8, 4, start=2.0)
    assert result.gender == "FEMALE"


def test_smoother_stale_person_get_without_samples_returns_unknown():
    smoother = GlobalPersonAttributeSmoother()
    result = smoother.get("NEVER-SEEN")
    assert result.gender == "UNKNOWN"
    assert result.status == "unknown"
    assert result.sample_count == 0


def test_smoother_forget_clears_person_state():
    smoother = GlobalPersonAttributeSmoother()
    _feed(smoother, "P1", "male", 0.9, 3, start=100.0)
    smoother.forget("P1")
    assert smoother.get("P1").sample_count == 0


def test_smoother_expires_a_confident_result_after_the_grace_period():
    # Real bug caught live: a person who turned their back kept showing
    # their earlier confidently-observed gender indefinitely. A confident result must revert to
    # UNKNOWN once no real add_sample() call has landed within the grace period.
    smoother = GlobalPersonAttributeSmoother(stale_grace_sec=8.0)
    _feed(smoother, "P1", "female", 0.9, 3, start=100.0)      # last sample at t=102
    fresh = smoother.get("P1", now=105.0)
    assert fresh.gender == "FEMALE"
    assert fresh.status == "ok"

    stale = smoother.get("P1", now=112.0)                      # 10 s after the last sample
    assert stale.gender == "UNKNOWN"
    assert stale.status == "unknown"
    assert stale.sample_count == 0


def test_smoother_expiring_clears_old_votes_instead_of_letting_them_resurrect():
    smoother = GlobalPersonAttributeSmoother(stale_grace_sec=8.0)
    _feed(smoother, "P1", "female", 0.95, 5, start=100.0)      # last sample at t=104
    smoother.get("P1", now=115.0)                              # triggers expiry
    result = smoother.add_sample("P1", "male", 0.9, "unknown", 0.0, now=115.5)
    # Only one fresh sample exists: not enough to label, and no trace of the old "female" votes.
    assert result.gender == "UNKNOWN"
    assert result.sample_count == 1
    result = smoother.add_sample("P1", "male", 0.9, "unknown", 0.0, now=116.0)
    assert result.gender == "MALE"


def test_smoother_get_defaults_to_real_wall_clock_when_now_omitted():
    import time as time_module

    smoother = GlobalPersonAttributeSmoother(stale_grace_sec=8.0)
    for _ in range(3):
        smoother.add_sample("P1", "female", 0.9, "20-29", 0.85, now=time_module.time())
    result = smoother.get("P1")
    assert result.gender == "FEMALE"


def test_smoother_confidence_is_mean_of_agreeing_votes():
    smoother = GlobalPersonAttributeSmoother()
    smoother.add_sample("P1", "male", 0.9, "unknown", 0.0, now=1.0)
    result = smoother.add_sample("P1", "male", 0.8, "unknown", 0.0, now=2.0)
    assert result.gender == "MALE"
    assert result.gender_confidence == pytest.approx((0.9 + 0.8) / 2)


def test_forward_averages_the_image_and_its_mirror(tmp_path):
    backend = FairFaceBackend(checkpoint_path=_fake_checkpoint(tmp_path),
                               face_detector=_FakeFaceDetector(), device="cpu")
    seen = {}

    def fake_model(batch):
        seen["shape"] = tuple(batch.shape)
        seen["mirror"] = bool(torch.equal(batch[1], torch.flip(batch[0], dims=[2])))
        return torch.zeros(batch.shape[0], FC_OUT_FEATURES)

    backend._model = fake_model
    face = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    logits = backend._forward(face)
    assert seen["shape"] == (2, 3, 224, 224)
    assert seen["mirror"]
    assert logits.shape == (FC_OUT_FEATURES,)


# ------------------------------------------------------------- sampler gate

def test_sampler_gates_small_bbox():
    sampler = AttributeSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.3, interval_sec=1.0)
    small_track = {"bbox": [0, 0, 50, 50], "first_seen": 100.0}
    assert sampler.should_analyze("P1", small_track, frame_height=480, now=100.0) is False


def test_sampler_gates_young_track():
    sampler = AttributeSampler(min_track_age_sec=0.5, min_bbox_height_norm=0.0, interval_sec=1.0)
    track = {"bbox": [0, 0, 50, 300], "first_seen": 100.0}
    assert sampler.should_analyze("P1", track, frame_height=480, now=100.0) is False
    assert sampler.should_analyze("P1", track, frame_height=480, now=100.6) is True


def test_sampler_throttles_by_global_id_not_track():
    """Spec section 8: two different local tracks resolving to the same
    Global Person must share one throttle clock, not get independent
    budgets — that's the whole point of keying by global_id."""
    sampler = AttributeSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.0, interval_sec=1.0)
    track_a = {"bbox": [0, 0, 50, 300], "first_seen": 100.0}
    track_b = {"bbox": [0, 0, 60, 320], "first_seen": 100.0}
    assert sampler.should_analyze("P1", track_a, frame_height=480, now=100.0) is True
    assert sampler.should_analyze("P1", track_b, frame_height=480, now=100.2) is False  # same global_id, too soon


def test_sampler_forget_clears_throttle():
    sampler = AttributeSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.0, interval_sec=1.0)
    track = {"bbox": [0, 0, 50, 300], "first_seen": 100.0}
    sampler.should_analyze("P1", track, frame_height=480, now=100.0)
    sampler.forget("P1")
    assert sampler.should_analyze("P1", track, frame_height=480, now=100.1) is True


def test_labels_match_spec_exactly():
    assert AGE_GROUP_LABELS == ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    assert GENDER_LABELS == ["male", "female"]
    assert ATTRIBUTE_GENDER_MIN_CONFIDENCE > 0
    assert ATTRIBUTE_AGE_MIN_CONFIDENCE > 0


def test_crop_face_adds_the_fairface_training_margin_and_clips_to_the_image():
    from core.attributes import crop_face
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_face(img, [80, 80, 120, 120])                 # 40x40 face -> +25% each side = 60x60
    assert crop.shape[:2] == (60, 60)
    assert crop_face(img, [80, 80, 120, 120], margin=0.0).shape[:2] == (40, 40)
    corner = crop_face(img, [0, 0, 40, 40])                    # margin is clipped at the border
    assert corner.shape[:2] == (50, 50)
