"""Gender for people who are far from the camera: small faces are analysed as weaker evidence instead of being
thrown away, and a person who is still UNKNOWN is sampled faster."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core import attributes as A
from core.attributes import (
    ATTRIBUTE_MIN_FACE_SIZE_PX, ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE, ATTRIBUTE_SMALL_FACE_MIN_PX, AttributeSampler,
    FairFaceBackend, GlobalPersonAttributeSmoother, YuNetFaceDetector, blur_floor_for, face_min_confidence,
    face_quality_ok, face_temperature)


# ------------------------------------------------------------------------------------ size policy
def test_temperature_is_one_for_big_faces_and_grows_as_the_face_shrinks():
    assert face_temperature(None) == 1.0 and face_temperature(ATTRIBUTE_MIN_FACE_SIZE_PX) == 1.0
    assert face_temperature(200) == 1.0
    ts = [face_temperature(px) for px in (60, 48, 36, 24, ATTRIBUTE_SMALL_FACE_MIN_PX)]
    assert ts == sorted(ts) and ts[0] > 1.0
    assert ts[-1] == pytest.approx(ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE)
    assert face_temperature(5) == pytest.approx(ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE)      # never below the floor


def test_a_small_face_must_be_more_confident_and_may_be_softer():
    assert face_min_confidence(None) == A.ATTRIBUTE_GENDER_MIN_CONFIDENCE == face_min_confidence(80)
    assert face_min_confidence(30) == A.ATTRIBUTE_SMALL_FACE_MIN_CONFIDENCE > A.ATTRIBUTE_GENDER_MIN_CONFIDENCE
    assert blur_floor_for(None) == blur_floor_for(80) == A.ATTRIBUTE_MIN_BLUR_VARIANCE
    assert blur_floor_for(14) == A.ATTRIBUTE_FAR_MIN_BLUR_VARIANCE < blur_floor_for(40) < A.ATTRIBUTE_MIN_BLUR_VARIANCE


def _frames_until_label(px, conf, gender="male"):
    sm = GlobalPersonAttributeSmoother()
    for i in range(1, 30):
        r = sm.add_sample("P", gender, conf, now=float(i), face_px=px)
        if r.gender != "UNKNOWN":
            return i
    return None


def test_a_far_person_needs_more_agreeing_frames_than_a_near_one():
    near, mid, far = (_frames_until_label(px, 0.9) for px in (90, 40, 18))
    assert near == 2 and near <= mid < far and far is not None and far <= 8      # a far face needs 3+ agreeing frames


def test_the_observed_low_confidence_small_face_does_not_count():
    assert _frames_until_label(58, 0.69, "female") is None


def test_the_legacy_call_without_a_face_size_is_unchanged():
    sm = GlobalPersonAttributeSmoother()
    sm.add_sample("P", "male", 0.9, now=1.0)
    assert sm.add_sample("P", "male", 0.9, now=2.0).gender == "MALE"


def test_person_level_error_stays_small_even_with_a_noisy_far_face_model():
    """Monte-Carlo under an ASSUMED model of a bad far-away face reader (78% right per frame, confidence only
    weakly related to being right). The measured accuracy on real footage will differ - run
    tools/eval_fairface_gender.py - but the evidence gate must keep the error of labelled people low."""
    rng = np.random.default_rng(7)
    labelled = wrong = total = 0
    for person in range(600):
        truth = "male" if person % 2 == 0 else "female"
        px = float(rng.integers(16, 40))
        sm = GlobalPersonAttributeSmoother()
        result = None
        for i in range(10):
            right = rng.random() < 0.78
            g = truth if right else ("female" if truth == "male" else "male")
            conf = float(np.clip(rng.normal(0.86 if right else 0.80, 0.07), 0.55, 0.99))
            result = sm.add_sample("p", g, conf, now=float(i), face_px=px)
        total += 1
        if result.gender != "UNKNOWN":
            labelled += 1
            wrong += result.gender.lower() != truth
    assert labelled / total > 0.55, labelled / total          # most far people do get a label ...
    assert wrong / labelled < 0.10, wrong / labelled          # ... and it is rarely wrong


# ------------------------------------------------------------------------------ head-region detector
class _FakeYuNet:
    def __init__(self, faces):
        self.faces, self.input_size, self.seen = faces, None, None

    def setInputSize(self, size):
        self.input_size = size

    def detect(self, img):
        self.seen = img.shape[:2]
        return 1, (None if self.faces is None else np.array(self.faces, dtype=np.float32))


def _row(x, y, w, h, score):
    r = np.zeros(15, np.float32)
    r[:4], r[14] = (x, y, w, h), score
    return r


def _detector(monkeypatch, faces, min_px=ATTRIBUTE_SMALL_FACE_MIN_PX):
    fake = _FakeYuNet(faces)
    monkeypatch.setattr(cv2, "FaceDetectorYN_create", lambda *a, **k: fake)
    return YuNetFaceDetector("fake.onnx", min_face_size_px=min_px), fake


def test_only_the_head_region_is_searched_and_it_is_upscaled(monkeypatch):
    det, fake = _detector(monkeypatch, [_row(100, 60, 80, 80, 0.9)])
    crop = np.zeros((120, 50, 3), np.uint8)                         # a far person: 120 px tall
    det.detect_in_person(crop)
    ih, iw = fake.seen
    assert ih >= 200 and ih / 60 == pytest.approx(iw / 50, rel=0.05)   # top half (60 rows) scaled up ~4x
    assert fake.input_size == (iw, ih)


def test_face_coordinates_come_back_in_original_crop_pixels(monkeypatch):
    det, fake = _detector(monkeypatch, [_row(80, 40, 80, 80, 0.9)])
    crop = np.zeros((120, 50, 3), np.uint8)
    bbox, score = det.detect_in_person(crop)
    scale = fake.seen[0] / 60
    assert bbox == pytest.approx([80 / scale, 40 / scale, 160 / scale, 120 / scale]) and score == pytest.approx(0.9)
    assert bbox[2] - bbox[0] == pytest.approx(80 / scale)          # a ~20 px face in the original image


def test_faces_smaller_than_the_minimum_are_ignored(monkeypatch):
    det, fake = _detector(monkeypatch, [_row(10, 10, 20, 20, 0.95)])
    assert det.detect_in_person(np.zeros((120, 50, 3), np.uint8)) is None      # 20 px up-scaled ~4x -> ~5 px source


def test_the_most_central_confident_face_wins(monkeypatch):
    det, fake = _detector(monkeypatch, [_row(0, 40, 60, 60, 0.92), _row(90, 40, 60, 60, 0.85)])
    crop = np.zeros((120, 60, 3), np.uint8)
    bbox, score = det.detect_in_person(crop)
    assert score == pytest.approx(0.85)                             # the edge face lost to the centred one


def test_no_face_no_crash(monkeypatch):
    det, _ = _detector(monkeypatch, None)
    assert det.detect_in_person(np.zeros((120, 50, 3), np.uint8)) is None
    assert det.detect_in_person(np.zeros((0, 0, 3), np.uint8)) is None


# ----------------------------------------------------------------------- backend locate / quality
def _backend_with(detector):
    b = object.__new__(FairFaceBackend)
    b._face_detector = detector
    return b


class _D:
    def __init__(self, bbox):
        self.bbox = bbox

    def detect(self, crop):
        return (self.bbox, 0.9)


def _soft_crop(size=120):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(60, 190, (size, size, 3), dtype=np.uint8), (0, 0), 1.2)
    return img


def test_a_soft_small_face_passes_the_relaxed_quality_floor_but_a_soft_big_one_does_not():
    crop = _soft_crop(40)
    assert cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < A.ATTRIBUTE_MIN_BLUR_VARIANCE
    assert face_quality_ok(crop, 16) is True
    assert face_quality_ok(crop, 90) is False
    assert face_quality_ok(crop) is False                           # legacy call = full floor


def test_locate_face_reports_the_face_size(monkeypatch):
    person = _soft_crop(200)
    located = _backend_with(_D([60, 20, 80, 42]))._locate_face(person)
    assert located is not None and located[1] == pytest.approx(20.0)
    assert located[0].shape[0] > 22                                  # padded by the 25 % margin


def test_locate_face_returns_none_without_a_face():
    class _None:
        def detect(self, crop):
            return None

    assert _backend_with(_None())._locate_face(_soft_crop(100)) is None


# --------------------------------------------------------------------------------- sampler cadence
def test_an_unknown_person_is_sampled_faster_than_a_labelled_one():
    s = AttributeSampler(interval_sec=0.75, min_track_age_sec=0.0, min_bbox_height_norm=0.0)
    track = {"bbox": [0, 0, 50, 200], "first_seen": 0.0}
    assert s.should_analyze("P", track, 480, 10.0) is True
    assert s.should_analyze("P", track, 480, 10.4) is False
    assert s.should_analyze("P", track, 480, 10.4, urgent=True) is True          # 0.4 s >= the urgent interval
    assert s.should_analyze("P", track, 480, 10.6, urgent=True) is False


def test_a_small_person_is_no_longer_skipped_outright():
    s = AttributeSampler(interval_sec=0.0, min_track_age_sec=0.0)
    far = {"bbox": [0, 0, 30, 60], "first_seen": 0.0}                            # 12.5 % of a 480 px frame
    assert s.should_analyze("P", far, 480, 5.0) is True


# ---------------------------------------------------------------- per-frame person classification
class _DetailBackend:
    def __init__(self, gender, confidence, face_px):
        self._d = {"gender": gender, "confidence": confidence, "face_px": face_px}

    def predict_gender(self, crop):  # pragma: no cover - must not be used when detail is available
        raise AssertionError("legacy path used")

    def predict_gender_detail(self, crop):
        return dict(self._d)


def _classify(backend):
    import numpy as np
    from core.vision import CameraWorker

    class _M:
        names = {}

    worker = CameraWorker(camera_id="T", device=0, model=_M(), allowed_classes=[], gender_age_backend=backend)
    return worker._classify_person(np.zeros((100, 100, 3), dtype=np.uint8), [10, 10, 50, 90])


def test_classify_person_full_size_face_keeps_confidence():
    gender, conf = _classify(_DetailBackend("male", 0.9, 120))
    assert gender == "male" and abs(conf - 0.9) < 1e-6


def test_classify_person_tiny_face_below_floor_is_unknown():
    assert _classify(_DetailBackend("male", 0.75, 16)) == ("unknown", 0.0)


def test_classify_person_small_confident_face_is_attenuated_but_still_counts():
    from core.attributes import attenuate_confidence
    gender, conf = _classify(_DetailBackend("female", 0.99, 20))
    assert gender == "unknown" or (gender == "female" and conf < 0.99)
    assert attenuate_confidence(0.99, 20) < 0.99
    assert attenuate_confidence(0.9, 120) == 0.9


# ---------------------------------------------------------------- calibration reaches the live prediction
def test_backend_applies_the_gender_bias_for_the_face_size():
    import numpy as np
    from core.attributes import FairFaceBackend, NUM_RACE_CLASSES
    from core.gender_calibration import GenderCalibration

    class _B(FairFaceBackend):
        def __init__(self):                                   # no torch model needed: only _forward is used
            self.calibration = GenderCalibration(checkpoint="m.pt", bias=0.0, bucket_bias=[1.5, None, None, None, None])

        def _forward(self, face_bgr):
            logits = np.zeros(18)
            logits[NUM_RACE_CLASSES:NUM_RACE_CLASSES + 2] = [0.4, 0.0]     # raw model: male by 0.4
            return logits

    b = _B()
    face = np.zeros((40, 40, 3), np.uint8)
    assert b._predict_face(face, face_px=100)[0] == "male"      # large face: no bias for that bucket
    assert b._predict_face(face, face_px=18)[0] == "female"     # small face: bias 1.5 tips it
    b.calibration = None
    assert b._predict_face(face, face_px=18)[0] == "male"


def test_resolve_prefers_the_webcam_checkpoint_unless_forced_back(tmp_path, monkeypatch):
    from core.attributes import WEBCAM_CHECKPOINT_NAME, resolve_fairface_checkpoint
    original = tmp_path / "best_model_state_dict.pt"
    original.write_bytes(b"x")
    assert resolve_fairface_checkpoint(original) == original
    (tmp_path / WEBCAM_CHECKPOINT_NAME).write_bytes(b"y")
    monkeypatch.delenv("MONGDEE_FAIRFACE_ORIGINAL", raising=False)
    assert resolve_fairface_checkpoint(original).name == WEBCAM_CHECKPOINT_NAME
    monkeypatch.setenv("MONGDEE_FAIRFACE_ORIGINAL", "1")
    assert resolve_fairface_checkpoint(original) == original
