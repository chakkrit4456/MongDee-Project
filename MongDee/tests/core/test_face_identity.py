from __future__ import annotations

import numpy as np

from core.face.detector import FaceBox
from core.face_identity import FACE_DIFFERENT, FACE_SAME, FaceIdentity, cosine, frontalness


class _Detector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return self.faces


class _Recognizer:
    def __init__(self):
        self.rows = []

    def alignCrop(self, img, row):
        self.rows.append(row)
        return np.zeros((112, 112, 3), np.uint8)

    def feature(self, aligned):
        return np.arange(1, 129, dtype=np.float32).reshape(1, -1)


def _lm(nose_x=50.0):
    return np.array([[35, 40], [65, 40], [nose_x, 55], [40, 70], [60, 70]], np.float32)


def _face(size=40.0, score=0.9, nose_x=50.0):
    return FaceBox(30, 20, 30 + size, 20 + size, score, _lm(nose_x))


CROP = np.zeros((200, 100, 3), np.uint8)


def test_embedding_is_unit_length_and_uses_the_landmarks():
    rec = _Recognizer()
    emb = FaceIdentity(_Detector([_face()]), rec).embed(CROP)
    assert emb is not None and emb.shape == (128,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5
    assert len(rec.rows) == 1 and rec.rows[0].shape == (15,)          # x,y,w,h + 5 landmarks + score


def test_no_face_or_unusable_face_gives_none():
    assert FaceIdentity(_Detector([]), _Recognizer()).embed(CROP) is None
    assert FaceIdentity(_Detector([_face(size=12)]), _Recognizer()).embed(CROP) is None           # too small
    assert FaceIdentity(_Detector([_face(score=0.4)]), _Recognizer()).embed(CROP) is None         # unsure detection
    assert FaceIdentity(_Detector([_face(nose_x=66)]), _Recognizer()).embed(CROP) is None         # profile view
    assert FaceIdentity(_Detector([_face()]), _Recognizer()).embed(np.zeros((0, 0, 3), np.uint8)) is None


def test_the_largest_usable_face_wins():
    rec = _Recognizer()
    FaceIdentity(_Detector([_face(size=30), _face(size=60)]), rec).embed(CROP)
    assert abs(float(rec.rows[0][2]) - 60.0) < 1e-3


def test_a_broken_recognizer_never_raises():
    class Boom(_Recognizer):
        def feature(self, aligned):
            raise RuntimeError("x")
    assert FaceIdentity(_Detector([_face()]), Boom()).embed(CROP) is None


def test_create_returns_none_when_the_model_files_are_missing(tmp_path):
    assert FaceIdentity.create(tmp_path / "yunet.onnx", tmp_path / "sface.onnx") is None
    assert FaceIdentity.create(None) is None


def test_thresholds_and_helpers():
    assert FACE_DIFFERENT < FACE_SAME
    assert frontalness(_lm()) > 0.95 and frontalness(_lm(nose_x=68)) < 0.2
    a = np.ones(4)
    assert abs(cosine(a, a) - 1.0) < 1e-6 and cosine(a, np.zeros(4)) == 0.0
