"""Face IDENTITY (who is this face?) as opposed to face DETECTION / gender.

Body appearance (clothes, shape) is a weak cue across two different cameras, so on its own it can glue two
different people together or split one person in two. A face embedding is the one cue that stays reliable
across cameras. This module wraps OpenCV's SFace recogniser (Apache-2.0, ~37 MB ONNX, CPU-friendly):

    person crop --YuNet--> 5 landmarks --alignCrop--> 112x112 --SFace--> 128-d unit vector

Two embeddings are compared with the cosine similarity; the SFace authors give 0.363 as the "same person"
threshold. `FACE_SAME` / `FACE_DIFFERENT` below are the (deliberately conservative) values the registry uses.

The ONNX file is NOT bundled: run `python tools/get_face_models.py` once (needs internet). Without it
everything degrades gracefully - `FaceIdentity.create()` returns None and Re-ID falls back to body cues,
with the stricter cross-camera rules in core.reid.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SFACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_recognizer" / "face_recognition_sface_2021dec.onnx"

FACE_SAME = 0.40        # cosine >= this: same person (SFace's own threshold is 0.363; a little margin)
FACE_DIFFERENT = 0.22   # cosine <  this: different people (different faces score around 0.0 - 0.15)
MIN_FACE_PX = 26        # smaller faces give unreliable embeddings
MIN_DET_SCORE = 0.70
MIN_FRONTAL = 0.45      # 0..1, from nose position between the eyes (profile faces embed badly)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def frontalness(landmarks: np.ndarray) -> float:
    """1.0 = nose centred between the eyes (frontal), -> 0 as the head turns."""
    re, le, nose = landmarks[0], landmarks[1], landmarks[2]
    dl, dr = abs(float(nose[0] - le[0])), abs(float(re[0] - nose[0]))
    return float(np.clip(1.0 - abs(dl - dr) / max(dl + dr, 1e-6) * 1.5, 0.0, 1.0))


class FaceIdentity:
    """Turns a person crop into a face embedding (or None when no usable face is visible)."""

    def __init__(self, detector, recognizer):
        self._detector = detector      # core.face.YuNetFaceDetector (needs landmarks)
        self._recognizer = recognizer  # cv2.FaceRecognizerSF

    @classmethod
    def create(cls, yunet_model, sface_model=None) -> "FaceIdentity | None":
        """None (never an exception) when OpenCV lacks FaceRecognizerSF or a model file is missing."""
        sface = Path(sface_model) if sface_model else DEFAULT_SFACE_MODEL
        if not sface.exists() or not yunet_model or not Path(yunet_model).exists():
            return None
        try:
            from core.face.detector import YuNetFaceDetector
            create = getattr(cv2, "FaceRecognizerSF_create", None) or cv2.FaceRecognizerSF.create
            recognizer = create(str(sface), "")
            detector = YuNetFaceDetector(str(yunet_model), score_thr=0.6, upscale=2.0)
            return cls(detector, recognizer)
        except Exception as exc:  # old OpenCV, corrupt file...
            logger.warning("face identity (SFace) unavailable: %s", exc)
            return None

    def embed(self, person_crop_bgr: np.ndarray) -> "np.ndarray | None":
        """Unit-length 128-d embedding of the best face in the crop, or None."""
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        try:
            faces = self._detector.detect(person_crop_bgr)
            best, best_size = None, 0.0
            for f in faces:
                size = float(min(f.w, f.h))
                if (f.landmarks is None or size < MIN_FACE_PX or f.score < MIN_DET_SCORE
                        or frontalness(f.landmarks) < MIN_FRONTAL):
                    continue
                if size > best_size:
                    best, best_size = f, size
            if best is None:
                return None
            row = np.concatenate([[best.x1, best.y1, best.w, best.h],
                                  np.asarray(best.landmarks, dtype=np.float32).reshape(-1),
                                  [best.score]]).astype(np.float32)
            aligned = self._recognizer.alignCrop(person_crop_bgr, row)
            feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feature))
            return feature / norm if norm > 0 else None
        except Exception as exc:
            logger.debug("face embedding failed: %s", exc)
            return None
