"""Concrete, OpenCV-DNN-backed implementation of GenderAgeBackend.

No gender/age model ships with this repo. vision/attributes/extractor.py's
module docstring explains why: this project deliberately never fakes a
prediction it can't actually produce — gender/age stay "unknown" unless a
real classifier is plugged in. This class is that plug-in point: it loads
whatever Caffe (.prototxt + .caffemodel) or ONNX two/multi-class model an
operator supplies and runs it via cv2.dnn.

The model file itself is intentionally NOT downloaded automatically or
hardcoded to a URL here — same "bring your own weights" pattern already
used for ReID (see reid.osnet_weights / scripts/download_models.py's
--osnet-url), because third-party model hosting isn't something this repo
can vouch for as stable or trustworthy. A commonly used, freely available
pair for gender is the Levi & Hassner (2015) "Age and Gender Classification
using Convolutional Neural Networks" Caffe model (gender_net.caffemodel +
deploy_gender.prototxt) — its [female, male] softmax output order and
227x227 BGR-mean-subtracted preprocessing are what the defaults below
match — but any equivalent model works as long as its output order lines
up with GENDER_LABELS (or you pass your own label list).
"""

from __future__ import annotations

import numpy as np

from vision.attributes.extractor import GenderAgeBackend

GENDER_LABELS = ["female", "male"]
AGE_GROUP_LABELS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]

# Standard preprocessing for the Levi & Hassner Caffe gender/age nets.
# If you're using a different model, pass its own input_size/mean instead.
_INPUT_SIZE = (227, 227)
_MEAN = (78.4263377603, 87.7689143744, 114.895847746)


class OpenCVDnnGenderAgeBackend(GenderAgeBackend):
    """Loads a gender classifier (and optionally an age classifier) via
    cv2.dnn and exposes them through the GenderAgeBackend interface that
    vision/attributes/extractor.py's AttributeExtractor expects.

    gender_model is required; gender_prototxt is required for a Caffe
    model and must be omitted for ONNX (cv2.dnn.readNet auto-detects the
    format from the file). age_model/age_prototxt are optional — leave
    them unset to keep age_group reported as unknown while still getting
    real gender predictions.
    """

    def __init__(self, gender_model: str, gender_prototxt: str | None = None,
                 age_model: str | None = None, age_prototxt: str | None = None,
                 gender_labels: list[str] | None = None, age_labels: list[str] | None = None,
                 input_size: tuple[int, int] = _INPUT_SIZE,
                 mean: tuple[float, float, float] = _MEAN):
        self._input_size = input_size
        self._mean = mean
        self._gender_labels = gender_labels or GENDER_LABELS
        self._age_labels = age_labels or AGE_GROUP_LABELS
        self._gender_net = self._load_net(gender_model, gender_prototxt)
        self._age_net = self._load_net(age_model, age_prototxt) if age_model else None

    @staticmethod
    def _load_net(model: str, prototxt: str | None):
        import cv2

        if prototxt:
            return cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        return cv2.dnn.readNet(str(model))  # ONNX (or anything else cv2.dnn can sniff)

    def _predict(self, net, crop_bgr: np.ndarray, labels: list[str]) -> tuple[str, float]:
        import cv2

        blob = cv2.dnn.blobFromImage(crop_bgr, 1.0, self._input_size, self._mean, swapRB=False, crop=False)
        net.setInput(blob)
        scores = net.forward().flatten()
        if scores.size == 0:
            return "unknown", 0.0
        idx = int(np.argmax(scores))
        if idx >= len(labels):
            return "unknown", 0.0
        return labels[idx], float(scores[idx])

    def predict_gender(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return "unknown", 0.0
        return self._predict(self._gender_net, person_crop_bgr, self._gender_labels)

    def predict_age_group(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        if self._age_net is None or person_crop_bgr is None or person_crop_bgr.size == 0:
            return "unknown", 0.0
        return self._predict(self._age_net, person_crop_bgr, self._age_labels)
