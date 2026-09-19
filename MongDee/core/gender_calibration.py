"""Decision-bias calibration for the FairFace gender head, fitted on webcam-like (degraded) faces.

A softmax over two logits decides male/female at exactly 0.5. On sharp FairFace photos that is the right point. On
small, blurry, dim faces a network is often *biased*: it collapses towards one class (typically "male") so one gender
is recognised much less often than the other. The cure that costs nothing at run time is to move the decision point:
add a bias `b` to the female-minus-male logit difference and pick `b` so that female recall and male recall are equal
(maximum balanced accuracy) on data that looks like the camera. The bias is fitted per face-size bucket because the
skew grows as the face shrinks. Pure numpy, no torch.

The fitted file is `gender_calibration.json` next to the checkpoint. It records which checkpoint it was fitted for and
is ignored for any other one, so swapping models can never apply a stale bias.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

CALIBRATION_FILENAME = "gender_calibration.json"
SIZE_EDGES = (24, 40, 64, 112)          # face pixels: buckets <24, 24-40, 40-64, 64-112, >=112
MAX_ABS_BIAS = 2.0                      # never move the decision point further than this (logit units)
MIN_PER_CLASS_PER_BUCKET = 60


def balanced_accuracy(diff: np.ndarray, is_female: np.ndarray, bias: float) -> float:
    pred_f = (diff + bias) > 0
    tpr_f = float(pred_f[is_female].mean()) if is_female.any() else 0.0
    tpr_m = float((~pred_f[~is_female]).mean()) if (~is_female).any() else 0.0
    return 0.5 * (tpr_f + tpr_m)


def fit_bias(diff: np.ndarray, is_female: np.ndarray) -> float:
    """The bias `b` that makes `diff + b > 0` call both genders equally well.

    Fitted as a class-balanced 1-D logistic regression P(female | diff) = sigmoid(a * diff + c); the decision point
    is diff = -c/a, i.e. b = c/a. That is a smooth, low-variance estimate of the point where female and male
    recall cross - unlike arg-maxing balanced accuracy directly, whose staircase is dominated by noise. The result
    is clamped to +-MAX_ABS_BIAS and discarded (0.0) if it does not beat no bias on the very data it was fitted on
    or if the scores carry no signal (a <= 0). 0.0 when a class is missing."""
    diff = np.asarray(diff, dtype=np.float64)
    is_female = np.asarray(is_female, dtype=bool)
    if diff.size == 0 or not is_female.any() or is_female.all():
        return 0.0
    y = is_female.astype(np.float64)
    w = np.where(is_female, 0.5 / is_female.sum(), 0.5 / (~is_female).sum())        # both classes weigh the same
    a, c = 1.0, 0.0
    for _ in range(50):                                                              # Newton-Raphson, tiny ridge on a
        z = np.clip(a * diff + c, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        r = w * p * (1 - p) + 1e-12
        g = np.array([np.sum(w * (p - y) * diff) + 1e-3 * (a - 1.0), np.sum(w * (p - y))])
        h = np.array([[np.sum(r * diff * diff) + 1e-3, np.sum(r * diff)], [np.sum(r * diff), np.sum(r)]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            return 0.0
        a, c = a - step[0], c - step[1]
        if np.abs(step).max() < 1e-8:
            break
    if not np.isfinite(a) or not np.isfinite(c) or a <= 1e-6:
        return 0.0
    b = float(np.clip(c / a, -MAX_ABS_BIAS, MAX_ABS_BIAS))
    return b if balanced_accuracy(diff, is_female, b) >= balanced_accuracy(diff, is_female, 0.0) else 0.0


def bucket_index(face_px: "float | None") -> int:
    if face_px is None:
        return len(SIZE_EDGES)                      # unknown size: treat as a large face
    for i, edge in enumerate(SIZE_EDGES):
        if face_px < edge:
            return i
    return len(SIZE_EDGES)


@dataclasses.dataclass
class GenderCalibration:
    checkpoint: str = ""
    bias: float = 0.0                                                   # used when no bucket applies
    bucket_bias: list = dataclasses.field(default_factory=lambda: [None] * (len(SIZE_EDGES) + 1))
    info: dict = dataclasses.field(default_factory=dict)

    def bias_for(self, face_px: "float | None") -> float:
        b = self.bucket_bias[bucket_index(face_px)] if len(self.bucket_bias) == len(SIZE_EDGES) + 1 else None
        return float(self.bias if b is None else b)

    def apply(self, gender_logits: np.ndarray, face_px: "float | None") -> np.ndarray:
        """gender_logits = [male, female]; returns a copy with the bias added to the female logit."""
        out = np.array(gender_logits, dtype=np.float64, copy=True)
        out[1] += self.bias_for(face_px)
        return out

    def to_dict(self) -> dict:
        return {"checkpoint": self.checkpoint, "bias": self.bias, "bucket_bias": self.bucket_bias,
                "size_edges": list(SIZE_EDGES), "info": self.info}

    @classmethod
    def from_dict(cls, data: dict) -> "GenderCalibration":
        edges = data.get("size_edges", list(SIZE_EDGES))
        buckets = data.get("bucket_bias") or [None] * (len(SIZE_EDGES) + 1)
        if list(edges) != list(SIZE_EDGES) or len(buckets) != len(SIZE_EDGES) + 1:
            buckets = [None] * (len(SIZE_EDGES) + 1)                     # incompatible bucket layout: global bias only
        clean = []
        for b in buckets:
            ok = isinstance(b, (int, float)) and math.isfinite(b)
            clean.append(float(np.clip(b, -MAX_ABS_BIAS, MAX_ABS_BIAS)) if ok else None)
        bias = data.get("bias", 0.0)
        bias = float(np.clip(bias, -MAX_ABS_BIAS, MAX_ABS_BIAS)) if isinstance(bias, (int, float)) and math.isfinite(bias) else 0.0
        return cls(checkpoint=str(data.get("checkpoint", "")), bias=bias, bucket_bias=clean, info=dict(data.get("info", {})))

    def save(self, path: "str | Path") -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_for(cls, checkpoint_path: "str | Path") -> "GenderCalibration | None":
        """The calibration stored beside `checkpoint_path` IF it was fitted for exactly that checkpoint, else None."""
        ckpt = Path(checkpoint_path)
        path = ckpt.parent / CALIBRATION_FILENAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("checkpoint") != ckpt.name:
            return None
        return cls.from_dict(data)


def fit_calibration(diff: np.ndarray, is_female: np.ndarray, face_px: np.ndarray, checkpoint: str,
                    info: "dict | None" = None) -> GenderCalibration:
    """Fit the global bias and one bias per face-size bucket that has enough samples of both genders."""
    diff = np.asarray(diff, dtype=np.float64)
    is_female = np.asarray(is_female, dtype=bool)
    face_px = np.asarray(face_px, dtype=np.float64)
    cal = GenderCalibration(checkpoint=checkpoint, bias=fit_bias(diff, is_female), info=dict(info or {}))
    buckets = np.array([bucket_index(v) for v in face_px])
    for i in range(len(SIZE_EDGES) + 1):
        m = buckets == i
        if (m & is_female).sum() >= MIN_PER_CLASS_PER_BUCKET and (m & ~is_female).sum() >= MIN_PER_CLASS_PER_BUCKET:
            cal.bucket_bias[i] = fit_bias(diff[m], is_female[m])
    return cal
