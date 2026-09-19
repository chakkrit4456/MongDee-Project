"""Track-level attribute aggregation.

A single frame's attribute read is noisy (lighting, motion blur, partial
occlusion). The master prompt wants per-*person* attributes, so aggregate
across a track's frames: for each categorical attribute take the
confidence-weighted majority vote; for the numeric appearance features
take the running mean. This also implements the "keyframe / track-level
aggregation" idea from section 23.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from vision.attributes.extractor import AttributeValue, PersonAttributes

_CATEGORICAL = ("shirt_color", "pants_color", "build", "gender", "age_group", "hair", "bag", "glasses", "mask")


class TrackAttributeAggregator:
    def __init__(self, max_samples: int = 30):
        self._votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._shirt_rgb: list[tuple[int, int, int]] = []
        self._pants_rgb: list[tuple[int, int, int]] = []
        self._aspect: list[float] = []
        self._n = 0
        self._max = max_samples

    def add(self, attrs: PersonAttributes) -> None:
        if self._n >= self._max:
            return
        self._n += 1
        for name in _CATEGORICAL:
            v: AttributeValue = getattr(attrs, name)
            if v.is_known:
                self._votes[name][v.value] += v.confidence
        if attrs.shirt_rgb != (0, 0, 0):
            self._shirt_rgb.append(attrs.shirt_rgb)
        if attrs.pants_rgb != (0, 0, 0):
            self._pants_rgb.append(attrs.pants_rgb)
        if attrs.aspect_ratio > 0:
            self._aspect.append(attrs.aspect_ratio)

    @property
    def sample_count(self) -> int:
        return self._n

    def result(self) -> PersonAttributes:
        out = PersonAttributes()
        for name, tally in self._votes.items():
            if not tally:
                continue
            best_value = max(tally, key=tally.get)
            total = sum(tally.values())
            confidence = tally[best_value] / total if total else 0.0
            setattr(out, name, AttributeValue(best_value, round(float(confidence), 3)))
        if self._shirt_rgb:
            out.shirt_rgb = tuple(int(v) for v in np.mean(self._shirt_rgb, axis=0))
        if self._pants_rgb:
            out.pants_rgb = tuple(int(v) for v in np.mean(self._pants_rgb, axis=0))
        if self._aspect:
            out.aspect_ratio = round(float(np.mean(self._aspect)), 3)
        return out
