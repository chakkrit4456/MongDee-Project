"""Turn noisy per-frame gender/age predictions into a stable, calibrated per-track label.

Fixes "man labelled as woman": the old flow committed to argmax of ONE full-body-crop
prediction from a 320x240 stream (tiny faces, no quality gate). Here:
  * only frames whose FACE quality >= min_quality contribute;
  * each vote is weighted by quality * classifier confidence and stored as log-odds;
  * a label is emitted only when >= min_votes votes AND |evidence| >= margin;
  * once emitted it changes only if opposite evidence exceeds `flip_margin` (hysteresis);
  * otherwise the answer is "unknown" - never a forced guess.
Age / child detection was removed: only male / female are reported.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrackAttributes:
    gender: str = "unknown"          # male | female | unknown
    gender_evidence: float = 0.0     # signed log-odds, + = male
    n_gender_votes: int = 0
    best_quality: float = 0.0


def _logit(p: float, eps: float = 1e-3) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class AttributeAggregator:
    def __init__(self, min_quality: float = 0.25, min_votes: int = 3, margin: float = 1.5,
                 flip_margin: float = 3.0, decay: float = 0.98,
                 max_tracks: int = 256):
        self.min_quality, self.min_votes = min_quality, min_votes
        self.margin, self.flip_margin = margin, flip_margin
        self.decay = decay
        self.max_tracks = max_tracks
        self._t: Dict[int, TrackAttributes] = {}

    def get(self, tid: int) -> TrackAttributes:
        return self._t.setdefault(tid, TrackAttributes())

    def forget(self, tid: int):
        self._t.pop(tid, None)

    def add(self, tid: int, quality: float, p_male: Optional[float] = None) -> TrackAttributes:
        a = self.get(tid)
        if len(self._t) > self.max_tracks:
            self._t.pop(next(iter(self._t)))
        if quality < self.min_quality:
            return a
        a.best_quality = max(a.best_quality, quality)
        w = quality
        if p_male is not None:
            a.gender_evidence = a.gender_evidence * self.decay + w * _logit(p_male)
            a.n_gender_votes += 1
            self._decide_gender(a)
        return a

    def _decide_gender(self, a: TrackAttributes):
        if a.n_gender_votes < self.min_votes:
            return
        e = a.gender_evidence
        cur = a.gender
        if cur == "unknown":
            if e >= self.margin: a.gender = "male"
            elif e <= -self.margin: a.gender = "female"
        elif cur == "male" and e <= -self.flip_margin:
            a.gender = "female"
        elif cur == "female" and e >= self.flip_margin:
            a.gender = "male"
