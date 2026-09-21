"""Temporal confirmation for embedding-matched products, plus distance-aware confidence scaling
shared by both product paths in core/vision.py (COCO-class YOLO detections and custom-trained
embedding matches).

A single AI pass can match a product against background clutter (a shadow, a bag, a poster) even
though nothing is really there. Real products stay put across passes, so a recognition is only
reported once the SAME product key was matched at an overlapping place in an earlier pass within
`window_sec`. One-off matches are silently dropped (they never draw a box and never reach analytics).

A flat confidence floor is not enough on its own: a small, low-detail box far from the camera can
clear the same bar as a close-up, sharp one and get reported with just as much certainty -- a
"random guess" at range. core/attributes.py already solved exactly this problem for faces
(face_min_confidence/attenuate_confidence, scaled by face_px) — product_min_confidence and
product_confirm_hits_for below apply the same idea here: a small box needs a higher score to be
believed at all, and needs to be seen agreeing across more frames before it is reported.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]

# A box at least this tall/wide (in source pixels) gets full trust: no confidence/hit-count
# penalty. Smaller than PRODUCT_FAR_MIN_PX is not scaled further -- a box that small is already
# governed by MIN_CROP_SIDE_PX (core/vision.py) or YOLO's own minimum object size, so there is
# nothing meaningful left to scale between there and zero.
PRODUCT_FULL_TRUST_PX = 80.0
PRODUCT_FAR_MIN_PX = 24.0
# Absolute confidence floor at the smallest attempted size -- set well above a typical base floor
# (YOLO's conf_threshold, or core.recognizer's MATCH_FLOOR/SINGLE_PRODUCT_MATCH_FLOOR) so a small,
# blurry box needs real, unambiguous evidence, not just "technically over the normal bar".
PRODUCT_FAR_MIN_CONFIDENCE = 0.75
PRODUCT_FAR_EXTRA_HITS = 2   # extra confirmation hits required at the smallest attempted size


def _far_fraction(box_px: "float | None") -> float:
    """0.0 at full-trust size or larger, 1.0 at the smallest attempted size, linear between."""
    if box_px is None or box_px >= PRODUCT_FULL_TRUST_PX:
        return 0.0
    px = max(float(box_px), PRODUCT_FAR_MIN_PX)
    return (PRODUCT_FULL_TRUST_PX - px) / (PRODUCT_FULL_TRUST_PX - PRODUCT_FAR_MIN_PX)


def product_min_confidence(box_px: "float | None", base_threshold: float) -> float:
    """Confidence a product detection/match needs to count at all, given its box size in source
    pixels. Equal to `base_threshold` at full-trust size; rises (linearly in box size) to
    PRODUCT_FAR_MIN_CONFIDENCE at the smallest attempted size -- never lower than base_threshold
    even if that is already stricter."""
    frac = _far_fraction(box_px)
    if frac <= 0.0:
        return base_threshold
    return max(base_threshold, base_threshold + frac * (PRODUCT_FAR_MIN_CONFIDENCE - base_threshold))


def product_confirm_hits_for(box_px: "float | None", base_hits: int) -> int:
    """How many agreeing frames a match at this box size needs before ProductConfirmer.confirm()
    reports it — base_hits at full-trust size, up to base_hits + PRODUCT_FAR_EXTRA_HITS at the
    smallest attempted size."""
    frac = _far_fraction(box_px)
    return int(base_hits + round(frac * PRODUCT_FAR_EXTRA_HITS))


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


PRODUCT_COAST_SEC = 0.6   # matches core.vision.PERSON_COAST_SEC -- see coasting()'s own docstring


class ProductConfirmer:
    def __init__(self, min_hits: int = 2, window_sec: float = 4.0, min_iou: float = 0.2,
                 coast_sec: float = PRODUCT_COAST_SEC):
        self.min_hits, self.window_sec, self.min_iou = max(1, int(min_hits)), float(window_sec), float(min_iou)
        self.coast_sec = float(coast_sec)
        self._recent: Dict[str, List[Tuple[float, Box]]] = {}
        # key -> (ts, bbox, score) of the most recent pass this key was actually CONFIRMED (not
        # just matched) on -- see coasting(). The score is carried through coasting too (rather
        # than some placeholder) so a momentary miss doesn't corrupt anything downstream that
        # tracks per-product confidence over time (core.aggregator.ProductAggregator keeps the
        # latest confidence per camera and would otherwise see a real match's score suddenly drop
        # to 0 for one pass, exactly the kind of instability this whole mechanism exists to avoid).
        self._last_confirmed: Dict[str, Tuple[float, Box, float]] = {}

    def reset(self) -> None:
        self._recent.clear()
        self._last_confirmed.clear()

    def confirm(self, key: str, bbox, now: float, min_hits: "int | None" = None,
                score: float = 0.0) -> bool:
        """Record this match; True once it has been seen `min_hits` times (this one included).
        `min_hits` overrides self.min_hits for this call only -- see product_confirm_hits_for:
        a small/far box should need more agreeing frames than a close, unambiguous one, without
        changing every OTHER product's requirement. `score`: this match's confidence, remembered
        for coasting() (see that method and _last_confirmed's own docstring)."""
        hits = [(t, b) for t, b in self._recent.get(key, []) if now - t <= self.window_sec]
        support = sum(1 for _, b in hits if _iou(b, bbox) >= self.min_iou)
        hits.append((now, tuple(float(v) for v in bbox)))
        self._recent[key] = hits[-12:]
        required = self.min_hits if min_hits is None else max(1, int(min_hits))
        confirmed = support + 1 >= required
        if confirmed:
            self._last_confirmed[key] = (now, tuple(float(v) for v in bbox), float(score))
        return confirmed

    def coasting(self, now: float, exclude: "set[str] | None" = None) -> List[Tuple[str, Box, float]]:
        """Already-confirmed products NOT re-matched on THIS pass (see `exclude`, the keys this
        pass's own confirm() calls already reported) but seen within coast_sec: their last known
        (box, score), exactly the same idea as core.tracker.PersonTracker.coasting_tracks -- one
        AI pass where an embedding score happens to dip a hair below the floor (motion blur, a
        hand briefly crossing it, a lighting flicker) keeps showing the box instead of blinking it
        off and back on a moment later. Static (no motion prediction, unlike person coasting): a
        product that is genuinely picked up and moved during the coast window will lag briefly
        before the next real match corrects it, which is preferable to disappearing outright for
        a window this short."""
        exclude = exclude or ()
        return [(k, b, s) for k, (t, b, s) in self._last_confirmed.items()
                if k not in exclude and now - t <= self.coast_sec]

    def forget_stale(self, now: float) -> None:
        for key in list(self._recent):
            kept = [(t, b) for t, b in self._recent[key] if now - t <= self.window_sec]
            if kept:
                self._recent[key] = kept
            else:
                del self._recent[key]
        stale_grace = max(self.window_sec, self.coast_sec)
        for key in list(self._last_confirmed):
            if now - self._last_confirmed[key][0] > stale_grace:
                del self._last_confirmed[key]
