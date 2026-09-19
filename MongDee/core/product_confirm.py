"""Temporal confirmation for embedding-matched products.

A single AI pass can match a product against background clutter (a shadow, a bag, a poster) even
though nothing is really there. Real products stay put across passes, so a recognition is only
reported once the SAME product key was matched at an overlapping place in an earlier pass within
`window_sec`. One-off matches are silently dropped (they never draw a box and never reach analytics).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

Box = Tuple[float, float, float, float]


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ProductConfirmer:
    def __init__(self, min_hits: int = 2, window_sec: float = 4.0, min_iou: float = 0.2):
        self.min_hits, self.window_sec, self.min_iou = max(1, int(min_hits)), float(window_sec), float(min_iou)
        self._recent: Dict[str, List[Tuple[float, Box]]] = {}

    def reset(self) -> None:
        self._recent.clear()

    def confirm(self, key: str, bbox, now: float) -> bool:
        """Record this match; True once it has been seen `min_hits` times (this one included)."""
        hits = [(t, b) for t, b in self._recent.get(key, []) if now - t <= self.window_sec]
        support = sum(1 for _, b in hits if _iou(b, bbox) >= self.min_iou)
        hits.append((now, tuple(float(v) for v in bbox)))
        self._recent[key] = hits[-12:]
        return support + 1 >= self.min_hits

    def forget_stale(self, now: float) -> None:
        for key in list(self._recent):
            kept = [(t, b) for t, b in self._recent[key] if now - t <= self.window_sec]
            if kept:
                self._recent[key] = kept
            else:
                del self._recent[key]
