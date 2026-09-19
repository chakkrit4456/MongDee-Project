"""Dominant-colour estimation for a person crop region.

Deliberately simple and honest: quantise the region's pixels into a small
set of named colours (the ones a human would name a shirt/trousers) and
report the most common one plus how dominant it was. No learned model —
so this is a real, checkable measurement, not a claimed attribute the
system can't actually produce (MongDee_Master_Prompt.md section 7:
"ห้ามอ้าง attribute ที่โมเดลไม่สามารถตรวจได้อย่างสมเหตุสมผล").
"""

from __future__ import annotations

import numpy as np

# name -> representative RGB. Kept small on purpose; ambiguous mid-tones
# resolve to the nearest of these.
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "red": (200, 30, 30),
    "orange": (230, 140, 30),
    "yellow": (235, 215, 50),
    "green": (40, 160, 60),
    "blue": (40, 70, 190),
    "navy": (25, 30, 90),
    "purple": (120, 50, 160),
    "pink": (235, 150, 190),
    "brown": (110, 70, 40),
    "beige": (210, 190, 150),
}

_NAMES = list(NAMED_COLORS)
_PALETTE = np.array([NAMED_COLORS[n] for n in _NAMES], dtype=np.float32)  # (K, 3) RGB


def _nearest_named(pixels_rgb: np.ndarray) -> np.ndarray:
    """pixels_rgb: (N, 3) float -> (N,) index into _NAMES of nearest palette colour."""
    d = np.linalg.norm(pixels_rgb[:, None, :] - _PALETTE[None, :, :], axis=2)  # (N, K)
    return np.argmin(d, axis=1)


def dominant_color(region_bgr: np.ndarray, sample_cap: int = 4000) -> tuple[str, tuple[int, int, int], float]:
    """Returns (color_name, mean_rgb, confidence). confidence is the
    fraction of sampled pixels that fell in the winning colour bucket, so
    a very mixed region reports low confidence."""
    if region_bgr.size == 0:
        return "unknown", (0, 0, 0), 0.0

    flat = region_bgr.reshape(-1, 3)
    if len(flat) > sample_cap:
        idx = np.random.default_rng(0).choice(len(flat), sample_cap, replace=False)
        flat = flat[idx]
    rgb = flat[:, ::-1].astype(np.float32)  # BGR -> RGB

    buckets = _nearest_named(rgb)
    counts = np.bincount(buckets, minlength=len(_NAMES))
    winner = int(np.argmax(counts))
    confidence = float(counts[winner] / len(rgb))
    mean_rgb = tuple(int(v) for v in rgb[buckets == winner].mean(axis=0))
    return _NAMES[winner], mean_rgb, confidence
