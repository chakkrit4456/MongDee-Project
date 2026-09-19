"""Crop quality gate for Re-ID (MongDee_Master_Prompt.md section 24).

A low-quality crop (tiny, blurry, wrong shape → probably occluded or a
detection error) must not be allowed to drive an identity decision. This
returns a 0..1 score and a `usable` flag; the extractor skips embedding
when not usable, and Phase 6 down-weights low-score evidence.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass
class CropQuality:
    score: float  # 0..1, higher is better
    usable: bool
    height_px: int
    blur_var: float
    aspect: float
    reason: str = ""


def assess_crop(
    image: np.ndarray,
    bbox: list[float],
    *,
    min_height_px: int,
    min_blur_var: float,
    min_aspect: float,
    max_aspect: float,
) -> CropQuality:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]

    ch, cw = crop.shape[:2] if crop.size else (0, 0)
    aspect = (cw / ch) if ch else 0.0

    if crop.size == 0 or ch < min_height_px:
        return CropQuality(0.0, False, ch, 0.0, aspect, "too small")
    if not (min_aspect <= aspect <= max_aspect):
        return CropQuality(0.2, False, ch, 0.0, aspect, f"aspect {aspect:.2f} out of range")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < min_blur_var:
        return CropQuality(0.3, False, ch, blur_var, aspect, f"too blurry ({blur_var:.1f})")

    # combine into a soft score
    size_term = min(1.0, ch / (min_height_px * 3))
    blur_term = min(1.0, blur_var / (min_blur_var * 6))
    score = round(0.5 * size_term + 0.5 * blur_term, 3)
    return CropQuality(score, True, ch, blur_var, aspect, "ok")
