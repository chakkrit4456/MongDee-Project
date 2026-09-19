"""Frame Processing / Normalization — Phase 2, the layer between the Camera
Gateway and the detector (MongDee_Master_Prompt.md section 2 diagram:
"Frame Normalization", and section 4 on decoupling input vs. processing
resolution).

Its one job: take a raw camera frame at whatever resolution the camera
delivers (could be 4K) and produce a smaller frame to run AI on (e.g.
960x540), while remembering the exact transform so a detection found in
the small frame can be mapped back to full-resolution coordinates for
storage and display.

Aspect-ratio-preserving resize only, no padding — the detector (ultralytics
YOLO) does its own letterboxing to a square internally, so padding here
would just be wasted pixels. That keeps the inverse transform a single
uniform scale factor.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass
class ProcessedFrame:
    image: np.ndarray  # BGR, resized for inference
    scale: float  # processed_pixel = original_pixel * scale  (uniform, aspect preserved)
    original_size: tuple[int, int]  # (width, height) of the frame before processing

    def to_original(self, bbox: tuple[float, float, float, float]) -> list[float]:
        """Map an (x1, y1, x2, y2) box from processed-frame coords back to
        original-frame coords, clamped to the original frame bounds."""
        inv = 1.0 / self.scale if self.scale else 1.0
        w, h = self.original_size
        x1, y1, x2, y2 = bbox
        return [
            max(0.0, min(x1 * inv, w)),
            max(0.0, min(y1 * inv, h)),
            max(0.0, min(x2 * inv, w)),
            max(0.0, min(y2 * inv, h)),
        ]


class FrameProcessor:
    """Resizes frames so the longer side fits within (max_width, max_height)
    while preserving aspect ratio. A frame already small enough is passed
    through untouched (scale = 1.0) rather than upscaled."""

    def __init__(self, max_width: int = 960, max_height: int = 540):
        if max_width <= 0 or max_height <= 0:
            raise ValueError(f"FrameProcessor: max_width/max_height must be positive, got {max_width}x{max_height}")
        self.max_width = max_width
        self.max_height = max_height

    def process(self, image: np.ndarray) -> ProcessedFrame:
        h, w = image.shape[:2]
        scale = min(self.max_width / w, self.max_height / h, 1.0)
        if scale == 1.0:
            return ProcessedFrame(image=image, scale=1.0, original_size=(w, h))
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # store the *actual* achieved scale (rounding new_w/new_h can nudge it)
        actual_scale = ((new_w / w) + (new_h / h)) / 2.0
        return ProcessedFrame(image=resized, scale=actual_scale, original_size=(w, h))
