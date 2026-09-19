"""Generic, class-agnostic "something is being shown to the camera" region proposer.

The built-in YOLO11 model only knows COCO's 80 classes, so it's useless for
locating a real exhibitor's product (a specific snack bag, a specific
gadget) that was never in COCO. This module finds the region a product
occupies in the frame *without knowing what it is* — via background
subtraction — so it works for literally any physical object. The crop it
returns then gets identified by core/recognizer.py's embedding matcher.

One ForegroundProposer instance is stateful per camera (the background
model adapts over time), so each CameraWorker owns its own.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_AREA_FRAC = 0.03   # ignore blobs smaller than 3% of the frame (noise)
MAX_AREA_FRAC = 0.75   # ignore blobs bigger than 75% of the frame (lighting shifts, whole-bg change)


class ForegroundProposer:
    def __init__(self):
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=True
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, frame_bgr) -> None:
        """Feed a frame into the rolling background model without asking for a
        proposal yet — call this every frame so the model adapts smoothly even
        on frames where a proposal isn't requested."""
        self._bg_subtractor.apply(frame_bgr, learningRate=-1)

    def propose(self, frame_bgr, exclude_boxes: list[list[float]] | None = None,
                max_regions: int = 2) -> list[list[float]]:
        """Return up to `max_regions` [x1,y1,x2,y2] boxes for foreground blobs
        that don't already overlap `exclude_boxes` (e.g. YOLO boxes already
        claimed by a person or a known COCO-class product)."""
        h, w = frame_bgr.shape[:2]
        frame_area = h * w

        mask = self._bg_subtractor.apply(frame_bgr, learningRate=-1)
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadow pixels (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            frac = area / frame_area
            if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            box = [float(x), float(y), float(x + bw), float(y + bh)]
            if exclude_boxes and any(_iou(box, other) > 0.3 for other in exclude_boxes):
                continue
            candidates.append((area, box))

        candidates.sort(key=lambda c: c[0], reverse=True)
        return [box for _, box in candidates[:max_regions]]


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_box(frame_bgr, box, padding_frac: float = 0.08) -> np.ndarray:
    """Crop `box` out of `frame_bgr` with a little padding so the object isn't
    cut off tight at the edges (helps both training and matching)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_frac
    pad_y = (y2 - y1) * padding_frac
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    return frame_bgr[y1:y2, x1:x2]
