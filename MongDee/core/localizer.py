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

MIN_AREA_FRAC = 0.015  # ignore blobs smaller than 1.5% of the frame (noise) -- was 0.03; a product
                        # held at arm's length or seen by a wide-angle camera can easily be smaller
                        # than 3% of the frame, and identify()'s own distance-scaled confidence/hit
                        # requirements (core/product_confirm.py) already raise the bar further for a
                        # small box, so this floor only needs to keep out single-pixel noise, not do
                        # the real far-object filtering itself.
MAX_AREA_FRAC = 0.90   # ignore blobs bigger than 90% of the frame (lighting shifts, whole-bg change)
                        # -- was 0.75, which rejected a product held up close enough to fill most of
                        # the frame (a completely normal "show it to the camera" pose). A genuinely
                        # unrelated whole-frame change (a light flicking on, someone walking behind
                        # the camera) still won't match any trained embedding, so identify()'s own
                        # floor is the real guard here, not this area cap.


class ForegroundProposer:
    def __init__(self):
        # history=700 (was 400): with learningRate=-1 the background model absorbs roughly 1/history
        # of each frame's content every pass, so a product held still in front of the camera for
        # ~13s (400 frames at 30fps) used to melt entirely into "background" and stop producing any
        # foreground blob at all -- i.e. the box would vanish even though the product never moved.
        # 700 roughly doubles that grace period without meaningfully slowing how fast the model
        # relearns a genuinely rearranged background between booth visitors.
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=700, varThreshold=40, detectShadows=True
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
        claimed by a person or a known COCO-class product).

        Reads the background model with learningRate=0 (no further
        adaptation here) -- the model is kept warm by a separate, much more
        frequent stream of update() calls (see CameraWorker.run()'s capture
        loop). Mixing a training update into a read call here as well would
        double-count whichever frame happens to be both captured and
        AI-passed, and -- far more importantly -- previously meant the
        background model only ever adapted on the sparse subset of frames
        that reached a custom-recognition AI pass, so it lagged reality by
        seconds. A stale model measures "foreground" against a background
        photo of a moment long past, so ordinary scene change (a person
        walking through, a lighting shift, camera shake) shows up as a
        large, confident, completely spurious blob -- the leading suspected
        cause of "ghost" product detections with no product in frame."""
        h, w = frame_bgr.shape[:2]
        frame_area = h * w

        mask = self._bg_subtractor.apply(frame_bgr, learningRate=0)
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


HUMAN_OVERLAP_REJECT_FRAC = 0.80  # candidate is mostly a hand/arm, not a product -> reject outright
                                   # -- was 0.60, which rejected the single most common presentation
                                   # pose (holding the product up toward the camera in-hand, so the
                                   # hand/wrist/forearm can easily cover half or more of the raw
                                   # foreground blob). The surviving non-human remainder still has to
                                   # clear MIN_CROP_SIDE_PX and identify()'s own confidence floor, so
                                   # this only needs to reject candidates that are ALMOST ENTIRELY
                                   # hand (a wave with nothing held), not merely hand-dominant.
HUMAN_OVERLAP_TRIM_FRAC = 0.02    # below this, the box is basically clean already -- skip retightening


def exclude_human_region(box, human_mask: np.ndarray | None) -> list[float] | None:
    """Tighten `box` to exclude pixels `human_mask` marks as a detected
    person (see core.person_segmenter.PersonSegmenter -- one instance mask
    per person, so it covers hand/arm/sleeve/torso alike, not just the
    whole-body bounding box). Returns None when the candidate is mostly
    human (a hand or arm waved through frame with no product at all) and
    should not become a product candidate.

    Deliberately geometry-based (bounding rect of the surviving non-human
    pixels within `box`) rather than a pixel-accurate silhouette: the
    downstream consumer is recognizer.py's rectangular-crop embedding
    either way, so a tighter human-free box is what actually improves the
    match -- a pixel mask would only matter for a renderer drawing the
    mask itself, which this project doesn't do."""
    if human_mask is None:
        return [float(v) for v in box]
    h, w = human_mask.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region = human_mask[y1:y2, x1:x2]
    if region.size == 0:
        return None
    human_frac = float(region.sum()) / region.size
    if human_frac >= HUMAN_OVERLAP_REJECT_FRAC:
        return None
    if human_frac < HUMAN_OVERLAP_TRIM_FRAC:
        return [float(x1), float(y1), float(x2), float(y2)]
    ys, xs = np.nonzero(~region)
    if ys.size == 0:
        return None
    ny1, ny2 = int(ys.min()), int(ys.max()) + 1
    nx1, nx2 = int(xs.min()), int(xs.max()) + 1
    return [float(x1 + nx1), float(y1 + ny1), float(x1 + nx2), float(y1 + ny2)]


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
