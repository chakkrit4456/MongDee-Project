"""Product interaction detection (MongDee_Master_Prompt.md section 57).

Full reach/pick/hold gestures need a pose model. Without one, this uses
two observable cues:

  1. the person's bounding box overlaps the product's box (they are close
     enough to touch it), and
  2. the product's box has moved from where it was resting (something
     picked it up / put it back).

Result is always POSSIBLE_INTERACTION / CONFIRMED_INTERACTION / UNKNOWN —
and never turned into "purchased" (section 57, 88, 92 rule 7).
"""

from __future__ import annotations

import dataclasses

POSSIBLE_INTERACTION = "POSSIBLE_INTERACTION"
CONFIRMED_INTERACTION = "CONFIRMED_INTERACTION"
UNKNOWN = "UNKNOWN"


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if (area_a + area_b - inter) > 0 else 0.0


def _centroid(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


@dataclasses.dataclass
class InteractionState:
    status: str = UNKNOWN
    contact_frames: int = 0
    product_moved_px: float = 0.0
    resting_centroid: tuple[float, float] | None = None


class InteractionDetector:
    def __init__(self, contact_iou: float = 0.02, move_threshold_px: float = 15.0, confirm_after_frames: int = 3):
        self._contact_iou = contact_iou
        self._move_threshold = move_threshold_px
        self._confirm_after = confirm_after_frames
        self._states: dict[tuple[int, int], InteractionState] = {}  # (person_track, product_track) -> state

    def observe(
        self, person_track_id: int, product_track_id: int, person_bbox: list[float], product_bbox: list[float]
    ) -> InteractionState:
        key = (person_track_id, product_track_id)
        state = self._states.setdefault(key, InteractionState())
        centroid = _centroid(product_bbox)
        if state.resting_centroid is None:
            state.resting_centroid = centroid

        contact = _iou(person_bbox, product_bbox) >= self._contact_iou
        moved = ((centroid[0] - state.resting_centroid[0]) ** 2 + (centroid[1] - state.resting_centroid[1]) ** 2) ** 0.5
        state.product_moved_px = max(state.product_moved_px, moved)

        if contact:
            state.contact_frames += 1
        else:
            state.contact_frames = max(0, state.contact_frames - 1)
            if state.contact_frames == 0:
                # settled again — reset the resting position
                state.resting_centroid = centroid
                state.product_moved_px = 0.0

        if state.contact_frames >= self._confirm_after and state.product_moved_px >= self._move_threshold:
            state.status = CONFIRMED_INTERACTION
        elif contact:
            state.status = POSSIBLE_INTERACTION
        else:
            state.status = UNKNOWN
        return state

    def state(self, person_track_id: int, product_track_id: int) -> InteractionState | None:
        return self._states.get((person_track_id, product_track_id))

    def forget_person(self, person_track_id: int) -> None:
        self._states = {k: v for k, v in self._states.items() if k[0] != person_track_id}
