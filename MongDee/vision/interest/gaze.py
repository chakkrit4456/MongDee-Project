"""Head pose & gaze (MongDee_Master_Prompt.md section 53).

A real implementation needs a face-landmark or head-pose model. There
isn't one wired in, so:

  - yaw / pitch / roll and true gaze return UNKNOWN unless a HeadPoseBackend
    is provided;
  - as the spec allows ("ให้ใช้ body/head orientation และ spatial behavior
    เป็นข้อมูลเสริม"), body_orientation_from_motion() gives a weak facing
    estimate from the track's own movement direction.

Never fabricate a gaze angle from nothing.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class HeadPose:
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    face_visible: bool = False
    confidence: float = 0.0

    @property
    def is_known(self) -> bool:
        return self.yaw is not None and self.confidence > 0.0


class HeadPoseBackend:
    """Interface for a real head-pose model."""

    def estimate(self, person_crop_bgr: np.ndarray) -> HeadPose:
        raise NotImplementedError


class HeadPoseEstimator:
    def __init__(self, backend: HeadPoseBackend | None = None):
        self._backend = backend

    def estimate(self, person_crop_bgr: np.ndarray) -> HeadPose:
        if self._backend is None:
            return HeadPose()  # all UNKNOWN — honest
        return self._backend.estimate(person_crop_bgr)


def body_orientation_from_motion(velocity: tuple[float, float]) -> float | None:
    """Heading in radians (world frame) if the person is actually moving,
    else None. atan2(vy, vx); assumes people generally face where they walk."""
    vx, vy = velocity
    if (vx * vx + vy * vy) ** 0.5 < 1e-3:
        return None
    return math.atan2(vy, vx)


def facing_score(
    person_xy: tuple[float, float],
    heading_rad: float | None,
    target_xy: tuple[float, float],
) -> float | None:
    """0..1 — how well the person's heading points at the target. None if
    heading is unknown. cos of the angle between heading and the
    person->target vector, clamped to [0, 1]."""
    if heading_rad is None:
        return None
    tx = target_xy[0] - person_xy[0]
    ty = target_xy[1] - person_xy[1]
    norm = (tx * tx + ty * ty) ** 0.5
    if norm < 1e-6:
        return 1.0
    to_target = math.atan2(ty, tx)
    diff = math.atan2(math.sin(to_target - heading_rad), math.cos(to_target - heading_rad))
    return max(0.0, math.cos(diff))
