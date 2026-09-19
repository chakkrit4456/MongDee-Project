"""Person <-> Product / Zone association (MongDee_Master_Prompt.md
section 52).

Given a person's world position and its recent movement, plus the booth's
products/zones, decide which product(s) the person is currently engaging
with — by proximity, whether they've stopped, whether they're approaching,
and (if a facing estimate is available) whether they're oriented toward it.

Explicitly NOT an ownership or purchase claim (section 52).
"""

from __future__ import annotations

import dataclasses

from vision.interest.config import InterestConfig
from vision.interest.gaze import body_orientation_from_motion, facing_score
from vision.spatial.booth import BoothLayout, BoothObject


@dataclasses.dataclass
class Association:
    global_id: str
    target_id: str          # zone id or product-object id
    target_type: str        # "product_zone" | "point_of_interest" | ...
    distance_m: float
    speed_mps: float
    approaching: bool
    facing_score: float | None
    confidence: float


class AssociationEngine:
    def __init__(self, layout: BoothLayout, config: InterestConfig | None = None):
        self._layout = layout
        self.config = config or InterestConfig()
        self._prev: dict[str, tuple[float, float, float]] = {}  # global_id -> (x, y, timestamp)

    def associate(
        self, global_id: str, world_x: float, world_y: float, timestamp: float
    ) -> list[Association]:
        cfg = self.config
        prev = self._prev.get(global_id)
        velocity = (0.0, 0.0)
        speed = 0.0
        if prev is not None:
            px, py, pts = prev
            dt = max(1e-3, timestamp - pts)
            velocity = ((world_x - px) / dt, (world_y - py) / dt)
            speed = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        self._prev[global_id] = (world_x, world_y, timestamp)

        heading = body_orientation_from_motion(velocity)
        candidates: list[Association] = []
        for obj in self._layout.objects:
            if not obj.active or obj.object_type not in ("product_zone", "point_of_interest", "shelf", "sales_point"):
                continue
            dist = obj.distance_to(world_x, world_y)
            if dist > cfg.near_distance_m * 2:
                continue

            approaching = False
            if prev is not None:
                prev_dist = obj.distance_to(prev[0], prev[1])
                approaching = dist < prev_dist - 0.05

            # only an actual candidate if they're near it, or clearly walking toward it
            if dist > cfg.near_distance_m and not (approaching and dist <= cfg.near_distance_m * 2):
                continue

            face = facing_score((world_x, world_y), heading, (obj.x, obj.y))
            confidence = _confidence(dist, cfg.near_distance_m, speed, cfg.stop_speed_mps, approaching, face)
            candidates.append(
                Association(
                    global_id=global_id, target_id=obj.id, target_type=obj.object_type,
                    distance_m=round(dist, 3), speed_mps=round(speed, 3),
                    approaching=approaching, facing_score=round(face, 3) if face is not None else None,
                    confidence=round(confidence, 3),
                )
            )
        candidates.sort(key=lambda a: a.confidence, reverse=True)
        return candidates

    def forget(self, global_id: str) -> None:
        self._prev.pop(global_id, None)


def _confidence(dist, near, speed, stop_speed, approaching, face) -> float:
    is_near = dist <= near
    score = 0.0
    score += 0.4 * max(0.0, 1.0 - dist / max(near, 1e-6)) if is_near else 0.0
    score += 0.2 if (is_near and speed <= stop_speed) else 0.0
    score += 0.2 if approaching else 0.0
    if face is not None:
        score += 0.2 * face
    return min(1.0, score)
