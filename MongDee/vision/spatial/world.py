"""Multi-camera spatial fusion (MongDee_Master_Prompt.md sections 58, 71,
72).

Turns a person detection/track in one camera into a world position on the
2D booth map, using that camera's calibration. The foot point (bottom-
centre of the bbox) is the thing that touches the floor plane the
homography was calibrated on.

Consistency checks (section 71): reject an impossible position jump (a
person can't move 8 m in 0.2 s), flag a position outside the booth,
lower confidence when the source calibration is unreliable.
"""

from __future__ import annotations

import dataclasses
import time

from vision.spatial.booth import BoothLayout
from vision.spatial.calibration import CalibrationStore


@dataclasses.dataclass
class WorldPosition:
    x: float
    y: float
    camera_id: str
    timestamp: float
    confidence: float
    zone_id: str | None = None
    in_bounds: bool = True


def foot_point(bbox: list[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


class WorldMapper:
    def __init__(
        self,
        calibrations: CalibrationStore,
        layout: BoothLayout | None = None,
        max_speed_mps: float = 3.0,  # a person walking fast; jumps beyond this are rejected
    ):
        self._calib = calibrations
        self._layout = layout
        self._max_speed = max_speed_mps
        self._last: dict[str, WorldPosition] = {}  # global_id -> last accepted position

    def locate(
        self, camera_id: str, bbox: list[float], global_id: str | None = None, timestamp: float | None = None
    ) -> WorldPosition | None:
        calib = self._calib.get(camera_id)
        if calib is None:
            return None
        if timestamp is None:
            timestamp = time.time()

        fx, fy = foot_point(bbox)
        wx, wy = calib.pixel_to_world(fx, fy)

        confidence = 0.9 if calib.is_reliable else 0.4
        in_bounds = self._layout.in_bounds(wx, wy) if self._layout else True
        if not in_bounds:
            confidence *= 0.5

        zone = self._layout.zone_at(wx, wy) if self._layout else None
        pos = WorldPosition(
            x=round(wx, 3), y=round(wy, 3), camera_id=camera_id, timestamp=timestamp,
            confidence=round(confidence, 3), zone_id=zone.id if zone else None, in_bounds=in_bounds,
        )

        if global_id is not None:
            prev = self._last.get(global_id)
            if prev is not None:
                dt = max(1e-3, timestamp - prev.timestamp)
                dist = ((wx - prev.x) ** 2 + (wy - prev.y) ** 2) ** 0.5
                if dist / dt > self._max_speed:
                    # impossible jump — return it but flag it very low confidence,
                    # and DON'T update `_last` so the next reading is compared to a sane anchor
                    pos.confidence = round(pos.confidence * 0.2, 3)
                    return pos
            self._last[global_id] = pos
        return pos

    def forget(self, global_id: str) -> None:
        self._last.pop(global_id, None)
