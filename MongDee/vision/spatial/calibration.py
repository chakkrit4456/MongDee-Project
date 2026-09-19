"""Camera calibration via homography (MongDee_Master_Prompt.md sections
59, 60, 71).

A booth camera looks at a roughly planar floor. A homography (3x3) maps
image pixels on that floor plane to world coordinates (metres, origin at
one corner of the booth). Calibrate by clicking >= 4 points in the camera
image whose real-world floor position you know (booth corners, tape
marks), then every person's foot point can be placed on the 2D booth map.

Stores a version + reference points + the measured reprojection error so a
bad calibration is visible and an old one is never silently reused after a
layout change (section 73).
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np


@dataclasses.dataclass
class CameraCalibration:
    camera_id: str
    homography: np.ndarray  # 3x3, image -> world
    image_points: list[tuple[float, float]]
    world_points: list[tuple[float, float]]
    reprojection_error: float  # mean pixel error of the reference points
    version: int = 1
    created_at: float = dataclasses.field(default_factory=time.time)
    unit: str = "m"

    @staticmethod
    def from_points(
        camera_id: str,
        image_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
        version: int = 1,
        unit: str = "m",
    ) -> "CameraCalibration":
        if len(image_points) < 4 or len(world_points) != len(image_points):
            raise ValueError("need >= 4 matching image/world point pairs")
        img = np.asarray(image_points, dtype=np.float64)
        wld = np.asarray(world_points, dtype=np.float64)
        H, mask = cv2.findHomography(img, wld, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            raise ValueError(f"calibration for {camera_id}: homography could not be estimated (degenerate points?)")

        # reprojection error: world -> image via H^-1, compare to given image points
        Hinv = np.linalg.inv(H)
        reproj = cv2.perspectiveTransform(wld.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(reproj - img, axis=1)))
        return CameraCalibration(
            camera_id=camera_id, homography=H,
            image_points=[tuple(p) for p in image_points],
            world_points=[tuple(p) for p in world_points],
            reprojection_error=round(error, 3), version=version, unit=unit,
        )

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.homography)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def world_to_pixel(self, wx: float, wy: float) -> tuple[float, float]:
        pt = np.array([[[float(wx), float(wy)]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, np.linalg.inv(self.homography))
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    @property
    def is_reliable(self) -> bool:
        return self.reprojection_error <= 15.0  # pixels; tune per install

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "homography": self.homography.tolist(),
            "image_points": [list(p) for p in self.image_points],
            "world_points": [list(p) for p in self.world_points],
            "reprojection_error": self.reprojection_error,
            "version": self.version,
            "created_at": self.created_at,
            "unit": self.unit,
        }

    @staticmethod
    def from_dict(d: dict) -> "CameraCalibration":
        return CameraCalibration(
            camera_id=d["camera_id"],
            homography=np.asarray(d["homography"], dtype=np.float64),
            image_points=[tuple(p) for p in d["image_points"]],
            world_points=[tuple(p) for p in d["world_points"]],
            reprojection_error=d.get("reprojection_error", 0.0),
            version=d.get("version", 1),
            created_at=d.get("created_at", time.time()),
            unit=d.get("unit", "m"),
        )


class CalibrationStore:
    """All cameras' calibrations, loadable from one JSON file."""

    def __init__(self, calibrations: dict[str, CameraCalibration] | None = None):
        self._by_camera = calibrations or {}

    def get(self, camera_id: str) -> CameraCalibration | None:
        return self._by_camera.get(camera_id)

    def set(self, calibration: CameraCalibration) -> None:
        self._by_camera[calibration.camera_id] = calibration

    def camera_ids(self) -> list[str]:
        return list(self._by_camera)

    def unreliable(self) -> list[str]:
        return [cid for cid, c in self._by_camera.items() if not c.is_reliable]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({cid: c.to_dict() for cid, c in self._by_camera.items()}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str | Path) -> "CalibrationStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return CalibrationStore({cid: CameraCalibration.from_dict(c) for cid, c in data.items()})
