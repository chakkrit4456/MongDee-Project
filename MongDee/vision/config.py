"""Detection configuration — Phase 2.

Model settings are global (one detector shared by every camera, since on a
CPU-only box a single YOLO instance serialised behind a lock is the
sensible arrangement — see vision/detection/detector.py). Per-camera
settings such as processing_fps stay in camera config (camera/base.py).

MongDee_Master_Prompt.md sections 5 (Person Detection), 29-30
(performance / GPU), 33 (config).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class DetectionConfig:
    model_path: str = "yolo11n.pt"
    device: str = "auto"  # "auto" -> cuda if available else cpu; or an explicit "cpu" / "cuda:0"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45  # NMS IoU
    imgsz: int = 640  # square size YOLO letterboxes to internally
    person_class_id: int = 0  # COCO "person"
    max_detections: int = 100

    # frame normalization (vision/frame.py) applied before inference
    processing_width: int = 960
    processing_height: int = 540

    # pipeline pacing (vision/pipeline.py): target detections/sec PER camera.
    # Actual rate is capped by CPU inference speed shared across all cameras.
    target_fps_per_camera: float = 4.0

    @staticmethod
    def from_dict(d: dict) -> "DetectionConfig":
        known = {f.name for f in dataclasses.fields(DetectionConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"detection config has unknown field(s): {unknown}")
        return DetectionConfig(**d)

    @staticmethod
    def load(path: str | Path) -> "DetectionConfig":
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read detection config file {path}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        # allow the detection block to be nested under "detection"
        if "detection" in data and isinstance(data["detection"], dict):
            data = data["detection"]
        try:
            return DetectionConfig.from_dict(data)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
