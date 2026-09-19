"""Person Detection — Phase 2 (MongDee_Master_Prompt.md section 5).

Wraps an ultralytics YOLO model, filters to the "person" class only, and
returns detections in the ORIGINAL camera-frame coordinate space (the
resize done for inference is undone here so nothing downstream has to know
about it).

This is a *separate* YOLO instance from the one in core/vision.py that the
booth's product recogniser uses — the person pipeline and the product
pipeline stay decoupled so neither can break the other (yolo11n weights
are ~5 MB, so a second copy costs almost nothing). Inference is serialised
behind a lock because neither ultralytics nor torch guarantees a
thread-safe forward pass, and the pipeline calls detect() from a worker
thread.
"""

from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np

from vision.config import DetectionConfig
from vision.frame import FrameProcessor


@dataclasses.dataclass
class Detection:
    """One detected person (section 5 output fields)."""

    bbox: list[float]  # [x1, y1, x2, y2] in original camera-frame pixels
    confidence: float
    camera_id: str
    timestamp: float
    class_name: str = "person"
    local_track_id: int | None = None  # filled by the tracker in Phase 3

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda:0' if a GPU is usable, else 'cpu' (section 30: CPU
    fallback). An explicit value is returned unchanged."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


class PersonDetector:
    def __init__(self, config: DetectionConfig | None = None, model=None):
        self.config = config or DetectionConfig()
        self.device = resolve_device(self.config.device)
        self._processor = FrameProcessor(self.config.processing_width, self.config.processing_height)
        self._lock = threading.Lock()

        if model is not None:
            self.model = model  # dependency injection for tests
        else:
            from ultralytics import YOLO

            self.model = YOLO(self.config.model_path)

        names = getattr(self.model, "names", {})
        resolved = names.get(self.config.person_class_id) if isinstance(names, dict) else None
        if resolved is not None and resolved != "person":
            raise ValueError(
                f"PersonDetector: person_class_id={self.config.person_class_id} is "
                f"{resolved!r} in this model, not 'person'"
            )

    # ------------------------------------------- adaptive control (backend/adaptive.py)
    def get_imgsz(self) -> int:
        return self.config.imgsz

    def set_imgsz(self, size: int) -> None:
        """Change the YOLO input size at runtime. self.config is frozen
        (dataclasses.replace makes a new instance) but the attribute
        itself is a plain mutable reference read fresh by detect_classes()
        on every call, so this takes effect on the very next inference."""
        self.config = dataclasses.replace(self.config, imgsz=size)

    def warmup(self) -> None:
        """Run one throwaway inference so the first real frame is not slowed
        by lazy model fusion / thread-pool init."""
        blank = np.zeros((self.config.processing_height, self.config.processing_width, 3), dtype=np.uint8)
        self.detect(blank, camera_id="__warmup__")

    def detect(self, image: np.ndarray, camera_id: str = "", timestamp: float | None = None) -> list[Detection]:
        return self.detect_classes(
            image, [self.config.person_class_id], camera_id, timestamp,
            class_name=lambda _cid: "person",
        )

    def detect_classes(
        self,
        image: np.ndarray,
        class_ids: list[int],
        camera_id: str = "",
        timestamp: float | None = None,
        conf: float | None = None,
        class_name=None,
    ) -> list[Detection]:
        """Generic single-model detection for an arbitrary class list. Used
        by detect() (person) and by the product module (see
        vision/product/pipeline.py) so the whole system shares one YOLO
        instance and one lock."""
        if timestamp is None:
            timestamp = time.time()
        processed = self._processor.process(image)
        names = getattr(self.model, "names", {})
        resolve_name = class_name or (lambda cid: names.get(cid, str(cid)))

        with self._lock:
            results = self.model.predict(
                source=processed.image,
                imgsz=self.config.imgsz,
                conf=self.config.confidence_threshold if conf is None else conf,
                iou=self.config.iou_threshold,
                classes=list(class_ids),
                max_det=self.config.max_detections,
                device=self.device,
                verbose=False,
            )

        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        allowed = set(class_ids)
        detections: list[Detection] = []
        for box in boxes:
            cid = int(box.cls.item())
            if cid not in allowed:
                continue  # defensive: predict(classes=...) should already guarantee this
            detections.append(
                Detection(
                    bbox=processed.to_original(box.xyxy[0].tolist()),
                    confidence=float(box.conf.item()),
                    camera_id=camera_id,
                    timestamp=timestamp,
                    class_name=resolve_name(cid),
                )
            )
        return detections
