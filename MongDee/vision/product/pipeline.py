"""Product vision module (MongDee_Master_Prompt.md sections 48-51).

Runs alongside the person pipeline on the same frames, sharing the one
YOLO instance (via PersonDetector.detect_classes). Per camera:

    detect product-candidate boxes -> track (stable local id) -> classify
    against the GI reference gallery -> attach GI info + booth zone

Output is a list of ProductDetection with product_class / product_id /
bbox / confidence / camera_id / local_track_id / timestamp / zone_id /
status (KNOWN / POSSIBLE / UNKNOWN).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from vision.detection.detector import PersonDetector
from vision.product.classifier import UNKNOWN, ProductClassifier, ProductMatch
from vision.product.config import ProductConfig
from vision.product.database import GIProductDatabase
from vision.product.tracker import ProductTracker
from vision.spatial.booth import BoothLayout
from vision.spatial.calibration import CameraCalibration


@dataclasses.dataclass
class ProductDetection:
    bbox: list[float]
    confidence: float
    camera_id: str
    timestamp: float
    local_track_id: int
    product_class: str = ""
    product_id: str | None = None
    status: str = UNKNOWN
    zone_id: str | None = None
    match: ProductMatch | None = None


class ProductVisionModule:
    def __init__(
        self,
        detector: PersonDetector,
        config: ProductConfig | None = None,
        classifier: ProductClassifier | None = None,
        gi_database: GIProductDatabase | None = None,
    ):
        self.config = config or ProductConfig()
        self._detector = detector
        self._classifier = classifier or ProductClassifier(self.config)
        self._gi = gi_database or GIProductDatabase()
        self._trackers: dict[str, ProductTracker] = {}
        self._classified: dict[tuple[str, int], ProductMatch] = {}  # cache per track (products don't change)

    def process(
        self,
        camera_id: str,
        image: np.ndarray,
        timestamp: float | None = None,
        calibration: CameraCalibration | None = None,
        layout: BoothLayout | None = None,
    ) -> list[ProductDetection]:
        if timestamp is None:
            timestamp = time.time()

        detections = self._detector.detect_classes(
            image, list(self.config.yolo_classes), camera_id, timestamp, conf=self.config.detection_confidence
        )
        tracker = self._trackers.setdefault(camera_id, ProductTracker(self.config))
        tracks = tracker.update([d.bbox for d in detections], [d.confidence for d in detections], timestamp)

        h, w = image.shape[:2]
        out: list[ProductDetection] = []
        for track in tracks:
            key = (camera_id, track.track_id)
            match = self._classified.get(key)
            if match is None or match.status == UNKNOWN:
                x1, y1, x2, y2 = (int(round(v)) for v in track.bbox)
                crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                match = self._classifier.classify(crop) if crop.size else ProductMatch(UNKNOWN, None, 0.0)
                self._classified[key] = match

            zone_id = None
            if calibration is not None and layout is not None:
                cx = (track.bbox[0] + track.bbox[2]) / 2.0
                cy = (track.bbox[1] + track.bbox[3]) / 2.0
                wx, wy = calibration.pixel_to_world(cx, cy)
                zone = layout.zone_at(wx, wy)
                zone_id = zone.id if zone else None

            gi = self._gi.get(match.product_id) if match.product_id else None
            out.append(
                ProductDetection(
                    bbox=track.bbox, confidence=round(track.confidence, 3), camera_id=camera_id,
                    timestamp=timestamp, local_track_id=track.track_id,
                    product_class=gi.class_name if gi else "", product_id=match.product_id,
                    status=match.status, zone_id=zone_id, match=match,
                )
            )
        return out

    def gi_info(self, product_id: str):
        return self._gi.get(product_id)

    def load_gallery_from_dir(self, gallery_dir: str) -> int:
        """gallery_dir/<product_id>/*.jpg -> reference embeddings. Returns
        the number of reference images loaded."""
        import cv2
        from pathlib import Path

        count = 0
        for product_dir in sorted(Path(gallery_dir).iterdir()):
            if not product_dir.is_dir():
                continue
            for img_path in sorted(product_dir.glob("*")):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self._classifier.add_reference(product_dir.name, img)
                    count += 1
        return count
