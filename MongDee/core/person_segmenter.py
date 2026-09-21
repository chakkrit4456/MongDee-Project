"""Per-pixel human-region segmentation, used to keep hands/arms/sleeves out
of the product representation -- both when training a product (core.training)
and when validating a live candidate before it can become a confirmed
product (core.vision's _run_custom_recognition).

Reuses the same YOLO11 family already vendored in this project (yolo11n.pt)
rather than adding a new dependency: yolo11n-seg.pt is the instance-
segmentation checkpoint from the same Ultralytics release, auto-downloaded
next to yolo11n.pt on first use. One mask per detected person instance --
since a person mask covers their whole visible silhouette (torso, sleeve,
arm, hand alike), excluding it is a solid proxy for "exclude human regions"
without needing a dedicated hand/pose model this project has never used.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger("mongdee.core.person_segmenter")

DEFAULT_SEG_WEIGHTS = Path(__file__).resolve().parent.parent / "yolo11n-seg.pt"
PERSON_CLASS_NAME = "person"
DEFAULT_PERSON_CONF = 0.35


class PersonSegmenter:
    """Wraps a lazily-loaded YOLO11-seg model. Never raises -- any failure
    to load or run (no weights, no internet on first use, GPU hiccup) just
    means person_mask() reports "no human found", which is the safe
    default: callers then treat every candidate as product-only instead of
    blocking product recognition entirely because segmentation is
    unavailable. See ProductRecognizer's own GPU-fallback comment for the
    same "a real but survivable degradation" reasoning."""

    def __init__(self, device="cpu", weights: Path | str = DEFAULT_SEG_WEIGHTS):
        self.device = device
        self._weights = weights
        self._model = None
        self._person_class_index: int | None = None
        self._load_lock = threading.Lock()
        self._load_failed = False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._load_lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            try:
                from ultralytics import YOLO

                model = YOLO(str(self._weights))
                index = next((i for i, n in model.names.items() if n == PERSON_CLASS_NAME), None)
                if index is None:
                    raise RuntimeError("yolo11n-seg.pt has no 'person' class")
                self._model = model
                self._person_class_index = index
                return True
            except Exception:
                logger.warning("PersonSegmenter: failed to load %s -- human-region exclusion "
                                "disabled until this succeeds", self._weights, exc_info=True)
                self._load_failed = True
                return False

    def person_mask(self, frame_bgr: np.ndarray, conf: float = DEFAULT_PERSON_CONF) -> np.ndarray:
        """Boolean [H, W] mask, True wherever any detected person instance's
        segmentation covers that pixel. All-False (never None) when
        unavailable or nobody is in frame, so callers can use it
        unconditionally without an availability check."""
        h, w = frame_bgr.shape[:2]
        empty = np.zeros((h, w), dtype=bool)
        if not self._ensure_loaded():
            return empty
        try:
            results = self._model.predict(
                source=frame_bgr, conf=conf, classes=[self._person_class_index],
                device=self.device, verbose=False, retina_masks=True,
            )
        except Exception:
            logger.warning("PersonSegmenter: predict() failed -- treating frame as human-free",
                            exc_info=True)
            return empty
        if not results or results[0].masks is None:
            return empty
        try:
            data = results[0].masks.data.detach().cpu().numpy()
        except Exception:
            return empty
        if data.size == 0:
            return empty
        combined = np.any(data > 0.5, axis=0)
        if combined.shape != (h, w):
            import cv2
            combined = cv2.resize(combined.astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        return combined


_default_lock = threading.Lock()
_default_segmenter: PersonSegmenter | None = None
_default_segmenter_device = None


def get_person_segmenter(device="cpu") -> PersonSegmenter:
    """Process-wide lazy singleton, mirroring core.vision._get_default_ai_worker's
    pattern -- one loaded model shared by every camera/import job instead of
    each caller loading its own copy."""
    global _default_segmenter, _default_segmenter_device
    with _default_lock:
        if _default_segmenter is None or _default_segmenter_device != device:
            _default_segmenter = PersonSegmenter(device=device)
            _default_segmenter_device = device
        return _default_segmenter
