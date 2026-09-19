"""Re-ID feature extraction + track-level aggregation
(MongDee_Master_Prompt.md sections 9, 23, 24).

ReIDExtractor.embed() returns an L2-normalised vector, or None when the
crop fails the quality gate (section 24: don't let a bad crop drive
identity). Cosine similarity between two normalised vectors is just their
dot product.

TrackEmbeddingAggregator keeps only the best few crops per track (by
quality score) and averages their embeddings into one representative
vector (section 23: keyframe sampling + track-level aggregation, not one
embedding per frame).
"""

from __future__ import annotations

import heapq

import numpy as np

from vision.reid.backends import ReIDBackend, build_backend
from vision.reid.config import ReIDConfig
from vision.reid.quality import CropQuality, assess_crop


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class ReIDExtractor:
    def __init__(self, config: ReIDConfig | None = None, backend: ReIDBackend | None = None):
        self.config = config or ReIDConfig()
        self.backend = backend or build_backend(self.config)

    @property
    def dim(self) -> int:
        return self.backend.dim

    def assess(self, image: np.ndarray, bbox: list[float]) -> CropQuality:
        c = self.config
        return assess_crop(
            image, bbox,
            min_height_px=c.min_height_px, min_blur_var=c.min_blur_var,
            min_aspect=c.min_aspect, max_aspect=c.max_aspect,
        )

    def embed(self, image: np.ndarray, bbox: list[float]) -> tuple[np.ndarray | None, CropQuality]:
        quality = self.assess(image, bbox)
        if not quality.usable:
            return None, quality
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        vec = l2_normalize(self.backend.embed(crop).astype(np.float32))
        return vec, quality


class TrackEmbeddingAggregator:
    """Per-track. Keeps the top-K (quality, embedding) pairs and produces a
    representative embedding on demand."""

    def __init__(self, max_embeddings: int = 8):
        self._max = max_embeddings
        self._heap: list[tuple[float, int, np.ndarray]] = []  # min-heap on quality
        self._counter = 0
        self._representative: np.ndarray | None = None
        self._dirty = False

    def add(self, embedding: np.ndarray, quality_score: float) -> None:
        self._counter += 1
        item = (quality_score, self._counter, embedding)
        if len(self._heap) < self._max:
            heapq.heappush(self._heap, item)
        elif quality_score > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)
        else:
            return
        self._dirty = True

    @property
    def sample_count(self) -> int:
        return len(self._heap)

    def representative(self) -> np.ndarray | None:
        if not self._heap:
            return None
        if self._dirty or self._representative is None:
            stacked = np.stack([e for _, _, e in self._heap])
            self._representative = l2_normalize(stacked.mean(axis=0))
            self._dirty = False
        return self._representative
