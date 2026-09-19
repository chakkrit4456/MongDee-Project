"""Product classification by reference-image embedding matching
(MongDee_Master_Prompt.md section 50).

Build a gallery: add a few reference crops per GI product. At runtime,
embed a detected crop and find the nearest gallery product. Report:

    KNOWN PRODUCT     high similarity AND a clear margin over the runner-up
    POSSIBLE PRODUCT  medium similarity, or a close call between two products
    UNKNOWN PRODUCT   low similarity to everything

Uses the same swappable embedding backends as person Re-ID
(vision/reid/backends.py). For real GI goods a trained model or the
booth's existing core/recognizer.py gallery will do better than the
colour histogram default.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision.product.config import ProductConfig
from vision.reid.backends import build_backend
from vision.reid.config import ReIDConfig
from vision.reid.extractor import cosine_similarity, l2_normalize

KNOWN = "KNOWN_PRODUCT"
POSSIBLE = "POSSIBLE_PRODUCT"
UNKNOWN = "UNKNOWN_PRODUCT"


@dataclasses.dataclass
class ProductMatch:
    status: str
    product_id: str | None
    confidence: float
    runner_up_id: str | None = None
    runner_up_confidence: float = 0.0


class ProductClassifier:
    def __init__(self, config: ProductConfig | None = None, backend=None):
        self.config = config or ProductConfig()
        self._backend = backend or build_backend(ReIDConfig(backend=self.config.reid_backend))
        self._gallery: dict[str, list[np.ndarray]] = {}

    def add_reference(self, product_id: str, crop_bgr: np.ndarray) -> None:
        vec = l2_normalize(self._backend.embed(crop_bgr).astype(np.float32))
        self._gallery.setdefault(product_id, []).append(vec)

    def reference_count(self, product_id: str) -> int:
        return len(self._gallery.get(product_id, []))

    @property
    def is_ready(self) -> bool:
        return bool(self._gallery)

    def classify(self, crop_bgr: np.ndarray) -> ProductMatch:
        if not self._gallery:
            return ProductMatch(UNKNOWN, None, 0.0)
        vec = l2_normalize(self._backend.embed(crop_bgr).astype(np.float32))

        scores: list[tuple[float, str]] = []
        for product_id, refs in self._gallery.items():
            best = max(cosine_similarity(vec, r) for r in refs)
            scores.append((best, product_id))
        scores.sort(reverse=True)

        top_score, top_id = scores[0]
        runner_score, runner_id = scores[1] if len(scores) > 1 else (0.0, None)
        cfg = self.config

        if top_score < cfg.possible_similarity:
            return ProductMatch(UNKNOWN, None, round(top_score, 3), runner_id, round(runner_score, 3))
        if top_score >= cfg.known_similarity and (top_score - runner_score) >= cfg.top2_margin:
            return ProductMatch(KNOWN, top_id, round(top_score, 3), runner_id, round(runner_score, 3))
        return ProductMatch(POSSIBLE, top_id, round(top_score, 3), runner_id, round(runner_score, 3))
