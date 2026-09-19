"""Product recognition: pretrained CNN embeddings + cosine-similarity match.

Why embeddings instead of training a classifier: products are added one at
a time through the enrollment flow (a handful of photos each), not as a
large labeled dataset — there's no batch to retrain a classifier on. An
embedding extracted from a generic pretrained CNN lets a brand-new product
be matchable immediately after enrollment, with no training step, and the
cosine-similarity score doubles as a natural confidence value.

MobileNetV3-Small is used for CPU-only inference speed on a booth laptop.
Its ImageNet weights are downloaded once by torchvision on first run and
then cached locally (~%USERPROFILE%\\.cache\\torch) — do this once with
internet access before going fully offline at the venue.

A CNN trained for ImageNet classification learns to be fairly invariant to
exact color (a red mug and a green mug are both still "mug") — great for
general recognition, bad for a product catalog where two SKUs are the same
box shape and differ mainly by an accent color (e.g. flavor variants of the
same packaging). So each embedding also carries an HSV color-histogram
component, concatenated onto the CNN features before normalizing, giving
color a real vote in the similarity score instead of being drowned out by
shape/texture.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app import config
from app.matching import MatchIndex


COLOR_HIST_DIM = 16 * 8  # bins used by _color_histogram


class FeatureExtractor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()  # drop the classifier head; keep pooled features
        model.eval()
        self._model = model
        self._transform = weights.transforms()
        with torch.no_grad():
            cnn_dim = self._model(torch.zeros(1, 3, 224, 224)).shape[1]
        self.output_dim = cnn_dim + COLOR_HIST_DIM

    @torch.no_grad()
    def _cnn_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(torch.from_numpy(rgb).permute(2, 0, 1)).unsqueeze(0)
        return self._model(tensor).squeeze(0).numpy()

    @staticmethod
    def _color_histogram(frame_bgr: np.ndarray) -> np.ndarray:
        """Coarse hue/saturation histogram — cheap, and exactly the signal a
        generic ImageNet CNN under-weights (see module docstring)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            shape_vec = self._unit(self._cnn_features(frame_bgr)) * config.EMBED_SHAPE_WEIGHT
            color_vec = self._unit(self._color_histogram(frame_bgr)) * config.EMBED_COLOR_WEIGHT
            return self._unit(np.concatenate([shape_vec, color_vec]))

    def embed_many(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        # Same preprocessing/weights as single-frame embeddings, evaluated
        # in small batches to reduce CPU overhead without changing old data.
        embeddings = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(frames), 4):
                batch = frames[start:start + 4]
                tensors = [self._transform(torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)) for f in batch]
                features = self._model(torch.stack(tensors)).numpy()
                for frame, feature in zip(batch, features):
                    shape = self._unit(feature) * config.EMBED_SHAPE_WEIGHT
                    color = self._unit(self._color_histogram(frame)) * config.EMBED_COLOR_WEIGHT
                    embeddings.append(self._unit(np.concatenate([shape, color])))
        return embeddings


feature_extractor = FeatureExtractor()
match_index = MatchIndex(feature_extractor.output_dim)
_refresh_lock = threading.Lock()


def refresh_match_index() -> None:
    import json

    from app import db

    with _refresh_lock:
        rows = []
        for row in db.all_embeddings():
            try:
                rows.append((row["product_id"], json.loads(row["embedding"])))
            except (ValueError, TypeError):
                print("[vision] ignoring malformed stored embedding")
        match_index.rebuild(rows)
        stale = db.stale_product_ids()
        if stale:
            print(f"[vision] {len(stale)} product(s) need a re-scan for embedding v{config.EMBED_VERSION}: {sorted(stale)}")
