"""Trainable product recognizer — few-shot image matching, no bounding-box
labeling and no training loop required.

How it works: a frozen, pretrained CNN (MobileNetV3-Small, ImageNet
weights) turns any product photo into a fixed-length "fingerprint" vector.
Uploading more reference photos of a product just adds more fingerprints to
its gallery — recognition accuracy improves immediately with each upload,
with no retraining step. At recognition time, a live camera crop's
fingerprint is compared (cosine similarity) against every product's
gallery; the closest match above a confidence threshold wins.

This is what makes "keep uploading images/video until it's accurate enough"
(the brief's requirement) actually work: every upload has an immediate,
visible effect, instead of needing a slow train/eval cycle.

Important detail — calibration: raw pooled CNN features are non-negative
(everything after a ReLU/Hardswish + global-average-pool stays >= 0), so
*any* two images — even completely unrelated ones — land unnervingly close
together in raw cosine-similarity terms (empirically ~0.85-0.95 between
random noise images, because they all share the same dominant "generic
photo" direction). Comparing raw embeddings directly would make almost
everything look like a match.

The fix is to mean-center every embedding before comparing — subtracting
out that shared direction so what's left is what's actually distinctive.
The mean MUST come from a fixed, gallery-independent calibration set (a
synthetic batch of varied colors/textures/noise, built once and cached to
disk), not from the product gallery itself: centering against the
gallery's own mean seems reasonable but silently breaks the moment a
product's gallery is homogeneous (e.g. one product, or many near-identical
photos) — the mean collapses to ~that product's own embedding and wipes out
its signal entirely. A fixed external calibration vector has no such
failure mode and works identically whether the catalog has 1 product or
50.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from core.product_lifecycle import ProductDeleted

logger = logging.getLogger("mongdee.core.recognizer")

DEFAULT_GALLERY_DIR = Path(__file__).resolve().parent.parent / "data" / "gallery"
MATCH_FLOOR = 0.45              # minimum centered similarity to even consider a match
MATCH_MARGIN = 0.10             # winner must beat the runner-up product by at least this much
# When the gallery holds only one product, there is no runner-up to require
# a margin against -- identify()'s "(best_score - runner_up_score) >= margin"
# check is then trivially satisfied by anything clearing MATCH_FLOOR alone
# (runner_up_score defaults to -1.0), so the margin provides zero real
# protection in exactly this state. A single-product catalog is a common,
# not edge-case, scenario (initial setup, a booth demoing one item), so
# demand a stronger floor when there's no competing product to differentiate
# against, instead of silently relying on a check that can't run. Like
# ATTRIBUTE_MIN_FACE_SIZE_PX in core/attributes.py, this specific value is
# an engineering judgment informed by this gap, not a rigorously swept
# optimum -- revisit once more real false-positive/true-positive examples
# are collected across different catalog sizes.
SINGLE_PRODUCT_MATCH_FLOOR = 0.65
RECOMMENDED_SAMPLES = 15        # UI guidance: "enough" data to be reliable
CALIBRATION_SEED = 42
CALIBRATION_SIZE = 30

_embed_lock = threading.Lock()  # serialize forward passes across camera threads


def _build_calibration_images() -> list[np.ndarray]:
    """A fixed, deterministic set of synthetic images spanning hue, noise
    texture and repeating patterns — standing in for "generic photo content"
    so the calibration mean isn't biased toward any particular product."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    images = []
    for hue in range(0, 180, 12):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        images.append(img)
    for _ in range(15):
        noise = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(cv2.GaussianBlur(noise, (21, 21), 0))
    return images


class ProductRecognizer:
    def __init__(self, gallery_dir: Path = DEFAULT_GALLERY_DIR, device: str = "cpu"):
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        backbone.classifier = torch.nn.Identity()  # keep the 576-d pooled feature, drop the head
        backbone.eval()
        try:
            backbone.to(self.device)
        except Exception as exc:
            # A GPU device string here already passed core.device.resolve_
            # device()'s own smoke test in the common case, but that's a
            # best-effort check, not a guarantee — a switchable-graphics/
            # power-managed GPU can still go from "just worked" to "busy or
            # unavailable" a moment later (this is exactly the failure this
            # project's own users have hit: torch.AcceleratorError:
            # "CUDA-capable device(s) is/are busy or unavailable"). Losing
            # GPU acceleration for the embedding recognizer is a real but
            # survivable degradation; crashing the whole booth process over
            # it is not — fall back to CPU instead.
            logger.warning("ProductRecognizer: failed to move model to %s (%s) — falling back to CPU",
                            self.device, exc)
            self.device = torch.device("cpu")
            backbone.to(self.device)
        self._model = backbone
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # product_key -> np.ndarray [N, 576] of raw (uncentered) L2-normalized embeddings
        self._gallery: dict[str, np.ndarray] = {}
        self._manifest: dict[str, dict] = {}  # product_key -> {"count": N, "updated_at": ts}
        # Guards _gallery/_manifest/.npy files so a delete (clear_product) can never
        # interleave with an in-flight add_sample from an import thread and leave a
        # "ghost" gallery behind for a product that no longer exists.
        self._lock = threading.RLock()
        # Optional callable returning the product keys that currently exist in the
        # catalog. When set, only those products can be matched or trained, no matter
        # what stale data sits in the gallery on disk. See set_active_provider().
        self._active_provider = None
        self._load()
        self._calibration_mean = self._load_or_build_calibration()

    # ------------------------------------------------------------- storage
    def _gallery_path(self, product_key: str) -> Path:
        safe = product_key.replace("/", "_")
        return self.gallery_dir / f"{safe}.npy"

    def _manifest_path(self) -> Path:
        return self.gallery_dir / "manifest.json"

    def _calibration_path(self) -> Path:
        return self.gallery_dir / "_calibration_mean.npy"

    def _load(self):
        manifest_path = self._manifest_path()
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        for product_key in list(self._manifest.keys()):
            path = self._gallery_path(product_key)
            if path.exists():
                self._gallery[product_key] = np.load(path)

    # --------------------------------------------------- catalog coupling
    def set_active_provider(self, provider) -> None:
        """Tie the gallery to the product catalog. `provider()` must return the keys of
        the products that currently exist. Root cause this fixes: the gallery on disk is
        independent of the catalog, so a product deleted from the catalog (or a stale
        import) left its embeddings behind and identify() kept reporting it even though
        it no longer existed. With a provider set, identify()/add_sample()/
        has_any_gallery() only ever see live products."""
        self._active_provider = provider

    def _active_keys(self):
        """Set of live product keys, or None when no provider is set (legacy behaviour)."""
        if self._active_provider is None:
            return None
        return set(self._active_provider())

    def _require_active(self, product_key: str) -> None:
        active = self._active_keys()
        if active is not None and product_key not in active:
            raise ProductDeleted(product_key)

    def prune_inactive(self, quarantine: bool = True) -> list[str]:
        """Remove gallery entries for products that are not in the catalog. Data files are
        MOVED to gallery_dir/_quarantine/<timestamp>/ (never deleted) so the operation is
        reversible. Returns the pruned keys."""
        active = self._active_keys()
        if active is None:
            return []
        with self._lock:
            stale = sorted((set(self._gallery) | set(self._manifest)) - active)
            if not stale:
                return []
            qdir = self.gallery_dir / "_quarantine" / time.strftime("%Y%m%d-%H%M%S")
            for key in stale:
                self._gallery.pop(key, None)
                self._manifest.pop(key, None)
                path = self._gallery_path(key)
                if path.exists():
                    if quarantine:
                        qdir.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), str(qdir / path.name))
                    else:
                        path.unlink()
            self._save_manifest()
        logger.warning("ProductRecognizer: pruned %d gallery entr%s not in the catalog: %s",
                       len(stale), "y" if len(stale) == 1 else "ies", ", ".join(stale))
        return stale

    def _save_manifest(self):
        with open(self._manifest_path(), "w", encoding="utf-8") as f:
            json.dump(self._manifest, f, ensure_ascii=False, indent=2)

    def _load_or_build_calibration(self) -> np.ndarray:
        path = self._calibration_path()
        if path.exists():
            return np.load(path)
        embeddings = np.array([self._embed_raw(img) for img in _build_calibration_images()])
        mean = embeddings.mean(axis=0)
        np.save(path, mean)
        return mean

    # ------------------------------------------------------------ embedding
    def _embed_raw(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty image")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(image_rgb).unsqueeze(0).to(self.device)
        with _embed_lock, torch.no_grad():
            feature = self._model(tensor).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Raw (uncentered) L2-normalized embedding — what's stored in the
        gallery on disk. Comparisons always go through _centered(), never
        this directly (see module docstring)."""
        return self._embed_raw(image_bgr)

    def _centered(self, vectors: np.ndarray) -> np.ndarray:
        centered = vectors - self._calibration_mean
        norms = np.linalg.norm(centered, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return centered / norms

    # ---------------------------------------------------------------- API
    def add_sample(self, product_key: str, image_bgr: np.ndarray) -> int:
        """Add one training image for a product. Returns the new sample count.
        Raises ProductDeleted if the product is not in the catalog (any more) - this is
        what stops a still-running import from resurrecting a deleted product."""
        self._require_active(product_key)        # fail fast before the expensive forward pass
        embedding = self.embed(image_bgr)
        with self._lock:
            # Re-check atomically with the mutation: a delete that ran while we were
            # embedding has already removed the catalog entry, so we must not write.
            self._require_active(product_key)
            existing = self._gallery.get(product_key)
            self._gallery[product_key] = (
                np.vstack([existing, embedding[None, :]]) if existing is not None
                else embedding[None, :]
            )
            np.save(self._gallery_path(product_key), self._gallery[product_key])
            self._manifest[product_key] = {
                "count": len(self._gallery[product_key]),
                "updated_at": time.time(),
            }
            self._save_manifest()
            return self._manifest[product_key]["count"]

    def clear_product(self, product_key: str) -> None:
        with self._lock:
            self._gallery.pop(product_key, None)
            self._manifest.pop(product_key, None)
            path = self._gallery_path(product_key)
            if path.exists():
                path.unlink()
            self._save_manifest()

    def sample_count(self, product_key: str) -> int:
        return self._manifest.get(product_key, {}).get("count", 0)

    def sample_counts(self) -> dict[str, int]:
        active = self._active_keys()
        with self._lock:
            return {k: v.get("count", 0) for k, v in self._manifest.items()
                    if active is None or k in active}

    def has_any_gallery(self) -> bool:
        active = self._active_keys()
        with self._lock:
            return any(v.shape[0] > 0 for k, v in self._gallery.items()
                       if active is None or k in active)

    def identify(self, image_bgr: np.ndarray, floor: float = MATCH_FLOOR,
                 margin: float = MATCH_MARGIN):
        """Returns (product_key, similarity) for the best match, or (None, best_score)
        if nothing clears the floor, or if the top match doesn't beat the runner-up
        product by enough of a margin to be confident it isn't a mix-up."""
        active = self._active_keys()
        with self._lock:
            gallery = {k: v for k, v in self._gallery.items() if active is None or k in active}
        if not gallery:
            return None, 0.0

        query = self._centered(self.embed(image_bgr)[None, :])[0]

        scores: dict[str, float] = {}
        for product_key, embeddings in gallery.items():
            if embeddings.shape[0] == 0:
                continue
            centered = self._centered(embeddings)
            scores[product_key] = float(np.max(centered @ query))

        if not scores:
            return None, 0.0

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_score = ranked[0]
        has_runner_up = len(ranked) > 1
        runner_up_score = ranked[1][1] if has_runner_up else -1.0
        # See SINGLE_PRODUCT_MATCH_FLOOR's comment: with no runner-up, the
        # margin check below is a no-op, so fall back to a stricter floor
        # instead of silently trusting a disabled safety check.
        effective_floor = floor if has_runner_up else max(floor, SINGLE_PRODUCT_MATCH_FLOOR)

        if best_score >= effective_floor and (best_score - runner_up_score) >= margin:
            return best_key, best_score
        return None, best_score
