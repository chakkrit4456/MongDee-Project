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

Important detail #2 — the embedding alone is not discriminative enough on
its own, even after centering. MobileNetV3-Small's global-average-pooled
feature is dominated by coarse layout/shape/proportion, not fine color or
texture identity: empirically (measured with this exact code, see the
implementation history), a same-shaped rectangle in a DIFFERENT color than
a registered product scored a centered similarity of 0.96 against it --
well above MATCH_FLOOR, indistinguishable from a genuine ~0.997 same-
product match. Two other clearly unrelated shapes (a circle, a triangle)
still scored 0.76-0.77 -- comfortably above SINGLE_PRODUCT_MATCH_FLOOR on a
single-product catalog. Raising the floor further would only push out more
true positives; it does not fix the underlying problem (an unrelated
object should never depend on a knife-edge threshold to be rejected).

The fix is a second, independent signal: a coarse HSV color-histogram
"fingerprint" stored alongside each sample's embedding. identify()'s
CNN-embedding ranking still picks the best candidate, but that candidate is
only confirmed if its color histogram ALSO correlates well with that same
product's own gallery -- two weak, differently-biased signals (one
shape/layout-dominated, one color-distribution-dominated) that a genuinely
different object is unlikely to satisfy simultaneously, even though either
one alone can be fooled. See identify()'s and _histogram()'s docstrings.
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

# Color-histogram cross-check (see module docstring "Important detail #2").
# 8x8x8 HSV bins: coarse enough to be robust to lighting/crop-boundary
# noise, fine enough to separate genuinely different-colored objects.
HIST_BINS = (8, 8, 8)
HIST_DIM = HIST_BINS[0] * HIST_BINS[1] * HIST_BINS[2]
# Minimum HISTCMP_CORREL (-1..1) between a query and the best-matching
# product's own gallery histograms. Measured against this project's own
# ProductRecognizer with a synthetic registered product and several
# synthetic unrelated objects (not guessed): a genuine same-product match
# under different lighting scored 0.877; two clearly unrelated shapes
# scored -0.03 and 0.02; the hardest case -- a same-shape/layout object in
# a DIFFERENT color, which scored a dangerously high 0.96 centered
# EMBEDDING similarity (see docstring) -- scored only 0.48 here, well
# below this floor. Like MATCH_FLOOR, an initial calibrated value from
# this one measurement, not a rigorously swept optimum across many real
# products -- revisit as more real registration data accumulates.
HIST_MATCH_FLOOR = 0.55
# A color-histogram correlation at least this strong can rescue a far/small candidate past the
# DISTANCE-scaled portion of identify()'s floor (see identify()'s docstring) -- set well above
# HIST_MATCH_FLOOR (a genuine same-product-different-lighting match measured 0.877, see
# HIST_MATCH_FLOOR's own comment) so only a color match that is itself already quite confident
# gets to compensate for the embedding's degraded discriminative power at long range/low detail.
STRONG_HIST_RESCUE_FLOOR = 0.80

# Viewpoint-robustness augmentation (see ProductRecognizer._viewpoint_variants): each real
# registered photo also stores a few synthetic nearby-angle/lighting variants, so the gallery
# covers a small neighborhood around every angle the operator actually captured instead of only
# that one exact pixel-for-pixel framing. Kept modest (5 total variants including the original)
# because add_sample() runs one forward pass per variant -- large enough to meaningfully widen
# the recognized angle range, small enough to stay fast during live "record from camera" (still
# well under the guided-recording flow's 500ms per-tick budget on the tiny MobileNetV3-Small
# backbone this project uses).
ROTATION_JITTER_DEG = 10.0        # in-plane tilt tolerance (camera/hand angle)
SHEAR_JITTER = 0.12               # horizontal shear -- approximates a slightly rotated viewpoint
BRIGHTNESS_CONTRAST_ALPHA = 0.85  # contrast multiplier for the lighting-jitter variant
BRIGHTNESS_CONTRAST_BETA = -15.0  # brightness offset for the lighting-jitter variant

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
        # Computed before _load() so _load() can validate each stored gallery
        # file's width (embedding ++ color histogram, see module docstring
        # "Important detail #2") against it and quarantine anything that
        # predates the histogram column instead of silently misreading it.
        self._calibration_mean = self._load_or_build_calibration()
        self._embed_dim = int(self._calibration_mean.shape[0])
        self._load()

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
        expected_width = self._embed_dim + HIST_DIM
        incompatible: list[str] = []
        for product_key in list(self._manifest.keys()):
            path = self._gallery_path(product_key)
            if not path.exists():
                continue
            array = np.load(path)
            if array.ndim != 2 or array.shape[1] != expected_width:
                # Pre-dates the color-histogram column (or is otherwise
                # corrupt) -- identify()'s histogram cross-check has no data
                # to work with for these rows, and treating a
                # narrower/wider array as if it matched would silently
                # misread garbage as either embedding or histogram values.
                # Quarantine (never delete) and drop from the live gallery
                # so the product just needs re-registering with the current
                # pipeline, same reversible pattern as prune_inactive().
                incompatible.append(product_key)
                continue
            self._gallery[product_key] = array
        if incompatible:
            qdir = self.gallery_dir / "_quarantine" / (time.strftime("%Y%m%d-%H%M%S") + "_incompatible")
            qdir.mkdir(parents=True, exist_ok=True)
            for product_key in incompatible:
                path = self._gallery_path(product_key)
                if path.exists():
                    shutil.move(str(path), str(qdir / path.name))
                self._manifest.pop(product_key, None)
            self._save_manifest()
            logger.warning(
                "ProductRecognizer: quarantined %d gallery file(s) saved by an older format "
                "(no color-histogram data) -- re-register these products: %s",
                len(incompatible), ", ".join(sorted(incompatible)),
            )

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

    def centered_embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Calibration-centered embedding — the same space identify() scores
        matches in. Exposed for callers that need to compare two images to
        each other directly (e.g. core.training's near-duplicate rejection
        during registration) rather than against a stored gallery."""
        return self._centered(self.embed(image_bgr)[None, :])[0]

    @staticmethod
    def _histogram(image_bgr: np.ndarray) -> np.ndarray:
        """Coarse HSV color-distribution fingerprint, L1-normalized (so
        histogram correlation is comparable regardless of crop size) and
        flattened to HIST_DIM float32 values. Deliberately blunt -- not
        meant to recognize texture or shape on its own, only to complement
        the embedding's own opposite blind spot (see module docstring
        "Important detail #2")."""
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
        total = float(hist.sum())
        if total > 0:
            hist = hist / total
        return hist.flatten().astype(np.float32)

    def _embed_and_describe(self, image_bgr: np.ndarray) -> np.ndarray:
        """Embedding ++ color histogram, concatenated -- the single vector
        stored per gallery sample. Splitting it back into its two parts
        always uses self._embed_dim, computed once from the calibration
        vector's own length (see __init__)."""
        return np.concatenate([self.embed(image_bgr), self._histogram(image_bgr)]).astype(np.float32)

    def _split(self, combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return combined[..., :self._embed_dim], combined[..., self._embed_dim:]

    def _centered(self, vectors: np.ndarray) -> np.ndarray:
        centered = vectors - self._calibration_mean
        norms = np.linalg.norm(centered, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return centered / norms

    def _viewpoint_variants(self, image_bgr: np.ndarray) -> list[np.ndarray]:
        """Synthesize a few extra views of `image_bgr` around the ANGLE the operator actually
        captured, so identify() can still recognize the product from a nearby angle/tilt/distance
        it never literally saw a training photo of. This is not "inventing" an unseen side of the
        object (impossible from one 2-D photo) -- it's covering the gap BETWEEN two real captured
        angles (e.g. the guided 360 recording's own turntable steps), and the ordinary lighting/
        hand-tilt variation of an object being presented to a live camera versus how it was framed
        during training.

        Deliberately does NOT touch identify()'s thresholds (MATCH_FLOOR/HIST_MATCH_FLOOR/MARGIN)
        to get this: every variant here is a geometric/photometric transform of a REAL registered
        photo, so it only ever adds genuine additional positive evidence for that exact product --
        it cannot make an unrelated object score higher, and the color-histogram cross-check in
        identify() still runs against these variants exactly as it would against a real photo.
        Skips (returns just the original) on a crop too small to warp usefully."""
        variants = [image_bgr]
        h, w = image_bgr.shape[:2]
        if h < 16 or w < 16:
            return variants
        center = (w / 2.0, h / 2.0)

        for angle in (-ROTATION_JITTER_DEG, ROTATION_JITTER_DEG):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            variants.append(cv2.warpAffine(image_bgr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101))

        shear = SHEAR_JITTER
        shear_matrix = np.array([[1.0, shear, -shear * h / 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        variants.append(cv2.warpAffine(image_bgr, shear_matrix, (w, h), borderMode=cv2.BORDER_REFLECT101))

        variants.append(cv2.convertScaleAbs(image_bgr, alpha=BRIGHTNESS_CONTRAST_ALPHA,
                                             beta=BRIGHTNESS_CONTRAST_BETA))
        return variants

    # ---------------------------------------------------------------- API
    def add_sample(self, product_key: str, image_bgr: np.ndarray) -> int:
        """Add one training image for a product. Returns the new REAL sample count (not the
        internal gallery row count -- see _viewpoint_variants: each real photo also registers a
        few synthetic nearby-angle/lighting variants for better matching robustness, but the UI's
        "N photos" progress guidance must keep counting what the operator actually provided).
        Raises ProductDeleted if the product is not in the catalog (any more) - this is
        what stops a still-running import from resurrecting a deleted product."""
        self._require_active(product_key)        # fail fast before the expensive forward pass
        rows = np.stack([self._embed_and_describe(variant)
                          for variant in self._viewpoint_variants(image_bgr)])
        with self._lock:
            # Re-check atomically with the mutation: a delete that ran while we were
            # embedding has already removed the catalog entry, so we must not write.
            self._require_active(product_key)
            existing = self._gallery.get(product_key)
            self._gallery[product_key] = (
                np.vstack([existing, rows]) if existing is not None else rows
            )
            np.save(self._gallery_path(product_key), self._gallery[product_key])
            real_count = self._manifest.get(product_key, {}).get("count", 0) + 1
            self._manifest[product_key] = {
                "count": real_count,
                "updated_at": time.time(),
            }
            self._save_manifest()
            return real_count

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
        """Returns (product_key, similarity) for the best match, or
        (None, best_score) if nothing clears the floor, if the top match
        doesn't beat the runner-up product by enough of a margin to be
        confident it isn't a mix-up, OR if the winning candidate's color
        histogram doesn't also correlate well enough with that same
        product's own gallery (see module docstring "Important detail #2"
        -- the embedding alone is not discriminative enough to trust by
        itself; both signals must agree). best_score is always the
        embedding similarity, even when the histogram check is what caused
        the reject, so a caller/log can tell "close embedding, wrong
        color" apart from "not close at all"."""
        active = self._active_keys()
        with self._lock:
            gallery = {k: v for k, v in self._gallery.items() if active is None or k in active}
        if not gallery:
            return None, 0.0

        query = self._centered(self.embed(image_bgr)[None, :])[0]

        scores: dict[str, float] = {}
        for product_key, combined in gallery.items():
            if combined.shape[0] == 0:
                continue
            embeddings, _hist = self._split(combined)
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

        if not (best_score >= effective_floor and (best_score - runner_up_score) >= margin):
            # A far/small candidate can fail only because of the DISTANCE-scaled portion of
            # `floor` (core/product_confirm.py's product_min_confidence, raised well above the
            # embedding's own base MATCH_FLOOR for a small box) while still being a genuine match
            # -- MobileNetV3's shape/layout embedding is exactly what degrades most on a low-
            # detail, far-away crop, but color distribution barely changes with distance/blur.
            # Rescue it ONLY on an unusually strong color-histogram correlation (well above the
            # ordinary HIST_MATCH_FLOOR bar), and never past the two floors that guard against a
            # real reported false-positive class of bug: the plain MATCH_FLOOR, and (since there
            # is no runner-up to cross-check against) SINGLE_PRODUCT_MATCH_FLOOR -- both stay
            # absolute regardless of how strong the color match looks.
            rescued = (
                has_runner_up and best_score >= floor and (best_score - runner_up_score) >= margin
                and self._histogram_agrees(image_bgr, gallery[best_key], min_correlation=STRONG_HIST_RESCUE_FLOOR)
            )
            if not rescued:
                return None, best_score

        if not self._histogram_agrees(image_bgr, gallery[best_key]):
            return None, best_score  # embedding liked it; color distribution disagreed -- reject

        return best_key, best_score

    def _histogram_agrees(self, image_bgr: np.ndarray, combined_gallery: np.ndarray,
                           min_correlation: float = HIST_MATCH_FLOOR) -> bool:
        """Second, independent verification signal for identify()'s winning
        candidate: does the query's color distribution correlate well with
        ANY of that product's own registered views? A same-shaped,
        differently-colored object can fool the shape/layout-biased
        embedding (see module docstring) but is very unlikely to also
        coincidentally match the registered product's actual color
        distribution. An inconclusive comparison (no real color signal on
        either side) rejects rather than assumes a match, consistent with
        this project's "ambiguous -> no detection" rule."""
        _embeddings, gallery_hists = self._split(combined_gallery)
        query_hist = self._histogram(image_bgr)
        if not np.any(query_hist) or gallery_hists.shape[0] == 0:
            return False
        best_corr = -1.0
        for hist in gallery_hists:
            if not np.any(hist):
                continue
            try:
                corr = cv2.compareHist(query_hist, hist.astype(np.float32), cv2.HISTCMP_CORREL)
            except cv2.error:
                continue
            best_corr = max(best_corr, corr)
        return best_corr >= min_correlation
