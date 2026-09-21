"""Regression tests for core/recognizer.py's product-identity verification.

This is a permanent regression test for the exact bug reported against the
running system: register Product A, then present a completely different,
never-registered object to the booth camera -- the system must NOT label
that object Product A. Uses the REAL ProductRecognizer (real MobileNetV3
forward passes), not a mock of the scoring logic, because the bug was in
the actual embedding space's discriminative power, which a mock cannot
reproduce or verify a fix for.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.recognizer import HIST_DIM, ProductRecognizer


def _obj(seed: int, shape: str = "rect", base=(60, 60, 60), size=(240, 320)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = base
    color = tuple(int(v) for v in rng.integers(60, 255, 3))
    if shape == "rect":
        cv2.rectangle(img, (80, 60), (220, 180), color, -1)
    elif shape == "circle":
        cv2.circle(img, (160, 120), 70, color, -1)
    elif shape == "triangle":
        pts = np.array([[160, 50], [240, 190], [80, 190]])
        cv2.fillPoly(img, [pts], color)
    noise = rng.integers(-15, 15, (h, w, 3))
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def _registered_product_a(seed: int) -> np.ndarray:
    """A dark-background image with a consistent RED rectangle -- stands in
    for one specific real registered product across all its training
    photos (same object, mild lighting/noise variation between shots)."""
    img = _obj(seed, "rect", base=(40, 40, 40))
    img[80:180, 100:200] = (30, 30, 220)
    return img


@pytest.fixture
def recognizer(tmp_path: Path) -> ProductRecognizer:
    return ProductRecognizer(gallery_dir=tmp_path / "gallery", device="cpu")


def _register(recognizer: ProductRecognizer, key: str, n: int = 6, seed_base: int = 100) -> None:
    for i in range(n):
        recognizer.add_sample(key, _registered_product_a(seed_base + i))


# ---------------------------------------------------------- the reported bug
def test_same_registered_product_is_identified(recognizer):
    _register(recognizer, "product-a")
    query = _registered_product_a(999)
    key, score = recognizer.identify(query)
    assert key == "product-a"
    assert score > 0.9


def test_unrelated_never_registered_object_is_rejected_not_labelled_product_a(recognizer):
    """The exact reported scenario: Product A is registered, then a
    completely different, never-trained object is shown. Must return no
    match, not Product A."""
    _register(recognizer, "product-a")
    unrelated_circle = _obj(5, "circle", base=(80, 80, 80))
    unrelated_circle[60:180, 120:260] = (40, 200, 40)
    key, _score = recognizer.identify(unrelated_circle)
    assert key is None

    unrelated_triangle = _obj(6, "triangle", base=(20, 20, 20))
    key, _score = recognizer.identify(unrelated_triangle)
    assert key is None


def test_same_shape_different_color_object_is_rejected():
    """The hardest case: an object with the SAME layout/shape as the
    registered product but a visibly different color. MobileNetV3's
    global-pooled embedding alone scored this 0.96 (near-indistinguishable
    from a genuine ~0.997 match) in the investigation that led to this
    fix -- only the color-histogram cross-check catches it. Uses a fresh
    recognizer (not the fixture) so this test's own gallery is isolated
    and its story is self-contained."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = ProductRecognizer(gallery_dir=Path(d) / "gallery", device="cpu")
        _register(rec, "product-a")
        same_layout_different_color = _obj(7, "rect", base=(40, 40, 40))
        same_layout_different_color[80:180, 100:200] = (200, 200, 30)  # cyan-ish, not red
        key, score = rec.identify(same_layout_different_color)
        assert key is None
        assert score > 0.45  # confirms this genuinely exercised the histogram veto, not just a weak embedding score


def test_registered_product_still_matches_under_different_lighting(recognizer):
    """The fix must not make the recognizer unusable -- a real photo of the
    SAME product taken under different lighting/crop position must still
    match."""
    _register(recognizer, "product-a")
    same_different_lighting = _obj(1234, "rect", base=(45, 42, 38))
    same_different_lighting[85:175, 105:195] = (25, 25, 200)
    key, _score = recognizer.identify(same_different_lighting)
    assert key == "product-a"


# ------------------------------------------------------ multiple products
def _registered_product_b(seed: int) -> np.ndarray:
    """A structurally different object (a green circle, not a rectangle)
    standing in for a second, distinct registered product. Deliberately
    NOT just "the same rectangle in a different color" -- two products
    that differ ONLY in fill color are the single hardest case for this
    embedding (see test_same_shape_different_color_object_is_rejected) and
    the spec this fix implements explicitly calls for REJECTING an
    embedding match that is too close to call, even between two real
    registered products -- that is a distinct, intentional scenario from
    "products a typical booth would register", which is what this test is
    about."""
    img = _obj(seed, "circle", base=(40, 40, 40))
    img[60:180, 120:260] = (40, 200, 40)
    return img


def test_multiple_registered_products_each_keep_their_own_identity(recognizer):
    _register(recognizer, "product-a", seed_base=100)
    for i in range(6):
        recognizer.add_sample("product-b", _registered_product_b(200 + i))

    key_a, _ = recognizer.identify(_registered_product_a(999))
    assert key_a == "product-a"
    key_b, _ = recognizer.identify(_registered_product_b(999))
    assert key_b == "product-b"


# --------------------------------------------------- storage/compatibility
def test_old_embedding_only_gallery_file_is_quarantined_not_misread(tmp_path: Path):
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    rec = ProductRecognizer(gallery_dir=gallery_dir, device="cpu")
    embed_dim = rec._embed_dim

    # Simulate a gallery saved by the pre-histogram format: embedding-only rows.
    old_format = np.random.default_rng(0).random((3, embed_dim)).astype(np.float32)
    np.save(gallery_dir / "legacy-product.npy", old_format)
    import json
    (gallery_dir / "manifest.json").write_text(
        json.dumps({"legacy-product": {"count": 3, "updated_at": 0}}), encoding="utf-8"
    )

    rec2 = ProductRecognizer(gallery_dir=gallery_dir, device="cpu")
    assert "legacy-product" not in rec2.sample_counts()
    assert not (gallery_dir / "legacy-product.npy").exists()
    quarantined = list((gallery_dir / "_quarantine").glob("*/legacy-product.npy"))
    assert len(quarantined) == 1  # moved, not deleted


def test_gallery_rows_store_embedding_plus_histogram_width(recognizer):
    _register(recognizer, "product-a", n=1)
    stored = recognizer._gallery["product-a"]
    assert stored.shape[1] == recognizer._embed_dim + HIST_DIM
