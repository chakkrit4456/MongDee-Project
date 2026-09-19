"""Unit tests for ProductRecognizer.identify()'s matching logic, in
particular the single-product-gallery gap found via a live camera session:
"product_found" events fired with no product actually shown to the camera,
while the catalog held exactly one product. Root cause (confirmed by code
inspection): identify()'s MATCH_MARGIN check compares the best score against
a runner-up product's score, defaulting runner_up_score to -1.0 when no
runner-up exists -- so with only one product in the gallery, the margin
check is trivially satisfied by anything clearing the bare MATCH_FLOOR,
providing no real protection against a false match.

These tests bypass the real MobileNetV3 embedding pipeline (monkeypatching
_centered to a no-op and embed() to return a chosen vector) so similarity
scores are exact and deterministic, instead of depending on real image
content.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.recognizer import MATCH_FLOOR, SINGLE_PRODUCT_MATCH_FLOOR, ProductRecognizer


def _recognizer_with_deterministic_scores(tmp_path, monkeypatch, gallery: dict, query_vec):
    recognizer = ProductRecognizer(gallery_dir=tmp_path, device="cpu")
    monkeypatch.setattr(recognizer, "_centered", lambda v: v)  # skip calibration centering
    recognizer._gallery = {k: np.array(v, dtype=np.float64) for k, v in gallery.items()}
    monkeypatch.setattr(recognizer, "embed", lambda image_bgr: np.array(query_vec, dtype=np.float64))
    return recognizer


def _dummy_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_single_product_gallery_rejects_a_score_between_the_two_floors(tmp_path, monkeypatch):
    # 0.55 clears the ordinary MATCH_FLOOR (0.45) but not
    # SINGLE_PRODUCT_MATCH_FLOOR (0.65) -- with only one product in the
    # gallery, this must now be rejected (the real live false-positive
    # this test encodes: a confident-looking but wrong match with no
    # competing product to weigh it against).
    assert MATCH_FLOOR < 0.55 < SINGLE_PRODUCT_MATCH_FLOOR
    recognizer = _recognizer_with_deterministic_scores(
        tmp_path, monkeypatch,
        gallery={"only-product": [[1.0, 0.0]]},
        query_vec=[0.55, (1 - 0.55 ** 2) ** 0.5],
    )

    key, score = recognizer.identify(_dummy_image())

    assert key is None
    assert score == pytest.approx(0.55)


def test_single_product_gallery_accepts_a_score_above_the_stricter_floor(tmp_path, monkeypatch):
    recognizer = _recognizer_with_deterministic_scores(
        tmp_path, monkeypatch,
        gallery={"only-product": [[1.0, 0.0]]},
        query_vec=[0.9, (1 - 0.9 ** 2) ** 0.5],
    )

    key, score = recognizer.identify(_dummy_image())

    assert key == "only-product"
    assert score == pytest.approx(0.9)


def test_multi_product_gallery_still_uses_the_ordinary_floor_and_margin(tmp_path, monkeypatch):
    # Backward compatibility: with a real runner-up to weigh against, a
    # score between the two floors (0.55) must still be accepted exactly
    # as before, as long as it clears MATCH_MARGIN over the runner-up.
    recognizer = _recognizer_with_deterministic_scores(
        tmp_path, monkeypatch,
        gallery={
            "product-a": [[1.0, 0.0]],
            "product-b": [[-1.0, 0.0]],  # opposite direction -- far below product-a, clear margin
        },
        query_vec=[0.55, (1 - 0.55 ** 2) ** 0.5],
    )

    key, score = recognizer.identify(_dummy_image())

    assert key == "product-a"
    assert score == pytest.approx(0.55)


def test_multi_product_gallery_still_rejects_a_too_close_runner_up(tmp_path, monkeypatch):
    # Two products with near-identical similarity to the query (both close
    # to 0.5) -- MATCH_MARGIN (0.10) must still reject this as ambiguous,
    # unaffected by the single-product-floor change.
    recognizer = _recognizer_with_deterministic_scores(
        tmp_path, monkeypatch,
        gallery={
            "product-a": [[0.51, (1 - 0.51 ** 2) ** 0.5]],
            "product-b": [[0.50, -(1 - 0.50 ** 2) ** 0.5]],
        },
        query_vec=[1.0, 0.0],
    )

    key, score = recognizer.identify(_dummy_image())

    assert key is None


def test_empty_gallery_returns_no_match(tmp_path, monkeypatch):
    recognizer = ProductRecognizer(gallery_dir=tmp_path, device="cpu")
    key, score = recognizer.identify(_dummy_image())
    assert key is None
    assert score == 0.0
