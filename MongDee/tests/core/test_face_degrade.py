from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.face_degrade import MAX_FACE_PX, MIN_FACE_PX, OUT_SIZE, degrade_face, sample_face_px


def _face(seed=0, size=224):
    r = np.random.default_rng(seed)
    img = np.full((size, size, 3), (120, 150, 190), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, (150, 180, 230), -1)
    for dx in (-size // 8, size // 8):
        cv2.circle(img, (size // 2 + dx, size // 2 - size // 12), size // 30, (20, 20, 20), -1)
    cv2.line(img, (size // 2 - size // 8, size // 2 + size // 6), (size // 2 + size // 8, size // 2 + size // 6), (40, 40, 160), 3)
    return np.clip(img + r.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def _sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_output_is_a_network_sized_uint8_image():
    out = degrade_face(_face(), np.random.default_rng(0))
    assert out.shape == (OUT_SIZE, OUT_SIZE, 3) and out.dtype == np.uint8


def test_same_generator_state_gives_the_same_picture():
    a = degrade_face(_face(), np.random.default_rng(5), face_px=30)
    b = degrade_face(_face(), np.random.default_rng(5), face_px=30)
    assert np.array_equal(a, b)


def test_smaller_faces_are_softer():
    sharp = np.mean([_sharpness(degrade_face(_face(i), np.random.default_rng(i), face_px=150, strength=0.0)) for i in range(4)])
    soft = np.mean([_sharpness(degrade_face(_face(i), np.random.default_rng(i), face_px=16, strength=0.0)) for i in range(4)])
    assert soft < 0.5 * sharp


def test_sampled_sizes_stay_in_range_and_favour_small_faces():
    rng = np.random.default_rng(1)
    sizes = np.array([sample_face_px(rng) for _ in range(2000)])
    assert sizes.min() >= MIN_FACE_PX - 1e-6 and sizes.max() <= MAX_FACE_PX + 1e-6
    assert np.median(sizes) < 0.5 * (MIN_FACE_PX + MAX_FACE_PX)          # log-uniform: median far below the midpoint


def test_full_strength_damages_more_than_zero_strength():
    face = _face()
    ref = degrade_face(face, np.random.default_rng(0), face_px=60, strength=0.0)
    diffs = [np.abs(degrade_face(face, np.random.default_rng(s), face_px=60, strength=1.0).astype(int) - ref.astype(int)).mean()
             for s in range(6)]
    assert np.mean(diffs) > 5.0


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        degrade_face(np.zeros((0, 0, 3), np.uint8), np.random.default_rng(0))
