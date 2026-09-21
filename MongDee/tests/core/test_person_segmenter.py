"""Unit tests for core/person_segmenter.py.

Deliberately never loads the real yolo11n-seg.pt weights (that's an
integration/manual-hardware concern, not a fast unit test) -- these tests
exercise the "model unavailable" degrade-gracefully path, which is the
behaviour every caller (core/training.py, core/vision.py) actually depends
on being safe.
"""

from __future__ import annotations

import numpy as np

from core.person_segmenter import PersonSegmenter, get_person_segmenter


def _frame():
    return np.zeros((100, 140, 3), dtype=np.uint8)


def test_missing_weights_never_raises_and_reports_unavailable():
    segmenter = PersonSegmenter(device="cpu", weights="__does_not_exist__.pt")
    assert segmenter.available is False
    mask = segmenter.person_mask(_frame())
    assert mask.shape == (100, 140)
    assert mask.dtype == bool
    assert not mask.any()


def test_person_mask_is_always_returned_never_none():
    segmenter = PersonSegmenter(device="cpu", weights="__does_not_exist__.pt")
    mask = segmenter.person_mask(_frame())
    assert mask is not None


def test_failed_load_is_cached_not_retried_every_call():
    segmenter = PersonSegmenter(device="cpu", weights="__does_not_exist__.pt")
    segmenter.person_mask(_frame())
    assert segmenter._load_failed is True
    # A second call must short-circuit on the cached failure (_ensure_loaded
    # returns False immediately) rather than trying to import/construct
    # YOLO again on every single frame.
    assert segmenter._ensure_loaded() is False


def test_get_person_segmenter_is_a_singleton_per_device():
    a = get_person_segmenter("cpu")
    b = get_person_segmenter("cpu")
    assert a is b


def test_get_person_segmenter_reloads_on_device_change():
    cpu_one = get_person_segmenter("cpu")
    gpu = get_person_segmenter(0)
    cpu_two = get_person_segmenter("cpu")
    assert gpu is not cpu_one
    # Switching back to "cpu" builds a fresh instance (the module caches
    # only the single most-recently-requested device), not the original.
    assert cpu_two is not cpu_one
