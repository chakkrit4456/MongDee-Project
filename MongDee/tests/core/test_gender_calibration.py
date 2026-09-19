from __future__ import annotations

import json

import numpy as np

from core.gender_calibration import (CALIBRATION_FILENAME, MAX_ABS_BIAS, GenderCalibration, balanced_accuracy,
                                     bucket_index, fit_bias, fit_calibration)


def _scores(n=600, female_shift=-1.2, seed=0):
    """Logit differences (female - male) of a model that under-calls women: female faces sit at +1.5 - shift."""
    r = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n), np.ones(n)]).astype(bool)
    diff = np.where(y, 1.5 + female_shift, -1.5) + r.normal(0, 1.2, 2 * n)
    return diff, y


def test_bias_recovers_a_skewed_model():
    diff, y = _scores()
    b = fit_bias(diff, y)
    assert 0.6 < b < 1.8
    assert balanced_accuracy(diff, y, b) > balanced_accuracy(diff, y, 0.0) + 0.03


def test_balanced_model_needs_no_bias():
    diff, y = _scores(female_shift=0.0, seed=3)
    assert abs(fit_bias(diff, y)) < 0.35


def test_bias_is_clamped_and_safe_on_degenerate_input():
    assert fit_bias(np.array([]), np.array([], bool)) == 0.0
    assert fit_bias(np.array([1.0, 2.0]), np.array([True, True])) == 0.0
    diff = np.concatenate([np.full(50, -20.0), np.full(50, -19.0)])
    assert abs(fit_bias(diff, np.array([False] * 50 + [True] * 50))) <= MAX_ABS_BIAS


def test_size_buckets_are_fitted_separately_and_fall_back_to_global():
    r = np.random.default_rng(1)
    n = 300
    y = np.tile([False, True], n)
    px = np.where(np.arange(2 * n) < n, 20.0, 90.0)                       # half small, half large faces
    shift = np.where(px < 24, -1.6, 0.0)                                  # only small faces are skewed
    diff = np.where(y, 1.5 + shift, -1.5) + r.normal(0, 1.0, 2 * n)
    cal = fit_calibration(diff, y, px, "m.pt")
    assert cal.bias_for(20) > 0.7 and abs(cal.bias_for(90)) < 0.5
    assert cal.bucket_bias[bucket_index(50)] is None                       # empty bucket -> falls back
    assert cal.bias_for(50) == cal.bias


def test_apply_moves_only_the_female_logit():
    cal = GenderCalibration(checkpoint="m.pt", bias=0.8)
    out = cal.apply(np.array([1.0, 0.5]), None)
    assert np.allclose(out, [1.0, 1.3])


def test_round_trip_and_checkpoint_guard(tmp_path):
    ckpt = tmp_path / "webcam_model_state_dict.pt"
    ckpt.write_bytes(b"x")
    cal = GenderCalibration(checkpoint=ckpt.name, bias=0.5, bucket_bias=[1.0, None, None, None, 0.0])
    cal.save(tmp_path / CALIBRATION_FILENAME)
    loaded = GenderCalibration.load_for(ckpt)
    assert loaded is not None and loaded.bias_for(10) == 1.0 and loaded.bias_for(200) == 0.0
    other = tmp_path / "another.pt"
    other.write_bytes(b"x")
    assert GenderCalibration.load_for(other) is None                       # fitted for a different checkpoint


def test_corrupt_or_hostile_file_is_ignored_or_clamped(tmp_path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")
    (tmp_path / CALIBRATION_FILENAME).write_text("{not json", encoding="utf-8")
    assert GenderCalibration.load_for(ckpt) is None
    (tmp_path / CALIBRATION_FILENAME).write_text(json.dumps({"checkpoint": "m.pt", "bias": 1e9, "bucket_bias": ["x", None, 5, None, None]}),
                                                 encoding="utf-8")
    cal = GenderCalibration.load_for(ckpt)
    assert cal.bias == MAX_ABS_BIAS and cal.bias_for(10) == cal.bias and cal.bucket_bias[2] == MAX_ABS_BIAS
