import cv2
import numpy as np
import pytest

from core.frame_integrity import FrameIntegrity
from core.hresult import decode, scan_log
from tests.core.camera_ai_fixtures import scenes

GOOD = scenes((320, 240))
RNG = np.random.default_rng(0)


def _sequence(base, n, dx=2, noise=2.0, seed=0):
    """A live-camera-like sequence: a 320x240 window sliding over a larger picture + sensor noise."""
    rng = np.random.default_rng(seed)
    big = cv2.resize(base, (420, 300), interpolation=cv2.INTER_AREA)
    out = []
    for k in range(n):
        x = (k * dx) % 100
        f = big[20:260, x:x + 320].astype(np.float32) + rng.normal(0, noise, (240, 320, 3))
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


def _run(frames, fi=None, t0=0.0, dt=0.1):
    fi = fi or FrameIntegrity()
    return [fi.check(f, t0 + i * dt) for i, f in enumerate(frames)]


def test_have_corpus():
    assert len(GOOD) >= 3


def test_clean_live_sequences_never_flagged():
    for base in GOOD:
        verdicts = _run(_sequence(base, 60))
        assert all(v.ok for v in verdicts), [v.reasons for v in verdicts if not v.ok][:3]


def test_blur_lowlight_jpeg_and_mild_noise_still_pass():
    for base in GOOD:
        seq = _sequence(base, 12)
        variants = [[cv2.GaussianBlur(f, (9, 9), 3) for f in seq],
                    [(f * 0.35).astype(np.uint8) for f in seq],
                    [cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 25])[1], 1) for f in seq],
                    _sequence(base, 12, noise=8.0)]
        for frames in variants:
            assert all(v.ok for v in _run(frames))


def test_dark_room_is_not_rejected_but_noted_unless_strict():
    dark = np.full((240, 320, 3), 2, np.uint8)
    v = FrameIntegrity().check(dark, 0)
    assert v.ok and v.notes == ["black"]
    assert "black" in FrameIntegrity(strict_exposure=True).check(dark, 0).reasons
    assert "white" in FrameIntegrity(strict_exposure=True).check(np.full((240, 320, 3), 255, np.uint8), 0).reasons
    assert "flat" in FrameIntegrity(strict_exposure=True).check(np.full((240, 320, 3), 120, np.uint8), 0).reasons


def test_empty_and_wrong_shapes():
    assert FrameIntegrity().check(None, 0).reasons == ["empty"]
    assert not FrameIntegrity().check(np.zeros((0, 0, 3), np.uint8), 0)
    assert FrameIntegrity().check(np.zeros((240, 320), np.uint8), 0).reasons == ["bad_shape_or_dtype"]
    assert FrameIntegrity().check(np.zeros((8, 8, 3), np.uint8), 0).reasons == ["too_small"]


def test_random_noise_garbage_rejected():
    bad = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    v = FrameIntegrity().check(bad, 0)
    assert not v.ok and "noise" in v.reasons


def test_truncated_mjpeg_tail_flagged_only_when_it_appears():
    for base in GOOD:
        seq = _sequence(base, 30)
        seq[20] = seq[20].copy(); seq[20][int(240 * 0.7):] = 128      # decoder gray fill
        verdicts = _run(seq)
        assert "truncated_tail" in verdicts[20].reasons
        assert [i for i, v in enumerate(verdicts) if not v.ok] == [20]


def test_permanent_black_bar_at_the_bottom_is_not_a_truncation():
    for base in GOOD:
        seq = _sequence(base, 40)
        for f in seq:
            f[int(240 * 0.8):] = 0                                        # letterbox present in every frame
        assert all(v.ok for v in _run(seq))


def test_tear_seam_flagged_when_it_appears_and_static_horizontal_edges_are_not():
    hit = 0
    for base in GOOD:
        seq = _sequence(base, 30)
        torn = seq[20].copy(); torn[120:] = np.roll(torn[120:], 60, axis=1)
        seq[20] = torn
        verdicts = _run(seq)
        assert all(v.ok for i, v in enumerate(verdicts) if i != 20), "false positive on a clean frame"
        hit += "tear_seam" in verdicts[20].reasons
    print("tear recall", hit, "/", len(GOOD))
    assert hit >= len(GOOD) - 1        # best-effort: a very low-texture scene can hide a seam
    # a strong permanent horizontal edge (window frame / table edge) in every frame
    for base in GOOD:
        seq = _sequence(base, 40)
        for f in seq:
            f[:120] = (f[:120] * 0.25).astype(np.uint8)                   # bright/dark boundary at row 120
        assert all(v.ok for v in _run(seq))


def test_green_cast_rejected():
    g = GOOD[0].copy(); g[..., 0] = 0; g[..., 2] = 0
    assert "color_cast" in FrameIntegrity().check(g, 0).reasons


def test_freeze_needs_time_count_and_bit_identical_frames():
    fi = FrameIntegrity(freeze_sec=2.0, freeze_min_frames=5)
    im = GOOD[0]; res = [fi.check(im.copy(), k * 0.5) for k in range(12)]
    assert res[3].ok                                     # too early
    assert not res[-1].ok and "frozen" in res[-1].reasons
    assert fi.check(GOOD[1], 7.0).ok                     # moving again clears it


def test_sensor_noise_is_not_freeze():
    fi = FrameIntegrity(freeze_sec=1.0, freeze_min_frames=3)
    rng = np.random.default_rng(3)
    for k in range(20):
        f = np.clip(GOOD[0] + rng.normal(0, 2, GOOD[0].shape), 0, 255).astype(np.uint8)
        assert fi.check(f, k * 0.3).ok


def test_hresult_decode_and_scan():
    assert decode(0xC00D3704)["action"] == "lower_profile"
    assert decode("error 0x8007001F occurred")["name"] == "ERROR_GEN_FAILURE"
    assert decode("no code here") is None
    c = scan_log("a 0xC00D3704 b 0xC00D3704 c 0x800703E3 d 0x12345678")
    assert c == {"0xC00D3704": 2, "0x800703E3": 1}


def test_frozen_for_reports_how_long_the_picture_has_been_identical():
    import numpy as np
    from core.frame_integrity import FrameIntegrity

    fi = FrameIntegrity(freeze_sec=10.0)
    rng = np.random.default_rng(0)
    frame = (np.tile(np.linspace(40, 200, 64), (48, 1))[:, :, None].repeat(3, 2) + rng.integers(0, 3, (48, 64, 3))).astype(np.uint8)
    assert fi.frozen_for(0.0) == 0.0
    fi.check(frame, 1.0)
    assert fi.frozen_for(1.0) == 0.0
    fi.check(frame.copy(), 1.5)
    assert fi.frozen_for(2.5) == pytest.approx(1.0)
    fi.check(frame + 1, 3.0)                       # the picture moved again
    assert fi.frozen_for(3.0) == 0.0
