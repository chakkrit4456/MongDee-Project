import numpy as np
import pytest

from vision.frame import FrameProcessor, ProcessedFrame


def _img(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_downscale_preserves_aspect_ratio():
    fp = FrameProcessor(max_width=960, max_height=540)
    out = fp.process(_img(1920, 1080))
    assert out.image.shape[:2] == (540, 960)
    assert out.scale == pytest.approx(0.5, abs=1e-6)
    assert out.original_size == (1920, 1080)


def test_downscale_limited_by_width():
    fp = FrameProcessor(max_width=960, max_height=540)
    out = fp.process(_img(1920, 200))  # very wide
    assert out.image.shape[1] == 960
    assert out.scale == pytest.approx(0.5, abs=1e-3)


def test_small_frame_passes_through_untouched():
    fp = FrameProcessor(max_width=960, max_height=540)
    src = _img(640, 480)
    out = fp.process(src)
    assert out.scale == 1.0
    assert out.image is src  # no copy, no resize
    assert out.original_size == (640, 480)


def test_to_original_round_trips_a_box():
    fp = FrameProcessor(max_width=960, max_height=540)
    out = fp.process(_img(1920, 1080))  # scale 0.5
    # a box in processed coords -> original coords
    assert out.to_original((100, 50, 200, 150)) == pytest.approx([200, 100, 400, 300])


def test_to_original_clamps_to_frame_bounds():
    fp = FrameProcessor(max_width=960, max_height=540)
    out = fp.process(_img(1920, 1080))
    mapped = out.to_original((-10, -10, 100000, 100000))
    assert mapped == [0.0, 0.0, 1920.0, 1080.0]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        FrameProcessor(max_width=0, max_height=540)


def test_processed_frame_to_original_with_unit_scale():
    pf = ProcessedFrame(image=_img(640, 480), scale=1.0, original_size=(640, 480))
    assert pf.to_original((10, 20, 30, 40)) == [10, 20, 30, 40]
