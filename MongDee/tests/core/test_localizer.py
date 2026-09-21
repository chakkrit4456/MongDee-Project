"""Unit tests for core/localizer.py: the class-agnostic foreground proposer
and the human-region exclusion helper used to keep hands/arms out of both
product registration and runtime product recognition.
"""

from __future__ import annotations

import numpy as np

from core.localizer import ForegroundProposer, exclude_human_region


def _frame(color=(0, 0, 0), size=(240, 320)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


# --------------------------------------------------------- exclude_human_region
def test_no_mask_returns_box_unchanged():
    box = [10.0, 20.0, 110.0, 220.0]
    assert exclude_human_region(box, None) == box


def test_box_mostly_covered_by_human_mask_is_rejected():
    mask = np.ones((240, 320), dtype=bool)  # whole frame is "human"
    box = [10.0, 10.0, 100.0, 100.0]
    assert exclude_human_region(box, mask) is None


def test_box_with_no_human_overlap_is_unchanged():
    mask = np.zeros((240, 320), dtype=bool)
    box = [10.0, 10.0, 100.0, 100.0]
    assert exclude_human_region(box, mask) == [10.0, 10.0, 100.0, 100.0]


def test_partial_human_overlap_tightens_the_box_to_the_clean_remainder():
    mask = np.zeros((240, 320), dtype=bool)
    # Human region covers the left half of the candidate box -- a hand
    # reaching in from one side while holding the product on the other.
    mask[10:110, 10:60] = True
    box = [10.0, 10.0, 110.0, 110.0]
    refined = exclude_human_region(box, mask)
    assert refined is not None
    rx1, ry1, rx2, ry2 = refined
    # The refined box must not start inside the human-covered strip.
    assert rx1 >= 55  # a few px of slack for the >= REJECT/TRIM boundary math
    assert rx2 <= 110.0


def test_degenerate_box_outside_frame_is_rejected():
    mask = np.zeros((240, 320), dtype=bool)
    assert exclude_human_region([400.0, 400.0, 500.0, 500.0], mask) is None


# --------------------------------------------------------------- ForegroundProposer
def test_update_does_not_raise_and_propose_still_works():
    proposer = ForegroundProposer()
    # Feed a stable background for a while (as CameraWorker.run() now does
    # every captured frame) so the model actually learns "empty scene".
    background = _frame((30, 30, 30))
    for _ in range(15):
        proposer.update(background)

    # Now something appears -- a bright rectangle standing in for a product
    # held up to the camera.
    foreground = background.copy()
    foreground[80:180, 100:220] = (220, 220, 220)
    boxes = proposer.propose(foreground)
    assert isinstance(boxes, list)  # never raises; may be empty on this one pass


def test_propose_without_update_does_not_crash_on_first_call():
    # propose() must remain safe to call even if update() was never called
    # (e.g. a gallery that only just gained its first trained product).
    proposer = ForegroundProposer()
    boxes = proposer.propose(_frame())
    assert boxes == []
