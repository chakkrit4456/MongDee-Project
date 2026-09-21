"""One-off product matches (background clutter) must not be reported; a product seen again must."""
import numpy as np

from core.product_confirm import (
    PRODUCT_FAR_MIN_CONFIDENCE,
    PRODUCT_FULL_TRUST_PX,
    ProductConfirmer,
    product_confirm_hits_for,
    product_min_confidence,
)
from core.vision import CameraWorker


def test_full_trust_size_box_is_never_scaled():
    assert product_min_confidence(PRODUCT_FULL_TRUST_PX, 0.45) == 0.45
    assert product_min_confidence(200.0, 0.45) == 0.45
    assert product_min_confidence(None, 0.45) == 0.45
    assert product_confirm_hits_for(PRODUCT_FULL_TRUST_PX, 2) == 2
    assert product_confirm_hits_for(200.0, 2) == 2


def test_small_box_needs_higher_confidence_and_more_hits():
    small = product_min_confidence(24.0, 0.45)      # smallest attempted size
    mid = product_min_confidence(52.0, 0.45)        # halfway between far/full-trust
    assert small == PRODUCT_FAR_MIN_CONFIDENCE
    assert 0.45 < mid < small
    assert product_confirm_hits_for(24.0, 2) > product_confirm_hits_for(52.0, 2) >= 2


def test_scaling_never_lowers_a_base_threshold_already_above_the_far_floor():
    # A caller whose own base floor already exceeds PRODUCT_FAR_MIN_CONFIDENCE must not be
    # weakened by this scaling -- it can only ever raise the bar, never lower it.
    assert product_min_confidence(24.0, 0.9) == 0.9


def test_coasting_keeps_a_confirmed_products_last_box_through_a_brief_miss():
    """A product confirmed once must not vanish on the very next pass just because that one
    pass's embedding match happened to fall short (motion blur, a hand crossing it, ...) -- see
    ProductConfirmer.coasting's own docstring for why this mirrors person-track coasting."""
    pc = ProductConfirmer(min_hits=2, coast_sec=0.6)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    assert pc.confirm("WATER", [11, 10, 61, 90], now=0.4, score=0.9) is True       # now confirmed
    coasting = pc.coasting(now=0.7, exclude=set())          # a miss on the very next pass
    assert coasting == [("WATER", (11.0, 10.0, 61.0, 90.0), 0.9)]


def test_coasting_excludes_keys_reconfirmed_this_pass():
    pc = ProductConfirmer(min_hits=2, coast_sec=0.6)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    pc.confirm("WATER", [11, 10, 61, 90], now=0.4)
    assert pc.coasting(now=0.5, exclude={"WATER"}) == []


def test_coasting_expires_after_coast_sec():
    pc = ProductConfirmer(min_hits=1, coast_sec=0.6)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    assert pc.coasting(now=0.5) != []
    assert pc.coasting(now=0.7) == []       # past coast_sec -- must not coast forever


def test_forget_stale_eventually_drops_coasting_state_too():
    pc = ProductConfirmer(min_hits=1, window_sec=1.0, coast_sec=0.6)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    pc.forget_stale(now=5.0)
    assert pc.coasting(now=5.0) == []


def test_first_sighting_is_not_confirmed_second_overlapping_one_is():
    pc = ProductConfirmer(min_hits=2)
    assert pc.confirm("WATER", [10, 10, 60, 90], now=0.0) is False
    assert pc.confirm("WATER", [12, 11, 62, 91], now=0.4) is True


def test_a_match_elsewhere_or_of_another_product_does_not_confirm():
    pc = ProductConfirmer(min_hits=2)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    assert pc.confirm("WATER", [200, 10, 250, 90], now=0.4) is False      # different place
    assert pc.confirm("COLA", [10, 10, 60, 90], now=0.8) is False          # different product


def test_old_sightings_expire():
    pc = ProductConfirmer(min_hits=2, window_sec=4.0)
    pc.confirm("WATER", [10, 10, 60, 90], now=0.0)
    assert pc.confirm("WATER", [10, 10, 60, 90], now=10.0) is False


def test_min_hits_one_disables_the_gate():
    assert ProductConfirmer(min_hits=1).confirm("WATER", [0, 0, 10, 10], now=0.0) is True


class _FakeModel:
    names = {0: "person"}


class _Recognizer:
    def __init__(self):
        self.answers = []

    def has_any_gallery(self):
        return True

    def identify(self, crop, floor=None, margin=None):
        return self.answers.pop(0)


class _Proposer:
    def propose(self, frame, exclude_boxes=None, max_regions=2):
        return [[10, 10, 80, 90]]


def test_worker_only_reports_a_product_after_it_is_seen_in_two_passes():
    worker = CameraWorker(camera_id="T", device=0, model=_FakeModel(), allowed_classes=[])
    worker.recognizer = _Recognizer()
    worker._proposer = _Proposer()
    frame = np.full((120, 120, 3), 90, dtype=np.uint8)

    worker.recognizer.answers = [("WATER", 0.9)]
    boxes = []
    assert worker._run_custom_recognition(frame, [], boxes) == []          # one-off: dropped
    assert boxes == []

    worker.recognizer.answers = [("WATER", 0.9)]
    detections = worker._run_custom_recognition(frame, [], boxes)          # seen again: confirmed
    assert [d["class_name"] for d in detections] == ["WATER"]


def test_a_confirmed_product_coasts_through_one_missed_pass():
    """The exact flicker this coasting mechanism exists to prevent: a product already confirmed
    and on screen must not vanish for a single pass where the embedding match momentarily misses
    (here, the proposer finds nothing at all that pass -- the strongest form of a "miss")."""
    worker = CameraWorker(camera_id="T", device=0, model=_FakeModel(), allowed_classes=[])
    worker.recognizer = _Recognizer()
    worker._proposer = _Proposer()
    frame = np.full((120, 120, 3), 90, dtype=np.uint8)

    worker.recognizer.answers = [("WATER", 0.9)]
    worker._run_custom_recognition(frame, [], [])                          # pass 1: one-off, dropped
    worker.recognizer.answers = [("WATER", 0.9)]
    first = worker._run_custom_recognition(frame, [], [])                  # pass 2: confirmed
    assert [d["class_name"] for d in first] == ["WATER"]

    class _EmptyProposer:
        def propose(self, frame, exclude_boxes=None, max_regions=2):
            return []

    worker._proposer = _EmptyProposer()                                    # pass 3: nothing detected at all
    boxes = []
    second = worker._run_custom_recognition(frame, [], boxes)
    assert [d["class_name"] for d in second] == ["WATER"]                  # still shown -- coasting
    assert second[0]["conf"] == 0.9                                        # the real last score, not 0.0
    assert len(boxes) == 1                                                 # the coasted box was drawn
