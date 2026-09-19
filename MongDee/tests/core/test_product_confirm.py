"""One-off product matches (background clutter) must not be reported; a product seen again must."""
import numpy as np

from core.product_confirm import ProductConfirmer
from core.vision import CameraWorker


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

    def identify(self, crop):
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
    assert [d["class_name"] for d in detections] == ["WATER"] and len(boxes) == 1
