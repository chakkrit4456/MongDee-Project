"""The FACE box must not blink: a face the detector misses for a pass stays on screen (following the person) for a
moment, a weak second look finds it again, and small detector jitter does not shake the box."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.face.detector import FaceBox
from core.face.service import FACE_HOLD_SEC, FaceService, FaceStabilizer

FRAME = np.full((240, 320, 3), 120, np.uint8)


def _track(tid=1, x=100.0, y=40.0):
    return {"track_id": tid, "bbox": [x, y, x + 60.0, y + 150.0]}


def _face(x, y, size=30.0, score=0.9):
    return FaceBox(x, y, x + size, y + size, score)


class _Detector:
    """Returns whatever the test queued for this pass ([] = the detector missed the face)."""

    def __init__(self):
        self.queue = []

    def detect(self, frame):
        return self.queue.pop(0) if self.queue else []


def test_a_missed_pass_does_not_make_the_face_box_disappear():
    det = _Detector()
    svc = FaceService(det, min_quality_for_shot=2.0)         # (no best-shot crops needed here)
    tracks = [_track()]
    shown = []
    t = 0.0
    for hit in [True, False, True, False, False, True, False]:
        t += 0.4
        det.queue = [[_face(115, 45)] if hit else []]
        svc.process(FRAME, tracks, now=t)
        shown.append(len(svc.stable_faces(tracks, now=t)))
    assert shown == [1] * 7                                  # never blinks off


def test_a_held_face_moves_with_the_person():
    det = _Detector()
    svc = FaceService(det, min_quality_for_shot=2.0)
    det.queue = [[_face(115, 45)]]
    svc.process(FRAME, [_track(x=100)], now=1.0)
    moved = [_track(x=160)]                                  # the person walked 60 px right; the detector misses
    svc.process(FRAME, moved, now=1.4)
    (face,) = svc.stable_faces(moved, now=1.4)
    assert abs(face.x1 - 175.0) < 2.0                        # 115 + 60


def test_the_face_box_goes_away_after_the_hold_time():
    det = _Detector()
    svc = FaceService(det, min_quality_for_shot=2.0)
    det.queue = [[_face(115, 45)]]
    svc.process(FRAME, [_track()], now=1.0)
    assert svc.stable_faces([_track()], now=1.0 + FACE_HOLD_SEC - 0.1)
    assert svc.stable_faces([_track()], now=1.0 + FACE_HOLD_SEC + 0.1) == []


def test_detector_jitter_is_smoothed():
    stab = FaceStabilizer()
    rng = np.random.default_rng(0)
    raw, smooth = [], []
    t = 0.0
    for _ in range(40):
        t += 0.3
        x = 100 + rng.normal(0, 4)
        stab.update([_face(x, 50)], [], t)
        (f,) = stab.visible([], t)
        raw.append(x)
        smooth.append(f.x1)
    assert np.std(smooth[5:]) < 0.75 * np.std(raw[5:])


def test_a_second_low_threshold_look_recovers_a_missed_face():
    class Recovering(_Detector):
        calls = 0

        def detect_in_region(self, frame, region, score_thr=0.35):
            Recovering.calls += 1
            return [_face(116, 46, score=0.4)]

    det = Recovering()
    svc = FaceService(det, min_quality_for_shot=2.0)
    det.queue = [[_face(115, 45)], []]
    svc.process(FRAME, [_track()], now=1.0)
    found = svc.process(FRAME, [_track()], now=1.5)          # the full-frame pass misses ...
    assert Recovering.calls == 1 and len(found) == 1          # ... the region search finds it


def test_an_unbound_box_does_not_stay_next_to_the_bound_one():
    stab = FaceStabilizer()
    f = _face(100, 50)
    stab.update([f], [], 1.0)                                # no track yet: kept under an anonymous key
    bound = _face(101, 50)
    bound.track_id = 1
    stab.update([bound], [_track()], 1.3)
    assert len(stab.visible([_track()], 1.3)) == 1


def test_camera_worker_publishes_the_steady_faces_before_the_slow_part_of_the_pass():
    from core.vision import CameraWorker

    class Model:
        names = {0: "person"}

    det = _Detector()
    worker = CameraWorker(camera_id="CAM-1", device=0, model=Model(), allowed_classes=[], face_detector=det)
    tracks = [_track()]
    det.queue = [[_face(115, 45)]]
    worker._face_service.process(FRAME, tracks)
    early = []
    worker._draw_stable_faces(tracks, early)                # what the early publish adds (previous pass's faces)
    assert len(early) == 1 and early[0][1].startswith("FACE")
    assert worker.get_face_count() == 1


_YUNET = Path(__file__).resolve().parent.parent.parent / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"


@pytest.mark.skipif(not _YUNET.exists(), reason="YuNet model not installed")
def test_region_search_on_a_blank_frame_finds_nothing_and_does_not_raise():
    from core.face.detector import YuNetFaceDetector
    det = YuNetFaceDetector(str(_YUNET), upscale=1.0)
    assert det.detect_in_region(FRAME, (100, 40, 200, 160)) == []
    assert det.detect_in_region(FRAME, (1000, 1000, 1010, 1010)) == []       # outside the frame
