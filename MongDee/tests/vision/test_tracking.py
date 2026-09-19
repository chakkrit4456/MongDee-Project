import os

import numpy as np
import pytest

from vision.detection.detector import Detection
from vision.tracking.bytetrack import ByteTracker
from vision.tracking.config import TrackingConfig
from vision.tracking.kalman import KalmanBoxFilter, bbox_to_measurement, mean_to_bbox
from vision.tracking.matching import greedy_match, iou_matrix
from vision.tracking.multi_tracker import MultiCameraTracker
from vision.tracking.track import Track, TrackState


def _det(x1, y1, x2, y2, conf=0.9, cam="CAM01", ts=0.0):
    return Detection(bbox=[float(x1), float(y1), float(x2), float(y2)], confidence=conf, camera_id=cam, timestamp=ts)


# --- kalman ---------------------------------------------------------------


def test_bbox_measurement_round_trip():
    bbox = [100.0, 50.0, 140.0, 210.0]
    m = bbox_to_measurement(bbox)
    assert m[0] == pytest.approx(120.0)  # cx
    assert m[1] == pytest.approx(130.0)  # cy
    assert m[3] == pytest.approx(160.0)  # h
    assert mean_to_bbox(np.r_[m, np.zeros(4)]) == pytest.approx(bbox)


def test_kalman_predicts_motion_after_updates():
    kf = KalmanBoxFilter()
    mean, cov = kf.initiate(bbox_to_measurement([100, 100, 140, 200]))
    for step in range(1, 5):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, bbox_to_measurement([100 + step * 10, 100, 140 + step * 10, 200]))
    # velocity learned -> next predict keeps moving right
    pred_mean, _ = kf.predict(mean, cov)
    assert pred_mean[0] > mean[0]


# --- matching -----------------------------------------------------------


def test_iou_matrix_values():
    iou = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(25 / 175)
    assert iou[0, 2] == pytest.approx(0.0)


def test_greedy_match_pairs_best_first():
    iou = np.array([[0.9, 0.2], [0.3, 0.8]])
    matches, un_r, un_c = greedy_match(iou, iou_threshold=0.3)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert un_r == [] and un_c == []


def test_greedy_match_respects_threshold():
    iou = np.array([[0.1, 0.15]])
    matches, un_r, un_c = greedy_match(iou, iou_threshold=0.3)
    assert matches == []
    assert un_r == [0] and un_c == [0, 1]


def test_greedy_match_empty():
    assert greedy_match(np.zeros((0, 3)), 0.3) == ([], [], [0, 1, 2])


# --- Track lifecycle --------------------------------------------------


def test_track_confirms_after_n_init_hits():
    kf = KalmanBoxFilter()
    t = Track(1, [0, 0, 10, 20], 0.9, 0.0, kf, n_init=3, max_age=5)
    assert t.is_tentative
    t.predict(); t.update([0, 0, 10, 20], 0.9, 1.0)
    assert t.is_tentative  # hits=2
    t.predict(); t.update([0, 0, 10, 20], 0.9, 2.0)
    assert t.is_confirmed  # hits=3


def test_track_tentative_dropped_on_miss():
    kf = KalmanBoxFilter()
    t = Track(1, [0, 0, 10, 20], 0.9, 0.0, kf, n_init=3, max_age=5)
    t.predict(); t.mark_missed()
    assert t.is_removed


def test_confirmed_track_goes_lost_then_removed():
    kf = KalmanBoxFilter()
    t = Track(1, [0, 0, 10, 20], 0.9, 0.0, kf, n_init=1, max_age=2)
    assert t.is_confirmed
    t.predict(); t.mark_missed()
    assert t.is_lost
    t.predict(); t.mark_missed()
    assert t.is_lost  # time_since_update=2, still <= max_age
    t.predict(); t.mark_missed()
    assert t.is_removed  # time_since_update=3 > max_age


# --- ByteTracker --------------------------------------------------------


def test_single_person_keeps_one_id():
    tracker = ByteTracker(TrackingConfig(n_init=2))
    ids = set()
    for i in range(8):
        x = 100 + i * 12
        tracks = tracker.update([_det(x, 100, x + 40, 200)], timestamp=float(i))
        ids.update(t.track_id for t in tracks)
    assert ids == {1}


def test_detection_gets_local_track_id_assigned():
    tracker = ByteTracker(TrackingConfig(n_init=1))
    d = _det(100, 100, 140, 200)
    tracker.update([d], 0.0)
    assert d.local_track_id == 1


def test_short_occlusion_recovers_same_id():
    cfg = TrackingConfig(n_init=2, track_buffer=15)
    tracker = ByteTracker(cfg)
    for i in range(4):
        tracker.update([_det(100, 100, 140, 200)], float(i))
    tid = tracker.active_tracks()[0].track_id

    for i in range(4, 8):  # occluded — no detections
        tracker.update([], float(i))
    assert tracker.active_tracks() == []  # lost, not "active"

    tracks = tracker.update([_det(104, 100, 144, 200)], 8.0)
    assert len(tracks) == 1
    assert tracks[0].track_id == tid


def test_long_absence_gets_new_id():
    cfg = TrackingConfig(n_init=2, track_buffer=3)
    tracker = ByteTracker(cfg)
    for i in range(4):
        tracker.update([_det(100, 100, 140, 200)], float(i))
    first_id = tracker.active_tracks()[0].track_id

    for i in range(4, 12):
        tracker.update([], float(i))

    tracker.update([_det(100, 100, 140, 200)], 12.0)
    live = tracker.tracks
    assert len(live) == 1
    assert live[0].track_id != first_id  # old track aged out, this is a fresh id


def test_low_confidence_detection_recovers_track():
    cfg = TrackingConfig(n_init=2, track_high_thresh=0.5, track_low_thresh=0.1, match_thresh=0.3, match_thresh_low=0.3)
    tracker = ByteTracker(cfg)
    for i in range(3):
        tracker.update([_det(100, 100, 140, 200, conf=0.9)], float(i))
    tid = tracker.active_tracks()[0].track_id

    low = _det(101, 100, 141, 200, conf=0.3)
    tracks = tracker.update([low], 3.0)
    assert tracks and tracks[0].track_id == tid
    assert low.local_track_id == tid


def test_new_track_needs_confidence_above_threshold():
    cfg = TrackingConfig(new_track_thresh=0.6)
    tracker = ByteTracker(cfg)
    tracker.update([_det(100, 100, 140, 200, conf=0.55)], 0.0)
    assert tracker.tracks == []
    tracker.update([_det(100, 100, 140, 200, conf=0.7)], 1.0)
    assert len(tracker.tracks) == 1


def test_two_people_keep_distinct_ids():
    tracker = ByteTracker(TrackingConfig(n_init=2))
    seen = set()
    for i in range(6):
        a_x = 100 + i * 5
        b_x = 400 - i * 5
        tracks = tracker.update(
            [_det(a_x, 100, a_x + 40, 200), _det(b_x, 100, b_x + 40, 200)], float(i)
        )
        seen.update(t.track_id for t in tracks)
    assert seen == {1, 2}


# --- MultiCameraTracker ----------------------------------------------


def test_multi_camera_ids_are_independent():
    mct = MultiCameraTracker(TrackingConfig(n_init=1))
    d1 = _det(100, 100, 140, 200, cam="CAM01")
    d2 = _det(500, 300, 540, 400, cam="CAM02")
    mct.update("CAM01", [d1], 0.0)
    mct.update("CAM02", [d2], 0.0)
    assert d1.local_track_id == 1
    assert d2.local_track_id == 1  # separate id space per camera
    assert set(mct.camera_ids()) == {"CAM01", "CAM02"}


def test_multi_camera_remove():
    mct = MultiCameraTracker(TrackingConfig(n_init=1))
    mct.update("CAM01", [_det(100, 100, 140, 200)], 0.0)
    mct.remove_camera("CAM01")
    assert mct.camera_ids() == []


@pytest.mark.slow
def test_real_detection_plus_tracking_is_stable_on_static_scene():
    """Opt-in (`pytest -m slow`): real YOLO on a real photo of people
    (ultralytics' bundled bus.jpg), fed as a static scene. The confirmed
    track IDs must not churn frame to frame."""
    import ultralytics
    import cv2

    from vision.config import DetectionConfig
    from vision.detection.detector import PersonDetector

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    assert img is not None

    detector = PersonDetector(DetectionConfig(device="cpu"))
    tracker = MultiCameraTracker(TrackingConfig(n_init=3))

    id_sets = []
    for i in range(8):
        dets = detector.detect(img, "CAM01", timestamp=i * 0.1)
        tracks = tracker.update("CAM01", dets, timestamp=i * 0.1)
        id_sets.append(frozenset(t.track_id for t in tracks))

    assert len(dets) >= 2  # bus.jpg has several people
    # once tracks confirm (by frame ~3) the set of active ids is constant
    assert len(set(id_sets[4:])) == 1
    assert len(id_sets[-1]) >= 2
