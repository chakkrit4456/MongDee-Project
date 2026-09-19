from __future__ import annotations

import time

import numpy as np
import pytest

import camera.gateway as gateway_mod
from camera.base import CameraConfig, CameraSource, Frame
from camera.gateway import CameraGateway
from vision.attributes import AttributeExtractor
from vision.config import DetectionConfig
from vision.detection.detector import PersonDetector
from vision.events import PERSON_NEW, PERSON_REIDENTIFIED
from vision.identity import GlobalIdentityManager, IdentityConfig
from vision.pipeline import DetectionPipeline, PipelineResult
from vision.reid import ReIDConfig, ReIDExtractor
from vision.track_features import FeatureStoreConfig, TrackFeatureStore
from vision.tracking import MultiCameraTracker, TrackingConfig
from tests.vision.conftest import FakeBox, FakeYOLO


def _wait_until(predicate, timeout=3.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeFrameSource:
    def __init__(self):
        self._frames: dict[str, Frame] = {}

    def set_frame(self, camera_id, image=None, frame_index=0, timestamp=None):
        img = image if image is not None else np.zeros((540, 960, 3), np.uint8)
        self._frames[camera_id] = Frame(camera_id, img, timestamp or time.time(), frame_index)

    def camera_ids(self):
        return list(self._frames.keys())

    def latest_frame(self, camera_id):
        return self._frames.get(camera_id)


def _detector(boxes=None, **cfg):
    return PersonDetector(DetectionConfig(device="cpu", **cfg), model=FakeYOLO(boxes=boxes))


def test_detections_reach_callback_and_latest():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=1)
    detector = _detector(boxes=[FakeBox(0, 0.9, [10, 20, 30, 40])])

    seen = []
    pipe = DetectionPipeline(source, detector, on_detections=lambda cid, dets, frame: seen.append((cid, dets, frame)))
    pipe.start()
    try:
        assert _wait_until(lambda: len(seen) >= 1)
        cid, dets, frame = seen[0]
        assert cid == "CAM01"
        assert len(dets) == 1 and dets[0].camera_id == "CAM01"
        assert frame.frame_index == 1
        assert len(pipe.latest_detections("CAM01")) == 1
    finally:
        pipe.stop()


def test_same_frame_index_not_reprocessed():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=7)
    detector = _detector(boxes=[])
    pipe = DetectionPipeline(source, detector)
    pipe.start()
    try:
        assert _wait_until(lambda: len(detector.model.predict_calls) >= 1)
        count = len(detector.model.predict_calls)
        time.sleep(0.3)
        assert len(detector.model.predict_calls) == count  # frame_index unchanged -> skipped

        source.set_frame("CAM01", frame_index=8)
        assert _wait_until(lambda: len(detector.model.predict_calls) > count)
    finally:
        pipe.stop()


def test_pacing_limits_processing_rate():
    source = FakeFrameSource()
    detector = _detector(boxes=[], target_fps_per_camera=5.0)  # min interval 0.2s
    pipe = DetectionPipeline(source, detector)
    pipe.start()
    try:
        start = time.monotonic()
        idx = 0
        while time.monotonic() - start < 0.6:
            idx += 1
            source.set_frame("CAM01", frame_index=idx)
            time.sleep(0.01)
        time.sleep(0.05)
    finally:
        pipe.stop()
    # ~5 fps for ~0.6s => about 3-4 detections; must be far below the ~60 frames pushed
    assert 1 <= len(detector.model.predict_calls) <= 10


def test_multiple_cameras_all_processed():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=1)
    source.set_frame("CAM02", frame_index=1)
    source.set_frame("CAM03", frame_index=1)
    detector = _detector(boxes=[FakeBox(0, 0.9, [1, 1, 2, 2])])
    pipe = DetectionPipeline(source, detector)
    pipe.start()
    try:
        assert _wait_until(
            lambda: all(len(pipe.latest_detections(c)) == 1 for c in ("CAM01", "CAM02", "CAM03"))
        )
        stats = pipe.all_stats()
        assert set(stats) == {"CAM01", "CAM02", "CAM03"}
        assert all(s.frames_processed >= 1 for s in stats.values())
    finally:
        pipe.stop()


def test_stats_report_detection_count_and_latency():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=1)
    detector = _detector(boxes=[FakeBox(0, 0.9, [1, 1, 2, 2]), FakeBox(0, 0.8, [3, 3, 4, 4])])
    pipe = DetectionPipeline(source, detector)
    pipe.start()
    try:
        assert _wait_until(lambda: pipe.stats("CAM01") is not None and pipe.stats("CAM01").frames_processed >= 1)
        s = pipe.stats("CAM01")
        assert s.last_detection_count == 2
        assert s.last_latency_sec >= 0.0
    finally:
        pipe.stop()


def test_callback_exception_does_not_kill_pipeline():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=1)
    detector = _detector(boxes=[FakeBox(0, 0.9, [1, 1, 2, 2])])

    def boom(cid, dets, frame):
        raise RuntimeError("callback broke")

    pipe = DetectionPipeline(source, detector, on_detections=boom)
    pipe.start()
    try:
        assert _wait_until(lambda: detector.model.predict_calls and pipe.stats("CAM01") is not None)
        # pipeline keeps going despite the raising callback
        count = len(detector.model.predict_calls)
        source.set_frame("CAM01", frame_index=2)
        assert _wait_until(lambda: len(detector.model.predict_calls) > count)
    finally:
        pipe.stop()


def test_pipeline_with_tracker_assigns_track_ids_and_emits_result():
    source = FakeFrameSource()
    detector = _detector(boxes=[FakeBox(0, 0.9, [100, 100, 140, 200])], target_fps_per_camera=1000.0)
    tracker = MultiCameraTracker(TrackingConfig(n_init=1))

    results: list[PipelineResult] = []
    pipe = DetectionPipeline(source, detector, tracker=tracker, on_result=results.append)
    pipe.start()
    try:
        for idx in range(1, 6):
            source.set_frame("CAM01", frame_index=idx)
            time.sleep(0.03)
        assert _wait_until(lambda: len(pipe.latest_tracks("CAM01")) == 1)
        track = pipe.latest_tracks("CAM01")[0]
        assert track.track_id == 1
        # detections carried a local_track_id, and results include tracks
        assert results and results[-1].tracks
        assert results[-1].detections[0].local_track_id == 1
        assert pipe.stats("CAM01").last_track_count == 1
    finally:
        pipe.stop()


def _person_frame(box, shirt=(190, 70, 40)):
    img = np.full((600, 500, 3), 128, np.uint8)
    x1, y1, x2, y2 = box
    mid = (y1 + y2) // 2
    img[y1:mid, x1:x2] = shirt
    img[mid:y2, x1:x2] = (20, 20, 20)
    img[y1:y2:4, x1:x2] = (shirt[0] // 2, shirt[1] // 2, shirt[2] // 2)
    return img


def test_full_person_pipeline_reidentifies_across_cameras():
    box = [150, 90, 250, 470]
    source = FakeFrameSource()
    detector = _detector(boxes=[FakeBox(0, 0.92, box)], target_fps_per_camera=1000.0)
    tracker = MultiCameraTracker(TrackingConfig(n_init=2, track_buffer=2))
    store = TrackFeatureStore(
        ReIDExtractor(ReIDConfig(backend="color")),
        AttributeExtractor(),
        FeatureStoreConfig(observe_interval_sec=0.0, min_samples_to_flush=1, track_end_timeout_sec=0.4),
    )
    identity = GlobalIdentityManager(IdentityConfig(match_threshold=0.6, uncertain_threshold=0.45))

    events = []
    pipe = DetectionPipeline(
        source, detector, tracker=tracker, feature_store=store, identity_manager=identity,
        on_person_event=events.append,
    )
    pipe.start()
    try:
        frame_a = _person_frame(box)
        for idx in range(1, 8):
            source._frames["CAM01"] = _frame_with_image("CAM01", frame_a, idx)
            time.sleep(0.03)
        assert _wait_until(lambda: any(e.event_type == PERSON_NEW for e in events), timeout=3.0)

        frame_b = _person_frame(box)
        for idx in range(1, 8):
            source._frames["CAM02"] = _frame_with_image("CAM02", frame_b, idx)
            time.sleep(0.03)
        assert _wait_until(
            lambda: any(e.event_type == PERSON_REIDENTIFIED for e in events), timeout=3.0
        )
    finally:
        pipe.stop()

    # the same-looking person seen by two cameras is ONE global id (section 16/47)
    assert identity.unique_count() == 1


def _frame_with_image(camera_id, image, frame_index):
    from camera.base import Frame

    return Frame(camera_id=camera_id, image=image, timestamp=time.time(), frame_index=frame_index)


def test_pipeline_rejects_feature_store_without_tracker():
    source = FakeFrameSource()
    store = TrackFeatureStore(ReIDExtractor(ReIDConfig(backend="color")), AttributeExtractor())
    with pytest.raises(ValueError, match="feature_store requires a tracker"):
        DetectionPipeline(source, _detector(), feature_store=store)


def test_stop_is_clean_and_idempotent():
    source = FakeFrameSource()
    source.set_frame("CAM01", frame_index=1)
    pipe = DetectionPipeline(source, _detector(boxes=[]))
    pipe.start()
    assert _wait_until(lambda: pipe.stats("CAM01") is not None)
    pipe.stop()
    pipe.stop()  # must not raise


# --- integration with the real CameraGateway -------------------------------


class _StillSource(CameraSource):
    """Always-connected camera that returns a constant frame."""

    def open(self):
        pass

    def read(self):
        return np.full((480, 640, 3), 100, np.uint8)

    def release(self):
        pass


def test_pipeline_consumes_real_camera_gateway(monkeypatch):
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda config: _StillSource(config))
    gateway = CameraGateway()
    gateway.add_camera(CameraConfig.from_dict({"id": "CAM01", "protocol": "usb", "processing_fps": 1000}))

    detector = _detector(boxes=[FakeBox(0, 0.95, [5, 5, 25, 55])])
    pipe = DetectionPipeline(gateway, detector)
    pipe.start()
    try:
        assert _wait_until(lambda: len(pipe.latest_detections("CAM01")) == 1, timeout=5.0)
        assert pipe.latest_detections("CAM01")[0].confidence == pytest.approx(0.95)
    finally:
        pipe.stop()
        gateway.stop_all()
