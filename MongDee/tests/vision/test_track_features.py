import numpy as np

from vision.attributes.extractor import AttributeExtractor
from vision.reid.config import ReIDConfig
from vision.reid.extractor import ReIDExtractor
from vision.track_features import FeatureStoreConfig, TrackFeatureStore
from vision.tracking.kalman import KalmanBoxFilter
from vision.tracking.track import Track


def _store(**cfg):
    return TrackFeatureStore(
        ReIDExtractor(ReIDConfig(backend="color")),
        AttributeExtractor(),
        FeatureStoreConfig(**cfg),
    )


def _track(track_id, bbox):
    return Track(track_id, bbox, 0.9, 0.0, KalmanBoxFilter(), n_init=1, max_age=30)


def _person_frame(w=400, h=600, box=(150, 100, 260, 480), shirt=(190, 70, 40)):
    img = np.full((h, w, 3), 128, np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = shirt
    img[y1:(y1 + y2) // 2, x1:x2] = shirt
    img[(y1 + y2) // 2:y2, x1:x2] = (20, 20, 20)
    img[y1:y2:4, x1:x2] = (shirt[0] // 2, shirt[1] // 2, shirt[2] // 2)  # texture for blur check
    return img


def test_observe_accumulates_and_build_summary():
    store = _store(observe_interval_sec=0.0)
    box = [150, 100, 260, 480]
    track = _track(1, box)
    frame = _person_frame(box=tuple(box))
    for i in range(5):
        store.observe("CAM01", track, frame, timestamp=float(i))

    summary = store.build_summary("CAM01", 1)
    assert summary is not None
    assert summary.camera_id == "CAM01"
    assert summary.embedding is not None
    assert np.linalg.norm(summary.embedding) > 0.99
    assert summary.embedding_samples >= 1
    assert summary.first_seen == 0.0 and summary.last_seen == 4.0
    assert summary.attributes.shirt_color.value in ("blue", "unknown")


def test_observe_is_rate_limited():
    store = _store(observe_interval_sec=1.0)
    box = [150, 100, 260, 480]
    track = _track(1, box)
    frame = _person_frame(box=tuple(box))
    store.observe("CAM01", track, frame, 0.0)
    store.observe("CAM01", track, frame, 0.3)  # too soon
    store.observe("CAM01", track, frame, 0.6)  # too soon
    summary = store.build_summary("CAM01", 1)
    assert summary.embedding_samples == 1


def test_low_quality_crop_contributes_no_embedding():
    store = _store(observe_interval_sec=0.0)
    track = _track(1, [10, 10, 30, 40])  # 30px tall -> fails quality gate
    flat = np.full((200, 200, 3), 120, np.uint8)
    store.observe("CAM01", track, flat, 0.0)
    summary = store.build_summary("CAM01", 1)
    assert summary.embedding is None
    assert summary.embedding_samples == 0


def test_pop_removes_entry():
    store = _store(observe_interval_sec=0.0)
    box = [150, 100, 260, 480]
    store.observe("CAM01", _track(1, box), _person_frame(box=tuple(box)), 0.0)
    assert store.track_ids_for_camera("CAM01") == {1}
    summary = store.pop("CAM01", 1)
    assert summary is not None
    assert store.track_ids_for_camera("CAM01") == set()
    assert store.pop("CAM01", 1) is None
