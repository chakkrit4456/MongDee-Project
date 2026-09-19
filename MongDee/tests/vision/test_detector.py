import os

import numpy as np
import pytest

from vision.config import DetectionConfig
from vision.detection.detector import Detection, PersonDetector, resolve_device
from tests.vision.conftest import FakeBox, FakeYOLO


def _detector(boxes=None, **cfg_overrides):
    cfg = DetectionConfig(device="cpu", **cfg_overrides)
    return PersonDetector(cfg, model=FakeYOLO(boxes=boxes))


def test_returns_person_detections_with_expected_fields():
    det = _detector(boxes=[FakeBox(0, 0.9, [10, 20, 30, 60])])
    out = det.detect(np.zeros((540, 960, 3), np.uint8), camera_id="CAM01", timestamp=123.0)
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.camera_id == "CAM01"
    assert d.timestamp == 123.0
    assert d.class_name == "person"
    assert d.confidence == pytest.approx(0.9)
    assert d.local_track_id is None
    assert d.bbox == pytest.approx([10, 20, 30, 60])


def test_non_person_classes_are_filtered_out():
    det = _detector(boxes=[FakeBox(0, 0.9, [1, 1, 2, 2]), FakeBox(2, 0.99, [3, 3, 4, 4])])  # car
    out = det.detect(np.zeros((540, 960, 3), np.uint8))
    assert len(out) == 1
    assert out[0].bbox == pytest.approx([1, 1, 2, 2])


def test_low_confidence_boxes_dropped():
    det = _detector(boxes=[FakeBox(0, 0.10, [1, 1, 2, 2])], confidence_threshold=0.35)
    assert det.detect(np.zeros((540, 960, 3), np.uint8)) == []


def test_boxes_mapped_back_to_original_resolution():
    # 1920x1080 frame, processed down to 960x540 -> scale 0.5, inverse 2.0
    det = _detector(boxes=[FakeBox(0, 0.8, [100, 50, 200, 150])])
    out = det.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert out[0].bbox == pytest.approx([200, 100, 400, 300])
    # and the model actually received the downscaled frame
    call = det.model.predict_calls[-1]
    assert call["source_shape"][:2] == (540, 960)
    assert call["classes"] == [0]


def test_empty_result_returns_empty_list():
    det = _detector(boxes=[])
    assert det.detect(np.zeros((540, 960, 3), np.uint8)) == []


def test_detect_defaults_timestamp_to_now():
    det = _detector(boxes=[FakeBox(0, 0.8, [1, 1, 2, 2])])
    before = __import__("time").time()
    out = det.detect(np.zeros((540, 960, 3), np.uint8))
    assert out[0].timestamp >= before


def test_person_class_id_validated_against_model_names():
    cfg = DetectionConfig(device="cpu", person_class_id=2)  # "car" in the fake names
    with pytest.raises(ValueError, match="not 'person'"):
        PersonDetector(cfg, model=FakeYOLO())


def test_warmup_runs_one_inference():
    det = _detector(boxes=[])
    det.warmup()
    assert len(det.model.predict_calls) == 1


def test_detection_geometry_helpers():
    d = Detection(bbox=[10, 20, 40, 60], confidence=0.9, camera_id="C", timestamp=0.0)
    assert d.width == 30
    assert d.height == 40
    assert d.center == (25, 40)


def test_resolve_device_explicit_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_returns_cpu_or_cuda():
    assert resolve_device("auto") in ("cpu", "cuda:0")


def test_config_from_dict_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown field"):
        DetectionConfig.from_dict({"confidence_threshold": 0.5, "bogus": 1})


def test_config_load_from_file(tmp_path):
    path = tmp_path / "det.json"
    path.write_text('{"detection": {"confidence_threshold": 0.5, "processing_width": 800}}', encoding="utf-8")
    cfg = DetectionConfig.load(path)
    assert cfg.confidence_threshold == 0.5
    assert cfg.processing_width == 800


@pytest.mark.slow
def test_real_yolo_model_loads_and_runs():
    """Opt-in (`pytest -m slow`): proves the ultralytics integration works
    end-to-end — real weights load, real inference runs, output parsing
    doesn't crash. Detection accuracy is YOLO's concern, not ours, so we
    only assert the call returns a well-formed list."""
    pytest.importorskip("ultralytics")
    if not os.path.exists("yolo11n.pt"):
        pytest.skip("yolo11n.pt not present")
    det = PersonDetector(DetectionConfig(device="cpu", model_path="yolo11n.pt"))
    det.warmup()
    out = det.detect(np.full((720, 1280, 3), 127, np.uint8), camera_id="CAM01")
    assert isinstance(out, list)
    for d in out:
        assert d.class_name == "person"
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.bbox) == 4
