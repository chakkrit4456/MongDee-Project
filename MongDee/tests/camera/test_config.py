import json

import pytest

from camera.base import CameraProtocol
from camera.config import load_camera_configs


def _write(tmp_path, obj):
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_load_camera_configs_object_form(tmp_path):
    path = _write(tmp_path, {"cameras": [{"id": "CAM01", "protocol": "usb"}]})
    configs = load_camera_configs(path)
    assert len(configs) == 1
    assert configs[0].id == "CAM01"
    assert configs[0].protocol is CameraProtocol.USB


def test_load_camera_configs_bare_list_form(tmp_path):
    path = _write(tmp_path, [{"id": "CAM01", "protocol": "usb"}, {"id": "CAM02", "protocol": "rtsp", "url": "rtsp://h/1"}])
    configs = load_camera_configs(path)
    assert [c.id for c in configs] == ["CAM01", "CAM02"]


def test_load_camera_configs_example_file_is_valid():
    # the checked-in example must always parse cleanly — it's user-facing documentation
    configs = load_camera_configs("configs/cameras.example.json")
    assert {c.id for c in configs} == {"CAM01", "CAM02", "CAM03", "CAM04", "CAM05"}
    protocols = {c.id: c.protocol for c in configs}
    assert protocols["CAM01"] is CameraProtocol.USB
    assert protocols["CAM02"] is CameraProtocol.RTSP
    assert protocols["CAM03"] is CameraProtocol.ONVIF
    assert protocols["CAM04"] is CameraProtocol.HTTP
    assert protocols["CAM05"] is CameraProtocol.HLS


def test_load_camera_configs_duplicate_id_rejected(tmp_path):
    path = _write(tmp_path, [{"id": "CAM01", "protocol": "usb"}, {"id": "CAM01", "protocol": "usb"}])
    with pytest.raises(ValueError, match="duplicate camera id"):
        load_camera_configs(path)


def test_load_camera_configs_bad_json(tmp_path):
    path = tmp_path / "cameras.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_camera_configs(path)


def test_load_camera_configs_wrong_shape(tmp_path):
    path = _write(tmp_path, {"not_cameras": []})
    with pytest.raises(ValueError, match="expected a top-level 'cameras' list"):
        load_camera_configs(path)


def test_load_camera_configs_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        load_camera_configs(tmp_path / "does_not_exist.json")
