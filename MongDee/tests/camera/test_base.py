import numpy as np
import pytest

from camera.base import CameraConfig, CameraProtocol, Frame, redact_url


def test_from_dict_minimal():
    cfg = CameraConfig.from_dict({"id": "CAM01", "protocol": "usb"})
    assert cfg.id == "CAM01"
    assert cfg.name == "CAM01"  # defaults to id
    assert cfg.protocol is CameraProtocol.USB
    assert cfg.enabled is True
    assert cfg.processing_fps == 10.0


def test_from_dict_protocol_case_insensitive():
    cfg = CameraConfig.from_dict({"id": "CAM01", "protocol": "RTSP", "url": "rtsp://x/y"})
    assert cfg.protocol is CameraProtocol.RTSP


def test_from_dict_missing_id():
    with pytest.raises(ValueError, match="missing required field 'id'"):
        CameraConfig.from_dict({"protocol": "usb"})


def test_from_dict_missing_protocol():
    with pytest.raises(ValueError, match="missing required field 'protocol'"):
        CameraConfig.from_dict({"id": "CAM01"})


def test_from_dict_unknown_protocol():
    with pytest.raises(ValueError, match="unknown protocol"):
        CameraConfig.from_dict({"id": "CAM01", "protocol": "carrier-pigeon"})


def test_from_dict_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown field"):
        CameraConfig.from_dict({"id": "CAM01", "protocol": "usb", "typo_field": 123})


def test_config_is_frozen():
    cfg = CameraConfig.from_dict({"id": "CAM01", "protocol": "usb"})
    with pytest.raises(Exception):
        cfg.id = "other"  # dataclasses.FrozenInstanceError, a subclass of AttributeError


@pytest.mark.parametrize(
    "url,expected",
    [
        ("rtsp://admin:secret@192.168.1.64:554/ch1", "rtsp://***@192.168.1.64:554/ch1"),
        ("rtsp://192.168.1.64:554/ch1", "rtsp://192.168.1.64:554/ch1"),
        ("http://192.168.1.66/video", "http://192.168.1.66/video"),
        ("http://user:pw@192.168.1.66/video", "http://***@192.168.1.66/video"),
        ("not-a-url", "not-a-url"),
    ],
)
def test_redact_url_never_leaks_credentials(url, expected):
    redacted = redact_url(url)
    assert redacted == expected
    if "@" in url:
        assert "secret" not in redacted
        assert "pw" not in redacted


def test_frame_dataclass():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    frame = Frame(camera_id="CAM01", image=image, timestamp=123.0, frame_index=0)
    assert frame.camera_id == "CAM01"
    assert frame.image.shape == (4, 4, 3)
