import pytest

import camera.rtsp as rtsp_mod
from camera.base import CameraConfig, CameraConnectionError
from camera.rtsp import RTSPCamera
from tests.camera.conftest import FakeVideoCapture


def _config(**overrides):
    base = {"id": "CAM02", "protocol": "rtsp", "url": "rtsp://admin:secret@192.168.1.64:554/ch1"}
    base.update(overrides)
    return CameraConfig.from_dict(base)


def test_open_and_read(monkeypatch):
    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend))
    cam = RTSPCamera(_config())
    cam.open()
    frame = cam.read()
    assert frame is not None
    cam.release()


def test_open_failure_message_redacts_credentials(monkeypatch):
    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend, opens=False))
    cam = RTSPCamera(_config())
    with pytest.raises(CameraConnectionError) as exc_info:
        cam.open()
    message = str(exc_info.value)
    assert "secret" not in message
    assert "admin" not in message
    assert "192.168.1.64" in message  # host itself is fine to log


def test_read_failure_message_redacts_credentials(monkeypatch):
    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend, frames=[]))
    cam = RTSPCamera(_config())
    cam.open()
    with pytest.raises(CameraConnectionError) as exc_info:
        cam.read()
    assert "secret" not in str(exc_info.value)


def test_missing_url_raises_at_construction():
    cfg = CameraConfig.from_dict({"id": "CAM02", "protocol": "rtsp"})
    with pytest.raises(ValueError, match="config.url is required"):
        RTSPCamera(cfg)


def test_url_override_used_over_config_url(monkeypatch):
    captured = {}

    def fake_ctor(url, backend):
        captured["url"] = url
        return FakeVideoCapture(url, backend)

    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", fake_ctor)
    cam = RTSPCamera(_config(), url="rtsp://resolved-by-onvif/stream1")
    cam.open()
    assert captured["url"] == "rtsp://resolved-by-onvif/stream1"
