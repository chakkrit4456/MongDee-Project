import pytest

import camera.hls as hls_mod
from camera.base import CameraConfig, CameraConnectionError
from camera.hls import HLSCamera
from tests.camera.conftest import FakeVideoCapture


def _config(**overrides):
    base = {"id": "CAM05", "protocol": "hls", "url": "https://example.com/stream/index.m3u8"}
    base.update(overrides)
    return CameraConfig.from_dict(base)


def test_open_and_read(monkeypatch):
    monkeypatch.setattr(hls_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend))
    cam = HLSCamera(_config())
    cam.open()
    frame = cam.read()
    assert frame is not None
    cam.release()


def test_open_failure_raises(monkeypatch):
    monkeypatch.setattr(hls_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend, opens=False))
    cam = HLSCamera(_config())
    with pytest.raises(CameraConnectionError, match="could not open playlist"):
        cam.open()


def test_missing_url_raises_at_construction():
    cfg = CameraConfig.from_dict({"id": "CAM05", "protocol": "hls"})
    with pytest.raises(ValueError, match="config.url is required"):
        HLSCamera(cfg)
