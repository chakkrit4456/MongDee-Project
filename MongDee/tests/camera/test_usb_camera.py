import cv2
import pytest

import camera.usb as usb_mod
from camera.base import CameraConfig, CameraConnectionError
from camera.usb import USBCamera
from tests.camera.conftest import FakeVideoCapture


def _config(**overrides):
    base = {"id": "CAM01", "protocol": "usb", "device_index": 0}
    base.update(overrides)
    return CameraConfig.from_dict(base)


def test_open_and_read(monkeypatch):
    monkeypatch.setattr(usb_mod.cv2, "VideoCapture", lambda idx, backend: FakeVideoCapture(idx, backend))
    cam = USBCamera(_config())
    cam.open()
    frame = cam.read()
    assert frame.shape == (480, 640, 3)
    assert cam.native_resolution() == (640, 480)
    cam.release()


def test_open_failure_raises(monkeypatch):
    monkeypatch.setattr(usb_mod.cv2, "VideoCapture", lambda idx, backend: FakeVideoCapture(idx, backend, opens=False))
    cam = USBCamera(_config())
    with pytest.raises(CameraConnectionError, match="did not open"):
        cam.open()


def test_read_failure_raises(monkeypatch):
    monkeypatch.setattr(usb_mod.cv2, "VideoCapture", lambda idx, backend: FakeVideoCapture(idx, backend, frames=[]))
    cam = USBCamera(_config())
    cam.open()
    with pytest.raises(CameraConnectionError, match=r"read\(\) failed"):
        cam.read()


def test_read_before_open_raises():
    cam = USBCamera(_config())
    with pytest.raises(CameraConnectionError, match="before open"):
        cam.read()


def test_release_is_idempotent(monkeypatch):
    monkeypatch.setattr(usb_mod.cv2, "VideoCapture", lambda idx, backend: FakeVideoCapture(idx, backend))
    cam = USBCamera(_config())
    cam.open()
    cam.release()
    cam.release()  # must not raise


def test_requested_resolution_is_applied(monkeypatch):
    monkeypatch.setattr(usb_mod.cv2, "VideoCapture", lambda idx, backend: FakeVideoCapture(idx, backend))
    cam = USBCamera(_config(request_width=800, request_height=600))
    cam.open()
    assert cam._cap._props[cv2.CAP_PROP_FRAME_WIDTH] == 800
    assert cam._cap._props[cv2.CAP_PROP_FRAME_HEIGHT] == 600
