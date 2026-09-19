"""MONGDEE_CAPTURE_SIZE: more pixels for far-away people, opt-in."""
from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from core import vision
from web.booth_manager import BoothManager, DISPLAY_MAX_WIDTH


@pytest.mark.parametrize("raw,expected", [
    ("1280x720", ("MJPG", 1280, 720, 30)), ("1920X1080@15", ("MJPG", 1920, 1080, 15)),
    (" 800x600 ", ("MJPG", 800, 600, 30)), ("abc", None), ("", None), ("1280*720", None)])
def test_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("MONGDEE_CAPTURE_SIZE", raw)
    assert vision.preferred_capture_profile() == expected


def test_the_default_profile_list_is_untouched_without_the_variable(monkeypatch):
    monkeypatch.delenv("MONGDEE_CAPTURE_SIZE", raising=False)
    assert vision._open_profiles() == vision._OPEN_PROFILES


def _capture_factory(created):
    class Cap:
        def __init__(self, source=None, backend=None):
            self.props, self.n = {}, 0
            created.append(self)

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.props[prop] = value
            return True

        def get(self, prop):
            return self.props.get(prop, 0)

        def read(self):
            self.n += 1
            y, x = np.mgrid[0:48, 0:64]
            g = (x * 3 + y).astype(np.uint8)
            return True, np.dstack([g, g, g])

        def release(self):
            pass

    return Cap


def test_the_requested_size_is_negotiated_first_and_applies_to_every_camera(monkeypatch):
    created = []
    monkeypatch.setattr(vision.cv2, "VideoCapture", _capture_factory(created))
    monkeypatch.setattr(vision, "_candidate_opens", lambda d: [(d, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda d: None)
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    monkeypatch.setenv("MONGDEE_CAPTURE_SIZE", "1280x720")
    cap = vision._open_capture(0)
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280 and cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    monkeypatch.setattr(vision, "_active_camera_count", 1)                 # a second camera while one is streaming
    model = type("Model", (), {"names": {0: "person"}})()
    w = vision.CameraWorker("CAM-2", 1, model, [], ai_worker=vision.AIWorker())
    seen = []
    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: seen.append(low_bandwidth_only) or cap)
    assert w._open() and seen == [False]                                   # NOT forced down to 320x240


def test_a_second_camera_still_uses_the_small_profile_when_no_size_is_requested(monkeypatch):
    monkeypatch.delenv("MONGDEE_CAPTURE_SIZE", raising=False)
    monkeypatch.setattr(vision, "_active_camera_count", 1)
    model = type("Model", (), {"names": {0: "person"}})()
    w = vision.CameraWorker("CAM-2", 1, model, [], ai_worker=vision.AIWorker())
    seen = []
    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: seen.append(low_bandwidth_only) or cv2.VideoCapture())
    w._open()
    assert seen == [True]


def test_the_browser_stream_is_capped_in_width_but_the_ai_frame_is_not():
    bm = object.__new__(BoothManager)
    bm._lock = threading.Lock()
    bm._latest_jpeg = {}
    frame = np.full((720, 1280, 3), 100, np.uint8)
    bm._on_frame("CAM-1", frame)
    decoded = cv2.imdecode(np.frombuffer(bm._latest_jpeg["CAM-1"], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == DISPLAY_MAX_WIDTH and decoded.shape[0] == 540
    assert frame.shape == (720, 1280, 3)                                    # the frame the AI sees is untouched
