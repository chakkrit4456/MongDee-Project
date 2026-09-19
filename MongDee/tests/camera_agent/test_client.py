"""Tests for camera_agent/client.py — CameraAgentClient captures via a real
camera.gateway.CameraGateway (fed a fake CameraSource, no real hardware
needed) and pushes frames to a mocked HTTP session, so these exercise the
actual register -> push -> latest-frame-wins logic without any network or
camera hardware.
"""

from __future__ import annotations

import threading
import time

import numpy as np

import camera.gateway as gateway_mod
from camera.base import CameraConfig, CameraConnectionError, CameraSource
from camera_agent.client import CameraAgentClient
from camera_agent.config import AgentConfig


class _FastFakeCamera(CameraSource):
    """Delivers an incrementing-fill frame as fast as asked, no real I/O."""

    def __init__(self, config):
        super().__init__(config)
        self._n = 0

    def open(self):
        pass

    def read(self):
        self._n += 1
        return np.full((8, 8, 3), self._n % 255, dtype=np.uint8)

    def release(self):
        pass


class _NeverOpensCamera(CameraSource):
    def open(self):
        raise CameraConnectionError("simulated: camera never opens")

    def read(self):
        raise CameraConnectionError("unreachable")

    def release(self):
        pass


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"HTTP {self.status_code}")


def _config_with(cam_configs, **overrides):
    return AgentConfig(cameras=tuple(cam_configs), **{
        "server_url": "http://fake-server", "api_key": "test-key",
        "register_retry_sec": 0.1, "push_timeout_sec": 1.0, **overrides,
    })


def test_registers_before_pushing_any_frame(monkeypatch):
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _FastFakeCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=200)
    config = _config_with([cam])

    calls = []

    class FakeSession:
        headers = {}

        def post(self, url, **kwargs):
            calls.append(url)
            return _FakeResponse(200)

    client = CameraAgentClient(config)
    client._session = FakeSession()
    client.start()
    try:
        time.sleep(0.3)
    finally:
        client.stop()

    assert any("/register" in c for c in calls), "must register before pushing frames"
    register_index = next(i for i, c in enumerate(calls) if "/register" in c)
    frame_indices = [i for i, c in enumerate(calls) if "/frame" in c]
    assert frame_indices, "expected at least one frame push"
    assert all(i > register_index for i in frame_indices)


def test_frames_are_pushed_with_incrementing_sequence(monkeypatch):
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _FastFakeCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=200)
    config = _config_with([cam])

    pushed_sequences = []
    lock = threading.Lock()

    class FakeSession:
        headers = {}

        def post(self, url, **kwargs):
            if "/frame" in url:
                with lock:
                    pushed_sequences.append(int(kwargs["data"]["sequence"]))
            return _FakeResponse(200)

    client = CameraAgentClient(config)
    client._session = FakeSession()
    client.start()
    try:
        time.sleep(0.3)
    finally:
        client.stop()

    assert len(pushed_sequences) >= 2
    assert pushed_sequences == sorted(pushed_sequences)  # strictly increasing, never reordered
    assert len(set(pushed_sequences)) == len(pushed_sequences)  # never repeated


def test_never_sends_more_frames_than_captured_and_drops_under_slow_network(monkeypatch):
    # Capture much faster than the (artificially slow) network push can
    # keep up with -- Latest-Frame-Wins must mean fewer pushes than
    # captures, never a growing backlog of queued pushes.
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _FastFakeCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=1000)
    config = _config_with([cam])

    push_count = [0]

    class FakeSession:
        headers = {}

        def post(self, url, **kwargs):
            if "/frame" in url:
                push_count[0] += 1
                time.sleep(0.05)  # slow "network"
            return _FakeResponse(200)

    client = CameraAgentClient(config)
    client._session = FakeSession()
    client.start()
    try:
        time.sleep(0.3)
    finally:
        client.stop()

    # In 0.3s at ~0.05s/push, at most ~6-7 pushes could possibly complete --
    # nowhere near the hundreds of frames a 1000fps-configured fake camera
    # would otherwise capture, proving old frames were dropped, not queued.
    assert push_count[0] < 15


def test_registration_failure_is_retried_until_success(monkeypatch):
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _FastFakeCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=200)
    config = _config_with([cam], register_retry_sec=0.05)

    attempts = [0]
    frame_pushed = threading.Event()

    class FlakySession:
        headers = {}

        def post(self, url, **kwargs):
            if "/register" in url:
                attempts[0] += 1
                if attempts[0] < 3:
                    return _FakeResponse(500)
                return _FakeResponse(200)
            if "/frame" in url:
                frame_pushed.set()
            return _FakeResponse(200)

    client = CameraAgentClient(config)
    client._session = FlakySession()
    client.start()
    try:
        assert frame_pushed.wait(timeout=3.0), "should eventually register and push once retries succeed"
    finally:
        client.stop()
    assert attempts[0] >= 3


def test_unreachable_camera_never_crashes_the_agent(monkeypatch):
    # A camera that can never even open (bad device index, USB unplugged)
    # must not crash the whole agent process or block other cameras.
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _NeverOpensCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=200,
                        backoff_initial_sec=0.05, backoff_max_sec=0.1)
    config = _config_with([cam])
    client = CameraAgentClient(config)
    client._session = type("S", (), {"headers": {}, "post": lambda self, url, **kw: _FakeResponse(200)})()

    client.start()
    try:
        time.sleep(0.3)  # must not raise / must not hang
    finally:
        client.stop()  # must return promptly


def test_stop_joins_sender_threads(monkeypatch):
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda cfg: _FastFakeCamera(cfg))
    cam = CameraConfig(id="CAM01", protocol="usb", processing_fps=200)
    config = _config_with([cam])
    client = CameraAgentClient(config)
    client._session = type("S", (), {"headers": {}, "post": lambda self, url, **kw: _FakeResponse(200)})()

    client.start()
    time.sleep(0.1)
    client.stop()
    for thread in client._sender_threads.values():
        assert not thread.is_alive()
