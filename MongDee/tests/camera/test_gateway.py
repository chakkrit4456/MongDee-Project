"""Tests camera.gateway.CameraGateway's supervision logic — reconnect with
backoff, FPS throttling, the "always latest, never queued" frame policy,
and clean shutdown — using a scripted CameraSource double instead of a
real camera, since that behaviour is independent of any particular
protocol adapter (those are tested separately, one file per adapter).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import camera.gateway as gateway_mod
from camera.base import CameraConfig, CameraConnectionError, CameraSource, CameraStatus
from camera.gateway import CameraGateway

_REGISTRY: dict[str, "_ScriptedSource"] = {}


class _ScriptedSource(CameraSource):
    """Reused across reconnect attempts within one test (gateway normally
    constructs a fresh adapter per connection attempt; reusing one Python
    object here is harmless and lets a test track state — like "how many
    times has open() been called overall" — across the whole scripted run)."""

    def __init__(self, config):
        super().__init__(config)
        self.open_calls = 0
        self.release_calls = 0
        self.read_calls = 0
        self.fail_open_times = 0  # raise on the first N open() calls
        self.fail_read_after: int | None = None  # raise starting from the (N+1)th read() call
        _REGISTRY[config.id] = self

    def open(self):
        self.open_calls += 1
        if self.open_calls <= self.fail_open_times:
            raise CameraConnectionError(f"scripted open failure #{self.open_calls}")

    def read(self):
        self.read_calls += 1
        if self.fail_read_after is not None and self.read_calls > self.fail_read_after:
            raise CameraConnectionError(f"scripted read failure #{self.read_calls}")
        return np.full((4, 4, 3), self.read_calls % 255, dtype=np.uint8)

    def release(self):
        self.release_calls += 1


@pytest.fixture(autouse=True)
def _patch_adapter_factory(monkeypatch):
    _REGISTRY.clear()
    monkeypatch.setattr(gateway_mod, "create_camera_source", lambda config: _REGISTRY[config.id])
    yield
    _REGISTRY.clear()


def _config(camera_id="CAM01", **overrides):
    base = {
        "id": camera_id,
        "protocol": "usb",  # irrelevant — create_camera_source is patched
        "processing_fps": 1000.0,  # effectively unthrottled unless a test overrides it
        "fail_threshold": 3,
        "backoff_initial_sec": 0.02,
        "backoff_max_sec": 0.05,
        "backoff_multiplier": 2.0,
    }
    base.update(overrides)
    return CameraConfig.from_dict(base)


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_frames_flow_through_and_status_goes_online():
    cfg = _config()
    source = _ScriptedSource(cfg)

    received = []
    statuses = []
    gateway = CameraGateway(on_frame=lambda f: received.append(f), on_status=lambda cid, s, m: statuses.append((cid, s)))
    gateway.add_camera(cfg)
    try:
        assert _wait_until(lambda: len(received) >= 3)
        assert gateway.status("CAM01")[0] == CameraStatus.ONLINE
        assert ("CAM01", CameraStatus.ONLINE) in statuses
        latest = gateway.latest_frame("CAM01")
        assert latest is not None
        assert latest.camera_id == "CAM01"
    finally:
        gateway.stop_all()


def test_fps_throttling_bounds_forwarded_frame_rate():
    # source can produce frames far faster than we want forwarded
    cfg = _config(processing_fps=10.0)
    _ScriptedSource(cfg)

    received = []
    gateway = CameraGateway(on_frame=lambda f: received.append(f))
    gateway.add_camera(cfg)
    try:
        time.sleep(0.5)
    finally:
        gateway.stop_all()
    # 10 fps for 0.5s ~= 5 frames; generous margin for scheduler jitter, but
    # this must be nowhere near the thousands of reads the source could produce.
    assert 1 <= len(received) <= 15


def test_reconnect_after_repeated_read_failures():
    cfg = _config(fail_threshold=3, backoff_initial_sec=0.02, backoff_max_sec=0.05)
    source = _ScriptedSource(cfg)
    source.fail_read_after = 2  # 2 good reads, then every read fails -> OFFLINE after fail_threshold=3 failures

    statuses = []
    gateway = CameraGateway(on_status=lambda cid, s, m: statuses.append((cid, s)))
    gateway.add_camera(cfg)
    try:
        assert _wait_until(lambda: source.open_calls >= 2, timeout=3.0)
        assert (("CAM01", CameraStatus.OFFLINE)) in statuses
        # after reconnecting, the source keeps failing every read forever in
        # this script, so it should keep cycling through OFFLINE repeatedly
        assert _wait_until(lambda: source.open_calls >= 3, timeout=3.0)
    finally:
        gateway.stop_all()


def test_connect_failure_retries_with_backoff():
    cfg = _config(backoff_initial_sec=0.02, backoff_max_sec=0.05)
    source = _ScriptedSource(cfg)
    source.fail_open_times = 3  # first 3 open() calls fail, 4th succeeds

    statuses = []
    gateway = CameraGateway(on_status=lambda cid, s, m: statuses.append((cid, s)))
    gateway.add_camera(cfg)
    try:
        assert _wait_until(lambda: gateway.status("CAM01")[0] == CameraStatus.ONLINE, timeout=3.0)
        assert source.open_calls >= 4
        assert ("CAM01", CameraStatus.OFFLINE) in statuses
    finally:
        gateway.stop_all()


def test_stop_all_halts_frame_delivery_and_releases_source():
    cfg = _config()
    source = _ScriptedSource(cfg)
    received = []
    gateway = CameraGateway(on_frame=lambda f: received.append(f))
    gateway.add_camera(cfg)
    assert _wait_until(lambda: len(received) >= 1)

    gateway.stop_all()
    assert gateway.status("CAM01")[0] == CameraStatus.STOPPED
    count_after_stop = len(received)
    time.sleep(0.1)
    assert len(received) == count_after_stop  # no more frames delivered once stopped


def test_multiple_cameras_run_independently():
    cfg_a = _config("CAM01")
    cfg_b = _config("CAM02")
    _ScriptedSource(cfg_a)
    _ScriptedSource(cfg_b)

    received: dict[str, int] = {}

    def on_frame(frame):
        received[frame.camera_id] = received.get(frame.camera_id, 0) + 1

    gateway = CameraGateway(on_frame=on_frame)
    gateway.add_camera(cfg_a)
    gateway.add_camera(cfg_b)
    try:
        assert _wait_until(lambda: received.get("CAM01", 0) >= 2 and received.get("CAM02", 0) >= 2)
        assert set(gateway.camera_ids()) == {"CAM01", "CAM02"}
    finally:
        gateway.stop_all()


def test_add_camera_duplicate_id_rejected():
    cfg = _config()
    _ScriptedSource(cfg)
    gateway = CameraGateway()
    gateway.add_camera(cfg)
    try:
        with pytest.raises(ValueError, match="already registered"):
            gateway.add_camera(cfg)
    finally:
        gateway.stop_all()


def test_disabled_camera_not_started_by_default():
    cfg = _config(enabled=False)
    source = _ScriptedSource(cfg)
    gateway = CameraGateway()
    gateway.add_camera(cfg)
    time.sleep(0.1)
    assert source.open_calls == 0
    assert gateway.status("CAM01")[0] == CameraStatus.STOPPED
