"""A USB camera that is unplugged must vanish from the tile within about a second, one that is plugged back
in must come back at once, a corrupt / frozen / hung stream must be handled the same way, and none of it may
hang the worker or the shared watchdog. Drives the real CameraWorker.run() loop with fake captures."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from core import vision


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    vision._active_camera_count = 0
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    yield
    vision._active_camera_count = 0


def _scene(i: int) -> np.ndarray:
    """A calm, real-looking frame that differs a little every frame (a live sensor is never bit-identical)."""
    gray = np.tile(np.linspace(50, 170, 160, dtype=np.float32), (120, 1)) + np.linspace(0, 30, 120, dtype=np.float32)[:, None]
    gray = np.clip(gray + (i % 11), 0, 255).astype(np.uint8)
    cv2.circle(gray, (40 + (i % 9) * 8, 60), 22, 235, -1)               # something moving in the scene
    gray = cv2.GaussianBlur(gray, (0, 0), 2.0)
    # the same luminance structure in every channel, like a real optical image
    return np.dstack([gray, np.clip(gray.astype(np.int16) + 8, 0, 255).astype(np.uint8),
                      np.clip(gray.astype(np.int16) + 16, 0, 255).astype(np.uint8)])


class FakeCap:
    """mode: 'live' (new frame each read), 'dead' (ok=False), 'frozen' (same frame forever), 'hung' (blocks)."""

    def __init__(self, mode="live", release_blocks=False):
        self.mode = mode
        self.n = 0
        self.released = False
        self.release_blocks = release_blocks
        self.gate = threading.Event()
        self._frozen = _scene(3)

    def isOpened(self):
        return not self.released

    def read(self):
        if self.released:
            return False, None
        self.n += 1
        time.sleep(0.005)
        if self.mode == "live":
            return True, _scene(self.n)
        if self.mode == "frozen":
            return True, self._frozen.copy()
        if self.mode == "hung":
            self.gate.wait(30)
            return False, None
        return False, None

    def release(self):
        if self.release_blocks:
            self.gate.wait(30)
        self.released = True


def _worker(monkeypatch, caps, present=None):
    """caps: list consumed by successive opens; the last one repeats."""
    opens = []

    def fake_open(device, label=None, low_bandwidth_only=False):
        cap = caps[min(len(opens), len(caps) - 1)]
        opens.append(low_bandwidth_only)
        if isinstance(cap, FakeCap) and cap.released:
            cap = FakeCap("dead")
        return cap if cap is not None else cv2.VideoCapture()

    monkeypatch.setattr(vision, "_open_capture", fake_open)
    model = type("Model", (), {"names": {0: "person"}})()
    statuses, frames = [], []
    w = vision.CameraWorker("CAM-1", 0, model, [], on_status=lambda c, s, m: statuses.append(s),
                            on_frame=lambda c, f: frames.append(time.monotonic()), ai_worker=vision.AIWorker())
    if present is not None:
        w.device_present_fn = present
    return w, statuses, frames, opens


def _start(w):
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    return t


def _stop(w, t):
    w._running = False
    w._stop_event.set()
    for _ in range(3):
        cap = w._cap
        if cap is not None and hasattr(cap, "gate"):
            cap.gate.set()
    t.join(timeout=3)


def _wait(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# ----------------------------------------------------------------------------- health check (unit level)
def _online_worker(monkeypatch, device=0):
    w, statuses, _, _ = _worker(monkeypatch, [FakeCap()])
    w.device = device
    w._last_status = "online"
    return w, statuses


def test_no_good_frame_for_a_second_reports_offline(monkeypatch):
    w, statuses = _online_worker(monkeypatch)
    now = time.monotonic()
    w._last_good_frame_ts = now - 0.5
    assert w.check_stream_health(now) is False
    w._last_good_frame_ts = now - 1.2
    assert w.check_stream_health(now) is True
    assert statuses[-1] == "offline"


def test_after_a_device_change_the_deadline_is_a_fraction_of_a_second(monkeypatch):
    w, statuses = _online_worker(monkeypatch)
    now = time.monotonic()
    w._last_good_frame_ts = now - 0.5
    w.note_device_change()
    assert w.check_stream_health(now) is True


def test_a_healthy_camera_is_untouched_by_a_device_change(monkeypatch):
    w, statuses = _online_worker(monkeypatch)
    now = time.monotonic()
    w._last_good_frame_ts = now - 0.03
    w.note_device_change()
    assert w.check_stream_health(now) is False and statuses == []


def test_health_check_ignores_non_usb_sources_and_offline_cameras(monkeypatch):
    w, statuses = _online_worker(monkeypatch, device="rtsp://cam/stream")
    w._last_good_frame_ts = time.monotonic() - 30
    assert w.check_stream_health() is False
    w2, _ = _online_worker(monkeypatch)
    w2._last_status = "offline"
    w2._last_good_frame_ts = time.monotonic() - 30
    assert w2.check_stream_health() is False


def test_stale_and_absent_device_frees_a_blocked_capture_immediately(monkeypatch):
    w, statuses = _online_worker(monkeypatch)
    cap = FakeCap()
    w._cap = cap
    w.device_present_fn = lambda d: False
    w._last_good_frame_ts = time.monotonic() - 5
    assert w.check_stream_health() is True
    assert w._cap is None and cap.released


# ----------------------------------------------------------------------------- release never blocks
def test_release_of_a_wedged_capture_does_not_block_the_caller(monkeypatch):
    monkeypatch.setattr(vision, "RELEASE_JOIN_TIMEOUT_SEC", 0.1)
    w, _ = _online_worker(monkeypatch)
    cap = FakeCap(release_blocks=True)
    w._cap = cap
    t0 = time.monotonic()
    w._release_capture()
    assert time.monotonic() - t0 < 1.0
    assert w._cap is None and not w._release_done.is_set()
    cap.gate.set()
    assert w._release_done.wait(2.0)


def test_reopen_is_postponed_while_the_old_handle_is_still_being_released(monkeypatch):
    monkeypatch.setattr(vision, "RELEASE_WAIT_BEFORE_REOPEN_SEC", 0.05)
    w, _ = _online_worker(monkeypatch)
    w._release_done.clear()                     # a wedged release is still running
    assert w._open() is False
    assert w.stuck_releases == 1


# ----------------------------------------------------------------------------- reopen gating on presence
def test_an_unplugged_camera_is_not_reopened_until_it_reappears(monkeypatch):
    state = {"present": False}
    w, statuses, _, opens = _worker(monkeypatch, [FakeCap()], present=lambda d: state["present"])
    w._ever_connected = True
    w._last_reopen_attempt = 0.0
    assert w._attempt_reopen() is False and opens == []          # nothing plugged in: no open attempt at all
    assert w._attempt_reopen() is False and opens == []
    state["present"] = True                                       # the device monitor saw it come back
    w.request_immediate_retry()
    assert w._attempt_reopen() is True and len(opens) == 1


def test_first_ever_open_is_not_gated_by_presence(monkeypatch):
    w, _, _, opens = _worker(monkeypatch, [FakeCap()], present=lambda d: False)
    assert w._ever_connected is False
    assert w._attempt_reopen() is True and len(opens) == 1


def test_a_listed_but_unopenable_camera_backs_off_only_briefly(monkeypatch):
    w, _, _, opens = _worker(monkeypatch, [None], present=lambda d: True)
    w._ever_connected = True
    for _ in range(6):
        w._last_reopen_attempt = 0.0
        w._attempt_reopen()
    assert w._reopen_backoff_sec == vision.PRESENT_REOPEN_BACKOFF_MAX_SEC
    w2, _, _, _ = _worker(monkeypatch, [None])                    # presence unknown: the old long back-off
    w2._ever_connected = True
    for _ in range(6):
        w2._last_reopen_attempt = 0.0
        w2._attempt_reopen()
    assert w2._reopen_backoff_sec == vision.REOPEN_BACKOFF_MAX_SEC


def test_camera_that_moved_to_another_index_is_followed_while_its_old_index_is_gone(monkeypatch):
    listed = {2}
    w, _, _, opens = _worker(monkeypatch, [FakeCap()], present=lambda d: int(d) in listed)
    w._ever_connected = True
    w._physical_id = "PID-1"
    w.device = 0
    monkeypatch.setattr(w, "_relocate_device_if_moved", lambda: setattr(w, "device", 2) or True)
    assert w._attempt_reopen() is True and w.device == 2


# ----------------------------------------------------------------------------- a fast-failing device open
def test_an_index_that_will_not_open_is_tried_once_per_backend_not_once_per_profile(monkeypatch):
    created = []

    class Closed:
        def __init__(self, *a, **k):
            created.append(a)

        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(vision.cv2, "VideoCapture", Closed)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW), (device, cv2.CAP_ANY)])
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda d: None)
    cap = vision._open_capture(3)
    assert not cap.isOpened()
    assert len(created) == 3          # 2 backends + the final closed placeholder, not 2 x len(profiles)


# ----------------------------------------------------------------------------- end-to-end through run()
def test_unplug_goes_offline_within_a_second_and_replug_comes_back_online(monkeypatch):
    live = FakeCap("live")
    replug = FakeCap("live")
    state = {"present": True}
    w, statuses, frames, opens = _worker(monkeypatch, [live, replug], present=lambda d: state["present"])
    t = _start(w)
    try:
        assert _wait(lambda: statuses[-1:] == ["online"]), statuses
        # ---- unplug: reads start failing and Windows stops listing the device
        t_unplug = time.monotonic()
        live.mode = "dead"
        state["present"] = False
        w.note_device_change()
        assert _wait(lambda: statuses[-1:] == ["offline"], 2.0), statuses
        assert time.monotonic() - t_unplug < 1.5
        n_frames = len(frames)
        time.sleep(0.3)
        assert len(frames) == n_frames                            # nothing is delivered while it is gone
        n_opens = len(opens)
        time.sleep(0.6)
        assert len(opens) == n_opens                              # and nothing keeps hammering the dead index
        # ---- replug
        t_plug = time.monotonic()
        state["present"] = True
        w.request_immediate_retry()
        assert _wait(lambda: statuses[-1:] == ["online"], 3.0), statuses
        assert _wait(lambda: len(frames) > n_frames, 2.0)
        assert time.monotonic() - t_plug < 2.5
    finally:
        _stop(w, t)


def test_a_frozen_picture_is_reported_offline_and_reconnected(monkeypatch):
    frozen = FakeCap("frozen")
    good = FakeCap("live")
    w, statuses, frames, opens = _worker(monkeypatch, [FakeCap("live"), good])
    monkeypatch.setattr(vision, "FROZEN_STREAM_SEC", 0.3)
    w._frame_integrity.freeze_sec = 0.3
    t = _start(w)
    try:
        assert _wait(lambda: statuses[-1:] == ["online"]), statuses
        w._cap.mode = "frozen"
        assert _wait(lambda: "offline" in statuses, 3.0), statuses
        assert _wait(lambda: statuses[-1:] == ["online"], 5.0), statuses
    finally:
        _stop(w, t)


def test_a_hung_read_is_recovered_without_blocking_the_watchdog(monkeypatch):
    monkeypatch.setattr(vision, "RELEASE_JOIN_TIMEOUT_SEC", 0.1)
    hung = FakeCap("live", release_blocks=True)
    w, statuses, frames, opens = _worker(monkeypatch, [hung, FakeCap("live")])
    t = _start(w)
    try:
        assert _wait(lambda: statuses[-1:] == ["online"]), statuses
        hung.mode = "hung"
        assert _wait(lambda: w.read_stall_seconds() > 0.2, 2.0)
        t0 = time.monotonic()
        assert w.recover_from_hung_read(stall_sec=0.2) is True     # must return even though release() wedges
        assert time.monotonic() - t0 < 1.5
        assert statuses[-1] == "offline"
    finally:
        _stop(w, t)


def test_repeated_faults_switch_to_the_low_bandwidth_profiles(monkeypatch):
    w, _, _, opens = _worker(monkeypatch, [FakeCap()])
    assert w._low_bandwidth_until == 0.0
    w._note_fault()
    assert w._low_bandwidth_until == 0.0
    w._note_fault()
    assert w._low_bandwidth_until > time.monotonic() + 100
    assert w._open() is True and opens[-1] is True


def test_a_flapping_stream_needs_a_few_good_frames_before_it_is_online_again(monkeypatch):
    w, statuses, frames, opens = _worker(monkeypatch, [FakeCap("live")])
    t = _start(w)
    try:
        assert _wait(lambda: statuses[-1:] == ["online"]), statuses
        w._emit_status("offline", "x")
        w._good_streak = 0
        assert _wait(lambda: statuses[-1:] == ["online"], 2.0)
        assert w._good_streak >= vision.ONLINE_GOOD_STREAK
    finally:
        _stop(w, t)
