"""A truly wedged cap.read() (release() called from the watchdog, but the underlying native call
still never returns -- see test_a_hung_read_is_recovered_without_blocking_the_watchdog in
tests/core/test_camera_unplug_replug.py, which only proves the watchdog itself doesn't block, not
that the camera ever comes back) permanently strands CameraWorker.run()'s own thread: nothing in
this process can preempt a blocked native call. core.capture_process.CaptureProcess exists exactly
for this (see its own module docstring) but was never wired into the production capture path.
This suite drives the real escalation wiring end-to-end: recover_from_hung_read counting
consecutive unresolved hangs, escalating to a REAL (spawned, not mocked) CaptureProcess via a
separate pump thread -- since the original thread can never itself reach the code that would start
one -- and the camera reporting "online" again afterwards despite its original run() thread
remaining permanently stuck. Uses core.capture_sim's fault-injection factories (as
tests/core/test_capture_process.py already does) instead of real camera hardware.
"""
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


class FakeCap:
    """Same shape as tests/core/test_camera_unplug_replug.py's FakeCap: 'hung' blocks forever in
    read() (a gate that is never set reproduces a genuinely unrecoverable stuck native call --
    release() flips `released` but a read() already inside gate.wait() does not notice)."""

    def __init__(self, mode="live"):
        self.mode = mode
        self.n = 0
        self.released = False
        self.gate = threading.Event()

    def isOpened(self):
        return not self.released

    def read(self):
        if self.released:
            return False, None
        self.n += 1
        if self.mode == "hung":
            self.gate.wait(30)
            return False, None
        img = np.full((120, 160, 3), 40 + (self.n % 50), dtype=np.uint8)
        return True, img

    def release(self):
        self.released = True


def _worker(monkeypatch, cap, escalate_after=2):
    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: cap)
    monkeypatch.setattr(vision, "HANG_ESCALATION_THRESHOLD", escalate_after)
    model = type("Model", (), {"names": {0: "person"}})()
    statuses = []
    w = vision.CameraWorker("CAM-1", 0, model, [], on_status=lambda c, s, m: statuses.append(s),
                            on_frame=lambda c, f: None, ai_worker=vision.AIWorker())
    return w, statuses


def _wait(cond, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_escalation_counter_resets_on_a_good_frame_in_between(monkeypatch):
    """Unit-level, no background thread (avoids racing a live capture loop against manual state
    pokes): one unresolved hang alone must not escalate, and a single good frame afterwards must
    fully reset the counter, exactly like check_stream_health's ONLINE_GOOD_STREAK reasoning."""
    cap = FakeCap("live")
    w, _ = _worker(monkeypatch, cap, escalate_after=2)
    w._read_started = time.monotonic() - 10
    assert w.recover_from_hung_read(stall_sec=0.01) is True
    assert w._consecutive_hang_recoveries == 1
    assert w.get_capture_mode() == "thread"          # one hang alone must not escalate

    frame = np.full((120, 160, 3), 50, dtype=np.uint8)
    assert w._process_captured_frame(True, frame) is True
    assert w._consecutive_hang_recoveries == 0


def test_repeated_unresolved_hangs_escalate_to_process_isolated_capture(monkeypatch):
    """The exact scenario that motivated core/capture_process.py: a native read that never
    returns even after release(). Escalation must happen from the WATCHDOG's own call (this test
    calls recover_from_hung_read directly, standing in for BoothManager's watchdog thread) since
    the run() thread itself never gets back to the top of its loop to do anything at all."""
    cap = FakeCap("live")
    w, statuses = _worker(monkeypatch, cap, escalate_after=2)
    # Swap in a hardware-free real CaptureProcess factory (core.capture_sim, same as
    # tests/core/test_capture_process.py) instead of the real negotiated-capture one, which needs
    # a real device.
    monkeypatch.setattr(
        vision.CameraWorker, "_process_capture_spec",
        lambda self: ("core.capture_sim:make_fake", {"mode": "ok", "marker": 3}),
    )
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    try:
        assert _wait(lambda: w._last_status == "online")
        cap.mode = "hung"                          # the next read() blocks forever
        assert _wait(lambda: w.read_stall_seconds() > 0.05, 5)

        # Two consecutive stalls with no good frame in between -- the run() thread is genuinely
        # stuck inside cap.read() (FakeCap.gate is never set), so it can never itself observe
        # escalation; only the watchdog-thread call below can act.
        assert w.recover_from_hung_read(stall_sec=0.01) is True
        assert w.get_capture_mode() == "thread"
        time.sleep(0.02)   # recover_from_hung_read only re-fires after another full stall window
        assert w.recover_from_hung_read(stall_sec=0.01) is True
        assert w.get_capture_mode() == "process"

        # The camera must actually come back online through the escalated path, even though the
        # original run() thread is permanently parked inside FakeCap's gate.wait(30) and this
        # worker's own `t` thread will never terminate on its own.
        assert _wait(lambda: statuses[-1:] == ["online"], 15), statuses
        assert w.get_worker_pid() is not None
        assert w.is_capture_open() is True
        assert w.get_frame_sequence() > 0
        assert w.get_last_capture_ts() is not None

        # A third call after escalation must be a pure no-op (nothing left for it to recover).
        assert w.recover_from_hung_read(stall_sec=0.01) is False
    finally:
        w._running = False       # unblocks the pump thread; the original stuck thread is leaked
        w._stop_event.set()
        if w._process_capture is not None:
            w._process_capture.stop()
        cap.gate.set()            # let the permanently-blocked FakeCap.read() return, for cleanup
        t.join(timeout=2)
        if w._pump_thread is not None:
            w._pump_thread.join(timeout=3)


def test_diagnostics_expose_capture_mode_and_counters(monkeypatch):
    cap = FakeCap("live")
    w, _ = _worker(monkeypatch, cap)
    d = w.get_diagnostics()
    assert d["capture_mode"] == "thread"
    assert d["worker_pid"] is None
    assert d["hang_recoveries"] == 0
    assert d["corrupt_frame_count"] == 0
    assert d["frozen_frame_count"] == 0
    assert d["reason_code"] == "OK"


@pytest.mark.parametrize("status,message,expected", [
    ("online", "ปกติ", "OK"),
    ("offline", "กล้องไม่ตอบสนอง (อ่านภาพค้าง) — กำลังรีเซ็ตการเชื่อมต่อ", "BLOCKED_READ"),
    # "หยุดหรือเสียหาย" (stopped-or-corrupted) covers both a genuinely frozen picture and a plain
    # stale-no-frame condition (see check_stream_health) -- with no frozen-frame evidence in
    # _frame_integrity (a fresh worker, as here), it must default to CORRUPT_FRAME, not guess
    # FROZEN_FRAME from the message text alone. See the dedicated frozen-state test below for the
    # other branch.
    ("offline", "ภาพจากกล้องหยุดหรือเสียหาย — กำลังเชื่อมต่อใหม่", "CORRUPT_FRAME"),
    ("offline", "เปิดกล้อง 0 ไม่สำเร็จ", "OPEN_FAILED"),
    ("offline", "อ่านภาพจากกล้องไม่ได้ต่อเนื่อง", "READ_FAILED"),
    ("disabled", "ปิดใช้งานกล้องนี้", "DISABLED"),
    ("offline", "something unexpected", "UNKNOWN"),
])
def test_classify_reason_maps_known_messages(monkeypatch, status, message, expected):
    cap = FakeCap("live")
    w, _ = _worker(monkeypatch, cap)
    assert w._classify_reason(status, message) == expected


def test_classify_reason_reports_frozen_frame_when_the_stream_is_actually_frozen(monkeypatch):
    cap = FakeCap("live")
    w, _ = _worker(monkeypatch, cap)
    frame = np.full((120, 160, 3), 77, dtype=np.uint8)
    now = 1000.0
    # Feed the same frame enough times, spaced past freeze_sec, for FrameIntegrity to actually
    # call it frozen -- mirrors what check_stream_health observes in production right before it
    # emits this exact message.
    for i in range(15):
        w._frame_integrity.check(frame, now + i * (vision.FROZEN_STREAM_SEC / 10))
    assert w._frame_integrity.frozen_for(now + 15 * (vision.FROZEN_STREAM_SEC / 10)) > 0
    monkeypatch.setattr("time.time", lambda: now + 15 * (vision.FROZEN_STREAM_SEC / 10))
    assert w._classify_reason("offline", "ภาพจากกล้องหยุดหรือเสียหาย — กำลังเชื่อมต่อใหม่") == "FROZEN_FRAME"


def test_classify_reason_flags_physical_disconnect_over_message_text(monkeypatch):
    cap = FakeCap("live")
    w, _ = _worker(monkeypatch, cap)
    w.device_present_fn = lambda device: False
    assert w._classify_reason("offline", "อ่านภาพจากกล้องไม่ได้ต่อเนื่อง") == "PHYSICAL_DISCONNECT"
