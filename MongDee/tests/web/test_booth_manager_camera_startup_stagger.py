"""Regression tests for staggering camera worker startup — spec: MongDee
multi-USB-camera root-cause repair ("why can't multiple USB cameras stream
at once"). Real hardware evidence (this session): opening two cameras that
share a USB controller/hub back-to-back can produce Windows'
MF_E_HW_MFT_FAILED_START_STREAMING (0xC00D3704) for whichever one loses the
race for hardware streaming resources — core.vision._open_capture's own
_open_lock only serializes the DirectShow/MSMF *open* handshake, not the
driver actually claiming USB bandwidth afterward. BoothManager.start() (and
the Hot-Plug Scan, when it discovers several new cameras in one tick) now
starts each camera worker CAMERA_STARTUP_STAGGER_SEC apart instead of all at
once, to give each one a real chance to finish claiming its hardware
resources before the next one tries.
"""

from __future__ import annotations

import threading
import time as real_time

import web.booth_manager as booth_manager_module
from web.booth_manager import CAMERA_STARTUP_STAGGER_SEC, BoothManager


class _FakeWorker:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def is_capture_open(self) -> bool:
        # bm.start() below leaves its real _hotplug_loop thread running in
        # the background (see this file's two bm.start()-calling tests,
        # which stop it via bm.stop() in a finally block) -- that loop calls
        # this on every worker on each scan tick, so it must exist even
        # though staggering itself never calls it.
        return False

    def stop(self):
        pass


class _FakeAdaptiveController:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class _FakePerformanceMonitor:
    def system_snapshot(self):
        return {"cpu_percent": 0.0}


class _FakeTime:
    """Stands in for booth_manager_module's own `time` name binding only —
    swapped in via monkeypatch.setattr(booth_manager_module, "time", ...)
    rather than mutating the real time module's own .sleep attribute (which
    would be a *global*, process-wide patch: `import time` always returns
    the one shared module object, so patching its .sleep would also stub
    out this test's own real-time polling waits below, starving the
    background thread being tested of real wall-clock progress)."""

    def __init__(self):
        self.sleep_calls: list[float] = []

    def sleep(self, seconds):
        self.sleep_calls.append(seconds)


def _bare_booth_manager(tmp_path, camera_ids):
    bm = object.__new__(BoothManager)
    bm.db_path = tmp_path / "mongdee.db"
    bm.booth_id = "BOOTH-1"
    bm.event_id = "EVT-1"
    bm._lock = threading.Lock()
    bm._running = False
    bm.camera_ids = list(camera_ids)
    bm.camera_devices = {cid: i for i, cid in enumerate(camera_ids)}
    bm.workers = {cid: _FakeWorker() for cid in camera_ids}
    bm.camera_status = {cid: {"status": "unknown", "message": ""} for cid in camera_ids}
    bm._ignored_usb_indices = set()
    bm.performance_monitor = _FakePerformanceMonitor()
    bm._adaptive_config = None
    bm.ai_worker = object()
    bm._latest_jpeg = {}  # bm.stop() (called by this file's tests to clean up their threads) clears this
    return bm


def test_start_stagger_sleeps_between_but_not_after_last_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(booth_manager_module, "AdaptiveController", _FakeAdaptiveController)
    monkeypatch.setattr(booth_manager_module.db, "log_heartbeat", lambda *a, **k: None)
    fake_time = _FakeTime()
    monkeypatch.setattr(booth_manager_module, "time", fake_time)

    bm = _bare_booth_manager(tmp_path, ["CAM-1", "CAM-2", "CAM-3"])
    try:
        bm.start()

        assert all(w.started for w in bm.workers.values())
        # 3 workers -> exactly 2 staggering sleeps, each of the configured length
        assert fake_time.sleep_calls == [CAMERA_STARTUP_STAGGER_SEC, CAMERA_STARTUP_STAGGER_SEC]
    finally:
        # bm.start() launches real background threads (heartbeat/hotplug)
        # that outlive this test unless stopped -- left running, the
        # hotplug loop would eventually call is_capture_open() on this
        # test's _FakeWorker from an unrelated later test, once its
        # real-time scan interval elapses.
        bm.stop()


def test_start_single_camera_never_sleeps(tmp_path, monkeypatch):
    monkeypatch.setattr(booth_manager_module, "AdaptiveController", _FakeAdaptiveController)
    monkeypatch.setattr(booth_manager_module.db, "log_heartbeat", lambda *a, **k: None)
    fake_time = _FakeTime()
    monkeypatch.setattr(booth_manager_module, "time", fake_time)

    bm = _bare_booth_manager(tmp_path, ["CAM-1"])
    try:
        bm.start()

        assert bm.workers["CAM-1"].started
        assert fake_time.sleep_calls == []
    finally:
        bm.stop()


def test_hotplug_stagger_between_multiple_new_cameras_in_one_scan(tmp_path, monkeypatch):
    bm = object.__new__(BoothManager)
    bm.db_path = tmp_path / "mongdee.db"
    bm._lock = threading.Lock()
    bm.camera_devices = {}
    bm.workers = {}
    bm._ignored_usb_indices = set()
    bm.performance_monitor = _FakePerformanceMonitor()
    bm._hotplug_stop = threading.Event()
    add_camera_calls = []
    bm.add_camera = lambda device: add_camera_calls.append(device)

    monkeypatch.setattr(booth_manager_module, "HOTPLUG_SCAN_INTERVAL_SEC", 0.01)
    calls = []

    def fake_discover_cameras(max_index, skip):
        calls.append(frozenset(skip))
        return [5, 6] if len(calls) == 1 else []

    monkeypatch.setattr(booth_manager_module, "discover_cameras", fake_discover_cameras)
    # _hotplug_loop's own scan-interval wait is threading.Event.wait(), not
    # time.sleep() -- only this module's own CAMERA_STARTUP_STAGGER_SEC call
    # goes through booth_manager_module.time.sleep on this code path.
    fake_time = _FakeTime()
    monkeypatch.setattr(booth_manager_module, "time", fake_time)

    thread = threading.Thread(target=bm._hotplug_loop, daemon=True)
    thread.start()
    deadline = real_time.monotonic() + 3.0
    while len(add_camera_calls) < 2 and real_time.monotonic() < deadline:
        real_time.sleep(0.01)
    bm._hotplug_stop.set()
    thread.join(timeout=2)

    assert add_camera_calls == [5, 6]
    # 2 new cameras found in one scan -> exactly 1 staggering sleep between them
    assert fake_time.sleep_calls == [CAMERA_STARTUP_STAGGER_SEC]
