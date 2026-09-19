"""BoothManager wiring of the device-presence monitor: unplug -> strict deadline, replug -> immediate
reconnect, no open-probing of known cameras, no duplicate camera for a camera that came back re-indexed."""
from __future__ import annotations

import threading
import time

import pytest

import web.booth_manager as bmm
from core.device_presence import DevicePresenceMonitor, PresenceEvent
from web.booth_manager import BoothManager


class _Worker:
    def __init__(self, capture_open, physical_id=None):
        self._open = capture_open
        self._physical_id = physical_id
        self.retries = 0
        self.changes = 0
        self.health = 0
        self.present_fn = None

    def is_capture_open(self):
        return self._open

    def request_immediate_retry(self):
        self.retries += 1

    def note_device_change(self):
        self.changes += 1

    def check_stream_health(self):
        self.health += 1


def _bare(workers, devices=None):
    bm = object.__new__(BoothManager)
    bm._lock = threading.Lock()
    bm.workers = dict(workers)
    bm.camera_devices = dict(devices or {})
    bm._hotplug_wake = threading.Event()
    bm._hotplug_stop = threading.Event()
    return bm


def test_a_camera_disappearing_puts_every_worker_on_the_strict_deadline():
    a, b = _Worker(True), _Worker(True)
    bm = _bare({"CAM-1": a, "CAM-2": b})
    bm._on_presence_event(PresenceEvent({}, (), ("USB Camera",)))
    assert (a.changes, b.changes) == (1, 1) and (a.retries, b.retries) == (0, 0)


def test_a_camera_appearing_reconnects_only_the_disconnected_workers_and_wakes_the_scan():
    live, gone = _Worker(True), _Worker(False)
    bm = _bare({"CAM-1": live, "CAM-2": gone})
    bm._on_presence_event(PresenceEvent({}, ("USB Camera",), ()))
    assert gone.retries == 1 and live.retries == 0
    assert bm._hotplug_wake.is_set()


def test_the_watchdog_runs_the_stream_health_check_every_tick(monkeypatch):
    monkeypatch.setattr(bmm, "CAPTURE_WATCHDOG_INTERVAL_SEC", 0.01)
    w = _Worker(True)
    bm = _bare({"CAM-1": w})
    stop = threading.Event()
    t = threading.Thread(target=bm._capture_watchdog_loop, args=(stop,), daemon=True)
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(2)
    assert w.health >= 5


def test_workers_get_the_presence_lookup(monkeypatch):
    class _Model:
        names = {0: "person"}

    bm = _bare({})
    bm.presence = DevicePresenceMonitor(enumerate_fn=lambda: {0: "USB Camera"})
    bm.presence.poll_once()
    bm.catalog = type("C", (), {"product_keys": lambda s: [], "get": lambda s, k: None})()
    bm.model, bm.recognizer, bm.model_device = _Model(), None, "cpu"
    bm._on_frame = bm._on_detections = bm._on_status = bm._on_person_tracks = lambda *a, **k: None
    bm.gender_age_backend = bm.performance_monitor = bm.ai_worker = None
    bm.reid_registry = type("R", (), {"get_confirmed_global_id_for": lambda s, *a: None})()
    bm.attribute_smoother = type("S", (), {"get": lambda s, *a: None})()
    w = bm._make_worker("CAM-1", 0)
    assert w.device_present_fn(0) is True and w.device_present_fn(3) is False


class _Presence:
    def __init__(self, known):
        self.known = known

    def is_present(self, index):
        return self.known.get(int(index))


def test_hotplug_does_not_open_probe_a_known_camera_when_the_os_list_is_known(tmp_path, monkeypatch):
    monkeypatch.setattr(bmm, "HOTPLUG_SCAN_INTERVAL_SEC", 0.01)
    bm = _bare({"CAM-1": _Worker(False)}, {"CAM-1": 0})
    bm.db_path = tmp_path / "x.db"
    bm._ignored_usb_indices = set()
    bm.performance_monitor = type("P", (), {"system_snapshot": lambda s: {"cpu_percent": 0.0}})()
    bm.presence = _Presence({0: True})
    calls = []
    monkeypatch.setattr(bmm, "discover_cameras", lambda max_index, skip: calls.append(frozenset(skip)) or [])
    t = threading.Thread(target=bm._hotplug_loop, daemon=True)
    t.start()
    end = time.monotonic() + 3
    while not calls and time.monotonic() < end:
        time.sleep(0.01)
    bm._hotplug_stop.set()
    t.join(2)
    assert calls and 0 in calls[0]           # the disconnected-but-listed camera is left to its own worker


def test_a_wake_up_shortens_the_hotplug_wait():
    bm = _bare({})
    bm._hotplug_wake.set()
    t0 = time.monotonic()
    assert bm._wait_hotplug_tick(30.0) == (False, True)
    assert time.monotonic() - t0 < 1.0
    bm._hotplug_stop.set()
    assert bm._wait_hotplug_tick(30.0) == (True, False)


def test_add_camera_refuses_a_re_indexed_camera_that_an_existing_worker_already_owns(monkeypatch):
    bm = _bare({"CAM-1": _Worker(False, physical_id="USB\\VID_4C4A&PID_4A55@port2")}, {"CAM-1": 0})
    ident = type("I", (), {"physical_id": "USB\\VID_4C4A&PID_4A55@port2"})()
    monkeypatch.setattr(bmm, "get_physical_camera_identities", lambda: {3: ident})
    with pytest.raises(ValueError):
        bm.add_camera(3)
