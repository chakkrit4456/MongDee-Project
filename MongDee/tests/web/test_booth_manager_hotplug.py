"""Unit tests for web/booth_manager.py's Hot-Plug Scan (_hotplug_loop) —
specifically the fast-reconnect nudge for an already-known camera whose
device dropped and came back, added alongside its existing "pick up a
genuinely new USB camera" behavior.

Root-cause context: each CameraWorker retries its own device on an
exponential backoff (core/vision.py's REOPEN_BACKOFF_*, capped at
REOPEN_BACKOFF_MAX_SEC = 30s) so a truly-dead camera isn't hammered forever.
But that same backoff means a camera unplugged long enough for it to reach
the cap could sit disconnected for up to 30s after being physically
replugged, before its own timer even tries again. _hotplug_loop already
runs independently every few seconds and already does a full device probe
(discover_cameras) — it now also probes already-known-but-currently-
disconnected camera indices (never a *live* one, to avoid disrupting a
connected camera) and, if found reachable, nudges that camera's own worker
to retry immediately (CameraWorker.request_immediate_retry) instead of
waiting on its own unrelated timer. Critically, this must never create a
second camera_id for the same physical device (see BoothManager.add_camera)
— reconnecting must always be the *same* logical camera's lifecycle.

Same "bare BoothManager" pattern as test_booth_manager_tripwire.py — the
real constructor loads a YOLO model/cameras/etc. none of this touches.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import web.booth_manager as booth_manager_module
from web.booth_manager import BoothManager


class _FakeWorker:
    def __init__(self, capture_open: bool):
        self._capture_open = capture_open
        self.retry_calls = 0

    def is_capture_open(self) -> bool:
        return self._capture_open

    def request_immediate_retry(self) -> None:
        self.retry_calls += 1


class _FakePerformanceMonitor:
    def system_snapshot(self):
        return {"cpu_percent": 0.0}


def _bare_booth_manager(tmp_path, camera_devices, workers):
    bm = object.__new__(BoothManager)
    bm.db_path = tmp_path / "mongdee.db"  # never created — load_camera_settings tolerates a missing file
    bm._lock = threading.Lock()
    bm.camera_devices = dict(camera_devices)
    bm.workers = dict(workers)
    bm._ignored_usb_indices = set()
    bm.performance_monitor = _FakePerformanceMonitor()
    bm._hotplug_stop = threading.Event()
    bm.add_camera_calls = []
    bm.add_camera = lambda device: bm.add_camera_calls.append(device)
    return bm


def _run_one_hotplug_tick(monkeypatch, bm, found):
    """Runs _hotplug_loop in a background thread with the scan interval
    shrunk down, captures the `skip` argument discover_cameras was actually
    called with, then stops the loop once it has produced some observable
    effect (or a short deadline passes).

    `discover_cameras` is faked to return `found` only on its first call and
    `[]` on every call after that — both so the loop's own interval reset
    (found something -> back to the base interval) doesn't matter for
    timing here, and so a real camera's worker method (request_immediate_
    retry/add_camera) is asserted on exactly once rather than however many
    times a tight polling loop happens to spin before the test tears it
    down."""
    monkeypatch.setattr(booth_manager_module, "HOTPLUG_SCAN_INTERVAL_SEC", 0.01)
    calls = []

    def fake_discover_cameras(max_index, skip):
        calls.append(frozenset(skip))
        return list(found) if len(calls) == 1 else []

    monkeypatch.setattr(booth_manager_module, "discover_cameras", fake_discover_cameras)

    thread = threading.Thread(target=bm._hotplug_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while not calls and time.monotonic() < deadline:
        time.sleep(0.01)
    # Give the loop a beat to finish acting on the first `found` result
    # (add_camera/request_immediate_retry) before tearing it down.
    time.sleep(0.1)
    bm._hotplug_stop.set()
    thread.join(timeout=2)
    return calls


def test_hotplug_nudges_disconnected_known_camera_instead_of_duplicating(tmp_path, monkeypatch):
    disconnected = _FakeWorker(capture_open=False)
    bm = _bare_booth_manager(tmp_path, {"CAM-1": 0}, {"CAM-1": disconnected})

    calls = _run_one_hotplug_tick(monkeypatch, bm, found=[0])

    assert calls, "discover_cameras was never called"
    assert 0 not in calls[0]  # a disconnected known camera's index must be probed, not skipped
    assert disconnected.retry_calls == 1
    assert bm.add_camera_calls == []  # must never create a duplicate camera_id for the same device


def test_hotplug_never_probes_index_of_a_live_camera(tmp_path, monkeypatch):
    live = _FakeWorker(capture_open=True)
    bm = _bare_booth_manager(tmp_path, {"CAM-1": 0}, {"CAM-1": live})

    calls = _run_one_hotplug_tick(monkeypatch, bm, found=[])

    assert calls, "discover_cameras was never called"
    assert 0 in calls[0]  # a live camera's index must stay skipped, never re-probed
    assert live.retry_calls == 0


def test_hotplug_adds_a_genuinely_new_camera(tmp_path, monkeypatch):
    live = _FakeWorker(capture_open=True)
    bm = _bare_booth_manager(tmp_path, {"CAM-1": 0}, {"CAM-1": live})

    calls = _run_one_hotplug_tick(monkeypatch, bm, found=[5])

    assert calls
    assert 5 not in calls[0]  # never owned by anyone, so never skipped either
    assert bm.add_camera_calls == [5]
    assert live.retry_calls == 0  # unrelated existing camera must be untouched


def test_hotplug_ignores_manually_removed_usb_index(tmp_path, monkeypatch):
    """An index the operator explicitly removed via /settings
    (BoothManager._ignored_usb_indices — see remove_camera) must stay
    excluded from both the "new camera" and the "reconnect nudge" paths."""
    bm = _bare_booth_manager(tmp_path, {}, {})
    bm._ignored_usb_indices = {3}

    calls = _run_one_hotplug_tick(monkeypatch, bm, found=[])

    assert calls
    assert 3 in calls[0]
    assert bm.add_camera_calls == []
