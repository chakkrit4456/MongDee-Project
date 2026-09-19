"""device_presence - notice a USB camera being unplugged / plugged in within about a second, WITHOUT opening it.

Why this exists: a worker that only learns about a dead camera through cap.read() finds out late (a
DirectShow read can block, or keep returning the last frame), and one that only tries to reopen on a
growing back-off (up to 30 s) finds out about a replug late. Enumerating the DirectShow device list is
cheap and never touches a camera that is streaming, so a small monitor thread can:

  * tell every worker "a device just disappeared" (they then hold the stream to a much stricter
    deadline instead of waiting for their own timeouts), and
  * tell every disconnected worker "a device just appeared" so it reopens NOW instead of at its next
    back-off tick, and
  * answer `is_present(index)` so a worker whose device is really gone does not burn 5-15 s of the
    process-wide DirectShow lock on futile open attempts (which also starved the other cameras).

Safety rules baked in:
  * "could not look" (enumerator unavailable, lock busy) is NEVER treated as "device gone";
  * a disappearance must be seen in `remove_confirmations` consecutive polls before it is believed
    (an appearance is believed at once) so one bad COM enumeration cannot drop a healthy camera;
  * no enumerator (non-Windows, no pygrabber) -> the monitor is inert and `is_present` returns None,
    which callers treat as "unknown" and fall back to the old behaviour.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("mongdee.core.device_presence")

DEFAULT_POLL_INTERVAL_SEC = 0.5
DEFAULT_REMOVE_CONFIRMATIONS = 2


@dataclass(frozen=True)
class PresenceEvent:
    devices: dict = field(default_factory=dict)      # index -> friendly name, after the change
    added: tuple = ()                                # friendly names that appeared
    removed: tuple = ()                              # friendly names that disappeared


def _default_enumerator():
    from core.camera_identity import list_directshow_devices_nowait
    return list_directshow_devices_nowait()


class DevicePresenceMonitor:
    def __init__(self, enumerate_fn: "Callable[[], dict[int, str] | None] | None" = None,
                 interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
                 remove_confirmations: int = DEFAULT_REMOVE_CONFIRMATIONS):
        self._enumerate = enumerate_fn or _default_enumerator
        self.interval_sec = interval_sec
        self.remove_confirmations = max(1, int(remove_confirmations))
        self._lock = threading.Lock()
        self._stable: "dict[int, str] | None" = None
        self._pending: "dict[int, str] | None" = None
        self._pending_count = 0
        self._listeners: list = []
        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None
        self.polls = 0
        self.events = 0

    # ---------------------------------------------------------------- queries
    def snapshot(self) -> "dict[int, str] | None":
        with self._lock:
            return None if self._stable is None else dict(self._stable)

    def is_present(self, device) -> "bool | None":
        """True / False when `device` is a camera index and the device list is known; None otherwise
        (path/URL devices, no enumerator yet, enumeration unavailable) - callers treat None as 'unknown'."""
        from core.camera_identity import device_as_index
        index = device_as_index(device)
        if index is None:
            return None
        with self._lock:
            return None if self._stable is None else index in self._stable

    def name_at(self, device) -> "str | None":
        from core.camera_identity import device_as_index
        index = device_as_index(device)
        with self._lock:
            return None if (index is None or self._stable is None) else self._stable.get(index)

    def add_listener(self, fn: "Callable[[PresenceEvent], None]") -> None:
        self._listeners.append(fn)

    # ------------------------------------------------------------------ core
    def poll_once(self) -> "PresenceEvent | None":
        """One enumeration + state update. Returns the event it emitted (or None). Public so tests and
        callers can drive it synchronously."""
        self.polls += 1
        try:
            devices = self._enumerate()
        except Exception:
            logger.debug("presence enumeration raised", exc_info=True)
            return None
        if devices is None:
            return None                                   # could not look: no information
        devices = dict(devices)
        event = None
        with self._lock:
            if self._stable is None:                      # first look: baseline, nothing "changed"
                self._stable = devices
                return None
            old_names, new_names = Counter(self._stable.values()), Counter(devices.values())
            added = tuple((new_names - old_names).elements())
            removed = tuple((old_names - new_names).elements())
            if added:                                     # believe an appearance immediately
                self._accept(devices)
                event = PresenceEvent(dict(devices), added, removed)
            elif removed:                                 # a disappearance must be seen repeatedly
                if self._pending == devices:
                    self._pending_count += 1
                else:
                    self._pending, self._pending_count = devices, 1
                if self._pending_count >= self.remove_confirmations:
                    self._accept(devices)
                    event = PresenceEvent(dict(devices), (), removed)
            else:
                self._pending, self._pending_count = None, 0
                if devices != self._stable:               # same cameras, different indices: just track it
                    self._stable = devices
        if event is not None:
            self.events += 1
            self._emit(event)
        return event

    def _accept(self, devices: dict) -> None:
        self._stable = devices
        self._pending, self._pending_count = None, 0

    def _emit(self, event: PresenceEvent) -> None:
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:
                logger.exception("device presence listener failed")

    # ---------------------------------------------------------------- thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mongdee-device-presence")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            try:
                self.poll_once()
            except Exception:
                logger.exception("device presence poll failed")
            if self._stop.wait(self.interval_sec):
                return
