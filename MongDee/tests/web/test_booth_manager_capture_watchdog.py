"""BoothManager's capture watchdog thread calls each worker's recover_from_hung_read()."""
from __future__ import annotations

import threading
import time

from web import booth_manager
from web.booth_manager import BoothManager


class _Worker:
    def __init__(self, boom=False):
        self.calls = 0
        self.boom = boom

    def recover_from_hung_read(self):
        self.calls += 1
        if self.boom:
            raise RuntimeError("boom")
        return False


class _OldWorker:          # workers without the method (older fakes) must be tolerated
    pass


def _bare():
    bm = object.__new__(BoothManager)
    bm._lock = threading.Lock()
    return bm


def test_watchdog_polls_every_worker_and_survives_errors_and_missing_method(monkeypatch):
    monkeypatch.setattr(booth_manager, "CAPTURE_WATCHDOG_INTERVAL_SEC", 0.01)
    bm = _bare()
    good, bad = _Worker(), _Worker(boom=True)
    bm.workers = {"CAM-1": good, "CAM-2": bad, "CAM-3": _OldWorker()}
    stop = threading.Event()
    t = threading.Thread(target=bm._capture_watchdog_loop, args=(stop,), daemon=True)
    t.start()
    time.sleep(0.25)
    stop.set(); t.join(2.0)
    assert not t.is_alive()
    assert good.calls >= 3 and bad.calls >= 3          # kept polling after the exception
