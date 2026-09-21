"""capture_process - run ONE camera in its OWN OS process with a watchdog.

Problem it solves: cv2.VideoCapture.read() blocks with no timeout on DirectShow. When one
USB camera stalls (0xC00D3704 / 0x8007001F) the blocked thread cannot be cancelled and, with
all cameras sharing one process + DIRECTSHOW_LOCK, it freezes the others and cannot be
recovered without restarting the app. Here the child process owns the device; the parent
only reads a shared-memory slot. If the child stops publishing frames (heartbeat stale), the
parent kills the child (TerminateProcess releases the driver handle) and respawns it.

    cap = CaptureProcess("core.capture_sim:make_fake", {"mode": "ok"}, name="cam1")
    cap.start()
    ok, frame, seq, ts = cap.read()      # never blocks
    cap.stop()

`factory` is "package.module:function"; it must return an object with
    read() -> (ok: bool, frame|None)   and   release()
Child imports it by name (spawn-safe on Windows). For real cameras use
`core.capture_sim:make_cv2` style factories in your own module.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, Optional, Tuple

import numpy as np

_HDR = struct.Struct("<QdiiI")  # seq, ts, h, w, state
_HDR_SIZE = 64
STATE_INIT, STATE_RUN, STATE_FAIL = 0, 1, 2

# Shared with core.vision._open_capture's cross-process serialization (see that function's
# docstring): a plain in-process threading.RLock (core.camera_identity.DIRECTSHOW_LOCK) cannot
# serialize a DirectShow/MSMF open happening in THIS module's child process against one happening
# in the parent (or another child) — they are different OS processes with independent memory. This
# multiprocessing.Lock is created once, here, at import time in the parent process and passed
# explicitly (via CaptureProcess kwargs, pickled through Process(args=...) at spawn time, not by a
# fresh child-side import of this module) to every escalated camera's real-capture factory, so an
# escalated camera's open and every other camera's open — thread- or process-isolated alike — still
# serialize against each other exactly as they did before any camera was ever escalated.
CROSS_PROCESS_OPEN_LOCK = mp.get_context("spawn").Lock()

# Whether any camera has EVER escalated to process isolation in this run. A multiprocessing.Lock
# is a real OS semaphore -- meaningfully slower per acquire/release than the plain in-process
# threading.RLock _open_capture already used -- so _open_capture only pays that cost once a second
# OS process genuinely exists to serialize against. Before the first escalation (the overwhelming
# common case: a healthy camera fleet that never needs process isolation at all), every open is
# exactly as fast as it was before CROSS_PROCESS_OPEN_LOCK existed. Set once escalation happens and
# never cleared: an escalated camera is never de-escalated (see core.vision.CameraWorker's own
# docstring on that design choice), so there is no point this flag would need to go back to False.
_cross_process_lock_needed = threading.Event()


def note_process_capture_starting() -> None:
    """Called by CaptureProcess.start() -- the exact moment a second OS process comes into
    existence that could open a camera concurrently with this one. See _cross_process_lock_needed."""
    _cross_process_lock_needed.set()


def cross_process_lock_active() -> bool:
    return _cross_process_lock_needed.is_set()


def _load(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


def _child_main(shm_name: str, cap_bytes: int, factory: str, kwargs: Dict[str, Any], stop_evt, fail_limit: int):
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    seq = 0
    try:
        cap = _load(factory)(**kwargs)
    except Exception:
        _HDR.pack_into(buf, 0, 0, time.time(), 0, 0, STATE_FAIL)
        return
    fails = 0
    try:
        while not stop_evt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                _HDR.pack_into(buf, 0, seq, time.time(), 0, 0, STATE_FAIL if fails >= fail_limit else STATE_RUN)
                if fails >= fail_limit:
                    return  # let the parent respawn us (fresh handle)
                time.sleep(0.02)
                continue
            fails = 0
            h, w = frame.shape[:2]
            n = h * w * 3
            if n > cap_bytes:
                continue
            seq += 1
            _HDR.pack_into(buf, 0, seq * 2 - 1, time.time(), h, w, STATE_RUN)  # odd => writing
            buf[_HDR_SIZE:_HDR_SIZE + n] = np.ascontiguousarray(frame).tobytes()
            _HDR.pack_into(buf, 0, seq * 2, time.time(), h, w, STATE_RUN)      # even => stable
    finally:
        try:
            cap.release()
        except Exception:
            pass
        del buf
        shm.close()


class CaptureProcess:
    def __init__(self, factory: str, kwargs: Optional[Dict[str, Any]] = None, name: str = "cam",
                 max_pixels: int = 1920 * 1080, stale_sec: float = 3.0, startup_grace_sec: float = 8.0,
                 backoff=(0.5, 1, 2, 4, 8), fail_limit: int = 20):
        self.factory, self.kwargs, self.name = factory, dict(kwargs or {}), name
        self._cap_bytes = max_pixels * 3
        self.stale_sec, self.startup_grace_sec = stale_sec, startup_grace_sec
        self.backoff, self.fail_limit = backoff, fail_limit
        self._ctx = mp.get_context("spawn")
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._proc = None
        self._stop = self._ctx.Event()
        self._wd: Optional[threading.Thread] = None
        self._running = False
        self.respawns = 0
        self.kills = 0
        self._spawned_at = 0.0
        self._last_seq = 0
        self._last_change = 0.0
        self._consec_fail = 0
        self._lock = threading.Lock()

    # ---- lifecycle ---------------------------------------------------------
    def _spawn(self):
        self._stop = self._ctx.Event()
        _HDR.pack_into(self._shm.buf, 0, 0, 0.0, 0, 0, STATE_INIT)
        self._proc = self._ctx.Process(target=_child_main, daemon=True, name=f"cap-{self.name}",
                                       args=(self._shm.name, self._cap_bytes, self.factory, self.kwargs,
                                             self._stop, self.fail_limit))
        self._proc.start()
        self._spawned_at = self._last_change = time.time()
        self._last_seq = 0

    def start(self):
        if self._running:
            return
        note_process_capture_starting()
        self._shm = shared_memory.SharedMemory(create=True, size=_HDR_SIZE + self._cap_bytes)
        self._running = True
        self._spawn()
        self._wd = threading.Thread(target=self._watchdog, daemon=True, name=f"wd-{self.name}")
        self._wd.start()

    def _kill(self):
        p = self._proc
        if p is None:
            return
        self._stop.set()
        p.join(timeout=0.3)
        if p.is_alive():
            p.terminate(); p.join(timeout=1.0)
        if p.is_alive():
            p.kill(); p.join(timeout=1.0)
        self.kills += 1

    def _watchdog(self):
        while self._running:
            time.sleep(0.1)
            try:
                seq, ts, h, w, state = _HDR.unpack_from(self._shm.buf, 0)
            except Exception:
                return
            now = time.time()
            if seq != self._last_seq:
                self._last_seq, self._last_change = seq, now
                self._consec_fail = 0
            dead = self._proc is None or not self._proc.is_alive()
            limit = self.startup_grace_sec if self._last_seq == 0 else self.stale_sec
            stale = (now - self._last_change) > limit
            if dead or stale or state == STATE_FAIL:
                with self._lock:
                    if not self._running:
                        return
                    if not dead:
                        self._kill()
                    delay = self.backoff[min(self._consec_fail, len(self.backoff) - 1)]
                    self._consec_fail += 1
                    t_end = time.time() + delay
                while time.time() < t_end and self._running:
                    time.sleep(0.05)
                with self._lock:
                    if self._running:
                        self.respawns += 1
                        self._spawn()

    def stop(self):
        with self._lock:
            self._running = False
        if self._proc is not None:
            self._kill()
        if self._wd:
            self._wd.join(timeout=2.0)
        if self._shm is not None:
            try:
                self._shm.close(); self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    # ---- consumer API --------------------------------------------------------
    def read(self, max_age_sec: Optional[float] = None) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Never blocks. Returns (ok, frame_copy|None, seq, capture_ts). ok=False if nothing
        fresh (older than max_age_sec, default stale_sec)."""
        if self._shm is None:
            return False, None, 0, 0.0
        max_age = self.stale_sec if max_age_sec is None else max_age_sec
        for _ in range(3):  # seqlock retry
            s1, ts, h, w, state = _HDR.unpack_from(self._shm.buf, 0)
            if s1 == 0 or s1 % 2 == 1 or h <= 0 or w <= 0:
                time.sleep(0.001); continue
            n = h * w * 3
            frame = np.frombuffer(self._shm.buf, np.uint8, n, _HDR_SIZE).reshape(h, w, 3).copy()
            s2 = _HDR.unpack_from(self._shm.buf, 0)[0]
            if s1 == s2:
                if time.time() - ts > max_age:
                    return False, None, s1 // 2, ts
                return True, frame, s1 // 2, ts
        return False, None, 0, 0.0

    @property
    def alive(self) -> bool:
        return bool(self._proc and self._proc.is_alive())

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None
