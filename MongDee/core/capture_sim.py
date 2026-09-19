"""Fault-injection capture doubles + factory for real cv2 devices (used by CaptureProcess).
No hardware needed: modes reproduce the failures seen on the real USB cameras."""
from __future__ import annotations
import os
import time
import numpy as np


class FakeCapture:
    """mode: ok | hang_after | fail_forever | intermittent | garbage | slow | crash_after
    A file `flag_path` (optional) lets a test flip a hung camera back to healthy after respawn:
    if it exists, the camera is healthy."""

    def __init__(self, mode="ok", after=10, fps=30.0, size=(240, 320), flag_path=None, marker=0):
        self.mode, self.after, self.fps, self.size = mode, after, fps, size
        self.n = 0
        self.marker = marker
        self.flag_path = flag_path
        self.t_last = 0.0
        self._rng = np.random.default_rng(marker)
        self._healthy = bool(flag_path and os.path.exists(flag_path))

    def _frame(self):
        h, w = self.size
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        ph = self.n * 0.15
        b = 128 + 90 * np.sin(x / 23.0 + ph) * np.cos(y / 31.0)
        g = 128 + 80 * np.sin((x + y) / 41.0 - ph)
        r = 110 + 60 * np.cos(x / 57.0 + y / 19.0 + self.marker)
        img = np.stack([b, g, r], -1) + self._rng.normal(0, 3, (h, w, 3))
        img = np.clip(img, 0, 255).astype(np.uint8)
        img[:, :4] = self.marker % 256   # per-camera signature stripe
        return img

    def read(self):
        dt = 1.0 / self.fps
        d = time.time() - self.t_last
        if d < dt:
            time.sleep(dt - d)
        self.t_last = time.time()
        self.n += 1
        m = self.mode
        if self._healthy:
            m = "ok"
        if m == "hang_after" and self.n > self.after:
            time.sleep(3600)          # blocking forever, exactly like a stalled DirectShow read
        if m == "crash_after" and self.n > self.after:
            os._exit(3)
        if m == "fail_forever":
            return False, None
        if m == "intermittent" and self.n % 3 == 0:
            return False, None
        if m == "garbage" and self.n % 2 == 0:
            return True, self._rng.integers(0, 256, (*self.size, 3), dtype=np.uint8)
        if m == "slow":
            time.sleep(0.3)
        return True, self._frame()

    def release(self):
        pass


def make_fake(**kw):
    return FakeCapture(**kw)


def make_cv2(index=0, width=320, height=240, fps=15, backend="dshow", fourcc="MJPG"):
    """Real device factory (Windows). Runs inside the child process only."""
    import cv2
    api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")

    class _W:
        def read(self_):
            return cap.read()
        def release(self_):
            cap.release()
    return _W()
