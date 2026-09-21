"""Qt adapter for core.vision.CameraWorker.

core/vision.py is intentionally framework-agnostic (plain threading +
callbacks) so the web server can use it directly. The PySide6 desktop UI
still wants Qt Signals (so cross-thread delivery onto the GUI thread is
handled automatically by Qt's queued connections) — this class wraps a
plain CameraWorker and re-exposes its three callbacks as signals, keeping
the exact same interface main_window.py/trainer_window.py already use.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal

from core.vision import CameraWorker as _PlainCameraWorker

UI_MAX_FPS = 20
UI_FRAME_INTERVAL_SEC = 1.0 / UI_MAX_FPS


class QtCameraWorker(QObject):
    frame_ready = Signal(str, object)          # camera_id, BGR ndarray (with boxes drawn)
    detections_ready = Signal(str, list)       # camera_id, [{"class_name","conf","bbox"}]
    status_changed = Signal(str, str, str)     # camera_id, status ("online"/"offline"), message

    def __init__(self, camera_id: str, device, model, allowed_classes: list[str],
                 recognizer=None, conf_threshold: float = 0.45, device_target="cpu",
                 gender_age_backend=None, performance_monitor=None,
                 parent=None):
        super().__init__(parent)
        self._last_frame_emit = 0.0
        self._worker = _PlainCameraWorker(
            camera_id=camera_id,
            device=device,
            model=model,
            allowed_classes=allowed_classes,
            recognizer=recognizer,
            conf_threshold=conf_threshold,
            device_target=device_target,
            gender_age_backend=gender_age_backend,
            performance_monitor=performance_monitor,
            on_frame=self._emit_frame,
            on_detections=self.detections_ready.emit,
            on_status=self.status_changed.emit,
        )

    def _emit_frame(self, camera_id, frame):
        """Bound queued UI updates so multiple cameras cannot build a stale
        frame backlog when detection or painting is slower than capture."""
        now = time.monotonic()
        if now - self._last_frame_emit < UI_FRAME_INTERVAL_SEC:
            return
        self._last_frame_emit = now
        self.frame_ready.emit(camera_id, frame)

    @property
    def camera_id(self) -> str:
        return self._worker.camera_id

    def start(self):
        self._worker.start()

    def stop(self):
        self._worker.stop()

    def set_enabled(self, enabled: bool):
        self._worker.set_enabled(enabled)

    # ---------------------------------------------- adaptive control knobs
    # Implements core.performance.ManagedCamera, same as the plain
    # CameraWorker it wraps — see core/vision.py.
    def get_detect_every_n_frames(self) -> int:
        return self._worker.get_detect_every_n_frames()

    def set_detect_every_n_frames(self, n: int) -> None:
        self._worker.set_detect_every_n_frames(n)

    def get_ai_imgsz(self) -> int:
        return self._worker.get_ai_imgsz()

    def set_ai_imgsz(self, size: int) -> None:
        self._worker.set_ai_imgsz(size)
