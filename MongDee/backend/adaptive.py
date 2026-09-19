"""Adaptive Performance Engine for the AI Server (backend/main.py) — brings
the same principles as core/performance.py's AdaptiveController (built for
the desktop/web app's per-camera CameraWorker pipeline) to this pipeline,
adapted to its actual knobs.

vision/pipeline.py's DetectionPipeline shares ONE detector/tracker across
every camera on a CPU-only box (see its module docstring) — so
target_fps_per_camera and the detector's YOLO input size are both
PROCESS-WIDE settings here, not per-camera. That is different enough from
core.performance.AdaptiveController's per-camera ManagedCamera protocol
that forcing a fit through it would be more confusing than reusing it
cleanly, so this is a small, dedicated controller instead — but it reuses
core.performance.PerformanceMonitor.system_snapshot() as-is for CPU/RAM,
since that part fits perfectly unchanged.

Priority order (MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md
section 36): reduce AI FPS first, then AI input resolution — restoring in
the opposite order once load is back to normal. Never touches display/
stream FPS, which is a separate, per-viewer concern the API's /stream
endpoint owns (section 93 rule: don't sacrifice live display for AI
headroom without cause).
"""

from __future__ import annotations

import logging
import threading

from core.performance import PerformanceMonitor

logger = logging.getLogger("mongdee.backend.adaptive")

DEFAULT_TICK_INTERVAL_SEC = 2.0
DEFAULT_CPU_HIGH_PCT = 85.0
DEFAULT_LATENCY_HIGH_MS = 300.0
DEFAULT_MIN_TARGET_FPS = 1.0
DEFAULT_IMGSZ_LEVELS = (640, 480, 320)


class PipelineAdaptiveController(threading.Thread):
    def __init__(
        self,
        pipeline,
        detector,
        monitor: PerformanceMonitor,
        *,
        tick_interval_sec: float = DEFAULT_TICK_INTERVAL_SEC,
        cpu_high_pct: float = DEFAULT_CPU_HIGH_PCT,
        latency_high_ms: float = DEFAULT_LATENCY_HIGH_MS,
        min_target_fps: float = DEFAULT_MIN_TARGET_FPS,
        max_target_fps: float | None = None,
        imgsz_levels: tuple[int, ...] = DEFAULT_IMGSZ_LEVELS,
    ):
        super().__init__(daemon=True, name="mongdee-backend-adaptive")
        self._pipeline = pipeline
        self._detector = detector
        self._monitor = monitor
        self._tick_interval_sec = tick_interval_sec
        self._cpu_high_pct = cpu_high_pct
        self._latency_high_ms = latency_high_ms
        self._min_fps = min_target_fps
        self._max_fps = max_target_fps if max_target_fps is not None else pipeline.get_target_fps()
        self._fps_step = max(0.5, (self._max_fps - self._min_fps) / 6) if self._max_fps > self._min_fps else 1.0
        self._imgsz_levels = tuple(sorted(set(int(s) for s in imgsz_levels), reverse=True)) or (detector.get_imgsz(),)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)

    def run(self) -> None:
        while not self._stop_event.wait(self._tick_interval_sec):
            try:
                self.tick()
            except Exception:
                logger.exception("adaptive tick failed")

    def tick(self) -> None:
        """One adjustment pass — public so tests/a benchmark run can drive
        it deterministically instead of waiting on the background timer."""
        system = self._monitor.system_snapshot()
        cpu_pct = system.get("cpu_percent")
        cpu_high = cpu_pct is not None and cpu_pct >= self._cpu_high_pct

        # If ANY camera is struggling, back off process-wide — one shared
        # detector serves every camera, so a single slow camera's latency
        # is evidence the whole pipeline is behind, not just that camera.
        all_stats = self._pipeline.all_stats()
        max_latency_ms = max((s.last_latency_sec * 1000.0 for s in all_stats.values()), default=0.0)
        overloaded = cpu_high or max_latency_ms >= self._latency_high_ms

        fps = self._pipeline.get_target_fps()
        imgsz = self._detector.get_imgsz()
        min_imgsz, max_imgsz = min(self._imgsz_levels), max(self._imgsz_levels)

        if overloaded:
            if fps > self._min_fps:
                new_fps = max(self._min_fps, fps - self._fps_step)
                self._pipeline.set_target_fps(new_fps)
                logger.info("[PERF] reducing AI FPS (%.1f -> %.1f)", fps, new_fps)
            elif imgsz > min_imgsz:
                lower = max((s for s in self._imgsz_levels if s < imgsz), default=imgsz)
                self._detector.set_imgsz(lower)
                logger.info("[PERF] reducing AI input resolution (%dpx -> %dpx)", imgsz, lower)
        else:
            if imgsz < max_imgsz:
                higher = min((s for s in self._imgsz_levels if s > imgsz), default=imgsz)
                self._detector.set_imgsz(higher)
                logger.info("[PERF] restoring AI input resolution (%dpx -> %dpx)", imgsz, higher)
            elif fps < self._max_fps:
                new_fps = min(self._max_fps, fps + self._fps_step)
                self._pipeline.set_target_fps(new_fps)
                logger.info("[PERF] restoring AI FPS (%.1f -> %.1f)", fps, new_fps)
