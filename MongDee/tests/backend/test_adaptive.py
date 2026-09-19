"""Unit tests for backend/adaptive.py — the AI Server's Adaptive
Performance Engine, adapted from core/performance.py's AdaptiveController
to vision.pipeline.DetectionPipeline's process-wide knobs (target FPS,
detector input size shared across every camera).
"""

from __future__ import annotations

import dataclasses

from backend.adaptive import PipelineAdaptiveController
from core.performance import PerformanceMonitor


@dataclasses.dataclass
class _FakeStats:
    last_latency_sec: float = 0.0


class _FakePipeline:
    def __init__(self, target_fps=6.0, stats=None):
        self._target_fps = target_fps
        self._stats = stats or {}

    def get_target_fps(self):
        return self._target_fps

    def set_target_fps(self, fps):
        self._target_fps = fps

    def all_stats(self):
        return self._stats


class _FakeDetector:
    def __init__(self, imgsz=640):
        self._imgsz = imgsz

    def get_imgsz(self):
        return self._imgsz

    def set_imgsz(self, size):
        self._imgsz = size


def _quiet_monitor():
    monitor = PerformanceMonitor()
    monitor.system_snapshot = lambda: {  # type: ignore[method-assign]
        "cpu_percent": 5.0, "ram_percent": 10.0, "ram_used_gb": 1.0, "gpu_vram_used_gb": None,
    }
    return monitor


def _loud_monitor():
    monitor = PerformanceMonitor()
    monitor.system_snapshot = lambda: {  # type: ignore[method-assign]
        "cpu_percent": 99.0, "ram_percent": 10.0, "ram_used_gb": 1.0, "gpu_vram_used_gb": None,
    }
    return monitor


def test_reduces_fps_before_resolution_when_cpu_high():
    pipeline = _FakePipeline(target_fps=6.0)
    detector = _FakeDetector(imgsz=640)
    controller = PipelineAdaptiveController(pipeline, detector, _loud_monitor(), min_target_fps=1.0)

    controller.tick()
    assert pipeline.get_target_fps() < 6.0
    assert detector.get_imgsz() == 640  # untouched while FPS still has headroom


def test_reduces_resolution_once_fps_is_at_floor():
    pipeline = _FakePipeline(target_fps=1.0)  # already at the floor
    detector = _FakeDetector(imgsz=640)
    controller = PipelineAdaptiveController(pipeline, detector, _loud_monitor(), min_target_fps=1.0,
                                             imgsz_levels=(640, 480, 320))

    controller.tick()
    assert pipeline.get_target_fps() == 1.0
    assert detector.get_imgsz() == 480


def test_high_latency_alone_triggers_backoff_even_with_low_cpu():
    pipeline = _FakePipeline(target_fps=6.0, stats={"CAM01": _FakeStats(last_latency_sec=0.5)})
    detector = _FakeDetector(imgsz=640)
    controller = PipelineAdaptiveController(pipeline, detector, _quiet_monitor(),
                                             latency_high_ms=300.0, min_target_fps=1.0)

    controller.tick()
    assert pipeline.get_target_fps() < 6.0


def test_restores_resolution_before_fps_once_load_is_normal():
    pipeline = _FakePipeline(target_fps=3.0)
    detector = _FakeDetector(imgsz=480)
    controller = PipelineAdaptiveController(pipeline, detector, _quiet_monitor(),
                                             max_target_fps=6.0, min_target_fps=1.0,
                                             imgsz_levels=(640, 480, 320))

    controller.tick()
    assert detector.get_imgsz() == 640  # resolution restores first
    assert pipeline.get_target_fps() == 3.0

    controller.tick()
    assert pipeline.get_target_fps() > 3.0  # then FPS climbs back up


def test_never_reduces_fps_below_configured_minimum():
    pipeline = _FakePipeline(target_fps=1.0)
    detector = _FakeDetector(imgsz=320)  # already at the lowest resolution too
    controller = PipelineAdaptiveController(pipeline, detector, _loud_monitor(), min_target_fps=1.0,
                                             imgsz_levels=(640, 480, 320))

    controller.tick()
    controller.tick()
    assert pipeline.get_target_fps() == 1.0
    assert detector.get_imgsz() == 320


def test_never_exceeds_configured_maximum_fps():
    pipeline = _FakePipeline(target_fps=6.0)
    detector = _FakeDetector(imgsz=640)  # already at the highest resolution too
    controller = PipelineAdaptiveController(pipeline, detector, _quiet_monitor(), max_target_fps=6.0)

    controller.tick()
    controller.tick()
    assert pipeline.get_target_fps() == 6.0
