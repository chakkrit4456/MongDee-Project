"""Unit tests for core/performance.py — all synthetic (no real camera/GPU
needed), since PerformanceMonitor/AdaptiveController only consume numbers
handed to them and detect_hardware() degrades gracefully without psutil/CUDA.
"""

from __future__ import annotations

import time

import pytest

from core.performance import (
    AdaptiveConfig,
    AdaptiveController,
    HardwareProfile,
    PerformanceMonitor,
    detect_hardware,
    format_benchmark_report,
    load_adaptive_config,
)


# ------------------------------------------------------------- hardware

def test_detect_hardware_never_raises_and_returns_sane_defaults():
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_logical_cores >= 1
    assert hw.tier in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(hw.summary_line(), str) and hw.summary_line()


# --------------------------------------------------------------- monitor

def test_camera_snapshot_defaults_for_unknown_camera():
    monitor = PerformanceMonitor()
    snap = monitor.camera_snapshot("NOPE")
    assert snap["captured_frames"] == 0
    assert snap["actual_fps"] == 0.0
    assert snap["resolution"] is None


def test_record_capture_tracks_fps_and_drop_rate():
    monitor = PerformanceMonitor()
    monitor.set_requested_fps("CAM-1", 30.0)
    now = time.time()
    # 10 successful captures spaced 0.1s apart (~10fps), then 2 drops.
    for i in range(10):
        monitor.record_capture("CAM-1", capture_ts=now - 1.0 + i * 0.1)
    monitor.record_dropped("CAM-1")
    monitor.record_dropped("CAM-1")

    snap = monitor.camera_snapshot("CAM-1")
    assert snap["captured_frames"] == 10
    assert snap["dropped_frames"] == 2
    assert snap["requested_fps"] == 30.0
    assert 5 < snap["actual_fps"] < 15  # ~10fps within the rolling window
    assert 0 < snap["drop_rate"] < 1


def test_record_display_tracks_latency():
    monitor = PerformanceMonitor()
    capture_ts = time.time() - 0.05
    monitor.record_display("CAM-1", capture_ts=capture_ts)
    snap = monitor.camera_snapshot("CAM-1")
    assert 30 <= snap["avg_display_latency_ms"] <= 200


def test_record_ai_tracks_ai_fps_and_latency():
    monitor = PerformanceMonitor()
    now = time.time()
    for i in range(4):
        monitor.record_ai("CAM-1", latency_sec=0.02)
    snap = monitor.camera_snapshot("CAM-1")
    assert snap["avg_ai_latency_ms"] == pytest.approx(20.0, abs=1.0)


def test_forget_camera_removes_it_from_snapshot():
    monitor = PerformanceMonitor()
    monitor.record_capture("CAM-1")
    assert "CAM-1" in monitor.camera_ids()
    monitor.forget_camera("CAM-1")
    assert "CAM-1" not in monitor.camera_ids()


def test_memory_is_bounded_regardless_of_event_count():
    monitor = PerformanceMonitor()
    for _ in range(10_000):
        monitor.record_capture("CAM-1")
    with monitor._lock:
        st = monitor._cameras["CAM-1"]
        assert len(st.events) <= 300  # MAX_SAMPLES_PER_CAMERA — never grows unbounded


def test_system_snapshot_never_raises():
    monitor = PerformanceMonitor()
    snap = monitor.system_snapshot()
    assert "cpu_percent" in snap and "ram_percent" in snap


# ------------------------------------------------------------- benchmark

def test_format_benchmark_report_contains_camera_and_system_lines():
    monitor = PerformanceMonitor()
    monitor.set_requested_fps("CAM-1", 24.0)
    monitor.set_resolution("CAM-1", (1280, 720))
    for i in range(5):
        monitor.record_capture("CAM-1", capture_ts=time.time() - 0.5 + i * 0.1)
    hw = detect_hardware()
    report = format_benchmark_report(hw, monitor, elapsed_sec=12.3)
    assert "MongDee Camera Benchmark" in report
    assert "CAM-1" in report
    assert "1280x720" in report
    assert "CPU:" in report and "RAM:" in report


# ------------------------------------------------------- adaptive config

def test_adaptive_config_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError):
        AdaptiveConfig.from_dict({"not_a_real_field": 1})


def test_adaptive_config_from_dict_normalizes_imgsz_levels():
    cfg = AdaptiveConfig.from_dict({"ai_imgsz_levels": [320, 640, 480, 640]})
    assert cfg.ai_imgsz_levels == (640, 480, 320)


def test_load_adaptive_config_reads_json_file(tmp_path):
    path = tmp_path / "performance.json"
    path.write_text('{"cpu_high_pct": 70, "max_detect_every_n_frames": 8}', encoding="utf-8")
    cfg = load_adaptive_config(path)
    assert cfg.cpu_high_pct == 70
    assert cfg.max_detect_every_n_frames == 8
    assert cfg.tick_interval_sec == AdaptiveConfig().tick_interval_sec  # unspecified fields keep defaults


def test_load_adaptive_config_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        load_adaptive_config(tmp_path / "does_not_exist.json")


def test_adaptive_config_from_dict_ignores_underscore_prefixed_comment_keys():
    cfg = AdaptiveConfig.from_dict({"_comment": "explanation text", "cpu_high_pct": 70})
    assert cfg.cpu_high_pct == 70


def test_shipped_performance_example_config_loads_successfully():
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent.parent / "configs" / "performance.example.json"
    cfg = load_adaptive_config(example)
    assert cfg.max_detect_every_n_frames == 8
    assert cfg.ai_imgsz_levels == (640, 480, 320)
    assert cfg.cpu_pause_pct == 95.0
    assert cfg.cpu_resume_pct == 70.0


# ---------------------------------------------------------- fake camera

class _FakeManagedCamera:
    """Stand-in for core.vision.CameraWorker's adaptive-control surface —
    lets AdaptiveController be tested without opening any real camera."""

    def __init__(self, camera_id: str, detect_every_n_frames: int = 1, ai_imgsz: int = 640):
        self.camera_id = camera_id
        self._n = detect_every_n_frames
        self._imgsz = ai_imgsz

    def get_detect_every_n_frames(self) -> int:
        return self._n

    def set_detect_every_n_frames(self, n: int) -> None:
        self._n = n

    def get_ai_imgsz(self) -> int:
        return self._imgsz

    def set_ai_imgsz(self, size: int) -> None:
        self._imgsz = size


def _quiet_system_snapshot(monitor: PerformanceMonitor) -> None:
    """AdaptiveController.tick() reads real CPU load via
    PerformanceMonitor.system_snapshot() — pin it to a low, deterministic
    value so these tests exercise AI-latency-driven decisions only,
    independent of whatever the real test machine's CPU is doing."""
    monitor.system_snapshot = lambda: {  # type: ignore[method-assign]
        "cpu_percent": 5.0, "ram_percent": 10.0, "ram_used_gb": 1.0, "gpu_vram_used_gb": None,
    }


def _overloaded_monitor(camera_id="CAM-1", samples=10):
    monitor = PerformanceMonitor()
    _quiet_system_snapshot(monitor)
    for _ in range(samples):
        monitor.record_ai(camera_id, latency_sec=1.0)  # 1000ms — far above any sane threshold
    for _ in range(samples):
        monitor.record_capture(camera_id)
    return monitor


def test_adaptive_controller_reduces_inference_rate_before_resolution():
    cam = _FakeManagedCamera("CAM-1", detect_every_n_frames=1, ai_imgsz=640)
    monitor = _overloaded_monitor()
    controller = AdaptiveController(monitor, {"CAM-1": cam}, AdaptiveConfig(min_samples_before_acting=1))

    controller.tick()
    assert cam.get_detect_every_n_frames() == 2  # rate backed off first
    assert cam.get_ai_imgsz() == 640             # resolution untouched while rate still has headroom


def test_adaptive_controller_reduces_resolution_once_rate_is_maxed():
    cam = _FakeManagedCamera("CAM-1", detect_every_n_frames=6, ai_imgsz=640)  # rate already at max
    monitor = _overloaded_monitor()
    config = AdaptiveConfig(min_samples_before_acting=1, max_detect_every_n_frames=6)
    controller = AdaptiveController(monitor, {"CAM-1": cam}, config)

    controller.tick()
    assert cam.get_detect_every_n_frames() == 6  # already maxed, unchanged
    assert cam.get_ai_imgsz() == 480             # next-lower resolution level


def test_adaptive_controller_marks_degraded_only_once_every_lever_exhausted():
    cam = _FakeManagedCamera("CAM-1", detect_every_n_frames=6, ai_imgsz=320)  # both levers maxed out
    monitor = _overloaded_monitor()
    config = AdaptiveConfig(min_samples_before_acting=1, max_detect_every_n_frames=6,
                             ai_imgsz_levels=(640, 480, 320))
    events = []
    controller = AdaptiveController(monitor, {"CAM-1": cam}, config,
                                     on_degraded=lambda cid, degraded: events.append((cid, degraded)))

    controller.tick()
    assert events == [("CAM-1", True)]
    # A second tick while still overloaded and maxed shouldn't re-fire the callback.
    controller.tick()
    assert events == [("CAM-1", True)]


def test_adaptive_controller_restores_settings_once_load_normal():
    cam = _FakeManagedCamera("CAM-1", detect_every_n_frames=3, ai_imgsz=480)
    monitor = PerformanceMonitor()  # no AI latency recorded at all -> not overloaded
    _quiet_system_snapshot(monitor)
    for _ in range(10):
        monitor.record_capture("CAM-1")
    config = AdaptiveConfig(min_samples_before_acting=1)
    controller = AdaptiveController(monitor, {"CAM-1": cam}, config)

    controller.tick()
    # Resolution restores toward the max level before the inference rate does (mirror of the backoff order).
    assert cam.get_ai_imgsz() == 640
    assert cam.get_detect_every_n_frames() == 3

    controller.tick()
    assert cam.get_detect_every_n_frames() == 2


def test_adaptive_controller_ignores_camera_with_too_few_samples():
    cam = _FakeManagedCamera("CAM-1")
    monitor = _overloaded_monitor(samples=2)  # below default min_samples_before_acting
    controller = AdaptiveController(monitor, {"CAM-1": cam})  # default config: min_samples_before_acting=5

    controller.tick()
    assert cam.get_detect_every_n_frames() == 1  # untouched — not enough data yet to act on


def test_adaptive_controller_does_not_throttle_on_drop_rate_alone():
    """Root-cause regression (camera-pipeline repair): a camera with a high
    drop_rate but healthy AI latency and CPU is a camera/USB capture
    problem, not AI running behind — it must never be marked overloaded/
    degraded, and its AI inference rate/resolution must stay untouched
    (throttling AI does nothing to fix a flaky camera). See core.vision.
    CameraWorker's own FAIL_THRESHOLD/offline handling for the correct way
    a capture problem is supposed to surface instead."""
    cam = _FakeManagedCamera("CAM-1", detect_every_n_frames=1, ai_imgsz=640)
    monitor = PerformanceMonitor()
    _quiet_system_snapshot(monitor)
    for _ in range(2):
        monitor.record_capture("CAM-1")
    for _ in range(8):
        monitor.record_dropped("CAM-1")  # 80% drop rate -- far above drop_rate_high (0.05)
    events = []
    config = AdaptiveConfig(min_samples_before_acting=1)
    controller = AdaptiveController(monitor, {"CAM-1": cam}, config,
                                     on_degraded=lambda cid, degraded: events.append((cid, degraded)))

    controller.tick()
    assert cam.get_detect_every_n_frames() == 1  # untouched
    assert cam.get_ai_imgsz() == 640              # untouched
    assert events == []                            # never marked degraded


def test_adaptive_controller_disabled_does_nothing_when_run():
    cam = _FakeManagedCamera("CAM-1")
    monitor = _overloaded_monitor()
    controller = AdaptiveController(monitor, {"CAM-1": cam}, AdaptiveConfig(enabled=False, tick_interval_sec=0.01))
    controller.start()
    time.sleep(0.1)
    controller.stop()
    assert cam.get_detect_every_n_frames() == 1  # run() returns immediately when disabled


# ----------------------------------------------------------------- AI pause

class _FakeAIScheduler:
    """Stand-in for core.vision.AIWorker's pause surface."""

    def __init__(self):
        self.paused = False
        self.pause_events: list[bool] = []

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.pause_events.append(paused)

    def is_paused(self) -> bool:
        return self.paused


def test_ai_pause_requires_sustained_high_cpu_then_resumes_after_sustained_low(monkeypatch):
    import core.performance as perf_module

    fake_now = [0.0]
    monkeypatch.setattr(perf_module.time, "monotonic", lambda: fake_now[0])

    scheduler = _FakeAIScheduler()
    config = AdaptiveConfig(cpu_pause_pct=95.0, cpu_resume_pct=70.0,
                             pause_sustain_sec=6.0, resume_sustain_sec=6.0)
    controller = AdaptiveController(PerformanceMonitor(), {}, config, ai_worker=scheduler)

    # High CPU, but not sustained long enough yet -> no pause.
    controller._update_ai_pause_state(96.0)
    fake_now[0] += 3.0
    controller._update_ai_pause_state(96.0)
    assert scheduler.paused is False

    # Sustained past pause_sustain_sec (7s >= 6s total) -> pauses AI, Preview untouched by this call.
    fake_now[0] += 4.0
    controller._update_ai_pause_state(96.0)
    assert scheduler.paused is True
    assert scheduler.pause_events == [True]

    # A brief dip below cpu_resume_pct, not yet sustained -> stays paused.
    fake_now[0] += 2.0
    controller._update_ai_pause_state(50.0)
    assert scheduler.paused is True

    # CPU spikes back up before resume_sustain_sec elapses -> resets the resume timer, stays paused.
    fake_now[0] += 1.0
    controller._update_ai_pause_state(96.0)
    assert scheduler.paused is True

    # Sustained low for long enough this time -> resumes automatically.
    fake_now[0] += 1.0
    controller._update_ai_pause_state(50.0)
    fake_now[0] += 6.0
    controller._update_ai_pause_state(50.0)
    assert scheduler.paused is False
    assert scheduler.pause_events == [True, False]  # never flapped in between — hysteresis held


def test_ai_pause_is_noop_without_an_ai_worker():
    controller = AdaptiveController(PerformanceMonitor(), {}, AdaptiveConfig(pause_sustain_sec=0.0))
    controller._update_ai_pause_state(99.0)  # must not raise with no ai_worker configured
    assert controller._ai_paused is False
