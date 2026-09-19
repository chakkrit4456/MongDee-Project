"""Performance Monitor + Adaptive Controller for the multi-camera pipeline.

This is the piece that answers, at runtime, "is this booth's hardware
keeping up with N cameras + AI?" and reacts if not — see
MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md sections 8-10, 29-34.

Three independent pieces:
  - HardwareProfile / detect_hardware() — a one-time snapshot at startup
    (CPU/RAM/GPU) used only as a *starting point*; real adjustments always
    come from live metrics, per section 33 ("ต้องใช้ข้อมูล Runtime จริง").
  - PerformanceMonitor — a thread-safe, bounded-memory recorder that camera
    threads report capture/display/drop/AI events into (bounded deques —
    never an unbounded queue, see rule #4 in the spec) and that anything
    (a dashboard, a benchmark run, the AdaptiveController) can snapshot.
  - AdaptiveController — a background tick that, per camera, backs off AI
    workload when this process's own measurements say it's overloaded
    (high CPU, high AI latency, or a high drop rate), in the priority order
    the spec mandates: inference rate first, then AI input resolution.
    Display FPS is deliberately never auto-reduced by this controller (see
    its docstring) — it only raises a "degraded" flag once every AI lever
    is exhausted and the camera is still overloaded, leaving that last,
    user-visible call to whoever owns the display pipeline.

Framework-agnostic like core/vision.py: no Qt, no FastAPI. A "managed
camera" here is any object satisfying the ManagedCamera protocol below —
core.vision.CameraWorker implements it directly, but nothing here imports
core.vision, so this module carries no camera-I/O dependency at all.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

logger = logging.getLogger("mongdee.core.performance")

try:
    import psutil
except Exception:  # pragma: no cover - ships in requirements.txt; degrade gracefully if absent
    psutil = None


# --------------------------------------------------------------- hardware

@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    cpu_logical_cores: int
    cpu_physical_cores: int | None
    ram_total_gb: float
    gpu_available: bool
    gpu_name: str | None
    vram_total_gb: float | None
    tier: str  # "LOW" | "MEDIUM" | "HIGH" -- an initial seed only, see module docstring

    def summary_line(self) -> str:
        gpu = f"{self.gpu_name} ({self.vram_total_gb:.1f} GB VRAM)" if self.gpu_available else "ไม่มี (CPU only)"
        cpu = f"{self.cpu_logical_cores} logical"
        if self.cpu_physical_cores:
            cpu += f" / {self.cpu_physical_cores} physical"
        return (f"[HW] CPU: {cpu} cores | RAM: {self.ram_total_gb:.1f} GB | "
                f"GPU: {gpu} | Tier: {self.tier}")


def detect_hardware() -> HardwareProfile:
    """Best-effort hardware snapshot — every probe degrades to a safe
    default instead of raising, since this must never be what stops the
    booth from starting (rule #16/#17: CPU fallback + error handling)."""
    cpu_logical = os.cpu_count() or 1
    cpu_physical = None
    ram_total_gb = 0.0
    if psutil is not None:
        try:
            cpu_physical = psutil.cpu_count(logical=False)
        except Exception:
            cpu_physical = None
        try:
            ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_total_gb = 0.0

    gpu_available = False
    gpu_name = None
    vram_total_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass

    if gpu_available and cpu_logical >= 8:
        tier = "HIGH"
    elif cpu_logical >= 4:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return HardwareProfile(
        cpu_logical_cores=cpu_logical, cpu_physical_cores=cpu_physical, ram_total_gb=ram_total_gb,
        gpu_available=gpu_available, gpu_name=gpu_name, vram_total_gb=vram_total_gb, tier=tier,
    )


# ----------------------------------------------------------- the monitor

FPS_WINDOW_SEC = 3.0          # rolling window used for actual-FPS/latency/drop-rate calculations
MAX_SAMPLES_PER_CAMERA = 300  # hard bound on memory per camera regardless of FPS/runtime (rule #4: no unbounded queues)


@dataclasses.dataclass
class _CameraStats:
    requested_fps: float = 0.0
    # (timestamp, dropped) for every capture attempt — successful and failed
    # alike — bounded by maxlen so a camera can run for hours without this
    # growing (rule #41: no unbounded memory growth).
    events: collections.deque = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=MAX_SAMPLES_PER_CAMERA))
    display_latencies_ms: collections.deque = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=MAX_SAMPLES_PER_CAMERA))
    ai_events: collections.deque = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=MAX_SAMPLES_PER_CAMERA))
    ai_latencies_ms: collections.deque = dataclasses.field(
        default_factory=lambda: collections.deque(maxlen=MAX_SAMPLES_PER_CAMERA))
    dropped_total: int = 0
    captured_total: int = 0
    resolution: tuple[int, int] | None = None


class PerformanceMonitor:
    """Thread-safe. Camera threads call record_*() from their own thread;
    anything else (dashboard route, AdaptiveController, benchmark run) can
    call the snapshot methods from any thread at any time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cameras: dict[str, _CameraStats] = {}

    def _stats(self, camera_id: str) -> _CameraStats:
        st = self._cameras.get(camera_id)
        if st is None:
            st = _CameraStats()
            self._cameras[camera_id] = st
        return st

    def set_requested_fps(self, camera_id: str, fps: float) -> None:
        with self._lock:
            self._stats(camera_id).requested_fps = fps

    def set_resolution(self, camera_id: str, resolution: tuple[int, int] | None) -> None:
        with self._lock:
            self._stats(camera_id).resolution = resolution

    def record_capture(self, camera_id: str, capture_ts: float | None = None) -> None:
        ts = capture_ts if capture_ts is not None else time.time()
        with self._lock:
            st = self._stats(camera_id)
            st.events.append((ts, False))
            st.captured_total += 1

    def record_dropped(self, camera_id: str) -> None:
        with self._lock:
            st = self._stats(camera_id)
            st.events.append((time.time(), True))
            st.dropped_total += 1

    def record_display(self, camera_id: str, capture_ts: float, display_ts: float | None = None) -> None:
        ts = display_ts if display_ts is not None else time.time()
        with self._lock:
            self._stats(camera_id).display_latencies_ms.append(max(0.0, (ts - capture_ts) * 1000.0))

    def record_ai(self, camera_id: str, latency_sec: float) -> None:
        with self._lock:
            st = self._stats(camera_id)
            st.ai_events.append(time.time())
            st.ai_latencies_ms.append(max(0.0, latency_sec * 1000.0))

    def forget_camera(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._cameras.keys())

    def camera_snapshot(self, camera_id: str) -> dict:
        now = time.time()
        with self._lock:
            st = self._cameras.get(camera_id)
            if st is None:
                return {
                    "camera_id": camera_id, "requested_fps": 0.0, "actual_fps": 0.0, "ai_fps": 0.0,
                    "dropped_frames": 0, "captured_frames": 0, "drop_rate": 0.0,
                    "avg_display_latency_ms": 0.0, "avg_ai_latency_ms": 0.0, "resolution": None,
                }
            recent_events = [(t, d) for t, d in st.events if now - t <= FPS_WINDOW_SEC]
            recent_ok = [t for t, d in recent_events if not d]
            actual_fps = 0.0
            if len(recent_ok) >= 2:
                span = recent_ok[-1] - recent_ok[0]
                if span > 0:
                    actual_fps = (len(recent_ok) - 1) / span
            recent_ai = [t for t in st.ai_events if now - t <= FPS_WINDOW_SEC]
            ai_fps = 0.0
            if len(recent_ai) >= 2:
                span = recent_ai[-1] - recent_ai[0]
                if span > 0:
                    ai_fps = (len(recent_ai) - 1) / span
            drop_rate = (
                sum(1 for _, d in recent_events if d) / len(recent_events) if recent_events else 0.0
            )
            avg_latency_ms = (
                sum(st.display_latencies_ms) / len(st.display_latencies_ms) if st.display_latencies_ms else 0.0
            )
            avg_ai_latency_ms = (
                sum(st.ai_latencies_ms) / len(st.ai_latencies_ms) if st.ai_latencies_ms else 0.0
            )
            return {
                "camera_id": camera_id,
                "requested_fps": round(st.requested_fps, 1),
                "actual_fps": round(actual_fps, 1),
                "ai_fps": round(ai_fps, 1),
                "dropped_frames": st.dropped_total,
                "captured_frames": st.captured_total,
                "drop_rate": round(drop_rate, 4),
                "avg_display_latency_ms": round(avg_latency_ms, 1),
                "avg_ai_latency_ms": round(avg_ai_latency_ms, 1),
                "resolution": list(st.resolution) if st.resolution else None,
            }

    def snapshot(self) -> dict[str, dict]:
        return {cid: self.camera_snapshot(cid) for cid in self.camera_ids()}

    def system_snapshot(self) -> dict:
        cpu_percent = None
        ram_percent = None
        ram_used_gb = None
        if psutil is not None:
            try:
                # non-blocking (interval=None) — compares against the last
                # call, which the AdaptiveController's own tick loop makes
                # regularly; the very first call in a process returns 0.0,
                # which is an acceptable, self-correcting cold-start blip.
                cpu_percent = psutil.cpu_percent(interval=None)
            except Exception:
                cpu_percent = None
            try:
                vm = psutil.virtual_memory()
                ram_percent = vm.percent
                ram_used_gb = vm.used / (1024 ** 3)
            except Exception:
                pass

        vram_used_gb = None
        try:
            import torch
            if torch.cuda.is_available():
                vram_used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        except Exception:
            pass

        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_used_gb": round(ram_used_gb, 2) if ram_used_gb is not None else None,
            "gpu_vram_used_gb": round(vram_used_gb, 2) if vram_used_gb is not None else None,
        }


# -------------------------------------------------------------- benchmark

def format_benchmark_report(hardware: HardwareProfile, monitor: PerformanceMonitor,
                             elapsed_sec: float) -> str:
    """Plain-text report matching the shape in
    MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 45 — used
    by both `--benchmark` CLI runs and (via the same data) the web
    Performance Monitor panel's underlying numbers."""
    system = monitor.system_snapshot()
    cam_snapshots = monitor.snapshot()
    lines = ["MongDee Camera Benchmark", f"Duration: {elapsed_sec:.1f}s", f"Cameras: {len(cam_snapshots)}"]
    for cam_id, m in sorted(cam_snapshots.items()):
        res = f"{m['resolution'][0]}x{m['resolution'][1]}" if m["resolution"] else "unknown"
        lines.append(
            f"[{cam_id}] Resolution: {res} | Display FPS: {m['actual_fps']:.1f} "
            f"(requested {m['requested_fps']:.1f}) | AI FPS: {m['ai_fps']:.1f} | "
            f"Frame Drop: {m['drop_rate'] * 100:.1f}% | Display Latency: {m['avg_display_latency_ms']:.0f} ms | "
            f"AI Latency: {m['avg_ai_latency_ms']:.0f} ms"
        )
    cpu = f"{system['cpu_percent']:.0f}%" if system["cpu_percent"] is not None else "n/a"
    ram = f"{system['ram_percent']:.0f}% ({system['ram_used_gb']:.1f} GB)" if system["ram_percent"] is not None else "n/a"
    vram = f"{system['gpu_vram_used_gb']:.2f} GB" if system["gpu_vram_used_gb"] is not None else ("n/a" if not hardware.gpu_available else "0.00 GB")
    lines.append(f"CPU: {cpu}")
    lines.append(f"RAM: {ram}")
    lines.append(f"GPU: {hardware.gpu_name if hardware.gpu_available else 'none'} | VRAM used: {vram}")
    return "\n".join(lines)


# ------------------------------------------------------- adaptive control

@dataclasses.dataclass(frozen=True)
class AdaptiveConfig:
    """Every threshold here is meant to be tuned per-deployment (rule #6:
    no hard-coded hardware-specific settings) — see
    configs/performance.example.json. Defaults are conservative starting
    points, not claims about any particular machine."""
    enabled: bool = True
    tick_interval_sec: float = 2.0
    cpu_high_pct: float = 85.0
    ai_latency_high_ms: float = 250.0
    # Kept for config-file backward compatibility (an existing
    # performance.json with this key must still load) but no longer feeds
    # AdaptiveController.tick()'s "overloaded" decision — see tick()'s own
    # comment. drop_rate reflects camera/capture health (failed reads,
    # frames core.vision._looks_like_noise() rejects), not AI load, so
    # throttling AI inference rate/resolution in response to it was
    # fixing the wrong thing.
    drop_rate_high: float = 0.05
    min_detect_every_n_frames: int = 1
    max_detect_every_n_frames: int = 8
    ai_imgsz_levels: tuple[int, ...] = (640, 480, 320)
    min_samples_before_acting: int = 5  # avoid reacting to a single noisy tick right after startup

    # AI Pause — the level past "every lever maxed out and still
    # overloaded": rather than staying pinned at the harshest inference
    # rate/resolution forever, sustained system-wide CPU pressure pauses AI
    # (YOLO + custom recognition) entirely for every camera, process-wide.
    # Camera Preview is untouched by this — it never depended on AI running
    # to begin with (see core.vision's AIWorker) — only detection/tracking/
    # product-recognition output stops until CPU recovers. Both the trigger
    # and the recovery require the condition to *hold* for `*_sustain_sec`
    # (hysteresis), so one brief spike/dip never flips this back and forth.
    cpu_pause_pct: float = 95.0
    cpu_resume_pct: float = 70.0
    pause_sustain_sec: float = 6.0
    resume_sustain_sec: float = 6.0

    @staticmethod
    def from_dict(d: dict) -> "AdaptiveConfig":
        # A leading "_" marks a documentation-only key (see
        # configs/performance.example.json) — never a real field, so it's
        # dropped before validation instead of tripping the unknown-field
        # check meant to catch genuine typos.
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        known = {f.name for f in dataclasses.fields(AdaptiveConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"performance config has unknown field(s): {unknown}")
        if "ai_imgsz_levels" in d:
            d["ai_imgsz_levels"] = tuple(sorted(set(int(s) for s in d["ai_imgsz_levels"]), reverse=True))
        return AdaptiveConfig(**d)


def load_adaptive_config(path: str | Path) -> AdaptiveConfig:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read performance config file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc
    return AdaptiveConfig.from_dict(data)


class ManagedCamera(Protocol):
    """Duck-typed interface AdaptiveController needs from a camera worker.
    core.vision.CameraWorker implements this directly; nothing here imports
    it, so core/performance.py has no camera-I/O dependency at all."""
    camera_id: str

    def get_detect_every_n_frames(self) -> int: ...
    def set_detect_every_n_frames(self, n: int) -> None: ...
    def get_ai_imgsz(self) -> int: ...
    def set_ai_imgsz(self, size: int) -> None: ...


class AIScheduler(Protocol):
    """Duck-typed interface for the AI Pause lever — core.vision.AIWorker
    implements this directly; nothing here imports it, same reasoning as
    ManagedCamera above."""

    def set_paused(self, paused: bool) -> None: ...
    def is_paused(self) -> bool: ...


class AdaptiveController(threading.Thread):
    """Background tick that watches PerformanceMonitor and, when a camera
    is sustainedly overloaded, backs off its AI workload in the order
    section 10 of the performance spec mandates: inference rate first, then
    AI input resolution. Display FPS is intentionally never touched here —
    once every AI lever is exhausted and the camera is still overloaded,
    this only flips a "degraded" flag (via on_degraded) rather than
    guessing at a display-FPS cut on its owner's behalf; whoever owns the
    display pipeline (BoothManager, MainWindow) can decide what "degraded"
    should do to its own stream, matching "อย่าลด Display FPS ก่อนโดยไม่มีเหตุผล"
    (don't reduce Display FPS without cause) — a controller two levers
    short of exhausted has no cause yet.

    Backs off one step per tick (never jumps straight to the harshest
    setting) and restores one step per tick once the camera is comfortably
    under threshold, so a brief spike doesn't overcorrect and settings
    don't flap every tick.

    Past every per-camera lever, there is one more, system-wide level: AI
    Pause (see AdaptiveConfig's cpu_pause_pct/cpu_resume_pct/*_sustain_sec).
    While system CPU stays at or above cpu_pause_pct for pause_sustain_sec,
    AI is paused for every camera at once via `ai_worker.set_paused(True)`
    — Camera Preview keeps running at full rate throughout (see
    core.vision.AIWorker); only detection/tracking/product-recognition
    output stops. Resumes automatically once CPU stays at or below
    cpu_resume_pct for resume_sustain_sec. `ai_worker` is optional — pass
    None (the default) to disable this level entirely and keep only the
    per-camera rate/resolution backoff above.
    """

    def __init__(self, monitor: PerformanceMonitor, cameras: dict[str, ManagedCamera],
                 config: AdaptiveConfig | None = None,
                 on_degraded: Callable[[str, bool], None] | None = None,
                 ai_worker: AIScheduler | None = None,
                 on_ai_paused: Callable[[bool], None] | None = None):
        super().__init__(daemon=True, name="mongdee-adaptive-controller")
        self._monitor = monitor
        self._cameras = cameras  # shared reference — caller may add/remove entries as cameras change
        self._config = config or AdaptiveConfig()
        self._on_degraded = on_degraded or (lambda camera_id, degraded: None)
        self._ai_worker = ai_worker
        self._on_ai_paused = on_ai_paused or (lambda paused: None)
        self._stop_event = threading.Event()
        self._degraded_state: dict[str, bool] = {}
        self._ai_paused = False
        self._cpu_high_since: float | None = None
        self._cpu_low_since: float | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)

    def run(self) -> None:
        if not self._config.enabled:
            return
        while not self._stop_event.wait(self._config.tick_interval_sec):
            try:
                self.tick()
            except Exception:
                logger.exception("adaptive controller tick failed")

    def tick(self) -> None:
        """One adjustment pass — public so tests/a manual `--benchmark`
        run can drive it deterministically instead of waiting on the
        background timer."""
        system = self._monitor.system_snapshot()
        cpu_pct = system.get("cpu_percent")
        self._update_ai_pause_state(cpu_pct)
        cpu_high = cpu_pct is not None and cpu_pct >= self._config.cpu_high_pct

        for camera_id, worker in list(self._cameras.items()):
            metrics = self._monitor.camera_snapshot(camera_id)
            if metrics["captured_frames"] < self._config.min_samples_before_acting:
                continue
            # drop_rate is deliberately NOT part of "overloaded" (root-cause
            # fix — see MongDee camera-pipeline repair notes /
            # docs/design/camera-pipeline-root-cause-repair.md): its only
            # producer is core.vision.CameraWorker.run()'s capture loop
            # calling PerformanceMonitor.record_dropped() on a failed read
            # or a frame core.vision._looks_like_noise() rejects — i.e. a
            # camera/USB/driver problem, never AI running behind. Folding it
            # into this AI-throttle decision meant a flaky camera got
            # mislabeled "degraded: AI ประมวลผลไม่ทัน" (and had its AI
            # inference rate/resolution needlessly reduced, which does
            # nothing to fix a capture problem) instead of the camera's own
            # already-correct offline/reconnecting handling (FAIL_THRESHOLD
            # in core/vision.py) ever getting a chance to report it
            # accurately. drop_rate is still computed and surfaced in the
            # Performance Monitor panel for human diagnosis; it just no
            # longer drives this decision.
            overloaded = (
                cpu_high
                or metrics["avg_ai_latency_ms"] >= self._config.ai_latency_high_ms
            )
            self._apply(camera_id, worker, overloaded)

    def _apply(self, camera_id: str, worker: ManagedCamera, overloaded: bool) -> None:
        n = worker.get_detect_every_n_frames()
        imgsz = worker.get_ai_imgsz()
        levels = self._config.ai_imgsz_levels or (imgsz,)
        min_imgsz, max_imgsz = min(levels), max(levels)

        if overloaded:
            if n < self._config.max_detect_every_n_frames:
                worker.set_detect_every_n_frames(n + 1)
                logger.info("[PERF] %s: reducing AI inference rate (every %d -> %d frames)",
                            camera_id, n, n + 1)
            elif imgsz > min_imgsz:
                lower = max((s for s in levels if s < imgsz), default=imgsz)
                worker.set_ai_imgsz(lower)
                logger.info("[PERF] %s: reducing AI input resolution (%dpx -> %dpx)",
                            camera_id, imgsz, lower)
        else:
            if imgsz < max_imgsz:
                higher = min((s for s in levels if s > imgsz), default=imgsz)
                worker.set_ai_imgsz(higher)
                logger.info("[PERF] %s: restoring AI input resolution (%dpx -> %dpx)",
                            camera_id, imgsz, higher)
            elif n > self._config.min_detect_every_n_frames:
                worker.set_detect_every_n_frames(n - 1)
                logger.info("[PERF] %s: restoring AI inference rate (every %d -> %d frames)",
                            camera_id, n, n - 1)

        is_degraded = overloaded and n >= self._config.max_detect_every_n_frames and imgsz <= min_imgsz
        was_degraded = self._degraded_state.get(camera_id, False)
        if is_degraded != was_degraded:
            self._degraded_state[camera_id] = is_degraded
            self._on_degraded(camera_id, is_degraded)

    def _update_ai_pause_state(self, cpu_pct: float | None) -> None:
        """AI Pause hysteresis — see AdaptiveController's docstring. A no-op
        when no ai_worker was given (e.g. the many tests/callers that only
        care about the per-camera rate/resolution levers) or CPU couldn't be
        read this tick (psutil missing/failed — never guess in that case)."""
        if self._ai_worker is None or cpu_pct is None:
            return
        now = time.monotonic()
        if not self._ai_paused:
            if cpu_pct >= self._config.cpu_pause_pct:
                if self._cpu_high_since is None:
                    self._cpu_high_since = now
                if now - self._cpu_high_since >= self._config.pause_sustain_sec:
                    self._ai_worker.set_paused(True)
                    self._ai_paused = True
                    self._cpu_low_since = None
                    logger.warning(
                        "[PERF] CPU สูงต่อเนื่อง (%.0f%% >= %.0f%% นาน %.0fs) — พัก AI ทั้งระบบชั่วคราว "
                        "(Camera Preview ยังทำงานปกติ)",
                        cpu_pct, self._config.cpu_pause_pct, self._config.pause_sustain_sec,
                    )
                    self._on_ai_paused(True)
            else:
                self._cpu_high_since = None
        else:
            if cpu_pct <= self._config.cpu_resume_pct:
                if self._cpu_low_since is None:
                    self._cpu_low_since = now
                if now - self._cpu_low_since >= self._config.resume_sustain_sec:
                    self._ai_worker.set_paused(False)
                    self._ai_paused = False
                    self._cpu_high_since = None
                    logger.info(
                        "[PERF] CPU กลับสู่ปกติ (%.0f%% <= %.0f%% นาน %.0fs) — เริ่ม AI ต่ออัตโนมัติ",
                        cpu_pct, self._config.cpu_resume_pct, self._config.resume_sustain_sec,
                    )
                    self._on_ai_paused(False)
            else:
                self._cpu_low_since = None
