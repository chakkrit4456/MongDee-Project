# Camera Pipeline — Root-Cause Diagnosis and Repair

Scope: the **active runtime** camera pipeline only (`web_server.py` →
`web/server.py` → `web/booth_manager.py` → `core/vision.py`). See §1 for how
that was confirmed. The separate `backend/`+`vision/`+`camera/` pipeline
(`python -m backend.main`, a different entry point/port) was inspected only
enough to confirm it is not what GUIDE.md documents users running, per this
repair's own scoping rule (§5 below) — it is not modified here.

## 1. Actual runtime path

Confirmed by tracing imports/entry points, not assumed:

```
web_server.py (CLI/argparse)
  -> web/server.py (FastAPI app, routes, /stream/{camera_id})
    -> web/booth_manager.py (BoothManager: owns CameraWorker instances,
       health/status, hot-plug scan, adaptive controller wiring)
      -> core/vision.py (CameraWorker: capture thread, AIWorker: shared
         inference thread, discover_cameras, _open_capture)
      -> core/performance.py (PerformanceMonitor, AdaptiveController)
      -> core/camera_identity.py (DirectShow name resolution, Windows-only)
```

GUIDE.md's own "how to run" instructions (`python web_server.py`) and
README.md confirm this is the primary, user-facing entry point. The
`backend/` + `vision/` + `camera/` tree is a separate pipeline for a
different spec (`MongDee_Master_Prompt_Complete-1.md`) with its own entry
point (`python -m backend.main`) and its own database — legacy/parallel,
not what this repair targets.

## 2. Failure symptoms (from the report)

A. corrupted/noisy camera frames
B. "AI ประมวลผลไม่ทัน" (AI can't keep up)
C. frozen/stale camera frames
D. USB hot-unplug failure
E. USB hot-plug/reconnect failure
F. multi-camera instability
G. camera stream/UI desynchronization

## 3-7. Root causes, by symptom

**A. Corrupted/noisy frames** — Already handled correctly. `core/vision.py`'s
`_looks_like_noise()` rejects both whole-frame garbage (a stale buffer kept
returning `ok=True` after a silent disconnect) and partial/banded garbage (a
torn USB transfer corrupting a scanline band), via two calibrated
pixel-roughness-vs-contrast ratios (`NOISE_ROUGHNESS_RATIO`,
`BANDING_ROUGHNESS_RATIO`), applied both at open-time (`_read_with_warmup`)
and continuously in the capture loop. No change needed — verified via
`tests/core/test_camera_open.py` (28 tests, all passing).

**B. "AI ประมวลผลไม่ทัน"** — **Root cause found and fixed.** The message
comes from `web/booth_manager.py`'s `_on_camera_degraded`, fired by
`core.performance.AdaptiveController` when a camera is "overloaded" for
`min_samples_before_acting` ticks with every AI lever (inference rate, then
resolution) already exhausted. Before this fix, `overloaded` was:

```python
overloaded = cpu_high or avg_ai_latency_ms >= ai_latency_high_ms or drop_rate >= drop_rate_high
```

`drop_rate`'s *only* producer is `core.vision.CameraWorker`'s capture loop
calling `PerformanceMonitor.record_dropped()` on a failed `read()` or a
frame `_looks_like_noise()` rejects — i.e. a **camera/USB/driver** problem,
never AI running behind (capture and AI are fully decoupled onto separate
threads via a single-slot frame hand-off — see §10 — so AI backlog cannot
itself cause a capture-side drop). A flaky camera producing intermittent
(but not 20-consecutive, so never tripping `FAIL_THRESHOLD`→offline)
corrupted frames was therefore mislabeled "AI ประมวลผลไม่ทัน" and had its AI
inference rate/resolution needlessly reduced — an action that does nothing
to fix a capture problem, while degrading detection quality on an otherwise
fine camera. **Fix**: removed `drop_rate` from the `overloaded` expression
in `AdaptiveController.tick()` (`core/performance.py`); `drop_rate` is still
computed and shown in the Performance Monitor panel for human diagnosis, it
just no longer drives AI throttling or the "degraded" status. Regression
test: `tests/core/test_performance.py::test_adaptive_controller_does_not_throttle_on_drop_rate_alone`.

**C. Frozen/stale frames** — Already handled correctly.
`BoothManager.get_latest_jpeg()` explicitly never serves `self._latest_jpeg`
(the last real frame) once a camera's status leaves `online`/`degraded` —
it serves a `"CAMERA RECONNECTING..."` placeholder instead, with an
explicit comment recording exactly this failure mode as the reason. No
change needed.

**D/E. Hot-unplug / hot-plug reconnect** — Already handled correctly.
Unplug: `FAIL_THRESHOLD` (20 consecutive bad reads) → `_release_capture()` →
status `offline`, never held "online" on stale reads. Reconnect (same
already-known camera): `_attempt_reopen()` on every loop iteration while
`self._cap` is closed, with exponential backoff (`REOPEN_BACKOFF_*`,
3s→30s) that resets to the initial delay the instant an open succeeds.
Reconnect (brand-new USB device at a previously-unused index):
`BoothManager._hotplug_loop` — a separate periodic `discover_cameras()`
scan (own exponential backoff, skips indices already owned by a running
`CameraWorker`, skips the scan entirely above `HOTPLUG_SKIP_SCAN_CPU_PCT`
system load). Both paths re-validate camera *identity* (DirectShow name via
`core.camera_identity`, not raw index) on every single open attempt, not
just once — this specifically closes a previously-observed bug where a
worker reconnecting after Windows silently re-enumerated devices picked up
the built-in webcam instead of the USB camera it was bound to. No change
needed.

**F. Multi-camera instability** — Structurally isolated already: each
`CameraWorker` owns its own `cv2.VideoCapture` on its own thread (capture),
one shared `AIWorker` thread round-robins AI passes so one camera's
inference never *starves* another's queue (each is skipped until its own
`ai_due()` cadence elapses) — but a single AI call that hangs unusually long
would still delay other cameras' *next* AI pass (not their capture/preview,
which is fully decoupled). This is an intentional, documented tradeoff (one
shared GPU/CPU inference budget, matching the GTX 1050 target hardware) —
not something this repair changes, since removing the shared-thread design
would reintroduce the GPU-contention problem it exists to prevent.

**G. Stream/UI desync** — Already handled correctly.
`web/static/booth.js`'s `updateViewerStatuses()` updates only the status
dot/text of the affected viewer(s) and only reloads an `<img>`'s `src` on a
genuine transition into `online`; it never rebuilds the camera grid DOM.
No change needed.

## 8. Existing architecture (confirmed, not assumed)

- Capture and AI inference are fully decoupled: `CameraWorker.run()`'s
  capture thread only ever does capture → draw last-known boxes → deliver
  via `on_frame()`; it never calls YOLO/the recognizer itself.
- Frame hand-off to AI is a **single mutable slot**
  (`self._latest_capture_frame`, guarded by `self._frame_slot_lock`), not a
  queue — every new capture simply overwrites it. This already *is* the
  "process the newest valid frame, never build a backlog" policy §11 of the
  original prompt asks for; there is nothing to add.
- One process-wide `AIWorker` thread runs YOLO + custom recognition for
  every camera, one at a time, matching the single-GPU-call-at-a-time
  constraint of the target hardware (GTX 1050).
- `core.performance.PerformanceMonitor` + `AdaptiveController` back off a
  camera's AI inference rate, then its input resolution, under sustained
  CPU/AI-latency pressure, and can pause AI process-wide (never capture)
  under sustained extreme CPU pressure.

## 9-15. Queue strategy / lifecycle / identity / recovery / UI / performance impact

All already implemented as described in §3-8 above; this repair's only
change is the `overloaded` classification fix in §3.B. No new queue, no new
lifecycle states, no new identity scheme, no new UI mechanism were needed.

## 16. Test plan (executed)

```
pytest tests/core/test_performance.py -q   # 24 passed (was 23; +1 regression test)
pytest tests/core/test_camera_open.py -q   # 28 passed (unchanged by this repair)
pytest tests/ -q                            # full suite, see final report
```

Real USB hardware / browser (Playwright) tests: **not available in this
environment** — see the final report's Runtime Verification / Remaining
Limitations sections for what was and wasn't verified.

## 17. Addendum (2026-09-18) — second concurrent USB camera still failing mid-stream

§7.F above scoped "multi-camera instability" to *thread/AI* isolation
(already correct) and explicitly left the underlying USB/driver contention
between two concurrently-streaming cameras untouched. Real hardware
evidence gathered separately (`.hw_validation/*.stderr`, same day) shows
that gap is real: with `CAMERA_STARTUP_STAGGER_SEC` staggering, both
cameras open and read cleanly at *startup* (6/6 clean trials at 2.5s), but
during **sustained** concurrent streaming the second-opened camera
eventually fails with Windows error `-1072875772` (`0xC00D3704` /
`MF_E_HW_MFT_FAILED_START_STREAMING`) and then cannot reopen at any
profile/backend — a signature consistent with a shared USB hub/controller
resource limit, not a per-camera or software defect.

**Fix applied**: `core/vision.py` now tracks how many `CameraWorker`s
currently have a live, reading capture (`_active_camera_count`,
`CameraWorker._mark_camera_active`/`_mark_camera_inactive`). Any camera
opening *while another is already active* requests only the two smallest,
already-compressed profiles (`_LOW_BANDWIDTH_OPEN_PROFILES` — MJPG/YUY2 at
320x240@15) instead of `_OPEN_PROFILES`' full list, which otherwise always
starts at MJPG 640x480@30 regardless of how many other cameras are already
consuming the same shared resource. The camera that opens first (nothing
else active yet) is unaffected — it still gets the full profile list.

This reduces the concurrent resource footprint the two evidenced-working
levers (stagger, FourCC verification) don't address, but it cannot
manufacture USB bandwidth or a second hardware MJPEG-decode grant that
genuinely isn't there — if the root cause is a hard per-controller limit of
one concurrent hardware-accelerated stream (plausible given YUY2 also
failed in the worst observed run, not just MJPG), this mitigates but does
not guarantee a fix. **Status: CODE VERIFIED (unit-tested with mocked
captures, `tests/core/test_camera_open.py`) — NOT YET HARDWARE VERIFIED.**
Needs a real rerun of `.hw_validation/probe_concurrent.py`-style sustained
concurrent streaming (several minutes, not just the 5-read open check) to
confirm whether CAM-2 now holds up. If it still fails identically, the next
evidence-gated step is testing the two cameras on genuinely separate USB
controllers/hubs (not yet done — see the final engineering report's
hardware-validation matrix), since that would confirm a true hardware
ceiling rather than a software-addressable one.
