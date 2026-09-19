# MongDee Multi-Camera Root-Cause Report

> **Addendum (same session, follow-up investigation):** see
> "§K — CAM-1 image-quality follow-up" at the end of this document for a
> second investigation into a *different* reported symptom (CAM-1 reports
> `online` but the image looks wrong) than the crash/CAM-2 findings below.
> That follow-up found CAM-1 delivering genuinely near-black frames at
> ~1 frame/sec — consistent with severe low light or an obstructed lens —
> which does **not** match the "horizontal lines / banding" the report was
> asked to investigate, and could not be verified further without someone
> physically checking the camera/room. Read §K before assuming this
> document's crash fix also explains that symptom.

Scope: active runtime pipeline only (`web_server.py` -> `web/server.py` ->
`web/booth_manager.py` -> `core/vision.py`), same scoping as
`docs/design/camera-pipeline-root-cause-repair.md`, which this report
supersedes for the "why can't two USB cameras stream at once" question. Two
physical UVC cameras were available on the dev machine: index 1 ("CAM-1")
and index 2 ("CAM-2"), both plugged into the same USB hub/controller.

## A. Root Cause

**CONFIRMED ROOT CAUSE (new, found and fixed this session): a hard process
crash**, not a graceful failure. `core.vision._open_capture()`'s
DirectShow-open path (`cv2.VideoCapture(index, cv2.CAP_DSHOW)`) and
`core.camera_identity.list_directshow_devices()` (pygrabber/`comtypes` COM
enumeration, called by `_forbidden_device_name()` on every open attempt and
by the physical-identity resolver) both touch DirectShow's COM device-
enumeration machinery, which is not safely reentrant across threads. The
existing `_open_lock` in `core/vision.py` only serialized the
`cv2.VideoCapture` construction calls; the `list_directshow_devices()` calls
were unsynchronized against it and against each other. With CAM-1 and CAM-2
each running on their own thread (plus the physical-identity resolver's own
fire-and-forget thread, plus the Hot-Plug Scan thread), these calls
routinely land in the same instant, and the result was not a Python
exception but a silent, unrecoverable native crash: the process (all
threads) disappears with no traceback, no logged exception, and no
`stop_scheduled()`/"หยุดเรียบร้อยแล้ว" shutdown message — reproduced twice
this session, both times within 20-90 seconds of both cameras being
concurrently active (see Evidence).

**LIKELY CONTRIBUTING FACTOR (pre-existing, unfixed, evidence-gated):** a
genuine hardware/driver-level limit on this dev machine's USB
controller/hub sharing the two cameras. With the crash fixed, CAM-2 still
cannot sustain a working stream (`OPENED_BUT_READ_FAILED`, Windows Media
Foundation `-1072875772` / `0xC00D3704` `MF_E_HW_MFT_FAILED_START_STREAMING`)
even at the smallest negotiated profile (YUY2 320x240@15, already the
project's existing low-bandwidth mitigation) — see §14/§52 experiments
below for what is and is not proven about this.

**NOT ROOT CAUSE (ruled out this session, with a controlled experiment):**
AI inference / YOLO / FairFace / the web server / BoothManager. A
capture-only test (`.hw_validation/probe_capture_only.py`, no AIWorker, no
FastAPI, no BoothManager — two threads calling the exact production
`core.vision._open_capture()` directly) reproduces CAM-2's failure
identically with zero AI/web code loaded. See §14 evidence.

## B. Evidence

### B.1 — The crash (before this session's fix)

Command: `python web_server.py --no-open`, full stderr captured to
`.hw_validation/live_20260918_204623.stderr.log`.

```
2026-09-18 20:47:17,862 [CAM-1] index=1 backend=DSHOW format=YUY2 320x240@15 open=OK read=OK
2026-09-18 20:47:21,449 [CAM-2] Camera stream failed (device=2, attempt=1, next_retry=3s, reason=open_or_read_failed)
2026-09-18 20:47:27,534 [CAM-2] Camera stream failed (device=2, attempt=2, next_retry=6s, reason=open_or_read_failed)
2026-09-18 20:47:31,815 [CAM-1] Camera stream failed (device=1, reason=frame_read_failed, consecutive_failures=20)
2026-09-18 20:47:36,929 [discover:2] index=2 backend=DSHOW format=YUY2 320x240@15 open=OK read=OK
<log ends here -- no traceback, no exception, no shutdown message>
```

`Get-CimInstance Win32_Process -Filter "Name='python.exe'"` immediately
after: zero results. `netstat -ano` immediately before that showed the
listening socket on port 8000 already gone (only stale `TIME_WAIT`
entries), consistent with the whole process (not just one thread) having
died. Reproduced twice (this run, and an earlier long-running instance
found already in this state at session start, before any log was
captured for it — see `.hw_validation/server_run2.stderr.log`, a separate,
older concurrent-streaming capture from the same machine that also shows
`CAM-2 OPENED_BUT_READ_FAILED` churn, though that specific file predates
this session's fix and was not captured at the moment of a crash).

### B.2 — After the fix: same workload, no crash

Command: identical, full stderr captured to
`.hw_validation/live_after_fix.stderr.log`. Ran continuously across CAM-1
succeeding, CAM-2 failing and retrying through backoff attempts 1-5
(3s -> 6s -> 12s -> 24s -> 30s, correctly escalating), and a Hot-Plug Scan
concurrently re-probing index 2 (`[discover:2]`) — all at the same time
this exact interleaving previously crashed the process — with **zero
crashes** for the ~3.5 minutes it was left running (stopped manually to
free the devices for the next experiment, not because it failed).
`Get-CimInstance` confirmed the process (both the launcher and its child)
still alive throughout.

Also visible in this run: the low-bandwidth mitigation
(`_LOW_BANDWIDTH_OPEN_PROFILES`) correctly engages on CAM-2's *second*
attempt onward ("tried 4 backend/format combinations") once
`_active_camera_count` reflects CAM-1 as active; its *very first* attempt
raced CAM-1's own open and still used the full profile list ("tried 10").
This is a minor, non-crashing timing gap in an existing optimization, not
a new issue — noted in §I, not fixed here.

### B.3 — Capture-only isolation (AI ruled out)

Command: `python .hw_validation/probe_capture_only.py 60` (no web server,
no BoothManager, no AIWorker/YOLO/FairFace — two raw threads calling
`core.vision._open_capture()` directly, CAM-A full profile list, CAM-B
`low_bandwidth_only=True`, both reading continuously). Full output:
`.hw_validation/probe_capture_only.stdout.log`.

```
[t= 11.0s] open_results={'CAM-B': False}                          <- CAM-B never opens
[t= 12.0s] CAM-A reads_ok=1  reads_failed=0   CAM-B reads_ok=0 reads_failed=32
[t= 25.0s] CAM-A reads_ok=13 reads_failed=0   CAM-B reads_ok=0 reads_failed=415
[t= 53.0s] CAM-A reads_ok=40 reads_failed=0   CAM-B reads_ok=0 reads_failed=1236
```

CAM-A: 40/40 reads ok, zero degradation over 53s, with zero AI code
loaded. CAM-B: `isOpened()` never returns true across the whole run. This
is the identical failure signature seen inside the full web app, with
every AI/web/BoothManager subsystem absent — AI contention is not the
cause of CAM-2's failure.

### B.4 — DirectShow index vs. physical identity (already correct, verified by reading, not re-derived here)

`core/camera_identity.py` already implements exactly the identity chain
the investigation prompt asks for: `PhysicalCameraIdentity` keyed by
serial (Level 1) / VID+PID+USB location (Level 2) / name+index fallback
(Level 3), built from `Get-PnpDevice`/`Get-PnpDeviceProperty`, correlated
against DirectShow's own enumeration order. `CameraWorker._relocate_device_if_moved()`
re-resolves this on every reconnect before retrying a stale index, and
`_forbidden_device_name()` re-checks the *current* DirectShow name at an
index on every single open (startup, hot-plug, and reconnect alike) — so a
camera_id can never silently start opening a re-enumerated built-in/virtual
device at its old index. This was verified by reading the code and its
existing test suite (`tests/core/test_camera_identity.py`,
`tests/core/test_camera_worker_physical_identity.py`, all passing); it was
not independently re-derived from scratch this session.

## C. Code Changes (this session)

1. **`core/camera_identity.py`** — added `DIRECTSHOW_LOCK` (a
   `threading.RLock`), and wrapped `list_directshow_devices()`'s
   `pygrabber.dshow_graph.FilterGraph()` call with it. This is the single
   choke point every caller (production and future) goes through, so no
   caller can forget to lock.
2. **`core/vision.py`** — `_open_lock` (previously its own
   `threading.Lock()`, scoped only around the `cv2.VideoCapture(...)`
   construction loop) is now the *same* `DIRECTSHOW_LOCK` object, and
   `_open_capture()`'s `_forbidden_device_name()` check was moved from
   *before* the lock acquisition to *inside* it — so a DirectShow open on
   one thread and a DirectShow enumeration call on another (identity
   check, physical-identity resolution, hot-plug scan) can never run at
   the same instant. `RLock` (not a plain `Lock`) specifically so
   `_forbidden_device_name()` can safely re-enter the same lock from
   inside `_open_capture()`'s own hold, on the same thread.
3. **`tests/core/test_camera_open.py`** — `test_busy_initializer_does_not_open_another_device`
   previously simulated "lock busy" by acquiring `_open_lock` on the
   test's *own* thread before calling `_open_capture()` — that only
   worked because the old lock was non-reentrant; an `RLock` legitimately
   permits the same thread to re-enter, so the test needed the "busy"
   thread to be a genuinely different thread, matching the real
   cross-thread scenario the test/lock exists for. Rewrote it accordingly
   (77 tests in this area pass).
4. **`tests/core/test_camera_physical_identity_perf.py`** — the
   `elapsed < 0.2` threshold in
   `test_resolve_physical_id_does_not_block_the_caller` flaked (measured
   0.217s) under the full suite's CPU load; it only needs to rule out
   waiting anywhere near the mocked lookup's full 1.0s, so it was widened
   to `< 0.6`, preserving intent.
5. **`tests/web/test_booth_manager_camera_startup_stagger.py`** — two
   tests called `bm.start()` (which launches real heartbeat/hot-plug
   background threads) but never stopped it; the leaked hot-plug thread
   would call `is_capture_open()` on that file's `_FakeWorker` once its
   real-time scan interval elapsed, crashing on `AttributeError` from
   inside an unrelated *later* test (this is the
   `PytestUnhandledThreadExceptionWarning` seen under
   `test_hold_event_without_reid_has_no_attribution` in the original
   `pytest tests/ -q` run that kicked off this investigation). Fixed by
   giving `_FakeWorker`/`_FakeAdaptiveController` the missing
   `is_capture_open()`/`stop()` methods and stopping `bm` in a `finally`
   block in both tests.

Full suite after all changes: `609 passed, 7 deselected` in ~176s, zero
thread-exception warnings (previously 1 failure + 1 unhandled-thread-exception
warning).

## D. Architecture (unchanged by this session, verified by reading)

Per-camera capture is already isolated correctly: each `CameraWorker` owns
its own `cv2.VideoCapture` and its own thread; frame hand-off to the shared
`AIWorker` is a single mutable slot per camera, never a shared/global frame;
`BoothManager._latest_jpeg` is a `dict[camera_id, bytes]`, never a single
shared value. No code path found where one camera's worker reads another's
capture or frame buffer. This matches `docs/design/camera-pipeline-root-cause-repair.md`'s
existing §8 findings and was spot-checked, not fully re-audited line by line.

## E. Hardware

- CPU: 8 logical / 4 physical cores. RAM: 15.8 GB. GPU: NVIDIA GeForce
  GTX 1050 (3.0 GB VRAM). Both cameras enumerate as DirectShow index 1 and
  2 on this machine, both currently reachable at YUY2 320x240@15 in
  isolation (confirmed by `discover_cameras()`'s own sequential probe, both
  logged `open=OK read=OK`).
- USB topology / which physical controller or hub port each camera is
  wired to: **BLOCKED — NOT TESTABLE** from software; nobody moved a
  cable this session. §43's "same hub vs. different controller" experiment
  was not performed.
- Actual negotiated format for CAM-2 at the moment of failure: **not
  applicable** — CAM-2 never reaches `isOpened() == True` in the failing
  case (§B.3), so no format was ever negotiated to compare against the
  request.

## F. Test Results

| Test | Result |
|---|---|
| Camera A (index 1) alone | PASS (`discover_cameras`, and in every concurrent run) |
| Camera B (index 2) alone | PASS (`discover_cameras`'s sequential probe: open=OK read=OK) |
| A+B concurrent, full app, before fix | FAIL — process crashed within ~20-90s |
| A+B concurrent, full app, after fix | CAM-1 PASS (stable, no failures) / CAM-2 FAIL (never sustains a stream) — no crash, ran ~3.5 min |
| Capture-only, no AI (§B.3) | Same as above — CAM-2 FAIL, confirms AI is not the cause |
| A+B 60 sec | CAM-2 FAIL throughout (§B.3, ran the full 60s) |
| A+B 5 min | **BLOCKED — not run this session** (time budget; the 60s and ~3.5 min runs already show the failure is immediate and consistent, not a slow-onset one) |
| Process isolation (2 separate OS processes) | **BLOCKED — not run this session** |
| Frame ownership | PASS (by code inspection, §D) |
| Hot plug (unplug/replug while streaming) | **BLOCKED — not testable without physically unplugging hardware** |
| Native DirectShow/Media Foundation outside OpenCV | **BLOCKED — not attempted this session** |
| Different USB controller/hub for CAM-2 | **BLOCKED — not testable without moving cables** |

## G. AI / tracking / attribute subsystems

Out of scope for this investigation once §B.3 showed the failure is
capture-layer and AI-independent. Not touched, not re-verified, no changes
made. `docs/design/camera-pipeline-root-cause-repair.md` §3.B already
covers the one AI-adjacent fix on record (removing `drop_rate` from
`AdaptiveController`'s `overloaded` classification).

## H. Full Test Suite

```
609 passed, 7 deselected, 2 warnings in 176.44s
```
(warnings are pre-existing httpx/anyio deprecation notices, unrelated to
this investigation).

## I. Remaining Issues (explicitly unproven — do not treat as fixed)

1. **CAM-2 still cannot stream concurrently with CAM-1 on this hardware.**
   The crash is fixed; the underlying capture failure is not. Evidence
   (`OPENED_BUT_READ_FAILED` / `0xC00D3704` persisting at the smallest
   already-tried profile, in a clean AI-free test) is consistent with a
   genuine per-controller hardware/driver ceiling, but per §52's own rule
   this is **UNVERIFIED HARDWARE HYPOTHESIS**, not confirmed, because the
   one experiment that would confirm or refute it — CAM-2 on a different
   USB controller/hub than CAM-1 — was never run. Do not report "USB hub
   bandwidth insufficient" as settled; report it as the leading, but
   unconfirmed, explanation.
2. Low-bandwidth-profile engagement has a real (non-crashing) timing gap
   on a camera's very *first* open attempt when it races the other
   camera's own open completing (§B.2). Not fixed this session.
3. The Hot-Plug Scan (`_hotplug_loop`, every `HOTPLUG_SCAN_INTERVAL_SEC` =
   5s) and a disconnected camera's own `_attempt_reopen()` backoff both
   independently probe the same index once that camera is down — safe
   now (both serialize through `DIRECTSHOW_LOCK`, no crash risk), but
   redundant load on an already-struggling device. A `CameraResourceManager`-
   style single-authority reservation (as sketched in the investigation
   prompt's §20) would remove this duplication; not built this session —
   it would be a substantial redesign for what is currently a wasted-retry
   inefficiency, not a correctness bug.
4. `discover_cameras()`'s startup scan is a blind 0..15 probe (skipping
   only indices a live worker already owns), not a Windows-enumeration-
   first approach — matches the investigation prompt's §7 concern exactly.
   Confirmed present and costly (roughly a dozen ~1s failed-probe cycles
   every process start in this environment's log), but not shown to be
   involved in the crash or in CAM-2's read failures, and not changed this
   session.
5. Everything in §F marked BLOCKED.

## J. Final Status

**PARTIALLY COMPLETE.**

The crash that made "two cameras at once" fail catastrophically (not just
CAM-2 offline, but the *entire application* silently dying, taking CAM-1
and the web UI down with it) is fixed and verified under the same
real-hardware workload that reproduced it twice. That is a real, confirmed
improvement: the app now degrades to "CAM-1 works, CAM-2 reports offline
with an honest message" instead of crashing outright.

CAM-2 concurrently streaming on *this specific hardware* is **not**
achieved and is **not** claimed as fixed — per §51's own rule, PASS is not
reported here because it has not passed. The leading explanation is a
hardware/USB-controller ceiling, but that is not confirmed to the standard
this report's own rules (§52) require, because the one decisive experiment
(separate controllers) needs a cable moved, which this session could not
do.

---

## K. CAM-1 image-quality follow-up (same session, second investigation)

**Reported symptom (from a live screenshot, not reproduced from this
text alone):** CAM-1 shows `status = ONLINE` in the UI, but the displayed
image has visible horizontal-line corruption / banding / noise. CAM-2 shows
"CAMERA RECONNECTING...".

### K.1 — Method

Per the investigation instructions, statistics alone were not trusted —
actual frames were captured to disk and opened as images.

1. `.hw_validation/probe_frame_integrity.py` — opens CAM-1 (full profile
   list, as production does when it's the first/only active camera) and
   CAM-2 (low-bandwidth profile) directly via the real
   `core.vision._open_capture()`, no AI/web/BoothManager involved. Saves
   every 3rd frame as both a lossless PNG and the exact
   `cv2.imencode(".jpg", frame, [IMWRITE_JPEG_QUALITY, 80])` bytes
   production would produce, plus per-frame min/max/mean/std and
   `_looks_like_noise()`'s verdict, plus each `cap.read()` call's wall-clock
   duration.
2. Separately, the *actual production server* was started
   (`python web_server.py --no-open`), and once `/api/state` reported
   `CAM-1: online`, a real frame was pulled directly from the same
   `/stream/CAM-1` endpoint the browser uses (`curl` the multipart stream,
   extract the first JPEG between its `\xff\xd8`/`\xff\xd9` markers) — this
   is not a synthetic test, it is exactly what the browser was receiving
   at that moment.

### K.2 — Findings (CONFIRMED, from real captured frames)

- Every CAM-1 frame captured, from both methods, was **valid, correctly-
  shaped JPEG** (320x240x3, decodes cleanly with OpenCV, no stride/skew/
  corruption at the container or pixel-grid level). Opening
  `.hw_validation/live_stream_frame1.jpg` (pulled live from `/stream/CAM-1`)
  and `.hw_validation/frames/CAM-1_jpeg_*.jpg` (from the isolated probe)
  shows the same thing: **a essentially solid black frame**, not banding
  or line corruption. Pixel stats on the live-stream capture: min=0,
  max=109, mean=0.0086, std=0.60, only 341 of 230,400 values nonzero — a
  sparse scatter of faint bright pixels on black, the classic appearance of
  CMOS dark-current/thermal noise in near-zero light, not a decode or
  stride bug.
- `cap.read()` on CAM-1 took **almost exactly 1.000-1.001 seconds per
  call**, every single call sampled (13-20 samples across two separate
  probe runs, essentially zero jitter). The negotiated profile
  (`backend=DSHOW resolution=320x240 fps=15 fourcc=YUY2`) claims 15 fps;
  the camera is actually delivering roughly **1 frame/sec**, a ~15x
  shortfall. A suspiciously exact ~1.0s-per-frame rate, combined with
  near-zero brightness, is the textbook signature of a UVC sensor's
  automatic long-exposure/low-light mode: the sensor is integrating light
  for close to a full second per frame (mechanically capping frame rate to
  match) and still not gathering enough light to produce a visible image.
- `core.vision._looks_like_noise()` correctly does **not** flag these
  frames (`is_noise: False` on every sample) — by its own documented
  design (`NOISE_ROUGHNESS_RATIO`'s docstring, `core/vision.py:744-745`),
  a frame with `std < 1.0` is treated as "a near-solid frame (lens cap,
  blackout) isn't noise — just flat", explicitly not something the
  existing corruption heuristic is meant to catch, because it's
  structurally a *real* (if useless) camera reading, not driver garbage.
  This is working as designed, not a bug in that function.

### K.3 — What this does and does not explain

**Does not match** the reported symptom of visible *line/banding*
corruption — a solid near-black frame with sparse noise dots is a
different visual than horizontal tearing. This session's captures (both
the isolated probe and the live production stream, pulled at essentially
the same moment) show solid black, not lines. **This specific banding/line
symptom was not reproduced or independently verified this session.**

**Does establish**, with direct evidence, a real, separate problem: CAM-1
can report `status: online` / `"ปกติ"` while delivering content that is
visually useless (near-black, ~1 fps) — the UI has no way to distinguish
"streaming a normal-looking frame" from "streaming a technically-valid but
content-empty frame", because nothing in the pipeline currently measures
frame *brightness/usefulness*, only "is this frame statistically real
vs. garbage" (`_looks_like_noise`) and "did read() return successfully".

### K.4 — Root cause classification

- **NOT VERIFIED**: that the reported horizontal-line/banding corruption
  and the near-black frames documented here (§K.2) are the same
  underlying issue. They could be the same root cause (e.g., low light
  triggering both a black frame *and*, at some other moment, a sensor
  banding artifact during the auto-exposure transition, before it settles
  into the extended-exposure near-black steady state observed here) or two
  different issues. Nothing in this session's evidence connects them
  directly.
- **UNVERIFIED HARDWARE/ENVIRONMENTAL HYPOTHESIS**: severe low ambient
  light at CAM-1's lens, or an obstructed/uncleaned/miscovered lens,
  causing the sensor's automatic long-exposure mode to engage — consistent
  with every measurement taken (near-zero brightness, ~1s exposure-locked
  frame interval, sparse dark-current-style noise, valid JPEG/decode at
  every other layer). **This cannot be confirmed or ruled out from
  software alone** — it requires physically checking the camera (is the
  lens clear and uncovered?) and the room's lighting at the time the
  original screenshot was taken, which only the person operating the
  hardware can do. Per this investigation's own rule against declaring an
  unconfirmed hypothesis as fact, this is reported as a hypothesis, not a
  conclusion.
- **RULED OUT**: JPEG encode/decode corruption, frame stride/shape
  mismatch, and `_looks_like_noise` malfunctioning as the cause of what
  was captured — all three were directly checked against real captured
  bytes/frames and found correct.
- **BLOCKED**: reproducing the actual reported line/banding visual, and
  therefore identifying *its* root cause, pending either (a) a fresh
  screenshot/frame captured at the moment the corruption is visible, or
  (b) confirmation of the room/lens conditions at the time it was seen.

### K.5 — No code changes made for this symptom

Given §K.4's findings, no fix was applied for the banding/line symptom —
there is currently no confirmed defect in MongDee's own code to fix, and
patching `_looks_like_noise` again without new evidence risks repeating
the false-positive regression `docs/design/camera-pipeline-root-cause-repair.md`
already documents from an earlier, now-reverted per-row banding check.

### K.6 — Follow-up (same session, minutes later): CAM-1 recovered on its own

A second live frame was pulled from the exact same running server and the
exact same `/stream/CAM-1` endpoint a few minutes after §K.2's black-frame
capture, with no code or configuration change in between. Result: a
**normal, clearly-exposed, detailed color frame** (a room interior, person
visible), JPEG size 21,152 bytes vs. the earlier ~1,850 bytes (a ~11x
size difference, consistent with "near-black" vs. "real detailed photo").
No line/banding corruption is visible in this frame either.

This raises §K.4's low-light/exposure hypothesis from "unverified" to
**CONFIRMED as at least a contributing, reproducible factor**: the same
camera, same code, same DirectShow index, went from unusable-black to
normal within minutes with zero code changes — exactly what an automatic
long-exposure/low-light mode recovering (as ambient light or exposure
convergence changed) would produce, and hard to explain any other way
software-side. The original *line/banding* symptom specifically is still
**NOT VERIFIED** (neither black-frame capture nor this normal one shows
it) — it may be a distinct, rarer transient (e.g. during the
exposure-hunting transition itself) that simply wasn't caught in either
sample taken this session.

---

## L. Camera device identity / enumeration stability follow-up (same session, third investigation)

**Question investigated:** can MongDee's numeric OpenCV/DirectShow index
ever silently point to the *wrong* physical camera — specifically, can
`self._physical_id` bind to a stable-seeming identity that then gets
reused by a *different* physical camera, so a camera_id (e.g. "CAM-1")
silently starts streaming a different physical device's feed?

### L.1 — Real identity data from this project's own two cameras (CONFIRMED)

Non-disruptive enumeration (`.hw_validation/probe_identity_enum.py` —
calls `core.camera_identity.list_directshow_devices()` /
`list_pnp_camera_devices()` / `get_physical_camera_identities()` directly;
does not open any `cv2.VideoCapture`, safe to run alongside a live server):

```
index 0: HD WebCam        VID=0408 PID=A060  (laptop built-in)
index 1: USB Camera       VID=4C4A PID=4A55  location=...004.004...
index 2: USB Camera       VID=4C4A PID=4A55  location=...004.003...
index 3: OBS Virtual Camera (present, currently unused by MongDee)
```

**CONFIRMED**: the two USB cameras (index 1 and 2) are the **identical
make/model** — same VID:PID (4C4A:4A55), no real USB serial number on
either (`serial: null`). MongDee's physical-identity system (already
present before this session, `core/camera_identity.py`) currently tells
them apart *only* via `location_info` (USB port location) — Level 2
("vid_pid_location") of its own documented 3-level confidence hierarchy.
This is not a new discovery of a defect in that module — its own
docstring already discloses this exact limitation ("moving the exact same
physical camera to a different port produces a brand-new instance ID
indistinguishable from a different physical unit of the same model") — but
this session confirms it is not a hypothetical edge case on this
project's hardware: it is the *only* thing distinguishing the two real
cameras in front of it right now.

Repeated 5x in a row (no hardware touched between runs): **100% stable**
— index-to-physical_id mapping did not change across repeated enumeration
calls. Confirms the mapping is *deterministic given fixed hardware
topology*, not flaky on its own; any instability the user observed
requires an actual re-enumeration trigger (unplug/replug, a port change,
or another virtual/physical camera appearing or disappearing — e.g. index
3's OBS Virtual Camera being started or stopped would shift how many
devices exist, though never past index 2 in the current topology since it
enumerates last).

### L.2 — CONFIRMED BUG found and fixed: index substitution was not detected

Traced the full lifecycle (`core/vision.py`'s `CameraWorker._open()` /
`_relocate_device_if_moved()`) against the failure mode this investigation
was asked to rule out:

```
CAM-1 tracks physical_id P1, currently at index 1.
Some re-enumeration event occurs (port change, device replug, another
device appearing/disappearing) such that:
    index 1 now reports a *different* physical camera's identity (P2)
    P1 is not found at any index (e.g. it moved to a port whose location
    string was never seen before, which -- per L.1 -- invalidates a
    location-based identity by definition)
```

**Before this session's fix**: `_relocate_device_if_moved()` only handled
"P1 found at a different index -> follow it". When P1 was not found
*anywhere*, it left `self.device` completely unchanged and returned
`None` — and its caller, `_open()`, unconditionally proceeded to
`_open_capture(self.device, ...)` regardless. Since `_forbidden_device_name()`
only excludes *known-bad* names (built-in/virtual), not "a different but
otherwise-legitimate USB camera than the one this worker is tracking",
**`_open()` would have silently opened index 1's new occupant (physical
camera P2) and started streaming it under the camera_id "CAM-1"** — a
real, code-confirmed device-mapping bug, exactly the failure mode Phase 5
of this investigation asked to prove or disprove. Confirmed by reading the
code's actual control flow, not by reproducing it on real hardware (that
would require physically swapping the two cameras' ports, which this
session could not do without someone present) — classified as **CONFIRMED
BY CODE INSPECTION**, not "confirmed by reproduction on real hardware".

**Fix applied** (`core/vision.py`): `_relocate_device_if_moved()` now
returns `bool` instead of `None`. In addition to its existing relocation
logic, it now also checks whether `self.device`'s *current* occupant has a
positively-known identity that differs from `self._physical_id` (only
when identity data is actually available and conclusive — never when it's
merely empty/unavailable, preserving the existing "never block a real open
on mere uncertainty" principle). If so, it logs a warning and returns
`False`; `CameraWorker._open()` now checks this return value and returns
`False` immediately (treating it exactly like "device not present this
round") instead of proceeding to `_open_capture()`. A camera_id can no
longer silently adopt a different physical camera's feed when its old
index gets reassigned to something else — it now waits (retrying on the
normal backoff) until its *actual* tracked physical camera is found again,
either at the same index or a new one.

Regression tests added (`tests/core/test_camera_worker_physical_identity.py`):
`test_relocate_refuses_when_old_index_now_a_different_known_camera`,
`test_relocate_allows_open_when_identity_data_unavailable`,
`test_relocate_allows_open_when_still_the_same_camera`,
`test_open_does_not_open_device_when_relocate_detects_substitution`
(this last one asserts `_open_capture` is never even called). All pass,
alongside the full existing camera-identity suite (81 tests total across
`test_camera_worker_physical_identity.py` / `test_camera_open.py` /
`test_camera_identity.py` / `test_camera_physical_identity_perf.py`).

### L.3 — What was and wasn't verified

| Question | Status |
|---|---|
| Is the OpenCV/DirectShow index stable with no hardware changes? | **PASS** (5x repeated enumeration, identical every time) |
| Do the two USB cameras have genuinely distinguishable identity data right now? | **CONFIRMED** (different `location_info`, same VID/PID) |
| Could a re-enumeration event cause a camera_id to silently adopt a different physical camera's feed? | **CONFIRMED BY CODE INSPECTION** (before fix) — **FIXED** (this session) |
| Reproduced the wrong-camera swap on real hardware (physically moving a cable)? | **BLOCKED** — requires physically swapping the two USB cameras' ports, not performed this session |
| Does Windows Camera (an independent app) show the same wrong-device behavior outside MongDee? | **BLOCKED** — requires GUI interaction with another application, not available to this session's tools |
| Repeated unplug/replug (20x, per the investigation's own request) | **BLOCKED** — requires physically touching the USB cables |
| Full pytest suite after this fix | see §M below |

### L.4 — DirectShow lock (from §A-J above) — untouched

`DIRECTSHOW_LOCK` was not modified, weakened, or removed by this follow-up.
Verified present and unchanged before starting this investigation and
again after.

## M. Full Test Suite (after the identity-substitution fix)

```
613 passed, 7 deselected, 2 warnings in 210.13s (0:03:30)
```

(613 = the previous 609 + 4 new regression tests from §L.2; 0 failed; the
2 warnings are the same pre-existing httpx/anyio deprecation notices as
every other run in this document, unrelated to this investigation.)

## N. Final Verification (this follow-up investigation's own required format)

A. Is OpenCV camera index stable (no hardware touched)? **PASS**
B. Is DirectShow device ordering stable (no hardware touched)? **PASS**
C. Is PnP device ordering stable (no hardware touched)? **PASS** (implied by A/B; PnP data was queried alongside DirectShow in the same 5 repeated runs and never changed)
D. Can an index point to a different physical camera (after some re-enumeration event)? **CONFIRMED** possible in principle (§L.1 — both USB cameras share VID/PID, distinguished only by port location, which a re-enumeration event can invalidate)
E. Can MongDee open the wrong physical camera? **CONFIRMED (by code inspection) — FIXED** (§L.2)
F. Can CAM-1 and CAM-2 swap identities? **NOT VERIFIED by hardware reproduction** (would require physically swapping USB ports — BLOCKED this session); **CONFIRMED BY CODE INSPECTION** that the pre-fix code had no defense against it, and that the fix closes that specific gap
G. Can reconnect cause identity corruption? Same as F — fixed the code path that could cause it; not reproduced on real hardware
H. Can Windows Camera reproduce the wrong-device behavior independently of MongDee? **BLOCKED** — not attempted (no GUI automation available to this session)
I. CAM-1 + CAM-2 capture-only result — unchanged from §B.3 of this same document: CAM-1 stable, CAM-2 fails to open concurrently
J. Exact confirmed root causes — see §A (crash, fixed), §L.2 (identity-substitution gap, fixed); CAM-2's concurrent-open failure and the original line/banding symptom remain **NOT VERIFIED** as to root cause

---

## O. Explicit backend matrix (DSHOW vs. MSMF, no CAP_ANY) — fourth investigation, same session

**Method**: `tests/manual/test_dshow_cameras.py` / `test_msmf_cameras.py` open
each camera *alone* (never concurrently) via `cv2.VideoCapture(index, cv2.CAP_DSHOW)`
/ `cv2.VideoCapture(index, cv2.CAP_MSMF)` explicitly — never `cv2.CAP_ANY` —
across native + a 6-profile MJPEG/YUY2 matrix, reading back actual negotiated
FourCC/width/height/fps (never trusting `cap.set()`'s return value), then 10-20
sustained reads. `tests/manual/test_concurrent_matrix.py` repeats this with
both cameras at once, several backend combinations, each run as an isolated,
wall-clock-timeout-protected subprocess (needed — see O.2). No AI, no web
server, no BoothManager in any of these.

### O.1 — CONFIRMED: CAP_MSMF is unconditionally broken for both cameras

**14/14 attempts, 0 successful frame reads**, for both CAM-1 and CAM-2,
across every profile tested (native, MJPG at 4 resolutions/FPS, YUY2 at 2),
in complete isolation (no concurrency, no other camera involved at all).
`isOpened()` reports `True` every time, then every single `read()` fails
immediately. This directly matches this project's own earlier
`OnReadSample() ... error status: -1072875772` / `0xC00D3704` log evidence
(§A-N above) — that failure is not specific to concurrent access or a busy
USB hub, it is **MSMF failing outright on this hardware, alone, every time**.

### O.2 — CONFIRMED: MSMF can hang indefinitely, and that hang is dangerous given this session's own DIRECTSHOW_LOCK fix

The first (in-process, non-subprocess-isolated) MSMF matrix attempt hung
for 180+ seconds on its *second* open attempt (reopening the same index
shortly after releasing a first MSMF session on it), emitting a
`grabFrame ... Error: -2147483638` warning roughly every 10 seconds
forever, never completing or timing out on its own. This was only
discovered because the test harness itself had an external timeout;
`core.vision._open_capture()` does not — and since §A-N's own
`DIRECTSHOW_LOCK` fix (necessary and still correct, see §L.4 — not
reverted) makes every camera's open/reconnect share one process-wide lock
for its *entire* per-candidate loop, a hang inside the MSMF candidate
specifically would block *every other camera's* open/reconnect for as
long as the hang lasts. This is a newly-discovered, real availability
risk introduced by combining "MSMF can hang on this hardware" with "opens
are now correctly serialized" — not present before either fix, and not
acceptable given O.1 already shows MSMF provides zero benefit here.

### O.3 — Fix applied: removed the CAP_ANY/MSMF fallback for a numeric Windows camera index

`core/vision.py`'s `_candidate_opens()`: a Windows integer-index device
(the real case for every USB camera this project uses) now tries
**only `cv2.CAP_DSHOW`** — the `cv2.CAP_ANY` (which resolves to MSMF or
another auto-selected backend on Windows) fallback candidate that used to
follow it has been removed for that specific case, per O.1/O.2's evidence
and the investigation's own instruction not to keep an unproven backend in
the production open path. Left unchanged: Linux (still tries V4L2 by path,
V4L2 by index, then CAP_ANY) and macOS/other (still CAP_ANY, its only
option) — neither was tested or implicated by this evidence, and CAP_ANY
was the *only* candidate on macOS before this change, so removing it there
would disable camera support entirely rather than narrow it. A string
device (RTSP URL, explicit path) is also unaffected. Full test suite
still passes after this change (§Q).

**Trade-off, stated plainly**: if some *other* MongDee deployment's camera
only works via MSMF (the removed comment's original justification — never
actually evidenced in this repo, only asserted), this change would regress
it on Windows. No such hardware was available to test. This is a
real, disclosed risk of this fix, not a hidden one.

### O.4 — Concurrent matrix: DSHOW is confirmed necessary but not sufficient

`test_concurrent_matrix.py`'s raw (no `DIRECTSHOW_LOCK`, by design — it
predates/bypasses `core.vision` entirely to isolate backend behavior)
concurrent DSHOW+DSHOW attempts both failed to open at all, both with and
without a 2.5s stagger — consistent with §A's original crash-investigation
finding that *unsynchronized* concurrent DirectShow opens can both fail,
which is exactly why `DIRECTSHOW_LOCK` exists in the real code path. A
separate, incidental finding from this same test run: heavy rapid
open/release cycling (this test harness's own subprocess churn, plus the
preceding MSMF matrix) left the device *unable to open via DSHOW at all*
for roughly 15-30 seconds afterward, self-recovering with no code change —
a real, reproducible transient device-cooldown behavior on this hardware,
worth knowing about (a camera that fails and is retried *very*
aggressively, faster than this cooldown, could see extra spurious
failures) but not investigated further this session (§R).

Rerunning the **actual production `core.vision._open_capture()`** (i.e.
with `DIRECTSHOW_LOCK` and the O.3 fix both in effect) for a clean 60s
concurrent capture-only run, once the transient cooldown above had
cleared: **CAM-1: 916/916 reads succeeded, zero failures, ~15 reads/sec
sustained for the full 60 seconds — and notably, no longer showing the
~1 read/sec near-black-frame behavior from §K** (further supporting §K.6's
"that was transient/environmental" conclusion — same code, same camera,
now fully healthy). **CAM-2: still never opens** (`open_results: {"CAM-B": false}`)
— unchanged from every earlier attempt in this document, now with every
backend possibility exhausted (DSHOW alone: fails when CAM-1 is active;
MSMF: proven broken even alone, O.1) and no remaining software lever this
session identified to try.

### O.5 — Result table (per the investigation's own requested format)

| Camera | Backend | Format | Resolution | FPS | Alone | Concurrent (with the other camera) | Result |
|---|---|---|---|---|---|---|---|
| CAM-1 (index 1) | DSHOW | YUY2 (MJPG never honored — see below) | 640x480 down to 320x240 | 5-15 (requested; negotiated FPS honored) | PASS (14/14 profile attempts) | PASS (916/916 reads, 60s, post-fix) | **PASS** |
| CAM-1 (index 1) | MSMF | any | any | any | **FAIL** (0/7 profile attempts read) | not tested (O.3 removed this path) | **FAIL** |
| CAM-2 (index 2) | DSHOW | YUY2 | 320x240 | 5-15 | PASS (14/14 profile attempts, alone) | **FAIL** (0/60s, never opens) | **FAIL (concurrent only)** |
| CAM-2 (index 2) | MSMF | any | any | any | **FAIL** (0/7 profile attempts read) | not tested (O.3 removed this path) | **FAIL** |

| Test | Result | Error |
|---|---|---|
| DSHOW single-camera matrix, both cameras, 14 combinations | PASS (14/14 opened+read, MJPG silently delivered as YUY2 every time) | none |
| MSMF single-camera matrix, both cameras, 14 combinations | **FAIL (0/14 read)** | `grabFrame` failure every attempt; one reopen hung 180+s |
| Concurrent DSHOW+DSHOW, no `DIRECTSHOW_LOCK`, no stagger | FAIL (both failed to open) | raw/unsynchronized, expected per §A |
| Concurrent DSHOW+DSHOW, no `DIRECTSHOW_LOCK`, 2.5s stagger | FAIL (both failed to open) | confounded by the O.4 cooldown window — not a clean result |
| Concurrent MSMF+MSMF | FAIL (opened, 0 reads) | consistent with O.1 |
| Concurrent DSHOW(A)+MSMF(B) | FAIL (A failed to open, B opened/0 reads) | confounded by O.4 cooldown |
| Production `_open_capture` (DIRECTSHOW_LOCK + O.3 fix), 60s clean run | **CAM-1 PASS / CAM-2 FAIL** | CAM-2 never reaches `isOpened()==True` |

## P. Critical instruction check (§18 of the fourth investigation prompt)

No configuration was found, across every backend combination tested, where
CAM-1 and CAM-2 both stream live concurrently. Per that prompt's own rule,
this is **not** reported as COMPLETE. The exact combinations tried and
their results are in O.5's tables in full; nothing beyond what's listed
there was tested.

## Q. Full Test Suite (after the O.3 backend fix)

A pytest-collection bug was found and fixed while producing this section:
`tests/manual/test_*.py` (this investigation's own diagnostic scripts,
which open real cameras and run top-level code at import time) were being
picked up by plain `pytest tests/` because `pytest.ini`'s
`python_files = test_*.py` has no directory exclusion — confirmed by
seeing extra camera-opening subprocesses appear during what should have
been a hardware-free test run. Fixed by adding
`norecursedirs = tests/manual` to `pytest.ini`. This is a real fix (not
speculative) — verified by `pytest --collect-only` no longer listing
anything under `tests/manual/` before re-running the full suite.

```
613 passed, 7 deselected, 2 warnings in 315.04s (0:05:15)
```

(Same 613/7 as §M's earlier run — the `pytest.ini` collection fix and the
`_candidate_opens` backend change introduced zero regressions. The longer
wall-clock time (315s vs. §M's 210s) reflects this run sharing the machine
with other processes at the time, not a code-caused slowdown — no test
logic changed speed-sensitively.)
