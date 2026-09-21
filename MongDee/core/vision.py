"""AI Vision: per-camera capture + detection threads.

Framework-agnostic on purpose: this module only depends on cv2/numpy/threading,
not on any GUI toolkit, so the exact same detection pipeline runs inside the
PySide6 desktop app (via ui/qt_camera_bridge.py, which adapts the callbacks
below into Qt Signals) and inside the browser-facing web server
(web/server.py, which reads the callbacks directly since it has no GUI
thread to marshal onto). Each camera runs in its own thread so N webcams run
concurrently. Two independent detectors feed the same overlay/aggregator:

  1. YOLO11 (COCO classes) — cheap, exact, but only knows 80 generic object
     types. Used for "person" (always) and for demo products whose catalog
     key happens to be a COCO class name (bottle, cup, ...).
  2. ForegroundProposer + ProductRecognizer (core/localizer.py,
     core/recognizer.py) — class-agnostic: finds "something is being shown"
     via background subtraction, then identifies *which* trained product it
     is via image-embedding matching. This is what makes real, arbitrary
     products (never in COCO) recognizable once someone has uploaded a few
     training photos through the AI Trainer.

Capture/display and AI inference are fully decoupled onto separate threads.
Each CameraWorker's own thread only ever does capture -> draw last-known
boxes -> deliver via on_frame() -> repeat; it never calls YOLO or the
embedding recognizer itself and so can never be stalled by them, regardless
of how slow inference is or how many other cameras are also asking for it.
All AI work (YOLO + custom recognition, for every camera) instead runs on
one shared AIWorker background thread that round-robins whichever
registered camera is next due for a pass (see AIWorker, CameraWorker.
run_ai_pass()). This keeps the total number of concurrent inference calls
in the whole process at exactly one no matter how many cameras are open —
adding a camera means it shares the same fixed AI budget with the others
instead of adding its own competing inference thread — which is also why
neither ultralytics' nor torch's forward pass needing to be single-threaded
is a real constraint here any more (there is only ever one caller), though
_inference_lock is kept as a defensive no-op for any less-common caller that
passes its own dedicated AIWorker per camera.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time

import cv2
import numpy as np

from core.attributes import attenuate_confidence, face_min_confidence
from core.box_motion import BoxFollower
from core.face import FaceService
from core.frame_integrity import FrameIntegrity
from core.localizer import ForegroundProposer, crop_box, exclude_human_region
from core.person_segmenter import get_person_segmenter
from core.product_confirm import ProductConfirmer, product_confirm_hits_for, product_min_confidence
from core.text_render import blit as blit_label, render_label
from core.tracker import PersonTracker

logger = logging.getLogger("mongdee.core.vision")

FAIL_THRESHOLD = 12          # consecutive failed reads before a camera is flagged offline (~0.5 s at 30 fps)
# cap.read() on DirectShow has no timeout: a stalled USB camera (0xC00D3704 / 0x8007001F) blocks its
# worker thread forever while the tile keeps showing the last frame as "online". A watchdog (see
# CameraWorker.recover_from_hung_read) treats a read that has not returned for this long as a hang.
CAPTURE_READ_STALL_SEC = 3.0
# recover_from_hung_read() releases the capture from the watchdog thread, which unblocks a stuck
# read on backends that honour a release from another thread — the fast path, proven in
# tests/core/test_camera_unplug_replug.py::test_a_hung_read_is_recovered_without_blocking_the_watchdog.
# But that same test also shows the underlying native call itself does not always return even after
# release() (a genuinely wedged DirectShow/MSMF graph): when that happens, this worker's OWN thread
# stays parked inside the OS driver call forever, and nothing short of killing the process can
# reclaim it (see core/capture_process.py's module docstring). After this many consecutive hang
# recoveries with no good frame in between, this camera permanently switches to process-isolated
# capture (core.capture_process.CaptureProcess) for the rest of this run: reads go through shared
# memory and never block this thread again, and a wedged child is killed/respawned by its own
# watchdog instead of leaking this camera's worker. 0 disables escalation (old behaviour only).
HANG_ESCALATION_THRESHOLD = int(os.environ.get("MONGDEE_CAPTURE_ESCALATE_AFTER", "3"))
# A USB camera that stops delivering GOOD frames (unplugged, frozen buffer, torn/noise burst, hung read)
# is reported offline after this long - the tile then shows the "reconnecting" card at once instead of a
# frozen or garbled picture. Shortened to STREAM_STALE_SUSPECT_SEC for SUSPECT_WINDOW_SEC after the
# device monitor (core/device_presence.py) sees a camera disappear from Windows' device list.
STREAM_STALE_SEC = 1.0
STREAM_STALE_SUSPECT_SEC = 0.4
SUSPECT_WINDOW_SEC = 3.0
# A stream that is bit-for-bit identical this long is frozen (a live sensor is never that still): its frames
# are dropped as corrupt (which reconnects the camera) and, in the strict window, it is reported offline sooner.
FROZEN_STREAM_SEC = 1.5
FROZEN_STREAM_SUSPECT_SEC = 0.5
ONLINE_GOOD_STREAK = 3            # good frames in a row before an offline camera is called online again
# cap.release() can itself block forever on a wedged DirectShow graph. It is done on a helper thread; the
# worker waits this long for it, then carries on (a reopen waits for it up to RELEASE_WAIT_BEFORE_REOPEN_SEC).
RELEASE_JOIN_TIMEOUT_SEC = 1.5
RELEASE_WAIT_BEFORE_REOPEN_SEC = 2.0
# While Windows still lists the camera, retry at most this far apart (an unplugged one is not retried at all
# until it re-appears). Repeated corruption/hang recoveries switch the camera to the small low-bandwidth
# profiles for a while: fast motion makes MJPEG frames bigger, and a marginal USB link then tears them.
PRESENT_REOPEN_BACKOFF_MAX_SEC = 6.0
FAULT_WINDOW_SEC = 120.0
FAULTS_BEFORE_LOW_BANDWIDTH = 2
LOW_BANDWIDTH_HOLD_SEC = 300.0
RELOCATE_PROBE_INTERVAL_SEC = 3.0
# Reopen retry delay backs off per camera (3s -> 6 -> 12 -> 24 -> 30, capped)
# instead of a fixed interval. A camera stuck on a hardware-level failure —
# e.g. Windows Media Foundation's MF_E_HW_MFT_FAILED_START_STREAMING
# (0xC00D3704), returned when a USB controller/hub can't grant a second
# camera the hardware resources to start streaming while another camera on
# the same controller is already active — will never open no matter how
# often it's retried, so hammering it every 3s forever only adds load
# (repeated DirectShow/MSMF enumeration) without any chance of success. The
# backoff resets to the initial delay the moment an open actually succeeds,
# so a camera that recovers (unplugged/replugged, other camera released)
# still reconnects promptly.
REOPEN_BACKOFF_INITIAL_SEC = 3.0
REOPEN_BACKOFF_MAX_SEC = 30.0
REOPEN_BACKOFF_MULTIPLIER = 2.0
OPEN_WARMUP_READS = 3        # a freshly opened webcam often needs a beat before its first real frame
OPEN_WARMUP_SLEEP_SEC = 0.05
# GPU-class defaults (CUDA available) — a modern GPU eats these easily.
DETECT_EVERY_N_FRAMES = 2    # run detection every Nth frame to keep multi-camera FPS reasonable
DEFAULT_AI_IMGSZ = 640       # YOLO input size; core.performance.AdaptiveController may lower this under load
# CPU-only defaults — CPU YOLO inference is the single heaviest cost in this
# whole pipeline (see AIWorker below), so a CPU-only box starts already
# throttled instead of paying full 640px/every-2nd-frame cost until
# AdaptiveController walks it down over several ticks. Only used when
# CameraWorker's device_target == "cpu" (see __init__); a CUDA GPU keeps the
# lighter-restriction defaults above untouched, matching "don't force CPU
# settings onto a GPU box".
DETECT_EVERY_N_FRAMES_CPU = 8
DEFAULT_AI_IMGSZ_CPU = 320
AI_BASE_FPS = 30.0            # matches the FPS requested from the camera in _open() — used to turn
                               # detect_every_n_frames into a wall-clock AI cadence (see AIWorker/run_ai_pass)
PRODUCT_CONFIRM_HITS = 2                  # an embedding-matched product must be seen in this many passes
# Reference floor for product_min_confidence's size-based scaling in _run_custom_recognition
# (below) -- an independent constant, not a live read of core.recognizer.MATCH_FLOOR, since
# CameraWorker deliberately never imports core.recognizer (the recognizer is a duck-typed
# constructor argument, not a concrete dependency). Set to MATCH_FLOOR's own default value; if an
# operator tunes recognizer.identify()'s floor separately, only the SIZE-BASED extra requirement
# below is affected, never the recognizer's own unscaled floor for full-trust-size boxes (this
# constant is only ever used to compute how much HIGHER a small box's floor should be, and is
# never passed through for a box at or above full-trust size -- see the "> CUSTOM_RECOGNITION_
# BASE_FLOOR" check at the call site).
CUSTOM_RECOGNITION_BASE_FLOOR = 0.45
CUSTOM_RECOGNITION_EVERY_N_AI_PASSES = 2  # embedding-based custom recognition runs at half the YOLO/AI pass rate
MIN_CROP_SIDE_PX = 24        # ignore foreground blobs too small to embed meaningfully
AI_RECOVERY_SUCCESS_COUNT = 3  # successful passes required before clearing an AI error
STOP_JOIN_TIMEOUT_SEC = 3.0
AI_WORKER_IDLE_SLEEP_SEC = 0.02  # AIWorker's poll granularity when nothing is due yet

PERSON_CLASS_NAME = "person"
# Spec (MongDee person-gender/age master prompt, section 3/5/78): a detected
# person must NEVER fall back to a generic red "Person" box — every person
# category, including "no confident gender/age guess", gets its own distinct
# non-red color from BOX_COLORS below. PERSON_UNKNOWN_COLOR (teal) is what a
# person renders as when gender_age_backend is unset, or its prediction
# didn't clear MIN_GENDER_CONFIDENCE/MIN_AGE_CONFIDENCE — an honest "category
# unknown", the 3rd Person Category alongside male/female, never a
# leftover "generic person" concept. (There is no child category: the product
# only reports male / female / product.)
PERSON_UNKNOWN_COLOR = (170, 170, 60)  # BGR teal — คน (ไม่ทราบเพศ/อายุ — Unknown)
PERSON_FEMALE_COLOR = (102, 102, 255)  # BGR red   — คนที่ระบบจำแนกว่าเป็นผู้หญิง
PERSON_MALE_COLOR = (255, 179, 94)     # BGR blue  — คนที่ระบบจำแนกว่าเป็นผู้ชาย
PRODUCT_COLOR = (0, 200, 0)       # BGR green — สินค้า (รู้จักแล้ว)
FACE_COLOR = (255, 200, 0)        # BGR cyan-blue — ใบหน้า (face detection overlay)
# STRICT UNKNOWN DETECTION rule (PRODUCTS ONLY, see _run_custom_recognition):
# an unmatched product candidate is never rendered at all -- a box promises
# the viewer "this is a real registered product", so an unresolved candidate
# gets no box, not a placeholder. This does NOT apply to people: a detected
# person is always shown (PERSON_UNKNOWN_COLOR / "PERSON NN%") the moment
# they're tracked, even before gender evidence is sufficient -- hiding the
# person track until classified would contradict detect-first tracking
# (person-gender evidence master prompt, "PERSON must be detected before
# gender is known"). Only products get the render-nothing-until-confident
# treatment; a pending person keeps a stable box and label.
TRIPWIRE_LINE_COLOR = (0, 165, 255)     # BGR orange — เส้นนับคน (Virtual Tripwire)
TRIPWIRE_TEXT_COLOR = (255, 255, 255)   # BGR white — ตัวเลข IN/OUT บนเส้น
BOX_THICKNESS = 2
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55

# Spec section 5's BOX_COLORS — one place mapping every rendered category to
# its color, so "which color means what" is never scattered ad hoc across
# _category_label_and_color's if/elif chain and the frontend legend.
BOX_COLORS = {
    "male": PERSON_MALE_COLOR,
    "female": PERSON_FEMALE_COLOR,
    "product": PRODUCT_COLOR,
    "unknown": PERSON_UNKNOWN_COLOR,
}

# A track's gender/age label needs this much summed classifier confidence AND this much lead over the
# runner-up category (see core.tracker._mode_category) before it is shown. Simulated at 82% per-frame
# accuracy: a single confident-looking wrong frame used to become the label 17.5% of the time; now it is
# "unknown" until a second agreeing frame arrives (error after 6 frames 3.3% -> 0.5%).
PERSON_CATEGORY_MIN_EVIDENCE = 1.6   # i.e. at least two agreeing frames at >= 0.8 confidence
PERSON_CATEGORY_MARGIN = 0.8


# Anti-flicker (person boxes blinking on/off and every blink minting a new ID):
#  * the detector runs at PERSON_LOW_CONF; detections below the camera's normal threshold may only extend
#    an existing track (they never start one), so a person whose confidence hovers around the threshold
#    keeps one steady track;
#  * a new track must be confirmed (seen twice, or detected with confidence >= PERSON_BIRTH_IMMEDIATE_SCORE)
#    before it is reported, so a one-frame false detection never becomes a person / Re-ID identity;
#  * a track survives PERSON_TRACK_MAX_AGE_SEC without detections and its predicted box is still drawn for
#    PERSON_COAST_SEC after a miss.
BOX_FOLLOW_ENABLED = True     # boxes follow the picture at the capture frame rate (core/box_motion.py)
PERSON_LOW_CONF = 0.25
PERSON_BIRTH_IMMEDIATE_SCORE = 0.70
PERSON_MIN_HITS = 2
PERSON_COAST_SEC = 0.6
PERSON_TRACK_MAX_AGE_SEC = 2.5


def _make_person_tracker() -> PersonTracker:
    return PersonTracker(category_min_evidence=PERSON_CATEGORY_MIN_EVIDENCE,
                         category_margin=PERSON_CATEGORY_MARGIN, min_hits=PERSON_MIN_HITS,
                         birth_immediate_score=PERSON_BIRTH_IMMEDIATE_SCORE, coast_sec=PERSON_COAST_SEC,
                         max_age_sec=PERSON_TRACK_MAX_AGE_SEC)
MIN_GENDER_CONFIDENCE = 0.70      # below this, the box stays the neutral PERSON_UNKNOWN_COLOR

_inference_lock = threading.Lock()

# core.camera_identity.get_physical_camera_identities() shells out to
# PowerShell (Get-PnpDevice) and has been measured taking several seconds on
# real hardware. _relocate_device_if_moved() (called synchronously at the
# start of every reconnect attempt, see CameraWorker._open()) needs its
# result before it can decide which index to retry, so unlike
# _resolve_physical_id() (made fully async — see its own docstring) it can't
# just fire-and-forget; this short cache instead collapses repeated calls
# across a burst of reconnect attempts (several cameras reconnecting at
# once, or the same camera retrying) into one real PowerShell call. Windows
# PnP topology doesn't meaningfully change moment to moment outside of an
# actual hot-plug event, so a few seconds of staleness here is an
# acceptable trade for not re-paying a multi-second cost on every single
# retry. Kept in core/vision.py rather than inside core/camera_identity.py's
# own list_pnp_camera_devices()/get_physical_camera_identities() so those
# functions keep their existing "always a fresh call" contract that
# tests/core/test_camera_identity.py's per-call mocking relies on.
_PHYSICAL_IDENTITY_CACHE_TTL_SEC = 10.0
_physical_identity_cache_lock = threading.Lock()
_physical_identity_cache: tuple[float, dict] | None = None


def _cached_physical_camera_identities() -> dict:
    global _physical_identity_cache
    now = time.monotonic()
    with _physical_identity_cache_lock:
        if _physical_identity_cache is not None and now - _physical_identity_cache[0] < _PHYSICAL_IDENTITY_CACHE_TTL_SEC:
            return _physical_identity_cache[1]
    from core.camera_identity import get_physical_camera_identities

    identities = get_physical_camera_identities()
    with _physical_identity_cache_lock:
        _physical_identity_cache = (now, identities)
    return identities


class AIWorker(threading.Thread):
    """Single shared background thread that runs AI (YOLO + custom
    recognition) for every registered camera, one at a time — see module
    docstring for why this replaces per-camera inline inference.

    Cameras register themselves at the start of CameraWorker.run() and
    unregister when that loop exits; nothing else needs to construct this
    directly unless it wants an *isolated* AI budget for a specific set of
    cameras (e.g. tests) — the default is one shared instance per process,
    lazily created by _get_default_ai_worker().

    Fairness: a plain round-robin scan of registered cameras, each skipped
    until its own `ai_due()` says its configured interval has elapsed. This
    naturally rate-limits the *total* inference throughput to whatever this
    one thread can sustain (bounded by how long a single YOLO call takes),
    instead of every camera independently trying to hit its own cadence and
    piling up CPU as more cameras are added.
    """

    def __init__(self):
        super().__init__(daemon=True, name="mongdee-ai-worker")
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._cameras: dict[str, "CameraWorker"] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._paused = False

    def register(self, worker: "CameraWorker") -> None:
        with self._lock:
            if worker.camera_id not in self._cameras:
                self._order.append(worker.camera_id)
            self._cameras[worker.camera_id] = worker

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)
            try:
                self._order.remove(camera_id)
            except ValueError:
                pass

    def set_paused(self, paused: bool) -> None:
        """core.performance.AdaptiveController's AI-Pause lever (see its
        docstring): while paused, this thread simply stops calling
        run_ai_pass() on anyone. Every camera's capture/display loop is
        completely unaffected, since it never depended on this thread to
        begin with — Camera Preview keeps running at full rate."""
        self._paused = paused

    def is_paused(self) -> bool:
        return self._paused

    def stop(self) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=STOP_JOIN_TIMEOUT_SEC)

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._paused:
                self._stop_event.wait(AI_WORKER_IDLE_SLEEP_SEC)
                continue
            worker = self._next_due_camera()
            if worker is None:
                self._stop_event.wait(AI_WORKER_IDLE_SLEEP_SEC)
                continue
            try:
                worker.run_ai_pass()
            except Exception:
                logger.exception("AI worker: run_ai_pass failed for camera %s", worker.camera_id)

    def _next_due_camera(self) -> "CameraWorker | None":
        now = time.monotonic()
        with self._lock:
            n = len(self._order)
            if n == 0:
                return None
            for _ in range(n):
                camera_id = self._order[self._cursor % n]
                self._cursor = (self._cursor + 1) % n
                worker = self._cameras.get(camera_id)
                if worker is not None and worker.ai_due(now):
                    return worker
        return None


_default_ai_worker_lock = threading.Lock()
_default_ai_worker: "AIWorker | None" = None


def _get_default_ai_worker() -> AIWorker:
    """The process-wide AIWorker every CameraWorker uses unless its caller
    passes its own (see web/booth_manager.py, which does — one explicit
    AIWorker per BoothManager, shared by every camera in that booth).
    Created lazily so merely importing/constructing a CameraWorker (e.g. in
    unit tests) never starts a background thread — only actually starting a
    camera (.start() -> run()) does."""
    global _default_ai_worker
    with _default_ai_worker_lock:
        if _default_ai_worker is None:
            _default_ai_worker = AIWorker()
            _default_ai_worker.start()
        return _default_ai_worker


def _draw_box(frame, bbox, label, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    if not label.isascii():
        # Thai product names: cv2.putText cannot draw them, so the label is rendered with a Thai font
        # (core/text_render.py) and pasted. Without a usable font fall through to an ASCII-safe label.
        patch = render_label(label, color)
        if patch is not None:
            blit_label(frame, patch, x1, y1)
            return
        label = label.encode("ascii", "ignore").decode().strip()
        if not any(ch.isalpha() for ch in label):
            label = f"PRODUCT {label}".strip()
    (tw, th), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, 2)
    label_top = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, label_top), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), LABEL_FONT, LABEL_SCALE, (255, 255, 255), 2)


def _draw_tripwire(frame, overlay: dict) -> None:
    """Draws a configured Virtual Tripwire line + a small IN/OUT arrow +
    running counters directly onto a captured frame — called from
    CameraWorker.run()'s own capture-thread draw loop (see its call site),
    the same loop that already draws person/product boxes every single
    captured frame regardless of AI pass rate. This is what keeps the line
    visible continuously even while AI is paused/throttled (core.performance
    .AdaptiveController's AI Pause, or a slow AIWorker cadence): drawing
    never depends on a fresh AI pass having just run, only on `overlay`
    holding whatever the last AI pass computed.

    `overlay`: {"x1","y1","x2","y2"} normalized 0..1, "inside_side" ("A"/"B"),
    "count_in", "count_out" — see CameraWorker.set_tripwire_overlay()."""
    h, w = frame.shape[:2]
    x1, y1 = int(overlay["x1"] * w), int(overlay["y1"] * h)
    x2, y2 = int(overlay["x2"] * w), int(overlay["y2"] * h)
    cv2.line(frame, (x1, y1), (x2, y2), TRIPWIRE_LINE_COLOR, 3)
    for cx, cy in ((x1, y1), (x2, y2)):
        cv2.circle(frame, (cx, cy), 6, TRIPWIRE_LINE_COLOR, -1)

    # A short perpendicular arrow at the line's midpoint pointing toward
    # "inside" — a plain line alone doesn't tell a viewer which direction
    # counts as IN, which is exactly the ambiguity the spec's "OUT <- LINE
    # -> IN" arrow requirement exists to remove.
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = (x2 - x1), (y2 - y1)
    length = (dx * dx + dy * dy) ** 0.5
    if length > 0:
        # Perpendicular unit vector; sign picked so it points toward
        # whichever side is configured as "inside" (matches
        # core.tripwire.TripwireLine.side_of's cross-product convention).
        px, py = -dy / length, dx / length
        if overlay["inside_side"] == "B":
            px, py = -px, -py
        arrow_len = 28
        tip = (int(mx + px * arrow_len), int(my + py * arrow_len))
        cv2.arrowedLine(frame, (int(mx), int(my)), tip, TRIPWIRE_LINE_COLOR, 2, tipLength=0.35)

    label = f"IN {overlay['count_in']}  OUT {overlay['count_out']}"
    (tw, th), baseline = cv2.getTextSize(label, LABEL_FONT, LABEL_SCALE, 2)
    label_x = max(0, min(int(mx) - tw // 2, w - tw - 6))
    label_y = max(th + baseline + 4, int(my) - 14)
    cv2.rectangle(frame, (label_x - 3, label_y - th - baseline - 2),
                  (label_x + tw + 3, label_y + baseline), TRIPWIRE_LINE_COLOR, -1)
    cv2.putText(frame, label, (label_x, label_y), LABEL_FONT, LABEL_SCALE, TRIPWIRE_TEXT_COLOR, 2)


def _ascii_label(text: str) -> str:
    """cv2.putText's built-in Hershey font can't render Thai glyphs (they come
    out as "??"), so any non-ASCII product key falls back to a generic word
    for the on-frame label; the real Thai name still shows in the AI Product
    Assistant panel, which Qt renders correctly."""
    return text.upper() if text.isascii() else "PRODUCT"


def _candidate_opens(device) -> list[tuple]:
    """Different OpenCV/V4L2 builds disagree about which of "open by device
    path" vs. "open by integer index" actually works — one build refuses
    "/dev/videoN" with a "can't be used to capture by name" warning, another
    refuses a plain index with "can't open camera by index", for the exact
    same physical webcam. So try every reasonable (device, backend)
    combination and let the caller use whichever one actually opens, instead
    of betting on one. Index-based open is also the only option at all on
    Windows/macOS, where /dev/video* doesn't exist.

    On Windows, cv2.CAP_ANY resolves to MSMF (or another auto-selected
    backend, e.g. obsensor), which on some laptops opens the device
    (isOpened() == True) but then fails every frame read with
    "OnReadSample() is called with error status". DirectShow is tried
    first; a numeric Windows index deliberately has *no* CAP_ANY/MSMF
    fallback below (see next paragraph) -- real hardware evidence on this
    project's own dev machine (`tests/manual/test_msmf_cameras.py`,
    `docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md`'s backend-matrix findings)
    measured CAP_MSMF failing every single frame read for *both* of its
    USB cameras, at every resolution/FourCC/FPS combination tried (14/14
    attempts, 0 successes) -- never a working fallback on this hardware,
    only wasted time. Worse, MSMF was also observed to *hang* indefinitely
    on a read call in one run; since _open_capture holds a process-wide
    lock (DIRECTSHOW_LOCK, shared with core.camera_identity) for its whole
    per-candidate loop, a hung MSMF candidate would block every *other*
    camera's open/reconnect for as long as the hang lasts -- a real
    availability risk with a measured zero-benefit, evidence-based reason
    to remove, not a hypothetical one. CAP_ANY is left in place for a
    string device path (the branch below) and for Linux, where it was not
    implicated by this evidence.
    """
    is_linux = sys.platform.startswith("linux")
    is_windows = sys.platform == "win32"
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    candidates = []
    if isinstance(device, int):
        if is_linux:
            candidates.append((f"/dev/video{device}", cv2.CAP_V4L2))
            candidates.append((device, cv2.CAP_V4L2))
            candidates.append((device, cv2.CAP_ANY))
        elif is_windows:
            candidates.append((device, cv2.CAP_DSHOW))
        else:
            candidates.append((device, cv2.CAP_ANY))
    else:  # explicit device path string, e.g. "/dev/video0"
        if is_linux:
            candidates.append((device, cv2.CAP_V4L2))
            match = re.search(r"(\d+)$", device)
            if match:
                candidates.append((int(match.group(1)), cv2.CAP_V4L2))
        candidates.append((device, cv2.CAP_ANY))
    return candidates


from core.camera_identity import DIRECTSHOW_LOCK

# Shared with core.camera_identity.list_directshow_devices() (see that
# lock's own docstring for the real crash this fixes) -- not a separate
# lock, so a DirectShow open here and a DirectShow enumeration call there
# (e.g. another CameraWorker's _forbidden_device_name()/
# _relocate_device_if_moved() running on its own thread) never touch
# DirectShow's COM machinery at the same moment. An RLock: _open_capture
# below holds it for its whole per-candidate loop and calls
# _forbidden_device_name() -- which itself acquires this same lock -- from
# inside that hold.
_open_lock = DIRECTSHOW_LOCK
OPEN_LOCK_TIMEOUT_SEC = 5.0

# See core.capture_process.CROSS_PROCESS_OPEN_LOCK's own docstring: this additionally serializes
# _open_capture against an escalated camera's own open, which happens in a different OS process and
# so cannot be reached by the in-process _open_lock above. Importing it here (module load time, in
# whichever process runs this code) is correct for every *normal* caller in this process — the
# parent only ever has one such lock instance for its whole lifetime. An escalated camera's child
# process must NOT rely on a fresh import of this to get "the same" lock (a spawned child's fresh
# import creates an unrelated instance); make_negotiated_capture_for_process below is always called
# with the parent's actual instance passed explicitly instead, for exactly that reason.
import core.capture_process as _capture_process
from core.capture_process import CROSS_PROCESS_OPEN_LOCK as _CROSS_PROCESS_OPEN_LOCK

# The (codec, width, height, fps) profiles _open_capture negotiates, in
# order — None means "leave the camera's own native/default format alone".
# Compressed formats first and shrinking resolution/FPS before ever falling
# to an uncompressed format: a raw/uncompressed stream can exhaust a shared
# USB bus (or a hub's hardware-encoder streaming budget — see
# MF_E_HW_MFT_FAILED_START_STREAMING / 0xC00D3704 in CameraWorker's
# docstring) before a lower-bandwidth format ever gets a chance to be tried.
_OPEN_PROFILES = (("MJPG", 640, 480, 30), ("MJPG", 640, 480, 15), ("MJPG", 320, 240, 15),
                   ("YUY2", 320, 240, 15), None)

# Real hardware evidence on this project (.hw_validation/*.stderr,
# 2026-09-18): a second USB camera opens and reads fine in isolation, and
# even opens fine at the full MJPG 640x480 profile when its start is
# staggered after the first camera -- but during sustained concurrent
# streaming it eventually fails with Windows MediaFoundation error
# -1072875772 (0xC00D3704, MF_E_HW_MFT_FAILED_START_STREAMING: the OS
# refusing to grant a second concurrent hardware MJPEG-decode/streaming
# resource) and then cannot reopen at any profile, consistent with a
# shared USB hub/controller resource limit rather than a per-camera
# defect (see docs/design/camera-pipeline-root-cause-repair.md and the
# CAM-2 shared-hub theory in CameraWorker._offline_message below).
# Software cannot manufacture USB bandwidth that isn't there, but it can
# stop *requesting* the largest, most resource-hungry profile for every
# camera after the first one is already streaming -- so a second/third
# concurrent camera tries only the smallest, already-compressed profiles
# first instead of unconditionally requesting 640x480@30 and only backing
# off after that request has already failed. Never applied to whichever
# camera opens first (no other camera is active yet, so there is nothing
# to conserve bandwidth for).
_LOW_BANDWIDTH_OPEN_PROFILES = (("MJPG", 320, 240, 15), ("YUY2", 320, 240, 15))

def preferred_capture_profile():
    """A capture profile requested with the MONGDEE_CAPTURE_SIZE environment variable, e.g. "1280x720" or
    "1280x720@15" (MJPG assumed), or None. A far-away person is only a few dozen pixels tall in a 640x480
    stream, and no software can recover a face from that; a larger capture size is the one thing that gives the
    face/body models more pixels to work with. It costs USB bandwidth and CPU, so it is opt-in, and when it is
    set it applies to every camera (the low-bandwidth rule for 2nd/3rd cameras is then off)."""
    raw = os.environ.get("MONGDEE_CAPTURE_SIZE", "").strip().lower()
    match = re.fullmatch(r"(\d{3,4})x(\d{3,4})(?:@(\d{1,3}))?", raw)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    fps = int(match.group(3)) if match.group(3) else 30
    return ("MJPG", width, height, fps)


def _open_profiles():
    preferred = preferred_capture_profile()
    return ((preferred,) + _OPEN_PROFILES) if preferred else _OPEN_PROFILES


_active_camera_count = 0
_active_camera_count_lock = threading.Lock()


def _low_bandwidth_open_preferred() -> bool:
    """True when at least one other CameraWorker already has an open,
    reading capture -- see _LOW_BANDWIDTH_OPEN_PROFILES's docstring for why
    that's when a *new* camera's own open should prefer the smallest
    profiles instead of the largest. Never counts the caller's own
    not-yet-opened stream (see CameraWorker._mark_camera_active/_inactive:
    a worker only increments the shared counter *after* its own open
    succeeds), so this is naturally "how many *other* cameras are active"
    with no self-exclusion bookkeeping needed."""
    with _active_camera_count_lock:
        return _active_camera_count > 0


def _profile_label(profile) -> str:
    if profile is None:
        return "native"
    codec, width, height, fps = profile
    return f"{codec} {width}x{height}@{fps}"


def _backend_name(backend: int) -> str:
    try:
        return cv2.videoio_registry.getBackendName(backend)
    except Exception:
        return str(backend)


def _describe_capture(cap) -> str:
    """Best-effort actual (not requested) stream properties for diagnostic
    logging — never raises, since a test double/mock capture won't implement
    getBackendName()/get()."""
    try:
        actual_backend = cap.getBackendName()
    except Exception:
        actual_backend = "?"
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"
    except Exception:
        w = h = fps = 0
        fourcc = "?"
    return f"backend={actual_backend} resolution={w}x{h} fps={fps:.0f} fourcc={fourcc}"


def _classify_open_failure(attempts: list[tuple[str, str, bool, bool]]) -> str:
    """attempts: list of (backend_name, profile_label, opened, read_ok).
    Turns raw attempt results into one of the failure-stage labels used
    throughout logging/diagnostics (see camera/README.md-style status
    vocabulary): every candidate either never opened at all
    (OPEN_FAILED — driver/backend/device-busy problem), opened but could
    never deliver a frame (OPENED_BUT_READ_FAILED — the isOpened()==True,
    read()==False case, e.g. Windows' MF_E_HW_MFT_FAILED_START_STREAMING /
    0xC00D3704 when a USB controller can't grant the hardware resources for
    a second concurrent camera stream), or there were no candidates to try
    at all (DISCOVERY_FAILED)."""
    if not attempts:
        return "DISCOVERY_FAILED"
    if any(opened for _, _, opened, _ in attempts):
        return "OPENED_BUT_READ_FAILED"
    return "OPEN_FAILED"


# Which DirectShow device *names* _open_capture must refuse to open — set
# once at startup from booth_settings.json (see
# web.booth_manager.load_camera_settings) via set_forbidden_device_names().
# Re-checked fresh on every single _open_capture call — initial discovery,
# Hot-Plug Scan, AND every per-camera reconnect — rather than only once at
# first assignment. This closes a real bug observed live during
# development: a CameraWorker already bound to an index (assigned when
# that index really was a USB camera) reconnected after a drop and picked
# up the *built-in* camera instead, because Windows had silently
# re-enumerated devices in the meantime and nothing ever re-validated the
# index's identity. A device's index is not a stable identity; its
# DirectShow name is comparatively stable (see core.camera_identity).
_forbidden_device_names: frozenset[str] = frozenset()


def set_forbidden_device_names(names) -> None:
    global _forbidden_device_names
    _forbidden_device_names = frozenset(str(n).lower() for n in names)


def _forbidden_device_name(device) -> str | None:
    """The current DirectShow name at `device`'s index, if it's one
    _open_capture must refuse — a virtual camera (always) or a configured
    built-in name — else None. Only meaningful for an integer/numeric
    index (Windows); a device path string (e.g. Linux's /dev/videoN) has
    nothing to resolve here and is always allowed through."""
    try:
        index = int(device)
    except (TypeError, ValueError):
        return None
    from core.camera_identity import is_virtual_camera_name, list_directshow_devices

    name = list_directshow_devices().get(index)
    if name is None:
        return None
    if is_virtual_camera_name(name) or name.lower() in _forbidden_device_names:
        return name
    return None


def _release_capture_safely(cap: cv2.VideoCapture, label: str) -> threading.Event:
    """Release `cap` on its own daemon thread and wait up to
    RELEASE_JOIN_TIMEOUT_SEC for it to finish, returning the completion
    Event either way -- never blocks the caller past that timeout.

    cap.release() can itself block forever on a wedged DirectShow graph --
    real, observed behaviour on this project's own hardware (see
    RELEASE_JOIN_TIMEOUT_SEC's own comment); CameraWorker._release_capture()
    already guards against it for a worker's own long-lived capture. This
    is the same guard as a free function, for every OTHER place a capture
    from _open_capture() gets released -- which matters even more there:
    a wedge inside _open_capture()'s own candidate-rejection loop (called
    while holding the shared _open_lock) would hold that lock forever and
    silently stop every camera in the process from ever opening again;
    a wedge inside discover_cameras() (which BoothManager's Hot-Plug Scan
    calls forever, every 5-30s, for the entire life of the process -- by
    far its most frequent caller) would permanently kill hot-plug
    recovery. Both match this project's own reported failure mode exactly
    ("works fine for a while, then one or all cameras get stuck
    'connecting' until the process is restarted") and neither used to go
    through this protection -- every release of an _open_capture() result
    must, from here on."""
    done = threading.Event()

    def _do_release():
        try:
            cap.release()
        except Exception:
            pass
        finally:
            done.set()

    threading.Thread(target=_do_release, daemon=True, name=f"mongdee-release-{label}").start()
    done.wait(RELEASE_JOIN_TIMEOUT_SEC)
    return done


def _open_capture(device, label: str | None = None, low_bandwidth_only: bool = False,
                   cross_process_lock=None) -> cv2.VideoCapture:
    """Open under a bounded process-wide lock and require one real frame.

    isOpened() alone isn't proof a backend works — MSMF (and, on other
    hardware, DSHOW) can happily open a device handle and then fail every
    frame read forever. So each candidate must prove it delivers a frame
    before it's accepted; a backend that opens but never reads is skipped
    in favor of the next candidate instead of being returned as "working".

    The bounded lock prevents concurrent DirectShow/MSMF initialization from
    making otherwise healthy cameras fail during startup.

    `label` tags every log line (e.g. "CAM-3") so a multi-camera failure can
    be told apart in the log; defaults to the raw device when the caller
    doesn't have a camera_id yet (discover_cameras' probing).
    """
    tag = label if label is not None else f"device={device!r}"
    # Opening several cameras at (nearly) the same moment is exactly what
    # happens on startup — BoothManager starts one CameraWorker thread per
    # camera back-to-back, and each immediately tries to open its device.
    # DirectShow/MSMF's device enumeration on Windows isn't reliably
    # reentrant across threads: opening two cameras concurrently can make
    # one of them fail to open even though it's plugged in and otherwise
    # fine (a well-known OpenCV/Windows quirk) -- and, far worse, can crash
    # the whole process outright (see DIRECTSHOW_LOCK's own docstring in
    # core/camera_identity.py for the real incident this fixes). The
    # forbidden-device-name check below calls list_directshow_devices(),
    # which touches the exact same DirectShow COM machinery as the
    # cv2.VideoCapture(..., CAP_DSHOW) calls further down -- both must be
    # inside this same lock's hold, not just the VideoCapture calls, or two
    # threads can still collide between them. Reads aren't affected — only
    # this open handshake is serialized — so this costs nothing once
    # cameras are up and running, and only adds a small, one-time delay
    # while several cameras start up together.
    #
    # A busy initializer must not trigger another concurrent backend open.
    # Return a closed handle on timeout; each worker retries independently.
    got_lock = _open_lock.acquire(timeout=OPEN_LOCK_TIMEOUT_SEC)
    if not got_lock:
        logger.debug("[%s] open skipped: initializer lock busy, will retry", tag)
        return cv2.VideoCapture()
    cp_lock = cross_process_lock if cross_process_lock is not None else _CROSS_PROCESS_OPEN_LOCK
    # An explicit cross_process_lock (an escalated camera's own child, see
    # make_negotiated_capture_for_process) always serializes -- it is only ever passed once a
    # second OS process genuinely exists. Otherwise, skip this real OS semaphore entirely until
    # note_process_capture_starting() reports that some camera, somewhere, has actually escalated:
    # before that, every open in the whole process's history has come from this one process, so
    # there is nothing to serialize against and paying a multiprocessing.Lock's per-call cost
    # (meaningfully higher than the plain in-process RLock above) on every single camera open
    # would be pure overhead for what is, in the overwhelming common case, its entire lifetime.
    need_cp_lock = cross_process_lock is not None or _capture_process.cross_process_lock_active()
    got_cp_lock = cp_lock.acquire(timeout=OPEN_LOCK_TIMEOUT_SEC) if need_cp_lock else True
    if not got_cp_lock:
        logger.debug("[%s] open skipped: cross-process initializer lock busy, will retry", tag)
        _open_lock.release()
        return cv2.VideoCapture()
    attempts: list[tuple[str, str, bool, bool]] = []
    try:
        forbidden_name = _forbidden_device_name(device)
        if forbidden_name is not None:
            logger.warning(
                "[%s] index=%s currently identifies as %r — excluded (built-in/virtual), refusing "
                "to open even though this index may previously have been a different, allowed "
                "device (Windows re-enumerated in the meantime)",
                tag, device, forbidden_name,
            )
            return cv2.VideoCapture()
        for dev, backend in _candidate_opens(device):
            backend_name = _backend_name(backend)
            # Configure before the first read: the default raw stream may
            # exhaust a shared USB bus before MJPEG can ever be selected.
            profiles = _LOW_BANDWIDTH_OPEN_PROFILES if low_bandwidth_only else _open_profiles()
            for profile in profiles:
                profile_name = _profile_label(profile)
                cap = cv2.VideoCapture(dev, backend)
                accepted = False
                opened = read_ok = False
                try:
                    opened = bool(cap.isOpened())
                    if opened:
                        requested_fourcc = None
                        if profile is not None:
                            codec, width, height, fps = profile
                            requested_fourcc = cv2.VideoWriter_fourcc(*codec)
                            cap.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            cap.set(cv2.CAP_PROP_FPS, fps)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        read_ok = _read_with_warmup(cap)
                        if read_ok:
                            # MJPG exists in this list specifically to keep a
                            # camera's USB bandwidth low on a shared hub. Real-
                            # hardware evidence (see docs/design/camera-pipeline-
                            # root-cause-repair.md): some UVC devices silently
                            # ignore the FourCC negotiation and deliver their
                            # uncompressed native format instead while still
                            # reporting read=OK -- accepting that here would make
                            # _open_capture "succeed" at exactly the highest-
                            # bandwidth outcome of every profile tried, before
                            # ever reaching the deliberately-small explicit YUY2
                            # fallback further down this same list, which is
                            # what actually fits a bandwidth-constrained bus.
                            fourcc_honored = (
                                requested_fourcc is None
                                or profile[0] != "MJPG"
                                or int(cap.get(cv2.CAP_PROP_FOURCC)) == requested_fourcc
                            )
                            if fourcc_honored:
                                accepted = True
                                logger.info(
                                    "[%s] index=%s backend=%s format=%s open=OK read=OK (%s)",
                                    tag, dev, backend_name, profile_name, _describe_capture(cap),
                                )
                                return cap
                            logger.debug(
                                "[%s] index=%s backend=%s format=%s: device ignored the MJPG "
                                "request and delivered %s instead -- rejecting so a smaller/"
                                "explicit-format profile is tried next instead of silently "
                                "accepting this bandwidth footprint",
                                tag, dev, backend_name, profile_name, _describe_capture(cap),
                            )
                except cv2.error:
                    pass  # try the next format/backend after releasing this handle
                finally:
                    attempts.append((backend_name, profile_name, opened, read_ok))
                    logger.debug(
                        "[%s] index=%s backend=%s format=%s open=%s read=%s",
                        tag, dev, backend_name, profile_name,
                        "OK" if opened else "FAILED", "OK" if read_ok else "FAILED",
                    )
                    if not accepted:
                        # Async + bounded (see _release_capture_safely's docstring): this runs
                        # while still holding _open_lock, so a wedged release here -- most likely
                        # exactly for a candidate that just failed to open/stream, i.e. this one --
                        # would otherwise hold that lock forever and stop every camera in the
                        # process from opening again, not just this one.
                        _release_capture_safely(cap, f"{tag}-reject")
                if not opened:
                    # isOpened() is False before any format is even requested, so the profile cannot be the
                    # cause: the device is absent or busy. Retrying it with every other profile only repeats
                    # the same ~1 s failure while holding the process-wide DirectShow lock.
                    break
        # Never reopen an unverified default backend after all probes fail.
        logger.warning(
            "[%s] index=%s state=%s (tried %d backend/format combinations, none produced a frame)",
            tag, device, _classify_open_failure(attempts), len(attempts),
        )
        return cv2.VideoCapture()
    finally:
        if need_cp_lock and got_cp_lock:
            cp_lock.release()
        if got_lock:
            _open_lock.release()


def make_negotiated_capture_for_process(device=None, label: str | None = None, cross_process_lock=None):
    """core.capture_process.CaptureProcess factory for an escalated camera's own OS process (see
    CameraWorker._ensure_process_capture). Runs the exact same backend/resolution negotiation as
    every other camera's open (_open_capture above) -- an escalated camera keeps the hard-won
    MJPG/YUY2/low-bandwidth fallback behaviour instead of falling back to a naive, unnegotiated
    cv2.VideoCapture(index) that would silently regress to whatever format the device defaults to.

    `cross_process_lock` is CROSS_PROCESS_OPEN_LOCK, passed in explicitly by the parent (see that
    lock's own docstring for why a plain import here would not be the same lock instance). Raises
    when every candidate fails, which is what CaptureProcess._child_main expects: it marks the
    shared-memory slot STATE_FAIL and returns, and the parent's watchdog kills+respawns this child
    on its own bounded backoff -- so an escalated camera's failed opens are retried exactly like a
    thread-mode camera's, just one level further out.
    """
    cap = _open_capture(device, label=label, cross_process_lock=cross_process_lock)
    if not cap.isOpened():
        raise RuntimeError(f"escalated capture: cannot open device={device!r}")
    return cap


# How much higher a frame's local pixel-to-pixel roughness is allowed to be
# relative to its own overall contrast before _looks_like_noise rejects it.
# Calibrated against real captures from this project: genuine photos (even
# dark/grainy ones) measured ~0.13; independent-random-per-pixel garbage
# measured ~1.1 — a wide margin either side of this threshold. Re-validated
# against 250 real photos from this project's own FairFace training set
# (dataset/train/) — max observed 0.393, comfortably under this threshold,
# zero false positives.
#
# An earlier version of this module also had a *per-row* variant of this
# same check (BANDING_ROUGHNESS_RATIO), meant to catch a corrupted band
# confined to a few rows that a whole-frame average dilutes away. It was
# removed after that same 250-photo validation turned up 2 real, legitimate
# photos (a sharp lighting/shadow edge, common in real portraits) scoring
# above its threshold — a real false-positive rate against exactly the
# content this booth's camera needs to keep showing. NOISE_CROSS_CHANNEL_CORR
# below already independently catches every banded/torn-frame case this was
# meant for (see its own docstring and this module's test suite), with a
# validated safety margin the roughness-based version didn't have, so
# removing it costs no real coverage.
NOISE_ROUGHNESS_RATIO = 0.5
# How correlated the R/G/B channels' horizontal-diff signals must be, at
# minimum, for a frame to still count as real. A real optical image's edges
# and texture are luminance-driven — a boundary/edge/hair strand moves R, G,
# and B together at the same pixel — so their horizontal-diff signals stay
# strongly correlated even in a genuinely noisy/grainy real photo. Sensor/
# decode garbage (no real optics behind it) does not share that constraint
# and comes out far less correlated across channels.
#
# This is the check that actually caught a real live incident this project
# hit: a physically-connected "USB Camera" producing colorful TV-static-like
# garbage (isOpened=True, read()=True, frame.size>0 — nothing else about the
# read looked wrong) scored well *inside* NOISE_ROUGHNESS_RATIO's "this is a
# real image" range, because real, detailed, high-contrast photos — this
# booth's actual normal workload, e.g. FairFace's face photos — turned out
# to score in that same roughness range too, so tightening that threshold to
# catch the garbage would also have started rejecting real people's faces.
# Cross-channel correlation is what actually separated them cleanly:
# checked against 250 real photos from this project's own FairFace training
# set (min observed 0.372) versus the real garbage frame (0.12-0.18) and
# synthetic pure noise (0.01) and banded/torn-frame garbage (0.03-0.11) —
# every real photo sampled stayed above this threshold, every garbage case
# stayed below it, with roughly a 2x margin on the garbage side and a
# smaller but still positive margin on the real-photo side (the closer of
# the two, since it's the side a false positive against real content would
# come from).
NOISE_CROSS_CHANNEL_CORR = 0.3


def _looks_like_noise(frame) -> bool:
    """True if `frame` is very likely decode/driver garbage rather than a
    real camera image.

    Catches two distinct failure shapes observed in practice:

    1. The *whole frame* is garbage — "device silently disconnected
       mid-stream but its capture handle kept returning ok=True with
       whatever was left in a stale buffer", which shows up as uniform,
       spatially-uncorrelated random noise across the entire frame. Caught
       by NOISE_ROUGHNESS_RATIO: mean absolute difference between
       horizontally adjacent pixels, relative to the frame's own standard
       deviation.
    2. The frame is garbage in a way that's *visually* rough enough to
       look exactly like real detailed content by (1)'s own measure — see
       NOISE_CROSS_CHANNEL_CORR's docstring for the real incident this
       exists for — but isn't real optics: its R/G/B channels vary too
       independently of each other for any real image to produce. This
       also covers a *partial*-garbage frame (a torn/corrupted band of rows
       among otherwise-normal ones), which (1)'s whole-frame average alone
       would dilute away — a corrupted band is exactly as channel-
       decorrelated as full-frame garbage, it just doesn't cover every row.

    Both measured on a cheap 64x48 downsample so this stays negligible per
    frame."""
    try:
        small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
    except cv2.error:
        return False  # never let a sanity check itself be the thing that breaks a real frame
    std = float(small.std())
    if std < 1.0:
        return False  # a near-solid frame (lens cap, blackout) isn't noise — just flat
    row_h_roughness = np.abs(small[:, 1:] - small[:, :-1]).mean()
    if (row_h_roughness / std) > NOISE_ROUGHNESS_RATIO:
        return True
    return _cross_channel_corr(small) < NOISE_CROSS_CHANNEL_CORR


def _cross_channel_corr(small) -> float:
    """Mean pairwise correlation between the three color channels' own
    horizontal-diff signals — see NOISE_CROSS_CHANNEL_CORR's docstring.
    `small` is the same int16 64x48 downsample _looks_like_noise already
    computed; a channel with zero local variance (a pure grayscale/solid
    frame — including one already known flat by the std<1.0 check above,
    which never reaches here) has nothing to correlate against, so that
    pair counts as fully correlated (1.0) rather than undefined, matching
    "this isn't the kind of frame this check is meant to reject."""
    diffs = small[:, 1:, :] - small[:, :-1, :]
    pairs = ((2, 1), (1, 0), (2, 0))  # (R,G), (G,B), (R,B) — BGR channel order
    correlations = []
    for a, b in pairs:
        da, db = diffs[:, :, a].ravel(), diffs[:, :, b].ravel()
        if da.std() < 1e-6 or db.std() < 1e-6:
            correlations.append(1.0)
        else:
            correlations.append(float(np.corrcoef(da, db)[0, 1]))
    return sum(correlations) / len(correlations)


def _read_with_warmup(cap, attempts: int = OPEN_WARMUP_READS) -> bool:
    """True once `cap` produces one real, non-garbage frame, giving it a
    few quick retries first — a webcam's very first read() right after
    open()/set() commonly fails while it's still negotiating
    format/exposure, which would otherwise make an actually-working camera
    look broken."""
    for attempt in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size and not _looks_like_noise(frame):
            return True
        if attempt < attempts - 1:
            time.sleep(OPEN_WARMUP_SLEEP_SEC)
    return False


DISCOVERY_INDEX_SAFETY_MARGIN = 2   # always probe this many indices past `known`'s own reported
                                     # max, in case that DirectShow enumeration snapshot is
                                     # incomplete (see discover_cameras' docstring). Deliberately
                                     # small, not a large floor: the whole point of consulting
                                     # `known` at all is avoiding a heavy blind sweep's open/close
                                     # churn (itself a cause of USB camera "cooldown"), so this
                                     # hedges against a *plausible* small enumeration undercount
                                     # (one race, one still-initializing driver) without giving up
                                     # most of that benefit the way a large fixed floor would.


def discover_cameras(max_index: int = 16, max_found: int = 16, warmup_reads: int = 5,
                      skip: frozenset[int] = frozenset()) -> list[int]:
    """Probe camera indices 0..max_index and return the ones that actually
    deliver frames (works the same way on Linux/Windows/macOS, and covers a
    laptop's built-in webcam, which is almost always index 0). Defaults are
    generous rather than a hard cap — a booth may plug in more than a
    handful of USB webcams, and more can always be added by hand afterwards
    via the /settings page (or CameraGateway, for non-USB protocols), which
    have no limit of their own.

    A webcam that just opened frequently fails its very first read() or two
    while it finishes format/exposure negotiation — trying only once (the
    old behaviour) would misreport a real, connected camera as absent and
    it would never get a CameraWorker created for it at all, i.e. it would
    never show up no matter how long the booth kept running. So each
    opened-but-not-yet-answering index gets a few quick retries before
    being written off as not actually there.

    `skip` excludes indices already owned by a running CameraWorker (see
    web.booth_manager's hot-plug scan) — opening a second VideoCapture on an
    index a worker thread already holds open can disrupt that worker's own
    connection on some backends (DirectShow in particular), so those indices
    are never probed here at all rather than opened-then-released.
    """
    found = []
    # Ask Windows how many DirectShow video devices exist and probe only those indices. The blind
    # 0..15 sweep used to open + close ~a dozen non-existent devices on every start, each attempt a
    # ~1 s failed open; heavy open/close churn is itself what leaves a USB camera in its 15-30 s
    # "cooldown" (E7) and hurts the following real open. Enumeration failing/unavailable (non-Windows,
    # no pygrabber) keeps the old blind sweep.
    #
    # CONFIRMED ROOT CAUSE (MongDee missing-cameras investigation): this used to trust `known`'s max
    # index as an exact, complete count and clamp the probe range to it (`max(known) + 1`). But
    # list_directshow_devices()'s own docstring already discloses that pygrabber's COM enumeration
    # can fail/be unavailable outright (handled -- an empty `known` keeps the full blind sweep,
    # below) -- it does NOT guarantee a *non-empty* result is complete. A DirectShow enumeration
    # snapshot taken while a USB camera's driver is still finishing initialization (very plausible
    # at startup with several cameras, or right after a hot-plug event -- exactly when this
    # function is called) can legitimately report fewer devices than physically exist right now.
    # Before this fix, any index beyond that incomplete count was never even attempted -- not
    # retried, not logged as missing, simply never probed -- which matches "Windows sees cameras
    # A/B/C/D but MongDee only shows a subset" exactly: not a per-camera open failure (that would at
    # least log something), but indices this function silently never tried in the first place.
    # DISCOVERY_MIN_PROBE_INDEX keeps a "handful or more" floor regardless of how few devices
    # `known` currently reports, and DISCOVERY_INDEX_SAFETY_MARGIN probes a few indices past
    # DirectShow's own reported max, so a momentarily-incomplete enumeration no longer permanently
    # hides a real camera -- while still skipping the pointless dead-index probes this
    # optimization exists to avoid when `known` genuinely is complete.
    try:
        from core.camera_identity import list_directshow_devices

        known = list_directshow_devices()
    except Exception:
        known = {}
    if known:
        max_index = min(max_index, max(known) + 1 + DISCOVERY_INDEX_SAFETY_MARGIN)
    for i in range(max_index):
        if i in skip:
            continue
        cap = _open_capture(i, label=f"discover:{i}")
        try:
            if cap.isOpened() and _read_with_warmup(cap, attempts=warmup_reads):
                found.append(i)
        finally:
            # Async + bounded (see _release_capture_safely's docstring) -- this function is
            # called forever by BoothManager's Hot-Plug Scan, so a wedged synchronous release
            # here used to be able to permanently hang that thread and kill hot-plug recovery
            # for the rest of the process's life.
            _release_capture_safely(cap, f"discover-{i}")
        if len(found) >= max_found:
            break
    return found


class CameraWorker(threading.Thread):
    """Runs one camera's capture+detection loop on a background thread.

    Callers get results via three optional callback attributes instead of
    Qt signals, so this class has no GUI dependency:
      - on_frame(camera_id, frame_bgr)          — every frame, with boxes drawn
      - on_detections(camera_id, detections)    — only on frames inference ran
      - on_status(camera_id, status, message)   — only on status transitions
    Each callback fires synchronously from this thread — callers touching a
    GUI must marshal onto their own main thread themselves (see
    ui/qt_camera_bridge.py for the Qt adapter); the web server just reads
    shared state directly since it's already thread-based.
    """

    def __init__(self, camera_id: str, device, model, allowed_classes: list[str],
                 recognizer=None, conf_threshold: float = 0.45, device_target="cpu",
                 on_frame=None, on_detections=None, on_status=None,
                 on_person_tracks=None, gender_age_backend=None, performance_monitor=None,
                 ai_worker: "AIWorker | None" = None, global_id_resolver=None,
                 global_attribute_resolver=None, is_product_live=None, face_detector=None,
                 product_name_resolver=None, gender_resolver=None):
        super().__init__(daemon=True)
        # Optional product_key -> display name (BoothManager wires the catalog's `name`). Product boxes are
        # labelled with that name, never with the internal id ("product-4621553d").
        self.product_name_resolver = product_name_resolver
        # Optional full-frame face detector (core.face.YuNetFaceDetector). When set, every AI pass that
        # sees at least one person also detects faces, binds each to a person track, draws a FACE
        # box and keeps the best-quality shot per track (self._face_service.store). None = feature off.
        # Per-camera corrupt-frame gate (torn / truncated / frozen / garbage frames) on top of
        # _looks_like_noise; see core/frame_integrity.py. Counters feed get_integrity_stats().
        self._read_started: float | None = None     # monotonic start of the read() in progress, if any
        self.hang_recoveries = 0
        self.stale_recoveries = 0
        self.stuck_releases = 0
        # See HANG_ESCALATION_THRESHOLD's docstring. _consecutive_hang_recoveries resets to 0 on
        # every good frame (thread mode only -- see run()); once it reaches the threshold this
        # camera escalates permanently to process-isolated capture for the rest of this run.
        self._consecutive_hang_recoveries = 0
        self._escalate_to_process = False
        self._process_capture = None   # core.capture_process.CaptureProcess, only once escalated
        self._pump_thread: "threading.Thread | None" = None  # see _run_process_capture_pump
        self._fail_count = 0
        self._last_reason_code = "OK"  # see _classify_reason
        self._frame_integrity = FrameIntegrity(freeze_sec=FROZEN_STREAM_SEC, freeze_min_frames=10)
        self._last_good_frame_ts: float | None = None   # monotonic time of the last frame that passed every check
        self._good_streak = 0
        self._suspect_until = 0.0                        # monotonic: device list just changed, be strict
        self._release_done = threading.Event()
        self._release_done.set()
        self._fault_times: list[float] = []
        self._low_bandwidth_until = 0.0
        self._last_relocate_probe = 0.0
        # Optional device -> bool | None ("is this camera index in Windows' device list right now?", None =
        # unknown). BoothManager wires its DevicePresenceMonitor.is_present here. Left unset: old behaviour.
        self.device_present_fn = None
        # One-off embedding matches (background clutter) are dropped until seen again (see core/product_confirm.py).
        self._product_confirmer = ProductConfirmer(min_hits=PRODUCT_CONFIRM_HITS)
        self._last_person_scores: list[float] = []
        self._integrity_counts: dict[str, int] = {}
        self._face_service = FaceService(face_detector) if face_detector is not None else None
        self._latest_faces: list[dict] = []
        self._face_error_logged = False
        self.camera_id = camera_id
        # Optional key -> bool. `allowed_classes` is frozen when this worker is built, so a
        # COCO-named product (e.g. "bottle") deleted afterwards would otherwise keep being
        # detected AND drawn until the app restarts. When set, a detection whose product
        # is no longer live is dropped before it is drawn or reported.
        self.is_product_live = is_product_live
        self.device = device
        self.model = model
        self.recognizer = recognizer
        # Optional (camera_id, local_track_id) -> global_id | None lookup —
        # duck-typed like gender_age_backend, so this module stays
        # dependency-free of core.reid. When set (web/booth_manager.py wires
        # its GlobalIdentityRegistry.get_confirmed_global_id_for here), a
        # person box's on-screen label gains "| P000001" once Re-ID has
        # CONFIRMED that track as a real identity (spec: MongDee AI Vision
        # master prompt section 19 — the live box must show the Global
        # Person ID, not just the category) — a track Re-ID has only just
        # matched once (still provisional/UNKNOWN, not yet counted a
        # visitor) never shows an ID, so a fleeting detection can never look
        # like it "is" some confirmed person. Left unset, behaves exactly
        # as before (category + confidence only).
        self.global_id_resolver = global_id_resolver
        # Optional global_id -> core.attributes.AttributeResult | None lookup
        # (web/booth_manager.py wires its GlobalPersonAttributeSmoother.get
        # here) — duck-typed, same dependency-free pattern as
        # global_id_resolver above. When set, a Re-ID-matched person box's
        # on-screen category is overridden by the Global Person's
        # quality-gated, confidence-weighted fused gender/age (shared across
        # every camera in the booth) instead of this camera's own
        # independent per-frame guess, whenever that fused result is
        # confident (status == "ok"). This exists because the same physical
        # person, correctly merged to one global_id by Re-ID, could
        # otherwise show a different category on each camera at the same
        # moment — each camera's local classification never saw the other's
        # evidence. Left unset, behaves exactly as before.
        self.global_attribute_resolver = global_attribute_resolver
        # Optional (camera_id, track_id) -> ("male"|"female", confidence) | None: the ONE gender this person is shown
        # with, decided per Global Person (BoothManager wires it), so the same person never appears as two different
        # genders on two cameras. An unclassified person is still drawn (as "PERSON NN%", PERSON_UNKNOWN_COLOR)
        # regardless of whether this is set -- only the label/gender it shows once classified depends on it.
        self.gender_resolver = gender_resolver
        self.conf_threshold = conf_threshold
        self.device_target = device_target
        # Optional core.performance.PerformanceMonitor (duck-typed, not
        # imported, matching gender_age_backend's pattern below): when set,
        # every capture/drop/display/AI event is reported to it so a
        # dashboard/benchmark/AdaptiveController can see real per-camera
        # FPS, latency and drop rate. Left unset, behaves exactly as before.
        self._performance_monitor = performance_monitor
        # Not resolved/registered until run() actually starts (see run()) —
        # constructing a CameraWorker must stay cheap and thread-free (unit
        # tests rely on this), so the process-wide default AIWorker is only
        # ever lazily created the first time some camera actually starts.
        self._ai_worker_override = ai_worker
        self._ai_worker: "AIWorker | None" = None
        self._ai_next_due = 0.0
        self._ai_pass_count = 0
        self._latest_capture_frame = None
        self._latest_capture_seq = -1
        self._latest_capture_ts: float | None = None
        self._frame_seq = 0
        # Moves the drawn boxes with the picture between AI passes (core/box_motion.py). None = draw the last
        # AI boxes as they are (the old behaviour).
        self._box_follower = BoxFollower() if BOX_FOLLOW_ENABLED else None
        self._frame_slot_lock = threading.Lock()
        # AdaptiveController's two levers (core/performance.py) — plain
        # instance attributes rather than the old module constants, so each
        # camera can be throttled independently based on its own measured
        # load. GIL-atomic get/set, same pattern as set_enabled below.
        # CPU-only starts already throttled (see DETECT_EVERY_N_FRAMES_CPU) —
        # a CUDA GPU keeps the lighter-restriction defaults.
        is_cpu_target = str(device_target) == "cpu"
        self._detect_every_n_frames = DETECT_EVERY_N_FRAMES_CPU if is_cpu_target else DETECT_EVERY_N_FRAMES
        self._ai_imgsz = DEFAULT_AI_IMGSZ_CPU if is_cpu_target else DEFAULT_AI_IMGSZ
        # Optional vision.attributes.extractor.GenderAgeBackend (duck-typed,
        # not imported, so this module stays dependency-free): when set,
        # every detected person box is colored per BOX_COLORS by predicted
        # gender/age (see _run_yolo). Left unset, every person box is
        # PERSON_UNKNOWN_COLOR — never a generic red "Person" box either way.
        self.gender_age_backend = gender_age_backend
        self.on_frame = on_frame or (lambda *a: None)
        self.on_detections = on_detections or (lambda *a: None)
        self.on_status = on_status or (lambda *a: None)
        # Optional, defaults to a no-op: on_person_tracks(camera_id,
        # visible_tracks, evicted_tracks, frame_width, frame_height, frame).
        # frame_width/frame_height (added for the Virtual Tripwire — see
        # core.tripwire — to convert a track's pixel-space foot point into
        # the same 0..1 normalized space a tripwire's line is stored in) and
        # the trailing BGR frame itself (added for Multi-Camera Person Re-ID
        # — see core.reid — so a qualifying track's bbox can be cropped and
        # embedded) are both purely additive — on_detections's
        # signature/contents are unchanged (person boxes still never appear
        # in it), and any caller still using an older-shape callback keeps
        # working unmodified as long as it accepts *args (e.g. the desktop
        # app via ui/qt_camera_bridge.py's typed Qt Signal(str, list), or
        # this default no-op itself).
        self.on_person_tracks = on_person_tracks or (lambda *a: None)
        self._person_tracker = _make_person_tracker()

        # Catalog keys that double as a YOLO/COCO class name get the cheap,
        # exact detection path; everything else (custom-trained products) is
        # only findable via the embedding recognizer.
        catalog_keys = set(allowed_classes)
        coco_names = set(model.names.values())
        self._legacy_coco_classes = catalog_keys & coco_names
        detect_classes = self._legacy_coco_classes | {PERSON_CLASS_NAME}
        self._class_indices = [
            idx for idx, name in model.names.items() if name in detect_classes
        ]

        self._proposer = ForegroundProposer()
        self._person_segmenter = get_person_segmenter(device_target)
        self._stop_event = threading.Event()
        self._running = False
        self._enabled = True
        self._ever_connected = False  # distinguishes "connecting" (first-ever attempt) from "reconnecting"
        self._reopen_backoff_sec = REOPEN_BACKOFF_INITIAL_SEC  # grows on repeated failure, resets on success
        self._reopen_attempt = 0  # count of consecutive failed reopen attempts, reset on success
        self._last_reopen_attempt = 0.0
        self._cap = None
        self._cap_lock = threading.Lock()
        # Whether this worker's current/most-recent open is counted in the
        # shared _active_camera_count (see _low_bandwidth_open_preferred) --
        # tracked per-worker so _release_capture never double-decrements a
        # stream that was never counted (e.g. an _open() attempt that never
        # got past isOpened()==False).
        self._counted_active = False
        # Best-effort physical-camera identity (core.camera_identity), set
        # once this worker's `device` (a DirectShow index) is first
        # successfully opened -- None forever for any device this can't
        # apply to (a path/RTSP URL, non-Windows, or identity data simply
        # unavailable). Lets a later reconnect re-locate the same physical
        # camera at a new index instead of retrying a now-stale one forever
        # (see _open()) -- an ephemeral OpenCV index is never assumed to be
        # a stable physical identity (spec: MongDee physical-camera-identity
        # follow-up).
        self._physical_id: str | None = None
        self._last_status = None
        self._last_boxes: list[tuple[list[float], str, tuple[int, int, int]]] = []
        self._ai_success_streak = 0
        # Virtual Tripwire overlay (see _draw_tripwire, set_tripwire_overlay
        # below) — a plain dict or None, replaced wholesale (never mutated
        # in place) by whoever owns crossing detection (web/booth_manager.py),
        # same lock-free single-writer/single-reader pattern as
        # self._last_boxes: safe for the capture thread to read while the
        # AI-pass thread replaces it.
        self._tripwire_overlay: dict | None = None

    def _open(self) -> bool:
        # A reconnect (not the very first-ever open) with a known physical
        # identity: check whether this camera's physical device simply
        # moved to a different DirectShow index (Windows re-enumeration)
        # before retrying the old, possibly-now-stale one. A no-op (never
        # raises, never blocks the open below) wherever identity data isn't
        # available -- see core.camera_identity's own module docstring for
        # exactly which cases that covers. It can also refuse outright (see
        # its own docstring) when self.device is positively known to now be
        # a *different* physical camera than the one this worker tracks --
        # opening it would silently mislabel that other camera's feed under
        # this camera_id.
        if not self._release_done.wait(RELEASE_WAIT_BEFORE_REOPEN_SEC):
            # The previous handle is still being released (a wedged driver call). Opening the same device
            # now would only fail; try again on the next retry instead of stacking up more native calls.
            self.stuck_releases += 1
            logger.warning("[%s] previous capture handle still not released -- postponing reopen", self.camera_id)
            return False
        if self._ever_connected and self._physical_id is not None:
            if not self._relocate_device_if_moved():
                return False

        force_low = time.monotonic() < self._low_bandwidth_until
        if force_low:
            logger.info("[%s] repeated stream faults -- reopening with the low-bandwidth profiles", self.camera_id)
        cap = _open_capture(
            self.device, label=self.camera_id,
            low_bandwidth_only=force_low or (_low_bandwidth_open_preferred() and preferred_capture_profile() is None),
        )
        if not cap.isOpened():
            _release_capture_safely(cap, self.camera_id)
            return False

        # _open_capture already configured and validated this exact stream.
        # Reapplying properties here can restart capture and invalidate it.
        if self._stop_event.is_set():
            _release_capture_safely(cap, self.camera_id)
            return False
        with self._cap_lock:
            self._cap = cap
        self._mark_camera_active()
        self._proposer = ForegroundProposer()
        self._person_tracker = _make_person_tracker()
        self._frame_integrity.reset()     # a new stream: previous frame no longer applies
        self._last_good_frame_ts = None
        self._good_streak = 0
        self._last_boxes = []
        if self._box_follower is not None:
            self._box_follower.reset()
        self._ai_success_streak = 0
        if self._physical_id is None:
            self._resolve_physical_id()
        return True

    def _resolve_physical_id(self) -> None:
        """Best-effort: record this worker's current `device`'s physical
        identity right after a successful open, so a later reconnect can
        recognize the same physical camera even if its index changes.

        Fire-and-forget on its own daemon thread rather than run inline:
        the underlying Windows PnP lookup (core.camera_identity.
        get_physical_camera_identities -> list_pnp_camera_devices) shells
        out to PowerShell and has been measured taking several seconds on
        real hardware. _open() calls this synchronously right after a
        capture already opened and read a real frame successfully -- paying
        that multi-second cost inline would delay this camera's very first
        "online" status (and its first delivered frame) by that same
        amount, exactly the kind of capture-path blocking this module's own
        architecture (see module docstring: capture/display never waits on
        anything slow) exists to avoid. Nothing on the hot capture/display
        path reads self._physical_id -- only a *later* reconnect
        (_relocate_device_if_moved) does, and by then this has had seconds
        to finish."""
        threading.Thread(
            target=self._resolve_physical_id_blocking, daemon=True,
            name=f"mongdee-physid-{self.camera_id}",
        ).start()

    def _resolve_physical_id_blocking(self) -> None:
        try:
            from core.camera_identity import device_as_index

            index = device_as_index(self.device)
            if index is None:
                return
            identity = _cached_physical_camera_identities().get(index)
            if identity is not None:
                self._physical_id = identity.physical_id
        except Exception:
            pass  # identity is strictly best-effort -- never let this affect a real, working connection

    def _relocate_device_if_moved(self) -> bool:
        """If this worker's physical camera now enumerates at a different
        DirectShow index than `self.device` currently points at, update
        `self.device` to follow it (logged) before the caller's next
        _open_capture attempt.

        Returns False in exactly one case: `self.device`'s *current*
        occupant is positively known (from live identity data) to be a
        *different* physical camera than the one this worker tracks, and
        that tracked camera wasn't found anywhere else either -- i.e. this
        isn't "my camera moved indices" (handled above) but "someone else's
        camera is now sitting at my old index" (spec: MongDee camera-
        identity investigation -- two USB cameras of the identical model,
        as this project's own dev hardware has, share the same VID/PID and
        have no real serial, so their only distinguishing identity is USB
        port location; if the physical units are swapped between ports --
        or Windows otherwise reassigns an index to a different physical
        camera between one reconnect attempt and the next -- opening
        `self.device` as-is would silently mislabel that other physical
        camera's feed under *this* camera_id). The caller must treat a
        False return exactly like "device not present yet" and must not
        open `self.device` this round.

        Returns True in every other case -- including every case where
        identity data is unavailable/inconclusive (never blocks a real
        open on mere uncertainty, only on positive contradicting evidence)."""
        try:
            from core.camera_identity import device_as_index

            current_index = device_as_index(self.device)
            identities = _cached_physical_camera_identities()
            new_index = None
            for index, identity in identities.items():
                if identity.physical_id == self._physical_id:
                    new_index = index
                    break
            if new_index is not None and new_index != current_index:
                logger.info(
                    "[%s] Physical camera relocated: index %s -> %s (physical_id=%s)",
                    self.camera_id, current_index, new_index, self._physical_id,
                )
                self.device = new_index
                return True
            if new_index is None and current_index is not None:
                current_identity = identities.get(current_index)
                if current_identity is not None and current_identity.physical_id != self._physical_id:
                    logger.warning(
                        "[%s] index %s now identifies as a different physical camera "
                        "(%s) than the one this worker tracks (%s) -- refusing to open "
                        "it as %s instead of silently mislabeling that other camera's "
                        "feed; will retry once the tracked camera is found again",
                        self.camera_id, current_index, current_identity.physical_id,
                        self._physical_id, self.camera_id,
                    )
                    return False
            return True
        except Exception:
            return True  # identity is strictly best-effort -- never block a real open on uncertainty

    def _mark_camera_active(self) -> None:
        """Increments the shared _active_camera_count exactly once per
        successful open (see _low_bandwidth_open_preferred) -- idempotent
        so an already-counted worker calling this again (there is no such
        call site today, but this keeps the invariant self-enforcing) never
        drifts the shared count upward."""
        if self._counted_active:
            return
        global _active_camera_count
        with _active_camera_count_lock:
            _active_camera_count += 1
        self._counted_active = True

    def _mark_camera_inactive(self) -> None:
        """Decrements the shared _active_camera_count exactly once per
        worker that was actually counted -- called from _release_capture so
        every path that stops this worker's stream (disabled, dropped,
        stopped) releases its share of the shared budget, never only a
        subset of them."""
        if not self._counted_active:
            return
        global _active_camera_count
        with _active_camera_count_lock:
            _active_camera_count = max(0, _active_camera_count - 1)
        self._counted_active = False

    def _release_capture(self):
        self._mark_camera_inactive()
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is None:
            return
        # See _release_capture_safely's docstring: cap.release() on a wedged DirectShow graph can
        # block forever, so this runs on a helper thread instead of the capture loop (or, worse,
        # the shared watchdog thread); a reopen waits for self._release_done up to
        # RELEASE_WAIT_BEFORE_REOPEN_SEC (see _open) because the same device cannot be opened twice.
        self._release_done = _release_capture_safely(cap, self.camera_id)

    def _process_capture_spec(self) -> tuple[str, dict]:
        """(factory, kwargs) for this camera's escalated CaptureProcess — a separate method
        purely so tests can substitute core.capture_sim's fault-injection factories for the real
        negotiated-capture one below, the same way core.capture_process's own test suite does,
        without needing real camera hardware to prove the escalation wiring itself is correct
        end-to-end (see tests/core/test_camera_process_escalation.py)."""
        from core.capture_process import CROSS_PROCESS_OPEN_LOCK
        return ("core.vision:make_negotiated_capture_for_process",
                {"device": self.device, "label": self.camera_id, "cross_process_lock": CROSS_PROCESS_OPEN_LOCK})

    def _ensure_process_capture(self) -> None:
        """Idempotent: starts this camera's CaptureProcess once, on first entry after escalation
        (see HANG_ESCALATION_THRESHOLD). Every later call while already escalated is a no-op --
        CaptureProcess's own watchdog thread handles that child's whole lifecycle from here on
        (stale/dead detection, kill, bounded-backoff respawn), same as core.vision.CameraWorker
        does for a thread-mode camera, just one process further out."""
        if self._process_capture is not None:
            return
        from core.capture_process import CaptureProcess
        factory, kwargs = self._process_capture_spec()
        self._process_capture = CaptureProcess(factory, kwargs, name=str(self.camera_id))
        self._process_capture.start()

    def get_capture_mode(self) -> str:
        """'process' once this camera has escalated (see HANG_ESCALATION_THRESHOLD), else
        'thread' -- exposed to BoothManager for the per-camera diagnostics panel (spec section 30:
        callers should not have to reach into private attributes to tell the two apart)."""
        return "process" if self._escalate_to_process else "thread"

    def get_worker_pid(self) -> int | None:
        """OS pid of this camera's own capture process once escalated, else None (thread-mode
        capture runs inside this process, under this worker's thread id, not a pid of its own)."""
        return self._process_capture.pid if self._process_capture is not None else None

    def get_hang_escalations(self) -> int:
        return 1 if self._escalate_to_process else 0

    def _note_fault(self) -> None:
        """Record a stream fault that forced a release. Two within FAULT_WINDOW_SEC -> reopen with the
        smaller, lower-bandwidth profiles for LOW_BANDWIDTH_HOLD_SEC (fast motion + a marginal USB link
        is what tears frames; a lighter stream survives it)."""
        now = time.monotonic()
        self._fault_times = [t for t in self._fault_times if now - t <= FAULT_WINDOW_SEC] + [now]
        if len(self._fault_times) >= FAULTS_BEFORE_LOW_BANDWIDTH:
            self._low_bandwidth_until = now + LOW_BANDWIDTH_HOLD_SEC

    # Spec-required machine-filterable failure classification (see module docstring for the full
    # set this project's own root-cause reports motivated). Free-text Thai messages stay the
    # human-facing UI copy; this is the companion code BoothManager exposes via get_state()/
    # get_camera_snapshot() for anything that needs to group/alert/dashboard on failure kind
    # instead of parsing localized strings. Best-effort: derived from the same message text and
    # instance state this worker already produces, not a second parallel source of truth.
    def _classify_reason(self, status: str, message: str) -> str:
        if status in ("online", "connecting", "degraded"):
            return "OK"
        if status == "disabled":
            return "DISABLED"
        present = self.device_present_fn(self.device) if self.device_present_fn else None
        if present is False:
            return "PHYSICAL_DISCONNECT"
        if "อ่านภาพค้าง" in message:
            return "BLOCKED_READ"
        if "หยุดหรือเสียหาย" in message:
            return "FROZEN_FRAME" if self._frame_integrity.frozen_for(time.time()) > 0 else "CORRUPT_FRAME"
        if "เปิดกล้อง" in message and "ไม่สำเร็จ" in message:
            return "OPEN_FAILED"
        if "ไม่ได้ต่อเนื่อง" in message:
            recent_corrupt = sum(v for k, v in self._integrity_counts.items() if k != "frozen")
            return "CORRUPT_FRAME" if recent_corrupt > 0 else "READ_FAILED"
        if "AI" in message:
            return "BACKEND_FAILURE"
        return "UNKNOWN"

    def get_last_reason_code(self) -> str:
        return self._last_reason_code

    def _emit_status(self, status: str, message: str = ""):
        self._last_reason_code = self._classify_reason(status, message)
        if status != self._last_status:
            self._last_status = status
            self.on_status(self.camera_id, status, message)

    def _run_yolo(self, frame, draw_boxes):
        """Returns (person_boxes, person_categories, person_confidences,
        legacy_product_detections, claimed_boxes). person_categories and
        person_confidences are parallel to person_boxes — one
        'female'/'male'/'unknown' guess and its classifier
        confidence per box (see _classify_person), the same guess drawn as
        that box's on-screen label/color and fed to self._person_tracker so
        a track's dashboard category never disagrees with what was on
        screen.

        `draw_boxes` accumulates (bbox, label, color) for this AI pass —
        appended here rather than mutated straight into self._last_boxes,
        because this method now runs on AIWorker's shared thread while the
        camera's own capture thread may concurrently be reading
        self._last_boxes to draw the live frame (see run_ai_pass, which
        replaces self._last_boxes with this list in one atomic assignment
        only once the whole pass is done)."""
        person_boxes = []
        person_categories = []
        person_confidences = []
        legacy_detections = []
        claimed_boxes = []
        if not self._class_indices:
            return person_boxes, person_categories, person_confidences, legacy_detections, claimed_boxes
        self._last_person_scores = []
        person_in_classes = any(self.model.names.get(i) == PERSON_CLASS_NAME for i in self._class_indices)
        detect_conf = min(self.conf_threshold, PERSON_LOW_CONF) if person_in_classes else self.conf_threshold
        with _inference_lock:
            results = self.model.predict(
                source=frame,
                imgsz=self._ai_imgsz,
                conf=detect_conf,
                classes=self._class_indices,
                device=self.device_target,
                verbose=False,
            )
        if not results or results[0].boxes is None:
            return person_boxes, person_categories, person_confidences, legacy_detections, claimed_boxes
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            conf = float(box.conf.item())
            bbox = [float(x) for x in box.xyxy[0].tolist()]
            if (class_name != PERSON_CLASS_NAME and self.is_product_live is not None
                    and not self.is_product_live(class_name)):
                continue  # deleted from the catalog after this worker was built
            if class_name != PERSON_CLASS_NAME:
                # A small, low-detail box far from the camera must clear a HIGHER bar than a
                # close, sharp one before it is trusted as a specific product -- otherwise a
                # distant, ambiguous detection is reported with exactly the same confidence as an
                # unambiguous close-up one (see core.product_confirm's module docstring, which
                # mirrors core.attributes.face_min_confidence's identical reasoning for faces).
                box_px = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
                if conf < product_min_confidence(box_px, self.conf_threshold):
                    continue  # the lowered detector threshold is for people only
            claimed_boxes.append(bbox)
            if class_name == PERSON_CLASS_NAME:
                person_boxes.append(bbox)
                self._last_person_scores.append(conf)
                if conf < self.conf_threshold:
                    # Low-score person: may only continue an existing track (see PERSON_LOW_CONF), so do
                    # not spend a classifier pass on it; the track's own votes already carry its category.
                    category, category_conf = "unknown", 0.0
                else:
                    category, category_conf = self._classify_person(frame, bbox)
                person_categories.append(category)
                person_confidences.append(category_conf)
                label, color = self._category_label_and_color(category, conf)
                draw_boxes.append((bbox, label, color))
            else:
                draw_boxes.append((bbox, f"{self._product_display_name(class_name, class_name.upper())} {conf:.0%}",
                                   PRODUCT_COLOR))
                legacy_detections.append({"class_name": class_name, "conf": conf, "bbox": bbox})
        return person_boxes, person_categories, person_confidences, legacy_detections, claimed_boxes

    PRODUCT_LABEL_MAX_CHARS = 28

    def _product_display_name(self, product_key: str, legacy: str) -> str:
        """The name shown on a product's box: the catalog name when a resolver is wired, else `legacy`
        (the old id-derived text, kept so callers without a catalog behave as before). When a catalog IS
        wired but has no name for the key, a generic word - never an id."""
        resolver = self.product_name_resolver
        if resolver is None:
            return legacy
        try:
            name = resolver(product_key)
        except Exception:
            name = None
        name = str(name).strip() if name else ""
        if not name:
            return "PRODUCT"
        limit = self.PRODUCT_LABEL_MAX_CHARS
        return name if len(name) <= limit else name[: limit - 1] + "…"

    def _classify_person(self, frame, bbox) -> tuple[str, float]:
        """Returns ('female'/'male'/'unknown', confidence) for one
        detected person box. Gender is only ever guessed when a real
        gender_age_backend is configured (see __init__) — with none set,
        this is always ('unknown', 0.0). A prediction below its confidence
        floor is treated as inconclusive rather than forced into a
        category, since a wrong category is worse than none here.

        The returned confidence is what PersonTracker.update() weights that
        frame's vote by (see core/tracker.py's person_confidences param) —
        so a low-confidence-but-still-above-floor guess influences a
        track's reported category less than a high-confidence one, instead
        of both counting as a flat +1. Age is not analysed at all."""
        if self.gender_age_backend is None:
            return "unknown", 0.0
        crop = crop_box(frame, bbox)
        if not crop.size:
            return "unknown", 0.0
        try:
            detail_fn = getattr(self.gender_age_backend, "predict_gender_detail", None)
            if callable(detail_fn):
                # The backend also reports how big the face was: a far-away face must be more confident to
                # count and is worth less (core.attributes.face_temperature) - the same policy the per-person
                # smoother applies, so this per-frame guess never out-votes it.
                detail = detail_fn(crop)
                if detail is None:
                    return "unknown", 0.0
                gender, gender_conf, face_px = detail["gender"], detail["confidence"], detail.get("face_px")
                if gender_conf < face_min_confidence(face_px):
                    return "unknown", 0.0
                gender_conf = attenuate_confidence(gender_conf, face_px)
            else:
                gender, gender_conf = self.gender_age_backend.predict_gender(crop)
        except Exception:
            gender, gender_conf = "unknown", 0.0
        if gender_conf >= MIN_GENDER_CONFIDENCE and gender in ("female", "male"):
            return gender, gender_conf
        return "unknown", 0.0

    @staticmethod
    def _category_label_and_color(category: str, conf: float) -> tuple[str, tuple[int, int, int]]:
        # cv2's Hershey font can't render Thai glyphs (see _ascii_label
        # above), so the on-frame label stays ASCII even though the UI
        # legend shows it in Thai.
        if category == "female":
            return f"FEMALE {conf:.0%}", PERSON_FEMALE_COLOR
        if category == "male":
            return f"MALE {conf:.0%}", PERSON_MALE_COLOR
        return f"UNKNOWN {conf:.0%}", PERSON_UNKNOWN_COLOR

    def _person_label_and_color(self, frame, bbox, conf) -> tuple[str, tuple[int, int, int]]:
        """Label + box color for one detected person — see _classify_person
        for how the category itself is decided. Kept as its own method
        (rather than inlined into _run_yolo) because it's directly
        unit-tested (tests/core/test_vision.py) without needing a full
        camera/YOLO round trip."""
        category, _classify_conf = self._classify_person(frame, bbox)
        return self._category_label_and_color(category, conf)

    def _run_custom_recognition(self, frame, claimed_boxes, draw_boxes):
        """Foreground blobs not already claimed by YOLO get identified against
        the trained embedding gallery (arbitrary, non-COCO products).
        `draw_boxes`: see _run_yolo.

        Skipped entirely (no proposer/embedding work at all) when there is
        no trained product to match against yet — the catalog starts empty
        (see GUIDE.md), so for a fresh install this is otherwise pure
        wasted CPU on every single AI pass for no possible result.

        Pipeline per proposal, in order: current-frame foreground proposal
        -> current-frame human-region segmentation/exclusion -> embedding
        match against the registered gallery -> temporal confirmation. A
        candidate is never matched against the gallery using pixels this
        pass's own current frame shows as human (hand/arm/sleeve) — that
        exclusion runs before recognition, not after, so a bare hand can
        never itself be identified as a product no matter what it happens
        to score.

        A candidate that does not clear identify()'s bar is dropped
        outright -- never drawn as "UNKNOWN". A rendered box promises the
        viewer "this is a real, specific thing"; an unresolved candidate is
        not one, so the correct UI state for it is no box at all, not a
        generic placeholder box (hard rule: STRICT UNKNOWN DETECTION)."""
        detections = []
        if self.recognizer is None or not self.recognizer.has_any_gallery():
            self._product_confirmer.reset()
            return detections
        now = time.time()
        self._product_confirmer.forget_stale(now)
        confirmed_this_pass: set[str] = set()
        proposals = self._proposer.propose(frame, exclude_boxes=claimed_boxes, max_regions=3)
        human_mask = self._person_segmenter.person_mask(frame) if proposals else None
        for bbox in proposals:
            refined = exclude_human_region(bbox, human_mask)
            if refined is None:
                continue  # mostly hand/arm in the current frame -- never a product candidate
            crop = crop_box(frame, refined)
            if crop.shape[0] < MIN_CROP_SIDE_PX or crop.shape[1] < MIN_CROP_SIDE_PX:
                continue
            box_px = min(refined[2] - refined[0], refined[3] - refined[1])
            # See core.product_confirm's module docstring: a small/far candidate must score higher
            # than recognizer.identify()'s own default floor before it is even considered a match
            # at all. product_min_confidence returns CUSTOM_RECOGNITION_BASE_FLOOR unchanged for a
            # full-trust-size box, so identify() only gets an explicit (stricter) floor when this
            # candidate is actually small -- never changes behaviour for the common case.
            required_floor = product_min_confidence(box_px, CUSTOM_RECOGNITION_BASE_FLOOR)
            identify_kwargs = {"floor": required_floor} if required_floor > CUSTOM_RECOGNITION_BASE_FLOOR else {}
            try:
                product_key, score = self.recognizer.identify(crop, **identify_kwargs)
            except Exception as exc:
                raise RuntimeError(f"custom recognizer failed: {exc}") from exc
            if product_key:
                required_hits = product_confirm_hits_for(box_px, PRODUCT_CONFIRM_HITS)
                if not self._product_confirmer.confirm(product_key, refined, now, min_hits=required_hits,
                                                        score=score):
                    continue  # first sighting: could be background clutter - wait for a second pass
                confirmed_this_pass.add(product_key)
                label = f"{self._product_display_name(product_key, _ascii_label(product_key))} {score:.0%}"
                draw_boxes.append((refined, label, PRODUCT_COLOR))
                detections.append({"class_name": product_key, "conf": score, "bbox": refined})
            # else: no registered product matched this candidate confidently enough --
            # dropped silently, never rendered (see docstring above).
        # A product confirmed on a recent pass but not re-matched on THIS one (a momentary
        # embedding-score dip from motion blur, a hand briefly crossing it, a lighting flicker)
        # keeps its last box (and last real score, not a placeholder) for a short grace window
        # instead of blinking off and back on -- see
        # core.product_confirm.ProductConfirmer.coasting's own docstring for why this mirrors
        # core.tracker.PersonTracker.coasting_tracks rather than being a new idea.
        for product_key, bbox, score in self._product_confirmer.coasting(now, exclude=confirmed_this_pass):
            label = f"{self._product_display_name(product_key, _ascii_label(product_key))} {score:.0%}"
            draw_boxes.append((list(bbox), label, PRODUCT_COLOR))
            detections.append({"class_name": product_key, "conf": score, "bbox": list(bbox)})
        return detections

    def _should_run_custom_recognition(self) -> bool:
        """Custom recognition (foreground proposal + embedding lookup) costs
        real CPU on top of YOLO, so it runs at a fraction of the AI pass
        rate instead of every pass."""
        return self._ai_pass_count % CUSTOM_RECOGNITION_EVERY_N_AI_PASSES == 0

    def _report_connected(self) -> None:
        self._ever_connected = True
        if self._performance_monitor is not None:
            self._performance_monitor.set_resolution(self.camera_id, self.get_resolution())
            self._performance_monitor.set_requested_fps(self.camera_id, 30.0)  # matches the FPS requested in _open()

    def _attempt_reopen(self) -> bool:
        """One gated reopen try. Does nothing and returns False if the
        per-camera exponential backoff (REOPEN_BACKOFF_*) hasn't elapsed yet.
        On success, resets the backoff and attempt counter so a camera that
        recovers reconnects promptly next time; on failure, grows the
        backoff (capped at REOPEN_BACKOFF_MAX_SEC) so a camera stuck on a
        hardware-level failure (see REOPEN_BACKOFF_INITIAL_SEC's docstring)
        is retried less and less often instead of in a tight loop forever.
        """
        now = time.monotonic()
        present = self._device_present()
        if present is False and self._ever_connected:
            # Windows no longer lists a camera at this index: it is unplugged. Opening it can only fail (and
            # each failed open holds the shared DirectShow lock, starving the other cameras), so do nothing
            # until the device monitor sees it come back - that calls request_immediate_retry(). The one
            # exception: a camera that merely moved to another index (identity known) is followed there.
            if not self._follow_relocation_while_absent(now):
                return False
            present = self._device_present()
        if now - self._last_reopen_attempt < self._reopen_backoff_sec:
            return False
        self._last_reopen_attempt = now
        self._reopen_attempt += 1
        self._emit_status(
            "reconnecting" if self._ever_connected else "connecting",
            "กำลังเชื่อมต่อกล้องใหม่..." if self._ever_connected else "กำลังเชื่อมต่อกล้อง...",
        )
        if self._open():
            was_reconnect = self._reopen_attempt > 1
            self._reopen_attempt = 0
            self._reopen_backoff_sec = REOPEN_BACKOFF_INITIAL_SEC
            self._report_connected()
            self._emit_status("online", "เชื่อมต่อกล้องสำเร็จ")
            if was_reconnect:
                logger.info(
                    "[%s] Stream recovered, backoff reset (device=%s)", self.camera_id, self.device,
                )
            return True
        logger.warning(
            "[%s] Camera stream failed (device=%s, attempt=%d, next_retry=%.0fs, "
            "reason=open_or_read_failed)",
            self.camera_id, self.device, self._reopen_attempt, self._reopen_backoff_sec,
        )
        self._emit_status("offline", self._offline_message())
        # A camera Windows still lists but that will not open is worth retrying soon (the driver may just
        # need a moment); one whose presence is unknown keeps the long back-off of old.
        ceiling = PRESENT_REOPEN_BACKOFF_MAX_SEC if present else REOPEN_BACKOFF_MAX_SEC
        self._reopen_backoff_sec = min(self._reopen_backoff_sec * REOPEN_BACKOFF_MULTIPLIER, ceiling)
        return False

    def _device_present(self) -> "bool | None":
        fn = self.device_present_fn
        if fn is None:
            return None
        try:
            return fn(self.device)
        except Exception:
            return None

    def _follow_relocation_while_absent(self, now: float) -> bool:
        """The index this worker used is gone from the device list. If the same physical camera now sits at
        another index, switch to it and say so (True); otherwise False. Throttled: identity lookups are slow."""
        if self._physical_id is None or now - self._last_relocate_probe < RELOCATE_PROBE_INTERVAL_SEC:
            return False
        self._last_relocate_probe = now
        old = self.device
        self._relocate_device_if_moved()
        return self.device != old and self._device_present() is not False

    def _offline_message(self) -> str:
        """"ไม่พบกล้อง" (camera not found) is actively misleading for a real
        failure mode verified on this project's own dev hardware: two
        physically-different USB cameras sharing one USB hub, where
        VideoCapture.open() fails outright for whichever one opens *second*
        -- every backend/format combination -- while the first stays
        perfectly healthy. The camera is very much present (Windows/
        DirectShow both still enumerate it); it just couldn't be *admitted*
        a second concurrent stream, a hub/driver-level resource limit no
        retry or format change works around (spec: MongDee multi-USB-camera
        capture-lifecycle investigation, Phase 5's "never hide the real
        reason" + Phase 14's "classify this separately from identity/stream/
        UI bugs"). Distinguishing this from a genuinely unplugged camera
        only costs a cheap DirectShow name lookup -- never a guess."""
        try:
            from core.camera_identity import device_as_index, list_directshow_devices

            index = device_as_index(self.device)
            if index is not None and index in list_directshow_devices():
                return (
                    f"กล้อง {self.device} ยังเชื่อมต่ออยู่แต่เปิดสตรีมไม่ได้ — "
                    f"อาจมีกล้อง USB ตัวอื่นที่ใช้ USB hub/พอร์ตเดียวกันเปิดอยู่ก่อนแล้ว "
                    f"(ทรัพยากรของ hub ไม่พอสำหรับหลายกล้องพร้อมกัน)"
                )
        except Exception:
            pass  # best-effort diagnostic only -- never let this block the offline status itself
        return f"ไม่พบกล้อง {self.device}"

    def run(self):
        self._running = True
        self._ai_worker = self._ai_worker_override or _get_default_ai_worker()
        self._ai_worker.register(self)
        self._emit_status("connecting", "กำลังเชื่อมต่อกล้อง...")
        if self._open():
            self._report_connected()
        else:
            self._emit_status("offline", f"เปิดกล้อง {self.device} ไม่สำเร็จ")

        self._fail_count = 0

        while self._running:
            if self._escalate_to_process:
                # Handed off to _run_process_capture_pump, started directly by
                # recover_from_hung_read the moment escalation was decided (see that method and
                # HANG_ESCALATION_THRESHOLD) -- this thread's job is done. Note this is *not*
                # self._running = False: the camera itself stays up, just served by a different
                # thread from here on. If this thread is the one that's genuinely, permanently
                # wedged inside cap.read() (the whole reason escalation exists), it will never
                # reach this check at all -- the pump thread already took over regardless, and
                # this one is simply abandoned (a leaked daemon thread, reclaimed at process
                # exit; see stop()'s own docstring for why that's an acceptable trade).
                break
            if not self._enabled:
                # Camera toggled off from the UI: release the device (so an
                # unused camera doesn't keep holding the OS handle/USB
                # bandwidth) and idle instead of trying to reopen it, until
                # set_enabled(True) is called again. _release_capture() (not
                # a bare self._cap.release()) so a wedged release can't hang
                # this camera's own capture/display loop forever either.
                if self._cap is not None:
                    self._release_capture()
                self._emit_status("disabled", "ปิดใช้งานกล้องนี้")
                time.sleep(0.2)
                continue

            if self._cap is None or not self._cap.isOpened():
                try:
                    reopened = self._attempt_reopen()
                except Exception:
                    # Same backstop as cap.read()/_process_captured_frame above: _attempt_reopen()
                    # calls into _open()/_open_capture()/device-identity lookups, none of which are
                    # guaranteed exception-free, and an uncaught raise here used to kill this
                    # worker's thread permanently -- silently, since a dead thread never re-enters
                    # this loop to try again. Treat a failed reopen attempt as just that (False),
                    # so it falls through to the same backoff wait below instead of ending the loop.
                    logger.exception(
                        "[%s] _attempt_reopen raised -- treating as a failed reopen instead of "
                        "letting it kill this camera's worker thread", self.camera_id,
                    )
                    reopened = False
                if reopened:
                    self._fail_count = 0
                self._stop_event.wait(0.2)
                continue

            cap = self._cap
            if cap is None:
                self._stop_event.wait(0.03)
                continue
            try:
                self._read_started = time.monotonic()
                ok, frame = cap.read()
            except Exception:
                # A native OpenCV/DirectShow exception here (observed live:
                # "cv2.error: Unknown C++ exception from OpenCV code" during
                # concurrent multi-camera capture) previously propagated
                # straight out of this thread's run() -- Python's threading
                # module logs "Exception in thread ..." and lets the thread
                # die, permanently, since nothing ever recreates it. That
                # camera would then stay offline until the whole process was
                # restarted, even though the exact same failure via a
                # returned ok=False (the far more common case, e.g. an
                # unplugged camera) already recovers fine through the
                # fail_count/_attempt_reopen backoff below. Converting the
                # exception into the same ok=False outcome routes it through
                # that already-tested recovery path instead of killing the
                # worker thread -- no change to backoff timing, the
                # DIRECTSHOW_LOCK, or any physical-identity/reconnect logic.
                # Broadened from `except cv2.error` to `except Exception`: a
                # bare cv2.error catch left every OTHER exception type (seen
                # live: a stuck camera whose read() eventually raises
                # something backend-specific, not always cv2.error) free to
                # kill this thread exactly the same way -- and once dead, no
                # watchdog can tell, because recover_from_hung_read() only
                # detects a read that is still IN PROGRESS (_read_started
                # set), never a thread that has already exited, and
                # check_stream_health() only acts on the online->offline
                # transition, never again once status is already "offline".
                # The camera then shows the "หยุดหรือเสียหาย" reconnecting
                # card forever, with frame_sequence/captured_frames frozen
                # and reconnect_count never advancing -- exactly the
                # permanently-stuck symptom this broadening closes.
                logger.warning(
                    "[%s] cap.read() raised %s (device=%s) -- treating as a "
                    "failed read instead of letting it kill this camera's worker thread",
                    self.camera_id, type(sys.exc_info()[1]).__name__ if sys.exc_info()[1] else "an exception",
                    self.device,
                )
                ok, frame = False, None
            finally:
                self._read_started = None
            if self._stop_event.is_set():
                break
            try:
                frame_accepted = self._process_captured_frame(ok, frame)
            except Exception:
                # _process_captured_frame's own docstring promises it "never raises" -- this is the
                # backstop for that promise, not a reason to rely on it: on_frame() (the
                # BoothManager/Qt-bridge callback), the box follower, and the frame-slot publish are
                # all called from here without their own try/except, and any one of them raising
                # would otherwise kill this thread exactly like the bare cap.read() exception above
                # used to (see that comment). Same treatment: log it, count it as a failed frame, and
                # let the existing fail_count/_attempt_reopen recovery path handle it instead of the
                # worker silently vanishing.
                logger.exception(
                    "[%s] _process_captured_frame raised -- treating as a failed frame instead of "
                    "letting it kill this camera's worker thread", self.camera_id,
                )
                frame_accepted = False
            if not frame_accepted:
                self._stop_event.wait(0.03)
                continue

        if self._escalate_to_process:
            # See the top-of-loop comment above: _run_process_capture_pump (already running, kept
            # going the whole time this thread was stuck) owns AI registration, self._running, and
            # self._process_capture from here on -- this thread has nothing left to clean up.
            return
        self._ai_worker.unregister(self.camera_id)
        self._release_capture()
        self._running = False

    def _process_captured_frame(self, ok: bool, frame) -> bool:
        """Everything after a frame is obtained that is identical regardless of whether it came
        from a thread-owned cv2.VideoCapture (run(), above) or an escalated camera's
        CaptureProcess (_run_process_capture_pump, below): noise/corruption/freeze checks, the
        online/offline state machine, the foreground-proposer update, box drawing/delivery, and
        publishing the latest-frame slot the AI pipeline and the stream endpoint both read from.
        Returns True for a frame that was accepted (state advanced to good), False for one that
        was rejected or missing (state advanced to failed) -- callers use this to decide how long
        to wait before trying again; it never raises and never blocks."""
        # A camera that's silently disconnected mid-stream (unplugged,
        # or a driver/USB glitch) can keep returning ok=True with
        # whatever garbage is left in a stale buffer instead of
        # cleanly failing — this is exactly as broken as a failed
        # read() and must not reach the UI or AI detection as if it
        # were a real frame (see _looks_like_noise's docstring).
        is_noise = ok and frame is not None and frame.size and _looks_like_noise(frame)
        integrity_reasons: list[str] = []
        if ok and frame is not None and frame.size and not is_noise:
            try:
                integrity_reasons = self._frame_integrity.check(frame, time.time()).reasons
            except Exception:
                integrity_reasons = []   # a bug in the checker must never take a camera down
            for reason in integrity_reasons:
                self._integrity_counts[reason] = self._integrity_counts.get(reason, 0) + 1
        if not ok or frame is None or frame.size == 0 or is_noise or integrity_reasons:
            self._fail_count += 1
            if self._performance_monitor is not None:
                self._performance_monitor.record_dropped(self.camera_id)
            if self._fail_count >= FAIL_THRESHOLD:
                logger.warning(
                    "[%s] Camera stream failed (device=%s, reason=%s, "
                    "consecutive_failures=%d)",
                    self.camera_id, self.device,
                    ("frame_looked_like_noise" if is_noise else
                     "frame_corrupt:" + ",".join(integrity_reasons) if integrity_reasons else "frame_read_failed"),
                    self._fail_count,
                )
                self._emit_status("offline", "อ่านภาพจากกล้องไม่ได้ต่อเนื่อง")
                self._note_fault()
                self._release_capture()   # a no-op once escalated -- self._cap is already None
            self._good_streak = 0
            return False

        self._fail_count = 0
        self._last_good_frame_ts = time.monotonic()
        self._good_streak += 1
        self._consecutive_hang_recoveries = 0
        # "connecting" is included here because the very first open at
        # the top of run() never itself emits "online" (only the
        # reconnect branch above does, on a *subsequent* open after a
        # drop) — without it, a camera that connects successfully on
        # its first try stays reported as "connecting" forever even
        # though frames are flowing, which get_latest_jpeg() (see
        # web/booth_manager.py) takes as "not really online yet" and
        # keeps serving the reconnecting placeholder instead of the
        # real video.
        if (self._last_status in (None, "offline", "connecting") and self._good_streak >= ONLINE_GOOD_STREAK
                and self._frame_integrity.frozen_for(time.time()) < FROZEN_STREAM_SUSPECT_SEC):
            self._emit_status("online", "ปกติ")
        capture_ts = time.time()
        if self._performance_monitor is not None:
            self._performance_monitor.record_capture(self.camera_id, capture_ts=capture_ts)

        # Keep the foreground-proposer's background model warm on every
        # captured frame (full capture fps), not just the sparse subset
        # that reaches a custom-recognition AI pass. A model fed only at
        # the throttled AI-pass rate lags real scene changes by seconds,
        # so ordinary motion (a person walking through, a lighting
        # shift, camera shake) reads as a large, confident, spurious
        # "foreground" blob against a stale background photo -- the
        # leading suspected cause of product detections with no real
        # product in frame. Gated on having a trained gallery at all
        # (see _run_custom_recognition's own gate) so this costs nothing
        # on a fresh install with no products trained yet.
        if self.recognizer is not None and self.recognizer.has_any_gallery():
            try:
                self._proposer.update(frame)
            except Exception:
                logger.debug("[%s] proposer.update failed", self.camera_id, exc_info=True)

        # Draw + deliver this frame, then hand the raw frame off for
        # AIWorker to pick up whenever this camera is next due (see
        # run_ai_pass) — capture/display never waits on AI in any way
        # any more, regardless of how slow inference is or how many
        # other cameras are also waiting their turn (see module
        # docstring). Boxes drawn here are whatever the last completed
        # AI pass produced; self._last_boxes is only ever replaced
        # wholesale by run_ai_pass (never appended to piecemeal here),
        # so reading it concurrently with that replace is safe without
        # a lock.
        overlay = self._tripwire_overlay  # local snapshot — see module-level lock-free comment
        self._frame_seq += 1
        frame_seq = self._frame_seq
        follower = self._box_follower
        if follower is not None:
            try:
                follower.push_frame(frame_seq, frame)
                shown_boxes = follower.boxes()
            except Exception:
                logger.debug("[%s] box follower failed", self.camera_id, exc_info=True)
                shown_boxes = self._last_boxes
        else:
            shown_boxes = self._last_boxes
        display_frame = frame.copy() if (shown_boxes or overlay) else frame
        for bbox, label, color in shown_boxes:
            _draw_box(display_frame, bbox, label, color)
        if overlay is not None:
            _draw_tripwire(display_frame, overlay)
        self.on_frame(self.camera_id, display_frame)
        if self._performance_monitor is not None:
            self._performance_monitor.record_display(self.camera_id, capture_ts=capture_ts)

        with self._frame_slot_lock:
            self._latest_capture_frame = frame
            self._latest_capture_seq = frame_seq
            self._latest_capture_ts = capture_ts
        return True

    def _run_process_capture_pump(self) -> None:
        """Runs this camera's whole capture+publish loop from a SEPARATE thread once escalated
        (see HANG_ESCALATION_THRESHOLD) -- started directly by recover_from_hung_read (the
        watchdog's thread), not by run()'s own loop, because run()'s thread may itself be the one
        permanently wedged inside a native cap.read() call that will never return: nothing in
        this process can preempt that thread, so the only way to make this camera recoverable
        again is to serve it from a different thread entirely and simply abandon (never join) the
        stuck one -- see run()'s own top-of-loop comment. CaptureProcess.read() never blocks (see
        its own docstring/tests), so this loop can never itself get stuck the same way; a wedged
        child process is killed and respawned entirely inside CaptureProcess's own watchdog."""
        while self._running:
            if not self._enabled:
                if self._process_capture is not None:
                    self._process_capture.stop()
                    self._process_capture = None
                time.sleep(0.2)
                continue
            self._ensure_process_capture()
            ok, frame, _proc_seq, _proc_ts = self._process_capture.read()
            self._process_captured_frame(ok, frame)
            time.sleep(0.01 if ok else 0.03)
        self._ai_worker.unregister(self.camera_id)
        if self._process_capture is not None:
            self._process_capture.stop()
            self._process_capture = None

    def ai_due(self, now: float) -> bool:
        """AIWorker asks every registered camera this on each scan pass —
        True once this camera's configured AI cadence (self._detect_every_n_
        frames, translated to a wall-clock interval via AI_BASE_FPS) has
        elapsed since its last pass."""
        return now >= self._ai_next_due

    @staticmethod
    def _fused_category_from_attribute_result(result) -> "tuple[str, float] | None":
        """Maps a core.attributes.AttributeResult (the Global-Person-scoped,
        quality-gated, confidence-weighted fused gender/age) onto this
        module's on-screen 'female'/'male' category scheme, or None
        if there's nothing confident enough yet to override a camera's own
        local guess with. Age is ignored."""
        if result is None or getattr(result, "status", "unknown") != "ok":
            return None
        gender = getattr(result, "gender", None)
        if gender in ("MALE", "FEMALE"):
            return gender.lower(), result.gender_confidence
        return None

    @staticmethod
    def _drop_unreported_person_boxes(draw_boxes: list, person_boxes: list, visible_tracks: list[dict]) -> None:
        """Person detections the tracker did not report (a low-score box that matched no track, or a still
        unconfirmed newborn track) are not drawn either - otherwise they would blink on and off."""
        reported = {id(t["bbox"]) for t in visible_tracks}
        unreported = {id(b) for b in person_boxes} - reported
        if unreported:
            draw_boxes[:] = [d for d in draw_boxes if id(d[0]) not in unreported]

    def _draw_coasting_tracks(self, draw_boxes: list) -> None:
        """Keep drawing a person's motion-predicted box for a moment after a missed detection."""
        for track in self._person_tracker.coasting_tracks():
            category, category_conf = self._resolved_category(track)
            _label, color = self._label_for(category, category_conf)
            text = "PERSON" if _label.startswith("PERSON") else category.upper()
            gid = self.global_id_resolver(self.camera_id, track["track_id"]) if self.global_id_resolver else None
            draw_boxes.append((track["bbox"], f"{text} | {gid}" if gid else text, color))

    def _detect_faces(self, frame, visible_tracks: list[dict], evicted_tracks: list[dict], draw_boxes: list) -> None:
        """Full-frame face detection for this AI pass (see face_detector in __init__). Skipped when
        nobody is on screen, so an empty booth costs nothing. Never raises: faces are an add-on and
        must not break person/product detection."""
        if self._face_service is None:
            return
        for track in evicted_tracks:
            self._face_service.forget(track["track_id"])
        if not visible_tracks:
            self._latest_faces = []
            return
        try:
            self._face_service.process(frame, visible_tracks)
        except Exception as exc:
            if not self._face_error_logged:
                self._face_error_logged = True
                logger.warning("[%s] face detection failed (%s) - continuing without faces", self.camera_id, exc)
            self._latest_faces = []
            return
        self._draw_stable_faces(visible_tracks, draw_boxes)

    def read_stall_seconds(self) -> float:
        """How long the read() currently in progress has been blocked (0.0 when not reading)."""
        started = self._read_started
        return 0.0 if started is None else max(time.monotonic() - started, 0.0)

    def recover_from_hung_read(self, stall_sec: float = CAPTURE_READ_STALL_SEC) -> bool:
        """Watchdog entry point (called from another thread, ~every 2 s by BoothManager). If the
        capture thread has been blocked in cap.read() for >= stall_sec, report the camera offline
        (instead of a frozen "online" tile) and release the capture from THIS thread, which
        unblocks a stuck DirectShow read on backends that honour it; the capture loop then
        reopens through the normal backoff. Returns True when a recovery was triggered. Repeats
        after another stall window if the read is still blocked (a truly stuck native call)."""
        if self._escalate_to_process:
            return False    # process-isolated reads never block -- nothing here to recover
        started = self._read_started
        if started is None or time.monotonic() - started < stall_sec:
            return False
        self.hang_recoveries += 1
        self._consecutive_hang_recoveries += 1
        logger.warning(
            "[%s] cap.read() blocked for %.1fs (device=%s, recovery #%d, consecutive #%d) -- "
            "releasing the capture from the watchdog and reopening", self.camera_id,
            time.monotonic() - started, self.device, self.hang_recoveries,
            self._consecutive_hang_recoveries)
        self._emit_status("offline", "กล้องไม่ตอบสนอง (อ่านภาพค้าง) — กำลังรีเซ็ตการเชื่อมต่อ")
        self._note_fault()
        self._release_capture()
        self._read_started = time.monotonic()   # next escalation only after another full window
        if (HANG_ESCALATION_THRESHOLD > 0
                and self._consecutive_hang_recoveries >= HANG_ESCALATION_THRESHOLD):
            logger.warning(
                "[%s] %d consecutive hang recoveries with no good frame in between -- escalating "
                "to process-isolated capture (a wedged native read can no longer freeze this "
                "camera's worker thread)", self.camera_id, self._consecutive_hang_recoveries)
            self._escalate_to_process = True
            self._read_started = None
            # Started here, from the watchdog's own thread, not left for run()'s loop to notice --
            # if THIS worker's own run() thread is the one genuinely, permanently wedged inside
            # cap.read(), it will never get back to the top of its loop to start anything itself
            # (see _run_process_capture_pump's docstring). Guarded so a second consecutive-hang
            # burst after a successful escalation (HANG_ESCALATION_THRESHOLD is never reached
            # again once _consecutive_hang_recoveries stops incrementing post-escalation, but this
            # stays defensive against any future path that re-enters here) never starts a second
            # pump thread for the same camera.
            if self._pump_thread is None:
                self._pump_thread = threading.Thread(
                    target=self._run_process_capture_pump, daemon=True, name=f"pump-{self.camera_id}")
                self._pump_thread.start()
        return True

    def _is_usb_like(self) -> bool:
        from core.camera_identity import device_as_index
        return device_as_index(self.device) is not None or str(self.device).startswith("/dev/video")

    def note_device_change(self) -> None:
        """The device monitor saw a camera vanish from Windows' device list. We cannot tell which one from
        the name alone (two identical webcams), so every online camera is held to the strict deadline for a
        few seconds: a healthy one keeps delivering frames and is unaffected, the unplugged one goes
        offline within a fraction of a second."""
        self._suspect_until = time.monotonic() + SUSPECT_WINDOW_SEC

    def check_stream_health(self, now: float | None = None) -> bool:
        """Watchdog entry point (BoothManager calls it ~2x/s from its own thread). An 'online' USB camera
        that has produced no good frame for STREAM_STALE_SEC - or whose picture has been bit-identical
        for FROZEN_STREAM_SEC - is reported offline immediately, so the tile stops showing a frozen or
        garbled picture and shows the reconnecting card. Going back online needs ONLINE_GOOD_STREAK good
        frames, so a flapping stream does not spam alerts. Returns True when it flagged the camera."""
        if self._last_status not in ("online", "degraded") or not self._is_usb_like():
            return False
        last = self._last_good_frame_ts
        if last is None:
            return False
        now = time.monotonic() if now is None else now
        suspect = now < self._suspect_until
        stale = now - last >= (STREAM_STALE_SUSPECT_SEC if suspect else STREAM_STALE_SEC)
        try:
            frozen = self._frame_integrity.frozen_for(time.time()) >= (
                FROZEN_STREAM_SUSPECT_SEC if suspect else FROZEN_STREAM_SEC)
        except Exception:
            frozen = False
        if not (stale or frozen):
            return False
        self.stale_recoveries += 1
        logger.warning("[%s] no good frame (%s) -- reporting the camera offline and reconnecting (device=%s)",
                       self.camera_id, "frozen picture" if frozen and not stale else
                       f"{now - last:.1f}s without a valid frame", self.device)
        self._good_streak = 0
        self._emit_status("offline", "ภาพจากกล้องหยุดหรือเสียหาย — กำลังเชื่อมต่อใหม่")
        if self._device_present() is False:
            # Windows no longer lists the device: free the (possibly blocked) capture right now instead of
            # waiting for the hung-read window.
            self._release_capture()
        return True

    def get_integrity_stats(self) -> dict[str, int]:
        """How many frames this camera's corrupt-frame gate rejected, by reason (since start)."""
        return dict(self._integrity_counts)

    def get_frame_sequence(self) -> int:
        """Monotonically increasing count of frames captured (this process's lifetime, from this
        open of the device -- not persisted across a reconnect). Lets a caller (the performance
        API) tell "genuinely new frames still arriving" apart from "the same status, no progress",
        which the status string alone cannot: a stalled camera can stay reported "online" for a
        watchdog window before check_stream_health notices."""
        return self._latest_capture_seq

    def get_last_capture_ts(self) -> float | None:
        """Wall-clock time.time() of the last frame that passed capture (not necessarily AI-
        processed) — spec section 12's freshness contract: paired with get_frame_sequence(), lets
        a caller tell a genuinely fresh frame from a stale one on its own, without trusting the
        status string alone."""
        return self._latest_capture_ts

    def get_diagnostics(self) -> dict:
        """Everything spec section 30 asks a camera to expose beyond the status/message BoothManager
        already tracks in camera_status — one call so BoothManager doesn't have to know this
        worker's internal attribute names."""
        frozen_reasons = {"frozen"}
        return {
            "capture_mode": self.get_capture_mode(),
            "worker_pid": self.get_worker_pid(),
            "worker_thread_alive": self.is_alive(),
            "frame_sequence": self._latest_capture_seq,
            "capture_timestamp": self._latest_capture_ts,
            "hang_recoveries": self.hang_recoveries,
            "consecutive_hang_recoveries": self._consecutive_hang_recoveries,
            "stale_recoveries": self.stale_recoveries,
            "stuck_releases": self.stuck_releases,
            "corrupt_frame_count": sum(v for k, v in self._integrity_counts.items() if k not in frozen_reasons),
            "frozen_frame_count": self._integrity_counts.get("frozen", 0),
            "worker_restart_count": (self.hang_recoveries + self.stale_recoveries
                                      + (self._process_capture.respawns if self._process_capture else 0)),
            "reason_code": self._last_reason_code,
        }

    def get_face_count(self) -> int:
        return len(self._latest_faces)

    def get_latest_faces(self) -> list[dict]:
        return list(self._latest_faces)

    def _apply_track_categories_to_labels(self, draw_boxes: list, visible_tracks: list[dict]) -> None:
        """Make the on-screen person label show the TRACK's aggregated category instead of the
        single-frame guess _run_yolo drew. Before this, the box flipped MALE/FEMALE from frame to
        frame (and showed one bad frame's guess) while the dashboard used the smoothed track
        category - the user saw a man labelled FEMALE for a moment. Matching is by bbox object
        identity, exactly like _append_global_ids_to_labels below. The trailing "NN%" of the
        original label is kept."""
        try:
            index_by_bbox_id = {id(bbox): idx for idx, (bbox, _label, _color) in enumerate(draw_boxes)}
            for track in visible_tracks:
                idx = index_by_bbox_id.get(id(track["bbox"]))
                if idx is None:
                    continue
                bbox, old_label, _old_color = draw_boxes[idx]
                category, category_conf = self._resolved_category(track)
                new_word, new_color = self._label_for(category, category_conf)
                number = old_label.rsplit(" ", 1)[-1] if "%" in old_label else ""
                draw_boxes[idx] = (bbox, f"{new_word.rsplit(' ', 1)[0]} {number}".strip(), new_color)
        except Exception:
            pass  # cosmetic only - never let it break an AI pass

    def _resolved_category(self, track: dict) -> "tuple[str, float]":
        """(category, confidence) drawn for this person: the shared per-person gender when the resolver has one, else
        the track's own aggregated category."""
        resolver = self.gender_resolver
        if resolver is not None:
            try:
                resolved = resolver(self.camera_id, track["track_id"])
                if resolved and resolved[0] in ("male", "female"):
                    return resolved[0], float(resolved[1])
            except Exception:
                pass
        category = track.get("category")
        return (category if category in ("male", "female") else "unknown"), 0.0

    def _label_for(self, category: str, conf: float) -> "tuple[str, tuple[int, int, int]]":
        if category == "unknown" and self.gender_resolver is not None:
            return f"PERSON {conf:.0%}", PERSON_UNKNOWN_COLOR       # no evidence yet: pending, not a gender guess
        return self._category_label_and_color(category, conf)

    def _draw_stable_faces(self, visible_tracks: list, draw_boxes: list) -> None:
        """Append the steady face boxes (held through detector misses, following the person) and refresh the count."""
        if self._face_service is None:
            return
        latest = []
        for face in self._face_service.stable_faces(visible_tracks):
            box = [float(face.x1), float(face.y1), float(face.x2), float(face.y2)]
            draw_boxes.append((box, f"FACE {face.score:.0%}", FACE_COLOR))
            latest.append({"bbox": box, "score": float(face.score), "quality": float(face.quality),
                           "track_id": face.track_id})
        self._latest_faces = latest

    def _append_global_ids_to_labels(self, draw_boxes: list, visible_tracks: list[dict]) -> None:
        """Patches each person box's already-built label (e.g. "MALE 90%")
        to "MALE 90% | P000001" once global_id_resolver (see __init__)
        confirms this track has been Re-ID-matched. A no-op wherever no
        resolver is configured, or for a track Re-ID hasn't sampled/matched
        yet (see core.reid.ReIDSampler's throttle) -- exactly today's
        category-only label in that case, never a fabricated ID.

        When global_attribute_resolver is also configured (see __init__),
        this also replaces the label's category with the matched Global
        Person's fused gender/age whenever that fused result is confident
        -- so the same physical person can't show a different category on
        two cameras at the same moment just because each camera's own
        per-frame classification saw different evidence. Falls back to the
        existing per-camera category exactly as before whenever no
        global_id is resolved yet, or the Global Person's fused result
        isn't confident enough.

        Matches a track back to its draw_boxes entry by bbox object
        identity: core.tracker.PersonTracker.update() returns each track's
        "bbox" as the literal same list object passed in (never a copy), and
        _run_yolo appended that same object into both person_boxes and
        draw_boxes -- so `is` comparison via id() reliably finds the right
        entry without needing draw_boxes to carry its own track_id."""
        if self.global_id_resolver is None or not visible_tracks:
            return
        try:
            draw_index_by_bbox_id = {id(bbox): idx for idx, (bbox, _label, _color) in enumerate(draw_boxes)}
            for track in visible_tracks:
                draw_index = draw_index_by_bbox_id.get(id(track["bbox"]))
                if draw_index is None:
                    continue
                global_id = self.global_id_resolver(self.camera_id, track["track_id"])
                if not global_id:
                    continue
                bbox, label, color = draw_boxes[draw_index]
                if self.global_attribute_resolver is not None:
                    try:
                        fused = self.global_attribute_resolver(global_id)
                        mapped = self._fused_category_from_attribute_result(fused)
                        if mapped is not None:
                            fused_category, fused_conf = mapped
                            label, color = self._category_label_and_color(fused_category, fused_conf)
                    except Exception:
                        pass  # fall back to this camera's own local label/color
                draw_boxes[draw_index] = (bbox, f"{label} | {global_id}", color)
        except Exception:
            pass  # the on-screen label must never break over this best-effort enrichment

    def _publish_boxes(self, frame_seq: int, boxes: list) -> None:
        """Make `boxes` (computed on the frame numbered `frame_seq`) the ones drawn from now on."""
        self._last_boxes = boxes  # single atomic replace, see run()'s comment on why this is safe
        follower = self._box_follower
        if follower is not None:
            try:
                follower.submit(frame_seq, boxes)
            except Exception:
                logger.debug("[%s] box follower submit failed", self.camera_id, exc_info=True)

    def run_ai_pass(self) -> None:
        """Runs one full AI pass (YOLO + throttled custom recognition) for
        this camera on whatever the *latest* captured frame is — called
        only from AIWorker's shared thread, never from this camera's own
        capture thread. Safe to call even if this camera has no fresh frame
        yet (returns immediately)."""
        interval = self._detect_every_n_frames / AI_BASE_FPS if self._detect_every_n_frames > 0 else 0.0
        self._ai_next_due = time.monotonic() + interval

        with self._frame_slot_lock:
            frame = self._latest_capture_frame
            frame_seq = self._latest_capture_seq
        if frame is None:
            return

        draw_boxes: list[tuple[list[float], str, tuple[int, int, int]]] = []
        detections = []
        ai_start = time.time()
        try:
            person_boxes, person_categories, person_confidences, legacy_detections, claimed_boxes = self._run_yolo(frame, draw_boxes)
            visible_tracks, evicted_tracks = self._person_tracker.update(
                person_boxes, person_categories, person_confidences,
                detection_scores=getattr(self, "_last_person_scores", None) or None,
                high_score=self.conf_threshold)
            self._drop_unreported_person_boxes(draw_boxes, person_boxes, visible_tracks)
            frame_h, frame_w = frame.shape[:2]
            self._apply_track_categories_to_labels(draw_boxes, visible_tracks)
            # Publish the boxes NOW, before the slow part of the pass (Re-ID, gender/face analysis, DB), so what
            # is on screen is only as old as YOLO + the tracker - not the whole pass. Labels for tracks Re-ID
            # already knows carry their id from the previous pass; the final publish below refreshes them.
            base_boxes = list(draw_boxes)
            early = list(base_boxes)
            self._append_global_ids_to_labels(early, visible_tracks)
            self._draw_coasting_tracks(early)
            self._draw_stable_faces(visible_tracks, early)   # the previous pass's steady face boxes: no blink mid-pass
            self._publish_boxes(frame_seq, early)
            # on_person_tracks (web/booth_manager.py's _on_person_tracks)
            # runs Re-ID resolution synchronously inside this call, so
            # global_id_resolver below sees this pass's just-resolved
            # mapping, not last pass's.
            self.on_person_tracks(self.camera_id, visible_tracks, evicted_tracks, frame_w, frame_h, frame)
            draw_boxes[:] = base_boxes
            self._apply_track_categories_to_labels(draw_boxes, visible_tracks)   # now with this pass's Re-ID result
            self._append_global_ids_to_labels(draw_boxes, visible_tracks)
            self._detect_faces(frame, visible_tracks, evicted_tracks, draw_boxes)
            self._draw_coasting_tracks(draw_boxes)
            detections.extend(legacy_detections)
            self._ai_pass_count += 1
            if self._should_run_custom_recognition():
                detections.extend(self._run_custom_recognition(frame, claimed_boxes, draw_boxes))
            self._ai_success_streak += 1
            if (self._last_status == "error" and
                    self._ai_success_streak >= AI_RECOVERY_SUCCESS_COUNT):
                self._emit_status("online", "AI กลับมาทำงานปกติ")
        except Exception as exc:
            self._ai_success_streak = 0
            self._emit_status("error", f"AI ตรวจจับล้มเหลว: {exc}")

        self._publish_boxes(frame_seq, draw_boxes)
        if self._performance_monitor is not None:
            self._performance_monitor.record_ai(self.camera_id, time.time() - ai_start)
        self.on_detections(self.camera_id, detections)

    def stop(self):
        self._stop_event.set()
        self._running = False
        # Helps unblock backends whose read() hangs after a disconnect.
        self._release_capture()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=STOP_JOIN_TIMEOUT_SEC)
        # Once escalated (see HANG_ESCALATION_THRESHOLD), _run_process_capture_pump -- not this
        # thread -- owns self._process_capture; it stops it as part of its own exit once it
        # observes self._running is False, above. This thread's own join, just above, can time
        # out without the ORIGINAL run() thread ever actually dying if it is the one permanently
        # wedged inside a native cap.read() call -- an already-tolerated, pre-existing trade-off
        # for a stuck camera at shutdown (a leaked daemon thread never blocks process exit), not a
        # regression introduced by escalation.
        pump = self._pump_thread
        if pump is not None and pump.is_alive() and threading.current_thread() is not pump:
            pump.join(timeout=STOP_JOIN_TIMEOUT_SEC)

    def set_enabled(self, enabled: bool) -> None:
        """Turn this camera on/off without tearing down its thread — for a
        camera that's plugged in but not currently needed at the booth.
        Cheap and safe to call from any thread; takes effect within one
        iteration of the capture loop (see run())."""
        self._enabled = enabled

    def is_capture_open(self) -> bool:
        """Whether this camera currently holds a live capture handle. Used
        by web/booth_manager.py's Hot-Plug Scan to tell which already-known
        camera indices are safe for it to also probe (one whose worker has
        already released its capture — offline/reconnecting — since nothing
        else has it open right now) from which must stay untouched (a live
        capture the Hot-Plug Scan must never also open — see _open_capture's
        own module-level comment on why). Once escalated (see
        HANG_ESCALATION_THRESHOLD), self._cap stays None forever but the
        physical device is still very much held open, just by this
        camera's own CaptureProcess child instead."""
        if self._process_capture is not None and self._process_capture.alive:
            return True
        with self._cap_lock:
            return self._cap is not None

    def request_immediate_retry(self) -> None:
        """Skips ahead of the current reopen backoff so the very next loop
        iteration (see run()) retries the connection right away, instead of
        waiting out whatever delay a preceding string of failed attempts
        grew _reopen_backoff_sec to (up to REOPEN_BACKOFF_MAX_SEC). Meant to
        be called by web/booth_manager.py's Hot-Plug Scan once it has
        independently confirmed (via discover_cameras) that this camera's
        device is actually reachable again — reconnecting right away at
        that point, rather than waiting on this worker's own unrelated
        timer, is what keeps "unplug, wait a while, then replug" fast even
        after the backoff has grown large. A no-op if this worker is
        already connected (nothing to retry) or hasn't started yet."""
        self._last_reopen_attempt = 0.0
        self._reopen_backoff_sec = REOPEN_BACKOFF_INITIAL_SEC

    # ---------------------------------------------- adaptive control knobs
    # Implements core.performance.ManagedCamera — AdaptiveController reads
    # and adjusts these under load. Plain attribute get/set, same
    # cheap-and-thread-safe pattern as set_enabled above.
    def get_detect_every_n_frames(self) -> int:
        return self._detect_every_n_frames

    def set_detect_every_n_frames(self, n: int) -> None:
        self._detect_every_n_frames = max(1, n)

    def get_ai_imgsz(self) -> int:
        return self._ai_imgsz

    def set_ai_imgsz(self, size: int) -> None:
        self._ai_imgsz = size

    def set_tripwire_overlay(self, overlay: dict | None) -> None:
        """Sets (or clears, with None) what the capture thread draws every
        frame on top of the video (see _draw_tripwire) — a plain dict
        {"x1","y1","x2","y2","inside_side","count_in","count_out"}, or None
        to draw nothing. Whoever owns crossing detection (web/booth_manager
        .py) calls this from the AI-worker thread after each pass; a single
        atomic attribute replacement is all that's needed for the capture
        thread to read it safely (same pattern as self._last_boxes — see
        run()'s comment), and it takes effect on the very next captured
        frame regardless of how often (or rarely, under AI Pause/throttle)
        the AI pass that updates the counts actually runs — the line itself
        never depends on AI running at all once set."""
        self._tripwire_overlay = overlay

    def get_resolution(self) -> tuple[int, int] | None:
        """Actual negotiated capture resolution, for diagnostics — None if
        the camera isn't currently open."""
        with self._cap_lock:
            cap = self._cap
        if cap is None or not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height) if width and height else None
