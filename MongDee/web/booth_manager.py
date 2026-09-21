"""Headless orchestration shared by the browser booth view — owns the camera
workers, the recognition state, and the in-memory state the API/streams read.

This is the web equivalent of ui/main_window.py, but with no GUI: instead of
Qt Signals updating widgets, core.vision.CameraWorker's plain callbacks
update shared dicts (each behind a lock) that FastAPI request handlers read
directly, since there's no GUI thread to marshal onto here.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core import database as db
from core import training
from core.aggregator import DetectionAggregator
from core.booth_sessions import BoothSessionManager
from core.camera_identity import device_as_index, get_physical_camera_identities, normalize_device_key
from core.device import device_backend, device_label
from core.device_presence import DevicePresenceMonitor, PresenceEvent
from core.face_identity import DEFAULT_SFACE_MODEL, FaceIdentity
from core.interest_tracker import ProductInterestTracker
from core.performance import AdaptiveConfig, AdaptiveController, PerformanceMonitor, detect_hardware
from core.person_segmenter import get_person_segmenter
from core.attributes import AttributeSampler, GlobalPersonAttributeSmoother
from core.body_cues import BODY_FEATURE_DIM, COLOR_DESCRIPTOR_DIM, body_features, hair_descriptor
from core.body_gender import BodyGenderLearner, BodyGenderModel
from core.clip_gender import ClipBodyGenderModel, try_load_backend as try_load_clip_backend
from core.readiness import readiness_to_json, run_readiness_check
from core.reid import GlobalIdentityRegistry, PersonReIDEmbedder, ReIDSampler
from core.tripwire import DIRECTION_IN, TripwireCounter, TripwireLine
from core.vision import AIWorker, CameraWorker, discover_cameras

logger = logging.getLogger("mongdee.web.booth_manager")

HEARTBEAT_INTERVAL_SEC = 30
CAPTURE_WATCHDOG_INTERVAL_SEC = 0.5
# How often the open/close Schedule re-checks the active booth's
# open_time/close_time against the clock — coarse on purpose: a booth
# opening/closing a few tens of seconds late is harmless, and a /settings
# edit doesn't need to take effect immediately.
SCHEDULE_CHECK_INTERVAL_SEC = 30
# How often the Hot-Plug Scan probes for USB webcams that weren't part of
# the original --cameras / discover_cameras() list at startup — a camera
# plugged in mid-session (or a fresh index left over after a replug shifted
# enumeration) is otherwise invisible until the process is restarted. Only
# indices *not already owned* by an existing camera are ever probed (see
# core.vision.discover_cameras's `skip`), so this never touches a camera
# that's already connected/reconnecting on its own.
#
# The interval backs off (doubling, capped at HOTPLUG_SCAN_MAX_INTERVAL_SEC)
# every scan that finds nothing new, and resets to the base interval the
# moment something changes — most of a booth's runtime has zero hot-plug
# activity, so steady-state this avoids opening/probing camera indices on a
# fixed aggressive cadence forever for no reason. A scan is also skipped
# outright (without resetting the backoff) whenever system CPU is already
# under AI-Pause-level pressure (see core.performance.AdaptiveConfig's
# cpu_pause_pct) — adding more device-probe overhead exactly when the
# system is already struggling is the one thing this must never do.
HOTPLUG_SCAN_INTERVAL_SEC = 5
HOTPLUG_SCAN_MAX_INTERVAL_SEC = 60
HOTPLUG_MAX_INDEX = 16
HOTPLUG_SKIP_SCAN_CPU_PCT = 90.0
# Starting N camera worker threads back-to-back means N cv2.VideoCapture
# opens can all try to begin actual hardware streaming at nearly the same
# instant. core.vision._open_capture's own _open_lock only serializes the
# DirectShow/MSMF *open* handshake itself (a known race during simultaneous
# enumeration) — it does not, and cannot, serialize what the camera driver
# does *after* that succeeds, when it actually claims USB
# bandwidth/hardware-encoder resources to start the stream. On a real
# multi-camera-over-one-USB-hub setup this is exactly what produces
# Windows' MF_E_HW_MFT_FAILED_START_STREAMING (0xC00D3704) — the driver
# already reported "open" but the *second* camera's actual streaming start
# is denied because the hub/controller can't grant a second concurrent
# hardware stream while the first one is still claiming it. Staggering each
# worker's start by this long gives the previous camera a real chance to
# finish claiming its hardware resources first, before the next one tries —
# it never fully eliminates a hardware-level bandwidth ceiling (see
# core.vision.CameraWorker._offline_message's own "ยังเชื่อมต่ออยู่... USB
# hub" distinction for when that's genuinely the wall), but it removes the
# purely-timing-caused instance of this failure, which is the one thing
# software here can actually fix.
#
# Value measured on real hardware (2x identical USB UVC cameras sharing one
# USB hub, see docs/design/camera-pipeline-root-cause-repair.md's real-
# hardware validation section): opening both cameras concurrently through
# core.vision._open_capture with a 1.5s stagger between them still failed
# or stalled the second camera in 3 of 5 trials (one exceeded a 30s wait
# entirely). 2.5s produced 0 failures in 6 consecutive trials, and 4.0s 0
# failures in 5 — 2.5s was picked as the smallest interval that measured
# reliably clean on this hardware, to avoid inflating multi-camera startup
# time further than the evidence requires.
CAMERA_STARTUP_STAGGER_SEC = 2.5
# Cameras themselves capture at up to ~30fps (see core/vision.py); this used
# to cap the browser-facing MJPEG stream at 12fps regardless, which read as
# visibly choppy compared to the live feed. Raised closer to capture rate so
# the stream looks smooth — this only changes how often the already-encoded
# latest JPEG is re-sent, not how much work the camera/detection threads do.
STREAM_FPS = 24
MAX_ALERTS = 50
JPEG_QUALITY = 80
DISPLAY_MAX_WIDTH = 960
TRACK_FACE_MIN_CONFIDENCE = 0.85   # a track's own face reading must be this confident ...
TRACK_FACE_MIN_PX = 24             # ... from a face at least this big ...
TRACK_FACE_MIN_VOTES = 2           # ... and repeated, before it may split the track off an identity of the other gender
MIN_PRESENCE_DURATION_SEC = 1.0   # denoise single-frame false-positive person detections

BOOTH_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "booth_settings.json"
# Default only — always prefer booth_settings_path_for(db_path) once a
# db_path is known (e.g. from --db), so a process pointed at an isolated/
# test database never clobbers the real project's active-booth marker.
# Discovered the hard way: a manually-isolated `--db` test run still wrote
# the *real* data/booth_settings.json (this constant, unconditionally)
# because activate_booth() used to reference it directly instead of
# deriving a path from self.db_path the way every other per-database file
# already does.


def booth_settings_path_for(db_path: Path) -> Path:
    return Path(db_path).resolve().parent / "booth_settings.json"
# Trainer image/video uploads stage here instead of the OS temp dir
# (tempfile's default) — %TEMP% on Windows is usually the C: drive, and a
# full/small system drive would otherwise break every upload with no
# indication it has anything to do with disk space on a completely
# different drive than the one this project lives on.
UPLOAD_TMP_ROOT = Path(__file__).resolve().parent.parent / "data" / "tmp"


def _build_placeholder_jpeg(text: str) -> bytes:
    """A static "camera is off" frame served in place of a stale last frame
    while a camera is disabled — a frozen frame would look like a bug rather
    than a deliberate off state."""
    frame = np.full((360, 480, 3), 24, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = max(0, (480 - tw) // 2), (360 + th) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (140, 140, 140), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


_DISABLED_CAMERA_JPEG = _build_placeholder_jpeg("CAMERA DISABLED")
_CLOSED_BOOTH_JPEG = _build_placeholder_jpeg("BOOTH CLOSED")
# Served instead of the last-known frame whenever a camera's own status
# isn't "online"/"degraded" (connecting, reconnecting after an unplug,
# offline, or erroring) — without this, get_latest_jpeg() would keep
# re-serving the last frame captured *before* the disconnect forever, which
# looks exactly like a frozen/stuck stream instead of an honest "not
# currently live" state. Cleared the instant the camera reports "online"
# again (see _on_frame), so a reconnect shows fresh video immediately.
_RECONNECTING_CAMERA_JPEG = _build_placeholder_jpeg("CAMERA RECONNECTING...")


def load_active_booth_id(path: Path = BOOTH_SETTINGS_PATH) -> str | None:
    """Which registry Booth (core.database's `booths` table) this process
    should report as, saved whenever /settings activates a different one —
    see web_server.py's startup, which reads this before falling back to a
    freshly-bootstrapped booth."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("active_booth_id")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_camera_settings(path: Path = BOOTH_SETTINGS_PATH) -> dict:
    """Built-in-camera policy for auto-discovery, read from the same
    booth_settings.json load_active_booth_id uses.

    Defaults to disabled with nothing configured. Prefer
    `builtin_camera_names` (e.g. `["HD WebCam"]`, exactly as
    core.camera_identity.list_directshow_devices() reports it) over
    `builtin_camera_indices` — a device's cv2 index is NOT a stable
    identity (Windows re-enumerates on reboot/replug/device-count changes,
    and this project has directly observed a built-in webcam's index shift
    after USB cameras were unplugged/replugged during the same session,
    silently un-excluding it), whereas its DirectShow friendly name stays
    the same. `builtin_camera_indices` is kept only as a last-resort
    fallback for a machine/build where core.camera_identity can't resolve
    names at all (no pygrabber, non-Windows). Either way this is an
    explicit opt-in the operator sets once, after checking the real name/
    index on this machine (e.g. via scripts/camera_diagnostic.py) — cv2
    alone can't reliably tell a laptop's integrated webcam apart from a USB
    one, so guessing risks silently excluding the wrong camera.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    return {
        "enable_builtin_camera": bool(data.get("enable_builtin_camera", False)),
        "builtin_camera_names": [str(n) for n in data.get("builtin_camera_names", [])],
        "builtin_camera_indices": [int(i) for i in data.get("builtin_camera_indices", [])],
    }


def compute_camera_skip_indices(camera_settings: dict) -> frozenset[int]:
    """Which cv2 device indices discovery must never open right now: any
    detected virtual camera (OBS Virtual Camera, ManyCam, ...) — never a
    legitimate physical booth source, excluded unconditionally — plus,
    unless enable_builtin_camera is set, whichever *currently* present
    index matches a configured built-in name (resolved fresh every call,
    so it tracks a re-enumeration instead of going stale) or a configured
    legacy index.

    Called identically at startup (app.py/web_server.py) and on every
    Hot-Plug Scan tick (BoothManager._hotplug_loop) — a built-in camera
    that wasn't present yet at startup must still never get picked up once
    the scan finds it.

    As a side effect, also refreshes core.vision's own forbidden-name
    registry (set_forbidden_device_names) — the actual safety net that
    matters isn't this function (which only guards *new* discovery), it's
    core.vision._open_capture re-checking identity on every single open,
    including a reconnect of an *already-assigned* camera_id. Without that,
    an index Windows re-enumerates out from under a running CameraWorker
    could silently start reading the built-in camera on its next
    reconnect — this was observed live during development. This function
    is called often enough (every startup, every Hot-Plug Scan tick) that
    piggy-backing the refresh here keeps it current without a separate
    call site to remember.
    """
    from core.camera_identity import is_virtual_camera_name, list_directshow_devices
    from core.vision import set_forbidden_device_names

    current_names = list_directshow_devices()
    skip = {idx for idx, name in current_names.items() if is_virtual_camera_name(name)}
    builtin_names = [] if camera_settings["enable_builtin_camera"] else camera_settings["builtin_camera_names"]
    set_forbidden_device_names(builtin_names)
    if not camera_settings["enable_builtin_camera"]:
        configured_names = {n.lower() for n in camera_settings["builtin_camera_names"]}
        skip |= {idx for idx, name in current_names.items() if name.lower() in configured_names}
        skip |= set(camera_settings["builtin_camera_indices"])
    return frozenset(skip)


def dedupe_camera_devices(camera_devices):
    """Drops any (camera_id, device) pair whose `device` is already bound
    to an earlier camera_id in the same list, keeping the first occurrence
    — the startup-time counterpart to BoothManager.add_camera()'s own
    duplicate-device rejection (spec: MongDee multi-USB-camera root-cause
    repair — "DEVICE INDEX COLLISION"). Confirmed on real hardware: two
    independent cv2.VideoCapture objects opened on the same physical camera
    index both succeed and both stream valid, live video — nothing about
    capture itself fails — so two camera_ids ended up silently showing the
    exact same physical camera's picture, which reads exactly like a
    cross-camera frame bug without actually being a frame-routing one.

    Startup (web_server.py's --cameras flag, app.py's discovered device
    list) degrades to "skip the duplicate, keep going" rather than raising,
    since crashing the whole booth over one bad --cameras value would be
    worse than just not creating the redundant camera_id; add_camera()
    (an interactive /settings action) raises instead, so the operator gets
    immediate, specific feedback.

    Two *different* raw device values (e.g. index 0 vs index 2) can still be
    the exact same physical camera if Windows re-enumerated between when
    each was resolved (spec: MongDee physical-camera-identity follow-up) —
    checked here too, via core.camera_identity's best-effort physical_id
    (see its own module docstring for the confidence levels/fallback this
    relies on), *in addition to* the raw-value check above, never instead of
    it: on any platform/environment where physical-identity data is
    unavailable (non-Windows, pygrabber/PowerShell missing), this layer is
    silently a no-op and behavior is identical to before this was added."""
    physical_ids_by_index = {
        idx: identity.physical_id for idx, identity in get_physical_camera_identities().items()
    }
    seen: dict[str, str] = {}          # normalized device key -> the camera_id that claimed it
    seen_physical: dict[str, str] = {}  # physical_id -> the camera_id that claimed it
    deduped = []
    for camera_id, device in camera_devices:
        key = normalize_device_key(device)
        if key in seen:
            logger.warning(
                "[STARTUP] ข้าม %s (device=%s) — กล้องตัวเดียวกันถูกกำหนดให้ %s ไปแล้ว "
                "(เปิดกล้องตัวเดียวกันสองครั้งจะทำให้เห็นภาพซ้ำกันในสองหน้าต่าง)",
                camera_id, device, seen[key],
            )
            continue
        index = device_as_index(device)
        physical_id = physical_ids_by_index.get(index) if index is not None else None
        if physical_id is not None and physical_id in seen_physical:
            logger.warning(
                "[STARTUP] ข้าม %s (device=%s) — เป็นกล้องตัวเดียวกันทางกายภาพกับ %s "
                "ที่ index อื่น (physical_id=%s)",
                camera_id, device, seen_physical[physical_id], physical_id,
            )
            continue
        seen[key] = camera_id
        if physical_id is not None:
            seen_physical[physical_id] = camera_id
        deduped.append((camera_id, device))
    return deduped


class BoothManager:
    def __init__(self, booth_id, booth_name, event_id, camera_devices, model,
                 model_device, catalog, recognizer, db_path, gender_age_backend=None,
                 attribute_backend=None, adaptive_config: AdaptiveConfig | None = None,
                 face_detector=None, face_identity=None):
        camera_devices = dedupe_camera_devices(camera_devices)
        self.booth_id = booth_id
        self.booth_name = booth_name
        self.event_id = event_id
        self.model = model
        self.model_device = model_device
        self.catalog = catalog
        self.recognizer = recognizer
        # Tie the product gallery to the catalog: only products that currently exist can be
        # matched or trained, and stale gallery data (a product deleted earlier, an import that
        # outlived its product) is quarantined instead of "detecting" a product that is gone.
        _bind = getattr(recognizer, "set_active_provider", None)
        if callable(_bind):
            _bind(catalog.product_keys)
            try:
                recognizer.prune_inactive()
            except Exception:
                logger.exception("could not prune stale product gallery entries")
        self.db_path = db_path
        self.started_at = time.time()
        # Optional — see core/vision.py's CameraWorker: when set, person
        # detection boxes are colored red/blue by predicted gender instead
        # of a flat color. None (the default) keeps today's behavior.
        self.gender_age_backend = gender_age_backend
        # Global-Person-scoped age/gender attribute analysis (core/attributes.py
        # — MongDee_Master_Prompt_FairFace_Age_Gender.md). Separate from
        # gender_age_backend above: that one drives the existing per-frame
        # on-screen box color/label via core.vision.CameraWorker
        # ._classify_person(), smoothed only within one local track's
        # lifetime. This one is keyed by Global Person ID (core.reid),
        # throttled independently, and feeds the Dashboard + tripwire
        # attribute snapshots. None (the default) when no --fairface-
        # checkpoint/--yunet-model pair was given — analysis is then simply
        # never run, exactly like gender_age_backend's own opt-in pattern.
        self.attribute_backend = attribute_backend
        # Optional full-frame face detector shared by every camera's AI pass (see
        # core.vision.CameraWorker's face_detector). All workers share one AIWorker thread, so one
        # detector instance is never used concurrently.
        self.face_detector = face_detector
        # Face IDENTITY (SFace embedding): the one cue that reliably tells two people apart / one person across two
        # cameras. Loaded automatically when models/face_recognizer/*.onnx exists (python tools/get_face_models.py);
        # without it Re-ID falls back to body appearance with strict cross-camera rules (core.reid).
        self.face_identity = face_identity if face_identity is not None else self._load_face_identity()

        self.aggregator = DetectionAggregator()
        self.interest_tracker = ProductInterestTracker()
        self.camera_ids = [cid for cid, _ in camera_devices]
        self.camera_devices = dict(camera_devices)

        self._lock = threading.Lock()
        self.camera_status: dict[str, dict] = {
            cid: {"status": "unknown", "message": ""} for cid in self.camera_ids
        }
        # Recovery observability (spec: "recovery latency must be measured", not just claimed) --
        # keyed by camera_id, read/written only from _on_status under self._lock.
        self._camera_failure_started: dict[str, float] = {}   # monotonic time the camera last left "online"
        self._camera_reconnect_count: dict[str, int] = {}     # completed offline/reconnecting -> online cycles
        self._camera_last_error: dict[str, str] = {}          # most recent offline/error message
        self._camera_last_recovery_latency_sec: dict[str, float] = {}
        self._latest_jpeg: dict[str, bytes] = {}
        self._camera_enabled: dict[str, bool] = {cid: True for cid in self.camera_ids}
        self._person_tracks: dict[str, list[dict]] = {cid: [] for cid in self.camera_ids}
        # Latest product detections per camera (see _on_detections) — kept
        # only for the live per-camera snapshot panel (get_camera_snapshot,
        # the popout camera view's right-side detail panel); never UNKNOWN
        # objects, since core.vision.CameraWorker._run_custom_recognition
        # only ever appends a detection here once it's actually matched to a
        # trained/catalog product (an UNKNOWN blob is drawn on-screen but
        # never added to this list).
        self._product_detections: dict[str, list[dict]] = {cid: [] for cid in self.camera_ids}
        # (camera_id, class_name, hold_start_ts) -> product_hold_events.id —
        # bridges core.interest_tracker's "confirmed" and "finalized" events
        # for the same interaction to the one DB row _on_interest_confirmed
        # opened for it (spec section 23/24: release finalizes, never opens
        # a second row). In-memory only, same lifetime as interest_tracker's
        # own state — a process restart mid-hold simply drops it, same as
        # interest_tracker's own state would.
        self._open_interest_rows: dict[tuple[str, str, float], int] = {}
        self.current_product: dict | None = None
        self.product_seq = 0
        self.recent_alerts: list[dict] = []
        self.import_progress: dict[str, dict] = {}
        self._live_training_sessions: dict[str, "training.LiveTrainingSession"] = {}

        self._next_camera_num = len(self.camera_ids) + 1
        # USB device indices the operator explicitly removed via /settings
        # (see remove_camera) — the Hot-Plug Scan (below) never re-adds one
        # of these on its own just because the same physical camera is still
        # plugged in; add_camera() (a deliberate re-add) clears an index back
        # out of this set.
        self._ignored_usb_indices: set[int] = set()

        # Performance Monitor + Adaptive Controller (see
        # MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md sections 8-10,
        # 29-34): every CameraWorker reports capture/drop/display/AI events
        # into self.performance_monitor; the AdaptiveController ticks in the
        # background and backs off a camera's own AI workload (inference
        # rate, then AI input resolution) when *this process's own
        # measurements* say it's overloaded — never a static hardware
        # guess. core.vision.CameraWorker implements the get/set methods
        # the controller needs directly, so self.workers (a plain dict) can
        # be handed to it as-is; adding/removing a camera later is enough to
        # keep the controller in sync since it holds this same dict.
        self.hardware_profile = detect_hardware()
        logger.info(self.hardware_profile.summary_line())
        self.performance_monitor = PerformanceMonitor()
        self._adaptive_config = adaptive_config or AdaptiveConfig()
        # One shared AI thread for every camera this booth owns (see
        # core.vision.AIWorker) — lives for the whole process, independent of
        # start()/stop(), since it does nothing on its own when no camera is
        # registered. This is what makes adding a camera share a fixed AI
        # budget instead of adding its own competing inference thread.
        self.ai_worker = AIWorker()
        self.ai_worker.start()
        self._ai_paused = False

        # Multi-Camera Person Re-ID (core/reid.py) — layered on top of, not
        # instead of, the per-camera PersonTracker: local track IDs stay
        # exactly as ephemeral/camera-scoped as before. Never allowed to
        # crash booth startup: a failed embedder load (e.g. a very
        # low-memory device) degrades to "no Re-ID" (unique-people counting
        # falls back to raw visible-track sums) rather than taking the whole
        # booth process down over an enhancement feature.
        # Must be created before the worker-creation loop below, since
        # _make_worker() reads self.reid_registry.get_confirmed_global_id_for.
        # require_confirmation: a one-blink false detection is never counted as a visitor; stitching/merging
        # stop a flickering box or a churned track from becoming a second (or third) ID for the same person.
        self.reid_registry = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
        self._reid_sampler = ReIDSampler()
        self._track_face_votes: dict = {}                  # (camera, local track) -> {"male": n, "female": n}
        try:
            self.reid_embedder: PersonReIDEmbedder | None = PersonReIDEmbedder(
                device=self.model_device, cache_dir=self.db_path.parent)
        except Exception as exc:
            logger.warning("PersonReIDEmbedder unavailable (%s) — Re-ID disabled, "
                            "falling back to per-camera visible-track counts", exc)
            self.reid_embedder = None

        # Global-Person-scoped attribute smoothing/throttle (core/attributes.py)
        # — cheap, model-free objects always created; actual FairFace inference
        # only ever runs when self.attribute_backend is configured (see
        # _update_attributes below). Must be created before the
        # worker-creation loop below, same reason as reid_registry above:
        # _make_worker() reads self.attribute_smoother.get.
        self.attribute_smoother = GlobalPersonAttributeSmoother()
        self._attribute_sampler = AttributeSampler()
        # Whole-person gender evidence for people too far away for a readable face (core/body_gender.py). It
        # learns on this booth from people the face model has already decided, and is saved next to the database.
        # With MONGDEE_CLIP_GENDER=1 (and open_clip installed) the whole-body opinion comes from a CLIP model with a
        # zero-shot prior instead - see core/clip_gender.py; otherwise the hand-crafted cues below are used.
        self._clip_backend = None
        try:
            clip_device = "cuda" if str(self.model_device).lower().startswith(("cuda", "0")) else "cpu"
            self._clip_backend = try_load_clip_backend(clip_device)
        except Exception:
            logger.exception("CLIP body-gender backend failed to start; using the hand-crafted body model")
        if self._clip_backend is not None:
            self._body_model_path = self.db_path.parent / "body_gender_clip_model.npz"
            self.body_model = ClipBodyGenderModel.load(self._body_model_path, self._clip_backend.embedding_dim)
        else:
            self._body_model_path = self.db_path.parent / "body_gender_model.npz"
            self.body_model = BodyGenderModel.load(self._body_model_path, BODY_FEATURE_DIM)
        self.body_learner = BodyGenderLearner(self.body_model, self._body_model_path)

        self.workers: dict[str, CameraWorker] = {}
        for camera_id, device in camera_devices:
            worker = self._make_worker(camera_id, device)
            self.workers[camera_id] = worker
        self.adaptive_controller: AdaptiveController | None = None

        # Virtual Tripwire — one TripwireCounter per camera (core/tripwire.py),
        # loaded from whatever was last saved to the `tripwires` table so a
        # configured line survives a process restart (per the spec: "ต้อง
        # บันทึก configuration ลง database และโหลดกลับอัตโนมัติเมื่อเปิดระบบใหม่").
        self._tripwire_counters: dict[str, TripwireCounter] = {}
        for camera_id in self.camera_ids:
            self._load_tripwire(camera_id)

        # Booth Session / Dwell Time (core/booth_sessions.py — spec:
        # MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md sections
        # 29-30): one continuous ENTRY->EXIT visit per identity, spanning
        # however many cameras that identity is seen at in between. Keyed
        # by whatever _session_key_for() resolves for a crossing's track —
        # see that method's docstring for the Re-ID-enabled vs. fallback
        # cases. In-memory only, same as reid_registry; web/booth_manager.py
        # (here) is what persists opens/closes to the `booth_sessions` table.
        self.session_manager = BoothSessionManager()

        # Camera workers / the heartbeat thread / the adaptive controller are
        # each threading.Thread subclasses — a Thread can only ever be
        # start()ed once, so the Schedule (below) recreates fresh instances
        # of all three on every start() rather than reusing the ones from
        # __init__ or a previous run.
        self._running = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

        # Booth Open/Close Schedule (open_time/close_time on the `booths`
        # registry row, HH:MM local time; both NULL = no schedule, booth
        # runs continuously like before this existed) — a low-frequency
        # watcher that starts/stops the camera pipeline itself so a booth
        # left running unattended overnight doesn't keep capturing/logging
        # data outside its own opening hours. Started separately via
        # start_scheduled() (see web_server.py), not by plain start().
        self._schedule_stop = threading.Event()
        self._schedule_thread = threading.Thread(
            target=self._schedule_loop, daemon=True, name="mongdee-schedule")

        # Hot-Plug Scan — see HOTPLUG_SCAN_INTERVAL_SEC. Only runs while the
        # booth is actually running (started/stopped alongside the heartbeat
        # thread in start()/stop()), same one-shot-Thread lifecycle as those.
        self._hotplug_stop = threading.Event()
        self._hotplug_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        # Watches Windows' camera list (no camera is opened) so an unplug is noticed in about a second and a
        # replug reconnects at once - see core/device_presence.py. Inert where the list is unavailable.
        self._hotplug_wake = threading.Event()
        self.presence = DevicePresenceMonitor()
        self.presence.add_listener(self._on_presence_event)

    def _product_display_name(self, key: str) -> "str | None":
        """Catalog name for a product key (what the boxes on the video show instead of the id)."""
        product = self.catalog.get(key)
        return None if product is None else product.get("name")

    def _on_presence_event(self, event: PresenceEvent) -> None:
        """Runs on the presence monitor thread whenever a camera appears in / disappears from the OS list."""
        with self._lock:
            workers = list(self.workers.values())
        if event.removed:
            logger.info("[PRESENCE] กล้องหายไปจากระบบ: %s", ", ".join(event.removed))
            for worker in workers:
                note = getattr(worker, "note_device_change", None)
                if callable(note):
                    note()
        if event.added:
            logger.info("[PRESENCE] พบกล้องเสียบเข้ามา: %s — เชื่อมต่อใหม่ทันที", ", ".join(event.added))
            for worker in workers:
                is_open = getattr(worker, "is_capture_open", None)
                retry = getattr(worker, "request_immediate_retry", None)
                if callable(retry) and callable(is_open) and not is_open():
                    retry()
            wake = getattr(self, "_hotplug_wake", None)
            if wake is not None:
                wake.set()        # also look for a camera we have never seen, right now

    def _make_worker(self, camera_id: str, device) -> CameraWorker:
        worker = self._make_worker_unwired(camera_id, device)
        presence = getattr(self, "presence", None)
        if presence is not None:
            worker.device_present_fn = presence.is_present
        return worker

    def _make_worker_unwired(self, camera_id: str, device) -> CameraWorker:
        return CameraWorker(
            camera_id=camera_id,
            device=device,
            model=self.model,
            allowed_classes=self.catalog.product_keys(),
            recognizer=self.recognizer,
            device_target=self.model_device,
            on_frame=self._on_frame,
            on_detections=self._on_detections,
            on_status=self._on_status,
            on_person_tracks=self._on_person_tracks,
            gender_age_backend=self.gender_age_backend,
            performance_monitor=self.performance_monitor,
            ai_worker=self.ai_worker,
            global_id_resolver=self.reid_registry.get_confirmed_global_id_for,
            global_attribute_resolver=self.attribute_smoother.get,
            gender_resolver=self._resolved_gender_for_track,
            is_product_live=lambda key: self.catalog.get(key) is not None,
            face_detector=getattr(self, "face_detector", None),
            product_name_resolver=self._product_display_name,
        )

    # ------------------------------------------------------------- lifecycle
    def start(self):
        """(Re)starts the camera pipeline. Safe to call more than once — a
        second call while already running is a no-op — and safe to call
        again after stop(), which a manual restart, add/remove-camera-while-
        stopped, or the Schedule all rely on."""
        with self._lock:
            if self._running:
                return
            self._running = True
            presence = getattr(self, "presence", None)
            if presence is not None:
                try:
                    presence.poll_once()      # baseline before any worker exists, so a first unplug is a change
                except Exception:
                    logger.exception("device presence baseline failed")
            for camera_id in self.camera_ids:
                if camera_id not in self.workers:
                    self.workers[camera_id] = self._make_worker(camera_id, self.camera_devices[camera_id])
            self.adaptive_controller = AdaptiveController(
                self.performance_monitor, self.workers, self._adaptive_config,
                on_degraded=self._on_camera_degraded,
                ai_worker=self.ai_worker, on_ai_paused=self._on_ai_paused,
            )
            self._heartbeat_stop = threading.Event()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hotplug_stop = threading.Event()
            self._hotplug_thread = threading.Thread(
                target=self._hotplug_loop, daemon=True, name="mongdee-hotplug")
            self._watchdog_thread = threading.Thread(
                target=self._capture_watchdog_loop, args=(self._heartbeat_stop,), daemon=True,
                name="mongdee-capture-watchdog")
            workers = list(self.workers.values())
            adaptive_controller = self.adaptive_controller
            heartbeat_thread = self._heartbeat_thread
            hotplug_thread = self._hotplug_thread
        # Staggered, not simultaneous -- see CAMERA_STARTUP_STAGGER_SEC's
        # own docstring for why. No stagger after the *last* one (nothing
        # left to protect it from), and none at all for the common single-
        # camera case.
        for i, worker in enumerate(workers):
            worker.start()
            if i < len(workers) - 1:
                time.sleep(CAMERA_STARTUP_STAGGER_SEC)
        heartbeat_thread.start()
        hotplug_thread.start()
        self._watchdog_thread.start()
        if getattr(self, "presence", None) is not None:
            self.presence.start()
        adaptive_controller.start()
        db.log_heartbeat(self.db_path, self.booth_id, self.event_id, "started", len(workers))

    def stop(self):
        """Stops the camera pipeline. Safe to call when already stopped."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            workers = list(self.workers.values())
            self.workers.clear()  # forces start() to build fresh (one-shot) worker threads next time
            adaptive_controller = self.adaptive_controller
            heartbeat_stop = self._heartbeat_stop
            hotplug_stop = self._hotplug_stop
            self._latest_jpeg.clear()
            for camera_id in self.camera_status:
                self.camera_status[camera_id] = {"status": "disabled", "message": "บูธปิดอยู่"}
            self._person_tracks = {cid: [] for cid in self.camera_ids}
        heartbeat_stop.set()
        hotplug_stop.set()
        if getattr(self, "presence", None) is not None:
            self.presence.stop()
        if adaptive_controller is not None:
            adaptive_controller.stop()
        for worker in workers:
            worker.stop()
        db.log_heartbeat(self.db_path, self.booth_id, self.event_id, "stopped", 0)

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    # ------------------------------------------------------- open/close schedule
    @staticmethod
    def _parse_hhmm(value: str | None) -> int | None:
        if not value:
            return None
        try:
            hh, mm = value.split(":")
            return int(hh) * 60 + int(mm)
        except (ValueError, AttributeError):
            return None

    @classmethod
    def is_open_at(cls, open_time: str | None, close_time: str | None, now_minutes: int) -> bool:
        """True if `now_minutes` (minutes since local midnight) falls inside
        [open_time, close_time). Either side blank means no schedule is
        configured, so the booth runs continuously (unchanged default
        behavior). close_time <= open_time is treated as an overnight
        window that wraps past midnight (e.g. 20:00-02:00)."""
        open_min = cls._parse_hhmm(open_time)
        close_min = cls._parse_hhmm(close_time)
        if open_min is None or close_min is None:
            return True
        if open_min == close_min:
            return True  # identical open/close = 24 ชม.
        if open_min < close_min:
            return open_min <= now_minutes < close_min
        return now_minutes >= open_min or now_minutes < close_min

    def start_scheduled(self) -> None:
        """Called once at process startup instead of start(): begins
        watching this booth's open_time/close_time and starts the camera
        pipeline immediately only if that schedule (or the lack of one)
        says the booth should be open right now."""
        self._schedule_thread.start()
        self._apply_schedule()

    def stop_scheduled(self) -> None:
        self._schedule_stop.set()
        self.stop()

    def _schedule_loop(self) -> None:
        while not self._schedule_stop.wait(SCHEDULE_CHECK_INTERVAL_SEC):
            self._apply_schedule()

    def _apply_schedule(self) -> None:
        booth_row = db.get_booth(self.db_path, self.booth_id)
        if not booth_row:
            return
        now = time.localtime()
        should_be_open = self.is_open_at(
            booth_row.get("open_time"), booth_row.get("close_time"), now.tm_hour * 60 + now.tm_min)
        if should_be_open and not self.is_running():
            logger.info("[SCHEDULE] ถึงเวลาเปิดบูธ (%s) — เริ่มกล้องอัตโนมัติ", booth_row.get("open_time") or "-")
            self.start()
        elif not should_be_open and self.is_running():
            logger.info("[SCHEDULE] ถึงเวลาปิดบูธ (%s) — หยุดกล้องอัตโนมัติ", booth_row.get("close_time") or "-")
            self.stop()

    def _capture_watchdog_loop(self, stop_event: threading.Event) -> None:
        """Every CAPTURE_WATCHDOG_INTERVAL_SEC: ask each camera worker whether its cap.read() has been
        blocked too long and let it recover (see CameraWorker.recover_from_hung_read). Runs on its own
        thread because a blocked capture thread cannot rescue itself."""
        while not stop_event.wait(CAPTURE_WATCHDOG_INTERVAL_SEC):
            with self._lock:
                workers = list(self.workers.items())
            for camera_id, worker in workers:
                for name in ("recover_from_hung_read", "check_stream_health"):
                    check = getattr(worker, name, None)
                    if callable(check):
                        try:
                            check()
                        except Exception:
                            logger.exception("capture watchdog: %s failed", name)
                self._replace_worker_if_dead(camera_id, worker)

    def _replace_worker_if_dead(self, camera_id: str, worker: CameraWorker) -> None:
        """Last-resort safety net for a CameraWorker thread that has exited on its own, outside
        every recovery path above -- recover_from_hung_read() only detects a read still IN
        PROGRESS, and check_stream_health() only fires once, on the online->offline transition,
        never again once a camera is already reported offline. Neither can rescue a thread that
        has simply died: an uncaught exception anywhere in CameraWorker.run() (cap.read() itself,
        _process_captured_frame() -- on_frame()/box-drawing/frame-slot-publish, none of which used
        to have their own guard -- or _attempt_reopen()) used to kill the whole thread silently,
        leaving the camera permanently stuck showing "ภาพจากกล้องหยุดหรือเสียหาย — กำลังเชื่อมต่อใหม่"
        with frame_sequence/captured_frames frozen and reconnect_count never advancing, exactly
        the live symptom this closes. core/vision.py's own exception handling around those three
        call sites should now make a dead thread rare, not impossible -- this is the backstop for
        whatever still isn't covered, not a replacement for fixing the actual exception sites.

        `worker._running` is only ever True once run() has actually started (see CameraWorker
        .__init__, where it starts False) and is set back to False by run()'s own normal-exit
        cleanup before that thread returns -- so `_running and not is_alive()` never true-positives
        on a worker that simply hasn't been .start()ed yet (staggered startup) or one that exited
        the ordinary way. `_escalate_to_process` is excluded because that path deliberately
        abandons this exact thread while a separate pump thread keeps serving the camera (see
        CameraWorker.run()'s own top-of-loop comment) -- an intentional "dead" original thread,
        not a failure."""
        if not (getattr(worker, "_running", False) and not getattr(worker, "_escalate_to_process", False)
                and not worker.is_alive()):
            return
        with self._lock:
            if not self._running or self.workers.get(camera_id) is not worker:
                return  # booth stopped, or already replaced/removed since the check above
            device = self.camera_devices.get(camera_id)
            if device is None:
                return
            logger.error(
                "[%s] capture watchdog: worker thread exited unexpectedly (not escalated, not "
                "stopped) -- replacing it with a fresh worker so this camera can recover instead "
                "of staying stuck offline forever", camera_id,
            )
            new_worker = self._make_worker(camera_id, device)
            self.workers[camera_id] = new_worker
        new_worker.start()

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SEC):
            with self._lock:
                active = sum(1 for s in self.camera_status.values() if s["status"] == "online")
            overall = "ok" if active > 0 else "degraded"
            db.log_heartbeat(self.db_path, self.booth_id, self.event_id, overall, active)

    def _wait_hotplug_tick(self, interval: float) -> tuple[bool, bool]:
        """Sleep up to `interval`; returns (stopped, woken). `woken` = the presence monitor saw a camera
        appear, so scan for new cameras now instead of waiting out the (backed-off) interval."""
        wake = getattr(self, "_hotplug_wake", None)
        if wake is None:
            return self._hotplug_stop.wait(interval), False
        end = time.monotonic() + interval
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False, False
            if self._hotplug_stop.wait(min(0.25, remaining)):
                return True, False
            if wake.is_set():
                wake.clear()
                return False, True

    def _hotplug_loop(self) -> None:
        """Periodically probes for a USB webcam that appeared after startup
        (freshly plugged in, or a replug that Windows/Linux re-enumerated to
        a different index than the one an existing camera already owns) and
        registers it the same way a manual /settings "add camera" would, so
        it shows up without restarting the process.

        A camera index that a worker currently holds a *live* capture on is
        never touched here — probing it too could disrupt that worker's own
        connection on some backends (see discover_cameras'/_open_capture's
        own comments) — that camera keeps managing its own reconnect via
        CameraWorker._open() on its own schedule. But an already-known
        camera that's currently disconnected (its worker released its
        capture and is waiting out its own reopen backoff — see
        REOPEN_BACKOFF_MAX_SEC) is safe to also probe here, since nothing
        else has that index open right now — and if this scan finds it
        reachable again, it nudges that worker to retry immediately (see
        CameraWorker.request_immediate_retry) instead of leaving it to wait
        out however large its own backoff grew from repeated failures.
        Without this, a camera unplugged long enough for its backoff to
        reach REOPEN_BACKOFF_MAX_SEC could sit disconnected for up to that
        long after being physically replugged, before its own timer even
        tries again.

        The scan interval backs off (doubling, capped at
        HOTPLUG_SCAN_MAX_INTERVAL_SEC) every time a scan finds nothing new,
        and resets to the base interval the moment something does change —
        most of a booth's runtime has zero hot-plug activity, so this avoids
        probing on a fixed aggressive cadence forever for no reason. A scan
        is also skipped outright (without touching the backoff) whenever
        system CPU is already under AI-Pause-level pressure — see
        HOTPLUG_SKIP_SCAN_CPU_PCT — since adding probe overhead on top of an
        already-struggling system is exactly what this must never do."""
        interval = HOTPLUG_SCAN_INTERVAL_SEC
        while True:
            stopped, woken = self._wait_hotplug_tick(interval)
            if stopped:
                break
            if woken:
                interval = HOTPLUG_SCAN_INTERVAL_SEC
            cpu_pct = self.performance_monitor.system_snapshot().get("cpu_percent")
            if cpu_pct is not None and cpu_pct >= HOTPLUG_SKIP_SCAN_CPU_PCT:
                continue  # backoff/interval unchanged — try again next tick, system is busy right now

            with self._lock:
                live_indices: set[int] = set()
                reconnect_candidates: dict[int, str] = {}  # index -> camera_id, disconnected right now
                for camera_id, dev in self.camera_devices.items():
                    if not str(dev).isdigit():
                        continue
                    index = int(dev)
                    worker = self.workers.get(camera_id)
                    if worker is not None and worker.is_capture_open():
                        live_indices.add(index)
                    else:
                        reconnect_candidates[index] = camera_id
                skip = live_indices | self._ignored_usb_indices
            # The startup discovery in app.py/web_server.py excludes the
            # built-in/virtual cameras, but this loop runs independently for
            # the rest of the process's life — without re-resolving the same
            # config here, a laptop's integrated webcam that wasn't plugged
            # in (or wasn't yet present) at startup would get silently
            # picked up and added as a real camera the moment the Hot-Plug
            # Scan next runs, exactly defeating "built-in camera is disabled
            # by default". Read + re-resolve fresh each tick (cheap) so a
            # config change, or a re-enumeration that moves the built-in
            # camera to a different index, takes effect without a restart.
            camera_settings = load_camera_settings(booth_settings_path_for(self.db_path))
            skip = skip | compute_camera_skip_indices(camera_settings)
            presence = getattr(self, "presence", None)
            if presence is not None:
                # Where the OS device list is known, a known-but-disconnected camera is never open-probed:
                # if it is listed its own worker reconnects (the presence monitor already nudged it the
                # moment it re-appeared), if it is not listed there is nothing to open. Probing by opening
                # it here only added open/close churn that leaves a USB camera in its cool-down.
                skip = skip | {i for i in reconnect_candidates if presence.is_present(i) is not None}
            try:
                found = discover_cameras(max_index=HOTPLUG_MAX_INDEX, skip=frozenset(skip))
            except Exception:
                logger.exception("hot-plug scan failed")
                continue

            if not found:
                interval = min(interval * 2, HOTPLUG_SCAN_MAX_INTERVAL_SEC)
                continue
            interval = HOTPLUG_SCAN_INTERVAL_SEC

            for index in found:
                if self._hotplug_stop.is_set():
                    return
                existing_camera_id = reconnect_candidates.get(index)
                if existing_camera_id is not None:
                    worker = self.workers.get(existing_camera_id)
                    if worker is not None:
                        logger.info(
                            "[HOTPLUG] %s (index %d) กลับมาแล้ว — เร่งให้เชื่อมต่อใหม่ทันที",
                            existing_camera_id, index,
                        )
                        worker.request_immediate_retry()
                    continue
                logger.info("[HOTPLUG] พบกล้อง USB ใหม่ที่ index %d — เพิ่มเข้าบูธอัตโนมัติ", index)
                try:
                    self.add_camera(index)
                except ValueError as exc:
                    # Already one of ours (same physical camera): its own worker reconnects it.
                    logger.info("[HOTPLUG] index %d ไม่ถูกเพิ่ม: %s", index, exc)
                except Exception:
                    logger.exception("hot-plug: failed to add camera at index %d", index)
                # Same reasoning as start()'s own stagger (see
                # CAMERA_STARTUP_STAGGER_SEC) -- several new USB cameras
                # plugged in between scans would otherwise all have their
                # worker threads started back-to-back here too.
                if index != found[-1]:
                    time.sleep(CAMERA_STARTUP_STAGGER_SEC)

    # ----------------------------------------------------------- callbacks
    def _on_frame(self, camera_id, frame):
        if frame.shape[1] > DISPLAY_MAX_WIDTH:
            # A large capture size (MONGDEE_CAPTURE_SIZE) is for the AI; the browser stream does not need it.
            scale = DISPLAY_MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self._lock:
                self._latest_jpeg[camera_id] = buf.tobytes()
        else:
            # CameraWorker already validated this frame (non-empty, read()
            # succeeded) before handing it here, so an encode failure means
            # something about this camera's frames confuses cv2.imencode
            # (unusual shape/dtype from a particular driver) — the camera
            # would otherwise report "online" while its tile never shows an
            # image, with nothing in the logs to explain why. Surface it.
            logger.warning(
                "camera %s: frame captured but JPEG encode failed (shape=%s, dtype=%s) — "
                "this camera's tile will stay blank until this is resolved",
                camera_id, getattr(frame, "shape", None), getattr(frame, "dtype", None),
            )

    def _on_detections(self, camera_id, detections):
        # Last line of defence against "ghost" products: whatever produced a detection, a product
        # that is not in the catalog right now is not a product (it must never reach the
        # aggregator, the interest tracker, the per-camera snapshot or the database).
        detections = [d for d in detections if self.catalog.get(d.get("class_name")) is not None]
        with self._lock:
            person_tracks = self._person_tracks.get(camera_id, [])
            self._product_detections[camera_id] = detections

        event = self.aggregator.update(camera_id, detections)
        if event:
            self._handle_product_recognized(event)
        elif self.aggregator.current_product is None and getattr(self, "current_product", None) is not None:
            # The aggregator just timed out its own confirmed product (no camera has
            # sighted it for IDLE_RESET_SEC -- see core/aggregator.py's _decide()) but
            # only ever tells us about a NEW confirmation via `event`, never about
            # going idle. Without this, self.current_product (and therefore
            # /api/state's current_product the UI polls) stayed permanently stuck on
            # the last product ever seen, even minutes after it was removed from
            # every camera -- a real stale-detection bug, not just a UI nicety: an
            # empty booth could show a product on screen indefinitely.
            self._clear_current_product()

        for hold in self.interest_tracker.update(camera_id, detections, person_tracks):
            if hold["event"] == "confirmed":
                self._on_interest_confirmed(camera_id, hold)
            else:
                self._on_interest_finalized(camera_id, hold)

    def _on_interest_confirmed(self, camera_id: str, hold: dict) -> None:
        """Spec: MongDee person-gender/age master continuation prompt
        section 20 — writes the Interest Event row the instant a hold
        clears INTEREST_CONFIRMATION_SECONDS, not deferred to release.
        gender/age/global_person_id are snapshotted once, right here (spec
        section 26), and _on_interest_finalized below never touches them
        again."""
        class_name = hold["class_name"]
        holder_track_id = hold["holder_track_id"]
        product = self.catalog.get(class_name)
        if product is None:
            return  # deleted while being held: never open an interest row for a ghost product
        # global_person_id is recorded whenever Re-ID has resolved this
        # holder at all (spec section 34: Unique Person vs Interest Events
        # needs it even before an attribute profile exists); gender/age
        # only come along once _attribute_snapshot_for has an actual "ok"
        # profile to snapshot (spec sections 26/27) — never guessed just
        # because an identity is known, and race is never part of either
        # (spec sections 5-7: FairFace's race output is discarded before it
        # ever reaches core.attributes — see that module's own docstring).
        global_id = self.reid_registry.get_global_id_for(camera_id, holder_track_id)
        snapshot = self._attribute_snapshot_for(camera_id, holder_track_id)
        row_id = db.open_product_interest_event(
            self.db_path, self.booth_id, self.event_id, camera_id, class_name,
            product["name"] if product else class_name, holder_track_id,
            hold["hold_start_ts"], hold["confirmed_at"],
            global_person_id=global_id,
            gender=snapshot["gender"] if snapshot else None,
            gender_confidence=snapshot["gender_confidence"] if snapshot else None,
            age_category=snapshot["age_category"] if snapshot else None,
            age_confidence=snapshot["age_confidence"] if snapshot else None,
        )
        with self._lock:
            self._open_interest_rows[(camera_id, class_name, hold["hold_start_ts"])] = row_id

    def _on_interest_finalized(self, camera_id: str, hold: dict) -> None:
        """Spec section 23: release/hand-off/disappearance finalizes the
        *same* interaction this camera+class+hold_start_ts already opened
        in _on_interest_confirmed — never a second DB row. If for some
        reason no open row is on record (e.g. process restarted mid-hold —
        core.interest_tracker's state is in-memory only), there is nothing
        to finalize and this is a silent no-op rather than inventing a row."""
        key = (camera_id, hold["class_name"], hold["hold_start_ts"])
        with self._lock:
            row_id = self._open_interest_rows.pop(key, None)
        if row_id is not None:
            db.finalize_product_interest_event(
                self.db_path, row_id, hold["hold_end_ts"], hold["duration_sec"])

    def _on_person_tracks(self, camera_id, visible_tracks, evicted_tracks, frame_width=0, frame_height=0,
                           frame=None):
        with self._lock:
            self._person_tracks[camera_id] = visible_tracks
        for track in evicted_tracks:
            duration = track["last_seen"] - track["first_seen"]
            if duration >= MIN_PRESENCE_DURATION_SEC:
                db.log_presence_session(self.db_path, self.booth_id, self.event_id, camera_id,
                                         track["track_id"], track["first_seen"], track["last_seen"],
                                         duration, category=track.get("category"))
            self.reid_registry.forget_local_track(camera_id, track["track_id"])
            self._reid_sampler.forget(camera_id, track["track_id"])
            self._face_votes().pop((camera_id, track["track_id"]), None)
        self.reid_registry.observe_frame(camera_id, [t["track_id"] for t in visible_tracks], time.time())
        self._update_tripwire(camera_id, visible_tracks, evicted_tracks, frame_width, frame_height)
        if frame is not None:
            self._update_reid(camera_id, visible_tracks, frame_width, frame_height, frame)

    def _update_reid(self, camera_id, visible_tracks, frame_width, frame_height, frame):
        """Throttled Multi-Camera Person Re-ID (spec sections 18-30): for
        each track that clears ReIDSampler's quality gate, crop its person
        box, embed it, and resolve it against every other camera's own
        GlobalIdentityRegistry so the Dashboard's Unique People count never
        just double-counts one visitor once per camera. Runs on the same
        AI-worker thread run_ai_pass() already runs on — cheap by design
        (the sampler skips almost every call; see REID_SAMPLE_INTERVAL_SEC),
        so this never meaningfully adds latency to a normal AI pass."""
        if self.reid_embedder is None:
            return
        now = time.time()
        for track in visible_tracks:
            if not self._reid_sampler.should_embed(camera_id, track, frame_height, now):
                # Not this track's Re-ID turn - but a person who already has an identity and still no gender
                # label is analysed on its own, faster clock (see AttributeSampler.should_analyze's `urgent`).
                self._analyze_known_person(camera_id, track, frame, frame_width, frame_height, now)
                continue
            crop = self._crop_track(frame, track, frame_width, frame_height)
            if crop is None:
                continue
            face_embedding = None
            face_identity = getattr(self, "face_identity", None)
            if face_identity is not None:
                face_embedding = face_identity.embed(crop)
            try:
                hair = hair_descriptor(crop)
            except Exception:
                hair = None
            try:
                embedding = self.reid_embedder.embed(crop)
            except Exception as exc:
                logger.warning("Re-ID embedding failed for %s track %s (%s)",
                                camera_id, track["track_id"], exc)
                continue
            global_id, _is_new = self.reid_registry.resolve(
                camera_id, track["track_id"], embedding, now, bbox=track["bbox"],
                gender=self._track_face_gender(camera_id, track["track_id"]), face=face_embedding, hair=hair)
            self._update_attributes(global_id, track, frame_height, crop, now, camera_id=camera_id)
            self._maybe_count_customer(global_id, now)
            self._note_session_camera(global_id, camera_id)
        self._apply_reid_merges()
        self._apply_reid_splits()

    def _apply_reid_splits(self) -> None:
        """A track that had been attached to somebody else's identity was split off: forget the gender evidence that
        identity collected (it may contain the other person's face) so it is re-learned from clean samples."""
        for gid in self.reid_registry.pop_splits():
            logger.info("Re-ID split a wrongly attached track off identity %s; re-learning its gender", gid)
            try:
                self.attribute_smoother.forget(gid)
                self._attribute_sampler.forget(gid)
                self.reid_registry.set_person_gender(gid, None)
            except Exception as exc:
                logger.warning("could not reset attributes of %s after a split (%s)", gid, exc)

    @staticmethod
    def _crop_track(frame, track, frame_width, frame_height):
        x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _analyze_known_person(self, camera_id, track, frame, frame_width, frame_height, now) -> None:
        if self.attribute_backend is None:
            return
        global_id = self.reid_registry.get_global_id_for(camera_id, track["track_id"])
        if not global_id or self.attribute_smoother.get(global_id, now).status == "ok":
            return
        if not self._attribute_sampler.should_analyze(global_id, track, frame_height, now, urgent=True):
            return
        crop = self._crop_track(frame, track, frame_width, frame_height)
        if crop is not None:
            self._update_attributes(global_id, track, frame_height, crop, now, gate_checked=True, camera_id=camera_id)
            self._maybe_count_customer(global_id, now)

    def _maybe_count_customer(self, global_id, now) -> None:
        """Persists a Global Person as a counted visitor (the `global_persons` row behind
        db.query_unique_people_count / the Dashboard's Unique People total) only once:
          * the identity itself is confirmed (core.reid.GlobalIdentityRegistry.is_counted —
            anti-flicker: repeated observations over a real span, never a one-blink ghost), AND
          * when this booth runs gender classification at all (self.attribute_backend is
            configured), that person's gender has reached confident, stable evidence
            (core.attributes.GlobalPersonAttributeSmoother status == "ok").
        A person who is still UNKNOWN — no readable evidence yet, or evidence too weak/mixed —
        is tracked and matched internally under their provisional global_id exactly as before
        (Re-ID needs a stable key to accumulate embeddings against), but is never written as a
        counted customer, and so never inflates the Unique People / gender dashboards, until
        their gender is confidently decided. The moment the smoother's status flips to "ok" this
        same call (reached again next frame, from either the Re-ID or the urgent-attribute path)
        writes the row under the SAME global_id already in use — no re-identification, no new ID.
        Booths with no gender backend configured have nothing to wait for, so they keep counting
        by identity confirmation alone, exactly as before this gate was added."""
        person = self.reid_registry.get_person(global_id)
        if person is None or not self.reid_registry.is_counted(global_id):
            return
        if self.attribute_backend is not None and self.attribute_smoother.get(global_id, now).status != "ok":
            return
        db.upsert_global_person(self.db_path, self.booth_id, self.event_id, global_id,
                                 person.first_seen, person.last_seen, len(person.cameras))

    def _apply_reid_merges(self) -> None:
        """Re-ID found that two identities were one person: point every row/state that used the duplicate at the
        surviving identity (database rows, gender evidence, body-model bookkeeping, open booth sessions)."""
        for loser, winner in self.reid_registry.pop_merges():
            logger.info("Re-ID merged duplicate identity %s into %s", loser, winner)
            try:
                db.merge_global_person(self.db_path, self.booth_id, loser, winner)
                self.attribute_smoother.merge(loser, winner)
                self.body_learner.merge(loser, winner)
                self._attribute_sampler.forget(loser)
                dropped = self.session_manager.rekey(loser, winner)
                if dropped is not None:
                    db.close_booth_session(self.db_path, self.booth_id, dropped.session_id, time.time(),
                                           time.time() - dropped.entered_at, dropped.cameras_seen)
            except Exception as exc:
                logger.warning("could not apply Re-ID merge %s -> %s (%s)", loser, winner, exc)

    def _note_session_camera(self, person_key: str, camera_id: str) -> None:
        """Spec section 21 ("person switches camera mid-visit"): tells the
        open BoothSession (if any) for this identity that it was also just
        observed at `camera_id`, even though no tripwire fired here — only
        writes to the DB when that's actually a *new* camera for this
        session (note_camera_seen's return value), not on every frame."""
        session = self.session_manager.get_open_session(person_key)
        if session is None:
            return
        if self.session_manager.note_camera_seen(person_key, camera_id):
            db.update_booth_session_cameras(self.db_path, self.booth_id, session.session_id,
                                             session.cameras_seen)

    def _note_track_face(self, camera_id, track_id, gender, confidence, face_px) -> None:
        """A decisive FACE reading of this one track (not of the identity it is attached to). Used only to notice that a
        track was attached to a person of the other gender - see GlobalIdentityRegistry.resolve(gender=...)."""
        if camera_id is None or gender not in ("male", "female"):
            return
        if confidence < TRACK_FACE_MIN_CONFIDENCE or (face_px is not None and face_px < TRACK_FACE_MIN_PX):
            return
        votes = self._face_votes().setdefault((camera_id, track_id), {"male": 0, "female": 0})
        votes[gender] += 1

    def _face_votes(self) -> dict:
        return self.__dict__.setdefault("_track_face_votes", {})

    def _track_face_gender(self, camera_id, track_id) -> "str | None":
        votes = self._face_votes().get((camera_id, track_id))
        if not votes:
            return None
        for g, other in (("male", "female"), ("female", "male")):
            if votes[g] >= TRACK_FACE_MIN_VOTES and votes[g] >= 3 * votes[other]:
                return g
        return None

    def _update_attributes(self, global_id, track, frame_height, crop, now, gate_checked=False, camera_id=None):
        """Global-Person-scoped FairFace age/gender analysis (spec:
        MongDee_Master_Prompt_FairFace_Age_Gender.md sections 7/8/9/12) —
        only ever runs when --fairface-checkpoint/--yunet-model were given
        (self.attribute_backend). Reuses the same person crop _update_reid
        already cropped for embedding; FairFaceBackend runs its own face
        detection on it internally (never a full-body prediction — spec
        section 4), and AttributeSampler throttles by global_id, not by
        camera/local-track, so two cameras (or a churned local track ID)
        resolving to the same Global Person share one analysis budget."""
        if self.attribute_backend is None:
            return
        if not gate_checked:
            urgent = self.attribute_smoother.get(global_id, now).status != "ok"
            if not self._attribute_sampler.should_analyze(global_id, track, frame_height, now, urgent=urgent):
                return
        face_px = None
        try:
            detail_fn = getattr(self.attribute_backend, "predict_gender_detail", None)
            if callable(detail_fn):
                detail = detail_fn(crop)
                if detail is None:
                    gender, gender_conf = "unknown", 0.0
                else:
                    gender, gender_conf, face_px = detail["gender"], detail["confidence"], detail.get("face_px")
            else:
                gender, gender_conf = self.attribute_backend.predict_gender(crop)
            age_group, age_conf = "unknown", 0.0  # age is not analysed: only male / female are reported
        except Exception as exc:
            logger.warning("FairFace attribute analysis failed for global_id %s (%s)", global_id, exc)
            return
        self._note_track_face(camera_id, track["track_id"], gender, gender_conf, face_px)
        result = self.attribute_smoother.add_sample(global_id, gender, gender_conf, age_group, age_conf, now,
                                                    face_px=face_px)
        result = self._fuse_body_evidence(global_id, crop, now) or result
        decided = str(result.gender).lower()
        self.reid_registry.set_person_gender(global_id, decided if decided in ("male", "female") else None)
        person = self.reid_registry.get_person(global_id)
        if person is None or not self.reid_registry.is_counted(global_id):
            return      # a still-unconfirmed identity keeps its evidence in memory but is not a row yet
        first_seen = person.first_seen
        db.upsert_person_attributes(
            self.db_path, self.booth_id, self.event_id, global_id,
            result.gender, result.gender_confidence, result.age_group, result.age_category,
            result.age_confidence, result.sample_count, first_seen, now,
        )

    def _fuse_body_evidence(self, global_id, crop, now):
        """Whole-person gender evidence (hair, clothing, build): teaches the body model from people whose face has
        already decided, and - once the model has proven itself on unseen people - adds its opinion for people
        too far away for a readable face. Never raises: it is an add-on to the face analysis."""
        learner = getattr(self, "body_learner", None)
        if learner is None:
            return None
        try:
            clip_backend = getattr(self, "_clip_backend", None)
            features = clip_backend.features(crop) if clip_backend is not None else body_features(crop)
            if features is None:
                return None
            learner.observe(global_id, features, self.attribute_smoother.face_label(global_id))
            prediction = learner.model.predict(features)
            if prediction is None:
                return None
            p_male, weight = prediction
            return self.attribute_smoother.add_body_sample(global_id, p_male, weight, now)
        except Exception as exc:
            logger.warning("body gender evidence failed for %s (%s)", global_id, exc)
            return None

    def _update_tripwire(self, camera_id, visible_tracks, evicted_tracks, frame_width, frame_height):
        """Feeds this camera's TripwireCounter (if any line is configured)
        and logs every crossing it fires. Runs on the same AI-worker thread
        run_ai_pass()/on_person_tracks already run on (see core/vision.py) —
        cheap by design (TripwireCounter.update() never touches camera I/O
        or the database itself), so this never adds latency to that pass."""
        counter = self._tripwire_counters.get(camera_id)
        if counter is None or counter.get_line() is None:
            return
        events = counter.update(visible_tracks, evicted_tracks, frame_width, frame_height, time.time())
        line = counter.get_line()
        for event in events:
            person_key, global_person_id = self._session_key_for(camera_id, event["track_id"])
            db.log_tripwire_crossing(
                self.db_path, self.booth_id, self.event_id, camera_id, line.id,
                event["track_id"], event["direction"], ts=event["ts"],
                attribute_snapshot=self._attribute_snapshot_for(camera_id, event["track_id"]),
                global_person_id=global_person_id,
            )
            self._handle_session_crossing(person_key, global_person_id, camera_id,
                                           event["direction"], event["ts"])
            self._push_alert(
                "tripwire_" + event["direction"],
                f"{camera_id}: มีคน{'เข้า' if event['direction'] == 'in' else 'ออก'}บูธ "
                f"(IN {counter.count_in} / OUT {counter.count_out})",
            )
        if events:
            worker = self.workers.get(camera_id)
            if worker is not None:
                worker.set_tripwire_overlay(self._overlay_dict(line, counter))

    def _session_key_for(self, camera_id: str, track_id: int) -> tuple[str, str | None]:
        """Returns (person_key, global_person_id) for a tripwire crossing's
        track. person_key is what core.booth_sessions.BoothSessionManager
        keys sessions by; global_person_id is the same value only when
        Re-ID actually resolved it for this track (else None) — both are
        stored on the session/crossing rows so a query can tell a true
        cross-camera visit apart from a Re-ID-less fallback one (see
        core/database.py's open_booth_session docstring). Falls back to a
        synthetic per-camera key when Re-ID is disabled or hasn't resolved
        this track yet, so dwell tracking still works (without cross-camera
        merging) instead of silently doing nothing."""
        global_id = self.reid_registry.get_global_id_for(camera_id, track_id)
        if global_id is not None:
            return global_id, global_id
        return f"local:{camera_id}:{track_id}", None

    def _handle_session_crossing(self, person_key: str, global_person_id: str | None,
                                  camera_id: str, direction: str, ts: float) -> None:
        """Turns one debounced tripwire ENTRY/EXIT into a BoothSession
        open/close (spec sections 19-21 and 29-30) — see
        core.booth_sessions.BoothSessionManager for the duplicate-ENTRY /
        nothing-open-to-EXIT no-op rules that prevent double-counting."""
        if direction == DIRECTION_IN:
            session = self.session_manager.on_entry(person_key, camera_id, ts)
            if session is not None:
                db.open_booth_session(self.db_path, self.booth_id, self.event_id, session.session_id,
                                       person_key, global_person_id, session.entered_at, session.cameras_seen)
        else:
            session = self.session_manager.on_exit(person_key, camera_id, ts)
            if session is not None:
                db.close_booth_session(self.db_path, self.booth_id, session.session_id,
                                        session.exited_at, session.dwell_seconds, session.cameras_seen)

    def _attribute_snapshot_for(self, camera_id, track_id) -> dict | None:
        """Spec section 13: a crossing event MAY carry a read-only copy of
        whatever the Global Person's attribute profile already was at that
        moment — never a fresh inference, never a second crossing/count.
        None whenever Re-ID isn't resolved for this track yet, or nothing
        has been analyzed for that Global Person yet (attribute_backend not
        configured, or AttributeSampler hasn't cleared this person for
        analysis yet) — that's an honest "no attribute data available",
        not a fabricated guess."""
        global_id = self.reid_registry.get_global_id_for(camera_id, track_id)
        if global_id is None:
            return None
        result = self.attribute_smoother.get(global_id)
        if result.status != "ok":
            return None
        return {
            "global_person_id": global_id,
            "gender": result.gender,
            "gender_confidence": result.gender_confidence,
            "age_group": result.age_group,
            "age_category": result.age_category,
            "age_confidence": result.age_confidence,
        }

    @staticmethod
    def _overlay_dict(line: TripwireLine, counter: TripwireCounter) -> dict:
        return {
            "x1": line.x1, "y1": line.y1, "x2": line.x2, "y2": line.y2,
            "inside_side": line.inside_side,
            "count_in": counter.count_in, "count_out": counter.count_out,
        }

    def _load_tripwire(self, camera_id: str) -> None:
        """(Re)creates this camera's TripwireCounter from whatever is saved
        in the `tripwires` table, and pushes the line (with fresh-start 0/0
        counts) to its CameraWorker so the overlay is visible immediately —
        called at startup for every configured camera and again whenever
        add_camera() brings a new one online."""
        row = db.get_tripwire(self.db_path, self.booth_id, camera_id)
        counter = TripwireCounter(TripwireLine.from_dict(row) if row else None)
        self._tripwire_counters[camera_id] = counter
        worker = self.workers.get(camera_id)
        if worker is not None:
            line = counter.get_line()
            worker.set_tripwire_overlay(self._overlay_dict(line, counter) if line and line.enabled else None)

    # ---------------------------------------------------------- tripwire API
    def set_tripwire(self, camera_id: str, x1: float, y1: float, x2: float, y2: float,
                      inside_side: str, enabled: bool = True) -> dict:
        if camera_id not in self.camera_ids:
            raise ValueError(f"ไม่พบกล้อง {camera_id}")
        if inside_side not in ("A", "B"):
            raise ValueError('inside_side ต้องเป็น "A" หรือ "B"')
        for name, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} ต้องอยู่ระหว่าง 0.0-1.0 (normalized coordinates)")
        tripwire_id = f"TW-{camera_id}"
        db.upsert_tripwire(self.db_path, tripwire_id, self.booth_id, camera_id, x1, y1, x2, y2,
                            inside_side, enabled)
        with self._lock:
            self._load_tripwire(camera_id)
        return db.get_tripwire(self.db_path, self.booth_id, camera_id)

    def get_tripwire(self, camera_id: str) -> dict | None:
        return db.get_tripwire(self.db_path, self.booth_id, camera_id)

    def list_tripwires(self) -> list[dict]:
        return db.list_tripwires(self.db_path, self.booth_id)

    def remove_tripwire(self, camera_id: str) -> None:
        db.delete_tripwire(self.db_path, self.booth_id, camera_id)
        with self._lock:
            counter = self._tripwire_counters.get(camera_id)
            if counter is not None:
                counter.set_line(None)
            worker = self.workers.get(camera_id)
            if worker is not None:
                worker.set_tripwire_overlay(None)

    def get_tripwire_status(self, camera_id: str) -> dict:
        counter = self._tripwire_counters.get(camera_id)
        if counter is None or counter.get_line() is None:
            return {"configured": False, "count_in": 0, "count_out": 0, "current_inside": 0}
        return {
            "configured": True,
            "line": dataclasses.asdict(counter.get_line()),
            "count_in": counter.count_in,
            "count_out": counter.count_out,
            "current_inside": counter.current_inside(),
        }

    def get_booth_session_status(self) -> dict:
        """Live (in-memory) snapshot of open booth visits (spec section 36:
        "Current Active Sessions") — deduplicated by identity, unlike
        get_tripwire_status's per-camera IN-OUT counters, which is exactly
        what makes this different from `current_inside` above (see spec
        section 30: never one metric standing in for both)."""
        return {
            "active_count": self.session_manager.active_count(),
            "sessions": [s.to_dict() for s in self.session_manager.all_open_sessions()],
        }

    def _on_status(self, camera_id, status, message):
        now = time.monotonic()
        with self._lock:
            previous = self.camera_status[camera_id]["status"]
            if previous == "online" and status != "online":
                # first step away from a healthy stream -- start the recovery clock. Repeated
                # offline<->reconnecting churn during one outage does not reset it: only leaving
                # "online" starts it, only reaching "online" again stops it (see below), so the
                # latency measures the whole outage, not just its last leg.
                self._camera_failure_started[camera_id] = now
            if status in ("offline", "error"):
                self._camera_last_error[camera_id] = message
            if status == "online" and previous != "online":
                started = self._camera_failure_started.pop(camera_id, None)
                if started is not None:
                    self._camera_last_recovery_latency_sec[camera_id] = now - started
                    self._camera_reconnect_count[camera_id] = self._camera_reconnect_count.get(camera_id, 0) + 1
            workers = getattr(self, "workers", None)
            worker = workers.get(camera_id) if isinstance(workers, dict) else None
            self.camera_status[camera_id] = {
                "status": status,
                "message": message,
                "reconnect_count": self._camera_reconnect_count.get(camera_id, 0),
                "last_error": self._camera_last_error.get(camera_id, ""),
                "last_recovery_latency_sec": self._camera_last_recovery_latency_sec.get(camera_id),
                # Machine-filterable companion to `message` (spec section 32) — see
                # core.vision.CameraWorker._classify_reason for what maps to what.
                "reason_code": worker.get_last_reason_code() if worker is not None else "UNKNOWN",
            }
            if status not in ("online", "degraded"):
                self._person_tracks[camera_id] = []
                self._product_detections[camera_id] = []
        if status == previous:
            return

        severity = {"online": "ok", "degraded": "warning"}.get(status, "warning" if status in ("connecting", "reconnecting", "disabled") else "error")
        db.log_health_event(self.db_path, self.booth_id, self.event_id, camera_id,
                             "camera", severity, message)

        # core.vision.CameraWorker's fuller status set (see
        # MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 4:
        # CONNECTING/ONLINE/DEGRADED/RECONNECTING/OFFLINE/ERROR) gets its
        # own alert type each so the alert feed reads accurately — a first
        # connect isn't alarming, so it's not alerted at all; a drop that
        # is actively retrying is worded differently from one that's given
        # up for now.
        if status == "online":
            if previous not in ("unknown", "connecting"):
                self._push_alert("camera_online", f"{camera_id}: กลับมาออนไลน์แล้ว")
            else:
                self._push_alert("camera_connected", f"{camera_id}: เชื่อมต่อสำเร็จ")
        elif status == "connecting":
            pass  # expected once per camera at startup — not alert-worthy
        elif status == "reconnecting":
            self._push_alert("camera_reconnecting", f"{camera_id}: กำลังพยายามเชื่อมต่อใหม่")
        elif status == "disabled":
            self._push_alert("camera_disabled", f"{camera_id}: ปิดใช้งานกล้องนี้")
        else:  # offline, error
            self._push_alert("camera_offline", f"{camera_id}: {message}")

    def _on_camera_degraded(self, camera_id: str, degraded: bool) -> None:
        """AdaptiveController callback (core/performance.py): fires once
        when a camera has exhausted every AI-side lever (inference rate,
        then input resolution) and is still overloaded, and once again when
        it recovers. A degraded camera is still online/streaming — this
        never touches person_tracks — it's purely a "the AI side of this
        camera is struggling, watch it" signal for the operator."""
        with self._lock:
            existing = self.camera_status.get(camera_id, {})
            current = existing.get("status")
            if degraded and current == "online":
                self.camera_status[camera_id] = {
                    **existing, "status": "degraded", "message": "AI ประมวลผลไม่ทัน กำลังลดภาระอัตโนมัติ",
                }
            elif not degraded and current == "degraded":
                self.camera_status[camera_id] = {**existing, "status": "online", "message": "ปกติ"}
            else:
                return
        if degraded:
            self._push_alert("camera_degraded", f"{camera_id}: ประสิทธิภาพลดลง ระบบกำลังปรับตัวอัตโนมัติ")
        else:
            self._push_alert("camera_online", f"{camera_id}: กลับมาทำงานปกติแล้ว")

    def _on_ai_paused(self, paused: bool) -> None:
        """AdaptiveController's AI-Pause callback (core/performance.py):
        fires once when sustained system CPU forces AI (YOLO + custom
        recognition) to pause for every camera at once, and once again on
        resume. Camera Preview is entirely unaffected — every camera keeps
        streaming live video throughout; only detection/tracking/product
        recognition output stops while paused."""
        with self._lock:
            self._ai_paused = paused
        if paused:
            self._push_alert("ai_paused", "CPU สูงต่อเนื่อง — พัก AI ทั้งระบบชั่วคราว (ภาพกล้องยังสดตามปกติ)")
        else:
            self._push_alert("ai_resumed", "CPU กลับสู่ปกติ — เริ่ม AI ต่ออัตโนมัติ")

    def _push_alert(self, alert_type: str, text: str):
        with self._lock:
            self.recent_alerts.insert(0, {"ts": time.time(), "type": alert_type, "text": text})
            self.recent_alerts = self.recent_alerts[:MAX_ALERTS]

    def _handle_product_recognized(self, event: dict):
        class_name = event["class_name"]
        product = self.catalog.get(class_name)
        if not product:
            return
        source = ", ".join(event["cameras"])
        with self._lock:
            self.current_product = {
                "key": class_name,
                "name": product["name"],
                "tagline": product.get("tagline", ""),
                "price": product.get("price", ""),
                "description": product.get("description", ""),
                "faq": product.get("faq", []),
                "confidence": event["confidence"],
                "cameras": source,
                "speak_text": f"{product['name']}. {product['description']}",
            }
            self.product_seq += 1
        db.log_interaction(self.db_path, self.booth_id, self.event_id, source, class_name,
                            product["name"], event["confidence"])
        self._push_alert("product_found", f"พบสินค้า: {product['name']} ({source})")

    def _clear_current_product(self) -> None:
        with self._lock:
            self.current_product = None

    # ------------------------------------------------------------- readers
    def get_latest_jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            if not self._running:
                return _CLOSED_BOOTH_JPEG
            if not self._camera_enabled.get(camera_id, True):
                return _DISABLED_CAMERA_JPEG
            status = self.camera_status.get(camera_id, {}).get("status")
            if status not in ("online", "degraded"):
                # Camera is mid-(re)connect, offline, or erroring — never
                # fall back to self._latest_jpeg here, or a viewer keeps
                # seeing the last frame captured before the disconnect,
                # frozen forever, with no visual sign anything is wrong.
                return _RECONNECTING_CAMERA_JPEG
            return self._latest_jpeg.get(camera_id) or _RECONNECTING_CAMERA_JPEG

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        """Turn a configured camera on/off without removing it — for a
        camera that's plugged in but not currently needed at the booth (see
        core.vision.CameraWorker.set_enabled, which this delegates to)."""
        worker = self.workers.get(camera_id)
        if worker is None:
            raise ValueError(f"ไม่พบกล้อง {camera_id}")
        worker.set_enabled(enabled)
        with self._lock:
            self._camera_enabled[camera_id] = enabled
            if not enabled:
                self._latest_jpeg.pop(camera_id, None)
                self._person_tracks[camera_id] = []
                self._product_detections[camera_id] = []

    def _camera_status_with_freshness(self) -> dict:
        """camera_status plus per-camera frame freshness (spec section 12/23's contract:
        camera_id, frame_sequence, capture_timestamp alongside state) — pulled live from each
        worker rather than cached on camera_status, since these change every frame, not just on
        the status transitions that update camera_status itself. Callers (both get_state(), which
        every client already polls every 1.5s, and get_camera_diagnostics()) must never present a
        cached frame as current without also being able to see how stale it actually is."""
        out = {}
        workers = getattr(self, "workers", None)
        for cid, status in self.camera_status.items():
            worker = workers.get(cid) if isinstance(workers, dict) else None
            out[cid] = {
                **status,
                "camera_id": cid,
                "frame_sequence": worker.get_frame_sequence() if worker else None,
                "capture_timestamp": worker.get_last_capture_ts() if worker else None,
            }
        return out

    def get_state(self) -> dict:
        booth_row = db.get_booth(self.db_path, self.booth_id) or {}
        with self._lock:
            return {
                "booth_id": self.booth_id,
                "booth_name": self.booth_name,
                "event_id": self.event_id,
                "running": self._running,
                "open_time": booth_row.get("open_time"),
                "close_time": booth_row.get("close_time"),
                "cameras": self._camera_status_with_freshness(),
                "current_product": dict(self.current_product) if self.current_product else None,
                "product_seq": self.product_seq,
                "recent_alerts": list(self.recent_alerts[:20]),
                "people_now": self._distinct_people_now(),
                "face_identity": getattr(self, "face_identity", None) is not None,
                "unique_people_total": self.reid_registry.unique_people_count(),
                "reid": self.reid_registry.stats(),
                "body_gender": self.body_model.stats() if getattr(self, "body_model", None) is not None else None,
                "ai_paused": self._ai_paused,
            }

    def get_performance_snapshot(self) -> dict:
        """Feeds the Performance Monitor panel (booth.html) and
        --benchmark (see web_server.py) — hardware profile, per-camera
        FPS/latency/drop-rate + current adaptive settings, and system
        CPU/RAM/VRAM, all from live measurements
        (MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 29)."""
        with self._lock:
            workers_snapshot = dict(self.workers)
            camera_ids = list(self.camera_ids)
        cameras = []
        for camera_id in camera_ids:
            metrics = self.performance_monitor.camera_snapshot(camera_id)
            worker = workers_snapshot.get(camera_id)
            with self._lock:
                status = self.camera_status.get(camera_id, {}).get("status", "unknown")
            cameras.append({
                **metrics,
                "status": status,
                "detect_every_n_frames": worker.get_detect_every_n_frames() if worker else None,
                "ai_imgsz": worker.get_ai_imgsz() if worker else None,
                "frame_sequence": worker.get_frame_sequence() if worker else None,
            })
        cameras.sort(key=lambda c: c["camera_id"])
        return {
            "hardware": dataclasses.asdict(self.hardware_profile),
            "system": self.performance_monitor.system_snapshot(),
            "cameras": cameras,
        }

    def get_live_analytics(self) -> dict:
        with self._lock:
            people_now = self._distinct_people_now()
        products = [p for p in self.interest_tracker.live_states()
                    if self.catalog.get(p["class_name"]) is not None]
        for p in products:
            p["product_name"] = self.catalog.get(p["class_name"])["name"]
        return {"people_now": people_now, "products": products}

    def get_camera_snapshot(self, camera_id: str) -> dict:
        """Live, right-now detail panel for one camera — the popout single-
        camera view (/booth/camera/{camera_id}) shows this beside its video
        stream. People are broken down by core.vision.CameraWorker.
        _classify_person's category ('male'/'female'/'unknown', the
        same category already drawn on that person's on-screen box and
        logged to presence_sessions) — never a historical/cumulative count,
        just who/what this camera's latest AI pass actually saw. products_
        detected counts only real catalog/COCO product detections (see
        _on_detections/_product_detections's docstring — an UNKNOWN blob is
        never included, so this never misleadingly counts un-trained
        objects as "products")."""
        with self._lock:
            tracks = list(self._person_tracks.get(camera_id, []))
            detections = list(self._product_detections.get(camera_id, []))
        counts = {"male": 0, "female": 0, "unknown": 0}
        people = self._distinct_people(camera_id, tracks)
        for category in people.values():
            counts[category] += 1
        return {
            "camera_id": camera_id,
            "people_total": len(people),
            "male": counts["male"],
            "female": counts["female"],
            "unknown": counts["unknown"],
            "products_detected": len(detections),
            "faces": self._face_count_for(camera_id),
        }

    def get_camera_diagnostics(self, camera_id: str) -> dict:
        """Full per-camera diagnostic record (spec section 30) — capture mode (thread/process),
        worker pid once escalated, hang/stale/restart counters, corrupt/frozen frame counts,
        current failure reason_code. Separate from get_camera_snapshot (people/product counts)
        and get_state (status/message/reconnect_count for the whole fleet at once) because this is
        the "why is this specific camera unhealthy" panel, not the "what does it currently see"
        or "is it up" ones — see MULTI_CAMERA_ROOT_CAUSE_REPORT.md for why those three questions
        need to stay independently answerable."""
        worker = self.workers.get(camera_id)
        if worker is None:
            raise ValueError(f"ไม่พบกล้อง {camera_id}")
        with self._lock:
            status = dict(self.camera_status.get(camera_id, {}))
        return {"camera_id": camera_id, **status, **worker.get_diagnostics()}

    @staticmethod
    def _load_face_identity():
        """FaceIdentity when the SFace model file is installed, else None (never raises)."""
        try:
            if not DEFAULT_SFACE_MODEL.exists():
                logger.info("face identity model not found (%s) - run: python tools/get_face_models.py "
                            "(Re-ID keeps working on body appearance, with strict cross-camera rules)", DEFAULT_SFACE_MODEL)
                return None
        except Exception:
            return None
        try:
            yunet = Path(__file__).resolve().parent.parent / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"
            identity = FaceIdentity.create(yunet)
            if identity is not None:
                logger.info("face identity (SFace) enabled: people are told apart by their faces across cameras")
            return identity
        except Exception:
            logger.exception("could not start face identity")
            return None

    def _resolved_gender_for_track(self, camera_id, track_id) -> "tuple[str, float] | None":
        """The ONE gender a person is shown with, on every camera: their Global Person's decided gender, else that
        person's best available guess (the sign of the accumulated evidence), else this track's own face reading.
        Two tracks with the same Global ID therefore always get the same answer. None = no evidence at all yet."""
        registry = getattr(self, "reid_registry", None)
        smoother = getattr(self, "attribute_smoother", None)
        gid = registry.get_global_id_for(camera_id, track_id) if registry is not None else None
        if gid and smoother is not None:
            result = smoother.get(gid)
            if result.status == "ok" and result.gender in ("MALE", "FEMALE"):
                return result.gender.lower(), float(result.gender_confidence)
            guess = smoother.guess(gid)
            if guess is not None:
                return guess
        own = self._track_face_gender(camera_id, track_id)
        return (own, TRACK_FACE_MIN_CONFIDENCE) if own else None

    def _distinct_people(self, camera_id: str, tracks) -> dict:
        """{person key: 'male'/'female'/'unknown'} for one camera's visible tracks. Two tracks that Re-ID mapped to the
        same Global Person ID are ONE person (a flickering box must never be counted twice)."""
        people: dict = {}
        for track in tracks:
            category = track.get("category")
            resolved = self._resolved_gender_for_track(camera_id, track.get("track_id"))
            if resolved is not None:
                category = resolved[0]
            if category not in ("male", "female"):
                category = "unknown"  # includes any legacy 'child' value: that category no longer exists
            registry = getattr(self, "reid_registry", None)
            gid = registry.get_global_id_for(camera_id, track.get("track_id")) if registry is not None else None
            key = gid or (camera_id, track.get("track_id"))
            if people.get(key, "unknown") == "unknown":
                people[key] = category
        return people

    def _distinct_people_now(self) -> int:
        """People on screen right now across ALL cameras, one per Global Person ID: the same ID seen by two cameras is
        one person, not two. (Call with self._lock held or not - it only reads.)"""
        keys: set = set()
        for camera_id, tracks in list(self._person_tracks.items()):
            keys |= set(self._distinct_people(camera_id, list(tracks)))
        return len(keys)

    def _face_count_for(self, camera_id: str) -> int:
        worker = self.workers.get(camera_id) if isinstance(getattr(self, "workers", None), dict) else None
        getter = getattr(worker, "get_face_count", None)
        try:
            return int(getter()) if callable(getter) else 0
        except Exception:
            return 0

    # -------------------------------------------------------- booth settings
    def get_settings(self) -> dict:
        with self._lock:
            return {
                "booth_id": self.booth_id,
                "booth_name": self.booth_name,
                "event_id": self.event_id,
                "cameras": [
                    {
                        "camera_id": cid,
                        "device": str(self.camera_devices.get(cid, "")),
                        "enabled": self._camera_enabled.get(cid, True),
                    }
                    for cid in self.camera_ids
                ],
            }

    def activate_booth(self, booth_id: str) -> None:
        """Switch which registry Booth (core.database's `booths` table) this
        running process reports as — the only way identity changes now that
        Booth ID / Event ID are real registry rows, not free text. Persists
        the choice so it survives a restart."""
        booth_row = db.get_booth(self.db_path, booth_id)
        if not booth_row:
            raise ValueError(f"ไม่พบบูธ {booth_id} ในระบบ")
        with self._lock:
            self.booth_id = booth_row["id"]
            self.booth_name = booth_row["name"]
            self.event_id = booth_row["event_id"] or ""
            # Tripwire configs are saved per (booth_id, camera_id) — see
            # core/database.py's `tripwires` table — so switching the
            # active booth must reload each camera's line from *this*
            # booth's own saved config, or a stale line (and running
            # IN/OUT counts) from whichever booth was active a moment ago
            # would keep showing/counting against the wrong booth.
            for camera_id in self.camera_ids:
                self._load_tripwire(camera_id)
        settings_path = booth_settings_path_for(self.db_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        # Merge rather than overwrite: this file also holds unrelated
        # settings (e.g. load_camera_settings' enable_builtin_camera /
        # builtin_camera_indices) that a blind overwrite would silently
        # wipe on every single startup, since web_server.py/app.py call
        # activate_booth() unconditionally on boot — discovered when the
        # built-in-camera config kept reverting after every restart.
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}
        data["active_booth_id"] = booth_id
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def remove_booth(self, booth_id: str) -> None:
        """Permanently delete a registry Booth and all its logged data.
        Refuses to delete the currently active one — it's still being
        written to live, so a one-time data wipe could never actually
        stick; the caller must activate a different booth first."""
        with self._lock:
            active = self.booth_id
        if booth_id == active:
            raise ValueError("ไม่สามารถลบบูธที่กำลังใช้งานอยู่ได้ กรุณาเปลี่ยนไปใช้บูธอื่นก่อน")
        db.delete_booth(self.db_path, booth_id)

    def add_camera(self, device) -> str:
        """Raises ValueError if `device` is already bound to an existing
        camera_id — two CameraWorkers independently opening the *same*
        physical camera each get their own valid, live cv2.VideoCapture
        (confirmed on real hardware: Windows/DirectShow happily grants
        concurrent access to one device), so nothing about capture itself
        fails; the two camera cards just end up showing the same physical
        camera's video, which reads exactly like a "duplicate/cross-camera
        frame" bug without being one (spec: MongDee multi-USB-camera
        root-cause repair — "DEVICE INDEX COLLISION"). Reject the collision
        here instead, at the one place both the /settings "add camera" form
        and any other caller of this method funnel through."""
        with self._lock:
            key = normalize_device_key(device)
            for existing_id, existing_device in self.camera_devices.items():
                if normalize_device_key(existing_device) == key:
                    raise ValueError(
                        f"กล้อง {device} ถูกใช้งานอยู่แล้วโดย {existing_id} — ไม่สามารถเพิ่มกล้องตัวเดียวกันซ้ำได้ "
                        f"(เปิดกล้องตัวเดียวกันสองครั้งจะทำให้เห็นภาพซ้ำกันในสองหน้าต่าง)"
                    )
            # Same physical_id layer as dedupe_camera_devices (see its own
            # docstring) — catches "device is a *different* raw index but
            # the exact same physical camera" (Windows re-enumerated between
            # when the existing camera and this new one were each resolved).
            # A no-op wherever physical-identity data is unavailable.
            new_index = device_as_index(device)
            if new_index is not None:
                identities = get_physical_camera_identities()
                new_identity = identities.get(new_index)
                if new_identity is not None:
                    for existing_id, existing_device in self.camera_devices.items():
                        existing_index = device_as_index(existing_device)
                        if existing_index is None:
                            continue
                        existing_identity = identities.get(existing_index)
                        if existing_identity is not None and existing_identity.physical_id == new_identity.physical_id:
                            raise ValueError(
                                f"กล้อง {device} เป็นกล้องตัวเดียวกันทางกายภาพกับ {existing_id} "
                                f"(device={existing_device}) — ไม่สามารถเพิ่มกล้องตัวเดียวกันซ้ำได้"
                            )
            if new_index is not None and new_identity is not None:
                # A camera that was unplugged and came back at a different index has no live device at its
                # OLD index to compare against above, but its worker still remembers who it is. It is the
                # same camera: that worker follows it to the new index (CameraWorker._open), so never add
                # a second camera_id for it.
                for existing_id, existing_worker in self.workers.items():
                    known = getattr(existing_worker, "_physical_id", None)
                    if known is not None and known == new_identity.physical_id:
                        raise ValueError(
                            f"กล้อง {device} คือกล้องตัวเดิมของ {existing_id} ที่เสียบกลับเข้ามาใหม่ — "
                            f"จะเชื่อมต่อกลับให้อัตโนมัติ ไม่เพิ่มซ้ำ"
                        )
            camera_id = f"CAM-{self._next_camera_num}"
            self._next_camera_num += 1
            self.camera_ids.append(camera_id)
            self.camera_devices[camera_id] = device
            self.camera_status[camera_id] = {"status": "unknown", "message": ""}
            self._camera_enabled[camera_id] = True
            self._person_tracks[camera_id] = []
            self._product_detections[camera_id] = []
            if str(device).isdigit():
                # A deliberate (re-)add always wins over a past removal —
                # otherwise re-adding the exact index the operator just
                # removed via /settings would have the Hot-Plug Scan
                # silently undo it again a few seconds later.
                self._ignored_usb_indices.discard(int(device))

        worker = self._make_worker(camera_id, device)
        with self._lock:
            self.workers[camera_id] = worker
            # Almost always a no-op (a brand-new camera_id has no saved
            # tripwire yet) — but on a process restart the same synthetic
            # CAM-N ids get handed out in the same order, so this is what
            # actually restores a previously-configured line for a camera
            # re-added here rather than through __init__'s own startup loop.
            self._load_tripwire(camera_id)
        worker.start()
        return camera_id

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            worker = self.workers.pop(camera_id, None)
            if camera_id in self.camera_ids:
                self.camera_ids.remove(camera_id)
            device = self.camera_devices.pop(camera_id, None)
            if device is not None and str(device).isdigit():
                # Removed on purpose — don't let the Hot-Plug Scan treat the
                # still-plugged-in physical camera as "new" and re-add it.
                self._ignored_usb_indices.add(int(device))
            self.camera_status.pop(camera_id, None)
            self._latest_jpeg.pop(camera_id, None)
            self._camera_enabled.pop(camera_id, None)
            self._person_tracks.pop(camera_id, None)
            self._product_detections.pop(camera_id, None)
            self._tripwire_counters.pop(camera_id, None)
        if worker is not None:
            worker.stop()  # blocks on thread join — never call while holding self._lock
        self.performance_monitor.forget_camera(camera_id)
        db.delete_tripwire(self.db_path, self.booth_id, camera_id)

    def reset_data(self) -> None:
        """Wipe every logged row (interactions, health, hold events, presence
        sessions, ...) for this booth_id — irreversible. The API layer is
        responsible for getting user confirmation before calling this."""
        db.delete_scope_data(self.db_path, booth_id=self.booth_id)

    def run_readiness(self) -> dict:
        with self._lock:
            statuses = {cid: v["status"] for cid, v in self.camera_status.items()}
            workers_snapshot = dict(self.workers)
        resolutions = {cid: w.get_resolution() for cid, w in workers_snapshot.items()}
        report = run_readiness_check(
            camera_statuses=statuses,
            model_loaded=self.model is not None,
            tts_available=True,  # spoken via the browser's Web Speech API, not the server
            db_path=self.db_path,
            camera_resolutions=resolutions,
            camera_devices=self.camera_devices,
            model_device=self.model_device,
            started_at=self.started_at,
        )
        db.log_readiness_check(self.db_path, self.booth_id, self.event_id,
                                "ready" if report["overall_ok"] else "not_ready",
                                readiness_to_json(report))
        return report

    # ------------------------------------------------------------ training
    def start_image_import(self, product_key: str, filenames_and_bytes: list[tuple[str, bytes]]):
        with self._lock:
            if self.import_progress.get(product_key, {}).get("status") == "running":
                raise RuntimeError("มีงานนำเข้าอยู่แล้วสำหรับสินค้านี้ กรุณารอให้เสร็จก่อน")
            self.import_progress[product_key] = {
                "done": 0, "total": len(filenames_and_bytes), "status": "running"
            }

        def job():
            UPLOAD_TMP_ROOT.mkdir(parents=True, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(prefix="mongdee_upload_", dir=UPLOAD_TMP_ROOT)
            try:
                paths = []
                for index, (name, content) in enumerate(filenames_and_bytes):
                    # Upload names are untrusted and may contain ../ or an
                    # absolute path. Keep every temporary file inside tmp_dir.
                    safe_name = Path(name).name or "image.bin"
                    path = Path(tmp_dir) / f"{index:04d}-{safe_name}"
                    path.write_bytes(content)
                    paths.append(str(path))

                def progress_cb(done, total):
                    self._set_import_progress(
                        product_key, {"done": done, "total": total, "status": "running"}
                    )

                added = training.import_images(paths, product_key, self.recognizer,
                                                self.model, self.model_device, progress_cb,
                                                person_segmenter=get_person_segmenter(self.model_device))
                self._set_import_progress(
                    product_key, {"done": added, "total": added, "status": "done"}
                )
            except Exception as exc:
                self._set_import_progress(product_key, {
                    "done": 0, "total": 0, "status": "error", "message": str(exc)
                })
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        threading.Thread(target=job, daemon=True).start()

    def start_video_import(self, product_key: str, filename: str, content: bytes):
        with self._lock:
            if self.import_progress.get(product_key, {}).get("status") == "running":
                raise RuntimeError("มีงานนำเข้าอยู่แล้วสำหรับสินค้านี้ กรุณารอให้เสร็จก่อน")
            self.import_progress[product_key] = {
                "done": 0, "total": 0, "status": "running"
            }

        def job():
            UPLOAD_TMP_ROOT.mkdir(parents=True, exist_ok=True)
            tmp_dir = tempfile.mkdtemp(prefix="mongdee_upload_", dir=UPLOAD_TMP_ROOT)
            try:
                safe_name = Path(filename).name or "video.bin"
                path = Path(tmp_dir) / safe_name
                path.write_bytes(content)

                def progress_cb(done, total):
                    self._set_import_progress(
                        product_key, {"done": done, "total": total, "status": "running"}
                    )

                added = training.import_video(str(path), product_key, self.recognizer,
                                               self.model, self.model_device, progress_cb,
                                               person_segmenter=get_person_segmenter(self.model_device))
                self._set_import_progress(
                    product_key, {"done": added, "total": added, "status": "done"}
                )
            except Exception as exc:
                self._set_import_progress(product_key, {
                    "done": 0, "total": 0, "status": "error", "message": str(exc)
                })
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        threading.Thread(target=job, daemon=True).start()

    def _set_import_progress(self, product_key: str, state: dict) -> None:
        with self._lock:
            self.import_progress[product_key] = state

    def get_import_progress(self, product_key: str) -> dict:
        with self._lock:
            return dict(self.import_progress.get(
                product_key, {"done": 0, "total": 0, "status": "idle"}
            ))

    # ------------------------------------------------- live "record from camera" training
    def start_live_training(self, product_key: str) -> None:
        """Begins (or restarts) a guided-rotation recording session for one product -- see
        core.training.LiveTrainingSession. One session per product at a time; starting again
        (e.g. the operator clicked "record" a second time after a failed attempt) simply replaces
        it, discarding whatever partial coverage the previous attempt had."""
        if not self.catalog.get(product_key):
            raise ValueError(f"ไม่พบสินค้า {product_key}")
        with self._lock:
            self._live_training_sessions[product_key] = training.LiveTrainingSession(
                product_key, self.recognizer, self.model, self.model_device,
                person_segmenter=get_person_segmenter(self.model_device),
            )

    def feed_live_training_frame(self, product_key: str, jpeg_bytes: bytes) -> dict:
        """One captured tick from the guided-rotation UI. Returns LiveTrainingSession.feed()'s
        dict so the frontend can show a live distinct-view count and decide whether to keep
        recording -- see that method's own docstring for exactly what "accepted" means here."""
        with self._lock:
            session = self._live_training_sessions.get(product_key)
        if session is None:
            raise ValueError("ยังไม่ได้เริ่มบันทึกสำหรับสินค้านี้")
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return {"accepted": False, "distinct_views": session.added,
                    "attempted": session.attempted, "reason": "unreadable"}
        return session.feed(frame)

    def finish_live_training(self, product_key: str) -> dict:
        """Ends the session (see LiveTrainingSession.finish -- raises the same "found nothing
        usable" error as a batch image/video import if the whole recording produced zero usable
        views) and forgets it either way, so a later start_live_training begins fresh."""
        with self._lock:
            session = self._live_training_sessions.pop(product_key, None)
        if session is None:
            raise ValueError("ยังไม่ได้เริ่มบันทึกสำหรับสินค้านี้")
        return session.finish()

    # ---------------------------------------------------------- GPU status
    # Acceleration is installed automatically at setup time now (see
    # requirements.txt / install.bat / install.sh) — this just reports what
    # core/device.py's resolve_device() ended up picking at startup.
    def get_gpu_status(self) -> dict:
        using_gpu = self.model_device not in ("cpu", None)
        return {
            "device": device_label(self.model_device),
            "using_gpu": using_gpu,
            # "backend"/"available" alongside the two keys above (which
            # settings.js already reads) rather than replacing them, so this
            # stays an additive change for any other consumer.
            "backend": device_backend(self.model_device),
            "available": using_gpu,
        }
