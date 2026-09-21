import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

import core.vision as vision
from core import capture_process


@pytest.fixture(autouse=True)
def _reset_active_camera_count():
    """core.vision._active_camera_count is process-wide shared state (see
    _low_bandwidth_open_preferred) -- reset it around every test in this
    module so one test's CameraWorker (or a threaded test whose worker
    thread hasn't finished unwinding by the time thread.join()'s timeout
    returns) can never leak a nonzero count into an unrelated later test
    and silently change which profile list that test's _open_capture call
    is expected to try."""
    vision._active_camera_count = 0
    yield


@pytest.fixture(autouse=True)
def _reset_cross_process_lock_flag():
    """core.capture_process._cross_process_lock_needed is also process-wide, one-way (set once
    escalation happens, never cleared -- see its own docstring) state: a real subprocess-spawning
    test in this SAME pytest process (test_camera_process_escalation.py, test_capture_process.py)
    running first would otherwise permanently make every _open_capture call in every later test
    pay for a real OS semaphore acquire/release it does not need, which is exactly the per-call
    overhead this flag exists to avoid in the common (never-escalated) case -- several of this
    module's own tests assert tight wall-clock bounds on _open_capture/discover_cameras and would
    flake under that extra cost."""
    capture_process._cross_process_lock_needed.clear()
    yield
    vision._active_camera_count = 0


class Capture:
    def __init__(self, source=None, backend=None, *, native_only=False, broken=False,
                 fourcc_liar=False):
        self.source = source
        self.props = {}
        self.released = False
        self.reads = 0
        self.native_only = native_only
        self.broken = broken
        # Simulates a real UVC device observed on real hardware: it accepts
        # a CAP_PROP_FOURCC set() (and reports read=OK regardless of which
        # codec was requested) but always actually streams its native
        # uncompressed format -- get(CAP_PROP_FOURCC) reports that reality,
        # not whatever was last set().
        self.fourcc_liar = fourcc_liar

    def isOpened(self):
        return self.source is not None and not self.released

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        if self.fourcc_liar and prop == cv2.CAP_PROP_FOURCC:
            return cv2.VideoWriter_fourcc(*"YUY2")
        return self.props.get(prop, 0)

    def read(self):
        self.reads += 1
        if self.broken or self.reads < 3:
            return False, None
        if self.fourcc_liar:
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        compressed = self.props.get(cv2.CAP_PROP_FOURCC) == cv2.VideoWriter_fourcc(*"MJPG")
        works = not compressed if self.native_only else compressed
        if not works:
            return False, None
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def setup(monkeypatch, **kwargs):
    captures = []
    def create(source=None, backend=None):
        cap = Capture(source, backend, **kwargs)
        captures.append(cap)
        return cap
    monkeypatch.setattr(vision.cv2, "VideoCapture", create)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    # Isolate from this machine's real DirectShow enumeration -- _forbidden_device_name()
    # otherwise calls it for real, and on a box with a real virtual-camera driver installed
    # (observed here: index 2 identifying as "OBS Virtual Camera" instead of a real USB camera,
    # varying across separate enumeration calls), that leaks unrelated real-world device state
    # into every test built on this fixture.
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda device: None)
    return captures


class WedgedReleaseCapture(Capture):
    """Like Capture(broken=True) (isOpened=True, read() always fails --
    exactly what a rejected/failed candidate looks like), but release()
    blocks on a gate the test controls, standing in for a real wedged
    DirectShow graph (see _release_capture_safely's docstring)."""

    def __init__(self, *a, **k):
        super().__init__(*a, broken=True, **k)
        self.gate = threading.Event()
        self.release_started = threading.Event()

    def release(self):
        self.release_started.set()
        self.gate.wait(5.0)
        super().release()


def _setup_wedged_release(monkeypatch):
    caps: list[WedgedReleaseCapture] = []

    def create(source=None, backend=None):
        cap = WedgedReleaseCapture(source, backend)
        caps.append(cap)
        return cap

    monkeypatch.setattr(vision.cv2, "VideoCapture", create)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    monkeypatch.setattr(vision, "RELEASE_JOIN_TIMEOUT_SEC", 0.1)
    return caps


def test_open_capture_does_not_hang_when_a_rejected_candidates_release_wedges(monkeypatch):
    """Regression test: _open_capture()'s own candidate-rejection loop used
    to call cap.release() synchronously, WHILE HOLDING the shared
    _open_lock -- a single wedged release there would hold that lock
    forever and silently stop every camera in the process from ever
    opening again (the reported "one or all cameras stuck 'connecting'
    until the server is restarted" symptom). Must stay bounded instead."""
    caps = _setup_wedged_release(monkeypatch)
    try:
        t0 = time.monotonic()
        result = vision._open_capture(0, label="TEST")
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0                    # bounded, not stuck forever on the wedge
        assert not result.isOpened()             # every profile failed -- an unopened capture returned
        # Release really was attempted for each rejected candidate (just didn't block on it) --
        # wait rather than a bare boolean check, since the background release thread's OS
        # scheduling isn't guaranteed to land inside RELEASE_JOIN_TIMEOUT_SEC under load.
        # Excludes the final cv2.VideoCapture() sentinel _open_capture() itself returns once every
        # candidate is exhausted (source=None) -- that one is `result`, never released by the function.
        real_candidates = [c for c in caps if c.source is not None]
        assert real_candidates and all(c.release_started.wait(2.0) for c in real_candidates)
        # The lock must be free again -- a second, unrelated open succeeds immediately.
        assert vision._open_lock.acquire(timeout=1.0)
        vision._open_lock.release()
    finally:
        for c in caps:
            c.gate.set()  # let the background release threads finish -- don't leak them past this test


def test_discover_cameras_does_not_hang_when_a_candidates_release_wedges(monkeypatch):
    """Same regression as above, for discover_cameras() -- called forever
    by BoothManager's Hot-Plug Scan, so a wedge here used to be able to
    permanently kill hot-plug recovery for the rest of the process's life."""
    caps = _setup_wedged_release(monkeypatch)
    try:
        t0 = time.monotonic()
        found = vision.discover_cameras(max_index=3)
        elapsed = time.monotonic() - t0
        # Only cv2.VideoCapture is mocked here -- _forbidden_device_name()'s real
        # list_directshow_devices() call (PowerShell/COM device enumeration) still runs for
        # real, and on a machine with real cameras attached that alone measured ~3.0-3.1s across
        # 3 probed indices, occasionally tripping a tighter bound. The bound only needs to rule
        # out "hangs forever on the wedge" (RELEASE_JOIN_TIMEOUT_SEC=0.1 per candidate), not match
        # a specific real-enumeration cost.
        assert elapsed < 8.0
        assert found == []
        assert caps and all(c.release_started.wait(2.0) for c in caps)
    finally:
        for c in caps:
            c.gate.set()


def test_multiple_cameras_configured_before_first_read(monkeypatch):
    captures = setup(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        opened = list(pool.map(vision._open_capture, range(4)))
    assert len(captures) == 4
    assert {cap.source for cap in opened} == {0, 1, 2, 3}
    for cap in opened:
        assert cap.isOpened()
        assert cap.read()[0]
        assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 640
        assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1
        cap.release()


def test_native_fallback_is_validated_and_failed_handle_released(monkeypatch):
    captures = setup(monkeypatch, native_only=True)
    cap = vision._open_capture(0)
    assert len(captures) == len(vision._OPEN_PROFILES) - 1
    assert all(item.released for item in captures[:-1])
    assert cap is captures[-1]
    assert cap.read()[0]


def test_open_capture_rejects_mjpg_profile_when_device_silently_delivers_native_format(monkeypatch):
    """Real-hardware evidence (docs/design/camera-pipeline-root-cause-repair.md):
    a UVC camera can report open=OK, read=OK for an MJPG-requested profile
    while actually delivering its uncompressed native format (a silently-
    ignored FourCC set()). _open_capture must not accept that as a genuine
    MJPG negotiation -- it would otherwise "succeed" at the highest-
    bandwidth outcome of every profile tried, defeating the whole point of
    trying MJPG first on a bandwidth-constrained shared USB hub -- and must
    keep trying until it reaches the explicit YUY2 profile designed for
    exactly this device behavior."""
    captures = setup(monkeypatch, fourcc_liar=True)
    cap = vision._open_capture(0)
    assert cap.isOpened()
    mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    yuy2_fourcc = cv2.VideoWriter_fourcc(*"YUY2")
    rejected = captures[:-1]
    assert len(rejected) == 3  # the 3 MJPG-labeled profiles in _OPEN_PROFILES
    assert all(item.props.get(cv2.CAP_PROP_FOURCC) == mjpg_fourcc for item in rejected)
    assert all(item.released for item in rejected)
    assert cap is captures[-1]
    assert cap.props[cv2.CAP_PROP_FOURCC] == yuy2_fourcc


def test_all_failed_candidates_return_closed_capture(monkeypatch):
    captures = setup(monkeypatch, broken=True)
    cap = vision._open_capture(0)
    assert not cap.isOpened()
    assert all(item.released for item in captures[:-1])
    assert [item.source for item in captures] == [0] * len(vision._OPEN_PROFILES) + [None]


def test_open_capture_low_bandwidth_only_skips_the_two_largest_profiles(monkeypatch):
    """Spec: when another camera is already streaming, a new camera's own
    open must not unconditionally request the largest/highest-bandwidth
    profile first -- real hardware evidence
    (.hw_validation/*.stderr, 2026-09-18) shows a second concurrent USB
    camera failing mid-stream (Windows MF_E_HW_MFT_FAILED_START_STREAMING)
    even though every profile opens and reads fine in isolation, consistent
    with a shared USB hub/controller resource limit. low_bandwidth_only=True
    must try only _LOW_BANDWIDTH_OPEN_PROFILES (the two smallest, already-
    compressed profiles), never the two full 640x480 MJPG profiles."""
    captures = setup(monkeypatch)
    cap = vision._open_capture(0, low_bandwidth_only=True)
    assert cap.isOpened()
    # Succeeds immediately at the first low-bandwidth profile (MJPG
    # 320x240) -- exactly like the existing default-profile-list behavior
    # succeeding at its own first (640x480) profile -- so only one Capture
    # is ever created, and it was never configured for 640x480.
    assert len(captures) == 1
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 320
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 240


def test_open_capture_low_bandwidth_only_defaults_to_false(monkeypatch):
    """Solo/first-camera behavior (the existing, already-covered default)
    must be completely unchanged -- low_bandwidth_only is opt-in, never
    assumed, so a caller that doesn't pass it (e.g. discover_cameras, or
    any pre-existing call site) keeps trying the full _OPEN_PROFILES list
    exactly as before this change."""
    captures = setup(monkeypatch)
    cap = vision._open_capture(0)
    assert cap.isOpened()
    assert len(captures) == 1  # first (largest) profile succeeds immediately, as before


def test_second_camera_worker_requests_low_bandwidth_profile_while_first_is_active(monkeypatch):
    """End-to-end through CameraWorker._open() (not just _open_capture
    directly): a first worker opening while no other camera is active must
    get the full-size profile; a second worker opening while the first is
    still active (counted via _mark_camera_active) must get the
    low-bandwidth profiles -- and once the first worker releases its
    capture, a third worker opening afterward is back to the full-size
    profile, since nothing else is active any more."""
    requested = []

    def fake_open_capture(device, label=None, low_bandwidth_only=False):
        requested.append(low_bandwidth_only)
        return Capture(device)

    monkeypatch.setattr(vision, "_open_capture", fake_open_capture)
    model = type("Model", (), {"names": {0: "person"}})()

    worker_a = vision.CameraWorker("CAM-1", 0, model, [])
    assert worker_a._open()
    assert requested[-1] is False  # no other camera active yet

    worker_b = vision.CameraWorker("CAM-2", 1, model, [])
    assert worker_b._open()
    assert requested[-1] is True  # CAM-1 is already active

    worker_a._release_capture()
    worker_b._release_capture()

    worker_c = vision.CameraWorker("CAM-3", 2, model, [])
    assert worker_c._open()
    assert requested[-1] is False  # both prior cameras released -- nothing active any more


def test_active_camera_count_never_goes_negative_on_double_release(monkeypatch):
    """_release_capture must be safe to call on a worker that was never
    successfully opened (e.g. _open() returning False before
    _mark_camera_active ever ran) or twice in a row -- _active_camera_count
    must never be decremented past zero, which would otherwise make a
    *later*, genuinely-solo camera wrongly prefer the low-bandwidth
    profiles because the shared counter looked positive when it shouldn't
    have been."""
    model = type("Model", (), {"names": {0: "person"}})()
    worker = vision.CameraWorker("CAM-1", 0, model, [])
    worker._release_capture()  # never opened -- must be a safe no-op
    worker._release_capture()  # calling it twice must also be a safe no-op
    assert vision._active_camera_count == 0


def test_worker_does_not_reconfigure_validated_stream(monkeypatch):
    cap = Capture(0)
    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: cap)
    model = type("Model", (), {"names": {0: "person"}})()
    worker = vision.CameraWorker("test", 0, model, [])
    assert worker._open()
    assert cap.props == {}
    worker._release_capture()
    assert cap.released


def test_busy_initializer_does_not_open_another_device(monkeypatch):
    """_open_lock is an RLock (shared with core.camera_identity's
    DIRECTSHOW_LOCK -- see its own docstring), so the *same* thread
    re-entering it must succeed, not time out -- only a genuinely
    different thread already holding it should block a new open attempt.
    A real background thread is used here (rather than acquiring on this
    test's own thread) to exercise that actual cross-thread case."""
    captures = setup(monkeypatch)
    monkeypatch.setattr(vision, "OPEN_LOCK_TIMEOUT_SEC", 0)
    holder_ready = threading.Event()
    release_holder = threading.Event()

    def hold_lock():
        with vision._open_lock:
            holder_ready.set()
            release_holder.wait(timeout=5)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert holder_ready.wait(timeout=5)
    try:
        cap = vision._open_capture(0)
    finally:
        release_holder.set()
        holder.join(timeout=5)
    assert not cap.isOpened()
    assert [item.source for item in captures] == [None]


def _make_worker(monkeypatch, open_results):
    """A CameraWorker whose _open() replays open_results in order (each a
    bool), without touching cv2 or threads — for exercising _attempt_reopen
    in isolation."""
    model = type("Model", (), {"names": {0: "person"}})()
    worker = vision.CameraWorker("CAM-2", 1, model, [])
    results = iter(open_results)
    monkeypatch.setattr(worker, "_open", lambda: next(results))
    return worker


def test_is_capture_open_reflects_cap_state():
    model = type("Model", (), {"names": {0: "person"}})()
    worker = vision.CameraWorker("CAM-1", 0, model, [])
    assert worker.is_capture_open() is False
    worker._cap = Capture(0)
    assert worker.is_capture_open() is True
    worker._cap = None
    assert worker.is_capture_open() is False


def test_request_immediate_retry_lets_the_very_next_attempt_through(monkeypatch):
    """web/booth_manager.py's Hot-Plug Scan calls this once it has
    independently confirmed a disconnected camera's device is reachable
    again, so the worker doesn't sit out whatever backoff its previous
    failures grew to (see REOPEN_BACKOFF_MAX_SEC)."""
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    worker = _make_worker(monkeypatch, [False] * 4 + [True])

    last_attempt_time = t[0]
    for _ in range(4):
        assert worker._attempt_reopen() is False
        last_attempt_time = t[0]
        t[0] += worker._reopen_backoff_sec  # advance exactly far enough for the next attempt to fire
    grown_backoff = worker._reopen_backoff_sec
    assert grown_backoff > vision.REOPEN_BACKOFF_INITIAL_SEC

    # Without the nudge, attempting again shortly after the *last actual
    # attempt* (not after the backoff has already fully elapsed again) is
    # still refused — gated before ever calling _open(), so it consumes no
    # item from the results queue either.
    t[0] = last_attempt_time + 0.1
    assert worker._attempt_reopen() is False

    worker.request_immediate_retry()
    assert worker._attempt_reopen() is True  # goes through immediately, no further wait needed
    assert worker._reopen_backoff_sec == vision.REOPEN_BACKOFF_INITIAL_SEC


def test_reopen_backoff_grows_on_repeated_failure(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    worker = _make_worker(monkeypatch, [False] * 6)

    delays = []
    for _ in range(6):
        assert worker._attempt_reopen() is False
        delays.append(worker._reopen_backoff_sec)
        t[0] += delays[-1]  # advance exactly far enough for the next attempt to fire
    assert delays == [6.0, 12.0, 24.0, 30.0, 30.0, 30.0]


def test_reopen_backoff_resets_on_success(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    worker = _make_worker(monkeypatch, [False, False, True])

    for _ in range(2):
        worker._attempt_reopen()
        t[0] += worker._reopen_backoff_sec
    assert worker._reopen_backoff_sec == vision.REOPEN_BACKOFF_INITIAL_SEC * (vision.REOPEN_BACKOFF_MULTIPLIER ** 2)

    assert worker._attempt_reopen() is True
    assert worker._reopen_backoff_sec == vision.REOPEN_BACKOFF_INITIAL_SEC
    assert worker._reopen_attempt == 0


def test_reopen_does_not_retry_before_backoff_elapses(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    worker = _make_worker(monkeypatch, [False, True])

    assert worker._attempt_reopen() is False
    open_calls_before = worker._reopen_attempt
    t[0] += 0.1  # far short of REOPEN_BACKOFF_INITIAL_SEC (3s)
    assert worker._attempt_reopen() is False
    assert worker._reopen_attempt == open_calls_before  # no new attempt was made


def test_one_camera_failure_does_not_affect_another_workers_backoff(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    failing = _make_worker(monkeypatch, [False] * 3)
    healthy = _make_worker(monkeypatch, [True])

    for _ in range(3):
        failing._attempt_reopen()
        t[0] += failing._reopen_backoff_sec
    assert failing._reopen_backoff_sec > vision.REOPEN_BACKOFF_INITIAL_SEC

    assert healthy._attempt_reopen() is True
    assert healthy._reopen_backoff_sec == vision.REOPEN_BACKOFF_INITIAL_SEC


def test_usb_streaming_resource_error_retry_never_substitutes_a_different_device(monkeypatch):
    """The exact hard requirement from the multi-USB-hub investigation: a
    camera stuck on a Windows hardware-streaming-resource error (isOpened
    True, read forever False — MF_E_HW_MFT_FAILED_START_STREAMING /
    0xC00D3704 in practice, seen when two USB cameras share one hub) must
    keep retrying its OWN configured device on every attempt. It must never
    substitute a different index (e.g. a built-in camera) as a fallback, no
    matter how many times it fails."""
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    # Isolate from this machine's real DirectShow enumeration (_forbidden_device_name() otherwise
    # calls it for real) -- on a box with a real virtual-camera driver installed, that enumeration
    # can legitimately vary between runs and has nothing to do with what this test checks.
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda device: None)
    opened_sources = []

    def create(source=None, backend=None):
        opened_sources.append(source)
        return Capture(source, backend, broken=True)  # isOpened True, read always False

    monkeypatch.setattr(vision.cv2, "VideoCapture", create)

    for _ in range(3):
        cap = vision._open_capture(2, label="CAM-2")
        assert not cap.isOpened()

    # Every VideoCapture() call across every retry either used device 2, or
    # (once all profiles are exhausted) took no device at all — cv2.VideoCapture()
    # with no arguments opens nothing, unlike VideoCapture(0)/(1) which would
    # actually try some other physical camera. Never 0 or 1 (a stand-in for
    # "some other/built-in camera").
    real_attempts = [s for s in opened_sources if s is not None]
    assert real_attempts == [2] * (len(vision._OPEN_PROFILES) * 3)
    assert opened_sources.count(None) == 3


def test_hardware_resource_error_on_one_camera_isolates_from_running_camera(monkeypatch):
    """CAM-1 = RUNNING, CAM-2 stuck on a hardware streaming-resource error
    (isOpened True, read forever False) must never restart or otherwise
    affect CAM-1's own backoff/attempt state."""
    t = [1000.0]
    monkeypatch.setattr(vision.time, "monotonic", lambda: t[0])
    cam1 = _make_worker(monkeypatch, [True])
    cam2 = _make_worker(monkeypatch, [False] * 5)

    assert cam1._attempt_reopen() is True
    baseline_backoff = cam1._reopen_backoff_sec

    for _ in range(5):
        cam2._attempt_reopen()
        t[0] += cam2._reopen_backoff_sec

    assert cam1._reopen_backoff_sec == baseline_backoff
    assert cam1._reopen_attempt == 0
    assert cam2._reopen_backoff_sec > vision.REOPEN_BACKOFF_INITIAL_SEC


def test_discover_cameras_never_opens_skipped_builtin_index(monkeypatch):
    opened = []
    def create(source=None, backend=None):
        opened.append(source)
        return Capture(source, backend)
    monkeypatch.setattr(vision.cv2, "VideoCapture", create)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    # See the same isolation note above: this machine's real DirectShow enumeration must not
    # decide which indices this test finds.
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda device: None)

    found = vision.discover_cameras(max_index=3, skip=frozenset({0}))
    assert 0 not in opened
    assert found == [1, 2]


def test_discover_cameras_probes_past_an_incomplete_directshow_enumeration(monkeypatch):
    """CONFIRMED ROOT CAUSE regression test (missing-cameras investigation): Windows/DirectShow's
    own enumeration (core.camera_identity.list_directshow_devices) can legitimately report fewer
    devices than physically exist right now (e.g. a camera whose driver is still initializing at
    the exact moment this is called -- plausible at startup with several cameras, or right after a
    hot-plug event). Before the fix, discover_cameras() trusted that count as exact and clamped its
    probe range to it (`max(known) + 1`), so real cameras beyond it were never even attempted --
    not retried, not logged, just silently never probed. Here `known` only reports index 0 while
    real (simulated) cameras exist at 0-2; discovery must still find all three (DISCOVERY_INDEX_
    SAFETY_MARGIN=2 reaches index 2; it is a deliberately small hedge, not unlimited, so this test
    stays within that margin rather than asserting an arbitrarily large undercount is recoverable)."""
    opened = []
    def create(source=None, backend=None):
        opened.append(source)
        return Capture(source, backend, broken=(source is not None and source >= 3))
    monkeypatch.setattr(vision.cv2, "VideoCapture", create)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    import core.camera_identity as camera_identity
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: {0: "USB Camera"})

    found = vision.discover_cameras(max_index=16)
    assert found == [0, 1, 2]


def test_discover_cameras_still_narrows_the_scan_for_a_larger_complete_enumeration(monkeypatch):
    """The original optimization (avoid a dozen wasted ~1s failed-open probes when only a couple of
    cameras exist) must survive the fix above: a `known` enumeration that already reports several
    devices still meaningfully narrows the scan, it just no longer narrows all the way down to
    exactly `max(known) + 1` with zero safety margin."""
    opened = []
    def create(source=None, backend=None):
        opened.append(source)
        return Capture(source, backend)
    monkeypatch.setattr(vision.cv2, "VideoCapture", create)
    monkeypatch.setattr(vision, "_candidate_opens", lambda device: [(device, cv2.CAP_DSHOW)])
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    import core.camera_identity as camera_identity
    monkeypatch.setattr(camera_identity, "list_directshow_devices",
                         lambda: {0: "USB Camera", 1: "USB Camera 2"})

    vision.discover_cameras(max_index=16)
    assert max(opened) < 10   # nowhere near the full 0..15 blind sweep
    assert max(opened) >= 3   # but still past `known`'s own max(1) + 1, per the safety margin


def _never_open(source=None, backend=None):
    """cv2.VideoCapture() with no arguments just constructs an always-closed
    handle — that's the legitimate "refused, nothing to open" return value.
    Only a call that actually names a device is the violation under test."""
    if source is not None:
        raise AssertionError(f"cv2.VideoCapture must not be called with a real device (source={source!r})")
    return Capture(None)


def test_open_capture_refuses_reconnect_when_index_now_identifies_as_builtin(monkeypatch):
    """The exact regression found live: CAM-2 was assigned device index 1
    when index 1 was a real USB camera. Windows later re-enumerated so
    index 1 now identifies as the built-in camera — a plain reconnect
    (CameraWorker._open() -> _open_capture(1, label='CAM-2')) must refuse
    outright, never calling cv2.VideoCapture at all, instead of silently
    reading the built-in camera because the caller still thinks index 1 is
    the same USB camera it was originally assigned."""
    monkeypatch.setattr(vision, "_forbidden_device_names", frozenset({"hd webcam"}))
    monkeypatch.setattr(
        vision, "_forbidden_device_name",
        lambda device: "HD WebCam" if int(device) == 1 else None,
    )
    monkeypatch.setattr(vision.cv2, "VideoCapture", _never_open)

    cap = vision._open_capture(1, label="CAM-2")
    assert not cap.isOpened()


def test_open_capture_refuses_virtual_camera_unconditionally(monkeypatch):
    """A virtual camera (OBS Virtual Camera, ...) is excluded regardless of
    enable_builtin_camera / configured names — it's never a legitimate
    physical booth source."""
    monkeypatch.setattr(vision, "_forbidden_device_names", frozenset())  # nothing built-in configured
    monkeypatch.setattr(
        vision, "_forbidden_device_name",
        lambda device: "OBS Virtual Camera" if int(device) == 2 else None,
    )
    monkeypatch.setattr(vision.cv2, "VideoCapture", _never_open)

    cap = vision._open_capture(2, label="CAM-3")
    assert not cap.isOpened()


def test_forbidden_device_name_resolves_current_name_at_index(monkeypatch):
    """_forbidden_device_name is the piece that actually re-resolves
    identity fresh every call — verify it end-to-end against a mocked
    core.camera_identity.list_directshow_devices, not just the wiring."""
    import core.camera_identity as camera_identity
    monkeypatch.setattr(
        camera_identity, "list_directshow_devices",
        lambda: {0: "USB Camera", 1: "HD WebCam", 2: "OBS Virtual Camera"},
    )
    vision.set_forbidden_device_names(["HD WebCam"])
    try:
        assert vision._forbidden_device_name(0) is None          # allowed USB camera
        assert vision._forbidden_device_name(1) == "HD WebCam"   # configured built-in
        assert vision._forbidden_device_name(2) == "OBS Virtual Camera"  # always-forbidden virtual
        assert vision._forbidden_device_name(99) is None          # unknown index — nothing to refuse
        assert vision._forbidden_device_name("/dev/video0") is None  # non-numeric device — not Windows-index-addressed
    finally:
        vision.set_forbidden_device_names([])  # setter, not monkeypatch — restore module state manually


def test_open_capture_proceeds_normally_for_a_non_forbidden_name(monkeypatch):
    """Sanity check that the new identity guard doesn't block ordinary,
    allowed USB cameras — same setup as test_multiple_cameras_configured_before_first_read
    but with a real (non-mocked) _forbidden_device_name resolving to None."""
    captures = setup(monkeypatch)
    monkeypatch.setattr(vision, "_forbidden_device_name", lambda device: None)
    cap = vision._open_capture(0, label="CAM-1")
    assert cap.isOpened()
    assert cap.read()[0]


def test_looks_like_noise_rejects_independent_random_pixels(monkeypatch):
    """The exact live-observed failure this exists for: a USB camera
    disconnects mid-stream but its capture handle keeps returning
    ok=True with whatever garbage is left in a stale buffer, which shows
    up as uniform, spatially-uncorrelated noise — unlike any real image."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert vision._looks_like_noise(noise)


def test_looks_like_noise_accepts_a_real_looking_image():
    # A smooth gradient has strong pixel-to-pixel correlation, like any
    # real camera image — must never be rejected.
    gradient = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    real_looking = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    assert not vision._looks_like_noise(real_looking)


def test_looks_like_noise_accepts_a_near_black_frame():
    # A lens cap / blacked-out room is flat, not noisy — must not be
    # confused with the independent-random-pixel garbage case.
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    assert not vision._looks_like_noise(black)


def _luminance_noise(rng, shape_2d, low, high):
    """Per-pixel noise shared across all 3 channels (broadcast, not drawn
    independently per channel) — real camera sensor noise is predominantly
    luminance noise; R/G/B move together at it the same way they do at a
    real edge or texture. Drawing independent noise per channel (an earlier
    version of these fixtures did this) artificially decorrelates the
    channels in a way no real image does, which would make these
    fixtures fail _cross_channel_corr's check for reasons that have
    nothing to do with what it actually exists to catch — see
    NOISE_CROSS_CHANNEL_CORR's docstring for the real incident (an actual
    physically-connected camera producing colorful garbage) that check was
    added for, and the real FairFace training photos it was validated
    against."""
    return rng.integers(low, high, shape_2d, dtype=np.int16)[:, :, None]


def _real_looking_gradient_frame(rng):
    """A smooth vertical brightness gradient + mild per-pixel texture —
    stand-in for an ordinary real camera frame, used as the base several
    corrupted-band tests below overlay a garbage band onto."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        frame[y, :, :] = int(80 + 60 * (y / 480))
    noise = _luminance_noise(rng, frame.shape[:2], -5, 5)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _garbled_scanline(rng, width=640):
    """One row of independent-random pixels, meant to be repeated down a
    band — models a stuck/repeated corrupted scanline (a common real
    USB/driver-tear shape: the same bad line held or duplicated across
    several rows), rather than fresh-independent-per-pixel-per-row static.
    That distinction matters here: cv2.resize(..., INTER_AREA)'s box-filter
    downsampling averages each destination pixel over a whole block of
    source pixels, which smooths *fresh*-independent-per-row noise toward
    the frame's mean far more aggressively (each destination row averages
    many statistically-independent source rows) than it does a repeated
    single bad line (every source row in the block is identical, so the
    block average keeps that line's own roughness instead of blurring it
    away) — i.e. a repeated/stuck scanline survives this downsample and
    fresh full-band static may not, at this resize ratio."""
    return rng.integers(0, 255, (1, width, 3), dtype=np.uint8)


def test_looks_like_noise_rejects_a_partial_garbage_band():
    """One reported bug symptom (camera card green/online, but the picture
    shows corrupted/noisy horizontal artifacts) matches a torn/partial
    frame where only a *band* of rows is garbage — e.g. a stuck/repeated
    corrupted scanline from a USB transfer interrupted mid-frame — while
    the rest of the frame still looks like a normal image. A whole-frame
    roughness average would dilute this away; NOISE_CROSS_CHANNEL_CORR's
    check (a corrupted band is exactly as channel-decorrelated as
    full-frame garbage) must still catch it — see its own docstring for
    why a per-row *roughness* variant of this check was tried first and
    then removed for being unsafe against real photos."""
    rng = np.random.default_rng(3)
    frame = _real_looking_gradient_frame(rng)
    frame[200:230, :, :] = _garbled_scanline(rng)
    assert vision._looks_like_noise(frame)


def test_looks_like_noise_rejects_multiple_thin_garbage_bands():
    rng = np.random.default_rng(4)
    frame = _real_looking_gradient_frame(rng)
    for band_start in range(0, 480, 30):
        frame[band_start:band_start + 10, :, :] = _garbled_scanline(rng)
    assert vision._looks_like_noise(frame)


def test_looks_like_noise_accepts_a_sharp_real_world_edge():
    """A hard real-world horizontal boundary (a table edge against a wall,
    for example) must never be mistaken for a corrupted band: unlike
    garbage, each side of the edge is internally smooth/correlated, just a
    different brightness than its neighbor."""
    rng = np.random.default_rng(5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:240, :, :] = 30
    frame[240:, :, :] = 220
    noise = _luminance_noise(rng, frame.shape[:2], -8, 8)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert not vision._looks_like_noise(frame)


def test_looks_like_noise_accepts_horizontal_blinds_pattern():
    """A repeating horizontal-stripe background (venetian blinds, a
    striped backdrop) must not be mistaken for a corrupted band either —
    same reasoning as the sharp-edge case above, just repeated."""
    rng = np.random.default_rng(6)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for band_start in range(0, 480, 8):
        frame[band_start:band_start + 4, :, :] = 200
        frame[band_start + 4:band_start + 8, :, :] = 40
    noise = _luminance_noise(rng, frame.shape[:2], -5, 5)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert not vision._looks_like_noise(frame)


def test_looks_like_noise_rejects_channel_decorrelated_garbage_within_roughness_range():
    """The exact real incident NOISE_CROSS_CHANNEL_CORR exists for: a
    physically-connected "USB Camera" on this project's own dev machine
    (isOpened=True, read()=True, frame.size>0 — nothing else about the read
    looked wrong) produced colorful TV-static-like garbage that scored
    *inside* NOISE_ROUGHNESS_RATIO's "this is a real image" range, because
    a real, detailed, high-contrast photo (this booth's actual normal
    workload — checked directly against a batch of this project's own
    FairFace training photos) scores in that same roughness range too.
    What actually separated the real garbage frame from real photos was
    cross-channel correlation: a real image's R/G/B channels move together
    at every edge/texture (same underlying scene luminance driving all
    three); decorrelated sensor/decode garbage does not. This reproduces
    that shape synthetically: three independently generated (not derived
    from one shared luminance field) smoothed random fields, tuned to land
    *below* the roughness threshold on their own — i.e. this frame alone
    proves NOISE_CROSS_CHANNEL_CORR is doing independent work, not just
    duplicating what the roughness check already catches."""
    rng = np.random.default_rng(11)
    h, w = 480, 640
    channels = [
        cv2.resize(rng.integers(60, 180, (h // 32, w // 32), dtype=np.uint8), (w, h),
                   interpolation=cv2.INTER_LINEAR)
        for _ in range(3)
    ]
    frame = np.stack(channels, axis=-1)

    small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
    std = float(small.std())
    row_h_roughness = float(np.abs(small[:, 1:] - small[:, :-1]).mean())
    assert row_h_roughness / std < vision.NOISE_ROUGHNESS_RATIO

    assert vision._looks_like_noise(frame)


def test_looks_like_noise_accepts_real_photos_with_similar_roughness_to_the_garbage_case():
    """The other half of the same real-incident check: real, detailed
    photos of actual people (this booth's real workload) that score in the
    *same* roughness range as the garbage case above must still be
    accepted — proving the fix didn't just lower a threshold until the one
    bad frame failed, at the cost of rejecting real content that happens to
    look similarly "rough" by that measure alone. Uses this project's own
    FairFace training photos (dataset/train/) if present; skipped (not
    failed) if that dataset isn't available in this checkout."""
    import glob
    import random

    files = glob.glob("dataset/train/*.jpg")
    if not files:
        import pytest
        pytest.skip("dataset/train/ not present in this checkout")
    random.Random(0).shuffle(files)
    for path in files[:20]:
        frame = cv2.imread(path)
        if frame is None:
            continue
        assert not vision._looks_like_noise(frame), f"false positive on real photo: {path}"


def test_read_with_warmup_rejects_noise_and_keeps_retrying(monkeypatch):
    monkeypatch.setattr(vision, "OPEN_WARMUP_SLEEP_SEC", 0)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    real = np.zeros((480, 640, 3), dtype=np.uint8)

    class NoiseThenReal:
        def __init__(self):
            self.calls = 0

        def read(self):
            self.calls += 1
            return True, noise if self.calls == 1 else real

    cap = NoiseThenReal()
    assert vision._read_with_warmup(cap, attempts=3) is True
    assert cap.calls == 2


def test_camera_worker_treats_mid_stream_noise_as_a_dropped_frame(monkeypatch):
    """The ongoing read loop (not just initial open) must also reject a
    frame that looks like garbage — this was the actual live bug: three
    already-"online" cameras kept streaming pure noise because nothing
    ever re-validated frame content after the initial connect. Drives the
    real CameraWorker.run() loop (in a background thread) rather than
    reimplementing its logic, so this actually exercises the production
    wiring, not just _looks_like_noise in isolation."""
    rng = np.random.default_rng(2)
    noise = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)

    class AlwaysNoise:
        def isOpened(self):
            return True

        def read(self):
            return True, noise

        def release(self):
            pass

    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: AlwaysNoise())
    model = type("Model", (), {"names": {0: "person"}})()
    statuses = []
    worker = vision.CameraWorker(
        "CAM-1", 0, model, [],
        on_status=lambda cam_id, status, msg: statuses.append(status),
        ai_worker=vision.AIWorker(),
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "offline" not in statuses and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        worker._running = False
        worker._stop_event.set()
        thread.join(timeout=2)

    assert "offline" in statuses


def test_camera_worker_survives_a_native_opencv_exception_from_read(monkeypatch):
    """Real bug caught live (multi-camera session): cap.read() can raise a
    native cv2.error ("Unknown C++ exception from OpenCV code") during
    concurrent capture. Previously this propagated straight out of
    CameraWorker.run(), which Python's threading module logs
    ("Exception in thread ...") and then lets die -- permanently, since
    nothing ever recreates that thread, so the camera stayed offline until
    the whole process was restarted. It must instead be treated exactly
    like an ordinary failed read (ok=False), i.e. counted by the existing
    fail_count/FAIL_THRESHOLD path and eventually reported offline, WITHOUT
    killing the worker thread -- same production-wiring style as
    test_camera_worker_treats_mid_stream_noise_as_a_dropped_frame."""

    class AlwaysRaises:
        def isOpened(self):
            return True

        def read(self):
            raise cv2.error("Unknown C++ exception from OpenCV code")

        def release(self):
            pass

    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: AlwaysRaises())
    model = type("Model", (), {"names": {0: "person"}})()
    statuses = []
    worker = vision.CameraWorker(
        "CAM-1", 0, model, [],
        on_status=lambda cam_id, status, msg: statuses.append(status),
        ai_worker=vision.AIWorker(),
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "offline" not in statuses and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        worker._running = False
        worker._stop_event.set()
        thread.join(timeout=2)

    # Reaching "offline" is only possible if the fail_count/FAIL_THRESHOLD
    # loop actually ran to completion (~20 iterations) despite every read()
    # raising -- if the exception had killed the thread (the pre-fix bug),
    # this status would never be appended and the loop above would time out.
    assert "offline" in statuses
    assert not thread.is_alive()  # clean shutdown via _running/_stop_event, not a crash


def test_camera_worker_treats_banded_garbage_as_a_dropped_frame(monkeypatch):
    """Same production-wiring check as
    test_camera_worker_treats_mid_stream_noise_as_a_dropped_frame, but for a
    *partial*-garbage frame (a corrupted band among otherwise-normal rows)
    — the reported bug's actual symptom (camera card green/online, picture
    shows corrupted/horizontal-artifact frames) rather than the
    whole-frame-noise case that test already covers."""
    rng = np.random.default_rng(7)
    frame = _real_looking_gradient_frame(rng)
    frame[200:230, :, :] = _garbled_scanline(rng)

    class AlwaysBanded:
        def isOpened(self):
            return True

        def read(self):
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: AlwaysBanded())
    model = type("Model", (), {"names": {0: "person"}})()
    statuses = []
    delivered_frames = []
    worker = vision.CameraWorker(
        "CAM-1", 0, model, [],
        on_status=lambda cam_id, status, msg: statuses.append(status),
        on_frame=lambda cam_id, f: delivered_frames.append(f),
        ai_worker=vision.AIWorker(),
    )

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while "offline" not in statuses and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        worker._running = False
        worker._stop_event.set()
        thread.join(timeout=2)

    assert "offline" in statuses
    # The banded frame must never have been handed to on_frame as if it
    # were a real, displayable frame.
    assert not delivered_frames
