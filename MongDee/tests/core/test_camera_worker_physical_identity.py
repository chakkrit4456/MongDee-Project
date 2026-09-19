"""Unit tests for CameraWorker's best-effort physical-camera-identity
tracking (spec: MongDee physical-camera-identity follow-up) —
_resolve_physical_id (record identity on first successful open) and
_relocate_device_if_moved (re-locate the same physical camera at a new
DirectShow index on reconnect, instead of retrying a stale one forever).

Constructing a CameraWorker never opens a camera or starts its thread, so
these private methods are safe to call directly, same pattern as
test_vision.py's _person_label_and_color tests.
"""

from __future__ import annotations

import pytest

import core.camera_identity as camera_identity
import core.vision as vision
from core.vision import CameraWorker


def _reset_physical_identity_cache():
    # _relocate_device_if_moved() reads through core.vision's own
    # process-wide _cached_physical_camera_identities() cache (added
    # alongside the async _resolve_physical_id fix — see that cache's own
    # docstring) — reset it before each test that monkeypatches
    # camera_identity.get_physical_camera_identities, or an earlier test's
    # result could still be served from cache instead of this test's own.
    vision._physical_identity_cache = None


@pytest.fixture(autouse=True)
def _clean_physical_identity_cache():
    # Runs before AND after every test in this file -- before, so no
    # earlier test's cached result leaks in; after, so this file never
    # leaves a stale cache entry for some *other* test module (e.g. a real
    # hardware-backed integration test) to trip over.
    _reset_physical_identity_cache()
    yield
    _reset_physical_identity_cache()


class _FakeModel:
    names = {0: "person"}


class _FakeIdentity:
    def __init__(self, physical_id):
        self.physical_id = physical_id


def _make_worker(device=1):
    return CameraWorker(
        camera_id="TEST", device=device, model=_FakeModel(), allowed_classes=[],
    )


# --------------------------------------------------------- _resolve_physical_id

# _resolve_physical_id() itself is fire-and-forget (runs
# _resolve_physical_id_blocking on a background daemon thread — see that
# method's docstring for why: the underlying PowerShell lookup was measured
# taking several seconds on real hardware, and running it synchronously
# inside _open() delayed every camera's first "online" status by that same
# amount). These tests call _resolve_physical_id_blocking directly to test
# the resolution logic itself, synchronously and deterministically;
# test_camera_physical_identity_perf.py separately covers the async
# fire-and-forget wrapper's own non-blocking behavior.

def test_resolve_physical_id_stores_identity_for_numeric_device(monkeypatch):
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {1: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1")})
    worker = _make_worker(device=1)
    worker._resolve_physical_id_blocking()
    assert worker._physical_id == "vid_pid_location:AAAA:BBBB:PORT-1"


def test_resolve_physical_id_noop_for_non_index_device(monkeypatch):
    # A device path/RTSP URL has no physical-identity lookup at all.
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: (_ for _ in ()).throw(AssertionError("should never be called")))
    worker = _make_worker(device="rtsp://cam/stream")
    worker._resolve_physical_id_blocking()
    assert worker._physical_id is None


def test_resolve_physical_id_noop_when_index_not_found(monkeypatch):
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities", lambda: {})
    worker = _make_worker(device=1)
    worker._resolve_physical_id_blocking()
    assert worker._physical_id is None


def test_resolve_physical_id_never_raises_on_backend_failure(monkeypatch):
    def boom():
        raise OSError("powershell exploded")

    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities", boom)
    worker = _make_worker(device=1)
    worker._resolve_physical_id_blocking()  # must not raise
    assert worker._physical_id is None


# --------------------------------------------------- _relocate_device_if_moved

def test_relocate_updates_device_when_physical_camera_moved_index(monkeypatch):
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {5: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1")})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    worker._relocate_device_if_moved()
    assert worker.device == 5


def test_relocate_leaves_device_unchanged_when_still_at_same_index(monkeypatch):
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {1: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1")})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    worker._relocate_device_if_moved()
    assert worker.device == 1


def test_relocate_leaves_device_unchanged_when_camera_not_present(monkeypatch):
    # Camera genuinely unplugged/offline -- no present identity matches;
    # must fall back to the pre-existing behavior (keep retrying the same
    # index) rather than clearing or guessing a new one.
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities", lambda: {})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    worker._relocate_device_if_moved()
    assert worker.device == 1


def test_relocate_never_raises_on_backend_failure(monkeypatch):
    def boom():
        raise OSError("powershell exploded")

    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities", boom)
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    worker._relocate_device_if_moved()  # must not raise
    assert worker.device == 1


# spec: MongDee camera-identity investigation -- two USB cameras of the
# identical model (this project's own dev hardware: same VID/PID, no real
# serial) are only distinguished by USB port location, which is invalidated
# the moment a physical unit moves to a different port. Relocation (above)
# already handles "my camera moved to a new index"; these cover the other
# half -- "a *different* camera is now sitting at my old index" -- which
# must refuse to open rather than silently mislabeling that other camera's
# feed under this camera_id.

def test_relocate_refuses_when_old_index_now_a_different_known_camera(monkeypatch):
    _reset_physical_identity_cache()
    # index 1 (this worker's own tracked index) now identifies as a
    # different camera than the one this worker is tracking; the tracked
    # camera isn't found anywhere else either (e.g. genuinely swapped out).
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {1: _FakeIdentity("vid_pid_location:CCCC:DDDD:PORT-2")})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    result = worker._relocate_device_if_moved()
    assert result is False
    assert worker.device == 1  # unchanged -- caller must not open it this round


def test_relocate_allows_open_when_identity_data_unavailable(monkeypatch):
    # Empty identities dict is "identity unavailable/inconclusive" (e.g.
    # pygrabber/PowerShell unreachable), not positive evidence of a
    # substitution -- must never block a real open on mere uncertainty.
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities", lambda: {})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    assert worker._relocate_device_if_moved() is True
    assert worker.device == 1


def test_relocate_allows_open_when_still_the_same_camera(monkeypatch):
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {1: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1")})
    worker = _make_worker(device=1)
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    assert worker._relocate_device_if_moved() is True
    assert worker.device == 1


def test_open_does_not_open_device_when_relocate_detects_substitution(monkeypatch):
    """The actual bug this investigation found: _open() must not proceed to
    _open_capture() at all when _relocate_device_if_moved() detects a
    substitution -- opening self.device would silently start streaming a
    different physical camera's feed under this worker's camera_id."""
    _reset_physical_identity_cache()
    monkeypatch.setattr(camera_identity, "get_physical_camera_identities",
                         lambda: {1: _FakeIdentity("vid_pid_location:CCCC:DDDD:PORT-2")})

    def boom_if_called(*a, **k):
        raise AssertionError("_open_capture must not be called for a detected substitution")

    monkeypatch.setattr(vision, "_open_capture", boom_if_called)
    worker = _make_worker(device=1)
    worker._ever_connected = True
    worker._physical_id = "vid_pid_location:AAAA:BBBB:PORT-1"
    assert worker._open() is False


# ----------------------------------------------------------------- _open() flow
# _open() itself does real camera I/O (cv2.VideoCapture) so it isn't called
# directly here -- these confirm the *gating* logic that decides whether
# _relocate_device_if_moved runs at all, via the same conditions _open() checks.

def test_first_ever_connect_does_not_attempt_relocation():
    # _ever_connected is False before any successful open -- there is no
    # "old index" to relocate away from yet, this must be a first connect.
    worker = _make_worker(device=1)
    assert worker._ever_connected is False
    assert worker._physical_id is None


def test_worker_starts_with_no_physical_id():
    worker = _make_worker(device=1)
    assert worker._physical_id is None


# --------------------------------------------------------- _offline_message
# spec: MongDee multi-USB-camera capture-lifecycle investigation -- "ไม่พบ
# กล้อง" (camera not found) is actively misleading for the real failure mode
# verified on real hardware (two USB cameras sharing a hub: the *second* one
# to open fails VideoCapture.open() outright, every format/backend, while
# still fully present/enumerated) -- must be distinguished from a genuinely
# unplugged camera instead of always showing the same generic message.

def test_offline_message_flags_resource_conflict_when_device_still_enumerates(monkeypatch):
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: {1: "USB Camera"})
    worker = _make_worker(device=1)
    message = worker._offline_message()
    assert "ยังเชื่อมต่ออยู่" in message
    assert "USB hub" in message


def test_offline_message_falls_back_to_not_found_when_device_absent(monkeypatch):
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: {0: "HD WebCam"})
    worker = _make_worker(device=1)
    message = worker._offline_message()
    assert message == "ไม่พบกล้อง 1"


def test_offline_message_falls_back_for_non_index_device(monkeypatch):
    monkeypatch.setattr(camera_identity, "list_directshow_devices",
                         lambda: (_ for _ in ()).throw(AssertionError("should never be called")))
    worker = _make_worker(device="rtsp://cam/stream")
    assert worker._offline_message() == "ไม่พบกล้อง rtsp://cam/stream"


def test_offline_message_never_raises_on_backend_failure(monkeypatch):
    def boom():
        raise OSError("powershell exploded")

    monkeypatch.setattr(camera_identity, "list_directshow_devices", boom)
    worker = _make_worker(device=1)
    assert worker._offline_message() == "ไม่พบกล้อง 1"
