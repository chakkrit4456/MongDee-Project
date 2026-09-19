"""Regression tests for the multi-USB-camera "DEVICE INDEX COLLISION" root
cause — spec: MongDee multi-USB-camera root-cause repair master prompt.

Confirmed on real hardware in this session: two independent
cv2.VideoCapture objects opened on the *same* physical camera index both
succeed and both stream valid, live video (nothing about capture itself
fails) — so binding two different camera_ids to the same device silently
produced two camera cards showing the identical physical camera, which
reads exactly like a cross-camera frame-routing bug without being one.
These tests cover the two places that binding could happen:
BoothManager.add_camera (the /settings "add camera" action) and
dedupe_camera_devices (BoothManager.__init__'s own defense-in-depth,
covering both web_server.py's --cameras flag and app.py's discovered
device list).
"""

from __future__ import annotations

import threading

import pytest

import web.booth_manager as booth_manager_module
from web.booth_manager import BoothManager, dedupe_camera_devices


@pytest.fixture(autouse=True)
def _no_real_physical_identity_lookup(monkeypatch):
    # dedupe_camera_devices also checks core.camera_identity's physical_id
    # (see its own docstring) as an *additional* layer on top of the raw
    # device-value check these tests target -- without this, every test
    # below would shell out to the real `Get-PnpDevice`/PowerShell on
    # Windows (slow, and dependent on whatever's actually plugged into this
    # specific machine) for no benefit to what they're actually testing.
    # Tests that specifically exercise the physical_id layer override this.
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {})


# --------------------------------------------------- dedupe_camera_devices

def test_dedupe_keeps_first_occurrence_and_drops_the_rest():
    result = dedupe_camera_devices([("CAM-1", 0), ("CAM-2", 0), ("CAM-3", 2)])
    assert result == [("CAM-1", 0), ("CAM-3", 2)]


def test_dedupe_catches_int_vs_str_of_the_same_index():
    result = dedupe_camera_devices([("CAM-1", 0), ("CAM-2", "0")])
    assert result == [("CAM-1", 0)]


def test_dedupe_never_touches_genuinely_different_devices():
    devices = [("CAM-1", 0), ("CAM-2", 1), ("CAM-3", 2)]
    assert dedupe_camera_devices(devices) == devices


def test_dedupe_handles_device_paths_case_insensitively():
    result = dedupe_camera_devices([("CAM-1", "/dev/video0"), ("CAM-2", "/DEV/VIDEO0")])
    assert result == [("CAM-1", "/dev/video0")]


def test_dedupe_empty_list():
    assert dedupe_camera_devices([]) == []


# ------------------------------- dedupe_camera_devices: physical_id layer
# (MongDee physical-camera-identity follow-up — catches two *different* raw
# index values that are actually the same physical camera, e.g. after
# Windows re-enumerated between when each index was resolved.)

class _FakeIdentity:
    def __init__(self, physical_id):
        self.physical_id = physical_id


def test_dedupe_catches_same_physical_camera_at_different_indices(monkeypatch):
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {
        0: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),
        2: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),  # same camera, reappeared at index 2
    })
    result = dedupe_camera_devices([("CAM-1", 0), ("CAM-2", 2)])
    assert result == [("CAM-1", 0)]


def test_dedupe_keeps_genuinely_different_physical_cameras(monkeypatch):
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {
        0: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),
        1: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-2"),
    })
    devices = [("CAM-1", 0), ("CAM-2", 1)]
    assert dedupe_camera_devices(devices) == devices


def test_dedupe_ignores_physical_id_for_non_index_devices(monkeypatch):
    # A device path/RTSP URL never has a physical_id lookup performed on it
    # (only numeric indices do) -- must never crash or misbehave.
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {})
    devices = [("CAM-1", "/dev/video0"), ("CAM-2", "rtsp://cam/stream")]
    assert dedupe_camera_devices(devices) == devices


# --------------------------------------------------------- add_camera

def _bare_booth_manager(camera_devices: dict):
    """The duplicate-device check in add_camera fires before any worker/
    model/catalog access, so a minimal bare instance is enough to exercise
    the rejection path without a real camera or YOLO model. The extra dicts/
    set below are only touched on the *success* path (past the rejection
    check) — present so a test can let add_camera actually succeed too, as
    long as it also stubs _make_worker/_load_tripwire (see
    test_add_camera_allows_genuinely_different_physical_camera) so nothing
    tries to open a real camera or start a real thread."""
    bm = object.__new__(BoothManager)
    bm._lock = threading.Lock()
    bm.camera_devices = dict(camera_devices)
    bm._next_camera_num = len(camera_devices) + 1
    bm.camera_ids = list(camera_devices.keys())
    bm.camera_status = {cid: {"status": "unknown", "message": ""} for cid in camera_devices}
    bm._camera_enabled = {cid: True for cid in camera_devices}
    bm._person_tracks = {cid: [] for cid in camera_devices}
    bm._product_detections = {cid: [] for cid in camera_devices}
    bm._ignored_usb_indices = set()
    bm.workers = {}
    return bm


def test_add_camera_rejects_a_device_already_bound_to_another_camera_id():
    bm = _bare_booth_manager({"CAM-1": 0})
    with pytest.raises(ValueError, match="CAM-1"):
        bm.add_camera(0)


def test_add_camera_rejects_int_vs_str_of_the_same_bound_device():
    bm = _bare_booth_manager({"CAM-1": 0})
    with pytest.raises(ValueError):
        bm.add_camera("0")


def test_add_camera_rejects_device_path_case_insensitively():
    bm = _bare_booth_manager({"CAM-1": "/dev/video0"})
    with pytest.raises(ValueError):
        bm.add_camera("/DEV/VIDEO0")


def test_add_camera_rejects_same_physical_camera_at_a_different_index(monkeypatch):
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {
        0: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),
        2: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),  # same camera, now at index 2
    })
    bm = _bare_booth_manager({"CAM-1": 0})
    with pytest.raises(ValueError, match="CAM-1"):
        bm.add_camera(2)


def test_add_camera_allows_genuinely_different_physical_camera(monkeypatch):
    monkeypatch.setattr(booth_manager_module, "get_physical_camera_identities", lambda: {
        0: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-1"),
        1: _FakeIdentity("vid_pid_location:AAAA:BBBB:PORT-2"),
    })
    bm = _bare_booth_manager({"CAM-1": 0})
    bm._make_worker = lambda camera_id, device: type("FakeWorker", (), {"start": lambda self: None})()
    bm._load_tripwire = lambda camera_id: None
    camera_id = bm.add_camera(1)
    assert camera_id != "CAM-1"
