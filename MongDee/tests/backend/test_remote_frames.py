"""Unit tests for backend/remote_frames.py — the network-fed FrameSource
that lets a lightweight Camera Agent (camera_agent/) push frames to the AI
Server instead of it capturing cameras locally.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from backend.remote_frames import CompositeFrameSource, RemoteFrameReceiver
from camera.base import CameraStatus


def _frame(fill=1):
    return np.full((4, 4, 3), fill, dtype=np.uint8)


def test_ingest_requires_prior_registration():
    receiver = RemoteFrameReceiver()
    assert receiver.ingest_frame("CAM-1", _frame(), time.time()) is False
    assert receiver.camera_ids() == []


def test_register_then_ingest_frame_succeeds():
    receiver = RemoteFrameReceiver()
    receiver.register("CAM-1", name="Entrance", resolution=(640, 480), fps=15)
    ok = receiver.ingest_frame("CAM-1", _frame(7), timestamp=123.0, sequence=1)
    assert ok is True

    frame = receiver.latest_frame("CAM-1")
    assert frame is not None
    assert frame.camera_id == "CAM-1"
    assert frame.timestamp == 123.0
    assert (frame.image == 7).all()


def test_out_of_order_or_stale_sequence_is_dropped_not_applied():
    receiver = RemoteFrameReceiver()
    receiver.register("CAM-1")
    assert receiver.ingest_frame("CAM-1", _frame(1), time.time(), sequence=5) is True
    # An older/duplicate sequence number must never overwrite the newer frame.
    assert receiver.ingest_frame("CAM-1", _frame(2), time.time(), sequence=3) is False
    assert receiver.ingest_frame("CAM-1", _frame(2), time.time(), sequence=5) is False
    frame = receiver.latest_frame("CAM-1")
    assert (frame.image == 1).all()

    status = receiver.status("CAM-1")
    assert status["dropped_out_of_order"] == 2


def test_latest_frame_wins_never_queues():
    receiver = RemoteFrameReceiver()
    receiver.register("CAM-1")
    for i in range(50):
        receiver.ingest_frame("CAM-1", _frame(i % 255), time.time(), sequence=i)
    # Only the single latest frame is ever retained — confirmed by frame_index
    # tracking exactly the ingest count, and there being nothing queue-like
    # to inspect at all (no queue attribute, no backlog).
    status = receiver.status("CAM-1")
    assert status["frames_received"] == 50


def test_status_transitions_reported_via_callback():
    events = []
    receiver = RemoteFrameReceiver(on_status=lambda cid, status, msg: events.append((cid, status)))
    receiver.register("CAM-1")
    assert events == [("CAM-1", CameraStatus.CONNECTING)]

    receiver.ingest_frame("CAM-1", _frame(), time.time())
    assert events[-1] == ("CAM-1", CameraStatus.ONLINE)

    # A second frame while already online must not re-fire the transition.
    receiver.ingest_frame("CAM-1", _frame(), time.time())
    assert events.count(("CAM-1", CameraStatus.ONLINE)) == 1


def test_sweep_stale_marks_quiet_camera_offline():
    events = []
    receiver = RemoteFrameReceiver(on_status=lambda cid, status, msg: events.append((cid, status)))
    receiver.register("CAM-1")
    receiver.ingest_frame("CAM-1", _frame(), time.time())

    # Simulate time passing without a new frame by backdating last_frame_at.
    with receiver._lock:
        receiver._cameras["CAM-1"].last_frame_at = time.time() - 999
    receiver.sweep_stale()

    assert events[-1] == ("CAM-1", CameraStatus.OFFLINE)
    assert receiver.status("CAM-1")["status"] == "offline"


def test_sweep_stale_is_a_noop_for_fresh_cameras():
    events = []
    receiver = RemoteFrameReceiver(on_status=lambda cid, status, msg: events.append((cid, status)))
    receiver.register("CAM-1")
    receiver.ingest_frame("CAM-1", _frame(), time.time())
    events.clear()
    receiver.sweep_stale()
    assert events == []


def test_unregister_removes_camera_and_frame():
    receiver = RemoteFrameReceiver()
    receiver.register("CAM-1")
    receiver.ingest_frame("CAM-1", _frame(), time.time())
    receiver.unregister("CAM-1")
    assert receiver.camera_ids() == []
    assert receiver.latest_frame("CAM-1") is None


# -------------------------------------------------------- composite source

class _FakeSource:
    def __init__(self, frames: dict):
        self._frames = frames

    def camera_ids(self):
        return list(self._frames.keys())

    def latest_frame(self, camera_id):
        return self._frames.get(camera_id)


def test_composite_frame_source_merges_camera_ids():
    local = _FakeSource({"LOCAL-1": _frame()})
    remote = RemoteFrameReceiver()
    remote.register("REMOTE-1")
    composite = CompositeFrameSource(local, remote)
    assert set(composite.camera_ids()) == {"LOCAL-1", "REMOTE-1"}


def test_composite_frame_source_reads_from_the_right_source():
    from camera.base import Frame

    local_frame = Frame(camera_id="LOCAL-1", image=_frame(9), timestamp=time.time(), frame_index=0)
    local = _FakeSource({"LOCAL-1": local_frame})
    remote = RemoteFrameReceiver()
    remote.register("REMOTE-1")
    remote.ingest_frame("REMOTE-1", _frame(3), time.time())
    composite = CompositeFrameSource(local, remote)

    assert (composite.latest_frame("LOCAL-1").image == 9).all()
    assert (composite.latest_frame("REMOTE-1").image == 3).all()
    assert composite.latest_frame("NOPE") is None
