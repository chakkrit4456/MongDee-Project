"""Frame source fed by network-connected Camera Agents (see
camera_agent/client.py) instead of local, in-process camera capture — this
is what lets a Camera Agent run on a completely different, GPU-less
machine from the AI Server (MongDee_Cloud_Vercel_Remote_AI_Server_Master_
Prompt.md sections 4, 32, 57, 58).

RemoteFrameReceiver implements the same FrameSource protocol
vision.pipeline.DetectionPipeline already accepts (camera_ids() +
latest_frame()) — camera.gateway.CameraGateway satisfies the same protocol
for locally-attached cameras, so CompositeFrameSource below can mix both
kinds of camera in one running AI Server without the pipeline ever needing
to know or care where a camera's frames physically come from.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable

from camera.base import CameraStatus, Frame

logger = logging.getLogger("mongdee.backend.remote_frames")

# No frame received in this long -> the camera agent (or its network path)
# is presumed down; report OFFLINE rather than silently keep serving a
# frozen last frame (section 43: Health Monitoring states).
STALE_AFTER_SEC = 5.0
WATCHDOG_TICK_SEC = 1.0

StatusCallback = Callable[[str, CameraStatus, str], None]


@dataclasses.dataclass
class _RemoteCameraState:
    name: str = ""
    resolution: tuple[int, int] | None = None
    requested_fps: float | None = None
    codec: str = ""
    registered_at: float = 0.0
    last_sequence: int | None = None
    last_frame_at: float = 0.0
    frame_index: int = 0
    dropped_out_of_order: int = 0
    status: CameraStatus = CameraStatus.CONNECTING


class RemoteFrameReceiver:
    """Thread-safe. API route handlers call register()/ingest_frame() from
    (potentially many concurrent) request-handling threads/tasks; the
    vision pipeline's worker thread calls camera_ids()/latest_frame()."""

    def __init__(self, on_status: StatusCallback | None = None):
        self._lock = threading.Lock()
        self._cameras: dict[str, _RemoteCameraState] = {}
        self._frames: dict[str, Frame] = {}
        self._on_status = on_status or (lambda *a: None)

    # ------------------------------------------------------------ writers
    def register(self, camera_id: str, name: str = "", resolution: tuple[int, int] | None = None,
                 fps: float | None = None, codec: str = "") -> None:
        """Camera Agent registration (section 32) — idempotent, so an
        agent re-registering after its own reconnect just refreshes
        metadata instead of erroring."""
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                state = _RemoteCameraState(registered_at=time.time())
                self._cameras[camera_id] = state
            state.name = name or state.name or camera_id
            state.resolution = resolution or state.resolution
            state.requested_fps = fps if fps is not None else state.requested_fps
            state.codec = codec or state.codec
        self._emit_status(camera_id, CameraStatus.CONNECTING, "registered, waiting for frames")

    def ingest_frame(self, camera_id: str, image, timestamp: float, sequence: int | None = None) -> bool:
        """Accepts one frame from a registered camera. Latest-Frame-Wins
        (section 5): a frame at or before the last accepted sequence
        number is dropped rather than queued or applied out of order —
        the network/agent may redeliver or reorder under load, and an
        older frame must never overwrite a newer one that already arrived.
        Returns False if the camera isn't registered or the frame was
        dropped as stale/out-of-order."""
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return False
            if sequence is not None and state.last_sequence is not None and sequence <= state.last_sequence:
                state.dropped_out_of_order += 1
                return False
            state.last_sequence = sequence if sequence is not None else (state.last_sequence or 0) + 1
            state.frame_index += 1
            state.last_frame_at = time.time()
            was_offline = state.status != CameraStatus.ONLINE
            state.status = CameraStatus.ONLINE
            self._frames[camera_id] = Frame(
                camera_id=camera_id, image=image, timestamp=timestamp, frame_index=state.frame_index,
            )
        if was_offline:
            self._emit_status(camera_id, CameraStatus.ONLINE, "receiving frames")
        return True

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)
            self._frames.pop(camera_id, None)
        self._emit_status(camera_id, CameraStatus.STOPPED, "unregistered")

    # ------------------------------------------------------------ readers
    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._cameras.keys())

    def latest_frame(self, camera_id: str) -> Frame | None:
        with self._lock:
            return self._frames.get(camera_id)

    def status(self, camera_id: str) -> dict | None:
        with self._lock:
            state = self._cameras.get(camera_id)
            if state is None:
                return None
            return {
                "camera_id": camera_id, "name": state.name, "status": state.status.value,
                "resolution": list(state.resolution) if state.resolution else None,
                "requested_fps": state.requested_fps, "codec": state.codec,
                "frames_received": state.frame_index, "dropped_out_of_order": state.dropped_out_of_order,
                "last_frame_age_sec": (time.time() - state.last_frame_at) if state.last_frame_at else None,
            }

    def all_statuses(self) -> dict[str, dict]:
        return {cid: self.status(cid) for cid in self.camera_ids()}

    # ---------------------------------------------------------- watchdog
    def sweep_stale(self) -> None:
        """Call periodically from a background thread (see
        RemoteFrameHealthWatchdog) — flips a camera that stopped sending
        frames to OFFLINE instead of leaving stale video/status forever."""
        now = time.time()
        went_offline = []
        with self._lock:
            for camera_id, state in self._cameras.items():
                if state.status == CameraStatus.ONLINE and state.last_frame_at and (now - state.last_frame_at) > STALE_AFTER_SEC:
                    state.status = CameraStatus.OFFLINE
                    went_offline.append(camera_id)
        for camera_id in went_offline:
            self._emit_status(camera_id, CameraStatus.OFFLINE, f"no frame received in over {STALE_AFTER_SEC:.0f}s")

    def _emit_status(self, camera_id: str, status: CameraStatus, message: str) -> None:
        try:
            self._on_status(camera_id, status, message)
        except Exception:
            logger.exception("camera %s: on_status callback raised", camera_id)


class RemoteFrameHealthWatchdog(threading.Thread):
    """Owns the periodic sweep_stale() tick so RemoteFrameReceiver itself
    stays a plain, easily-unit-tested object with no thread of its own."""

    def __init__(self, receiver: RemoteFrameReceiver, tick_sec: float = WATCHDOG_TICK_SEC):
        super().__init__(daemon=True, name="mongdee-remote-frame-watchdog")
        self._receiver = receiver
        self._tick_sec = tick_sec
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._tick_sec):
            self._receiver.sweep_stale()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2)


class CompositeFrameSource:
    """Merges any number of FrameSource-protocol objects (e.g. a local
    camera.gateway.CameraGateway for on-machine webcams + a
    RemoteFrameReceiver for network Camera Agents) into one, so
    DetectionPipeline never needs to know or care where a camera's frames
    physically come from (section 57: Server Independence). Camera IDs
    must be unique across all sources — first match wins."""

    def __init__(self, *sources):
        self._sources = list(sources)

    def camera_ids(self) -> list[str]:
        seen: list[str] = []
        for source in self._sources:
            for cid in source.camera_ids():
                if cid not in seen:
                    seen.append(cid)
        return seen

    def latest_frame(self, camera_id: str) -> Frame | None:
        for source in self._sources:
            frame = source.latest_frame(camera_id)
            if frame is not None:
                return frame
        return None
