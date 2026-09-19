"""Camera Gateway — owns one worker thread per configured camera, handles
reconnects with exponential backoff, and hands frames to a callback at a
configured rate. This is the only thing later pipeline stages should talk
to; they never import a protocol adapter directly (Master Prompt section
3: "เมื่อเพิ่มกล้องใหม่ ห้ามแก้ AI Core โดยตรง").

Frame delivery policy (Master Prompt section 31, "Stream Management"):
each camera keeps only its single latest frame. If a consumer is slower
than the camera, old frames are silently dropped rather than queued —
bounded memory, bounded latency, by design ("หาก AI ประมวลผลไม่ทัน ให้ทิ้ง
frame เก่าและใช้ frame ล่าสุด แทนการสะสม queue จน latency สูง").
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from camera.base import (
    CameraConfig,
    CameraConnectionError,
    CameraProtocol,
    CameraSource,
    CameraStatus,
    Frame,
)
from camera.hls import HLSCamera
from camera.http import HTTPCamera
from camera.onvif import ONVIFCamera
from camera.rtsp import RTSPCamera
from camera.usb import USBCamera

logger = logging.getLogger("mongdee.camera.gateway")

_ADAPTERS: dict[CameraProtocol, type[CameraSource]] = {
    CameraProtocol.USB: USBCamera,
    CameraProtocol.RTSP: RTSPCamera,
    CameraProtocol.HTTP: HTTPCamera,
    CameraProtocol.HLS: HLSCamera,
    CameraProtocol.ONVIF: ONVIFCamera,
}

StatusCallback = Callable[[str, CameraStatus, str], None]  # (camera_id, status, message)
FrameCallback = Callable[[Frame], None]


def create_camera_source(config: CameraConfig) -> CameraSource:
    try:
        adapter_cls = _ADAPTERS[config.protocol]
    except KeyError:
        raise ValueError(f"no adapter registered for protocol {config.protocol!r}") from None
    return adapter_cls(config)


class _CameraWorker:
    """Runs the connect -> read-loop -> reconnect cycle for one camera on
    its own thread. Internal to CameraGateway."""

    def __init__(self, config: CameraConfig, on_frame: FrameCallback | None, on_status: StatusCallback | None):
        self.config = config
        self._on_frame = on_frame
        self._on_status = on_status
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._lock = threading.Lock()
        self._latest_frame: Frame | None = None
        self._status = CameraStatus.STOPPED
        self._status_message = "not started"
        self._frame_index = 0
        self._consecutive_failures = 0
        self._last_forward_time = 0.0

    # -- public, thread-safe --------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.config.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            # The worker may be blocked inside a single open()/read() call
            # right now (requests/cv2 have no way to be interrupted
            # mid-call), so the join must outlast the longest such call can
            # possibly take, or it returns while the thread is still alive —
            # letting a late status update from that stale thread land
            # *after* the STOPPED status set below and silently overwrite it.
            join_timeout = max(self.config.connect_timeout_sec, self.config.read_timeout_sec) + 2.0
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                logger.warning(
                    "camera %s: worker thread still alive %.1fs after stop() (stuck in a blocking call?)",
                    self.config.id, join_timeout,
                )
        self._set_status(CameraStatus.STOPPED, "stopped")

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._latest_frame

    def status(self) -> tuple[CameraStatus, str]:
        with self._lock:
            return self._status, self._status_message

    # -- worker thread body -----------------------------------------------------
    def _run(self) -> None:
        backoff = self.config.backoff_initial_sec
        while not self._stop_event.is_set():
            source = create_camera_source(self.config)
            self._set_status(CameraStatus.CONNECTING, "connecting")
            try:
                source.open()
            except CameraConnectionError as exc:
                self._set_status(CameraStatus.OFFLINE, str(exc))
                logger.warning("camera %s: connect failed: %s", self.config.id, exc)
                if self._wait_or_stop(backoff):
                    break
                backoff = min(backoff * self.config.backoff_multiplier, self.config.backoff_max_sec)
                continue

            backoff = self.config.backoff_initial_sec
            self._consecutive_failures = 0
            self._set_status(CameraStatus.ONLINE, "connected")
            self._read_loop(source)
            source.release()
            if self._stop_event.is_set():
                break
            self._set_status(CameraStatus.OFFLINE, self._status_message or "disconnected")
            if self._wait_or_stop(backoff):
                break
            backoff = min(backoff * self.config.backoff_multiplier, self.config.backoff_max_sec)

    def _read_loop(self, source: CameraSource) -> None:
        min_interval = 1.0 / self.config.processing_fps if self.config.processing_fps > 0 else 0.0
        pull_model = source.is_pull_model()
        while not self._stop_event.is_set():
            if pull_model and min_interval:
                # Pull source (e.g. HTTP snapshot poll): sleep until the next
                # frame is actually due instead of firing requests as fast as
                # possible and discarding most of them.
                remaining = min_interval - (time.time() - self._last_forward_time)
                if remaining > 0 and self._stop_event.wait(remaining):
                    return

            try:
                image = source.read()
            except CameraConnectionError as exc:
                self._consecutive_failures += 1
                logger.debug(
                    "camera %s: read failed (%d/%d): %s",
                    self.config.id, self._consecutive_failures, self.config.fail_threshold, exc,
                )
                if self._consecutive_failures >= self.config.fail_threshold:
                    self._status_message = str(exc)
                    return  # gives up on this connection; _run() reconnects with backoff
                continue

            self._consecutive_failures = 0
            now = time.time()
            if not pull_model and now - self._last_forward_time < min_interval:
                continue  # push/continuous source: keep draining the decoder, drop this frame, never queue it
            self._last_forward_time = now
            frame = Frame(camera_id=self.config.id, image=image, timestamp=now, frame_index=self._frame_index)
            self._frame_index += 1
            with self._lock:
                self._latest_frame = frame
            if self._on_frame is not None:
                try:
                    self._on_frame(frame)
                except Exception:
                    logger.exception("camera %s: on_frame callback raised", self.config.id)

    def _wait_or_stop(self, seconds: float) -> bool:
        """Sleep up to `seconds`, waking immediately if stop() is called. Returns True if stopped."""
        return self._stop_event.wait(seconds)

    def _set_status(self, status: CameraStatus, message: str) -> None:
        with self._lock:
            self._status = status
            self._status_message = message
        if self._on_status is not None:
            try:
                self._on_status(self.config.id, status, message)
            except Exception:
                logger.exception("camera %s: on_status callback raised", self.config.id)


class CameraGateway:
    """Owns and supervises every configured camera. See camera/README.md
    for a usage example."""

    def __init__(self, on_frame: FrameCallback | None = None, on_status: StatusCallback | None = None):
        self._on_frame = on_frame
        self._on_status = on_status
        self._workers: dict[str, _CameraWorker] = {}
        self._lock = threading.Lock()

    def add_camera(self, config: CameraConfig, start: bool = True) -> None:
        with self._lock:
            if config.id in self._workers:
                raise ValueError(f"camera id {config.id!r} already registered")
            worker = _CameraWorker(config, self._on_frame, self._on_status)
            self._workers[config.id] = worker
        if start and config.enabled:
            worker.start()

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()

    def start_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            if worker.config.enabled:
                worker.start()

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop()

    def latest_frame(self, camera_id: str) -> Frame | None:
        worker = self._workers.get(camera_id)
        return worker.latest_frame() if worker else None

    def status(self, camera_id: str) -> tuple[CameraStatus, str] | None:
        worker = self._workers.get(camera_id)
        return worker.status() if worker else None

    def all_statuses(self) -> dict[str, tuple[CameraStatus, str]]:
        with self._lock:
            workers = dict(self._workers)
        return {cam_id: w.status() for cam_id, w in workers.items()}

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._workers.keys())
