"""Camera Agent runtime — captures locally via camera.gateway.CameraGateway
(the same robust, reconnecting, torch-free capture code used everywhere
else in this project) and pushes each camera's latest frame to a Remote AI
Server over HTTP (MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md
sections 4, 5, 32, 33, 59).

One sender thread per camera, each with its own single-slot "pending
frame" mailbox: CameraGateway already keeps only the latest captured frame
per camera (see camera/gateway.py's frame delivery policy) and this client
applies the exact same "Latest Frame Wins" rule at the network boundary —
if the previous push to the server hasn't finished yet, a newly captured
frame simply replaces whatever was still waiting to be sent, never queues.
This means a slow or temporarily unreachable server degrades to a lower
effective frame rate instead of ever blocking capture or growing memory
(sections 33, 59, 93 rule 6).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
import requests

from camera.base import CameraStatus, Frame
from camera.gateway import CameraGateway
from camera_agent.config import AgentConfig

logger = logging.getLogger("mongdee.camera_agent.client")

StatusCallback = Callable[[str, CameraStatus, str], None]


class CameraAgentClient:
    def __init__(self, config: AgentConfig, on_status: StatusCallback | None = None):
        self._config = config
        self._on_status = on_status or (lambda *a: None)

        self._session = requests.Session()
        if config.api_key:
            self._session.headers["X-API-Key"] = config.api_key

        self._pending: dict[str, Frame] = {}
        self._pending_lock = threading.Lock()
        self._pending_events: dict[str, threading.Event] = {cam.id: threading.Event() for cam in config.cameras}
        self._stop_event = threading.Event()
        self._sender_threads: dict[str, threading.Thread] = {}
        self._sequence: dict[str, int] = {}

        self._gateway = CameraGateway(on_frame=self._on_frame, on_status=self._on_camera_status)
        for cam in config.cameras:
            self._gateway.add_camera(cam, start=False)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._stop_event.clear()
        for cam in self._config.cameras:
            thread = threading.Thread(
                target=self._sender_loop, args=(cam.id,), daemon=True, name=f"camera-agent-sender-{cam.id}",
            )
            self._sender_threads[cam.id] = thread
            thread.start()
        self._gateway.start_all()

    def stop(self) -> None:
        self._stop_event.set()
        for event in self._pending_events.values():
            event.set()  # wake every sender thread so it notices _stop_event promptly
        self._gateway.stop_all()
        for thread in self._sender_threads.values():
            thread.join(timeout=5)

    def camera_ids(self) -> list[str]:
        return self._gateway.camera_ids()

    def local_status(self, camera_id: str):
        """Local capture health (open/reading), as opposed to whether the
        AI Server currently has this camera registered — see the AI
        Server's own /api/cameras/{id}/status for that."""
        return self._gateway.status(camera_id)

    # ------------------------------------------------------------ capture
    def _on_frame(self, frame: Frame) -> None:
        with self._pending_lock:
            self._pending[frame.camera_id] = frame
        self._pending_events[frame.camera_id].set()

    def _on_camera_status(self, camera_id: str, status: CameraStatus, message: str) -> None:
        self._on_status(camera_id, status, message)

    def _take_pending(self, camera_id: str) -> Frame | None:
        with self._pending_lock:
            return self._pending.pop(camera_id, None)

    # -------------------------------------------------------------- upload
    def _sender_loop(self, camera_id: str) -> None:
        cam_config = next(c for c in self._config.cameras if c.id == camera_id)
        event = self._pending_events[camera_id]
        registered = False
        last_register_attempt = 0.0
        retry_delay = max(0.1, self._config.register_retry_sec)

        while not self._stop_event.is_set():
            if not registered:
                now = time.time()
                if now - last_register_attempt >= retry_delay:
                    last_register_attempt = now
                    registered = self._register(cam_config)
                    if registered:
                        retry_delay = max(0.1, self._config.register_retry_sec)
                    else:
                        retry_delay = min(
                            self._config.retry_backoff_max_sec,
                            retry_delay * self._config.retry_backoff_multiplier,
                        )
                if not registered:
                    self._stop_event.wait(min(0.5, retry_delay))
                    continue

            frame = self._take_pending(camera_id)
            if frame is None:
                event.wait(0.2)  # woken immediately by a new frame; this timeout is just a safety-net poll
                event.clear()
                continue

            if not self._push_frame(camera_id, frame):
                registered = False  # re-register next loop -- covers an AI Server restart losing its state
                retry_delay = max(0.1, self._config.register_retry_sec)
                last_register_attempt = time.time()

    def _register(self, cam_config) -> bool:
        try:
            resp = self._session.post(
                f"{self._config.server_url}/api/cameras/register",
                json={
                    "cameraId": cam_config.id, "name": cam_config.name or cam_config.id,
                    "location": cam_config.location,
                },
                timeout=self._config.push_timeout_sec,
            )
            resp.raise_for_status()
            logger.info("camera %s: registered with AI server at %s", cam_config.id, self._config.server_url)
            return True
        except requests.RequestException as exc:
            logger.warning("camera %s: could not register with AI server (%s) -- will retry", cam_config.id, exc)
            return False

    def _push_frame(self, camera_id: str, frame: Frame) -> bool:
        ok, buf = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
        if not ok:
            logger.warning("camera %s: JPEG encode failed for this frame, skipping", camera_id)
            return True  # not a network/registration problem -- don't force a re-register over it
        seq = self._sequence.get(camera_id, 0) + 1
        try:
            resp = self._session.post(
                f"{self._config.server_url}/api/cameras/{camera_id}/frame",
                files={"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                data={"timestamp": frame.timestamp, "sequence": seq},
                timeout=self._config.push_timeout_sec,
            )
            resp.raise_for_status()
            self._sequence[camera_id] = seq
            return True
        except requests.RequestException as exc:
            logger.warning("camera %s: frame push failed (%s)", camera_id, exc)
            return False
