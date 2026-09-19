"""Camera Gateway — common interface every camera adapter implements.

Phase 1 of the MongDee multi-camera person-tracking pipeline (see
MongDee_Master_Prompt.md, sections 2-4, 31-33, 46). Scope of this module is
deliberately narrow: connect to a camera, decode frames, survive
disconnects, and report health — nothing about detection/tracking/re-id
lives here (that starts one layer up, in a not-yet-built "Frame
Processing" stage). Downstream code depends only on CameraSource /
CameraGateway, never on a concrete protocol class, so adding a new camera
protocol never touches AI code (Master Prompt section 3: "ห้ามให้ AI Core
ผูกกับกล้องตัวใดตัวหนึ่งโดยตรง").

Sized for the MongDee booth deployment described in the project README,
not literal enterprise CCTV: a handful of cameras per booth, CPU inference
(this dev machine has no CUDA), reconnect/backoff tuned for a LAN, not a
WAN of hundreds of cameras. See camera/README.md for details.
"""

from __future__ import annotations

import dataclasses
import enum

import cv2
import numpy as np


class CameraProtocol(str, enum.Enum):
    USB = "usb"
    RTSP = "rtsp"
    HTTP = "http"
    HLS = "hls"
    ONVIF = "onvif"


class CameraStatus(str, enum.Enum):
    CONNECTING = "connecting"
    ONLINE = "online"
    OFFLINE = "offline"
    STOPPED = "stopped"


class CameraConnectionError(RuntimeError):
    """Raised by a CameraSource when open()/read() fails in a way the
    gateway should treat as a disconnect (not a programming error) —
    triggers the reconnect/backoff policy in camera.gateway."""


def redact_url(url: str) -> str:
    """Strip embedded credentials before a URL ever reaches a log line or
    exception message (Master Prompt section 34: "ห้ามเก็บ username/password
    ของกล้องไว้ใน log")."""
    scheme_sep = url.find("://")
    if scheme_sep == -1:
        return url
    scheme = url[: scheme_sep + 3]
    rest = url[scheme_sep + 3 :]
    creds, sep, host_part = rest.partition("@")
    return f"{scheme}***@{host_part}" if sep else url


def enable_hw_video_acceleration(cap: "cv2.VideoCapture") -> None:
    """Ask the FFmpeg backend to decode compressed video (H.264/H.265) on
    whatever GPU decode block the OS exposes instead of doing it in software
    on the CPU — this is what actually costs CPU for rtsp/hls, not the
    post-decode resize/AI work already handled elsewhere.

    `cv2.VIDEO_ACCELERATION_ANY` is vendor-agnostic: it goes through the OS's
    own hardware-decode API (D3D11VA/DXVA2 on Windows, VAAPI/VDPAU/CUDA on
    Linux), which NVIDIA, AMD and Intel GPUs all implement, so there is no
    per-vendor branch to maintain here. Best-effort by design: on an older
    OpenCV build without these properties, or a machine with no usable GPU
    decoder, `cap.set()` just returns False / FFmpeg silently keeps using
    software decode — capture still works either way, we only miss the
    CPU-offload opportunity.
    """
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_HW_DEVICE, -1)
    except (AttributeError, cv2.error):
        pass


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str = ""
    protocol: CameraProtocol = CameraProtocol.USB
    location: str = ""
    enabled: bool = True

    # USB
    device_index: int = 0

    # RTSP / ONVIF (ONVIF resolves to an rtsp:// url at connect time, then
    # reuses this same adapter — see camera/onvif.py)
    url: str = ""
    username: str = ""
    password: str = ""

    # ONVIF device service (used to resolve `url` when `url` is blank)
    host: str = ""
    port: int = 80
    onvif_path: str = "/onvif/device_service"
    profile_token: str = ""  # blank = use the first profile GetProfiles returns

    # HTTP
    http_mode: str = "mjpeg"  # "mjpeg" (multipart push stream) or "snapshot" (poll a still-image URL)

    # capture behaviour
    request_width: int = 0  # 0 = accept the camera's native resolution (USB only)
    request_height: int = 0
    processing_fps: float = 10.0  # gateway forwards at most this many frames/sec to consumers
    connect_timeout_sec: float = 8.0
    read_timeout_sec: float = 5.0
    hw_accel: bool = True  # rtsp/hls: decode compressed video on GPU when available, see enable_hw_video_acceleration()

    # reconnect policy (Master Prompt section 32)
    fail_threshold: int = 20  # consecutive failed reads before declaring OFFLINE and reconnecting
    backoff_initial_sec: float = 1.0
    backoff_max_sec: float = 30.0
    backoff_multiplier: float = 2.0

    @staticmethod
    def from_dict(d: dict) -> "CameraConfig":
        d = dict(d)
        cam_id = d.get("id")
        if not cam_id:
            raise ValueError("camera config missing required field 'id'")
        proto = d.get("protocol")
        if not proto:
            raise ValueError(f"camera config {cam_id!r} missing required field 'protocol'")
        try:
            d["protocol"] = CameraProtocol(str(proto).lower())
        except ValueError:
            valid = ", ".join(p.value for p in CameraProtocol)
            raise ValueError(
                f"camera config {cam_id!r} has unknown protocol {proto!r}; expected one of: {valid}"
            ) from None
        d.setdefault("name", cam_id)
        known = {f.name for f in dataclasses.fields(CameraConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"camera config {cam_id!r} has unknown field(s): {unknown}")
        return CameraConfig(**d)


@dataclasses.dataclass
class Frame:
    camera_id: str
    image: np.ndarray  # BGR, native captured resolution — no resizing at this layer
    timestamp: float  # time.time() at the moment this frame was captured
    frame_index: int  # monotonic per-camera counter, starts at 0


class CameraSource:
    """Abstract adapter. One instance = one camera, one physical
    connection. Not thread-safe by itself — CameraGateway owns exactly one
    worker thread per instance and never calls it from two threads at
    once."""

    def __init__(self, config: CameraConfig):
        self.config = config

    def open(self) -> None:
        """Establish the connection. Raise CameraConnectionError on failure."""
        raise NotImplementedError

    def read(self) -> np.ndarray:
        """Return one BGR frame (H, W, 3) uint8. Raise CameraConnectionError
        if a frame could not be obtained (source down, decode failure,
        timeout — the gateway treats all of these as "one failed read")."""
        raise NotImplementedError

    def release(self) -> None:
        """Release underlying resources. Must be safe to call more than
        once and safe to call even if open() was never called or failed."""
        raise NotImplementedError

    def native_resolution(self) -> tuple[int, int] | None:
        """(width, height) once known (usually only after a successful
        open()/read()), else None."""
        return None

    def is_pull_model(self) -> bool:
        """False (default) for a continuous/push source — a live decoder
        feed (RTSP, USB, HLS, MJPEG push) where read() should be called as
        fast as possible even when most results get thrown away, so the
        underlying buffer never accumulates stale frames (Master Prompt
        section 31: keep draining, never queue).

        True for a pull/on-demand source — e.g. an HTTP snapshot poll,
        where every read() is its own network request and there is no
        buffer to drain, so the gateway should sleep between reads instead
        of firing requests far faster than the target FPS."""
        return False
