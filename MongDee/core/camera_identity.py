"""Windows-only helper for telling *what* a cv2.VideoCapture(index) index
currently points at, since OpenCV's Python API exposes no device name,
VID/PID, or instance path of its own.

This exists because a device's *index* is not a stable identity — Windows
re-enumerates video capture devices on reboot, on unplug/replug, and when
a device count changes, so "index 0 is the built-in webcam" can silently
become false the next time a USB camera is plugged in or removed (see
web/booth_manager.py's load_camera_settings and its callers, which use this
module to resolve *names* to the current index instead of trusting a
hardcoded one). A device's DirectShow friendly name is comparatively
stable — it's a property of the driver, not of enumeration order.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
import threading

_VIRTUAL_NAME_MARKERS = ("virtual camera", "obs virtual", "manycam", "droidcam", "snap camera")

# DirectShow's device-enumeration COM objects (what both pygrabber's
# FilterGraph() below and cv2.VideoCapture(index, cv2.CAP_DSHOW) use under
# the hood) are not safely reentrant across threads on Windows -- two
# threads touching DirectShow enumeration/open at the same moment is a
# known cause of hard, unrecoverable process crashes (access violations
# that kill the interpreter with no Python traceback at all, not a
# catchable exception). Real evidence from this project's own dev hardware
# (session hw-validation, 2026-09-18): running two CameraWorker threads
# concurrently -- one calling cv2.VideoCapture(..., CAP_DSHOW) to (re)open
# its camera while the other's _forbidden_device_name()/
# _relocate_device_if_moved() called list_directshow_devices() to check the
# current device identity -- crashed the entire process outright within
# ~90 seconds, with no exception logged and no shutdown message, exactly
# the signature of a native crash rather than a Python-level bug. An RLock
# (not a plain Lock) so the same thread can safely re-enter it: core.vision
# _open_capture holds this for its whole per-candidate open loop and calls
# _forbidden_device_name() (which calls list_directshow_devices(), below)
# from inside that same hold.
DIRECTSHOW_LOCK = threading.RLock()


def list_directshow_devices() -> dict[int, str]:
    """{index: friendly_name} for every DirectShow video input device, in
    the same enumeration order cv2.VideoCapture(index, cv2.CAP_DSHOW) uses
    — that correspondence is exactly what pygrabber's ICreateDevEnum wrapper
    is for. Returns {} on non-Windows platforms, or if pygrabber isn't
    installed, or if the underlying COM enumeration fails for any reason
    (e.g. no camera driver at all) — callers must treat that as "identity
    unavailable", not "no devices exist", and fall back to their own
    explicit configuration rather than guessing.
    """
    if sys.platform != "win32":
        return {}
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return {}
    try:
        with DIRECTSHOW_LOCK:
            names = FilterGraph().get_input_devices()
    except Exception:
        return {}
    return dict(enumerate(names))


def list_directshow_devices_nowait(lock_timeout: float = 0.3) -> dict[int, str] | None:
    """Like list_directshow_devices, for a fast polling monitor (core/device_presence.py):

    * never waits long for DIRECTSHOW_LOCK - a camera open can hold it for seconds, and a presence
      poll must not queue behind it (returns None = "could not look this time", NOT "no cameras");
    * distinguishes "enumeration unavailable" (non-Windows, no pygrabber, COM failure -> None) from
      "no video devices at all" ({}), so the caller never mistakes a broken enumerator for an unplug.
    """
    if sys.platform != "win32":
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception:
        return None
    if not DIRECTSHOW_LOCK.acquire(timeout=lock_timeout):
        return None
    try:
        names = FilterGraph().get_input_devices()
    except Exception:
        return None
    finally:
        DIRECTSHOW_LOCK.release()
    return dict(enumerate(names))


def normalize_device_key(device) -> str:
    """Canonical form for comparing two `device` values (as passed to
    core.vision.CameraWorker/BoothManager) for *physical-device* equality —
    used to reject binding two different camera_ids to the same physical
    camera (spec: MongDee multi-USB-camera root-cause repair — "DEVICE
    INDEX COLLISION"). `device` can arrive as either an int (auto-discovery,
    core.vision.discover_cameras) or a str (the /settings "add camera" form,
    the --cameras CLI flag) for what may be the exact same numeric index —
    0 and "0" must compare equal. A non-numeric device (a path like
    "/dev/video0", an RTSP URL, ...) is never coerced into looking numeric;
    it's compared case-insensitively as a plain string instead."""
    s = str(device).strip()
    return str(int(s)) if s.isdigit() else s.lower()


def device_as_index(device) -> int | None:
    """`device` as a plain int index if it's one (int, or a numeric str like
    "0"/"2") -- physical-identity lookups only ever apply to a DirectShow
    index (what get_physical_camera_identities() is keyed by); a device
    path or RTSP URL has no such lookup and this returns None for it."""
    if isinstance(device, bool):  # bool is an int subclass -- never a camera index
        return None
    if isinstance(device, int):
        return device
    if isinstance(device, str) and device.strip().isdigit():
        return int(device.strip())
    return None


def is_virtual_camera_name(name: str) -> bool:
    """True for known virtual-camera software (OBS Virtual Camera, ManyCam,
    DroidCam, Snap Camera, ...) — never a legitimate physical USB booth
    camera, so excluded unconditionally (not just when built-in cameras
    are excluded) wherever this module's names are used to filter."""
    lname = name.lower()
    return any(marker in lname for marker in _VIRTUAL_NAME_MARKERS)


# ---------------------------------------------------------- physical identity
#
# A DirectShow *index* (what cv2.VideoCapture(index) actually opens) is not a
# stable physical identity — Windows can reassign it any time the device
# count changes (unplug/replug, another camera added/removed, reboot). This
# section adds a best-effort *physical* identity, built from Windows PnP data
# (VID/PID, USB port location, container ID — pulled via `Get-PnpDevice`,
# since neither cv2 nor pygrabber's public API exposes any of this), so a
# camera can in principle be recognized as "still the same physical device"
# across an index change.
#
# Verified on this project's own dev machine's real registry (see the
# session's own hardware audit): a webcam with no real USB serial number
# (most cheap UVC cameras) gets a Windows-synthesized instance ID that is
# tied to *USB port location*, not to the physical unit itself — unplugging
# and replugging into the SAME port reuses the same instance ID (so identity
# survives that), but moving the exact same physical camera to a DIFFERENT
# port produces a brand-new instance ID indistinguishable from a different
# physical unit of the same model. This is a genuine hardware/OS limitation,
# not something any software here can paper over — hence the explicit,
# reported identity_source/confidence level below instead of silently
# pretending every camera has a stable serial.
#
# Capability hierarchy (highest confidence first), matching the level a
# given camera's Windows registration actually supports:
#   LEVEL 1 "serial"              — a real USB iSerialNumber string (rare on
#                                    cheap UVC cameras, but when present,
#                                    survives being moved to any port).
#   LEVEL 2 "vid_pid_location"    — VID+PID+USB port location. Survives
#                                    same-port unplug/replug; changes if the
#                                    device moves to a different port.
#   LEVEL 3 "name_index_fallback" — DirectShow friendly name + current index
#                                    only (today's pre-existing behavior) —
#                                    used whenever PnP correlation is
#                                    unavailable (non-Windows, pygrabber/
#                                    PowerShell missing, or no matching
#                                    present PnP entry found for that index).

_INSTANCE_ID_VID_PID_RE = re.compile(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", re.IGNORECASE)


def list_pnp_camera_devices() -> list[dict]:
    """{friendly_name, instance_id, location_info, container_id} for every
    currently-*present* Windows PnP device in the Camera class — i.e. only
    devices actually connected right now, not every device Windows has ever
    seen plugged in (Get-PnpDevice without -PresentOnly returns those too;
    confirmed on this project's own dev machine to include many stale/absent
    entries from cameras previously connected to other USB ports).

    Returns [] on non-Windows platforms, or if PowerShell/Get-PnpDevice is
    unavailable or fails for any reason — callers must treat that exactly
    like list_directshow_devices()'s own empty-dict case: "identity
    unavailable", never "no cameras exist"."""
    if sys.platform != "win32":
        return []
    script = (
        "$ErrorActionPreference = 'SilentlyContinue'; "
        "@(Get-PnpDevice -Class Camera -PresentOnly | ForEach-Object { "
        "$id = $_.InstanceId; "
        "$loc = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_LocationInfo').Data; "
        "$cid = (Get-PnpDeviceProperty -InstanceId $id -KeyName 'DEVPKEY_Device_ContainerId').Data; "
        "[PSCustomObject]@{ FriendlyName = $_.FriendlyName; InstanceId = $id; "
        "LocationInfo = $loc; ContainerId = $(if ($cid) { $cid.ToString() } else { $null }) } "
        "}) | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except Exception:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [
        {
            "friendly_name": entry.get("FriendlyName") or "",
            "instance_id": entry.get("InstanceId"),
            "location_info": entry.get("LocationInfo"),
            "container_id": entry.get("ContainerId"),
        }
        for entry in data if isinstance(entry, dict)
    ]


def _parse_instance_id(instance_id: str | None) -> tuple[str | None, str | None, str | None]:
    """(vid, pid, serial) from a Windows PnP InstanceId like
    "USB\\VID_4C4A&PID_4A55&MI_00\\7&7199EFD&0&0000". `serial` is only
    returned when the trailing segment looks like a real USB iSerialNumber
    (no '&') rather than a Windows-synthesized per-port instance suffix
    (contains '&', e.g. "7&7199EFD&0&0000") -- see the module-level
    docstring above for why that distinction matters."""
    if not instance_id:
        return None, None, None
    m = _INSTANCE_ID_VID_PID_RE.search(instance_id)
    vid, pid = (m.group(1).upper(), m.group(2).upper()) if m else (None, None)
    suffix = instance_id.rsplit("\\", 1)[-1]
    serial = suffix if suffix and "&" not in suffix else None
    return vid, pid, serial


@dataclasses.dataclass(frozen=True)
class PhysicalCameraIdentity:
    physical_id: str
    device_index: int
    device_name: str
    instance_id: str | None
    vid: str | None
    pid: str | None
    serial: str | None
    location_info: str | None
    container_id: str | None
    identity_source: str  # "serial" | "vid_pid_location" | "name_index_fallback"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build_identity(index: int, name: str, pnp_entry: dict | None) -> PhysicalCameraIdentity:
    if pnp_entry is None:
        return PhysicalCameraIdentity(
            physical_id=f"name_index:{name.lower()}:{index}", device_index=index, device_name=name,
            instance_id=None, vid=None, pid=None, serial=None, location_info=None, container_id=None,
            identity_source="name_index_fallback",
        )
    vid, pid, serial = _parse_instance_id(pnp_entry.get("instance_id"))
    location = pnp_entry.get("location_info")
    if serial:
        physical_id, source = f"serial:{vid}:{pid}:{serial}", "serial"
    elif vid and pid and location:
        physical_id, source = f"vid_pid_location:{vid}:{pid}:{location}", "vid_pid_location"
    else:
        physical_id, source = f"name_index:{name.lower()}:{index}", "name_index_fallback"
    return PhysicalCameraIdentity(
        physical_id=physical_id, device_index=index, device_name=name,
        instance_id=pnp_entry.get("instance_id"), vid=vid, pid=pid, serial=serial,
        location_info=location, container_id=pnp_entry.get("container_id"),
        identity_source=source,
    )


def get_physical_camera_identities(
    directshow_devices: dict[int, str] | None = None,
    pnp_devices: list[dict] | None = None,
) -> dict[int, PhysicalCameraIdentity]:
    """{device_index: PhysicalCameraIdentity} for every currently-enumerated
    DirectShow camera, correlated against present Windows PnP devices.

    Two devices sharing the exact same friendly name (spec: "USB Camera" x N)
    are paired with present PnP entries of that same name *in relative
    enumeration order* (DirectShow index order <-> Get-PnpDevice list order)
    -- a best-effort heuristic, since neither cv2 nor pygrabber's public API
    exposes a DirectShow moniker's own DevicePath to correlate on exactly.
    When a name has more DirectShow indices than matching present PnP
    entries (or PnP data is unavailable at all), the unmatched indices fall
    back to identity_source="name_index_fallback" -- never silently missing,
    never fabricated.

    directshow_devices/pnp_devices can be injected for testing (see
    tests/core/test_camera_identity.py); omitted, they're fetched live via
    list_directshow_devices()/list_pnp_camera_devices()."""
    if directshow_devices is None:
        directshow_devices = list_directshow_devices()
    if pnp_devices is None:
        pnp_devices = list_pnp_camera_devices()

    by_name_ds: dict[str, list[int]] = {}
    for index in sorted(directshow_devices):
        by_name_ds.setdefault(directshow_devices[index].lower(), []).append(index)

    by_name_pnp: dict[str, list[dict]] = {}
    for entry in pnp_devices:
        by_name_pnp.setdefault(entry["friendly_name"].lower(), []).append(entry)

    identities: dict[int, PhysicalCameraIdentity] = {}
    for name_lower, indices in by_name_ds.items():
        pnp_entries = by_name_pnp.get(name_lower, [])
        for position, index in enumerate(indices):
            entry = pnp_entries[position] if position < len(pnp_entries) else None
            identities[index] = _build_identity(index, directshow_devices[index], entry)
    return identities


def resolve_current_index_for_physical_id(physical_id: str) -> int | None:
    """The DirectShow index a previously-identified physical camera
    currently enumerates at, or None if it isn't present right now (or
    identity data is unavailable). Used by a CameraWorker whose camera
    stopped responding at its original index, to check whether the same
    physical device simply moved to a different index (Windows
    re-enumeration) before giving up and reporting it offline."""
    for index, identity in get_physical_camera_identities().items():
        if identity.physical_id == physical_id:
            return index
    return None
