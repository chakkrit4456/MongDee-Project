"""Decode Windows/DirectShow/Media Foundation HRESULTs seen in MongDee logs into an
action class so the reopen logic can react differently (wait vs. replug vs. lower profile)."""
from __future__ import annotations
import re
from typing import Dict, Optional

# code -> (name, meaning, action)   action: 'backoff' | 'lower_profile' | 'reset_usb' | 'busy' | 'fatal'
_TABLE: Dict[int, tuple] = {
    0xC00D3704: ("MF_E_HW_MFT_FAILED_START_STREAMING", "device refused to start streaming (bandwidth/driver/busy)", "lower_profile"),
    0x8007001F: ("ERROR_GEN_FAILURE", "device attached to the system is not functioning (USB stall)", "reset_usb"),
    0x800703E3: ("ERROR_OPERATION_ABORTED", "I/O aborted (device removed or stream stopped)", "backoff"),
    0x80070005: ("E_ACCESSDENIED", "camera in use by another process or blocked by privacy setting", "busy"),
    0xC00D3EA2: ("MF_E_VIDEO_RECORDING_DEVICE_INVALIDATED", "device unplugged/invalidated", "reset_usb"),
    0xC00D3EA3: ("MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED", "another app took the device", "busy"),
    0x80070020: ("ERROR_SHARING_VIOLATION", "device in use", "busy"),
    0x8007048F: ("ERROR_DEVICE_NOT_CONNECTED", "device not connected", "reset_usb"),
}
_RE = re.compile(r"(?:0x)?([0-9A-Fa-f]{8})\b")


def decode(code) -> Optional[Dict[str, str]]:
    if isinstance(code, str):
        m = _RE.search(code)
        if not m:
            return None
        code = int(m.group(1), 16)
    code &= 0xFFFFFFFF
    row = _TABLE.get(code)
    if not row:
        return {"code": f"0x{code:08X}", "name": "UNKNOWN", "meaning": "unrecognised", "action": "backoff"}
    return {"code": f"0x{code:08X}", "name": row[0], "meaning": row[1], "action": row[2]}


def scan_log(text: str) -> Dict[str, int]:
    """Count known HRESULT occurrences in a log body -> {'0xC00D3704': n, ...}."""
    counts: Dict[str, int] = {}
    for m in re.finditer(r"0x[0-9A-Fa-f]{8}", text):
        d = decode(m.group(0))
        if d and d["name"] != "UNKNOWN":
            counts[d["code"]] = counts.get(d["code"], 0) + 1
    return counts
