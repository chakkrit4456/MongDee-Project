"""Structured per-device open/read diagnostic for troubleshooting USB
webcams that won't connect — e.g. CAM-1/CAM-2 work but CAM-3/CAM-4 don't.

Usage:
    python scripts/camera_diagnostic.py               # probe index 0..7
    python scripts/camera_diagnostic.py --max-index 12

Run this with the main app (app.py / web_server.py) NOT running — they hold
their own cameras open, and this script trying the same index at the same
time would just show a false OPEN_FAILED for whichever one loses the race.

Unlike normal startup, this probes every index (including any Windows
built-in webcam — it is not excluded here) and keeps trying every
backend/format combination for that index even after one already produced
a frame, so every failure stage is visible in one run instead of stopping
at the first success like the real app does. It never touches
app.py/web_server.py's own startup path — Built-in Camera stays excluded
there by booth_settings.json's "enable_builtin_camera" regardless of what
this script finds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core.camera_identity import list_directshow_devices
from core.vision import (
    _OPEN_PROFILES, _backend_name, _candidate_opens, _describe_capture,
    _profile_label, _read_with_warmup,
)
from web.booth_manager import BOOTH_SETTINGS_PATH, compute_camera_skip_indices, load_camera_settings

# This script prints its own structured table below; the library's own INFO/
# DEBUG lines (core.vision's per-attempt logging) would just duplicate it.
logging.basicConfig(level=logging.WARNING, format="%(message)s")


def probe_index(index: int) -> list[dict]:
    """Try every (backend, format) combination for one device index, without
    stopping at the first success — troubleshooting wants the full picture,
    unlike _open_capture's production early-return."""
    results = []
    for dev, backend in _candidate_opens(index):
        backend_name = _backend_name(backend)
        for profile in _OPEN_PROFILES:
            cap = cv2.VideoCapture(dev, backend)
            opened = read_ok = False
            detail = ""
            try:
                opened = bool(cap.isOpened())
                if opened:
                    if profile is not None:
                        codec, width, height, fps = profile
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        cap.set(cv2.CAP_PROP_FPS, fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    read_ok = _read_with_warmup(cap)
                    if read_ok:
                        detail = _describe_capture(cap)
            except cv2.error as exc:
                detail = f"cv2.error: {exc}"
            finally:
                cap.release()
            results.append({
                "backend": backend_name, "format": _profile_label(profile),
                "opened": opened, "read_ok": read_ok, "detail": detail,
            })
    return results


def classify(results: list[dict]) -> str:
    """Same failure-stage vocabulary core.vision._classify_open_failure uses
    (see its docstring) plus the success case."""
    if not results:
        return "DISCOVERY_FAILED"
    if any(r["read_ok"] for r in results):
        return "OK"
    if any(r["opened"] for r in results):
        return "OPENED_BUT_READ_FAILED"
    return "OPEN_FAILED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-index", type=int, default=8, help="probe device index 0..N-1 (default 8)")
    args = parser.parse_args()

    camera_settings = load_camera_settings(BOOTH_SETTINGS_PATH)
    skip_indices = compute_camera_skip_indices(camera_settings)
    device_names = list_directshow_devices()  # {} off-Windows or without pygrabber installed

    print("Camera Diagnostic")
    print(f"(probing index 0..{args.max_index - 1}, every backend/format combination, "
          f"does not stop at first success)")
    if device_names:
        print("DirectShow device names by index (this is the ground truth app.py/web_server.py's "
              "built-in/virtual-camera exclusion actually matches against, not a guess):")
        for idx, name in sorted(device_names.items()):
            print(f"  {idx}: {name}")
    else:
        print("DirectShow device names unavailable (pygrabber not installed, or not on Windows) — "
              "built-in/virtual-camera exclusion can only use booth_settings.json's static "
              "builtin_camera_indices here, which goes stale across replug/reboot.")
    if skip_indices:
        print(f"Excluded at normal startup (built-in/virtual, by current name or configured index): "
              f"{sorted(skip_indices)}")
    print()

    for i in range(args.max_index):
        name = device_names.get(i)
        name_part = f" name={name!r}" if name else ""
        tag = " [would be SKIPPED at normal startup: built-in/virtual]" if i in skip_indices else ""
        print(f"Device {i}{name_part}{tag}")
        results = probe_index(i)
        if not results:
            print("  DISCOVERY_FAILED — no backend/candidate applicable on this platform for this index")
            print()
            continue
        for r in results:
            if r["read_ok"]:
                status = "READ OK"
            elif r["opened"]:
                status = "OPEN OK, READ FAILED"
            else:
                status = "OPEN FAILED"
            line = f"  backend={r['backend']:<8} format={r['format']:<16} {status}"
            if r["detail"]:
                line += f"  ({r['detail']})"
            print(line)
        print(f"  => {classify(results)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
