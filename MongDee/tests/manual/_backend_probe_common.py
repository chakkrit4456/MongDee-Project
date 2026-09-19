"""Shared, AI-free, web-server-free camera probing logic for the manual
backend-comparison scripts in this directory (test_dshow_cameras.py,
test_msmf_cameras.py, test_concurrent_matrix.py).

Deliberately bypasses core.vision._open_capture()'s own backend-fallback/
profile-list logic -- the whole point of these scripts is to test each
backend (CAP_DSHOW, CAP_MSMF) *explicitly and separately*, with format
negotiation verified by reading properties back (never trusting cap.set()'s
return value alone), instead of letting MongDee's own candidate-list guess
for us. Not a replacement for core.vision's real code path -- a diagnostic
to decide, with evidence, which backend/profile combination that code path
should prefer.
"""
from __future__ import annotations

import time
import cv2


def fourcc_to_str(value: float) -> str:
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"


def backend_name(backend: int) -> str:
    try:
        return cv2.videoio_registry.getBackendName(backend)
    except Exception:
        return str(backend)


def probe_camera(index: int, backend: int, profile: tuple | None,
                  warmup_reads: int = 5, sustained_reads: int = 20) -> dict:
    """profile: (fourcc_str, width, height, fps) or None for native/default.
    Returns a fully-populated result dict; never raises."""
    result = {
        "index": index, "backend": backend_name(backend),
        "requested_profile": profile if profile is None else
            f"{profile[0]} {profile[1]}x{profile[2]}@{profile[3]}",
        "open": False, "isOpened": False,
        "actual_fourcc": None, "actual_width": None, "actual_height": None, "actual_fps": None,
        "warmup_read_ok": False, "sustained_read_ok": 0, "sustained_read_failed": 0,
        "read_durations_sample": [], "error": None,
    }
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        result["open"] = True
        result["isOpened"] = bool(cap.isOpened())
        if not result["isOpened"]:
            return result
        if profile is not None:
            fourcc_str, width, height, fps = profile
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Never trust cap.set()'s return value -- read the properties back.
        result["actual_fourcc"] = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        result["actual_width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["actual_height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["actual_fps"] = cap.get(cv2.CAP_PROP_FPS)

        for _ in range(warmup_reads):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                result["warmup_read_ok"] = True
                break
            time.sleep(0.05)

        for _ in range(sustained_reads):
            t0 = time.monotonic()
            ok, frame = cap.read()
            dur = time.monotonic() - t0
            if len(result["read_durations_sample"]) < sustained_reads:
                result["read_durations_sample"].append(round(dur, 3))
            if ok and frame is not None and frame.size:
                result["sustained_read_ok"] += 1
            else:
                result["sustained_read_failed"] += 1
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
    return result


def print_result_row(r: dict) -> None:
    print(
        f"  index={r['index']} backend={r['backend']:6s} profile={r['requested_profile'] or 'native':20s} "
        f"open={r['open']} isOpened={r['isOpened']} "
        f"actual={r['actual_fourcc']}/{r['actual_width']}x{r['actual_height']}@{r['actual_fps']} "
        f"warmup_ok={r['warmup_read_ok']} sustained={r['sustained_read_ok']}/{r['sustained_read_ok']+r['sustained_read_failed']} "
        f"error={r['error']}",
        flush=True,
    )
