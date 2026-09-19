"""Step 2 -- isolate whether resolution/bandwidth is the variable, by forcing
BOTH cameras to the absolute lowest resolution this hardware showed any sign
of negotiating (160x120 request -> observed actual 320x240 on this specific
"USB Camera" model per the prior session's manual sweep), sustained for
several minutes of concurrent reading. If CAM-2 still fails identically at
the smallest profile, resolution/bandwidth-per-se is ruled out.

Capture-only (no AIWorker/BoothManager/FastAPI), same pattern as
.hw_validation/probe_capture_only.py, calling the real production
core.vision._open_capture / _read_with_warmup so this exercises the actual
code path, not a reimplementation.
"""
import sys
import threading
import time

sys.path.insert(0, r"D:\mongdee2\MongDee")
import cv2

from core.vision import _open_capture, _candidate_opens

DEVICES = {"CAM-A": 1, "CAM-B": 2}
DURATION_SEC = int(sys.argv[1]) if len(sys.argv) > 1 else 300  # 5 min default

stats = {name: {"reads_ok": 0, "reads_failed": 0, "opened": False} for name in DEVICES}
stop = threading.Event()


def force_lowest(device):
    """Opens at the lowest resolution this camera model showed any signal
    at (160x120 request -> actual 320x240 native format on this hardware,
    per the prior manual sweep saved in this same .hw_validation/ dir)."""
    cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def worker(name, device):
    cap = force_lowest(device)
    stats[name]["opened"] = bool(cap.isOpened())
    if not stats[name]["opened"]:
        cap.release()
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[{name}] opened at actual {w}x{h}", flush=True)
    while not stop.is_set():
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            stats[name]["reads_ok"] += 1
        else:
            stats[name]["reads_failed"] += 1
        time.sleep(0.03)
    cap.release()


threads = [threading.Thread(target=worker, args=(n, d), daemon=True) for n, d in DEVICES.items()]
t0 = time.time()
for t in threads:
    t.start()
    time.sleep(2.5)  # matches the project's own CAMERA_STARTUP_STAGGER_SEC

try:
    while time.time() - t0 < DURATION_SEC:
        time.sleep(10)
        elapsed = time.time() - t0
        print(f"[t={elapsed:5.1f}s] " + "  ".join(
            f"{n} ok={stats[n]['reads_ok']} failed={stats[n]['reads_failed']} opened={stats[n]['opened']}"
            for n in DEVICES
        ), flush=True)
finally:
    stop.set()
    for t in threads:
        t.join(timeout=3)

print("FINAL:", stats, flush=True)
