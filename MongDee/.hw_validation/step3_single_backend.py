"""Step 3 -- isolate whether backend choice (DSHOW vs MSMF) changes the
failure. Forces BOTH cameras to the SAME explicit backend (passed as argv[1]:
"DSHOW" or "MSMF") for the whole run, instead of production's DSHOW-first-
then-CAP_ANY-fallback per open. Capture-only, same pattern as step2.
"""
import sys
import threading
import time

sys.path.insert(0, r"D:\mongdee2\MongDee")
import cv2

BACKEND_NAME = sys.argv[1] if len(sys.argv) > 1 else "DSHOW"
BACKEND = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}[BACKEND_NAME]
DURATION_SEC = int(sys.argv[2]) if len(sys.argv) > 2 else 120

DEVICES = {"CAM-A": 1, "CAM-B": 2}
stats = {name: {"reads_ok": 0, "reads_failed": 0, "opened": False} for name in DEVICES}
stop = threading.Event()


def worker(name, device):
    cap = cv2.VideoCapture(device, BACKEND)
    stats[name]["opened"] = bool(cap.isOpened())
    if not stats[name]["opened"]:
        print(f"[{name}] backend={BACKEND_NAME} OPEN FAILED", flush=True)
        cap.release()
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[{name}] backend={BACKEND_NAME} opened", flush=True)
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
    time.sleep(2.5)

try:
    while time.time() - t0 < DURATION_SEC:
        time.sleep(10)
        elapsed = time.time() - t0
        print(f"[t={elapsed:5.1f}s backend={BACKEND_NAME}] " + "  ".join(
            f"{n} ok={stats[n]['reads_ok']} failed={stats[n]['reads_failed']} opened={stats[n]['opened']}"
            for n in DEVICES
        ), flush=True)
finally:
    stop.set()
    for t in threads:
        t.join(timeout=3)

print("FINAL:", BACKEND_NAME, stats, flush=True)
