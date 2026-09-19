"""Open BOTH camera indices concurrently via threads, mimicking exactly how
BoothManager.start() launches one CameraWorker thread per camera. Compares
NO-STAGGER (both threads start() back-to-back with 0 delay) against the
production CAMERA_STARTUP_STAGGER_SEC gap, using the real _open_capture()
function (same code path CameraWorker._open() uses)."""
import sys
import threading
import time
import json

sys.path.insert(0, ".")
from core.vision import _open_capture

INDEX_A = 1
INDEX_B = 2
stagger = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

results = {}
lock = threading.Lock()


def worker(label, index, start_delay):
    time.sleep(start_delay)
    t_request = time.monotonic()
    cap = _open_capture(index, label=label)
    t_done = time.monotonic()
    opened = cap.isOpened()
    frame_ok = False
    if opened:
        ok, frame = cap.read()
        frame_ok = bool(ok and frame is not None)
        cap.release()
    with lock:
        results[label] = {
            "index": index,
            "start_delay": start_delay,
            "request_to_done_s": round(t_done - t_request, 2),
            "isOpened": opened,
            "frame_ok": frame_ok,
        }


t0 = time.monotonic()
tA = threading.Thread(target=worker, args=("CAM-A", INDEX_A, 0.0))
tB = threading.Thread(target=worker, args=("CAM-B", INDEX_B, stagger))
tA.start()
tB.start()
tA.join(timeout=30)
tB.join(timeout=30)

print(json.dumps({"stagger_sec": stagger, "total_wall_s": round(time.monotonic()-t0, 2), "results": results}, indent=2))
