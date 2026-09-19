"""CAPTURE_ONLY sustained concurrent test -- no YOLO, no AIWorker, no web
server, no BoothManager. Uses the exact production core.vision._open_capture
function (same DIRECTSHOW_LOCK-protected code path CameraWorker._open() uses)
so this is a faithful "is AI/web contention the cause" isolation test, not a
toy re-implementation.

Opens CAM-A (index 1) first with the full profile list, marks it active
(mirroring CameraWorker._mark_camera_active), then opens CAM-B (index 2)
with low_bandwidth_only=True (mirroring what a second concurrent camera
gets in production), then reads both in a tight loop for DURATION_SEC,
tallying read successes/failures per camera every second.
"""
import sys
import threading
import time
import json

sys.path.insert(0, ".")
from core.vision import _open_capture

DURATION_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
INDEX_A = 1
INDEX_B = 2

stats = {"CAM-A": {"reads_ok": 0, "reads_failed": 0}, "CAM-B": {"reads_ok": 0, "reads_failed": 0}}
stop = threading.Event()
open_results = {}


def reader(label, index, low_bandwidth, start_barrier):
    cap = _open_capture(index, label=label, low_bandwidth_only=low_bandwidth)
    open_results[label] = bool(cap.isOpened())
    start_barrier.wait()
    while not stop.is_set():
        ok, frame = cap.read()
        if ok and frame is not None:
            stats[label]["reads_ok"] += 1
        else:
            stats[label]["reads_failed"] += 1
        time.sleep(1 / 30)
    cap.release()


barrier = threading.Barrier(2)
tA = threading.Thread(target=reader, args=("CAM-A", INDEX_A, False, barrier))
tB = threading.Thread(target=reader, args=("CAM-B", INDEX_B, True, barrier))

print(f"[probe] opening CAM-A (index {INDEX_A}, full profiles) and CAM-B "
      f"(index {INDEX_B}, low-bandwidth profiles) concurrently...", flush=True)
t0 = time.monotonic()
tA.start()
tB.start()

last_print = time.monotonic()
while time.monotonic() - t0 < DURATION_SEC:
    time.sleep(1.0)
    now = time.monotonic()
    print(f"[t={now - t0:5.1f}s] {json.dumps(stats)} open_results={open_results}", flush=True)

stop.set()
tA.join(timeout=10)
tB.join(timeout=10)
print("[probe] FINAL", json.dumps({"duration_sec": DURATION_SEC, "open_results": open_results, "stats": stats}, indent=2), flush=True)
