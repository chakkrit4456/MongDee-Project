"""Frame-integrity diagnostic: opens CAM-1 and CAM-2 concurrently exactly the
way production does (CAM-1 full profile list since it's "first active", CAM-2
low_bandwidth_only=True since CAM-1 is already active), reads continuously
for DURATION_SEC, and for every frame records:
  - shape, dtype, min/max/mean/std
  - core.vision._looks_like_noise() verdict
  - actual negotiated backend/resolution/fps/fourcc (once, at open)
Saves every SAVE_EVERY_N-th raw frame (both as read from cv2, AND after a
JPEG encode/decode round-trip -- the exact bytes production's own
cv2.imencode(".jpg", frame) / booth.get_latest_jpeg() would produce) to
.hw_validation/frames/ for direct visual inspection, per the investigation
prompt's "do not judge from statistics alone, open the image" instruction.
"""
import sys
import threading
import time
import json
import os

sys.path.insert(0, ".")
import cv2
import numpy as np
from core.vision import _open_capture, _looks_like_noise, _describe_capture

DURATION_SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
SAVE_EVERY_N = 3
OUT_DIR = ".hw_validation/frames"
os.makedirs(OUT_DIR, exist_ok=True)

stats = {}
stop = threading.Event()


def reader(label, index, low_bandwidth, start_barrier):
    cap = _open_capture(index, label=label, low_bandwidth_only=low_bandwidth)
    opened = bool(cap.isOpened())
    desc = _describe_capture(cap) if opened else "NOT OPENED"
    stats[label] = {"opened": opened, "describe": desc, "frames": [], "noise_count": 0,
                     "read_ok": 0, "read_failed": 0}
    print(f"[{label}] open={opened} {desc}", flush=True)
    start_barrier.wait()
    n = 0
    last_read_ts = time.monotonic()
    while not stop.is_set():
        t_before = time.monotonic()
        ok, frame = cap.read()
        read_dur = time.monotonic() - t_before
        n += 1
        if not ok or frame is None or frame.size == 0:
            stats[label]["read_failed"] += 1
            stats[label].setdefault("read_durations_sample", []).append(round(read_dur, 3))
            time.sleep(1 / 15)
            continue
        stats[label]["read_ok"] += 1
        stats[label].setdefault("read_durations_sample", [])
        if len(stats[label]["read_durations_sample"]) < 40:
            stats[label]["read_durations_sample"].append(round(read_dur, 3))
        is_noise = _looks_like_noise(frame)
        if is_noise:
            stats[label]["noise_count"] += 1
        if n % SAVE_EVERY_N == 0:
            small = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
            rec = {
                "n": n, "shape": list(frame.shape), "dtype": str(frame.dtype),
                "min": int(frame.min()), "max": int(frame.max()),
                "mean": round(float(frame.mean()), 2), "std": round(float(frame.std()), 2),
                "is_noise": is_noise,
            }
            stats[label]["frames"].append(rec)
            raw_path = f"{OUT_DIR}/{label}_raw_{n:05d}.png"
            cv2.imwrite(raw_path, frame)
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok_enc:
                jpeg_path = f"{OUT_DIR}/{label}_jpeg_{n:05d}.jpg"
                with open(jpeg_path, "wb") as f:
                    f.write(buf.tobytes())
                decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                rec["jpeg_bytes"] = len(buf.tobytes())
                rec["jpeg_decodes_ok"] = decoded is not None
            print(f"[{label}] saved frame {n}: {rec}", flush=True)
        # No sleep here on purpose (unlike the previous version of this
        # probe) -- reveals cap.read()'s own real blocking duration/rate
        # instead of a rate this script itself imposes.
    cap.release()


barrier = threading.Barrier(2)
tA = threading.Thread(target=reader, args=("CAM-1", 1, False, barrier))
tB = threading.Thread(target=reader, args=("CAM-2", 2, True, barrier))
tA.start()
tB.start()
time.sleep(DURATION_SEC)
stop.set()
tA.join(timeout=10)
tB.join(timeout=10)

summary = {label: {k: v for k, v in s.items() if k != "frames"} for label, s in stats.items()}
print("[probe] SUMMARY", json.dumps(summary, indent=2), flush=True)
with open(f"{OUT_DIR}/../frame_integrity_summary.json", "w") as f:
    json.dump(stats, f, indent=2)
