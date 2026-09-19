"""Open ONE camera index in isolation, prove it streams real distinct frames,
and report success/failure with the raw OpenCV error context. No other
camera process may be running while this executes (exclusive access)."""
import sys
import time
import hashlib
import json
import cv2

index = int(sys.argv[1])
backend_name = sys.argv[2] if len(sys.argv) > 2 else "DSHOW"
backend = cv2.CAP_DSHOW if backend_name == "DSHOW" else (cv2.CAP_MSMF if backend_name == "MSMF" else cv2.CAP_ANY)

t_open_start = time.monotonic()
cap = cv2.VideoCapture(index, backend)
t_open_done = time.monotonic()
opened = cap.isOpened()

result = {
    "index": index,
    "backend": backend_name,
    "isOpened": opened,
    "open_ms": round((t_open_done - t_open_start) * 1000, 1),
}

frames = []
if opened:
    for i in range(5):
        t0 = time.monotonic()
        ok, frame = cap.read()
        t1 = time.monotonic()
        if not ok or frame is None:
            frames.append({"i": i, "ok": False, "read_ms": round((t1-t0)*1000, 1)})
            continue
        h = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
        frames.append({
            "i": i,
            "ok": True,
            "read_ms": round((t1 - t0) * 1000, 1),
            "shape": list(frame.shape),
            "mean_bgr": [round(float(x), 2) for x in frame.reshape(-1, frame.shape[-1]).mean(axis=0)],
            "sha256_16": h,
            "t_wall": time.time(),
        })
        if i == 0:
            cv2.imwrite(f".hw_validation/snap_idx{index}_{backend_name}.jpg", frame)
        time.sleep(0.05)
    cap.release()

result["frames"] = frames
result["first_frame_wall_ts"] = frames[0]["t_wall"] if frames and frames[0]["ok"] else None
print(json.dumps(result))
