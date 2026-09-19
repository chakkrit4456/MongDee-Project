"""Concurrent capture-only matrix (Phase 5/8/9 of the investigation): both
cameras opened at once, in every backend combination, no AI/web/BoothManager.
Each combination runs as its own subprocess pair with a hard wall-clock
timeout (see test_msmf_cameras.py's docstring for why -- MSMF can hang).

Usage: python tests/manual/test_concurrent_matrix.py
"""
import subprocess
import sys
import time
import json

TIMEOUT_SEC = 30
RUN_SECONDS = 8  # per combination -- kept short since this is a matrix of several combinations


def run_pair(index_a, backend_a, index_b, backend_b, stagger_sec=0.0):
    script = f"""
import sys, time, threading, json
sys.path.insert(0, ".")
import cv2

def reader(label, index, backend, delay, out):
    time.sleep(delay)
    cap = cv2.VideoCapture(index, backend)
    opened = bool(cap.isOpened())
    ok_count = 0
    fail_count = 0
    t_end = time.monotonic() + {RUN_SECONDS}
    while time.monotonic() < t_end:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            ok_count += 1
        else:
            fail_count += 1
    cap.release()
    out[label] = {{"opened": opened, "read_ok": ok_count, "read_failed": fail_count}}

out = {{}}
tA = threading.Thread(target=reader, args=("A", {index_a}, {backend_a}, 0.0, out))
tB = threading.Thread(target=reader, args=("B", {index_b}, {backend_b}, {stagger_sec}, out))
tA.start(); tB.start()
tA.join(); tB.join()
print(json.dumps(out))
"""
    try:
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=TIMEOUT_SEC)
        if proc.returncode != 0:
            return {"error": f"exit {proc.returncode}: {proc.stderr[-500:]}"}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": f"HUNG (no result within {TIMEOUT_SEC}s)"}


import cv2

COMBOS = [
    ("DSHOW+DSHOW, no stagger", 1, cv2.CAP_DSHOW, 2, cv2.CAP_DSHOW, 0.0),
    ("DSHOW+DSHOW, 2.5s stagger", 1, cv2.CAP_DSHOW, 2, cv2.CAP_DSHOW, 2.5),
    ("MSMF+MSMF, no stagger", 1, cv2.CAP_MSMF, 2, cv2.CAP_MSMF, 0.0),
    ("DSHOW(A)+MSMF(B), no stagger", 1, cv2.CAP_DSHOW, 2, cv2.CAP_MSMF, 0.0),
    ("MSMF(A)+DSHOW(B), no stagger", 1, cv2.CAP_MSMF, 2, cv2.CAP_DSHOW, 0.0),
]

results = {}
print(f"=== Concurrent matrix ({RUN_SECONDS}s per combo, {TIMEOUT_SEC}s hard timeout) ===", flush=True)
for label, ia, ba, ib, bb, stagger in COMBOS:
    r = run_pair(ia, ba, ib, bb, stagger)
    print(f"  {label}: {r}", flush=True)
    results[label] = r

with open("tests/manual/concurrent_matrix_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved tests/manual/concurrent_matrix_results.json", flush=True)
