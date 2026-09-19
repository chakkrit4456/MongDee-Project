"""Same as test_dshow_cameras.py but for cv2.CAP_MSMF explicitly (never
cv2.CAP_ANY). See that file's docstring for the shared method/rationale.

Each profile attempt runs as its own subprocess with a hard wall-clock
timeout: MSMF was found (this session, on this project's own hardware) to
hang indefinitely on a cap.read() call after a previous MSMF session on the
same index had just been released -- see
docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md's MSMF findings -- so an in-process
loop risks never finishing the matrix at all.

Usage: python tests/manual/test_msmf_cameras.py [index ...]
"""
import sys
import json
import subprocess

PER_ATTEMPT_TIMEOUT_SEC = 20

indices = [int(a) for a in sys.argv[1:]] or [1, 2]

PROFILES = [
    ("native",),
    ("MJPG", "640", "480", "10"),
    ("MJPG", "640", "480", "5"),
    ("MJPG", "320", "240", "10"),
    ("MJPG", "320", "240", "5"),
    ("YUY2", "320", "240", "10"),
    ("YUY2", "320", "240", "5"),
]

results = []
print(f"=== CAP_MSMF single-camera matrix, indices={indices} (subprocess-isolated, "
      f"{PER_ATTEMPT_TIMEOUT_SEC}s hard timeout per attempt) ===", flush=True)
for index in indices:
    for profile_args in PROFILES:
        label = profile_args[0] if profile_args[0] == "native" else \
            f"{profile_args[0]} {profile_args[1]}x{profile_args[2]}@{profile_args[3]}"
        args = [sys.executable, "tests/manual/_single_probe.py", str(index), "MSMF", *profile_args]
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=PER_ATTEMPT_TIMEOUT_SEC)
            if proc.returncode != 0:
                r = {"index": index, "backend": "MSMF", "requested_profile": label,
                     "error": f"subprocess exited {proc.returncode}: {proc.stderr[-500:]}"}
            else:
                r = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            r = {"index": index, "backend": "MSMF", "requested_profile": label,
                 "open": None, "isOpened": None, "error": f"HUNG (no result within {PER_ATTEMPT_TIMEOUT_SEC}s)"}
        print(f"  index={index} backend=MSMF   profile={label:20s} -> {r}", flush=True)
        results.append(r)

with open("tests/manual/msmf_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved tests/manual/msmf_results.json", flush=True)
