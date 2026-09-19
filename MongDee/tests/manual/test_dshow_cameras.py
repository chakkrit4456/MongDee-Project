"""Standalone, AI-free, web-server-free DirectShow (cv2.CAP_DSHOW) backend
test for every USB camera index given on the command line (default: 1 2,
this project's own two USB cameras -- see docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md).

Usage: python tests/manual/test_dshow_cameras.py [index ...]

Tests each camera ALONE (one at a time, never concurrently -- see
test_concurrent_matrix.py for that), through the MJPEG-then-YUY2 profile
matrix the investigation prompt asked for, plus each camera's native
(no explicit format) default. Never uses cv2.CAP_ANY.
"""
import sys
import json

sys.path.insert(0, ".")
from tests.manual._backend_probe_common import probe_camera, print_result_row
import cv2

indices = [int(a) for a in sys.argv[1:]] or [1, 2]

PROFILES = [
    None,  # native/default -- whatever the camera opens with, unconfigured
    ("MJPG", 640, 480, 10),
    ("MJPG", 640, 480, 5),
    ("MJPG", 320, 240, 10),
    ("MJPG", 320, 240, 5),
    ("YUY2", 320, 240, 10),
    ("YUY2", 320, 240, 5),
]

results = []
print(f"=== CAP_DSHOW single-camera matrix, indices={indices} ===", flush=True)
for index in indices:
    for profile in PROFILES:
        r = probe_camera(index, cv2.CAP_DSHOW, profile)
        print_result_row(r)
        results.append(r)

with open("tests/manual/dshow_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved tests/manual/dshow_results.json", flush=True)
