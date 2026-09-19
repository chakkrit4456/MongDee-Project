"""Runs exactly one probe_camera() call and prints its result as JSON on
stdout. Meant to be invoked as its own subprocess with an external
wall-clock timeout (see test_msmf_cameras.py) so a single hung backend call
(observed: MSMF can hang indefinitely on this hardware, see
docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md) can't block an entire test matrix.

Usage: python tests/manual/_single_probe.py <index> <backend> <fourcc|native> <w> <h> <fps>
"""
import sys
import json

sys.path.insert(0, ".")
from tests.manual._backend_probe_common import probe_camera
import cv2

index = int(sys.argv[1])
backend = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}[sys.argv[2]]
if sys.argv[3] == "native":
    profile = None
else:
    profile = (sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

result = probe_camera(index, backend, profile, warmup_reads=5, sustained_reads=10)
print(json.dumps(result))
