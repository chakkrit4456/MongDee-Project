"""Non-disruptive camera identity enumeration -- does NOT open any
cv2.VideoCapture, so it's safe to run alongside a live MongDee server that
already holds the cameras open. Reports exactly what core.camera_identity
sees: DirectShow enumeration, Windows PnP enumeration, and the correlated
PhysicalCameraIdentity MongDee actually computes and uses -- the same data
CameraWorker._relocate_device_if_moved()/_forbidden_device_name() rely on.
"""
import sys
import json

sys.path.insert(0, ".")
from core.camera_identity import (
    list_directshow_devices, list_pnp_camera_devices, get_physical_camera_identities,
)

ds = list_directshow_devices()
pnp = list_pnp_camera_devices()
identities = get_physical_camera_identities(ds, pnp)

print("=== DirectShow enumeration (index -> friendly name) ===")
print(json.dumps(ds, indent=2, ensure_ascii=False))

print("\n=== Windows PnP Camera-class devices (present only) ===")
print(json.dumps(pnp, indent=2, ensure_ascii=False))

print("\n=== MongDee's resolved PhysicalCameraIdentity per DirectShow index ===")
print(json.dumps({str(k): v.to_dict() for k, v in identities.items()}, indent=2, ensure_ascii=False))

print("\n=== Distinguishability check ===")
seen = {}
for idx, ident in identities.items():
    key = (ident.vid, ident.pid, ident.location_info, ident.serial)
    seen.setdefault(key, []).append(idx)
for key, idxs in seen.items():
    vid, pid, loc, serial = key
    if len(idxs) > 1:
        print(f"AMBIGUOUS: indices {idxs} share identical VID={vid} PID={pid} "
              f"location={loc} serial={serial} -- cannot be told apart by MongDee's "
              f"own identity data alone")
    else:
        print(f"index {idxs[0]}: VID={vid} PID={pid} location={loc} serial={serial} "
              f"source={identities[idxs[0]].identity_source} -- unique")
