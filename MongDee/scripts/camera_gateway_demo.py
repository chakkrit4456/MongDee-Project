"""Runnable demo/smoke-test for the Phase 1 Camera Gateway.

Usage:
    python scripts/camera_gateway_demo.py configs/cameras.example.json
    python scripts/camera_gateway_demo.py configs/cameras.example.json --show
    python scripts/camera_gateway_demo.py configs/cameras.example.json --seconds 15

Connects every *enabled* camera in the given config file through
camera.CameraGateway, prints status transitions and a per-camera FPS
counter to the console, and (with --show) opens a live cv2.imshow preview
window per camera. Ctrl+C to stop cleanly.

Edit configs/cameras.example.json (or copy it) to point at real cameras —
see camera/README.md for the URL format each protocol expects.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from camera import CameraGateway, CameraStatus, Frame, load_camera_configs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a cameras JSON config file")
    parser.add_argument("--show", action="store_true", help="open a live preview window per camera")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop automatically after N seconds (0 = run until Ctrl+C)")
    args = parser.parse_args()

    try:
        configs = load_camera_configs(args.config)
    except ValueError as exc:
        print(f"error loading {args.config}: {exc}", file=sys.stderr)
        return 1

    enabled = [c for c in configs if c.enabled]
    if not enabled:
        print(f"no enabled cameras in {args.config}", file=sys.stderr)
        return 1
    print(f"loaded {len(configs)} camera(s), {len(enabled)} enabled: {[c.id for c in enabled]}")

    frame_counts: dict[str, int] = {c.id: 0 for c in enabled}
    last_frames: dict[str, Frame] = {}

    def on_frame(frame: Frame) -> None:
        frame_counts[frame.camera_id] = frame_counts.get(frame.camera_id, 0) + 1
        last_frames[frame.camera_id] = frame

    def on_status(camera_id: str, status: CameraStatus, message: str) -> None:
        print(f"[{camera_id}] {status.value}: {message}")

    gateway = CameraGateway(on_frame=on_frame, on_status=on_status)
    for cfg in enabled:
        gateway.add_camera(cfg, start=False)
    gateway.start_all()

    start = time.time()
    last_report = start
    try:
        while True:
            now = time.time()
            if args.seconds and now - start >= args.seconds:
                print(f"--seconds {args.seconds} elapsed, stopping")
                break
            if args.show:
                for camera_id, frame in list(last_frames.items()):
                    cv2.imshow(camera_id, frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if now - last_report >= 1.0:
                elapsed = now - start
                report = ", ".join(f"{cid}={count / elapsed:.1f}fps" for cid, count in frame_counts.items())
                print(f"[{elapsed:5.1f}s] {report}")
                last_report = now
            time.sleep(0.05 if args.show else 0.2)
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping...")
    finally:
        gateway.stop_all()
        if args.show:
            cv2.destroyAllWindows()

    print("final status:")
    for camera_id, (status, message) in gateway.all_statuses().items():
        print(f"  {camera_id}: {status.value} ({message}) - {frame_counts.get(camera_id, 0)} frames received")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
