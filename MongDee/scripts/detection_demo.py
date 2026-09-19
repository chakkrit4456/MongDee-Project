"""Runnable demo for Phase 1 + Phase 2: Camera Gateway -> Frame Processing
-> Person Detection.

Usage:
    python scripts/detection_demo.py configs/cameras.example.json
    python scripts/detection_demo.py configs/cameras.example.json --detection configs/detection.example.json
    python scripts/detection_demo.py configs/cameras.example.json --show --seconds 20

Connects every enabled camera, runs one shared YOLO person detector over
them (round-robin, CPU-friendly), and prints a per-camera person count +
detection FPS once a second. With --show, opens a preview window per
camera with red person boxes drawn on the live frame. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from camera import CameraGateway, load_camera_configs
from vision import DetectionConfig, DetectionPipeline, MultiCameraTracker, PersonDetector, TrackingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cameras", help="path to a cameras JSON config file")
    parser.add_argument("--detection", help="path to a detection JSON config file (optional; defaults used otherwise)")
    parser.add_argument("--show", action="store_true", help="open a live preview window per camera with boxes drawn")
    parser.add_argument("--track", action="store_true", help="also run ByteTrack multi-object tracking (Phase 3)")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop automatically after N seconds (0 = until Ctrl+C)")
    args = parser.parse_args()

    try:
        camera_configs = [c for c in load_camera_configs(args.cameras) if c.enabled]
    except ValueError as exc:
        print(f"error loading {args.cameras}: {exc}", file=sys.stderr)
        return 1
    if not camera_configs:
        print(f"no enabled cameras in {args.cameras}", file=sys.stderr)
        return 1

    det_config = DetectionConfig.load(args.detection) if args.detection else DetectionConfig()
    print(f"cameras: {[c.id for c in camera_configs]}")
    print(f"detection: model={det_config.model_path} device={det_config.device} -> loading...")
    detector = PersonDetector(det_config)
    detector.warmup()
    print(f"detector ready on {detector.device}")

    latest_frames: dict[str, object] = {}
    latest_dets: dict[str, list] = {}
    latest_tracks: dict[str, list] = {}

    def on_result(result):
        latest_frames[result.camera_id] = result.frame
        latest_dets[result.camera_id] = result.detections
        latest_tracks[result.camera_id] = result.tracks

    gateway = CameraGateway(
        on_status=lambda cid, status, msg: print(f"[{cid}] {status.value}: {msg}")
    )
    for cfg in camera_configs:
        gateway.add_camera(cfg, start=False)
    tracker = MultiCameraTracker(TrackingConfig()) if args.track else None
    pipeline = DetectionPipeline(gateway, detector, tracker=tracker, on_result=on_result)

    gateway.start_all()
    pipeline.start()

    start = time.time()
    last_report = start
    try:
        while True:
            now = time.time()
            if args.seconds and now - start >= args.seconds:
                print(f"--seconds {args.seconds} elapsed, stopping")
                break
            if args.show:
                for camera_id, frame in list(latest_frames.items()):
                    img = frame.image.copy()
                    if args.track:
                        for t in latest_tracks.get(camera_id, []):
                            x1, y1, x2, y2 = (int(v) for v in t.bbox)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 60, 235), 2)
                            cv2.putText(img, f"#{t.track_id}", (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        for d in latest_dets.get(camera_id, []):
                            x1, y1, x2, y2 = (int(v) for v in d.bbox)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 60, 235), 2)
                            cv2.putText(img, f"PERSON {d.confidence:.0%}", (x1, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.imshow(camera_id, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if now - last_report >= 1.0:
                parts = []
                for cid, stats in pipeline.all_stats().items():
                    label = f"{stats.last_track_count} tracks" if args.track else f"{stats.last_detection_count} ppl"
                    parts.append(f"{cid}: {label} @ {stats.effective_fps:.1f}fps")
                print(f"[{now - start:5.1f}s] " + " | ".join(parts) if parts else f"[{now - start:5.1f}s] (no detections yet)")
                last_report = now
            time.sleep(0.03 if args.show else 0.2)
    except KeyboardInterrupt:
        print("\nCtrl+C received, stopping...")
    finally:
        pipeline.stop()
        gateway.stop_all()
        if args.show:
            cv2.destroyAllWindows()

    print("final per-camera stats:")
    for cid, stats in pipeline.all_stats().items():
        print(f"  {cid}: {stats.frames_processed} frames detected, last {stats.last_detection_count} people, "
              f"last latency {stats.last_latency_sec * 1000:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
