"""Real-time FairFace Age/Gender/Race webcam demo — up to two cameras at once.

    python detect.py                                   # camera 0 + camera 1 (if present)
    python detect.py --camera1 none                     # single camera only
    python detect.py --checkpoint models/fairface/last_model.pt

Pipeline per camera, every captured frame:
    Webcam -> Face Detection (every --detect-every-n-frames frames; the
    face box + attribute prediction are reused/redrawn on the frames in
    between, matching the spec's "detect every N frames, reuse prediction"
    perf strategy) -> FairFace model -> draw box + Age/Gender/Race -> display.

Press 'q' or Ctrl+C to stop — both release every open camera and close all
windows before the process exits.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera0", default="0", help="Device index/path for the first camera.")
    parser.add_argument("--camera1", default="1",
                         help="Device index/path for the second camera, or 'none' to run single-camera.")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "fairface" / "best_model.pt"))
    _default_yunet = ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"
    parser.add_argument("--yunet-model", default=str(_default_yunet) if _default_yunet.is_file() else None,
                         help="YuNet ONNX face detector (auto-detected under models/face_detector/ if present); "
                              "falls back to the bundled Haar cascade otherwise.")
    parser.add_argument("--detect-every-n-frames", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reopen-interval", type=float, default=3.0,
                         help="Seconds between reopen attempts for a camera that failed to open.")
    return parser.parse_args(argv)


_GENDER_COLOR = {"male": (255, 179, 94), "female": (102, 102, 255)}
_DEFAULT_COLOR = (150, 150, 150)


def _draw_result(frame, bbox, result, det_conf):
    import cv2

    x1, y1, x2, y2 = (int(v) for v in bbox)
    color = _GENDER_COLOR.get(result["gender"]["label"].lower(), _DEFAULT_COLOR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"{result['gender']['label']} {result['gender']['confidence']:.0%}",
        f"{result['age']['display_label']} {result['age']['confidence']:.0%}",
        f"{result['race']['label']} {result['race']['confidence']:.0%}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x1, max(15, y1 - 10 - 18 * (len(lines) - 1 - i))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _placeholder_frame(label: str, message: str):
    import numpy as np
    import cv2

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, message, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


class CameraSlot:
    def __init__(self, label: str, device, reopen_interval: float):
        from src.camera import Camera

        self.label = label
        self.camera = Camera(device, label)
        self.reopen_interval = reopen_interval
        self.last_attempt = 0.0
        self.frame_count = 0
        self.last_result = None  # (bbox, result, det_conf) reused between detect passes
        self.timestamps: deque = deque(maxlen=30)
        self.attempt_open()

    def attempt_open(self) -> bool:
        self.last_attempt = time.monotonic()
        ok = self.camera.open()
        if ok:
            print(f"{self.label}: connected")
        return ok

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / span if span > 0 else 0.0


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.device != "cpu":
        from src.gpu_probe import ensure_cuda_env
        ensure_cuda_env()

    print(f"Loading model: {args.checkpoint}")
    try:
        from src.inference import FairFacePredictor
        predictor = FairFacePredictor(args.checkpoint, device=args.device)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    print(f"Device: {predictor.device.type}")

    from src.face_detector import FaceDetector, crop_face
    try:
        detector = FaceDetector(yunet_model_path=args.yunet_model)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    print(f"Face detector backend: {detector.backend_name}")

    device_specs = [("Camera 0", args.camera0)]
    if args.camera1 and args.camera1.lower() != "none":
        device_specs.append(("Camera 1", args.camera1))

    slots = [CameraSlot(label, device, args.reopen_interval) for label, device in device_specs]
    for slot in slots:
        if not slot.camera.is_open:
            print(f"{slot.label}: unavailable ({slot.camera.device})")

    import cv2

    print("\nPress 'q' in a video window (or Ctrl+C here) to stop.\n")
    try:
        while True:
            for slot in slots:
                if not slot.camera.is_open:
                    now = time.monotonic()
                    if now - slot.last_attempt >= slot.reopen_interval:
                        slot.attempt_open()
                    cv2.imshow(slot.label, _placeholder_frame(slot.label, f"unavailable ({slot.camera.device})"))
                    continue

                ok, frame = slot.camera.read()
                if not ok or frame is None:
                    slot.camera.release()
                    print(f"{slot.label}: stream lost, will retry")
                    continue

                slot.frame_count += 1
                slot.timestamps.append(time.monotonic())

                infer_ms = None
                if slot.frame_count % args.detect_every_n_frames == 0:
                    t0 = time.monotonic()
                    face = detector.detect_largest(frame)
                    if face is not None:
                        bbox, det_conf = face
                        crop = crop_face(frame, bbox)
                        result = predictor.predict(crop)
                        slot.last_result = (bbox, result, det_conf)
                    else:
                        slot.last_result = None
                    infer_ms = (time.monotonic() - t0) * 1000

                if slot.last_result is not None:
                    bbox, result, det_conf = slot.last_result
                    _draw_result(frame, bbox, result, det_conf)

                overlay = f"FPS: {slot.fps:.1f}  Device: {predictor.device.type.upper()}"
                if infer_ms is not None:
                    overlay += f"  Inference: {infer_ms:.1f} ms"
                cv2.putText(frame, overlay, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(slot.label, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Releasing webcam(s)...")
        for slot in slots:
            slot.camera.release()
        print("Closing windows...")
        cv2.destroyAllWindows()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
