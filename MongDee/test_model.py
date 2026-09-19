"""Test a trained FairFace checkpoint against a single image.

    python test_model.py --image test.jpg
    python test_model.py --image test.jpg --checkpoint models/fairface/last_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to a JPG/PNG image containing a face.")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "fairface" / "best_model.pt"))
    _default_yunet = ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"
    parser.add_argument("--yunet-model", default=str(_default_yunet) if _default_yunet.is_file() else None,
                         help="YuNet ONNX face detector (auto-detected under models/face_detector/ if present); "
                              "falls back to the bundled Haar cascade otherwise (only available on some "
                              "opencv-python builds — see --skip-face-detection).")
    parser.add_argument("--skip-face-detection", action="store_true",
                         help="Treat the whole image as an already-cropped face (matches FairFace's own "
                              "val/ images) instead of running a face detector.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"ERROR: image was not found.\n\nExpected:\n{image_path}\n")
        return 1

    if args.device != "cpu":
        from src.gpu_probe import ensure_cuda_env
        ensure_cuda_env()

    import cv2

    from src.face_detector import FaceDetector, crop_face

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"ERROR: could not read image (unsupported format or corrupt file): {image_path}")
        return 1

    try:
        from src.inference import FairFacePredictor
        predictor = FairFacePredictor(args.checkpoint, device=args.device)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if args.skip_face_detection:
        face_crop = frame
        backend_name, det_conf = "skipped", 1.0
    else:
        try:
            detector = FaceDetector(yunet_model_path=args.yunet_model)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        face = detector.detect_largest(frame)
        if face is None:
            print(f"No face detected in {image_path} (face detector backend: {detector.backend_name}).")
            return 1
        bbox, det_conf = face
        face_crop = crop_face(frame, bbox)
        backend_name = detector.backend_name

    result = predictor.predict(face_crop)
    print(f"Image: {image_path}")
    print(f"Face detected (backend={backend_name}, confidence={det_conf:.2f})\n")
    print(f"Age: {result['age']['display_label']} (confidence: {result['age']['confidence']:.1%})")
    print(f"Gender: {result['gender']['label']} (confidence: {result['gender']['confidence']:.1%})")
    print(f"Race: {result['race']['label']} (confidence: {result['race']['confidence']:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
