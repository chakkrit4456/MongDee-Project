"""Download the SFace face-recognition model (face IDENTITY) used by core/face_identity.py.

    python tools/get_face_models.py

Source: OpenCV Zoo (Apache-2.0). ~37 MB. Saved to models/face_recognizer/. Also fetches the YuNet detector
into models/face_detector/ if it is missing. Safe to re-run (existing files are kept, use --force to replace).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    ROOT / "models" / "face_recognizer" / "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()
    failed = 0
    for path, url in MODELS.items():
        if path.exists() and path.stat().st_size > 100_000 and not args.force:
            print(f"[OK] {path.name} already present")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[..] downloading {path.name} ...")
        try:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 100_000:
                raise RuntimeError("downloaded file is too small (blocked / not the model)")
            tmp.replace(path)
            print(f"[OK] saved {path}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path.name}: {exc}\n       download it manually from {url}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
