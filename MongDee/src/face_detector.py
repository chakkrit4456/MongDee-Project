"""Face detector for the standalone test_model.py / detect.py tools —
deliberately never the FairFace attribute classifier itself (spec: a
classifier must never double as its own detector).

Two backends:
  - YuNet (cv2.FaceDetectorYN): accurate DNN detector, but its ONNX weights
    aren't bundled with opencv-python (same "bring your own weights" policy
    as vision/attributes/gender_age_backend.py) — used when --yunet-model
    points at a real file.
  - Haar cascade (cv2.CascadeClassifier): used automatically otherwise.
    Its .xml ships inside the opencv-python wheel itself
    (cv2.data.haarcascades), so this backend needs no download at all and
    is always available as the zero-setup default.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, yunet_model_path: str | Path | None = None, score_threshold: float = 0.6):
        self._backend_name = "haar"
        self._yunet = None
        self._haar = None

        if yunet_model_path and Path(yunet_model_path).is_file():
            self._yunet = cv2.FaceDetectorYN_create(
                str(yunet_model_path), "", (320, 320), score_threshold, 0.3, 5000)
            self._backend_name = "yunet"
            return

        # Haar's cascade data isn't bundled by every opencv-python build (this
        # project has observed builds — see core/attributes.py's
        # YuNetFaceDetector docstring — that ship neither the .xml files nor
        # even the cv2.CascadeClassifier binding at all). Detected here
        # rather than assumed, so the failure is an explicit, actionable
        # error instead of a confusing AttributeError deep in detect().
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not hasattr(cv2, "CascadeClassifier") or not cascade_path.is_file():
            raise RuntimeError(
                "No face detector is available: this opencv-python build ships neither "
                "cv2.CascadeClassifier nor its Haar cascade data, and no --yunet-model was "
                "given.\n\nPass --yunet-model pointing at face_detection_yunet_2023mar.onnx "
                "(https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), "
                "or use --skip-face-detection if your input images are already tightly "
                "cropped to a single face (e.g. FairFace's own val/ images)."
            )
        self._haar = cv2.CascadeClassifier(str(cascade_path))
        if self._haar.empty():
            raise RuntimeError(f"Failed to load bundled Haar cascade from {cascade_path}")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[list[float], float]]:
        """Returns [(bbox_xyxy, confidence), ...] sorted largest-first.
        Confidence is a real calibrated score for YuNet, or a fixed 1.0
        placeholder for Haar (which doesn't produce one)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        results = []
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _retval, faces = self._yunet.detect(frame_bgr)
            if faces is not None:
                for f in faces:
                    x, y, fw, fh, score = f[0], f[1], f[2], f[3], f[14]
                    results.append(([float(x), float(y), float(x + fw), float(y + fh)], float(score)))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            for (x, y, fw, fh) in boxes:
                results.append(([float(x), float(y), float(x + fw), float(y + fh)], 1.0))

        results.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
        return results

    def detect_largest(self, frame_bgr: np.ndarray) -> tuple[list[float], float] | None:
        results = self.detect(frame_bgr)
        return results[0] if results else None


def crop_face(frame_bgr: np.ndarray, bbox: list[float], margin: float = 0.2) -> np.ndarray:
    """Crop with a small margin around the detected face box — FairFace's
    own training preprocessing (align, padding=0.25) already bakes in
    context around the face, so inference crops match that expectation
    better than a tight box."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return frame_bgr[y1:y2, x1:x2]
