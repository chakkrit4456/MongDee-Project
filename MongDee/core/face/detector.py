"""Full-frame face detection with pluggable backends + a face-quality score.

YuNetFaceDetector : primary (opencv_zoo YuNet, MIT/Apache; model file supplied by the caller).
HaarFaceDetector  : dependency-free fallback used in tests / when the ONNX is missing.
Both return List[FaceBox] in ORIGINAL frame pixel coordinates.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import cv2
import numpy as np


@dataclass
class FaceBox:
    x1: float; y1: float; x2: float; y2: float
    score: float = 1.0
    landmarks: Optional[np.ndarray] = None   # (5,2): r-eye, l-eye, nose, r-mouth, l-mouth (YuNet order)
    quality: float = 0.0
    track_id: Optional[int] = None

    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1
    @property
    def center(self): return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def xyxy(self): return (self.x1, self.y1, self.x2, self.y2)


def _clip(box, W, H):
    return (max(0, int(round(box[0]))), max(0, int(round(box[1]))),
            min(W, int(round(box[2]))), min(H, int(round(box[3]))))


def crop_with_margin(frame, box: FaceBox, margin: float = 0.25):
    H, W = frame.shape[:2]
    mx, my = box.w * margin, box.h * margin
    x1, y1, x2, y2 = _clip((box.x1 - mx, box.y1 - my, box.x2 + mx, box.y2 + my), W, H)
    return frame[y1:y2, x1:x2]


def face_quality(frame, box: FaceBox) -> float:
    """0..1 = size * sharpness * exposure * (frontalness if landmarks). Used for best-shot
    selection and to WEIGHT attribute votes (small/blurry faces barely count)."""
    crop = crop_with_margin(frame, box, 0.0)
    if crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    size = float(np.clip((min(box.w, box.h) - 16) / 64.0, 0.0, 1.0))               # 16px->0, 80px->1
    g64 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    sharp = float(np.clip(cv2.Laplacian(g64, cv2.CV_32F).var() / 150.0, 0.0, 1.0))
    m = float(g.mean())
    expo = 1.0 - min(abs(m - 118.0) / 118.0, 1.0) ** 2
    front = 1.0
    lm = box.landmarks
    if lm is not None and len(lm) >= 3:
        re, le, nose = lm[0], lm[1], lm[2]
        dl, dr = abs(nose[0] - le[0]), abs(re[0] - nose[0])  # symmetric nose position => frontal
        front = float(np.clip(1.0 - abs(dl - dr) / max(dl + dr, 1e-6) * 1.5, 0.1, 1.0))
    return float(np.clip(size * (0.35 + 0.65 * sharp) * expo * front * min(box.score * 1.2, 1.0), 0.0, 1.0))


class YuNetFaceDetector:
    def __init__(self, model_path: str, score_thr: float = 0.6, nms_thr: float = 0.3,
                 top_k: int = 20, upscale: float = 1.0):
        if not hasattr(cv2, "FaceDetectorYN_create") and not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV without FaceDetectorYN")
        create = getattr(cv2, "FaceDetectorYN_create", None) or cv2.FaceDetectorYN.create
        self._det = create(model_path, "", (320, 320), score_thr, nms_thr, top_k)
        self.upscale = float(upscale)
        self.score_thr = float(score_thr)

    def detect_in_region(self, frame, region, score_thr: float = 0.35, target_px: int = 160) -> List[FaceBox]:
        """Second-chance search for a face we ALREADY know is around here (a tracked person's face that the
        full-frame pass just missed): crop `region` (x1,y1,x2,y2), enlarge it and accept a lower score. The
        boxes come back in full-frame coordinates. Never raises."""
        try:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = _clip(region, W, H)
            if x2 - x1 < 12 or y2 - y1 < 12:
                return []
            crop = frame[y1:y2, x1:x2]
            scale = float(np.clip(target_px / max(x2 - x1, y2 - y1), 1.0, 4.0))
            img = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1.0 else crop
            h, w = img.shape[:2]
            self._det.setInputSize((w, h))
            self._det.setScoreThreshold(float(score_thr))
            try:
                _, faces = self._det.detect(img)
            finally:
                self._det.setScoreThreshold(self.score_thr)
            out: List[FaceBox] = []
            for f in (faces if faces is not None else []):
                fx, fy, fw, fh = f[:4] / scale
                lm = (f[4:14].reshape(5, 2) / scale + np.array([x1, y1], np.float32)).astype(np.float32)
                out.append(FaceBox(float(fx + x1), float(fy + y1), float(fx + x1 + fw), float(fy + y1 + fh),
                                   float(f[14]), lm))
            return out
        except Exception:
            return []

    def detect(self, frame) -> List[FaceBox]:
        img = frame
        s = self.upscale
        if s != 1.0:
            img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img)
        out: List[FaceBox] = []
        if faces is None:
            return out
        for f in faces:
            x, y, fw, fh = f[:4] / s
            lm = (f[4:14].reshape(5, 2) / s).astype(np.float32)
            out.append(FaceBox(float(x), float(y), float(x + fw), float(y + fh), float(f[14]), lm))
        return out


class HaarFaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=24, upscale: float = 1.0):
        path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self._c = cv2.CascadeClassifier(path)
        if self._c.empty():
            raise RuntimeError("haar cascade not found")
        self.sf, self.mn, self.ms, self.upscale = scale_factor, min_neighbors, min_size, upscale

    def detect(self, frame) -> List[FaceBox]:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = self.upscale
        if s != 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        g = cv2.equalizeHist(g)
        rects = self._c.detectMultiScale(g, self.sf, self.mn, minSize=(int(self.ms), int(self.ms)))
        return [FaceBox(x / s, y / s, (x + w) / s, (y + h) / s, 0.8) for (x, y, w, h) in rects]
