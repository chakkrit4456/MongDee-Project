"""Shared "upload photos/video to train a product" logic.

Framework-agnostic (no Qt) so both the desktop AI Trainer
(ui/trainer_window.py, which wraps these in a QThread for progress signals)
and the browser-based trainer (web/server.py, which calls these directly
from a background thread) share one implementation instead of drifting.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

import cv2

from core.localizer import crop_box

MAX_FRAMES_PER_VIDEO = 60
VIDEO_SAMPLE_INTERVAL_SEC = 0.4

ProgressCallback = Callable[[int, int], None]


def slugify(name: str, prefix: str = "product") -> str:
    """ASCII-safe id from a (often Thai) display name. `prefix` names the
    fallback used when the name has no ASCII characters at all to slugify
    (e.g. an all-Thai booth/event name) — callers outside the product
    catalog should pass their own so the fallback id reads sensibly."""
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return ascii_part or f"{prefix}-{uuid.uuid4().hex[:8]}"


def auto_crop(frame_bgr, model, model_device):
    """Best-effort localization of the product within a single uploaded photo
    or video frame: try a class-agnostic YOLO pass (anything but a person),
    keep the largest box; fall back to the whole image if nothing is found —
    reasonable for typical product photos where the item fills most of the frame."""
    try:
        results = model.predict(source=frame_bgr, imgsz=640, conf=0.35,
                                 device=model_device, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = [b for b in results[0].boxes if model.names[int(b.cls.item())] != "person"]
            if boxes:
                best = max(boxes, key=lambda b: float(b.conf.item()))
                bbox = [float(x) for x in best.xyxy[0].tolist()]
                return crop_box(frame_bgr, bbox)
    except Exception:
        pass
    return frame_bgr


def import_images(paths: list[str], product_key: str, recognizer, model, model_device,
                   progress_cb: ProgressCallback | None = None) -> int:
    added = 0
    total = len(paths)
    for i, path in enumerate(paths):
        image = cv2.imread(path)
        if image is not None:
            crop = auto_crop(image, model, model_device)
            recognizer.add_sample(product_key, crop)
            added += 1
        if progress_cb:
            progress_cb(i + 1, total)
    return added


def import_video(path: str, product_key: str, recognizer, model, model_device,
                  progress_cb: ProgressCallback | None = None) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดไฟล์วิดีโอไม่ได้: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps * VIDEO_SAMPLE_INTERVAL_SEC))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        expected = min(MAX_FRAMES_PER_VIDEO, (total_frames // step) if total_frames else MAX_FRAMES_PER_VIDEO)

        added = 0
        frame_idx = 0
        while added < MAX_FRAMES_PER_VIDEO:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                crop = auto_crop(frame, model, model_device)
                recognizer.add_sample(product_key, crop)
                added += 1
                if progress_cb:
                    progress_cb(added, max(expected, added))
            frame_idx += 1
        return added
    finally:
        cap.release()
