"""Shared "upload photos/video to train a product" logic.

Framework-agnostic (no Qt) so both the desktop AI Trainer
(ui/trainer_window.py, which wraps these in a QThread for progress signals)
and the browser-based trainer (web/server.py, which calls these directly
from a background thread) share one implementation instead of drifting.
This is also what the web trainer's 15-second guided-rotation "record from
camera" capture (web/static/trainer.js) ultimately feeds into, one JPEG
frame at a time, via import_images.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable

import cv2
import numpy as np

from core.localizer import crop_box, exclude_human_region

# Safety ceiling only -- NOT a target sample count. At VIDEO_SAMPLE_INTERVAL_SEC=0.4s this covers
# ~33 minutes of source video before sampling stops; a normal product-training clip (walkarounds,
# multiple angles/distances/lighting per spec) never gets near it. It exists purely so an
# accidentally-uploaded hours-long file can't sample forever -- the real limiter on how many
# samples actually get kept is _try_add's quality/near-duplicate gating, not this constant.
MAX_FRAMES_PER_VIDEO = 5000
VIDEO_SAMPLE_INTERVAL_SEC = 0.4

MIN_CROP_SIDE_PX = 24          # a crop smaller than this has no usable detail to embed
# Laplacian-variance floor for rejecting motion-blurred/out-of-focus frames.
# An initial, deliberately conservative heuristic (only catches frames that
# are genuinely blurry, not just low-texture products) -- not a rigorously
# swept optimum, same caveat as recognizer.py's MATCH_FLOOR/
# SINGLE_PRODUCT_MATCH_FLOOR: revisit once real registration footage is
# available to calibrate against.
BLUR_VARIANCE_MIN = 15.0
# Mean-brightness band (0-255 grayscale) outside which a crop is treated as
# unusably under/overexposed. Deliberately wide (only rejects the extremes
# -- a near-black or near-blown-out frame) for the same "conservative
# initial heuristic, not a swept optimum" reason as BLUR_VARIANCE_MIN.
EXPOSURE_MIN_MEAN = 15.0
EXPOSURE_MAX_MEAN = 240.0
# Above this centered-embedding cosine similarity to an already-accepted
# frame of the same product, a new frame is "the same view again" rather
# than a new, useful viewpoint -- matches the dedup threshold this
# project's own prior investigation proposed (see
# MongDee_Master_Prompt_Camera_Detection_Face_Fix.md's quality-filter plan).
DUPLICATE_COSINE_MAX = 0.98

ProgressCallback = Callable[[int, int], None]


def slugify(name: str, prefix: str = "product") -> str:
    """ASCII-safe id from a (often Thai) display name. `prefix` names the
    fallback used when the name has no ASCII characters at all to slugify
    (e.g. an all-Thai booth/event name) — callers outside the product
    catalog should pass their own so the fallback id reads sensibly."""
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return ascii_part or f"{prefix}-{uuid.uuid4().hex[:8]}"


def auto_crop(frame_bgr, model, model_device, person_segmenter=None) -> np.ndarray | None:
    """Localizes the product within a single uploaded photo or video frame:
    a class-agnostic YOLO pass (anything but a person), keep the highest-
    confidence box; when `person_segmenter` is given, tighten that box to
    exclude any detected human region (hand/arm/sleeve) inside it.

    Returns None (reject this frame) when no plausible product box exists,
    or when the box is mostly human -- this NEVER falls back to the whole
    raw frame any more. That fallback used to fire exactly when YOLO found
    nothing, i.e. exactly the frames most likely to be a hand/arm reaching
    toward the camera or bare background -- silently training the gallery
    on it. That contamination is the leading suspected root cause of both
    poor product representations and runtime false-positive matches
    against hands/background (see the project's own prior investigation
    doc, MongDee_Master_Prompt_Camera_Detection_Face_Fix.md)."""
    try:
        results = model.predict(source=frame_bgr, imgsz=640, conf=0.35,
                                 device=model_device, verbose=False)
    except Exception:
        return None
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = [b for b in results[0].boxes if model.names[int(b.cls.item())] != "person"]
    if not boxes:
        return None
    best = max(boxes, key=lambda b: float(b.conf.item()))
    bbox = [float(x) for x in best.xyxy[0].tolist()]
    if person_segmenter is not None:
        human_mask = person_segmenter.person_mask(frame_bgr)
        refined = exclude_human_region(bbox, human_mask)
        if refined is None:
            return None
        bbox = refined
    crop = crop_box(frame_bgr, bbox)
    return crop if crop.size else None


def _quality_reject_reason(crop: np.ndarray | None) -> str | None:
    if crop is None:
        return "no_product_region"
    h, w = crop.shape[:2]
    if h < MIN_CROP_SIDE_PX or w < MIN_CROP_SIDE_PX:
        return "too_small"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_VARIANCE_MIN:
        return "blurry"
    mean_brightness = float(gray.mean())
    if mean_brightness < EXPOSURE_MIN_MEAN:
        return "underexposed"
    if mean_brightness > EXPOSURE_MAX_MEAN:
        return "overexposed"
    return None


def _try_add(crop: np.ndarray | None, product_key: str, recognizer,
             seen_embeddings: list[np.ndarray], counts: dict[str, int]) -> bool:
    """Quality-gates and near-duplicate-gates `crop` before adding it as a
    training sample. `seen_embeddings` accumulates across one whole
    import_images/import_video call (not persisted) so a slow 360° turn
    doesn't add 30 near-identical frames of the same three seconds."""
    reason = _quality_reject_reason(crop)
    if reason:
        counts[reason] = counts.get(reason, 0) + 1
        return False
    embedding = recognizer.centered_embed(crop)
    if any(float(np.dot(embedding, other)) >= DUPLICATE_COSINE_MAX for other in seen_embeddings):
        counts["duplicate"] = counts.get("duplicate", 0) + 1
        return False
    recognizer.add_sample(product_key, crop)
    seen_embeddings.append(embedding)
    return True


def _validate_import(added: int, attempted: int, counts: dict[str, int]) -> None:
    """Registration must not silently "succeed" with zero usable frames —
    the caller needs a meaningful reason to show the user, not a product
    with an empty (or worse, contaminated) gallery."""
    if added > 0 or attempted == 0:
        return
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "unknown"
    raise RuntimeError(
        f"ไม่พบภาพสินค้าที่ใช้ได้เลยจาก {attempted} เฟรมที่ส่งมา ({reasons}) "
        "กรุณาถ่ายใหม่ให้เห็นสินค้าชัดเจน ไม่มีมือ/แขนบังเกินไป และภาพไม่เบลอ"
    )


def import_images(paths: list[str], product_key: str, recognizer, model, model_device,
                   progress_cb: ProgressCallback | None = None, person_segmenter=None) -> int:
    added = 0
    seen_embeddings: list[np.ndarray] = []
    counts: dict[str, int] = {}
    total = len(paths)
    for i, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            counts["unreadable"] = counts.get("unreadable", 0) + 1
        else:
            crop = auto_crop(image, model, model_device, person_segmenter)
            if _try_add(crop, product_key, recognizer, seen_embeddings, counts):
                added += 1
        if progress_cb:
            progress_cb(i + 1, total)
    _validate_import(added, total, counts)
    return added


def import_video(path: str, product_key: str, recognizer, model, model_device,
                  progress_cb: ProgressCallback | None = None, person_segmenter=None) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดไฟล์วิดีโอไม่ได้: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(fps * VIDEO_SAMPLE_INTERVAL_SEC))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        expected = min(MAX_FRAMES_PER_VIDEO, (total_frames // step) if total_frames else MAX_FRAMES_PER_VIDEO)

        added = 0
        sampled = 0
        seen_embeddings: list[np.ndarray] = []
        counts: dict[str, int] = {}
        frame_idx = 0
        while sampled < MAX_FRAMES_PER_VIDEO:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                sampled += 1
                crop = auto_crop(frame, model, model_device, person_segmenter)
                if _try_add(crop, product_key, recognizer, seen_embeddings, counts):
                    added += 1
                if progress_cb:
                    progress_cb(sampled, max(expected, sampled))
            frame_idx += 1
        _validate_import(added, sampled, counts)
        return added
    finally:
        cap.release()


class LiveTrainingSession:
    """Server-side state for the web trainer's guided-rotation "record from camera" flow
    (web/static/trainer.js): each captured tick is fed here IMMEDIATELY (one HTTP round-trip per
    tick, not one big batch upload at the end), quality/duplicate-gated and added to the gallery
    the exact same way import_images/import_video do (_try_add) -- so the frontend gets a live
    "N distinct views collected so far" count back from every feed() call, and can keep the
    recording going until that count is actually comprehensive instead of stopping after a blind
    fixed-duration timer. `added` only ever counts frames that were NOT near-duplicates of an
    already-accepted view (see DUPLICATE_COSINE_MAX) -- so it is a genuine coverage/diversity
    measure, not just a frame-processed count."""

    def __init__(self, product_key: str, recognizer, model, model_device, person_segmenter=None):
        self.product_key = product_key
        self.recognizer = recognizer
        self.model = model
        self.model_device = model_device
        self.person_segmenter = person_segmenter
        self.seen_embeddings: list[np.ndarray] = []
        self.counts: dict[str, int] = {}
        self.added = 0
        self.attempted = 0

    def feed(self, frame_bgr: np.ndarray) -> dict:
        """One captured tick. Never raises -- a single bad frame (blurry, hand-covered, a near-
        duplicate of one already collected) is reported back via `reason`, not treated as fatal;
        only finish() decides whether the WHOLE session produced anything usable."""
        self.attempted += 1
        crop = auto_crop(frame_bgr, self.model, self.model_device, self.person_segmenter)
        accepted = _try_add(crop, self.product_key, self.recognizer, self.seen_embeddings, self.counts)
        if accepted:
            self.added += 1
        reason = None if accepted else (_quality_reject_reason(crop) or "duplicate")
        return {"accepted": accepted, "distinct_views": self.added, "attempted": self.attempted, "reason": reason}

    def finish(self) -> dict:
        """Ends the session. Raises the same "found nothing usable" error import_images/
        import_video raise if literally nothing was accepted across the whole recording, so a
        product genuinely never rotated into view still gets a clear, actionable message instead
        of silently registering with an empty gallery."""
        _validate_import(self.added, self.attempted, self.counts)
        return {"added": self.added, "attempted": self.attempted, "counts": dict(self.counts)}
