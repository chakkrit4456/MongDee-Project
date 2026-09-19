"""Fresh frames, fixed empty reference and debounced presence detection."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app import config


def imread_unicode(path) -> Optional[np.ndarray]:
    # cv2.imread/imwrite can't open non-ASCII paths on Windows (e.g. a Thai user folder).
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite_unicode(path, image: np.ndarray, params=None) -> bool:
    ok, buf = cv2.imencode(Path(path).suffix, image, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True


class PresenceState(str, Enum):
    EMPTY = "empty"
    PRESENT = "present"


@dataclass(frozen=True)
class CameraSnapshot:
    frame: Optional[np.ndarray]
    state: PresenceState
    foreground_ratio: float
    updated_at: float
    generation: int = 0
    captured_at: float = 0.0  # monotonic clock for freshness
    motion_ratio: float = 0.0
    sharpness: float = 0.0

    @property
    def fresh(self) -> bool:
        return self.frame is not None and time.monotonic() - self.captured_at <= config.FRAME_MAX_AGE_S


def roi_box(frame: np.ndarray, fractions=None) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    fx, fy, fw, fh = fractions or config.ROI_FRACTION
    x = min(w - 1, max(0, int(fx * w)))
    y = min(h - 1, max(0, int(fy * h)))
    return x, y, max(1, min(w - x, int(fw * w))), max(1, min(h - y, int(fh * h)))


def crop_to_roi(frame: np.ndarray, fractions=None) -> np.ndarray:
    x, y, w, h = roi_box(frame, fractions)
    return frame[y:y + h, x:x + w]


def build_masked_crop(frame: np.ndarray, reference: Optional[np.ndarray], fractions=None) -> np.ndarray:
    """ROI crop with the booth/turntable background suppressed.

    Diffs the ROI against the fixed empty-platform reference (with a coarse
    exposure correction, like presence detection), keeps the product blob,
    feathers the mask, and blends non-product pixels toward a heavy blur of
    the crop itself — removing background structure without hard cut edges.

    Deterministic: enrollment and recognition run this identically, so their
    embeddings stay comparable. If no reference is available, or the product
    can't be isolated, the plain crop is returned (same choice both sides).
    """
    crop = crop_to_roi(frame, fractions)
    if reference is None or reference.shape != frame.shape:
        return crop
    ref_crop = crop_to_roi(reference, fractions)
    if ref_crop.shape != crop.shape:
        return crop
    live = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    base = cv2.GaussianBlur(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    live -= float(np.clip(np.median(live - base), -config.MAX_EXPOSURE_OFFSET, config.MAX_EXPOSURE_OFFSET))
    mask = (np.abs(live - base) > config.MASK_DIFF_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        min_area = max(64, int(mask.size * config.MIN_COMPONENT_RATIO))
        largest = 1 + int(np.argmax(stats[1:count, cv2.CC_STAT_AREA]))
        keep = {i for i in range(1, count) if i == largest or stats[i, cv2.CC_STAT_AREA] >= min_area}
        mask = np.isin(labels, list(keep)).astype(np.uint8)
    if float(mask.mean()) < config.MASK_MIN_COVERAGE:
        return crop
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    alpha = np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6.0), 0.0, 1.0)[:, :, None]
    background = cv2.GaussianBlur(crop, (0, 0), 25.0)
    return (crop * alpha + background * (1.0 - alpha)).astype(np.uint8)


def _small_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > config.PRESENCE_PROCESS_WIDTH:
        height = max(1, round(gray.shape[0] * config.PRESENCE_PROCESS_WIDTH / gray.shape[1]))
        gray = cv2.resize(gray, (config.PRESENCE_PROCESS_WIDTH, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _resized_roi_gray(frame: np.ndarray, fractions=None) -> np.ndarray:
    gray = cv2.cvtColor(crop_to_roi(frame, fractions), cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 320:
        gray = cv2.resize(gray, (320, max(1, round(gray.shape[0] * 320 / gray.shape[1]))))
    return gray


def frame_sharpness(frame: np.ndarray, fractions=None) -> float:
    """Laplacian-variance focus score of the ROI, same scale as select_quality_frames
    uses per-frame — lets the live status endpoint flag a blurry feed before capture."""
    return float(cv2.Laplacian(_resized_roi_gray(frame, fractions), cv2.CV_64F).var())


def select_quality_frames(frames: list[np.ndarray], fractions=None) -> list[np.ndarray]:
    """Drop dark/clipped and blurred captures, preserving rotation order."""
    scores = []
    for frame in frames:
        gray = _resized_roi_gray(frame, fractions)
        clipped = float(np.mean((gray < 4) | (gray > 251)))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        scores.append(sharpness if clipped < 0.90 else 0.0)
    if not scores:
        return []
    cutoff = max(config.MIN_SHARPNESS, float(np.percentile(scores, 75)) * 0.25)
    return [frame for frame, score in zip(frames, scores) if score >= cutoff]


class CameraManager:
    def __init__(self, index: int = config.CAMERA_INDEX):
        self._index = index
        self._cap = None
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._snapshot = CameraSnapshot(None, PresenceState.EMPTY, 0.0, 0.0)
        self._reference_frame = None
        self._frame_buffer = deque(maxlen=config.NOISE_SMOOTHING_FRAMES)
        self._recent_frames = deque(maxlen=7)
        self._previous_gray = None
        self._roi = tuple(config.ROI_FRACTION)
        self._on_streak = self._off_streak = 0
        self._on_since = self._off_since = None
        self._stop = threading.Event()
        self._thread = None
        self._opened_at = 0.0
        self._last_attempt = 0.0
        self._capture_revision = 0

    @property
    def capture_context(self):
        with self._lock:
            return self._index, self._capture_revision, tuple(config.ROI_FRACTION)

    @property
    def reference_path(self):
        # Never reuse a baseline whose camera identity is unknown.
        return config.DATA_DIR / f"reference_camera_{self._index}.png"

    def _try_open(self, index: int) -> bool:
        self._last_attempt = time.monotonic()
        if config.FAKE_CAMERA:
            from app.fake_camera import FakeCapture
            cap = FakeCapture()
        else:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._opened_at = time.monotonic()
        return True

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._load_reference_from_disk()
            self._try_open(self._index)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=3.0)
                if self._thread.is_alive():
                    raise RuntimeError("กล้องยังไม่หยุดทำงาน กรุณาลองใหม่")
            if self._cap:
                self._cap.release()
                self._cap = None
            with self._lock:
                self._invalidate()

    @property
    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened() and self.snapshot().fresh)

    @property
    def index(self) -> int:
        return self._index

    @property
    def has_reference(self) -> bool:
        with self._lock:
            return self._reference_frame is not None

    def masked_crop(self, frame: np.ndarray, fractions=None) -> np.ndarray:
        """ROI crop with the booth background removed against this camera's
        empty-platform reference — see build_masked_crop."""
        with self._lock:
            reference = self._reference_frame
        return build_masked_crop(frame, reference, fractions)

    def switch_index(self, new_index: int) -> bool:
        with self._lifecycle_lock:
            self.stop()
            self._index = new_index
            self._reference_frame = None
            self.start()
            return bool(self._cap and self._cap.isOpened())

    def _invalidate(self) -> None:
        self._capture_revision += 1
        self._frame_buffer.clear()
        self._recent_frames.clear()
        self._previous_gray = None
        self._on_streak = self._off_streak = 0
        self._on_since = self._off_since = None
        self._snapshot = CameraSnapshot(None, PresenceState.EMPTY, 0.0, 0.0, self._snapshot.generation + 1)

    def capture_reference_now(self) -> bool:
        with self._lock:
            if not self._snapshot.fresh or len(self._recent_frames) < 5:
                return False
            # Median rejects isolated noise; operator must empty the platform.
            frame = np.median(np.stack(self._recent_frames), axis=0).astype(np.uint8)
            self.reference_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.reference_path.with_name(f"reference_camera_{self._index}.tmp.png")
            if not imwrite_unicode(temp_path, frame):
                return False
            temp_path.replace(self.reference_path)
            self._reference_frame = frame
            snap = self._snapshot
            self._invalidate()
            self._snapshot = replace(snap, state=PresenceState.EMPTY, foreground_ratio=0.0,
                                     generation=self._snapshot.generation)
            return True

    def _load_reference_from_disk(self) -> None:
        with self._lock:
            self._reference_frame = imread_unicode(self.reference_path) if self.reference_path.exists() else None

    def _diff_ratio(self, frame: np.ndarray, roi=None) -> float:
        if self._reference_frame is None or self._reference_frame.shape != frame.shape:
            return 0.0
        live = _small_gray(frame).astype(np.float32)
        ref = _small_gray(self._reference_frame).astype(np.float32)
        x, y, w, h = roi_box(live, self._roi)
        # Estimate exposure OUTSIDE the ROI so the product isn't subtracted.
        outside = np.ones(live.shape, dtype=bool)
        outside[y:y + h, x:x + w] = False
        offset = 0.0
        if np.count_nonzero(outside) >= max(32, live.size * 0.01):
            offset = float(np.clip(np.median((live - ref)[outside]),
                                   -config.MAX_EXPOSURE_OFFSET, config.MAX_EXPOSURE_OFFSET))
        corrected = live[y:y + h, x:x + w] - offset
        self._frame_buffer.append(corrected)
        smoothed = np.mean(self._frame_buffer, axis=0)
        mask = (np.abs(smoothed - ref[y:y + h, x:x + w]) > config.DIFF_PIXEL_THRESHOLD).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        minimum = max(4, int(mask.size * config.MIN_COMPONENT_RATIO))
        areas = stats[1:count, cv2.CC_STAT_AREA]
        return float(areas[areas >= minimum].sum()) / mask.size

    def _process_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            roi = tuple(config.ROI_FRACTION)
            if roi != self._roi:
                self._roi = roi
                self._invalidate()
            if self._reference_frame is not None and self._reference_frame.shape != frame.shape:
                self._reference_frame = None
                self._invalidate()
            gray = _small_gray(crop_to_roi(frame, self._roi))
            motion = 0.0
            if self._previous_gray is not None and self._previous_gray.shape == gray.shape:
                motion = float(np.mean(cv2.absdiff(gray, self._previous_gray) > 20))
            self._previous_gray = gray
            self._recent_frames.append(frame.copy())
            ratio = self._diff_ratio(frame)
            state = self._snapshot.state
            generation = self._snapshot.generation
            now = time.monotonic()
            if state == PresenceState.EMPTY:
                self._on_streak = self._on_streak + 1 if self.has_reference and ratio >= config.PRESENCE_ON_RATIO else 0
                self._on_since = (now if self._on_since is None else self._on_since) if self._on_streak else None
                if self._on_streak >= config.PRESENCE_ON_FRAMES and now - self._on_since >= config.PRESENCE_ON_HOLD_S:
                    state = PresenceState.PRESENT
                    self._on_streak = 0
                    self._on_since = None
            else:
                self._off_streak = self._off_streak + 1 if ratio <= config.PRESENCE_OFF_RATIO else 0
                self._off_since = (now if self._off_since is None else self._off_since) if self._off_streak else None
                if self._off_streak >= config.PRESENCE_OFF_FRAMES and now - self._off_since >= config.PRESENCE_OFF_HOLD_S:
                    state = PresenceState.EMPTY
                    self._off_streak = 0
                    self._off_since = None
            if state != self._snapshot.state:
                generation += 1
            # Only the PRESENT state ever surfaces sharpness (see /api/camera/status'
            # box_status) — skip the extra Laplacian pass while the platform is empty.
            sharpness = frame_sharpness(frame, self._roi) if state == PresenceState.PRESENT else 0.0
            self._snapshot = CameraSnapshot(frame, state, ratio, time.time(), generation, time.monotonic(), motion, sharpness)

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    if self._cap is None or not self._cap.isOpened():
                        if time.monotonic() - self._last_attempt >= config.CAMERA_RECONNECT_INTERVAL_S:
                            self._try_open(self._index)
                        self._stop.wait(0.1)
                        continue
                    ok, frame = self._cap.read()
                    if ok:
                        if time.monotonic() - self._opened_at >= config.CAMERA_WARMUP_S:
                            self._process_frame(frame)
                    elif time.monotonic() - max(self._snapshot.captured_at, self._opened_at) > config.FRAME_MAX_AGE_S:
                        self._cap.release()
                        self._cap = None
                        with self._lock:
                            self._invalidate()
                    self._stop.wait(0.03)
                except Exception as exc:
                    print(f"[camera] capture error: {exc}")
                    with self._lock:
                        self._invalidate()
                    self._stop.wait(0.2)
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None

    def snapshot(self) -> CameraSnapshot:
        with self._lock:
            return self._snapshot

    def latest_jpeg(self) -> Optional[bytes]:
        snap = self.snapshot()
        if not snap.fresh:
            return None
        ok, buf = cv2.imencode(".jpg", snap.frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def capture_burst_frames(self, require_present: bool = True) -> list[np.ndarray]:
        initial = self.snapshot()
        context = self.capture_context
        roi = tuple(config.ROI_FRACTION)
        if not initial.fresh or (require_present and (not self.has_reference or initial.state != PresenceState.PRESENT)):
            return []
        deadline = time.monotonic() + config.BURST_DURATION_S
        next_sample = time.monotonic() + config.CAPTURE_SETTLE_S
        frames = []
        last_timestamp = 0.0
        while time.monotonic() < deadline and not self._stop.is_set():
            snap = self.snapshot()
            if not snap.fresh or self.capture_context != context:
                return []
            if require_present and (snap.state != PresenceState.PRESENT or snap.generation != initial.generation):
                return []
            if snap.motion_ratio > config.MAX_CAPTURE_MOTION:
                next_sample = time.monotonic() + config.CAPTURE_SETTLE_S
            elif (time.monotonic() >= next_sample and snap.captured_at != last_timestamp
                  and (not frames or not np.array_equal(frames[-1], snap.frame))):
                frames.append(snap.frame.copy())
                last_timestamp = snap.captured_at
                next_sample = time.monotonic() + config.BURST_SAMPLE_INTERVAL_S
            self._stop.wait(0.05)
        if self._stop.is_set():
            return []
        return select_quality_frames(frames, roi)


camera_manager = CameraManager()


def list_camera_devices(max_probe: int = 6) -> list[dict]:
    if config.FAKE_CAMERA:
        return [{"index": config.FAKE_CAMERA_INDEX, "active": True,
                 "width": config.FRAME_WIDTH, "height": config.FRAME_HEIGHT}]
    devices = []
    active_index = camera_manager.index if camera_manager._cap is not None else None
    for idx in range(max_probe):
        if idx == active_index:
            devices.append({"index": idx, "active": True, "width": None, "height": None})
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            devices.append({"index": idx, "active": False, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})
        cap.release()
    return devices
