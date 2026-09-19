"""Simulated webcam for developing without hardware (KIOSK_FAKE_CAMERA=1).

FakeCapture mimics the slice of cv2.VideoCapture that CameraManager uses. It
renders an empty turntable (any photo of it, else a synthetic one) and, on
demand, composites a product image onto the platform that "rotates" by
squashing/mirroring it — enough change per frame for presence detection,
burst capture, masking and embedding to all run for real.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app import config
from app.camera import imread_unicode

EMPTY = "empty"
FPS = 15
ROTATION_PERIOD_S = 6.0
PRODUCT_BOX = 300  # max product edge in px at 1280x720
PLATFORM_CENTER = (0.43, 0.70)  # x, y as fractions of the frame; product base sits here
AUTO_EMPTY_S = 10.0
AUTO_PRODUCT_S = 40.0
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


_read_image = imread_unicode


def _synthetic_background() -> np.ndarray:
    w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    frame = np.full((h, w, 3), (150, 145, 140), np.uint8)
    cx, cy = int(PLATFORM_CENTER[0] * w), int(PLATFORM_CENTER[1] * h)
    cv2.ellipse(frame, (cx, cy), (260, 70), 0, 0, 360, (235, 235, 235), -1)
    return frame


class SceneControl:
    """Which scene the fake camera shows; shared with the dev router."""

    def __init__(self):
        self._lock = threading.Lock()
        self._scene = EMPTY
        self._auto = config.FAKE_CAMERA_AUTO
        self._auto_t0 = time.monotonic()
        self._background: Optional[np.ndarray] = None
        self._products: dict[str, tuple[np.ndarray, int, int]] = {}
        self.reload()

    def reload(self) -> None:
        folder = config.FAKE_CAMERA_DIR
        background = None
        for candidate in (folder / "empty.png", folder / "empty.jpg",
                          config.DATA_DIR / "reference_camera_0.png"):
            if candidate.exists():
                background = _read_image(candidate)
                if background is not None:
                    break
        if background is None:
            background = _synthetic_background()
        background = cv2.resize(background, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        sources = folder / "products"
        if not sources.is_dir() or not any(p.suffix.lower() in _IMAGE_EXTS for p in sources.iterdir()):
            sources = config.CAPTURES_DIR / "covers"
        products = {}
        if sources.is_dir():
            for path in sorted(sources.iterdir()):
                if path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                img = _read_image(path)
                if img is None:
                    continue
                scale = PRODUCT_BOX / max(img.shape[:2])
                w, h = max(8, round(img.shape[1] * scale)), max(8, round(img.shape[0] * scale))
                products[path.stem] = (cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), w, h)
        with self._lock:
            self._background, self._products = background, products
            if self._scene != EMPTY and self._scene not in products:
                self._scene = EMPTY

    def scenes(self) -> list[str]:
        with self._lock:
            return [EMPTY, *self._products]

    def set_scene(self, name: str) -> bool:
        with self._lock:
            if name != EMPTY and name not in self._products:
                return False
            self._scene, self._auto = name, False
            return True

    def set_auto(self, on: bool) -> None:
        with self._lock:
            self._auto = on
            self._auto_t0 = time.monotonic()

    def state(self) -> dict:
        return {"scene": self.current(), "auto": self._auto, "scenes": self.scenes()}

    def current(self) -> str:
        with self._lock:
            if not self._auto or not self._products:
                return self._scene
            names = list(self._products)
            cycle = AUTO_EMPTY_S + AUTO_PRODUCT_S
            elapsed = time.monotonic() - self._auto_t0
            index = int(elapsed // cycle) % len(names)
            return names[index] if elapsed % cycle >= AUTO_EMPTY_S else EMPTY

    def render(self, t: float) -> np.ndarray:
        scene = self.current()
        with self._lock:
            frame = self._background.copy()
            product = self._products.get(scene)
        if product is not None:
            img, w0, h = product
            c = math.cos(2 * math.pi * t / ROTATION_PERIOD_S)
            view = img if c >= 0 else cv2.flip(img, 1)
            w = max(8, int(w0 * (0.55 + 0.45 * abs(c))))
            view = cv2.resize(view, (w, h), interpolation=cv2.INTER_AREA)
            cx = int(PLATFORM_CENTER[0] * frame.shape[1])
            base = int(PLATFORM_CENTER[1] * frame.shape[0]) + 15
            x0, y0 = cx - w // 2, base - h
            shadow = frame.copy()
            cv2.ellipse(shadow, (cx, base), (w // 2 + 25, 16), 0, 0, 360, (40, 40, 40), -1)
            frame = cv2.addWeighted(shadow, 0.35, frame, 0.65, 0)
            frame[y0:y0 + h, x0:x0 + w] = view
        noise = np.random.normal(0, 2.0, frame.shape[:2] + (1,))
        return np.clip(frame + noise, 0, 255).astype(np.uint8)


scene_control = SceneControl()


class FakeCapture:
    def __init__(self):
        self._open = True
        self._t0 = time.monotonic()
        self._next = self._t0

    def isOpened(self) -> bool:
        return self._open

    def set(self, *_args) -> bool:
        return True

    def get(self, prop) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: float(config.FRAME_WIDTH),
                cv2.CAP_PROP_FRAME_HEIGHT: float(config.FRAME_HEIGHT)}.get(prop, 0.0)

    def read(self):
        if not self._open:
            return False, None
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next, time.monotonic() - 1.0 / FPS) + 1.0 / FPS
        return True, scene_control.render(time.monotonic() - self._t0)

    def release(self) -> None:
        self._open = False
