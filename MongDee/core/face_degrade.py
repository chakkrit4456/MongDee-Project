"""Turn a clean face photo into what a booth webcam actually delivers - for training and for honest evaluation.

The FairFace network was trained on sharp, well-lit, roughly frontal faces. A face 5-8 m from a 320x240 or 640x480
USB camera arrives as 15-60 pixels, soft, JPEG-blocky, noisy, colour-cast, badly exposed and often half covered by
a hat / mask / hand, and it is then blown up to 224 px with a cubic filter before it reaches the network. That gap is
the main reason a model that scores ~95% on FairFace guesses much worse (and, on some faces, systematically towards
one gender) on the live picture. Fine-tuning on faces degraded in exactly this way, and measuring on them, closes it.

Pure numpy + OpenCV (no torch), deterministic for a given numpy Generator, so it is unit-tested and shared by
tools/finetune_fairface_gender.py and tools/eval_fairface_gender.py.
"""
from __future__ import annotations

import cv2
import numpy as np

OUT_SIZE = 224                 # what the network is fed
MIN_FACE_PX = 14               # smallest face the pipeline still analyses (core.attributes.FAR_FACE_MIN_PX)
MAX_FACE_PX = 160


def sample_face_px(rng: np.random.Generator) -> float:
    """Face size (source pixels) to simulate: log-uniform, so far away (small) faces are well represented."""
    return float(np.exp(rng.uniform(np.log(MIN_FACE_PX), np.log(MAX_FACE_PX))))


def _motion_blur(img: np.ndarray, rng: np.random.Generator, max_len: int) -> np.ndarray:
    length = int(rng.integers(3, max(4, max_len + 1)))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (length, length))
    s = kernel.sum()
    return cv2.filter2D(img, -1, kernel / s) if s > 0 else img


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def _occlude(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A hat brim, a mask, a hand or a shadow: a flat-coloured rectangle over the top / bottom / a side of the face."""
    h, w = img.shape[:2]
    out = img.copy()
    kind = int(rng.integers(0, 4))
    colour = tuple(int(c) for c in rng.integers(20, 235, 3))
    if kind == 0:                                   # hat / hair fringe: top band
        out[: int(h * rng.uniform(0.12, 0.30))] = colour
    elif kind == 1:                                 # surgical mask: lower third
        out[int(h * rng.uniform(0.60, 0.72)):] = colour
    elif kind == 2:                                 # hand / object at one side
        x = int(w * rng.uniform(0.0, 0.55))
        out[int(h * rng.uniform(0.3, 0.6)):, x: x + int(w * rng.uniform(0.2, 0.4))] = colour
    else:                                           # shadow across a random half
        shade = np.ones_like(out, dtype=np.float32)
        if rng.random() < 0.5:
            shade[:, : w // 2] = rng.uniform(0.35, 0.7)
        else:
            shade[:, w // 2:] = rng.uniform(0.35, 0.7)
        out = np.clip(out * shade, 0, 255).astype(np.uint8)
    return out


def degrade_face(face_bgr: np.ndarray, rng: np.random.Generator, face_px: "float | None" = None,
                 strength: float = 1.0, out_size: int = OUT_SIZE) -> np.ndarray:
    """Simulate one webcam observation of a clean face photo. Returns a BGR uint8 image of out_size x out_size.

    `face_px` - the size (in source pixels) the face should end up at before it is enlarged again; None draws one.
    `strength` - 0 gives only the size change, 1 the full set of photometric / compression / occlusion damage.
    The crop is also jittered (scale / shift / small rotation) because a detector box is never the aligned
    FairFace crop."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("empty face image")
    strength = float(np.clip(strength, 0.0, 1.0))
    if face_px is None:
        face_px = sample_face_px(rng)
    img = face_bgr
    h, w = img.shape[:2]

    # 1. detector-box jitter: crop scale, shift and a small in-plane rotation
    if strength > 0:
        scale = float(rng.uniform(0.82, 1.18))
        angle = float(rng.uniform(-12, 12)) * strength
        dx, dy = (float(v) * w for v in rng.uniform(-0.07, 0.07, 2))
        m = cv2.getRotationMatrix2D((w / 2 + dx, h / 2 + dy), angle, 1.0 / scale)
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img)

    # 2. optical blur BEFORE the sampling (lens / defocus / motion), stronger for small faces
    if strength > 0:
        if rng.random() < 0.55:
            img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.4, 1.6)) * strength * max(0.5, 60.0 / face_px))
        if rng.random() < 0.25:
            img = _motion_blur(img, rng, max_len=max(3, int(h * 0.06)))

    # 3. the actual loss of resolution
    px = max(8, int(round(face_px)))
    small = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)

    # 4. sensor / codec / exposure damage at the small size
    if strength > 0:
        x = small.astype(np.float32)
        if rng.random() < 0.9:                                          # exposure, gamma and per-channel white balance
            gain = rng.uniform(0.85, 1.18, 3) if rng.random() < 0.6 else np.ones(3)
            gamma = float(np.exp(rng.uniform(np.log(0.6), np.log(1.7))))
            bright = float(rng.uniform(0.35, 1.35))
            x = 255.0 * np.clip((x / 255.0) * gain, 0, 1) ** gamma * bright
        if rng.random() < 0.05:
            x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)      # IR / greyscale camera
        if rng.random() < 0.7:
            x = x + rng.normal(0, rng.uniform(1.0, 12.0) * strength, x.shape)
        small = np.clip(x, 0, 255).astype(np.uint8)
        if rng.random() < 0.7:
            small = _jpeg(small, int(rng.integers(20, 91)))
        if rng.random() < 0.15:
            small = _occlude(small, rng)

    # 5. what the pipeline does before the network: enlarge with a cubic filter
    return cv2.resize(small, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
