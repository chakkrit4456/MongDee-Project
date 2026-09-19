"""Hand-crafted whole-body appearance cues from a person crop (numpy + OpenCV only, no model).

Used for two things:
  * `color_descriptor` - clothing colour layout of torso and legs, blended into the Re-ID embedding so two
    people in clearly different clothes are not merged and one person is easier to re-find;
  * `body_features` - a small feature vector (head/hair, torso, legs, silhouette proportions, clothing colour)
    for the online body-gender model in core/body_gender.py.

Everything is computed on a 48x96 (w x h) resize, so it works on the tiny far-away crops of a 320x240 stream.
The cues are deliberately weak individually (hair length, exposed skin, colour, proportions): they only become
useful when a model learns how they relate to gender on THIS booth's own people (see core/body_gender.py).
"""
from __future__ import annotations

import cv2
import numpy as np

CROP_W, CROP_H = 48, 96
MIN_SIDE_PX = 12

# vertical bands as fractions of the person box height
HEAD_BAND = (0.00, 0.17)
NECK_BAND = (0.17, 0.26)
TORSO_BAND = (0.22, 0.55)
LEGS_BAND = (0.55, 0.95)

_HUE_BINS = 12
_ACHRO_BINS = 4
COLOR_BINS = _HUE_BINS + _ACHRO_BINS
COLOR_DESCRIPTOR_DIM = 2 * COLOR_BINS      # torso + legs


def _prepare(crop_bgr: np.ndarray) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.shape[0] < MIN_SIDE_PX or crop_bgr.shape[1] < MIN_SIDE_PX // 2:
        return None
    return cv2.resize(crop_bgr, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def _band(img: np.ndarray, band: tuple[float, float], side_margin: float = 0.12) -> np.ndarray:
    h, w = img.shape[:2]
    y1, y2 = int(band[0] * h), max(int(band[1] * h), int(band[0] * h) + 1)
    x1, x2 = int(side_margin * w), max(int((1 - side_margin) * w), int(side_margin * w) + 1)
    return img[y1:y2, x1:x2]


def _color_hist(band_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    chromatic = (sat >= 45) & (val >= 45)
    hist = np.zeros(COLOR_BINS, dtype=np.float32)
    if chromatic.any():
        # Soft, circular binning: red sits on the hue wrap-around (hue ~0 and ~179 are the same colour), so each
        # pixel is shared between its two nearest bin centres instead of being cut off at a bin edge.
        pos = hue[chromatic] / 180.0 * _HUE_BINS - 0.5
        lo = np.floor(pos).astype(int)
        frac = (pos - lo).astype(np.float32)
        np.add.at(hist, lo % _HUE_BINS, 1.0 - frac)
        np.add.at(hist, (lo + 1) % _HUE_BINS, frac)
    achro = ~chromatic
    if achro.any():
        hist[_HUE_BINS:] = np.bincount(np.minimum((val[achro] / 256.0 * _ACHRO_BINS).astype(int), _ACHRO_BINS - 1),
                                       minlength=_ACHRO_BINS)
    total = hist.sum()
    return hist / total if total > 0 else hist


# ------------------------------------------------- camera-independent clothing colour (used for Re-ID only)
_WB_GAIN_CLAMP = (0.70, 1.40)      # a colour cast is never stronger than this; more would erase real clothing colour
_EXPOSURE_TARGET = 190.0           # the crop's 90th-percentile brightness is brought to this
_EXPOSURE_CLAMP = (0.60, 1.80)


def normalize_illumination(img_bgr: np.ndarray) -> np.ndarray:
    """Undo the part of a crop's colour that comes from the CAMERA, not the clothes: a gray-world white balance
    (clamped, so a person in an all-red shirt keeps a red shirt) and an exposure normalisation. Two cameras with
    different white balance / auto-exposure looking at the same person then produce nearly the same colours."""
    x = img_bgr.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1.0), *_WB_GAIN_CLAMP)
    x = x * gains
    p90 = float(np.percentile(x.mean(axis=2), 90))
    x = x * float(np.clip(_EXPOSURE_TARGET / max(p90, 1.0), *_EXPOSURE_CLAMP))
    return np.clip(x, 0, 255).astype(np.uint8)


def _soft_color_hist(band_bgr: np.ndarray) -> np.ndarray:
    """Like _color_hist, but every pixel is shared between the two nearest hue / brightness bins, so a small hue
    shift between two cameras moves weight to the neighbouring bin instead of jumping across a bin edge."""
    hsv = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hue, sat, val = hsv[:, 0] * 2.0, hsv[:, 1], hsv[:, 2]          # hue in degrees
    chromatic = (sat >= 45) & (val >= 45)
    hist = np.zeros(COLOR_BINS, dtype=np.float32)
    if chromatic.any():
        centre = hue[chromatic] / (360.0 / _HUE_BINS) - 0.5
        lo = np.floor(centre).astype(int)
        frac = centre - lo
        np.add.at(hist, lo % _HUE_BINS, 1.0 - frac)
        np.add.at(hist, (lo + 1) % _HUE_BINS, frac)
    achro = ~chromatic
    if achro.any():
        centre = val[achro] / 256.0 * _ACHRO_BINS - 0.5
        lo = np.floor(centre).astype(int)
        frac = centre - lo
        np.add.at(hist, _HUE_BINS + np.clip(lo, 0, _ACHRO_BINS - 1), 1.0 - frac)
        np.add.at(hist, _HUE_BINS + np.clip(lo + 1, 0, _ACHRO_BINS - 1), frac)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _centered_unit(hist: np.ndarray) -> np.ndarray:
    v = np.sqrt(hist)                      # Hellinger: robust to a few stray pixels
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.zeros_like(v)


def color_descriptor(crop_bgr: np.ndarray) -> np.ndarray:
    """Clothing colour layout (torso + legs), each band a centred unit vector; the whole thing has norm sqrt(2)
    when both bands are informative. Zeros if the crop is unusable."""
    img = _prepare(crop_bgr)
    if img is None:
        return np.zeros(COLOR_DESCRIPTOR_DIM, dtype=np.float32)
    img = normalize_illumination(img)
    parts = [_centered_unit(_soft_color_hist(_band(img, b))) for b in (TORSO_BAND, LEGS_BAND)]
    return np.concatenate(parts).astype(np.float32)


# ------------------------------------------------------------------ gender-related body cues
def _skin_fraction(band_bgr: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
    return float(skin.mean())


def _dark_fraction(band_bgr: np.ndarray, thresh: int = 80) -> float:
    return float((cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY) < thresh).mean())


def _edge_width(band_bgr: np.ndarray) -> float:
    """Horizontal extent (0..1) of the textured/foreground part of a band: a crude silhouette width."""
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_energy = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    if col_energy.size == 0 or col_energy.max() <= 1e-6:
        return 0.0
    active = col_energy >= 0.35 * col_energy.max()
    idx = np.flatnonzero(active)
    return float((idx[-1] - idx[0] + 1) / len(col_energy)) if idx.size else 0.0


BODY_FEATURE_NAMES = (
    ["aspect", "head_dark", "head_skin", "head_width", "hair_side_dark", "hair_side_dark_low",
     "neck_skin", "torso_skin", "legs_skin", "torso_width", "legs_width", "torso_legs_width_ratio",
     "torso_sat", "legs_sat", "torso_val", "legs_val", "torso_dark", "legs_dark"]
    + [f"torso_c{i}" for i in range(COLOR_BINS)] + [f"legs_c{i}" for i in range(COLOR_BINS)]
)
BODY_FEATURE_DIM = len(BODY_FEATURE_NAMES)


def body_features(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Fixed-length feature vector for one person crop, or None if the crop is too small to say anything."""
    img = _prepare(crop_bgr)
    if img is None:
        return None
    h0, w0 = crop_bgr.shape[:2]
    head = _band(img, HEAD_BAND, 0.05)
    neck_l = _band(img, NECK_BAND, 0.0)[:, : CROP_W // 4]                 # left quarter at neck height
    neck_r = _band(img, NECK_BAND, 0.0)[:, -CROP_W // 4:]                 # right quarter at neck height
    neck_c = _band(img, NECK_BAND, 0.30)
    below_l = _band(img, (0.26, 0.40), 0.0)[:, : CROP_W // 4]
    below_r = _band(img, (0.26, 0.40), 0.0)[:, -CROP_W // 4:]
    torso, legs = _band(img, TORSO_BAND), _band(img, LEGS_BAND)
    hsv_t = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hsv_l = cv2.cvtColor(legs, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    tw, lw = _edge_width(torso), _edge_width(legs)
    scalar = [
        w0 / max(h0, 1),
        _dark_fraction(head), _skin_fraction(head), _edge_width(head),
        0.5 * (_dark_fraction(neck_l) + _dark_fraction(neck_r)),          # hair falling beside the neck
        0.5 * (_dark_fraction(below_l) + _dark_fraction(below_r)),        # ... and below it (long hair)
        _skin_fraction(neck_c), _skin_fraction(torso), _skin_fraction(legs),
        tw, lw, tw / lw if lw > 1e-3 else 1.0,
        hsv_t[:, 1].mean() / 255.0, hsv_l[:, 1].mean() / 255.0,
        hsv_t[:, 2].mean() / 255.0, hsv_l[:, 2].mean() / 255.0,
        _dark_fraction(torso), _dark_fraction(legs),
    ]
    colours = np.concatenate([_color_hist(torso), _color_hist(legs)])
    return np.concatenate([np.asarray(scalar, dtype=np.float32), colours.astype(np.float32)])


# ------------------------------------------------------------------ hairstyle cue (for identity, not gender)
HAIR_DESCRIPTOR_DIM = 6


def hair_descriptor(crop_bgr: np.ndarray) -> "np.ndarray | None":
    """Coarse hairstyle of one person crop, six values in 0..1 (illumination-normalised so two cameras agree):
    how much of the head band is dark, how wide the head+hair is, how much hair falls beside the neck and below it
    (long hair), and the head band's saturation / brightness (hair colour - black, brown, dyed...). None when the crop
    is too small. Two looks of one person differ by little; a bob and a ponytail, or black and blonde, differ a lot."""
    img = _prepare(crop_bgr)
    if img is None:
        return None
    img = normalize_illumination(img)
    head = _band(img, HEAD_BAND, 0.05)
    neck_l = _band(img, NECK_BAND, 0.0)[:, : CROP_W // 4]
    neck_r = _band(img, NECK_BAND, 0.0)[:, -CROP_W // 4:]
    below_l = _band(img, (0.26, 0.40), 0.0)[:, : CROP_W // 4]
    below_r = _band(img, (0.26, 0.40), 0.0)[:, -CROP_W // 4:]
    hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    dark = hsv[:, 2] < 90
    hair_sat = float(hsv[dark, 1].mean() / 255.0) if dark.any() else 0.0
    hair_val = float(hsv[dark, 2].mean() / 255.0) if dark.any() else float(hsv[:, 2].mean() / 255.0)
    return np.asarray([
        _dark_fraction(head, 90), _edge_width(head),
        0.5 * (_dark_fraction(neck_l, 90) + _dark_fraction(neck_r, 90)),
        0.5 * (_dark_fraction(below_l, 90) + _dark_fraction(below_r, 90)),
        hair_sat, hair_val,
    ], dtype=np.float32)


def hair_similarity(a: "np.ndarray | None", b: "np.ndarray | None") -> "float | None":
    """1.0 = same hairstyle; falls as the most different aspect (length, width, colour) grows. None if unknown."""
    if a is None or b is None:
        return None
    return float(1.0 - np.max(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32))))
