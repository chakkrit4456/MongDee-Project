"""text_render - draw Unicode (Thai) labels onto OpenCV frames.

cv2.putText's Hershey fonts cannot draw Thai (every glyph becomes "?"), which is why product boxes used to
say "PRODUCT" or a raw id such as "PRODUCT-4621553D" instead of the product's name. This module renders a label
with Pillow and a Thai-capable TrueType font (Leelawadee UI / Tahoma on Windows, Noto Sans Thai / Loma on Linux,
or the file named in the MONGDEE_LABEL_FONT environment variable), caches the rendered label as a small BGR patch
and blits it, so drawing it every captured frame costs a slice copy, not a font rasteriser call.

If no usable font is found `unicode_labels_available()` is False and callers fall back to an ASCII label.
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict

import cv2
import numpy as np

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\LeelawUI.ttf", r"C:\Windows\Fonts\leelawui.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\LeelUIsl.ttf", r"C:\Windows\Fonts\cordia.ttc", r"C:\Windows\Fonts\angsa.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf", "/usr/share/fonts/truetype/tlwg/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/Garuda.ttf", "/usr/share/fonts/opentype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansThai-Regular.ttf", "/System/Library/Fonts/Supplemental/Thonburi.ttc",
    "/System/Library/Fonts/Thonburi.ttc",
)
_CACHE_MAX = 256
_lock = threading.Lock()
_font_path: "str | None | bool" = False          # False = not resolved yet, None = none available
_fonts: dict = {}
_patches: "OrderedDict[tuple, np.ndarray]" = OrderedDict()


def find_label_font() -> "str | None":
    global _font_path
    if _font_path is not False:
        return _font_path
    candidates = [os.environ.get("MONGDEE_LABEL_FONT", "")] + list(_FONT_CANDIDATES)
    _font_path = next((p for p in candidates if p and os.path.isfile(p)), None)
    return _font_path


def unicode_labels_available() -> bool:
    if find_label_font() is None:
        return False
    try:
        import PIL.ImageFont  # noqa: F401
    except Exception:
        return False
    return True


def _font(size_px: int):
    from PIL import ImageFont
    f = _fonts.get(size_px)
    if f is None:
        f = ImageFont.truetype(find_label_font(), size_px)
        _fonts[size_px] = f
    return f


def render_label(text: str, bg_bgr, fg_bgr=(255, 255, 255), size_px: int = 17, pad: int = 3) -> "np.ndarray | None":
    """A BGR patch with `text` on a `bg_bgr` background, or None when no Thai-capable font is available.
    Cached per (text, colours, size)."""
    if not unicode_labels_available():
        return None
    key = (text, tuple(int(v) for v in bg_bgr), tuple(int(v) for v in fg_bgr), size_px, pad)
    with _lock:
        patch = _patches.get(key)
        if patch is not None:
            _patches.move_to_end(key)
            return patch
    try:
        from PIL import Image, ImageDraw
        font = _font(size_px)
        left, top, right, bottom = font.getbbox(text)
        w, h = int(right - left) + 2 * pad, int(bottom - top) + 2 * pad
        img = Image.new("RGB", (max(w, 1), max(h, 1)), (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])))
        ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=(int(fg_bgr[2]), int(fg_bgr[1]), int(fg_bgr[0])))
        patch = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None
    with _lock:
        _patches[key] = patch
        while len(_patches) > _CACHE_MAX:
            _patches.popitem(last=False)
    return patch


def blit(frame: np.ndarray, patch: np.ndarray, x: int, y_bottom: int) -> None:
    """Paste `patch` with its bottom edge at y_bottom (flips below the point when there is no room above)."""
    fh, fw = frame.shape[:2]
    ph, pw = patch.shape[:2]
    y = y_bottom - ph
    if y < 0:
        y = min(y_bottom, max(fh - ph, 0))
    x = int(min(max(x, 0), max(fw - pw, 0)))
    h, w = min(ph, fh - y), min(pw, fw - x)
    if h > 0 and w > 0:
        frame[y:y + h, x:x + w] = patch[:h, :w]
