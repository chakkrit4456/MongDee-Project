"""Shared photo fixtures (public-domain / CC0 sample images from scikit-image's data set)."""
import os
import cv2

_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "camera_ai")


def load(name, size=None):
    im = cv2.imread(os.path.join(_DIR, f"{name}.jpg"))
    if im is None:
        raise FileNotFoundError(name)
    return cv2.resize(im, size, interpolation=cv2.INTER_AREA) if size else im


def scenes(size=(320, 240)):
    return [load(n, size) for n in ("astronaut", "coffee", "chelsea", "rocket", "cat") if os.path.exists(os.path.join(_DIR, f"{n}.jpg"))]
