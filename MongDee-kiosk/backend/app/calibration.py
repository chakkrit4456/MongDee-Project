"""Runtime-tunable presence-detection settings.

Unlike most of config.py, these need to change without a restart: staff
tune them by watching the live foreground ratio while setting up at the
actual venue (see README "On-site calibration"). camera.py's capture loop
reads config.ROI_FRACTION / config.PRESENCE_*_RATIO fresh on every frame
(never caches them at construction time), so mutating the config module's
attributes here takes effect on the very next frame — no restart required.
"""
from __future__ import annotations

import copy
import json

from app import config, db
from app.schemas import CalibrationSettings

_SETTINGS_KEY = "calibration"


def current() -> dict:
    x, y, w, h = config.ROI_FRACTION
    return {
        "roi": {"x": x, "y": y, "w": w, "h": h},
        "presence_on_ratio": config.PRESENCE_ON_RATIO,
        "presence_off_ratio": config.PRESENCE_OFF_RATIO,
    }


def apply(data: dict) -> None:
    data = CalibrationSettings.model_validate(data).model_dump()
    roi = data["roi"]
    config.ROI_FRACTION = (roi["x"], roi["y"], roi["w"], roi["h"])
    config.PRESENCE_ON_RATIO = data["presence_on_ratio"]
    config.PRESENCE_OFF_RATIO = data["presence_off_ratio"]


def save(data: dict) -> None:
    apply(data)
    db.set_setting(_SETTINGS_KEY, json.dumps(data))


def load_persisted() -> None:
    raw = db.get_setting(_SETTINGS_KEY)
    if raw:
        try:
            apply(json.loads(raw))
        except (ValueError, TypeError, KeyError) as exc:
            print(f"[calibration] invalid saved settings; using defaults: {exc}")


def defaults() -> dict:
    return copy.deepcopy(_DEFAULTS)


# Captured once at import time, before load_persisted()/apply() ever run, so
# it reflects the values shipped in config.py regardless of what gets tuned
# on-site later.
_DEFAULTS = current()
