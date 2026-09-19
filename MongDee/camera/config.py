"""Load camera configuration from JSON.

See configs/cameras.example.json for the file format and camera/README.md
for URL formats per protocol (Master Prompt section 33: "ทุกค่าที่สำคัญต้อง
สามารถกำหนดผ่าน config").
"""

from __future__ import annotations

import json
from pathlib import Path

from camera.base import CameraConfig


def load_camera_configs(path: str | Path) -> list[CameraConfig]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read camera config file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON: {exc}") from exc

    if isinstance(data, list):
        cameras = data
    elif isinstance(data, dict) and isinstance(data.get("cameras"), list):
        cameras = data["cameras"]
    else:
        raise ValueError(f"{path}: expected a top-level 'cameras' list (or a bare JSON list)")

    seen_ids: set[str] = set()
    configs: list[CameraConfig] = []
    for i, entry in enumerate(cameras):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: cameras[{i}] must be an object, got {type(entry).__name__}")
        try:
            cfg = CameraConfig.from_dict(entry)
        except ValueError as exc:
            raise ValueError(f"{path}: cameras[{i}]: {exc}") from exc
        if cfg.id in seen_ids:
            raise ValueError(f"{path}: duplicate camera id {cfg.id!r}")
        seen_ids.add(cfg.id)
        configs.append(cfg)
    return configs
