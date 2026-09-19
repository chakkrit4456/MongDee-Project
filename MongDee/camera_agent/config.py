"""Camera Agent configuration. Reuses camera.base.CameraConfig for the
camera list (same format as configs/cameras.example.json) plus a small
block of agent-specific settings (server URL, credentials, push behaviour)
— see configs/camera_agent.example.json.

Credentials may be given as "${ENV_VAR}" to pull from the environment
instead of committing them (matching backend/config.py's convention).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

from camera.base import CameraConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env(value):
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    server_url: str = "http://127.0.0.1:8100"
    api_key: str = ""
    cameras: tuple[CameraConfig, ...] = ()

    # How often (seconds) to attempt registering/re-registering a camera
    # with the server before it has confirmed success, and after a push
    # failure (section 33/59: auto reconnect with backoff).
    register_retry_sec: float = 5.0
    retry_backoff_multiplier: float = 2.0
    retry_backoff_max_sec: float = 30.0
    push_timeout_sec: float = 5.0
    jpeg_quality: int = 80

    @staticmethod
    def load(path: str | Path) -> "AgentConfig":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read agent config {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
        data = _resolve_env(data)

        cameras = []
        seen = set()
        for i, entry in enumerate(data.get("cameras", [])):
            cfg = CameraConfig.from_dict(entry)
            if cfg.id in seen:
                raise ValueError(f"{path}: duplicate camera id {cfg.id!r}")
            seen.add(cfg.id)
            cameras.append(cfg)
        if not cameras:
            raise ValueError(f"{path}: at least one camera is required")

        known = {
            "server_url", "api_key", "register_retry_sec", "retry_backoff_multiplier",
            "retry_backoff_max_sec", "push_timeout_sec", "jpeg_quality",
        }
        agent_block = {k: v for k, v in data.items() if k in known}
        unknown = sorted(set(data) - known - {"cameras"})
        if unknown:
            raise ValueError(f"{path}: unknown field(s): {unknown}")

        return AgentConfig(cameras=tuple(cameras), **agent_block)
