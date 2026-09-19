"""Top-level MongDee configuration (MongDee_Master_Prompt.md section 33).

One JSON file holds every sub-config. Sections:

    cameras     list, see camera/config.py
    detection   see vision/config.py DetectionConfig
    tracking    see vision/tracking/config.py TrackingConfig
    reid        see vision/reid/config.py ReIDConfig
    attributes  see vision/attributes/config.py AttributeConfig
    identity    see vision/identity/config.py IdentityConfig
    topology    see vision/identity/topology.py CameraTopology
    features    see vision/track_features.py FeatureStoreConfig
    database    {"path": "...", "log_detections": false}
    api         {"host": "0.0.0.0", "port": 8100, "api_key": "..."}

Anything omitted falls back to that component's own defaults. Credentials
(camera passwords, api_key) may be given as "${ENV_VAR}" to pull from the
environment instead of committing them (section 34).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

from camera.base import CameraConfig
from vision.attributes.config import AttributeConfig
from vision.config import DetectionConfig
from vision.identity.config import IdentityConfig
from vision.identity.topology import CameraTopology
from vision.interest.config import InterestConfig
from vision.product.config import ProductConfig
from vision.reid.config import ReIDConfig
from vision.track_features import FeatureStoreConfig
from vision.tracking.config import TrackingConfig

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env(value):
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


@dataclasses.dataclass
class DatabaseConfig:
    path: str = ""  # "" -> backend.database.db.DEFAULT_DB_PATH
    log_detections: bool = False
    detection_sample_interval_sec: float = 1.0


@dataclasses.dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    api_key: str = ""  # single admin key (back-compat shortcut)
    # role-based keys (section 74): {"<key>": "admin|operator|viewer"}
    keys: dict = dataclasses.field(default_factory=dict)
    # Origins allowed to call this API cross-origin (e.g. the Vercel
    # dashboard's URL) — see MongDee_Cloud_Vercel_Remote_AI_Server_Master_
    # Prompt.md sections 27-28. Empty means "allow every origin", which is
    # fine for local development but not a public deployment.
    cors_origins: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SpatialConfig:
    enabled: bool = False
    booth_id: str = "BOOTH-01"
    layout_path: str = ""       # BoothLayout JSON (vision/spatial/booth.py)
    calibration_path: str = ""  # CalibrationStore JSON (vision/spatial/calibration.py)


@dataclasses.dataclass
class ProductPipelineConfig:
    enabled: bool = False
    gi_database_path: str = ""  # GI product JSON (vision/product/database.py)
    gallery_dir: str = ""       # <gallery_dir>/<product_id>/*.jpg reference crops


@dataclasses.dataclass
class AdaptiveConfig:
    """Adaptive Performance Engine (backend/adaptive.py) — see
    MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md section 34.
    Every threshold here is meant to be tuned per-deployment, not
    hard-coded (section 93 rule 10)."""
    enabled: bool = True
    tick_interval_sec: float = 2.0
    cpu_high_pct: float = 85.0
    latency_high_ms: float = 300.0
    min_target_fps: float = 1.0
    max_target_fps: float = 0.0  # 0 -> use detection.target_fps_per_camera as the ceiling
    imgsz_levels: list = dataclasses.field(default_factory=lambda: [640, 480, 320])


@dataclasses.dataclass
class MongDeeConfig:
    cameras: list[CameraConfig] = dataclasses.field(default_factory=list)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    reid: ReIDConfig = dataclasses.field(default_factory=ReIDConfig)
    attributes: AttributeConfig = dataclasses.field(default_factory=AttributeConfig)
    identity: IdentityConfig = dataclasses.field(default_factory=IdentityConfig)
    topology: CameraTopology = dataclasses.field(default_factory=CameraTopology)
    features: FeatureStoreConfig = dataclasses.field(default_factory=FeatureStoreConfig)
    interest: InterestConfig = dataclasses.field(default_factory=InterestConfig)
    product: ProductConfig = dataclasses.field(default_factory=ProductConfig)
    database: DatabaseConfig = dataclasses.field(default_factory=DatabaseConfig)
    api: ApiConfig = dataclasses.field(default_factory=ApiConfig)
    spatial: SpatialConfig = dataclasses.field(default_factory=SpatialConfig)
    product_pipeline: ProductPipelineConfig = dataclasses.field(default_factory=ProductPipelineConfig)
    adaptive: AdaptiveConfig = dataclasses.field(default_factory=AdaptiveConfig)

    @staticmethod
    def load(path: str | Path) -> "MongDeeConfig":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read config {path}: {exc}") from exc
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

        def sub(key, cls):
            block = data.get(key)
            return cls.from_dict(block) if isinstance(block, dict) else cls()

        return MongDeeConfig(
            cameras=cameras,
            detection=sub("detection", DetectionConfig),
            tracking=sub("tracking", TrackingConfig),
            reid=sub("reid", ReIDConfig),
            attributes=sub("attributes", AttributeConfig),
            identity=sub("identity", IdentityConfig),
            features=sub("features", FeatureStoreConfig),
            interest=sub("interest", InterestConfig),
            product=sub("product", ProductConfig),
            topology=CameraTopology.from_dict(data["topology"]) if isinstance(data.get("topology"), dict) else CameraTopology(),
            database=_dc_from_dict(DatabaseConfig, data.get("database")),
            api=_dc_from_dict(ApiConfig, data.get("api")),
            spatial=_dc_from_dict(SpatialConfig, data.get("spatial")),
            product_pipeline=_dc_from_dict(ProductPipelineConfig, data.get("product_pipeline")),
            adaptive=_dc_from_dict(AdaptiveConfig, data.get("adaptive")),
        )


def _dc_from_dict(cls, block):
    if not isinstance(block, dict):
        return cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown field(s): {unknown}")
    return cls(**block)
