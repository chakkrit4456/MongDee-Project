"""Re-ID configuration — Phase 5 (MongDee_Master_Prompt.md sections 9, 23, 24)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ReIDConfig:
    backend: str = "auto"  # "auto" (torchvision if usable, else color) | "torchvision" | "color" | "osnet"
    torchvision_model: str = "resnet18"  # resnet18 | resnet34 | resnet50
    osnet_weights: str = ""  # path to an osnet_*.pth checkpoint (required for backend="osnet")
    osnet_variant: str = "osnet_x1_0"  # osnet_x1_0 | osnet_x0_25
    device: str = "auto"

    # crop resize before embedding (person aspect: taller than wide)
    input_width: int = 128
    input_height: int = 256

    # quality gate (section 24) — a crop failing this is not embedded
    min_height_px: int = 80
    min_blur_var: float = 15.0  # variance-of-Laplacian below this = too blurry
    min_aspect: float = 0.15  # w/h; a standing person's box
    max_aspect: float = 0.9

    # track-level aggregation (section 23)
    max_embeddings_per_track: int = 8

    @staticmethod
    def from_dict(d: dict) -> "ReIDConfig":
        known = {f.name for f in dataclasses.fields(ReIDConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"reid config has unknown field(s): {unknown}")
        return ReIDConfig(**d)
