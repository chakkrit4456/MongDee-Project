"""Attribute extraction config — Phase 4."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AttributeConfig:
    # vertical slices of the person bbox (fraction of height, top=0)
    upper_body: tuple[float, float] = (0.15, 0.50)  # shirt region
    lower_body: tuple[float, float] = (0.50, 0.90)  # trousers region
    # horizontal inset to avoid background at the edges of the bbox
    horizontal_inset: float = 0.15

    min_color_confidence: float = 0.25  # below this, report the colour as "unknown"
    min_crop_height_px: int = 60  # crops shorter than this are too small to read attributes from

    @staticmethod
    def from_dict(d: dict) -> "AttributeConfig":
        d = dict(d)
        for key in ("upper_body", "lower_body"):
            if key in d and isinstance(d[key], list):
                d[key] = tuple(d[key])
        known = {f.name for f in dataclasses.fields(AttributeConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"attribute config has unknown field(s): {unknown}")
        return AttributeConfig(**d)
