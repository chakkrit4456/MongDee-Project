"""Product vision config (MongDee_Master_Prompt.md sections 48, 50)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ProductConfig:
    # detection: which YOLO/COCO classes to treat as candidate products
    # (booth GI goods rarely match COCO — a trained model or the class-agnostic
    # foreground path from core/localizer.py is better; this is the cheap default)
    yolo_classes: tuple[int, ...] = (39, 40, 41, 45, 46, 47, 63, 67, 73)  # bottle, wine glass, cup, bowl, banana, apple, laptop, cell phone, book
    detection_confidence: float = 0.35
    processing_width: int = 960
    processing_height: int = 540

    # classification: nearest-embedding match against the reference gallery
    reid_backend: str = "color"  # "color" | "torchvision" | "osnet"
    known_similarity: float = 0.75      # >= this to a gallery item -> KNOWN PRODUCT
    possible_similarity: float = 0.55   # [possible, known) -> POSSIBLE PRODUCT; below -> UNKNOWN
    top2_margin: float = 0.05           # winner must beat runner-up by this, else POSSIBLE

    # tracking: products barely move; a light IoU-persistence tracker
    track_match_iou: float = 0.4
    track_max_age_sec: float = 3.0
    track_min_hits: int = 2

    target_fps_per_camera: float = 2.0  # products are ~static; run this rarely

    @staticmethod
    def from_dict(d: dict) -> "ProductConfig":
        d = dict(d)
        if "yolo_classes" in d and isinstance(d["yolo_classes"], list):
            d["yolo_classes"] = tuple(d["yolo_classes"])
        known = {f.name for f in dataclasses.fields(ProductConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"product config has unknown field(s): {unknown}")
        return ProductConfig(**d)
