"""Person Attribute Extraction — Phase 4 (MongDee_Master_Prompt.md
section 7).

What this actually produces from pixels alone, with a confidence on each:
    shirt_color, pants_color   (named dominant colour of the upper / lower body region)
    dominant_colors            (overall, for the multi-feature matcher in Phase 6)
    aspect_ratio, build        (rough: "slim" / "average" / "broad" from bbox w/h)

Everything else the master prompt lists (hair length/colour, glasses,
mask, bag type, gender, age, ...) needs a trained model. Rather than fake
those, the extractor returns them as `unknown` (confidence 0) unless a
`model_backend` that can actually produce them is plugged in. Sections 17
(gender) and 18 (age) have their own optional classifier hooks for the
same reason.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision.attributes.color import dominant_color
from vision.attributes.config import AttributeConfig


@dataclasses.dataclass
class AttributeValue:
    value: str
    confidence: float

    @staticmethod
    def unknown() -> "AttributeValue":
        return AttributeValue("unknown", 0.0)

    @property
    def is_known(self) -> bool:
        return self.value != "unknown" and self.confidence > 0.0


@dataclasses.dataclass
class PersonAttributes:
    shirt_color: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    pants_color: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    build: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    gender: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    age_group: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    hair: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    bag: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    glasses: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)
    mask: AttributeValue = dataclasses.field(default_factory=AttributeValue.unknown)

    # numeric appearance features for the matcher (not user-facing)
    shirt_rgb: tuple[int, int, int] = (0, 0, 0)
    pants_rgb: tuple[int, int, int] = (0, 0, 0)
    aspect_ratio: float = 0.0

    def to_dict(self) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = {"value": v.value, "confidence": round(v.confidence, 3)} if isinstance(v, AttributeValue) else v
        return out

    def known_items(self) -> dict[str, AttributeValue]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), AttributeValue) and getattr(self, f.name).is_known
        }


class GenderAgeBackend:
    """Interface for an optional gender/age classifier. A real one returns
    (label, confidence); the label for gender must include 'unknown' as a
    possibility (section 17), and age must be a group not a number
    (section 18)."""

    def predict_gender(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError

    def predict_age_group(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError


class AttributeExtractor:
    def __init__(self, config: AttributeConfig | None = None, gender_age_backend: GenderAgeBackend | None = None):
        self.config = config or AttributeConfig()
        self._gender_age = gender_age_backend

    def extract(self, image: np.ndarray, bbox: list[float]) -> PersonAttributes:
        cfg = self.config
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        attrs = PersonAttributes()
        if crop.size == 0 or crop.shape[0] < cfg.min_crop_height_px:
            return attrs  # too small — everything stays unknown, honestly

        ch, cw = crop.shape[:2]
        attrs.aspect_ratio = round(cw / ch, 3) if ch else 0.0
        attrs.build = _build_from_aspect(attrs.aspect_ratio)

        inset = int(cw * cfg.horizontal_inset)
        x_lo, x_hi = inset, max(inset + 1, cw - inset)

        up = crop[int(ch * cfg.upper_body[0]) : int(ch * cfg.upper_body[1]), x_lo:x_hi]
        lo = crop[int(ch * cfg.lower_body[0]) : int(ch * cfg.lower_body[1]), x_lo:x_hi]

        attrs.shirt_color, attrs.shirt_rgb = self._region_color(up)
        attrs.pants_color, attrs.pants_rgb = self._region_color(lo)

        if self._gender_age is not None:
            g_label, g_conf = self._gender_age.predict_gender(crop)
            attrs.gender = AttributeValue(g_label, g_conf)
            a_label, a_conf = self._gender_age.predict_age_group(crop)
            attrs.age_group = AttributeValue(a_label, a_conf)

        return attrs

    def _region_color(self, region: np.ndarray) -> tuple[AttributeValue, tuple[int, int, int]]:
        if region.size == 0:
            return AttributeValue.unknown(), (0, 0, 0)
        name, rgb, conf = dominant_color(region)
        if conf < self.config.min_color_confidence:
            return AttributeValue("unknown", round(conf, 3)), rgb
        return AttributeValue(name, round(conf, 3)), rgb


def _build_from_aspect(aspect: float) -> AttributeValue:
    if aspect <= 0:
        return AttributeValue.unknown()
    # width/height of a standing person's bbox: slimmer people have a smaller ratio.
    # This is a very rough cue and reported with low confidence on purpose.
    if aspect < 0.33:
        return AttributeValue("slim", 0.4)
    if aspect < 0.5:
        return AttributeValue("average", 0.4)
    return AttributeValue("broad", 0.4)
