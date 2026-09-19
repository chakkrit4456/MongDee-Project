"""Booth layout model (MongDee_Master_Prompt.md sections 51, 61-66, 70).

A booth is a rectangle (width x length, metres) containing objects placed
at real-world coordinates: cameras, shelves, product zones, tables, walls,
doors, entrances/exits, points of interest, no-entry zones, sales points.

Every layout change that matters creates a new version (section 66); a
historical event references the layout_version it happened under so its
meaning never shifts under it.
"""

from __future__ import annotations

import dataclasses
import time

OBJECT_TYPES = {
    "camera", "shelf", "product", "table", "wall", "door",
    "entrance", "exit", "product_zone", "point_of_interest",
    "no_entry_zone", "sales_point",
}


@dataclasses.dataclass
class BoothObject:
    id: str
    object_type: str
    x: float          # world X of the object's centre, metres
    y: float          # world Y of the object's centre, metres
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0  # degrees, CCW
    metadata: dict = dataclasses.field(default_factory=dict)
    active: bool = True

    def __post_init__(self):
        if self.object_type not in OBJECT_TYPES:
            raise ValueError(f"unknown booth object_type {self.object_type!r}; expected one of {sorted(OBJECT_TYPES)}")

    def contains(self, wx: float, wy: float) -> bool:
        """Axis-aligned test for un-rotated objects; rotation-aware for the rest."""
        if self.width <= 0 or self.height <= 0:
            return False
        dx, dy = wx - self.x, wy - self.y
        if self.rotation:
            rad = -self.rotation * 3.141592653589793 / 180.0
            cos_r = _cos(rad)
            sin_r = _sin(rad)
            dx, dy = dx * cos_r - dy * sin_r, dx * sin_r + dy * cos_r
        return abs(dx) <= self.width / 2 and abs(dy) <= self.height / 2

    def distance_to(self, wx: float, wy: float) -> float:
        """Distance from a world point to this object's edge (0 if inside)."""
        dx = max(abs(wx - self.x) - self.width / 2, 0.0)
        dy = max(abs(wy - self.y) - self.height / 2, 0.0)
        return (dx * dx + dy * dy) ** 0.5

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BoothObject":
        known = {f.name for f in dataclasses.fields(BoothObject)}
        return BoothObject(**{k: v for k, v in d.items() if k in known})


def _cos(x: float) -> float:
    import math
    return math.cos(x)


def _sin(x: float) -> float:
    import math
    return math.sin(x)


@dataclasses.dataclass
class BoothLayout:
    booth_id: str
    width: float          # metres (X extent)
    length: float         # metres (Y extent)
    version: int = 1
    unit: str = "m"
    objects: list[BoothObject] = dataclasses.field(default_factory=list)
    active: bool = True
    created_at: float = dataclasses.field(default_factory=time.time)

    def add(self, obj: BoothObject) -> None:
        if any(o.id == obj.id for o in self.objects):
            raise ValueError(f"booth object id {obj.id!r} already in layout")
        self.objects.append(obj)

    def get(self, object_id: str) -> BoothObject | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def of_type(self, object_type: str) -> list[BoothObject]:
        return [o for o in self.objects if o.object_type == object_type and o.active]

    def zone_at(self, wx: float, wy: float) -> BoothObject | None:
        """The product_zone (or point_of_interest) containing a world point."""
        for o in self.objects:
            if o.active and o.object_type in ("product_zone", "point_of_interest", "no_entry_zone") and o.contains(wx, wy):
                return o
        return None

    def in_bounds(self, wx: float, wy: float, margin: float = 0.5) -> bool:
        return -margin <= wx <= self.width + margin and -margin <= wy <= self.length + margin

    def bumped_version(self) -> "BoothLayout":
        """A copy with version+1 and a fresh timestamp — call after an edit
        that needs recalibration/history (section 66, 73)."""
        return dataclasses.replace(
            self, version=self.version + 1, created_at=time.time(),
            objects=[dataclasses.replace(o) for o in self.objects],
        )

    def to_dict(self) -> dict:
        return {
            "booth_id": self.booth_id, "width": self.width, "length": self.length,
            "version": self.version, "unit": self.unit, "active": self.active,
            "created_at": self.created_at, "objects": [o.to_dict() for o in self.objects],
        }

    @staticmethod
    def from_dict(d: dict) -> "BoothLayout":
        return BoothLayout(
            booth_id=d["booth_id"], width=d["width"], length=d["length"],
            version=d.get("version", 1), unit=d.get("unit", "m"), active=d.get("active", True),
            created_at=d.get("created_at", time.time()),
            objects=[BoothObject.from_dict(o) for o in d.get("objects", [])],
        )
