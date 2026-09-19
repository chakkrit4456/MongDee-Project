"""GI Product database (MongDee_Master_Prompt.md section 49).

Holds the reference information the system attaches to a product once the
vision layer has identified it — no QR needed as the primary path
(section 49: "โดยไม่ต้องพึ่ง QR เป็นวิธีหลัก").

Loadable from one JSON file. `class_name` links a record to whatever label
the detector/classifier produces.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class GIProduct:
    id: str
    name: str
    class_name: str = ""
    gi_reference: str = ""
    province: str = ""
    production_area: str = ""
    producer: str = ""
    community: str = ""
    category: str = ""
    description: str = ""
    ingredients: str = ""
    production_process: str = ""
    image_url: str = ""
    selling_info: str = ""
    additional_info: str = ""
    active: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GIProduct":
        known = {f.name for f in dataclasses.fields(GIProduct)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"GI product {d.get('id')!r} has unknown field(s): {unknown}")
        return GIProduct(**d)


class GIProductDatabase:
    def __init__(self, products: list[GIProduct] | None = None):
        self._by_id: dict[str, GIProduct] = {}
        self._by_class: dict[str, str] = {}  # class_name -> product_id
        for p in products or []:
            self.add(p)

    def add(self, product: GIProduct) -> None:
        if product.id in self._by_id:
            raise ValueError(f"duplicate GI product id {product.id!r}")
        self._by_id[product.id] = product
        if product.class_name:
            self._by_class[product.class_name] = product.id

    def get(self, product_id: str) -> GIProduct | None:
        return self._by_id.get(product_id)

    def by_class_name(self, class_name: str) -> GIProduct | None:
        pid = self._by_class.get(class_name)
        return self._by_id.get(pid) if pid else None

    def all(self) -> list[GIProduct]:
        return [p for p in self._by_id.values() if p.active]

    def ids(self) -> list[str]:
        return list(self._by_id)

    @staticmethod
    def load(path: str | Path) -> "GIProductDatabase":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read GI product file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
        items = data.get("products", data if isinstance(data, list) else None)
        if items is None:
            raise ValueError(f"{path}: expected a 'products' list")
        db = GIProductDatabase()
        for i, entry in enumerate(items):
            try:
                db.add(GIProduct.from_dict(entry))
            except ValueError as exc:
                raise ValueError(f"{path}: products[{i}]: {exc}") from exc
        return db
