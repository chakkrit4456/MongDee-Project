"""product_lifecycle - helpers for keeping the product gallery in sync with the catalog.

Why this exists (root cause of "products I deleted are still detected"):
  * data/gallery/manifest.json + <key>.npy kept a gallery (product-4621553d, 30 samples) for a
    product that is no longer in products.json; ProductRecognizer loaded every manifest key, so
    the ghost kept matching whatever was in front of the camera (or the empty room).
  * an import thread still running when a product was deleted could call add_sample() and
    re-create the gallery.
The runtime guard lives in core.recognizer.ProductRecognizer (set_active_provider / prune_inactive /
add_sample -> ProductDeleted). This module holds the exception and the offline audit used by
tools/audit_gallery.py. Pure stdlib.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Optional, Set

_ID_FIELDS = ("id", "key", "product_id", "slug", "product_key")


class ProductDeleted(Exception):
    """The product no longer exists in the catalog (raised by ProductRecognizer.add_sample)."""


def load_product_keys(products_json: str) -> Set[str]:
    """Tolerant loader. MongDee's products.json is {"_comment": ..., "<key>": {...}, ...};
    also accepts list[dict] and {"products": [...]}. Keys starting with "_" are metadata."""
    if not os.path.exists(products_json):
        return set()
    with open(products_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("products"), (list, dict)):
        data = data["products"]
    keys: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for fld in _ID_FIELDS:
                    if item.get(fld):
                        keys.add(str(item[fld])); break
            elif isinstance(item, str):
                keys.add(item)
    elif isinstance(data, dict):
        keys = {str(k) for k in data.keys() if not str(k).startswith("_")}
    return keys


def _npy_name(key: str) -> str:
    return key.replace("/", "_") + ".npy"          # mirrors ProductRecognizer._gallery_path


def reconcile(products_json: str, gallery_dir: str, apply: bool = False,
              quarantine_root: Optional[str] = None) -> Dict[str, List[str]]:
    """Compare the catalog with the gallery on disk.

    orphans        - gallery entries (manifest key and/or .npy file) with no catalog product
    no_gallery     - catalog products with no samples (cannot be recognised until trained)
    apply=True     - orphans are removed from manifest.json and their .npy MOVED to
                     <gallery_dir>/_quarantine/<timestamp>/ (never deleted, reversible).
    """
    manifest_path = os.path.join(gallery_dir, "manifest.json")
    catalog = load_product_keys(products_json)
    manifest: Dict[str, dict] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    npy_files = {n for n in os.listdir(gallery_dir)} if os.path.isdir(gallery_dir) else set()
    stray_npy = {n for n in npy_files if n.endswith(".npy") and not n.startswith("_")}
    catalog_npy = {_npy_name(k): k for k in catalog}
    orphan_keys = {k for k in manifest if k not in catalog}
    orphan_files = {n for n in stray_npy if n not in catalog_npy}
    orphans = sorted(orphan_keys | {n[:-4] for n in orphan_files})
    no_gallery = sorted(k for k in catalog if k not in manifest and _npy_name(k) not in npy_files)
    moved: List[str] = []
    if apply and orphans:
        qdir = os.path.join(quarantine_root or os.path.join(gallery_dir, "_quarantine"), time.strftime("%Y%m%d-%H%M%S"))
        for key in orphan_keys:
            manifest.pop(key, None)
        for n in sorted(orphan_files | {_npy_name(k) for k in orphan_keys}):
            src = os.path.join(gallery_dir, n)
            if os.path.exists(src):
                os.makedirs(qdir, exist_ok=True)
                dst = os.path.join(qdir, n)
                shutil.move(src, dst)
                moved.append(dst)
        tmp = f"{manifest_path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp, manifest_path)
    return {"orphans": orphans, "no_gallery": no_gallery, "quarantined": moved}
