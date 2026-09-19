#!/usr/bin/env python
"""Find/fix ghost products. Dry-run by default.
    python tools/audit_gallery.py                 # report only
    python tools/audit_gallery.py --apply         # quarantine orphans (moved, never deleted)
Uses ./products.json (the catalog) and <data>/gallery (manifest.json + <key>.npy)."""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.product_lifecycle import reconcile

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--products", default="products.json"); ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
r = reconcile(a.products, os.path.join(a.data, "gallery"), apply=a.apply)
print(json.dumps(r, indent=2, ensure_ascii=False))
if r["orphans"] and not a.apply:
    print("\nGhost gallery keys found. Re-run with --apply to quarantine them (reversible).")
