"""Export one booth's local SQLite data to JSON for central aggregation.

Run this on a booth machine, then load the resulting file into the
Dashboard (📂 นำเข้าข้อมูลบูธอื่น) on the central/organizer machine to combine
data from many booths without needing a live network connection between them.

Usage:
    python scripts/export_booth_data.py --db data/mongdee.db --out booth_a_export.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database as db  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"ไม่พบไฟล์ฐานข้อมูล: {db_path}")

    payload = {
        "exported_at": time.time(),
        "source_db": str(db_path),
        "interactions": db.query_interactions(db_path, limit=1_000_000),
        "health_events": db.query_health_events(db_path, limit=1_000_000),
    }

    out_path = Path(args.out) if args.out else db_path.with_suffix(".export.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"ส่งออกข้อมูลสำเร็จ: {out_path} "
          f"({len(payload['interactions'])} interactions, {len(payload['health_events'])} health events)")


if __name__ == "__main__":
    main()
