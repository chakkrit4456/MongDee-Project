"""Standalone AI Dashboard & Analytics viewer (no cameras needed).

Useful for an organizer/central machine that just wants to watch analytics
across booths, importing exports from scripts/export_booth_data.py.

Usage:
    python dashboard.py --db data/mongdee.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from core import database as db
from ui.dashboard_window import DashboardWindow

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    db.init_db(db_path)

    app = QApplication(sys.argv)
    window = DashboardWindow(db_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
