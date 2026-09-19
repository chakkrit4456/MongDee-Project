"""Removes the admin PIN (for a forgotten code). Run on the kiosk PC itself:

    venv\\Scripts\\python.exe scripts\\reset_admin_pin.py

The next delete then asks you to choose a new PIN.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import admin_auth, db  # noqa: E402

db.init_db()
admin_auth.clear_pin()
print("admin PIN removed - the next delete will ask for a new one")
