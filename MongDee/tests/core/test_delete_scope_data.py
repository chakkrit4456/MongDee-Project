"""Unit tests for core/database.py's delete_scope_data() — backs the
Dashboard's single "ล้างข้อมูล" (clear data) button, which must support
clearing by booth only, event only, one day only, or any combination, while
never touching registry/config tables (booths/events/tripwires) and never
wiping everything by accident on an empty/default form submission.

Always against an isolated tmp_path DB — never data/mongdee.db (see
core.database's own production-DB guard, exercised separately in
test_database_guard.py).
"""

from __future__ import annotations

import time

import pytest

from core import database as db


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    return db_path


def test_no_scope_and_no_confirm_all_raises(tmp_path):
    db_path = _make_db(tmp_path)
    with pytest.raises(ValueError):
        db.delete_scope_data(db_path)


def test_booth_only_scope_leaves_other_booths_and_config_intact(tmp_path):
    db_path = _make_db(tmp_path)
    db.create_booth(db_path, "BOOTH-A", "A", None)
    db.create_booth(db_path, "BOOTH-B", "B", None)
    db.upsert_tripwire(db_path, "TW-A", "BOOTH-A", "CAM-1", 0.5, 0.0, 0.5, 1.0, "A", True)
    db.log_interaction(db_path, "BOOTH-A", "EVT-1", "CAM-1", "product-1", "Widget", 0.9)
    db.log_interaction(db_path, "BOOTH-B", "EVT-1", "CAM-1", "product-1", "Widget", 0.9)

    db.delete_scope_data(db_path, booth_id="BOOTH-A")

    assert db.query_interactions(db_path, booth_id="BOOTH-A") == []
    assert len(db.query_interactions(db_path, booth_id="BOOTH-B")) == 1
    # Registry/config tables are never touched by a log-data wipe.
    assert db.get_booth(db_path, "BOOTH-A") is not None
    assert db.get_tripwire(db_path, "BOOTH-A", "CAM-1") is not None


def test_event_only_scope(tmp_path):
    db_path = _make_db(tmp_path)
    db.log_interaction(db_path, "BOOTH-1", "EVT-A", "CAM-1", "product-1", "Widget", 0.9)
    db.log_interaction(db_path, "BOOTH-1", "EVT-B", "CAM-1", "product-1", "Widget", 0.9)

    db.delete_scope_data(db_path, event_id="EVT-A")

    assert db.query_interactions(db_path, event_id="EVT-A") == []
    assert len(db.query_interactions(db_path, event_id="EVT-B")) == 1


def test_date_only_scope(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    today = time.mktime(time.strptime("2026-09-10", "%Y-%m-%d")) + 3600  # noon-ish
    yesterday = today - 86400

    monkeypatch.setattr(time, "time", lambda: today)
    db.log_interaction(db_path, "BOOTH-1", "EVT-1", "CAM-1", "product-1", "Today", 0.9)
    monkeypatch.setattr(time, "time", lambda: yesterday)
    db.log_interaction(db_path, "BOOTH-1", "EVT-1", "CAM-1", "product-1", "Yesterday", 0.9)
    monkeypatch.undo()

    db.delete_scope_data(db_path, date="2026-09-10")

    remaining = db.query_interactions(db_path)
    assert len(remaining) == 1
    assert remaining[0]["product_name"] == "Yesterday"


def test_confirm_all_wipes_every_scope(tmp_path):
    db_path = _make_db(tmp_path)
    db.log_interaction(db_path, "BOOTH-A", "EVT-1", "CAM-1", "product-1", "Widget", 0.9)
    db.log_interaction(db_path, "BOOTH-B", "EVT-2", "CAM-1", "product-1", "Widget", 0.9)

    db.delete_scope_data(db_path, confirm_all=True)

    assert db.query_interactions(db_path) == []


def test_combined_booth_and_date_scope(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    day = time.mktime(time.strptime("2026-09-10", "%Y-%m-%d")) + 3600
    monkeypatch.setattr(time, "time", lambda: day)
    db.log_interaction(db_path, "BOOTH-A", "EVT-1", "CAM-1", "product-1", "A-that-day", 0.9)
    db.log_interaction(db_path, "BOOTH-B", "EVT-1", "CAM-1", "product-1", "B-that-day", 0.9)
    monkeypatch.undo()

    db.delete_scope_data(db_path, booth_id="BOOTH-A", date="2026-09-10")

    remaining = db.query_interactions(db_path)
    assert len(remaining) == 1
    assert remaining[0]["product_name"] == "B-that-day"
