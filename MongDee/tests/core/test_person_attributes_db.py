"""Unit tests for core/database.py's person_attributes persistence + the
attribute_snapshot_json column on tripwire_crossings — spec:
MongDee_Master_Prompt_FairFace_Age_Gender.md sections 13/14/17.

Always against an isolated tmp_path DB — see core.database's own
production-DB guard (test_database_guard.py).
"""

from __future__ import annotations

import json

from core import database as db


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    return db_path


def test_upsert_person_attributes_insert_then_update(tmp_path):
    db_path = _make_db(tmp_path)
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P000001",
                                 "MALE", 0.9, "20-29", "ADULT", 0.8, 2, 100.0, 101.0)
    row = db.get_tripwire(db_path, "BOOTH-1", "CAM-1")  # sanity: unrelated table untouched
    assert row is None

    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 1
    assert breakdown["age_category"]["ADULT"] == 1

    # Refine the same global_id — must update in place, not insert a second row.
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P000001",
                                 "MALE", 0.95, "20-29", "ADULT", 0.85, 5, 100.0, 110.0)
    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 1  # still one distinct Global Person, not two


def test_attribute_breakdown_separates_unknown_bucket(tmp_path):
    db_path = _make_db(tmp_path)
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P1",
                                 "UNKNOWN", 0.0, "UNKNOWN", "UNKNOWN", 0.0, 1, 100.0, 100.0)
    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["UNKNOWN"] == 1
    assert breakdown["age_category"]["UNKNOWN"] == 1
    assert breakdown["gender"]["MALE"] == 0


def test_attribute_breakdown_scoped_by_booth(tmp_path):
    db_path = _make_db(tmp_path)
    db.upsert_person_attributes(db_path, "BOOTH-A", "EVT-1", "P1",
                                 "MALE", 0.9, "20-29", "ADULT", 0.8, 1, 100.0, 100.0)
    db.upsert_person_attributes(db_path, "BOOTH-B", "EVT-1", "P1",
                                 "FEMALE", 0.9, "20-29", "ADULT", 0.8, 1, 100.0, 100.0)
    breakdown_a = db.query_attribute_breakdown(db_path, booth_id="BOOTH-A")
    assert breakdown_a["gender"]["MALE"] == 1
    assert breakdown_a["gender"]["FEMALE"] == 0


def test_person_attributes_wiped_by_delete_scope_data(tmp_path):
    db_path = _make_db(tmp_path)
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P1",
                                 "MALE", 0.9, "20-29", "ADULT", 0.8, 1, 100.0, 100.0)
    db.delete_scope_data(db_path, booth_id="BOOTH-1")
    breakdown = db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")
    assert breakdown["gender"]["MALE"] == 0


def test_tripwire_crossing_attribute_snapshot_round_trips(tmp_path):
    db_path = _make_db(tmp_path)
    db.create_booth(db_path, "BOOTH-1", "B", "EVT-1")
    snapshot = {"global_person_id": "P103", "gender": "MALE", "age_group": "20-29",
                "age_category": "ADULT", "gender_confidence": 0.91, "age_confidence": 0.76}
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-1", 7, "in",
                              ts=100.0, attribute_snapshot=snapshot)
    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(crossings) == 1
    stored = json.loads(crossings[0]["attribute_snapshot_json"])
    assert stored == snapshot


def test_tripwire_crossing_without_snapshot_has_null_column(tmp_path):
    db_path = _make_db(tmp_path)
    db.create_booth(db_path, "BOOTH-1", "B", "EVT-1")
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-1", 7, "in", ts=100.0)
    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert crossings[0]["attribute_snapshot_json"] is None


def test_one_crossing_stays_one_row_with_or_without_snapshot(tmp_path):
    """Spec section 13: attaching an attribute snapshot must never create a
    duplicate crossing row — re-delivering the identical event (same
    camera/track/tripwire/direction/ts) with a snapshot attached the second
    time still dedupes via the existing event_key idempotency."""
    db_path = _make_db(tmp_path)
    db.create_booth(db_path, "BOOTH-1", "B", "EVT-1")
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-1", 7, "in", ts=100.0)
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-1", 7, "in", ts=100.0,
                              attribute_snapshot={"gender": "MALE"})
    crossings = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(crossings) == 1
