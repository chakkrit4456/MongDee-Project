"""Unit tests for core/database.py's open_product_interest_event /
finalize_product_interest_event — spec: MongDee person-gender/age master
continuation prompt sections 20/23/26/35/37 (confirm-at-2s, finalize-at-
release, snapshot-once attribution, backward compatibility with legacy
insert-at-release rows).
"""

from __future__ import annotations

from core import database as db


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    return db_path


def test_open_product_interest_event_writes_confirmed_status(tmp_path):
    db_path = _make_db(tmp_path)
    row_id = db.open_product_interest_event(
        db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
        hold_start_ts=100.0, confirmed_at=102.0,
        global_person_id="P000001", gender="FEMALE", gender_confidence=0.9,
        age_category="ADULT", age_confidence=0.8,
    )
    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == row_id
    assert row["status"] == "confirmed"
    assert row["confirmed_at"] == 102.0
    assert row["hold_start_ts"] == 100.0
    # provisional until finalized — see the function's own docstring
    assert row["hold_end_ts"] == 102.0
    assert row["duration_sec"] == 2.0
    assert row["global_person_id"] == "P000001"
    assert row["gender"] == "FEMALE"
    assert row["age_category"] == "ADULT"


def test_finalize_overwrites_only_end_of_interaction_fields(tmp_path):
    db_path = _make_db(tmp_path)
    row_id = db.open_product_interest_event(
        db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
        hold_start_ts=100.0, confirmed_at=102.0,
        global_person_id="P000001", gender="FEMALE", gender_confidence=0.9,
        age_category="ADULT", age_confidence=0.8,
    )
    db.finalize_product_interest_event(db_path, row_id, hold_end_ts=110.0, duration_sec=10.0)

    row = db.query_product_hold_events(db_path, booth_id="BOOTH-1")[0]
    assert row["status"] == "finalized"
    assert row["hold_end_ts"] == 110.0
    assert row["duration_sec"] == 10.0
    # attribution snapshot from confirmation time is untouched
    assert row["gender"] == "FEMALE"
    assert row["global_person_id"] == "P000001"
    assert row["hold_start_ts"] == 100.0


def test_finalize_on_unknown_row_id_is_a_no_op(tmp_path):
    db_path = _make_db(tmp_path)
    db.finalize_product_interest_event(db_path, 999999, hold_end_ts=10.0, duration_sec=1.0)
    assert db.query_product_hold_events(db_path, booth_id="BOOTH-1") == []


def test_legacy_insert_at_release_rows_default_to_finalized_status(tmp_path):
    """Spec section 72/73: old rows (inserted via log_product_hold_event,
    the original insert-at-release path) never had a status column — the
    DEFAULT must read as 'finalized' for them since they always carried a
    real, already-known hold_end_ts/duration_sec."""
    db_path = _make_db(tmp_path)
    db.log_product_hold_event(db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
                               100.0, 105.0, 5.0)
    row = db.query_product_hold_events(db_path, booth_id="BOOTH-1")[0]
    assert row["status"] == "finalized"
    assert row["confirmed_at"] is None
    assert row["global_person_id"] is None  # honest UNKNOWN, no crash


def test_analytics_count_open_and_finalized_rows_together(tmp_path):
    """A still-open (confirmed, not yet finalized) interaction is a real
    confirmed interest and must already be counted, not undercounted while
    it's still in progress."""
    db_path = _make_db(tmp_path)
    db.open_product_interest_event(db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
                                    100.0, 102.0, global_person_id="P1", gender="MALE")
    db.log_product_hold_event(db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 8,
                               200.0, 205.0, 5.0, global_person_id="P2", gender="FEMALE")

    overview = db.query_interest_overview(db_path, booth_id="BOOTH-1")
    assert overview == {"total_interest_events": 2, "unique_interested_persons": 2}
    totals = db.query_gender_interest_totals(db_path, booth_id="BOOTH-1")
    assert totals == {"MALE": 1, "FEMALE": 1, "UNKNOWN": 0}


def test_camera_filter_on_interest_analytics(tmp_path):
    db_path = _make_db(tmp_path)
    db.log_product_hold_event(db_path, "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
                               100.0, 105.0, 5.0, gender="MALE")
    db.log_product_hold_event(db_path, "BOOTH-1", "EVT-1", "CAM-2", "WATER", "น้ำดื่ม", 8,
                               100.0, 105.0, 5.0, gender="FEMALE")

    assert db.query_gender_interest_totals(db_path, booth_id="BOOTH-1", camera_id="CAM-1") == \
        {"MALE": 1, "FEMALE": 0, "UNKNOWN": 0}
    assert db.query_gender_interest_totals(db_path, booth_id="BOOTH-1", camera_id="CAM-2") == \
        {"MALE": 0, "FEMALE": 1, "UNKNOWN": 0}
    assert db.query_gender_interest_totals(db_path, booth_id="BOOTH-1") == \
        {"MALE": 1, "FEMALE": 1, "UNKNOWN": 0}
