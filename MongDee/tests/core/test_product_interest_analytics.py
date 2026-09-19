"""Unit tests for core/database.py's product-interest gender/age analytics —
spec: MongDee person-gender/age master prompt sections 22/26-45.

Always against an isolated tmp_path DB — see core.database's own
production-DB guard (test_database_guard.py).
"""

from __future__ import annotations

from core import database as db


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    return db_path


def _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", holder_track_id=1,
          global_person_id=None, gender=None, gender_confidence=None,
          age_category=None, age_confidence=None, duration_sec=2.5, camera_id="CAM-1"):
    db.log_product_hold_event(
        db_path, "BOOTH-1", "EVT-1", camera_id, product_key, product_name, holder_track_id,
        100.0, 100.0 + duration_sec, duration_sec,
        global_person_id=global_person_id, gender=gender, gender_confidence=gender_confidence,
        age_category=age_category, age_confidence=age_confidence,
    )


def test_log_product_hold_event_stores_attribution(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, global_person_id="P000001", gender="FEMALE", gender_confidence=0.9,
          age_category="ADULT", age_confidence=0.8)
    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert len(rows) == 1
    assert rows[0]["global_person_id"] == "P000001"
    assert rows[0]["gender"] == "FEMALE"
    assert rows[0]["age_category"] == "ADULT"


def test_log_product_hold_event_without_attribution_is_null(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path)
    rows = db.query_product_hold_events(db_path, booth_id="BOOTH-1")
    assert rows[0]["global_person_id"] is None
    assert rows[0]["gender"] is None
    assert rows[0]["age_category"] is None


def test_product_movers_reports_unique_persons_separately_from_mover_count(tmp_path):
    db_path = _make_db(tmp_path)
    # same Global Person holds WATER twice -> mover_count=2, unique_persons=1
    _hold(db_path, global_person_id="P1")
    _hold(db_path, global_person_id="P1")
    # a hold with unresolved identity still counts toward mover_count, never
    # toward unique_persons (spec: never guess an identity into existence)
    _hold(db_path, global_person_id=None)

    movers = db.query_product_movers(db_path, booth_id="BOOTH-1")
    assert len(movers) == 1
    assert movers[0]["mover_count"] == 3
    assert movers[0]["unique_persons"] == 1


def test_product_interest_summary_counts_and_scopes_correctly(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", global_person_id="P1")
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", global_person_id="P2")
    _hold(db_path, product_key="SNACK", product_name="ขนม", global_person_id="P1")

    summary = db.query_product_interest_summary(db_path, booth_id="BOOTH-1")
    by_key = {r["product_key"]: r for r in summary}
    assert by_key["WATER"]["interest_count"] == 2
    assert by_key["WATER"]["unique_persons"] == 2
    assert by_key["SNACK"]["interest_count"] == 1
    assert by_key["SNACK"]["unique_persons"] == 1


def test_product_interest_summary_omits_products_with_no_holds(tmp_path):
    """Spec section 37: this function alone never invents a zero-interest
    row — the caller merges this against the full product catalog for
    that (see the function's own docstring)."""
    db_path = _make_db(tmp_path)
    _hold(db_path, product_key="WATER")
    summary = db.query_product_interest_summary(db_path, booth_id="BOOTH-1")
    assert {r["product_key"] for r in summary} == {"WATER"}


def test_product_gender_matrix(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", gender="MALE")
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", gender="MALE")
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", gender="FEMALE")
    _hold(db_path, product_key="WATER", product_name="น้ำดื่ม", gender=None)  # -> UNKNOWN

    matrix = db.query_product_gender_matrix(db_path, booth_id="BOOTH-1")
    assert len(matrix) == 1
    row = matrix[0]
    assert row["product_key"] == "WATER"
    assert row["MALE"] == 2
    assert row["FEMALE"] == 1
    assert row["UNKNOWN"] == 1


def test_product_age_matrix(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, product_key="TOY", product_name="ของเล่น", age_category="CHILD")
    _hold(db_path, product_key="TOY", product_name="ของเล่น", age_category="CHILD")
    _hold(db_path, product_key="TOY", product_name="ของเล่น", age_category="ADULT")

    matrix = db.query_product_age_matrix(db_path, booth_id="BOOTH-1")
    row = matrix[0]
    assert row["CHILD"] == 2
    assert row["ADULT"] == 1
    assert row["UNKNOWN"] == 0


def test_gender_interest_totals(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, gender="MALE")
    _hold(db_path, gender="MALE")
    _hold(db_path, gender="FEMALE")
    _hold(db_path, gender=None)

    totals = db.query_gender_interest_totals(db_path, booth_id="BOOTH-1")
    assert totals == {"MALE": 2, "FEMALE": 1, "UNKNOWN": 1}


def test_age_interest_totals(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, age_category="CHILD")
    _hold(db_path, age_category="ADULT")
    _hold(db_path, age_category="ADULT")
    _hold(db_path, age_category=None)

    totals = db.query_age_interest_totals(db_path, booth_id="BOOTH-1")
    assert totals == {"CHILD": 1, "ADULT": 2, "UNKNOWN": 1}


def test_interest_overview(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, global_person_id="P1")
    _hold(db_path, global_person_id="P1")
    _hold(db_path, global_person_id="P2")
    _hold(db_path, global_person_id=None)

    overview = db.query_interest_overview(db_path, booth_id="BOOTH-1")
    assert overview == {"total_interest_events": 4, "unique_interested_persons": 2}


def test_analytics_scoped_by_booth(tmp_path):
    db_path = _make_db(tmp_path)
    _hold(db_path, gender="MALE")
    db.log_product_hold_event(db_path, "BOOTH-2", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 1,
                               100.0, 102.0, 2.0, gender="FEMALE")

    assert db.query_gender_interest_totals(db_path, booth_id="BOOTH-1") == {"MALE": 1, "FEMALE": 0, "UNKNOWN": 0}
    assert db.query_gender_interest_totals(db_path, booth_id="BOOTH-2") == {"MALE": 0, "FEMALE": 1, "UNKNOWN": 0}


# ------------------------------------------------------------- Re-ID merge bookkeeping
def test_merging_two_identities_repoints_history_and_stops_counting_the_duplicate(tmp_path):
    db_path = tmp_path / "merge.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    db.upsert_global_person(db_path, "BOOTH-1", "EVT-1", "P000001", 10.0, 20.0, 1)
    db.upsert_global_person(db_path, "BOOTH-1", "EVT-1", "P000002", 15.0, 40.0, 2)
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P000001", "MALE", 0.9, "UNKNOWN", "UNKNOWN", 0.0, 3, 10.0, 20.0)
    db.upsert_person_attributes(db_path, "BOOTH-1", "EVT-1", "P000002", "MALE", 0.8, "UNKNOWN", "UNKNOWN", 0.0, 2, 15.0, 40.0)
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 2

    db.merge_global_person(db_path, "BOOTH-1", "P000002", "P000001")

    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
    assert sum(db.query_attribute_breakdown(db_path, booth_id="BOOTH-1")["gender"].values()) == 1
    db.merge_global_person(db_path, "BOOTH-1", "P000002", "P000001")       # idempotent: merging again is harmless
    assert db.query_unique_people_count(db_path, booth_id="BOOTH-1") == 1
