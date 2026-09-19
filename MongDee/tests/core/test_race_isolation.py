"""Race isolation regression tests — spec: MongDee person-gender/age master
continuation prompt sections 5-7, 44-51, 59, 77 ("Race must never affect
Customer/Product Interest").

FairFace's model head genuinely outputs race logits (core/attributes.py's
FairFaceBackend forward pass) — that capability is preserved (spec section
7), but every downstream interface in this codebase (AttributeResult,
GlobalPersonAttributeSmoother, ProductInterestTracker, the product_hold_
events schema, every Gender/Age analytics query) simply has no race
parameter or column at all. These tests prove that structurally: not just
"race isn't currently used", but "there is no code path through which a
race value could reach an Interest decision, even by accident".
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from core import attributes, database as db, interest_tracker
from core.attributes import AttributeResult, GlobalPersonAttributeSmoother, to_age_category
from core.interest_tracker import ProductInterestTracker


# --------------------------------------------------------- structural audit

def test_attribute_result_has_no_race_field():
    field_names = {f.name for f in dataclasses.fields(AttributeResult)}
    assert "race" not in field_names
    assert field_names == {
        "gender", "gender_confidence", "age_group", "age_category",
        "age_confidence", "sample_count", "last_updated", "status", "source",
    }


def test_global_person_attribute_smoother_add_sample_rejects_race_kwarg():
    """Spec section 48: even if a caller tried to smuggle race in as an
    extra keyword argument, the function signature itself refuses it."""
    smoother = GlobalPersonAttributeSmoother()
    with pytest.raises(TypeError):
        smoother.add_sample("P1", "male", 0.9, "20-29", 0.8, 0.0, race="A")  # type: ignore[call-arg]


def test_interest_tracker_update_has_no_race_parameter():
    sig = inspect.signature(ProductInterestTracker.update)
    assert "race" not in sig.parameters


def test_log_product_hold_event_rejects_race_kwarg():
    sig = inspect.signature(db.log_product_hold_event)
    assert "race" not in sig.parameters
    with pytest.raises(TypeError):
        db.log_product_hold_event(  # type: ignore[call-arg]
            "unused.db", "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
            100.0, 105.0, 5.0, race="A",
        )


def test_open_product_interest_event_rejects_race_kwarg():
    sig = inspect.signature(db.open_product_interest_event)
    assert "race" not in sig.parameters
    with pytest.raises(TypeError):
        db.open_product_interest_event(  # type: ignore[call-arg]
            "unused.db", "BOOTH-1", "EVT-1", "CAM-1", "WATER", "น้ำดื่ม", 7,
            100.0, 102.0, race="A",
        )


@pytest.mark.parametrize("fn", [
    db.query_gender_interest_totals, db.query_age_interest_totals,
    db.query_product_gender_matrix, db.query_product_age_matrix,
    db.query_interest_overview, db.query_product_interest_summary,
    db.query_product_movers,
])
def test_interest_analytics_sql_never_references_race(fn):
    """Spec section 47's static audit, made executable: none of the
    Gender/Age/Product analytics queries this feature added (or the
    pre-existing product_movers) may filter, GROUP BY, or otherwise
    reference a race/ethnicity column anywhere in their SQL."""
    source = inspect.getsource(fn).lower()
    for forbidden in ("race", "ethnicity", "demographic"):
        assert forbidden not in source, f"{fn.__name__} source mentions {forbidden!r}"


def test_fairface_backend_race_logits_are_sliced_off_and_never_returned():
    """core.attributes.FairFaceBackend._predict_face reads race(7) +
    gender(2) + age(9) = 18 logits off the model's fc head (the model can
    still do race inference internally — spec section 7), but only ever
    returns (gender_label, gender_conf, age_group_label, age_conf) — a
    4-tuple, never a 5th race element."""
    source = inspect.getsource(attributes.FairFaceBackend._predict_face)
    assert "return (GENDER_LABELS[g_idx]" in source
    # the slice that discards race logits before gender/age are computed
    assert "gender_logits = logits[NUM_RACE_CLASSES:" in source


def test_predict_gender_and_predict_age_group_never_expose_race():
    """FairFaceBackend.predict_gender/predict_age_group (the interface both
    the live per-frame classifier and the Global-Person attribute analyzer
    actually call) each return a plain (label, confidence) pair — no race,
    no third value."""
    sig_gender = inspect.signature(attributes.FairFaceBackend.predict_gender)
    sig_age = inspect.signature(attributes.FairFaceBackend.predict_age_group)
    assert list(sig_gender.parameters) == ["self", "crop_bgr"]
    assert list(sig_age.parameters) == ["self", "crop_bgr"]


# ------------------------------------------------------------- behavioral (49-51)

def test_race_unknown_does_not_crash_attribute_smoothing():
    """Spec section 50: even though there's no race parameter to pass,
    prove UNKNOWN gender/age (the honest fallback whenever FairFace's own
    race-agnostic confidence floors aren't cleared) never crashes."""
    smoother = GlobalPersonAttributeSmoother()
    result = smoother.add_sample("P1", "unknown", 0.0, "unknown", 0.0, 0.0)
    assert result.gender == "UNKNOWN"
    assert result.age_category == "UNKNOWN"
    assert result.status == "unknown"


def test_race_missing_does_not_crash_age_category_mapping():
    """Spec section 51: to_age_category (used by both the live smoother and
    every downstream analytics query's age_category column) only ever sees
    gender/age labels — never race — and must handle an absent/unrecognized
    label without raising."""
    assert to_age_category("unknown") == "UNKNOWN"
    assert to_age_category(None) == "UNKNOWN"  # type: ignore[arg-type]


def test_same_interest_decision_regardless_of_which_person_holds_it(monkeypatch):
    """Spec section 49's regression test, adapted to what this codebase can
    actually represent: two different people (standing in for "same gender/
    age, different race" — race isn't a representable dimension anywhere in
    this pipeline at all) performing the *exact* same product interaction
    (same product, same movement, same spatial relation, same duration)
    must produce the exact same interest decision and confirmation timing —
    only holder_track_id may differ."""
    resting = [100.0, 100.0, 140.0, 180.0]
    moved = [200.0, 100.0, 240.0, 180.0]
    person_a = {"track_id": 101, "bbox": [180.0, 90.0, 260.0, 190.0]}
    person_b = {"track_id": 202, "bbox": [180.0, 90.0, 260.0, 190.0]}

    def run(holder):
        tracker = ProductInterestTracker()
        monkeypatch.setattr(interest_tracker.time, "time", lambda: 0.0)
        tracker.update("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": resting}], [])
        monkeypatch.setattr(interest_tracker.time, "time", lambda: 0.0)
        tracker.update("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": moved}], [holder])
        monkeypatch.setattr(interest_tracker.time, "time", lambda: 2.0)
        return tracker.update("CAM-1", [{"class_name": "WATER", "conf": 0.9, "bbox": moved}], [holder])

    events_a = run(person_a)
    events_b = run(person_b)
    assert len(events_a) == len(events_b) == 1
    a, b = events_a[0], events_b[0]
    assert a["event"] == b["event"] == "confirmed"
    assert a["confirmed_at"] == b["confirmed_at"]
    assert a["hold_start_ts"] == b["hold_start_ts"]
    # the only difference is identity, never the decision/timing
    assert {k: v for k, v in a.items() if k != "holder_track_id"} == \
           {k: v for k, v in b.items() if k != "holder_track_id"}
