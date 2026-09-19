"""Unit tests for core/interest_tracker.py's Real-Time 2-Second Confirmation
lifecycle — spec: MongDee person-gender/age master continuation prompt
sections 17-27, 52-59.

Uses monkeypatch.setattr(interest_tracker.time, "time", ...) to control
"now" precisely (same pattern as tests/core/test_delete_scope_data.py) so
boundary cases (exactly 2.0s, still-holding, hand-off dedup, ...) are exact
and instant rather than relying on real sleeps. Timestamps are deliberately
kept to clean integers so `start + INTEREST_CONFIRMATION_SECONDS` lands on
an exact float (avoids spurious floating-point-rounding test failures like
2.01 - 0.01 != 2.0).
"""

from __future__ import annotations

from core import interest_tracker
from core.interest_tracker import INTEREST_CONFIRMATION_SECONDS, ProductInterestTracker

RESTING = [100.0, 100.0, 140.0, 180.0]     # centroid (120, 140)
MOVED = [200.0, 100.0, 240.0, 180.0]       # centroid (220, 140) — far from RESTING
MOVED2 = [300.0, 100.0, 340.0, 180.0]      # centroid (320, 140) — far from MOVED too (for hand-offs)
HOLDER_A = {"track_id": 1, "bbox": [180.0, 90.0, 260.0, 190.0]}   # covers MOVED's centroid
HOLDER_B = {"track_id": 2, "bbox": [280.0, 90.0, 360.0, 190.0]}   # covers MOVED2's centroid
RELEASED = [202.0, 102.0, 242.0, 182.0]    # close to MOVED -> reads as "not moved"


def _det(bbox, conf=0.9, class_name="WATER"):
    return [{"class_name": class_name, "conf": conf, "bbox": bbox}]


def _set_now(monkeypatch, t):
    monkeypatch.setattr(interest_tracker.time, "time", lambda: t)


def _confirmed_events(events):
    return [e for e in events if e["event"] == "confirmed"]


def _finalized_events(events):
    return [e for e in events if e["event"] == "finalized"]


# --------------------------------------------------------------- Test A/B/C/D
# (spec section 52: 0.5s/1.9s -> no interest, 2.0s/3.5s -> confirmed)

def test_released_before_2_seconds_produces_no_event_at_all(monkeypatch):
    """Test A/B: pick up, release at 0.5s and 1.9s — spec sections 19/27/29."""
    for release_after in (0.5, 1.9):
        tracker = ProductInterestTracker()
        _set_now(monkeypatch, 0.0)
        tracker.update("CAM-1", _det(RESTING), [])
        _set_now(monkeypatch, 0.0)
        tracker.update("CAM-1", _det(MOVED), [HOLDER_A])  # hold begins

        _set_now(monkeypatch, release_after)
        events = tracker.update("CAM-1", _det(RELEASED), [])  # released, no holder
        assert events == [], f"release_after={release_after}"


def test_exactly_2_seconds_confirms_while_still_held(monkeypatch):
    """Test C: spec section 20 — confirmation must fire AT 2.0s, without
    waiting for release."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 1.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])  # hold begins at t=1.0

    _set_now(monkeypatch, 1.0 + INTEREST_CONFIRMATION_SECONDS)
    events = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])  # still held
    confirmed = _confirmed_events(events)
    assert len(confirmed) == 1
    assert confirmed[0]["holder_track_id"] == 1
    assert _finalized_events(events) == []  # not released yet


def test_more_than_2_seconds_confirms(monkeypatch):
    """Test D: 3.5s of continuous holding -> confirmed."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, 3.5)
    events = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    assert len(_confirmed_events(events)) == 1


# ------------------------------------------------------- Test E/F (22/23/24)

def test_still_holding_after_confirm_does_not_refire_or_duplicate(monkeypatch):
    """Test E: t=0 pick, t=2 confirmed, t=5 still holding -> still exactly
    one confirmed event total, no duplicate."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, INTEREST_CONFIRMATION_SECONDS)
    first = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    assert len(_confirmed_events(first)) == 1

    _set_now(monkeypatch, 5.0)
    second = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    assert second == []  # no repeat confirm, no premature finalize


def test_release_after_confirm_produces_exactly_one_finalized_event(monkeypatch):
    """Test F: t=0 pick, t=2 confirm, t=4 release -> exactly 1 total
    Interest Event across the whole lifecycle (1 confirmed + 1 finalized,
    but only the finalized one is the durable record — spec section 23:
    release finalizes, it never creates a second event)."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, INTEREST_CONFIRMATION_SECONDS)
    confirmed = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    assert len(_confirmed_events(confirmed)) == 1

    _set_now(monkeypatch, 4.0)
    released = tracker.update("CAM-1", _det(RELEASED), [])
    finalized = _finalized_events(released)
    assert len(finalized) == 1
    assert finalized[0]["duration_sec"] == 4.0


# --------------------------------------------------------------- Test 27/28/29

def test_cancel_before_confirmation_leaves_no_trace(monkeypatch):
    """Spec section 27: CONFIRMING -> invalid before 2s -> cancelled, no event."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    _set_now(monkeypatch, 1.0)
    events = tracker.update("CAM-1", _det(RELEASED), [])  # released before 2s
    assert events == []


def test_person_walks_away_without_product_moving_is_no_interest(monkeypatch):
    """Spec section 28: person nearby, product stationary, person leaves."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 1.0)
    events = tracker.update("CAM-1", _det(RESTING), [HOLDER_A])  # near, not moved
    assert events == []
    _set_now(monkeypatch, 2.0)
    events = tracker.update("CAM-1", _det(RESTING), [])  # walked away
    assert events == []


def test_product_moves_without_any_person_is_not_customer_interest(monkeypatch):
    """Spec section 30: product displaced, no attributable holder -> never
    an Interest Event (may be a product_movement_event elsewhere, but this
    tracker's whole output vocabulary here is Interest Events)."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 1.0)
    events = tracker.update("CAM-1", _det(MOVED), [])  # moved, no holder
    assert events == []
    _set_now(monkeypatch, 4.0)
    events = tracker.update("CAM-1", _det(MOVED), [])
    assert events == []


def test_camera_jitter_does_not_confirm_interest(monkeypatch):
    """Spec section 31/57: sub-threshold bbox wobble must never read as a
    pickup at all."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    jittered = [101.0, 99.0, 141.0, 179.0]  # a couple px of jitter, well under threshold
    for t in (0.5, 1.0, 2.5, 5.0):
        _set_now(monkeypatch, t)
        events = tracker.update("CAM-1", _det(jittered), [HOLDER_A])
        assert events == [], f"t={t}"


# ----------------------------------------------------------------- Test 53/54/55

def test_event_dedup_across_many_polls_while_confirming_and_held(monkeypatch):
    """Spec section 53: many update() calls across the confirming period
    and while held after confirmation must total exactly one confirmed +
    one finalized event, never one per poll."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    all_events = []
    for t in (0.5, 1.0, 1.5, 1.9, 2.0, 2.5, 3.0, 4.0):
        _set_now(monkeypatch, t)
        all_events.extend(tracker.update("CAM-1", _det(MOVED), [HOLDER_A]))

    assert len(_confirmed_events(all_events)) == 1

    _set_now(monkeypatch, 5.0)
    all_events.extend(tracker.update("CAM-1", _det(RELEASED), []))
    assert len(_confirmed_events(all_events)) == 1
    assert len(_finalized_events(all_events)) == 1


def test_multiple_person_test_only_actual_holder_is_attributed(monkeypatch):
    """Spec section 54: Person A and Person B both in frame, only Person A
    actually touches/holds the product -> event attributes to A, never B."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    bystander = {"track_id": 99, "bbox": [0.0, 0.0, 20.0, 20.0]}  # far away, never touches
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A, bystander])

    _set_now(monkeypatch, 2.5)
    events = tracker.update("CAM-1", _det(MOVED), [HOLDER_A, bystander])
    confirmed = _confirmed_events(events)
    assert len(confirmed) == 1
    assert confirmed[0]["holder_track_id"] == 1  # never 99


def test_multiple_product_test_only_touched_product_gets_an_event(monkeypatch):
    """Spec section 55: two products in frame, only one is picked up."""
    tracker = ProductInterestTracker()
    snack_resting = [400.0, 400.0, 440.0, 480.0]
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING, class_name="WATER") + _det(snack_resting, class_name="SNACK"), [])

    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED, class_name="WATER") + _det(snack_resting, class_name="SNACK"),
                    [HOLDER_A])

    _set_now(monkeypatch, 2.5)
    events = tracker.update(
        "CAM-1", _det(MOVED, class_name="WATER") + _det(snack_resting, class_name="SNACK"), [HOLDER_A])
    confirmed = _confirmed_events(events)
    assert len(confirmed) == 1
    assert confirmed[0]["class_name"] == "WATER"


def test_multi_camera_isolation(monkeypatch):
    """Spec section 56: two cameras tracking their own products must never
    cross-assign camera_id."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-A", _det(RESTING), [])
    tracker.update("CAM-B", _det(RESTING), [])

    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-A", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, 2.5)
    events_a = tracker.update("CAM-A", _det(MOVED), [HOLDER_A])
    events_b = tracker.update("CAM-B", _det(RESTING), [])  # never moved on CAM-B
    assert len(_confirmed_events(events_a)) == 1
    assert events_a[0]["camera_id"] == "CAM-A"
    assert events_b == []


# --------------------------------------------------------------- hand-off (25)

def test_handoff_to_a_different_holder_before_confirmation_drops_silently(monkeypatch):
    """A hand-off before 2s clears has nothing confirmed to finalize — spec
    section 19/25: only a genuinely-evidenced, sufficiently-long interaction
    produces an event. The new holder's own clock starts at the hand-off."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, 1.0)  # before confirmation; product moves again to B
    events = tracker.update("CAM-1", _det(MOVED2), [HOLDER_B])  # hand-off A -> B
    assert events == []

    _set_now(monkeypatch, 1.0 + INTEREST_CONFIRMATION_SECONDS)
    events = tracker.update("CAM-1", _det(MOVED2), [HOLDER_B])
    confirmed = _confirmed_events(events)
    assert len(confirmed) == 1
    assert confirmed[0]["holder_track_id"] == 2  # confirmed under B's hold, started at the handoff


def test_handoff_after_confirmation_finalizes_the_first_holder(monkeypatch):
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, 2.5)
    events = tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    assert _confirmed_events(events)[0]["holder_track_id"] == 1

    _set_now(monkeypatch, 3.0)
    events = tracker.update("CAM-1", _det(MOVED2), [HOLDER_B])  # hand-off after confirm
    finalized = _finalized_events(events)
    assert len(finalized) == 1
    assert finalized[0]["holder_track_id"] == 1  # A's confirmed segment closes out


# ------------------------------------------------------------------ live_states

def test_live_states_reports_possible_interaction_then_confirmed(monkeypatch):
    """Spec sections 32-34: backend-decided live status for polling UIs."""
    tracker = ProductInterestTracker()
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(RESTING), [])
    _set_now(monkeypatch, 0.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])

    _set_now(monkeypatch, 1.0)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    live = {(s["camera_id"], s["class_name"]): s for s in tracker.live_states()}
    assert live[("CAM-1", "WATER")]["live_status"] == "POSSIBLE_INTERACTION"
    assert live[("CAM-1", "WATER")]["confirmed"] is False

    _set_now(monkeypatch, 2.5)
    tracker.update("CAM-1", _det(MOVED), [HOLDER_A])
    live = {(s["camera_id"], s["class_name"]): s for s in tracker.live_states()}
    assert live[("CAM-1", "WATER")]["live_status"] == "INTEREST_CONFIRMED"
    assert live[("CAM-1", "WATER")]["confirmed"] is True
