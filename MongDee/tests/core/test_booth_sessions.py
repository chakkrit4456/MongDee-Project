"""Unit tests for core/booth_sessions.py's BoothSessionManager — spec:
MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md sections 29-30.
"""

from __future__ import annotations

from core.booth_sessions import BoothSessionManager


def test_entry_opens_a_session():
    mgr = BoothSessionManager()
    session = mgr.on_entry("P001", "CAM-1", ts=100.0)
    assert session is not None
    assert session.person_key == "P001"
    assert session.entered_at == 100.0
    assert session.exited_at is None
    assert session.status == "active"
    assert session.cameras_seen == {"CAM-1"}
    assert mgr.active_count() == 1


def test_exit_closes_the_session_and_computes_dwell_from_real_timestamps():
    mgr = BoothSessionManager()
    mgr.on_entry("P001", "CAM-1", ts=100.0)
    closed = mgr.on_exit("P001", "CAM-1", ts=427.0)
    assert closed is not None
    assert closed.exited_at == 427.0
    assert closed.dwell_seconds == 327.0
    assert closed.status == "closed"
    assert mgr.active_count() == 0
    assert mgr.get_open_session("P001") is None


def test_duplicate_entry_for_someone_already_inside_is_a_no_op():
    """Spec section 30: CAM-A P103 IN then CAM-B P103 IN must not open a
    second session / restart entered_at — only record the second camera."""
    mgr = BoothSessionManager()
    first = mgr.on_entry("P103", "CAM-A", ts=100.0)
    second = mgr.on_entry("P103", "CAM-B", ts=105.0)
    assert second is None  # no new session opened
    assert mgr.active_count() == 1
    session = mgr.get_open_session("P103")
    assert session.entered_at == 100.0  # unchanged
    assert session.session_id == first.session_id
    assert session.cameras_seen == {"CAM-A", "CAM-B"}


def test_exit_with_nothing_open_is_a_no_op():
    mgr = BoothSessionManager()
    assert mgr.on_exit("P999", "CAM-1", ts=100.0) is None
    assert mgr.active_count() == 0


def test_note_camera_seen_adds_camera_to_open_session_without_a_crossing():
    """Spec section 21: a person switching cameras mid-visit must show up
    in cameras_seen even with no tripwire on that second camera at all."""
    mgr = BoothSessionManager()
    mgr.on_entry("P001", "CAM-1", ts=100.0)
    changed = mgr.note_camera_seen("P001", "CAM-2")
    assert changed is True
    assert mgr.get_open_session("P001").cameras_seen == {"CAM-1", "CAM-2"}

    # seeing the same camera again is not a "new" camera
    assert mgr.note_camera_seen("P001", "CAM-2") is False


def test_note_camera_seen_with_no_open_session_is_a_no_op():
    mgr = BoothSessionManager()
    assert mgr.note_camera_seen("P404", "CAM-1") is False


def test_cross_camera_visit_survives_a_camera_switch_before_exit():
    """Full spec-29/21 scenario: enters on CAM-1, is later seen (Re-ID
    resolved) on CAM-2 with no crossing there, then exits on CAM-2 — one
    session throughout, dwell computed from the original entry."""
    mgr = BoothSessionManager()
    mgr.on_entry("P042", "CAM-1", ts=1000.0)
    mgr.note_camera_seen("P042", "CAM-2")
    closed = mgr.on_exit("P042", "CAM-2", ts=1327.0)
    assert closed.dwell_seconds == 327.0
    assert closed.cameras_seen == {"CAM-1", "CAM-2"}


def test_independent_people_get_independent_sessions():
    mgr = BoothSessionManager()
    mgr.on_entry("P001", "CAM-1", ts=100.0)
    mgr.on_entry("P002", "CAM-1", ts=101.0)
    assert mgr.active_count() == 2
    ids = {s.session_id for s in mgr.all_open_sessions()}
    assert len(ids) == 2


def test_session_ids_are_unique_across_open_and_closed_sessions():
    mgr = BoothSessionManager()
    s1 = mgr.on_entry("P001", "CAM-1", ts=100.0)
    mgr.on_exit("P001", "CAM-1", ts=110.0)
    s2 = mgr.on_entry("P001", "CAM-1", ts=200.0)  # a second, separate visit
    assert s1.session_id != s2.session_id


def test_rekey_moves_an_open_session_to_the_surviving_identity():
    m = BoothSessionManager()
    m.on_entry("P000002", "CAM-1", 1.0)
    assert m.rekey("P000002", "P000001") is None
    assert m.get_open_session("P000002") is None
    assert m.get_open_session("P000001").person_key == "P000001"


def test_rekey_drops_the_duplicate_when_the_winner_already_has_a_session():
    m = BoothSessionManager()
    m.on_entry("P000001", "CAM-1", 1.0)
    dup = m.on_entry("P000002", "CAM-2", 2.0)
    assert m.rekey("P000002", "P000001") is dup
    assert m.active_count() == 1 and m.get_open_session("P000001").cameras_seen == {"CAM-1", "CAM-2"}
