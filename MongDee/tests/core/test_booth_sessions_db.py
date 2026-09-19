"""Unit tests for core/database.py's booth_sessions persistence + the
global_person_id column on tripwire_crossings — spec:
MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md sections 29/30/36.

Always against an isolated tmp_path DB — see core.database's own
production-DB guard (test_database_guard.py).
"""

from __future__ import annotations

from core import database as db


def _make_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    return db_path


def test_log_tripwire_crossing_stores_global_person_id(tmp_path):
    db_path = _make_db(tmp_path)
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-CAM-1", 7, "in",
                              ts=100.0, global_person_id="P000001")
    rows = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert len(rows) == 1
    assert rows[0]["global_person_id"] == "P000001"


def test_log_tripwire_crossing_without_global_person_id_is_null(tmp_path):
    db_path = _make_db(tmp_path)
    db.log_tripwire_crossing(db_path, "BOOTH-1", "EVT-1", "CAM-1", "TW-CAM-1", 7, "in", ts=100.0)
    rows = db.query_tripwire_crossings(db_path, booth_id="BOOTH-1")
    assert rows[0]["global_person_id"] is None


def test_open_then_close_booth_session(tmp_path):
    db_path = _make_db(tmp_path)
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-00000001", "P000001", "P000001",
                           entered_at=100.0, cameras_seen={"CAM-1"})

    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert len(sessions) == 1
    assert sessions[0]["status"] == "active"
    assert sessions[0]["exited_at"] is None
    assert sessions[0]["cameras_seen"] == ["CAM-1"]

    db.close_booth_session(db_path, "BOOTH-1", "BS-00000001", exited_at=427.0,
                            dwell_seconds=327.0, cameras_seen={"CAM-1", "CAM-2"})
    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert sessions[0]["status"] == "closed"
    assert sessions[0]["exited_at"] == 427.0
    assert sessions[0]["dwell_seconds"] == 327.0
    assert sorted(sessions[0]["cameras_seen"]) == ["CAM-1", "CAM-2"]


def test_update_booth_session_cameras_only_touches_active_sessions(tmp_path):
    db_path = _make_db(tmp_path)
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-1", "P1", "P1",
                           entered_at=100.0, cameras_seen={"CAM-1"})
    db.close_booth_session(db_path, "BOOTH-1", "BS-1", exited_at=110.0,
                            dwell_seconds=10.0, cameras_seen={"CAM-1"})

    # A late/racy camera-seen update after the session already closed must
    # not resurrect or mutate the closed row.
    db.update_booth_session_cameras(db_path, "BOOTH-1", "BS-1", {"CAM-1", "CAM-2"})
    sessions = db.query_booth_sessions(db_path, booth_id="BOOTH-1")
    assert sessions[0]["cameras_seen"] == ["CAM-1"]
    assert sessions[0]["status"] == "closed"


def test_booth_session_stats_separates_active_from_dwell_averages(tmp_path):
    db_path = _make_db(tmp_path)
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-1", "P1", "P1",
                           entered_at=0.0, cameras_seen={"CAM-1"})
    db.close_booth_session(db_path, "BOOTH-1", "BS-1", exited_at=100.0,
                            dwell_seconds=100.0, cameras_seen={"CAM-1"})
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-2", "P2", "P2",
                           entered_at=0.0, cameras_seen={"CAM-1"})
    db.close_booth_session(db_path, "BOOTH-1", "BS-2", exited_at=300.0,
                            dwell_seconds=300.0, cameras_seen={"CAM-1"})
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-3", "P3", "P3",
                           entered_at=0.0, cameras_seen={"CAM-1"})  # still active

    stats = db.query_booth_session_stats(db_path, booth_id="BOOTH-1")
    assert stats["total_sessions"] == 3
    assert stats["active_count"] == 1
    assert stats["avg_dwell_sec"] == 200.0
    assert stats["min_dwell_sec"] == 100.0
    assert stats["max_dwell_sec"] == 300.0


def test_booth_sessions_scoped_by_booth(tmp_path):
    db_path = _make_db(tmp_path)
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-1", "P1", "P1",
                           entered_at=0.0, cameras_seen={"CAM-1"})
    db.open_booth_session(db_path, "BOOTH-2", "EVT-1", "BS-2", "P1", "P1",
                           entered_at=0.0, cameras_seen={"CAM-1"})
    assert len(db.query_booth_sessions(db_path, booth_id="BOOTH-1")) == 1
    assert len(db.query_booth_sessions(db_path, booth_id="BOOTH-2")) == 1


def test_delete_scope_data_clears_booth_sessions(tmp_path):
    db_path = _make_db(tmp_path)
    db.open_booth_session(db_path, "BOOTH-1", "EVT-1", "BS-1", "P1", "P1",
                           entered_at=0.0, cameras_seen={"CAM-1"})
    db.delete_scope_data(db_path, booth_id="BOOTH-1")
    assert db.query_booth_sessions(db_path, booth_id="BOOTH-1") == []
