"""Camera recovery must be MEASURED, not just claimed (cross-camera-identity
+ camera-recovery master prompt, section 38/55/56): web/booth_manager.py's
_on_status now tracks, per camera, how many offline->online recovery cycles
happened, how long the most recent one took, and the last failure message --
exposed straight through BoothManager.get_state()['cameras'][camera_id]
since that already does `dict(v) for cid, v in self.camera_status.items()`.

Same "bare" BoothManager construction pattern as
test_booth_manager_product_idle_clear.py, since _on_status calls
core.database.log_health_event (needs a real initialized db) and
self._push_alert (needs self.recent_alerts).
"""

from __future__ import annotations

import threading

from core import database as db
from web.booth_manager import BoothManager


def _bare_booth_manager(db_path):
    bm = object.__new__(BoothManager)
    bm.db_path = db_path
    bm.booth_id = "BOOTH-1"
    bm.event_id = "EVT-1"
    bm.camera_ids = ["CAM-1"]
    bm._lock = threading.Lock()
    bm.camera_status = {"CAM-1": {"status": "unknown", "message": ""}}
    bm._camera_failure_started = {}
    bm._camera_reconnect_count = {}
    bm._camera_last_error = {}
    bm._camera_last_recovery_latency_sec = {}
    bm._person_tracks = {"CAM-1": []}
    bm._product_detections = {"CAM-1": []}
    bm.recent_alerts = []
    return bm


def _init_db(tmp_path):
    db_path = tmp_path / "test.db"
    db.init_db(db_path)
    db.create_event(db_path, "EVT-1", "Event 1")
    db.create_booth(db_path, "BOOTH-1", "Booth 1", "EVT-1")
    return db_path


def test_fresh_camera_has_zero_reconnects_and_no_error(tmp_path):
    bm = _bare_booth_manager(_init_db(tmp_path))
    bm._on_status("CAM-1", "online", "ปกติ")
    entry = bm.camera_status["CAM-1"]
    assert entry["reconnect_count"] == 0
    assert entry["last_error"] == ""
    assert entry["last_recovery_latency_sec"] is None


def test_recovery_latency_is_measured_from_the_moment_it_left_online(tmp_path, monkeypatch):
    bm = _bare_booth_manager(_init_db(tmp_path))
    fake_now = [100.0]
    monkeypatch.setattr("web.booth_manager.time.monotonic", lambda: fake_now[0])

    bm._on_status("CAM-1", "online", "ปกติ")
    fake_now[0] = 105.0
    bm._on_status("CAM-1", "offline", "อ่านภาพจากกล้องไม่ได้ต่อเนื่อง")
    fake_now[0] = 107.5
    bm._on_status("CAM-1", "reconnecting", "กำลังพยายามเชื่อมต่อใหม่")
    fake_now[0] = 109.2
    bm._on_status("CAM-1", "online", "ปกติ")

    entry = bm.camera_status["CAM-1"]
    assert entry["reconnect_count"] == 1
    # measured from when it left "online" (105.0), not from the last "reconnecting" step (107.5)
    assert entry["last_recovery_latency_sec"] == 109.2 - 105.0
    assert entry["last_error"] == "อ่านภาพจากกล้องไม่ได้ต่อเนื่อง"


def test_repeated_outages_increment_reconnect_count_each_time(tmp_path):
    bm = _bare_booth_manager(_init_db(tmp_path))
    bm._on_status("CAM-1", "online", "ปกติ")
    for _ in range(3):
        bm._on_status("CAM-1", "offline", "เชื่อมต่อขาดหาย")
        bm._on_status("CAM-1", "online", "ปกติ")
    assert bm.camera_status["CAM-1"]["reconnect_count"] == 3


def test_degraded_transition_preserves_recovery_fields(tmp_path):
    """_on_camera_degraded writes camera_status directly (AdaptiveController callback, not a
    CameraWorker status), and must not silently drop the reconnect_count/last_error fields
    _on_status just populated -- a camera going degraded-then-back-online is not a new outage."""
    bm = _bare_booth_manager(_init_db(tmp_path))
    bm._on_status("CAM-1", "online", "ปกติ")
    bm._on_status("CAM-1", "offline", "เชื่อมต่อขาดหาย")
    bm._on_status("CAM-1", "online", "ปกติ")
    assert bm.camera_status["CAM-1"]["reconnect_count"] == 1

    bm._on_camera_degraded("CAM-1", True)
    assert bm.camera_status["CAM-1"]["status"] == "degraded"
    assert bm.camera_status["CAM-1"]["reconnect_count"] == 1   # not reset by the degraded transition

    bm._on_camera_degraded("CAM-1", False)
    assert bm.camera_status["CAM-1"]["status"] == "online"
    assert bm.camera_status["CAM-1"]["reconnect_count"] == 1


def test_state_only_toggle_between_offline_and_reconnecting_does_not_double_count(tmp_path):
    """One outage that churns offline<->reconnecting several times before finally recovering is
    ONE reconnect, not several -- only reaching "online" again closes the cycle."""
    bm = _bare_booth_manager(_init_db(tmp_path))
    bm._on_status("CAM-1", "online", "ปกติ")
    bm._on_status("CAM-1", "offline", "a")
    bm._on_status("CAM-1", "reconnecting", "b")
    bm._on_status("CAM-1", "offline", "c")
    bm._on_status("CAM-1", "reconnecting", "d")
    bm._on_status("CAM-1", "online", "e")
    assert bm.camera_status["CAM-1"]["reconnect_count"] == 1
