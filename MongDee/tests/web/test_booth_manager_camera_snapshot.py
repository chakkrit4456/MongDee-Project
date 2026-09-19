"""Unit tests for web/booth_manager.py's live per-camera detail-panel
snapshot (BoothManager.get_camera_snapshot) — backs the popout single-camera
view's right-side panel (people by category + live product-detection count
for that one camera).
"""

from __future__ import annotations

import threading

from web.booth_manager import BoothManager


def _bare_booth_manager(camera_ids=("CAM-1", "CAM-2")):
    bm = object.__new__(BoothManager)
    bm._lock = threading.Lock()
    bm._person_tracks = {cid: [] for cid in camera_ids}
    bm._product_detections = {cid: [] for cid in camera_ids}
    return bm


def _track(track_id, category):
    return {"track_id": track_id, "bbox": [0, 0, 10, 10], "category": category}


def test_snapshot_with_no_data_is_all_zero():
    bm = _bare_booth_manager()
    snap = bm.get_camera_snapshot("CAM-1")
    assert snap == {
        "camera_id": "CAM-1", "people_total": 0, "male": 0, "female": 0,
        "unknown": 0, "products_detected": 0, "faces": 0,
    }


def test_snapshot_counts_people_by_category():
    bm = _bare_booth_manager()
    bm._person_tracks["CAM-1"] = [
        _track(1, "male"), _track(2, "male"), _track(3, "female"),
        _track(4, "child"), _track(5, "unknown"),  # a legacy "child" value counts as unknown
    ]
    snap = bm.get_camera_snapshot("CAM-1")
    assert snap["people_total"] == 5
    assert snap["male"] == 2
    assert snap["female"] == 1
    assert "child" not in snap
    assert snap["unknown"] == 2


def test_snapshot_treats_missing_category_as_unknown():
    bm = _bare_booth_manager()
    bm._person_tracks["CAM-1"] = [{"track_id": 1, "bbox": [0, 0, 10, 10]}]  # no "category" key
    snap = bm.get_camera_snapshot("CAM-1")
    assert snap["unknown"] == 1
    assert snap["people_total"] == 1


def test_snapshot_counts_product_detections():
    bm = _bare_booth_manager()
    bm._product_detections["CAM-1"] = [
        {"class_name": "bottle", "conf": 0.9, "bbox": [0, 0, 1, 1]},
        {"class_name": "cup", "conf": 0.8, "bbox": [0, 0, 1, 1]},
    ]
    snap = bm.get_camera_snapshot("CAM-1")
    assert snap["products_detected"] == 2


def test_snapshot_is_scoped_to_its_own_camera():
    bm = _bare_booth_manager()
    bm._person_tracks["CAM-1"] = [_track(1, "male")]
    bm._person_tracks["CAM-2"] = [_track(2, "female"), _track(3, "female")]
    bm._product_detections["CAM-2"] = [{"class_name": "bottle", "conf": 0.9, "bbox": [0, 0, 1, 1]}]

    snap1 = bm.get_camera_snapshot("CAM-1")
    snap2 = bm.get_camera_snapshot("CAM-2")
    assert snap1["male"] == 1 and snap1["female"] == 0 and snap1["products_detected"] == 0
    assert snap2["female"] == 2 and snap2["products_detected"] == 1


def test_snapshot_for_unknown_camera_id_is_empty_not_an_error():
    bm = _bare_booth_manager()
    snap = bm.get_camera_snapshot("CAM-DOES-NOT-EXIST")
    assert snap["people_total"] == 0
    assert snap["products_detected"] == 0
