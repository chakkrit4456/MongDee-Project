import json

import numpy as np
import pytest

from vision.product.classifier import KNOWN, POSSIBLE, UNKNOWN, ProductClassifier
from vision.product.config import ProductConfig
from vision.product.database import GIProduct, GIProductDatabase
from vision.product.tracker import ProductTracker
from tests.vision.conftest import FakeBox, FakeYOLO


def _solid(color, size=64):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = color
    img[::4] = tuple(c // 2 for c in color)  # texture
    return img


# --- GI database ------------------------------------------------------


def test_gi_database_lookup(tmp_path):
    path = tmp_path / "gi.json"
    path.write_text(json.dumps({"products": [
        {"id": "GI-001", "name": "ข้าวหอมมะลิทุ่งกุลา", "class_name": "rice_bag", "province": "Roi Et"},
        {"id": "GI-002", "name": "Doi Chaang Coffee", "class_name": "coffee_bag"},
    ]}), encoding="utf-8")
    db = GIProductDatabase.load(path)
    assert db.get("GI-001").province == "Roi Et"
    assert db.by_class_name("coffee_bag").id == "GI-002"
    assert len(db.all()) == 2


def test_gi_database_rejects_unknown_field(tmp_path):
    path = tmp_path / "gi.json"
    path.write_text(json.dumps({"products": [{"id": "GI-1", "name": "x", "bogus": 1}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown field"):
        GIProductDatabase.load(path)


def test_gi_database_duplicate_id():
    db = GIProductDatabase()
    db.add(GIProduct("GI-1", "a"))
    with pytest.raises(ValueError, match="duplicate"):
        db.add(GIProduct("GI-1", "b"))


# --- classifier ------------------------------------------------------


def test_classifier_empty_gallery_is_unknown():
    clf = ProductClassifier(ProductConfig(reid_backend="color"))
    assert clf.classify(_solid((200, 50, 50))).status == UNKNOWN
    assert not clf.is_ready


def test_classifier_known_vs_unknown():
    clf = ProductClassifier(ProductConfig(reid_backend="color", known_similarity=0.7, possible_similarity=0.4))
    clf.add_reference("GI-001", _solid((200, 40, 40)))
    clf.add_reference("GI-001", _solid((205, 45, 42)))
    clf.add_reference("GI-002", _solid((40, 40, 200)))

    m_known = clf.classify(_solid((200, 42, 41)))
    assert m_known.status == KNOWN
    assert m_known.product_id == "GI-001"

    m_unknown = clf.classify(_solid((40, 200, 40)))  # green — matches neither
    assert m_unknown.status == UNKNOWN


def test_classifier_close_call_is_possible():
    clf = ProductClassifier(ProductConfig(reid_backend="color", known_similarity=0.6, possible_similarity=0.3, top2_margin=0.2))
    clf.add_reference("GI-001", _solid((150, 120, 100)))
    clf.add_reference("GI-002", _solid((150, 120, 105)))  # nearly identical -> no clear winner
    m = clf.classify(_solid((150, 120, 102)))
    assert m.status == POSSIBLE


# --- tracker -------------------------------------------------------


def test_product_tracker_stable_ids():
    tracker = ProductTracker(ProductConfig(track_min_hits=2, track_match_iou=0.3))
    ids = set()
    for i in range(5):
        tracks = tracker.update([[100, 100, 150, 200]], [0.9], timestamp=float(i))
        ids.update(t.track_id for t in tracks)
    assert ids == {1}


def test_product_tracker_ages_out():
    tracker = ProductTracker(ProductConfig(track_min_hits=1, track_max_age_sec=1.0))
    tracker.update([[100, 100, 150, 200]], [0.9], timestamp=0.0)
    tracker.update([], [], timestamp=2.0)  # 2s later, no detection
    assert tracker.confirmed_tracks() == []


# --- module (with fake YOLO) ---------------------------------------


def test_product_module_end_to_end():
    from vision.config import DetectionConfig
    from vision.detection.detector import PersonDetector
    from vision.product.pipeline import ProductVisionModule

    # fake YOLO returns a "bottle" (class 39) box
    detector = PersonDetector(DetectionConfig(device="cpu"), model=FakeYOLO(
        boxes=[FakeBox(39, 0.9, [100, 100, 150, 250])],
        names={0: "person", 39: "bottle"},
    ))
    cfg = ProductConfig(reid_backend="color", known_similarity=0.6, possible_similarity=0.3, track_min_hits=1)
    db = GIProductDatabase([GIProduct("GI-001", "Local Honey", class_name="bottle")])
    clf = ProductClassifier(cfg)
    clf.add_reference("GI-001", _solid((60, 140, 200)))
    module = ProductVisionModule(detector, cfg, classifier=clf, gi_database=db)

    frame = np.full((400, 400, 3), 128, np.uint8)
    frame[100:250, 100:150] = (60, 140, 200)  # match the honey reference
    frame[100:250:4, 100:150] = (30, 70, 100)

    results = []
    for i in range(3):
        results = module.process("CAM01", frame, timestamp=float(i))
    assert len(results) == 1
    pd = results[0]
    assert pd.local_track_id == 1
    assert pd.product_id == "GI-001"
    assert pd.status in (KNOWN, POSSIBLE)
    assert module.gi_info(pd.product_id).name == "Local Honey"
