import numpy as np
import pytest

from vision.spatial.booth import BoothLayout, BoothObject
from vision.spatial.calibration import CalibrationStore, CameraCalibration
from vision.spatial.heatmap import HeatmapAccumulator, MovementPath
from vision.spatial.world import WorldMapper, foot_point


# --- calibration --------------------------------------------------------


def _square_calib(camera_id="CAM01"):
    # image corners of a 1000x1000 view map to a 10m x 6m floor
    return CameraCalibration.from_points(
        camera_id,
        image_points=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
        world_points=[(0, 0), (10, 0), (10, 6), (0, 6)],
    )


def test_calibration_pixel_to_world_round_trip():
    calib = _square_calib()
    wx, wy = calib.pixel_to_world(500, 500)
    assert wx == pytest.approx(5.0, abs=0.01)
    assert wy == pytest.approx(3.0, abs=0.01)
    px, py = calib.world_to_pixel(wx, wy)
    assert px == pytest.approx(500, abs=1)
    assert py == pytest.approx(500, abs=1)
    assert calib.reprojection_error < 1.0
    assert calib.is_reliable


def test_calibration_needs_four_points():
    with pytest.raises(ValueError, match=">= 4"):
        CameraCalibration.from_points("C", [(0, 0), (1, 1)], [(0, 0), (2, 2)])


def test_calibration_store_save_load(tmp_path):
    store = CalibrationStore()
    store.set(_square_calib("CAM01"))
    store.set(_square_calib("CAM02"))
    path = tmp_path / "calib.json"
    store.save(path)
    loaded = CalibrationStore.load(path)
    assert set(loaded.camera_ids()) == {"CAM01", "CAM02"}
    wx, wy = loaded.get("CAM01").pixel_to_world(0, 0)
    assert (round(wx), round(wy)) == (0, 0)


# --- booth layout -----------------------------------------------------


def test_booth_object_contains_and_distance():
    shelf = BoothObject("SHELF-A", "shelf", x=4.0, y=2.0, width=1.5, height=0.5)
    assert shelf.contains(4.0, 2.0)
    assert shelf.contains(4.6, 2.1)
    assert not shelf.contains(6.0, 2.0)
    assert shelf.distance_to(4.0, 2.0) == 0.0
    assert shelf.distance_to(4.0, 4.25) == pytest.approx(2.0, abs=0.01)


def test_booth_object_rejects_bad_type():
    with pytest.raises(ValueError, match="unknown booth object_type"):
        BoothObject("X", "teleporter", 0, 0)


def test_booth_layout_zone_lookup_and_versioning():
    layout = BoothLayout("BOOTH-01", width=10, length=6)
    layout.add(BoothObject("ZONE-A1", "product_zone", x=3, y=2, width=2, height=2, metadata={"products": ["GI-001"]}))
    layout.add(BoothObject("CAM01", "camera", x=0, y=0))
    assert layout.zone_at(3, 2).id == "ZONE-A1"
    assert layout.zone_at(9, 5) is None
    assert layout.in_bounds(5, 3)
    assert not layout.in_bounds(20, 3)
    assert [o.id for o in layout.of_type("camera")] == ["CAM01"]

    v2 = layout.bumped_version()
    assert v2.version == 2
    assert layout.version == 1  # original untouched
    with pytest.raises(ValueError, match="already in layout"):
        layout.add(BoothObject("ZONE-A1", "product_zone", 0, 0))


def test_booth_layout_dict_round_trip():
    layout = BoothLayout("BOOTH-01", width=8, length=5)
    layout.add(BoothObject("SHELF-A", "shelf", x=2, y=2, width=1, height=0.5, rotation=45))
    restored = BoothLayout.from_dict(layout.to_dict())
    assert restored.booth_id == "BOOTH-01"
    assert restored.get("SHELF-A").rotation == 45


# --- world mapping ---------------------------------------------------


def test_world_mapper_locates_foot_point_in_zone():
    store = CalibrationStore()
    store.set(_square_calib("CAM01"))
    layout = BoothLayout("BOOTH-01", width=10, length=6)
    layout.add(BoothObject("ZONE-A1", "product_zone", x=5, y=3, width=4, height=4))
    mapper = WorldMapper(store, layout)

    # bbox whose bottom-centre is image (500, 1000) -> world (5, 6)... let's aim for centre
    pos = mapper.locate("CAM01", bbox=[400, 100, 600, 500], global_id="PERSON-0001", timestamp=0.0)
    assert pos is not None
    assert pos.x == pytest.approx(5.0, abs=0.1)
    assert pos.zone_id == "ZONE-A1"
    assert pos.confidence > 0.5


def test_world_mapper_rejects_impossible_jump():
    store = CalibrationStore()
    store.set(_square_calib("CAM01"))
    mapper = WorldMapper(store, max_speed_mps=3.0)
    p1 = mapper.locate("CAM01", [0, 0, 100, 100], global_id="P", timestamp=0.0)
    p2 = mapper.locate("CAM01", [900, 900, 1000, 1000], global_id="P", timestamp=0.1)  # ~11m in 0.1s
    assert p1.confidence > 0.5
    assert p2.confidence < 0.3  # flagged as impossible


def test_world_mapper_unknown_camera_returns_none():
    mapper = WorldMapper(CalibrationStore())
    assert mapper.locate("NOPE", [0, 0, 10, 10]) is None


def test_foot_point():
    assert foot_point([10, 20, 30, 80]) == (20.0, 80.0)


# --- heatmap --------------------------------------------------------


def test_heatmap_accumulates_and_finds_hotspots():
    hm = HeatmapAccumulator(booth_width=10, booth_length=6, cell_size=1.0)
    for _ in range(10):
        hm.add(3.5, 2.5, timestamp=0.0)
    hm.add(8.0, 5.0, timestamp=0.0)
    grid = hm.grid()
    assert grid.shape == (6, 10)
    assert grid[2, 3] == 10
    top = hm.top_cells(n=2)
    assert top[0]["value"] == 10
    assert (top[0]["x"], top[0]["y"]) == (3.5, 2.5)  # centre of cell (row 2, col 3)


def test_heatmap_time_window():
    hm = HeatmapAccumulator(4, 4, cell_size=1.0)
    hm.add(1, 1, timestamp=0.0)
    hm.add(1, 1, timestamp=100.0)
    assert hm.grid(since=50.0).sum() == 1
    hm.prune(older_than=50.0)
    assert hm.sample_count == 1


def test_movement_path():
    path = MovementPath("PERSON-0001")
    path.add(0, 0, timestamp=0.0, camera_id="CAM01", zone_id=None, confidence=0.9)
    path.add(3, 4, timestamp=2.0, camera_id="CAM01", zone_id="ZONE-A1", confidence=0.9)
    path.add(3, 4, timestamp=3.0, camera_id="CAM02", zone_id="ZONE-A1", confidence=0.8)
    assert path.total_distance() == pytest.approx(5.0)
    assert path.zones_visited() == ["ZONE-A1"]
    assert path.is_consistent()

    bad = MovementPath("P2")
    bad.add(0, 0, 0.0, "CAM01", None, 0.9)
    bad.add(50, 0, 0.5, "CAM02", None, 0.9)  # 100 m/s
    assert not bad.is_consistent()
