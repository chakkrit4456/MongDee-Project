import pytest

from vision.booth_analytics import BoothAnalytics
from vision.interest.config import InterestConfig
from vision.spatial.booth import BoothLayout, BoothObject
from vision.spatial.calibration import CalibrationStore, CameraCalibration


def _calib(camera_id="CAM01"):
    return CameraCalibration.from_points(
        camera_id,
        image_points=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
        world_points=[(0, 0), (10, 0), (10, 6), (0, 6)],
    )


def _layout():
    layout = BoothLayout("BOOTH-01", width=10, length=6)
    layout.add(BoothObject("ZONE-A", "product_zone", x=5, y=3, width=3, height=3))
    return layout


def _analytics(on_event=None):
    store = CalibrationStore()
    store.set(_calib("CAM01"))
    return BoothAnalytics(
        _layout(), store,
        InterestConfig(near_distance_m=2.0, min_look_duration_sec=1.0, min_dwell_duration_sec=1.0,
                       high_interest_threshold=0.55, possible_interest_threshold=0.3, min_signals_for_high=2),
        on_interest_event=on_event,
    )


def test_observe_person_produces_world_position_and_zone():
    ba = _analytics()
    # bbox foot point ~ image (500, 600) -> world (5, 3.6) -> inside ZONE-A
    pos = ba.observe_person("CAM01", "PERSON-0001", bbox=[400, 200, 600, 600], timestamp=0.0)
    assert pos is not None
    assert pos.x == pytest.approx(5.0, abs=0.2)
    assert pos.zone_id == "ZONE-A"
    assert "PERSON-0001" in ba.positions()


def test_unknown_camera_yields_no_position():
    ba = _analytics()
    assert ba.observe_person("CAM99", "P1", [0, 0, 10, 10], 0.0) is None


def test_movement_path_and_traffic_heatmap_accumulate():
    ba = _analytics()
    for i in range(6):
        ba.observe_person("CAM01", "PERSON-0001", bbox=[400 + i * 20, 200, 600 + i * 20, 600], timestamp=float(i))
    path = ba.movement_path("PERSON-0001")
    assert path is not None and len(path.points) == 6
    assert path.total_distance() > 0
    hm = ba.heatmap("traffic")
    assert hm["kind"] == "traffic"
    assert max(max(row) for row in hm["grid"]) == pytest.approx(1.0)  # normalized


def test_interest_events_emitted_for_dwelling_person():
    events = []
    ba = _analytics(on_event=events.append)
    for i in range(8):
        ba.observe_person("CAM01", "PERSON-0001", bbox=[440, 380, 560, 600], timestamp=float(i))
    types = {e.event_type for e in events}
    assert "CUSTOMER_INTEREST_STARTED" in types
    assert "CUSTOMER_INTEREST_UPDATED" in types
    # person walks away and is later observed far from ZONE-A -> that session ends
    ba.observe_person("CAM01", "PERSON-0001", bbox=[850, 850, 950, 990], timestamp=30.0)
    assert any(e.event_type == "CUSTOMER_INTEREST_ENDED" for e in events)


def test_heatmap_bad_kind():
    with pytest.raises(ValueError, match="unknown heatmap kind"):
        _analytics().heatmap("nope")


def test_analytics_without_layout_still_maps_positions():
    store = CalibrationStore()
    store.set(_calib("CAM01"))
    ba = BoothAnalytics(None, store)
    pos = ba.observe_person("CAM01", "P1", [400, 200, 600, 600], 0.0)
    assert pos is not None
    assert ba.active_interest() == []  # no layout -> no interest sessions
