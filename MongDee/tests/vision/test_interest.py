import math

import pytest

from vision.interest.association import AssociationEngine
from vision.interest.config import InterestConfig
from vision.interest.estimator import HIGH, LOW, POSSIBLE, UNKNOWN, InterestObservation, InterestSession
from vision.interest.gaze import HeadPose, HeadPoseEstimator, body_orientation_from_motion, facing_score
from vision.interest.interaction import (
    CONFIRMED_INTERACTION,
    POSSIBLE_INTERACTION,
    InteractionDetector,
)
from vision.interest.interaction import UNKNOWN as I_UNKNOWN
from vision.spatial.booth import BoothLayout, BoothObject


# --- gaze / head pose --------------------------------------------------


def test_headpose_without_backend_is_unknown():
    hp = HeadPoseEstimator().estimate(None)
    assert not hp.is_known
    assert hp.yaw is None


def test_body_orientation_from_motion():
    assert body_orientation_from_motion((0.0, 0.0)) is None
    assert body_orientation_from_motion((1.0, 0.0)) == pytest.approx(0.0)
    assert body_orientation_from_motion((0.0, 1.0)) == pytest.approx(math.pi / 2)


def test_facing_score():
    # person at origin heading +x, target on +x axis -> facing it
    assert facing_score((0, 0), 0.0, (5, 0)) == pytest.approx(1.0)
    # heading +x, target behind on -x -> not facing
    assert facing_score((0, 0), 0.0, (-5, 0)) == pytest.approx(0.0)
    assert facing_score((0, 0), None, (5, 0)) is None


# --- interaction --------------------------------------------------


def test_interaction_contact_only_is_possible():
    det = InteractionDetector(confirm_after_frames=3, move_threshold_px=15)
    st = det.observe(1, 10, person_bbox=[0, 0, 100, 200], product_bbox=[80, 80, 120, 120])
    assert st.status == POSSIBLE_INTERACTION


def test_interaction_contact_plus_movement_confirms():
    det = InteractionDetector(confirm_after_frames=2, move_threshold_px=10)
    for _ in range(3):
        det.observe(1, 10, [0, 0, 100, 200], [80, 80, 120, 120])
    st = det.observe(1, 10, [0, 0, 100, 200], [130, 80, 170, 120])  # product moved ~50px
    assert st.status == CONFIRMED_INTERACTION


def test_interaction_no_contact_is_unknown():
    det = InteractionDetector()
    st = det.observe(1, 10, [0, 0, 50, 100], [500, 500, 540, 540])
    assert st.status == I_UNKNOWN


# --- interest session -------------------------------------------


def _cfg():
    return InterestConfig(near_distance_m=1.5, min_look_duration_sec=1.0, min_dwell_duration_sec=1.0,
                          high_interest_threshold=0.6, possible_interest_threshold=0.3, min_signals_for_high=3)


def test_short_glance_is_not_interest():
    s = InterestSession("P1", "ZONE-A", _cfg())
    s.observe(InterestObservation(timestamp=0.0, distance_m=1.0, speed_mps=1.2, approaching=True))
    s.observe(InterestObservation(timestamp=0.3, distance_m=1.4, speed_mps=1.2, approaching=False))
    res = s.result()
    assert res.look_duration < 1.0
    assert res.status in (LOW, UNKNOWN)


def test_sustained_engagement_is_high_interest():
    s = InterestSession("P1", "ZONE-A", _cfg())
    for i in range(8):
        s.observe(InterestObservation(
            timestamp=float(i), distance_m=0.6, speed_mps=0.1, approaching=(i < 2),
            facing_score=0.9, gaze_score=0.85, interacting=(i >= 6),
        ))
    latest = InterestObservation(timestamp=8.0, distance_m=0.6, speed_mps=0.1, approaching=False,
                                 facing_score=0.9, gaze_score=0.85, interacting=True)
    res = s.result(latest)
    assert res.look_duration >= 1.0
    assert res.dwell_duration >= 1.0
    assert len(res.signals_used) >= 3
    assert res.status == HIGH


def test_high_interest_needs_multiple_signals():
    s = InterestSession("P1", "ZONE-A", _cfg())
    # only one signal: a long look, nothing else
    for i in range(12):
        s.observe(InterestObservation(timestamp=float(i), distance_m=1.0, speed_mps=1.0, approaching=False))
    res = s.result()
    assert res.status != HIGH  # capped: too few independent signals


def test_session_staleness():
    s = InterestSession("P1", "ZONE-A", _cfg())
    s.observe(InterestObservation(timestamp=0.0, distance_m=1.0, speed_mps=0.1, approaching=False))
    assert not s.is_stale(2.0)
    assert s.is_stale(10.0)


# --- association -----------------------------------------------


def _layout():
    layout = BoothLayout("BOOTH-01", width=10, length=6)
    layout.add(BoothObject("ZONE-A", "product_zone", x=2, y=2, width=1.5, height=1.5))
    layout.add(BoothObject("ZONE-B", "product_zone", x=8, y=5, width=1.5, height=1.5))
    return layout


def test_association_prefers_nearer_zone():
    engine = AssociationEngine(_layout(), _cfg())
    engine.associate("P1", 2.0, 3.5, timestamp=0.0)          # establish prev
    assocs = engine.associate("P1", 2.2, 2.9, timestamp=1.0)  # walked toward ZONE-A
    assert assocs
    assert assocs[0].target_id == "ZONE-A"
    assert assocs[0].approaching
    assert assocs[0].confidence > 0


def test_association_none_when_far_from_everything():
    engine = AssociationEngine(_layout(), _cfg())
    assert engine.associate("P1", 5.0, 0.5, timestamp=0.0) == []
