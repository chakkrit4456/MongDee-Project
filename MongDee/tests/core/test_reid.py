"""Unit tests for core/reid.py's GlobalIdentityRegistry and ReIDSampler —
all synthetic (deterministic fake embeddings, no CNN/camera needed). The
real PersonReIDEmbedder (which does load a CNN) is exercised only by
web/booth_manager.py's live integration, not here — this file tests the
matching *logic*, independent of how the embedding vectors were produced.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.reid import (
    GlobalIdentityRegistry,
    ReIDSampler,
    apply_camera_layout,
    camera_transitions_from_booth_layout,
    transition_window_from_distance,
)

# Two clearly distinct synthetic "identities" in embedding space, plus a
# small-noise variant of each to stand in for "the same person seen again,
# slightly different pose/lighting".
rng = np.random.default_rng(1234)
_BASE_A = rng.normal(size=64)
_BASE_B = rng.normal(size=64)


def _variant(base: np.ndarray, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    return base + local_rng.normal(scale=noise, size=base.shape)


# --------------------------------------------------------- 39: basic Re-ID


def test_same_appearance_different_camera_same_global_id():
    reg = GlobalIdentityRegistry()
    gid1, is_new1 = reg.resolve("CAM-1", 17, _variant(_BASE_A, seed=1), ts=100.0)
    assert is_new1 is True
    gid2, is_new2 = reg.resolve("CAM-2", 4, _variant(_BASE_A, seed=2), ts=103.0)
    assert is_new2 is False
    assert gid1 == gid2
    assert reg.unique_people_count() == 1


def test_similar_but_distinct_people_not_auto_merged():
    reg = GlobalIdentityRegistry()
    gid_a, _ = reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=1), ts=100.0)
    gid_b, is_new = reg.resolve("CAM-1", 2, _variant(_BASE_B, seed=1), ts=100.0)
    assert is_new is True
    assert gid_a != gid_b
    assert reg.unique_people_count() == 2


def test_unknown_person_not_force_matched():
    reg = GlobalIdentityRegistry()
    reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=1), ts=100.0)
    # A completely unrelated embedding must never be force-matched onto an
    # existing person just because it's the "closest" of one candidate.
    unrelated = rng.normal(size=64)
    gid, is_new = reg.resolve("CAM-1", 2, unrelated, ts=100.1)
    assert is_new is True
    assert reg.unique_people_count() == 2


# --------------------------------------------------------- 40: multi-camera


def test_multi_camera_dedup_counts_unique_people():
    reg = GlobalIdentityRegistry()
    reg.resolve("CAM-01", 1, _variant(_BASE_A, seed=1), ts=100.0)
    reg.resolve("CAM-02", 8, _variant(_BASE_A, seed=2), ts=102.0)  # same person, other camera
    assert reg.unique_people_count() == 1

    reg.resolve("CAM-02", 9, _variant(_BASE_B, seed=1), ts=102.0)  # a genuinely new person
    assert reg.unique_people_count() == 2


def test_local_track_id_never_assumed_same_person_across_cameras():
    """Spec section 6: CAM-01 Track 17 and CAM-02 Track 17 sharing a number
    is coincidence, not identity — only appearance decides."""
    reg = GlobalIdentityRegistry()
    reg.resolve("CAM-01", 17, _variant(_BASE_A, seed=1), ts=100.0)
    gid_b, is_new = reg.resolve("CAM-02", 17, _variant(_BASE_B, seed=1), ts=100.0)
    assert is_new is True
    assert reg.unique_people_count() == 2


# -------------------------------------------------- camera topology gating


def test_implausible_camera_transition_rejected_even_if_similar():
    reg = GlobalIdentityRegistry()
    reg.set_camera_transition("CAM-1", "CAM-2", min_transition_sec=5.0, max_transition_sec=15.0)
    reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=1), ts=100.0)
    # Only 1 second later on a camera that needs at least 5s to walk to —
    # appearance matches, but the timing makes it impossible, so this must
    # be treated as a different (or at least unconfirmed) person.
    gid, is_new = reg.resolve("CAM-2", 2, _variant(_BASE_A, seed=2), ts=101.0)
    assert is_new is True
    assert reg.unique_people_count() == 2


def test_plausible_camera_transition_within_topology_window_matches():
    reg = GlobalIdentityRegistry()
    reg.set_camera_transition("CAM-1", "CAM-2", min_transition_sec=1.0, max_transition_sec=10.0)
    gid1, _ = reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=1), ts=100.0)
    gid2, is_new = reg.resolve("CAM-2", 2, _variant(_BASE_A, seed=2), ts=105.0)
    assert is_new is False
    assert gid1 == gid2


# --------------------------------------------------- 13/41: track ID churn


def test_forgotten_local_track_can_rematch_same_global_person():
    """When core.tracker.PersonTracker evicts a track and later hands out a
    brand-new local ID for the same physical person, Re-ID should recover
    the *same* global identity from appearance rather than minting a new
    one (spec section 13's "รักษา identity ตาม architecture ที่มีอยู่")."""
    reg = GlobalIdentityRegistry()
    gid1, _ = reg.resolve("CAM-1", 5, _variant(_BASE_A, seed=1), ts=100.0)
    reg.forget_local_track("CAM-1", 5)  # local track evicted
    # A new local track_id (29) appears later, same physical person.
    gid2, is_new = reg.resolve("CAM-1", 29, _variant(_BASE_A, seed=3), ts=101.0)
    assert is_new is False
    assert gid1 == gid2
    assert reg.unique_people_count() == 1


def test_already_mapped_local_track_refines_not_rematches():
    reg = GlobalIdentityRegistry()
    gid1, _ = reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=1), ts=100.0)
    gid2, is_new = reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=4), ts=100.5)
    assert is_new is False
    assert gid1 == gid2
    person = reg.get_person(gid1)
    assert len(person.embeddings) == 2  # both observations kept (up to the cap)


def test_embedding_memory_capped_per_person():
    from core.reid import MAX_EMBEDDINGS_PER_PERSON
    reg = GlobalIdentityRegistry()
    gid, _ = reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=0), ts=100.0)
    for i in range(MAX_EMBEDDINGS_PER_PERSON + 5):
        reg.resolve("CAM-1", 1, _variant(_BASE_A, seed=i + 1), ts=100.0 + i)
    person = reg.get_person(gid)
    assert len(person.embeddings) == MAX_EMBEDDINGS_PER_PERSON


# --------------------------------------------------------------- ReIDSampler


def test_sampler_gates_new_track_until_min_age():
    sampler = ReIDSampler(min_track_age_sec=0.5, min_bbox_height_norm=0.1, sample_interval_sec=1.0)
    track = {"track_id": 1, "bbox": [0, 0, 100, 100], "first_seen": 100.0}
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=100.0) is False
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=100.6) is True


def test_sampler_gates_small_bbox():
    sampler = ReIDSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.2, sample_interval_sec=1.0)
    small_track = {"track_id": 1, "bbox": [0, 0, 20, 20], "first_seen": 100.0}
    assert sampler.should_embed("CAM-1", small_track, frame_height=480, now=100.0) is False


def test_sampler_throttles_repeat_embeds():
    sampler = ReIDSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.1, sample_interval_sec=1.0)
    track = {"track_id": 1, "bbox": [0, 0, 100, 200], "first_seen": 100.0}
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=100.0) is True
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=100.5) is False  # too soon
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=101.1) is True


def test_sampler_forget_clears_throttle_state():
    sampler = ReIDSampler(min_track_age_sec=0.0, min_bbox_height_norm=0.1, sample_interval_sec=1.0)
    track = {"track_id": 1, "bbox": [0, 0, 100, 200], "first_seen": 100.0}
    sampler.should_embed("CAM-1", track, frame_height=480, now=100.0)
    sampler.forget("CAM-1", 1)
    assert sampler.should_embed("CAM-1", track, frame_height=480, now=100.1) is True


# ------------------------------------- camera-layout-derived transition topology


def test_transition_window_from_distance_scales_with_distance():
    near_min, near_max = transition_window_from_distance(1.0)
    far_min, far_max = transition_window_from_distance(10.0)
    assert 0 < near_min < near_max
    assert far_min > near_min and far_max > near_max


def test_transition_window_from_zero_distance_is_near_instant():
    min_sec, max_sec = transition_window_from_distance(0.0)
    assert min_sec == 0.0
    assert max_sec > 0.0   # still allows the pause allowance, not a zero-width window


def test_camera_transitions_from_booth_layout_matches_the_example_schema():
    # Same shape as configs/booth_layout.example.json.
    layout = {
        "unit": "m",
        "objects": [
            {"id": "CAM01", "object_type": "camera", "x": 0.0, "y": 0.0,
             "metadata": {"camera_id": "CAM01"}},
            {"id": "CAM02", "object_type": "camera", "x": 9.0, "y": 0.0,
             "metadata": {"camera_id": "CAM02"}},
            {"id": "SHELF-A", "object_type": "shelf", "x": 2.5, "y": 3.0},  # not a camera -- ignored
        ],
    }
    windows = camera_transitions_from_booth_layout(layout)
    assert set(windows) == {("CAM01", "CAM02")}
    min_sec, max_sec = windows[("CAM01", "CAM02")]
    expected_min, expected_max = transition_window_from_distance(9.0)
    assert min_sec == pytest.approx(expected_min)
    assert max_sec == pytest.approx(expected_max)


def test_camera_transitions_from_booth_layout_converts_feet_to_meters():
    layout = {
        "unit": "ft",
        "objects": [
            {"id": "A", "object_type": "camera", "x": 0.0, "y": 0.0},
            {"id": "B", "object_type": "camera", "x": 10.0, "y": 0.0},
        ],
    }
    windows = camera_transitions_from_booth_layout(layout)
    min_sec, _max_sec = windows[("A", "B")]
    expected_min, _ = transition_window_from_distance(10.0 * 0.3048)
    assert min_sec == pytest.approx(expected_min)


def test_camera_transitions_from_booth_layout_with_fewer_than_two_cameras_is_empty():
    layout = {"unit": "m", "objects": [{"id": "A", "object_type": "camera", "x": 0.0, "y": 0.0}]}
    assert camera_transitions_from_booth_layout(layout) == {}
    assert camera_transitions_from_booth_layout({"objects": []}) == {}
    assert camera_transitions_from_booth_layout({}) == {}


def test_apply_camera_layout_configures_the_registry():
    reg = GlobalIdentityRegistry()
    layout = {
        "unit": "m",
        "objects": [
            {"id": "CAM-1", "object_type": "camera", "x": 0.0, "y": 0.0},
            {"id": "CAM-2", "object_type": "camera", "x": 5.0, "y": 0.0},
        ],
    }
    configured = apply_camera_layout(reg, layout)
    assert configured == 1
    expected = transition_window_from_distance(5.0)
    assert reg._transition_window("CAM-1", "CAM-2") == pytest.approx(expected)


def test_apply_camera_layout_with_no_usable_cameras_configures_nothing():
    reg = GlobalIdentityRegistry()
    assert apply_camera_layout(reg, {"objects": []}) == 0
    # unconfigured pair still falls back to the existing topology-blind default
    assert reg._transition_window("CAM-1", "CAM-2") == (0.0, 30.0)
