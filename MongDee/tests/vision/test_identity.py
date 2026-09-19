import json

import numpy as np
import pytest

from vision.attributes.extractor import PersonAttributes
from vision.identity.config import IdentityConfig
from vision.identity.global_person import GlobalPerson, TrackSummary
from vision.identity.manager import GlobalIdentityManager
from vision.identity.matcher import MATCH, NEW_PERSON, UNCERTAIN, IdentityMatcher
from vision.identity.topology import CameraTopology


def _emb(vec):
    a = np.array(vec, dtype=float)
    return a / np.linalg.norm(a)


def _summary(cam, tid, emb, t0, t1, shirt=(0, 0, 200), pants=(20, 20, 20), aspect=0.4, samples=3):
    attrs = PersonAttributes(shirt_rgb=shirt, pants_rgb=pants, aspect_ratio=aspect)
    return TrackSummary(
        camera_id=cam, local_track_id=tid, first_seen=t0, last_seen=t1,
        embedding=_emb(emb) if emb is not None else None,
        embedding_samples=samples, attributes=attrs, quality=0.7,
    )


# --- topology ------------------------------------------------------------


def test_temporal_feasibility_windows():
    topo = CameraTopology(transitions={("A", "B"): (3.0, 60.0)})
    assert topo.temporal_feasibility("A", "B", 10.0) == 1.0
    assert topo.temporal_feasibility("A", "B", -1.0) == 0.0  # impossible: arrived before leaving
    assert topo.temporal_feasibility("A", "B", 1.0) < 0.5    # too fast
    assert 0.0 < topo.temporal_feasibility("A", "B", 90.0) < 1.0  # a bit slow
    assert topo.temporal_feasibility("A", "C", 10.0) == topo.neutral_temporal  # unknown pair


def test_spatial_feasibility_graph():
    topo = CameraTopology(graph={"A": {"B"}, "B": {"A", "C"}, "C": {"B"}})
    assert topo.spatial_feasibility("A", "B") == 1.0
    assert topo.spatial_feasibility("A", "C") == 0.6  # two hops
    assert topo.spatial_feasibility("A", "D") == 0.15  # not connected
    assert CameraTopology().spatial_feasibility("A", "B") == 0.6  # no graph -> neutral


def test_topology_load(tmp_path):
    path = tmp_path / "topo.json"
    path.write_text(json.dumps({
        "transitions": [{"from": "CAM01", "to": "CAM02", "min_seconds": 3, "max_seconds": 60}],
        "graph": {"CAM01": ["CAM02"]},
        "locations": {"CAM01": "Lobby"},
    }), encoding="utf-8")
    topo = CameraTopology.load(path)
    assert topo.transitions[("CAM01", "CAM02")] == (3.0, 60.0)
    assert topo.location("CAM01") == "Lobby"
    assert topo.spatial_feasibility("CAM01", "CAM02") == 1.0


# --- matcher ------------------------------------------------------------


def test_matcher_scores_similar_high_dissimilar_low():
    matcher = IdentityMatcher()
    person = GlobalPerson("PERSON-0001", _summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))

    similar = _summary("CAM02", 5, [0.98, 0.1, 0.0], 15.0, 25.0)
    s_sim, _ = matcher.score(person, similar)
    assert s_sim > 0.7

    different = _summary("CAM02", 6, [0, 1, 0], 15.0, 25.0, shirt=(0, 200, 0), pants=(200, 200, 0), aspect=0.8)
    s_diff, _ = matcher.score(person, different)
    assert s_diff < 0.5


def test_matcher_single_signal_cannot_reach_match():
    cfg = IdentityConfig()
    matcher = IdentityMatcher(cfg)
    person = GlobalPerson("PERSON-0001", _summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    # only Re-ID available: no colours, no aspect
    only_reid = _summary("CAM02", 5, [1, 0, 0], 15.0, 25.0, shirt=(0, 0, 0), pants=(0, 0, 0), aspect=0.0)
    score, _ = matcher.score(person, only_reid)
    assert score <= cfg.match_threshold - 0.01


def test_matcher_hard_temporal_reject():
    matcher = IdentityMatcher(IdentityConfig(hard_temporal_reject=True))
    person = GlobalPerson("PERSON-0001", _summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    impossible = _summary("CAM02", 5, [1, 0, 0], 5.0, 8.0)  # seen at CAM02 before leaving CAM01
    score, breakdown = matcher.score(person, impossible)
    assert score == 0.0
    assert breakdown.get("rejected") == 1.0


# --- manager ----------------------------------------------------------


def test_first_track_creates_person():
    mgr = GlobalIdentityManager()
    gid, result = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    assert gid == "PERSON-0001"
    assert result.status == NEW_PERSON
    assert mgr.unique_count() == 1


def test_same_person_across_cameras_is_merged():
    mgr = GlobalIdentityManager()
    gid1, _ = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    gid2, result = mgr.submit_track(_summary("CAM02", 7, [0.97, 0.05, 0.0], 14.0, 24.0))
    assert result.status == MATCH
    assert gid2 == gid1
    assert mgr.unique_count() == 1
    person = mgr.get(gid1)
    assert set(person.cameras_seen) == {"CAM01", "CAM02"}


def test_different_people_stay_separate():
    mgr = GlobalIdentityManager()
    mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0, shirt=(0, 0, 200)))
    _gid, result = mgr.submit_track(
        _summary("CAM02", 7, [0, 1, 0], 14.0, 24.0, shirt=(0, 200, 0), pants=(200, 200, 0), aspect=0.75)
    )
    assert result.status == NEW_PERSON
    assert mgr.unique_count() == 2


def test_uncertain_creates_new_person_with_link():
    # tuned so the score lands in the uncertain band: decent Re-ID, mismatched clothing
    cfg = IdentityConfig(match_threshold=0.8, uncertain_threshold=0.45)
    mgr = GlobalIdentityManager(cfg)
    g1, _ = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0, shirt=(0, 0, 200), pants=(20, 20, 20)))
    g2, result = mgr.submit_track(
        _summary("CAM02", 7, [0.8, 0.3, 0.1], 14.0, 24.0, shirt=(0, 120, 120), pants=(120, 120, 0), aspect=0.5)
    )
    assert result.status == UNCERTAIN
    assert g2 != g1
    assert mgr.unique_count() == 2
    assert g1 in mgr.get(g2).uncertain_links
    assert g2 in mgr.get(g1).uncertain_links


def test_resubmitting_same_track_merges_not_duplicates():
    mgr = GlobalIdentityManager()
    gid1, _ = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    gid2, result = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 10.0, 20.0))
    assert gid2 == gid1
    assert result.breakdown.get("same_track") == 1.0
    assert mgr.unique_count() == 1


def test_ttl_prunes_stale_persons():
    mgr = GlobalIdentityManager(IdentityConfig(person_ttl_sec=100.0))
    mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    # a much later track: the first person is now stale
    mgr.submit_track(_summary("CAM02", 2, [0, 1, 0], 500.0, 510.0, shirt=(0, 200, 0), aspect=0.7))
    assert mgr.unique_count() == 1


def test_camera_person_counts():
    mgr = GlobalIdentityManager()
    g1, _ = mgr.submit_track(_summary("CAM01", 1, [1, 0, 0], 0.0, 10.0))
    mgr.submit_track(_summary("CAM02", 7, [0.97, 0.05, 0.0], 14.0, 24.0))  # merges into g1
    mgr.submit_track(_summary("CAM01", 2, [0, 1, 0], 30.0, 40.0, shirt=(0, 200, 0), aspect=0.7))  # new
    counts = mgr.camera_person_counts()
    assert counts["CAM01"] == 2
    assert counts["CAM02"] == 1
