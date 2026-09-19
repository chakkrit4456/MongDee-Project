import numpy as np
from core.tracker_v2 import TrackerV2, iou_matrix
from core.mot_sim import make_scene, evaluate, GreedyBaseline


def test_iou_basic():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1 / 3) < 1e-6 and m[0, 1] == 0


def test_single_frame_false_positive_never_gets_id():
    tr = TrackerV2(min_hits=2)
    vis, _ = tr.update([[10, 10, 50, 90]], [None], [0.9], now=0.0)
    assert vis == []
    vis, _ = tr.update([], None, [], now=0.5)
    assert vis == [] and tr._tracks == []


def test_confirmed_after_min_hits_and_id_stable():
    tr = TrackerV2(min_hits=2)
    ids = set()
    for k in range(10):
        vis, _ = tr.update([[10 + k * 3, 10, 50 + k * 3, 90]], ["male"], [0.9], now=k * 0.4)
        ids |= {v["track_id"] for v in vis}
    assert len(ids) == 1


def test_survives_occlusion_gap_and_keeps_id():
    tr = TrackerV2(min_hits=2, max_age_sec=2.0)
    ids = []
    t = 0.0
    for k in range(6):
        vis, _ = tr.update([[10 + k * 5, 10, 50 + k * 5, 90]], None, [0.9], now=t); t += 0.3
    first = vis[0]["track_id"]
    for _ in range(3):  # 0.9 s no detections
        tr.update([], None, [], now=t); t += 0.3
    vis, _ = tr.update([[10 + 9 * 5, 10, 50 + 9 * 5, 90]], None, [0.9], now=t)
    assert vis and vis[0]["track_id"] == first


def test_eviction_reported_after_max_age():
    tr = TrackerV2(min_hits=1, max_age_sec=1.0)
    tr.update([[0, 0, 40, 80]], None, [0.9], now=0.0)
    ev_all = []
    for k in range(1, 6):
        _, ev = tr.update([], None, [], now=k * 0.5); ev_all += ev
    assert len(ev_all) == 1 and ev_all[0]["track_id"] == 1


def test_low_conf_detection_rescues_track():
    tr = TrackerV2(min_hits=2)
    for k in range(4):
        tr.update([[10, 10, 50, 90]], None, [0.9], now=k * 0.3)
    vis, _ = tr.update([[11, 10, 51, 90]], None, [0.2], now=1.2)
    assert len(vis) == 1 and vis[0]["track_id"] == 1


def test_category_vote_resists_single_flip():
    tr = TrackerV2(min_hits=1)
    cats = ["male"] * 6 + ["female"] + ["male"] * 3
    for k, c in enumerate(cats):
        vis, _ = tr.update([[10, 10, 50, 90]], [c], [0.9], now=k * 0.3)
    assert vis[0]["category"] == "male"


def test_two_crossing_people_no_swap():
    tr = TrackerV2(min_hits=2)
    first = {}
    for k in range(30):
        a = [10 + k * 8, 20, 60 + k * 8, 120]      # moves right
        b = [250 - k * 8, 20, 300 - k * 8, 120]    # moves left
        vis, _ = tr.update([a, b], None, [0.9, 0.9], now=k * 0.2)
        if k == 3:
            first = {v["track_id"]: v["bbox"][0] for v in vis}
    left_id = min(first, key=first.get); right_id = max(first, key=first.get)
    final = {v["track_id"]: v["bbox"][0] for v in vis}
    # after crossing, the track that started on the left must now be on the right
    assert final[left_id] > final[right_id]


def _bench(fast, n=12):
    new = dict(idsw=0, idf1=0.0); old = dict(idsw=0, idf1=0.0)
    for seed in range(n):
        sc = make_scene(seed=seed, n_people=3, det_hz=3.0, fast=fast)
        a = evaluate(sc, lambda: TrackerV2()); b = evaluate(sc, lambda: GreedyBaseline())
        for k in new: new[k] += a[k]; old[k] += b[k]
    new["idf1"] /= n; old["idf1"] /= n
    return new, old


def test_synthetic_walking_beats_greedy_baseline():
    new, old = _bench(fast=False)
    print("walking new", new, "old", old)
    assert new["idf1"] > old["idf1"] + 0.10
    assert new["idsw"] < old["idsw"]
    assert new["idf1"] >= 0.85


def test_synthetic_fast_motion_not_worse_than_baseline():
    # 40-100 px/s at 3 Hz with wall bounces: hard for any IoU tracker; only require improvement.
    new, old = _bench(fast=True)
    print("fast new", new, "old", old)
    assert new["idf1"] > old["idf1"]
    assert new["idsw"] < old["idsw"]
