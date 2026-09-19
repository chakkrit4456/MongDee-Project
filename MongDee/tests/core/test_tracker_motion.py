"""PersonTracker: motion-compensated matching vs the old greedy last-box IoU matching.

The AI runs every 0.2-1 s, so a walking person moves a large fraction of their own width between
two passes. use_motion=False is the exact legacy behaviour, so these are real A/B tests of the
production class (not a re-implementation)."""
import numpy as np

from core.mot_sim import evaluate, make_scene
from core.tracker import PersonTracker


def test_person_walking_faster_than_iou_gate_keeps_one_id_with_motion_only():
    # 50px-wide person; after a short ramp-up they walk 35 px per AI pass (0.5 s): the last-box IoU
    # (15/85 = 0.18) is under the 0.25 gate, so the legacy matcher loses them every pass
    xs = [0, 20, 45, 75, 110, 145, 180, 215, 250, 285, 320, 355]
    ids = {True: set(), False: set()}
    for use_motion in (True, False):
        tr = PersonTracker(use_motion=use_motion)
        for k, x in enumerate(xs):
            visible, _ = tr.update([[x, 20, x + 50, 140]], now=k * 0.5)
            ids[use_motion] |= {v["track_id"] for v in visible}
    assert len(ids[True]) == 1
    assert len(ids[False]) > 1          # the bug: ID churn for a normally walking person


class _YoloLike:
    """Feeds the tracker what production feeds it: only boxes above YOLO's conf threshold."""
    def __init__(self, **kw):
        self.t = PersonTracker(**kw)

    def update(self, boxes, cats=None, confs=None, now=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        keep = [i for i, c in enumerate(confs) if c >= 0.45]
        return self.t.update([list(boxes[i]) for i in keep], None, None, now=now)


def _bench(use_motion, hz, fast, n=12):
    idsw = 0; idf1 = 0.0
    for seed in range(n):
        r = evaluate(make_scene(seed=seed, det_hz=hz, fast=fast), lambda: _YoloLike(use_motion=use_motion))
        idsw += r["idsw"]; idf1 += r["idf1"]
    return idsw, idf1 / n


def test_synthetic_walking_scenes_fewer_id_switches_and_no_worse_idf1():
    for hz in (2.0, 3.0, 5.0):
        old, new = _bench(False, hz, False), _bench(True, hz, False)
        print(f"walk {hz}Hz old(idsw,idf1)={old} new={new}")
        assert new[0] < old[0] and new[1] >= old[1]


def test_synthetic_fast_scenes_improve_but_remain_hard():
    for hz in (3.0, 5.0):
        old, new = _bench(False, hz, True), _bench(True, hz, True)
        print(f"fast {hz}Hz old(idsw,idf1)={old} new={new}")
        assert new[0] < old[0] and new[1] > old[1]


def test_legacy_mode_and_categories_unchanged_api():
    tr = PersonTracker(use_motion=False)
    visible, evicted = tr.update([[0, 0, 40, 80]], ["male"], [0.9], now=0.0)
    assert visible[0]["category"] == "male" and evicted == []


# --------------------------------------------- category evidence gate (man labelled as woman)
def _category_error_rate(kwargs, n_frames, tracks=400, acc=0.82, seed=1):
    rng = np.random.default_rng(seed)
    wrong = unknown = 0
    for i in range(tracks):
        male = bool(i % 2)
        tr = PersonTracker(**kwargs)
        for f in range(n_frames):
            correct = rng.random() < acc
            conf = rng.uniform(0.7, 0.95)   # frames below MIN_GENDER_CONFIDENCE are never voted
            cat = ("male" if male else "female") if correct else ("female" if male else "male")
            visible, _ = tr.update([[10, 10, 50, 90]], [cat], [conf], now=f * 0.4)
        cat = visible[0]["category"]
        unknown += cat == "unknown"
        wrong += cat not in ("unknown", "male" if male else "female")
    return wrong / tracks, unknown / tracks


def test_evidence_gate_turns_single_bad_frames_into_unknown_and_lowers_long_run_error():
    gate = dict(category_min_evidence=1.2, category_margin=0.6)
    old1, new1 = _category_error_rate({}, 1), _category_error_rate(gate, 1)
    old6, new6 = _category_error_rate({}, 6), _category_error_rate(gate, 6)
    print("1 frame old/new (wrong,unknown):", old1, new1, "| 6 frames:", old6, new6)
    assert old1[0] > 0.10 and new1[0] == 0.0 and new1[1] == 1.0        # never a wrong label from 1 frame
    assert new6[0] < old6[0] and new6[1] < 0.15


def test_defaults_keep_the_original_plurality_behaviour():
    tr = PersonTracker()
    visible, _ = tr.update([[0, 0, 40, 80]], ["female"], [0.65], now=0.0)
    assert visible[0]["category"] == "female"


def test_production_gate_constants_never_label_from_one_frame_and_are_accurate_after_a_few():
    from core.vision import PERSON_CATEGORY_MARGIN, PERSON_CATEGORY_MIN_EVIDENCE
    gate = dict(category_min_evidence=PERSON_CATEGORY_MIN_EVIDENCE, category_margin=PERSON_CATEGORY_MARGIN)
    one = _category_error_rate(gate, 1)
    six = _category_error_rate(gate, 6)
    ten = _category_error_rate(gate, 10)
    print("production gate: 1 frame", one, "6 frames", six, "10 frames", ten)
    assert one[0] == 0.0 and one[1] == 1.0
    assert six[0] < 0.02
    assert ten[0] < 0.01 and ten[1] < 0.10


# --------------------------------------------- anti-flicker: two-threshold rescue, tentative tracks, coasting
def _box(x):
    return [x, 20.0, x + 40.0, 100.0]


def test_low_score_detection_keeps_the_same_track_instead_of_blinking_out():
    tr = PersonTracker(min_hits=2, coast_sec=0.6)
    v, _ = tr.update([_box(10)], ["unknown"], [0.0], now=0.0, detection_scores=[0.9], high_score=0.45)
    tid = v[0]["track_id"]
    # Confidence dips below the high threshold for several passes: still the same track, still visible.
    for i in range(1, 6):
        v, _ = tr.update([_box(10 + i)], ["unknown"], [0.0], now=i * 0.3, detection_scores=[0.3], high_score=0.45)
        assert [t["track_id"] for t in v] == [tid]


def test_low_score_detection_never_starts_a_track_or_a_visible_person():
    tr = PersonTracker(min_hits=1)
    v, _ = tr.update([_box(10)], ["unknown"], [0.0], now=0.0, detection_scores=[0.3], high_score=0.45)
    assert v == []
    v, _ = tr.update([_box(10)], ["unknown"], [0.0], now=0.3, detection_scores=[0.3], high_score=0.45)
    assert v == []


def test_one_frame_false_detection_is_never_reported_but_a_confident_one_is_immediate():
    tr = PersonTracker(min_hits=2, birth_immediate_score=0.7)
    v, ev = tr.update([_box(10)], ["unknown"], [0.0], now=0.0, detection_scores=[0.5], high_score=0.45)
    assert v == []
    v, ev = tr.update([], [], [], now=0.3, detection_scores=[], high_score=0.45)
    assert v == [] and ev == []                        # vanished: dropped silently, no phantom eviction
    v, _ = tr.update([_box(100)], ["unknown"], [0.0], now=0.6, detection_scores=[0.9], high_score=0.45)
    assert len(v) == 1                                  # confident birth is trusted at once
    tr2 = PersonTracker(min_hits=2)
    tr2.update([_box(10)], ["unknown"], [0.0], now=0.0, detection_scores=[0.5], high_score=0.45)
    v, _ = tr2.update([_box(12)], ["unknown"], [0.0], now=0.3, detection_scores=[0.5], high_score=0.45)
    assert len(v) == 1                                  # seen twice -> confirmed


def test_coasting_reports_the_predicted_box_briefly_then_stops():
    tr = PersonTracker(coast_sec=0.6)
    for i in range(4):
        tr.update([_box(10 + 10 * i)], ["unknown"], [0.0], now=i * 0.3)
    tr.update([], [], [], now=1.2)                       # detector missed this pass
    c = tr.coasting_tracks()
    assert len(c) == 1 and c[0]["bbox"][0] > 40          # keeps moving along the same direction
    tr.update([], [], [], now=2.0)
    assert tr.coasting_tracks() == []                     # gap 1.1 s > coast_sec


def test_defaults_are_unchanged_by_the_flicker_options():
    tr = PersonTracker()
    v, _ = tr.update([_box(10)], ["male"], [0.9], now=0.0)
    assert len(v) == 1 and tr.coasting_tracks() == []
