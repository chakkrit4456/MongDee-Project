"""Unit tests for core/tracker.py's per-track category (gender/age) voting."""

from __future__ import annotations

from core.tracker import PersonTracker


def test_no_categories_passed_reports_unknown():
    tracker = PersonTracker()
    visible, _ = tracker.update([[0, 0, 10, 10]])
    assert visible[0]["category"] == "unknown"


def test_single_category_is_reported():
    tracker = PersonTracker()
    visible, _ = tracker.update([[0, 0, 10, 10]], ["male"])
    assert visible[0]["category"] == "male"


def test_majority_category_wins_over_noise():
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    # Same track (same box, high IoU with itself) seen across several
    # frames — mostly "female", with a couple of "unknown" frames noise.
    for cat in ["female", "unknown", "female", "female", "unknown"]:
        visible, _ = tracker.update([box], [cat])
    assert visible[0]["category"] == "female"


def test_unknown_frames_never_outvote_a_confident_category():
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    # One single confident "male" frame plus many "unknown" ones — a plain
    # mode would say "unknown"; the tracker should still report "male"
    # since any confident guess outranks inconclusive noise (see
    # core/tracker.py's _mode_category).
    for cat in ["unknown", "unknown", "male", "unknown", "unknown"]:
        visible, _ = tracker.update([box], [cat])
    assert visible[0]["category"] == "male"


def test_evicted_track_carries_its_final_category():
    import time

    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    tracker.update([box], ["child"])
    # Force eviction by not re-matching the box and manually aging the
    # track past TRACK_MAX_AGE_SEC (avoids a real sleep in the test).
    track_id = next(iter(tracker._tracks))
    tracker._tracks[track_id]["last_seen"] -= 10
    _, evicted = tracker.update([], [])
    assert len(evicted) == 1
    assert evicted[0]["track_id"] == track_id
    assert evicted[0]["category"] == "child"


def test_confidences_omitted_falls_back_to_unweighted_counting():
    # Same scenario as test_majority_category_wins_over_noise, but this
    # time confirms the *default* (no person_confidences passed at all)
    # behaves identically to before this parameter existed.
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    for cat in ["female", "unknown", "female", "female", "unknown"]:
        visible, _ = tracker.update([box], [cat])
    assert visible[0]["category"] == "female"


def test_one_high_confidence_vote_outweighs_several_low_confidence_votes():
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    # Three low-confidence "male" frames (0.15 each, sum 0.45) followed by
    # one high-confidence "female" frame (0.9) — a flat unweighted count
    # would report "male" (3 votes > 1 vote); confidence-weighted voting
    # should report "female" since its single vote outweighs the sum of
    # the three low-confidence ones.
    for cat, conf in [("male", 0.15), ("male", 0.15), ("male", 0.15), ("female", 0.9)]:
        visible, _ = tracker.update([box], [cat], [conf])
    assert visible[0]["category"] == "female"


def test_confidence_weighted_voting_still_ignores_unknown_noise():
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    for cat, conf in [("unknown", 0.0), ("unknown", 0.0), ("male", 0.7), ("unknown", 0.0)]:
        visible, _ = tracker.update([box], [cat], [conf])
    assert visible[0]["category"] == "male"


def test_stale_category_reverts_to_unknown_while_track_stays_matched():
    # Real bug caught live: a person who turns their back while remaining
    # continuously tracked (their body silhouette still matches the same
    # track every frame, so it never ages out via TRACK_MAX_AGE_SEC) kept
    # showing their last confidently-seen category indefinitely. Simulates
    # "a long time has passed with no confident guess" the same way
    # test_evicted_track_carries_its_final_category simulates aging, by
    # directly rewinding the recorded timestamp rather than sleeping.
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    visible, _ = tracker.update([box], ["female"])
    assert visible[0]["category"] == "female"

    track_id = next(iter(tracker._tracks))
    tracker._tracks[track_id]["last_confident_category_ts"] -= 100  # far past the grace period

    # Box keeps matching (still "seen"), but with no category info this
    # frame (e.g. face no longer visible) -- category must now read
    # 'unknown', not the stale 'female'.
    visible, _ = tracker.update([box], ["unknown"])
    assert visible[0]["category"] == "unknown"


def test_stale_category_votes_are_cleared_not_just_hidden():
    # After expiry, a single new low-weight guess must be judged on its
    # own, not instantly overpowered by the old vote sum still sitting in
    # category_votes.
    tracker = PersonTracker()
    box = [0, 0, 10, 10]
    tracker.update([box], ["female"], [0.95])
    track_id = next(iter(tracker._tracks))
    tracker._tracks[track_id]["last_confident_category_ts"] -= 100

    # One weak "male" guess after expiry -- if the old female vote (0.95)
    # were still present, "male" (0.3) would lose; it must win instead.
    visible, _ = tracker.update([box], ["male"], [0.3])
    assert visible[0]["category"] == "male"
