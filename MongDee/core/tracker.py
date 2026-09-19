"""Lightweight per-camera person tracker — ephemeral IDs only.

Not re-identification: a track ID persists only while a person stays
continuously visible (matched frame-to-frame) in one camera's feed. Leaving
the frame and coming back always starts a new ID, and the same physical
person seen by two different cameras gets two independent IDs. This is a
deliberate, documented limitation (there is no face/person re-id in this
codebase) — good enough for "how many people are on screen right now" and
"which track picked up this product", not for cross-camera or cross-visit
identity.

Matching: each track's box is first PREDICTED forward with a constant-velocity
Kalman filter (core.tracker_v2.KalmanBox) to where the person should be now, then
detections are assigned to tracks with the Hungarian algorithm on IoU. The AI runs
at a low, uneven cadence (0.2-1 s between passes, shared across cameras), and a
person walking at normal speed moves a large fraction of their own width in that
time - plain "IoU against the last box" then drops the match and hands out a new
ID (fragmented counts, lost product-holder links, gender re-guessed from scratch).
The old greedy/no-motion behaviour is still available as PersonTracker(use_motion=
False) and is what tests/core/test_tracker_motion.py benchmarks against.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from core.tracker_v2 import KalmanBox, iou_matrix, linear_sum_assignment

TRACK_MAX_AGE_SEC = 1.5   # wall-clock: inference cadence varies with cross-camera lock contention
MIN_MATCH_IOU = 0.25
MOTION_MIN_MATCH_IOU = 0.20   # gate for the Kalman-predicted path (0.2 vs 0.25: fewer ID switches
                              # on fast walkers in tests/core/test_tracker_motion.py; 0.15 and below
                              # started adding switches in the slow scenes)

# Found via a live camera test (the same gap independently existed here and
# in core.attributes.GlobalPersonAttributeSmoother, see
# ATTRIBUTE_STALE_GRACE_SEC): a person who turns away while staying
# continuously tracked (their body silhouette still matches the same track
# frame-to-frame, so the track never ages out via TRACK_MAX_AGE_SEC) kept
# showing their last confidently-seen category indefinitely, since
# category_votes never decayed. If a track hasn't received a real
# non-'unknown' guess within this long, its accumulated vote is now treated
# as stale and reported as 'unknown' instead -- long enough to survive a
# normal brief head-turn without flicker, short enough to stop showing a
# guess that's no longer backed by any recent evidence. Same
# evidence-informed-judgment-call caveat as that constant.
CATEGORY_STALE_GRACE_SEC = 8.0


def _mode_category(votes: Counter, last_confident_ts: float, now: float,
                   min_evidence: float = 0.0, margin: float = 0.0) -> str:
    """Highest-weight non-'unknown' category seen for a track, or 'unknown'
    if every guess (or none at all) was inconclusive, or if the newest
    confident guess is older than CATEGORY_STALE_GRACE_SEC. A plain mode
    across every frame would report 'unknown' for most people — a side
    profile, motion blur, or just being far from the camera often reads as
    inconclusive on a given frame even when the confident frames agree — so
    any category that was ever confidently seen outranks 'unknown' noise,
    as long as that evidence isn't stale.

    votes holds a running weight sum per category, not a plain frame count:
    PersonTracker.update()'s optional person_confidences lets each frame's
    guess contribute its own classifier confidence instead of a flat +1, so
    one high-confidence frame can outweigh several low-confidence ones of a
    different category (see PersonTracker.update's docstring). Callers that
    never pass confidences get the old flat +1-per-frame behavior for free,
    since weight defaults to 1.0."""
    if now - last_confident_ts > CATEGORY_STALE_GRACE_SEC:
        return "unknown"
    real_votes = {k: v for k, v in votes.items() if k != "unknown"}
    if not real_votes:
        return "unknown"
    ranked = sorted(real_votes.items(), key=lambda kv: kv[1], reverse=True)
    best_category, best_weight = ranked[0]
    # Optional evidence gate (both default to 0 = the original "plurality wins" behaviour): a
    # label needs a minimum total classifier weight AND a clear lead over the runner-up,
    # otherwise the honest answer is "unknown". A single confident-looking wrong frame from a
    # tiny 320x240 face then can't become a man's label ("wrong is worse than none").
    if best_weight < min_evidence:
        return "unknown"
    if len(ranked) > 1 and (best_weight - ranked[1][1]) < margin:
        return "unknown"
    return best_category


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """One instance per camera. Call update() once per inference frame, even
    with an empty box list, so stale tracks age out promptly."""

    def __init__(self, use_motion: bool = True, category_min_evidence: float = 0.0,
                 category_margin: float = 0.0, min_hits: int = 1, birth_immediate_score: float = 0.7,
                 coast_sec: float = 0.0, max_age_sec: float = TRACK_MAX_AGE_SEC):
        # Anti-flicker options (all off by default = the original behaviour):
        #   min_hits              a new track stays invisible ("tentative") until it has been matched
        #                         this many passes, so a one-frame false detection never becomes a
        #                         person, a Re-ID identity or a presence session;
        #   birth_immediate_score a detection at least this confident is trusted at once;
        #   coast_sec             how long an unmatched track is still reported by coasting_tracks()
        #                         (its motion-predicted box) so the on-screen box does not blink;
        #   max_age_sec           how long an unmatched confirmed track survives before eviction.
        # Detections with a low score (update(detection_scores=...) below the caller's own high
        # threshold) may only extend an existing track, never start one (ByteTrack-style rescue).
        self._min_hits = max(1, int(min_hits))
        self._birth_immediate_score = birth_immediate_score
        self._coast_sec = coast_sec
        self._max_age_sec = max_age_sec
        self._last_update_ts = 0.0
        # track_id -> {"bbox", "first_seen", "last_seen", "category_votes",
        # "last_confident_category_ts"}. category_votes is a Counter over
        # every 'female'/'male'/'unknown' guess seen for this track
        # (see core/vision.py's _classify_person) — a per-frame guess is
        # noisy, so the track's reported category is the mode across its
        # whole lifetime, not just its latest frame.
        # last_confident_category_ts is the wall-clock time of this track's
        # most recent non-'unknown' guess (see CATEGORY_STALE_GRACE_SEC) --
        # distinct from "last_seen", which updates on every matched frame
        # regardless of whether that frame's guess was confident.
        self._tracks: dict[int, dict] = {}
        self._next_id = 1
        self._use_motion = use_motion
        self._category_min_evidence = category_min_evidence
        self._category_margin = category_margin

    def update(self, person_boxes: list[list[float]],
               person_categories: list[str] | None = None,
               person_confidences: list[float] | None = None,
               now: float | None = None,
               detection_scores: list[float] | None = None,
               high_score: float = 0.0) -> tuple[list[dict], list[dict]]:
        """Returns (visible_tracks, evicted_tracks).

        person_categories, if given, must be the same length as
        person_boxes — one 'female'/'male'/'unknown' guess per box,
        from the same frame. Omit it (or pass None) to track boxes without
        recording any category, e.g. when no gender_age_backend is
        configured.

        person_confidences, if given, must also be the same length as
        person_boxes — the classifier confidence backing each entry in
        person_categories (see core/vision.py's _classify_person), used to
        weight that frame's vote instead of counting it as a flat +1. Omit
        it (or pass None) to fall back to the original unweighted
        behavior — every existing caller that doesn't pass it is
        unaffected.

        visible_tracks: one entry per input box —
          {"track_id", "bbox", "first_seen", "last_seen", "category"}
        evicted_tracks: tracks not re-matched for TRACK_MAX_AGE_SEC, one-shot —
          {"track_id", "first_seen", "last_seen", "category"}
        category on both is the mode of every guess seen for that track so
        far ("unknown" if none were ever categorized).
        """
        now = time.time() if now is None else float(now)
        self._last_update_ts = now
        scores = detection_scores if detection_scores is not None else [1.0] * len(person_boxes)
        is_low = [s < high_score for s in scores]

        matched_tracks: set[int] = set()
        matched_boxes: set[int] = set()
        box_to_track: dict[int, int] = {}
        if self._use_motion:
            self._match_with_motion(person_boxes, now, matched_tracks, matched_boxes, box_to_track, is_low)
        else:
            pairs = []
            for track_id, track in self._tracks.items():
                for box_idx, box in enumerate(person_boxes):
                    score = _iou(track["bbox"], box)
                    if score >= MIN_MATCH_IOU:
                        pairs.append((score, track_id, box_idx))
            pairs.sort(key=lambda p: p[0], reverse=True)
            for _score, track_id, box_idx in pairs:
                if track_id in matched_tracks or box_idx in matched_boxes:
                    continue
                matched_tracks.add(track_id)
                matched_boxes.add(box_idx)
                box_to_track[box_idx] = track_id

        born: set[int] = set()
        visible_tracks = []
        for box_idx, box in enumerate(person_boxes):
            track_id = box_to_track.get(box_idx)
            if track_id is None:
                if is_low[box_idx]:
                    continue      # a low-score detection never starts a track
                track_id = self._next_id
                self._next_id += 1
                born.add(track_id)
                self._tracks[track_id] = {
                    "bbox": box, "first_seen": now, "last_seen": now, "hits": 1,
                    "confirmed": self._min_hits <= 1 or scores[box_idx] >= self._birth_immediate_score,
                    "category_votes": Counter(), "last_confident_category_ts": 0.0,
                }
                if self._use_motion:
                    self._tracks[track_id]["kf"] = KalmanBox(box)
                    self._tracks[track_id]["kf_time"] = now
            else:
                track = self._tracks[track_id]
                track["bbox"] = box
                track["last_seen"] = now
                track["hits"] = track.get("hits", 1) + 1
                if track["hits"] >= self._min_hits:
                    track["confirmed"] = True
                if self._use_motion:
                    track["kf"].update(box)
            track = self._tracks[track_id]
            if person_categories is not None:
                # Staleness is judged against the timestamp from *before*
                # this frame's sample is folded in -- checking after would
                # let this very sample's own "if category != 'unknown':
                # update the timestamp" step make the check always see
                # "fresh", defeating it entirely.
                was_stale = (now - track["last_confident_category_ts"]) > CATEGORY_STALE_GRACE_SEC
                if was_stale:
                    # Discard a stale vote sum so a person who returns to
                    # view after a long absence (while somehow never having
                    # the track itself evicted) starts from a clean slate
                    # instead of an old answer instantly winning again the
                    # moment any new sample -- even a weak one -- arrives.
                    track["category_votes"] = Counter()
                weight = person_confidences[box_idx] if person_confidences is not None else 1.0
                category = person_categories[box_idx]
                track["category_votes"][category] += weight
                if category != "unknown":
                    track["last_confident_category_ts"] = now
            reported_category = _mode_category(
                track["category_votes"], track["last_confident_category_ts"], now,
                self._category_min_evidence, self._category_margin)
            if not track.get("confirmed", True):
                continue          # tentative: votes are kept, but not reported until seen again
            visible_tracks.append({
                "track_id": track_id,
                "bbox": box,
                "first_seen": track["first_seen"],
                "last_seen": track["last_seen"],
                "category": reported_category,
            })

        evicted_tracks = []
        for track_id in list(self._tracks.keys()):
            if track_id in matched_tracks or track_id in born:
                continue
            track = self._tracks[track_id]
            if not track.get("confirmed", True):
                del self._tracks[track_id]      # never confirmed: it was a false detection, drop silently
                continue
            if now - track["last_seen"] > self._max_age_sec:
                evicted_tracks.append({
                    "track_id": track_id,
                    "first_seen": track["first_seen"],
                    "last_seen": track["last_seen"],
                    "category": _mode_category(
                        track["category_votes"], track["last_confident_category_ts"], now,
                        self._category_min_evidence, self._category_margin),
                })
                del self._tracks[track_id]

        return visible_tracks, evicted_tracks

    def coasting_tracks(self, now: float | None = None) -> list[dict]:
        """Confirmed tracks that were NOT matched in the latest update but were seen within coast_sec:
        their motion-predicted box, so the on-screen box keeps following the person through a short
        detector miss instead of blinking. Purely cosmetic: never fed to Re-ID/tripwire/analytics."""
        if self._coast_sec <= 0 or not self._use_motion:
            return []
        now = self._last_update_ts if now is None else now
        out = []
        for track_id, track in self._tracks.items():
            if not track.get("confirmed", True) or track["last_seen"] >= self._last_update_ts:
                continue
            if now - track["last_seen"] > self._coast_sec:
                continue
            out.append({"track_id": track_id, "bbox": [float(v) for v in track["kf"].xyxy],
                        "category": _mode_category(track["category_votes"], track["last_confident_category_ts"], now,
                                                   self._category_min_evidence, self._category_margin)})
        return out

    def _match_with_motion(self, person_boxes, now, matched_tracks, matched_boxes, box_to_track,
                           is_low=None) -> None:
        """Hungarian assignment of detections to tracks on IoU. A track is scored by the better
        of (a) its motion-predicted box and (b) its last observed box, so for a still or slow
        person this can never do worse than the old behaviour, while a walking person is still
        matched when they moved further than their own width between two AI passes."""
        if not self._tracks:
            return
        ids = list(self._tracks.keys())
        for track_id in ids:                       # advance every filter to `now` exactly once
            track = self._tracks[track_id]
            track["kf"].predict(max(now - track["kf_time"], 0.0))
            track["kf_time"] = now
        if not len(person_boxes):
            return
        predicted = np.stack([self._tracks[i]["kf"].xyxy for i in ids])
        last = np.asarray([self._tracks[i]["bbox"] for i in ids], dtype=float)
        dets = np.asarray(person_boxes, dtype=float).reshape(-1, 4)
        iou = np.maximum(iou_matrix(predicted, dets), iou_matrix(last, dets))
        low = list(is_low) if is_low is not None else [False] * len(person_boxes)
        # Stage 1: confident detections against every track. Stage 2 (ByteTrack-style rescue): the
        # low-score detections against the tracks that are still unmatched, so a person whose
        # confidence dips for a moment keeps the same track instead of blinking out and being reborn.
        for stage_low in (False, True):
            cols_idx = [j for j in range(len(person_boxes)) if low[j] == stage_low and j not in matched_boxes]
            rows_idx = [i for i in range(len(ids)) if ids[i] not in matched_tracks]
            if not cols_idx or not rows_idx:
                continue
            sub = iou[np.ix_(rows_idx, cols_idx)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= MOTION_MIN_MATCH_IOU:
                    matched_tracks.add(ids[rows_idx[r]])
                    matched_boxes.add(int(cols_idx[c]))
                    box_to_track[int(cols_idx[c])] = ids[rows_idx[r]]
