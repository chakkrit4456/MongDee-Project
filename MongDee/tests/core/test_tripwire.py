"""Unit tests for core/tripwire.py's crossing-detection algorithm — all
synthetic (no camera/YOLO needed), driving TripwireCounter with the exact
visible_tracks/evicted_tracks shape core.tracker.PersonTracker.update()
produces.
"""

from __future__ import annotations

from core.tripwire import (
    DEAD_ZONE_NORM,
    DIRECTION_IN,
    DIRECTION_OUT,
    SIDE_A,
    SIDE_B,
    TRIPWIRE_CONFIRM_FRAMES,
    TripwireCounter,
    TripwireLine,
    foot_point,
)

assert TRIPWIRE_CONFIRM_FRAMES == 2, "tests below assume exactly 2 confirmation frames"

FRAME_W = 640
FRAME_H = 480

# Vertical line straight down the middle of the frame (x=0.5 normalized).
# side_of() puts nx < 0.5 on SIDE_A, nx > 0.5 on SIDE_B (see core/tripwire.py's
# docstring on the cross-product convention) — "inside the booth" = SIDE_A
# (the left half), so walking left-to-right (A -> B) is OUT and
# right-to-left (B -> A) is IN.
LINE = TripwireLine(id="TW-1", camera_id="CAM-1", x1=0.5, y1=0.0, x2=0.5, y2=1.0,
                     inside_side=SIDE_A, enabled=True)


def _track(track_id: int, nx: float, ny: float = 0.5) -> dict:
    """A visible_tracks-shaped dict whose foot point (bottom-center of its
    bbox) lands exactly at normalized (nx, ny)."""
    px, py = nx * FRAME_W, ny * FRAME_H
    bbox = [px - 10, py - 40, px + 10, py]
    assert foot_point(bbox) == (px, py)
    return {"track_id": track_id, "bbox": bbox, "first_seen": 0.0, "last_seen": 0.0, "category": "unknown"}


def _counter() -> TripwireCounter:
    return TripwireCounter(LINE)


# --------------------------------------------------------------- 1 & 2: IN/OUT

def test_outside_to_inside_counts_in():
    c = _counter()
    now = 100.0
    events1 = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)  # SIDE_B, first reading
    assert events1 == []
    events2 = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1)  # candidate: SIDE_A, frame 1/2
    assert events2 == []
    events3 = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 2)  # confirmed: SIDE_A, frame 2/2
    assert len(events3) == 1
    assert events3[0]["track_id"] == 1
    assert events3[0]["direction"] == DIRECTION_IN
    assert c.count_in == 1
    assert c.count_out == 0


def test_inside_to_outside_counts_out():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now)  # SIDE_A, first reading
    events1 = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 1)  # candidate: SIDE_B, frame 1/2
    assert events1 == []
    events = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 2)  # confirmed: SIDE_B, frame 2/2
    assert len(events) == 1
    assert events[0]["direction"] == DIRECTION_OUT
    assert c.count_in == 0
    assert c.count_out == 1


def test_single_frame_flip_does_not_count_needs_confirmation():
    """Spec section 11: a lone opposite-side frame (one lean over the line,
    or one bad detection) must not fire a crossing by itself — only a
    sustained (TRIPWIRE_CONFIRM_FRAMES-consecutive) reading does."""
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)  # SIDE_B established
    events = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1)  # one frame on SIDE_A
    assert events == []
    # Immediately reads back on the original side — the pending confirmation
    # must be cancelled, not carried over.
    events = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 2)
    assert events == []
    assert c.count_in == 0 and c.count_out == 0


# ------------------------------------------------------------- 3: parallel walk

def test_walking_parallel_to_line_never_counted():
    c = _counter()
    now = 100.0
    # Stays firmly on SIDE_B (nx=0.8) the entire time, only y (ny) changes —
    # walking parallel to the line, never approaching or crossing it.
    for i, ny in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        events = c.update([_track(1, 0.8, ny)], [], FRAME_W, FRAME_H, now + i)
        assert events == []
    assert c.count_in == 0
    assert c.count_out == 0


# --------------------------------------------------------- 4: standing on line

def test_standing_on_line_does_not_double_count():
    c = _counter()
    now = 100.0
    # Establish a firm SIDE_A reading first.
    c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now)
    assert DEAD_ZONE_NORM < 0.02  # sanity: keep the jitter below the dead zone
    # Jitters right on top of the line (well inside DEAD_ZONE_NORM) for
    # several frames — every one of these must be ambiguous, so the track's
    # last *solid* side (A) is never overwritten and nothing fires.
    for i, nx in enumerate([0.495, 0.505, 0.498, 0.502, 0.500]):
        events = c.update([_track(1, nx)], [], FRAME_W, FRAME_H, now + 1 + i)
        assert events == [], f"frame {i} (nx={nx}) should be inside the dead zone"
    assert c.count_in == 0
    assert c.count_out == 0
    # Only once they decisively commit to the other side (for
    # TRIPWIRE_CONFIRM_FRAMES consecutive frames) does a single, correct
    # crossing fire.
    events = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 10)
    assert events == []
    events = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 11)
    assert len(events) == 1
    assert events[0]["direction"] == DIRECTION_OUT
    assert c.count_out == 1


# ------------------------------------------------------- 5: many frames, once

def test_same_person_across_many_frames_counted_once():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)
    assert c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1) == []  # confirmation frame 1/2
    first_cross = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1.5)  # confirmation frame 2/2
    assert len(first_cross) == 1
    # Keeps standing firmly inside (SIDE_A) for many more AI passes — no
    # further crossings should ever fire since the side never changes again.
    for i in range(20):
        events = c.update([_track(1, 0.3 + 0.001 * i)], [], FRAME_W, FRAME_H, now + 2 + i)
        assert events == []
    assert c.count_in == 1
    assert c.count_out == 0


# ------------------------------------------------------ 6: temporary track gap

def test_temporary_track_gap_does_not_create_duplicate():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)  # SIDE_B established
    # Track briefly missing from visible_tracks (occluded) for a couple of
    # passes — core.tracker.PersonTracker keeps the same track_id alive
    # through this (TRACK_MAX_AGE_SEC), so it's simply absent from the
    # visible_tracks list handed in, not evicted.
    assert c.update([], [], FRAME_W, FRAME_H, now + 1) == []
    assert c.update([], [], FRAME_W, FRAME_H, now + 2) == []
    # Reappears already on the other side — exactly one crossing (after
    # confirmation), not one per missed frame, and nothing was invented
    # during the gap itself.
    assert c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 3) == []
    events = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 3.5)
    assert len(events) == 1
    assert events[0]["direction"] == DIRECTION_IN
    assert c.count_in == 1
    assert c.count_out == 0


def test_real_eviction_forgets_track_state_cleanly():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)
    evicted = [{"track_id": 1, "first_seen": now, "last_seen": now}]
    assert c.update([], evicted, FRAME_W, FRAME_H, now + 2) == []
    # A brand new track_id (as core.tracker.PersonTracker would hand out
    # after a real eviction) starting fresh on the other side must not be
    # treated as a continuation of track 1 — first reading, no crossing.
    events = c.update([_track(2, 0.3)], [], FRAME_W, FRAME_H, now + 5)
    assert events == []
    assert c.count_in == 0


# --------------------------------------------------- 7: resolution independence

def test_line_position_stable_across_resolution_change():
    c = _counter()
    now = 100.0
    # Established at 640x480.
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)
    # Same *normalized* position, but the frame (and so the bbox in pixel
    # space) is now a completely different resolution — a naive pixel-space
    # comparison would misplace the line; the normalized conversion must
    # keep it correct.
    big_w, big_h = 1920, 1080
    # Bbox height kept at the same *proportion* of frame height as _track()'s
    # (40/480) — a real person's box scales with resolution, it doesn't stay
    # a fixed pixel size, so this is what an actual resolution change looks
    # like (and it keeps this track above TRACK_MIN_BBOX_HEIGHT_NORM).
    box_h = (40 / FRAME_H) * big_h  # same proportion of frame height as _track()'s 40px/480px
    track_same_spot = {  # bbox built directly at the new resolution's 0.3 normalized x
        "track_id": 1,
        "bbox": [0.3 * big_w - 10, 0.5 * big_h - box_h, 0.3 * big_w + 10, 0.5 * big_h],
        "first_seen": 0.0, "last_seen": 0.0, "category": "unknown",
    }
    assert c.update([track_same_spot], [], big_w, big_h, now + 1) == []
    events = c.update([track_same_spot], [], big_w, big_h, now + 1.5)
    assert len(events) == 1
    assert events[0]["direction"] == DIRECTION_IN


# ------------------------------------------------------- 8: independent cameras

def test_multiple_cameras_have_independent_counters():
    cam1 = TripwireCounter(TripwireLine(id="TW-1", camera_id="CAM-1", x1=0.5, y1=0.0, x2=0.5, y2=1.0,
                                         inside_side=SIDE_A))
    cam2 = TripwireCounter(TripwireLine(id="TW-2", camera_id="CAM-2", x1=0.5, y1=0.0, x2=0.5, y2=1.0,
                                         inside_side=SIDE_A))
    now = 100.0
    cam1.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)
    cam1.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1)
    cam1.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1.5)  # cam1: +1 IN (confirmed)
    cam2.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now)
    cam2.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 1)
    cam2.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 1.5)  # cam2: +1 OUT (confirmed)
    assert cam1.count_in == 1 and cam1.count_out == 0
    assert cam2.count_in == 0 and cam2.count_out == 1


# --------------------------------------------------------------- other safety

def test_no_line_configured_is_inert():
    c = TripwireCounter(None)
    events = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, 100.0)
    assert events == []
    assert c.count_in == 0 and c.count_out == 0


def test_disabled_line_is_inert():
    disabled = TripwireLine(id="TW-1", camera_id="CAM-1", x1=0.5, y1=0.0, x2=0.5, y2=1.0,
                             inside_side=SIDE_A, enabled=False)
    c = TripwireCounter(disabled)
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, 100.0)
    events = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, 101.0)
    assert events == []


def test_reconfiguring_line_resets_side_memory_not_counts():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now)
    c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1)
    c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 1.5)  # +1 IN (confirmed)
    assert c.count_in == 1
    new_line = TripwireLine(id="TW-1", camera_id="CAM-1", x1=0.2, y1=0.0, x2=0.2, y2=1.0, inside_side=SIDE_A)
    c.set_line(new_line)
    assert c.count_in == 1  # counts survive a reconfigure
    # But side memory was cleared — the very next reading for track 1 is
    # treated as a fresh first observation, not compared against its old
    # (now-meaningless, against a different line) side.
    events = c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now + 2)
    assert events == []


# ------------------------------------------------------- 12: track quality gate

def test_too_small_bbox_never_counted():
    """Spec section 12: a track whose box is too short (relative to frame
    height) is noise, not a trustworthy foot-point reading — it must never
    register a crossing, no matter how many frames it "crosses" on."""
    from core.tripwire import TRACK_MIN_BBOX_HEIGHT_NORM
    c = _counter()
    now = 100.0
    tiny_h = TRACK_MIN_BBOX_HEIGHT_NORM * FRAME_H * 0.5  # well under the gate

    def tiny_track(nx: float) -> dict:
        px, py = nx * FRAME_W, 0.5 * FRAME_H
        return {"track_id": 1, "bbox": [px - 5, py - tiny_h, px + 5, py],
                "first_seen": 0.0, "last_seen": 0.0, "category": "unknown"}

    for i, nx in enumerate([0.7, 0.3, 0.3, 0.7, 0.7]):
        events = c.update([tiny_track(nx)], [], FRAME_W, FRAME_H, now + i)
        assert events == []
    assert c.count_in == 0 and c.count_out == 0


def test_too_new_track_gated_until_it_ages():
    """Spec section 12: a track younger than TRACK_MIN_AGE_SEC is not yet
    trustworthy — it must not seed a baseline side (let alone fire a
    crossing) until it has aged past the gate."""
    from core.tripwire import TRACK_MIN_AGE_SEC
    c = _counter()
    born = 100.0
    track_id = 1

    def aged_track(nx: float, ts: float) -> dict:
        px, py = nx * FRAME_W, 0.5 * FRAME_H
        return {"track_id": track_id, "bbox": [px - 10, py - 40, px + 10, py],
                "first_seen": born, "last_seen": ts, "category": "unknown"}

    # Still too young — gated, no baseline established yet.
    events = c.update([aged_track(0.7, born)], [], FRAME_W, FRAME_H, born)
    assert events == []
    # Now old enough — this becomes its first trusted (baseline) reading.
    now = born + TRACK_MIN_AGE_SEC + 0.01
    events = c.update([aged_track(0.7, now)], [], FRAME_W, FRAME_H, now)
    assert events == []
    # Confirmed flip to the other side now counts normally.
    events = c.update([aged_track(0.3, now + 1)], [], FRAME_W, FRAME_H, now + 1)
    assert events == []
    events = c.update([aged_track(0.3, now + 1.5)], [], FRAME_W, FRAME_H, now + 1.5)
    assert len(events) == 1
    assert events[0]["direction"] == DIRECTION_IN


def test_current_inside_never_negative():
    c = _counter()
    now = 100.0
    c.update([_track(1, 0.3)], [], FRAME_W, FRAME_H, now)
    c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 1)
    events = c.update([_track(1, 0.7)], [], FRAME_W, FRAME_H, now + 1.5)  # OUT with no prior IN (confirmed)
    assert events[0]["direction"] == DIRECTION_OUT
    assert c.count_out == 1
    assert c.current_inside() == 0  # floored, never -1
