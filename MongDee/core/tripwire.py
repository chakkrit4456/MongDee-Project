"""Virtual Tripwire — a configurable line on one camera's frame that counts
people crossing it (IN toward "inside the booth", OUT away from it).

Detection strategy (per the spec): never trust raw per-frame YOLO detections
alone — reuse core.tracker.PersonTracker's track IDs (each one already
survives brief occlusion via TRACK_MAX_AGE_SEC) and watch each track's own
foot-point (bottom-center of its bounding box) move from one side of the
line to the other. A crossing only fires on an actually-observed
*transition* between two solid (non-ambiguous) side readings for the same
track_id, never inferred from a single frame's position alone. That one
rule is what naturally rules out counting a person more than once across
frames (nothing repeats every frame — only the one frame the side actually
changes fires an event) and rules out inventing a crossing just because a
track resumed on the other side after a brief tracking gap (no crossing is
ever fired without *observing* both sides in sequence).

Line and foot-point coordinates are both normalized to the frame's own
width/height (0.0-1.0) at the moment of each check, so a saved tripwire
never drifts out of position if the camera's capture resolution changes.

Framework-agnostic like core/tracker.py, core/interest_tracker.py and
core/aggregator.py: no camera I/O, no database — web/booth_manager.py owns
persistence and wires this to core.vision.CameraWorker's on_person_tracks
callback, the same place presence-session logging already consumes
visible_tracks/evicted_tracks.
"""

from __future__ import annotations

import dataclasses

# How close (in normalized units, a fraction of frame width/height) a foot
# point has to be to the line before its side reading becomes ambiguous
# rather than solid. Without this, a person standing still exactly on the
# line would flicker between "just barely A" and "just barely B" every
# frame from ordinary detection jitter, each flicker firing a fake
# crossing. While a track's foot point is inside this band, its last solid
# side is simply left unchanged instead of being overwritten — see
# TripwireCounter.update().
DEAD_ZONE_NORM = 0.015

# Once a track has just registered a crossing, further crossings from that
# same track are ignored for this long. A person genuinely lingering near
# the line can otherwise wobble across the dead zone boundary repeatedly,
# each edge potentially reading as another crossing a moment later.
CROSSING_COOLDOWN_SEC = 1.5

# A side reading opposite the track's last solid side must repeat this many
# consecutive AI passes before it is accepted as a real crossing (rather
# than firing the instant the foot point first steps past the line). This
# is what tells apart someone genuinely walking through from someone who
# leans over the line for one frame and pulls back — a single-frame flip
# would otherwise count the lean as a crossing. Any frame that reads back on
# the original side (or into the dead zone) cancels the pending count.
TRIPWIRE_CONFIRM_FRAMES = 2

# Track quality gate (spec section 12): a track has to have existed for at
# least this long before its foot point is allowed to register a crossing.
# Brand-new tracks are the ones most likely to be detector noise (a
# one-frame false positive, or a real person's track ID that just churned)
# rather than an established, trustworthy trajectory.
TRACK_MIN_AGE_SEC = 0.3

# ...and its bounding box must be at least this tall (as a fraction of
# frame height) — a tiny/far/partially-visible box's foot point is too
# noisy to trust for a line-crossing decision. Deliberately low (5%): this
# only needs to reject degenerate near-zero-size detector noise, not
# legitimately small/far people still meaningfully approaching the line.
TRACK_MIN_BBOX_HEIGHT_NORM = 0.05

SIDE_A = "A"
SIDE_B = "B"
DIRECTION_IN = "in"
DIRECTION_OUT = "out"


@dataclasses.dataclass(frozen=True)
class TripwireLine:
    """One configured line for one camera. Coordinates are normalized
    (0.0-1.0) against that camera's own frame width/height, so the same
    config is valid regardless of the camera's actual capture resolution."""
    id: str
    camera_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    inside_side: str  # SIDE_A or SIDE_B — which side of the line is "inside the booth"
    enabled: bool = True

    def side_of(self, nx: float, ny: float) -> tuple[str | None, float]:
        """Returns (side, signed_distance) for a normalized point (nx, ny).

        side is SIDE_A, SIDE_B, or None if the point falls inside
        DEAD_ZONE_NORM of the line (ambiguous — straddling it). The line is
        treated as infinite (extended past its two endpoints) rather than a
        strict segment — simpler and matches how a tripwire is meant to be
        drawn (spanning the full width a person could plausibly enter
        through), at the cost of also reacting to someone crossing its
        far-off imaginary extension; acceptable for a booth-entrance line.

        signed_distance is returned even inside the dead zone, for anything
        that wants the raw value (e.g. tests, or a future "how close" UI
        hint) — it is not itself used for crossing decisions there.
        """
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return None, 0.0
        cross = dx * (ny - self.y1) - dy * (nx - self.x1)
        distance = cross / length
        if abs(distance) < DEAD_ZONE_NORM:
            return None, distance
        return (SIDE_A if distance >= 0 else SIDE_B), distance

    def raw_side_of(self, nx: float, ny: float) -> str:
        """Like side_of(), but never returns None/ambiguous -- used only when the caller has
        independently established (via bbox_touches_line) that the detection box is physically
        overlapping the line, which is itself strong evidence this is a real crossing attempt
        rather than detector jitter far from the line. See update()'s docstring for why this is
        what lets a crossing register without requiring the whole box to have passed fully
        through to the other side first."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = (dx * dx + dy * dy) ** 0.5
        if length == 0:
            return SIDE_A
        cross = dx * (ny - self.y1) - dy * (nx - self.x1)
        return SIDE_A if cross >= 0 else SIDE_B

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TripwireLine":
        return TripwireLine(
            id=d["id"], camera_id=d["camera_id"],
            x1=float(d["x1"]), y1=float(d["y1"]), x2=float(d["x2"]), y2=float(d["y2"]),
            inside_side=d["inside_side"], enabled=bool(d.get("enabled", True)),
        )


def bbox_touches_line(bbox: list[float], line: "TripwireLine",
                       frame_width: int, frame_height: int) -> bool:
    """True when the detection box (pixel [x1,y1,x2,y2]) physically overlaps the tripwire line --
    i.e. the line passes through the box's rectangle, not just its foot point. Checked via the
    box's four corners: if they aren't all on the same raw side (no dead zone -- any straddle
    counts), the line cuts through the box somewhere. This is what lets a crossing be counted the
    moment a person's detection box reaches the line, instead of requiring the WHOLE box (and
    specifically its foot point, past a further DEAD_ZONE_NORM margin) to have already crossed
    all the way to the other side."""
    if frame_width <= 0 or frame_height <= 0:
        return False
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    if dx == 0 and dy == 0:
        return False
    x1, y1, x2, y2 = bbox
    signs = []
    for px, py in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        nx, ny = px / frame_width, py / frame_height
        signs.append(dx * (ny - line.y1) - dy * (nx - line.x1))
    return min(signs) < 0 < max(signs)


def foot_point(bbox: list[float]) -> tuple[float, float]:
    """Bottom-center of a [x1,y1,x2,y2] pixel bbox — the reference point the
    spec calls for ("จุดกึ่งกลางด้านล่างของ Bounding Box"), i.e. roughly where
    a standing person's feet are, which tracks a doorway crossing far more
    reliably than the box center would (the center lags behind the feet by
    half a body height while walking toward/away from the camera)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


class TripwireCounter:
    """Owns crossing detection and running IN/OUT counts for exactly one
    camera's tripwire. One instance per camera, fed every AI pass with the
    same visible_tracks/evicted_tracks core.tracker.PersonTracker.update()
    just produced (see web/booth_manager.py's _on_person_tracks, which
    already receives both for presence-session logging)."""

    def __init__(self, line: TripwireLine | None = None):
        self._line = line
        self._track_side: dict[int, str] = {}
        self._track_pending: dict[int, tuple[str, int]] = {}  # track_id -> (candidate_side, count)
        self._track_cooldown_until: dict[int, float] = {}
        self.count_in = 0
        self.count_out = 0

    def set_line(self, line: TripwireLine | None) -> None:
        """Reconfiguring (or disabling/removing) the line resets all
        per-track side memory — not the running counters. A moved line
        makes any previously-recorded "which side was this track on"
        reading meaningless, and keeping it could fire a spurious crossing
        against a line the track never actually walked past."""
        self._line = line
        self._track_side.clear()
        self._track_pending.clear()
        self._track_cooldown_until.clear()

    def get_line(self) -> TripwireLine | None:
        return self._line

    def reset_counts(self) -> None:
        self.count_in = 0
        self.count_out = 0

    def current_inside(self) -> int:
        """IN minus OUT, floored at 0 — a simple running occupancy count.
        Never goes negative: a booth that was already occupied before the
        tripwire was configured (or that briefly mis-tracked an OUT for
        someone who was never counted IN) would otherwise show an
        impossible negative headcount."""
        return max(0, self.count_in - self.count_out)

    def update(self, visible_tracks: list[dict], evicted_tracks: list[dict],
               frame_width: int, frame_height: int, now: float) -> list[dict]:
        """Call once per AI pass. Returns newly-fired crossing events from
        this call only: [{"track_id", "direction", "ts"}, ...] (direction
        is DIRECTION_IN or DIRECTION_OUT). Safe to call with no configured
        line, a disabled line, or a zero-size frame — always returns []
        rather than raising, and never blocks/slows the caller (this is
        called from the same AI-worker thread run_ai_pass() already runs
        on, so it must stay cheap and never touch camera I/O)."""
        events: list[dict] = []
        line = self._line
        if line is None or not line.enabled or frame_width <= 0 or frame_height <= 0:
            # No active line to evaluate against, but still forget evicted
            # tracks' state so it can never leak indefinitely.
            for t in evicted_tracks:
                self._forget_track(t["track_id"])
            return events

        for t in visible_tracks:
            track_id = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            bbox_height_norm = (y2 - y1) / frame_height if frame_height > 0 else 0.0
            track_age_sec = now - t.get("first_seen", now)

            fx, fy = foot_point(t["bbox"])
            nx, ny = fx / frame_width, fy / frame_height
            if bbox_touches_line(t["bbox"], line, frame_width, frame_height):
                # The box is physically overlapping the line right now -- that overlap is
                # itself strong evidence this is a genuine crossing attempt, not detector
                # jitter, so trust the foot point's raw side immediately instead of also
                # requiring it to clear DEAD_ZONE_NORM past the line. TRIPWIRE_CONFIRM_FRAMES
                # and CROSSING_COOLDOWN_SEC below still guard against noise the same as always
                # -- this only changes WHEN a side reading is trusted, not how many consistent
                # readings are required before a crossing fires.
                side = line.raw_side_of(nx, ny)
            else:
                side, _dist = line.side_of(nx, ny)
                if side is None:
                    continue  # inside the dead zone — last solid side (if any) stands unchanged

            if (track_age_sec < TRACK_MIN_AGE_SEC or
                    bbox_height_norm < TRACK_MIN_BBOX_HEIGHT_NORM):
                # Track quality gate (spec section 12): too new / too small
                # to trust yet. Don't touch side memory at all — wait for a
                # frame where this same track has aged/grown enough, then
                # treat that as its first reading (still no crossing fires
                # off a gated track's position, only off an observed change
                # after it becomes trustworthy).
                continue

            prev_side = self._track_side.get(track_id)

            if prev_side is None:
                self._track_side[track_id] = side  # first trusted reading — baseline, no crossing
                self._track_pending.pop(track_id, None)
                continue

            if side == prev_side:
                self._track_pending.pop(track_id, None)  # back on the known side — cancel any pending flip
                continue

            # Reads opposite the last known side — candidate crossing.
            # Require TRIPWIRE_CONFIRM_FRAMES consecutive confirmations
            # (spec section 11) before treating it as real, so one noisy
            # frame near the line can't fire an event on its own.
            pending_side, pending_count = self._track_pending.get(track_id, (side, 0))
            if pending_side != side:
                pending_count = 0
            pending_count += 1

            if pending_count < TRIPWIRE_CONFIRM_FRAMES:
                self._track_pending[track_id] = (side, pending_count)
                continue

            self._track_pending.pop(track_id, None)
            self._track_side[track_id] = side

            cooldown_until = self._track_cooldown_until.get(track_id, 0.0)
            if now < cooldown_until:
                continue  # debounced — this track crossed very recently already

            direction = DIRECTION_IN if side == line.inside_side else DIRECTION_OUT
            self._track_cooldown_until[track_id] = now + CROSSING_COOLDOWN_SEC
            if direction == DIRECTION_IN:
                self.count_in += 1
            else:
                self.count_out += 1
            events.append({"track_id": track_id, "direction": direction, "ts": now})

        for t in evicted_tracks:
            self._forget_track(t["track_id"])

        return events

    def _forget_track(self, track_id: int) -> None:
        self._track_side.pop(track_id, None)
        self._track_pending.pop(track_id, None)
        self._track_cooldown_until.pop(track_id, None)
