"""Per-camera product movement / "interest" tracking.

Implements the rule the booth needs for its Dashboard: a product resting
untouched for STATIONARY_BASELINE_SEC is "not moved" (idle baseline). But as
soon as a person's hand overlaps it and it moves, the 10-second countdown
does not apply — instead the time spent held counts toward
INTEREST_CONFIRMATION_SECONDS (spec: MongDee person-gender/age master
continuation prompt sections 17-27 — "Real-Time 2-Second Confirmation").

Two events now come out of update(), not one:
  - {"event": "confirmed", ...} fires exactly once, the instant a
    continuous hold reaches INTEREST_CONFIRMATION_SECONDS — *while the
    product is still being held*, not deferred to release (spec section 20:
    "ยืนยัน Interest ตอนครบ 2 seconds ไม่ต้องรอ release").
  - {"event": "finalized", ...} fires once the hold actually ends (release,
    hand-off to a different holder, or the product disappearing for too
    long), but *only* if that hold was actually confirmed first — an
    unconfirmed (<2s) hold produces no event at all, exactly as before.
    Release's job is to close out the same interaction, never to open a
    second one (spec section 23) — see web/booth_manager.py's open/finalize
    DB wiring, which keys off this same confirmed/finalized pairing.

Deliberately separate from core/aggregator.py: the aggregator answers "what
is THE current product for the whole booth" via cross-camera agreement at
class-name granularity, for the product-info panel. This module answers a
different, inherently per-camera question — bounding boxes and person track
IDs from core/tracker.py are per-camera and not comparable across cameras —
so it is keyed by (camera_id, class_name) instead.
"""

from __future__ import annotations

import threading
import time

STATIONARY_BASELINE_SEC = 10.0     # sit still this long, untouched -> "idle" (not moved)
MOVEMENT_PX_THRESHOLD = 20.0       # centroid displacement vs. resting position, at 640x480 capture
HOLD_MIN_IOU = 0.02                # product bbox overlaps a person bbox at least this much...
RELEASE_GRACE_SEC = 1.0            # ...or stop counting as "held" after this long without contact/movement
PRODUCT_ABSENCE_TIMEOUT_SEC = 3.0  # matches aggregator.IDLE_RESET_SEC

# Spec sections 15/20: a continuous, evidenced hold must last at least this
# long — using real elapsed wall-clock time (time.time()), never a frame
# count — before it becomes a confirmed Interest Event. Supersedes the old
# MIN_HOLD_DURATION_SEC noise filter (0.5s): 2.0s is strictly stricter, so a
# hold that clears this also clears what that filter was ever for.
INTEREST_CONFIRMATION_SECONDS = 2.0

STATE_UNSETTLED = "unsettled"
STATE_IDLE = "idle"
STATE_HELD = "held"


def _centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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


def _center_inside(inner_bbox, outer_bbox) -> bool:
    cx, cy = _centroid(inner_bbox)
    x1, y1, x2, y2 = outer_bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _find_holder(product_bbox, person_tracks: list[dict]) -> dict | None:
    best = None
    best_score = 0.0
    for track in person_tracks:
        score = _iou(product_bbox, track["bbox"])
        touching = score >= HOLD_MIN_IOU or _center_inside(product_bbox, track["bbox"])
        if touching and score >= best_score:
            best = track
            best_score = score
    return best


class _ProductState:
    def __init__(self, bbox, now):
        self.state = STATE_UNSETTLED
        self.resting_bbox = bbox
        self.settle_start_ts = now
        self.last_seen_ts = now
        self.hold_start_ts: float | None = None
        self.holder_track_id: int | None = None
        self.confirmed = False               # has this hold already fired its "confirmed" event?
        self.confirmed_at: float | None = None

    def interest_seconds(self, now) -> float:
        if self.state == STATE_HELD and self.hold_start_ts is not None:
            return now - self.hold_start_ts
        return 0.0

    def live_status(self, now) -> str:
        """For ProductInterestTracker.live_states()'s polling-based UI feed
        (spec sections 32-34): NOT_INTERACTING while idle/unsettled,
        POSSIBLE_INTERACTION while held but under INTEREST_CONFIRMATION_
        SECONDS, INTEREST_CONFIRMED once it's cleared that mark — backend is
        the source of truth here, the dashboard only ever renders this."""
        if self.state != STATE_HELD:
            return "NOT_INTERACTING"
        if self.confirmed or self.interest_seconds(now) >= INTEREST_CONFIRMATION_SECONDS:
            return "INTEREST_CONFIRMED"
        return "POSSIBLE_INTERACTION"


class ProductInterestTracker:
    def __init__(self):
        self._states: dict[tuple[str, str], _ProductState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _maybe_confirm(state: "_ProductState", now: float, camera_id: str, class_name: str,
                        events: list[dict]) -> None:
        """Fires the one-time "confirmed" event the instant a continuous
        hold reaches INTEREST_CONFIRMATION_SECONDS (spec sections 17-22) —
        called on every tick a hold is still active, so confirmation always
        happens live regardless of whether the product is moving on this
        exact frame or just being held steady."""
        if state.confirmed or state.hold_start_ts is None:
            return
        if now - state.hold_start_ts >= INTEREST_CONFIRMATION_SECONDS:
            state.confirmed = True
            state.confirmed_at = now
            events.append({
                "event": "confirmed", "camera_id": camera_id, "class_name": class_name,
                "holder_track_id": state.holder_track_id,
                "hold_start_ts": state.hold_start_ts, "confirmed_at": now,
            })

    @staticmethod
    def _finalize(state: "_ProductState", end_ts: float, camera_id: str, class_name: str,
                   events: list[dict]) -> None:
        """Ends the current HELD segment. Emits a "finalized" event only
        when that segment actually got confirmed first (spec sections 19/
        23/27/29) — an unconfirmed (<2s) hold produces no event at all, and
        release/hand-off/disappearance never invents a second event for an
        already-confirmed one (spec section 24)."""
        if state.confirmed:
            events.append({
                "event": "finalized", "camera_id": camera_id, "class_name": class_name,
                "holder_track_id": state.holder_track_id,
                "hold_start_ts": state.hold_start_ts, "hold_end_ts": end_ts,
                "confirmed_at": state.confirmed_at,
                "duration_sec": end_ts - state.hold_start_ts,
            })
        state.state = STATE_UNSETTLED
        state.hold_start_ts = None
        state.holder_track_id = None
        state.confirmed = False
        state.confirmed_at = None

    def update(self, camera_id: str, product_detections: list[dict],
               person_tracks: list[dict]) -> list[dict]:
        """Call once per inference frame per camera. Returns zero or more
        lifecycle events — {"event": "confirmed", camera_id, class_name,
        holder_track_id, hold_start_ts, confirmed_at} the instant a hold
        clears INTEREST_CONFIRMATION_SECONDS, and/or {"event": "finalized",
        ..., hold_end_ts, duration_sec} once an already-confirmed hold ends
        — see this module's docstring and _maybe_confirm/_finalize above."""
        now = time.time()
        events: list[dict] = []

        best_by_class: dict[str, dict] = {}
        for det in product_detections:
            name = det["class_name"]
            if name not in best_by_class or det["conf"] > best_by_class[name]["conf"]:
                best_by_class[name] = det

        with self._lock:
            for class_name, det in best_by_class.items():
                key = (camera_id, class_name)
                bbox = det["bbox"]
                state = self._states.get(key)
                if state is None:
                    self._states[key] = _ProductState(bbox, now)
                    continue

                state.last_seen_ts = now
                moved = _dist(_centroid(bbox), _centroid(state.resting_bbox)) > MOVEMENT_PX_THRESHOLD
                holder = _find_holder(bbox, person_tracks)

                if moved and holder is not None:
                    if state.state != STATE_HELD:
                        state.state = STATE_HELD
                        state.hold_start_ts = now
                        state.holder_track_id = holder["track_id"]
                    elif holder["track_id"] != state.holder_track_id:
                        # hand-off: finalize the previous holder's segment
                        # (only emits if it had actually been confirmed),
                        # then start a fresh one for the new holder.
                        self._finalize(state, now, camera_id, class_name, events)
                        state.state = STATE_HELD
                        state.hold_start_ts = now
                        state.holder_track_id = holder["track_id"]
                    state.resting_bbox = bbox
                    self._maybe_confirm(state, now, camera_id, class_name, events)
                elif state.state == STATE_HELD:
                    if now - state.last_seen_ts >= RELEASE_GRACE_SEC or (not moved and holder is None):
                        self._finalize(state, now, camera_id, class_name, events)
                        state.resting_bbox = bbox
                        state.settle_start_ts = now
                    else:
                        # still within grace / holding steady (not actively
                        # moving this exact tick, but not released either) —
                        # confirmation must still fire live at exactly 2s.
                        self._maybe_confirm(state, now, camera_id, class_name, events)
                elif moved:
                    # displaced without an attributable holder (e.g. nudged) —
                    # can't log an event without a holder, just re-arm the baseline
                    state.resting_bbox = bbox
                    state.settle_start_ts = now
                    state.state = STATE_UNSETTLED
                else:
                    if state.state == STATE_UNSETTLED and now - state.settle_start_ts >= STATIONARY_BASELINE_SEC:
                        state.state = STATE_IDLE

            # drop products not seen recently; finalize an in-progress hold first
            for key in list(self._states.keys()):
                cam, class_name = key
                if cam != camera_id or class_name in best_by_class:
                    continue
                state = self._states[key]
                if now - state.last_seen_ts > PRODUCT_ABSENCE_TIMEOUT_SEC:
                    if state.state == STATE_HELD:
                        self._finalize(state, state.last_seen_ts, cam, class_name, events)
                    del self._states[key]

        return events

    def live_states(self) -> list[dict]:
        """Polling-based live feed (spec sections 32-34: no WebSocket in
        this app, and none is added — the existing /api/dashboard/live poll
        is the realtime transport) — every currently-tracked product's
        live interest status, backend-decided (never inferred by the UI)."""
        now = time.time()
        with self._lock:
            return [
                {
                    "camera_id": camera_id,
                    "class_name": class_name,
                    "state": state.state,
                    "interest_seconds": state.interest_seconds(now),
                    "holder_track_id": state.holder_track_id,
                    "confirmed": state.confirmed,
                    "live_status": state.live_status(now),
                }
                for (camera_id, class_name), state in self._states.items()
            ]
