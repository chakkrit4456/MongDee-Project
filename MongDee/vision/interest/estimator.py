"""Estimated Customer Interest (MongDee_Master_Prompt.md sections 54-56, 82).

One InterestSession per (global person, product/zone). Each pipeline tick,
feed it the current observation (distance, whether the person is stopped,
whether they're approaching, a facing/gaze score if available, interaction
status). It accumulates look/dwell time and produces:

    InterestScore = weighted(gaze, head_pose, distance, look_duration, stop, approach, interaction)
    status        = LOW / POSSIBLE / HIGH / UNKNOWN

Rules enforced here:
  - HIGH needs >= min_signals_for_high independent signals (section 82)
  - a look/dwell below the minimum duration never becomes interest (section 56)
  - "Estimated" — this is behaviour, not a psychological fact (section 88)
"""

from __future__ import annotations

import dataclasses

from vision.interest.config import InterestConfig

LOW = "LOW_INTEREST"
POSSIBLE = "POSSIBLE_INTEREST"
HIGH = "HIGH_INTEREST"
UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class InterestObservation:
    timestamp: float
    distance_m: float
    speed_mps: float
    approaching: bool
    facing_score: float | None = None   # 0..1 from head pose / body orientation, or None
    gaze_score: float | None = None     # 0..1 from a real gaze model, or None
    interacting: bool = False


@dataclasses.dataclass
class InterestResult:
    status: str
    score: float
    look_duration: float
    dwell_duration: float
    signals_used: list[str]
    breakdown: dict[str, float]


class InterestSession:
    def __init__(self, global_id: str, target_id: str, config: InterestConfig | None = None):
        self.global_id = global_id
        self.target_id = target_id
        self.config = config or InterestConfig()
        self.look_start: float | None = None
        self.look_end: float | None = None
        self.first_seen: float | None = None
        self.last_seen: float | None = None
        self._near_time = 0.0
        self._prev_ts: float | None = None
        self._prev_distance: float | None = None
        self._interacted = False

    def observe(self, obs: InterestObservation) -> None:
        cfg = self.config
        if self.first_seen is None:
            self.first_seen = obs.timestamp
        self.last_seen = obs.timestamp

        near = obs.distance_m <= cfg.near_distance_m
        if near:
            if self._prev_ts is not None:
                self._near_time += max(0.0, obs.timestamp - self._prev_ts)
            if self.look_start is None:
                self.look_start = obs.timestamp
            self.look_end = obs.timestamp
        self._prev_ts = obs.timestamp
        self._prev_distance = obs.distance_m
        if obs.interacting:
            self._interacted = True

    @property
    def look_duration(self) -> float:
        if self.look_start is None or self.look_end is None:
            return 0.0
        return round(self.look_end - self.look_start, 3)

    @property
    def dwell_duration(self) -> float:
        return round(self._near_time, 3)

    def is_stale(self, now: float) -> bool:
        return self.last_seen is not None and (now - self.last_seen) > self.config.association_timeout_sec

    def result(self, latest: InterestObservation | None = None) -> InterestResult:
        cfg = self.config
        components: dict[str, tuple[float, float]] = {}

        look = self.look_duration
        dwell = self.dwell_duration
        if look >= cfg.min_look_duration_sec:
            components["look_duration"] = (min(1.0, look / 10.0), cfg.w_look_duration)
        if dwell >= cfg.min_dwell_duration_sec:
            components["stop"] = (min(1.0, dwell / 15.0), cfg.w_stop)

        if latest is not None:
            components["distance"] = (
                max(0.0, 1.0 - latest.distance_m / max(cfg.near_distance_m, 1e-6)) if latest.distance_m <= cfg.near_distance_m else 0.0,
                cfg.w_distance,
            )
            if latest.approaching:
                components["approach"] = (1.0, cfg.w_approach)
            if latest.facing_score is not None:
                components["head_pose"] = (latest.facing_score, cfg.w_head_pose)
            if latest.gaze_score is not None:
                components["gaze"] = (latest.gaze_score, cfg.w_gaze)
        if self._interacted:
            components["interaction"] = (1.0, cfg.w_interaction)

        if not components:
            return InterestResult(UNKNOWN, 0.0, look, dwell, [], {})

        total_w = sum(w for _s, w in components.values())
        score = sum(s * w for s, w in components.values()) / total_w if total_w else 0.0
        breakdown = {k: round(v[0], 3) for k, v in components.items()}
        signals = sorted(components)

        if score >= cfg.high_interest_threshold and len(signals) >= cfg.min_signals_for_high:
            status = HIGH
        elif score >= cfg.possible_interest_threshold:
            status = POSSIBLE
        else:
            status = LOW
        return InterestResult(status, round(score, 4), look, dwell, signals, breakdown)
