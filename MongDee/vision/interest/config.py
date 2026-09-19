"""Customer-interest estimation config (MongDee_Master_Prompt.md sections
54-57, 82).

Every weight and threshold here is a knob to tune on real footage. The
naming is deliberate: this estimates *behavioural* interest, it does not
read minds (section 54, 83, 88, 92 rule 6).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class InterestConfig:
    # a person within this distance (metres) of a product/zone is "near" it
    near_distance_m: float = 1.5
    # velocity below this (m/s) counts as "stopped"
    stop_speed_mps: float = 0.3
    # a look/dwell shorter than this is treated as walking past, not engaging
    min_look_duration_sec: float = 1.5
    min_dwell_duration_sec: float = 2.0
    # an association is dropped this long after the person was last near
    association_timeout_sec: float = 3.0

    # InterestScore = weighted(...) — renormalised over available signals
    w_gaze: float = 0.30
    w_head_pose: float = 0.15
    w_distance: float = 0.15
    w_look_duration: float = 0.15
    w_stop: float = 0.10
    w_approach: float = 0.10
    w_interaction: float = 0.05

    high_interest_threshold: float = 0.70
    possible_interest_threshold: float = 0.40
    # section 82: never HIGH from a single signal
    min_signals_for_high: int = 3

    @staticmethod
    def from_dict(d: dict) -> "InterestConfig":
        known = {f.name for f in dataclasses.fields(InterestConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"interest config has unknown field(s): {unknown}")
        return InterestConfig(**d)
