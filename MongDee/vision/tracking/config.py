"""Tracking configuration — Phase 3.

All thresholds are config, not hard-coded (MongDee_Master_Prompt.md
section 92 rule 10). Defaults follow the ByteTrack paper, adjusted for
booth-scale scenes and the ~2-8 fps effective rate the CPU pipeline runs
at (so `track_buffer` is in *frames*, and a smaller number of frames now
covers the same wall-clock occlusion window).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class TrackingConfig:
    track_high_thresh: float = 0.5   # detection score >= this -> first association pass
    track_low_thresh: float = 0.1    # score in [low, high) -> second (recovery) pass
    new_track_thresh: float = 0.6    # unmatched detection this confident -> start a new track
    match_thresh: float = 0.3        # min IoU to match in pass 1 (IoU, not IoU-distance)
    match_thresh_low: float = 0.4    # min IoU to match a low-conf detection in pass 2
    n_init: int = 3                  # consecutive hits before a track is CONFIRMED
    track_buffer: int = 30           # frames a lost track is kept before removal
    min_box_area: float = 64.0       # ignore detections smaller than this (px^2)

    @staticmethod
    def from_dict(d: dict) -> "TrackingConfig":
        known = {f.name for f in dataclasses.fields(TrackingConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"tracking config has unknown field(s): {unknown}")
        return TrackingConfig(**d)
