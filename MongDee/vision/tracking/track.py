"""A single person track within one camera (MongDee_Master_Prompt.md
section 6). Owns its Kalman filter and its lifecycle state.

Lifecycle:
    TENTATIVE  just created, not yet confirmed (needs `n_init` hits in a row)
    CONFIRMED  a real track, currently matched frame-to-frame
    LOST       confirmed but currently unmatched (occluded / left frame);
               kept for up to `max_age` frames so it can be re-matched
    REMOVED    aged out — will be dropped by the tracker

`local_track_id` is unique per camera and per tracker run. The same
physical person in two cameras gets two unrelated local ids — reconciling
those into one Global Person ID is Phase 6's job, not this module's.
"""

from __future__ import annotations

import enum

import numpy as np

from vision.tracking.kalman import KalmanBoxFilter, bbox_to_measurement, mean_to_bbox


class TrackState(str, enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class Track:
    def __init__(
        self,
        track_id: int,
        bbox: list[float],
        confidence: float,
        timestamp: float,
        kf: KalmanBoxFilter,
        n_init: int = 3,
        max_age: int = 30,
    ):
        self.track_id = track_id
        self.confidence = confidence
        self.first_seen = timestamp
        self.last_seen = timestamp

        self._kf = kf
        self.mean, self.covariance = kf.initiate(bbox_to_measurement(bbox))

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self._n_init = n_init
        self._max_age = max_age
        # confirmed as soon as hits >= n_init (so n_init=1 confirms on creation)
        self.state = TrackState.CONFIRMED if self.hits >= n_init else TrackState.TENTATIVE

    # -- geometry -------------------------------------------------------------
    @property
    def bbox(self) -> list[float]:
        return mean_to_bbox(self.mean)

    @property
    def center(self) -> tuple[float, float]:
        return float(self.mean[0]), float(self.mean[1])

    # -- lifecycle ----------------------------------------------------------
    def predict(self) -> None:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: list[float], confidence: float, timestamp: float) -> None:
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, bbox_to_measurement(bbox)
        )
        self.confidence = confidence
        self.last_seen = timestamp
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Called for a confirmed/tentative track that matched nothing this
        frame. A tentative track is dropped immediately; a confirmed track
        goes LOST and is kept until it exceeds max_age missed frames."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.REMOVED
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    # -- predicates -------------------------------------------------------
    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED
