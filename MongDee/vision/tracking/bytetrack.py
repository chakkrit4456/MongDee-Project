"""ByteTrack-style multi-object tracker for one camera
(MongDee_Master_Prompt.md section 6).

The idea that makes ByteTrack better than plain SORT: don't throw away
low-confidence detections. First match tracks against high-confidence
detections; then take the tracks still unmatched and try to match them
against the *low*-confidence detections too. A partially-occluded person
often only produces a weak detection for a few frames — this second pass
keeps their ID alive instead of dropping the track and re-numbering them
when they reappear.

Kalman prediction (vision/tracking/kalman.py) carries a track's position
forward through frames where it is missed entirely, so short occlusions
(up to `max_age` frames) don't break the ID.

Long re-entries (person leaves the frame, comes back much later) are NOT
solved here by design — motion prediction can't bridge that. That needs
appearance re-identification, which Phase 5 adds; this tracker exposes a
hook (`appearance_gate`) for it.
"""

from __future__ import annotations

from typing import Callable

from vision.detection.detector import Detection
from vision.tracking.config import TrackingConfig
from vision.tracking.kalman import KalmanBoxFilter
from vision.tracking.matching import greedy_match, iou_matrix
from vision.tracking.track import Track, TrackState

AppearanceGate = Callable[[Track, Detection], bool]


class ByteTracker:
    """One instance per camera. Call update() once per processed frame."""

    def __init__(self, config: TrackingConfig | None = None, appearance_gate: AppearanceGate | None = None):
        self.config = config or TrackingConfig()
        self._kf = KalmanBoxFilter()
        self._tracks: list[Track] = []
        self._next_id = 1
        self._appearance_gate = appearance_gate

    @property
    def tracks(self) -> list[Track]:
        """All live tracks (confirmed + lost + tentative), newest state."""
        return list(self._tracks)

    def active_tracks(self) -> list[Track]:
        """Tracks worth reporting downstream: confirmed and seen this frame."""
        return [t for t in self._tracks if t.state == TrackState.CONFIRMED and t.time_since_update == 0]

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        cfg = self.config

        dets = [d for d in detections if _area(d.bbox) >= cfg.min_box_area]
        high = [d for d in dets if d.confidence >= cfg.track_high_thresh]
        low = [d for d in dets if cfg.track_low_thresh <= d.confidence < cfg.track_high_thresh]

        for track in self._tracks:
            track.predict()

        # --- pass 1: (confirmed | lost | tentative) vs high-confidence detections
        pool_idx = list(range(len(self._tracks)))
        matches1, un_tracks1, un_high = self._associate(pool_idx, high, cfg.match_thresh)
        for ti, di in matches1:
            self._tracks[ti].update(high[di].bbox, high[di].confidence, timestamp)
            high[di].local_track_id = self._tracks[ti].track_id

        # --- pass 2: tracks still unmatched (only confirmed/lost) vs low-confidence detections
        second_pool = [
            ti for ti in un_tracks1 if self._tracks[ti].state in (TrackState.CONFIRMED, TrackState.LOST)
        ]
        matches2, un_tracks2, _un_low = self._associate(second_pool, low, cfg.match_thresh_low)
        for ti, di in matches2:
            self._tracks[ti].update(low[di].bbox, low[di].confidence, timestamp)
            low[di].local_track_id = self._tracks[ti].track_id

        # --- tracks that matched nothing this frame
        matched_track_ids = {ti for ti, _ in matches1} | {ti for ti, _ in matches2}
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.mark_missed()

        # --- unmatched high detections spawn new tentative tracks
        for di in un_high:
            det = high[di]
            if det.confidence >= cfg.new_track_thresh:
                track = Track(
                    self._next_id, det.bbox, det.confidence, timestamp, self._kf,
                    n_init=cfg.n_init, max_age=cfg.track_buffer,
                )
                det.local_track_id = self._next_id
                self._next_id += 1
                self._tracks.append(track)

        self._tracks = [t for t in self._tracks if t.state != TrackState.REMOVED]
        return self.active_tracks()

    def _associate(self, track_indices, dets, thresh):
        if not track_indices or not dets:
            return [], list(track_indices), list(range(len(dets)))
        track_bboxes = [self._tracks[ti].bbox for ti in track_indices]
        det_bboxes = [d.bbox for d in dets]
        iou = iou_matrix(track_bboxes, det_bboxes)

        if self._appearance_gate is not None:
            for r, ti in enumerate(track_indices):
                for c, det in enumerate(dets):
                    if iou[r, c] >= thresh and not self._appearance_gate(self._tracks[ti], det):
                        iou[r, c] = 0.0

        matches, un_rows, un_cols = greedy_match(iou, thresh)
        matches = [(track_indices[r], c) for r, c in matches]
        un_rows = [track_indices[r] for r in un_rows]
        return matches, un_rows, un_cols


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
