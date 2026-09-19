"""Light product tracker (MongDee_Master_Prompt.md section 48: "ห้ามสร้าง
Product Track ใหม่ทุก frame").

Products on a shelf barely move, so full Kalman/ByteTrack is overkill —
greedy IoU persistence with an age-out is enough to give each product a
stable local track id and smooth over one-frame detection dropouts.
"""

from __future__ import annotations

import dataclasses
import time

from vision.product.config import ProductConfig
from vision.tracking.matching import greedy_match, iou_matrix


@dataclasses.dataclass
class ProductTrack:
    track_id: int
    bbox: list[float]
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1

    @property
    def is_confirmed_at(self):
        return self.hits


class ProductTracker:
    """One per camera."""

    def __init__(self, config: ProductConfig | None = None):
        self.config = config or ProductConfig()
        self._tracks: dict[int, ProductTrack] = {}
        self._next_id = 1

    def update(self, boxes: list[list[float]], confidences: list[float], timestamp: float | None = None) -> list[ProductTrack]:
        if timestamp is None:
            timestamp = time.time()
        cfg = self.config

        track_ids = list(self._tracks)
        if track_ids and boxes:
            iou = iou_matrix([self._tracks[t].bbox for t in track_ids], boxes)
            matches, un_tracks, un_boxes = greedy_match(iou, cfg.track_match_iou)
        else:
            matches, un_tracks, un_boxes = [], list(range(len(track_ids))), list(range(len(boxes)))

        for ti, bi in matches:
            tr = self._tracks[track_ids[ti]]
            tr.bbox = boxes[bi]
            tr.confidence = confidences[bi]
            tr.last_seen = timestamp
            tr.hits += 1

        for bi in un_boxes:
            self._tracks[self._next_id] = ProductTrack(
                self._next_id, boxes[bi], confidences[bi], timestamp, timestamp
            )
            self._next_id += 1

        # age out
        stale = [tid for tid, tr in self._tracks.items() if timestamp - tr.last_seen > cfg.track_max_age_sec]
        for tid in stale:
            del self._tracks[tid]

        return [tr for tr in self._tracks.values() if tr.hits >= cfg.track_min_hits and tr.last_seen == timestamp]

    def confirmed_tracks(self) -> list[ProductTrack]:
        return [tr for tr in self._tracks.values() if tr.hits >= self.config.track_min_hits]
