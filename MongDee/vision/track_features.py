"""TrackFeatureStore — accumulates per-track appearance evidence
(Re-ID embeddings + attributes) while a local track is alive, then hands a
single TrackSummary to the Global Identity layer when the track ends.

This is where "keyframe sampling + track-level aggregation" from section
23 actually happens: observe() is rate-limited, and only good-quality
crops contribute (the Re-ID quality gate, section 24).
"""

from __future__ import annotations

import dataclasses
import threading

import numpy as np

from vision.attributes.aggregator import TrackAttributeAggregator
from vision.attributes.extractor import AttributeExtractor
from vision.identity.global_person import TrackSummary
from vision.reid.extractor import ReIDExtractor, TrackEmbeddingAggregator
from vision.tracking.track import Track


@dataclasses.dataclass(frozen=True)
class FeatureStoreConfig:
    observe_interval_sec: float = 0.5   # don't re-embed the same track faster than this
    max_reid_samples: int = 8
    max_attr_samples: int = 30
    min_samples_to_flush: int = 1       # a track needs at least this many good embeddings to be submitted
    track_end_timeout_sec: float = 3.0  # flush a track this long after it was last observed (camera went quiet)
    eager_submit_samples: int = 3       # submit a still-live track for a Global ID once it has this many samples (0 = only on end)

    @staticmethod
    def from_dict(d: dict) -> "FeatureStoreConfig":
        known = {f.name for f in dataclasses.fields(FeatureStoreConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"features config has unknown field(s): {unknown}")
        return FeatureStoreConfig(**d)


@dataclasses.dataclass
class _Entry:
    first_seen: float
    last_seen: float
    last_observe: float
    reid: TrackEmbeddingAggregator
    attrs: TrackAttributeAggregator
    quality_sum: float = 0.0
    quality_n: int = 0


class TrackFeatureStore:
    def __init__(
        self,
        reid_extractor: ReIDExtractor,
        attribute_extractor: AttributeExtractor,
        config: FeatureStoreConfig | None = None,
    ):
        self._reid = reid_extractor
        self._attrs = attribute_extractor
        self._config = config or FeatureStoreConfig()
        self._entries: dict[tuple[str, int], _Entry] = {}
        self._lock = threading.Lock()

    def observe(self, camera_id: str, track: Track, image: np.ndarray, timestamp: float) -> None:
        key = (camera_id, track.track_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _Entry(
                    first_seen=timestamp, last_seen=timestamp, last_observe=-1e9,
                    reid=TrackEmbeddingAggregator(self._config.max_reid_samples),
                    attrs=TrackAttributeAggregator(self._config.max_attr_samples),
                )
                self._entries[key] = entry
            entry.last_seen = timestamp
            due = timestamp - entry.last_observe >= self._config.observe_interval_sec
        if not due:
            return

        vec, quality = self._reid.embed(image, track.bbox)
        attrs = self._attrs.extract(image, track.bbox)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.last_observe = timestamp
            if vec is not None:
                entry.reid.add(vec, quality.score)
                entry.quality_sum += quality.score
                entry.quality_n += 1
            entry.attrs.add(attrs)

    def build_summary(self, camera_id: str, track_id: int) -> TrackSummary | None:
        with self._lock:
            entry = self._entries.get((camera_id, track_id))
            if entry is None:
                return None
            rep = entry.reid.representative()
            quality = entry.quality_sum / entry.quality_n if entry.quality_n else 0.4
            return TrackSummary(
                camera_id=camera_id,
                local_track_id=track_id,
                first_seen=entry.first_seen,
                last_seen=entry.last_seen,
                embedding=rep,
                embedding_samples=entry.reid.sample_count,
                attributes=entry.attrs.result(),
                quality=round(quality, 3),
            )

    def pop(self, camera_id: str, track_id: int) -> TrackSummary | None:
        summary = self.build_summary(camera_id, track_id)
        with self._lock:
            self._entries.pop((camera_id, track_id), None)
        return summary

    def track_ids_for_camera(self, camera_id: str) -> set[int]:
        with self._lock:
            return {tid for (cid, tid) in self._entries if cid == camera_id}

    def last_seen(self, camera_id: str, track_id: int) -> float | None:
        with self._lock:
            entry = self._entries.get((camera_id, track_id))
            return entry.last_seen if entry else None

    def camera_ids(self) -> set[str]:
        with self._lock:
            return {cid for (cid, _tid) in self._entries}

    def reid_sample_count(self, camera_id: str, track_id: int) -> int:
        with self._lock:
            entry = self._entries.get((camera_id, track_id))
            return entry.reid.sample_count if entry else 0

    @property
    def min_samples_to_flush(self) -> int:
        return self._config.min_samples_to_flush

    @property
    def track_end_timeout_sec(self) -> float:
        return self._config.track_end_timeout_sec

    @property
    def eager_submit_samples(self) -> int:
        return self._config.eager_submit_samples
