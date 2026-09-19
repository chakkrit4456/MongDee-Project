"""A Global Person — one physical individual, reconciled across cameras
(MongDee_Master_Prompt.md sections 12, 19, 26).

Holds the running appearance model (representative Re-ID embedding +
aggregated colours), where and when they've been seen, and which local
tracks were merged into them. Never stores raw face/biometric data — only
derived vectors (section 35).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision.attributes.extractor import PersonAttributes
from vision.reid.extractor import l2_normalize


@dataclasses.dataclass
class TrackSummary:
    """What one finished (or periodically-flushed) local track contributes."""

    camera_id: str
    local_track_id: int
    first_seen: float
    last_seen: float
    embedding: np.ndarray | None  # representative, L2-normalised
    embedding_samples: int
    attributes: PersonAttributes
    quality: float = 0.5  # 0..1 overall crop quality for this track


@dataclasses.dataclass
class CameraVisit:
    camera_id: str
    first_seen: float
    last_seen: float


class GlobalPerson:
    def __init__(self, global_id: str, summary: TrackSummary):
        self.global_id = global_id
        self.embedding = summary.embedding if summary.embedding is not None else None
        self._embedding_weight = float(summary.embedding_samples) if summary.embedding is not None else 0.0
        self.attributes = summary.attributes
        self.first_seen = summary.first_seen
        self.last_seen = summary.last_seen
        self.last_camera = summary.camera_id
        self.visits: list[CameraVisit] = [
            CameraVisit(summary.camera_id, summary.first_seen, summary.last_seen)
        ]
        self.track_refs: list[tuple[str, int]] = [(summary.camera_id, summary.local_track_id)]
        self.uncertain_links: list[str] = []  # other global_ids this one might actually be

    @property
    def cameras_seen(self) -> list[str]:
        seen = []
        for v in self.visits:
            if v.camera_id not in seen:
                seen.append(v.camera_id)
        return seen

    def merge(self, summary: TrackSummary) -> None:
        # weighted running mean of the embedding
        if summary.embedding is not None:
            if self.embedding is None:
                self.embedding = summary.embedding
                self._embedding_weight = float(summary.embedding_samples)
            else:
                w_new = float(summary.embedding_samples)
                blended = self.embedding * self._embedding_weight + summary.embedding * w_new
                self.embedding = l2_normalize(blended)
                self._embedding_weight = min(self._embedding_weight + w_new, 100.0)

        self._merge_attributes(summary.attributes)

        self.first_seen = min(self.first_seen, summary.first_seen)
        self.last_seen = max(self.last_seen, summary.last_seen)
        self.last_camera = summary.camera_id if summary.last_seen >= self.last_seen else self.last_camera
        self.track_refs.append((summary.camera_id, summary.local_track_id))

        # extend an existing visit to the same camera if it overlaps/adjoins, else add one
        for v in self.visits:
            if v.camera_id == summary.camera_id and summary.first_seen <= v.last_seen + 5.0:
                v.first_seen = min(v.first_seen, summary.first_seen)
                v.last_seen = max(v.last_seen, summary.last_seen)
                break
        else:
            self.visits.append(CameraVisit(summary.camera_id, summary.first_seen, summary.last_seen))
        self.visits.sort(key=lambda v: v.first_seen)

    def _merge_attributes(self, other: PersonAttributes) -> None:
        import dataclasses as _dc

        from vision.attributes.extractor import AttributeValue

        for f in _dc.fields(PersonAttributes):
            cur = getattr(self.attributes, f.name)
            new = getattr(other, f.name)
            if isinstance(cur, AttributeValue):
                if new.is_known and (not cur.is_known or new.confidence > cur.confidence):
                    setattr(self.attributes, f.name, new)

    def timeline(self) -> list[dict]:
        return [
            {"camera_id": v.camera_id, "first_seen": v.first_seen, "last_seen": v.last_seen}
            for v in self.visits
        ]

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "cameras_seen": self.cameras_seen,
            "attributes": self.attributes.to_dict(),
            "track_refs": [{"camera_id": c, "local_track_id": t} for c, t in self.track_refs],
            "uncertain_links": list(self.uncertain_links),
            "timeline": self.timeline(),
        }
