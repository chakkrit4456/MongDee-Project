"""Multi-feature identity matching (MongDee_Master_Prompt.md sections 10,
13, 38, 39).

Given a finished local track (TrackSummary) and the current set of Global
Persons, score the track against each candidate using several independent
appearance + space + time signals, combine them as a weighted average
over *whichever signals are actually available*, and return one of:

    MATCH        best score >= match_threshold      -> merge into that person
    UNCERTAIN    uncertain_threshold <= best < match -> new person, link recorded
    NEW_PERSON   best score < uncertain_threshold    -> brand new person

Rule (section 38): never MATCH on a single signal. A candidate whose score
comes almost entirely from one component is capped below the match
threshold.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision.identity.config import IdentityConfig
from vision.identity.global_person import GlobalPerson, TrackSummary
from vision.identity.topology import CameraTopology
from vision.reid.extractor import cosine_similarity

MATCH = "match"
UNCERTAIN = "uncertain"
NEW_PERSON = "new_person"

_MAX_RGB_DIST = float(np.linalg.norm([255, 255, 255]))


@dataclasses.dataclass
class MatchResult:
    status: str
    score: float
    candidate_id: str | None
    breakdown: dict[str, float]


def _clothing_score(a, b) -> float | None:
    parts = []
    for attr_rgb in ("shirt_rgb", "pants_rgb"):
        ra = getattr(a, attr_rgb)
        rb = getattr(b, attr_rgb)
        if ra != (0, 0, 0) and rb != (0, 0, 0):
            dist = float(np.linalg.norm(np.array(ra, float) - np.array(rb, float)))
            parts.append(1.0 - min(1.0, dist / (_MAX_RGB_DIST * 0.5)))
    return sum(parts) / len(parts) if parts else None


def _body_score(a, b) -> float | None:
    if a.aspect_ratio > 0 and b.aspect_ratio > 0:
        return 1.0 - min(1.0, abs(a.aspect_ratio - b.aspect_ratio) / 0.3)
    return None


class IdentityMatcher:
    def __init__(self, config: IdentityConfig | None = None, topology: CameraTopology | None = None):
        self.config = config or IdentityConfig()
        self.topology = topology or CameraTopology()

    def score(self, person: GlobalPerson, summary: TrackSummary) -> tuple[float, dict[str, float]]:
        cfg = self.config
        components: dict[str, tuple[float, float]] = {}  # name -> (score, weight)

        if (
            person.embedding is not None
            and summary.embedding is not None
            and summary.embedding_samples >= cfg.min_reid_samples
        ):
            reid = max(0.0, cosine_similarity(person.embedding, summary.embedding))
            components["reid"] = (reid, cfg.reid_weight)

        clothing = _clothing_score(person.attributes, summary.attributes)
        if clothing is not None:
            components["clothing"] = (clothing, cfg.clothing_weight)

        body = _body_score(person.attributes, summary.attributes)
        if body is not None:
            components["body"] = (body, cfg.body_weight)

        dt = summary.first_seen - person.last_seen
        temporal = self.topology.temporal_feasibility(person.last_camera, summary.camera_id, dt)
        spatial = self.topology.spatial_feasibility(person.last_camera, summary.camera_id)
        components["temporal"] = (temporal, cfg.temporal_weight)
        components["spatial"] = (spatial, cfg.spatial_weight)

        breakdown = {name: round(s, 3) for name, (s, _w) in components.items()}

        # hard physical rejects
        if cfg.hard_temporal_reject and temporal <= 0.0:
            return 0.0, {**breakdown, "rejected": 1.0}
        if cfg.hard_spatial_reject and spatial < 0.2:
            return 0.0, {**breakdown, "rejected": 1.0}

        total_w = sum(w for _s, w in components.values())
        if total_w <= 0:
            return 0.0, breakdown
        combined = sum(s * w for s, w in components.values()) / total_w

        # section 38: don't let one appearance signal carry a match by itself
        appearance = {k: v for k, v in components.items() if k in ("reid", "clothing", "body")}
        if len(appearance) < 2:
            combined = min(combined, cfg.match_threshold - 0.01)

        return round(combined, 4), breakdown

    def match(self, summary: TrackSummary, candidates: list[GlobalPerson]) -> MatchResult:
        cfg = self.config
        best_score = 0.0
        best_id: str | None = None
        best_breakdown: dict[str, float] = {}

        for person in candidates:
            if (summary.camera_id, summary.local_track_id) in person.track_refs:
                continue
            s, breakdown = self.score(person, summary)
            if s > best_score:
                best_score, best_id, best_breakdown = s, person.global_id, breakdown

        if best_score >= cfg.match_threshold:
            return MatchResult(MATCH, best_score, best_id, best_breakdown)
        if best_score >= cfg.uncertain_threshold:
            return MatchResult(UNCERTAIN, best_score, best_id, best_breakdown)
        return MatchResult(NEW_PERSON, best_score, best_id, best_breakdown)
