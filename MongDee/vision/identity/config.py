"""Global identity matching config — Phase 6
(MongDee_Master_Prompt.md sections 10, 13, 38, 39).

Weights are per *evidence component*. Components with no evidence for a
given comparison (e.g. no face, no hair model) are dropped and the
remaining weights renormalised, rather than silently capping the max
achievable score — this is the section 24 idea ("reduce the weight of a
feature you can't trust") applied to missing features.

Thresholds follow section 13's three-way split and MUST be tuned on real
validation data before production (section 13, section 39, section 92
rule 10).
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class IdentityConfig:
    # component weights (relative; renormalised over whatever is available)
    reid_weight: float = 0.40
    clothing_weight: float = 0.20
    body_weight: float = 0.10
    face_weight: float = 0.10
    hair_weight: float = 0.10
    temporal_weight: float = 0.05
    spatial_weight: float = 0.05

    # section 13 suggests 0.85 / 0.65; kept slightly lower here because the
    # default Re-ID backend (ImageNet features) is weaker than a trained
    # Re-ID model. MUST be re-tuned on real footage before production — err
    # toward NOT merging (section 47: never merge the wrong people).
    match_threshold: float = 0.80       # >= this -> MATCH (merge)
    uncertain_threshold: float = 0.62   # [uncertain, match) -> UNCERTAIN (new person, flagged link)

    # a match is rejected outright if the topology says the move was
    # physically impossible, regardless of appearance score
    hard_temporal_reject: bool = True
    hard_spatial_reject: bool = False   # off by default: topology is often incomplete

    person_ttl_sec: float = 900.0       # drop a global person not seen this long (0 = never)
    min_reid_samples: int = 1           # a track needs at least this many good embeddings to match

    @staticmethod
    def from_dict(d: dict) -> "IdentityConfig":
        known = {f.name for f in dataclasses.fields(IdentityConfig)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"identity config has unknown field(s): {unknown}")
        return IdentityConfig(**d)
