"""Person-level events (MongDee_Master_Prompt.md section 27).

Emitted by the pipeline as tracks resolve to Global Person IDs. The
Database / API / Dashboard layers (Phases 7-9) consume these.
"""

from __future__ import annotations

import dataclasses
import time

PERSON_NEW = "NEW_PERSON"
PERSON_REIDENTIFIED = "PERSON_REIDENTIFIED"
POSSIBLE_MATCH = "POSSIBLE_MATCH"
PERSON_CAMERA_TRANSITION = "PERSON_CAMERA_TRANSITION"
PERSON_TRACK_ENDED = "PERSON_TRACK_ENDED"


@dataclasses.dataclass
class PersonEvent:
    event_type: str
    timestamp: float
    camera_id: str
    global_id: str
    local_track_id: int
    match_score: float = 0.0
    payload: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def now(**kwargs) -> "PersonEvent":
        kwargs.setdefault("timestamp", time.time())
        return PersonEvent(**kwargs)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "global_id": self.global_id,
            "local_track_id": self.local_track_id,
            "match_score": round(self.match_score, 4),
            "payload": self.payload,
        }
