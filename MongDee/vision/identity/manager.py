"""GlobalIdentityManager — owns the set of Global Persons and the
one-track-at-a-time reconciliation loop (MongDee_Master_Prompt.md
sections 12-16, 22, 47).

submit_track(summary):
  MATCH      -> merge into the winning person, return its id
  UNCERTAIN  -> create a NEW person, but record a link both ways (section 13:
               "ในกรณีคะแนนต่ำ ห้าม merge" — no automatic merge on a weak score)
  NEW_PERSON -> create a new person

unique_count() is just len(persons) — the Global-ID aggregation the master
prompt wants instead of summing per-camera detections (section 16).
"""

from __future__ import annotations

import threading

from vision.identity.config import IdentityConfig
from vision.identity.global_person import GlobalPerson, TrackSummary
from vision.identity.matcher import MATCH, NEW_PERSON, UNCERTAIN, IdentityMatcher, MatchResult
from vision.identity.topology import CameraTopology


class GlobalIdentityManager:
    def __init__(
        self,
        config: IdentityConfig | None = None,
        topology: CameraTopology | None = None,
        id_prefix: str = "PERSON",
    ):
        self.config = config or IdentityConfig()
        self._matcher = IdentityMatcher(self.config, topology or CameraTopology())
        self._persons: dict[str, GlobalPerson] = {}
        self._by_track: dict[tuple[str, int], str] = {}
        self._next_num = 1
        self._id_prefix = id_prefix
        self._lock = threading.Lock()

    # -- public --------------------------------------------------------------
    def submit_track(self, summary: TrackSummary) -> tuple[str, MatchResult]:
        with self._lock:
            self._prune(summary.last_seen)

            existing = self._by_track.get((summary.camera_id, summary.local_track_id))
            if existing and existing in self._persons:
                self._persons[existing].merge(summary)
                return existing, MatchResult(MATCH, 1.0, existing, {"same_track": 1.0})

            result = self._matcher.match(summary, list(self._persons.values()))

            if result.status == MATCH and result.candidate_id in self._persons:
                person = self._persons[result.candidate_id]
                person.merge(summary)
                self._by_track[(summary.camera_id, summary.local_track_id)] = person.global_id
                return person.global_id, result

            new_person = self._create(summary)
            if result.status == UNCERTAIN and result.candidate_id in self._persons:
                new_person.uncertain_links.append(result.candidate_id)
                self._persons[result.candidate_id].uncertain_links.append(new_person.global_id)
            return new_person.global_id, result

    def persons(self) -> list[GlobalPerson]:
        with self._lock:
            return list(self._persons.values())

    def get(self, global_id: str) -> GlobalPerson | None:
        with self._lock:
            return self._persons.get(global_id)

    def global_id_for_track(self, camera_id: str, local_track_id: int) -> str | None:
        """The Global Person ID a local track has been assigned to, if any
        (set once the track has been submitted — see feature-store eager
        submit). Lets the live booth map show a person before their track
        has ended."""
        with self._lock:
            gid = self._by_track.get((camera_id, local_track_id))
            return gid if gid in self._persons else None

    def unique_count(self) -> int:
        with self._lock:
            return len(self._persons)

    def camera_person_counts(self) -> dict[str, int]:
        """How many distinct global persons each camera has seen."""
        with self._lock:
            counts: dict[str, int] = {}
            for p in self._persons.values():
                for cam in p.cameras_seen:
                    counts[cam] = counts.get(cam, 0) + 1
            return counts

    # -- internal -----------------------------------------------------------
    def _create(self, summary: TrackSummary) -> GlobalPerson:
        gid = f"{self._id_prefix}-{self._next_num:04d}"
        self._next_num += 1
        person = GlobalPerson(gid, summary)
        self._persons[gid] = person
        self._by_track[(summary.camera_id, summary.local_track_id)] = gid
        return person

    def _prune(self, now: float) -> None:
        ttl = self.config.person_ttl_sec
        if ttl <= 0:
            return
        stale = [gid for gid, p in self._persons.items() if now - p.last_seen > ttl]
        for gid in stale:
            person = self._persons.pop(gid)
            for ref in person.track_refs:
                if self._by_track.get(ref) == gid:
                    del self._by_track[ref]
