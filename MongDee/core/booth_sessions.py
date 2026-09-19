"""Booth Session tracking — one continuous ENTRY -> EXIT visit per identity,
spanning however many cameras/tripwires that identity is seen at in between.

Spec: MongDee_Master_Prompt_Accurate_Person_Counting_ReID.md section 29
("Tripwire กับ Global Identity" — a tripwire crossing's local_track_id must
be mapped to global_person_id so "CAM-01 Track-17" and "CAM-02 Track-4"
resolving to the same P103 are known to be the same person) and section 30
("Double Count Prevention" — CAM-A P103 IN then CAM-B P103 IN must not add a
second Unique Person / re-open a second visit, while still keeping each
camera's own observation). This module is what turns that mapping into an
actual dwell-time visit record.

Deliberately layered on top of, not merged into, core.tripwire.TripwireCounter
and core.reid.GlobalIdentityRegistry: TripwireCounter already turns per-camera
foot-point crossings into debounced ENTRY/EXIT direction events (it has no
notion of Global Person identity — its track_id is camera-local).
GlobalIdentityRegistry already resolves that camera-local track_id to a
Global Person id. This module is the third layer: given an already-resolved
identity string (web/booth_manager.py passes a Global Person id when Re-ID
is enabled and has resolved that track, else a synthetic per-camera fallback
key — see booth_manager._update_tripwire) and an ENTRY/EXIT direction, it
owns exactly one open BoothSession per identity so:

  - a second ENTRY for someone already inside is a no-op other than
    recording the camera as seen (section 30's double-count prevention) —
    it never restarts entered_at or opens a second session;
  - an EXIT with no open session for that identity is also a no-op —
    nothing to close, never invents a session just to immediately close it
    (e.g. a stale/duplicate event, or an EXIT for an identity whose ENTRY
    was never resolved because Re-ID hadn't matched yet);
  - dwell_seconds is always computed from real entered_at/exited_at
    timestamps, never frame counts.

No camera I/O, no database — same shape as core.tripwire and
core.interest_tracker. web/booth_manager.py owns persistence.
"""

from __future__ import annotations

import dataclasses
import itertools

_id_counter = itertools.count(1)


@dataclasses.dataclass
class BoothSession:
    session_id: str
    person_key: str
    entered_at: float
    cameras_seen: set = dataclasses.field(default_factory=set)
    exited_at: float | None = None
    dwell_seconds: float | None = None

    @property
    def status(self) -> str:
        return "closed" if self.exited_at is not None else "active"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "person_key": self.person_key,
            "entered_at": self.entered_at,
            "exited_at": self.exited_at,
            "dwell_seconds": self.dwell_seconds,
            "cameras_seen": sorted(self.cameras_seen),
            "status": self.status,
        }


class BoothSessionManager:
    """One instance per booth. Tracks at most one open BoothSession per
    `person_key` at a time."""

    def __init__(self):
        self._open: dict[str, BoothSession] = {}

    def on_entry(self, person_key: str, camera_id: str, ts: float) -> BoothSession | None:
        """Returns the newly-opened BoothSession, or None if `person_key`
        already had one open (duplicate ENTRY, e.g. a second camera's
        tripwire firing for the same visit — spec section 30) — still
        records this camera as seen either way."""
        existing = self._open.get(person_key)
        if existing is not None:
            existing.cameras_seen.add(camera_id)
            return None
        session = BoothSession(
            session_id=f"BS-{next(_id_counter):08d}",
            person_key=person_key,
            entered_at=ts,
            cameras_seen={camera_id},
        )
        self._open[person_key] = session
        return session

    def on_exit(self, person_key: str, camera_id: str, ts: float) -> BoothSession | None:
        """Returns the just-closed BoothSession, or None if nothing was open
        for `person_key` (nothing to close)."""
        session = self._open.pop(person_key, None)
        if session is None:
            return None
        session.cameras_seen.add(camera_id)
        session.exited_at = ts
        session.dwell_seconds = ts - session.entered_at
        return session

    def note_camera_seen(self, person_key: str, camera_id: str) -> bool:
        """Records that `person_key`'s open session (if any) was also
        observed at `camera_id` — called from Re-ID resolution, not just
        tripwire crossings, so switching cameras mid-visit without ever
        tripping a second tripwire still shows up in cameras_seen. Returns
        True only when this added a *new* camera (the caller's cue to
        persist the change — calling this every frame a track is visible
        would otherwise write to the DB far more often than needed)."""
        session = self._open.get(person_key)
        if session is None or camera_id in session.cameras_seen:
            return False
        session.cameras_seen.add(camera_id)
        return True

    def rekey(self, old_key: str, new_key: str) -> BoothSession | None:
        """Two identities were merged into one person (core.reid). An open session under `old_key` moves to
        `new_key`; if `new_key` already has an open session the duplicate is dropped and RETURNED so the caller
        can close its database row. Returns None when nothing had to be dropped."""
        session = self._open.pop(old_key, None)
        if session is None:
            return None
        if new_key in self._open:
            self._open[new_key].cameras_seen |= session.cameras_seen
            return session
        session.person_key = new_key
        self._open[new_key] = session
        return None

    def get_open_session(self, person_key: str) -> BoothSession | None:
        return self._open.get(person_key)

    def active_count(self) -> int:
        return len(self._open)

    def all_open_sessions(self) -> list[BoothSession]:
        return list(self._open.values())
