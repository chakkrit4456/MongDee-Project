"""BoothAnalytics — the spatial / product / interest layer that sits on
top of the person pipeline (MongDee_Master_Prompt.md sections 58, 67-69,
79, 86).

Feed it, per processed frame:
  - the confirmed person tracks (with their Global Person IDs once known)
  - the product detections for that camera

and it maintains:
  - each person's world position + movement path (2D booth map)
  - traffic / dwell / interest heatmaps
  - per (person, zone/product) interest sessions -> interest events
  - person <-> product associations

Everything degrades gracefully: no calibration for a camera -> that
camera contributes detections/tracks but no world position; no layout ->
no zones/associations/interest, just heatmap-less positions.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time

from vision.interest.association import AssociationEngine
from vision.interest.config import InterestConfig
from vision.interest.estimator import InterestObservation, InterestResult, InterestSession
from vision.spatial.booth import BoothLayout
from vision.spatial.calibration import CalibrationStore
from vision.spatial.heatmap import HeatmapAccumulator, MovementPath
from vision.spatial.world import WorldMapper, WorldPosition

logger = logging.getLogger("mongdee.vision.booth")


@dataclasses.dataclass
class InterestEvent:
    event_type: str            # CUSTOMER_INTEREST_STARTED / UPDATED / ENDED
    global_id: str
    target_id: str
    timestamp: float
    status: str
    score: float
    look_duration: float
    dwell_duration: float
    signals: list[str]


class BoothAnalytics:
    def __init__(
        self,
        layout: BoothLayout | None,
        calibrations: CalibrationStore,
        interest_config: InterestConfig | None = None,
        heatmap_cell_size: float = 0.25,
        on_interest_event=None,
    ):
        self.layout = layout
        self._calib = calibrations
        self._cfg = interest_config or InterestConfig()
        self._mapper = WorldMapper(calibrations, layout)
        self._assoc = AssociationEngine(layout, self._cfg) if layout is not None else None
        self._on_interest_event = on_interest_event

        w = layout.width if layout else 10.0
        length = layout.length if layout else 6.0
        self._traffic = HeatmapAccumulator(w, length, heatmap_cell_size)
        self._dwell = HeatmapAccumulator(w, length, heatmap_cell_size)
        self._interest_hm = HeatmapAccumulator(w, length, heatmap_cell_size)

        self._paths: dict[str, MovementPath] = {}
        self._sessions: dict[tuple[str, str], InterestSession] = {}
        self._positions: dict[str, WorldPosition] = {}
        self._lock = threading.Lock()

    # -- ingest -------------------------------------------------------------
    def observe_person(self, camera_id: str, global_id: str, bbox: list[float], timestamp: float | None = None):
        if timestamp is None:
            timestamp = time.time()
        pos = self._mapper.locate(camera_id, bbox, global_id=global_id, timestamp=timestamp)
        if pos is None:
            return None

        with self._lock:
            self._positions[global_id] = pos
            path = self._paths.setdefault(global_id, MovementPath(global_id))
            path.add(pos.x, pos.y, timestamp, camera_id, pos.zone_id, pos.confidence)

            self._traffic.add(pos.x, pos.y, timestamp=timestamp)
            if len(path.points) >= 2:
                dt = path.points[-1]["timestamp"] - path.points[-2]["timestamp"]
                if 0 < dt < 5.0:
                    self._dwell.add(pos.x, pos.y, weight=dt, timestamp=timestamp)

            if self._assoc is not None:
                self._update_interest(global_id, pos, timestamp)
        return pos

    def _update_interest(self, global_id: str, pos: WorldPosition, timestamp: float) -> None:
        associations = self._assoc.associate(global_id, pos.x, pos.y, timestamp)
        for a in associations:
            key = (global_id, a.target_id)
            session = self._sessions.get(key)
            is_new = session is None
            if is_new:
                session = InterestSession(global_id, a.target_id, self._cfg)
                self._sessions[key] = session

            obs = InterestObservation(
                timestamp=timestamp, distance_m=a.distance_m, speed_mps=a.speed_mps,
                approaching=a.approaching, facing_score=a.facing_score,
            )
            session.observe(obs)
            result = session.result(obs)
            self._interest_hm.add(pos.x, pos.y, weight=max(0.0, result.score), timestamp=timestamp)
            self._emit_interest(
                "CUSTOMER_INTEREST_STARTED" if is_new else "CUSTOMER_INTEREST_UPDATED",
                session, result, timestamp,
            )

        # end stale sessions
        for key, session in list(self._sessions.items()):
            if key[0] == global_id and session.is_stale(timestamp):
                result = session.result()
                self._emit_interest("CUSTOMER_INTEREST_ENDED", session, result, timestamp)
                del self._sessions[key]

    def _emit_interest(self, event_type: str, session: InterestSession, result: InterestResult, timestamp: float) -> None:
        if self._on_interest_event is None:
            return
        try:
            self._on_interest_event(
                InterestEvent(
                    event_type=event_type, global_id=session.global_id, target_id=session.target_id,
                    timestamp=timestamp, status=result.status, score=result.score,
                    look_duration=result.look_duration, dwell_duration=result.dwell_duration,
                    signals=result.signals_used,
                )
            )
        except Exception:
            logger.exception("on_interest_event callback raised")

    def forget_person(self, global_id: str) -> None:
        with self._lock:
            self._mapper.forget(global_id)
            if self._assoc is not None:
                self._assoc.forget(global_id)
            for key in [k for k in self._sessions if k[0] == global_id]:
                del self._sessions[key]

    # -- read ---------------------------------------------------------
    def positions(self) -> dict[str, WorldPosition]:
        with self._lock:
            return dict(self._positions)

    def movement_path(self, global_id: str) -> MovementPath | None:
        with self._lock:
            return self._paths.get(global_id)

    def heatmap(self, kind: str = "traffic", since: float | None = None) -> list[dict]:
        hm = {"traffic": self._traffic, "dwell": self._dwell, "interest": self._interest_hm}.get(kind)
        if hm is None:
            raise ValueError(f"unknown heatmap kind {kind!r}")
        with self._lock:
            grid = hm.grid(since=since, normalize=True)
        return {
            "kind": kind, "cell_size": hm.cell_size, "cols": hm.cols, "rows": hm.rows,
            "grid": grid.round(4).tolist(),
        }

    def active_interest(self) -> list[dict]:
        with self._lock:
            out = []
            for (gid, target), session in self._sessions.items():
                r = session.result()
                out.append({
                    "global_id": gid, "target_id": target, "status": r.status, "score": r.score,
                    "look_duration": r.look_duration, "dwell_duration": r.dwell_duration,
                })
            return out
