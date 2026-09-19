"""Camera topology — the Camera Transition Matrix (section 14) and Camera
Graph (section 15) that keep the matcher from linking people across
physically impossible moves.

transitions:  (from_camera, to_camera) -> (min_seconds, max_seconds)
              how long it can take a person to walk between the two views.
graph:        which cameras are directly reachable from which (adjacency).
locations:    human-readable place name per camera (Lobby, Booth A, ...).

Everything is optional: an unknown pair returns a neutral feasibility so
an incomplete topology never blocks matching, it only helps when present.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class CameraTopology:
    transitions: dict[tuple[str, str], tuple[float, float]] = dataclasses.field(default_factory=dict)
    graph: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    locations: dict[str, str] = dataclasses.field(default_factory=dict)

    # feasibility returned when the pair is not in the topology at all
    neutral_temporal: float = 0.6
    neutral_spatial: float = 0.6

    def temporal_feasibility(self, from_cam: str, to_cam: str, dt_sec: float) -> float:
        """1.0 when dt is inside the expected window, decaying to 0 as it
        falls outside. Negative dt (to-camera saw them first) is impossible."""
        if from_cam == to_cam:
            return 1.0
        if dt_sec < 0:
            return 0.0  # arrived at the next camera before leaving this one — impossible
        window = self.transitions.get((from_cam, to_cam))
        if window is None:
            return self.neutral_temporal
        lo, hi = window
        if lo <= dt_sec <= hi:
            return 1.0
        if dt_sec < lo:
            # too fast — sharp penalty (they can't teleport)
            return max(0.0, dt_sec / lo) ** 2
        # too slow — gentle penalty (they may have loitered)
        overshoot = dt_sec - hi
        return max(0.0, 1.0 - overshoot / (hi if hi > 0 else 1.0))

    def spatial_feasibility(self, from_cam: str, to_cam: str) -> float:
        if from_cam == to_cam:
            return 1.0
        if not self.graph:
            return self.neutral_spatial
        neighbours = self.graph.get(from_cam, set())
        if to_cam in neighbours:
            return 1.0
        # reachable in 2 hops?
        for mid in neighbours:
            if to_cam in self.graph.get(mid, set()):
                return 0.6
        return 0.15  # not connected in the known graph

    def location(self, camera_id: str) -> str:
        return self.locations.get(camera_id, camera_id)

    @staticmethod
    def from_dict(d: dict) -> "CameraTopology":
        transitions: dict[tuple[str, str], tuple[float, float]] = {}
        for entry in d.get("transitions", []):
            transitions[(entry["from"], entry["to"])] = (
                float(entry.get("min_seconds", 0.0)),
                float(entry.get("max_seconds", 1e9)),
            )
        graph = {cam: set(neigh) for cam, neigh in d.get("graph", {}).items()}
        topo = CameraTopology(transitions=transitions, graph=graph, locations=dict(d.get("locations", {})))
        if "neutral_temporal" in d:
            topo.neutral_temporal = float(d["neutral_temporal"])
        if "neutral_spatial" in d:
            topo.neutral_spatial = float(d["neutral_spatial"])
        return topo

    @staticmethod
    def load(path: str | Path) -> "CameraTopology":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not read topology file {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON: {exc}") from exc
        return CameraTopology.from_dict(data)
