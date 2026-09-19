"""Heatmaps & movement analytics (MongDee_Master_Prompt.md sections 67-69).

Accumulate world positions into a grid over the booth. Three flavours:

    traffic   one hit per position sample (where people are)
    dwell     hits weighted by time spent (where people linger)
    interest  hits weighted by an interest score (where people engage)

Time-windowed: add() takes a timestamp, grid() can restrict to the last
N seconds so "Last Hour" / "Now" views work.
"""

from __future__ import annotations

import time

import numpy as np


class HeatmapAccumulator:
    def __init__(self, booth_width: float, booth_length: float, cell_size: float = 0.25):
        self.width = booth_width
        self.length = booth_length
        self.cell_size = cell_size
        self.cols = max(1, int(np.ceil(booth_width / cell_size)))
        self.rows = max(1, int(np.ceil(booth_length / cell_size)))
        # each sample: (row, col, weight, timestamp)
        self._samples: list[tuple[int, int, float, float]] = []

    def _cell(self, x: float, y: float) -> tuple[int, int] | None:
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def add(self, x: float, y: float, weight: float = 1.0, timestamp: float | None = None) -> None:
        cell = self._cell(x, y)
        if cell is None:
            return
        self._samples.append((cell[0], cell[1], float(weight), timestamp if timestamp is not None else time.time()))

    def grid(self, since: float | None = None, normalize: bool = False) -> np.ndarray:
        g = np.zeros((self.rows, self.cols), dtype=np.float64)
        for row, col, weight, ts in self._samples:
            if since is not None and ts < since:
                continue
            g[row, col] += weight
        if normalize and g.max() > 0:
            g = g / g.max()
        return g

    def prune(self, older_than: float) -> None:
        self._samples = [s for s in self._samples if s[3] >= older_than]

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def top_cells(self, n: int = 5, since: float | None = None) -> list[dict]:
        g = self.grid(since=since)
        flat = np.argsort(g, axis=None)[::-1][:n]
        out = []
        for idx in flat:
            row, col = divmod(int(idx), self.cols)
            if g[row, col] <= 0:
                break
            out.append({
                "x": round((col + 0.5) * self.cell_size, 3),
                "y": round((row + 0.5) * self.cell_size, 3),
                "value": round(float(g[row, col]), 3),
            })
        return out


class MovementPath:
    """One Global Person's trail through the booth (section 68). Stores the
    ordered world positions; reports total distance and whether the path is
    self-consistent."""

    def __init__(self, global_id: str):
        self.global_id = global_id
        self.points: list[dict] = []

    def add(self, x: float, y: float, timestamp: float, camera_id: str, zone_id: str | None, confidence: float) -> None:
        self.points.append({
            "x": x, "y": y, "timestamp": timestamp, "camera_id": camera_id,
            "zone_id": zone_id, "confidence": confidence,
        })
        self.points.sort(key=lambda p: p["timestamp"])

    def total_distance(self) -> float:
        d = 0.0
        for a, b in zip(self.points, self.points[1:]):
            d += ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
        return round(d, 3)

    def zones_visited(self) -> list[str]:
        seen = []
        for p in self.points:
            if p["zone_id"] and (not seen or seen[-1] != p["zone_id"]):
                seen.append(p["zone_id"])
        return seen

    def is_consistent(self, max_speed_mps: float = 3.5) -> bool:
        for a, b in zip(self.points, self.points[1:]):
            dt = max(1e-3, b["timestamp"] - a["timestamp"])
            dist = ((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2) ** 0.5
            if dist / dt > max_speed_mps:
                return False
        return True
