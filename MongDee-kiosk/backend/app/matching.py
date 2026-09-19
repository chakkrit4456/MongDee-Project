"""Multi-view consensus using the existing (compatible) embeddings."""
from dataclasses import dataclass
import threading

import numpy as np

from app import config


@dataclass(frozen=True)
class MatchResult:
    ranked: list[tuple[int, float]]
    frame_count: int = 0
    agreement: float = 0.0
    support: float = 0.0

    @property
    def status(self) -> str:
        if self.frame_count < config.RECOGNITION_MIN_FRAMES:
            return "unstable"
        if not self.ranked or self.ranked[0][1] < config.MATCH_CONFIDENCE_THRESHOLD:
            return "unknown"
        # A runner-up just below threshold is still a dangerous near-tie.
        close = len(self.ranked) > 1 and self.ranked[0][1] - self.ranked[1][1] < config.AMBIGUOUS_MARGIN
        if close or self.agreement < config.MATCH_MIN_AGREEMENT or self.support < config.MATCH_MIN_SUPPORT:
            return "ambiguous"
        return "matched"


class MatchIndex:
    def __init__(self, expected_dim: int):
        self.expected_dim = expected_dim
        self._lock = threading.Lock()
        self._state = (np.empty(0, dtype=np.int64), None, 0)

    @property
    def version(self) -> int:
        return self._state[2]

    def rebuild(self, rows) -> None:
        ids, vectors = [], []
        for pid, vector in rows:
            try:
                vec = np.asarray(vector, dtype=np.float32)
            except (ValueError, TypeError):
                continue
            if vec.shape != (self.expected_dim,) or not np.isfinite(vec).all():
                continue
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm < 1e-8:
                continue
            ids.append(pid)
            vectors.append(vec / norm)
        matrix = np.stack(vectors) if vectors else None
        with self._lock:
            # Publish IDs and vectors together; readers never see a half rebuild.
            self._state = (np.asarray(ids), matrix, self.version + 1)

    def match_burst(self, live_embeddings) -> MatchResult:
        ids, stored, _ = self._state
        if not len(live_embeddings):
            return MatchResult([])
        live = np.asarray(live_embeddings, dtype=np.float32)
        if live.ndim != 2 or live.shape[1] != self.expected_dim:
            return MatchResult([])
        norms = np.linalg.norm(live, axis=1)
        valid = np.isfinite(live).all(axis=1) & np.isfinite(norms) & (norms > 1e-8)
        live = live[valid] / norms[valid, None]
        if stored is None or not len(live):
            return MatchResult([], len(live))
        sims = np.clip(live @ stored.T, -1.0, 1.0)
        products = np.unique(ids)
        # Each live angle gets one vote per product, irrespective of the
        # number of saved templates. Never use the maximum over live frames.
        per_frame = np.column_stack([sims[:, ids == pid].max(axis=1) for pid in products])
        ordered = np.sort(per_frame, axis=0)
        trim = int(len(live) * 0.20)
        middle = ordered[trim:len(live) - trim] if trim else ordered
        aggregate = 0.5 * np.median(per_frame, axis=0) + 0.5 * np.mean(middle, axis=0)
        order = np.argsort(-aggregate, kind="stable")
        best = order[0]
        winners = np.argmax(per_frame, axis=1)
        agreement = float(np.mean(winners == best))
        support = float(np.mean(per_frame[:, best] >= config.MATCH_CONFIDENCE_THRESHOLD))
        ranked = [(int(products[i]), float(aggregate[i])) for i in order[:config.TOP_K_CANDIDATES]]
        return MatchResult(ranked, len(live), agreement, support)
