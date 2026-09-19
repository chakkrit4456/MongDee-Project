"""tracker_v2 - Kalman + two-stage (ByteTrack-style) association tracker.

Original implementation (numpy/scipy only, no third-party tracker code, no AGPL).
Drop-in shape-compatible with core.tracker.PersonTracker:
    update(person_boxes, person_categories=None, person_confidences=None, now=None)
        -> (visible_tracks, evicted_tracks)
Track dict keys: track_id, bbox, first_seen, last_seen, category (+ extras: hits, confidence).
Boxes are (x1, y1, x2, y2) in any consistent pixel unit.

Why it fixes the old behaviour (greedy IoU, no motion model, TRACK_MAX_AGE 1.5 s):
  * constant-velocity Kalman predicts where a person moved between sparse AI passes,
    so IoU stays high when detections arrive every ~0.3-1 s;
  * low-confidence detections rescue existing tracks (2nd stage) instead of dropping them;
  * new tracks need `min_hits` confirmations -> single-frame false positives never get an ID;
  * category (male/female) is a confidence-weighted sliding vote, so one bad
    frame cannot flip the label.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
try:  # scipy is optional: a pure-numpy Hungarian fallback keeps the module dependency-light
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised in environments without scipy
    def linear_sum_assignment(cost):
        """Minimise sum(cost[r, c]) over a partial matching (rectangular ok). O(n^3) Hungarian."""
        cost = np.asarray(cost, float)
        transposed = cost.shape[0] > cost.shape[1]
        if transposed:
            cost = cost.T
        n, m = cost.shape  # n <= m
        if n == 0:
            return np.array([], int), np.array([], int)
        INF = float("inf")
        u = np.zeros(n + 1); v = np.zeros(m + 1)
        p = np.zeros(m + 1, int); way = np.zeros(m + 1, int)
        for i in range(1, n + 1):
            p[0] = i; j0 = 0
            minv = np.full(m + 1, INF); used = np.zeros(m + 1, bool)
            while True:
                used[j0] = True
                i0 = p[j0]; delta = INF; j1 = 0
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur; way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]; j1 = j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta; v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]; p[j0] = p[j1]; j0 = j1
                if j0 == 0:
                    break
        rows, cols = [], []
        for j in range(1, m + 1):
            if p[j]:
                rows.append(p[j] - 1); cols.append(j - 1)
        rows, cols = np.array(rows, int), np.array(cols, int)
        if transposed:
            rows, cols = cols, rows
        order = np.argsort(rows)
        return rows[order], cols[order]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class KalmanBox:
    """Constant-velocity Kalman over state [cx, cy, w, h, vx, vy, vw, vh] with
    time-dependent transition (dt in seconds) so uneven AI cadence is handled."""

    def __init__(self, xyxy: Sequence[float]):
        x1, y1, x2, y2 = map(float, xyxy)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        self.x = np.array([(x1 + x2) / 2, (y1 + y2) / 2, w, h, 0, 0, 0, 0], float)
        # position uncertainty ~ box size, velocity very uncertain at birth
        self.P = np.diag([w * .1] * 2 + [w * .1, h * .1] + [w * 2, h * 2, w, h]).astype(float) ** 2
        self._std_pos, self._std_vel, self._std_meas = 1 / 20., 1 / 160., 1 / 20.

    def _F(self, dt):
        F = np.eye(8)
        for i in range(4):
            F[i, i + 4] = dt
        return F

    def predict(self, dt: float):
        dt = float(np.clip(dt, 0.0, 5.0))
        F = self._F(dt)
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        sp, sv = self._std_pos, self._std_vel
        q = np.array([sp * w, sp * h, sp * w, sp * h, sv * w, sv * h, sv * w, sv * h]) ** 2
        Q = np.diag(q) * max(dt, 0.05) * 4
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, xyxy: Sequence[float], conf: float = 1.0):
        x1, y1, x2, y2 = map(float, xyxy)
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1), max(y2 - y1, 1)])
        H = np.zeros((4, 8)); H[:, :4] = np.eye(4)
        w, h = max(self.x[2], 1.0), max(self.x[3], 1.0)
        sm = self._std_meas / max(conf, 0.2)
        R = np.diag([sm * w, sm * h, sm * w, sm * h]) ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def xyxy(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class _Track:
    __slots__ = ("id", "kf", "first_seen", "last_seen", "hits", "misses", "confirmed",
                 "votes", "conf", "last_det_bbox")

    def __init__(self, tid, box, conf, cat, now, vote_window):
        self.id = tid
        self.kf = KalmanBox(box)
        self.first_seen = self.last_seen = now
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.votes: deque = deque(maxlen=vote_window)
        self.conf = float(conf)
        self.last_det_bbox = tuple(map(float, box))
        if cat is not None:
            self.votes.append((cat, float(conf)))

    def category(self):
        if not self.votes:
            return None
        tally: Counter = Counter()
        for c, w in self.votes:
            tally[c] += w
        return tally.most_common(1)[0][0]


class TrackerV2:
    def __init__(self, high_conf: float = 0.5, low_conf: float = 0.1,
                 iou_high: float = 0.25, iou_low: float = 0.2, min_hits: int = 2,
                 max_age_sec: float = 2.0, vote_window: int = 15, visible_max_gap_sec: float = 1.0):
        self.high_conf, self.low_conf = high_conf, low_conf
        self.iou_high, self.iou_low = iou_high, iou_low
        self.min_hits, self.max_age_sec = min_hits, max_age_sec
        self.vote_window = vote_window
        self.visible_max_gap_sec = visible_max_gap_sec
        self._tracks: List[_Track] = []
        self._next_id = 1
        self._last_now: Optional[float] = None

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _assign(tracks: List[_Track], boxes: np.ndarray, thr: float):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = np.stack([t.kf.xyxy for t in tracks])
        iou = iou_matrix(tb, boxes)
        r, c = linear_sum_assignment(-iou)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(boxes)))
        for i, j in zip(r, c):
            if iou[i, j] >= thr:
                matches.append((i, j)); ut.discard(i); ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def _to_dict(self, t: _Track) -> Dict:
        return {"track_id": t.id, "bbox": tuple(float(v) for v in t.kf.xyxy),
                "first_seen": t.first_seen, "last_seen": t.last_seen,
                "category": t.category(), "hits": t.hits, "confidence": t.conf}

    # ---- main API ------------------------------------------------------
    def update(self, person_boxes, person_categories=None, person_confidences=None,
               now: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
        now = time.time() if now is None else float(now)
        boxes = np.asarray(person_boxes, float).reshape(-1, 4) if len(person_boxes) else np.zeros((0, 4))
        n = len(boxes)
        cats = list(person_categories) if person_categories is not None else [None] * n
        confs = np.asarray(person_confidences if person_confidences is not None else [1.0] * n, float)
        dt = 0.0 if self._last_now is None else max(now - self._last_now, 0.0)
        self._last_now = now
        for t in self._tracks:
            t.kf.predict(dt)

        hi = [i for i in range(n) if confs[i] >= self.high_conf]
        lo = [i for i in range(n) if self.low_conf <= confs[i] < self.high_conf]
        confirmed = [t for t in self._tracks if t.confirmed]
        tentative = [t for t in self._tracks if not t.confirmed]

        def apply(t: _Track, i: int):
            t.kf.update(boxes[i], confs[i]); t.hits += 1; t.misses = 0
            t.last_seen = now; t.conf = float(confs[i]); t.last_det_bbox = tuple(boxes[i])
            if cats[i] is not None:
                t.votes.append((cats[i], float(confs[i])))
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True

        # stage 1: confirmed+tentative vs high-conf
        pool = confirmed + tentative
        m1, ut1, ud1 = self._assign(pool, boxes[hi] if hi else np.zeros((0, 4)), self.iou_high)
        for ti, di in m1:
            apply(pool[ti], hi[di])
        rest_tracks = [pool[i] for i in ut1]
        # stage 2: remaining confirmed tracks vs low-conf detections (rescue)
        rescue = [t for t in rest_tracks if t.confirmed]
        m2, ut2, _ = self._assign(rescue, boxes[lo] if lo else np.zeros((0, 4)), self.iou_low)
        matched2 = set()
        for ti, di in m2:
            apply(rescue[ti], lo[di]); matched2.add(id(rescue[ti]))
        # births from unmatched high-conf detections
        for di in ud1:
            i = hi[di]
            t = _Track(self._next_id, boxes[i], confs[i], cats[i], now, self.vote_window)
            if self.min_hits <= 1:
                t.confirmed = True
            self._next_id += 1
            self._tracks.append(t)
        # aging / eviction
        evicted, keep = [], []
        matched_ids = {pool[ti].id for ti, _ in m1} | {t.id for t in rescue if id(t) in matched2}
        for t in self._tracks:
            if t.last_seen == now or t.id in matched_ids:
                keep.append(t); continue
            t.misses += 1
            age = now - t.last_seen
            if (not t.confirmed and age > 0) or age > self.max_age_sec:
                if t.confirmed:
                    evicted.append(self._to_dict(t))
            else:
                keep.append(t)
        self._tracks = keep
        visible = [self._to_dict(t) for t in self._tracks
                   if t.confirmed and (now - t.last_seen) <= self.visible_max_gap_sec]
        return visible, evicted

    def reset(self):
        self._tracks.clear(); self._last_now = None
