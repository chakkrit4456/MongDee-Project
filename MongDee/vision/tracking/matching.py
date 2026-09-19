"""Association helpers for the tracker.

Greedy IoU matching rather than Hungarian/Jonker-Volgenant: at booth scale
(a handful to a couple dozen people per camera) the optimal-assignment gap
is negligible, and greedy keeps this module dependency-free and trivially
verifiable. ByteTrack's real advantage over plain SORT comes from its
two-stage high/low-confidence split and Kalman prediction (see
bytetrack.py), not from the assignment algorithm.
"""

from __future__ import annotations

import numpy as np


def iou_matrix(tracks_bboxes: list[list[float]], det_bboxes: list[list[float]]) -> np.ndarray:
    """IoU between every track bbox (rows) and every detection bbox (cols)."""
    if not tracks_bboxes or not det_bboxes:
        return np.zeros((len(tracks_bboxes), len(det_bboxes)), dtype=float)

    t = np.asarray(tracks_bboxes, dtype=float)  # (T, 4)
    d = np.asarray(det_bboxes, dtype=float)  # (D, 4)

    tl = np.maximum(t[:, None, :2], d[None, :, :2])  # (T, D, 2)
    br = np.minimum(t[:, None, 2:], d[None, :, 2:])
    wh = np.clip(br - tl, a_min=0.0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    area_t = np.clip(t[:, 2] - t[:, 0], 0, None) * np.clip(t[:, 3] - t[:, 1], 0, None)
    area_d = np.clip(d[:, 2] - d[:, 0], 0, None) * np.clip(d[:, 3] - d[:, 1], 0, None)
    union = area_t[:, None] + area_d[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def greedy_match(
    iou: np.ndarray, iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily pair rows (tracks) to columns (detections) by descending
    IoU. Returns (matches, unmatched_rows, unmatched_cols) where matches is
    a list of (row_idx, col_idx). Only pairs with IoU >= iou_threshold are
    matched."""
    n_rows, n_cols = iou.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    pairs = [
        (iou[r, c], r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if iou[r, c] >= iou_threshold
    ]
    pairs.sort(reverse=True)

    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, r, c in pairs:
        if r in matched_rows or c in matched_cols:
            continue
        matched_rows.add(r)
        matched_cols.add(c)
        matches.append((r, c))

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
