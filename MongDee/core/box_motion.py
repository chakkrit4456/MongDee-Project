"""box_motion - keep the drawn detection boxes glued to the moving picture between (and after) AI passes.

The AI pass (YOLO + tracker + Re-ID + faces ...) takes ~100-300 ms and runs a few times a second, so a box
drawn straight from it jumps a few times a second and is always a fraction of a second behind the video. This
follower fixes both, cheaply, without another neural network:

  * every captured frame is reduced to a small grey image and kept in a short ring buffer;
  * for every box currently shown, the median optical flow (Lucas-Kanade on corner features inside the box)
    between consecutive frames moves the box, so it follows the person / product at the CAPTURE frame rate;
  * when an AI pass finishes, its boxes describe the frame the AI *started* on, so they are first carried
    forward through the frames that were captured while the AI was busy (catch-up), then blended with the
    boxes already on screen (so a fresh detection corrects drift without a visible jump);
  * boxes that no AI pass has refreshed for `max_age_sec` disappear instead of freezing on screen.

Everything degrades gracefully: too few trackable features, a huge flow estimate or a missing ring frame just
means the box stays where the AI put it (the previous behaviour). Thread model: `push_frame` from the capture
thread, `submit` from the AI thread, `boxes` from the capture thread; one lock, held only for list swaps.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

WORK_WIDTH = 320             # optical flow runs on frames shrunk to this width
RING_FRAMES = 24
MAX_AGE_SEC = 1.5
MIN_POINTS = 5
MAX_STEP_FRACTION = 0.35     # a per-frame shift larger than this fraction of the box is not believed
BLEND_NEW = 0.6              # weight of a fresh AI box when it is blended with the tracked one it matches
MATCH_IOU = 0.3
_LK = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def flow_shift(prev_gray: np.ndarray, cur_gray: np.ndarray, box) -> "tuple[float, float] | None":
    """Median (dx, dy), in the images' pixels, of the features inside `box` (x1,y1,x2,y2 in the same
    pixels) between two grey frames; None when it cannot be measured reliably."""
    h, w = prev_gray.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 10 or bh < 10:
        return None
    mx, my = 0.15 * bw, 0.12 * bh                       # the central part of a person box is mostly the person
    rx1, ry1 = int(max(x1 + mx, 0)), int(max(y1 + my, 0))
    rx2, ry2 = int(min(x2 - mx, w)), int(min(y2 - my, h))
    if rx2 - rx1 < 8 or ry2 - ry1 < 8:
        return None
    roi = prev_gray[ry1:ry2, rx1:rx2]
    pts = cv2.goodFeaturesToTrack(roi, maxCorners=40, qualityLevel=0.01, minDistance=3)
    if pts is None or len(pts) < MIN_POINTS:
        return None
    pts = pts.reshape(-1, 1, 2).astype(np.float32) + np.array([[[rx1, ry1]]], dtype=np.float32)
    nxt, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, pts, None, **_LK)
    if nxt is None:
        return None
    ok = status.reshape(-1) == 1
    if int(ok.sum()) < MIN_POINTS:
        return None
    flow = (nxt - pts).reshape(-1, 2)[ok]
    med = np.median(flow, axis=0)
    inliers = flow[np.linalg.norm(flow - med, axis=1) <= max(1.5, 0.5 * float(np.linalg.norm(med)) + 1.0)]
    if len(inliers) < MIN_POINTS:
        return None
    dx, dy = np.median(inliers, axis=0)
    if abs(dx) > MAX_STEP_FRACTION * bw or abs(dy) > MAX_STEP_FRACTION * bh:
        return None
    return float(dx), float(dy)


class BoxFollower:
    def __init__(self, work_width: int = WORK_WIDTH, ring_frames: int = RING_FRAMES, max_age_sec: float = MAX_AGE_SEC,
                 clock=time.monotonic):
        self.work_width = work_width
        self.max_age_sec = max_age_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._ring: deque = deque(maxlen=ring_frames)     # (seq, small_gray)
        self._scale = 1.0
        self._boxes: list = []                            # [bbox(list of 4 floats, full-res), label, color]
        self._stamp = 0.0
        self._shown_seq = -1
        self.steps = 0
        self.catchups = 0

    # ----------------------------------------------------------------- helpers
    def _small(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = min(1.0, self.work_width / float(w))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._scale < 1.0:
            gray = cv2.resize(gray, (int(round(w * self._scale)), int(round(h * self._scale))), interpolation=cv2.INTER_AREA)
        return gray

    def _advance(self, bbox: list, prev_gray: np.ndarray, cur_gray: np.ndarray) -> list:
        s = self._scale
        shift = flow_shift(prev_gray, cur_gray, [v * s for v in bbox])
        if shift is None:
            return bbox
        dx, dy = shift[0] / s, shift[1] / s
        return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]

    # ----------------------------------------------------------- capture thread
    def push_frame(self, seq: int, frame: np.ndarray) -> None:
        """A new captured frame: remember it and move every shown box along with the picture."""
        gray = self._small(frame)
        with self._lock:
            prev = self._ring[-1] if self._ring else None
            self._ring.append((seq, gray))
            if prev is None or not self._boxes or prev[1].shape != gray.shape:
                self._shown_seq = seq
                return
            for entry in self._boxes:
                entry[0] = self._advance(entry[0], prev[1], gray)
            self._shown_seq = seq
            self.steps += 1

    def boxes(self) -> list:
        """(bbox, label, color) tuples for the frame just pushed; [] once the AI has gone quiet too long."""
        with self._lock:
            if not self._boxes or self._clock() - self._stamp > self.max_age_sec:
                return []
            return [(list(b[0]), b[1], b[2]) for b in self._boxes]

    def reset(self) -> None:
        with self._lock:
            self._ring.clear()
            self._boxes = []
            self._shown_seq = -1

    # ---------------------------------------------------------------- AI thread
    def submit(self, frame_seq: int, boxes: list) -> None:
        """`boxes` = [(bbox, label, color)] the AI computed for the frame numbered `frame_seq`."""
        fresh = [[[float(v) for v in bbox[:4]], label, color] for bbox, label, color in boxes]
        with self._lock:
            ring = list(self._ring)
            scale = self._scale
        idx = next((i for i, (s, _g) in enumerate(ring) if s == frame_seq), None)
        # catch up through the frames captured while the AI was busy (outside the lock: this is the slow part)
        done_seq = frame_seq
        if idx is not None and fresh:
            for i in range(idx, len(ring) - 1):
                prev, cur = ring[i][1], ring[i + 1][1]
                if prev.shape != cur.shape:
                    break
                for entry in fresh:
                    entry[0] = self._advance(entry[0], prev, cur)
                done_seq = ring[i + 1][0]
            self.catchups += 1
        with self._lock:
            # frames pushed while we were catching up (normally 0-2): finish the job under the lock
            newer = [(s, g) for s, g in self._ring if s > done_seq]
            if idx is not None and fresh and newer:
                prev_gray = next((g for s, g in self._ring if s == done_seq), None)
                for s, g in newer:
                    if prev_gray is not None and prev_gray.shape == g.shape:
                        for entry in fresh:
                            entry[0] = self._advance(entry[0], prev_gray, g)
                    prev_gray = g
            # blend with what is on screen so a fresh detection corrects drift without a visible jump
            for entry in fresh:
                best, best_iou = None, MATCH_IOU
                for old in self._boxes:
                    iou = _iou(entry[0], old[0])
                    if iou >= best_iou:
                        best, best_iou = old, iou
                if best is not None:
                    entry[0] = [BLEND_NEW * n + (1.0 - BLEND_NEW) * o for n, o in zip(entry[0], best[0])]
            self._boxes = fresh
            self._stamp = self._clock()
