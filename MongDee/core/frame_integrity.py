"""frame_integrity - decide whether a captured BGR frame is trustworthy.

Detects (hard failures -> `Verdict.ok == False`):
  empty/None/bad dtype, truncated-MJPEG flat tail, horizontal tearing seam, random-noise garbage,
  solid colour cast (broken YUV/MJPEG decode) and a FROZEN stream (bit-identical frames for seconds).
Soft observations (`Verdict.notes`, only failures when `strict_exposure=True`): black / white / flat.
Dark rooms are legitimate, so exposure alone never rejects a frame in production.

Why the seam and tail checks are TEMPORAL: a strong horizontal edge in a real scene (table edge,
window frame) or a black letterbox bar at the bottom looks exactly like a tear / truncated tail in a
single frame, but it is present in every frame. A corrupt frame is one whose seam / flat tail
APPEARS relative to the previous frame. So a candidate is only reported when it is new. The very
first frame after reset() can therefore never be flagged for those two.

Cheap: a few row/column statistics plus one full-frame equality test.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class Verdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)   # hard failures
    notes: List[str] = field(default_factory=list)     # soft observations

    def __bool__(self):
        return self.ok


class FrameIntegrity:
    def __init__(self, min_std=3.0, dark_mean=6.0, bright_mean=249.0, freeze_sec=8.0,
                 freeze_min_frames=8, tail_frac=0.08, seam_ratio=1.8, seam_jump=20.0,
                 noise_hf_ratio=0.55, strict_exposure=False):
        self.min_std, self.dark_mean, self.bright_mean = min_std, dark_mean, bright_mean
        self.freeze_sec, self.freeze_min_frames = freeze_sec, freeze_min_frames
        self.tail_frac, self.seam_ratio, self.seam_jump = tail_frac, seam_ratio, seam_jump
        self.noise_hf_ratio = noise_hf_ratio
        self.strict_exposure = strict_exposure
        self.reset()

    def reset(self):
        self._prev_frame: Optional[np.ndarray] = None
        self._same_since: Optional[float] = None
        self._same_count = 0
        self._prev_rowdiff: Optional[np.ndarray] = None
        self._prev_tail_rows: Optional[int] = None

    def frozen_for(self, now: float) -> float:
        """Seconds the stream has been bit-for-bit identical (0.0 when it is moving)."""
        return 0.0 if self._same_since is None else max(now - self._same_since, 0.0)

    # -- individual checks -------------------------------------------------
    @staticmethod
    def _tail_rows(gray) -> int:
        """Number of contiguous perfectly-flat rows at the bottom (decoder gray fill)."""
        flat = gray.std(axis=1) < 1.0
        n = 0
        for v in flat[::-1]:
            if not v:
                break
            n += 1
        return n

    def _seam_row(self, rowdiff: np.ndarray) -> Optional[int]:
        """Row boundary that is both dominant in this frame and NEW versus the previous frame."""
        if self._prev_rowdiff is None or len(self._prev_rowdiff) != len(rowdiff):
            return None
        top = np.sort(rowdiff)[::-1]
        r = int(rowdiff.argmax())
        if float(top[0]) < 25.0 or float(top[0]) / (float(top[1]) + 1.0) < self.seam_ratio:
            return None
        prev = float(self._prev_rowdiff[r])
        if float(top[0]) - prev >= self.seam_jump and float(top[0]) >= 1.8 * prev:
            return r
        return None

    def _noise(self, gray) -> bool:
        """Garbage/static: almost all energy is pixel-to-pixel high-frequency."""
        g = gray.astype(np.float32)
        if g.std() < 1e-3:
            return False
        hf = np.abs(np.diff(g, axis=1)).mean()
        lf = np.abs(cv2.blur(g, (9, 9)) - g.mean()).mean()
        spread = g.std()
        return float(hf) / (spread + 1e-6) > self.noise_hf_ratio * 2.0 and float(lf) < 0.6 * float(spread)

    @staticmethod
    def _cast(frame) -> bool:
        """Solid colour cast from a broken YUV/MJPEG decode: two channels ~dead, one alive."""
        m = np.sort(frame.reshape(-1, 3).mean(axis=0))
        return float(m[2]) > 25.0 and float(m[1]) < 0.05 * float(m[2])

    # -- main ----------------------------------------------------------------
    def check(self, frame, now: float) -> Verdict:
        r: List[str] = []
        notes: List[str] = []
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return Verdict(False, ["empty"])
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            return Verdict(False, ["bad_shape_or_dtype"])
        if frame.shape[0] < 16 or frame.shape[1] < 16:
            return Verdict(False, ["too_small"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std = float(gray.mean()), float(gray.std())
        exposure = None
        if mean < self.dark_mean:
            exposure = "black"
        elif mean > self.bright_mean:
            exposure = "white"
        elif std < self.min_std:
            exposure = "flat"
        (r if self.strict_exposure else notes).extend([exposure] if exposure else [])

        rowdiff = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=1)
        tail = self._tail_rows(gray)
        if exposure is None:
            min_tail = max(int(gray.shape[0] * self.tail_frac), 4)
            body_std = float(gray[: gray.shape[0] - tail].std()) if tail < gray.shape[0] else 0.0
            if (self._prev_tail_rows is not None and tail >= min_tail
                    and self._prev_tail_rows < 0.5 * tail and body_std > 8.0):
                r.append("truncated_tail")
            if self._seam_row(rowdiff) is not None:
                r.append("tear_seam")
            if self._noise(gray):
                r.append("noise")
            if self._cast(frame):
                r.append("color_cast")
        self._prev_rowdiff, self._prev_tail_rows = rowdiff, tail

        # freeze: bit-identical full frame for freeze_sec AND >= N frames. Sensor noise makes this
        # impossible for a live camera, so it reliably means a stuck driver buffer.
        if self._prev_frame is not None and self._prev_frame.shape == frame.shape \
                and np.array_equal(self._prev_frame, frame):
            self._same_count += 1
            if self._same_since is None:
                self._same_since = now
            if self._same_count >= self.freeze_min_frames and now - self._same_since >= self.freeze_sec:
                r.append("frozen")
        else:
            self._same_since, self._same_count = None, 0
        self._prev_frame = frame.copy() if (self._prev_frame is None or self._prev_frame.shape != frame.shape
                                            or self._same_count == 0) else self._prev_frame
        return Verdict(not r, r, notes)
