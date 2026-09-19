"""FaceService: detect faces on the full frame, bind each face to a person track, keep the
best-quality shot per track. Independent of the person detector; consumes the tracker's
visible tracks (dicts with track_id + bbox)."""
from __future__ import annotations
import heapq
import time
from typing import Dict, List, Optional, Sequence
import numpy as np
from .detector import FaceBox, crop_with_margin, face_quality

# --- keeping a face box steady between AI passes ---------------------------------------------------------------
FACE_HOLD_SEC = 1.6        # a face that was seen is kept on screen this long after the detector misses it
FACE_SMOOTH = 0.6          # weight of a new detection when it moves an existing box (removes jitter)
FACE_MATCH_IOU = 0.2       # an un-bound detection continues an existing face box when they overlap this much
FACE_RECOVER_MAX = 3       # second-chance (low-threshold) searches per pass
FACE_RECOVER_SCORE = 0.35


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class _StableFace:
    __slots__ = ("box", "score", "quality", "track_id", "last_hit", "rel", "hits")

    def __init__(self, box, score, quality, track_id, now):
        self.box, self.score, self.quality, self.track_id = list(box), float(score), float(quality), track_id
        self.last_hit, self.rel, self.hits = now, None, 1


class FaceStabilizer:
    """Turns the detector's per-pass face boxes (which blink: a turned head, motion blur or one weak frame drops
    the face for a pass) into steady boxes. A detection continues the box of the same person track (or the
    overlapping box); a miss keeps the box for FACE_HOLD_SEC, moving and scaling with the person's own box, so it
    never vanishes and reappears; a new detection is blended with the old box so it does not jitter."""

    def __init__(self, hold_sec: float = FACE_HOLD_SEC, smooth: float = FACE_SMOOTH):
        self.hold_sec, self.smooth = hold_sec, smooth
        self._faces: Dict[object, _StableFace] = {}
        self._n = 0

    @staticmethod
    def _track_box(tracks, track_id):
        for t in tracks:
            if t["track_id"] == track_id:
                return t["bbox"]
        return None

    @staticmethod
    def _relative(box, tbox):
        tw, th = max(tbox[2] - tbox[0], 1.0), max(tbox[3] - tbox[1], 1.0)
        return ((box[0] - tbox[0]) / tw, (box[1] - tbox[1]) / th, (box[2] - tbox[0]) / tw, (box[3] - tbox[1]) / th)

    def update(self, faces: Sequence[FaceBox], tracks: Sequence[dict], now: float) -> set:
        """Feed this pass's detections; returns the keys that were hit (seen this pass)."""
        hit: set = set()
        for f in sorted(faces, key=lambda x: -x.score):
            box = [float(f.x1), float(f.y1), float(f.x2), float(f.y2)]
            key = None
            if f.track_id is not None and f.track_id not in hit:
                key = ("t", f.track_id)
            else:
                best = 0.0
                for k, s in self._faces.items():
                    if k in hit:
                        continue
                    v = _iou(s.box, box)
                    if v > best and v >= FACE_MATCH_IOU:
                        key, best = k, v
                if key is None:
                    self._n += 1
                    key = ("u", self._n)
            state = self._faces.get(key)
            if state is None or _iou(state.box, box) < 0.1:
                state = _StableFace(box, f.score, f.quality, f.track_id, now)
                self._faces[key] = state
            else:
                a = self.smooth
                state.box = [a * n + (1 - a) * o for n, o in zip(box, state.box)]
                state.score, state.quality, state.last_hit = float(f.score), float(f.quality), now
                state.hits += 1
                state.track_id = f.track_id if f.track_id is not None else state.track_id
            tbox = self._track_box(tracks, state.track_id) if state.track_id is not None else None
            state.rel = self._relative(state.box, tbox) if tbox is not None else None
            hit.add(key)
        for k in [k for k in self._faces if k[0] == "u" and k not in hit]:      # an un-bound box the bound one replaced
            if any(h[0] == "t" and _iou(self._faces[k].box, self._faces[h].box) > 0.5 for h in hit):
                del self._faces[k]
        return hit

    def _predicted(self, state: _StableFace, tracks) -> list:
        tbox = self._track_box(tracks, state.track_id) if state.track_id is not None else None
        if tbox is None or state.rel is None:
            return list(state.box)
        tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
        r = state.rel
        return [tbox[0] + r[0] * tw, tbox[1] + r[1] * th, tbox[0] + r[2] * tw, tbox[1] + r[3] * th]

    def missed(self, hit: set, tracks, now: float) -> list:
        """(key, predicted box) of faces seen recently but not detected this pass - candidates for a second look."""
        return [(k, self._predicted(s, tracks)) for k, s in self._faces.items()
                if k not in hit and now - s.last_hit <= self.hold_sec]

    def visible(self, tracks: Sequence[dict], now: float) -> List[FaceBox]:
        """The steady set to draw: every face seen within the hold time, at its motion-predicted position."""
        for k in [k for k, s in self._faces.items() if now - s.last_hit > self.hold_sec]:
            del self._faces[k]
        out: List[FaceBox] = []
        for s in self._faces.values():
            b = self._predicted(s, tracks)
            s.box = b
            out.append(FaceBox(b[0], b[1], b[2], b[3], s.score, None, s.quality, s.track_id))
        return out

    def forget(self, track_id) -> None:
        self._faces.pop(("t", track_id), None)


class BestShotStore:
    def __init__(self, keep: int = 3, max_tracks: int = 64):
        self.keep, self.max_tracks = keep, max_tracks
        self._shots: Dict[int, list] = {}     # tid -> min-heap of (quality, counter, crop)
        self._n = 0

    def offer(self, tid: int, quality: float, crop: np.ndarray):
        h = self._shots.setdefault(tid, [])
        self._n += 1
        item = (quality, self._n, crop)
        if len(h) < self.keep:
            heapq.heappush(h, item)
        elif quality > h[0][0]:
            heapq.heapreplace(h, item)
        if len(self._shots) > self.max_tracks:  # drop oldest track
            self._shots.pop(next(iter(self._shots)))

    def best(self, tid: int) -> Optional[np.ndarray]:
        h = self._shots.get(tid)
        return max(h)[2] if h else None

    def best_quality(self, tid: int) -> float:
        h = self._shots.get(tid)
        return max(h)[0] if h else 0.0

    def drop(self, tid: int):
        self._shots.pop(tid, None)


class FaceService:
    def __init__(self, detector, min_quality_for_shot: float = 0.15, min_face_px: int = 20,
                 shot_margin: float = 0.3):
        self.detector, self.min_q, self.min_px, self.margin = detector, min_quality_for_shot, min_face_px, shot_margin
        self.store = BestShotStore()
        self.stabilizer = FaceStabilizer()

    @staticmethod
    def _bind(face: FaceBox, tracks: Sequence[dict]) -> Optional[int]:
        """Face belongs to the track whose box contains the face centre inside its upper 65%,
        preferring the smallest such box (avoids a big box swallowing a nearby person's face)."""
        cx, cy = face.center
        best, best_area = None, None
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.65 * (y2 - y1):
                a = (x2 - x1) * (y2 - y1)
                if best is None or a < best_area:
                    best, best_area = t["track_id"], a
        return best

    def process(self, frame, tracks: Sequence[dict], now: Optional[float] = None) -> List[FaceBox]:
        """Detect faces (raw detections, as before) and feed the stabiliser; a face the full-frame pass missed but
        that was there a moment ago gets a second, lower-threshold look around where it should be."""
        now = time.monotonic() if now is None else now
        faces = [f for f in self.detector.detect(frame) if min(f.w, f.h) >= self.min_px]
        for f in faces:
            f.quality = face_quality(frame, f)
            f.track_id = self._bind(f, tracks)
            if f.track_id is not None and f.quality >= self.min_q:
                self.store.offer(f.track_id, f.quality, crop_with_margin(frame, f, self.margin).copy())
        hit = self.stabilizer.update(faces, tracks, now)
        recover = getattr(self.detector, "detect_in_region", None)
        if recover is not None:
            for key, box in self.stabilizer.missed(hit, tracks, now)[:FACE_RECOVER_MAX]:
                w, h = box[2] - box[0], box[3] - box[1]
                region = (box[0] - 0.8 * w, box[1] - 0.8 * h, box[2] + 0.8 * w, box[3] + 0.8 * h)
                found = [f for f in recover(frame, region, FACE_RECOVER_SCORE) if min(f.w, f.h) >= self.min_px]
                if not found:
                    continue
                best = max(found, key=lambda f: _iou((f.x1, f.y1, f.x2, f.y2), box))
                if _iou((best.x1, best.y1, best.x2, best.y2), box) < 0.15:
                    continue
                best.quality = face_quality(frame, best)
                best.track_id = self._bind(best, tracks)
                self.stabilizer.update([best], tracks, now)
                faces.append(best)
        return faces

    def stable_faces(self, tracks: Sequence[dict], now: Optional[float] = None) -> List[FaceBox]:
        """Steady boxes to draw (see FaceStabilizer)."""
        return self.stabilizer.visible(tracks, time.monotonic() if now is None else now)

    def forget(self, track_id: int):
        self.store.drop(track_id)
        self.stabilizer.forget(track_id)
