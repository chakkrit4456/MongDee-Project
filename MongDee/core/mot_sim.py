"""Synthetic multi-person scenes + MOT metrics (ID switches, IDF1, fragmentation).
Pure numpy/scipy so it runs anywhere (no motmetrics)."""
from __future__ import annotations
import numpy as np
from .tracker_v2 import iou_matrix, linear_sum_assignment


def make_scene(seed=0, n_people=3, duration=20.0, det_hz=3.0, frame=(320, 240),
               miss_p=0.10, low_conf_p=0.15, fp_rate=0.05, jitter=2.0, fast=False, cross=True):
    """Return list of (t, gt{id:box}, det_boxes, det_conf). People walk on lines that may cross."""
    rng = np.random.default_rng(seed)
    W, H = frame
    people = []
    for pid in range(n_people):
        w, h = rng.uniform(40, 70), rng.uniform(100, 170)
        sx, sy = rng.uniform(0, W - w), rng.uniform(0, max(H - h, 1))
        if cross and pid % 2 == 1:
            sx = W - w - sx
        speed = rng.uniform(10, 25) * (4.0 if fast else 1.0)
        ang = rng.uniform(0, 2 * np.pi)
        people.append(dict(w=w, h=h, x=sx, y=sy, vx=speed * np.cos(ang), vy=speed * np.sin(ang) * 0.3))
    out, t, dt = [], 0.0, 1.0 / det_hz
    while t <= duration:
        gt, det, conf = {}, [], []
        for pid, p in enumerate(people):
            p["x"] += p["vx"] * dt; p["y"] += p["vy"] * dt
            if p["x"] < 0 or p["x"] + p["w"] > W: p["vx"] *= -1; p["x"] = np.clip(p["x"], 0, W - p["w"])
            if p["y"] < 0 or p["y"] + p["h"] > H: p["vy"] *= -1; p["y"] = np.clip(p["y"], 0, max(H - p["h"], 0))
            box = np.array([p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]])
            gt[pid] = box
            if rng.random() < miss_p:
                continue
            b = box + rng.normal(0, jitter, 4)
            det.append(b)
            conf.append(rng.uniform(0.15, 0.45) if rng.random() < low_conf_p else rng.uniform(0.55, 0.95))
        if rng.random() < fp_rate:  # single-frame false positive
            x, y = rng.uniform(0, W - 40), rng.uniform(0, H - 80)
            det.append(np.array([x, y, x + 40, y + 80])); conf.append(rng.uniform(0.5, 0.7))
        out.append((t, gt, det, conf))
        t += dt
    return out


def evaluate(scene, tracker_factory, iou_thr=0.4):
    """Feed detections to tracker; return dict(idsw, idf1, frag, fp_ids)."""
    tr = tracker_factory()
    gt_ids = set(); pair_frames = {}; gt_frames = {}; trk_frames = {}
    last_match = {}; idsw = 0; frag = 0; last_seen_gt = {}
    for t, gt, det, conf in scene:
        vis, _ = tr.update(np.array(det).reshape(-1, 4), None, conf, now=t) if _accepts_now(tr) else tr.update(np.array(det).reshape(-1, 4), None, conf)
        ids = list(gt.keys())
        gtb = np.stack([gt[i] for i in ids]) if ids else np.zeros((0, 4))
        trb = np.array([v["bbox"] for v in vis]).reshape(-1, 4)
        iou = iou_matrix(gtb, trb)
        r, c = linear_sum_assignment(-iou) if iou.size else ([], [])
        matched = set()
        for i, j in zip(r, c):
            if iou[i, j] >= iou_thr:
                g, tid = ids[i], vis[j]["track_id"]
                matched.add(g)
                pair_frames[(g, tid)] = pair_frames.get((g, tid), 0) + 1
                if g in last_match and last_match[g] != tid: idsw += 1
                last_match[g] = tid
                if last_seen_gt.get(g) is False: frag += 1
                last_seen_gt[g] = True
        for g in ids:
            gt_frames[g] = gt_frames.get(g, 0) + 1
            if g not in matched: last_seen_gt[g] = False
        for v in vis:
            trk_frames[v["track_id"]] = trk_frames.get(v["track_id"], 0) + 1
    # IDF1 via global assignment on pair_frames
    gl, tl = sorted(gt_frames), sorted(trk_frames)
    if not gl or not tl:
        return dict(idsw=idsw, idf1=0.0, frag=frag, n_track_ids=len(tl))
    M = np.zeros((len(gl), len(tl)))
    for (g, t_), n in pair_frames.items(): M[gl.index(g), tl.index(t_)] = n
    r, c = linear_sum_assignment(-M)
    idtp = M[r, c].sum()
    idf1 = 2 * idtp / (sum(gt_frames.values()) + sum(trk_frames.values()))
    return dict(idsw=idsw, idf1=float(idf1), frag=frag, n_track_ids=len(tl))


def _accepts_now(tr):
    import inspect
    return "now" in inspect.signature(tr.update).parameters


class GreedyBaseline:
    """Replica of the OLD behaviour described in the audit (greedy IoU>=0.25, max age 1.5 s,
    no motion model, birth on first detection). For comparison only."""
    def __init__(self, iou=0.25, max_age=1.5):
        self.iou, self.max_age, self.t, self.n = iou, max_age, [], 1
    def update(self, boxes, cats=None, confs=None, now=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        confs = [1.0] * len(boxes) if confs is None else confs
        used = set()
        for tr in self.t:
            best, bj = 0, None
            for j, b in enumerate(boxes):
                if j in used or confs[j] < 0.45: continue
                v = iou_matrix(np.array([tr["bbox"]]), b[None])[0, 0]
                if v > best: best, bj = v, j
            if bj is not None and best >= self.iou:
                tr["bbox"], tr["last"] = tuple(boxes[bj]), now; used.add(bj)
        for j, b in enumerate(boxes):
            if j not in used and confs[j] >= 0.45:
                self.t.append({"track_id": self.n, "bbox": tuple(b), "last": now}); self.n += 1
        self.t = [x for x in self.t if now - x["last"] <= self.max_age]
        return [dict(x) for x in self.t if x["last"] == now], []
