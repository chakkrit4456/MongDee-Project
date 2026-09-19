#!/usr/bin/env python
"""Measure the REAL gender accuracy of the FairFace model MongDee uses, the way a webcam sees it.

Run on the Windows machine (needs the real torch venv):

    python tools/eval_fairface_gender.py
    python tools/eval_fairface_gender.py --n 1000 --frames 6 --device cuda

It uses dataset/val (labelled FairFace validation faces, never used for training) and degrades every
face like a 320x240 USB stream (down-scale, blur, JPEG, noise, low light). Three questions are answered:

  1. FRAME level  - accuracy / coverage of the raw model, with and without the mirror trick (TTA),
                    for each degradation and each confidence floor.
  2. PERSON level - the same faces pushed through the REAL GlobalPersonAttributeSmoother for
                    --frames frames each: how often a person ends up labelled, and how often that label
                    is WRONG (the "man labelled as woman" rate). This is the number that matters.
  3. Gate check   - PASS when the person-level error among labelled people is <= --max-error.

Writes reports/camera-ai-fix/fairface_gender_eval.json. Nothing is modified; no camera needed.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

KINDS = ["clean", "down-64px", "down-48px", "down-32px", "down-24px", "blur", "jpeg-q20", "noise", "lowlight", "combined"]
# face size (px) a "down-Npx" degradation stands for - the far-range person-level test passes it to the smoother
KIND_FACE_PX = {"down-64px": 64.0, "down-48px": 48.0, "down-32px": 32.0, "down-24px": 24.0}


def degrade(img, kind, rng):
    h, w = img.shape[:2]

    def down(px, im=img):
        s = cv2.resize(im, (px, px), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)

    def jpg(q, im=img):
        return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)

    if kind == "clean":
        return img
    if kind == "down-64px":
        return down(64)
    if kind == "down-48px":
        return down(48)
    if kind == "down-32px":
        return down(32)
    if kind == "down-24px":
        return down(24)
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    if kind == "jpeg-q20":
        return jpg(20)
    if kind == "noise":
        return np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    if kind == "lowlight":
        return np.clip(img * 0.35 + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    if kind == "combined":
        return jpg(30, cv2.GaussianBlur(down(56), (0, 0), 1.2))
    raise ValueError(kind)


def load_items(csv_path, img_root, n, seed):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rng = np.random.default_rng(seed)
    males = [r for r in rows if r["gender"].lower() == "male"]
    females = [r for r in rows if r["gender"].lower() == "female"]
    rng.shuffle(males); rng.shuffle(females)
    picked = males[: n // 2] + females[: n // 2]
    items = []
    for r in picked:
        path = os.path.join(img_root, os.path.basename(r["file"]))
        im = cv2.imread(path)
        if im is not None:
            items.append((im, r["gender"].lower()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "val", "fairface_label_val.csv"))
    ap.add_argument("--img-root", default=os.path.join(ROOT, "dataset", "val"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "fairface", "best_model_state_dict.pt"))
    ap.add_argument("--n", type=int, default=600, help="faces (half male, half female)")
    ap.add_argument("--frames", type=int, default=6, help="frames per person in the person-level test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--far", action="store_true", help="person-level test with far-range faces (24-48 px)")
    ap.add_argument("--max-error", type=float, default=0.02, help="person-level error among labelled people")
    a = ap.parse_args()

    import torch
    from core.attributes import (ATTRIBUTE_GENDER_MIN_CONFIDENCE, FairFaceBackend, GlobalPersonAttributeSmoother)

    class NoDetector:               # val images already are faces; we call the network directly
        def detect(self, crop):
            return None

    backend = FairFaceBackend(a.ckpt, NoDetector(), device=a.device)
    tta_forward = backend._forward

    def single_forward(face_bgr):   # the old, single-view behaviour, for comparison
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        t = backend._preprocess(face_rgb).unsqueeze(0).to(backend.device)
        with torch.no_grad():
            return backend._model(t).squeeze(0).cpu().numpy()

    def p_male(face, tta):
        backend._forward = tta_forward if tta else single_forward
        g, gc, _age, _ac = backend._predict_face(face)
        return gc if g == "male" else 1.0 - gc

    items = load_items(a.csv, a.img_root, a.n, a.seed)
    print(f"{len(items)} validation faces, device={backend.device}")
    rng = np.random.default_rng(a.seed)
    report = {"n": len(items), "frame_level": {}, "person_level": {}}

    # ---------------------------------------------------------------- 1. frame level
    for kind in KINDS:
        deg = [degrade(im, kind, rng) for im, _ in items]
        for tta in (False, True):
            ps = np.array([p_male(d, tta) for d in deg])
            truth = np.array([lab == "male" for _, lab in items])
            pred = ps >= 0.5
            conf = np.maximum(ps, 1 - ps)
            row = {"acc_all": float((pred == truth).mean())}
            for floor in (0.6, 0.7, 0.8, 0.9):
                m = conf >= floor
                row[f"floor{floor}"] = {"coverage": float(m.mean()),
                                        "acc_answered": float((pred[m] == truth[m]).mean()) if m.any() else None}
            report["frame_level"][f"{kind}|{'tta' if tta else 'single'}"] = row
            print(f"{kind:10s} {'TTA   ' if tta else 'single'} acc={row['acc_all']:.3f} "
                  f"@0.7 cov={row['floor0.7']['coverage']:.2f} acc={row['floor0.7']['acc_answered']}")

    # ---------------------------------------------------------------- 2. person level (real smoother)
    frame_kinds = ["down-64px", "down-48px", "blur", "jpeg-q20", "noise", "combined"]
    if a.far:                                   # far-range mix: people whose faces are only 24-48 px in the frame
        frame_kinds = ["down-48px", "down-32px", "down-24px", "blur", "combined"]
    cache = {}
    for pid, (im, lab) in enumerate(items):
        ps = []
        for f in range(a.frames):
            k = frame_kinds[int(rng.integers(len(frame_kinds)))]
            ps.append((p_male(degrade(im, k, rng), True), KIND_FACE_PX.get(k)))
        cache[pid] = (lab, ps)
    labelled = wrong = male_as_female = males = 0
    for nframes in sorted({1, 2, 3, a.frames}):
        sm = GlobalPersonAttributeSmoother()
        labelled = wrong = male_as_female = males = 0
        for pid, (lab, ps) in cache.items():
            res = None
            for f, (p, px) in enumerate(ps[:nframes]):
                g = "male" if p >= 0.5 else "female"
                res = sm.add_sample(str(pid), g, max(p, 1 - p), "unknown", 0.0, now=float(f), face_px=px)
            males += lab == "male"
            if res.gender != "UNKNOWN":
                labelled += 1
                if res.gender.lower() != lab:
                    wrong += 1
                    male_as_female += lab == "male"
        n = len(cache)
        row = {"frames": nframes, "labelled": labelled / n, "wrong_among_labelled": wrong / max(labelled, 1),
               "wrong_overall": wrong / n, "male_labelled_female": male_as_female / max(males, 1)}
        report["person_level"][f"{nframes}_frames"] = row
        print(f"PERSON {nframes} frames: labelled {row['labelled']:.0%}  wrong among labelled "
              f"{row['wrong_among_labelled']:.2%}  male->female {row['male_labelled_female']:.2%}")

    final = report["person_level"][f"{a.frames}_frames"]
    report["gate"] = {"max_error": a.max_error, "value": final["wrong_among_labelled"],
                      "pass": final["wrong_among_labelled"] <= a.max_error,
                      "min_confidence_used": ATTRIBUTE_GENDER_MIN_CONFIDENCE}
    print("GATE person-level error <= %.1f%% :" % (a.max_error * 100), "PASS" if report["gate"]["pass"] else "FAIL",
          f"({final['wrong_among_labelled']:.2%}, {final['labelled']:.0%} of people labelled)")
    out = os.path.join(ROOT, "reports", "camera-ai-fix"); os.makedirs(out, exist_ok=True)
    json.dump(report, open(os.path.join(out, "fairface_gender_eval.json"), "w", encoding="utf-8"), indent=2)
    print("-> reports/camera-ai-fix/fairface_gender_eval.json")


if __name__ == "__main__":
    t0 = time.time(); main(); print(f"done in {time.time() - t0:.0f}s")
