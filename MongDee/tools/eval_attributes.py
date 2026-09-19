#!/usr/bin/env python
"""Measure gender accuracy of ANY classifier under the degradations of a 320x240 USB stream.

Dataset layout (any face images; FairFace val, LFW, your own consented photos):
    <root>/male/*.jpg   <root>/female/*.jpg
Classifier: "package.module:function"  taking a BGR uint8 face crop and returning
    p_male (float 0..1)  OR  ("male"/"female", confidence).

    python tools/eval_attributes.py --root D:/data/val --clf my_adapter:predict --n 500
Reports accuracy AND coverage at abstain thresholds, per degradation:
    clean | down-32px | down-48px | blur | jpeg-q20 | noise | lowlight | combined(48px+blur+jpeg)
Reproducible: fixed seed. Writes reports/camera-ai-fix/attr_eval.json.
"""
import argparse, glob, importlib, json, os, random, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def degrade(img, kind, rng):
    h, w = img.shape[:2]
    def down(px, im=img):
        s = cv2.resize(im, (px, px), interpolation=cv2.INTER_AREA)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    def jpg(q, im=img):
        return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    if kind == "clean": return img
    if kind == "down-32px": return down(32)
    if kind == "down-48px": return down(48)
    if kind == "blur": return cv2.GaussianBlur(img, (0, 0), 2.0)
    if kind == "jpeg-q20": return jpg(20)
    if kind == "noise": return np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    if kind == "lowlight": return np.clip(img * 0.3 + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    if kind == "combined": return jpg(30, cv2.GaussianBlur(down(48), (0, 0), 1.2))
    raise ValueError(kind)


KINDS = ["clean", "down-32px", "down-48px", "blur", "jpeg-q20", "noise", "lowlight", "combined"]


def load(root, n, rng):
    items = []
    for lab in ("male", "female"):
        fs = sorted(glob.glob(os.path.join(root, lab, "*.*")))
        rng.shuffle(fs); items += [(f, lab) for f in fs[: n // 2]]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--clf", required=True)
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--abstain", type=float, default=0.75, help="abstain when max(p,1-p) < this")
    a = ap.parse_args()
    mod, fn = a.clf.split(":"); clf = getattr(importlib.import_module(mod), fn)
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    items = load(a.root, a.n, random.Random(a.seed))
    imgs = [(cv2.imread(f), lab) for f, lab in items]; imgs = [(i, l) for i, l in imgs if i is not None]
    rep = {}
    for k in KINDS:
        right = wrong = abst = 0; man_as_woman = 0; men = 0
        for im, lab in imgs:
            r = clf(degrade(im, k, rng))
            p = float(r) if not isinstance(r, tuple) else (r[1] if r[0] == "male" else 1 - r[1])
            men += lab == "male"
            if max(p, 1 - p) < a.abstain: abst += 1; continue
            pred = "male" if p >= 0.5 else "female"
            if pred == lab: right += 1
            else:
                wrong += 1; man_as_woman += (lab == "male")
        n = len(imgs); answered = right + wrong
        rep[k] = dict(n=n, acc_all=round(right / max(n, 1), 4), acc_answered=round(right / max(answered, 1), 4),
                      coverage=round(answered / max(n, 1), 4), male_labelled_female=round(man_as_woman / max(men, 1), 4))
        print(k, rep[k])
    os.makedirs(os.path.join(ROOT, "reports", "camera-ai-fix"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "reports", "camera-ai-fix", "attr_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
