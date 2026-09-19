#!/usr/bin/env python
"""Make the FairFace GENDER head reliable on what the booth webcams really see (small, soft, dim, half-hidden faces).

    python tools/finetune_fairface_gender.py --dry-run          # no torch: writes a picture of the degraded training faces
    python tools/finetune_fairface_gender.py                    # fine-tune + calibrate + (only if better) promote
    python tools/finetune_fairface_gender.py --epochs 3 --train-n 60000 --device cuda

What it does (uses dataset/train for training and dataset/val for measuring - never the same images):
  1. Measures the CURRENT model on webcam-like degraded validation faces: accuracy, female recall, male recall,
     per face size (this is the honest baseline; on clean FairFace it is ~95%).
  2. Fits a decision-bias calibration (core/gender_calibration.py) so female and male recall are equal - free at run time.
  3. Fine-tunes ResNet34 on degraded faces (core/face_degrade.py: 14-160 px faces, blur, JPEG, noise, exposure, colour
     cast, hats / masks / shadows, crop jitter) with balanced classes; only the gender output is trained, so the age /
     race outputs of the new file are NOT meaningful (MongDee does not use them).
  4. Calibrates the fine-tuned model on one half of the validation faces and scores it on the other half.
  5. PROMOTES only what is measurably better, and never at the cost of clean accuracy:
       models/fairface/webcam_model_state_dict.pt   (picked up automatically by app.py / web_server.py)
       models/fairface/gender_calibration.json      (bias fitted for whichever checkpoint won)
     Otherwise nothing changes. The full comparison is written to reports/camera-ai-fix/finetune_gender_report.json.
Set MONGDEE_FAIRFACE_ORIGINAL=1 at run time to force the original weights back.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.face_degrade import OUT_SIZE, degrade_face, sample_face_px  # noqa: E402
from core.gender_calibration import (SIZE_EDGES, GenderCalibration, bucket_index, fit_calibration)  # noqa: E402

try:                                    # torch is optional (--dry-run needs none); the Dataset must live at module
    import torch                        # level so DataLoader workers can pickle it on Windows (spawn)
    from torch.utils.data import Dataset
except Exception:                       # pragma: no cover
    torch = None
    Dataset = object

NUM_RACE = 7                      # fc layout: race(7) + gender(2) + age(9); gender = [male, female]
BUCKET_NAMES = [f"<{SIZE_EDGES[0]}px"] + [f"{a}-{b}px" for a, b in zip(SIZE_EDGES, SIZE_EDGES[1:])] + [f">={SIZE_EDGES[-1]}px"]
CLEAN_PX = 224.0
ASIAN_RACES = ("east asian", "southeast asian")     # FairFace race labels; Thai faces are 'Southeast Asian'


# ------------------------------------------------------------------ data
def read_rows(dataset_root: str, split: str):
    path = os.path.join(dataset_root, split, f"fairface_label_{split}.csv")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g = r["gender"].strip().lower()
            if g not in ("male", "female"):
                continue
            img = os.path.join(dataset_root, r["file"])
            if not os.path.isfile(img):
                img = os.path.join(dataset_root, split, os.path.basename(r["file"]))
            rows.append((img, 1 if g == "female" else 0, r.get("race", "").strip().lower() in ASIAN_RACES))
    return rows


def read_extra(directory):
    """Faces this booth saved (MONGDEE_SAVE_FACES=1) and a person sorted into <directory>/male and <directory>/female.
    Row = (path, gender, is_asian=True, raw=True): already webcam-quality, so they are used as they are."""
    rows = []
    for name, y in (("male", 0), ("female", 1)):
        folder = os.path.join(directory, name)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                rows.append((os.path.join(folder, fn), y, True, True))
    return rows


def split_extra(rows, holdout_every=5):
    """Deterministic split of the booth's own faces: every 5th goes to the test set, never trained on."""
    train = [r for i, r in enumerate(rows) if i % holdout_every != 0]
    test = [r for i, r in enumerate(rows) if i % holdout_every == 0]
    return train, test


def balanced_subset(rows, n, seed):
    rng = np.random.default_rng(seed)
    m = [r for r in rows if r[1] == 0]
    f = [r for r in rows if r[1] == 1]
    rng.shuffle(m)
    rng.shuffle(f)
    half = n // 2
    return m[:half], f[:half]


def make_item(img_bgr, rng, p_clean, min_strength):
    """One training / evaluation sample: (uint8 BGR 224x224, face_px)."""
    if rng.random() < p_clean:
        out = cv2.resize(img_bgr, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA if min(img_bgr.shape[:2]) > OUT_SIZE
                         else cv2.INTER_CUBIC)
        return (out[:, ::-1] if rng.random() < 0.5 else out), CLEAN_PX
    px = sample_face_px(rng)
    strength = float(rng.uniform(min_strength, 1.0))
    return degrade_face(img_bgr, rng, face_px=px, strength=strength), px



_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def to_tensor(bgr):
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array(_MEAN, np.float32)) / np.array(_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class FaceSet(Dataset):
    """(image tensor, gender 0=male/1=female, face_px). A fixed set is reproducible; a training set re-rolls every epoch."""

    def __init__(self, rows, seed, p_clean, min_strength, fixed):
        self.rows, self.seed, self.p_clean, self.min_strength, self.fixed = rows, seed, p_clean, min_strength, fixed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, y = self.rows[i][0], self.rows[i][1]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((224, 224, 3), np.uint8)
        rng = np.random.default_rng([self.seed, 0 if self.fixed else self.epoch, i])
        if len(self.rows[i]) > 3 and self.rows[i][3]:          # a real booth face: keep its own size / quality, only mild jitter
            px = float(min(img.shape[:2]))
            arr, px = degrade_face(img, rng, face_px=max(px, 8.0), strength=0.0 if self.fixed else 0.2), px
        else:
            arr, px = make_item(img, rng, self.p_clean, self.min_strength)
        return to_tensor(arr), y, float(px)


# ------------------------------------------------------------------ metrics
def summarise(diff, y, px, bias_fn=None):
    """Accuracy / recalls overall and per face-size bucket for predictions `diff + bias > 0` = female."""
    diff, y, px = np.asarray(diff, float), np.asarray(y, int), np.asarray(px, float)
    bias = np.array([bias_fn(p) for p in px]) if bias_fn else np.zeros_like(diff)
    pred = (diff + bias) > 0

    def block(m):
        if not m.any():
            return None
        f, mm = m & (y == 1), m & (y == 0)
        rf = float(pred[f].mean()) if f.any() else None
        rm = float((~pred[mm]).mean()) if mm.any() else None
        vals = [v for v in (rf, rm) if v is not None]
        return {"n": int(m.sum()), "accuracy": float((pred[m] == (y[m] == 1)).mean()),
                "female_recall": rf, "male_recall": rm, "balanced": float(np.mean(vals)) if vals else None}

    out = {"all": block(np.ones_like(y, bool))}
    b = np.array([bucket_index(p) for p in px])
    for i, name in enumerate(BUCKET_NAMES):
        out[name] = block(b == i)
    return out


def fmt(block):
    if not block:
        return "   -"
    rf = "  - " if block["female_recall"] is None else f"{block['female_recall']:.3f}"
    rm = "  - " if block["male_recall"] is None else f"{block['male_recall']:.3f}"
    return f"n={block['n']:5d} acc={block['accuracy']:.3f} female_recall={rf} male_recall={rm}"


def print_table(title, summary):
    print(f"\n== {title}")
    for k, v in summary.items():
        print(f"  {k:10s} {fmt(v)}")


# ------------------------------------------------------------------ dry run (no torch)
def dry_run(args):
    rows = read_rows(args.dataset, "val")
    m, f = balanced_subset(rows, 16, args.seed)
    rng = np.random.default_rng(args.seed)
    tiles = []
    for path, *_rest in (m[:8] + f[:8]):
        img = cv2.imread(path)
        if img is None:
            continue
        clean = cv2.resize(img, (112, 112))
        row = [clean] + [cv2.resize(make_item(img, rng, 0.0, 0.4)[0], (112, 112)) for _ in range(5)]
        tiles.append(np.hstack(row))
    sheet = np.vstack(tiles)
    out = os.path.join(ROOT, "reports", "camera-ai-fix")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "degrade_preview.png")
    cv2.imwrite(p, sheet)
    print("first column = clean photo, the other five = what the network is trained / tested on ->", p)


# ------------------------------------------------------------------ torch part
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "fairface", "best_model_state_dict.pt"))
    ap.add_argument("--out-dir", default=None, help="where to write the new files (default: next to --ckpt)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-n", type=int, default=40000, help="training faces per epoch (half male, half female)")
    ap.add_argument("--val-n", type=int, default=3000, help="validation faces for calibration and for testing each")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--asian-boost", type=float, default=2.0,
                    help="how much more often East / Southeast Asian faces are drawn for training (1 = no boost)")
    ap.add_argument("--extra-faces", default=os.path.join(ROOT, "data", "face_labels"),
                    help="folder with male/ and female/ sub-folders of this booth's own faces (see MONGDEE_SAVE_FACES)")
    ap.add_argument("--extra-repeat", type=int, default=3, help="how many times each booth face is repeated per epoch")
    ap.add_argument("--clean-fraction", type=float, default=0.15, help="share of training faces left undamaged")
    ap.add_argument("--min-gain", type=float, default=0.01, help="balanced-accuracy gain needed to promote")
    ap.add_argument("--max-clean-drop", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-promote", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="only write a picture of the degraded faces (needs no torch)")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)

    if torch is None:
        sys.exit("PyTorch is not installed in this environment (use the project's .venv).")
    from torch.utils.data import DataLoader
    from torchvision.models import resnet34

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    if device.type == "cpu":
        print("WARNING: no CUDA device - training on CPU is very slow; reduce --train-n (e.g. 8000) or use a GPU.")
    ckpt_dir = args.out_dir or os.path.dirname(args.ckpt)
    orig_name = os.path.basename(args.ckpt)
    def build_model(path):
        m = resnet34()
        m.fc = torch.nn.Linear(m.fc.in_features, NUM_RACE + 2 + 9)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.to(device)

    @torch.no_grad()
    def gender_diffs(model, ds):
        """female-minus-male logit per sample, with the same mirror averaging the live pipeline uses."""
        model.eval()
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.num_workers)
        diffs, ys, pxs = [], [], []
        for x, y, px in dl:
            x = x.to(device)
            lo = (model(x) + model(torch.flip(x, dims=[3]))) / 2
            g = lo[:, NUM_RACE:NUM_RACE + 2].float().cpu().numpy()
            diffs.append(g[:, 1] - g[:, 0])
            ys.append(y.numpy())
            pxs.append(px.numpy())
        return np.concatenate(diffs), np.concatenate(ys), np.concatenate(pxs)

    # ---------------- data
    train_rows_all = read_rows(args.dataset, "train")
    val_rows_all = read_rows(args.dataset, "val")
    print(f"dataset: {len(train_rows_all)} train / {len(val_rows_all)} val faces with a gender label")
    vm, vf = balanced_subset(val_rows_all, args.val_n * 2, args.seed + 1)
    half = len(vm) // 2
    calib_rows = vm[:half] + vf[:half]
    test_rows = vm[half:] + vf[half:]
    print(f"validation: {len(calib_rows)} faces to calibrate on, {len(test_rows)} to test on (different images)")

    def val_sets():
        return (FaceSet(calib_rows, args.seed + 11, 0.0, 0.5, True), FaceSet(test_rows, args.seed + 12, 0.0, 0.5, True),
                FaceSet(test_rows[::2], args.seed + 13, 1.0, 0.0, True))     # degraded calib, degraded test, CLEAN test

    extra_train, extra_test = split_extra(read_extra(args.extra_faces))
    if extra_train or extra_test:
        print(f"booth's own faces: {len(extra_train)} for training, {len(extra_test)} held out for testing ({args.extra_faces})")
    report = {"args": vars(args), "device": str(device), "extra_faces": {"train": len(extra_train), "test": len(extra_test)}}

    def evaluate(model, name, calibrate=True):
        calib_ds, test_ds, clean_ds = val_sets()
        d_c, y_c, p_c = gender_diffs(model, calib_ds)
        d_t, y_t, p_t = gender_diffs(model, test_ds)
        d_k, y_k, p_k = gender_diffs(model, clean_ds)
        asian = np.array([bool(r[2]) for r in test_ds.rows])
        raw = summarise(d_t, y_t, p_t)
        cal = fit_calibration(d_c, y_c == 1, p_c, checkpoint="", info={"fitted_on": len(y_c)})
        calibrated = summarise(d_t, y_t, p_t, cal.bias_for)
        asian_cal = summarise(d_t[asian], y_t[asian], p_t[asian], cal.bias_for) if asian.any() else None
        asian_raw = summarise(d_t[asian], y_t[asian], p_t[asian]) if asian.any() else None
        booth = None
        if len(extra_test) >= 20:
            d_b, y_b, p_b = gender_diffs(model, FaceSet(extra_test, args.seed + 14, 0.0, 0.0, True))
            booth = summarise(d_b, y_b, p_b, cal.bias_for)
            print_table(f"{name}: THIS BOOTH's own held-out faces, calibrated", booth)
        if asian_cal:
            print_table(f"{name}: East/Southeast Asian faces only (Thai people are in this group), calibrated", asian_cal)
        clean = summarise(d_k, y_k, p_k)
        clean_cal = summarise(d_k, y_k, p_k, cal.bias_for)
        print_table(f"{name}: webcam-like faces, raw", raw)
        print_table(f"{name}: webcam-like faces, calibrated (bias {cal.bias:+.2f}, per size {cal.bucket_bias})", calibrated)
        print(f"  clean FairFace faces: raw acc {clean['all']['accuracy']:.3f}, calibrated acc {clean_cal['all']['accuracy']:.3f}")
        report[name] = {"booth": booth, "asian_raw": asian_raw, "asian_calibrated": asian_cal, "raw": raw, "calibrated": calibrated, "clean_raw": clean, "clean_calibrated": clean_cal,
                        "calibration": cal.to_dict()}
        return cal

    # ---------------- 1. baseline
    t0 = time.time()
    baseline = build_model(args.ckpt)
    base_cal = evaluate(baseline, "baseline")
    base_raw_bal = report["baseline"]["raw"]["all"]["balanced"]
    base_cal_bal = report["baseline"]["calibrated"]["all"]["balanced"]
    base_raw_bal_focus = 0.5 * (base_raw_bal + report["baseline"]["asian_raw"]["all"]["balanced"]) if report["baseline"]["asian_raw"] else base_raw_bal
    base_cal_bal_focus = 0.5 * (base_cal_bal + report["baseline"]["asian_calibrated"]["all"]["balanced"]) if report["baseline"]["asian_calibrated"] else base_cal_bal
    print(f"\nbaseline evaluated in {time.time() - t0:.0f}s")

    # ---------------- 2. fine-tune
    model = build_model(args.ckpt)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")], "lr": args.lr},
                             {"params": model.fc.parameters(), "lr": args.lr * 3}], weight_decay=1e-4)
    steps_per_epoch = (args.train_n + len(extra_train) * max(0, args.extra_repeat)) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 3], total_steps=max(1, steps_per_epoch * args.epochs),
                                                pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    tm, tf = balanced_subset(train_rows_all, len(train_rows_all), args.seed)
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + 100 + epoch)

        def pick(rows_, n):                                   # East / Southeast Asian faces are drawn `asian_boost` x as often
            w = np.array([args.asian_boost if r[2] else 1.0 for r in rows_], dtype=np.float64)
            return rng.choice(len(rows_), size=min(n, len(rows_)), replace=False, p=w / w.sum())

        m_pick = pick(tm, args.train_n // 2)
        f_pick = pick(tf, args.train_n // 2)
        rows = [tm[i] for i in m_pick] + [tf[i] for i in f_pick] + extra_train * max(0, args.extra_repeat)
        ds = FaceSet(rows, args.seed + 1000, args.clean_fraction, 0.35, False)
        ds.epoch = epoch
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, drop_last=True,
                        persistent_workers=args.num_workers > 0)
        model.train()
        run_loss = run_acc = seen = 0.0
        t1 = time.time()
        for step, (x, y, _px) in enumerate(dl):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
                loss = crit(out[:, NUM_RACE:NUM_RACE + 2].float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss) * len(y)
            run_acc += float((out[:, NUM_RACE:NUM_RACE + 2].argmax(1) == y).float().sum())
            seen += len(y)
            if (step + 1) % 100 == 0:
                eta = (time.time() - t1) / (step + 1) * (steps_per_epoch - step - 1)
                print(f"epoch {epoch + 1}/{args.epochs} step {step + 1}/{steps_per_epoch} loss {run_loss / seen:.4f} "
                      f"acc {run_acc / seen:.3f}  ~{eta / 60:.0f} min left in this epoch", flush=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "webcam_candidate.pt"))
        print(f"epoch {epoch + 1} done in {(time.time() - t1) / 60:.1f} min -> candidate saved")

    # ---------------- 3. compare and promote
    new_cal = evaluate(model, "finetuned")
    def focus(name, kind="calibrated"):
        """The number the decision uses: mean of overall and Asian-only balanced accuracy (the booth is in Thailand)."""
        overall = report[name][kind]["all"]["balanced"]
        asian_ = report[name]["asian_" + kind]
        return overall if not asian_ else 0.5 * (overall + asian_["all"]["balanced"])

    new_bal = focus("finetuned")
    base_clean = report["baseline"]["clean_calibrated"]["all"]["accuracy"]
    new_clean = report["finetuned"]["clean_calibrated"]["all"]["accuracy"]
    decision = {"metric": "mean of overall and East/Southeast-Asian balanced accuracy on webcam-like faces",
                "baseline_raw": base_raw_bal_focus, "baseline_calibrated": base_cal_bal_focus,
                "finetuned_calibrated": new_bal, "clean_acc_baseline": base_clean, "clean_acc_finetuned": new_clean}
    print(f"\nbalanced accuracy on webcam-like faces (mean of all + Asian-only): baseline {base_raw_bal_focus:.3f} -> "
          f"calibrated {base_cal_bal_focus:.3f} -> fine-tuned+calibrated {new_bal:.3f}")
    promoted = "nothing"
    cal_path = os.path.join(ckpt_dir, "gender_calibration.json")
    booth_ok = True
    if report["baseline"].get("booth") and report["finetuned"].get("booth"):
        booth_ok = report["finetuned"]["booth"]["all"]["accuracy"] >= report["baseline"]["booth"]["all"]["accuracy"] - 0.02
        decision["booth_accuracy"] = {"baseline": report["baseline"]["booth"]["all"]["accuracy"],
                                      "finetuned": report["finetuned"]["booth"]["all"]["accuracy"]}
    if (new_bal >= max(base_raw_bal_focus, base_cal_bal_focus) + args.min_gain and booth_ok
            and new_clean >= base_clean - args.max_clean_drop):
        promoted = "finetuned"
    elif base_cal_bal_focus >= base_raw_bal_focus + args.min_gain:
        promoted = "calibration_only"
    print(f"decision: {promoted}")
    decision["promoted"] = promoted
    if promoted != "nothing" and not args.no_promote:
        if promoted == "finetuned":
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "webcam_model_state_dict.pt"))
            new_cal.checkpoint = "webcam_model_state_dict.pt"
            new_cal.info.update(source="finetune_fairface_gender.py", balanced_accuracy_focus=new_bal)
            new_cal.save(cal_path)
            print("wrote webcam_model_state_dict.pt + gender_calibration.json - restart MongDee to use them")
        else:
            base_cal.checkpoint = orig_name
            base_cal.info.update(source="finetune_fairface_gender.py (calibration only)", balanced_accuracy_focus=base_cal_bal_focus)
            base_cal.save(cal_path)
            print("wrote gender_calibration.json for the original checkpoint - restart MongDee to use it")
    report["decision"] = decision
    out = os.path.join(ROOT, "reports", "camera-ai-fix")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "finetune_gender_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("-> reports/camera-ai-fix/finetune_gender_report.json")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done in {(time.time() - t) / 60:.1f} min")
