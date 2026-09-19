"""FairFace Age/Gender/Race multi-task training.

    python train.py --check-dataset          # validate dataset/, no training
    python train.py                           # train with config.yaml / built-in defaults
    python train.py --epochs 20 --batch-size 32 --lr 0.0001 --num-workers 4
    python train.py --resume models/fairface/last_model.pt

Dataset root defaults to "dataset/" (see config.yaml) and is never
hard-coded anywhere else in this project.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_yaml_defaults(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        print(f"[WARN] pyyaml not installed — ignoring {path}, using built-in defaults.")
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv=None):
    defaults = _load_yaml_defaults(ROOT / "config.yaml")
    d_dataset = defaults.get("dataset", {})
    d_train = defaults.get("training", {})
    d_loss = defaults.get("loss_weights", {})
    d_model = defaults.get("model", {})
    d_output = defaults.get("output", {})

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=d_dataset.get("root", "dataset"))
    parser.add_argument("--img-size", type=int, default=d_dataset.get("image_size", 224))
    parser.add_argument("--num-workers", type=int, default=d_dataset.get("num_workers", 4))

    parser.add_argument("--epochs", type=int, default=d_train.get("epochs", 20))
    parser.add_argument("--batch-size", type=int, default=d_train.get("batch_size", 32))
    parser.add_argument("--lr", type=float, default=d_train.get("lr", 1e-4))
    parser.add_argument("--weight-decay", type=float, default=d_train.get("weight_decay", 1e-4))
    parser.add_argument("--seed", type=int, default=d_train.get("seed", 42))
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    parser.add_argument("--age-weight", type=float, default=d_loss.get("age_weight", 1.0))
    parser.add_argument("--gender-weight", type=float, default=d_loss.get("gender_weight", 1.0))
    parser.add_argument("--race-weight", type=float, default=d_loss.get("race_weight", 1.0))

    parser.add_argument("--architecture", default=d_model.get("architecture", "resnet34"),
                         choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=d_model.get("pretrained", True))
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--model-dir", default=d_output.get("model_dir", "models/fairface"))
    parser.add_argument("--log-dir", default=d_output.get("log_dir", "logs"))

    parser.add_argument("--resume", default=None, metavar="CHECKPOINT_PATH")
    parser.add_argument("--check-dataset", action="store_true", help="Validate the dataset and exit, no training.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Debug/smoke-test: truncate the train set to this many rows.")
    parser.add_argument("--max-val-samples", type=int, default=None,
                         help="Debug/smoke-test: truncate the val set to this many rows.")
    return parser.parse_args(argv)


def cmd_check_dataset(args) -> int:
    from src.dataset import DatasetError, validate_dataset

    print(f"Checking dataset at: {Path(args.dataset).resolve()}\n")
    try:
        report, _train_rows, _val_rows = validate_dataset(args.dataset, check_images=True)
    except DatasetError as exc:
        print(str(exc))
        return 1
    print(report.format_report())
    return 0 if report.is_ready else 1


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.check_dataset:
        return cmd_check_dataset(args)

    # Heavy imports deferred past --check-dataset so that path stays fast
    # and doesn't require torch/torchvision to be importable at all.
    # Must run before the first `import torch` — see src/gpu_probe.py.
    if args.device != "cpu":
        from src.gpu_probe import ensure_cuda_env
        ensure_cuda_env()

    import torch
    from torch.utils.data import DataLoader

    from src.dataset import (
        DatasetError, FairFaceDataset, build_label_mappings, build_transforms, validate_dataset,
    )
    from src.model import build_model
    from src.train_utils import (
        EpochLogger, load_checkpoint, resolve_device_safe, run_epoch, save_checkpoint,
        save_label_mappings, save_training_config, set_seed,
    )

    set_seed(args.seed)

    print(f"Checking dataset at: {Path(args.dataset).resolve()}")
    try:
        report, train_rows, val_rows = validate_dataset(args.dataset, check_images=True)
    except DatasetError as exc:
        print(str(exc))
        return 1
    if not report.is_ready:
        print(report.format_report())
        print("\nERROR: dataset is not ready for training (see report above).")
        return 1
    print(f"Train images: {report.train.usable_rows}  |  Validation images: {report.val.usable_rows}")

    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
        print(f"[SMOKE TEST] train set truncated to {len(train_rows)} samples (--max-train-samples)")
    if args.max_val_samples:
        val_rows = val_rows[: args.max_val_samples]
        print(f"[SMOKE TEST] val set truncated to {len(val_rows)} samples (--max-val-samples)")

    label_mappings = build_label_mappings(report.age_classes, report.gender_classes, report.race_classes)

    device = resolve_device_safe(args.device)
    print(f"Device: {device.type}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    amp_enabled = args.amp if args.amp is not None else (device.type == "cuda")
    if amp_enabled and device.type != "cuda":
        print("[WARN] Mixed precision requested but no CUDA device — disabling AMP.")
        amp_enabled = False

    train_dataset = FairFaceDataset(train_rows, label_mappings, transform=build_transforms(args.img_size, train=True))
    val_dataset = FairFaceDataset(val_rows, label_mappings, transform=build_transforms(args.img_size, train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                               persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=args.num_workers > 0)

    model = build_model(
        architecture=args.architecture,
        num_age=len(label_mappings["age"]["classes"]),
        num_gender=len(label_mappings["gender"]["classes"]),
        num_race=len(label_mappings["race"]["classes"]),
        pretrained=args.pretrained,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device="cuda") if amp_enabled else None
    loss_weights = {"age_weight": args.age_weight, "gender_weight": args.gender_weight, "race_weight": args.race_weight}

    config = {
        "architecture": args.architecture,
        "image_size": args.img_size,
        "loss_weights": loss_weights,
        "dataset_root": str(Path(args.dataset).resolve()),
    }

    start_epoch = 0
    best_val_score = -1.0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, map_location=device)
        if checkpoint["label_mappings"] != label_mappings:
            print("[WARN] Resumed checkpoint's label mappings differ from the current dataset's — "
                  "continuing with the checkpoint's own mappings to keep the model's output layout consistent.")
            label_mappings = checkpoint["label_mappings"]
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_score = checkpoint["best_val_score"]
        print(f"Resumed at epoch {start_epoch}, best_val_score={best_val_score:.4f}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_label_mappings(model_dir / "label_mappings.json", label_mappings)
    save_training_config(model_dir / "training_config.json", {**config, "epochs": args.epochs,
                                                                "batch_size": args.batch_size, "lr": args.lr})
    logger = EpochLogger(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        import time
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, loss_weights, device,
                                   optimizer=optimizer, scaler=scaler, desc="train")
        val_metrics = run_epoch(model, val_loader, criterion, loss_weights, device, desc="val")
        elapsed = time.time() - t0

        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Age Accuracy: {val_metrics['age_accuracy']:.4f}")
        print(f"Gender Accuracy: {val_metrics['gender_accuracy']:.4f}")
        print(f"Race Accuracy: {val_metrics['race_accuracy']:.4f}")
        print(f"Epoch time: {elapsed:.1f}s")

        logger.log(epoch=epoch + 1, train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
                   age_accuracy=val_metrics["age_accuracy"], gender_accuracy=val_metrics["gender_accuracy"],
                   race_accuracy=val_metrics["race_accuracy"], lr=args.lr, seconds=round(elapsed, 1))

        # best_val_score must be updated *before* last_model.pt is saved —
        # --resume trusts last_model.pt's stored best_val_score to decide
        # whether a later epoch counts as a new best, so saving it with a
        # stale value could let a genuinely worse epoch overwrite an
        # already-better best_model.pt after a resume.
        val_score = (val_metrics["age_accuracy"] + val_metrics["gender_accuracy"] + val_metrics["race_accuracy"]) / 3
        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score

        save_checkpoint(model_dir / "last_model.pt", model, optimizer, epoch, best_val_score, label_mappings, config)

        if is_best:
            print(f"Saving best model... (val_score={val_score:.4f})")
            save_checkpoint(model_dir / "best_model.pt", model, optimizer, epoch, best_val_score, label_mappings, config)
            if len(label_mappings["race"]["classes"]) == 7 and len(label_mappings["gender"]["classes"]) == 2:
                import torch as _torch
                _torch.save(model.export_combined_state_dict(), model_dir / "best_model_state_dict.pt")

    print("\nTraining complete.")
    print(f"Best model: {model_dir / 'best_model.pt'} (val_score={best_val_score:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
