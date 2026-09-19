"""Training loop building blocks: device resolution, one train/val epoch,
checkpoint save/resume, and a plain CSV epoch logger."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device cuda was requested but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _disable_accelerator_health_check() -> None:
    """torch 2.x's Adam optimizer calls torch.accelerator.current_stream()
    as a "graph capture" safety check on every single .step() call — even
    when every tensor involved lives on CPU — which itself queries the CUDA
    runtime. Setting CUDA_VISIBLE_DEVICES="" doesn't reliably stop this on
    every Windows/driver combination (observed: still raises
    cudaErrorDevicesUnavailable on this project's GTX 1050). Patched to a
    harmless stub only once this process has already committed to running
    on CPU (see resolve_device_safe), so it can never mask a real problem
    on a machine where CUDA genuinely works."""

    class _NullStream:
        def is_capturing(self) -> bool:
            return False

    try:
        torch.accelerator.current_stream = lambda *a, **k: _NullStream()
    except Exception:
        pass


def resolve_device_safe(requested: str = "auto") -> torch.device:
    """Like resolve_device, but also probes the device with a real tensor
    move before committing to it. torch.cuda.is_available() only checks
    that a CUDA-capable device enumerates — on this project's own
    documented hardware (GTX 1050 under Windows/Optimus-style power
    management, see core/attributes.py's FairFaceBackend), a GPU can
    enumerate as available and then still refuse every actual operation
    with cudaErrorDevicesUnavailable. Same "never crash the run over a GPU
    that isn't really there" fallback already used elsewhere in this repo,
    applied here so a multi-hour training run doesn't die a few seconds in."""
    device = resolve_device(requested)
    if device.type != "cuda":
        _disable_accelerator_health_check()
        return device
    try:
        torch.zeros(1, device=device)
        return device
    except Exception as exc:
        print(f"[WARN] CUDA device enumerated but is not actually usable ({exc}); falling back to CPU.")
        _disable_accelerator_health_check()
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class EpochLogger:
    """Appends one row per epoch to logs/<run_name>.csv — plain CSV so it's
    readable without any extra dependency, per spec section 8."""

    FIELDS = ["epoch", "train_loss", "val_loss", "age_accuracy", "gender_accuracy",
              "race_accuracy", "lr", "seconds"]

    def __init__(self, log_dir: str | Path, run_name: str = "fairface_train"):
        self.path = Path(log_dir) / f"{run_name}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, **row) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.FIELDS])


def compute_loss(outputs: dict, labels: dict, weights: dict, criterion: nn.Module) -> tuple[torch.Tensor, dict]:
    losses = {task: criterion(outputs[task], labels[task]) for task in ("age", "gender", "race")}
    total = sum(weights[f"{task}_weight"] * losses[task] for task in losses)
    return total, {task: loss.item() for task, loss in losses.items()}


def _accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    preds = outputs.argmax(dim=1)
    correct = int((preds == targets).sum().item())
    return correct, targets.size(0)


def run_epoch(model: nn.Module, loader, criterion, weights: dict, device: torch.device,
              optimizer=None, scaler=None, max_batches: int | None = None, desc: str | None = None) -> dict:
    """One pass over `loader`. Training when optimizer is given, evaluation
    (no_grad, no augmentation-sensitive state change) otherwise. `max_batches`
    lets --check/smoke runs cap iteration count without a separate code path."""
    from tqdm import tqdm

    train_mode = optimizer is not None
    model.train(train_mode)

    loss_meter = AverageMeter()
    correct = {"age": 0, "gender": 0, "race": 0}
    total_samples = 0

    total_batches = max_batches if max_batches is not None else len(loader)
    iterable = tqdm(loader, desc=desc, leave=False, total=total_batches) if desc else loader
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(iterable):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in labels.items()}

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                    loss, _ = compute_loss(outputs, labels, weights, criterion)
                if train_mode:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(images)
                loss, _ = compute_loss(outputs, labels, weights, criterion)
                if train_mode:
                    loss.backward()
                    optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            total_samples += batch_size
            for task in ("age", "gender", "race"):
                c, _ = _accuracy(outputs[task], labels[task])
                correct[task] += c
            if desc:
                iterable.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return {
        "loss": loss_meter.avg,
        "age_accuracy": correct["age"] / total_samples if total_samples else 0.0,
        "gender_accuracy": correct["gender"] / total_samples if total_samples else 0.0,
        "race_accuracy": correct["race"] / total_samples if total_samples else 0.0,
        "samples": total_samples,
    }


def save_checkpoint(path: str | Path, model: nn.Module, optimizer, epoch: int,
                     best_val_score: float, label_mappings: dict, config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_score": best_val_score,
        "label_mappings": label_mappings,
        "config": config,
    }, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ERROR: model checkpoint was not found.\n\nExpected:\n{checkpoint_path}\n"
        )
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"ERROR: model checkpoint at {checkpoint_path} could not be loaded "
            f"(file may be corrupt or from an incompatible torch version).\nDetails: {exc}"
        ) from exc


def save_label_mappings(path: str | Path, label_mappings: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(label_mappings, indent=2, ensure_ascii=False), encoding="utf-8")


def save_training_config(path: str | Path, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


class Timer:
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._start
