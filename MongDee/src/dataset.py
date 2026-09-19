"""FairFace dataset discovery, validation and torch Dataset.

Never assumes a fixed layout or hard-codes an image path — everything here
is derived by inspecting whatever is actually on disk under the dataset
root (default: "dataset/", never hard-coded anywhere else; always plumbed
through from train.py's --dataset flag / config.yaml's dataset.root).

Two layouts are recognized automatically:
  1. dataset/train_labels.csv + dataset/val_labels.csv, images under
     dataset/train/ + dataset/val/  (the layout named in the spec)
  2. dataset/train/fairface_label_train.csv + dataset/val/fairface_label_val.csv,
     images alongside their own CSV (the official FairFace release layout —
     what this project's dataset/ folder actually contains)
A generic fallback additionally globs for any *.csv under the dataset root
(depth <= 2) and classifies each as train/val by filename, so a differently
named dataset still works without code changes.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from pathlib import Path

logger = logging.getLogger("mongdee.fairface.dataset")

REQUIRED_COLUMNS = ("file", "age", "gender", "race")

# Canonical label orders — used only when they *match* the real values found
# in the CSV (verified, never assumed). Preferred because they line up with
# this repo's existing vision/core.attributes.py FairFaceBackend conventions
# (gender: male=0/female=1, age: 9 FairFace buckets) and the official
# joojs/fairface predict.py race ordering, which keeps a trained checkpoint
# here interoperable with that existing code without a remapping step.
GENDER_CANONICAL = ["Male", "Female"]
AGE_CANONICAL = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
RACE_CANONICAL = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian",
                   "Indian", "Middle Eastern"]

# Display-only normalization so the on-screen/report label matches this
# repo's existing "70+" convention (core/attributes.py) instead of the raw
# CSV string "more than 70" — purely cosmetic, never used as a dict key.
AGE_DISPLAY_OVERRIDES = {"more than 70": "70+"}


def age_display(label: str) -> str:
    return AGE_DISPLAY_OVERRIDES.get(label, label)


class DatasetError(Exception):
    """Raised only for hard failures (nothing usable found at all) — soft
    issues (some missing images, a few bad rows) are collected into
    DatasetReport instead so --check-dataset can report them without
    crashing, per spec."""


@dataclasses.dataclass
class SplitLayout:
    name: str               # "train" | "val"
    csv_path: Path
    image_root: Path        # directory that `file` column paths are relative to


@dataclasses.dataclass
class DatasetLayout:
    root: Path
    train: SplitLayout
    val: SplitLayout


@dataclasses.dataclass
class SplitReport:
    name: str
    csv_path: str
    total_rows: int = 0
    usable_rows: int = 0
    missing_images: int = 0
    missing_image_samples: list = dataclasses.field(default_factory=list)
    empty_label_rows: int = 0
    missing_columns: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DatasetReport:
    layout: DatasetLayout
    train: SplitReport
    val: SplitReport
    age_classes: list
    gender_classes: list
    race_classes: list
    warnings: list = dataclasses.field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return (self.train.usable_rows > 0 and self.val.usable_rows > 0
                and not self.train.missing_columns and not self.val.missing_columns)

    def format_report(self) -> str:
        lines = [
            "FairFace Dataset Check",
            "=" * 22,
            "",
            f"Dataset: {self.layout.root}",
            "",
            f"Train images: {self.train.usable_rows} (of {self.train.total_rows} labeled rows)",
            f"Validation images: {self.val.usable_rows} (of {self.val.total_rows} labeled rows)",
            "",
            f"Age classes ({len(self.age_classes)}): {', '.join(self.age_classes)}",
            f"Gender classes ({len(self.gender_classes)}): {', '.join(self.gender_classes)}",
            f"Race classes ({len(self.race_classes)}): {', '.join(self.race_classes)}",
            "",
            f"Missing images: {self.train.missing_images + self.val.missing_images}",
            f"Empty/invalid label rows: {self.train.empty_label_rows + self.val.empty_label_rows}",
        ]
        for split in (self.train, self.val):
            if split.missing_columns:
                lines.append(f"  [{split.name}] MISSING REQUIRED COLUMNS: {split.missing_columns}")
            if split.missing_image_samples:
                sample = ", ".join(split.missing_image_samples[:5])
                lines.append(f"  [{split.name}] sample missing files: {sample}")
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        lines.append("")
        lines.append("Dataset is ready for training." if self.is_ready
                      else "Dataset is NOT ready for training (see issues above).")
        return "\n".join(lines)


def _candidate_csvs(root: Path, split: str) -> list[Path]:
    """Every plausible CSV location for `split` ("train"/"val"), most
    conventional first — see module docstring for the two named layouts."""
    candidates = [
        root / f"{split}_labels.csv",
        root / split / f"fairface_label_{split}.csv",
        root / f"fairface_label_{split}.csv",
        root / split / f"{split}_labels.csv",
    ]
    # Generic fallback: any csv anywhere up to two levels deep whose name
    # contains the split name, so a differently-named dataset still works.
    if root.is_dir():
        for csv_path in root.glob("*.csv"):
            candidates.append(csv_path)
        for csv_path in root.glob("*/*.csv"):
            candidates.append(csv_path)
    seen = set()
    out = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_file() and split in c.stem.lower():
            out.append(c)
    return out


def _resolve_image_root(csv_path: Path, root: Path, sample_file_value: str | None) -> Path:
    """The directory `file` column values are relative to: usually the
    dataset root itself (FairFace's own CSVs use paths like "train/1.jpg"),
    but falls back to the CSV's own directory for a layout where `file` is
    just a bare filename (e.g. "1.jpg")."""
    if sample_file_value and (root / sample_file_value).exists():
        return root
    if sample_file_value and (csv_path.parent / sample_file_value).exists():
        return csv_path.parent
    # Neither resolved yet (e.g. checking during a dry validation pass
    # before any file existence check) — default to root, matching the
    # official FairFace CSV convention ("train/1.jpg", "val/1.jpg").
    return root


def discover_dataset(root: str | Path) -> DatasetLayout:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(
            f"FairFace dataset was not found.\n\nExpected:\n{root}\n\n"
            "Please check that the dataset is inside the project directory."
        )

    splits = {}
    for split in ("train", "val"):
        candidates = _candidate_csvs(root, split)
        if not candidates:
            raise DatasetError(
                f"No {split} label CSV found under {root}.\n\n"
                f"Expected one of:\n"
                f"  {root / (split + '_labels.csv')}\n"
                f"  {root / split / ('fairface_label_' + split + '.csv')}\n\n"
                "FairFace's label CSVs (fairface_label_train.csv / fairface_label_val.csv) "
                "ship separately from the image archive on the official FairFace Google "
                "Drive (linked from https://github.com/joojs/fairface) — download them and "
                "place them under the dataset folder."
            )
        csv_path = candidates[0]
        with csv_path.open(newline="", encoding="utf-8") as f:
            first_row = next(csv.DictReader(f), None)
        sample_file = first_row.get("file") if first_row else None
        image_root = _resolve_image_root(csv_path, root, sample_file)
        splits[split] = SplitLayout(name=split, csv_path=csv_path, image_root=image_root)

    return DatasetLayout(root=root, train=splits["train"], val=splits["val"])


def _pick_class_order(values: set, canonical: list[str]) -> list[str]:
    if values == set(canonical):
        return list(canonical)
    return sorted(values)


def _validate_split(layout: SplitLayout, check_images: bool = True) -> tuple[SplitReport, list[dict]]:
    report = SplitReport(name=layout.name, csv_path=str(layout.csv_path))
    rows: list[dict] = []

    with layout.csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            report.missing_columns = missing_cols
            return report, rows

        for row in reader:
            report.total_rows += 1
            file_value = (row.get("file") or "").strip()
            age = (row.get("age") or "").strip()
            gender = (row.get("gender") or "").strip()
            race = (row.get("race") or "").strip()
            if not (file_value and age and gender and race):
                report.empty_label_rows += 1
                continue

            image_path = layout.image_root / file_value
            if check_images and not image_path.is_file():
                report.missing_images += 1
                if len(report.missing_image_samples) < 20:
                    report.missing_image_samples.append(file_value)
                continue

            rows.append({"path": str(image_path), "age": age, "gender": gender, "race": race})
            report.usable_rows += 1

    return report, rows


def validate_dataset(root: str | Path, check_images: bool = True) -> tuple[DatasetReport, list[dict], list[dict]]:
    """Full dataset validation — never raises for soft issues (missing
    images, empty labels); only discover_dataset()'s hard "nothing found at
    all" cases raise DatasetError. Returns (report, train_rows, val_rows)
    where *_rows are the usable, verified rows ready to feed FairFaceDataset."""
    layout = discover_dataset(root)
    train_report, train_rows = _validate_split(layout.train, check_images=check_images)
    val_report, val_rows = _validate_split(layout.val, check_images=check_images)

    warnings = []
    all_values = {"age": set(), "gender": set(), "race": set()}
    for rows in (train_rows, val_rows):
        for r in rows:
            all_values["age"].add(r["age"])
            all_values["gender"].add(r["gender"])
            all_values["race"].add(r["race"])

    age_classes = _pick_class_order(all_values["age"], AGE_CANONICAL)
    gender_classes = _pick_class_order(all_values["gender"], GENDER_CANONICAL)
    race_classes = _pick_class_order(all_values["race"], RACE_CANONICAL)
    for task, classes, canonical in (("age", age_classes, AGE_CANONICAL),
                                      ("gender", gender_classes, GENDER_CANONICAL),
                                      ("race", race_classes, RACE_CANONICAL)):
        if classes != canonical:
            warnings.append(
                f"{task} labels in the dataset don't match the expected FairFace set "
                f"({canonical}); using classes found in the data instead: {classes}"
            )

    report = DatasetReport(
        layout=layout, train=train_report, val=val_report,
        age_classes=age_classes, gender_classes=gender_classes, race_classes=race_classes,
        warnings=warnings,
    )
    return report, train_rows, val_rows


def build_label_mappings(age_classes: list, gender_classes: list, race_classes: list) -> dict:
    return {
        "age": {"classes": age_classes, "label_to_idx": {c: i for i, c in enumerate(age_classes)}},
        "gender": {"classes": gender_classes, "label_to_idx": {c: i for i, c in enumerate(gender_classes)}},
        "race": {"classes": race_classes, "label_to_idx": {c: i for i, c in enumerate(race_classes)}},
    }


class FairFaceDataset:
    """torch Dataset over already-validated rows (see validate_dataset) —
    kept dependency-light (torch imported lazily) so src/dataset.py's
    discovery/validation logic can be unit-tested / used by --check-dataset
    without requiring torch or torchvision to be importable.

    A row whose image fails to decode at runtime (corrupt file that
    happened to pass the existence check) is skipped by resampling a
    neighboring index rather than crashing the whole training run — real
    but rare on a 97k-image dataset, and one bad JPEG must never take down
    a multi-hour training job."""

    def __init__(self, rows: list[dict], label_mappings: dict, transform=None):
        self.rows = rows
        self.label_mappings = label_mappings
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image, UnidentifiedImageError
        import torch

        for attempt in range(5):
            i = (index + attempt) % len(self.rows)
            row = self.rows[i]
            try:
                image = Image.open(row["path"]).convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                logger.warning("Skipping unreadable image %s: %s", row["path"], exc)
                continue
            if self.transform is not None:
                image = self.transform(image)
            labels = {
                "age": torch.tensor(self.label_mappings["age"]["label_to_idx"][row["age"]], dtype=torch.long),
                "gender": torch.tensor(self.label_mappings["gender"]["label_to_idx"][row["gender"]], dtype=torch.long),
                "race": torch.tensor(self.label_mappings["race"]["label_to_idx"][row["race"]], dtype=torch.long),
            }
            return image, labels
        raise RuntimeError(f"5 consecutive unreadable images starting at index {index}")


def build_transforms(image_size: int, train: bool):
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    # Validation must never use random augmentation.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
