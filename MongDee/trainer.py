"""Standalone AI Trainer — upload product photos/video and train recognition
without opening the full booth window (only needed if you use "ทดสอบด้วยกล้องสด").

Usage:
    python trainer.py
    python trainer.py --products products.json --gallery data/gallery
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from core.device import device_label, resolve_device
from core.products import ProductCatalog
from core.recognizer import DEFAULT_GALLERY_DIR, ProductRecognizer
from ui.trainer_window import TrainerWindow

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", default=str(ROOT / "products.json"))
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY_DIR))
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    args = parser.parse_args()

    from ultralytics import YOLO

    app = QApplication(sys.argv)

    catalog = ProductCatalog(Path(args.products))
    model_device = resolve_device()
    print(f"[TRAINER] ใช้งานอุปกรณ์ประมวลผล: {device_label(model_device)}")
    model = YOLO(args.model)
    recognizer = ProductRecognizer(gallery_dir=Path(args.gallery), device=model_device)

    window = TrainerWindow(catalog, recognizer, model, model_device)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
