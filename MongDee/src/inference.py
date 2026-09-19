"""Loads a trained checkpoint (models/fairface/best_model.pt by default)
and runs age/gender/race prediction on a face crop — shared by
test_model.py and detect.py so both tools stay in exact agreement about
preprocessing and label decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import age_display, build_transforms
from src.model import build_model
from src.train_utils import load_checkpoint


class FairFacePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        from src.train_utils import resolve_device_safe

        self.device = resolve_device_safe(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        self.label_mappings = checkpoint["label_mappings"]
        config = checkpoint.get("config", {})
        self.image_size = config.get("image_size", 224)
        architecture = config.get("architecture", "resnet34")

        self.model = build_model(
            architecture=architecture,
            num_age=len(self.label_mappings["age"]["classes"]),
            num_gender=len(self.label_mappings["gender"]["classes"]),
            num_race=len(self.label_mappings["race"]["classes"]),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self._transform = build_transforms(self.image_size, train=False)

    def predict(self, face_crop_bgr: np.ndarray) -> dict:
        """face_crop_bgr: an already-cropped face (BGR, as OpenCV reads it).
        Returns {"age": {...}, "gender": {...}, "race": {...}}, each with
        label/display_label/confidence, plus the full probability
        distribution for that task."""
        import cv2
        from PIL import Image

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return self._unknown_result()

        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self._transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)

        result = {}
        for task in ("age", "gender", "race"):
            probs = F.softmax(outputs[task], dim=1).squeeze(0).cpu().numpy()
            idx = int(np.argmax(probs))
            classes = self.label_mappings[task]["classes"]
            label = classes[idx]
            result[task] = {
                "label": label,
                "display_label": age_display(label) if task == "age" else label,
                "confidence": float(probs[idx]),
                "probabilities": {c: float(p) for c, p in zip(classes, probs)},
            }
        return result

    @staticmethod
    def _unknown_result() -> dict:
        return {task: {"label": "unknown", "display_label": "unknown", "confidence": 0.0, "probabilities": {}}
                for task in ("age", "gender", "race")}

    # ---- GenderAgeBackend-compatible interface (drop-in for
    # vision/attributes/extractor.py's duck-typed hook / core/vision.py) ----
    def predict_gender(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        r = self.predict(person_crop_bgr)["gender"]
        return r["label"].lower(), r["confidence"]

    def predict_age_group(self, person_crop_bgr: np.ndarray) -> tuple[str, float]:
        r = self.predict(person_crop_bgr)["age"]
        return r["display_label"], r["confidence"]
