"""The pure (torch-free) parts of tools/finetune_fairface_gender.py and the booth-face dumper that feeds it."""
from __future__ import annotations

import csv
import importlib.util
import os

import cv2
import numpy as np

from core.attributes import FaceDumper, FairFaceBackend, NUM_RACE_CLASSES

_spec = importlib.util.spec_from_file_location(
    "finetune_tool", os.path.join(os.path.dirname(__file__), "..", "..", "tools", "finetune_fairface_gender.py"))
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _write_dataset(root, n=12):
    (root / "val").mkdir(parents=True)
    with open(root / "val" / "fairface_label_val.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "age", "gender", "race", "service_test"])
        for i in range(n):
            cv2.imwrite(str(root / "val" / f"{i}.jpg"), np.full((64, 64, 3), 100 + i, np.uint8))
            w.writerow([f"val/{i}.jpg", "20-29", "Male" if i % 2 == 0 else "Female",
                        "Southeast Asian" if i % 3 == 0 else "White", "False"])


def test_read_rows_marks_east_and_southeast_asian_faces(tmp_path):
    _write_dataset(tmp_path)
    rows = tool.read_rows(str(tmp_path), "val")
    assert len(rows) == 12
    assert {os.path.basename(r[0]) for r in rows if r[2]} == {"0.jpg", "3.jpg", "6.jpg", "9.jpg"}
    m, f = tool.balanced_subset(rows, 6, 0)
    assert len(m) == len(f) == 3 and all(r[1] == 0 for r in m) and all(r[1] == 1 for r in f)


def test_summarise_reports_female_and_male_recall_per_face_size():
    y = np.array([1, 1, 0, 0])
    diff = np.array([0.5, -0.2, -1.0, -0.4])          # the second woman is called male
    px = np.array([100.0, 20.0, 100.0, 20.0])
    s = tool.summarise(diff, y, px)
    assert s["all"]["female_recall"] == 0.5 and s["all"]["male_recall"] == 1.0
    assert s["<24px"]["female_recall"] == 0.0 and s[">=112px"] is None
    calibrated = tool.summarise(diff, y, px, lambda p: 0.5)
    assert calibrated["all"]["female_recall"] == 1.0


def test_booth_faces_are_read_by_folder_and_split_deterministically(tmp_path):
    for name in ("male", "female"):
        (tmp_path / name).mkdir()
        for i in range(10):
            cv2.imwrite(str(tmp_path / name / f"{i}.jpg"), np.zeros((30, 30, 3), np.uint8))
    rows = tool.read_extra(str(tmp_path))
    assert len(rows) == 20 and {r[1] for r in rows} == {0, 1} and all(r[2] and r[3] for r in rows)
    train, test = tool.split_extra(rows)
    assert len(test) == 4 and len(train) == 16 and not set(train) & set(test)
    assert tool.read_extra(str(tmp_path / "nope")) == []


def test_face_dumper_is_rate_limited_and_capped(tmp_path):
    d = FaceDumper(tmp_path, min_interval_sec=2.0, max_files=2)
    face = np.full((40, 40, 3), 120, np.uint8)
    assert d.maybe_save(face, "female", 0.93, 31.0, now=100.0).name.startswith("female_93_31px_")
    assert d.maybe_save(face, "male", 0.9, 31.0, now=100.5) is None              # too soon
    assert d.maybe_save(face, "male", 0.9, 31.0, now=103.0) is not None
    assert d.maybe_save(face, "male", 0.9, 31.0, now=106.0) is None              # cap reached
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_backend_dumps_the_faces_it_analyses_only_when_enabled(tmp_path):
    class _B(FairFaceBackend):
        def __init__(self, dump):
            self.calibration = None
            self.face_dump = dump

        def _locate_face(self, crop):
            return np.full((40, 40, 3), 120, np.uint8), 40.0

        def _forward(self, face_bgr):
            logits = np.zeros(18)
            logits[NUM_RACE_CLASSES:NUM_RACE_CLASSES + 2] = [0.0, 2.0]
            return logits

    crop = np.zeros((100, 60, 3), np.uint8)
    _B(None).predict_gender_detail(crop)
    assert not list(tmp_path.glob("*.jpg"))
    detail = _B(FaceDumper(tmp_path, min_interval_sec=0.0)).predict_gender_detail(crop)
    assert detail["gender"] == "female" and len(list(tmp_path.glob("female_*.jpg"))) == 1


def test_env_switch(monkeypatch, tmp_path):
    monkeypatch.delenv("MONGDEE_SAVE_FACES", raising=False)
    assert FaceDumper.from_env() is None
    monkeypatch.setenv("MONGDEE_SAVE_FACES", "1")
    monkeypatch.setenv("MONGDEE_SAVE_FACES_DIR", str(tmp_path))
    assert FaceDumper.from_env().directory == tmp_path
