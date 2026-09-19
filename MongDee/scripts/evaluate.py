"""Run the evaluation metrics (MongDee_Master_Prompt.md section 37)
against a labelled annotation file from real footage.

    python scripts/evaluate.py annotations.json

Annotation file format (JSON):
{
  "detection": {
     "frames": [
        {"pred": [[x1,y1,x2,y2], ...], "gt": [[x1,y1,x2,y2], ...]},
        ...
     ]
  },
  "tracking": {
     "frames": [
        {"pred": [[track_id, [x1,y1,x2,y2]], ...], "gt": [[track_id, [x1,y1,x2,y2]], ...]},
        ...
     ]
  },
  "counting": {"predicted_unique": 137, "true_unique": 140},
  "matching": {
     "predicted": {"CAM01:15": "PERSON-0001", ...},
     "ground_truth": {"CAM01:15": "person_A", ...}
  }
}

Every section is optional — include only what you have labels for.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.evaluation import (
    evaluate_counting,
    evaluate_detection,
    evaluate_matching,
    evaluate_tracking,
)


def _parse_key(k: str) -> tuple[str, int]:
    cam, tid = k.rsplit(":", 1)
    return (cam, int(tid))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations", help="path to the annotation JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    report: dict = {}

    if "detection" in data:
        frames = data["detection"]["frames"]
        m = evaluate_detection([f["pred"] for f in frames], [f["gt"] for f in frames])
        report["detection"] = dataclasses.asdict(m)

    if "tracking" in data:
        frames = data["tracking"]["frames"]
        pred = [[(t[0], t[1]) for t in f["pred"]] for f in frames]
        gt = [[(t[0], t[1]) for t in f["gt"]] for f in frames]
        report["tracking"] = dataclasses.asdict(evaluate_tracking(pred, gt))

    if "counting" in data:
        c = data["counting"]
        report["counting"] = dataclasses.asdict(evaluate_counting(c["predicted_unique"], c["true_unique"]))

    if "matching" in data:
        pred = {_parse_key(k): v for k, v in data["matching"]["predicted"].items()}
        gt = {_parse_key(k): v for k, v in data["matching"]["ground_truth"].items()}
        report["matching"] = dataclasses.asdict(evaluate_matching(pred, gt))

    print(json.dumps(report, indent=2))
    print("\nNote: these are measured on YOUR test set. Do not report them as the system's universal accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
