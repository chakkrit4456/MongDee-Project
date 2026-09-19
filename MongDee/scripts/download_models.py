"""Pre-fetch model weights so the vision system can start offline
(MongDee_Master_Prompt.md section 43: "model installation").

    python scripts/download_models.py                 # YOLO + torchvision resnet18
    python scripts/download_models.py --torchvision resnet50
    python scripts/download_models.py --osnet-url <URL> --osnet-out models/osnet_x1_0.pth

YOLO weights (yolo11n.pt) are already committed in the repo; this just
verifies ultralytics can load them. torchvision weights cache to
~/.cache/torch. OSNet weights are not hosted anywhere stable — pass a URL
you trust (e.g. from the deep-person-reid model zoo) to fetch one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo", default="yolo11n.pt", help="YOLO weights path (default: yolo11n.pt)")
    parser.add_argument("--torchvision", default="resnet18", choices=["resnet18", "resnet34", "resnet50", "none"])
    parser.add_argument("--osnet-url", help="URL to download an osnet_*.pth from")
    parser.add_argument("--osnet-out", default="models/osnet_x1_0.pth")
    args = parser.parse_args()

    print(f"[1] YOLO: loading {args.yolo} ...")
    from ultralytics import YOLO

    YOLO(args.yolo)
    print(f"    ok")

    if args.torchvision != "none":
        print(f"[2] torchvision: fetching {args.torchvision} ImageNet weights ...")
        import torchvision

        weights = {
            "resnet18": torchvision.models.ResNet18_Weights,
            "resnet34": torchvision.models.ResNet34_Weights,
            "resnet50": torchvision.models.ResNet50_Weights,
        }[args.torchvision]
        getattr(torchvision.models, args.torchvision)(weights=weights.DEFAULT)
        print("    ok (cached to ~/.cache/torch)")

    if args.osnet_url:
        import urllib.request

        out = Path(args.osnet_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[3] OSNet: downloading {args.osnet_url} -> {out} ...")
        urllib.request.urlretrieve(args.osnet_url, out)
        size_mb = out.stat().st_size / 1e6
        print(f"    ok ({size_mb:.1f} MB). Set reid.backend='osnet' and reid.osnet_weights='{out}' in your config.")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
