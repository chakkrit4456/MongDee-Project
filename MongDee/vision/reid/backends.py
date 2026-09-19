"""Re-ID embedding backends.

Two are shipped:

  ColorHistogramBackend  pure numpy, always available. Region-split HSV
                         histogram — a genuine clothing/colour appearance
                         descriptor (the signal the master prompt leans on
                         in sections 10, 11, 38). Weak against lighting
                         changes between cameras; that's why Phase 6 never
                         decides identity on it alone.

  TorchvisionBackend     ImageNet-pretrained ResNet (torchvision) global
                         average-pool features, 512-d (resnet18/34) or
                         2048-d (resnet50). A generic appearance embedding,
                         NOT a purpose-trained person Re-ID model like
                         OSNet — accuracy will be lower, but it needs no
                         extra download beyond torchvision's cached
                         weights and runs on CPU. Swappable: implement
                         `ReIDBackend` with an OSNet/ONNX model and pass it
                         to ReIDExtractor.

Weights for TorchvisionBackend download once to ~/.cache/torch on first
use — run scripts/download_models.py ahead of an offline deployment.
"""

from __future__ import annotations

import numpy as np


class ReIDBackend:
    dim: int

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a 1-D float32 feature vector for a person crop (BGR).
        Normalisation is the extractor's job, not the backend's."""
        raise NotImplementedError


class ColorHistogramBackend(ReIDBackend):
    def __init__(self, h_bins: int = 8, s_bins: int = 4, stripes: int = 3):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.stripes = stripes
        self.dim = h_bins * s_bins * stripes

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h = crop_bgr.shape[0]
        feats = []
        for i in range(self.stripes):
            y0 = int(h * i / self.stripes)
            y1 = int(h * (i + 1) / self.stripes)
            band = hsv[y0:y1]
            hist = cv2.calcHist([band], [0, 1], None, [self.h_bins, self.s_bins], [0, 180, 0, 256])
            hist = hist.flatten().astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            feats.append(hist)
        return np.concatenate(feats)


class TorchvisionBackend(ReIDBackend):
    _DIMS = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}

    def __init__(self, model_name: str = "resnet18", device: str = "cpu", input_size: tuple[int, int] = (128, 256)):
        import torch
        import torchvision
        from torchvision import transforms

        if model_name not in self._DIMS:
            raise ValueError(f"TorchvisionBackend: unsupported model {model_name!r}")
        weights_enum = {
            "resnet18": torchvision.models.ResNet18_Weights,
            "resnet34": torchvision.models.ResNet34_Weights,
            "resnet50": torchvision.models.ResNet50_Weights,
        }[model_name]
        net = getattr(torchvision.models, model_name)(weights=weights_enum.DEFAULT)
        net.fc = torch.nn.Identity()  # keep the 512/2048-d avgpool output
        net.eval()

        self._torch = torch
        self.device = device
        self._net = net.to(device)
        self.dim = self._DIMS[model_name]
        self._tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size[1], input_size[0])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self._tf(rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self._net(tensor).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)


class OSNetBackend(ReIDBackend):
    """OSNet (vision/reid/osnet.py) — a purpose-trained person Re-ID
    architecture. This is the backend to use in production: unlike the
    ImageNet/colour fallbacks it actually discriminates *identity*.

    It needs pretrained weights, which are NOT bundled (licensing +
    they're hosted on Google Drive). Point `weights_path` at a downloaded
    osnet_*.pth (see scripts/download_models.py and vision/reid/README
    notes). Without weights it raises rather than silently running an
    untrained net.
    """

    def __init__(self, weights_path: str, variant: str = "osnet_x1_0", device: str = "cpu", input_size=(128, 256)):
        import os

        import torch
        from torchvision import transforms

        from vision.reid.osnet import osnet_x0_25, osnet_x1_0

        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"OSNetBackend: weights_path {weights_path!r} not found. Download an osnet_*.pth "
                f"(e.g. via scripts/download_models.py) and set reid.osnet_weights in your config."
            )
        net = {"osnet_x1_0": osnet_x1_0, "osnet_x0_25": osnet_x0_25}.get(variant, osnet_x1_0)()
        state = torch.load(weights_path, map_location="cpu")
        state = state.get("state_dict", state)
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            import logging

            logging.getLogger("mongdee.vision.reid").warning(
                "OSNet weights loaded with %d missing / %d unexpected keys — embedding quality may be degraded "
                "if the checkpoint architecture differs", len(missing), len(unexpected),
            )
        net.eval()
        self._torch = torch
        self.device = device
        self._net = net.to(device)
        self.dim = net.feature_dim
        self._tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((input_size[1], input_size[0])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self._tf(rgb).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            feat = self._net(tensor).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)


def build_backend(config) -> ReIDBackend:
    """config: ReIDConfig. Resolves 'auto' to torchvision if it imports and
    its weights are reachable, else the always-available colour backend.
    'osnet' requires config.osnet_weights to point at a real checkpoint."""
    from vision.detection.detector import resolve_device

    want = config.backend
    if want == "color":
        return ColorHistogramBackend()
    if want == "osnet":
        return OSNetBackend(
            config.osnet_weights, config.osnet_variant,
            resolve_device(config.device), (config.input_width, config.input_height),
        )
    if want in ("torchvision", "auto"):
        try:
            device = resolve_device(config.device)
            return TorchvisionBackend(
                config.torchvision_model, device, (config.input_width, config.input_height)
            )
        except Exception as exc:
            if want == "torchvision":
                raise
            import logging

            logging.getLogger("mongdee.vision.reid").warning(
                "torchvision Re-ID backend unavailable (%s); falling back to colour histogram", exc
            )
            return ColorHistogramBackend()
    raise ValueError(f"unknown reid backend {want!r}")
