"""Spec section 42 (GPU Fallback Tests), ProductRecognizer half: when moving
the backbone to the requested device raises (a GPU that enumerates but is
actually busy/unavailable — the real incident core/device.py's smoke test
and this fallback both exist to survive), __init__ must degrade to CPU and
finish constructing instead of propagating the exception and crashing
web_server.py's startup.

Uses the real MobileNetV3-Small backbone (weights are already cached
locally from prior runs — see torch.hub's checkpoint dir — so this doesn't
hit the network), only faking device placement failure via monkeypatch.
"""

from __future__ import annotations

import torch

from core.recognizer import ProductRecognizer


def test_gpu_move_failure_falls_back_to_cpu_without_crashing(tmp_path, monkeypatch):
    real_to = torch.nn.Module.to

    def _raise_busy_for_cuda(self, device, *a, **kw):
        if torch.device(device).type == "cuda":
            raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
        return real_to(self, device, *a, **kw)

    monkeypatch.setattr(torch.nn.Module, "to", _raise_busy_for_cuda)

    recognizer = ProductRecognizer(gallery_dir=tmp_path, device="cuda")
    assert recognizer.device == torch.device("cpu")


def test_cpu_device_construction_succeeds(tmp_path):
    recognizer = ProductRecognizer(gallery_dir=tmp_path, device="cpu")
    assert recognizer.device == torch.device("cpu")
    assert recognizer.has_any_gallery() is False
