"""Unit tests for web/booth_manager.py's get_gpu_status() — the read-only
status the Booth Settings page's GPU panel displays (see
web/static/settings.js's loadGpuStatus()) now that the manual GPU
install/update controls have been removed in favor of automatic
acceleration selection at startup (core/device.py's resolve_device()).

Only self.model_device is touched, so a minimal bare instance (same
`object.__new__(BoothManager)` pattern as test_booth_manager_attributes.py)
is enough — no cameras/db/YOLO model needed.
"""

from __future__ import annotations

import torch

from web.booth_manager import BoothManager


def _bare_booth_manager(model_device):
    bm = object.__new__(BoothManager)
    bm.model_device = model_device
    return bm


def test_gpu_status_reports_cuda_in_use(monkeypatch):
    # Mocked rather than relying on real hardware — see core/device.py's own
    # tests for why (CI/dev boxes without an NVIDIA GPU must still pass).
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    bm = _bare_booth_manager(0)
    status = bm.get_gpu_status()
    assert status["using_gpu"] is True
    assert status["available"] is True
    assert status["backend"] == "cuda"
    assert status["device"] == "cuda:0 (Fake GPU)"


def test_gpu_status_reports_cpu_not_in_use():
    bm = _bare_booth_manager("cpu")
    status = bm.get_gpu_status()
    assert status["using_gpu"] is False
    assert status["available"] is False
    assert status["backend"] == "cpu"
    assert status["device"] == "cpu"


def test_gpu_status_reports_cpu_when_model_device_is_none():
    bm = _bare_booth_manager(None)
    status = bm.get_gpu_status()
    assert status["using_gpu"] is False
    assert status["available"] is False
    assert status["backend"] == "cpu"
    assert status["device"] == "cpu"
