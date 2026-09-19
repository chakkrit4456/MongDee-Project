"""Unit tests for core/device.py's resolve_device() — spec section 42 (GPU
Fallback Tests): a CUDA device that *enumerates* but can't actually create a
context (the real incident this module exists to catch — see its docstring
and the torch.AcceleratorError "CUDA-capable device(s) is/are busy or
unavailable" traceback that motivated it) must fall back to CPU, never raise.

None of these need a physical NVIDIA/AMD/Intel GPU: CUDA is mocked via
monkeypatching torch.cuda, and DirectML (never installed in this dev/CI
environment — see GPU-Automatic-Acceleration-Audit notes) is mocked by
injecting a fake `torch_directml` module into sys.modules.
"""

from __future__ import annotations

import sys
import types

import torch

from core.device import device_backend, device_label, resolve_device


class _FakeDirectMLDevice:
    """Stand-in for whatever object type torch_directml.device() actually
    returns — only its __module__ (checked by device_backend/device_label)
    and str() need to look the part."""


_FakeDirectMLDevice.__module__ = "torch_directml"


def _install_fake_torch_directml(monkeypatch, *, available: bool, device_raises: bool = False,
                                  is_available_raises: bool = False):
    """Injects a fake torch_directml module for the duration of a test and
    removes it again afterward (monkeypatch.setitem handles the teardown)."""
    fake = types.ModuleType("torch_directml")

    def _is_available():
        if is_available_raises:
            raise RuntimeError("DirectML driver query failed")
        return available

    def _device():
        if device_raises:
            raise RuntimeError("DirectML device creation failed")
        return _FakeDirectMLDevice()

    fake.is_available = _is_available
    fake.device = _device
    fake.device_name = lambda i=0: "Fake DirectML GPU"
    monkeypatch.setitem(sys.modules, "torch_directml", fake)
    return fake


def test_cuda_unavailable_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == "cpu"


def test_cuda_enumerates_but_context_creation_fails_falls_back_to_cpu(monkeypatch):
    """Reproduces the real crash: torch.cuda.is_available() says yes, but the
    actual allocation raises (busy/unavailable device) — resolve_device()
    must catch that and return "cpu", not propagate the exception."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _raise_busy(*a, **kw):
        raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")

    monkeypatch.setattr(torch, "zeros", _raise_busy)
    assert resolve_device() == "cpu"


def test_cuda_available_and_working_returns_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: None)  # smoke test "succeeds"
    assert resolve_device() == 0


# --------------------------------------------------------- CUDA > DirectML > CPU priority

def test_cuda_available_wins_over_directml_even_if_installed(monkeypatch):
    """Case A: both CUDA and DirectML are usable — CUDA must be picked, never
    DirectML, since there's no reason to prefer it on an NVIDIA machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: None)
    _install_fake_torch_directml(monkeypatch, available=True)
    assert resolve_device() == 0


def test_cuda_unavailable_falls_back_to_directml_when_available(monkeypatch):
    """Case B: no CUDA, but DirectML is installed and usable — must pick
    DirectML, not drop straight to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: None)
    _install_fake_torch_directml(monkeypatch, available=True)
    device = resolve_device()
    assert isinstance(device, _FakeDirectMLDevice)


def test_neither_cuda_nor_directml_available_falls_back_to_cpu(monkeypatch):
    """Case C: CUDA unavailable, DirectML installed but reports unusable —
    must land on CPU, not raise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _install_fake_torch_directml(monkeypatch, available=False)
    assert resolve_device() == "cpu"


def test_directml_import_failure_falls_back_to_cpu(monkeypatch):
    """Case D: torch_directml isn't installed at all (import fails) — must
    land on CPU, not raise. Simulated by making the import raise instead of
    relying on the package genuinely being absent."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "torch_directml", None)  # forces ImportError on `import torch_directml`
    assert resolve_device() == "cpu"


def test_cuda_exception_then_directml_available_picks_directml(monkeypatch):
    """Case E: CUDA initialization throws — must still try DirectML next
    (and succeed here), not give up straight to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    calls = {"n": 0}

    def _zeros(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
        return None

    monkeypatch.setattr(torch, "zeros", _zeros)
    _install_fake_torch_directml(monkeypatch, available=True)
    device = resolve_device()
    assert isinstance(device, _FakeDirectMLDevice)


def test_cuda_exception_then_directml_also_fails_falls_back_to_cpu(monkeypatch):
    """Case E continued: CUDA throws, and DirectML's own device() throws too
    — must still land safely on CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "zeros", lambda *a, **kw: (_ for _ in ()).throw(
        RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")))
    _install_fake_torch_directml(monkeypatch, available=True, device_raises=True)
    assert resolve_device() == "cpu"


# --------------------------------------------------------------------- labels

def test_device_label_cpu():
    assert device_label("cpu") == "cpu"
    assert device_label(None) == "cpu"


def test_device_label_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Fake GPU")
    assert device_label(0) == "cuda:0 (Fake GPU)"


def test_device_label_directml(monkeypatch):
    _install_fake_torch_directml(monkeypatch, available=True)
    assert device_label(_FakeDirectMLDevice()) == "directml (Fake DirectML GPU)"


# --------------------------------------------------------------- device_backend

def test_device_backend_cpu():
    assert device_backend("cpu") == "cpu"
    assert device_backend(None) == "cpu"


def test_device_backend_cuda():
    assert device_backend(0) == "cuda"


def test_device_backend_directml(monkeypatch):
    _install_fake_torch_directml(monkeypatch, available=True)
    assert device_backend(_FakeDirectMLDevice()) == "directml"
