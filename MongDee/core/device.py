"""Compute-device resolution for YOLO/torch, shared by web_server.py and
app.py so both entrypoints pick the same accelerator the same way.

Order tried: CUDA (NVIDIA) -> DirectML (any DirectX12 GPU on Windows --
AMD, Intel, or an NVIDIA card without a CUDA-capable torch build -- via the
optional `torch-directml` package) -> CPU. install.bat's `--gpu` install
path auto-detects the GPU vendor and installs whichever package this needs
(CUDA torch for NVIDIA, base torch + torch-directml for anything else with
a GPU) -- see its "detect GPU vendor" step. Every step here is
independently wrapped in try/except so a partially set up GPU stack (driver
present but no matching package, or vice versa) degrades to the next option
instead of crashing the whole app.
"""

from __future__ import annotations


def resolve_device():
    """Best acceleration this process can actually use right now. Never
    raises -- always returns something valid to hand to ultralytics/torch's
    `device=` argument (an int CUDA index, a torch_directml device, or the
    string "cpu").

    `torch.cuda.is_available()` (and torch_directml's own is_available())
    only confirm the driver enumerated a device -- not that a context can
    actually be created on it right now. On a switchable-graphics/Optimus
    laptop (a discrete NVIDIA GPU alongside an Intel iGPU -- exactly the
    hardware docs/gpu-setup.md was written against) the dGPU can enumerate
    fine and then reject the very first real use with
    `torch.AcceleratorError: CUDA-capable device(s) is/are busy or
    unavailable` -- which used to crash the whole app at startup instead of
    just falling back to CPU. A tiny real allocation below is the only way
    to catch that before every other part of the app finds out the hard
    way."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")  # smoke test -- see docstring
            return 0
    except Exception:
        pass

    try:
        import torch
        import torch_directml

        if torch_directml.is_available():
            device = torch_directml.device()
            torch.zeros(1, device=device)  # same smoke test, same reasoning
            return device
    except Exception:
        pass

    return "cpu"


def device_backend(device) -> str:
    """Coarse backend name behind a resolve_device() result -- "cuda",
    "directml", or "cpu" -- for status APIs/telemetry that need to say which
    branch was picked, not just a human-readable label."""
    if device == "cpu" or device is None:
        return "cpu"
    if isinstance(device, int):
        return "cuda"
    try:
        import torch_directml  # noqa: F401  -- import only proves the module exists

        if "directml" in type(device).__module__.lower() or "privateuseone" in str(device).lower():
            return "directml"
    except Exception:
        pass
    return "cpu"


def device_label(device) -> str:
    """Human-readable name for startup logs -- so "GPU detected but not
    actually being used" (or the reverse) is never a silent guess."""
    backend = device_backend(device)
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        try:
            import torch

            return f"cuda:{device} ({torch.cuda.get_device_name(device)})"
        except Exception:
            return str(device)
    try:
        import torch_directml

        return f"directml ({torch_directml.device_name(0)})"
    except Exception:
        return str(device)
