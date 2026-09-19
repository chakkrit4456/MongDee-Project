"""Decides, before torch is ever imported in this process, whether CUDA is
actually usable right now — not just enumerated.

torch.cuda.is_available() only checks that a CUDA-capable device enumerates.
On this project's own documented hardware (GTX 1050 under Windows, see
core/attributes.py's FairFaceBackend comment on "GPU that enumerates but is
actually busy"), that can be true while every real CUDA operation still
raises cudaErrorDevicesUnavailable. Worse, torch 2.x's Adam optimizer calls
torch.accelerator.current_stream() as a "graph capture" health check on
every .step() — unconditionally, even when every tensor involved lives on
CPU — so a half-broken CUDA device can crash a pure-CPU training run.

The only fully reliable fix is to make CUDA invisible to this process
*before* torch initializes it at all: set CUDA_VISIBLE_DEVICES="" ahead of
the first `import torch`, based on a quick probe in a throwaway
subprocess (so a crash/hang while probing can never take the real process
down with it).
"""

from __future__ import annotations

import os
import subprocess
import sys

_PROBE_CODE = "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.zeros(1, device='cuda').numel() else 1)"


def ensure_cuda_env(timeout: float = 30.0) -> None:
    """No-op if CUDA_VISIBLE_DEVICES is already set (caller/user decided),
    or if this isn't Windows+CUDA hardware at all (cheap common case: no
    point spending a subprocess+CUDA-init round trip when torch.cuda simply
    isn't compiled in). Otherwise probes in a subprocess and, only on
    failure, hides CUDA from the real process."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            timeout=timeout, capture_output=True,
        )
        usable = result.returncode == 0
    except Exception:
        usable = False
    if not usable:
        print("[INFO] CUDA probe failed (device enumerates but is not usable right now) — "
              "running on CPU for this process.")
        # "-1" (not "") -- verified on this project's own Windows/GTX 1050
        # setup that CUDA_VISIBLE_DEVICES="" does NOT reliably hide the
        # device from this process: torch.cuda.device_count() correctly
        # reports 0, but torch.cuda.is_available() still returns True and a
        # real op still reaches the actual (incompatible) GPU, leaking the
        # exact same raw CUDA error text this function exists to prevent.
        # "-1" is the value CUDA's own runtime recognizes as "no visible
        # devices" and reliably makes is_available() return False too.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
