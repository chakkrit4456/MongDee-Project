# GPU setup

The AI Server auto-detects what it's running on
(`core/performance.py:detect_hardware()`, printed at startup and available
at `GET /metrics`) and never assumes a specific GPU is present — CPU-only
is a fully supported mode, just slower.

There are two independent places GPU can help, and they're set up differently:

- **Video decode** (camera → display): automatic, vendor-agnostic, no setup — see
  below.
- **AI inference** (detection/re-id models): NVIDIA/CUDA-specific, needs a matching
  torch build — see "NVIDIA GPU" below.

## Camera decode acceleration (display) — automatic, any GPU vendor

`camera/rtsp.py` and `camera/hls.py` (both FFmpeg-backed) ask OpenCV to decode
compressed video (H.264/H.265) with `cv2.VIDEO_ACCELERATION_ANY`
(`camera/base.py:enable_hw_video_acceleration()`), which goes through the OS's own
hardware-decode API — D3D11VA/DXVA2 on Windows, VAAPI/VDPAU/CUDA on Linux — instead
of a vendor SDK. **NVIDIA, AMD, and Intel GPUs all work through this same path**, so
there is nothing to install or configure: it's on by default (`CameraConfig.hw_accel
= True`) and silently falls back to CPU software decode if the build/machine can't
do hardware decode. Set `"hw_accel": false` on a specific camera in
`configs/cameras.example.json`-style config if a particular camera/driver combination
misbehaves with it.

This is separate from, and does not require, the CUDA/torch setup below — a
machine with an AMD GPU and no NVIDIA driver still gets GPU-accelerated camera
decode, just CPU-only AI inference.

## CPU (default)

Nothing to configure. `docker/Dockerfile` installs the CPU torch wheels by
default and `detection.device: "auto"` (the config default) resolves to
`"cpu"` when no CUDA device is visible.

## NVIDIA GPU

1. **Host driver**: install the NVIDIA driver for your GPU and confirm
   `nvidia-smi` works on the host.
2. **Docker**: install the
   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
   then:
   - Rebuild `docker/Dockerfile` `FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04`
     instead of `python:3.12-slim`, and change the torch install line to
     the CUDA wheels (both are called out in comments right at the top of
     the Dockerfile).
   - Uncomment the `deploy: resources: reservations: devices:` block in
     `docker/docker-compose.yml`.
3. **Bare metal / no Docker**: install CUDA-enabled `torch`/`torchvision`
   matching your driver's CUDA version, per
   [pytorch.org](https://pytorch.org/get-started/locally/).
4. Set `"detection": {"device": "auto"}` (or an explicit `"cuda:0"`) in
   your config — `vision/detection/detector.py:resolve_device()` picks
   `cuda:0` automatically when `torch.cuda.is_available()`.

## Verifying it's actually being used

- Startup log line: `loading detector (... on cuda:0)` vs `... on cpu`.
- `GET /metrics` → `hardware.gpu_available` / `hardware.gpu_name` /
  `hardware.vram_total_gb`.
- Compare FPS via `--benchmark`-style measurement (see
  `docs/performance.md`) before/after — don't take "GPU is on" as proof
  it's faster for your specific model/resolution without measuring.

## Camera Agent needs none of this

`camera_agent/` never imports torch or ultralytics — it runs identically
on a GPU-less machine. See `docs/camera-agent.md`.
