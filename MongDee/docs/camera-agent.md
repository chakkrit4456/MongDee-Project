# Camera Agent (`camera_agent/`)

A lightweight process that captures webcams (or RTSP/ONVIF/HTTP/HLS
sources) on one machine and streams them over HTTP to an AI Server running
somewhere else — see `docs/architecture.md`. It deliberately does **not**
depend on torch, ultralytics, or anything in `vision/` — "Camera Client =
Lightweight, AI Server = Heavy"
(`MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md` section 58).

## Installing

```bash
pip install opencv-python numpy requests
```

(All three are already in the repo's `requirements.txt` if you're running
from a full checkout — this is just the minimum for a camera-only machine
that will never run AI locally.)

## Configuring and running

```bash
cp configs/camera_agent.example.json configs/camera_agent.json   # edit as needed
python -m camera_agent.main configs/camera_agent.json
```

`configs/camera_agent.example.json` shows the shape: a `server_url`
pointing at the AI Server, an `api_key` (needs at least the `operator`
role there — see `docs/security.md`), and a `"cameras"` list using the
exact same format as `configs/cameras.example.json` (USB by device index,
or RTSP/ONVIF/HTTP/HLS by URL). Credentials can be written as
`"${ENV_VAR}"` and are resolved from the environment at load time.

## How it behaves under load or network trouble

- **Latest Frame Wins, end to end.** `camera.gateway.CameraGateway`
  already keeps only the latest captured frame per camera; this agent
  applies the same rule to the network hop — if the previous HTTP push
  hasn't finished, a newly captured frame simply replaces it. A slow or
  temporarily unreachable AI Server degrades the effective frame rate; it
  never queues frames or grows memory.
- **A camera that can't open never blocks another.** Each camera has its
  own capture thread (via `CameraGateway`) and its own sender thread; one
  camera's USB failure or a specific push failing doesn't affect the
  others.
- **Registration retries with backoff**, and re-registers automatically if
  a push starts failing (covers the AI Server restarting and losing its
  in-memory camera list).
- **Never crashes the process** on a camera or network failure — verified
  by `tests/camera_agent/test_client.py`'s
  `test_unreachable_camera_never_crashes_the_agent`.

Retry timing starts at `register_retry_sec`, multiplies by
`retry_backoff_multiplier`, and is capped at `retry_backoff_max_sec`. A
a successful registration resets the delay. A failed push also enters this
backoff before registration is attempted again.

## What it sends

Each push is `POST {server_url}/api/cameras/{camera_id}/frame`, a
multipart JPEG plus form fields `timestamp` (the capture time) and
`sequence` (a monotonically increasing per-camera counter, so the AI
Server can drop a reordered/stale delivery instead of applying it —
see `backend/remote_frames.py`).

## Limits

- Sends MJPEG-over-HTTP, not H.264/WebRTC/RTSP — see
  `docs/networking.md` for why, and what a future upgrade path looks like.
- No TLS is configured by this project's example config; a production
  deployment across an untrusted network should put a reverse proxy (with
  HTTPS) or a VPN (Tailscale/WireGuard) in front of the AI Server rather
  than exposing plain HTTP — see `docs/security.md`.
