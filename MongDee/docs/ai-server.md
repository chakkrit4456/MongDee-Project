# AI Server (`backend/`)

The AI Server runs the real computer-vision pipeline: person detection,
tracking, Re-ID, global identity, optionally product detection and spatial/
booth analytics — see `MongDee_Master_Prompt_Complete-1.md` for the full
pipeline spec and `docs/architecture.md` for how the Cloud/Remote work fits
around it.

## Running it

```bash
pip install -r requirements.txt
cp configs/mongdee.example.json configs/mongdee.json   # edit as needed
python -m backend.main configs/mongdee.json
```

It prints a hardware summary line (CPU/RAM/GPU, via `core/performance.py`)
on startup, then serves the API + dashboard at `http://<api.host>:<api.port>/`
(default `127.0.0.1:8100`).

Docker: `docker/Dockerfile` + `docker/docker-compose.yml` build a CPU image
that runs the same `python -m backend.main` entrypoint. For a GPU host, see
`docs/gpu-setup.md`.

## Configuration

One JSON file (`backend/config.py`'s `MongDeeConfig`) holds everything —
see `configs/mongdee.example.json` for a fully-annotated example covering
cameras, detection, tracking, Re-ID, identity, spatial/booth, products,
database, the API, and the Adaptive Performance Engine. Secrets can be
written as `"${ENV_VAR}"` in the JSON and are resolved from the process
environment at load time — never commit a real password or API key.

## Cameras: local and remote at once

A camera can arrive two ways, and both work simultaneously on the same AI
Server instance:

- **Locally attached**: listed in the config's `"cameras"` array, captured
  directly by `camera.gateway.CameraGateway` (USB/RTSP/ONVIF/HTTP/HLS).
- **Remote, via a Camera Agent**: not listed in the config at all — a
  Camera Agent (see `docs/camera-agent.md`) registers it at runtime by
  calling `POST /api/cameras/register`, then keeps pushing frames to
  `POST /api/cameras/{id}/frame`.

Both feed the exact same `DetectionPipeline` through
`backend.remote_frames.CompositeFrameSource` — the pipeline has no idea
which kind of camera it's looking at, and never needs to.

## API surface

Full route list is in `backend/api/app.py`; the additions for this Cloud/
Remote work are:

| Method & path | Role required | Purpose |
|---|---|---|
| `POST /api/cameras/register` | operator | Camera Agent registers a camera_id |
| `POST /api/cameras/{id}/frame` | operator | Camera Agent pushes one JPEG frame |
| `GET /api/cameras/{id}/stream` | viewer | Live MJPEG feed (works for local or remote cameras) |
| `GET /health` | none | Bare-path liveness check (in addition to `/api/health`) |
| `GET /metrics` | none | Hardware profile, remote camera health, live pipeline FPS/latency |
| `GET /config.js` | none | Dashboard's own AI-server-URL config (empty/same-origin by default) |

Authentication is the existing role-based API-key system
(`api.keys` / `api.api_key` in config) — `viewer` < `operator` < `admin`.
A Camera Agent's key needs at least `operator`; the dashboard's read-only
views work with `viewer`. See `docs/security.md`.

## Adaptive Performance Engine

`backend/adaptive.py`'s `PipelineAdaptiveController` ticks every
`adaptive.tick_interval_sec` (config), reads real CPU% (`core/performance.py`)
and the pipeline's own measured per-camera latency, and — only when
actually overloaded — reduces `detection.target_fps_per_camera` first,
then the YOLO input resolution (`imgsz_levels`, high to low). It restores
in the opposite order once load is back to normal. It never touches
display/stream FPS. See `docs/performance.md` for the full priority
rationale and how to tune the thresholds.

## Known limits (see also `docs/troubleshooting.md`)

- One shared detector/tracker per process — CPU-bound deployments should
  expect per-camera FPS to divide roughly by camera count, same as the
  local/self-hosted stack.
- No horizontal scaling (multiple AI Server instances behind a load
  balancer) is implemented — `MongDee_Cloud_Vercel_Remote_AI_Server_Master_
  Prompt.md` section 80 explicitly says this is a "design must not block
  it" requirement, not a "build it now" one; the `FrameSource` protocol
  seam this work introduced is exactly what a future sharded deployment
  would build on, but no such sharding exists yet.
