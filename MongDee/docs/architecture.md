# Architecture

MongDee has two independent ways to run today, and they share almost all
of their underlying code:

1. **Local / self-hosted** (`web_server.py`, `app.py`) — one machine does
   everything: capture, AI, database, and serves either a browser UI
   (`web/`) or a PySide6 desktop UI (`ui/`). This is what `GUIDE.md`
   documents and is unaffected by everything below.
2. **Cloud / Remote AI Server** (`backend/`, `camera_agent/`, `apps/web/`)
   — the architecture this document describes, built to
   `MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md`. A lightweight
   Camera Agent captures webcams and pushes frames over HTTP to a
   separately-hosted AI Server, which runs the actual computer vision
   pipeline; a dashboard (deployable to Vercel as a static site, or served
   directly by the AI Server for LAN use) shows the results.

```text
                    INTERNET / LAN
                         │
                         ▼
              ┌─────────────────────┐
              │   apps/web (Vercel) │   static HTML/JS dashboard,
              │   or backend's own  │   no build step, no framework
              │   self-hosted copy  │
              └──────────┬──────────┘
                         │ HTTPS + WSS (REST + WebSocket)
                         ▼
              ┌─────────────────────┐
              │   backend/ (FastAPI)│   AI Server
              │                     │
              │  RemoteFrameReceiver│ ← Camera Agent frames arrive here
              │  CameraGateway      │ ← locally-attached cameras (if any)
              │  CompositeFrameSource
              │         │           │
              │  DetectionPipeline  │   detection → tracking → Re-ID →
              │  (vision/)          │   global identity → analytics
              │         │           │
              │  DatabaseSink       │   → backend/database (SQLite)
              │  PipelineAdaptiveController (backend/adaptive.py)
              └──────────▲──────────┘
                         │ HTTP: register + push JPEG frames
              ┌──────────┴──────────┐
              │  camera_agent/      │   lightweight, no torch/ultralytics
              │  (uses camera/      │
              │   package directly) │
              └──────────┬──────────┘
                ┌────────┼────────┐
                ▼        ▼        ▼
              CAM 1    CAM 2    CAM 3 ...
```

## Why this split, not a rewrite

`backend/` + `vision/` + `camera/` already existed before this Cloud/
Vercel work started — they implement the full person/product/Re-ID/
spatial pipeline against `MongDee_Master_Prompt_Complete-1.md`, running
entirely on one machine with locally-attached cameras
(`camera.gateway.CameraGateway`). That pipeline consumes frames through a
small `FrameSource` protocol (`camera_ids()` + `latest_frame()`, see
`vision/pipeline.py`) — `CameraGateway` was already the only implementation
of it.

The actual gap for "Remote AI Server" was purely the *transport*: nothing
let a camera on one machine feed a pipeline running on another. That is
what this work added:

- `backend/remote_frames.py` — `RemoteFrameReceiver`, a second
  `FrameSource` implementation fed by HTTP instead of local capture, plus
  `CompositeFrameSource` to merge it with `CameraGateway` so one AI Server
  can serve local and remote cameras at once.
- `camera_agent/` — a new, deliberately dependency-light client that
  reuses `camera.gateway.CameraGateway` for capture (the exact same
  reconnect/backoff/DSHOW-on-Windows logic already used elsewhere in this
  project) and pushes frames to the AI Server's new endpoints.
- New endpoints on the existing `backend/api/app.py` FastAPI app:
  `POST /api/cameras/register`, `POST /api/cameras/{id}/frame`,
  `GET /api/cameras/{id}/stream`, `GET /health`, `GET /metrics`, plus CORS
  support and a query-param API-key fallback for `<img>` tags.
- `backend/adaptive.py` — an Adaptive Performance Engine for this
  pipeline's actual knobs (shared detector's target FPS and YOLO input
  size), following the same priority order (AI rate before AI resolution,
  display never touched) as the Multi-Webcam Performance Engine built for
  the local desktop/web app (`core/performance.py`).
- The dashboard (`backend/api/dashboard.html`, `designer.html`) gained a
  configurable AI Server base URL (`config.js` + a per-browser "server"
  settings dialog) and CORS so it can be hosted on a different origin —
  which is what makes deploying it to Vercel meaningful.

No existing feature was removed. The local/self-hosted stack (`web/`,
`ui/`, `core/vision.py`) is untouched by any of this.

## Where each piece lives

| Concern | Local/self-hosted | Cloud/Remote |
|---|---|---|
| Capture | `core/vision.py` (`CameraWorker`) | `camera/` package (`CameraGateway`) via `camera_agent/` |
| AI pipeline | `core/vision.py` + `core/recognizer.py` | `vision/` package (`DetectionPipeline`) |
| API/UI | `web/` (Jinja2 + FastAPI) or `ui/` (PySide6) | `backend/api/app.py` (FastAPI) + `apps/web/` (static, Vercel) |
| Database | `core/database.py` (SQLite) | `backend/database/` (separate SQLite schema) |
| Performance | `core/performance.py` | `core/performance.py` (hardware detection reused) + `backend/adaptive.py` |

See also: `docs/ai-server.md`, `docs/camera-agent.md`,
`docs/vercel-deployment.md`, `docs/networking.md`, `docs/security.md`,
`docs/performance.md`, `docs/troubleshooting.md`,
`docs/local-development.md`.
