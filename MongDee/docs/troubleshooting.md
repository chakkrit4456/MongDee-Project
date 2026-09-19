# Troubleshooting

## Dashboard shows "disconnected" forever

1. Open the browser console — a CORS error means the AI Server's
   `configs/mongdee.json` needs `api.cors_origins` to include the
   dashboard's actual origin (see `docs/vercel-deployment.md`).
2. Click the **server ⚙** button and confirm the AI Server URL is right
   for where you actually deployed it — a stale value in that browser's
   `localStorage` overrides the build-time default until changed.
3. Confirm the AI Server itself is reachable: `curl https://your-ai-server/health`.
4. If an API key is configured, the dashboard needs `?api_key=...` in its
   own URL (`https://your-app.vercel.app/?api_key=...`) — it isn't stored
   anywhere else.

## A camera never shows up on the dashboard

- **Locally attached** (in the config's `"cameras"` list): check the AI
  Server's log for that camera's connect/reconnect messages —
  `camera.gateway.CameraGateway` logs status transitions. See also
  `MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md`'s notes on
  Windows DirectShow quirks, which `camera/usb.py` already accounts for.
- **Remote, via a Camera Agent**: confirm the agent's own log shows
  `"registered with AI server"` for that camera_id — if it's stuck
  retrying, the `server_url`/`api_key` in `configs/camera_agent.json` is
  probably wrong, or the AI Server isn't reachable from the agent's
  machine (check firewall/VPN — `docs/networking.md`). A registered
  camera that stops sending frames is marked `OFFLINE` after
  `backend/remote_frames.py`'s `STALE_AFTER_SEC` (5s) with no new frame.

## "this server was not started with remote camera support" (503)

`POST /api/cameras/register` and `.../frame` need the AI Server to have
been built with a `RemoteFrameReceiver` — this always happens when running
via `python -m backend.main` (see `backend/main.py`). This error only
happens if you're constructing `create_app()` yourself (e.g. in a test or
a custom script) without passing `remote_frames=`.

## Video is choppy / high latency

- Check `GET /metrics` for `pipeline.cameras.<id>.effective_fps` vs the
  camera's requested rate, and whether `hardware.cpu_percent` (via a
  fresh `/metrics` call — the first ever call in a process reads 0%) is
  pinned near 100%. If so, the Adaptive Performance Engine
  (`docs/performance.md`) should already be backing off automatically —
  confirm `adaptive.enabled: true` in your config.
- MJPEG-over-HTTP has more latency than WebRTC would at the same network
  quality — see `docs/networking.md` for why that trade-off was made and
  what upgrading would involve.
- A Camera Agent on a slow/high-latency link to the AI Server will drop
  frames (by design — Latest Frame Wins, see `docs/camera-agent.md`)
  rather than build up a backlog; a lower effective FPS there is expected
  behavior, not a bug, under a genuinely constrained network.

## `psutil` not installed

`core/performance.py` degrades gracefully (CPU/RAM read as `None`,
hardware tier defaults conservatively) rather than crashing, but the
Adaptive Performance Engine can't detect CPU overload without it. Install
it: `pip install psutil` (already in `requirements.txt`; the Docker image
installs it explicitly in `docker/Dockerfile`).

## Full C: drive (Windows) breaking things unrelated to MongDee

If SQLite writes, `pytest`, or even `git status` start failing with
"No space left on device" on a Windows machine with plenty of disk space
on other drives, check `%TEMP%` specifically — Python's default temp
directory (and `pytest`'s `tmp_path` fixture) resolve there, and a full
system drive breaks them even when a data drive has plenty of room. This
project's own `pytest.ini` already pins its test temp directory next to
the repo (`--basetemp=.pytest_tmp`) for exactly this reason; the same fix
(set `TEMP`/`TMP` to a folder on a drive with space, or free up the system
drive) applies to anything else on the machine using the OS default.
