# Networking

## Ports

| Service | Default port | Notes |
|---|---|---|
| AI Server (`backend/main.py`) | 8100 | `api.host`/`api.port` in config |
| Vercel dashboard | n/a | Vercel's own HTTPS, no port to manage |
| Camera Agent | none listening | it only makes outbound calls to the AI Server |

## Protocol choice: MJPEG-over-HTTP, not WebRTC/RTSP

`MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md` section 7 asks to
choose a streaming protocol on its merits (latency, reliability, browser
compatibility, bandwidth, multi-camera scalability, server complexity,
deployment environment) rather than by popularity, and to justify the
choice rather than default to WebRTC. For this project, that evaluation
favors plain HTTP + MJPEG for v1:

- **Reliability & simplicity**: `requests.post()` with a timeout and a
  simple retry/backoff loop (`camera_agent/client.py`) is trivial to
  reason about, test without mocking a signaling server, and debug with
  ordinary HTTP tooling (curl, browser devtools). WebRTC needs a
  signaling channel, ICE/STUN/TURN for NAT traversal, and a much larger
  implementation surface.
- **Browser compatibility**: a plain `<img src="...">` pointed at a
  `multipart/x-mixed-replace` MJPEG endpoint works in every browser with
  zero JS beyond setting `src` once — this is exactly what the dashboard
  already does for the local/self-hosted stack (`web/server.py`'s
  `/stream/{camera_id}`), so the Cloud/Remote AI Server's
  `/api/cameras/{id}/stream` reuses the same proven approach
  (`backend/api/app.py`).
- **Deployment environment**: the primary target is a LAN-attached booth
  with a handful of cameras, not hundreds of viewers on the public
  internet — WebRTC's main advantages (sub-second latency at scale, P2P
  bandwidth savings) matter most exactly where this project isn't
  operating yet.
- **Trade-off, stated honestly**: MJPEG-over-HTTP has higher latency and
  bandwidth per stream than H.264/WebRTC would. If a future deployment
  needs many concurrent viewers or sub-200ms glass-to-glass latency,
  WebRTC (or at minimum switching the codec to H.264 with a small
  WebSocket-based player) is the documented upgrade path — nothing in the
  `FrameSource`/`CompositeFrameSource` design couples the AI pipeline to
  MJPEG, so that change would only touch the streaming route, not the
  detection/tracking/Re-ID pipeline.

Frame delivery from Camera Agent to AI Server is a normal HTTP POST per
frame (JPEG body), not a persistent stream — this keeps the agent's
network code trivial (no long-lived connection to manage/reconnect) at
the cost of per-request HTTP overhead, which is small relative to a JPEG
frame's own size at the frame rates this project targets (a handful of
FPS per camera, not high-speed video).

## Exposing the AI Server safely

Don't expose the AI Server's port directly to the internet. For a Camera
Agent or dashboard connecting from outside your LAN:

- Put a reverse proxy (nginx, Caddy, Traefik) in front of it with TLS, so
  traffic is HTTPS/WSS rather than plain HTTP/WS.
- Or use a mesh VPN (Tailscale, WireGuard) between the AI Server, Camera
  Agent machines, and anyone administering the system — the Camera Agent
  and dashboard then just point at the VPN-internal address.
- Either way, `configs/mongdee.json`'s `api.cors_origins` should list only
  the dashboard origin(s) you actually deploy, not `["*"]`. See
  `docs/security.md`.

## Firewall

Only the AI Server's port needs an inbound rule (from wherever Camera
Agents and dashboard viewers connect from). The Camera Agent needs no
inbound rule at all — it only makes outbound HTTP calls.
