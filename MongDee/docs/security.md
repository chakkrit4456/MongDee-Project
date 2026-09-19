# Security

## Authentication

The AI Server (`backend/api/app.py`) supports role-based API keys:
`viewer` < `operator` < `admin`. Configure via `configs/mongdee.json`'s
`api.keys` (`{"<key>": "role"}`) — `api.api_key` is a shortcut for a single
admin key. **Leaving both unset runs the API open** (logged as a warning
on startup); fine for a trusted LAN during development, not for anything
reachable beyond it.

Minimum roles for the new Camera Agent / dashboard endpoints:

| Endpoint | Minimum role |
|---|---|
| `POST /api/cameras/register` | operator |
| `POST /api/cameras/{id}/frame` | operator |
| `GET /api/cameras/{id}/stream` | viewer |
| Everything under `/api/*` except `/api/health` | viewer (read) or higher |
| `GET /api/audit` | admin |

A key is sent as `X-API-Key: <key>` for normal requests. Because a plain
`<img>`/`<video>` tag or a `WebSocket` connection can't set a custom
header, those specific routes (the live stream, and `/ws/*`) also accept
`?api_key=<key>` in the URL — treat such URLs as sensitive (don't paste
them somewhere with unrelated viewers) since a query string can end up in
server access logs or browser history.

## CORS

`configs/mongdee.json`'s `api.cors_origins` should list the exact
dashboard origin(s) you deploy (e.g. `["https://your-app.vercel.app"]`).
Leaving it unset allows every origin — logged as a warning, and something
to tighten before a public deployment. See `docs/vercel-deployment.md`.

## Secrets

Nothing in this repository should ever contain a real API key, camera
password, or database credential. The pattern used throughout
(`backend/config.py`, `camera_agent/config.py`) is `"${ENV_VAR}"` inside a
committed JSON config file, resolved from the process environment at load
time. See `.env.example` at the repo root for the full list of variables
this project reads, and `docker/docker-compose.yml` for how they reach a
containerized AI Server.

## Transport security

Neither the AI Server nor the Camera Agent's example config sets up TLS
itself — see `docs/networking.md` for putting a reverse proxy (HTTPS/WSS)
or a mesh VPN (Tailscale/WireGuard) in front of any connection that
crosses an untrusted network. A Vercel-hosted dashboard is HTTPS by
default (Vercel's own certificate); it's the AI Server's own exposure that
needs deliberate TLS termination if it's reachable outside a trusted LAN.

## Data handling

Person Re-ID here uses appearance/clothing-color embeddings, not facial
recognition, by default (`vision/reid/`) — see
`MongDee_Master_Prompt_Complete-1.md` sections 9-10 for what signals are
and aren't used. If a deployment adds face-based features, treat that data
as sensitive: apply a retention policy, and don't log raw embeddings or
frames alongside personally-identifying labels. This project does not
currently implement automatic data retention/expiry for the AI Server's
database — logged events accumulate until manually pruned; factor this
into any real deployment's privacy posture before going live.

## Rate limiting

The API applies bounded in-process sliding-window limits to the two remote
camera mutation endpoints: registration is limited per client IP, while
frame upload is limited per client IP and camera. The defaults are 30
registrations per minute and 120 frame uploads per second per camera, which
leaves headroom for normal camera rates while returning `429` during bursts.
These limits are process-local; keep a reverse proxy rate limit in front of
Internet-facing deployments as a second layer, especially when running
multiple API workers.

`create_app()` accepts `register_rate_limit` and `frame_rate_limit` for
deployment-specific tuning. Set either to `0` only for a controlled local
test, not for a public server.
