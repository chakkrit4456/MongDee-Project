# Local development (Cloud/Remote AI Server stack)

For the local/self-hosted stack instead (`web_server.py`/`app.py`, one
machine does everything), see the repo's `GUIDE.md` — this document is
specifically about running the AI Server + Camera Agent + dashboard as
separate processes, as they'd run in a real Cloud deployment.

There's no separate database service to start — `backend/database/` is
SQLite, a single file created automatically on first run.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure

```bash
cp configs/mongdee.example.json configs/mongdee.json
cp configs/camera_agent.example.json configs/camera_agent.json
```

Edit `configs/mongdee.json`: at minimum, set `"cameras": []` if you're
testing with a Camera Agent only (no locally-attached camera on this
machine), and set an `api.api_key` (or `api.keys`) so the Camera Agent has
something to authenticate with.

Edit `configs/camera_agent.json`: point `server_url` at wherever you'll
run the AI Server (`http://127.0.0.1:8100` if it's the same machine),
matching `api_key`, and list your actual camera(s).

## 3. Start the AI Server

```bash
python -m backend.main configs/mongdee.json
```

Confirm it's up: `curl http://127.0.0.1:8100/health`.

## 4. Start the Camera Agent

In a second terminal (can be the same machine, or a different one on the
same network):

```bash
python -m camera_agent.main configs/camera_agent.json
```

Its log should show `registered with AI server` for each configured
camera. Confirm frames are arriving: `curl http://127.0.0.1:8100/metrics`
should show the camera under `remote_cameras` with `frames_received`
climbing.

## 5. View the dashboard

Open `http://127.0.0.1:8100/` — this is the AI Server serving its own
copy of `backend/api/dashboard.html` directly (same-origin, no CORS/
`config.js` complexity needed for local dev). You should see the
camera's live feed and, once someone walks in front of it, person counts
and events.

## 6. (Optional) Preview the Vercel build locally

```bash
MONGDEE_AI_SERVER_URL=http://127.0.0.1:8100 python scripts/build_web_dashboard.py apps/web/dist
python -m http.server 5500 --directory apps/web/dist
```

Open `http://127.0.0.1:5500/` — same dashboard, but now served from a
different origin than the AI Server, exercising the actual CORS +
`config.js` path a real Vercel deployment uses. If `configs/mongdee.json`
doesn't list `http://127.0.0.1:5500` in `api.cors_origins`, either add it
or leave `cors_origins` unset (open CORS, fine for this local check).

## Running the test suite

```bash
pytest                 # fast tests (excludes @pytest.mark.slow)
pytest -m slow         # real-YOLO end-to-end tests, including the
                        # Camera Agent -> AI Server -> real pipeline path
                        # (tests/backend/test_remote_camera_e2e.py)
```
