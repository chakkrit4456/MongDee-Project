# Deploying the dashboard to Vercel

The dashboard's actual source is `backend/api/dashboard.html` and
`backend/api/designer.html` — the same pages `python -m backend.main`
serves directly. `apps/web/` has no source files of its own; it exists so
Vercel has something to build and serve. See `apps/web/README.md`.

## What Vercel is (and is not) responsible for

Per `MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md` section 2:
Vercel serves the static dashboard only. It never runs AI inference and
never proxies raw video — the dashboard's browser calls the AI Server
directly for both its JSON API and its MJPEG camera streams. Vercel is a
pure application/presentation layer here.

## One-time setup

1. **Push this repository to GitHub** (or GitLab/Bitbucket) if it isn't
   already, and import it as a new Vercel project.
2. In the Vercel project's **Settings → Environment Variables**, add:
   - `MONGDEE_AI_SERVER_URL` — the AI Server's public URL, e.g.
     `https://ai.your-domain.com:8100` (or a URL you've put behind a
     reverse proxy — see `docs/networking.md`). This becomes every
     viewer's *default*; nothing here is a secret.
3. Vercel reads `vercel.json` at the repo root automatically — no other
   configuration is needed. It runs:

   ```text
   buildCommand:     python3 scripts/build_web_dashboard.py apps/web/dist
   outputDirectory:  apps/web/dist
   ```

4. Deploy. Vercel's build environment includes Python 3 by default; if a
   given deployment target does not, set the build command to `python
   scripts/build_web_dashboard.py apps/web/dist` instead (same script,
   different interpreter name).

## On the AI Server side

The dashboard now calls the AI Server from a **different origin**
(`your-app.vercel.app` calling `ai.your-domain.com`), so the AI Server
needs CORS enabled for that origin. In `configs/mongdee.json`:

```json
"api": {
  "cors_origins": ["https://your-app.vercel.app"]
}
```

Omitting `cors_origins` allows every origin, which is fine to get started
locally but is logged as a warning and should not ship to production — see
`docs/security.md`.

## Verifying it worked

Open the deployed URL. The header's connection indicator should go from
"connecting…" to "live" within a few seconds once it can reach the AI
Server's `/api/*` and `/ws/dashboard`. If it can't:

- Check the browser console for a CORS error — confirms `cors_origins`
  needs the deployed Vercel URL added.
- Click the **server ⚙** button in the header to confirm/override which
  AI Server URL this browser is actually using (saved in that browser's
  `localStorage`, independent of the build-time default — useful for
  pointing one deployed dashboard at a different AI Server temporarily,
  e.g. while testing).
- See `docs/troubleshooting.md`.

## Previewing the exact build locally

```bash
MONGDEE_AI_SERVER_URL=http://127.0.0.1:8100 python scripts/build_web_dashboard.py apps/web/dist
python -m http.server 5500 --directory apps/web/dist
# open http://127.0.0.1:5500/
```

## What this deployment does *not* do (be honest about scope)

This documents how to build and deploy the static site itself; actually
running `vercel` / connecting a Vercel account and clicking deploy is a
step only you can do (it needs your Vercel login). Nothing in this repo
can complete that step on your behalf.
