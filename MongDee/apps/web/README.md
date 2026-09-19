# MongDee Dashboard — Vercel deployment

This directory has no source files of its own on purpose. The dashboard's
actual source lives in `backend/api/dashboard.html` and
`backend/api/designer.html` — the same pages `python -m backend.main`
serves directly for local/self-hosted use — so there is exactly one copy
to maintain instead of two that can silently drift apart.

`dist/` here is **generated**, never committed:

```bash
python scripts/build_web_dashboard.py apps/web/dist
```

That copies both HTML files into `dist/` and writes `dist/config.js` with
the AI Server URL viewers should default to (from the `MONGDEE_AI_SERVER_URL`
environment variable — see repository root `.env.example`).

Vercel runs exactly that command itself (see `vercel.json` at the
repository root) and serves `apps/web/dist/` as a static site. See
`docs/vercel-deployment.md` for the full walkthrough (env vars, CORS on
the AI Server side, custom domains).

To preview locally before deploying:

```bash
MONGDEE_AI_SERVER_URL=http://127.0.0.1:8100 python scripts/build_web_dashboard.py apps/web/dist
python -m http.server 5500 --directory apps/web/dist
# open http://127.0.0.1:5500/
```
