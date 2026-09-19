"""Public-facing read-only proxy for the Dashboard page only.

`uvicorn app.main:app` (this machine, LAN-only by default, see `start.bat`)
serves the whole kiosk app: the live scanner, camera control, product
enrollment/management, and calibration. Exposing that whole app to the
internet would also expose camera control and product data changes. This
proxy sits in front of it and forwards only the handful of GET requests the
dashboard page actually needs — everything else (every other page, every
POST/DELETE) gets a 404 — so it is safe to tunnel to the public internet
(e.g. with `cloudflared tunnel --url http://127.0.0.1:8091`) while the real
app stays reachable only on the LAN.

Usage:
    python dashboard_proxy.py                        # proxies 127.0.0.1:8000, listens on :8091
    python dashboard_proxy.py --target-port 8000 --port 8091
"""

from __future__ import annotations

import argparse

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response

# Exact paths and path-prefix patterns the dashboard page needs. Every one
# of these is read-only on the origin server (GET only — see below), so
# this allowlist cannot be used to control the camera, change products, or
# delete data even though it shares the origin server's process.
ALLOWED_EXACT = {"/dashboard"}
ALLOWED_PREFIXES = ("/static/", "/api/analytics/")


def is_allowed(method: str, path: str) -> bool:
    if method != "GET":
        return False
    if path in ALLOWED_EXACT:
        return True
    return any(path.startswith(p) for p in ALLOWED_PREFIXES)


def create_app(target_base: str) -> FastAPI:
    app = FastAPI()
    client = httpx.AsyncClient(base_url=target_base, timeout=30.0)

    @app.get("/")
    async def root():
        return Response(status_code=307, headers={"Location": "/dashboard"})

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request):
        full_path = "/" + path
        if not is_allowed(request.method, full_path):
            return Response(status_code=404, content="not found")

        upstream = await client.get(full_path, params=request.query_params)
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        headers = {k: v for k, v in upstream.headers.items() if k.lower() not in excluded}
        return Response(content=upstream.content, status_code=upstream.status_code, headers=headers)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-host", default="127.0.0.1", help="Host the real app (app.main) listens on")
    parser.add_argument("--target-port", type=int, default=8000, help="Port the real app (app.main) listens on")
    parser.add_argument("--host", default="127.0.0.1", help="Host this proxy binds to")
    parser.add_argument("--port", type=int, default=8091, help="Port this proxy listens on")
    args = parser.parse_args()

    target_base = f"http://{args.target_host}:{args.target_port}"
    print(f"[dashboard_proxy] forwarding dashboard-only GET requests to {target_base}")
    print(f"[dashboard_proxy] listening on http://{args.host}:{args.port}/dashboard")
    uvicorn.run(create_app(target_base), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
