"""Smoke-test driver for MONGDEE MINI KIOSK.

Drives the running FastAPI app the way a browser / the kiosk frontend does:
hits every page route and the JSON API, prints a pass/fail table, exits
non-zero if anything is wrong.

Usage (from repo root, with the server already running on :8000):

    backend/venv/Scripts/python.exe .claude/skills/run-mongdee-kiosk/smoke.py

Options:
    --base URL     base URL of a running server (default http://127.0.0.1:8000)
    --launch       start the server itself (backend/venv uvicorn), wait for it,
                   run the checks, then leave it running in the foreground
                   (Ctrl+C to stop). Without this, the server must already run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND = REPO_ROOT / "backend"

PAGE_ROUTES = ["/", "/products", "/enroll", "/manage", "/dashboard", "/calibrate"]
API_ROUTES = [
    "/api/camera/status",
    "/api/camera/devices",
    "/api/products",
    "/api/recognition/current",
    "/api/analytics/summary",
    "/api/analytics/hourly",
]


def get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:  # noqa: F821
        return e.code, e.read()


def wait_for(base: str, tries: int = 40) -> bool:
    for _ in range(tries):
        try:
            code, _ = get(base + "/api/camera/status", timeout=2)
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_checks(base: str) -> int:
    failures = 0
    for route in PAGE_ROUTES + API_ROUTES:
        code, body = get(base + route)
        ok = code == 200
        failures += not ok
        note = ""
        if route == "/api/camera/status" and ok:
            note = body.decode("utf-8", "replace")[:120]
        print(f"  {'OK ' if ok else 'FAIL'}  {route:<28} {code}  {note}")
    print()
    if failures:
        print(f"{failures} check(s) FAILED")
    else:
        print("all checks passed")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    proc = None
    if args.launch:
        py = BACKEND / "venv" / "Scripts" / "python.exe"
        if not py.exists():
            py = BACKEND / "venv" / "bin" / "python"
        print(f"launching: {py} -m uvicorn app.main:app")
        proc = subprocess.Popen(
            [str(py), "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            cwd=str(BACKEND),
        )
        if not wait_for(args.base):
            print("server did not become ready")
            proc.terminate()
            return 1

    print(f"smoke-testing {args.base}\n")
    rc = run_checks(args.base)

    if proc is not None:
        print("\nserver still running (pid %d). Ctrl+C to stop." % proc.pid)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    return rc


if __name__ == "__main__":
    sys.exit(main())
