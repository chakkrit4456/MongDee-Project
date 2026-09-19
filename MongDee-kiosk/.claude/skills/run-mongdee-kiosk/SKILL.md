---
name: run-mongdee-kiosk
description: Launch and drive the MONGDEE MINI KIOSK app (FastAPI backend + static frontend, webcam product scanner). Use when asked to run, start, serve, smoke-test, or screenshot the kiosk / dashboard / enroll pages, or to confirm a backend change works in the real app.
---

# Run MONGDEE MINI KIOSK

Windows-only local web app. A Python/FastAPI backend (`backend/app`) serves the
plain-HTML frontend (`frontend/`) and an MJPEG webcam stream on
`http://127.0.0.1:8000`. There is **no build step** and no system Python — the
app runs from the checked-in venv at `backend/venv`.

Driver: [`.claude/skills/run-mongdee-kiosk/smoke.py`](.claude/skills/run-mongdee-kiosk/smoke.py) —
launches the server and hits every page route + JSON API, prints a pass/fail
table. This is the agent path; a browser is the human path.

All paths below are relative to the repo root (the directory containing
`start.bat`). Commands are PowerShell unless noted.

## Prerequisites (one-time, needs internet)

The repo ships `backend/venv`, but it was created on a different machine
(`D:\`, user `wachi`) so its interpreter link is dead, and this machine has
**no Python on PATH**. Repair it once:

```powershell
# 1. Install Python 3.11 (venv is 3.11.7; 3.11.9 is ABI-compatible)
winget install --id Python.Python.3.11 --source winget `
  --accept-package-agreements --accept-source-agreements --disable-interactivity

$py = "$env:LOCALAPPDATA\Programs\Python\Python311"

# 2. Drop a working interpreter + its DLLs into the venv
copy "$py\python.exe","$py\pythonw.exe","$py\python3.dll","$py\python311.dll",`
     "$py\vcruntime140.dll","$py\vcruntime140_1.dll" `
     "backend\venv\Scripts\"

# 3. Repoint the venv config
@"
home = $py
include-system-site-packages = false
version = 3.11.9
executable = $py\python.exe
"@ | Set-Content -Encoding utf8 backend\venv\pyvenv.cfg

# 4. The shipped opencv build (5.0.0.93, a broken pre-release) fails to load.
#    Replace it with a known-good headless build (server needs no GUI window).
backend\venv\Scripts\python.exe -m pip install --force-reinstall --no-cache-dir `
  "opencv-python-headless==4.10.0.84"
```

Verify:

```powershell
cd backend
venv\Scripts\python.exe -c "import fastapi, uvicorn, cv2, torch, torchvision; from app.main import app; print('ok')"
cd ..
```

torchvision downloads MobileNetV3 weights (~10 MB) on first real scan, cached
under `%USERPROFILE%\.cache\torch`. After that the app is fully offline.

## Run — agent path (driver)

From the repo root:

```powershell
backend\venv\Scripts\python.exe .claude\skills\run-mongdee-kiosk\smoke.py --launch
```

Starts uvicorn, waits for readiness, checks all routes, prints:

```
  OK   /                            200
  OK   /enroll                      200
  OK   /manage                      200
  OK   /dashboard                   200
  OK   /calibrate                   200
  OK   /api/camera/status           200  {"state":"empty","camera_open":false,...}
  OK   /api/camera/devices          200
  OK   /api/products                200
  OK   /api/recognition/current     200
  OK   /api/analytics/summary       200
  OK   /api/analytics/hourly        200

all checks passed
```

Then it leaves the server running in the foreground (Ctrl+C to stop). Drop
`--launch` to check a server you started separately, or pass
`--base http://127.0.0.1:8000`.

To screenshot a page, launch the server (above) then drive a browser with
`chromium-cli` against `http://127.0.0.1:8000/` (kiosk), `/dashboard`, or
`/enroll`.

## Run — human path

```powershell
.\start.bat
```
Serves on `http://127.0.0.1:8000/`. Open that in a browser, F11 for fullscreen
kiosk. Pages: `/`, `/enroll`, `/manage`, `/dashboard`, `/calibrate`.

Caveat: `start.bat` only skips dependency install when `backend\.venv\Scripts\python.exe`
exists (note the dot). With the repaired `backend\venv` (no dot) it runs
`pip install -r requirements.txt` on **every** launch, which reinstalls the
broken `opencv-python` 5.x — so after any `start.bat` run, redo Prerequisites
step 4. The driver avoids this entirely; prefer it.

## Run without the full server (direct invocation)

```powershell
cd backend
venv\Scripts\python.exe -m unittest discover -s tests -v    # 36 tests, no camera/network
```
(the venv has no `pytest`; the tests are stdlib `unittest`.)
`app.config` has every threshold; `app.db` opens `backend/data/mongdee.db`
(SQLite, checked in with sample products). No env vars or init guards.

## Gotchas

- **The camera almost never works headless / under automation.** OpenCV's
  DSHOW backend logs `backend is generally available but can't be used to
  capture by index` and `camera_open` stays `false`. This is expected and the
  app is built for it: `/api/camera/status` returns `state:"empty"` or
  `needs_reference`, recognition sits in `camera_unavailable`, and **every page
  and API still serves 200**. Don't treat a missing camera as a failed run —
  the smoke table passing is the bar.
- **An unclean shutdown wedges the process.** `app.main`'s lifespan calls
  `camera_manager.stop()`, which can raise (`RuntimeError: กล้องยังไม่หยุดทำงาน`)
  and leave a Python process holding port 8000 (and the camera device). It
  ignores `taskkill /F /PID` and `Stop-Process -Id` (both report success or
  "no running instance" while the process keeps listening). What works:
  ```powershell
  $c = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
  if ($c) { (Get-Process -Id $c.OwningProcess).Kill() }
  ```
  Always clear port 8000 this way before a fresh launch, or uvicorn dies with
  `[Errno 10048] ... only one usage of each socket address`.
- **Don't pipe the server's stdout into a command that exits** (`... | head`).
  On Windows uvicorn then blocks on log writes and the server hangs while still
  holding the port. Run the driver without a trailing pipe.
- **`opencv-python` 5.x and 4.11+ pre-release wheels** fail with
  `ImportError: DLL load failed while importing cv2` even with VC++ redist
  present — the missing piece is `python3.dll` (step 2 above) plus staying on
  `opencv-python-headless==4.10.0.84`.
- **`start.bat` reinstalls the broken opencv.** It skips `pip install -r
  requirements.txt` only when `backend\.venv\Scripts\python.exe` exists; the
  repaired venv is `backend\venv` (no dot), so every `start.bat` run reinstalls
  `opencv-python` 5.x. Prefer the driver; if you must use `start.bat`, redo
  Prerequisites step 4 afterward.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No Python at '...\wachi\...python.exe'` | venv not repaired — do Prerequisites. |
| `ImportError: DLL load failed while importing cv2` | Copy `python3.dll` into `backend\venv\Scripts`; reinstall `opencv-python-headless==4.10.0.84`. |
| `[Errno 10048]` / `only one usage of each socket address` | Stale server on :8000 — kill via `(Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess).Kill()`. |
| Server "started" but every request times out | Wedged process (its stdout was piped to something that exited, or a prior crash) — kill as above, relaunch without any stdout pipe. |
| `camera_open: false`, DSHOW warnings | Expected without a dedicated webcam; not a failure. |
