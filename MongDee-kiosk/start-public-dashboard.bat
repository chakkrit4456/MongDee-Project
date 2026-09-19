@echo off
rem Exposes ONLY the dashboard (read-only) for viewing from outside the LAN.
rem Run this alongside start.bat (the kiosk app must already be running on :8000).
rem
rem 1. This window: proxies :8000 -> :8091, serving GET-only dashboard routes.
rem 2. In a separate window, tunnel :8091 to the internet, e.g.:
rem      cloudflared tunnel --url http://127.0.0.1:8091
rem    then use the https://*.trycloudflare.com URL it prints as the
rem    "Mini Kiosk" dashboard link on the main portal page.
cd /d "%~dp0backend"

if exist .venv\Scripts\python.exe (
    set "KIOSK_PYTHON=.venv\Scripts\python.exe"
    goto run
)
if exist venv\Scripts\python.exe (
    set "KIOSK_PYTHON=venv\Scripts\python.exe"
    goto run
)
set "KIOSK_PYTHON=python"

:run
echo.
echo Public dashboard proxy: http://127.0.0.1:8091/dashboard
echo (forwards to the kiosk app on http://127.0.0.1:8000 — start.bat must be running)
echo.
"%KIOSK_PYTHON%" dashboard_proxy.py
