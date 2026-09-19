@echo off
rem Exposes ONLY the dashboard (read-only) for viewing from outside the LAN.
rem Run this alongside run_web.bat (web_server.py must already be running on :8000).
rem
rem 1. This window: proxies :8000 -> :8090, serving GET-only dashboard routes.
rem 2. In a separate window, tunnel :8090 to the internet, e.g.:
rem      cloudflared tunnel --url http://127.0.0.1:8090
rem    then use the https://*.trycloudflare.com URL it prints as the
rem    "MongDee" dashboard link on the main portal page.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] ยังไม่ได้ติดตั้งโปรแกรม กรุณารัน install.bat ก่อน
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
echo.
echo Public dashboard proxy: http://127.0.0.1:8090/dashboard
echo (forwards to web_server.py on http://127.0.0.1:8000 — run_web.bat must be running)
echo.
python dashboard_proxy.py
pause
