@echo off
cd /d "%~dp0backend"

if exist .venv\Scripts\python.exe (
    set "KIOSK_PYTHON=.venv\Scripts\python.exe"
    goto run
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/checking dependencies...
pip install -r requirements.txt
if errorlevel 1 exit /b 1
set "KIOSK_PYTHON=venv\Scripts\python.exe"

:run
echo.
echo Starting MONGDEE MINI KIOSK on http://127.0.0.1:8000
echo   Kiosk display : http://127.0.0.1:8000/
echo   Add product   : http://127.0.0.1:8000/enroll
echo   Reports       : http://127.0.0.1:8000/dashboard
echo.

"%KIOSK_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
