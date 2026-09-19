@echo off
REM ตัวติดตั้ง MONGDEE AI Booth OS สำหรับ Windows
REM
REM วิธีใช้:
REM   install.bat          ติดตั้งรุ่น CPU (ใช้ได้ทุกเครื่อง แนะนำสำหรับส่วนใหญ่)
REM   install.bat --gpu    ตรวจจับการ์ดจอในเครื่องอัตโนมัติแล้วติดตั้งตัวเร่งความเร็วที่เหมาะสม
REM                        (NVIDIA -> PyTorch CUDA build, AMD/Intel/การ์ดจออื่น ๆ ที่รองรับ
REM                        DirectX12 -> PyTorch CPU build + torch-directml) ไม่ต้องรู้เองว่า
REM                        เครื่องมีการ์ดจอยี่ห้อไหน
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ================================================
echo  MONGDEE AI Booth OS - Installer (Windows)
echo ================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo         Be sure to check "Add python.exe to PATH" during setup.
    exit /b 1
)

if not exist ".venv" (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
) else (
    echo [1/5] Virtual environment already exists, skipping.
)

call ".venv\Scripts\activate.bat"

echo [2/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

set GPU_MODE=0
if "%1"=="--gpu" set GPU_MODE=1

if "%GPU_MODE%"=="1" (
    echo [3/5] Detecting graphics card for hardware acceleration...
    set GPU_NAMES=
    for /f "delims=" %%G in ('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue).Name -join ';'" 2^>nul') do set GPU_NAMES=%%G
    echo       Found: !GPU_NAMES!

    echo !GPU_NAMES! | findstr /I "NVIDIA" >nul
    if not errorlevel 1 (
        echo       NVIDIA GPU detected - installing PyTorch CUDA build ^(large download, ~2-3GB^)...
        REM --force-reinstall --no-deps: without this, pip sees "torch is
        REM already installed" (true, but as the CPU-only build from a
        REM previous plain install) and does *nothing* -- it doesn't check
        REM whether the installed build variant actually matches the index
        REM being requested. That silently leaves CPU torch in place while
        REM this script reports success, which is exactly what happened the
        REM first time this was tested for real. --no-deps skips redundant
        REM re-downloads of torch's own already-satisfied dependencies
        REM (numpy, pillow, ...) -- only the two build-specific wheels
        REM themselves need swapping.
        pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ) else (
        echo !GPU_NAMES! | findstr /I "AMD Radeon Intel" >nul
        if not errorlevel 1 (
            REM DirectML works through any DirectX12-capable GPU's normal
            REM Windows driver (WDDM) -- no vendor SDK to install separately,
            REM which is what makes this "any model, any driver" rather than
            REM needing a specific AMD/Intel toolkit picked by hand -- this
            REM also covers onboard/integrated Intel and AMD graphics, not
            REM just discrete cards.
            REM
            REM torch-directml has no wheel at all for Python 3.13+ (its
            REM newest release only ships cp38-cp312) -- pip would otherwise
            REM fail the whole install right here. Detect that up front and
            REM fall back to the plain CPU build instead of a hard failure.
            python -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)"
            if errorlevel 1 (
                echo       Non-NVIDIA GPU detected, but this Python version is too new for DirectML ^(needs 3.12 or older^) - installing PyTorch ^(CPU build^) instead...
                pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cpu
            ) else (
                echo       Non-NVIDIA GPU detected - installing PyTorch ^(CPU build^) + DirectML acceleration...
                REM torch-directml pins its own exact torch/torchvision
                REM versions (see its Requires-Dist on PyPI) -- pinning them
                REM here too instead of leaving them unpinned means pip
                REM resolves both installs to the same versions from the
                REM start, rather than installing latest torch here and then
                REM having the next pip call silently re-resolve/downgrade
                REM it out from under this step to satisfy torch-directml's
                REM pin. Bump these two together with whatever torch-directml
                REM version is in use, never independently.
                pip install --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
                pip install torch-directml --quiet
            )
        ) else (
            echo       No dedicated GPU detected ^(or detection failed^) - installing PyTorch ^(CPU build^)...
            pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cpu
        )
    )
) else (
    echo [3/5] Installing PyTorch ^(CPU build - works on any PC, ~200MB^)...
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo [4/5] Installing remaining libraries (OpenCV, YOLO, PySide6, voice, web server, ...)...
pip install opencv-python ultralytics PySide6 pyttsx3 SpeechRecognition sounddevice numpy fastapi "uvicorn[standard]" jinja2 python-multipart openpyxl requests --quiet

echo [5/5] Creating shortcuts...
(
echo @echo off
echo cd /d "%%~dp0"
echo call ".venv\Scripts\activate.bat"
echo python launcher.py
) > run_launcher.bat

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%USERPROFILE%\Desktop\MONGDEE AI Booth OS.lnk');" ^
    "$s.TargetPath = '%~dp0run_launcher.bat';" ^
    "$s.WorkingDirectory = '%~dp0';" ^
    "$s.IconLocation = '%~dp0assets\icon.ico';" ^
    "$s.Save()"

echo.
echo Installation complete!
echo Launch from the Desktop shortcut "MONGDEE AI Booth OS", or run run_launcher.bat
REM No `pause` here on purpose: launcher.py runs this script headlessly via
REM QProcess (see its _on_run_setup) with no interactive console attached,
REM so a trailing `pause` would block forever waiting for a keypress that
REM can never arrive — the whole script would look "stuck" even though
REM installation already finished, and the launcher's Setup button would
REM never re-enable. Anyone running install.bat by double-clicking it
REM directly still sees every line above before the window closes; only the
REM final "press any key" hold is gone.
