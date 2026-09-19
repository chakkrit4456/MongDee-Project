@echo off
REM สร้างไฟล์ .exe แบบ standalone ของ "ตัวเปิดแอป" (launcher.py) สำหรับ Windows
REM โดยใช้ PyInstaller — รันสคริปต์นี้บนเครื่อง Windows เท่านั้น (PyInstaller ไม่รองรับ
REM cross-compile ข้าม OS ต้องรันบน OS เป้าหมายจริง)
REM
REM หมายเหตุสำคัญ: ไฟล์ที่ได้ (dist\MONGDEE-AI-Booth-OS.exe) เป็น "ตัวเปิดแอป" ที่เบา
REM (แค่ PySide6) ไม่ได้รวม PyTorch/OpenCV/YOLO ไว้ในไฟล์เดียว (ไลบรารีเหล่านั้นรวมกัน
REM หนักหลาย GB โดยเฉพาะรุ่น GPU) — ตัวเปิดแอปนี้จะพาไปติดตั้งไลบรารีจริงแยกต่างหาก
REM ผ่าน install.bat ในครั้งแรกที่รัน (ดู BUILD.md)
setlocal
cd /d "%~dp0"

set BUILD_VENV=.build_venv

echo ================================================
echo  Building MONGDEE-AI-Booth-OS.exe (Windows)
echo ================================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist "%BUILD_VENV%" (
    python -m venv "%BUILD_VENV%"
)
call "%BUILD_VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
pip install PySide6 pyinstaller --quiet

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist launcher.spec del /q launcher.spec

pyinstaller --onefile --windowed ^
    --name "MONGDEE-AI-Booth-OS" ^
    --icon assets\icon.ico ^
    launcher.py

echo.
echo Build complete: dist\MONGDEE-AI-Booth-OS.exe
echo Copy this file into the same folder as app.py / install.bat, then double-click to run.
pause
