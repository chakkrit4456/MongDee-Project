@echo off
REM ดับเบิลคลิกไฟล์นี้เพื่อเปิด MONGDEE AI Booth OS ผ่านเบราว์เซอร์
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] ยังไม่ได้ติดตั้งโปรแกรม กรุณารัน install.bat ก่อน
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python web_server.py
pause
