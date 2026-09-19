@echo off
rem Runs the kiosk with a simulated camera (no webcam needed).
rem Scene switcher: http://127.0.0.1:8000/dev/fake-camera
rem Use "set KIOSK_FAKE_CAMERA=auto" for a self-cycling empty/product loop.
set "KIOSK_FAKE_CAMERA=1"
call "%~dp0start.bat"
