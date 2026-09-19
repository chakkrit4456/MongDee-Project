#!/usr/bin/env bash
# สร้างไฟล์ execuTABLE แบบ standalone ของ "ตัวเปิดแอป" (launcher.py) สำหรับ Linux
# โดยใช้ PyInstaller — รันสคริปต์นี้บนเครื่อง Linux เท่านั้น (PyInstaller ไม่รองรับ
# cross-compile ข้าม OS)
#
# หมายเหตุสำคัญ: ไฟล์ที่ได้ (dist/MONGDEE-AI-Booth-OS) เป็น "ตัวเปิดแอป" ที่เบา
# (แค่ PySide6) ไม่ได้รวม PyTorch/OpenCV/YOLO ไว้ในไฟล์เดียว (ไลบรารีเหล่านั้นรวมกัน
# หนักหลาย GB โดยเฉพาะรุ่น GPU) — ตัวเปิดแอปนี้จะพาไปติดตั้งไลบรารีจริงแยกต่างหาก
# ผ่าน install.sh ในครั้งแรกที่รัน (ดู BUILD.md)
set -e
cd "$(dirname "$0")"

BUILD_VENV=".build_venv"

echo "================================================"
echo " กำลังสร้าง MONGDEE-AI-Booth-OS (Linux executable)"
echo "================================================"

if [ ! -d "$BUILD_VENV" ]; then
    python3 -m venv "$BUILD_VENV"
fi
# shellcheck disable=SC1091
source "$BUILD_VENV/bin/activate"
pip install --upgrade pip --quiet
pip install PySide6 pyinstaller --quiet

rm -rf build dist launcher.spec
pyinstaller --onefile --windowed \
    --name "MONGDEE-AI-Booth-OS" \
    --icon assets/icon.png \
    launcher.py

echo ""
echo "✅ สร้างเสร็จแล้ว: dist/MONGDEE-AI-Booth-OS"
echo "   วางไฟล์นี้ไว้ที่โฟลเดอร์โปรเจกต์เดียวกับ app.py/install.sh แล้วดับเบิลคลิกเปิดได้เลย"
