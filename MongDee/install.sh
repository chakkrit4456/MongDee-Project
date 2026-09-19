#!/usr/bin/env bash
# ตัวติดตั้ง MONGDEE AI Booth OS สำหรับ Linux
#
# วิธีใช้:
#   bash install.sh          ติดตั้งรุ่น CPU (ใช้ได้ทุกเครื่อง แนะนำสำหรับส่วนใหญ่)
#   bash install.sh --gpu    ติดตั้งรุ่นเร่งความเร็วด้วย NVIDIA GPU (ต้องมีการ์ดจอ NVIDIA)
#
# สร้าง Python virtual environment ในโฟลเดอร์ .venv/ ติดตั้งไลบรารีที่จำเป็นทั้งหมด
# และสร้างทางลัดเปิดแอปบนเมนู Applications (ไม่แตะระบบ Python หลักของเครื่อง)
set -e
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
GPU_MODE=false
[ "$1" = "--gpu" ] && GPU_MODE=true

echo "================================================"
echo " MONGDEE AI Booth OS — ตัวติดตั้ง (Linux)"
echo "================================================"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] ไม่พบ $PYTHON_BIN กรุณาติดตั้ง Python 3.10 ขึ้นไปก่อน (เช่น sudo apt install python3 python3-venv)"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] กำลังสร้าง Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[1/5] พบ virtual environment เดิมแล้ว ข้ามขั้นตอนนี้"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] กำลังอัปเดต pip..."
pip install --upgrade pip --quiet

# --force-reinstall --no-deps: pip treats "torch is already installed" as
# satisfied even when the installed build variant (CPU vs CUDA) doesn't
# match the index being requested here -- without this it silently leaves
# whatever was already there in place while this script reports success.
# --no-deps skips redundant re-downloads of torch's own already-satisfied
# dependencies; only the two build-specific wheels need swapping.
if $GPU_MODE; then
    echo "[3/5] กำลังติดตั้ง PyTorch รุ่น GPU/NVIDIA CUDA (ไฟล์ใหญ่ ~2-3GB ใช้เวลานาน)..."
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
else
    echo "[3/5] กำลังติดตั้ง PyTorch รุ่น CPU (~200MB ใช้ได้ทุกเครื่อง)..."
    pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[4/5] กำลังติดตั้งไลบรารีที่เหลือ (OpenCV, YOLO, PySide6, เสียงพูด ฯลฯ)..."
pip install opencv-python ultralytics PySide6 pyttsx3 SpeechRecognition sounddevice numpy \
    fastapi "uvicorn[standard]" jinja2 python-multipart openpyxl requests --quiet

echo "[5/5] กำลังสร้างทางลัดเปิดแอป..."
cat > run_launcher.sh <<EOF
#!/usr/bin/env bash
cd "$(pwd)"
source "$VENV_DIR/bin/activate"
exec python launcher.py
EOF
chmod +x run_launcher.sh

if [ -d "$HOME/.local/share/applications" ] || mkdir -p "$HOME/.local/share/applications" 2>/dev/null; then
    cat > "$HOME/.local/share/applications/mongdee-ai-booth-os.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=MONGDEE AI Booth OS
Comment=ระบบปฏิบัติการสำหรับบูธอัจฉริยะ
Exec=$(pwd)/run_launcher.sh
Icon=$(pwd)/assets/icon.png
Terminal=false
Categories=Utility;
EOF
    echo "      เพิ่มทางลัดในเมนู Applications แล้ว (ค้นหา \"MONGDEE AI Booth OS\")"
fi

echo ""
echo "✅ ติดตั้งเสร็จสมบูรณ์!"
echo "เปิดแอปได้โดย:  ./run_launcher.sh"
echo "หรือค้นหา \"MONGDEE AI Booth OS\" ในเมนูแอปพลิเคชันของระบบ"
[ -n "$(command -v espeak-ng)" ] || echo "หมายเหตุ: ยังไม่มี espeak-ng (เสียงพูด) ติดตั้งเพิ่มด้วย: sudo apt install espeak-ng"
