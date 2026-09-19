# คู่มือการใช้งาน MONGDEE AI Booth OS

คู่มือติดตั้งและรันแบบสั้น ๆ ตั้งแต่โคลนโปรเจกต์จนเปิดใช้งานได้ (รายละเอียดเชิงลึกดู [README.md](README.md))

## 1. โคลนโปรเจกต์

```bash
git clone https://github.com/chakkrit4456/MongDee.git
cd MongDee
```

## 2. ติดตั้ง


### แบบ command line (นักพัฒนา)

```bash
python3 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
python web_server.py
```

## 3. วิธีรัน

เปิด virtual environment ก่อนทุกครั้ง (`source .venv/bin/activate` หรือ Windows: `.venv\Scripts\activate`)

### ผ่านเบราว์เซอร์ (แนะนำ — ครบทุกฟีเจอร์ในหน้าเว็บเดียว)

```bash
python web_server.py
# เปิดเฉพาะเครื่องนี้: http://127.0.0.1:8000/

python web_server.py --host 0.0.0.0 --port 8000
# ให้อุปกรณ์อื่นในวง LAN เดียวกันเข้าถึงได้ด้วย (เช่น แท็บเล็ตหน้าบูธ)
# เปิดจากอุปกรณ์อื่น: http://<IP เครื่องที่รัน>:8000/
```

ตัวเลือกเพิ่มเติม: `--booth-id`, `--booth-name`, `--event-id`, `--cameras /dev/video0,/dev/video2` (ไม่ระบุ = ค้นหากล้องอัตโนมัติ), `--no-open` (ไม่เปิดเบราว์เซอร์อัตโนมัติ)

### ติดตั้งบนเครื่องใหม่ (สเปคต่างจากเครื่องพัฒนา)

ส่วนใหญ่ทำงานเหมือนกันทุกเครื่องโดยอัตโนมัติ ไม่ต้องแก้อะไร:

- **GPU/CPU** — `core/device.py` ตรวจหา CUDA (NVIDIA) → DirectML (AMD/Intel รวมถึงการ์ดจอออนบอร์ด) → CPU ให้เองตามลำดับ ไม่มี GPU ก็รันบน CPU ได้ปกติ ไม่ต้องกดอะไรในหน้า Booth Settings — ตรวจสถานะจริงได้ที่ `GET /api/system/gpu_status` (หน้า `/settings` ก็แสดงผลเดียวกัน) หรือถ้าเป็นการ์ด NVIDIA ใช้ `nvidia-smi` ดู GPU-Util/Memory ระหว่างรันเพื่อยืนยันว่ามีการใช้งานจริง ไม่ใช่แค่ตรวจพบว่ามี
- **ปรับภาระ AI ตามสเปค** — `core/performance.py` วัด CPU/RAM ของเครื่องแล้วปรับ FPS/ความละเอียดที่ AI ประมวลผลให้เหมาะสมอัตโนมัติ (ดูข้อความ `[HW] ... Tier: ...` ตอน server เริ่ม)
- **ติดตั้ง dependency** — `pip install -r requirements.txt` เหมือนกันทุก OS (แพ็กเกจที่ใช้เฉพาะ Windows เช่น `pygrabber` ถูกกำกับเงื่อนไข OS ไว้แล้ว ข้ามเองบน Linux/macOS)

จุดเดียวที่ **ต้องตั้งเองใหม่ทุกเครื่อง** (ทำครั้งเดียวตอนติดตั้ง) — การแยกกล้อง built-in ของโน้ตบุ๊คออกจากกล้อง USB จริง เพราะชื่อกล้อง built-in ต่างกันไปตามรุ่นเครื่อง (`data/booth_settings.json` เองก็ไม่ถูก commit ขึ้น git อยู่แล้ว เพื่อไม่ให้ค่าที่ตั้งไว้บนเครื่องหนึ่งไปทับเครื่องอื่น):

```bash
python scripts/camera_diagnostic.py
# ดูคอลัมน์ชื่อกล้อง — เครื่องนี้จะเห็นประมาณ:
#   0: HD WebCam            <- นี่คือกล้อง built-in ของโน้ตบุ๊คเครื่องนี้
#   1: USB Camera
```

จากนั้นแก้/สร้าง `data/booth_settings.json` (ถ้ายังไม่มีให้สร้างไฟล์ใหม่) ใส่ชื่อกล้อง built-in ของ**เครื่องนั้น**ตามที่เห็นจริงจาก diagnostic:

```json
{
  "enable_builtin_camera": false,
  "builtin_camera_names": ["ชื่อกล้อง built-in ที่เห็นจริงบนเครื่องนี้"]
}
```

ถ้าไม่ตั้งค่านี้ไว้ ระบบจะยังทำงานได้ปกติ เพียงแต่กล้อง built-in ของโน้ตบุ๊คเครื่องนั้นอาจถูกนับเป็นกล้อง USB ตัวหนึ่งไปด้วย (ไม่เดาให้เองเพราะเสี่ยงตัดกล้อง USB ตัวจริงผิดตัว)

**กรอบ detect แยกสีตามเพศ/เด็ก (ตัวเลือกเสริม):** ปกติกรอบรอบตัวคนจะเป็นสีแดงเหมือนกันหมด ถ้าต้องการให้กรอบเปลี่ยนเป็น **สีแดง = ผู้หญิง / สีน้ำเงิน = ผู้ชาย** ต้องเตรียมโมเดลจำแนกเพศ (Caffe หรือ ONNX) เอง แล้วระบุ `--gender-model path/to/gender.caffemodel --gender-prototxt path/to/deploy_gender.prototxt` (ONNX ไม่ต้องใส่ `--gender-prototxt`) ตอนรัน `python web_server.py` หรือ `python app.py` — ไม่ระบุก็ใช้งานได้ตามปกติ กรอบคนยังเป็นสีแดงเดียวเหมือนเดิม เพราะระบบไม่เดาเพศเองถ้าไม่มีโมเดลจริงมายืนยัน

ถ้าอยากแยก **เด็ก** ออกมาเป็นสีเหลือง/ส้มเพิ่มด้วย (แซงหน้าสีเพศ — พบว่าเป็นเด็กจะไม่โชว์ชาย/หญิงซ้อน) ต้องมีโมเดลจำแนกช่วงอายุเพิ่มอีกตัว แล้วระบุ `--age-model path/to/age.caffemodel --age-prototxt path/to/deploy_age.prototxt` คู่กับ `--gender-model` (ต้องมี `--gender-model` ด้วยเสมอ ใช้ backend เดียวกัน) โมเดลตัวอย่างที่ใช้ได้คือ Levi & Hassner age net (ช่วงอายุ 0-2, 4-6, 8-12 ถือเป็นเด็ก ดูได้ที่ `core/vision.py`'s `CHILD_AGE_GROUPS`)

หน้าเว็บที่ได้: `/booth` (สตรีมกล้องสด + ปุ่มเต็มจอ/เปิดกล้องแยกหน้าต่างต่อจอ), `/product-view` (ดูรายละเอียดสินค้าที่สแกนเจอ แยกหน้าต่างต่างหาก), `/dashboard` (วิเคราะห์ข้อมูล), `/trainer` (เทรน AI ให้รู้จักสินค้า) — เปิดพร้อมกันหลายแท็บ/หลายอุปกรณ์/หลายจอได้

### แบบ Desktop (PySide6)

```bash
python launcher.py                          # หน้าต่างเดียว มีปุ่มเปิดทุกโหมด
python app.py --cameras /dev/video0,/dev/video2   # เปิดบูธตรง ๆ
python dashboard.py                         # เปิด Dashboard แยก
python trainer.py                           # เปิด AI Trainer แยก
```

### สร้างไฟล์ .exe/executable ตัวเดียว (แจกให้คนอื่นใช้)

```bash
bash build_linux.sh        # Linux
build_windows.bat          # Windows
```
ดูรายละเอียดที่ [BUILD.md](BUILD.md)

## 4. เริ่มใช้งานสินค้าจริง

แคตตาล็อกสินค้าเริ่มต้นว่างเปล่า — เข้าหน้า `/trainer` แล้ว "เพิ่มสินค้าใหม่" (กรอกชื่อ/แท็กไลน์/ราคา/รายละเอียด/FAQ) จากนั้นอัปโหลดรูป/วิดีโอ หรือใช้ปุ่ม "บันทึกจากกล้อง" ให้ระบบเก็บภาพจากกล้องสดให้อัตโนมัติ
