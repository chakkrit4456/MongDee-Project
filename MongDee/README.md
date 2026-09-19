# MONGDEE "AI Booth OS"

ระบบปฏิบัติการสำหรับบูธอัจฉริยะ (Intelligent Booth Operating System) — สร้างขึ้นตามโจทย์
รอบคัดเลือก **"ออกแบบและพัฒนาสิ่งประดิษฐ์ อุปกรณ์ หรือเทคโนโลยี เพื่อยกระดับการทำงานและ
สร้างประสบการณ์ใหม่ในธุรกิจ MICE"** ของ IMPACT

โปรเจกต์นี้เป็น **ซอฟต์แวร์ที่ใช้งานได้จริง** (ไม่ใช่แค่เอกสารสรุป) เขียนด้วย Python +
YOLO11 (Ultralytics) รันบนเว็บแคมจริงตั้งแต่ 1 ตัวขึ้นไป และรองรับหลายกล้องทำงานพร้อมกัน
เพื่อช่วยกันตรวจจับ ตามที่ได้ทดสอบจริงบนเครื่องนี้ — ใช้งานได้ **2 แบบ**:
- **โปรแกรม Desktop** (PySide6) — `app.py` / `dashboard.py` / `trainer.py`
- **ผ่านเบราว์เซอร์** (FastAPI + HTML/JS ธรรมดา ไม่ต้องติดตั้งอะไรเพิ่มบนอุปกรณ์ที่ใช้ดู) —
  `web_server.py` เปิดครบทุกฟีเจอร์ (Booth, Dashboard, AI Trainer) ในหน้าเว็บเดียว รองรับทั้ง
  เปิดดูบนเครื่องเดียวกัน หรือจากอุปกรณ์อื่นในวง LAN เดียวกัน (เช่น แท็บเล็ตหน้าบูธ)

ทั้งสองแบบใช้โค้ดตรวจจับ/จดจำสินค้าชุดเดียวกัน (`core/`) จึงพฤติกรรมเหมือนกันทุกประการ

---

## ส่วนที่ 1 — สรุปความเข้าใจจากโจทย์ (จากเอกสารที่ให้มา)

### แนวคิดของทีม
ทีมต้องการพัฒนา **"MONGDEE AI Booth OS"** — ไม่ใช่ Kiosk เครื่องเดียว แต่เป็น **Platform**
ที่เชื่อมต่อและบริหารจัดการบูธอัจฉริยะได้หลายร้อยบูธพร้อมกันในงานเดียว โดยใช้กล้อง
Webcam, จอแสดงผล, ไมโครโฟน, ลำโพง และคอมพิวเตอร์ประมวลผล ร่วมกับซอฟต์แวร์ส่วนกลาง
ให้บูธสามารถ **"มองเห็น ตรวจสอบ สื่อสาร และรายงานข้อมูล"** ได้อย่างอัจฉริยะ

### ที่มาของปัญหา
ในงานแสดงสินค้า/อีเวนต์ขนาดใหญ่ เจ้าหน้าที่ประจำบูธมีจำกัด แต่ผู้เข้าชมมีจำนวนมาก ทำให้:
- ให้ข้อมูลผู้เข้าชมได้ไม่ทั่วถึง ต้องตอบคำถามซ้ำ ๆ
- การตรวจสอบความพร้อมของอุปกรณ์ก่อนงาน และระหว่างงาน ต้องอาศัยคนตรวจเองทั้งหมด
- ข้อมูลความสนใจของผู้เข้าชมไม่ถูกเก็บ/วิเคราะห์อย่างเป็นระบบ

จึงนำ AI และระบบอัตโนมัติเข้ามาช่วยลดภาระเจ้าหน้าที่ เพิ่มความถูกต้องและความสม่ำเสมอ
ของข้อมูล ตรวจสอบความพร้อม-แจ้งปัญหาอุปกรณ์อัตโนมัติ และเก็บข้อมูลไว้วิเคราะห์ภายหลัง

### ระบบย่อยหลัก 5 ส่วน (ตามที่ระบุในโจทย์)
| # | ระบบย่อย | หน้าที่ |
|---|----------|---------|
| 1 | **AI Vision** | กล้องตรวจจับและจดจำสินค้าที่วางอยู่ในพื้นที่บูธ |
| 2 | **AI Product Assistant** | เมื่อรู้จักสินค้าแล้ว อธิบายรายละเอียด/จุดเด่น/วิธีใช้งาน และตอบคำถามผู้เข้าชม (เสียง/หน้าจอ) |
| 3 | **Booth Readiness Check** | ก่อนเริ่มงาน ตรวจสอบความพร้อมของกล้อง จอ ระบบ AI ฐานข้อมูล และการเชื่อมต่อ |
| 4 | **Booth Health Monitoring** | ระหว่างงาน ติดตามสถานะอุปกรณ์ต่อเนื่อง แจ้งเตือนเจ้าหน้าที่ทันทีเมื่อมีปัญหา |
| 5 | **AI Dashboard & Analytics** | รวบรวมข้อมูลการใช้งาน (สินค้ายอดนิยม, คำถามที่พบบ่อย, ช่วงเวลาที่มีการใช้งานสูง) แยกตาม Event ID / Booth ID เพื่อบริหารจัดการหลายบูธจากส่วนกลาง |

### จุดเด่น / ความแตกต่างจาก Kiosk ทั่วไป
- Kiosk ทั่วไปแค่ **แสดง** ข้อมูล — MONGDEE ทำได้ครบ **มองเห็น → เข้าใจ → สื่อสาร →
  ตรวจสอบ → วิเคราะห์** ด้วยระบบเดียวกัน


- ออกแบบให้ **หลายบูธ** ใช้ฐานความรู้ของตัวเอง แต่บริหารจัดการผ่านศูนย์กลางได้ (ขยายจาก
  ต้นแบบ 1 บูธ ไปสู่งานขนาดใหญ่หลายร้อยบูธ)
- สถาปัตยกรรมแบบ **Modular** ต่อยอดไปงานอื่นของ IMPACT ได้ ไม่จำกัดแค่บูธแสดงสินค้า
  (Event Setup, Technical Support, Facility Tools, Food & Beverage Tools ตามที่ระบุใน
  โจทย์รอบคัดเลือก)

### การประยุกต์ใช้กับงานของ IMPACT
- **Event Setup**: ตรวจสอบความพร้อมอุปกรณ์บูธก่อนเปิดงานอัตโนมัติ
- **Technical Support**: ติดตามสถานะกล้อง/จอ/ระบบ แจ้งเตือนทีมเทคนิคเมื่อพบปัญหาพร้อม
  ตำแหน่งบูธที่เกิดปัญหา
- **การบริการผู้เข้าชม**: AI ตอบคำถามเบื้องต้นแทนเจ้าหน้าที่ ลดภาระงานซ้ำ
- **การบริหารจัดการ**: Dashboard ส่วนกลางเห็นภาพรวมทุกบูธพร้อมกัน
- **การวิเคราะห์หลังจบงาน**: ข้อมูลที่เก็บได้นำไปปรับปรุงการจัดงานครั้งถัดไป

---

## ส่วนที่ 2 — สิ่งที่สร้างขึ้นจริง (Implementation)

### สถาปัตยกรรม

```
webcam-ai/
├── launcher.py             # 🖱️ ตัวเปิดแอป — หน้าต่างเดียวมีปุ่มเปิดบูธ/Dashboard/Trainer
├── install.sh / install.bat  # ตัวติดตั้ง (สร้าง .venv + ติดตั้งไลบรารี + สร้างทางลัด)
├── build_linux.sh / build_windows.bat  # คอมไพล์ launcher.py เป็น .exe/binary ตัวเดียว
├── BUILD.md                 # คู่มือติดตั้ง/สร้างตัวเปิดแอปแบบละเอียด
├── assets/icon.png, icon.ico # ไอคอนแอป
├── app.py                  # เปิดแอปบูธหลัก Desktop (multi-camera + AI Vision + Assistant)
├── dashboard.py            # เปิด Dashboard Desktop แบบแยกต่างหาก (ไม่ต้องมีกล้อง)
├── trainer.py               # เปิด AI Trainer Desktop แบบแยกต่างหาก (อัปโหลดรูป/วิดีโอเทรนสินค้า)
├── web_server.py            # เปิดทุกฟีเจอร์ผ่านเบราว์เซอร์ (FastAPI) — ดูหัวข้อ "ใช้งานผ่านเบราว์เซอร์"
├── products.json            # แคตตาล็อกสินค้า (ทั้งสินค้าสาธิตและสินค้าที่เทรนเอง)
├── core/                     # ไม่ขึ้นกับ GUI ใด ๆ (ไม่มี Qt) — ใช้ร่วมกันทั้ง Desktop และเว็บ
│   ├── vision.py              # AI Vision: CameraWorker ต่อกล้อง 1 ตัว (plain thread + callback) — รวม YOLO11 + ตัวจดจำสินค้าที่เทรนเอง
│   ├── localizer.py           # หา "บริเวณที่มีวัตถุ" แบบไม่ขึ้นกับคลาส (background subtraction) — ใช้กับสินค้าที่เทรนเอง
│   ├── recognizer.py          # จดจำสินค้าที่เทรนเองด้วย image embedding (few-shot, ไม่ต้อง train loop)
│   ├── training.py             # ตรรกะอัปโหลดรูป/วิดีโอเทรน ใช้ร่วมกันทั้ง Desktop Trainer และเว็บ Trainer
│   ├── aggregator.py          # รวมผลตรวจจับจากทุกกล้อง → ตัดสินใจว่าเจอสินค้าอะไร ("กล้องช่วยกัน")
│   ├── products.py            # แคตตาล็อกสินค้า + จับคู่คำถาม-คำตอบ (FAQ matching)
│   ├── assistant.py           # AI Product Assistant (Desktop): พูด (TTS ผ่าน pyttsx3) + ตอบคำถาม
│   ├── readiness.py           # Booth Readiness Check
│   └── database.py            # เก็บ Interaction / Health Event / Heartbeat ลง SQLite
├── ui/                       # ฝั่ง Desktop (PySide6)
│   ├── main_window.py         # หน้าจอหลักของบูธ
│   ├── dashboard_window.py    # AI Dashboard & Analytics
│   ├── trainer_window.py      # AI Trainer: อัปโหลดรูป/วิดีโอ, จัดการสินค้า, ทดสอบด้วยกล้องสด
│   └── qt_camera_bridge.py    # แปลง callback ของ core/vision.py เป็น Qt Signal (ให้ UI ใช้ต่อได้)
├── web/                      # ฝั่งเบราว์เซอร์ (FastAPI) — ใช้ core/ ชุดเดียวกับ Desktop ทั้งหมด
│   ├── server.py              # FastAPI routes: หน้าเว็บ, MJPEG stream, REST API
│   ├── booth_manager.py       # เวอร์ชันไม่มีหน้าจอของ ui/main_window.py (คุมกล้อง+state ทั้งหมด)
│   ├── templates/              # booth.html, dashboard.html, trainer.html, index.html
│   └── static/                  # style.css, booth.js, dashboard.js, trainer.js
├── scripts/
│   └── export_booth_data.py  # export ข้อมูลบูธเป็น JSON เพื่อรวมกับบูธอื่นใน Dashboard
├── yolo11n.pt                # น้ำหนักโมเดล YOLO11 ที่ core/vision.py โหลดใช้จริง
└── data/
    ├── mongdee.db             # ฐานข้อมูล SQLite (สร้างอัตโนมัติตอนรันครั้งแรก)
    └── gallery/                # ข้อมูลเทรน AI ต่อสินค้า (เฉพาะ embedding ไม่เก็บรูปต้นฉบับ — ดูหัวข้อการเทรน)
```

### แต่ละฟีเจอร์ตามโจทย์ ทำงานอย่างไรจริง

**1) AI Vision — กล้องหลายตัวช่วยกัน + จดจำสินค้าจริงได้ (ไม่ใช่แค่สินค้าสาธิต)**
`core/vision.py` เปิดกล้องแต่ละตัวใน thread ของตัวเอง (`CameraWorker`) และรันตัวตรวจจับ 2
แบบร่วมกันในทุกเฟรม:
- **YOLO11** (`yolo11n.pt`, ใช้ GPU/CUDA อัตโนมัติถ้ามี) — ตรวจจับ "คน" (กรอบสีแดง) เสมอ
  และตรวจจับสินค้าสาธิต 12 ชนิดที่ตรงกับคลาส COCO ได้ทันทีแบบ zero-setup
- **`core/localizer.py` + `core/recognizer.py`** — หาตำแหน่งวัตถุที่ถูกยกมาหน้ากล้องแบบ
  ไม่ขึ้นกับคลาส (background subtraction) แล้วจดจำว่าเป็นสินค้าอะไรด้วย image-embedding
  matching กับรูปที่อัปโหลดผ่าน **AI Trainer** — นี่คือเส้นทางสำหรับ **สินค้าจริงของผู้จัดบูธ**
  ที่ YOLO มาตรฐานไม่รู้จัก (ดูหัวข้อ "การเทรน AI ให้รู้จักสินค้าจริง" ด้านล่าง)

วัตถุที่ตรวจพบแต่ยังไม่ถูกเทรนจะขึ้นกรอบสีเทา "UNKNOWN" — เป็นสัญญาณบอกว่าควรไปเทรน
สินค้านั้นเพิ่มใน AI Trainer

`core/aggregator.py` คือส่วน **"กล้องหลายตัวช่วยกัน"** ตามที่โจทย์ระบุ: ถ้ากล้อง ≥ 2 ตัว
เห็นสินค้าชิ้นเดียวกัน**พร้อมกัน** ระบบยืนยันผลทันที (มุมมองที่กล้องหนึ่งมองไม่เห็น อีกกล้อง
ช่วยเห็นแทน) ถ้ามีกล้องเดียวก็ยังทำงานได้ปกติ โดยต้องเห็นสินค้าต่อเนื่องสักครู่ก่อนยืนยันผล
(กันสัญญาณรบกวน/วัตถุผ่านหน้ากล้องเร็ว ๆ)

**2) AI Product Assistant**
เมื่อยืนยันสินค้าได้ ระบบแสดงข้อมูลสินค้า (ชื่อ/จุดเด่น/รายละเอียด) บนแผงด้านขวาของ
หน้าจอทันที และพูดออกลำโพงผ่าน Text-to-Speech (`pyttsx3`) ผู้เข้าชมพิมพ์คำถามได้ในช่อง
"พิมพ์คำถามเกี่ยวกับสินค้านี้" ระบบจับคู่คำถามกับชุด FAQ ของสินค้านั้น (`core/products.py`)
แล้วตอบทั้งบนจอและด้วยเสียง

> **หมายเหตุเรื่องเสียง**: pyttsx3 บน Linux ต้องพึ่งเอนจิน `espeak-ng` ของระบบปฏิบัติการ
> เครื่องทดสอบนี้ไม่มี `espeak-ng` ติดตั้งไว้ (ต้องใช้สิทธิ์ผู้ดูแลระบบติดตั้ง) ระบบจึงตรวจพบ
> อัตโนมัติและตัดไปทำงานแบบข้อความอย่างเดียวโดยไม่ error หากต้องการเสียงพูดจริง ให้รัน
> `sudo apt install espeak-ng` ก่อนเปิดแอป (ดูหัวข้อ "การติดตั้ง")

**3) Booth Readiness Check**
กดปุ่ม **"🔍 ตรวจสอบความพร้อมบูธ"** ที่แถบด้านบน ระบบตรวจกล้องทุกตัว, โมเดล AI, ฐานข้อมูล,
ระบบเสียงพูด และไมโครโฟน แสดงผลเป็นรายการ ✅/❌ พร้อมสรุป READY/NOT READY และบันทึกผล
ลงฐานข้อมูล (`readiness_checks` table)

**4) Booth Health Monitoring**
แต่ละกล้องมี "จุดสถานะ" (เขียว = ปกติ, แดง = หลุดการเชื่อมต่อ) หากอ่านภาพจากกล้องไม่ได้
ต่อเนื่อง ระบบจะเปลี่ยนสถานะเป็นออฟไลน์ แจ้งเตือนในแถบ "🔔 การแจ้งเตือนล่าสุด" ทันที และ
พยายามเชื่อมต่อใหม่อัตโนมัติทุก 3 วินาที เมื่อกลับมาใช้งานได้จะแจ้งเตือนอีกครั้ง — ทุกเหตุการณ์
บันทึกลง `health_events` table พร้อม heartbeat ทุก 30 วินาทีเพื่อดูสถานะ uptime

**5) AI Dashboard & Analytics**
กดปุ่ม **"📊 เปิด Dashboard"** (หรือรัน `python dashboard.py` แยก) เพื่อดู: จำนวน Interaction
ทั้งหมด, สินค้ายอดนิยม, ปฏิสัมพันธ์ล่าสุด (คำถาม-คำตอบ), และ Log การแจ้งเตือนอุปกรณ์
กรองข้อมูลตาม **Event ID / Booth ID** ได้ และเพราะโจทย์ต้องรองรับ "หลายร้อยบูธในงานเดียวกัน"
— Dashboard จึงมีปุ่ม **"📂 นำเข้าข้อมูลบูธอื่น"** ให้โหลดไฟล์ที่ export จากบูธอื่น
(`scripts/export_booth_data.py`) มารวมแสดงผลได้ทันที จำลองการบริหารจัดการจากส่วนกลาง
โดยไม่ต้องมีเซิร์ฟเวอร์เครือข่ายส่วนกลาง (เหมาะกับการสาธิต/รันแยกเครื่องกันในงานจริง)

### ทดสอบแล้วบนฮาร์ดแวร์จริง
รันแอปด้วยเว็บแคมจริง 3 ตัวพร้อมกัน (`/dev/video0`, `/dev/video2`, `/dev/video4`) บนเครื่องนี้
— ทุกกล้อง stream ภาพสด, สถานะขึ้นเขียว (ปกติ), YOLO11 รันบน GPU (CUDA) ได้ราบรื่น ไม่มี
error และข้อมูลถูกบันทึกลงฐานข้อมูลถูกต้องตามที่ตรวจสอบด้วย `sqlite3`

### สิ่งที่เป็น "ข้อจำกัดของการสาธิต" อย่างตรงไปตรงมา
- **สินค้าสาธิต 12 ชนิดยังอิงคลาส COCO ของ YOLO11** (ขวดน้ำ, แก้ว, มือถือ, โน้ตบุ๊ก ฯลฯ)
  เพื่อให้ทดสอบได้ทันทีแบบ zero-setup — แต่สำหรับ **สินค้าจริงของผู้จัดบูธ** (เช่น บรรจุภัณฑ์
  เฉพาะแบรนด์ที่ YOLO ไม่รู้จัก) ระบบมี **AI Trainer** ให้อัปโหลดรูป/วิดีโอเทรนได้จริงแล้ว — ดู
  หัวข้อ "การเทรน AI ให้รู้จักสินค้าจริง" ด้านล่าง
- **AI Trainer เป็นการจดจำแบบ few-shot image matching** (เทียบรูปคล้ายรูป ไม่ใช่การเทรน
  object-detection model เต็มรูปแบบด้วย bounding-box) ความแม่นยำจึงขึ้นกับความหลากหลายของ
  รูปที่อัปโหลด (มุม/แสง/พื้นหลัง) มากกว่าความแม่นยำระดับ production-grade object detector —
  เหมาะสำหรับสาธิตและใช้งานบูธจริงในระดับที่ตั้งค่าได้เอง แต่หากต้องการความแม่นยำสูงสุดสำหรับ
  งานขนาดใหญ่ ควรพิจารณาเทรน YOLO11 เต็มรูปแบบด้วยชุดข้อมูล labeled จริง (Ultralytics
  รองรับ fine-tune ได้ตรง)
- การถามตอบเป็นการ **พิมพ์คำถาม** ไม่ใช่การพูดถาม (ยังไม่เปิดใช้ speech-to-text อัตโนมัติ
  เพราะการฟังเสียงต่อเนื่องต้องส่งไปประมวลผลผ่านบริการภายนอก ซึ่งควรเป็นทางเลือกที่ผู้จัดงาน
  เปิดเองอย่างชัดเจน ไม่ใช่ค่าเริ่มต้นที่ดักฟังตลอดเวลา) โครงสร้างรองรับส่วนนี้ไว้แล้ว
  (`SpeechRecognition` ติดตั้งไว้ใน `requirements.txt`) เป็นจุดต่อยอดง่าย ๆ

---

## การติดตั้งและเปิดแอป — แบบง่าย (แนะนำ)

ไม่ต้องพิมพ์คำสั่งใด ๆ เลย — มีปุ่มให้กดทั้งหมด:

**Linux:** `bash install.sh` (ครั้งแรกครั้งเดียว) → เปิดแอปจาก `./run_launcher.sh` หรือไอคอน
"MONGDEE AI Booth OS" ในเมนู Applications
**Windows:** ดับเบิลคลิก `install.bat` (ครั้งแรกครั้งเดียว) → เปิดแอปจากไอคอนบน Desktop

ทั้งสองจะเปิด **`launcher.py`** — หน้าต่างเดียวมีปุ่ม "🏬 เปิดบูธ", "📊 เปิด Dashboard",
"🎓 เปิด AI Trainer", "🌐 เปิดผ่านเบราว์เซอร์" แทนการพิมพ์คำสั่งใน terminal ทุกครั้ง

ต้องการไฟล์ `.exe`/execuTABLE ตัวเดียวดับเบิลคลิกได้ทันทีแม้ไม่เคยติดตั้งอะไรเลย (สำหรับแจก
ให้คนอื่นใช้)? ดู **[BUILD.md](BUILD.md)** — มีสคริปต์ `build_linux.sh` / `build_windows.bat`
ให้คอมไพล์ตัวเปิดแอปเป็นไฟล์เดียว (ทดสอบแล้วบน Linux ได้ไฟล์ขนาด 83MB รันได้ปกติ)

**เสียงพูด (ถ้าต้องการ)**: `sudo apt install espeak-ng` (Linux)

---

## การติดตั้งและเปิดแอป — แบบ command line (สำหรับนักพัฒนา)

โปรเจกต์นี้มี virtualenv (`.venv/`) พร้อม PyTorch (CUDA), Ultralytics, OpenCV อยู่แล้ว หากต้อง
ติดตั้งใหม่:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## วิธีรัน

```bash
source .venv/bin/activate

# เปิดตัวเปิดแอป (หน้าต่างเดียวมีปุ่มทั้งหมด)
python launcher.py

# หรือเปิดตรง ๆ ทีละส่วน:
# เปิดบูธ (ไม่ระบุ --cameras = ค้นหากล้องอัตโนมัติ)
python app.py --booth-id BOOTH-01 --booth-name "MONGDEE Demo Booth" --event-id "1-Day-at-IMPACT"

# ระบุกล้องเอง (รองรับหลายตัว คั่นด้วย comma)
python app.py --cameras /dev/video0,/dev/video2,/dev/video4

# เปิด Dashboard แยกต่างหาก (ไม่ต้องมีกล้อง)
python dashboard.py

# หรือเปิดผ่านเบราว์เซอร์แทน — ดูหัวข้อ "ใช้งานผ่านเบราว์เซอร์" ด้านล่าง
python web_server.py

# จำลองบูธที่สอง แล้ว export ไปรวมกับบูธแรกใน Dashboard
python app.py --booth-id BOOTH-02 --db data/booth2.db
python scripts/export_booth_data.py --db data/booth2.db --out booth2.json
# แล้วกด "📂 นำเข้าข้อมูลบูธอื่น" ใน Dashboard เลือกไฟล์ booth2.json
```

วิธีทดสอบ: เปิดแอป วางขวดน้ำ/แก้ว/มือถือ/โน้ตบุ๊ก ฯลฯ ให้เข้ากรอบกล้อง รอ ~1 วินาที ระบบจะ
ขึ้นข้อมูลสินค้าที่แผงด้านขวาอัตโนมัติ แล้วลองพิมพ์คำถาม เช่น "ราคาเท่าไหร่"

---

## ใช้งานผ่านเบราว์เซอร์

ทุกฟีเจอร์ (AI Vision, AI Product Assistant, Booth Readiness Check, Booth Health
Monitoring, AI Dashboard & Analytics, AI Trainer) ใช้งานผ่านเบราว์เซอร์ได้ครบ ไม่ต้องติดตั้ง
PySide6 หรือโปรแกรมใด ๆ บนอุปกรณ์ที่ใช้ "ดู" — กล้องเว็บแคมยังต่อกับเครื่องที่รัน
`web_server.py` เหมือนเดิม (เครื่องบูธจริง) เบราว์เซอร์แค่เชื่อมต่อเข้ามาดู/สั่งงานผ่าน HTTP

```bash
source .venv/bin/activate

# เปิดบนเครื่องเดียวกัน (เข้าถึงได้เฉพาะ http://127.0.0.1:8000/)
python web_server.py --booth-name "MONGDEE Demo Booth" --event-id "1-Day-at-IMPACT"

# เปิดให้อุปกรณ์อื่นในวง LAN เดียวกันเข้าถึงได้ด้วย (เช่น แท็บเล็ตหน้าบูธ)
python web_server.py --host 0.0.0.0 --port 8000
# แล้วเปิด http://<IP ของเครื่องบูธ>:8000/ จากอุปกรณ์อื่น
```

เปิดเบราว์เซอร์ไปที่ URL ที่ขึ้นในเทอร์มินัล จะเจอหน้า Landing page ลิงก์ไปยัง:
- **`/booth`** — วิดีโอสดจากทุกกล้อง (สตรีมแบบ MJPEG พร้อมกรอบสีตรวจจับวาดสดบนภาพ), ปุ่ม
  ตรวจสอบความพร้อมบูธ, และรายการแจ้งเตือน แต่ละกล้องมีปุ่ม "เต็มจอ" (ขยายกล้องนั้นแบบเต็ม
  หน้าจอในหน้าต่างปัจจุบัน) และปุ่ม "เปิดในหน้าต่างใหม่" (ย้ายไปแสดงเต็มจอบนอีกจอภาพได้ —
  แต่ละหน้าต่างเต็มจอได้แค่จอเดียว ถ้าต้องการหลายกล้องเต็มจอพร้อมกันบนหลายจอ ต้องเปิดแยก
  หน้าต่างต่อกล้อง)
- **`/product-view`** — แผง AI Product Assistant แยกออกมาเป็นหน้าต่างหาก (พิมพ์ถาม-ตอบ พร้อม
  เสียงพูดผ่าน **Web Speech API ของเบราว์เซอร์เอง** — ไม่ต้องพึ่ง `espeak-ng` บนเซิร์ฟเวอร์อีก
  ต่อไป) เปิดจากปุ่ม "เปิดหน้าดูสินค้า" ในหน้า `/booth` เพื่อย้ายไปแสดงบนอีกจอภาพได้เช่นกัน
- **`/dashboard`** — วิเคราะห์ข้อมูลเหมือนหน้าต่าง Desktop ทุกประการ รวมถึงปุ่ม "นำเข้าข้อมูล
  บูธอื่น" (ประมวลผลไฟล์ JSON ที่เลือกทั้งหมดในเบราว์เซอร์เอง ไม่ต้องส่งขึ้นเซิร์ฟเวอร์)
- **`/trainer`** — เพิ่มสินค้าใหม่, อัปโหลดรูปภาพ/วิดีโอเทรน, ดูสถานะความพร้อม, ทดสอบด้วย
  กล้องที่บูธกำลังใช้งานอยู่ (ไม่ต้องเปิดกล้องซ้ำ)

**สถาปัตยกรรม**: `web/server.py` (FastAPI) + `web/booth_manager.py` เป็นเวอร์ชันไม่มีหน้าจอของ
`ui/main_window.py` — ใช้ `core/vision.py` ชุดเดียวกับ Desktop ทุกประการ (ปรับ `core/vision.py`
ให้เป็น plain thread + callback แทน Qt Signal ตั้งแต่แรก เพื่อให้ทั้ง Desktop และเว็บใช้โค้ด
ตรวจจับ/จดจำสินค้าชุดเดียวกัน ไม่มีของสองชุดที่พฤติกรรมเพี้ยนกัน — ดู `ui/qt_camera_bridge.py`
สำหรับตัวแปลง callback → Qt Signal ที่ฝั่ง Desktop ใช้) วิดีโอส่งเป็น MJPEG stream (เบราว์เซอร์
รองรับ native ไม่ต้องมี JS พิเศษ), ส่วนสถานะ/การแจ้งเตือนอัปเดตผ่าน polling REST API ทุก
1.5 วินาที (ไม่ใช้ WebSocket เพื่อความเรียบง่าย เพียงพอกับความหน่วงที่ยอมรับได้ของ use case นี้)

**ทดสอบแล้วจริง**: รันด้วยเว็บแคมจริง เปิดใน Chromium จริง — เห็นวิดีโอสตรีมสด พร้อมกรอบ
ตรวจจับสีแดง "PERSON 91%" วาดทับภาพถูกต้อง, แผง AI Product Assistant อัปเดตข้อมูลสินค้า
ภาษาไทยถูกต้องสมบูรณ์ (ต่างจากภาพวิดีโอที่ font ของ OpenCV วาดภาษาไทยไม่ได้ — ฝั่งเบราว์เซอร์
ไม่มีข้อจำกัดนี้เลย), ประวัติการโต้ตอบและการแจ้งเตือนอัปเดตถูกต้อง, และ API ทุกตัว (state,
readiness, dashboard summary/top-products, products) ตอบกลับข้อมูลถูกต้องตามที่ตรวจสอบด้วย
`curl`

---

## กำลังพัฒนา — Multi-Camera Person Counting (`camera/`)

โมดูลใหม่ที่สร้างต่อยอดจากระบบข้างบน ตามสเปกใน
[`MongDee_Master_Prompt_Complete-1.md`](MongDee_Master_Prompt_Complete-1.md): นับผู้เข้าชมไม่ซ้ำ
ข้ามกล้องหลายตัวรอบบูธเดียวกัน (แยกจาก AI Vision ที่จดจำ**สินค้า**ข้างบนนี้ ซึ่งใช้งานได้จริงและทดสอบครบ)

**เฟส 1-9 (core person pipeline) เสร็จแล้ว:**

| เฟส | ได้อะไร |
|---|---|
| 1 Camera Gateway | ต่อกล้อง USB/RTSP/ONVIF/HTTP(MJPEG+snapshot)/HLS ผ่าน interface เดียว + reconnect/backoff |
| 2 Detection | ย่อเฟรม → YOLO ตรวจเฉพาะ "คน" → map bbox กลับพิกัดต้นฉบับ |
| 3 Tracking | ByteTrack + Kalman ต่อกล้อง → `local_track_id` ต่อเนื่อง, กัน ID switch, รองรับ occlusion |
| 4 Attributes | สีเสื้อ/กางเกง + รูปร่าง (ที่เหลือ UNKNOWN จนกว่าจะมีโมเดล) |
| 5 Re-ID | embedding + quality gate + track-level aggregation (backend: color/torchvision/OSNet) |
| 6 Global Identity | multi-feature weighted match + temporal/spatial constraint → MATCH/UNCERTAIN/NEW → Global Person ID |
| 7 Database | SQLite + migrations (`data/mongdee_vision.db` แยกจากระบบสินค้า) |
| 8 API | REST + WebSocket (`/api/persons`, `/api/events`, `/api/stats`, `/ws/events`, ...) + API-key auth |
| 9 Dashboard | หน้าเว็บ live: unique count, gender/age estimate, camera grid, event log, person list |

**Sections 48-95 (ฉบับขยาย) เสร็จแล้วเช่นกัน:** product detection + GI database, person↔product
association, head pose/gaze (proxy), estimated customer interest + look/dwell, product interaction,
camera calibration (homography) + 2D world coordinates + spatial fusion, real-time booth map +
movement paths + traffic/dwell/interest heatmaps, **Booth Designer** (2D drag-drop editor ที่ `/designer`),
layout versioning, user roles (ADMIN/OPERATOR/VIEWER) + audit log, และ API/dashboard extensions

```bash
python -m backend.main configs/mongdee.example.json   # รันทั้งระบบ → http://127.0.0.1:8100/
python -m pytest -q                                    # 299 เทส (+ 7 slow ที่รัน YOLO/model จริง)
```

ดู [`camera/README.md`](camera/README.md), [`vision/README.md`](vision/README.md),
[`backend/README.md`](backend/README.md)

**ต่อยอดเป็น Cloud + Remote AI Server + Vercel Dashboard แล้ว** ตามสเปกใน
[`MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md`](MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md):
กล้องแยกเป็น **Camera Agent** เบา ๆ (`camera_agent/`, ไม่มี torch/ultralytics) ที่ push เฟรมข้ามเครื่อง
มาที่ AI Server (`backend/`) ผ่าน HTTP ได้แล้ว, Dashboard เดิม deploy เป็น static site บน Vercel ได้
(`apps/web/`), และมี Adaptive Performance Engine เฉพาะของ pipeline นี้ (`backend/adaptive.py`)
รายละเอียดทั้งหมดอยู่ที่ [`docs/`](docs/) — เริ่มจาก [`docs/architecture.md`](docs/architecture.md)

**ยังเป็น "พร้อมพัฒนาต่อ" ไม่ใช่ "พร้อม deploy ทันที":**
- เครื่องพัฒนานี้ **CPU-only** (~2-4 fps/กล้องเมื่อมีหลายกล้อง) — โค้ดใช้ GPU เองถ้ามี (`device=auto`) แต่ยังไม่เคยทดสอบบน GPU
- **ยังไม่มี Re-ID weights เฉพาะทาง** (OSNet) → ความแม่นในการ merge ข้ามกล้องยังต่ำ; threshold ตั้งแบบ
  conservative (เอนไปทาง "ไม่ merge" ตามส่วนที่ 47) และ **ต้อง tune กับ footage จริงก่อนใช้งาน** (`scripts/evaluate.py`)
- **ยังไม่มีโมเดล** สำหรับ face / gender / age / head-pose / pedestrian-attributes → คืน `UNKNOWN` อย่างซื่อสัตย์
  (มี hook ไว้เสียบโมเดลแล้วทุกจุด)
- ONVIF / HLS ยังไม่เคยทดสอบกับฮาร์ดแวร์จริง (protocol เขียนตามสเปก + เทสกับ mock server)
- **Camera Agent / AI Server ผ่าน network** ทดสอบจริงแล้วทั้งแบบ automated test (YOLO จริง, ไม่ mock)
  และรัน process จริงแยกต่างหากยิง HTTP เข้าไป (ดู `docs/local-development.md`) — แต่ยังไม่เคยรันข้าม
  เครื่องจริงบนเครือข่ายจริง (LAN/VPN) มีแต่ localhost
- **สตรีมวิดีโอเป็น MJPEG-over-HTTP** ไม่ใช่ WebRTC — ตัดสินใจไว้ใน `docs/networking.md` (เหตุผลและ
  trade-off), เป็น upgrade path ที่เปิดไว้ไม่ได้ปิดทาง
- **การ deploy ขึ้น Vercel จริง** (`vercel deploy` เข้าบัญชีจริง) เป็นขั้นที่ผู้ใช้ต้องทำเอง — เตรียม build
  script + `vercel.json` + เอกสารไว้ครบแล้วที่ `docs/vercel-deployment.md` แต่ไม่มีใครนอกจากเจ้าของบัญชี
  Vercel กด deploy แทนได้
- ยังไม่มี horizontal scaling (AI Server หลายตัวหลัง load balancer) — ออกแบบไม่ปิดทางไว้
  (`vision.pipeline.FrameSource` เป็น protocol แยกจาก implementation) แต่ยังไม่ได้ implement

---

## การเทรน AI ให้รู้จักสินค้าจริง (Custom Product Training)

สินค้าจริงของผู้จัดบูธแทบไม่มีทางตรงกับ 80 คลาสของ YOLO ที่เทรนจาก COCO — จึงต้องมีวิธี
สอน AI ให้รู้จักสินค้าที่ไม่เคยเห็นมาก่อน **โดยไม่ต้องวาด bounding box เอง และไม่ต้องรอ
เทรนโมเดลใหม่ทุกครั้ง** นี่คือสิ่งที่ **AI Trainer** ทำ:

### เปิดใช้งาน
```bash
# เปิดจากแอปบูธหลัก
python app.py            # แล้วกดปุ่ม "🎓 เทรน AI จดจำสินค้า" ที่แถบด้านบน

# หรือเปิดแยกต่างหาก (ไม่ต้องมีกล้อง ยกเว้นตอนกด "ทดสอบด้วยกล้องสด")
python trainer.py
```

### ขั้นตอน
1. กด **"➕ เพิ่มสินค้าใหม่"** ใส่ชื่อ/แท็กไลน์/รายละเอียดสินค้า (บันทึกลง `products.json`
   ทันที เหมือนสินค้าสาธิตทุกประการ — ใช้ต่อกับ AI Product Assistant, Dashboard,
   Health Monitoring ได้เลยโดยไม่ต้องแก้โค้ด)
2. เลือกสินค้าจากรายการทางซ้าย แล้ว **"📁 อัปโหลดรูปภาพ"** (เลือกได้หลายไฟล์พร้อมกัน) หรือ
   **"🎞️ อัปโหลดวิดีโอ"** (ระบบตัดเฟรมให้อัตโนมัติทุก 0.4 วินาที สูงสุด 60 ภาพต่อวิดีโอ)
3. ระบบจะพยายามหาตำแหน่งสินค้าในภาพให้อัตโนมัติ (`_auto_crop` ใน `ui/trainer_window.py`)
   แล้วแปลงเป็น "ลายนิ้วมือ" ตัวเลข (embedding) เก็บเข้าคลังของสินค้านั้นทันที — **ไม่มีขั้นตอน
   เทรนแยกต่างหาก** อัปโหลดเสร็จ ใช้งานจดจำได้เลย
4. แถบสถานะจะบอกว่าอัปโหลดพอหรือยัง: ⚪ ยังไม่มีข้อมูล → 🟡 มีข้อมูลแล้วแต่ควรเพิ่ม → 🟢
   พร้อมใช้งาน (แนะนำอย่างน้อย **15 ภาพต่อสินค้า จากหลายมุมและสภาพแสงที่ต่างกัน** — ยิ่งอัปโหลด
   หลากหลายมาก ยิ่งแม่นยำขึ้น เพราะระบบเทียบภาพใหม่กับภาพตัวอย่างทั้งหมดที่มี)
5. กด **"▶️ ทดสอบด้วยกล้องสด"** เพื่อดูผลจริงทันทีโดยไม่ต้องออกจากหน้าเทรน แล้วอัปโหลด
   เพิ่มได้เรื่อย ๆ จนกว่าจะแม่นยำพอ — ตรงตามที่โจทย์ต้องการ ("อัปโหลดจนกว่าข้อมูลจะครบถ้วนพอ")

### หลักการทำงาน (สรุปสั้น ๆ)
- ภาพทุกภาพถูกแปลงเป็นเวกเตอร์ตัวเลขด้วยโครงข่ายที่ผ่านการเทรนมาแล้ว (MobileNetV3, ImageNet
  weights, ไม่ปรับน้ำหนักเพิ่ม) — ยิ่งมีตัวอย่างของสินค้านั้นมาก ยิ่งครอบคลุมมุม/แสงต่าง ๆ
- ตอนตรวจจับสด ระบบตัดบริเวณที่ "มีอะไรบางอย่างถูกยกขึ้นมา" ด้วย background subtraction
  (`core/localizer.py` — ไม่ขึ้นกับว่าเป็นวัตถุประเภทไหน ใช้ได้กับสินค้าทุกรูปทรง) แล้วเทียบ
  ลายนิ้วมือของบริเวณนั้นกับคลังของทุกสินค้าที่เทรนไว้ ใครใกล้เคียงที่สุดและทิ้งห่างอันดับสอง
  พอสมควรถึงจะตัดสินว่าใช่ (กันการจับคู่ผิดระหว่างสินค้าที่หน้าตาคล้ายกัน)
- **ไม่เก็บรูปต้นฉบับที่อัปโหลด** เก็บเฉพาะเวกเตอร์ embedding ไว้ที่ `data/gallery/` — ลด
  พื้นที่จัดเก็บและข้อมูลที่ต้องดูแลด้านความเป็นส่วนตัว
- ลบข้อมูลเทรนของสินค้าใดสินค้าหนึ่งได้ทุกเมื่อด้วยปุ่ม "🗑️ ลบข้อมูลเทรนทั้งหมด" หากอัปโหลด
  รูปผิดหรือคุณภาพไม่ดี

### ข้อควรรู้
- ระบบต้องมีข้อมูลรวมกันอย่างน้อย ~4 ภาพในคลังทั้งหมด (ไม่ว่าสินค้าเดียวหรือหลายสินค้า) ก่อน
  จึงจะเริ่มจดจำได้แม่นยำ — ช่วงเริ่มต้นที่มีภาพน้อยมาก ระบบจะยังไม่ฟันธงว่าใช่สินค้าไหน
  (แสดง UNKNOWN ไปก่อน) เพื่อกันการจับคู่ผิดพลาด แล้วจะเริ่มทำงานได้เองเมื่ออัปโหลดเพิ่ม
- ใช้ได้กับสินค้าที่มีรูปทรง/ลวดลายชัดเจน ถ้าสินค้าหน้าตาคล้ายกันมาก (เช่น สินค้าสีเดียวกัน
  ทรงกล่องเหมือนกันทุกกล่อง) แนะนำให้ถ่ายให้เห็นจุดต่าง เช่น โลโก้/ฉลาก เพื่อความแม่นยำ

---

## ส่วนที่ FairFace — เทรนโมเดล Age / Gender / Race และใช้งานกับเว็บแคม

โมดูลแยกต่างหากจากระบบบูธหลักด้านบน สำหรับเทรนโมเดลจำแนก **อายุ / เพศ / เชื้อชาติ**
จาก [FairFace dataset](https://github.com/joojs/fairface) (CC BY 4.0) ตั้งแต่ต้น (ไม่ใช้
โมเดล FairFace สำเร็จรูปมาแทนการเทรน — ใช้ได้แค่ ImageNet pretrained backbone เป็นจุดเริ่มต้น)
แล้วนำโมเดลที่เทรนเสร็จไปรันกับเว็บแคมแบบ real-time ได้ (`detect.py`)

โมเดลที่เทรนจากที่นี่ยังบันทึกไฟล์ `best_model_state_dict.pt` เพิ่มให้อัตโนมัติ ซึ่งใช้แทน
`res34_fair_align_multi_7_20190809.pt` ได้ทันทีกับ `--fairface-checkpoint` ของ `web_server.py`
(ดูส่วน AI Vision ด้านบน) — คือโมเดลเดียวกัน แค่เทรนจากข้อมูลของโปรเจกต์นี้เอง

### 1. ติดตั้ง

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. เตรียม Dataset

วาง FairFace dataset ไว้ที่ `dataset/` (ไม่ hard-code path อื่น) รองรับทั้งโครงสร้าง:
```text
dataset/
├── train_labels.csv          # หรือ dataset/train/fairface_label_train.csv
├── val_labels.csv            # หรือ dataset/val/fairface_label_val.csv
├── train/*.jpg
└── val/*.jpg
```
รูปภาพ + ไฟล์ label CSV (`fairface_label_train.csv` / `fairface_label_val.csv`) ต้องดาวน์โหลด
แยกจาก Google Drive อย่างเป็นทางการ (ลิงก์อยู่ใน README ของ https://github.com/joojs/fairface)
— ตรวจสอบ license/เงื่อนไขการใช้งานก่อนนำไป deploy จริง

### 3. ตรวจสอบ Dataset ก่อนเทรน

```bash
python train.py --check-dataset
```
รายงานจำนวนรูปที่ใช้ได้จริง, class ของแต่ละ task, ไฟล์ที่หายไป โดยไม่ทำให้โปรแกรม crash แม้มี
ไฟล์เสียบางไฟล์

### 4. Train

```bash
python train.py                                    # ใช้ค่า default จาก config.yaml
python train.py --epochs 20 --batch-size 32 --lr 0.0001 --num-workers 4
python train.py --resume models/fairface/last_model.pt   # เทรนต่อจากที่ค้างไว้
```
ใช้ GPU อัตโนมัติถ้ามี (`torch.cuda.is_available()`) และ fallback ไป CPU ถ้าไม่มี/ใช้งานไม่ได้
จริง (บางเครื่อง GPU enumerate ว่ามีแต่ใช้งานจริงไม่ได้ — โค้ดตรวจสอบและ fallback ให้อัตโนมัติ
ไม่ทำให้ training crash) ผลลัพธ์ที่ได้:
```text
models/fairface/
├── best_model.pt              # checkpoint ที่ val accuracy เฉลี่ยดีที่สุด (มี optimizer/epoch/label mapping ครบ, resume ต่อได้)
├── last_model.pt               # checkpoint ของ epoch ล่าสุด (สำหรับ --resume)
├── best_model_state_dict.pt    # เวอร์ชัน state_dict ล้วน ใช้กับ --fairface-checkpoint ของ web_server.py ได้ทันที
├── label_mappings.json
└── training_config.json

logs/
└── fairface_train.csv          # loss/accuracy ทุก epoch
```

ทดสอบ pipeline เร็ว ๆ ก่อนเทรนเต็มด้วย subset ได้:
```bash
python train.py --epochs 1 --max-train-samples 200 --max-val-samples 50
```

#### GPU รุ่นเก่า (Pascal เช่น GTX 10xx) — ต้องใช้ torch build ที่ตรงกัน

`pip install torch torchvision` ธรรมดา (หรือ build cu128 ขึ้นไป) จะไม่รองรับการ์ดตระกูล Pascal
(compute capability sm_60/61/62 เช่น GTX 1050/1060/1070/1080) อีกต่อไป — รันได้แต่จะ error
`CUDA error: no kernel image is available for execution on the device` ตอนใช้งานจริง (ทั้งที่
`torch.cuda.is_available()` ยังขึ้น `True` ก็ตาม) ต้องใช้ torch build ที่ยังคอมไพล์ kernel สำหรับ
sm_6x ไว้ (เช่น tag `cu126`) — ยืนยันแล้วว่าใช้งานได้จริงบน GTX 1050 (forward/backward/optimizer
step บน CUDA จริง ไม่ใช่แค่ `is_available()`):
```bash
python -m venv .venv-fairface-gpu
.venv-fairface-gpu\Scripts\activate        # Windows
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126 torchvision==0.29.0+cu126
pip install opencv-python Pillow tqdm pyyaml numpy
python train.py --max-train-samples 64 --max-val-samples 32 --epochs 1   # smoke test บน GPU จริง
```
`src/train_utils.py`'s `resolve_device_safe()` ตรวจด้วย tensor op จริง (ไม่ใช่แค่
`is_available()`) จึง fallback ไป CPU ให้อัตโนมัติถ้า build ที่ติดตั้งไม่รองรับการ์ดจริง — ไม่ทำให้
training crash แม้ config CUDA ไม่ตรงกับฮาร์ดแวร์

### 5. ทดสอบโมเดลกับรูปเดียว

```bash
python test_model.py --image test.jpg
python test_model.py --image test.jpg --checkpoint models/fairface/last_model.pt
python test_model.py --image dataset/val/1.jpg --skip-face-detection   # รูปที่ครอปหน้าไว้แล้ว
```

### 6. ใช้งานกับเว็บแคม (real-time, รองรับ 2 กล้อง)

```bash
python detect.py                          # กล้อง 0 และ 1
python detect.py --camera1 none           # กล้องเดียว
python detect.py --checkpoint models/fairface/last_model.pt
```
กด `q` ในหน้าต่างวิดีโอ หรือ Ctrl+C ในเทอร์มินัล เพื่อหยุด — จะ release กล้องทุกตัวและปิด
หน้าต่างทั้งหมดก่อนโปรแกรมจบจริง กล้องที่เปิดไม่ได้จะขึ้น "unavailable" โดยไม่ทำให้โปรแกรม
ทั้งหมด crash และพยายามเปิดใหม่เป็นระยะ

### หมายเหตุเรื่อง Face Detector

`test_model.py`/`detect.py` ต้องการตัวตรวจจับใบหน้า (ไม่ใช้ FairFace classifier เป็นตัวตรวจจับ
เอง) — ถ้ามีไฟล์ `models/face_detector/face_detection_yunet_2023mar.onnx` อยู่แล้ว ทั้งสอง
คำสั่งจะใช้ YuNet (DNN) โดยอัตโนมัติ ไม่ต้องระบุ flag ใด ๆ เพิ่ม

ถ้ายังไม่มีไฟล์นั้น ค่าเริ่มต้นจะ fallback ไปใช้ Haar cascade ที่มากับ opencv-python แต่บาง
build (เช่น opencv-python 5.x ที่ใช้ในโปรเจกต์นี้) ไม่มี Haar มาให้เลย ในกรณีนั้นต้องดาวน์โหลด
YuNet มาเอง แล้วระบุ path ตรง ๆ:
```bash
python detect.py --yunet-model path/to/face_detection_yunet_2023mar.onnx
```
ดาวน์โหลดได้จาก https://github.com/opencv/opencv_zoo (LFS asset — ใช้ลิงก์
`https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
ถ้า `git clone` ธรรมดาได้แค่ pointer file)
