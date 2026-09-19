# Camera Gateway (Phase 1)

ส่วนแรกของ pipeline "นับคนไม่ซ้ำข้ามกล้อง" ตามที่ระบุใน
[`MongDee_Master_Prompt.md`](../MongDee_Master_Prompt.md) (ส่วนที่ 2-4, 31-33, 46) — โมดูลนี้ทำหน้าที่
**เชื่อมต่อกล้องเท่านั้น**: เปิดสตรีม, ถอดรหัสเฟรม, reconnect อัตโนมัติเมื่อหลุด, และรายงานสถานะ
online/offline ยังไม่มี AI ใด ๆ อยู่ในนี้ — detection/tracking/re-id เป็น layer ถัดไปที่ยังไม่ได้สร้าง
(ตาม diagram ในส่วนที่ 2 ของ master prompt: Camera Layer → **Camera Gateway** → Frame Processing →
Object Detection → ...)

**ขนาดที่ออกแบบไว้**: เครื่องพัฒนาโปรเจกต์นี้ไม่มี GPU (CPU-only, ตรวจสอบแล้วด้วย
`torch.cuda.is_available() == False`) และแอปหลัก (`core/`, `app.py`, `web_server.py`) เป็นระบบสำหรับ
บูธเดียว/ไม่กี่บูธ ไม่ใช่ CCTV องค์กรขนาดใหญ่ — โมดูลนี้จึงถูกปรับให้เหมาะกับกล้องจำนวนไม่มาก
(เว็บแคม + กล้อง IP ไม่กี่ตัวรอบบูธ), ไม่ผูกกับ GPU/TensorRT, ใช้ SQLite ต่อจาก `core/database.py`
เดิมได้ในเฟสถัดไป (ไม่ใช่ PostgreSQL/Redis ตามที่ master prompt เขียนไว้แบบ enterprise-scale)

## ติดตั้ง

```bash
pip install requests pytest   # เพิ่มจาก requirements.txt เดิม — ดู requirements.txt
```

(`opencv-python` ที่มีอยู่แล้วในโปรเจกต์มาพร้อม FFmpeg แบบ prebuilt แล้ว ตรวจสอบด้วย
`python -c "import cv2; print(cv2.getBuildInformation())"` — บรรทัด `FFMPEG: YES (prebuilt binaries)` —
จึงรองรับ RTSP/HLS ได้ทันทีแม้เครื่องไม่มี `ffmpeg` binary ติดตั้งแยก)

## รันทดสอบจริง (demo)

```bash
python scripts/camera_gateway_demo.py configs/cameras.example.json
python scripts/camera_gateway_demo.py configs/cameras.example.json --show       # เปิดหน้าต่างพรีวิวสด
python scripts/camera_gateway_demo.py configs/cameras.example.json --seconds 15 # หยุดเองหลัง 15 วินาที
```

คัดลอก `configs/cameras.example.json` แล้วแก้ให้ตรงกับกล้องจริงของคุณ (ดูรูปแบบ URL ด้านล่าง)
สคริปต์จะพิมพ์การเปลี่ยนสถานะ (connecting/online/offline) และ FPS ที่ได้รับจริงของแต่ละกล้องทุกวินาที

## รันเทส

```bash
python -m pytest tests/camera -q
```

59 เทสทั้งหมดรันจริง ไม่มี mock ที่ปลอมง่ายเกินไป — `test_http_camera.py` และ `test_onvif.py`
เปิดเซิร์ฟเวอร์ HTTP/SOAP จริงบน localhost (ผ่าน `http.server`) แล้วยิง request จริงผ่าน `requests`
ไปหามัน เพื่อพิสูจน์ว่า parser (multipart MJPEG, ONVIF SOAP + WS-Security digest) ทำงานถูกต้องกับ
bytes จริงบน the wire ไม่ใช่แค่ "ไม่ error" กับ mock ที่เขียนขึ้นมาเอง (`test_onvif.py` ถึงขั้นให้
เซิร์ฟเวอร์จำลองคำนวณ WS-Security digest กลับมาตรวจสอบเองว่าตรงกับที่ client ส่งมาหรือไม่ — พิสูจน์ว่า
สูตร digest ถูกต้องตามสเปกจริง ไม่ใช่แค่รูปแบบ XML ที่หน้าตาคล้าย)

## รูปแบบ config (`configs/cameras.example.json`)

```json
{
  "cameras": [
    { "id": "CAM01", "protocol": "usb", "device_index": 0, "processing_fps": 10 },
    { "id": "CAM02", "protocol": "rtsp", "url": "rtsp://user:pass@192.168.1.64:554/ch1", "processing_fps": 8 },
    { "id": "CAM03", "protocol": "onvif", "host": "192.168.1.65", "username": "admin", "password": "secret" },
    { "id": "CAM04", "protocol": "http", "url": "http://192.168.1.66/video", "http_mode": "mjpeg" },
    { "id": "CAM05", "protocol": "hls", "url": "https://example.com/stream/index.m3u8", "enabled": false }
  ]
}
```

ฟิลด์ทั้งหมด (ดู `CameraConfig` ใน `camera/base.py` สำหรับ default ของแต่ละตัว):

| ฟิลด์ | ใช้กับ | ความหมาย |
|---|---|---|
| `id`, `name`, `location`, `enabled` | ทุก protocol | ตัวระบุ, ชื่อแสดงผล, ตำแหน่งติดตั้ง, เปิด/ปิดโดยไม่ต้องลบออกจาก config |
| `device_index` | usb | ลำดับกล้องตามที่ OS/OpenCV เห็น (0 = ตัวแรก) |
| `url`, `username`, `password` | rtsp, http, hls, (onvif เป็น override) | ที่อยู่สตรีมและ credential |
| `host`, `port`, `onvif_path`, `profile_token` | onvif | ที่อยู่ ONVIF device service (ข้ามได้ถ้าใส่ `url` ตรง ๆ) |
| `http_mode` | http | `"mjpeg"` (สตรีมต่อเนื่อง) หรือ `"snapshot"` (poll ภาพนิ่ง) |
| `request_width`/`request_height` | usb | ขอ resolution จากกล้อง (0 = ตามค่า native) |
| `processing_fps` | ทุก protocol | อัตราสูงสุดที่ gateway จะส่งเฟรมต่อไปให้ผู้ใช้งาน (ไม่ผูกกับ FPS จริงของกล้อง) |
| `connect_timeout_sec`, `read_timeout_sec` | ทุก protocol | timeout ต่อการเชื่อมต่อ/อ่านหนึ่งครั้ง |
| `hw_accel` | rtsp, hls | ถอดรหัสวิดีโอ (H.264/H.265) ด้วย GPU ของเครื่องแทน CPU เมื่อทำได้ — ผ่าน API ระดับ OS
  (D3D11VA/DXVA2 บน Windows, VAAPI/VDPAU/CUDA บน Linux) จึงใช้ได้กับ GPU ทุกยี่ห้อ (NVIDIA, AMD, Intel)
  โดยไม่ต้องตั้งค่ายี่ห้อ, เปิดอัตโนมัติ (`true` เป็นค่า default), ถ้าเครื่อง/build ไม่รองรับจะ fallback
  กลับไปถอดรหัสด้วย CPU เหมือนเดิมโดยไม่ error — ปิดได้ด้วย `"hw_accel": false` ถ้ากล้อง/ไดรเวอร์บางรุ่นมีปัญหา |
| `fail_threshold`, `backoff_initial_sec`, `backoff_max_sec`, `backoff_multiplier` | ทุก protocol | นโยบาย reconnect (ส่วนที่ 32 ของ master prompt) |

ฟิลด์ที่ไม่รู้จักใน config จะทำให้โหลดไม่ผ่านทันที (กัน typo เงียบ ๆ)

## รูปแบบ URL ต่อ protocol

- **usb** — ไม่มี URL ใช้ `device_index` แทน (0, 1, 2, ... ตามลำดับที่ระบบเห็นกล้อง —
  บน Windows ถ้ามีเว็บแคมหลายตัวลองไล่เลขดูได้ตรง ๆ ด้วย `python scripts/camera_gateway_demo.py`)
- **rtsp** — `rtsp://[user:pass@]host[:port]/path` เช่น
  `rtsp://admin:secret@192.168.1.64:554/Streaming/Channels/101` (รูปแบบ path จริงแตกต่างกันไปตามยี่ห้อ
  กล้อง — ดูคู่มือกล้องนั้น ๆ)
- **onvif** — ใส่ `host`/`port`/`username`/`password` ของ ONVIF device service (ปกติ port 80 หรือ 8000)
  โมดูลนี้จะเรียก `GetCapabilities` → `GetProfiles` → `GetStreamUri` ให้อัตโนมัติเพื่อหา URL ของ RTSP จริง
  (ไม่ต้องรู้ path ล่วงหน้า) หรือใส่ `url` ตรง ๆ เพื่อข้ามขั้นตอนนี้ทั้งหมด
- **http** — mode `mjpeg`: URL ของ multipart push stream เช่น `http://192.168.1.66/video`;
  mode `snapshot`: URL ของภาพนิ่งที่ poll ซ้ำได้ เช่น `http://192.168.1.66/snapshot.jpg`
- **hls** — URL ของไฟล์ `.m3u8` (master หรือ media playlist) เช่น
  `https://example.com/stream/index.m3u8`

## สิ่งที่ยังไม่ทำ / ข้อจำกัดที่รู้ตัว

- **ONVIF ยังไม่เคยทดสอบกับกล้องจริง** — ไม่มีกล้อง ONVIF ต่ออยู่กับเครื่องพัฒนานี้ ตัว SOAP/WS-Security
  client ถูกเขียนตามสเปก ONVIF core spec จริงและมีเทสยืนยัน protocol correctness กับเซิร์ฟเวอร์จำลอง
  (รวมถึงตรวจ digest ถูกต้องแบบ end-to-end) แต่ integration กับฮาร์ดแวร์จริงยังไม่เคยตรวจ — ยี่ห้อกล้อง
  บางรุ่นอาจมี quirk นอกสเปกที่ต้องปรับเพิ่ม
- **WS-Discovery (`camera.onvif.discover()`) เป็น best-effort** — ขึ้นกับว่าเครือข่ายอนุญาต UDP multicast
  หรือไม่ (หลายเครือข่าย/VLAN บล็อก) ทางที่แน่นอนกว่าคือใส่ `host`/`port` ตรง ๆ ใน config
- **การปิดกล้องกลางคันขณะกำลัง connect อาจช้า** — `cv2.VideoCapture.open()`/`requests` เป็น blocking call
  ที่ Python ขัดจังหวะจากภายนอกไม่ได้ ถ้าเรียก `stop()` ขณะกำลังพยายามเชื่อมต่อกล้องที่ไม่ตอบสนอง อาจต้องรอ
  จนครบ `connect_timeout_sec` ก่อน thread จะจบจริง (ระบบ join ด้วย timeout ที่ครอบคลุมกรณีนี้แล้ว และจะ
  log คำเตือนถ้า thread ค้างนานผิดปกติ แต่ไม่มีทางบังคับ kill mid-call ได้จริงจาก Python)
- **HLS ยังไม่ได้ทดสอบกับ live stream จริง** — ยืนยันแค่ว่า `cv2`'s FFmpeg backend มีของอยู่ (build info)
  และ error path (open ไม่สำเร็จ) ทำงานถูกต้อง ยังไม่มี HLS server จริงให้ทดสอบ end-to-end ในสภาพแวดล้อมนี้
- **ยังไม่ resize/normalize เฟรม** — ตั้งใจ: ตาม architecture diagram ในส่วนที่ 2 ของ master prompt
  "Frame Normalization" เป็น layer ถัดไป ไม่ใช่หน้าที่ของ Camera Gateway — เฟรมที่ได้จาก
  `CameraGateway.latest_frame()` เป็น native resolution เสมอ

## Phase ถัดไป

- **Phase 2 — Frame Processing + Person Detection: เสร็จแล้ว** → [`vision/README.md`](../vision/README.md)
  (รับเฟรมจาก `CameraGateway` → ย่อขนาด → YOLO ตรวจจับเฉพาะ "คน" ด้วย instance แยกจาก
  `core/vision.py` เพื่อไม่กระทบการตรวจจับสินค้าเดิม)
- Phase 3 — Multi-Object Tracking (ByteTrack/BoT-SORT): ยังไม่เริ่ม
