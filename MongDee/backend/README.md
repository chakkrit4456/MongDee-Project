# MongDee Vision Backend (`backend/`)

Phase 7-9 — database, REST + WebSocket API, dashboard — and the top-level
entrypoint that wires every phase together.

## รันทั้งระบบ

```bash
python -m backend.main configs/mongdee.example.json
```

จะ: โหลด YOLO → ต่อกล้องทุกตัวใน config → รัน pipeline (detect → track → reid → identity)
→ เขียนลง SQLite → เปิด API ที่ `http://127.0.0.1:8100`

เปิดเบราว์เซอร์ที่ `http://127.0.0.1:8100/` เพื่อดู **Dashboard** (Unique People, gender/age estimate,
camera grid, event log, person list — polling + WebSocket)

Ctrl+C ปิดสะอาด (flush track ที่ค้างลง DB ก่อน)

## config

ไฟล์เดียว `configs/mongdee.example.json` — section: `cameras`, `detection`, `tracking`, `reid`,
`attributes`, `identity`, `topology`, `features`, `interest`, `product`, `spatial`,
`product_pipeline`, `database`, `api` (ดู `backend/config.py`)

เปิด spatial layer (world coords + booth map + heatmap + customer interest): ตั้ง
`spatial.enabled = true` + ชี้ `layout_path` (ดู `configs/booth_layout.example.json`) และ
`calibration_path` (ดู `configs/calibration.example.json` — calibrate ด้วยจุดอ้างอิง >= 4 จุดต่อกล้อง)
เปิด product layer: `product_pipeline.enabled = true` + `gi_database_path`
(ดู `configs/gi_products.example.json`) และ `gallery_dir` (`<dir>/<product_id>/*.jpg` เป็นภาพอ้างอิง)

รหัสผ่านกล้อง / API key ใส่เป็น `"${ENV_VAR}"` ให้ดึงจาก environment (ส่วนที่ 34 — ห้าม hard-code):

```bash
export CAM02_PASSWORD='...'  MONGDEE_API_KEY='...'
python -m backend.main configs/mongdee.example.json
```

## หน้าเว็บ

| path | คือ |
|---|---|
| `/` , `/dashboard` | Dashboard — unique count, gender/age estimate, camera grid, event log, person list, live booth map |
| `/designer` | Booth Designer — 2D drag-drop editor: ตั้งขนาดบูธ วางกล้อง/ชั้น/zone, snap-to-grid, undo/redo, save version, import/export |

## API (ส่วนที่ 28, 76)

| endpoint | คืนอะไร |
|---|---|
| `GET /api/health` | สถานะ + schema version (public) |
| `GET /api/cameras` , `/api/cameras/{id}/status` | กล้อง + สถานะ + live count/fps/latency |
| `GET /api/persons` , `/api/persons/{id}` | Global Persons + timeline + tracks |
| `GET /api/persons/{id}/movement` | เส้นทางเดินในพิกัดจริง (2D booth) |
| `GET /api/persons/{id}/interests` | interest events ของคนนั้น |
| `GET /api/events` | Event log (person + interest) |
| `GET /api/stats` | unique count, gender/age breakdown, camera counts |
| `GET /api/booths` , `/api/booths/{id}/layout` | booth + layout ที่ active |
| `POST /api/booths/{id}/layout` | บันทึก layout version ใหม่ (admin) |
| `GET /api/booths/{id}/layout/versions` , `POST .../{v}/activate` | ดู/สลับ version (activate = admin) |
| `GET /api/products` , `/api/products/{id}` | GI products + analytics |
| `GET /api/zones/{id}/analytics` | unique visitors + interest breakdown ของ zone |
| `GET /api/analytics/heatmap?kind=traffic\|dwell\|interest` | heatmap grid (ต้องมี spatial layer) |
| `GET /api/map` | ตำแหน่งคนบน booth map ตอนนี้ + interest sessions |
| `GET /api/audit` | audit log ของ admin actions (admin) |
| `WS /ws/events` , `/ws/dashboard` , `/ws/map` | push real-time |
| `GET /health` | alias ของ `/api/health` แบบ bare path (public) |
| `GET /metrics` | hardware profile + remote camera health + live pipeline FPS/latency ต่อกล้อง (public) |
| `POST /api/cameras/register` | Camera Agent ลงทะเบียนกล้อง (operator+) — ดู `docs/camera-agent.md` |
| `POST /api/cameras/{id}/frame` | Camera Agent push เฟรม JPEG (operator+) |
| `GET /api/cameras/{id}/stream` | MJPEG live feed ต่อกล้อง (viewer+, ใช้ได้ทั้งกล้อง local และ remote) |

**Camera Agent + Cloud/Vercel:** ตอนนี้กล้องมาได้ 2 ทาง — ต่อตรงใน config (`CameraGateway`)
หรือมาจาก **Camera Agent** เครื่องอื่นผ่านเครือข่าย (`camera_agent/`, ไม่ต้องมี torch/ultralytics)
ที่ push เฟรมมาที่ endpoint ข้างบน ทั้งสองทางเข้า pipeline เดียวกันผ่าน
`backend/remote_frames.py`'s `CompositeFrameSource` — ดู `docs/architecture.md`,
`docs/camera-agent.md`, `docs/ai-server.md`

**Adaptive Performance Engine** (`backend/adaptive.py`): ลด AI FPS ก่อน แล้วค่อยลด resolution
เมื่อ CPU/latency สูงจริง (ไม่แตะ display FPS) — config ที่ `adaptive` block ใน
`configs/mongdee.example.json`, รายละเอียดที่ `docs/performance.md`

**Deploy Dashboard บน Vercel:** `docs/vercel-deployment.md` (dashboard เดิมหน้านี้ deploy เป็น
static site แยก origin จาก AI Server ได้ ผ่าน `scripts/build_web_dashboard.py` + `vercel.json`)

**Auth + roles (ส่วนที่ 34, 74):** `api.api_key` = admin key เดี่ยว (shortcut) หรือ `api.keys = {"<key>": "admin|operator|viewer"}`
— reads ต้อง viewer+, แก้ layout ต้อง admin, ลงทะเบียน/push เฟรมกล้องต้อง operator+, admin actions ถูก
audit ทุกครั้ง `/api/health` และหน้า dashboard/designer เป็น public (แต่ data ที่หน้านั้นเรียกยังต้องมี key)
ดู `docs/security.md` สำหรับรายละเอียดเต็ม

**CORS:** ตั้ง `api.cors_origins` เป็น list ของ origin ที่อนุญาต (เช่น URL ของ Vercel dashboard) — ไม่ตั้ง
= เปิดทุก origin (มี warning log, เหมาะกับ dev เท่านั้น)

## Database (ส่วนที่ 22)

SQLite ที่ `data/mongdee_vision.db` (แยกไฟล์จาก `data/mongdee.db` ของระบบสินค้าเดิม — ส่วนที่ 87
modular, ล้มแยกกัน) ตาราง: `cameras`, `global_persons`, `camera_tracks`, `detections`,
`reid_embeddings`, `events`, `camera_transitions`, `system_logs`

Migration แบบมี version (`backend/database/schema.py` → `MIGRATIONS`) — ห้ามแก้ migration ที่ ship
ไปแล้ว ให้เพิ่มอันใหม่ WAL mode เปิดอยู่ (reader ไม่บล็อก writer)

`detections` log ปิดโดย default (`database.log_detections`) — เปิดแล้วเขียนแบบ sample
(1 แถว/กล้อง/วินาที) กัน DB บวม

## ประเมินผล (ส่วนที่ 37)

```bash
python scripts/evaluate.py annotations.json
```

คำนวณ detection precision/recall, tracking MOTA/ID-switch, counting error, cross-camera
false/missed match rate จากไฟล์ annotation ที่ label จาก footage จริง (ดู format ใน `scripts/evaluate.py`)
**อย่ารายงานตัวเลขเหล่านี้เป็นความแม่นยำสากลของระบบ** — เป็นผลบน test set ของคุณเท่านั้น

## Deploy

```bash
docker compose -f docker/docker-compose.yml up --build
```

รุ่น CPU (image ~2GB) — GPU ดูคอมเมนต์ใน `docker/Dockerfile` และ `docker-compose.yml`
