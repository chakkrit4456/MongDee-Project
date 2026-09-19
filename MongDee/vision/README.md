# Person Vision Pipeline (`vision/`)

Phase 2-6 ของ pipeline "นับคนไม่ซ้ำข้ามกล้อง" ตาม
[`MongDee_Master_Prompt_Complete-1.md`](../MongDee_Master_Prompt_Complete-1.md)

> **แยกจาก `core/vision.py`** — `core/` คือระบบจดจำ**สินค้า**ของบูธ (เดิม ใช้งานได้จริง ทดสอบครบ)
> `vision/` (โฟลเดอร์นี้) คือ pipeline ใหม่สำหรับ ตรวจจับ/ติดตาม/ระบุตัว**บุคคล**ข้ามกล้อง

## สถานะแต่ละเฟส

| เฟส | สถานะ | โมดูล |
|---|---|---|
| Phase 1 — Camera Gateway | เสร็จ | [`camera/`](../camera/README.md) |
| Phase 2 — Frame Processing + Person Detection | เสร็จ | `vision/frame.py`, `vision/detection/` |
| Phase 3 — Multi-Object Tracking (ByteTrack + Kalman) | เสร็จ | `vision/tracking/` |
| Phase 4 — Person Attributes (สี/รูปร่าง; ที่เหลือ = UNKNOWN จนกว่าจะมีโมเดล) | เสร็จ | `vision/attributes/` |
| Phase 5 — Person Re-ID (embedding + quality gate + track aggregation) | เสร็จ | `vision/reid/` |
| Phase 6 — Multi-Camera Identity (multi-feature weighted match + temporal/spatial) | เสร็จ | `vision/identity/` |
| Phase 7 — Database (SQLite + migrations) | เสร็จ | [`backend/`](../backend/README.md) |
| Phase 8 — REST + WebSocket API | เสร็จ | `backend/api/` |
| Phase 9 — Dashboard | เสร็จ | `backend/api/dashboard.html` |
| Phase 10-12 — Optimization / Testing / Deployment | บางส่วน | config knobs, `scripts/evaluate.py`, `docker/` |

### Sections 48-95 (Complete additions)

| หมวด | สถานะ | โมดูล |
|---|---|---|
| 48-51 Product detection + GI database + classification | เสร็จ | `vision/product/` |
| 52 Person ↔ Product association | เสร็จ | `vision/interest/association.py` |
| 53 Head pose / gaze (UNKNOWN จนกว่าจะมีโมเดล + body-orientation proxy) | เสร็จ | `vision/interest/gaze.py` |
| 54-56 Estimated Customer Interest + look/dwell | เสร็จ | `vision/interest/estimator.py` |
| 57 Product interaction (POSSIBLE/CONFIRMED/UNKNOWN) | เสร็จ | `vision/interest/interaction.py` |
| 58-60, 71-72 Camera calibration (homography), world coords, spatial fusion | เสร็จ | `vision/spatial/` |
| 61-64 Booth Designer (2D drag-drop editor) | เสร็จ | `backend/api/designer.html` |
| 65-66, 73 Multi-booth + layout versioning | เสร็จ | `backend/database/` (`layouts`, `activate_layout`) |
| 67-69 Real-time booth map + movement paths + heatmaps | เสร็จ | `vision/spatial/heatmap.py`, `vision/booth_analytics.py` |
| 70, 80 Spatial database | เสร็จ | `backend/database/schema.py` (migration 2) |
| 74 User roles (ADMIN/OPERATOR/VIEWER) + audit log | เสร็จ | `backend/api/app.py` |
| 75-77 Event / API / Dashboard extensions | เสร็จ | `backend/api/` |
| 79 Customer journey (path + zones visited) | เสร็จ | `MovementPath`, `/api/persons/{id}/movement` |

## โครงสร้าง pipeline

```
CameraGateway ──► PersonDetector ──► MultiCameraTracker ──► TrackFeatureStore ──► GlobalIdentityManager
 latest_frame      YOLO (person)      ByteTrack + Kalman     Re-ID embedding +      multi-feature weighted
 (native res)      → Detection        → local_track_id       attributes ต่อ track    match → MATCH/UNCERTAIN/NEW
                                                             (keyframe sampled)     → Global Person ID + PersonEvent
                         │                                          │
                         └──► ProductVisionModule (vision/product/)  └──► BoothAnalytics (vision/booth_analytics.py)
                              detect → track → classify → GI info         world position (calibration/homography)
                              → KNOWN/POSSIBLE/UNKNOWN + zone              → movement path + traffic/dwell/interest heatmap
                                                                          → person↔zone association → InterestSession
                                                                          → CUSTOMER_INTEREST_STARTED/UPDATED/ENDED
```

`DetectionPipeline` (`vision/pipeline.py`) รัน worker thread เดียว วนทุกกล้อง แชร์ detector/extractor
ชุดเดียว (เหมาะกับ CPU ที่รัน inference ขนานกันไม่ได้จริง) — แต่ละ stage หลัง detection เป็น optional
ProductVisionModule และ BoothAnalytics ประกอบเข้าที่ `backend/main.py` (เปิดผ่าน config `spatial` / `product_pipeline`)

## รันทดสอบจริง

```bash
python scripts/detection_demo.py configs/cameras.example.json --track --show   # detection + tracking
python -m backend.main configs/mongdee.example.json                            # ทั้งระบบ + API + dashboard
```

## รันเทส

```bash
python -m pytest -q            # ~175 เทส (camera + vision + backend) — ใช้ YOLO/model ปลอม
python -m pytest -q -m slow    # + เทสที่โหลด yolo11n.pt / torchvision จริง และรัน inference จริง
```

## Re-ID backend — ข้อควรรู้สำคัญ (accuracy)

`ReIDConfig.backend`:

| backend | ได้อะไร | ข้อจำกัด |
|---|---|---|
| `color` | HSV histogram แบ่งโซน — pure numpy ไม่ต้องมีโมเดล | อ่อนมากต่อการแยก**ตัวบุคคล** ใช้เป็นสัญญาณ "เสื้อผ้า/สี" ประกอบเท่านั้น |
| `torchvision` (ค่า default ของ `auto`) | ResNet18 ImageNet features 512 มิติ | เป็น feature ประเภทวัตถุ ไม่ใช่ identity — คนละคนก็ยังคล้ายกันสูง |
| `osnet` | OSNet ([`vision/reid/osnet.py`](reid/osnet.py)) — โมเดล Re-ID เฉพาะทาง | **ต้องมี weights** (`reid.osnet_weights`) — ไม่ได้ bundle มา ดู `scripts/download_models.py --osnet-url` |

**สำหรับ production จริงต้องใช้ `osnet` + weights ที่เทรนกับ Market-1501/MSMT17** — `color`/`torchvision`
เป็น fallback ที่ซื่อสัตย์ ไม่ใช่ identity discrimination ระดับใช้งานจริง จนกว่าจะมี weights จริง
threshold ใน `IdentityConfig` ตั้งไว้แบบ conservative (เอนไปทาง "ไม่ merge") ตาม master prompt ส่วนที่ 47
และต้อง tune ใหม่กับ footage จริง (ส่วนที่ 13, 37, `scripts/evaluate.py`)

## ข้อจำกัดที่รู้ตัว

- **CPU-only บนเครื่องนี้** — yolo11n ~110-130ms/เฟรม → ต่อกล้อง ~2-4 fps เมื่อมีหลายกล้อง
  (`device=auto` จะใช้ GPU เองถ้ามี แต่ยังไม่เคยทดสอบบน GPU)
- **Re-ID** — ดูด้านบน; ไม่มี weights เฉพาะทาง = ความแม่นในการ merge ข้ามกล้องต่ำ
- **Attributes** — สกัดได้จริงแค่สีเสื้อ/กางเกง + รูปร่างหยาบ ๆ; เพศ/อายุ/ผม/แว่น/หน้ากาก = UNKNOWN
  จนกว่าจะเสียบโมเดล PAR/gender/age (มี hook `GenderAgeBackend` ไว้แล้ว)
- **Face** (ส่วนที่ 8) — ยังไม่ทำ (เป็นข้อมูลเสริม ไม่ใช่ข้อกำหนด)
- **ความแม่นของ YOLO เป็นเรื่องของ YOLO** — เทสตรวจแค่ว่า pipeline ประมวลผล/map พิกัดถูก
  ไม่ได้ประเมิน detection precision/recall (ต้องทำกับชุดข้อมูลจริง — ส่วนที่ 37)
