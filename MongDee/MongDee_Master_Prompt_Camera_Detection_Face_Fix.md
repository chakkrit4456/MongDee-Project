# MongDee Master Prompt — เสถียรภาพกล้อง USB · ความแม่นยำ Detect/Track · Face Detection · แก้ Ghost Product

> **วิธีใช้:** เปิด AI coding agent (Claude Code หรือเทียบเท่า) ที่ root ของโปรเจกต์ `D:\mongdee2\MongDee`
> บนเครื่อง Windows เครื่องเดียวกับที่เสียบกล้อง USB อยู่ แล้ววางไฟล์นี้ทั้งไฟล์เป็นคำสั่งเดียว
> Agent ต้องทำต่อเนื่องจนจบทุก Phase โดย **ไม่ถามผู้ใช้กลางทาง และไม่ขอให้ผู้ใช้ไปยืน/ขยับ/เสียบถอดกล้อง**
> (ดู §3 กฎข้อ R2) ถ้าติดจริงให้บันทึกเป็น BLOCKED พร้อมหลักฐานแล้วทำส่วนอื่นต่อ

---

## 1. ภารกิจ

คุณคือวิศวกรอาวุโสด้าน computer vision / Windows camera stack ที่รับช่วงต่อโปรเจกต์ MongDee AI Booth OS
(FastAPI + ultralytics YOLO11 + FairFace + Re-ID, เอกสารหลัก: `README.md`, `GUIDE.md`,
`docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md`, `docs/design/camera-pipeline-root-cause-repair.md`,
`third_party/MODEL_LICENSES.md`, `FAIRFACE_TRAINING_REPORT.md`)

แก้ 6 อาการต่อไปนี้ให้เสร็จสมบูรณ์ **และพิสูจน์ด้วยการทดสอบอัตโนมัติที่คุณรันเอง**:

| # | อาการที่ผู้ใช้รายงาน |
|---|---|
| S1 | กล้อง USB หลายตัวสตรีมพร้อมกันไม่นิ่ง (ตัวที่สองไม่เปิด / หลุดกลางทาง) |
| S2 | ขยับกล้องเร็วๆ แล้วภาพพัง (torn / banding / ดำ / ค้าง) |
| S3 | หลังภาพพัง บางครั้งเชื่อมต่อกล้องไม่ได้อีกเลย (ต้องรีสตาร์ททั้งแอป) |
| S4 | Detect/Track คนไม่แม่น — โดยเฉพาะเพศ (ผู้ชายถูกจำแนกเป็นผู้หญิง) และ track ID เปลี่ยนบ่อย |
| S5 | Detect สินค้าที่ลบออกจากฐานข้อมูลแล้ว และตรวจพบทั้งที่ไม่มีสินค้าอยู่หน้ากล้อง (ghost detection) |
| S6 | ต้องมี **Face Detection** เป็นโมดูลจริง (ตอนนี้มีแค่ YuNet ซ่อนอยู่ในเส้นทาง attribute) |

**Definition of Done:** ทุก Gate ใน §7 เป็น PASS หรือ BLOCKED-with-evidence, `pytest` ผ่านทั้งหมด,
มีรายงาน `docs/CAMERA_AI_FIX_REPORT.md` ตามรูปแบบ §8, และข้อมูลจริงของผู้ใช้ไม่ถูกแตะ

---

## 2. ข้อเท็จจริงที่ตรวจพบแล้วจากการอ่านโปรเจกต์

> ข้อมูลด้านล่างมาจากการอ่านโค้ด เอกสาร log และไฟล์ข้อมูลของโปรเจกต์นี้ — **ยังไม่มีการรันของจริง**
> ป้าย **[CONFIRMED-CODE]** = ยืนยันจากโค้ด/ไฟล์ข้อมูลที่เห็นตรงๆ, **[HYPOTHESIS]** = สมมติฐานที่ต้องพิสูจน์ด้วยการวัด
> ห้ามนำสมมติฐานไปเขียนในรายงานว่าเป็นข้อเท็จจริงจนกว่าจะมีตัวเลขรองรับ

### 2.1 สภาพแวดล้อม (จากเอกสาร/venv ที่มีอยู่)
Windows 10 · Python 3.14.7 (`.venv`) · opencv-python 5.0.0.93 · torch 2.14.0+cu126 (ต้อง build cu126 เท่านั้น เพราะ GPU เป็น
GTX 1050 Pascal sm_61 3 GB) · ultralytics 8.4.153 · RAM ~16 GB · 8 logical cores ·
กล้อง: index0 "HD WebCam" (built-in, ถูก exclude), index1 และ index2 "USB Camera" **รุ่นเดียวกัน VID_4C4A PID_4A55 ไม่มี serial**
(แยกกันด้วย USB port location เท่านั้น), index3 "OBS Virtual Camera" · Baseline ทดสอบ: `613 passed, 7 deselected`

### 2.2 กล้อง (S1–S3)
- **[CONFIRMED-CODE]** `core/vision.py::CameraWorker.run()` เรียก `cap.read()` แบบ blocking **ไม่มี timeout/watchdog** —
  ถ้า read ค้าง (เคยเห็น MSMF ค้าง 180+ วินาที) สถานะยังเป็น online และภาพเก่าค้างจอ
- **[CONFIRMED-CODE]** ทุกกล้องอยู่ใน process เดียวกัน ใช้ `DIRECTSHOW_LOCK` ร่วมกัน ถ้า DirectShow handle เสียสภาพ
  ไม่มีทางรีเซ็ตได้นอกจากรีสตาร์ททั้งแอป (ตรงกับ S3) รายงานเดิมระบุ "process isolation — BLOCKED ยังไม่เคยทดลอง"
- **[CONFIRMED-CODE]** `FAIL_THRESHOLD=20` ครั้งติดกัน (รวมเฟรมที่ `_looks_like_noise` ปฏิเสธ) → ปล่อย capture → reopen ด้วย backoff 3→30 วินาที
  รายงาน O.4 วัดได้ว่าหลัง open/release ถี่ๆ อุปกรณ์เปิดไม่ได้ ~15–30 วินาที
- **[CONFIRMED-CODE]** `discover_cameras()` เปิดสุ่มทีละ index 0..15 ทุกครั้งที่สตาร์ท และ `_hotplug_loop` สแกนทุก 5 วินาที
  พร้อมกับ `_attempt_reopen()` ของแต่ละกล้อง → เปิด/ปิดอุปกรณ์ซ้ำซ้อน (รายงานข้อ I.3, I.4 ระบุแล้วแต่ยังไม่แก้)
- **[CONFIRMED-CODE]** ตรวจภาพเสียด้วย pixel heuristic เท่านั้น (`_looks_like_noise`: roughness + cross-channel correlation) เพราะ
  per-row banding check เคยถูกถอดออกด้วย false positive → เฟรม torn บางแบบ (แถบเทา/ดำ/แถบซ้ำ) อาจหลุดเข้าจอ ยังไม่มี freeze detection
  และไม่มีการวัด "ความมืดจนใช้ไม่ได้" (รายงาน §K: กล้องส่งภาพดำ 1 fps แต่สถานะ online)
- **[CONFIRMED-DATA]** log ใน `.hw_validation/*` มีรหัสข้อผิดพลาด: `-1072875772` (0xC00D3704 MF_E_HW_MFT_FAILED_START_STREAMING) ~2,600 ครั้ง,
  `-2147024865` (0x8007001F ERROR_GEN_FAILURE) ~195 ครั้ง, `-2147023901` (0x800703E3 OPERATION_ABORTED), `-2147024891` (0x80070005 ACCESS_DENIED)
  — สองตัวหลังเป็นลายเซ็นของ USB glitch/รีเซ็ตอุปกรณ์ ซึ่งตรงกับอาการ "ขยับกล้องแล้วพัง" (ต้องตรวจซ้ำตาม timestamp ว่ามาจากช่วง MSMF ก่อนตัด MSMF ออกหรือไม่)
- **[CONFIRMED-DATA]** รายงานเดิม: กล้องที่สองเปิดไม่ได้ตอนใช้พร้อมกัน แม้ที่ YUY2 320x240@15 (ราว 18 Mbps) → คำอธิบาย "แบนด์วิดท์ USB ไม่พอ" **ไม่ครบ**
  อาจเป็นการจอง isochronous bandwidth ตาม alt-setting ของกล้องราคาถูก, ปัญหา driver/instance ของกล้องรุ่นเดียวกันสองตัว, หรือ power/hub — **[HYPOTHESIS]** ต้องทดลองตาม §5.2
- **[HYPOTHESIS]** รายงานระบุว่า "MJPG ไม่เคยถูกยอมรับ กล้องส่ง YUY2 เสมอ" — อาจเป็นจริง (กล้องไม่มี MJPG ที่ขนาดนั้น) หรืออาจเป็น artifact ของการอ่านกลับ
  `CAP_PROP_FOURCC` บน DSHOW และลำดับการ set (FOURCC ก่อนขนาด) — ตรวจด้วยการ enumerate format จริงผ่าน `pygrabber` (มีอยู่ใน requirements แล้ว)
- **[HYPOTHESIS]** MSMF ถูกตัดทิ้งเพราะล้มเหลว 0/14 แต่ยังไม่เคยทดลองตั้ง `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS` (ต้องตั้งก่อน `import cv2`;
  OpenCV รุ่นใหม่ปิด hardware transforms เป็นค่าเริ่มต้นอยู่แล้ว จึงต้องวัดจริงทั้ง `=0` และ `=1` ไม่ใช่เดาว่าแก้ได้) และยังไม่เคยลอง FFmpeg/dshow เป็น backend ทางเลือก

### 2.3 คน / เพศ / อายุ / Tracking (S4)
- **[CONFIRMED-CODE]** กรอบเพศบนจอมาจาก `CameraWorker._classify_person()` → `FairFaceBackend` (`core/attributes.py`) ทำงานบน **person crop ทั้งตัว**
  แล้วให้ YuNet หาใบหน้าในนั้น กล้องถูกบังคับที่ **320×240** (low-bandwidth profile) ใบหน้าจึงเล็ก; มี `ATTRIBUTE_MIN_FACE_SIZE_PX=64`
  (บันทึกไว้ว่าเคยเจอ "FEMALE 69%" ผิด จาก face 58×68 px ที่ถูกขยาย ~4×)
- **[CONFIRMED-CODE]** `core/attributes.py::crop_face` ตัดกรอบ YuNet **แบบไม่มี margin** ขณะที่ `src/face_detector.py::crop_face` (ที่ `detect.py`/`test_model.py` ใช้) เผื่อ margin 0.2
  → preprocessing ไม่ตรงกันระหว่างเครื่องมือออฟไลน์กับ pipeline จริง และไม่มีหลักฐานว่าตรงกับลักษณะภาพที่ใช้เทรน **[HYPOTHESIS]** ทำให้เพศเพี้ยน — ต้องวัด
- **[CONFIRMED-CODE]** `_classify_person` ให้ "อายุชนะเพศ": ถ้า age ∈ {0-2, 3-9, 10-19} และ conf ≥ 0.6 จะเป็น CHILD ทันที (โมเดลอายุแม่นแค่ ~58% ในรายงานเทรน)
  ส่วนเพศใช้เกณฑ์ conf ≥ 0.6 เท่านั้น (ต่ำมากสำหรับ binary classifier)
- **[CONFIRMED-CODE]** ทั้งกรอบบนจอและ `AttributeAnalyzer` รัน FairFace ซ้ำซ้อน (ทุก person box ทุก AI pass + อีกเส้นทางแบบ throttle ต่อ Global Person) เปลืองงบ GTX 1050
- **[CONFIRMED-DATA]** โมเดลเทรนเอง 20 epoch (best = epoch 6, gender 95.18% บน FairFace val ซึ่งเป็นภาพ in-distribution) `build_transforms` มีแค่ flip/หมุน 10°/ColorJitter อ่อนๆ
  → **ไม่มี augmentation แบบ blur / ความละเอียดต่ำ / JPEG / noise / แสงน้อย / มุมข้าง** เลยไม่มีหลักฐานว่าใช้ได้กับ webcam 320×240 (domain gap)
- **[CONFIRMED-CODE]** `core/tracker.py::PersonTracker` เป็น greedy IoU (MIN_MATCH_IOU=0.25) ไม่มี motion model/Kalman อายุ track 1.5 วินาที
  และอัปเดตเฉพาะตอน AI pass (ความถี่ต่ำและไม่สม่ำเสมอเพราะใช้ AIWorker ตัวเดียวร่วมทุกกล้อง) → ID switch ง่าย
  ใน repo มี ByteTrack+Kalman พร้อม test อยู่แล้วที่ `vision/tracking/` แต่เป็น pipeline ขนานที่ runtime (`web_server.py`) ไม่ได้ใช้
- **[CONFIRMED-CODE]** detector = `yolo11n.pt` (nano) `conf=0.45` `imgsz` 640 (GPU) / 320 (CPU)

### 2.4 สินค้า Ghost (S5)
- **[CONFIRMED-DATA]** ตอนนี้ `data/gallery/manifest.json` มี `product-4621553d` (30 samples, ไฟล์ .npy อยู่) แต่ `products.json` มีสินค้า 3 ตัว
  (`product-4b11ff33`, `product-227cc848`, `product-15bda15f`) **ไม่มีตัวไหนเป็น `product-4621553d`** และทั้ง 3 ตัวไม่มี gallery เลย — คือ orphan จริงที่ระบบยังจดจำอยู่
- **[CONFIRMED-CODE]** `ProductRecognizer._load()` โหลดทุก key ใน manifest โดยไม่เทียบกับ catalog
- **[CONFIRMED-CODE]** `web/server.py::api_delete_product` เรียกแค่ `catalog.remove_product` + `recognizer.clear_product` — **ไม่ยกเลิกงาน import ที่กำลังรัน**
  `start_image_import/start_video_import` (thread เบื้องหลัง) จะ `recognizer.add_sample(key, …)` ต่อ → สร้าง gallery ของสินค้าที่ลบแล้วกลับมาใหม่ และ `add_sample/clear_product` ไม่มี lock ป้องกัน race
- **[CONFIRMED-CODE]** เส้นทางหลังตรวจพบไม่เช็ก catalog: `_on_detections` → `interest_tracker.update` → `_on_interest_confirmed` เขียนแถวลง DB โดยใช้ `class_name` เป็นชื่อสินค้าถ้าไม่พบใน catalog
  (`product["name"] if product else class_name`), `get_camera_snapshot.products_detected` นับทุก detection; มีเพียง `_handle_product_recognized` ที่เช็ก
- **[CONFIRMED-CODE]** `CameraWorker._legacy_coco_classes/_class_indices` คำนวณครั้งเดียวตอนสร้าง worker จาก `catalog.product_keys()` (`BoothManager._make_worker`) →
  ลบ/เพิ่มสินค้าที่ key ตรงชื่อ COCO (`cup`, `mouse`, `bottle`) ไม่มีผลจนกว่าจะรีสตาร์ท
- **[HYPOTHESIS ที่น่าจะเป็นสาเหตุหลักของ "ไม่มีสินค้าในกล้องก็ยังเจอ"]** `ForegroundProposer` (MOG2 history 400, เรียกเฉพาะทุก 2 AI pass และ **ไม่เคยเรียก `update()`**)
  เสนอ blob ใดๆ ที่ใหญ่ 3–75% ของเฟรม — คนเดินผ่าน มือ แสง/auto-exposure เปลี่ยน **หรือการสั่นของกล้อง (S2)** — แล้ว `identify()` ใช้ MobileNetV3-Small (ImageNet features)
  ที่ floor 0.45 (สินค้าเดียว 0.65) margin 0.10 ไม่มี background/negative class ไม่มีการ calibrate แบบ open-set
  และ `training.auto_crop` **fallback เป็น "ทั้งเฟรม"** เมื่อ YOLO ไม่พบวัตถุ → gallery อาจเก็บ "ห้อง/พื้นหลัง" แทนตัวสินค้า (30 samples ของวิดีโอเดียว ≈ ภาพเกือบซ้ำกัน) ทำให้ฉากว่างก็ match ได้

### 2.5 Face Detection (S6)
มี `models/face_detector/face_detection_yunet_2023mar.onnx` และ `core/attributes.py::YuNetFaceDetector` แต่ใช้ภายใน person crop เท่านั้น (min 64 px) — **ไม่มี face detection ทั้งเฟรม,
ไม่มีกรอบใบหน้าบนจอ/API, ไม่มี face track, ไม่มี best-shot selection**

---

## 3. กฎเหล็ก

- **R1 หลักฐานก่อนคำกล่าว** ทุกข้อสรุปต้องมีป้าย CONFIRMED / HYPOTHESIS / BLOCKED และไฟล์หลักฐานใน `reports/camera-ai-fix/` (log, CSV, ภาพ, JSON) —
  รายงาน PASS ได้เมื่อมีตัวเลขที่วัดเองเท่านั้น ห้ามอ้างว่าแก้ฮาร์ดแวร์ได้ถ้าไม่ได้พิสูจน์
- **R2 ไม่มีมนุษย์ใน loop** ห้ามขอให้ผู้ใช้ยืนหน้ากล้อง ขยับกล้อง ถอด/เสียบสาย หรือรันคำสั่งแทน ใช้: (ก) fault-injection + วิดีโอ/ภาพสังเคราะห์, (ข) กล้องจริงที่เสียบอยู่สำหรับ soak test อัตโนมัติ,
  (ค) จำลองถอด/เสียบด้วย `pnputil /disable-device` + `/enable-device` **เฉพาะ InstanceId ของ "USB Camera" VID_4C4A&PID_4A55 ที่ตรวจซ้ำแล้ว**, ห้ามแตะอุปกรณ์อื่น (เมาส์/คีย์บอร์ด/hub),
  ต้อง re-enable ใน `finally` เสมอ, ต้องมีสิทธิ์ admin — ถ้าไม่มีให้ข้ามเป็น BLOCKED ห้ามเปลี่ยนการตั้งค่าระบบ/พลังงาน/USB ของ Windows (แนะนำในรายงานเท่านั้น)
- **R3 ปกป้องข้อมูลผู้ใช้** ก่อนเริ่มสำรอง `products.json`, `data/mongdee.db*`, `data/gallery/`, `data/booth_settings.json`, `models/` ไปที่ `backups/<timestamp>/` (gitignore)
  การทดสอบ/E2E ทุกชนิดต้องใช้สำเนาชั่วคราว (`--db`, `--products` มีอยู่แล้ว; gallery **ยังไม่มีตัวเลือก** → ต้องเพิ่ม `MONGDEE_GALLERY_DIR`/พารามิเตอร์ก่อนรัน E2E ใดๆ)
  ก่อน–หลังการทดสอบให้เทียบ SHA256 ของไฟล์จริงต้องเท่าเดิม ห้ามลบ orphan gallery ทิ้ง ให้ย้ายไป quarantine `data/gallery/_orphaned/`
- **R4 Git** สร้าง branch `fix/camera-ai-accuracy` commit ทีละชุดเล็กๆ ไม่ push ตอนนี้ working tree มีไฟล์ ~76 ไฟล์ที่ต่างแค่ CRLF/LF (insertions = deletions) →
  ใช้ `git diff --ignore-cr-at-eol --stat` ดูการเปลี่ยนจริง, `git add <path>` เป็นรายไฟล์, **ห้าม** normalize line ending ทั้ง repo, ห้าม commit โมเดล/dataset/.venv/.exe/ข้อมูลผู้ใช้/ฟุตเทจ
- **R5 ไม่ทำของเดิมพัง** `pytest` ต้องผ่านทั้งหมด (baseline 613) + test ใหม่ของทุกการแก้ไข, ไม่มี thread-exception warning ใหม่, API/JSON ของเดิมเข้ากันได้ย้อนหลัง (เพิ่ม field ได้ ห้ามเปลี่ยนความหมาย),
  ข้อความ UI ภาษาไทยคงเดิม, การเปลี่ยนพฤติกรรมสำคัญต้องอยู่หลัง config flag พร้อมค่า default ที่ผ่านการวัดแล้ว
- **R6 License gate** ก่อนใช้ dependency/โมเดล/ข้อมูลใหม่ทุกตัว ดึง LICENSE จริงจาก GitHub API/README (ไม่เดาจากความจำ) แล้วต่อท้าย `third_party/MODEL_LICENSES.md`
  ห้ามใช้ของที่ non-commercial (ดู §4.3) และ **ต้องแจ้งผู้ใช้ในรายงาน**ว่า ultralytics เป็น AGPL-3.0 (มี Enterprise License) — MongDee แจกจ่ายเป็น .exe จึงควรให้ผู้ใช้ตัดสินใจเรื่องนี้ (ไม่ใช่งานที่ agent ตัดสินแทน)
- **R7 วัดก่อนแก้ วัดหลังแก้** เก็บ baseline ของทุกตัวชี้วัดก่อน การเปลี่ยนที่ไม่ทำให้ตัวชี้วัดดีขึ้นอย่างมีนัยสำคัญให้ revert (ตามหลัก "accepted only if measurable improvement" ของ `MODEL_LICENSES.md`)
- **R8 ความเข้ากันได้** ทดลองติดตั้งแพ็กเกจใหม่ใน venv แยก (`python -m venv .venv-eval`) ก่อนแตะ `.venv`; ห้ามอัปเกรด torch/torchvision/numpy/opencv ของ `.venv`; ตรวจ wheel cp314-win_amd64 และ CUDA kernel sm_61 จริง
- **R9 ห้ามวนเกิน** ทดลองแนวทางเดียวกันที่ล้มเหลวไม่เกิน 3 ครั้ง แล้วบันทึกผลและเปลี่ยนแนวทาง อย่าปล่อยให้ค้าง/ไม่รายงาน
- **R10 ทำให้จบ** อย่าหยุดที่แผน — ทำถึง §8 ทุกครั้ง แม้บางข้อเป็น BLOCKED

---

## 4. โปรเจกต์ที่ต้องโคลน/ติดตั้งเป็นโมดูลเสริม

วิธีจัดการ: ติดตั้งด้วย `pip` (pin เวอร์ชันใน `requirements.txt`) ตรวจ wheel ใน `.venv-eval` ก่อน; ถ้าต้องอ่านซอร์สหรือแพตช์ ให้ clone ไป `third_party/_src/<ชื่อ>/`
(gitignore) และบันทึก commit SHA ที่ใช้ใน `third_party/PINNED.md` — ห้ามคัดลอกซอร์สบุคคลที่สามเข้า repo ถ้า license ไม่ชัด

### 4.1 Tier A — ทำแน่นอน (หลักฐานรองรับ)

| โปรเจกต์ | License | ใช้ทำอะไร | หมายเหตุสำคัญ |
|---|---|---|---|
| **roboflow/trackers** — `github.com/roboflow/trackers` (`pip install trackers`) | Apache-2.0 (ยืนยันจากหน้า repo) มี SORT / ByteTrack / OC-SORT / BoT-SORT, Python ≥ 3.10, ไม่ผูกกับ detector | แทน greedy-IoU ของ `core/tracker.py` (S4) | สร้าง **tracker 1 instance ต่อกล้อง** — เหตุที่ไม่ใช้ `model.track(persist=True)` ของ ultralytics: MongDee ใช้ YOLO ตัวเดียวร่วมทุกกล้องใน AIWorker state ของ tracker จะปนกันข้ามกล้อง ต้องตรวจก่อนว่าเข้ากับ Python 3.14 และเทียบกับ `vision/tracking/` ที่มีอยู่แล้ว (ทางเลือกที่ต้นทุนต่ำสุด — เลือกตามค่าวัด) |
| **chinaheyu/cv2_enumerate_cameras** (`pip install cv2-enumerate-cameras`) | MIT (ยืนยัน) คืนชื่อ กล้อง VID/PID device path และ index ของทั้ง DSHOW และ MSMF | enumerate กล้องแทนการเปิดสุ่ม index 0..15 และใช้ device path เป็นตัวระบุกล้องรุ่นเดียวกันสองตัวให้เสถียรกว่า location string (S1, S3) | ใช้คู่กับ `pygrabber`/`core/camera_identity.py` ที่มีอยู่ ไม่ต้องทิ้งของเดิม |
| **opencv/opencv_zoo — YuNet** | โมเดล `2023mar` ที่ใช้อยู่แล้ว (ตรวจ LICENSE ในโฟลเดอร์โมเดลจริง) | Face Detection เป็น baseline (S6) | ตรวจว่ามีรุ่นใหม่/int8 หรือไม่ ปรับ input size, score, NMS และทดสอบ upscale เฉพาะขั้น detect สำหรับใบหน้าเล็ก |
| **PyAV** — `github.com/PyAV-Org/PyAV` (`pip install av`) หรือ ffmpeg CLI | BSD-3 (ต้องตรวจซ้ำ) และต้องตรวจ license ของ FFmpeg ที่ wheel ฝังมา (LGPL vs GPL) | backend จับภาพทางเลือกผ่าน `dshow` — เลือก device ตามชื่อ/หมายเลข, บังคับ pixel format, และ **รายงาน decode error ของ MJPEG จริง** ซึ่ง OpenCV ซ่อนไว้ (ช่วย S2) | ทำเป็น `CaptureBackend` เสริมหลัง interface เดียวกัน เลือกใช้ตามผลทดลอง §5.2 ไม่ใช่ทดแทนแบบไม่วัด |
| เครื่องมือทดสอบ: **Playwright**, **py-motmetrics** หรือ **TrackEval** | Apache-2.0 / MIT (ตรวจซ้ำ) | E2E ผ่านเบราว์เซอร์ และคำนวณ IDF1 / ID switches / MOTA | ติดตั้งใน `requirements-dev.txt` ไม่ใช่ runtime |

### 4.2 Tier B — ใช้เมื่อ Gate ไม่ผ่านด้วยการแก้ Tier A เท่านั้น (ต้องมีตัวเลขพิสูจน์)

| โปรเจกต์ | เงื่อนไข |
|---|---|
| **MiVOLO** `github.com/WildChlamydia/MiVOLO` — Apache-2.0 (โค้ดและ weights ตาม README) | ใช้เมื่อเพศ/อายุจาก FairFace ที่ปรับปรุงแล้วยังไม่ถึง G5 หรือใบหน้าไม่ชัด (README รายงาน gender face+body 97.99% แต่ body-only 96.48%) ต้องอ่านเงื่อนไขของ dataset (IMDB-clean/LAGENDA) ก่อนใช้เชิงพาณิชย์, ไม่อยู่บน PyPI ต้อง clone, ทดสอบใน venv แยกหรือแปลงเป็น ONNX (ยังไม่ทราบว่าเข้ากับ Python 3.14) |
| **DINOv2** `facebookresearch/dinov2` (ตรวจ license ของโค้ดและ weights จริง) หรือ embedder อื่นที่ license ผ่าน | ใช้แทน MobileNetV3-Small ใน `core/recognizer.py` เมื่อแก้ logic §5.3 แล้วยังไม่ผ่าน G4 |
| **FastReID** `JDAI-CV/fast-reid` (Apache-2.0) | เมื่อ Re-ID ข้ามกล้องอ่อนจากการวัด ใช้เฉพาะ checkpoint Market-1501/MSMT17 **ห้ามใช้ DukeMTMC** |
| detector ใบหน้าอื่น (เช่น MediaPipe Face Detector, UltraFace) | เมื่อ YuNet ไม่ผ่าน G6 และ license/wheel ผ่านการตรวจแล้วเท่านั้น |

### 4.3 ห้ามใช้ / ห้ามฝัง (ตามผล audit เดิม `third_party/MODEL_LICENSES.md`)
InsightFace ชุด pretrained (SCRFD / ArcFace / buffalo_*) — **non-commercial** · checkpoint ของ FastReID ที่เทรนด้วย DukeMTMC · **BoxMOT** (AGPL-3.0 ซ้อนเพิ่ม) ·
โมเดล/ชุดข้อมูลที่ไม่มีไฟล์ LICENSE ชัดเจน · ชุดข้อมูล non-commercial (CelebA, UTKFace, IMDB-WIKI ฯลฯ) ห้ามนำมาเทรน/ส่งมอบโมเดล ใช้วัดผลภายในเครื่องได้เท่านั้นโดยต้องบันทึก license ไว้

---

## 5. แผนงานตาม Phase (ทำตามลำดับ ทุก Phase จบด้วยไฟล์หลักฐานใน `reports/camera-ai-fix/`)

### 5.0 Phase 0 — Preflight และ Baseline
1. อ่านเอกสารใน §1 ให้ครบ ตรวจ `python --version`, `cv2.__version__`, `torch.cuda.is_available()` + smoke test บน GPU, `pip list` ของ `.venv`
2. `git status` / สร้าง branch / สำรองข้อมูลตาม R3 / บันทึก SHA256 ของไฟล์ข้อมูลจริง
3. รัน `python -m pytest tests/ -q` เก็บผล baseline; รายชื่อกล้องที่ enumerate ได้ (DSHOW name / VID / PID / path) เก็บเป็น JSON
4. เพิ่มตัวเลือกกำหนดโฟลเดอร์ gallery (env/param) และ test ที่พิสูจน์ว่าไม่แตะ `data/gallery/` จริงเมื่อใช้ temp
5. เขียน `reports/camera-ai-fix/00_baseline.md`

### 5.1 Phase 1 — "ห้องทดลองที่ไม่ต้องใช้คน" (สร้างก่อนแก้อะไร)
1. **FakeCapture + fault injection** ที่มี interface เหมือน `cv2.VideoCapture` (`isOpened/read/release/get/set`) เล่นวิดีโอ/ลำดับภาพพร้อมสคริปต์ความผิดปกติ:
   `hang(sec)`, `raise cv2.error`, `return False × N`, เฟรม torn (ตัด JPEG ครึ่งหนึ่งแล้ว `imdecode`, bit-flip ใน entropy segment, เลื่อน stride, แถบแนวนอนซ้ำ/เทา, YUY2 เพี้ยนสี),
   เฟรมดำ, เฟรมค้าง (ซ้ำเดิม), fps ตก, อุปกรณ์หายแล้วกลับมา, index ถูกสลับ, เปิดไม่ติดแล้วติด
   ต่อเข้ากับ production ได้ 2 ทาง: monkeypatch `cv2.VideoCapture` ใน unit test และ device ชนิด `sim:<scenario.json>` ที่ `_open_capture` รองรับเมื่อตั้ง `MONGDEE_SIM=1` เท่านั้น (สำหรับ E2E) —
   `--cameras <ไฟล์วิดีโอ>` ที่มีอยู่แล้วก็ใช้เป็นกล้องเสมือนพื้นฐานได้
2. **คลังเฟรมเสีย** สร้างจากภาพจริงหลากหลาย (มืด/สว่าง/คอนทราสต์จัด/มีคน) ไม่น้อยกว่า 500 ภาพสะอาด + 500 ภาพที่ทำเสียตามแบบข้างบน ใช้วัด recall/FPR ของตัวตรวจเฟรมเสีย
3. **`tools/camera_soak.py`** เปิดกล้อง USB จริงผ่านโค้ด production พร้อมกัน บันทึกต่อวินาที: fps, จำนวน read ล้มเหลว, อายุเฟรมล่าสุด, สถานะ, restart count, HRESULT (ถอดรหัสเป็นชื่อ) ลง CSV
4. **`tools/make_synth_scenes.py`** สร้างฉากสังเคราะห์พร้อม ground truth จาก `dataset/val` ของ FairFace (มี label เพศ/อายุ) วางบนพื้นหลังหลากหลาย
   คุมขนาดใบหน้า 24–160 px แล้วทำให้เสื่อมด้วย: ลดความละเอียดเป็น 320×240/640×480, blur (Gaussian/motion), noise, JPEG q20–80, แสงน้อย/แสงจัด, color cast, มุมหน้า (flip/หมุน/perspective)
5. **สินค้าสังเคราะห์และฉากว่าง** สร้างวัตถุ/ฉลากสมมติหลายแบบ (หรือภาพสินค้า public-domain) สำหรับทดสอบ add → train → detect → delete;
   ฉากว่างใช้วิดีโอสังเคราะห์ที่มีคนเดินผ่าน แสง/exposure กระเพื่อม และกล้องสั่น และ **ให้บันทึกภาพห้องจริงจากกล้องที่เสียบอยู่ 60 วินาทีต่อกล้องเป็น negative data ในเครื่อง** (gitignore, ไม่ส่งออก, ไม่ commit)
6. **ชุดวัด Tracking**: ฉากสังเคราะห์คนเดินตัดกัน/บังกัน/หยุด/กลับตัว พร้อม ground truth ที่แน่นอน; ทดสอบที่ความถี่ AI 2 / 4 / 8 / 15 Hz เพื่อจำลองสภาพจริงที่ AIWorker ช้า
7. เก็บ baseline ของ **ทุก** ตัวชี้วัดใน §7 ก่อนแก้โค้ดผลิตภัณฑ์ (เพื่อจะพิสูจน์ได้ว่า "ดีขึ้น") → `reports/camera-ai-fix/01_baseline_metrics.json`

### 5.2 Phase 2 — เสถียรภาพกล้อง (S1, S2, S3)

**2.1 การทดลองวินิจฉัย S1 (รันเองบนเครื่องนี้ ทีละข้อ บันทึกผลทุกข้อ แต่ละข้อรันใน process แยกพร้อม timeout เพื่อกันค้าง)**
- **E1** enumerate format จริงของกล้องแต่ละตัวด้วย `pygrabber` (media subtype, ขนาด, ช่วง fps) — ยืนยันว่ามี MJPG หรือไม่ที่ขนาดไหน และเทียบกับ `cap.get(CAP_PROP_FOURCC)`; ทดลองลำดับ set (FOURCC→ขนาด→fps และกลับกัน)
- **E2** DSHOW เปิดกล้อง 2 ตัวพร้อมกันใน process เดียว, stagger 0 / 2.5 / 10 วินาที, เปิดสลับลำดับ, format ต่างๆ — อ่านต่อเนื่อง 5 นาที/เงื่อนไข
- **E3** **แยกกล้องละ OS process** (DSHOW) — พิสูจน์ว่าปัญหาอยู่ที่ COM/DirectShow ใน process เดียวหรืออยู่ที่ตัวอุปกรณ์/USB
- **E4** MSMF ใน process แยก ตั้ง `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` และ `=1` ก่อน `import cv2` (ต้องมี timeout ฆ่า process — เคยพบ hang 180 วินาที)
- **E5** FFmpeg dshow (PyAV หรือ ffmpeg CLI): `-video_size 320x240 -framerate 15` เทียบ `-vcodec mjpeg` กับ `rawvideo`, เลือกกล้องด้วยชื่อ+`video_device_number`, สองกล้องพร้อมกันคนละ process
- **E6** อ่านทอพอโลยี USB แบบ **read-only** ด้วย PowerShell (`Get-PnpDevice`, `Get-PnpDeviceProperty` ค่า Parent/LocationInfo/BusReportedDeviceDesc; `powercfg /q` ดูค่า USB selective suspend) —
  สองกล้องอยู่ controller/hub เดียวกันหรือไม่, ความเร็ว (High-Speed/SuperSpeed) — ห้ามแก้ค่าใดๆ
- **E7** ทดลองรอบเปิด/ปิดถี่ที่ไม่ควรทำเพื่อวัด cooldown ของอุปกรณ์ (รายงานเดิม 15–30 วินาที) แล้วเอาไปกำหนดนโยบาย reopen
- **กฎตัดสิน:** เลือก backend/โมเดล process ที่ให้กล้องคู่นี้ทำงานพร้อมกันได้เสถียรที่สุดตามค่าวัด 30 นาที ถ้าทุกเส้นทางซอฟต์แวร์ล้มเหลวที่กล้องตัวที่สองทั้งที่แต่ละตัวเดี่ยวๆ ใช้ได้
  ให้สรุปเป็น "ขีดจำกัดฮาร์ดแวร์/ไดรเวอร์ (มีหลักฐาน)" พร้อมทอพอโลยีจาก E6 และคำแนะนำฮาร์ดแวร์แยกไว้ต่างหาก (ไม่นับเป็นการแก้ซอฟต์แวร์) แล้วทำ 2.2 ต่ออยู่ดี

**2.2 สิ่งที่ต้องสร้างโดยไม่ขึ้นกับผลทดลอง (ทำให้ระบบ "พังแล้วต้องกลับมาได้เสมอ")**
- **ก. Capture isolation:** แต่ละกล้องรันใน OS process ของตัวเอง (`multiprocessing` แบบ spawn) ส่งเฟรมกลับผ่าน shared memory (หรือ pipe แบบ JPEG ถ้าวัดแล้วเร็วพอ);
  parent มี watchdog: ไม่มีเฟรมใหม่ภายใน T วินาที → `kill` process → spawn ใหม่ด้วย backoff (รีเซ็ต DirectShow state ทั้งหมด) คง interface `on_frame/on_status` ของ `CameraWorker` เพื่อให้ `BoothManager`/UI ไม่ต้องเปลี่ยน
  เปิด/ปิดด้วย config `capture.isolation = process | thread` (default `process` บน Windows เมื่อผ่าน Gate)
- **ข. Read watchdog + freshness:** สถานะ online ต้องอ้างอิง "อายุของเฟรมล่าสุด" ไม่ใช่แค่ว่า thread ยังอยู่; ตรวจเฟรมค้าง (hash/diff ซ้ำนานเกิน X วินาที) → `stale`
- **ค. Frame integrity v2:** รวมสัญญาณหลายชั้น — (1) decode error จริงจาก FFmpeg path ถ้าใช้, (2) `_looks_like_noise` เดิม, (3) ตัวตรวจแถบ/torn ที่ **ผ่านการวัด FPR กับคลังเฟรมสะอาดใน §5.1** (ห้ามซ้ำรอย banding check ที่เคยถูกถอด),
  (4) ตัวตรวจ "มืด/ไร้ประโยชน์" (ความสว่างเฉลี่ยและ std ต่ำต่อเนื่อง) → สถานะ `degraded: dark` ไม่ใช่ online ปกติ
- **ง. จัดการเฟรมเสียแบบนุ่มนวล:** เฟรมเสียชั่วคราว → ทิ้งและแสดงเฟรมดีล่าสุด (ไม่นับเข้า FAIL_THRESHOLD จนกว่าจะเสียต่อเนื่องเกินเวลา T) → เกินแล้วค่อย release+reopen ในที่เดิม → ยังไม่กลับ → ยกระดับเป็น kill process
  (ให้ `/stream` **ไม่มีทาง**ส่งเฟรมที่ตรวจว่าเสียออกไป)
- **จ. ผู้ควบคุมทรัพยากรกล้องเดียว (`CameraResourceManager`):** รวม hot-plug scan และ `_attempt_reopen` ของแต่ละกล้องให้มีตารางจองเดียว ไม่มีสอง thread probe index เดียวกัน; ใช้ enumeration (`cv2_enumerate_cameras`/`pygrabber`)
  แทนการเปิดสุ่ม 0..15; เคารพ cooldown จาก E7 พร้อม jitter
- **ฉ. ตัวระบุกล้อง:** เก็บ device path จาก enumeration เป็นตัวระบุหลักเสริม physical-id เดิม (รวมการป้องกันสลับกล้องที่แก้ไว้แล้ว ต้องไม่ถอยหลัง)
- **ช. Observability:** เพิ่มใน `/api/state`: backend, format จริง, fps จริง, `last_frame_age_ms`, `restart_count`, `last_error` (ถอดรหัส HRESULT เป็นชื่อ เช่น 0xC00D3704, 0x8007001F, 0x800703E3, 0x80070005 พร้อมคำอธิบายภาษาไทยสั้นๆ)
- **ซ. ถ้าผลทดลองแสดงว่าความละเอียดสูงขึ้นทำได้** ให้ทำ adaptive profile (ขอ 640×480 เมื่อกล้องเดียว/มีงบ) เพราะช่วย S4 โดยตรง — ตัดสินด้วยตัวเลขจาก §5.4

**2.3 การพิสูจน์:** unit test ด้วย FakeCapture ครบทุกแบบความผิดปกติ · E2E ด้วยกล้อง `sim:` · soak กล้องจริง 60 นาที · kill process จับภาพสุ่ม 20 ครั้ง (กลับมา ≤ 10 วินาที/ครั้ง) ·
ฉีด torn/black burst 50 ครั้ง · read hang 60 วินาที ต้องตรวจพบ ≤ 5 วินาที · (ถ้ามี admin) `pnputil` ปิด/เปิด USB Camera 20 รอบ ต้องกลับทุกรอบ ≤ 30 วินาที

### 5.3 Phase 3 — Ghost Product (S5)

**3.1 เขียน test ที่ล้มเหลวก่อน (reproduce):**
T1 orphan ใน manifest ต้องไม่ถูกตรวจพบ/ไม่เขียน DB หลังสตาร์ท ·
T2 ลบสินค้าระหว่างที่ import ภาพ/วิดีโอกำลังรัน → ต้องไม่มี gallery/manifest กลับมา ·
T3 ลบสินค้าที่ key ตรงชื่อ COCO ขณะ worker กำลังทำงาน → หยุดตรวจพบทันทีโดยไม่ต้องรีสตาร์ท ·
T4 ฉากว่างยาวนาน (คนเดินผ่าน + exposure กระเพื่อม + กล้องสั่น) → 0 detection / 0 แถว interest ·
T5 สินค้าอยู่หน้ากล้อง → ตรวจพบ, เอาออก → หายภายใน ≤ 2 วินาที

**3.2 แก้:**
- `ProductRecognizer`: `reconcile(valid_keys)` ตอนสตาร์ท (ย้าย orphan ไป `_orphaned/` ไม่ลบ) · `RLock` ครอบ `_gallery/_manifest/add_sample/clear_product` · tombstone/cancellation token ให้ job import ยกเลิกได้และ `add_sample` ปฏิเสธ key ที่ไม่อยู่ใน catalog ·
  `identify()` กรองด้วยชุด key ที่ valid ณ ขณะนั้น
- `api_delete_product`: ยกเลิกและรอ job import → ลบ catalog + gallery + สถานะ import → **ล้างสถานะใน memory ทันที** (`aggregator`, `interest_tracker`, `_product_detections`, `current_product`, `_last_boxes` ของทุก worker) →
  เรียก `refresh_catalog()` ที่คำนวณ `_legacy_coco_classes/_class_indices` ใหม่ทุก worker โดยไม่ต้องรีสตาร์ท (เพิ่มเมื่อ `add`/`update` ด้วย)
- **Defense in depth ที่จุดใช้งาน:** ทิ้ง detection ที่ `class_name` ไม่อยู่ใน catalog (หรือชุด COCO ที่เปิดใช้) ก่อนวาด/aggregate/log; เอา fallback `else class_name` ใน `_on_interest_confirmed` ออก; `products_detected` นับเฉพาะสินค้าที่ valid
- **ลด false positive จริง (ตรงกับสมมติฐานหลัก):** (ก) เก็บ **negative/background samples** (จากภาพห้องจริงตอนไม่มีสินค้า + ขอบภาพ) เข้า gallery เป็นคลาสปฏิเสธ;
  (ข) เลิก fallback ครอบ "ทั้งเฟรม" แบบเงียบๆ — ใช้ crop ที่ผ่านการตรวจคุณภาพ/แจ้งเตือนใน UI เมื่อ crop เป็นทั้งเฟรม; ตัด sample ที่ซ้ำกัน (cosine > 0.98) และ augment (flip/สี/สเกล);
  (ค) calibrate threshold แบบ open-set ต่อสินค้า (leave-one-out + เทียบ negative) แทนค่าคงที่ 0.45/0.65, ใช้ top-k vote + margin ต่อ background;
  (ง) ยืนยันแบบเชิงเวลา N-of-M เฟรมและกล่องนิ่งก่อนแสดง/นับ; (จ) `ForegroundProposer`: ระงับ proposal และ re-init MOG2 เมื่อฉากเปลี่ยนทั้งภาพ (สัดส่วน foreground สูงผิดปกติ, ความสว่างเปลี่ยนพรวด, กล้องสั่น) และเรียก `update()` ทุกเฟรมให้ background model ทันสมัย ·
  (ฉ) เปลี่ยน embedder เป็น Tier B เฉพาะเมื่อข้างบนแล้วยังไม่ผ่าน G4

### 5.4 Phase 4 — Face Detection และความแม่นยำเพศ/อายุ (S6, S4)

**4.1 โมดูล Face Detection ระดับ first-class** (`core/face/`) — interface `FaceDetector` คืน `Face(bbox, score, landmarks5, quality)`; baseline = YuNet ปรับได้ (score, NMS, input size, upscale เฉพาะตอน detect);
โหมด (ก) ทั้งเฟรม (ข) ภายใน ROI ของคน; เมตริกที่ต้องรายงาน: recall ตามขนาดใบหน้า, false positive ต่อเฟรมบนฉากไม่มีหน้า, latency บนเครื่องนี้ (CPU และ GPU)
ห้ามสลับไปใช้ detector อื่นถ้าไม่ผ่าน R6/R7

**4.2 ผูกใบหน้ากับคน + Best-shot:** จับคู่ใบหน้ากับ person track (จุดกึ่งกลางใบหน้าอยู่ในส่วนบนของกล่องคน, Hungarian) → track ใบหน้าระยะสั้น →
เก็บ **best-shot Top-K ต่อคน** จากคะแนนคุณภาพ = ขนาด × ความคม (Laplacian) × ความหันตรง (ประมาณจาก 5 landmarks) × ความสว่างที่พอดี;
รัน FairFace เฉพาะ best-shot (ไม่ใช่ทุกเฟรม/ทุก AI pass) และรวมเส้นทางกรอบบนจอกับ `AttributeAnalyzer` ให้ใช้ผลชุดเดียวกัน (เลิกคำนวณซ้ำซ้อน)

**4.3 แสดงผล/API:** วาดกรอบใบหน้าบางๆ (สีต่างจากกรอบคน), เพิ่ม `faces_total` ใน `get_camera_snapshot` และ face ต่อ track ใน state, ปรับ dashboard/หน้า booth เท่าที่จำเป็น คงข้อความไทยเดิม

**4.4 โปรแกรมปรับความแม่นยำเพศ/อายุ (วัดทีละข้อ เก็บเฉพาะที่ดีขึ้น):**
1. **Preprocessing ให้ตรงกัน:** ฟังก์ชันเดียวใช้ทั้งเทรนและ inference; วัดว่าภาพ FairFace ใน `dataset/` เป็น crop แบบไหน (สัดส่วนใบหน้าต่อภาพ) แล้วทดสอบ margin {0, 0.1, 0.2, 0.25, 0.3, 0.4} และการ align ด้วย 5 landmarks; ตรวจ RGB/BGR
2. **เกณฑ์จากข้อมูล:** วัดความแม่นยำตามขนาดใบหน้า/blur/ความสว่าง แล้วกำหนด min-face-px และ threshold จากเป้า precision (ไม่ใช้ 64 px / 0.6 เพราะเดา); ทำ temperature scaling ให้ confidence มีความหมาย; "ผิดแย่กว่าไม่ตอบ" → ไม่มั่นใจ = UNKNOWN
3. **นโยบายตัดสินต่อคน:** ลงคะแนนถ่วงน้ำหนักจาก best-shot พร้อม hysteresis (ต้องนำอีกฝั่งเกิน margin จึงจะพลิก, ล็อกเมื่อสอดคล้องกัน k ครั้ง) — ผลไม่กะพริบ M/F
4. **กติกา CHILD:** ใช้ผลรวมความน่าจะเป็นช่วง 0–9 เป็นหลัก ห้ามให้ "10–19" เพียงบัคเก็ตเดียวเอาชนะเพศ; วัด confusion adult↔child
5. **Fine-tune ปรับโดเมน (ถ้า baseline ไม่ผ่าน G5):** เทรนต่อจาก checkpoint เดิมด้วย FairFace + degradation augmentation (ลดความละเอียด 24–112 px แล้วขยาย, blur, JPEG, noise, แสงน้อย, color cast, jitter ของ margin/ตำแหน่ง, occlusion) เพียงไม่กี่ epoch บน GTX 1050,
   early stop, ประเมินบนชุด val ที่ทำให้เสื่อมด้วย seed คงที่; **ห้ามทับ** `best_model_state_dict.pt` เดิม — บันทึกชื่อใหม่ + config สลับ + เก็บของเดิมเป็น fallback; ห้ามใช้ชุดข้อมูล non-commercial เทรน
6. รายงาน confusion matrix แยกเพศ และช่องว่าง recall ชาย vs หญิง (ตรงกับอาการที่ผู้ใช้เจอ: ชาย→หญิง) รวมทั้งแยกตามขนาดใบหน้า
7. กรณีไม่เห็นใบหน้า → UNKNOWN (ไม่เดาจากรูปร่าง) เว้นแต่ MiVOLO (Tier B) พิสูจน์แล้วว่าดีกว่าและผ่าน license

### 5.5 Phase 5 — ตรวจจับคนและ Tracking (S4)
1. **Detector:** เทียบ `yolo11n` @320/@640, `yolo11s` @640, ค่า `conf`/NMS-IoU บน GPU ของเครื่องนี้ ด้วย AP ของคลาส person บนชุดวัดที่มี label
   (ใช้ COCO val2017 เฉพาะภาพที่มีคน ลดขนาดเป็น 320×240/640×480 เพื่อจำลองกล้อง — **ใช้วัดผลภายในเท่านั้น ห้ามส่งออก/commit**) รายงาน AP และ FPS ต่อโมเดล
2. **Tracker:** สร้าง `PersonTrackerV2` ที่ signature เดียวกับ `PersonTracker.update()` (คง field `track_id, bbox, first_seen, last_seen, category` และ `evicted_tracks`) ข้างหลังใช้ ByteTrack/BoT-SORT (Tier A; เทียบกับ `vision/tracking/` ที่มีอยู่)
   ต้อง (ก) 1 instance ต่อกล้อง, (ข) ใช้ dt จริงจาก timestamp เพราะความถี่ AI ไม่คงที่, (ค) ใช้ detection คะแนนต่ำในขั้น association รอบสอง (แนวคิด ByteTrack), (ง) ตั้ง track buffer เป็นวินาที, (จ) reset เมื่อกล้อง reconnect
   เปิดใช้ผ่าน config `tracking.backend = greedy_iou | bytetrack | botsort`
3. **งบเวลา AIWorker:** profile ต่อขั้น (YOLO / face / FairFace / recognizer) บน GTX 1050, ทำ batch ที่ทำได้, throttle `_classify_person` ต่อ track;
   เป้า ≥ 8 Hz ต่อกล้องเมื่อรัน 2 กล้อง หรือบันทึกตัวเลขจริงและคอขวดถ้าไม่ถึง
4. **Re-ID ข้ามกล้อง:** วัดก่อน ถ้าอ่อนจริงจึงพิจารณา FastReID (Tier B)

### 5.6 Phase 6–7 — E2E อัตโนมัติและ Soak
- **E2E (ไม่มีคน):** รัน `web_server.py` ด้วย `--db`/`--products`/gallery ชั่วคราว และกล้อง `sim:`/ไฟล์วิดีโอ; Playwright เปิด `/booth`, `/trainer`, `/dashboard` ตรวจไม่มี console error, `/stream/<cam>` ส่ง JPEG จริงต่อเนื่อง (ตรวจ magic bytes และภาพไม่ใช่เฟรมเสีย),
  ทำ flow สินค้าครบวงจรผ่าน REST/UI (สร้าง → อัปโหลด → ตรวจพบ → ลบระหว่าง import → ยืนยันไม่มี ghost), ฉีดความผิดปกติระหว่างสตรีมแล้วตรวจสถานะ/การกลับมา, เก็บภาพหน้าจอเป็นหลักฐาน
- **Soak กล้องจริง 60 นาที** สองกล้องพร้อมกัน (ทำหลังผ่านการทดลอง §5.2) + soak แบบ `sim:` 8 ชั่วโมงเร่งเวลา ตรวจ memory/handle/thread รั่ว (psutil) และ log ไม่มี exception ซ้ำ
- ก่อน–หลัง E2E เทียบ SHA256 ข้อมูลจริงต้องเท่าเดิม (G9)

---

## 6. แหล่งข้อมูลทดสอบที่ใช้ได้โดยไม่ต้องมีคน
- `dataset/val` (FairFace, มี label เพศ/อายุ) สำหรับ face/attribute — ห้ามนำชุดข้อมูลไปเผยแพร่
- ภาพ/วิดีโอสังเคราะห์ที่สร้างเอง + ภาพห้องจริงจากกล้องที่เสียบอยู่ (เก็บในเครื่อง gitignore) เป็น negative data
- ฟุตเทจคนเดินที่ license อนุญาต (CC0/Public Domain/Pexels/Pixabay/Wikimedia) — บันทึกที่มา+license ใน `tests/assets/SOURCES.md` ไม่ commit ไฟล์ขนาดใหญ่
- COCO val2017 (คลาส person) สำหรับ AP ภายในเท่านั้น
- ถ้าจำเป็นต้องใช้ข้อมูลที่ต้องมีคนจริง ให้ทำเป็น BLOCKED — ห้ามขอให้ผู้ใช้ไปถ่าย

---

## 7. Gate การยอมรับ (ตัวเลขเป็นค่าเริ่มต้น ปรับได้ถ้ามีเหตุผลจากข้อมูลและบันทึกไว้)

| Gate | เงื่อนไข PASS |
|---|---|
| G1 กล้องหลายตัว | สองกล้อง USB สตรีมพร้อมกัน 60 นาที fps ≥ 10 (320×240) ไม่มี process crash; **หรือ** สรุป "ขีดจำกัดฮาร์ดแวร์" พร้อมหลักฐาน E1–E7 ครบและคำแนะนำฮาร์ดแวร์ (ระบุ BLOCKED ส่วนที่ทำไม่ได้) |
| G2 ภาพเสีย | คลังเฟรมเสีย: recall ≥ 95% · FPR บนเฟรมสะอาด ≤ 0.5% · ไม่มีเฟรมที่ถูกตรวจว่าเสียหลุดออก `/stream` |
| G3 การกลับมา | ทุกความผิดปกติที่ฉีด (hang, kill, torn burst, หายแล้วกลับ) กลับมา p95 ≤ 15 วินาที, สูงสุด ≤ 30 วินาที, **ไม่มี "ต้องรีสตาร์ททั้งแอป"**; hang ตรวจพบ ≤ 5 วินาที |
| G4 สินค้า | T1–T5 ผ่านทั้งหมด · ฉากว่าง ≥ 30 นาที (หรือ ≥ 5,000 AI pass) มี false positive = 0 · recall ≥ 0.9 ที่ ≥ 15 samples บนมุมมองที่ไม่ได้ใช้เทรน · รีสตาร์ทแล้วไม่มี orphan |
| G5 เพศ/อายุ | บนชุดเสื่อม ใบหน้า ≥ 48 px: precision ของคำตอบที่ "มั่นใจ" ≥ 95% · ชาย→หญิง ≤ 3% · ช่องว่าง recall ชาย/หญิง ≤ 5 จุด · อัตราพลิกเพศในลำดับเฟรมเดียวกัน ≤ 2% · รายงานช่วง 30–48 px แยกต่างหาก |
| G6 Face detection | recall ≥ 90% ที่ใบหน้า ≥ 40 px (ชุดเสื่อม) · false positive ≤ 0.02/เฟรมบนฉากไร้ใบหน้า · latency ≤ 15 ms/เฟรม (320×240) บนเครื่องนี้ |
| G7 Tracking | ID switch ลดลง ≥ 50% จาก baseline · IDF1 ≥ 0.85 บนชุดสังเคราะห์ที่ 4 Hz |
| G8 ไม่ถดถอย | `pytest` ผ่านทั้งหมด (≥ 613 + ใหม่) ไม่มี thread-exception warning ใหม่ |
| G9 ข้อมูลผู้ใช้ | SHA256 ของ `products.json`, `data/mongdee.db`, `data/gallery/` ก่อน–หลังเท่าเดิม (orphan ถูกย้ายไป quarantine เฉพาะตอนผู้ใช้เปิดแอปจริง ไม่ใช่ตอนทดสอบ) |
| G10 License | `third_party/MODEL_LICENSES.md` + `PINNED.md` ครบทุก dependency/โมเดลใหม่ ไม่มีรายการ non-commercial |

---

## 8. รายงานฉบับสุดท้าย `docs/CAMERA_AI_FIX_REPORT.md` (ภาษาไทย ตารางเป็นหลัก)
A. สาเหตุราก ต่ออาการ S1–S6 พร้อมป้าย CONFIRMED/HYPOTHESIS/BLOCKED และหลักฐานที่อ้างถึงไฟล์ ·
B. รายการเปลี่ยนแปลง (ไฟล์/เหตุผล/flag/commit) ·
C. ตาราง before/after ของทุกตัวชี้วัดใน §7 ·
D. ผล E1–E7 ครบทุกข้อ (รวมข้อที่ล้มเหลว) และเหตุผลที่เลือก backend ·
E. สถานะ G1–G10 = PASS / FAIL / BLOCKED (ห้ามใช้คำอื่น) ·
F. ปัญหาที่ยังเหลือและสิ่งที่พิสูจน์ไม่ได้ ·
G. คำแนะนำฮาร์ดแวร์ (แยกจากการแก้ซอฟต์แวร์: ต่อ controller/พอร์ตแยก, hub มีไฟเลี้ยง, สายสั้น/มีฟิลเตอร์, แสง) ·
H. เรื่องที่ผู้ใช้ต้องตัดสินใจเอง (AGPL ของ ultralytics, license ของ dataset, การเปิดใช้ `capture.isolation`) ·
I. วิธีรันชุดตรวจทั้งหมดซ้ำด้วยคำสั่งเดียว (`tools/run_all_checks.ps1` หรือเทียบเท่า)

## 9. เมื่อติดขัด
ทำส่วนที่ทำได้ให้เสร็จก่อน · บันทึกสิ่งที่ลอง ผล และสิ่งที่ขาด · ห้ามข้าม Gate เงียบๆ ห้ามปรับตัวเลข Gate ให้ผ่านโดยไม่บันทึกเหตุผล · ห้ามแก้ระบบ Windows นอกโปรเจกต์ ·
ถ้าต้องตัดสินใจที่เป็นของผู้ใช้จริงๆ (เช่น license เชิงพาณิชย์) ให้ทำงานที่เกี่ยวข้องต่อด้วยทางเลือกที่ปลอดภัยที่สุดและระบุในหัวข้อ H ของรายงาน

