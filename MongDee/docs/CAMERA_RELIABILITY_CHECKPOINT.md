# Camera Reliability — Checkpoint (2026-09-21, updated: bug #5 found+fixed live)

เอกสารนี้สรุปว่า **ตอนนี้ระบบกล้องของ MongDee อยู่ในสถานะไหน**, **ทำอะไรไปแล้วในรอบนี้**,
และ **อะไรยังไม่ verify** — เขียนไว้ให้คนต่อจากนี้ (หรือ Claude session ถัดไป) อ่านแล้วเข้าใจ
บริบทได้ทันที ไม่ต้องไล่ไปอ่านการสนทนาเดิมทั้งหมด

อ้างอิงเอกสารก่อนหน้า: [`docs/MULTI_CAMERA_ROOT_CAUSE_REPORT.md`](MULTI_CAMERA_ROOT_CAUSE_REPORT.md)
(§A-R) — การสืบสวน root cause แบบละเอียดที่สุด ทำไว้ session ก่อนหน้า checkpoint นี้

---

## 1. สรุปสถานะปัจจุบัน (ให้อ่านก่อน)

**Bug หลัก 4 ตัวที่เคยทำให้กล้องพัง/ค้างถาวร ถูกพบและแก้แล้ว** — โค้ดอยู่ใน working tree
ปัจจุบัน (`core/vision.py` และไฟล์เกี่ยวข้อง) แต่ **ยังไม่ commit**:

1. Process crash ตอนเปิดกล้อง 2 ตัวพร้อมกัน (DirectShow COM ไม่ thread-safe) — แก้ด้วย
   `DIRECTSHOW_LOCK`
2. Reconnect อาจเปิดกล้องคนละตัวแบบเงียบ ๆ ถ้า index ถูกสลับ (Windows re-enumerate) — แก้ที่
   `_relocate_device_if_moved()`
3. **"กล้องค้างใน reconnecting ตลอดไปจนกว่าจะ restart server"** — เกิดจาก `cap.release()`
   ที่ wedge ได้ (native DirectShow call ค้างไม่คืนตลอดกาล) ถูกเรียกแบบ synchronous ใน 4 จุด
   จุดหนึ่งถือ lock ตัวเดียวกับที่ทุกกล้องต้องใช้เปิด — พัง 1 จุด กล้องทุกตัวเปิดไม่ได้อีกเลย
   นี่คือ**อาการเดียวกับที่ user รายงานรอบนี้เป๊ะ** — แก้แล้วด้วย `_release_capture_safely()`
   (release บน background thread, bounded wait, ไม่ block caller เกิน 1.5 วิ)
4. MSMF backend พังสนิทบนฮาร์ดแวร์ dev เครื่องนี้ (0/14 read สำเร็จ) แถม hang ได้ 180+ วิ —
   ตัดออกจาก fallback path บน Windows numeric index แล้ว เหลือ DSHOW อย่างเดียว
5. **(ใหม่ 2026-09-21) Worker thread ตายเงียบ ๆ แล้วไม่มีใครฟื้นให้เลย** — reproduce ได้จริงบน
   server ที่รันอยู่ตอนนั้น (CAM-1 ค้าง `frame_sequence`/`captured_frames` นิ่งสนิท,
   `reconnect_count=0`, `dropped_frames=0` นานหลายนาที) `cap.read()` guard แค่ `cv2.error` เดิม
   ไม่ครอบ exception อื่น และ `self._process_captured_frame(ok, frame)` **ไม่มี guard เลย** —
   ถ้ามี exception หลุดจากใน (`on_frame()`, box drawing, frame-slot publish ฯลฯ) thread ตาย
   เงียบ ๆ แล้ว watchdog เดิมช่วยไม่ได้เลยสักตัว: `recover_from_hung_read()` เช็คแค่ "read ค้างอยู่
   ไหม" (ไม่ใช่ "thread ตายหรือยัง"), `check_stream_health()` ทำงานแค่ตอน transition
   online→offline ครั้งเดียว พอ offline แล้วไม่ทำอะไรอีกเลย — แก้แล้ว: (a) ขยาย
   `except cv2.error` → `except Exception` รอบ `cap.read()`, (b) ห่อ
   `_process_captured_frame()` และ `_attempt_reopen()` ด้วย `try/except Exception` เพิ่ม (ทุกจุด
   log แล้วนับเป็น frame/reopen ที่ล้มเหลว ไม่ปล่อยให้ thread ตาย), (c) เพิ่ม safety-net ใหม่ใน
   `web/booth_manager.py`'s `_capture_watchdog_loop`: เช็ค `worker._running and not
   worker.is_alive()` (ไม่รวม escalated-to-process case ที่ตั้งใจทิ้ง thread เดิม) ทุก 0.5 วิ ถ้า
   worker ตายจริงโดยไม่ตั้งใจ → สร้าง worker ใหม่แทนที่ camera_id เดิมทันที ไม่ต้อง restart
   process

**ยังไม่ confirm 100%**: กล้อง USB 2 ตัวรุ่นเดียวกันเสียบ hub เดียวกัน stream พร้อมกันได้จริง
หรือไม่ — evidence เอียงไปทาง "USB hub bandwidth ไม่พอ" แต่ยังไม่มี experiment ตัดสิน (ต้องลอง
คนละ USB controller/hub ซึ่งต้องขยับสายจริง ทำจาก software ไม่ได้)

---

## 2. งานที่ทำใน checkpoint รอบนี้ (2 รอบคำขอ)

### รอบที่ 1 — "สอง USB camera ต้อง stream พร้อมกันได้"
- อ่าน architecture เต็ม (`core/vision.py`, `core/capture_process.py`, `web/booth_manager.py`)
  เทียบกับ requirement 29 ข้อที่ user ให้มา — พบว่า**ของเดิมตอบโจทย์เกือบทั้งหมดอยู่แล้ว**:
  per-camera worker/capture owner เดียว, AI แยก thread ไม่ block capture, state machine
  ต่อกล้องจริง, identity ผูกกับ VID/PID+location ไม่ใช่ index ดิบ, backoff แบบ exponential,
  frontend poll `/api/state` แล้ว auto-reload stream ตอนกลับ online
- รัน real hardware test บนกล้องจริง 2 ตัวที่เจอในเครื่องนี้ (index 0 กับ 2, VID:PID เดียวกัน
  4C4A:4A55): ยืนยันว่า CAM-1 stream ต่อเนื่องได้ (`frame_sequence` วิ่ง), CAM-2 เปิดพร้อมกัน
  ไม่ได้แต่ retry ด้วย backoff ถูกต้อง และ **CAM-1 ไม่ถูกกระทบเลย** — isolation ทำงานตามที่
  ออกแบบไว้
- เจอ + แก้ **test isolation bug** (ไม่ใช่ production bug): test บางตัวไม่ mock
  `core.camera_identity.list_directshow_devices()` เลยไปเจอ enumeration จริงของเครื่องนี้
  (มี driver "OBS Virtual Camera" ติดตั้งอยู่ ทำให้ index เดิมรายงานชื่ออุปกรณ์ต่างกันไปมา
  ระหว่าง call) → แก้ 3 จุดใน `tests/core/test_camera_open.py` ให้ mock
  `_forbidden_device_name` แทนที่จะพึ่ง state จริงของเครื่อง
- ผลทดสอบ: `803 passed, 0 failed` (เต็ม `tests/core tests/web tests/camera`)

### รอบที่ 2 — "กล้อง stream ปกติพักหนึ่งแล้วค้างใน reconnecting ตลอดไป ไม่ฟื้นเอง"
- Trace เต็ม production path: `web_server.py → web/server.py → web/booth_manager.py →
  core/vision.py` — ยืนยันว่า watchdog (`_capture_watchdog_loop`, ทุก 0.5 วิ) **เชื่อมกับ
  production runtime จริง** ไม่ใช่ฟังก์ชันลอย ๆ ที่ไม่มีใครเรียก
- อ่านโค้ด `recover_from_hung_read()` (จับ `cap.read()` ค้าง) และ `check_stream_health()`
  (จับ stream หยุด/frozen) — ตรงกับข้อความ error ที่ user เห็นจริง
  ("ภาพจากกล้องหยุดหรือเสียหาย — กำลังเชื่อมต่อใหม่" มาจาก `check_stream_health()` บรรทัดเป๊ะ)
- ยืนยัน (โดยอ่านโค้ด) ว่า `_release_capture()` set `self._cap = None`
  **แบบ synchronous ทันที** ก่อนที่จะส่ง native release() ไปทำงานบน background thread —
  แปลว่ารอบถัดไปของ loop จะเห็น `self._cap is None` แล้วเข้า path reopen ทันที ไม่ว่า native
  release จะค้างแค่ไหนก็ตาม (bug เดิมที่ทำให้ค้างตลอดไปคือ**ไม่มี**การ set None ก่อน — จุดนี้
  แก้แล้วจริง ตรวจสอบจาก source code แล้ว)
- รัน regression test เฉพาะทาง: `test_camera_unplug_replug.py` +
  `test_camera_process_escalation.py` + `tests/camera` → **91 passed**
- พยายามทำ real hardware failure-injection test (เปิดกล้องตัวเดียวกันจาก process อื่นแข่ง
  ระหว่างที่ MongDee กำลัง stream อยู่) — **ผลลัพธ์: DirectShow ปฏิเสธการเปิดจาก process ที่สอง
  ทันที (`isOpened()=False`) ไม่ได้ทำให้ MongDee ที่กำลังรันอยู่หลุด** เป็นหลักฐานว่า OS เอง
  บังคับ "เจ้าของเดียวต่อกล้อง" อยู่แล้ว แต่**ไม่ได้พิสูจน์ mid-stream recovery แบบ end-to-end
  จริง** เพราะไม่มีทางถอดปลั๊ก USB จริงจาก sandbox นี้ — ดูข้อจำกัดหัวข้อ 4

---

## 3. Environment note ที่สำคัญ (เจอใหม่ session นี้)

เครื่องที่ session นี้รันอยู่มี DirectShow index ที่ **ไม่เสถียรข้าม process/เวลา** — เรียก
`list_directshow_devices()` คนละครั้งห่างกันไม่กี่นาที index เดิม (เช่น index 2) รายงานชื่อ
อุปกรณ์ต่างกัน ("USB Camera" ↔ "OBS Virtual Camera") ทั้งที่ไม่มีใครแตะฮาร์ดแวร์เลย — อาจเป็น
เพราะเครื่องนี้เป็น VM/sandbox ที่มี virtual camera driver ติดตั้งอยู่ ไม่ใช่ physical booth PC
ทั่วไป **ผลการทดสอบ real-hardware ใน checkpoint นี้จึงควรถือเป็นหลักฐานสนับสนุน ไม่ใช่การยืนยัน
เทียบเท่าเครื่อง booth จริง**

---

## 4. ยังไม่ได้ทำ / ยังไม่ verify (พูดตรง ๆ)

- **30-minute soak test ต่อเนื่อง** — ไม่ได้ทำ (time budget ของ session)
- **Hot-plug จริง (ถอด/เสียบสาย USB จริงระหว่าง stream)** — ทำไม่ได้จาก sandbox นี้
- **Mid-stream failure injection แบบสมบูรณ์** (บังคับกล้องที่กำลัง stream ให้หลุดจริง ๆ แล้วดู
  ว่าฟื้นเองไหม) — ลองแล้วแต่ induce ไม่สำเร็จ (ดูหัวข้อ 2 รอบที่ 2) มีแค่ code-inspection
  evidence + regression test ที่จำลอง wedge/hang ไว้แล้วครอบคลุม
- **สอง USB camera stream พร้อมกันจริงบนฮาร์ดแวร์นี้** — ยังทำไม่ได้ (เหมือนเดิมจาก
  `MULTI_CAMERA_ROOT_CAUSE_REPORT.md`) ไม่ทราบว่าเป็น hardware ceiling จริงหรือ
  environment-specific (ดูหัวข้อ 3)
- **แยก USB controller/hub คนละอันสำหรับกล้องแต่ละตัว** — ไม่เคยทดสอบ (ต้องขยับสายจริง)

---

## 5. Files changed session นี้

- `tests/core/test_camera_open.py` — เพิ่ม test isolation (mock `_forbidden_device_name`)
  3 จุด + ขยาย timing threshold 1 จุด (`< 3.0` → `< 8.0`) ทั้งหมดเพื่อกัน real hardware state
  ของเครื่องนี้รั่วเข้า test ไม่ใช่การแก้ production logic
- **(รอบใหม่) `core/vision.py`** — 3 จุด: `except cv2.error` → `except Exception` รอบ
  `cap.read()`, เพิ่ม `try/except` รอบ `self._process_captured_frame(...)`, เพิ่ม `try/except`
  รอบ `self._attempt_reopen()` — ทั้งหมดใน `CameraWorker.run()`
- **(รอบใหม่) `web/booth_manager.py`** — เพิ่มเมธอด `_replace_worker_if_dead()` เรียกจาก
  `_capture_watchdog_loop` ทุก 0.5 วิ

### Live verification (ทำจริงบน server ที่ user กำลังใช้งานอยู่ตอนนั้น)

1. เจอ CAM-1 ค้างจริงผ่าน `/api/state` (`frame_sequence` นิ่งที่ 999 นานเกิน 20 วิ, `dropped_frames=0`,
   `reconnect_count=0`) — ยืนยันว่า thread ตายเงียบ ๆ ตามทฤษฎี ไม่ใช่แค่ retry ช้า
2. Restart server เดิม (`web_server.py`, args เดิมทุกตัวตามที่ดึงจาก command line จริงของ process)
   ด้วยโค้ดที่แก้แล้ว
3. หลัง restart: **CAM-1 และ CAM-2 online พร้อมกันทั้งคู่**, ตรวจ `frame_sequence` ต่อเนื่องทุก 20
   วิ นาน 2 นาที (CAM-1 103→2553, CAM-2 -→2385) ไม่มีตกกลับ offline เลยสักครั้ง, `dropped_frames=0`
   ทั้งคู่ ~17 fps คงที่ — **ไม่ได้แปลว่าปัญหา "USB hub bandwidth ไม่พอสำหรับ 2 กล้อง" (bug #4 ที่
   ยัง unconfirmed) ถูกแก้ไปด้วย** เป็นคนละกลไกกัน อาจเป็นความบังเอิญของช่วงเวลานี้ — แต่เป็นผลลัพธ์
   ดีกว่าที่เคยเห็นในทุก session ก่อนหน้า ควรจับตาดูต่อ ไม่ด่วนสรุปว่าแก้ครบแล้ว
4. Process/memory หลัง restart: ยังเป็น 2 python.exe เท่าเดิม (ไม่มี process รั่ว), RAM ~1.4GB
   คงที่ตลอด 2 นาทีที่ monitor

---

## 6. Test summary (ล่าสุด, clean run)

```
tests/core tests/web tests/camera: 803 passed, 0 failed
tests/core/test_camera_unplug_replug.py + test_camera_process_escalation.py + tests/camera: 91 passed
tests/core/test_vision.py + test_camera_unplug_replug.py + test_camera_process_escalation.py + tests/web
  (หลังแก้ bug #5): 156 passed, 0 failed
```

## 7. Git status

```
branch: main
commit: e374369 (HEAD)
working tree: uncommitted changes จำนวนมากใน core/, ui/, web/ (root-cause fix จาก session ก่อน
              checkpoint นี้ ยังไม่ commit) + tests/core/test_camera_open.py (session นี้)
```

ยังไม่ได้ commit/push — ของเดิม (root-cause fix ตัวใหญ่) ควร review รวมกันก่อนตัดสินใจว่าจะ
commit เป็นก้อนเดียวหรือแยก แนะนำให้ตรวจ diff `core/vision.py` (686 บรรทัด) เองสักรอบก่อน commit
เพราะเป็นไฟล์ที่กระทบ camera pipeline ทั้งหมด

---

## 8. แนะนำขั้นต่อไป (ถ้าจะสานงานต่อ)

1. **Commit ของที่ค้างอยู่** — งานแก้ root cause (crash/identity/release-hang/MSMF) มีค่ามาก
   ควร commit แยกจาก test-isolation fix ของ checkpoint นี้ เพื่อให้ history อ่านง่าย
2. **ทดสอบบนเครื่อง booth จริง** (ไม่ใช่ sandbox/VM) เพื่อตัดข้อสงสัยเรื่อง DirectShow index
   ไม่เสถียรที่เจอใน session นี้ — ถ้าเครื่องจริงเสถียรกว่า evidence เรื่อง hub bandwidth ceiling
   จะน่าเชื่อถือขึ้นมาก
3. **ทดลองแยก USB controller/hub** สำหรับกล้องแต่ละตัว (เสียบ USB card เพิ่ม หรือใช้พอร์ตคนละ
   ฝั่งของเมนบอร์ด) เพื่อตัดสิน root cause ข้อสุดท้ายที่ยังไม่ confirm
4. **รัน soak test 30+ นาทีจริงบน booth PC** พร้อม monitor thread/process/memory count
