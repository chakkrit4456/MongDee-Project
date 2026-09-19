# MongDee Master Prompt — FairFace Age/Gender Attribute Analysis

## เป้าหมาย

แก้ไขโปรเจกต์ MongDee ที่มีอยู่แล้ว เพื่อเพิ่มระบบวิเคราะห์ **เพศที่โมเดลคาดการณ์ (Male/Female)** และ **ช่วงอายุ (Age Group)** จากใบหน้าของบุคคล โดยต้องทำงานร่วมกับระบบเดิม:

- YOLO Person Detection
- PersonTracker
- Tripwire
- Re-ID / Global Person ID
- Multi-camera
- Dashboard
- GPU/CPU fallback
- Camera capture/reconnect
- Existing tests

**ห้ามรื้อระบบเดิมที่ทำงานอยู่แล้ว และห้ามสร้างระบบตรวจคนใหม่ซ้ำซ้อน**

---

# 1. Dataset / Model ที่ต้องใช้

ใช้ **FairFace** เป็น baseline สำหรับ age/gender attribute model

Official GitHub:
https://github.com/joojs/fairface

Hugging Face dataset:
https://huggingface.co/datasets/HuggingFaceM4/FairFace

FairFace มีภาพประมาณ 108,501 ภาพ และ label สำหรับ age, gender และ race โดย age มี 9 กลุ่ม:

- 0-2
- 3-9
- 10-19
- 20-29
- 30-39
- 40-49
- 50-59
- 60-69
- more than 70

Gender:
- Male
- Female

Dataset license: CC BY 4.0

Official pretrained model / resources:
https://github.com/joojs/fairface

ให้ตรวจสอบ license และเงื่อนไขการใช้งานก่อนนำ model/data ไป deploy จริง

**ห้ามดาวน์โหลด dataset ขนาดใหญ่โดยอัตโนมัติถ้าไม่จำเป็น**
ให้ตรวจสอบก่อนว่า pretrained model สามารถใช้งานได้เพียงพอหรือไม่

---

# 2. Architecture ที่ต้องการ

Camera
↓
YOLO Person Detection
↓
PersonTracker
↓
Local Track ID
↓
Re-ID
↓
Global Person ID
↓
Face Detection / Face Crop
↓
FairFace Age/Gender Model
↓
Temporal Smoothing
↓
Attribute Result
↓
Dashboard / Analytics

**YOLO ยังตรวจ class `person` เหมือนเดิม**

ห้ามเปลี่ยน YOLO class เป็น male/female/child/adult เพราะ person detector กับ attribute classifier เป็นคนละหน้าที่

---

# 3. สร้าง Attribute Analyzer

เพิ่ม module ใหม่โดยรักษาโครงสร้าง project ปัจจุบัน

แนะนำ:

```text
vision/
  attributes/
    __init__.py
    fairface.py
    age_gender.py
```

หรือเลือกตำแหน่งที่เหมาะกับ architecture ปัจจุบันหลังจากตรวจ repository แล้ว

ต้องมี API ประมาณ:

```python
result = analyzer.predict(face_crop)
```

ผลลัพธ์:

```python
{
    "gender": "Male",
    "gender_confidence": 0.91,
    "age_group": "20-29",
    "age_confidence": 0.74,
    "status": "ok"
}
```

หากไม่สามารถวิเคราะห์ได้ ให้คืน UNKNOWN แทนการเดา

---

# 4. Face Detection

ห้ามส่ง full-body crop เข้า FairFace โดยตรง

ต้อง:

person bbox → face detector → face bbox → face crop → FairFace

ตรวจสอบก่อนว่า project มี face detector อยู่แล้วหรือไม่ ถ้ามีให้ reuse ถ้าไม่มีให้เลือก lightweight face detector ที่เหมาะกับ GTX 1050

ต้องมี:

- minimum face size
- face confidence threshold
- invalid crop protection
- image boundary clipping

---

# 5. Age Group

เก็บ age group แบบละเอียด:

```text
0-2
3-9
10-19
20-29
30-39
40-49
50-59
60-69
70+
```

จากนั้นค่อย map เป็น business category:

```python
def to_age_category(age_group):
    if age_group in ["0-2", "3-9", "10-19"]:
        return "CHILD"

    if age_group in [
        "20-29", "30-39", "40-49",
        "50-59", "60-69", "70+",
        "more than 70"
    ]:
        return "ADULT"

    return "UNKNOWN"
```

ต้องเก็บทั้ง `age_group` และ `age_category` เพื่อเปลี่ยน business rule ภายหลังได้

---

# 6. Gender

ผลลัพธ์:

```text
Male
Female
UNKNOWN
```

ห้ามบังคับ classification หาก confidence ต่ำ

threshold ต้องเป็น config ไม่ hard-code กระจายหลายไฟล์ เช่น:

```text
ATTRIBUTE_GENDER_MIN_CONFIDENCE
ATTRIBUTE_AGE_MIN_CONFIDENCE
```

อย่าเปลี่ยน threshold แบบสุ่มเพื่อให้ test ผ่าน

---

# 7. Temporal Smoothing

ห้ามใช้ผลจาก frame เดียวเป็น profile สุดท้าย

สำหรับแต่ละ Global Person ID ให้เก็บผลหลาย frame แล้ว aggregate

ตัวอย่าง:

```text
P103

Frame 1: Male 0.82 / 20-29 0.71
Frame 2: Male 0.91 / 20-29 0.76
Frame 3: Male 0.89 / 30-39 0.55
Frame 4: Female 0.51 / 20-29 0.79

Final:
Gender = Male
Age = 20-29
```

ต้องมี aggregate confidence และ sample count

---

# 8. วิเคราะห์เฉพาะ Global Person

ห้ามรัน FairFace ทุก frame

ใช้ Global Person ID เป็น key:

```text
CAM-01 Track-17
       ↓
Global P103
       ↓
Attribute Analyzer
```

เมื่อคนเดิมปรากฏใน CAM-02:

```text
CAM-02 Track-4
       ↓
Re-ID
       ↓
Global P103
       ↓
reuse / update attribute
```

เป้าหมายคือประหยัด GPU บน GTX 1050

---

# 9. Attribute Cache

สร้าง in-memory cache สำหรับ:

```text
global_person_id
gender
gender_confidence
age_group
age_category
age_confidence
sample_count
last_updated
status
```

ห้าม query SQLite ทุก frame

Database ใช้สำหรับ persistence/analytics ไม่ใช่ hot path

---

# 10. Track Quality Gate

ก่อนวิเคราะห์ face ต้องตรวจ:

- track confidence
- bbox size
- face size
- blur
- brightness
- occlusion
- face confidence
- pose / visibility ถ้ามี

หากคุณภาพไม่พอ ให้ UNKNOWN

---

# 11. GTX 1050 Optimization

รักษา `resolve_device()` และ CUDA smoke test/CPU fallback ของระบบเดิม

ห้ามทำให้ server crash หาก CUDA ใช้งานไม่ได้

ห้ามสร้าง CUDA context หลายครั้งโดยไม่จำเป็น

---

# 12. AI Throttle

Attribute analysis ต้อง throttle ไม่ต้องวิเคราะห์ทุก frame

ค่าเริ่มต้นแนะนำ:

```text
ATTRIBUTE_ANALYSIS_INTERVAL=0.75
```

หรือ 0.5–1.0 sec / Global Person

ต้องเป็น configuration

ห้าม block camera capture thread

---

# 13. Tripwire Integration

เมื่อคนผ่าน tripwire ให้ event สามารถเก็บ attribute snapshot ได้:

```json
{
  "global_person_id": "P103",
  "gender": "Male",
  "age_group": "20-29",
  "age_category": "ADULT",
  "gender_confidence": 0.91,
  "age_confidence": 0.76
}
```

แต่:

**หนึ่ง crossing = หนึ่ง event**

ห้าม attribute inference ทำให้ tripwire count ซ้ำ

---

# 14. Database

ตรวจ schema ปัจจุบันก่อน หากมี person/global identity tables ให้ reuse

ถ้าต้องเพิ่ม ให้สร้าง migration ใหม่ ห้ามแก้ migration production เดิมย้อนหลังโดยไม่จำเป็น

แนะนำ:

```text
person_attributes
-----------------
id
global_person_id
gender
gender_confidence
age_group
age_category
age_confidence
sample_count
first_seen
last_seen
created_at
updated_at
```

อย่าเก็บ raw face image โดย default

---

# 15. Privacy / Data Minimization

ระบบนี้เป็นการประเมิน age/gender จากภาพบุคคล จึงต้อง:

- ไม่เก็บภาพใบหน้าโดย default
- ไม่เก็บ face crop ถ้าไม่จำเป็น
- เก็บ prediction + confidence เท่าที่จำเป็น
- มี UNKNOWN
- มี config ปิด attribute analysis
- ตรวจสอบสิทธิ์/นโยบาย/การแจ้งให้ทราบที่เกี่ยวข้อง โดยเฉพาะเมื่อมีกลุ่มผู้เยาว์
- ไม่ใช้ prediction เป็นหลักฐานยืนยันตัวตนหรืออัตลักษณ์
- UI/DB ควรสื่อว่าเป็น `model-predicted gender`

---

# 16. Dashboard

เพิ่มข้อมูลใน person/camera UI โดยไม่ทำให้ dashboard เดิมเสีย:

```text
Person P103

เพศที่โมเดลคาดการณ์:
Male 91%

ช่วงอายุ:
20-29 76%

กลุ่ม:
ADULT

ตัวอย่าง:
12 frames
```

หากไม่มั่นใจ:

```text
Gender: UNKNOWN
Age: UNKNOWN
```

ห้ามแสดงเหมือนเป็นผลที่ถูกต้อง 100%

---

# 17. Analytics

เพิ่มแบบ optional:

```text
People observed
Male predicted
Female predicted
Unknown gender

Age:
Child
Adult
Unknown
```

ต้องแยกจาก:

```text
Tripwire IN
Tripwire OUT
Unique Global Persons
```

ห้ามใช้ attribute totals แทนจำนวนคนจริง

---

# 18. Re-ID Interaction

รักษาหลัก:

```text
Local Track ID != Global Person ID
```

Attribute ต้องผูกกับ Global Person ID ไม่ใช่ local Track ID

---

# 19. Testing

ทุก test ต้องใช้ isolated temporary DB

เพิ่ม production guard:

```python
if is_production_database(db_path):
    raise RuntimeError(
        "REFUSING TO RUN TESTS AGAINST PRODUCTION DATABASE"
    )
```

Required tests:

### Attribute
- model loads
- CPU fallback
- CUDA fallback
- valid face crop
- invalid crop
- no face
- low confidence
- gender prediction
- age prediction
- UNKNOWN behavior

### Smoothing
- multiple frames
- conflicting predictions
- dominant prediction
- confidence aggregation
- stale person

### Re-ID
- same global person across cameras
- local ID changes but global ID stays
- unknown identity
- duplicate prevention

### Tripwire
- one crossing = one event
- attribute inference doesn't create duplicate crossing
- IN/OUT remains correct

### Performance
- attribute worker doesn't block camera capture
- queue doesn't grow without limit
- stale tasks are discarded safely

### Production safety
- test cannot use production DB
- no production data mutation
- no booth deletion
- no booth settings overwrite

---

# 20. Real Camera Validation

หลัง tests ผ่าน ต้องทดสอบกับกล้องจริงเมื่อ hardware พร้อม:

- person walking toward camera
- person walking away
- tripwire crossing
- standing
- partial occlusion
- far distance
- poor lighting
- multiple people
- same person across two cameras
- permitted child/minor scenarios
- adult
- different viewing angles

ห้าม claim 100% accuracy

รายงาน:

```text
Total samples
Known predictions
Unknown predictions
Correct
Incorrect
Precision
Recall
F1
```

รายงาน age/gender แยกกัน

---

# 21. Domain Validation

FairFace เป็น baseline ไม่ใช่ guarantee สำหรับ CCTV/booth

หากมีข้อมูลจริงของ MongDee ที่ได้รับอนุญาตให้ใช้ ให้สร้าง domain validation dataset:

```text
datasets/
  mongdee_attributes/
    train/
    val/
    test/
```

ห้ามมีภาพเดียวกันใน train และ test

หาก fine-tune ให้บันทึก:

```text
model version
dataset version
threshold
accuracy
precision
recall
F1
```

---

# 22. ห้ามทำ

ห้าม:

- เปลี่ยน YOLO person detector เป็น gender detector
- วิเคราะห์ทุก frame
- query DB ทุก frame
- block camera capture ด้วย FairFace
- force prediction เมื่อ confidence ต่ำ
- ลบ Re-ID
- ลบ Tripwire
- ลบ camera reconnect
- ลบ GPU fallback
- แก้ migration เดิมย้อนหลังโดยไม่จำเป็น
- test บน production DB
- reset production database
- overwrite booth settings
- ลบข้อมูล booth จริง
- claim accuracy 100%
- เพิ่ม dependency หนักโดยไม่ตรวจ GTX 1050
- ดาวน์โหลด dataset ใหญ่โดยไม่จำเป็น
- เก็บ face images เป็นค่าเริ่มต้น

---

# 23. Implementation Strategy

1. Inspect repository
2. Inspect detector/tracker/Re-ID
3. Inspect device.py
4. Inspect database/migrations
5. Inspect dashboard
6. ตรวจว่ามี face detector อยู่แล้วหรือไม่
7. Implement Attribute Analyzer
8. FairFace integration
9. Confidence thresholds
10. Temporal smoothing
11. Global Person attribute cache
12. Re-ID integration
13. Tripwire integration
14. Database persistence
15. Dashboard
16. Tests
17. Complete test suite
18. Real-camera validation

ห้ามเดาไฟล์/class/function ต้องค้นจาก repository จริงก่อนแก้

---

# 24. Acceptance Criteria

- [ ] FairFace integration ทำงานจริง
- [ ] Person detector เดิมยังทำงาน
- [ ] PersonTracker เดิมยังทำงาน
- [ ] Re-ID เดิมยังทำงาน
- [ ] Tripwire เดิมยังทำงาน
- [ ] Camera capture ไม่ถูก block
- [ ] GPU fallback ยังทำงาน
- [ ] CPU fallback ยังทำงาน
- [ ] Attribute inference มี throttle
- [ ] Attribute มี confidence
- [ ] Low confidence = UNKNOWN
- [ ] Temporal smoothing ทำงาน
- [ ] Global Person รักษา attribute ข้าม camera ได้
- [ ] Tripwire event ไม่ถูกนับซ้ำ
- [ ] Dashboard แสดงผล
- [ ] ไม่มี raw face image persistence โดย default
- [ ] Production DB ไม่ถูกแตะระหว่าง tests
- [ ] Existing tests ผ่าน
- [ ] New tests ผ่าน
- [ ] Real camera smoke test ผ่านเมื่อ hardware พร้อม

---

# 25. Final Report

รายงาน:

```text
Files changed:
...

Files added:
...

Model:
...

Dataset:
FairFace

Age classes:
...

Gender classes:
...

Threshold:
...

Attribute interval:
...

Device:
CUDA / CPU

Tests:
XXX passed

Real camera:
PASS / FAIL / NOT RUN

Known limitations:
...
```

ห้ามรายงานว่า accuracy ดี/แม่น หากยังไม่ได้ทำ real validation

---

# 26. Priority

P0:
- Production DB safety
- Existing system preservation
- CPU/GPU safety

P1:
- FairFace integration
- Face detection
- Attribute prediction
- UNKNOWN handling

P2:
- Temporal smoothing
- Global Person integration
- Re-ID integration

P3:
- Tripwire
- Database
- Dashboard

P4:
- Performance
- Tests
- Real-camera validation

---

# Dataset / Official Resources

FairFace official GitHub:
https://github.com/joojs/fairface

FairFace dataset on Hugging Face:
https://huggingface.co/datasets/HuggingFaceM4/FairFace

FairFace official pretrained model/resources:
https://github.com/joojs/fairface

Do not substitute an unrelated dataset without documenting why.

# END MASTER PROMPT
