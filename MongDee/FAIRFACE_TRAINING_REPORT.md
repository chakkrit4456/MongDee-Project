# รายงานผล Full Training โมเดล FairFace (Age / Gender / Race)

เอกสารนี้สรุปผลการเทรนโมเดลจำแนก **อายุ / เพศ / เชื้อชาติ** จาก [FairFace dataset](https://github.com/joojs/fairface)
แบบเต็มรูปแบบ (20 epochs, ข้อมูลทั้งชุดจริง ไม่ใช่ subset) บนเครื่องนี้ — ดูวิธีใช้งานทั่วไปที่
[README.md](README.md#ส่วนที่-fairface--เทรนโมเดล-age--gender--race-และใช้งานกับเว็บแคม)

สถานะ: **เทรนครบ 20/20 epoch สำเร็จ ไม่มี error/OOM/NaN ระหว่างเทรน**

---

## 1. สภาพแวดล้อมที่ใช้เทรน

| รายการ | ค่า |
|---|---|
| OS | Windows 10 |
| Python | 3.14.6 |
| PyTorch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| CUDA runtime | 12.6 |
| GPU | NVIDIA GeForce GTX 1050 (Pascal, compute capability sm_61) |
| VRAM | 3072 MiB |
| Environment | `.venv-fairface-gpu/` (แยกจาก environment หลักของโปรเจกต์ ซึ่งยังคงเป็น CPU-only `torch==2.14.0+cpu` ตามเดิม ไม่ถูกแตะต้อง) |

**หมายเหตุสำคัญเรื่อง GPU:** GTX 1050 เป็นการ์ดรุ่น Pascal — PyTorch build ปกติ (`pip install torch`
หรือ build `cu128` ขึ้นไป) **ไม่รองรับการ์ดรุ่นนี้แล้ว** (จะ error
`CUDA error: no kernel image is available for execution on the device` แม้
`torch.cuda.is_available()` จะขึ้น `True` ก็ตาม) ต้องใช้ build tag `cu126` โดยเฉพาะถึงจะใช้ GPU
ตัวนี้เทรนได้จริง — วิธีติดตั้งอยู่ใน [README.md](README.md#gpu-รุ่นเก่า-pascal-เช่น-gtx-10xx--ต้องใช้-torch-build-ที่ตรงกัน)

---

## 2. Dataset

ยืนยันด้วย `python train.py --check-dataset` (ผลลัพธ์จริง ณ ตอนเทรนเสร็จ):

| รายการ | ค่า |
|---|---|
| Train images | 86,744 |
| Validation images | 10,954 |
| Missing images | 0 |
| Invalid/empty label rows | 0 |
| Age classes (9) | 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70 |
| Gender classes (2) | Male, Female |
| Race classes (7) | White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern |

---

## 3. ค่า Config ที่ใช้เทรน (ตรึงค่าตลอดทั้ง 20 epoch ไม่มีการเปลี่ยนระหว่างทาง)

| Hyperparameter | ค่า |
|---|---|
| Architecture | ResNet34 (backbone แบบ ImageNet-pretrained) |
| Image size | 224×224 |
| Epochs | 20 |
| Batch size | 32 |
| Learning rate | 0.0001 (Adam) |
| num_workers | 4 |
| Loss weights | age=1.0, gender=1.0, race=1.0 (multi-task cross-entropy รวมกัน) |
| AMP (mixed precision) | เปิดใช้งาน (`torch.amp.autocast` + `GradScaler`) |
| Device | CUDA (GTX 1050) |

---

## 4. ผลลัพธ์แต่ละ Epoch (ข้อมูลจริงจาก `logs/fairface_train.csv`)

Overall = ค่าเฉลี่ยของ Age/Gender/Race accuracy (ตัวชี้วัดที่ใช้เลือก best checkpoint)

| Epoch | Train Loss | Val Loss | Age Acc. | Gender Acc. | Race Acc. | Overall | เวลา/epoch |
|------:|-----------:|---------:|---------:|-------------:|----------:|--------:|-----------:|
| 1  | 2.4981 | 2.2308 | 52.64% | 93.19% | 64.04% | 69.96% | 65.0 นาที |
| 2  | 2.0755 | 2.0249 | 56.14% | 94.39% | 67.74% | 72.76% | 40.1 นาที |
| 3  | 1.9252 | 1.9756 | 56.90% | 95.16% | 67.73% | 73.26% | 26.9 นาที |
| 4  | 1.8105 | 1.9544 | 56.48% | 95.40% | 68.93% | 73.60% | 25.7 นาที |
| 5  | 1.7172 | 1.9297 | 57.78% | 95.47% | 69.57% | 74.27% | 25.0 นาที |
| **6**  | **1.6336** | **1.9153** | **58.42%** | **95.18%** | **70.13%** | **74.58%** ⭐ | 25.0 นาที |
| 7  | 1.5487 | 1.9322 | 58.28% | 95.56% | 69.74% | 74.53% | 26.7 นาที |
| 8  | 1.4685 | 2.0110 | 58.41% | 95.50% | 69.26% | 74.39% | 24.9 นาที |
| 9  | 1.3954 | 2.0107 | 58.08% | 95.50% | 69.37% | 74.32% | 26.3 นาที |
| 10 | 1.3184 | 2.0555 | 58.56% | 95.75% | 69.15% | 74.49% | 26.7 นาที |
| 11 | 1.2529 | 2.1046 | 57.96% | 95.32% | 69.03% | 74.10% | 24.7 นาที |
| 12 | 1.1805 | 2.1283 | 57.42% | 95.46% | 69.39% | 74.09% | 28.1 นาที |
| 13 | 1.1165 | 2.2620 | 57.39% | 95.12% | 69.02% | 73.84% | 24.8 นาที |
| 14 | 1.0587 | 2.2664 | 57.94% | 95.29% | 69.19% | 74.14% | 25.1 นาที |
| 15 | 0.9975 | 2.3993 | 57.51% | 95.55% | 67.73% | 73.60% | 25.5 นาที |
| 16 | 0.9490 | 2.4263 | 56.03% | 95.37% | 68.09% | 73.17% | 24.7 นาที |
| 17 | 0.8921 | 2.4908 | 55.21% | 95.45% | 68.35% | 73.01% | 28.3 นาที |
| 18 | 0.8490 | 2.5944 | 56.87% | 95.32% | 68.08% | 73.42% | 25.5 นาที |
| 19 | 0.8003 | 2.6712 | 56.08% | 94.97% | 68.15% | 73.07% | 25.7 นาที |
| 20 | 0.7596 | 2.7588 | 54.32% | 95.66% | 67.80% | 72.59% | 31.8 นาที |

**เวลาเทรนรวม: 34,583 วินาที ≈ 9.6 ชั่วโมง** (เฉลี่ย ~28.8 นาที/epoch, epoch แรกช้ากว่าปกติเพราะ
disk cache ยังไม่ warm)

### ข้อสังเกตเรื่อง Overfitting

Train loss ลดลงต่อเนื่องตลอด (2.50 → 0.76 — โมเดล fit กับข้อมูล train ได้ดีขึ้นเรื่อยๆ) แต่ **Val
loss ต่ำสุดที่ epoch 6 (1.9153) แล้วค่อยๆ สูงขึ้นจนถึง 2.76 ที่ epoch 20** — เป็นรูปแบบ
overfitting ทั่วไปเมื่อไม่มี learning-rate scheduler หรือ early stopping (ค่า config ถูกล็อกไว้ตาม
ที่ authorize ไว้ ไม่ได้ปรับระหว่างเทรน) **นี่คือเหตุผลที่ระบบเลือก `best_model.pt` จาก epoch 6 แทนที่จะใช้
`last_model.pt` (epoch 20)** — checkpoint ระบบทำงานถูกต้องตามที่ออกแบบไว้

---

## 5. Best Model

| รายการ | ค่า |
|---|---|
| Best epoch | 6 |
| Best validation score (overall) | **0.7458** (74.58%) |
| Best checkpoint | `models/fairface/best_model.pt` |
| Last checkpoint | `models/fairface/last_model.pt` (epoch 20) |

**สรุปความแม่นยำของ best model (epoch 6):**
- **Age (ช่วงอายุ, 9 classes):** 58.42%
- **Gender (เพศ, 2 classes):** 95.18%
- **Race (เชื้อชาติ, 7 classes):** 70.13%
- **Overall เฉลี่ย 3 task:** 74.58%

Gender แม่นยำสูงสุด (งานจำแนก 2 class ง่ายกว่า) ส่วน Age เป็น task ที่ยากที่สุด (9 ช่วงอายุ
ใกล้เคียงกันมาก เช่น 20-29 กับ 30-39 แยกยากแม้แต่คนดูเอง) — ตัวเลขนี้สอดคล้องกับ
[ผลการทดลองต้นฉบับของ FairFace paper](https://github.com/joojs/fairface) ที่ก็รายงาน age accuracy
ต่ำกว่า gender/race เช่นกัน

---

## 6. หลักฐานการใช้ GPU จริง (ไม่ใช่แค่ `torch.cuda.is_available()`)

| การตรวจสอบ | ผล |
|---|---|
| `torch.cuda.is_available()` | True |
| Real CUDA tensor/matmul op | ผ่าน (ไม่ error) |
| Real CUDA backward + optimizer.step() | ผ่าน |
| FairFace model forward/backward บน CUDA จริง | ผ่าน (`next(model.parameters()).device` = `cuda:0`) |
| GPU process ปรากฏใน `nvidia-smi` | ใช่ (เห็น PID ของ training process ตลอดการเทรน) |
| GPU utilization ขณะเทรน | สูงสุด 100% (วัดจาก `nvidia-smi` ระหว่างเทรนจริง) |
| VRAM ที่ใช้ | ~1,247 MiB จากทั้งหมด 3,072 MiB (เหลือพอสำหรับ batch size 32 สบายๆ) |
| CUDA error / OOM ระหว่างเทรนเต็ม | ไม่มี |
| NaN / Inf ระหว่างเทรนเต็ม | ไม่มี |

---

## 7. การตรวจสอบ Checkpoint หลังเทรนเสร็จ

| รายการตรวจสอบ | ผล |
|---|---|
| โหลด `best_model.pt` กลับมาใช้ | ผ่าน |
| โหลด `last_model.pt` กลับมาใช้ | ผ่าน |
| `load_state_dict(strict=True)` (ไม่มี key ขาด/เกิน) | ผ่าน |
| Export ไปใช้กับ `core/attributes.py`'s `FairFaceBackend` (ResNet34 + `Linear(512, 18)`) | ผ่าน — โหลดแบบ strict ได้ไม่มี key ขาด/เกิน |
| ทดสอบ inference กับรูปจริงจาก `dataset/val/1.jpg` (ground truth: age 3-9, Male, East Asian) | ทำนาย Age: 3-9 (78.5%, ✅ถูก), Gender: Male (99.6%, ✅ถูก), Race: Middle Eastern (46.9%, ❌ผิด — สมเหตุสมผลเมื่อ race accuracy โดยรวม ~70%) |
| Output เป็นตัวเลขปกติ (ไม่มี NaN/Inf) | ผ่าน |

---

## 8. ไฟล์ผลลัพธ์ (Output Artifacts)

```text
models/fairface/
├── best_model.pt              # checkpoint เต็ม (epoch 6) — มี optimizer state, resume ต่อได้
├── last_model.pt              # checkpoint เต็ม (epoch 20)
├── best_model_state_dict.pt   # state_dict ล้วนจาก best_model.pt ใช้กับ --fairface-checkpoint ของ web_server.py ได้ทันที
├── label_mappings.json        # mapping label -> index ของทั้ง 3 task
└── training_config.json       # ค่า config ที่ใช้เทรนจริง

logs/
└── fairface_train.csv         # log ทุก epoch (20 แถว ไม่มีข้อมูลซ้ำ)
```

**ขนาดไฟล์:** `best_model.pt` / `last_model.pt` ~244 MB ต่อไฟล์, `best_model_state_dict.pt` ~81 MB
(ไฟล์เหล่านี้ถูก gitignore ไว้ — ไม่ถูก commit ขึ้น git เพราะเป็นไฟล์ขนาดใหญ่/generated)

---

## 9. วิธีนำโมเดลไปใช้งานต่อ

```bash
# ทดสอบกับรูปเดียว
python test_model.py --image path/to/image.jpg

# ใช้งานกับเว็บแคมแบบ real-time (รองรับ 2 กล้อง)
python detect.py

# เทรนต่อจาก epoch ที่ดีที่สุด (ถ้าต้องการปรับปรุงต่อ เช่น เพิ่ม LR scheduler)
python train.py --resume models/fairface/last_model.pt
```

`models/fairface/best_model_state_dict.pt` ใช้แทน
`res34_fair_align_multi_7_20190809.pt` ได้ทันทีกับ `--fairface-checkpoint` ของ `web_server.py`
(โมเดลเดียวกันในเชิงโครงสร้าง เพียงแต่เทรนจากข้อมูลของโปรเจกต์นี้เอง)

---

## 10. ข้อเสนอแนะสำหรับการเทรนรอบถัดไป (ถ้าต้องการความแม่นยำสูงขึ้น)

จากรูปแบบ overfitting ที่เห็นใน val loss (ข้อ 4) แนวทางที่น่าจะช่วยได้โดยไม่ต้องเปลี่ยนสถาปัตยกรรม:
- เพิ่ม learning-rate scheduler (เช่น cosine หรือ step decay) แทนการใช้ LR คงที่ 0.0001 ตลอด
- เพิ่ม early stopping หรือหยุดที่ราว epoch 6-10 (จุดที่ val loss ต่ำสุด) แทนที่จะฝืนเทรนครบ 20
- เพิ่ม weight decay หรือ dropout เพื่อลด overfitting
- Age accuracy ต่ำสุดในสาม task — อาจพิจารณาจัดกลุ่มช่วงอายุใหม่ให้กว้างขึ้น หากงานที่ใช้จริงไม่ต้องการความละเอียดถึง 9 ช่วง

รายการนี้เป็นข้อเสนอแนะเท่านั้น ยังไม่ได้ทดลอง — การเทรนที่รายงานในเอกสารนี้ใช้ค่า config ที่ authorize
ไว้แบบตรึงค่าตลอด (section 3) ตามที่ตกลงกันไว้
