"""Adds ~30 display-only mock products (with drawn cover images) for UI testing.

    venv\\Scripts\\python.exe scripts\\seed_mock_products.py           # add
    venv\\Scripts\\python.exe scripts\\seed_mock_products.py --remove  # delete exactly what it added

Mock products have no embeddings, so the kiosk cannot recognize them (the
admin list flags them "ต้องสแกนใหม่"). The ids it created are kept in
data/mock_products.json so --remove never touches real products.
"""
import colorsys
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app import config, db  # noqa: E402

REGISTRY = config.DATA_DIR / "mock_products.json"
COVERS = config.CAPTURES_DIR / "covers"

# name, category, kind, price, origin, material, process, story (None -> generated)
PRODUCTS = [
    ("กระเป๋าสานย่านลิเภา", "ของใช้", "bag", 1290, "นครศรีธรรมราช", "เถาย่านลิเภา หวายแท้", "จักสานด้วยมือทีละเส้น ใช้เวลาประมาณ 14 วัน", None),
    ("ตะกร้าไม้ไผ่ลายขัด", "ของใช้", "bag", 260, "เชียงใหม่", "ไม้ไผ่บ้าน", "จักสานลายขัดด้วยมือ", None),
    ("ผ้าไหมมัดหมี่ผืนใหญ่", "ผ้าและเครื่องแต่งกาย", "cloth", 2850, "สุรินทร์", "ไหมแท้ 100% ย้อมสีธรรมชาติ", "มัดย้อมและทอด้วยกี่พื้นบ้าน", None),
    ("ผ้าฝ้ายทอมือย้อมครามธรรมชาติ", "ผ้าและเครื่องแต่งกาย", "cloth", 690, "สกลนคร", "ฝ้ายแท้ ครามธรรมชาติ", "ย้อมคราม 7 ครั้งแล้วตากแดด", None),
    ("ผ้าขาวม้าลายสก๊อต", None, "cloth", 150, "อุบลราชธานี", "ฝ้าย", "ทอมือ", ""),
    ("น้ำผึ้งดอกลำไยแท้ 100%", "อาหาร", "jar", 220, "ลำพูน", "น้ำผึ้งดอกลำไย", "เก็บจากรังผึ้งเลี้ยง กรองด้วยผ้าขาวบาง", None),
    ("น้ำพริกหนุ่มแม่ประไพ", "อาหาร", "jar", 65, "เชียงราย", "พริกหนุ่ม หอมแดง กระเทียม", "ย่างแล้วโขลกสด ไม่ใส่วัตถุกันเสีย", None),
    ("กล้วยตากบางกระทุ่ม", "อาหาร", "box", 89.5, "พิษณุโลก", "กล้วยน้ำว้าสุก", "ตากแดดธรรมชาติ 3 แดด", None),
    ("มะม่วงกวนรสธรรมชาติ", "อาหาร", "box", 75, "ราชบุรี", "มะม่วงสุก น้ำตาลทรายแดง", "กวนด้วยไฟอ่อนแล้วตัดเป็นชิ้น", None),
    ("ข้าวกล้องหอมมะลิอินทรีย์ 5 กก.", "อาหาร", "box", 320, "ยโสธร", "ข้าวหอมมะลิ 105", "สีกล้อง ไม่ขัดขาว", None),
    ("ชาใบหม่อนอบแห้ง", "เครื่องดื่ม", "box", 120, "น่าน", "ใบหม่อนอ่อน", "อบแห้งด้วยความร้อนต่ำ", None),
    ("กาแฟดอยช้างคั่วกลาง", "เครื่องดื่ม", "jar", 350, "เชียงราย", "เมล็ดกาแฟอาราบิก้า", "คั่วสดตามออเดอร์", None),
    ("น้ำสมุนไพรอัญชันมะนาว", "เครื่องดื่ม", "bottle", 35, "นครปฐม", "ดอกอัญชัน มะนาว", "ต้มสกัดและบรรจุขวด", None),
    ("สบู่ก้อนน้ำมันมะพร้าวสกัดเย็น", "ของใช้", "box", None, "ชุมพร", "น้ำมันมะพร้าว", "หมักบ่มนาน 4 สัปดาห์", None),
    ("แชมพูอัญชันมะกรูด", "ผลิตภัณฑ์ดูแลผม", "bottle", 180, "จันทบุรี", "อัญชัน มะกรูด", "สกัดเย็นแล้วผสมสูตรอ่อนโยน", None),
    ("เทียนหอมกลิ่นดอกจำปา", "ของตกแต่งบ้าน", "jar", 260, "เพชรบุรี", "ไขถั่วเหลือง น้ำมันหอมระเหย", "เทมือทีละชิ้น", None),
    ("โคมไฟไม้สักแกะสลัก", "ของตกแต่งบ้าน", "bowl", 3400, "แพร่", "ไม้สัก", "แกะสลักลายด้วยมือ", None),
    ("ชามเซรามิกเคลือบเขียวไข่กา", "เครื่องปั้นดินเผา", "bowl", 480, "ลำปาง", "ดินขาวลำปาง", "เผาที่อุณหภูมิ 1,250 องศา", None),
    ("แจกันดินเผาบ้านเชียง", "เครื่องปั้นดินเผา", "bottle", 1650, "อุดรธานี", "ดินเผา", "ปั้นขึ้นรูปด้วยมือ เขียนลายสีแดง", None),
    ("ถ้วยกาแฟลายคราม (คู่)", "เครื่องปั้นดินเผา", "bowl", 390, "ราชบุรี", "เซรามิก", "เขียนลายครามด้วยมือ", None),
    ("ตุ๊กตาไม้แกะสลักช้างไทย", "ของที่ระลึก", "box", 750, "ลำปาง", "ไม้มะม่วง", "แกะสลักและขัดเงา", None),
    ("พวงกุญแจผ้าลายไทย", "ของที่ระลึก", "box", 29, None, "ผ้าฝ้ายพิมพ์ลาย", "เย็บมือ", None),
    ("หมวกปีกกว้างสานใบลาน", "ผ้าและเครื่องแต่งกาย", "bowl", 350, "สุพรรณบุรี", "ใบลาน", "สานลายขัดและขึ้นทรง", None),
    ("รองเท้าแตะหนังแท้เย็บมือ", "ผ้าและเครื่องแต่งกาย", "box", 890, "ขอนแก่น", "หนังวัวฟอกฝาด", "ตัดเย็บและตอกหมุดด้วยมือ", None),
    ("ชุดเครื่องประดับเงินลายดอกพิกุลสร้อยและต่างหูสำหรับงานมงคลสมรสและพิธีสำคัญ", "เครื่องประดับ", "box", 12500, "เชียงใหม่", "เงินแท้ 92.5%", "ตีขึ้นรูปและสลักลายด้วยมือ", "ชุดเครื่องประดับเงินลายดอกพิกุลตามแบบโบราณ ช่างเงินใช้เวลาหลายสัปดาห์ในการตีขึ้นรูปและสลักลายทีละดอก เหมาะสำหรับสวมใส่ในงานมงคลสมรสและพิธีสำคัญ พร้อมกล่องผ้าไหมสำหรับมอบเป็นของขวัญ"),
    ("ผ้าคลุมไหล่ไหมยกดอก", "ผ้าและเครื่องแต่งกาย", "cloth", 4200, "น่าน", "ไหมแท้", "ทอยกดอกด้วยกี่โบราณ", None),
    ("ขนมทองม้วนกรอบ", "อาหาร", "box", 55, "อ่างทอง", "แป้งข้าวเจ้า น้ำตาลมะพร้าว", "ม้วนด้วยมือขณะร้อน", None),
    ("ปลาร้าบองสูตรโบราณ", "อาหาร", "jar", 80, "มหาสารคาม", "ปลาร้า ข่า ตะไคร้", "หมักและโขลกตามสูตรเดิม", None),
    ("น้ำมันมะพร้าวสกัดเย็นบริสุทธิ์", "ผลิตภัณฑ์ดูแลผิว", "bottle", 240, "สมุทรสงคราม", "มะพร้าวน้ำหอม", "สกัดเย็นโดยไม่ผ่านความร้อน", None),
    ("กระเช้าของขวัญปีใหม่ MONGDEE-GIFT-SET-PREMIUM-EDITION-2569-LIMITED", "ของขวัญ", "box", 1999, "กรุงเทพมหานคร", "รวมสินค้า OTOP 6 ชิ้น", "คัดสรรและจัดชุดโดยทีมงาน", None),
]

STORIES = [
    "ผลิตโดยกลุ่มวิสาหกิจชุมชน ใช้วัตถุดิบในท้องถิ่น สืบทอดภูมิปัญญาจากรุ่นสู่รุ่น",
    "ทำมือทีละชิ้น จึงไม่มีชิ้นไหนเหมือนกัน เหมาะเป็นของฝากและของใช้ในบ้าน",
    "คัดวัตถุดิบคุณภาพดี ผ่านขั้นตอนตามสูตรดั้งเดิม ปลอดภัยและใช้งานได้นาน",
]


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_cover(kind, index):
    hue = (index * 0.083) % 1.0
    scale = 2
    w, h = 1200 * scale, 900 * scale
    img = Image.new("RGB", (w, h))
    px = ImageDraw.Draw(img)
    top, bottom = hsv(hue, 0.10, 0.99), hsv(hue, 0.20, 0.90)
    for y in range(h):
        t = y / h
        px.line([(0, y), (w, y)], fill=tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))

    cx, base = w // 2, int(h * 0.80)
    shadow = Image.new("L", (w, h), 0)
    ImageDraw.Draw(shadow).ellipse([cx - 380 * scale, base - 30 * scale, cx + 380 * scale, base + 50 * scale], fill=110)
    shadow = shadow.filter(ImageFilter.GaussianBlur(28 * scale))
    img = Image.composite(Image.new("RGB", (w, h), (30, 30, 40)), img, shadow)
    px = ImageDraw.Draw(img)

    body, dark, light = hsv(hue + 0.02, 0.60, 0.78), hsv(hue + 0.02, 0.70, 0.55), hsv(hue + 0.02, 0.30, 0.95)
    s = scale

    if kind == "jar":
        px.rounded_rectangle([cx - 230 * s, base - 430 * s, cx + 230 * s, base], radius=60 * s, fill=body)
        px.rounded_rectangle([cx - 250 * s, base - 520 * s, cx + 250 * s, base - 420 * s], radius=30 * s, fill=dark)
        px.rounded_rectangle([cx - 150 * s, base - 320 * s, cx + 150 * s, base - 120 * s], radius=30 * s, fill=light)
    elif kind == "bottle":
        px.rounded_rectangle([cx - 170 * s, base - 470 * s, cx + 170 * s, base], radius=70 * s, fill=body)
        px.rectangle([cx - 70 * s, base - 620 * s, cx + 70 * s, base - 460 * s], fill=body)
        px.rounded_rectangle([cx - 90 * s, base - 690 * s, cx + 90 * s, base - 610 * s], radius=20 * s, fill=dark)
        px.rounded_rectangle([cx - 120 * s, base - 330 * s, cx + 120 * s, base - 130 * s], radius=20 * s, fill=light)
    elif kind == "bag":
        px.arc([cx - 170 * s, base - 620 * s, cx + 170 * s, base - 260 * s], 180, 360, fill=dark, width=34 * s)
        px.polygon([(cx - 260 * s, base - 380 * s), (cx + 260 * s, base - 380 * s), (cx + 320 * s, base), (cx - 320 * s, base)], fill=body)
        px.rectangle([cx - 260 * s, base - 400 * s, cx + 260 * s, base - 350 * s], fill=dark)
        for i in range(-2, 3):
            px.line([(cx + i * 90 * s, base - 350 * s), (cx + i * 105 * s, base)], fill=dark, width=8 * s)
    elif kind == "bowl":
        px.pieslice([cx - 330 * s, base - 420 * s, cx + 330 * s, base + 100 * s], 0, 180, fill=body)
        px.ellipse([cx - 330 * s, base - 260 * s, cx + 330 * s, base - 130 * s], fill=dark)
        px.rounded_rectangle([cx - 150 * s, base + 20 * s, cx + 150 * s, base + 70 * s], radius=20 * s, fill=dark)
    elif kind == "cloth":
        px.rounded_rectangle([cx - 340 * s, base - 260 * s, cx + 340 * s, base], radius=40 * s, fill=body)
        for i in range(8):
            x = cx - 340 * s + i * 85 * s
            px.rectangle([x, base - 260 * s, x + 40 * s, base], fill=dark if i % 2 == 0 else light)
        px.ellipse([cx - 380 * s, base - 300 * s, cx - 280 * s, base + 40 * s], fill=dark)
    else:  # box
        px.rounded_rectangle([cx - 300 * s, base - 380 * s, cx + 300 * s, base], radius=30 * s, fill=body)
        px.rectangle([cx - 40 * s, base - 380 * s, cx + 40 * s, base], fill=light)
        px.rectangle([cx - 300 * s, base - 240 * s, cx + 300 * s, base - 170 * s], fill=light)
        px.polygon([(cx, base - 380 * s), (cx - 110 * s, base - 470 * s), (cx - 40 * s, base - 380 * s)], fill=dark)
        px.polygon([(cx, base - 380 * s), (cx + 110 * s, base - 470 * s), (cx + 40 * s, base - 380 * s)], fill=dark)

    return img.resize((1200, 900), Image.LANCZOS)


def seed():
    if REGISTRY.exists():
        sys.exit(f"{REGISTRY} already exists — run with --remove first to avoid duplicates.")
    COVERS.mkdir(parents=True, exist_ok=True)
    db.init_db()
    created = []
    for i, (name, category, kind, price, origin, material, process, story) in enumerate(PRODUCTS):
        cover_name = f"mock_cover_{i + 1:02d}.jpg"
        draw_cover(kind, i).save(COVERS / cover_name, quality=90)
        cover_path = f"captures/covers/{cover_name}"
        if story is None:
            story = f"{name} {STORIES[i % len(STORIES)]}"
        month = 1 + i % 9
        production = f"2026-{month:02d}-{1 + i % 27:02d}" if i % 3 else None
        expiry = f"2027-{month:02d}-{1 + i % 27:02d}" if production and category in ("อาหาร", "เครื่องดื่ม") else None
        contact = f"https://example.com/shop/{i + 1}" if i % 4 == 0 else None
        product_id = db.create_product(
            name, category, origin, material, process, story, contact, cover_path,
            price=price, production_date=production, expiry_date=expiry,
        )
        db.set_cover_image(product_id, cover_path)
        created.append({"id": product_id, "cover": cover_path})
    REGISTRY.write_text(json.dumps(created, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"added {len(created)} mock products (ids {created[0]['id']}..{created[-1]['id']})")


def remove():
    if not REGISTRY.exists():
        sys.exit("nothing to remove (no data/mock_products.json)")
    created = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for entry in created:
        db.delete_product(entry["id"])
        (config.DATA_DIR / entry["cover"]).unlink(missing_ok=True)
    REGISTRY.unlink()
    print(f"removed {len(created)} mock products")


if __name__ == "__main__":
    remove() if "--remove" in sys.argv else seed()
