"""Excel report of scan analytics (one workbook, several sheets)."""
from datetime import datetime
from io import BytesIO
from typing import Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from app import db

RANGE_LABELS = {"today": "วันนี้", "7d": "7 วันล่าสุด", "30d": "30 วันล่าสุด", "all": "ทั้งหมด"}
_HEADER_FILL = PatternFill("solid", fgColor="0A63FF")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"


def _header(ws, titles, widths):
    ws.append(titles)
    for cell in ws[1]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    ws.freeze_panes = "A2"


def _when(timestamp: Optional[float]):
    return datetime.fromtimestamp(timestamp) if timestamp else None


def build_report(range_key: str, now: Optional[datetime] = None) -> bytes:
    now = now or datetime.now()
    overview = db.analytics_overview(range_key)
    start = db.range_start(range_key, now=now)

    wb = Workbook()

    # --- Summary ---------------------------------------------------------
    ws = wb.active
    ws.title = "สรุป"
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 30
    ws["A1"] = "รายงานการสแกน MONGDEE MINI KIOSK"
    ws["A1"].font = Font(bold=True, size=14)
    period_start = "ตั้งแต่เริ่มใช้งาน" if not start else datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M")
    rows = [
        ("ช่วงเวลา", RANGE_LABELS.get(range_key, range_key)),
        ("เริ่มนับตั้งแต่", period_start),
        ("ออกรายงานเมื่อ", now.strftime("%Y-%m-%d %H:%M:%S")),
        (None, None),
        ("สแกนสำเร็จ (ครั้ง)", overview["successful"]),
        ("สแกนไม่รู้จัก / ไม่แน่ใจ (ครั้ง)", overview["unmatched"]),
        ("อัตราสแกนสำเร็จ", overview["success_rate"]),
        ("สินค้าที่มีคนสแกน", overview["products_scanned"]),
        ("สินค้าทั้งหมดในระบบ", overview["products_total"]),
        ("พนักงานแก้ผลสแกนเอง (ครั้ง)", overview["corrected"]),
        ("ไม่นับ: สแกนของสินค้าที่ถูกลบแล้ว (ครั้ง)", overview["orphaned"]),
    ]
    for label, value in rows:
        ws.append([label, value])
    for cell in ws["A"][3:]:
        cell.font = Font(bold=True)
    rate_cell = ws.cell(row=9, column=2)
    rate_cell.number_format = "0.0%"
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=2, max_col=2):
        row[0].alignment = Alignment(horizontal="right")

    # --- Products --------------------------------------------------------
    ws = wb.create_sheet("สินค้า")
    _header(ws, ["สินค้า", "หมวดหมู่", "ราคา (บาท)", "จำนวนสแกน", "ความคล้ายเฉลี่ย", "สแกนล่าสุด"], [40, 22, 14, 14, 16, 22])
    for r in db.analytics_product_rows(range_key):
        ws.append([r["name"], r["category"], r["price"], r["scan_count"], r["avg_confidence"], _when(r["last_scan"])])
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
        row[0].number_format = "0%"
    for row in ws.iter_rows(min_row=2, min_col=6, max_col=6):
        row[0].number_format = _DATETIME_FORMAT

    # --- Daily -----------------------------------------------------------
    ws = wb.create_sheet("รายวัน")
    _header(ws, ["วันที่", "จำนวนสแกน"], [16, 14])
    for r in db.analytics_daily(range_key, today=now.date()):
        ws.append([datetime.fromisoformat(r["day"]), r["count"]])
    for row in ws.iter_rows(min_row=2, max_col=1):
        row[0].number_format = "yyyy-mm-dd"

    # --- Hourly ----------------------------------------------------------
    ws = wb.create_sheet("รายชั่วโมง")
    _header(ws, ["ช่วงเวลา", "จำนวนสแกน"], [18, 14])
    by_hour = {r["hour"]: r["count"] for r in db.analytics_hourly(range_key)}
    for hour in range(24):
        ws.append([f"{hour:02d}:00–{hour:02d}:59", by_hour.get(hour, 0)])

    # --- Every scan ------------------------------------------------------
    ws = wb.create_sheet("รายการสแกน")
    _header(ws, ["เวลา", "สินค้า", "ผล", "พนักงานแก้เอง", "ความคล้าย"], [22, 40, 28, 16, 12])
    for e in db.analytics_events(range_key):
        if e["matched"] and e["product_name"]:
            result, product = "สำเร็จ", e["product_name"]
        elif e["matched"]:
            result, product = "สำเร็จ (สินค้าถูกลบแล้ว)", None
        else:
            result, product = "ไม่รู้จัก / ไม่แน่ใจ", None
        ws.append([_when(e["created_at"]), product, result, "ใช่" if e["corrected"] else "", e["confidence"]])
    for row in ws.iter_rows(min_row=2, min_col=1, max_col=1):
        row[0].number_format = _DATETIME_FORMAT
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
        row[0].number_format = "0%"

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
