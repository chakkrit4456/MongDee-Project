"""Booth Readiness Check — a pre-event diagnostic of every subsystem.

Mirrors spec item 3 (Booth Readiness Check): before a booth opens, staff
run this once to confirm cameras, the AI model, storage, TTS and the mic
are all working, instead of checking each device by hand. Each component
also carries an "advanced" dict of deeper diagnostic fields (resolution,
row counts, disk space, ...) for staff who want the full picture, not just
a pass/fail line.
"""

from __future__ import annotations

import json
import platform
import shutil
import sqlite3
import time
from pathlib import Path


def check_database(db_path) -> tuple[bool, str]:
    try:
        with sqlite3.connect(db_path, timeout=3) as conn:
            conn.execute("SELECT 1")
        return True, "เชื่อมต่อฐานข้อมูลสำเร็จ"
    except Exception as exc:
        return False, f"เชื่อมต่อฐานข้อมูลไม่สำเร็จ: {exc}"


def check_microphone() -> tuple[bool, str]:
    try:
        import sounddevice as sd

        devices = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if devices:
            return True, f"พบไมโครโฟน {len(devices)} อุปกรณ์"
        return False, "ไม่พบไมโครโฟนที่ใช้งานได้"
    except Exception as exc:
        return False, f"ตรวจสอบไมโครโฟนไม่สำเร็จ: {exc}"


def _microphone_advanced() -> dict:
    try:
        import sounddevice as sd

        names = [d["name"] for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        return {"อุปกรณ์ที่พบ": ", ".join(names) if names else "ไม่มี"}
    except Exception as exc:
        return {"ข้อผิดพลาด": str(exc)}


def _database_advanced(db_path) -> dict:
    from core.database import SCOPED_TABLES  # local import: avoid a cross-module import cycle at load time

    advanced: dict = {}
    try:
        size_bytes = Path(db_path).stat().st_size
        advanced["ขนาดไฟล์"] = f"{size_bytes / 1024:.1f} KB"
    except OSError:
        advanced["ขนาดไฟล์"] = "ไม่ทราบ"
    try:
        with sqlite3.connect(db_path, timeout=3) as conn:
            counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in SCOPED_TABLES}
        advanced["จำนวนแถวทั้งหมด"] = sum(counts.values())
        for table, count in counts.items():
            advanced[f"แถวใน {table}"] = count
    except Exception as exc:
        advanced["ข้อผิดพลาด"] = str(exc)
    return advanced


def _model_advanced(model_device) -> dict:
    advanced = {"อุปกรณ์ประมวลผล": str(model_device)}
    try:
        import torch

        advanced["เวอร์ชัน PyTorch"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        advanced["CUDA พร้อมใช้งาน"] = "ใช่" if cuda_available else "ไม่"
        if cuda_available:
            try:
                advanced["ชื่อ GPU"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    except Exception as exc:
        advanced["ข้อผิดพลาด"] = str(exc)
    try:
        import ultralytics

        advanced["เวอร์ชัน Ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return advanced


def _system_advanced(started_at: float | None, db_path) -> dict:
    advanced: dict = {"Python": platform.python_version(), "ระบบปฏิบัติการ": platform.platform()}
    if started_at:
        uptime_sec = time.time() - started_at
        h, rem = divmod(int(uptime_sec), 3600)
        m, s = divmod(rem, 60)
        advanced["เวลาที่ทำงานมาแล้ว"] = f"{h} ชม. {m} นาที {s} วินาที"
    try:
        usage = shutil.disk_usage(Path(db_path).resolve().parent)
        advanced["พื้นที่ว่างบนดิสก์"] = f"{usage.free / (1024 ** 3):.1f} GB จากทั้งหมด {usage.total / (1024 ** 3):.1f} GB"
    except OSError:
        pass
    return advanced


def _camera_advanced(resolution: tuple[int, int] | None, device) -> dict:
    return {
        "อุปกรณ์": str(device),
        "ความละเอียด": f"{resolution[0]}x{resolution[1]}" if resolution else "ไม่ทราบ (กล้องยังไม่เปิดหรือออฟไลน์)",
    }


def run_readiness_check(camera_statuses: dict[str, str], model_loaded: bool,
                         tts_available: bool, db_path,
                         camera_resolutions: dict[str, tuple[int, int] | None] | None = None,
                         camera_devices: dict[str, object] | None = None,
                         model_device=None, started_at: float | None = None) -> dict:
    """camera_statuses: {camera_id: "online"/"offline"/"error"}."""
    components = []
    camera_resolutions = camera_resolutions or {}
    camera_devices = camera_devices or {}

    for camera_id, status in camera_statuses.items():
        ok = status == "online"
        components.append({
            "component": f"กล้อง {camera_id}",
            "critical": True,
            "ok": ok,
            "detail": "พร้อมใช้งาน" if ok else f"สถานะ: {status}",
            "advanced": _camera_advanced(camera_resolutions.get(camera_id), camera_devices.get(camera_id, camera_id)),
        })

    components.append({
        "component": "โมเดล AI Vision (YOLO11)",
        "critical": True,
        "ok": model_loaded,
        "detail": "โหลดโมเดลสำเร็จ" if model_loaded else "โหลดโมเดลไม่สำเร็จ",
        "advanced": _model_advanced(model_device),
    })

    db_ok, db_msg = check_database(db_path)
    components.append({
        "component": "ฐานข้อมูล", "critical": True, "ok": db_ok, "detail": db_msg,
        "advanced": _database_advanced(db_path),
    })

    components.append({
        "component": "ระบบเสียงพูด (Text-to-Speech)",
        "critical": False,
        "ok": tts_available,
        "detail": "พร้อมใช้งาน" if tts_available else "ไม่พร้อมใช้งาน (ระบบจะยังทำงานได้แบบข้อความ)",
        "advanced": {"หมายเหตุ": "พูดผ่าน Web Speech API ของเบราว์เซอร์ ไม่ใช่เซิร์ฟเวอร์"},
    })

    mic_ok, mic_msg = check_microphone()
    components.append({
        "component": "ไมโครโฟน", "critical": False, "ok": mic_ok, "detail": mic_msg,
        "advanced": _microphone_advanced(),
    })

    components.append({
        "component": "ระบบ",
        "critical": False,
        "ok": True,
        "detail": "ข้อมูลเซิร์ฟเวอร์",
        "advanced": _system_advanced(started_at, db_path),
    })

    critical_failed = [c for c in components if c["critical"] and not c["ok"]]
    overall_ok = len(critical_failed) == 0

    return {
        "ts": time.time(),
        "overall_ok": overall_ok,
        "components": components,
    }


def readiness_to_json(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False)
