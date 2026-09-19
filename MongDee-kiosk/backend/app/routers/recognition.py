from fastapi import APIRouter, HTTPException

from app import db
from app.camera import camera_manager, PresenceState
from app.recognition_loop import recognition_state
from app.schemas import CorrectionRequest, RecognitionResult, ProductOut
import time

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


@router.get("/current", response_model=RecognitionResult)
async def current():
    result = await recognition_state.get()
    snap = camera_manager.snapshot()
    # The capture/inference task may still be finishing. Never serve its
    # previous product while the camera is offline or the platform is empty.
    if not snap.fresh:
        return RecognitionResult(status="camera_unavailable", message="ไม่พบภาพสดจากกล้อง กำลังเชื่อมต่อใหม่", updated_at=result.updated_at)
    if not camera_manager.has_reference:
        return RecognitionResult(status="needs_reference", message="ยกสินค้าออก แล้วบันทึกภาพพื้นเปล่าในหน้าปรับตั้งค่า", updated_at=result.updated_at)
    if snap.state == PresenceState.EMPTY:
        return RecognitionResult(status="idle", updated_at=result.updated_at)
    if result.camera_generation is not None and result.camera_generation != snap.generation:
        return RecognitionResult(status="scanning", updated_at=result.updated_at)
    return result


@router.post("/correct")
async def correct(body: CorrectionRequest):
    """Manual fallback when confidence was too low: staff/visitor picks the
    right product, the scan event is corrected for analytics, and — as the
    pitch script promises — this correction is exactly the signal a future
    iteration would feed back into the match index to improve accuracy."""
    product = db.get_product(body.product_id)
    if not product:
        raise HTTPException(404, "Product not found")
    current_result = await recognition_state.get()
    snap = camera_manager.snapshot()
    if (current_result.scan_event_id != body.scan_event_id or not snap.fresh
            or snap.state != PresenceState.PRESENT or current_result.camera_generation != snap.generation):
        raise HTTPException(409, "ผลสแกนเปลี่ยนไปแล้ว กรุณาเลือกจากผลล่าสุด")
    db.correct_scan_event(body.scan_event_id, body.product_id)
    await recognition_state.set(RecognitionResult(
        status="matched", product=ProductOut(**db.product_out_fields(product)), scan_event_id=body.scan_event_id,
        manually_confirmed=True, camera_generation=snap.generation, updated_at=time.time(),
    ))
    return {"ok": True}
