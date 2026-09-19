"""Enrollment flow: capture a burst, let staff select which frames are
usable (repeating capture if needed), then confirm with product metadata.

A session can also arrive pre-loaded: when the kiosk fails to recognize a
product, recognition_loop.py stashes its captured frames under a session
id via enroll_sessions and hands that id to the UI, so "add this as a new
product" skips straight to frame selection instead of recapturing.
"""
from __future__ import annotations

import asyncio
import base64
import uuid
import threading

import cv2
from fastapi import APIRouter, HTTPException

from app import camera as camera_module
from app import config, db, enroll_sessions, vision
from app.schemas import EnrollBatchResponse, EnrollConfirmRequest, ProductOut

router = APIRouter(prefix="/api/enroll", tags=["enrollment"])
_confirm_lock = threading.Lock()


async def _capture_batch() -> tuple[list, list]:
    context = camera_module.camera_manager.capture_context
    roi = tuple(config.ROI_FRACTION)
    # An explicit enrollment scan is operator-directed: automatic presence
    # calibration must not prevent teaching the system a new product.
    # The empty-platform reference is required now: enrollment embeds the
    # product with the booth background masked out (camera.build_masked_crop),
    # and recognition does the same — without the baseline the two wouldn't
    # match. Set it once on /calibrate before enrolling.
    if not camera_module.camera_manager.has_reference:
        raise HTTPException(400, "ยังไม่มีภาพพื้นเปล่าสำหรับกล้องนี้ — ไปที่หน้าปรับตั้งค่า ยกสินค้าออก แล้วกด \"บันทึกภาพพื้นเปล่า\" ก่อนเพิ่มสินค้า")
    frames = await asyncio.to_thread(camera_module.camera_manager.capture_burst_frames, require_present=False)
    if len(frames) < config.ENROLL_MIN_IMAGES:
        raise HTTPException(400, "ภาพชัดยังไม่พอ: ตรวจว่ากล้องมีภาพสด วางสินค้าในกรอบ เอามือออก และเพิ่มแสง แล้วสแกนอีกครั้ง")
    # Embed the platform region with the background masked out — see
    # camera.build_masked_crop. `frames` (full, uncropped) is still what
    # gets kept for thumbnails.
    cropped = [camera_module.camera_manager.masked_crop(f, roi) for f in frames]
    embeddings = await asyncio.to_thread(vision.feature_extractor.embed_many, cropped)
    snap = camera_module.camera_manager.snapshot()
    if not snap.fresh or camera_module.camera_manager.capture_context != context:
        raise HTTPException(409, "กล้องหรือกรอบตรวจจับเปลี่ยนระหว่างเก็บภาพ กรุณาถ่ายใหม่")
    return frames, embeddings


@router.post("/start", response_model=EnrollBatchResponse)
async def start_enrollment():
    frames, embeddings = await _capture_batch()
    session_id = enroll_sessions.create_session()
    added = enroll_sessions.add_batch(session_id, frames, embeddings)
    return EnrollBatchResponse(session_id=session_id, frames=added)


@router.get("/{session_id}", response_model=EnrollBatchResponse)
def get_enrollment_session(session_id: str):
    session = enroll_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Enrollment session expired or not found — please capture again")
    frames = []
    for frame_id, frame in session["frames"].items():
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        thumb_url = ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else ""
        frames.append({"frame_id": frame_id, "thumbnail_data_url": thumb_url})
    return EnrollBatchResponse(session_id=session_id, frames=frames)


@router.post("/{session_id}/more", response_model=EnrollBatchResponse)
async def capture_more(session_id: str):
    if enroll_sessions.get_session(session_id) is None:
        raise HTTPException(404, "Enrollment session expired or not found — please capture again")
    frames, embeddings = await _capture_batch()
    added = enroll_sessions.add_batch(session_id, frames, embeddings)
    return EnrollBatchResponse(session_id=session_id, frames=added)


@router.post("/{session_id}/confirm", response_model=ProductOut)
def confirm_enrollment(session_id: str, body: EnrollConfirmRequest):
    with _confirm_lock:
        return _confirm_enrollment(session_id, body)


def _confirm_enrollment(session_id: str, body: EnrollConfirmRequest):
    session = enroll_sessions.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Enrollment session expired or not found — please capture again")

    selected_ids = list(dict.fromkeys(body.selected_frame_ids))
    if len(selected_ids) < config.ENROLL_MIN_IMAGES:
        raise HTTPException(400, f"กรุณาเลือกภาพอย่างน้อย {config.ENROLL_MIN_IMAGES} รูป")

    frames_by_id = session["frames"]
    embeddings_by_id = session["embeddings"]
    missing = [fid for fid in selected_ids if fid not in frames_by_id]
    if missing:
        raise HTTPException(400, "พบรหัสภาพที่ไม่ถูกต้องในชุดที่เลือก — กรุณาลองใหม่")

    # Every selected frame is saved (not just one thumbnail) so the kiosk
    # popup can play them back as a 360-style spin view — see
    # db.get_product_frames. Cropped to the platform, not the whole desk,
    # same as the old single-thumbnail behavior.
    frames_dir = config.CAPTURES_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for fid in selected_ids:
        name = f"{uuid.uuid4().hex}.jpg"
        cropped = camera_module.crop_to_roi(frames_by_id[fid])
        if not camera_module.imwrite_unicode(frames_dir / name, cropped):
            raise HTTPException(500, "บันทึกภาพสินค้าไม่สำเร็จ กรุณาลองใหม่")
        frame_paths.append(f"captures/frames/{name}")

    # The middle angle doubles as the kiosk display thumbnail until/unless
    # staff upload a nicer cover via /manage — no separate file needed.
    thumb_path = frame_paths[len(frame_paths) // 2]

    product_id = db.create_product(
        name=body.name,
        category=body.category,
        origin=body.origin,
        material=body.material,
        process=body.process,
        story=body.story,
        video_url=body.video_url,
        video_link=body.video_link,
        thumbnail_path=thumb_path,
        price=body.price,
        production_date=body.production_date,
        expiry_date=body.expiry_date,
        embeddings=[embeddings_by_id[fid].tolist() for fid in selected_ids],
        frame_image_paths=frame_paths,
    )

    vision.refresh_match_index()
    enroll_sessions.discard_session(session_id)

    return ProductOut(**db.product_out_fields(db.get_product(product_id)))


@router.delete("/{session_id}")
def cancel_enrollment(session_id: str):
    enroll_sessions.discard_session(session_id)
    return {"ok": True}
