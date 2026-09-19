"""Guarded multi-view recognition with bounded retries and verification."""
from __future__ import annotations

import asyncio
import time

from app import config, db, enroll_sessions, vision
from app.camera import CameraManager, PresenceState, select_quality_frames
from app.schemas import ProductOut, RecognitionCandidate, RecognitionResult


class RecognitionState:
    def __init__(self):
        self._result = RecognitionResult(status="idle", updated_at=time.time())
        self._lock = asyncio.Lock()

    async def set(self, result):
        async with self._lock:
            self._result = result

    async def get(self):
        async with self._lock:
            return self._result


recognition_state = RecognitionState()

# Guards a *change* to what is already on screen: a scan whose outcome differs
# from the displayed one must repeat SWITCH_CONFIRM_SCANS times in a row before
# it takes over, so one shaky burst can't flip the kiosk. Reset on a new
# placement and whenever a scan confirms the current display.
_pending_switch = {"key": None, "count": 0}


def _clear_pending_switch() -> None:
    _pending_switch["key"] = None
    _pending_switch["count"] = 0


def _switch_confirmed(key: str) -> bool:
    if _pending_switch["key"] == key:
        _pending_switch["count"] += 1
    else:
        _pending_switch["key"] = key
        _pending_switch["count"] = 1
    return _pending_switch["count"] >= config.SWITCH_CONFIRM_SCANS


async def _set_status(status: str, message: str | None = None):
    current = await recognition_state.get()
    if current.status != status or current.message != message:
        await recognition_state.set(RecognitionResult(status=status, message=message, updated_at=time.time()))


def _valid_scan(camera, initial, roi, version):
    snap = camera.snapshot()
    return (snap.fresh and snap.state == PresenceState.PRESENT
            and snap.generation == initial.generation
            and tuple(config.ROI_FRACTION) == roi and vision.match_index.version == version)


async def _run_scan(camera_manager: CameraManager) -> bool:
    initial = camera_manager.snapshot()
    roi = tuple(config.ROI_FRACTION)
    version = vision.match_index.version
    previous = await recognition_state.get()
    if previous.status != "matched":
        await _set_status("scanning")
    frames = await asyncio.to_thread(camera_manager.capture_burst_frames)
    if not _valid_scan(camera_manager, initial, roi, version):
        return False
    if len(frames) < config.RECOGNITION_MIN_FRAMES:
        await _set_status("unstable", "ภาพยังไม่ชัดพอ วางสินค้าให้อยู่ในกรอบ เอามือออก และเพิ่มแสง")
        return False
    crops = [camera_manager.masked_crop(f, roi) for f in frames]
    embeddings = await asyncio.to_thread(vision.feature_extractor.embed_many, crops)
    match = await asyncio.to_thread(vision.match_index.match_burst, embeddings)
    if not _valid_scan(camera_manager, initial, roi, version):
        return False  # Never display/log the item that has already been removed.
    ranked = match.ranked
    candidates = []
    for pid, score in ranked:
        product = db.get_product(pid)
        if product:
            candidates.append(RecognitionCandidate(product_id=pid, name=product["name"], score=score))
    best_score = ranked[0][1] if ranked else None
    status = match.status
    if status == "unstable":
        await _set_status("unstable", "ภาพที่ใช้งานได้ยังไม่พอ กรุณาวางสินค้าไว้ในกรอบ")
        return False
    showing_match = previous.status == "matched" and previous.product is not None
    if status == "matched":
        best_id = ranked[0][0]
        product = db.get_product(best_id)
        if product is None:
            return False
        if showing_match and previous.product.id == best_id:
            _clear_pending_switch()
            return True  # No duplicate analytics/video restart for a verification.
        # Switching from one shown product to a different one: make a second
        # scan agree first, otherwise keep the current display steady.
        if showing_match and not _switch_confirmed(f"match:{best_id}"):
            return False
        _clear_pending_switch()
        scan_id = db.log_scan_event(best_id, best_score, True)
        result = RecognitionResult(status=status, scan_event_id=scan_id, product=ProductOut(**db.product_out_fields(product)),
                                   confidence=best_score, candidates=candidates, camera_generation=initial.generation, updated_at=time.time())
    else:
        # Don't drop a good match to unknown/ambiguous on a single weak burst.
        if showing_match and not _switch_confirmed(f"{status}"):
            return False
        _clear_pending_switch()
        pending_id = enroll_sessions.create_session()
        enroll_sessions.add_batch(pending_id, frames, embeddings)
        scan_id = db.log_scan_event(None, best_score, False)
        result = RecognitionResult(status=status, scan_event_id=scan_id, confidence=best_score,
                                   candidates=candidates, pending_enroll_session_id=pending_id,
                                   camera_generation=initial.generation, updated_at=time.time())
    await recognition_state.set(result)
    return True


async def _still_matches(camera_manager: CameraManager, product_id: int) -> bool | None:
    """Three fresh views; unclear/tied/blurred evidence cannot trigger a swap."""
    initial = camera_manager.snapshot()
    roi = tuple(config.ROI_FRACTION)
    version = vision.match_index.version
    frames = []
    last_timestamp = 0.0
    for _ in range(3):
        snap = camera_manager.snapshot()
        if not _valid_scan(camera_manager, initial, roi, version):
            return None
        if snap.captured_at != last_timestamp and snap.motion_ratio <= config.MAX_CAPTURE_MOTION:
            frames.append(snap.frame.copy())
            last_timestamp = snap.captured_at
        await asyncio.sleep(0.25)
    frames = select_quality_frames(frames, roi)
    if len(frames) < 3:
        return None
    crops = [camera_manager.masked_crop(f, roi) for f in frames]
    embeddings = await asyncio.to_thread(vision.feature_extractor.embed_many, crops)
    if not _valid_scan(camera_manager, initial, roi, version):
        return None
    match = vision.match_index.match_burst(embeddings)
    if not match.ranked:
        return False
    best_id, score = match.ranked[0]
    if best_id == product_id and score >= config.MATCH_CONFIDENCE_THRESHOLD - config.AMBIGUOUS_MARGIN:
        return True
    if match.agreement < config.MATCH_MIN_AGREEMENT:
        return None
    if best_id != product_id and len(match.ranked) > 1 and score - match.ranked[1][1] < config.AMBIGUOUS_MARGIN:
        return None
    return False


async def recognition_loop(camera_manager: CameraManager) -> None:
    generation = None
    completed = False
    retries = failures = 0
    last_attempt = last_verify = 0.0
    while True:
        try:
            snap = camera_manager.snapshot()
            if not snap.fresh:
                await _set_status("camera_unavailable", "ไม่พบภาพสดจากกล้อง กำลังเชื่อมต่อใหม่")
                generation = None
            elif not camera_manager.has_reference:
                await _set_status("needs_reference", "ยกสินค้าออก แล้วบันทึกภาพพื้นเปล่าในหน้าปรับตั้งค่า")
                generation = None
            elif snap.state == PresenceState.EMPTY:
                await _set_status("idle")
                generation = None
            else:
                if generation != snap.generation:
                    generation = snap.generation
                    completed = False
                    retries = failures = 0
                    last_attempt = 0.0
                    _clear_pending_switch()
                    # A new placement must not inherit a prior matched result.
                    await _set_status("scanning")
                now = time.monotonic()
                if not completed and retries <= config.SCAN_MAX_RETRIES and now - last_attempt >= config.SCAN_RETRY_INTERVAL_S:
                    completed = await _run_scan(camera_manager)
                    retries += 1
                    last_attempt = last_verify = time.monotonic()
                elif completed and now - last_verify >= config.REVERIFY_INTERVAL_S:
                    last_verify = now
                    current = await recognition_state.get()
                    if current.status == "matched" and current.product and not current.manually_confirmed:
                        verdict = await _still_matches(camera_manager, current.product.id)
                        failures = failures + 1 if verdict is False else 0
                        if failures >= config.REVERIFY_FAILURES:
                            completed = await _run_scan(camera_manager)
                            retries = failures = 0
                            last_attempt = last_verify = time.monotonic()
        except Exception as exc:
            print(f"[recognition_loop] error: {exc}")
            await _set_status("unstable", "วิเคราะห์ภาพไม่สำเร็จ กรุณายกสินค้าออกแล้ววางใหม่")
            last_attempt = time.monotonic()
            retries += 1
        await asyncio.sleep(0.2)
