import asyncio
import time
from contextlib import suppress
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from app import calibration
from app import camera as camera_module
from app import config, db, vision
from app.recognition_loop import recognition_loop
from app import admin_auth
from app.routers import analytics, enrollment, products, recognition
from app.schemas import CalibrationSettings, CameraDevicesResponse, CameraSelectRequest

# Directories must exist before StaticFiles mounts below (mounts happen at
# import time, before the lifespan startup hook runs).
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    vision.refresh_match_index()
    calibration.load_persisted()

    saved_index = None if config.FAKE_CAMERA else db.get_setting("camera_index")
    if saved_index is not None:
        camera_module.camera_manager.switch_index(int(saved_index))
    else:
        camera_module.camera_manager.start()

    task = asyncio.create_task(recognition_loop(camera_module.camera_manager))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.to_thread(camera_module.camera_manager.stop)


app = FastAPI(title="MONGDEE MINI KIOSK", lifespan=lifespan)

app.include_router(products.router)
app.include_router(enrollment.router)
app.include_router(recognition.router)
app.include_router(analytics.router)
app.include_router(admin_auth.router)
if config.FAKE_CAMERA:
    from app.routers import dev
    app.include_router(dev.router)

app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR / "static")), name="static")
app.mount("/captures", StaticFiles(directory=str(config.CAPTURES_DIR)), name="captures")


@app.get("/api/camera/status")
def camera_status():
    snap = camera_module.camera_manager.snapshot()
    camera_open = camera_module.camera_manager.is_open
    has_reference = camera_module.camera_manager.has_reference
    if not camera_open or not has_reference or snap.state != camera_module.PresenceState.PRESENT:
        box_status = "none"
    elif snap.sharpness < config.MIN_SHARPNESS:
        box_status = "blurry"
    else:
        box_status = "ok"
    return {
        "state": snap.state.value,
        "foreground_ratio": snap.foreground_ratio,
        "camera_open": camera_open,
        "has_reference": has_reference,
        "frame_fresh": snap.fresh,
        "motion_ratio": snap.motion_ratio,
        "sharpness": snap.sharpness,
        "box_status": box_status,
        "updated_at": snap.updated_at,
    }


@app.post("/api/calibration/reference")
def capture_reference():
    """Remembers the current live frame as the "empty platform" baseline —
    call this while the platform is genuinely empty. Every later frame is
    compared against this fixed snapshot instead of an adaptive model."""
    ok = camera_module.camera_manager.capture_reference_now()
    if not ok:
        raise HTTPException(400, "รอภาพสดจากกล้องให้นิ่ง แล้วลองบันทึกพื้นเปล่าอีกครั้ง")
    return {"ok": True}


@app.get("/api/camera/devices", response_model=CameraDevicesResponse)
def camera_devices():
    return CameraDevicesResponse(
        devices=camera_module.list_camera_devices(),
        current_index=camera_module.camera_manager.index,
    )


@app.post("/api/camera/select")
def camera_select(body: CameraSelectRequest):
    if config.FAKE_CAMERA:
        raise HTTPException(400, "กำลังใช้กล้องจำลอง (KIOSK_FAKE_CAMERA) เลือกกล้องจริงไม่ได้")
    ok = camera_module.camera_manager.switch_index(body.index)
    if not ok:
        raise HTTPException(400, f"เปิดกล้องหมายเลข {body.index} ไม่สำเร็จ — อาจถูกใช้งานโดยโปรแกรมอื่นอยู่")
    db.set_setting("camera_index", str(body.index))
    return {"ok": True, "current_index": camera_module.camera_manager.index}


@app.get("/api/calibration", response_model=CalibrationSettings)
def get_calibration():
    return calibration.current()


@app.post("/api/calibration")
def post_calibration(body: CalibrationSettings):
    data = body.model_dump()
    calibration.save(data)
    return {"ok": True}


@app.post("/api/calibration/reset", response_model=CalibrationSettings)
def reset_calibration():
    calibration.save(calibration.defaults())
    return calibration.current()


def _mjpeg_stream():
    boundary = b"--frame"
    while True:
        jpeg = camera_module.camera_manager.latest_jpeg()
        if jpeg:
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.1)


@app.get("/api/camera/stream")
def camera_stream():
    return StreamingResponse(_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
def kiosk_page():
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))


@app.get("/products")
def products_page():
    return FileResponse(str(config.FRONTEND_DIR / "products.html"))


@app.get("/enroll")
def enroll_page():
    return FileResponse(str(config.FRONTEND_DIR / "enroll.html"))


@app.get("/manage")
def manage_page():
    return FileResponse(str(config.FRONTEND_DIR / "manage.html"))


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(str(config.FRONTEND_DIR / "dashboard.html"))


@app.get("/calibrate")
def calibrate_page():
    return FileResponse(str(config.FRONTEND_DIR / "calibrate.html"))
