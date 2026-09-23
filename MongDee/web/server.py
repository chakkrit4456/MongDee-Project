"""FastAPI app: everything the desktop app does, reachable from any browser
on the same machine or the local network — booth view (live streams + AI
Product Assistant + readiness check + health alerts), Dashboard, and AI
Trainer. See web_server.py for the CLI entrypoint that builds this app.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from core import database as db
from core.export import build_workbook
from core.recognizer import RECOMMENDED_SAMPLES
from web.booth_manager import STREAM_FPS, BoothManager

WEB_ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BATCH_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"ไฟล์ {upload.filename or ''} มีขนาดใหญ่เกินกำหนด")
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_hhmm(value, field_label: str) -> str | None:
    """Booth Schedule (open_time/close_time) input from /settings — "" or
    missing means "no schedule" (NULL, always-on), anything else must be a
    plain HH:MM the same way <input type="time"> sends it."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        hh, mm = value.split(":")
        if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
            raise ValueError
    except ValueError:
        raise HTTPException(400, f"{field_label}ไม่ถูกต้อง (รูปแบบ HH:MM)")
    return f"{int(hh):02d}:{int(mm):02d}"


def create_app(booth: BoothManager) -> FastAPI:
    app = FastAPI(title="MONGDEE AI Booth OS")
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")

    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        """Starlette's StaticFiles sends only an ETag/Last-Modified by
        default (no Cache-Control) -- a browser is still free to skip
        revalidation and reuse whatever it already has on disk, which is
        exactly the class of bug that made a CSS/HTML change (e.g. the
        blue-black-white theme rework) look "distorted" for anyone whose
        browser already had the site open: the new HTML loaded against
        old, still-cached CSS. Forcing revalidation on every /static
        request (not `no-store`, so a valid If-None-Match hit still
        short-circuits to a cheap 304) means the next reload always picks
        up a real edit, with no separate cache-busting scheme to remember
        to touch."""
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.on_event("startup")
    def _start_booth():
        booth.start_scheduled()

    @app.on_event("shutdown")
    def _stop_booth():
        booth.stop_scheduled()

    # --------------------------------------------------------------- pages
    @app.get("/")
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "booth_id": booth.booth_id, "booth_name": booth.booth_name,
            "event_id": booth.event_id,
        })

    @app.get("/booth")
    def booth_page(request: Request):
        settings = booth.get_settings()
        camera_enabled = {c["camera_id"]: c["enabled"] for c in settings["cameras"]}
        return templates.TemplateResponse(request, "booth.html", {
            "booth_id": booth.booth_id, "booth_name": booth.booth_name,
            "event_id": booth.event_id, "camera_ids": booth.camera_ids,
            "camera_enabled": camera_enabled,
        })

    @app.get("/product-view")
    def product_view_page(request: Request):
        return templates.TemplateResponse(request, "product_view.html", {
            "booth_id": booth.booth_id, "booth_name": booth.booth_name,
            "event_id": booth.event_id, "camera_ids": booth.camera_ids,
        })

    @app.get("/booth/camera/{camera_id}")
    def booth_camera_page(request: Request, camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        return templates.TemplateResponse(request, "camera_view.html", {
            "booth_id": booth.booth_id, "booth_name": booth.booth_name,
            "event_id": booth.event_id, "camera_id": camera_id,
        })

    @app.get("/dashboard")
    def dashboard_page(request: Request):
        return templates.TemplateResponse(request, "dashboard.html", {
            "booth_id": booth.booth_id, "event_id": booth.event_id,
        })

    @app.get("/trainer")
    def trainer_page(request: Request):
        return templates.TemplateResponse(request, "trainer.html", {
            "camera_ids": booth.camera_ids,
            "recommended_samples": RECOMMENDED_SAMPLES,
        })

    @app.get("/settings")
    def settings_page(request: Request):
        return templates.TemplateResponse(request, "settings.html", {})

    # ------------------------------------------------------------ streaming
    @app.get("/stream/{camera_id}")
    def stream(camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")

        async def generate():
            # An async generator + asyncio.sleep (instead of a sync
            # generator + time.sleep) so each open camera stream just parks
            # on the event loop between frames rather than tying up a
            # worker thread from Starlette's threadpool for its whole
            # lifetime. That threadpool has a fixed size — with several
            # cameras open across several viewers/tabs at once, sync
            # generators could exhaust it and make every stream (and every
            # other request) stall; this keeps N simultaneous streams cheap
            # regardless of how many cameras/devices are watching.
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                frame = booth.get_latest_jpeg(camera_id)
                if frame:
                    yield boundary + frame + b"\r\n"
                await asyncio.sleep(1 / STREAM_FPS)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    # ---------------------------------------------------------------- booth API
    @app.get("/api/state")
    def api_state():
        return booth.get_state()

    @app.get("/api/booth/cameras/{camera_id}/snapshot")
    def api_camera_snapshot(camera_id: str):
        """Live people (by category)/product-count detail panel for one
        camera — see BoothManager.get_camera_snapshot; used by the popout
        single-camera view's right-side panel (camera_view.html)."""
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        return booth.get_camera_snapshot(camera_id)

    @app.get("/api/booth/cameras/{camera_id}/diagnostics")
    def api_camera_diagnostics(camera_id: str):
        """Per-camera failure diagnostics (capture mode, worker pid once escalated to process
        isolation, hang/stale/restart counters, corrupt/frozen frame counts, reason_code) — see
        BoothManager.get_camera_diagnostics. Separate from /snapshot (people/product counts)."""
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        return booth.get_camera_diagnostics(camera_id)

    @app.get("/api/performance")
    def api_performance():
        return booth.get_performance_snapshot()

    @app.post("/api/readiness")
    def api_readiness():
        return booth.run_readiness()

    # -------------------------------------------------------- booth settings
    @app.get("/api/booth/settings")
    def api_booth_settings():
        return booth.get_settings()

    # -------------------------------------------------------- event/booth registry
    @app.get("/api/registry/events")
    def api_list_events():
        return db.list_events(booth.db_path)

    @app.post("/api/registry/events")
    def api_create_event(payload: dict):
        from core.training import slugify

        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "กรุณาระบุชื่อ Event")
        existing = {e["id"] for e in db.list_events(booth.db_path)}
        event_id = slugify(name, prefix="event")
        while event_id in existing:
            event_id = f"{event_id}-{int(time.time() * 1000) % 10000}"
        db.create_event(booth.db_path, event_id, name)
        return {"id": event_id}

    @app.put("/api/registry/events/{event_id}")
    def api_rename_event(event_id: str, payload: dict):
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "กรุณาระบุชื่อ Event")
        db.rename_event(booth.db_path, event_id, name)
        return {"ok": True}

    @app.delete("/api/registry/events/{event_id}")
    def api_delete_event(event_id: str):
        db.delete_event(booth.db_path, event_id)
        booth.activate_booth(booth.booth_id)  # refresh event_id if the active booth was a member
        return {"ok": True}

    @app.get("/api/registry/booths")
    def api_list_booths():
        return [{**b, "active": b["id"] == booth.booth_id} for b in db.list_booths(booth.db_path)]

    @app.post("/api/registry/booths")
    def api_create_booth(payload: dict):
        from core.training import slugify

        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "กรุณาระบุชื่อบูธ")
        existing = {b["id"] for b in db.list_booths(booth.db_path)}
        booth_id = slugify(name, prefix="booth").upper()
        while booth_id in existing:
            booth_id = f"{booth_id}-{int(time.time() * 1000) % 10000}"
        db.create_booth(
            booth.db_path, booth_id, name, payload.get("event_id") or None,
            _validate_hhmm(payload.get("open_time"), "เวลาเปิด"),
            _validate_hhmm(payload.get("close_time"), "เวลาปิด"),
        )
        return {"id": booth_id}

    @app.put("/api/registry/booths/{booth_id}")
    def api_update_booth(booth_id: str, payload: dict):
        if not db.get_booth(booth.db_path, booth_id):
            raise HTTPException(404, "ไม่พบบูธนี้")
        kwargs = {}
        if "name" in payload:
            kwargs["name"] = payload.get("name")
        if "event_id" in payload:
            kwargs["event_id"] = payload.get("event_id") or None  # "" or missing = unassign
        if "open_time" in payload:
            kwargs["open_time"] = _validate_hhmm(payload.get("open_time"), "เวลาเปิด")
        if "close_time" in payload:
            kwargs["close_time"] = _validate_hhmm(payload.get("close_time"), "เวลาปิด")
        db.update_booth(booth.db_path, booth_id, **kwargs)
        if booth_id == booth.booth_id:
            booth.activate_booth(booth_id)  # refresh live name/event_id
        return {"ok": True}

    @app.delete("/api/registry/booths/{booth_id}")
    def api_delete_booth(booth_id: str):
        if not db.get_booth(booth.db_path, booth_id):
            raise HTTPException(404, "ไม่พบบูธนี้")
        try:
            booth.remove_booth(booth_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        return {"ok": True}

    @app.post("/api/registry/booths/{booth_id}/activate")
    def api_activate_booth(booth_id: str):
        try:
            booth.activate_booth(booth_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        return booth.get_settings()

    @app.post("/api/booth/cameras")
    def api_add_camera(payload: dict):
        device = (payload.get("device") or "").strip()
        if not device:
            raise HTTPException(400, "กรุณาระบุกล้อง เช่น 0 หรือ /dev/video0")
        try:
            camera_id = booth.add_camera(device)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"camera_id": camera_id}

    @app.delete("/api/booth/cameras/{camera_id}")
    def api_remove_camera(camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        booth.remove_camera(camera_id)
        return {"ok": True}

    @app.post("/api/booth/cameras/{camera_id}/enabled")
    def api_set_camera_enabled(camera_id: str, payload: dict):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        enabled = bool(payload.get("enabled", True))
        booth.set_camera_enabled(camera_id, enabled)
        return {"camera_id": camera_id, "enabled": enabled}

    @app.post("/api/booth/reset_data")
    def api_reset_booth_data():
        booth.reset_data()
        return {"ok": True}

    # --------------------------------------------------- built-in camera policy
    @app.get("/api/booth/camera_settings")
    def api_get_camera_settings():
        return booth.get_camera_settings()

    @app.post("/api/booth/camera_settings")
    def api_set_camera_settings(payload: dict):
        return booth.set_builtin_camera_enabled(bool(payload.get("enable_builtin_camera", False)))

    # -------------------------------------------------------------- tripwire
    @app.get("/api/booth/cameras/{camera_id}/tripwire")
    def api_get_tripwire(camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        return booth.get_tripwire(camera_id)

    @app.post("/api/booth/cameras/{camera_id}/tripwire")
    def api_set_tripwire(camera_id: str, payload: dict):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        try:
            return booth.set_tripwire(
                camera_id,
                x1=float(payload["x1"]), y1=float(payload["y1"]),
                x2=float(payload["x2"]), y2=float(payload["y2"]),
                inside_side=payload["inside_side"],
                enabled=bool(payload.get("enabled", True)),
            )
        except (KeyError, TypeError):
            raise HTTPException(400, "ข้อมูลเส้นไม่ครบ ต้องมี x1,y1,x2,y2,inside_side")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.delete("/api/booth/cameras/{camera_id}/tripwire")
    def api_delete_tripwire(camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        booth.remove_tripwire(camera_id)
        return {"ok": True}

    @app.get("/api/booth/cameras/{camera_id}/tripwire/status")
    def api_tripwire_status(camera_id: str):
        if camera_id not in booth.camera_ids:
            raise HTTPException(404, "ไม่พบกล้องนี้")
        return booth.get_tripwire_status(camera_id)

    @app.get("/api/booth/session_status")
    def api_booth_session_status():
        return booth.get_booth_session_status()

    # ------------------------------------------------------------------ GPU
    @app.get("/api/system/gpu_status")
    def api_gpu_status():
        return booth.get_gpu_status()

    # ------------------------------------------------------------- products API
    @app.get("/api/products")
    def api_products():
        counts = booth.recognizer.sample_counts()
        return [
            {"key": key, **booth.catalog.get(key), "sample_count": counts.get(key, 0)}
            for key in booth.catalog.product_keys()
        ]

    @app.post("/api/products")
    def api_add_product(payload: dict):
        from core.training import slugify

        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "กรุณาระบุชื่อสินค้า")
        key = slugify(name)
        while booth.catalog.get(key):
            key = f"{key}-{int(time.time() * 1000) % 10000}"
        booth.catalog.add_product(key, name, payload.get("tagline", ""), payload.get("description", ""),
                                   payload.get("faq"), payload.get("price", ""))
        return {"key": key}

    @app.put("/api/products/{key}")
    def api_update_product(key: str, payload: dict):
        if not booth.catalog.get(key):
            raise HTTPException(404, "ไม่พบสินค้านี้")
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "กรุณาระบุชื่อสินค้า")
        booth.catalog.add_product(key, name, payload.get("tagline", ""), payload.get("description", ""),
                                   payload.get("faq"), payload.get("price", ""))
        return {"key": key}

    @app.delete("/api/products/{key}")
    def api_delete_product(key: str):
        # Order matters. Remove from the catalog FIRST: the recognizer only trains/matches products
        # that are in the catalog, so from this line on any import thread still running for this
        # key is refused by add_sample() (ProductDeleted) and cannot re-create the gallery.
        # Then wipe what is already stored (memory + manifest + .npy).
        booth.catalog.remove_product(key)
        booth.recognizer.clear_product(key)
        return {"ok": True}

    @app.post("/api/products/{key}/upload_images")
    async def api_upload_images(key: str, files: list[UploadFile] = File(...)):
        if not booth.catalog.get(key):
            raise HTTPException(404, "ไม่พบสินค้านี้")
        payload = []
        batch_size = 0
        for upload in files:
            content = await _read_upload_limited(upload, MAX_IMAGE_BYTES)
            batch_size += len(content)
            if batch_size > MAX_IMAGE_BATCH_BYTES:
                raise HTTPException(413, "รูปภาพทั้งหมดมีขนาดรวมใหญ่เกินกำหนด")
            payload.append((upload.filename or "image.jpg", content))
        try:
            booth.start_image_import(key, payload)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"status": "started"}

    @app.post("/api/products/{key}/upload_video")
    async def api_upload_video(key: str, file: UploadFile = File(...)):
        if not booth.catalog.get(key):
            raise HTTPException(404, "ไม่พบสินค้านี้")
        content = await _read_upload_limited(file, MAX_VIDEO_BYTES)
        try:
            booth.start_video_import(key, file.filename or "video.mp4", content)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"status": "started"}

    @app.get("/api/products/{key}/import_progress")
    def api_import_progress(key: str):
        return booth.get_import_progress(key)

    # ----------------------------------------------- live "record from camera" training session
    # One frame per HTTP call (not a batch upload) so the frontend gets an immediate, real
    # distinct-view count back from every captured tick -- see core.training.LiveTrainingSession
    # and web/static/trainer.js's guided-rotation recording loop, which uses this to keep
    # recording until the data is actually comprehensive instead of stopping at a fixed timer.
    @app.post("/api/products/{key}/training_session/start")
    def api_start_live_training(key: str):
        try:
            booth.start_live_training(key)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        return {"status": "started"}

    @app.post("/api/products/{key}/training_session/frame")
    async def api_live_training_frame(key: str, file: UploadFile = File(...)):
        content = await _read_upload_limited(file, MAX_IMAGE_BYTES)
        try:
            return booth.feed_live_training_frame(key, content)
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.post("/api/products/{key}/training_session/finish")
    def api_finish_live_training(key: str):
        try:
            return booth.finish_live_training(key)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/products/{key}/clear_samples")
    def api_clear_samples(key: str):
        booth.recognizer.clear_product(key)
        return {"ok": True}

    # ------------------------------------------------------------ dashboard API
    @app.get("/api/dashboard/summary")
    def api_dashboard_summary(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_summary(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/top_products")
    def api_dashboard_top_products(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_top_products(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/interactions")
    def api_dashboard_interactions(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_interactions(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/health")
    def api_dashboard_health(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_health_events(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/live")
    def api_dashboard_live():
        return booth.get_live_analytics()

    @app.get("/api/dashboard/product_movers")
    def api_dashboard_product_movers(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_product_movers(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/presence_stats")
    def api_dashboard_presence_stats(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_presence_stats(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/presence_breakdown")
    def api_dashboard_presence_breakdown(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_presence_breakdown(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/product_hold_history")
    def api_dashboard_product_hold_history(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_product_hold_events(booth.db_path, event_id, booth_id, date)

    # --------------------------------------------- product interest analytics
    # Spec: MongDee person-gender/age master prompt sections 28-45/65-68.
    @app.get("/api/dashboard/interest_overview")
    def api_dashboard_interest_overview(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_interest_overview(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/product_interest_summary")
    def api_dashboard_product_interest_summary(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_product_interest_summary(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/product_gender_matrix")
    def api_dashboard_product_gender_matrix(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_product_gender_matrix(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/product_age_matrix")
    def api_dashboard_product_age_matrix(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_product_age_matrix(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/gender_interest_totals")
    def api_dashboard_gender_interest_totals(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_gender_interest_totals(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/age_interest_totals")
    def api_dashboard_age_interest_totals(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_age_interest_totals(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/presence_sessions")
    def api_dashboard_presence_sessions(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_presence_sessions(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/available_dates")
    def api_dashboard_available_dates(event_id: str | None = None, booth_id: str | None = None):
        return db.query_available_dates(booth.db_path, event_id, booth_id)

    @app.get("/api/dashboard/tripwire_stats")
    def api_dashboard_tripwire_stats(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_tripwire_stats(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/tripwire_crossings")
    def api_dashboard_tripwire_crossings(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_tripwire_crossings(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/booth_sessions")
    def api_dashboard_booth_sessions(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        return db.query_booth_sessions(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/booth_session_stats")
    def api_dashboard_booth_session_stats(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        """Spec section 36: Average/Longest Dwell Time + Current Active
        Sessions, kept separate from Tripwire IN/OUT and Unique People."""
        return db.query_booth_session_stats(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/event_sessions")
    def api_dashboard_event_sessions(event_id: str):
        return db.query_event_sessions(booth.db_path, event_id)

    @app.get("/api/dashboard/event_daily_breakdown")
    def api_dashboard_event_daily_breakdown(event_id: str, booth_id: str | None = None):
        return db.query_event_daily_breakdown(booth.db_path, event_id, booth_id)

    @app.get("/api/dashboard/unique_people")
    def api_dashboard_unique_people(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        """Spec section 28: Unique People (distinct Multi-Camera Person
        Re-ID global identities), kept separate from the raw per-camera
        visible-detection sum shown elsewhere — never the same number."""
        return {"unique_people": db.query_unique_people_count(booth.db_path, event_id, booth_id, date)}

    @app.get("/api/dashboard/attribute_breakdown")
    def api_dashboard_attribute_breakdown(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        """Spec: MongDee_Master_Prompt_FairFace_Age_Gender.md section 17 —
        model-predicted gender/age-category counts of distinct Global
        Persons, kept separate from Tripwire IN/OUT and Unique People."""
        return db.query_attribute_breakdown(booth.db_path, event_id, booth_id, date)

    @app.get("/api/dashboard/known_ids")
    def api_dashboard_known_ids():
        return db.query_known_ids(booth.db_path)

    @app.post("/api/dashboard/delete_scope")
    def api_dashboard_delete_scope(payload: dict):
        """Backs the Dashboard's single "ล้างข้อมูล" (clear data) button —
        booth/event/date are each optional and independently combinable
        (booth-only, event-only, day-only, or any combination); leaving all
        three unset only wipes anything if confirm_all is explicitly true,
        so a bare/default form submission can never nuke the whole
        database by accident."""
        scope_booth_id = (payload.get("booth_id") or "").strip() or None
        scope_event_id = (payload.get("event_id") or "").strip() or None
        scope_date = (payload.get("date") or "").strip() or None
        confirm_all = bool(payload.get("confirm_all"))
        if not scope_booth_id and not scope_event_id and not scope_date and not confirm_all:
            raise HTTPException(400, "กรุณาระบุ booth_id, event_id หรือ date อย่างน้อยหนึ่งอย่าง")
        db.delete_scope_data(booth.db_path, booth_id=scope_booth_id, event_id=scope_event_id,
                              date=scope_date, confirm_all=confirm_all)
        return {"ok": True}

    @app.get("/api/dashboard/export.xlsx")
    def api_dashboard_export(event_id: str | None = None, booth_id: str | None = None, date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
        content = build_workbook(booth.db_path, event_id, booth_id, date)
        scope = booth_id or event_id or "all"
        if date:
            scope += f"-{date}"
        filename = f"mongdee-export-{scope}-{int(time.time())}.xlsx"
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return app
