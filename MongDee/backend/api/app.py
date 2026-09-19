"""MongDee vision REST + WebSocket API (MongDee_Master_Prompt.md section 28;
Camera Agent / remote streaming additions per
MongDee_Cloud_Vercel_Remote_AI_Server_Master_Prompt.md sections 4, 29, 32).

create_app(db, live=...) builds a FastAPI app. It always serves read
endpoints from the database; if a `live` LiveState is passed it also
serves current per-camera detection/track counts and pushes real-time
updates over WebSocket. Passing `remote_frames` additionally exposes the
camera-agent registration/frame-ingest endpoints a network Camera Agent
uses; passing `frame_source` (anything satisfying vision.pipeline's
FrameSource protocol — a CameraGateway, a RemoteFrameReceiver, or a
backend.remote_frames.CompositeFrameSource of both) exposes a live MJPEG
stream endpoint the dashboard can point a plain <img> at.

Auth (section 34): if `api_key` is set, every /api/* and /ws/* request must
present it (X-API-Key header, or ?api_key= for websockets). If it is not
set the app logs a warning and runs open — fine for a trusted LAN, not for
production.

CORS: the Vercel-hosted dashboard (see apps/web/) calls this API from a
different origin than the AI Server itself, so cross-origin requests need
CORS enabled. Pass `cors_origins` (a list of allowed origins) in
production; omitting it allows every origin, which is fine for local
development but logged as a warning since it is not something a public
deployment should ship with.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from backend.database.db import Database
from backend.remote_frames import RemoteFrameReceiver
from vision.events import PersonEvent

logger = logging.getLogger("mongdee.backend.api")
_DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")
_DESIGNER_HTML = Path(__file__).with_name("designer.html")

STREAM_FPS = 15
JPEG_QUALITY = 80
MAX_FRAME_UPLOAD_BYTES = 10 * 1024 * 1024


class _SlidingWindowLimiter:
    """Small process-local limiter for mutation endpoints.

    It intentionally has bounded state and fails open only when configured
    with a non-positive limit, which keeps local development compatible while
    preventing one client from exhausting memory or CPU in production.
    """

    def __init__(self, limit: int, window_sec: float):
        self._limit = max(0, int(limit))
        self._window_sec = max(0.1, float(window_sec))
        self._lock = threading.Lock()
        self._hits: dict[str, collections.deque[float]] = {}

    def allow(self, key: str) -> bool:
        if self._limit == 0:
            return True
        now = time.monotonic()
        cutoff = now - self._window_sec
        with self._lock:
            hits = self._hits.setdefault(key, collections.deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            if len(self._hits) > 2048:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
            return True


async def _mjpeg_chunks(frame_source, camera_id: str, stream_fps: float = STREAM_FPS):
    """The actual MJPEG frame-serving loop, pulled out of the route so it
    can be driven directly (a bounded number of `__anext__()` calls) in
    tests instead of through a real HTTP connection — this generator runs
    forever by design (a live stream has no natural end), which hangs
    TestClient if driven through the full ASGI stack."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        current = frame_source.latest_frame(camera_id)
        if current is not None:
            ok, buf = cv2.imencode(".jpg", current.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                yield boundary + buf.tobytes() + b"\r\n"
        await asyncio.sleep(1 / stream_fps)

_VALID_OBJECT_TYPES = {
    "camera", "shelf", "product", "table", "wall", "door", "entrance", "exit",
    "product_zone", "point_of_interest", "no_entry_zone", "sales_point",
}


def _validate_layout(layout: dict) -> None:
    for field in ("width", "length"):
        if not isinstance(layout.get(field), (int, float)) or layout[field] <= 0:
            raise ValueError(f"layout.{field} must be a positive number")
    seen_ids = set()
    for i, obj in enumerate(layout.get("objects", [])):
        if not isinstance(obj, dict) or "id" not in obj or "object_type" not in obj:
            raise ValueError(f"objects[{i}] needs at least id and object_type")
        if obj["object_type"] not in _VALID_OBJECT_TYPES:
            raise ValueError(f"objects[{i}]: unknown object_type {obj['object_type']!r}")
        if obj["id"] in seen_ids:
            raise ValueError(f"objects[{i}]: duplicate id {obj['id']!r}")
        seen_ids.add(obj["id"])


class LiveState:
    """Snapshot of what the running pipeline currently sees. The server
    reads this; the pipeline writes it via update_*()."""

    def __init__(self):
        self._camera_status: dict[str, dict] = {}
        self._camera_counts: dict[str, dict] = {}
        self._unique_count = 0

    def update_camera_status(self, camera_id: str, status: str, message: str = "") -> None:
        self._camera_status[camera_id] = {"status": status, "message": message, "updated_at": time.time()}

    def update_camera_counts(self, camera_id: str, detections: int, tracks: int, fps: float, latency_ms: float) -> None:
        self._camera_counts[camera_id] = {
            "detections": detections, "tracks": tracks, "fps": round(fps, 2),
            "latency_ms": round(latency_ms, 1), "updated_at": time.time(),
        }

    def set_unique_count(self, n: int) -> None:
        self._unique_count = n

    def snapshot(self) -> dict:
        return {
            "cameras": {
                cid: {**self._camera_status.get(cid, {}), **self._camera_counts.get(cid, {})}
                for cid in set(self._camera_status) | set(self._camera_counts)
            },
            "unique_persons": self._unique_count,
        }


class EventHub:
    """Fan-out of PersonEvents to connected WebSocket clients."""

    def __init__(self):
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: PersonEvent) -> None:
        """Thread-safe: called from the pipeline thread."""
        if self._loop is None:
            return
        payload = event.to_dict()
        self._loop.call_soon_threadsafe(self._fan_out, payload)

    def _fan_out(self, payload: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)


_ROLE_RANK = {"viewer": 0, "operator": 1, "admin": 2}


def create_app(
    db: Database,
    live: LiveState | None = None,
    event_hub: EventHub | None = None,
    api_key: str | None = None,
    analytics=None,  # optional vision.booth_analytics.BoothAnalytics for live map/heatmap
    keys: dict | None = None,  # {"<key>": "admin|operator|viewer"} (section 74)
    remote_frames: RemoteFrameReceiver | None = None,  # enables /api/cameras/register + frame ingest
    frame_source=None,  # anything satisfying vision.pipeline's FrameSource protocol; enables /stream
    hardware_profile=None,  # optional core.performance.HardwareProfile, for /metrics
    pipeline=None,  # optional vision.pipeline.DetectionPipeline, for /metrics (current AI FPS/latency)
    cors_origins: list[str] | None = None,
    register_rate_limit: int = 30,
    frame_rate_limit: int = 120,
):
    # single api_key is a back-compat shortcut for one admin key
    key_roles = dict(keys or {})
    if api_key:
        key_roles.setdefault(api_key, "admin")
    auth_enabled = bool(key_roles)
    if not auth_enabled:
        logger.warning("MongDee API running WITHOUT authentication — set api_key/keys before exposing beyond localhost")

    @asynccontextmanager
    async def lifespan(_app):
        if event_hub is not None:
            event_hub.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="MongDee Vision API", version="1.0", lifespan=lifespan)

    # The Vercel-hosted dashboard (apps/web/) calls this API from a
    # different origin than wherever the AI Server itself runs, so it
    # needs CORS. cors_origins=None (the default) allows every origin,
    # which is convenient for local development but not something a
    # public deployment should ship with — same "logs a warning and runs
    # open" posture as an unset api_key above.
    if cors_origins is None:
        logger.warning("MongDee API running with CORS open to ALL origins — set cors_origins for a public deployment")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _role_for(x_api_key: str | None) -> str | None:
        if not auth_enabled:
            return "admin"
        return key_roles.get(x_api_key)

    def require_role(min_role: str):
        # A plain <img>/<video> tag (the live camera stream on the
        # dashboard) can't set a custom header, so this also accepts
        # ?api_key= as a fallback — the same accommodation the WebSocket
        # endpoints below already need for the same reason.
        def dep(x_api_key: str | None = Header(default=None), api_key: str | None = Query(default=None)) -> str:
            role = _role_for(x_api_key or api_key)
            if role is None:
                raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
            if _ROLE_RANK[role] < _ROLE_RANK[min_role]:
                raise HTTPException(status_code=403, detail=f"requires {min_role} role, you are {role}")
            return role
        return dep

    guarded = [Depends(require_role("viewer"))]
    operator_or_above = [Depends(require_role("operator"))]
    admin_only = [Depends(require_role("admin"))]
    register_limiter = _SlidingWindowLimiter(register_rate_limit, 60.0)
    frame_limiter = _SlidingWindowLimiter(frame_rate_limit, 1.0)

    def rate_limit(kind: str):
        limiter = register_limiter if kind == "register" else frame_limiter

        def dependency(request: Request) -> None:
            camera_id = request.path_params.get("camera_id", "")
            client = request.client.host if request.client else "unknown"
            key = f"{client}:{camera_id}" if camera_id else client
            if not limiter.allow(key):
                logger.warning("rate limited %s request from %s camera=%s", kind, client, camera_id or "-")
                raise HTTPException(status_code=429, detail="rate limit exceeded")

        return dependency

    def _audit(role: str, action: str, detail: str) -> None:
        db.log("AUDIT", f"api/{role}", f"{action}: {detail}")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        # the page itself is public; its data endpoints still enforce api_key
        return _DASHBOARD_HTML.read_text(encoding="utf-8")

    @app.get("/config.js")
    def dashboard_config():
        # No-op default (same-origin) for this self-hosted copy of the
        # dashboard. A Vercel deployment (apps/web/) generates its own
        # config.js at build time instead — see scripts/build_web_dashboard.py.
        return Response(content="window.MONGDEE_API_BASE = \"\";", media_type="application/javascript")

    @app.get("/api/health")
    @app.get("/health")  # bare path some uptime/orchestration probes expect (section 29)
    def health():
        return {"status": "ok", "schema_version": db.schema_version, "live": live is not None}

    @app.get("/metrics")
    def metrics():
        payload: dict = {}
        if hardware_profile is not None:
            payload["hardware"] = dataclasses.asdict(hardware_profile)
        if remote_frames is not None:
            payload["remote_cameras"] = remote_frames.all_statuses()
        if live is not None:
            payload["live"] = live.snapshot()
        if pipeline is not None:
            payload["pipeline"] = {
                "target_fps_per_camera": pipeline.get_target_fps(),
                "cameras": {
                    cid: {
                        "frames_processed": s.frames_processed,
                        "effective_fps": round(s.effective_fps, 2),
                        "last_latency_ms": round(s.last_latency_sec * 1000, 1),
                        "last_detection_count": s.last_detection_count,
                        "last_track_count": s.last_track_count,
                    }
                    for cid, s in pipeline.all_stats().items()
                },
            }
        return payload

    # ------------------------------------------------ Camera Agent (remote)
    # See camera_agent/client.py and MongDee_Cloud_Vercel_Remote_AI_Server_
    # Master_Prompt.md sections 4, 32: a lightweight, network-connected
    # Camera Agent registers each of its cameras once, then pushes frames
    # continuously. Both require at least "operator" — these mutate live
    # state and, unlike a viewer's read-only GETs, cost real CPU/GPU time
    # once the pipeline picks the frame up.
    @app.post("/api/cameras/register", dependencies=operator_or_above + [Depends(rate_limit("register"))])
    def register_camera(payload: dict, role: str = Depends(require_role("operator"))):
        if remote_frames is None:
            raise HTTPException(status_code=503, detail="this server was not started with remote camera support")
        camera_id = payload.get("cameraId") or payload.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id.strip() or len(camera_id) > 128:
            raise HTTPException(status_code=400, detail="cameraId is required")
        camera_id = camera_id.strip()
        resolution = payload.get("resolution")
        try:
            if isinstance(resolution, str) and "x" in resolution:
                w, h = resolution.lower().split("x", 1)
                resolution = (int(w), int(h))
            elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
                resolution = (int(resolution[0]), int(resolution[1]))
            else:
                resolution = None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="resolution must be WIDTHxHEIGHT") from None
        if resolution is not None and (not all(1 <= value <= 16384 for value in resolution)):
            raise HTTPException(status_code=400, detail="resolution is out of range")
        fps = payload.get("fps")
        if fps is not None:
            try:
                fps = float(fps)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="fps must be a positive number") from None
            if not 0 < fps <= 240:
                raise HTTPException(status_code=400, detail="fps must be between 0 and 240")
        remote_frames.register(
            camera_id, name=payload.get("name", camera_id), resolution=resolution,
            fps=fps, codec=payload.get("codec", ""),
        )
        db.upsert_camera({
            "id": camera_id, "name": payload.get("name", camera_id), "protocol": "remote_agent",
            "url_redacted": "", "location": payload.get("location", ""), "status": "connecting",
            "resolution": f"{resolution[0]}x{resolution[1]}" if resolution else None, "fps": payload.get("fps"),
        })
        _audit(role, "camera_registered", camera_id)
        return {"cameraId": camera_id, "status": "connecting"}

    @app.post("/api/cameras/{camera_id}/frame", dependencies=operator_or_above + [Depends(rate_limit("frame"))])
    async def ingest_frame(
        camera_id: str,
        frame: UploadFile = File(...),
        timestamp: float | None = Form(default=None),
        sequence: int | None = Form(default=None),
    ):
        if remote_frames is None:
            raise HTTPException(status_code=503, detail="this server was not started with remote camera support")
        if camera_id not in remote_frames.camera_ids():
            raise HTTPException(status_code=404, detail=f"camera {camera_id!r} is not registered")
        raw = await frame.read(MAX_FRAME_UPLOAD_BYTES + 1)
        if len(raw) > MAX_FRAME_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="frame exceeds maximum upload size")
        buf = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="could not decode frame as an image")
        accepted = remote_frames.ingest_frame(
            camera_id, image, timestamp if timestamp is not None else time.time(), sequence,
        )
        return {"accepted": accepted}

    @app.get("/api/cameras/{camera_id}/stream", dependencies=guarded)
    def camera_stream(camera_id: str):
        if frame_source is None:
            raise HTTPException(status_code=503, detail="this server was not started with a live frame source")
        if camera_id not in frame_source.camera_ids():
            raise HTTPException(status_code=404, detail="camera not found")

        return StreamingResponse(
            _mjpeg_chunks(frame_source, camera_id), media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/cameras", dependencies=guarded)
    def cameras():
        rows = db.cameras()
        if live:
            snap = live.snapshot()["cameras"]
            for r in rows:
                r["live"] = snap.get(r["id"], {})
        return {"cameras": rows}

    @app.get("/api/cameras/{camera_id}/status", dependencies=guarded)
    def camera_status(camera_id: str):
        row = db.query_one("SELECT * FROM cameras WHERE id=?", (camera_id,))
        if row is None:
            raise HTTPException(status_code=404, detail="camera not found")
        if live:
            row["live"] = live.snapshot()["cameras"].get(camera_id, {})
        return row

    @app.get("/api/persons", dependencies=guarded)
    def persons(limit: int = Query(200, le=1000), since: float | None = None):
        return {"persons": db.global_persons(limit=limit, since=since)}

    @app.get("/api/persons/{person_id}", dependencies=guarded)
    def person(person_id: str):
        row = db.global_person(person_id)
        if row is None:
            raise HTTPException(status_code=404, detail="person not found")
        return row

    @app.get("/api/events", dependencies=guarded)
    def events(limit: int = Query(200, le=1000), event_type: str | None = None, since: float | None = None):
        return {"events": db.events(limit=limit, event_type=event_type, since=since)}

    @app.get("/api/stats", dependencies=guarded)
    def stats():
        s = db.stats()
        s["camera_person_counts"] = db.camera_person_counts()
        if live:
            s["live"] = live.snapshot()
        return s

    # --- spatial / product / interest (sections 76-77) ---------------

    @app.get("/designer", response_class=HTMLResponse)
    def designer():
        return _DESIGNER_HTML.read_text(encoding="utf-8")

    @app.get("/api/booths", dependencies=guarded)
    def booths():
        return {"booths": db.booths()}

    @app.get("/api/booths/{booth_id}/layout", dependencies=guarded)
    def get_layout(booth_id: str):
        layout = db.active_layout(booth_id)
        if layout is None:
            raise HTTPException(status_code=404, detail="no active layout for booth")
        return layout

    @app.get("/api/booths/{booth_id}/layout/versions", dependencies=guarded)
    def layout_versions(booth_id: str):
        return {"versions": db.layout_versions(booth_id)}

    @app.post("/api/booths/{booth_id}/layout")
    def post_layout(booth_id: str, layout: dict, role: str = Depends(require_role("admin"))):
        try:
            _validate_layout(layout)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        db.upsert_booth({"id": booth_id, "name": layout.get("name", booth_id), "width": layout["width"], "length": layout["length"], "unit": layout.get("unit", "m")})
        version = db.save_layout(booth_id, layout, created_by=role)
        _audit(role, "layout_saved", f"{booth_id} v{version}")
        return {"booth_id": booth_id, "version": version, "active": True}

    @app.post("/api/booths/{booth_id}/layout/{version}/activate")
    def activate_layout(booth_id: str, version: int, role: str = Depends(require_role("admin"))):
        if not db.activate_layout(booth_id, version):
            raise HTTPException(status_code=404, detail="layout version not found")
        _audit(role, "layout_activated", f"{booth_id} v{version}")
        return {"booth_id": booth_id, "active_version": version}

    @app.get("/api/audit", dependencies=admin_only)
    def audit_log(limit: int = Query(200, le=1000)):
        return {"entries": db.query(
            "SELECT * FROM system_logs WHERE level='AUDIT' ORDER BY timestamp DESC LIMIT ?", (limit,)
        )}

    @app.get("/api/products", dependencies=guarded)
    def products():
        return {"products": db.products()}

    @app.get("/api/products/{product_id}", dependencies=guarded)
    def product(product_id: str):
        row = db.product(product_id)
        if row is None:
            raise HTTPException(status_code=404, detail="product not found")
        row["analytics"] = db.product_analytics(product_id)
        return row

    @app.get("/api/products/{product_id}/interest", dependencies=guarded)
    def product_interest(product_id: str):
        return db.product_analytics(product_id)

    @app.get("/api/persons/{person_id}/movement", dependencies=guarded)
    def person_movement(person_id: str):
        return {"person_id": person_id, "path": db.person_movement(person_id)}

    @app.get("/api/persons/{person_id}/interests", dependencies=guarded)
    def person_interests(person_id: str):
        return {"person_id": person_id, "interests": db.person_interests(person_id)}

    @app.get("/api/zones/{zone_id}/analytics", dependencies=guarded)
    def zone_analytics(zone_id: str):
        return db.zone_analytics(zone_id)

    @app.get("/api/analytics/heatmap", dependencies=guarded)
    def heatmap(kind: str = Query("traffic"), since: float | None = None):
        if analytics is None:
            raise HTTPException(status_code=503, detail="no live analytics available")
        try:
            return analytics.heatmap(kind, since=since)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/map", dependencies=guarded)
    def live_map():
        if analytics is None:
            return {"positions": [], "interest": []}
        return {
            "positions": [
                {"global_id": gid, "x": p.x, "y": p.y, "zone_id": p.zone_id,
                 "camera_id": p.camera_id, "confidence": p.confidence, "timestamp": p.timestamp}
                for gid, p in analytics.positions().items()
            ],
            "interest": analytics.active_interest(),
        }

    def _ws_key_ok(websocket: WebSocket) -> bool:
        return not api_key or websocket.query_params.get("api_key") == api_key

    @app.websocket("/ws/map")
    async def ws_map(websocket: WebSocket):
        if not _ws_key_ok(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                if analytics is not None:
                    await websocket.send_json({
                        "positions": [
                            {"global_id": gid, "x": p.x, "y": p.y, "zone_id": p.zone_id}
                            for gid, p in analytics.positions().items()
                        ],
                        "interest": analytics.active_interest(),
                    })
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        if not _ws_key_ok(websocket):
            await websocket.close(code=1008)
            return
        if event_hub is None:
            await websocket.close(code=1011)
            return
        await websocket.accept()
        q = await event_hub.subscribe()
        try:
            while True:
                payload = await q.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            event_hub.unsubscribe(q)

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(websocket: WebSocket):
        if not _ws_key_ok(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                payload = {"stats": db.stats()}
                if live:
                    payload["live"] = live.snapshot()
                await websocket.send_json(payload)
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            pass

    return app
