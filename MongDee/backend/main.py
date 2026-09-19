"""MongDee vision system entrypoint — wires every phase together and runs
the API (MongDee_Master_Prompt.md sections 44, 46, 86).

    python -m backend.main configs/mongdee.example.json

Pipeline:  cameras -> CameraGateway -> DetectionPipeline
           (detector -> tracker -> feature store -> global identity)
           -> DatabaseSink + EventHub + LiveState -> FastAPI (REST + WS)

Ctrl+C shuts the pipeline and gateway down cleanly, flushing any
in-progress tracks to the database first.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.adaptive import PipelineAdaptiveController
from backend.api.app import EventHub, LiveState, create_app
from backend.config import MongDeeConfig
from backend.database.db import DEFAULT_DB_PATH, Database
from backend.database.sink import DatabaseSink
from backend.remote_frames import CompositeFrameSource, RemoteFrameHealthWatchdog, RemoteFrameReceiver
from camera import CameraGateway
from core.performance import PerformanceMonitor, detect_hardware
from vision.attributes.extractor import AttributeExtractor
from vision.booth_analytics import BoothAnalytics
from vision.detection.detector import PersonDetector
from vision.identity.manager import GlobalIdentityManager
from vision.pipeline import DetectionPipeline
from vision.product.classifier import ProductClassifier
from vision.product.database import GIProductDatabase
from vision.product.pipeline import ProductVisionModule
from vision.reid.extractor import ReIDExtractor
from vision.spatial.calibration import CalibrationStore
from vision.spatial.booth import BoothLayout
from vision.track_features import TrackFeatureStore
from vision.tracking.multi_tracker import MultiCameraTracker

logger = logging.getLogger("mongdee.backend.main")


class MongDeeSystem:
    def __init__(self, config: MongDeeConfig):
        self.config = config

        db_path = config.database.path or DEFAULT_DB_PATH
        self.db = Database(db_path)
        self.live = LiveState()
        self.event_hub = EventHub()

        self.detector = PersonDetector(config.detection)
        self.tracker = MultiCameraTracker(config.tracking)
        self.feature_store = TrackFeatureStore(
            ReIDExtractor(config.reid), AttributeExtractor(config.attributes), config.features
        )
        self.identity = GlobalIdentityManager(config.identity, config.topology)
        self.sink = DatabaseSink(
            self.db, self.identity,
            log_detections=config.database.log_detections,
            detection_sample_interval_sec=config.database.detection_sample_interval_sec,
        )

        self.hardware_profile = detect_hardware()
        logger.info(self.hardware_profile.summary_line())

        self.gateway = CameraGateway(on_status=self._on_camera_status)
        for cam in config.cameras:
            self.sink.register_camera(cam)
            self.gateway.add_camera(cam, start=False)

        # Camera Agent support (MongDee_Cloud_Vercel_Remote_AI_Server_Master_
        # Prompt.md sections 4, 32, 57): a lightweight, network-connected
        # agent (camera_agent/) pushes frames here via the API instead of
        # this process capturing them locally. frame_source merges both
        # kinds of camera into one FrameSource so the pipeline below never
        # needs to know or care which is which.
        self.remote_frames = RemoteFrameReceiver(on_status=self._on_camera_status)
        self.remote_watchdog = RemoteFrameHealthWatchdog(self.remote_frames)
        self.frame_source = CompositeFrameSource(self.gateway, self.remote_frames)

        # --- optional spatial layer (sections 58-69) ---
        self.calibrations = CalibrationStore()
        self.layout: BoothLayout | None = None
        self.analytics: BoothAnalytics | None = None
        self._layout_version = 1
        if config.spatial.enabled:
            self._build_spatial()

        # --- optional product layer (sections 48-51) ---
        self.product_module: ProductVisionModule | None = None
        self._last_product_write: dict[str, float] = {}
        if config.product_pipeline.enabled:
            self._build_product()

        self.pipeline = DetectionPipeline(
            self.frame_source, self.detector, tracker=self.tracker,
            feature_store=self.feature_store, identity_manager=self.identity,
            on_result=self._on_result, on_person_event=self._on_person_event,
        )

        # Adaptive Performance Engine (section 34) — reacts to this
        # process's own measured CPU load and detection latency, not a
        # static hardware guess (section 93 rule 27).
        self.performance_monitor = PerformanceMonitor()
        ac = config.adaptive
        self.adaptive_controller = PipelineAdaptiveController(
            self.pipeline, self.detector, self.performance_monitor,
            tick_interval_sec=ac.tick_interval_sec, cpu_high_pct=ac.cpu_high_pct,
            latency_high_ms=ac.latency_high_ms, min_target_fps=ac.min_target_fps,
            max_target_fps=ac.max_target_fps or None, imgsz_levels=tuple(ac.imgsz_levels),
        ) if ac.enabled else None

    def _build_spatial(self):
        sp = self.config.spatial
        if sp.calibration_path:
            self.calibrations = CalibrationStore.load(sp.calibration_path)
        if sp.layout_path:
            self.layout = BoothLayout.from_dict(json.loads(Path(sp.layout_path).read_text(encoding="utf-8")))
            self._layout_version = self.layout.version
            self.db.upsert_booth({
                "id": self.layout.booth_id, "name": self.layout.booth_id,
                "width": self.layout.width, "length": self.layout.length, "unit": self.layout.unit,
            })
            self.db.save_layout(self.layout.booth_id, self.layout.to_dict())
        self.analytics = BoothAnalytics(
            self.layout, self.calibrations, self.config.interest,
            on_interest_event=self._on_interest_event,
        )
        logger.info("spatial layer on: %d calibrated camera(s), layout=%s",
                    len(self.calibrations.camera_ids()), self.layout.booth_id if self.layout else "none")

    def _build_product(self):
        pp = self.config.product_pipeline
        gi_db = GIProductDatabase.load(pp.gi_database_path) if pp.gi_database_path else GIProductDatabase()
        for product in gi_db.all():
            self.db.upsert_product(product.to_dict())
        classifier = ProductClassifier(self.config.product)
        self.product_module = ProductVisionModule(self.detector, self.config.product, classifier, gi_db)
        if pp.gallery_dir:
            n = self.product_module.load_gallery_from_dir(pp.gallery_dir)
            logger.info("product layer on: %d GI products, %d reference images", len(gi_db.ids()), n)

    # -- callbacks -------------------------------------------------------
    def _on_camera_status(self, camera_id, status, message):
        self.live.update_camera_status(camera_id, getattr(status, "value", str(status)), message)
        self.sink.on_camera_status(camera_id, status, message)

    def _on_result(self, result):
        stats = self.pipeline.stats(result.camera_id)
        if stats:
            self.live.update_camera_counts(
                result.camera_id, stats.last_detection_count, stats.last_track_count,
                stats.effective_fps, stats.last_latency_sec * 1000,
            )
        self.live.set_unique_count(self.identity.unique_count())
        self.sink.on_result(result)

        if self.analytics is not None:
            for track in result.tracks:
                gid = self.identity.global_id_for_track(result.camera_id, track.track_id)
                if gid is None:
                    continue
                pos = self.analytics.observe_person(result.camera_id, gid, track.bbox, result.frame.timestamp)
                if pos is not None:
                    self._safe(lambda: self.db.insert_position({
                        "global_person_id": gid, "camera_id": result.camera_id, "x": pos.x, "y": pos.y,
                        "zone_id": pos.zone_id, "confidence": pos.confidence, "timestamp": pos.timestamp,
                    }))

        if self.product_module is not None:
            self._run_product(result)

    def _run_product(self, result):
        # products are near-static — run this rarely (config: product.target_fps_per_camera)
        interval = 1.0 / max(self.config.product.target_fps_per_camera, 1e-6)
        now = time.time()
        if now - self._last_product_write.get(result.camera_id, 0.0) < interval:
            return
        self._last_product_write[result.camera_id] = now

        calib = self.calibrations.get(result.camera_id)
        try:
            detections = self.product_module.process(
                result.camera_id, result.frame.image, result.frame.timestamp, calibration=calib, layout=self.layout
            )
        except Exception:
            logger.exception("product module failed for %s", result.camera_id)
            return
        for pd in detections:
            self._safe(lambda pd=pd: self.db.insert_product_detection({
                "product_id": pd.product_id, "camera_id": pd.camera_id, "local_track_id": pd.local_track_id,
                "status": pd.status, "zone_id": pd.zone_id, "bbox": pd.bbox,
                "confidence": pd.confidence, "timestamp": pd.timestamp,
            }))

    def _on_interest_event(self, event):
        self._safe(lambda: self.db.insert_interest_event({
            "event_type": event.event_type, "global_id": event.global_id, "target_id": event.target_id,
            "status": event.status, "score": event.score, "look_duration": event.look_duration,
            "dwell_duration": event.dwell_duration, "signals": event.signals,
            "timestamp": event.timestamp, "layout_version": self._layout_version,
        }))

    def _on_person_event(self, event):
        self.sink.on_person_event(event)
        self.event_hub.publish(event)
        self.live.set_unique_count(self.identity.unique_count())

    @staticmethod
    def _safe(fn):
        try:
            fn()
        except Exception:
            logger.exception("sink write failed")

    # -- lifecycle -----------------------------------------------------
    def start(self):
        logger.info("loading detector (%s on %s)...", self.config.detection.model_path, self.detector.device)
        self.detector.warmup()
        self.gateway.start_all()
        self.remote_watchdog.start()
        self.pipeline.start()
        if self.adaptive_controller is not None:
            self.adaptive_controller.start()
        logger.info("MongDee vision pipeline running: %d local camera(s), remote agents connect via the API",
                    len(self.config.cameras))

    def stop(self):
        logger.info("stopping MongDee vision pipeline...")
        if self.adaptive_controller is not None:
            self.adaptive_controller.stop()
        self.pipeline.stop()   # flushes in-progress tracks to identity/db
        self.remote_watchdog.stop()
        self.gateway.stop_all()
        logger.info("stopped. unique persons this session: %d", self.identity.unique_count())

    def close(self):
        """Release the database. Call after stop() when fully done."""
        self.db.close()

    def build_app(self):
        return create_app(
            self.db, live=self.live, event_hub=self.event_hub,
            api_key=self.config.api.api_key or None, analytics=self.analytics,
            keys=self.config.api.keys or None,
            remote_frames=self.remote_frames, frame_source=self.frame_source,
            hardware_profile=self.hardware_profile, pipeline=self.pipeline,
            cors_origins=self.config.api.cors_origins or None,
        )


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(argv) != 2:
        print("usage: python -m backend.main <config.json>", file=sys.stderr)
        return 2
    try:
        config = MongDeeConfig.load(argv[1])
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    import uvicorn

    system = MongDeeSystem(config)
    system.start()

    stopping = {"done": False}

    def handle_signal(signum, frame):
        if not stopping["done"]:
            stopping["done"] = True
            system.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        uvicorn.run(system.build_app(), host=config.api.host, port=config.api.port, log_level="warning")
    finally:
        if not stopping["done"]:
            system.stop()
        system.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
