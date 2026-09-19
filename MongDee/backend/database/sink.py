"""DatabaseSink — persists the person pipeline's output.

Wire it to a running system by passing its handler methods as the
pipeline / gateway callbacks:

    sink = DatabaseSink(Database())
    gateway = CameraGateway(on_status=sink.on_camera_status)
    pipeline = DetectionPipeline(..., on_person_event=sink.on_person_event,
                                 on_result=sink.on_result)

It writes: global_persons + camera_tracks + events on every person event,
camera status changes, and (optionally, sampled) a detection log. All
writes are best-effort — a DB error is logged, never raised into the
pipeline thread (section 32 / 87: one component failing must not stop the
rest).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from backend.database.db import Database
from vision.events import PERSON_CAMERA_TRANSITION, PERSON_REIDENTIFIED, PersonEvent
from vision.identity.manager import GlobalIdentityManager
from vision.pipeline import PipelineResult

logger = logging.getLogger("mongdee.backend.sink")


class DatabaseSink:
    def __init__(
        self,
        db: Database,
        identity_manager: GlobalIdentityManager | None = None,
        log_detections: bool = False,
        detection_sample_interval_sec: float = 1.0,
    ):
        self._db = db
        self._identity = identity_manager
        self._log_detections = log_detections
        self._detection_interval = detection_sample_interval_sec
        self._last_detection_write: dict[str, float] = {}

    # -- callbacks ---------------------------------------------------------
    def on_camera_status(self, camera_id: str, status, message: str = "") -> None:
        try:
            self._db.set_camera_status(camera_id, getattr(status, "value", str(status)))
        except Exception:
            logger.exception("failed to persist camera status for %s", camera_id)

    def register_camera(self, camera_config) -> None:
        from camera.base import redact_url

        try:
            self._db.upsert_camera(
                {
                    "id": camera_config.id,
                    "name": camera_config.name,
                    "protocol": camera_config.protocol.value,
                    "url_redacted": redact_url(camera_config.url) if camera_config.url else "",
                    "location": camera_config.location,
                    "status": "unknown",
                    "resolution": "",
                    "fps": camera_config.processing_fps,
                    "enabled": 1 if camera_config.enabled else 0,
                    "updated_at": time.time(),
                }
            )
        except Exception:
            logger.exception("failed to register camera %s", camera_config.id)

    def on_person_event(self, event: PersonEvent) -> None:
        try:
            self._db.insert_event(event.to_dict())
            if event.event_type == PERSON_CAMERA_TRANSITION:
                p = event.payload
                self._db.insert_transition(
                    p.get("from_camera", ""), p.get("to_camera", ""), event.global_id, p.get("transit_seconds", 0.0)
                )
            if event.global_id and self._identity is not None:
                self._persist_person(event)
        except Exception:
            logger.exception("failed to persist person event %s", event.event_type)

    def on_result(self, result: PipelineResult) -> None:
        if not self._log_detections or not result.detections:
            return
        now = time.time()
        if now - self._last_detection_write.get(result.camera_id, 0.0) < self._detection_interval:
            return
        self._last_detection_write[result.camera_id] = now
        try:
            self._db.insert_detections(
                [
                    {
                        "camera_id": result.camera_id,
                        "local_track_id": d.local_track_id,
                        "x1": d.bbox[0], "y1": d.bbox[1], "x2": d.bbox[2], "y2": d.bbox[3],
                        "confidence": d.confidence, "timestamp": d.timestamp,
                    }
                    for d in result.detections
                ]
            )
        except Exception:
            logger.exception("failed to persist detections for %s", result.camera_id)

    # -- helpers --------------------------------------------------------
    def _persist_person(self, event: PersonEvent) -> None:
        person = self._identity.get(event.global_id)
        if person is None:
            return
        attrs = person.attributes
        self._db.upsert_global_person(
            {
                "id": person.global_id,
                "first_seen": person.first_seen,
                "last_seen": person.last_seen,
                "estimated_gender": attrs.gender.value,
                "gender_confidence": attrs.gender.confidence,
                "estimated_age_group": attrs.age_group.value,
                "age_confidence": attrs.age_group.confidence,
                "shirt_color": attrs.shirt_color.value,
                "pants_color": attrs.pants_color.value,
                "cameras_seen": person.cameras_seen,
                "track_count": len(person.track_refs),
                "uncertain_links": list(person.uncertain_links),
            }
        )
        match_status = "match" if event.event_type == PERSON_REIDENTIFIED else "new"
        self._db.insert_track(
            {
                "camera_id": event.camera_id,
                "local_track_id": event.local_track_id,
                "global_person_id": person.global_id,
                "start_time": person.last_seen,
                "end_time": person.last_seen,
                "embedding_samples": event.payload.get("embedding_samples", 0),
                "quality": 0.0,
                "match_status": match_status,
                "match_score": event.match_score,
            }
        )
        if person.embedding is not None:
            vec = np.asarray(person.embedding, dtype=np.float32)
            self._db.insert_embedding(
                person.global_id, event.camera_id, event.local_track_id,
                vec.tobytes(), int(vec.size), 0.0,
            )
