"""Vision Pipeline — the glue that pulls the latest frame from each camera
(via the Camera Gateway) and runs it through the full person pipeline:

    detection  ->  tracking  ->  per-track feature accumulation  ->  global identity

Each stage after detection is optional (pass the corresponding object to
the constructor). With all of them wired up, the pipeline emits
PersonEvents (NEW_PERSON / PERSON_REIDENTIFIED / POSSIBLE_MATCH /
PERSON_CAMERA_TRANSITION) as local tracks resolve to Global Person IDs.

Design for a CPU-only box (this project's constraint): ONE worker thread
that round-robins every camera and shares ONE detector / extractor set.
Effective per-camera rate is roughly (1 / stage_time) / camera_count
(MongDee_Master_Prompt.md section 29). Frame selection follows the
gateway's "always latest, never queue" policy (section 31).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Protocol

from camera.base import Frame
from vision.detection.detector import Detection, PersonDetector
from vision.events import (
    PERSON_CAMERA_TRANSITION,
    PERSON_NEW,
    PERSON_REIDENTIFIED,
    PERSON_TRACK_ENDED,
    POSSIBLE_MATCH,
    PersonEvent,
)
from vision.identity.manager import GlobalIdentityManager
from vision.identity.matcher import MATCH, UNCERTAIN
from vision.track_features import TrackFeatureStore
from vision.tracking.multi_tracker import MultiCameraTracker
from vision.tracking.track import Track

logger = logging.getLogger("mongdee.vision.pipeline")


@dataclasses.dataclass
class PipelineResult:
    camera_id: str
    frame: Frame
    detections: list[Detection]
    tracks: list[Track]  # empty when no tracker is configured


ResultCallback = Callable[[PipelineResult], None]
DetectionsCallback = Callable[[str, list[Detection], Frame], None]
PersonEventCallback = Callable[[PersonEvent], None]


class FrameSource(Protocol):
    def camera_ids(self) -> list[str]: ...
    def latest_frame(self, camera_id: str) -> Frame | None: ...


@dataclasses.dataclass
class CameraDetectionStats:
    frames_processed: int = 0
    last_detection_count: int = 0
    last_track_count: int = 0
    last_latency_sec: float = 0.0
    effective_fps: float = 0.0
    _window_start: float = 0.0
    _window_count: int = 0


class DetectionPipeline:
    def __init__(
        self,
        frame_source: FrameSource,
        detector: PersonDetector,
        tracker: MultiCameraTracker | None = None,
        feature_store: TrackFeatureStore | None = None,
        identity_manager: GlobalIdentityManager | None = None,
        on_detections: DetectionsCallback | None = None,
        on_result: ResultCallback | None = None,
        on_person_event: PersonEventCallback | None = None,
    ):
        if feature_store is not None and tracker is None:
            raise ValueError("feature_store requires a tracker (features are keyed by local track id)")
        if identity_manager is not None and feature_store is None:
            raise ValueError("identity_manager requires a feature_store (it consumes TrackSummary objects)")

        self._source = frame_source
        self._detector = detector
        self._tracker = tracker
        self._feature_store = feature_store
        self._identity = identity_manager
        self._on_detections = on_detections
        self._on_result = on_result
        self._on_person_event = on_person_event
        self._target_fps_per_camera = detector.config.target_fps_per_camera
        self._min_interval = (
            1.0 / self._target_fps_per_camera if self._target_fps_per_camera > 0 else 0.0
        )

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, list[Detection]] = {}
        self._latest_tracks: dict[str, list[Track]] = {}
        self._stats: dict[str, CameraDetectionStats] = {}
        self._last_index: dict[str, int] = {}
        self._last_run: dict[str, float] = {}
        self._eager_submitted: set[tuple[str, int]] = set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="vision-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=15.0)
            if thread.is_alive():
                logger.warning("vision pipeline worker still alive 15s after stop()")
        self._flush_all_tracks()

    def latest_detections(self, camera_id: str) -> list[Detection]:
        with self._lock:
            return list(self._latest.get(camera_id, []))

    def latest_tracks(self, camera_id: str) -> list[Track]:
        with self._lock:
            return list(self._latest_tracks.get(camera_id, []))

    def unique_person_count(self) -> int:
        return self._identity.unique_count() if self._identity else 0

    def stats(self, camera_id: str) -> CameraDetectionStats | None:
        with self._lock:
            found = self._stats.get(camera_id)
            return dataclasses.replace(found) if found is not None else None

    def all_stats(self) -> dict[str, CameraDetectionStats]:
        with self._lock:
            return {cid: dataclasses.replace(s) for cid, s in self._stats.items()}

    # ---------------------------------------- adaptive control (backend/adaptive.py)
    def get_target_fps(self) -> float:
        return self._target_fps_per_camera

    def set_target_fps(self, fps: float) -> None:
        """Change the detection pacing at runtime — takes effect on the
        very next _process_one() call, no restart needed. One shared
        detector serves every camera (see module docstring), so this rate
        applies process-wide, not per-camera."""
        self._target_fps_per_camera = max(0.0, fps)
        self._min_interval = 1.0 / self._target_fps_per_camera if self._target_fps_per_camera > 0 else 0.0

    # -- worker -----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            progressed = False
            for camera_id in self._source.camera_ids():
                if self._stop_event.is_set():
                    return
                if self._process_one(camera_id):
                    progressed = True
            self._reconcile_all()
            if not progressed and self._stop_event.wait(0.05):
                return

    def _process_one(self, camera_id: str) -> bool:
        now = time.monotonic()
        if self._min_interval and now - self._last_run.get(camera_id, 0.0) < self._min_interval:
            return False

        frame = self._source.latest_frame(camera_id)
        if frame is None or frame.frame_index == self._last_index.get(camera_id):
            return False

        t0 = time.perf_counter()
        try:
            detections = self._detector.detect(frame.image, camera_id, frame.timestamp)
            tracks: list[Track] = []
            if self._tracker is not None:
                tracks = self._tracker.update(camera_id, detections, frame.timestamp)
            if self._feature_store is not None:
                for track in tracks:
                    self._feature_store.observe(camera_id, track, frame.image, frame.timestamp)
                if self._identity is not None:
                    self._eager_submit(camera_id, tracks)
        except Exception:
            logger.exception("vision pipeline failed for camera %s", camera_id)
            return False
        latency = time.perf_counter() - t0

        self._last_run[camera_id] = now
        self._last_index[camera_id] = frame.frame_index
        with self._lock:
            self._latest[camera_id] = detections
            self._latest_tracks[camera_id] = tracks
            self._update_stats_locked(camera_id, len(detections), len(tracks), latency)

        result = PipelineResult(camera_id=camera_id, frame=frame, detections=detections, tracks=tracks)
        self._emit(self._on_detections, camera_id, detections, frame, name="on_detections")
        self._emit(self._on_result, result, name="on_result")
        return True

    def _eager_submit(self, camera_id: str, tracks: list[Track]) -> None:
        """Give a still-live track a provisional Global Person ID once it
        has enough Re-ID evidence, so the booth map can show the person
        before their track ends. The final (better-aggregated) summary is
        re-submitted on flush and merges into the same person via the
        manager's per-track mapping."""
        threshold = self._feature_store.eager_submit_samples
        if threshold <= 0:
            return
        for track in tracks:
            key = (camera_id, track.track_id)
            if key in self._eager_submitted:
                continue
            if self._feature_store.reid_sample_count(camera_id, track.track_id) < threshold:
                continue
            summary = self._feature_store.build_summary(camera_id, track.track_id)
            if summary is None or summary.embedding is None:
                continue
            self._eager_submitted.add(key)
            global_id, result = self._identity.submit_track(summary)
            self._record_identity_result(camera_id, track.track_id, global_id, result)

    def _reconcile_all(self) -> None:
        """Flush any track that has ended: removed by its tracker, or not
        observed for track_end_timeout_sec (its camera went quiet)."""
        if self._feature_store is None or self._identity is None:
            return
        now = time.time()
        timeout = self._feature_store.track_end_timeout_sec
        for camera_id in self._feature_store.camera_ids():
            live = {t.track_id for t in self._tracker.tracks(camera_id)}
            for track_id in list(self._feature_store.track_ids_for_camera(camera_id)):
                last_seen = self._feature_store.last_seen(camera_id, track_id)
                ended = track_id not in live
                timed_out = last_seen is not None and (now - last_seen) > timeout
                if ended or timed_out:
                    self._flush_track(camera_id, track_id)

    def _flush_track(self, camera_id: str, track_id: int) -> None:
        summary = self._feature_store.pop(camera_id, track_id)
        if summary is None:
            return
        self._emit(
            self._on_person_event,
            PersonEvent.now(
                event_type=PERSON_TRACK_ENDED, camera_id=camera_id, global_id="",
                local_track_id=track_id, payload={"embedding_samples": summary.embedding_samples},
            ),
            name="on_person_event",
        )
        if summary.embedding_samples < self._feature_store.min_samples_to_flush:
            return

        global_id, result = self._identity.submit_track(summary)
        self._record_identity_result(camera_id, track_id, global_id, result)

    def _record_identity_result(self, camera_id: str, track_id: int, global_id: str, result) -> None:
        if result.breakdown.get("same_track"):
            return  # a re-submit of an already-announced track (eager submit -> flush); nothing new to report
        person = self._identity.get(global_id)
        if result.status == MATCH:
            self._emit_person_event(
                PERSON_REIDENTIFIED, camera_id, global_id, track_id, result.score,
                {"breakdown": result.breakdown},
            )
            if person is not None and len(person.cameras_seen) >= 2 and person.cameras_seen[-1] == camera_id:
                prev_cam = person.cameras_seen[-2]
                self._emit_person_event(
                    PERSON_CAMERA_TRANSITION, camera_id, global_id, track_id, result.score,
                    {"from_camera": prev_cam, "to_camera": camera_id},
                )
        elif result.status == UNCERTAIN:
            self._emit_person_event(
                POSSIBLE_MATCH, camera_id, global_id, track_id, result.score,
                {"candidate": result.candidate_id, "breakdown": result.breakdown},
            )
        else:
            self._emit_person_event(PERSON_NEW, camera_id, global_id, track_id, result.score, {})

    def _emit_person_event(self, event_type, camera_id, global_id, track_id, score, payload) -> None:
        self._emit(
            self._on_person_event,
            PersonEvent.now(
                event_type=event_type, camera_id=camera_id, global_id=global_id,
                local_track_id=track_id, match_score=score, payload=payload,
            ),
            name="on_person_event",
        )

    def _flush_all_tracks(self) -> None:
        if self._feature_store is None or self._identity is None:
            return
        for camera_id in self._source.camera_ids():
            for track_id in list(self._feature_store.track_ids_for_camera(camera_id)):
                self._flush_track(camera_id, track_id)

    def _emit(self, callback, *args, name: str) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("%s callback raised", name)

    def _update_stats_locked(self, camera_id: str, det_count: int, track_count: int, latency: float) -> None:
        stats = self._stats.setdefault(camera_id, CameraDetectionStats())
        stats.frames_processed += 1
        stats.last_detection_count = det_count
        stats.last_track_count = track_count
        stats.last_latency_sec = latency
        now = time.monotonic()
        if stats._window_start == 0.0:
            stats._window_start = now
        stats._window_count += 1
        elapsed = now - stats._window_start
        if elapsed >= 2.0:
            stats.effective_fps = stats._window_count / elapsed
            stats._window_start = now
            stats._window_count = 0
