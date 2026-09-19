"""MultiCameraTracker — one ByteTracker per camera, created on demand.

This is the object the pipeline talks to. Each camera's tracker is fully
independent: `local_track_id` values are unique only within a camera, and
the same physical person seen by two cameras gets two unrelated ids.
Merging those into a Global Person ID is Phase 6.
"""

from __future__ import annotations

import threading

from vision.detection.detector import Detection
from vision.tracking.bytetrack import ByteTracker
from vision.tracking.config import TrackingConfig
from vision.tracking.track import Track


class MultiCameraTracker:
    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self._trackers: dict[str, ByteTracker] = {}
        self._lock = threading.Lock()

    def update(self, camera_id: str, detections: list[Detection], timestamp: float) -> list[Track]:
        """Runs the camera's tracker for one frame. Mutates each matched
        Detection's `local_track_id` in place, and returns the camera's
        active (confirmed, seen-this-frame) tracks."""
        with self._lock:
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = ByteTracker(self.config)
                self._trackers[camera_id] = tracker
        return tracker.update(detections, timestamp)

    def tracks(self, camera_id: str) -> list[Track]:
        tracker = self._trackers.get(camera_id)
        return tracker.tracks if tracker else []

    def active_tracks(self, camera_id: str) -> list[Track]:
        tracker = self._trackers.get(camera_id)
        return tracker.active_tracks() if tracker else []

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._trackers)
