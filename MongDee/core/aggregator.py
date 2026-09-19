"""Combines detections from every active camera into one booth-level decision.

This is the "หลายกล้องช่วยกัน" behaviour: a product is confirmed as soon as
two or more cameras agree on it at the same time (fast path — one camera's
blind spot is covered by another), or after a short continuous sighting from
a single camera (slow path — still works with just one webcam).
"""

from __future__ import annotations

import threading
import time

CONCURRENT_WINDOW_SEC = 0.6      # cameras "agree" if both saw it within this window
SINGLE_CAMERA_STABLE_SEC = 1.2   # continuous sighting needed from just one camera
IDLE_RESET_SEC = 3.0             # no sightings at all for this long -> booth goes idle
RECHANGE_COOLDOWN_SEC = 2.0      # minimum time before swapping to a different product


class DetectionAggregator:
    def __init__(self):
        # class_name -> {camera_id: last_seen_ts}
        self._sightings: dict[str, dict[str, float]] = {}
        # class_name -> first continuous sighting start ts (any camera)
        self._first_seen: dict[str, float] = {}
        # class_name -> {camera_id: latest confidence}
        self._confidences: dict[str, dict[str, float]] = {}
        self.current_product: str | None = None
        self._current_since = 0.0
        self._last_change_ts = 0.0
        # Browser-mode callbacks arrive concurrently from camera threads.
        self._lock = threading.RLock()

    def update(self, camera_id: str, detections: list[dict]) -> dict | None:
        """Feed one camera's latest detections. Returns a confirmation event or None."""
        now = time.monotonic()
        with self._lock:
            # A camera can disappear without ever sending an empty result.
            # Remove expired evidence globally before accepting this update so
            # a new camera cannot inherit another camera's old stable period.
            self._drop_expired(now)

            seen_classes = set()
            for det in detections:
                name = det["class_name"]
                seen_classes.add(name)
                self._sightings.setdefault(name, {})[camera_id] = now
                self._confidences.setdefault(name, {})[camera_id] = float(det.get("conf", 0.0))
                self._first_seen.setdefault(name, now)

            # Drop stale evidence for classes absent from this camera's newest result.
            for name, cams in list(self._sightings.items()):
                if camera_id in cams and name not in seen_classes:
                    del cams[camera_id]
                    self._confidences.get(name, {}).pop(camera_id, None)
                    if not cams:
                        del self._sightings[name]
                        self._first_seen.pop(name, None)
                        self._confidences.pop(name, None)

            return self._decide(now)

    def _drop_expired(self, now: float) -> None:
        for name, cameras in list(self._sightings.items()):
            for stale_camera in [
                camera_id
                for camera_id, seen_at in cameras.items()
                if now - seen_at > CONCURRENT_WINDOW_SEC
            ]:
                del cameras[stale_camera]
                self._confidences.get(name, {}).pop(stale_camera, None)
            if not cameras:
                del self._sightings[name]
                self._first_seen.pop(name, None)
                self._confidences.pop(name, None)

    def _decide(self, now: float) -> dict | None:
        # Nothing seen by anyone recently -> go idle.
        most_recent = max(
            (ts for cams in self._sightings.values() for ts in cams.values()),
            default=0.0,
        )
        if self.current_product and now - most_recent > IDLE_RESET_SEC:
            self.current_product = None

        best_class = None
        best_reason = None
        for name, cams in self._sightings.items():
            active = [t for t in cams.values() if now - t <= CONCURRENT_WINDOW_SEC]
            if len(set(cams.keys())) >= 2 and len(active) >= 2:
                best_class, best_reason = name, "multi_camera_agreement"
                break
            since = self._first_seen.get(name, now)
            if now - since >= SINGLE_CAMERA_STABLE_SEC and active:
                if best_class is None:
                    best_class, best_reason = name, "single_camera_stable"

        if best_class is None or best_class == self.current_product:
            return None
        if now - self._last_change_ts < RECHANGE_COOLDOWN_SEC:
            return None

        active_cameras = {
            camera_id
            for camera_id, seen_at in self._sightings.get(best_class, {}).items()
            if now - seen_at <= CONCURRENT_WINDOW_SEC
        }
        confidence = max(
            (conf for camera_id, conf in self._confidences.get(best_class, {}).items()
             if camera_id in active_cameras),
            default=0.0,
        )

        self.current_product = best_class
        self._last_change_ts = now
        self._current_since = now
        cameras = sorted(active_cameras)
        return {
            "class_name": best_class,
            "confidence": confidence,
            "reason": best_reason,
            "cameras": cameras,
        }
