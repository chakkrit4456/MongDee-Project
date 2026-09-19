"""Kalman filter for single-object bounding-box tracking, in the
(center-x, center-y, aspect, height) state space used by SORT / DeepSORT /
ByteTrack.

State vector (8-dim): [cx, cy, a, h, vcx, vcy, va, vh]
Measurement (4-dim):  [cx, cy, a, h]   (a = width / height)

Process and measurement noise are scaled by the object's height, the
standard DeepSORT trick — a person far from the camera (small h) moves
fewer pixels per frame than one close up, so the uncertainty should scale
with apparent size.

Pure numpy, no filterpy dependency — the maths here is small and fixed.
"""

from __future__ import annotations

import numpy as np


class KalmanBoxFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # uncertainty weights (relative to measured height)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(measurement)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)

        kalman_gain = np.linalg.solve(projected_cov, (covariance @ self._update_mat.T).T).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


def bbox_to_measurement(bbox: list[float]) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, aspect, height]"""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=float)


def mean_to_bbox(mean: np.ndarray) -> list[float]:
    """[cx, cy, aspect, height, ...] -> [x1, y1, x2, y2]"""
    cx, cy, a, h = mean[:4]
    w = a * h
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
