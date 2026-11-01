"""
Kalman filter for temporal smoothing of eye detections.

State vector: [cx, cy, r_major, r_minor, angle, vx, vy]
  - position (center x, y)
  - shape (semi-major, semi-minor, angle)
  - velocity (dx/dt, dy/dt)

The filter provides:
  - Noise reduction between frames
  - Prediction during temporary detection failures (blinks, blur)
  - Velocity estimation for motion compensation
  - Smooth output for surgical overlay
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from pupil_tracking.utils.types import (
    EllipseParams,
    PupilDetection,
    LimbusDetection,
    EyeDetectionResult,
    DetectionMethod,
    assign_quality_grade,
)
from pupil_tracking.utils.config import get_config


class EllipseKalmanFilter:
    """Kalman filter for a single ellipse (pupil OR limbus).

    State: [cx, cy, semi_major, semi_minor, angle_sin, angle_cos, vx, vy]
    Using sin/cos for angle avoids the 0°/180° wraparound discontinuity.
    """

    # state indices
    _CX, _CY = 0, 1
    _SA, _SB = 2, 3
    _ASIN, _ACOS = 4, 5
    _VX, _VY = 6, 7
    _DIM = 8

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        self.dim = self._DIM
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.eye(self.dim, dtype=np.float64) * 100.0
        self._initialised = False

        # transition: constant-velocity model for cx, cy
        self.F = np.eye(self.dim, dtype=np.float64)
        self.F[self._CX, self._VX] = 1.0
        self.F[self._CY, self._VY] = 1.0

        # process noise
        self.Q = np.eye(self.dim, dtype=np.float64) * process_noise
        self.Q[self._VX, self._VX] = process_noise * 2.0
        self.Q[self._VY, self._VY] = process_noise * 2.0

        # measurement: we observe [cx, cy, sa, sb, asin, acos]
        self.H = np.zeros((6, self.dim), dtype=np.float64)
        for i in range(6):
            self.H[i, i] = 1.0

        # measurement noise
        self.R = np.eye(6, dtype=np.float64) * measurement_noise

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # normalise angle sin/cos
        norm = math.sqrt(
            self.x[self._ASIN] ** 2 + self.x[self._ACOS] ** 2
        )
        if norm > 0.01:
            self.x[self._ASIN] /= norm
            self.x[self._ACOS] /= norm

        return self.x.copy()

    def update(
        self,
        ellipse: EllipseParams,
        confidence: float = 1.0,
    ) -> np.ndarray:
        """Update with a measurement."""
        angle_rad = math.radians(ellipse.angle_deg)
        z = np.array([
            ellipse.center_x,
            ellipse.center_y,
            ellipse.semi_major,
            ellipse.semi_minor,
            math.sin(angle_rad),
            math.cos(angle_rad),
        ], dtype=np.float64)

        if not self._initialised:
            self.x[:6] = z
            self.x[6:] = 0.0
            self._initialised = True
            # tighten initial covariance
            self.P = np.eye(self.dim, dtype=np.float64) * 5.0
            return self.x.copy()

        # scale measurement noise by inverse confidence
        # low confidence → high noise → measurement weighted less
        noise_scale = 1.0 / max(confidence, 0.1)
        R_scaled = self.R * noise_scale

        # innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R_scaled
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()

        self.x = self.x + K @ y
        I = np.eye(self.dim)
        self.P = (I - K @ self.H) @ self.P

        # ensure positive semi-axes
        self.x[self._SA] = max(1.0, self.x[self._SA])
        self.x[self._SB] = max(1.0, self.x[self._SB])
        if self.x[self._SA] < self.x[self._SB]:
            self.x[self._SA], self.x[self._SB] = (
                self.x[self._SB],
                self.x[self._SA],
            )

        return self.x.copy()

    def get_ellipse(self) -> EllipseParams:
        """Current state as an EllipseParams."""
        angle_rad = math.atan2(self.x[self._ASIN], self.x[self._ACOS])
        angle_deg = math.degrees(angle_rad) % 180.0

        sa = max(1.0, self.x[self._SA])
        sb = max(1.0, min(self.x[self._SB], sa))

        return EllipseParams(
            center_x=self.x[self._CX],
            center_y=self.x[self._CY],
            semi_major=sa,
            semi_minor=sb,
            angle_deg=angle_deg,
            eccentricity=(
                math.sqrt(max(0.0, 1.0 - (sb / sa) ** 2))
                if sa > 0 else 0.0
            ),
            circularity=sb / sa if sa > 0 else 1.0,
        )

    @property
    def initialised(self) -> bool:
        return self._initialised

    def reset(self) -> None:
        self.x = np.zeros(self.dim, dtype=np.float64)
        self.P = np.eye(self.dim, dtype=np.float64) * 100.0
        self._initialised = False


class EyeKalmanTracker:
    """Dual Kalman filter tracker for pupil + limbus.

    Manages:
    - Separate Kalman filters for pupil and limbus
    - Carry-forward with confidence decay on missed detections
    - Blink detection (both lost → pause tracking)
    """

    def __init__(self, config=None) -> None:
        self.cfg = config or get_config()
        vc = self.cfg.video

        self.pupil_kf = EllipseKalmanFilter(
            process_noise=vc.kalman_process_noise,
            measurement_noise=vc.kalman_measurement_noise,
        )
        self.limbus_kf = EllipseKalmanFilter(
            process_noise=vc.kalman_process_noise * 0.5,  # limbus moves less
            measurement_noise=vc.kalman_measurement_noise,
        )

        self._frames_without_pupil = 0
        self._frames_without_limbus = 0
        self._max_carry = vc.max_carry_forward_frames
        self._decay = vc.carry_forward_decay

    def update(self, result: EyeDetectionResult) -> EyeDetectionResult:
        """Apply Kalman filtering to a detection result.

        Returns a new result with smoothed positions.
        """
        smoothed = EyeDetectionResult()
        smoothed.metadata = result.metadata
        smoothed.calibration = result.calibration
        smoothed.alerts = list(result.alerts)

        # ── pupil ───────────────────────────────────────────────
        self.pupil_kf.predict()

        if result.pupil.detected and result.pupil.ellipse is not None:
            self.pupil_kf.update(
                result.pupil.ellipse, result.pupil.confidence
            )
            self._frames_without_pupil = 0

            smoothed.pupil = PupilDetection(
                detected=True,
                ellipse=self.pupil_kf.get_ellipse(),
                confidence=result.pupil.confidence,
                quality=result.pupil.quality,
                method=DetectionMethod.KALMAN,
            )
        elif (
            self.pupil_kf.initialised
            and self._frames_without_pupil < self._max_carry
        ):
            self._frames_without_pupil += 1
            decay_factor = self._decay ** self._frames_without_pupil
            carry_conf = (
                result.pupil.confidence
                if result.pupil.confidence > 0
                else 0.5
            ) * decay_factor

            smoothed.pupil = PupilDetection(
                detected=True,
                ellipse=self.pupil_kf.get_ellipse(),
                confidence=carry_conf,
                quality=assign_quality_grade(carry_conf),
                method=DetectionMethod.CARRY_FORWARD,
            )
        else:
            self._frames_without_pupil += 1
            smoothed.pupil = PupilDetection()

        # ── limbus ──────────────────────────────────────────────
        self.limbus_kf.predict()

        if result.limbus.detected and result.limbus.ellipse is not None:
            self.limbus_kf.update(
                result.limbus.ellipse, result.limbus.confidence
            )
            self._frames_without_limbus = 0

            smoothed.limbus = LimbusDetection(
                detected=True,
                ellipse=self.limbus_kf.get_ellipse(),
                confidence=result.limbus.confidence,
                quality=result.limbus.quality,
                method=DetectionMethod.KALMAN,
            )
        elif (
            self.limbus_kf.initialised
            and self._frames_without_limbus < self._max_carry
        ):
            self._frames_without_limbus += 1
            decay_factor = self._decay ** self._frames_without_limbus
            carry_conf = (
                result.limbus.confidence
                if result.limbus.confidence > 0
                else 0.5
            ) * decay_factor

            smoothed.limbus = LimbusDetection(
                detected=True,
                ellipse=self.limbus_kf.get_ellipse(),
                confidence=carry_conf,
                quality=assign_quality_grade(carry_conf),
                method=DetectionMethod.CARRY_FORWARD,
            )
        else:
            self._frames_without_limbus += 1
            smoothed.limbus = LimbusDetection()

        # ── blink detection ─────────────────────────────────────
        if (
            self._frames_without_pupil > 3
            and self._frames_without_limbus > 3
        ):
            smoothed.alerts.append("Possible blink detected")

        # ── corneal centre ──────────────────────────────────────
        smoothed.corneal_center = result.corneal_center

        # ── overall ─────────────────────────────────────────────
        confs = []
        if smoothed.pupil.detected:
            confs.append(smoothed.pupil.confidence)
        if smoothed.limbus.detected:
            confs.append(smoothed.limbus.confidence)
        if confs:
            smoothed.overall_confidence = float(np.mean(confs))
        smoothed.overall_quality = assign_quality_grade(
            smoothed.overall_confidence
        )

        return smoothed

    def reset(self) -> None:
        self.pupil_kf.reset()
        self.limbus_kf.reset()
        self._frames_without_pupil = 0
        self._frames_without_limbus = 0