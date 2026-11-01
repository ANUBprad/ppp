"""
Temporal Smoother — Lightweight Kalman-filter for detection smoothing.

Provides frame-to-frame smoothing of pupil/limbus detection results
using a simple Kalman filter for position and size parameters.

Used in OptimizedVideoProcessor to reduce jitter in real-time processing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Lightweight Kalman filter for smoothing pupil/limbus detection.
    
    Smooths:
    - Pupil center (cx, cy)
    - Pupil semi-axes (sa, sb)
    - Limbus center (limbus_cx, limbus_cy)
    - Limbus semi-axes (limbus_sa, limbus_sb)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 1.0,
    ):
        """
        Initialize temporal smoother.
        
        Args:
            process_noise: Process noise (how much we expect state to change)
            measurement_noise: Measurement noise (detection uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # State variables for each tracked parameter.
        # Keys must match FastInference / OptimizedVideoProcessor dict keys.
        self.pupil_state = {
            'pupil_x': None,
            'pupil_y': None,
            'pupil_radius': None,
            'pupil_major': None,
            'pupil_minor': None,
        }

        self.limbus_state = {
            'limbus_x': None,
            'limbus_y': None,
            'limbus_radius': None,
            'limbus_major': None,
            'limbus_minor': None,
        }
        self.ring_state = {
            'ring_center_x': None,
            'ring_center_y': None,
            'ring_radius': None,
        }
        
        # Covariance for each parameter (uncertainty)
        self.pupil_cov = {k: 100.0 for k in self.pupil_state.keys()}
        self.limbus_cov = {k: 100.0 for k in self.limbus_state.keys()}
        self.ring_cov = {k: 100.0 for k in self.ring_state.keys()}
        
        # Track if we have valid detections
        self.pupil_detected = False
        self.limbus_detected = False
        self.ring_detected = False
        self.ring_detect_streak = 0
        self.ring_miss_streak = 0
        self.ring_dot_count = 0
        self.ring_locked = False
        self.ring_lock_streak = 0
        self.ring_lock_center = None
        self.ring_lock_radius = None
        self.ring_lock_dot_count = 0
        self.ring_lock_confidence = 0.0
        self.ring_lock_boundary_ratio = 1.0
        self.ring_lock_boundary_ratio = 1.0
        self._last_measured_centers = {
            'pupil': None,
            'limbus': None,
        }
        self._last_motion_mode = {
            'pupil': 'unknown',
            'limbus': 'unknown',
        }
        self._pupil_miss_count = 0
        self._limbus_miss_count = 0

    def smooth(self, raw_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Kalman smoothing to detection results.
        
        Args:
            raw_dict: Detection results dictionary containing:
                - pupil_detected (bool)
                - pupil_cx, pupil_cy, pupil_sa, pupil_sb (float)
                - limbus_detected (bool)
                - limbus_cx, limbus_cy, limbus_sa, limbus_sb (float)
                - Other fields passed through unchanged
        
        Returns:
            Smoothed dictionary with same keys as input
        """
        result = dict(raw_dict)  # Copy input
        
        # Check if detections are valid
        pupil_detected = raw_dict.get('pupil_detected', False)
        limbus_detected = raw_dict.get('limbus_detected', False)
        pupil_profile = self._motion_profile('pupil', raw_dict) if pupil_detected else None
        limbus_profile = self._motion_profile('limbus', raw_dict) if limbus_detected else None
        
        # Smooth pupil if detected
        if pupil_detected:
            self._pupil_miss_count = 0
            for key in self.pupil_state.keys():
                if key in raw_dict and raw_dict[key] is not None:
                    measurement = raw_dict[key]
                    smoothed = self._kalman_update(
                        key,
                        measurement,
                        self.pupil_state,
                        self.pupil_cov,
                        process_scale=(pupil_profile or {}).get('process_scale', 1.0),
                        measurement_scale=(pupil_profile or {}).get('measurement_scale', 1.0),
                    )
                    result[key] = smoothed
            # Sync radius alias
            if 'pupil_radius' in result:
                result['pupil_r'] = result['pupil_radius']
            self.pupil_detected = True
            result['pupil_interpolated'] = False
            if pupil_profile is not None:
                result['pupil_motion_mode'] = pupil_profile['mode']
        else:
            self._pupil_miss_count += 1
            if self._pupil_miss_count <= 4:
                # Short blink bridging: carry forward prediction but flag as interpolated
                for key in self.pupil_state.keys():
                    if key in result and self.pupil_state[key] is not None:
                        result[key] = self.pupil_state[key]
                if 'pupil_radius' in result:
                    result['pupil_r'] = result['pupil_radius']
                result['pupil_interpolated'] = True
            else:
                # Expired: clear stale state so stale measurements are not displayed
                for key in self.pupil_state.keys():
                    self.pupil_state[key] = None
                    self.pupil_cov[key] = 100.0
                    if key in result:
                        result[key] = None
                result['pupil_r'] = None
                result['pupil_confidence'] = 0.0
                result['pupil_interpolated'] = False
            self.pupil_detected = False
            result['pupil_motion_mode'] = self._last_motion_mode.get('pupil', 'unknown')

        # Smooth limbus if detected
        if limbus_detected:
            self._limbus_miss_count = 0
            for key in self.limbus_state.keys():
                if key in raw_dict and raw_dict[key] is not None:
                    measurement = raw_dict[key]
                    smoothed = self._kalman_update(
                        key,
                        measurement,
                        self.limbus_state,
                        self.limbus_cov,
                        process_scale=(limbus_profile or {}).get('process_scale', 1.0),
                        measurement_scale=(limbus_profile or {}).get('measurement_scale', 1.0),
                    )
                    result[key] = smoothed
            # Sync radius alias
            if 'limbus_radius' in result:
                result['limbus_r'] = result['limbus_radius']
            self.limbus_detected = True
            result['limbus_interpolated'] = False
            if limbus_profile is not None:
                result['limbus_motion_mode'] = limbus_profile['mode']
        else:
            self._limbus_miss_count += 1
            if self._limbus_miss_count <= 4:
                # Short loss bridging
                for key in self.limbus_state.keys():
                    if key in result and self.limbus_state[key] is not None:
                        result[key] = self.limbus_state[key]
                if 'limbus_radius' in result:
                    result['limbus_r'] = result['limbus_radius']
                result['limbus_interpolated'] = True
            else:
                # Expired: clear stale state
                for key in self.limbus_state.keys():
                    self.limbus_state[key] = None
                    self.limbus_cov[key] = 100.0
                    if key in result:
                        result[key] = None
                result['limbus_r'] = None
                result['limbus_confidence'] = 0.0
                result['limbus_interpolated'] = False
            self.limbus_detected = False
            result['limbus_motion_mode'] = self._last_motion_mode.get('limbus', 'unknown')

        ring_present = (
            raw_dict.get('ring_status') == 'ring_present'
            and raw_dict.get('ring_center_x') is not None
            and raw_dict.get('ring_center_y') is not None
            and raw_dict.get('ring_radius') is not None
        )
        ring_confidence = float(raw_dict.get('ring_confidence', 0.0) or 0.0)
        ring_dot_count = int(raw_dict.get('ring_dot_count', 0) or 0)
        ring_boundary_ratio = float(raw_dict.get('ring_boundary_ratio', 1.0) or 1.0)

        if ring_present:
            measurement_center = (
                float(raw_dict.get('ring_center_x')),
                float(raw_dict.get('ring_center_y')),
            )
            measurement_radius = float(raw_dict.get('ring_radius'))
            if self.ring_locked and self.ring_lock_center is not None and self.ring_lock_radius is not None:
                delta = float(np.hypot(
                    measurement_center[0] - self.ring_lock_center[0],
                    measurement_center[1] - self.ring_lock_center[1],
                ))
                radius_delta = abs(measurement_radius - self.ring_lock_radius)
                if delta <= max(7.0, self.ring_lock_radius * 0.018) and radius_delta <= max(4.0, self.ring_lock_radius * 0.016):
                    self.ring_miss_streak = 0
                    self.ring_dot_count = max(self.ring_lock_dot_count, ring_dot_count)
                    result['ring_status'] = 'ring_present'
                    result['ring_center_x'] = self.ring_lock_center[0]
                    result['ring_center_y'] = self.ring_lock_center[1]
                    result['ring_radius'] = self.ring_lock_radius
                    result['ring_dot_count'] = self.ring_dot_count
                    result['ring_locked'] = True
                    result['ring_lock_confidence'] = max(self.ring_lock_confidence, ring_confidence)
                    result['ring_boundary_ratio'] = self.ring_lock_boundary_ratio
                    return result
                self.ring_miss_streak += 1
                if self.ring_miss_streak < 5:
                    result['ring_status'] = 'ring_present'
                    result['ring_center_x'] = self.ring_lock_center[0]
                    result['ring_center_y'] = self.ring_lock_center[1]
                    result['ring_radius'] = self.ring_lock_radius
                    result['ring_dot_count'] = self.ring_lock_dot_count
                    result['ring_locked'] = True
                    result['ring_lock_confidence'] = self.ring_lock_confidence
                    result['ring_boundary_ratio'] = self.ring_lock_boundary_ratio
                    return result
                self.ring_locked = False
                self.ring_miss_streak = 0
            for key in self.ring_state.keys():
                measurement = raw_dict.get(key)
                if measurement is not None:
                    result[key] = self._kalman_update(
                        key,
                        measurement,
                        self.ring_state,
                        self.ring_cov,
                    )
            self.ring_dot_count = ring_dot_count
            self.ring_detect_streak += 1
            self.ring_miss_streak = 0
            self.ring_detected = self.ring_detected or self.ring_detect_streak >= 2
            smoothed_center = (
                float(self.ring_state['ring_center_x']),
                float(self.ring_state['ring_center_y']),
            )
            smoothed_radius = float(self.ring_state['ring_radius'])
            center_residual = float(np.hypot(
                measurement_center[0] - smoothed_center[0],
                measurement_center[1] - smoothed_center[1],
            ))
            radius_residual = abs(measurement_radius - smoothed_radius)
            stable_measurement = (
                center_residual <= max(6.0, smoothed_radius * 0.02)
                and radius_residual <= max(4.0, smoothed_radius * 0.015)
            )
            if (
                ring_confidence >= 0.90
                and ring_dot_count >= 8
                and ring_boundary_ratio <= 0.0065
                and stable_measurement
            ):
                self.ring_lock_streak += 1
            else:
                self.ring_lock_streak = 0
            if self.ring_lock_streak >= 8:
                self.ring_locked = True
                if self.ring_lock_center is None or self.ring_lock_radius is None:
                    self.ring_lock_center = smoothed_center
                    self.ring_lock_radius = smoothed_radius
                else:
                    self.ring_lock_center = (
                        self.ring_lock_center[0] * 0.7 + smoothed_center[0] * 0.3,
                        self.ring_lock_center[1] * 0.7 + smoothed_center[1] * 0.3,
                    )
                    self.ring_lock_radius = self.ring_lock_radius * 0.7 + smoothed_radius * 0.3
                self.ring_lock_dot_count = self.ring_dot_count
                self.ring_lock_confidence = max(self.ring_lock_confidence, ring_confidence)
                self.ring_lock_boundary_ratio = min(self.ring_lock_boundary_ratio, ring_boundary_ratio)
            if self.ring_detected:
                result['ring_status'] = 'ring_present'
                if self.ring_locked and self.ring_lock_center is not None and self.ring_lock_radius is not None:
                    result['ring_center_x'] = self.ring_lock_center[0]
                    result['ring_center_y'] = self.ring_lock_center[1]
                    result['ring_radius'] = self.ring_lock_radius
                else:
                    result['ring_center_x'] = self.ring_state['ring_center_x']
                    result['ring_center_y'] = self.ring_state['ring_center_y']
                    result['ring_radius'] = self.ring_state['ring_radius']
                result['ring_dot_count'] = self.ring_dot_count
                result['ring_locked'] = self.ring_locked
                result['ring_lock_confidence'] = self.ring_lock_confidence if self.ring_locked else ring_confidence
                result['ring_boundary_ratio'] = self.ring_lock_boundary_ratio if self.ring_locked else ring_boundary_ratio
            else:
                result['ring_status'] = 'ring_absent'
                result['ring_center_x'] = None
                result['ring_center_y'] = None
                result['ring_radius'] = None
                result['ring_dot_count'] = 0
                result['ring_locked'] = False
                result['ring_lock_confidence'] = 0.0
                result['ring_boundary_ratio'] = 1.0
        else:
            self.ring_detect_streak = 0
            self.ring_lock_streak = 0
            if self.ring_locked and self.ring_lock_center is not None and self.ring_lock_radius is not None and self.ring_miss_streak < 6:
                self.ring_miss_streak += 1
                result['ring_status'] = 'ring_present'
                result['ring_center_x'] = self.ring_lock_center[0]
                result['ring_center_y'] = self.ring_lock_center[1]
                result['ring_radius'] = self.ring_lock_radius
                result['ring_dot_count'] = self.ring_lock_dot_count
                result['ring_locked'] = True
                result['ring_lock_confidence'] = self.ring_lock_confidence
                result['ring_boundary_ratio'] = self.ring_lock_boundary_ratio
            elif self.ring_detected and self.ring_state['ring_center_x'] is not None and self.ring_miss_streak < 2:
                self.ring_miss_streak += 1
                result['ring_status'] = 'ring_present'
                result['ring_center_x'] = self.ring_state['ring_center_x']
                result['ring_center_y'] = self.ring_state['ring_center_y']
                result['ring_radius'] = self.ring_state['ring_radius']
                result['ring_dot_count'] = self.ring_dot_count
                result['ring_locked'] = False
                result['ring_lock_confidence'] = 0.0
                result['ring_boundary_ratio'] = 1.0
            else:
                self.ring_detected = False
                self.ring_locked = False
                self.ring_miss_streak = 0
                result['ring_status'] = 'ring_absent'
                result['ring_center_x'] = None
                result['ring_center_y'] = None
                result['ring_radius'] = None
                result['ring_dot_count'] = 0
                result['ring_locked'] = False
                result['ring_lock_confidence'] = 0.0
                result['ring_boundary_ratio'] = 1.0
        
        return result

    @property
    def is_ring_locked(self) -> bool:
        return bool(
            self.ring_locked
            and self.ring_lock_center is not None
            and self.ring_lock_radius is not None
        )

    def get_locked_ring_measurement(self) -> Optional[Dict[str, Any]]:
        if not self.is_ring_locked:
            return None
        return {
            'ring_status': 'ring_present',
            'ring_center_x': float(self.ring_lock_center[0]),
            'ring_center_y': float(self.ring_lock_center[1]),
            'ring_radius': float(self.ring_lock_radius),
            'ring_dot_count': int(self.ring_lock_dot_count),
            'ring_confidence': float(self.ring_lock_confidence),
            'ring_boundary_ratio': float(self.ring_lock_boundary_ratio),
            'ring_locked': True,
        }

    def _kalman_update(
        self,
        param_name: str,
        measurement: float,
        state: Dict[str, Optional[float]],
        covariance: Dict[str, float],
        process_scale: float = 1.0,
        measurement_scale: float = 1.0,
    ) -> float:
        """
        Single-parameter Kalman filter step.
        
        Args:
            param_name: Parameter name (key in state dict)
            measurement: New measurement value
            state: State dictionary for this parameter group
            covariance: Covariance dictionary for uncertainty
        
        Returns:
            Smoothed estimate (weighted between prediction and measurement)
        """
        # If we don't have a prior state, initialize it
        if state[param_name] is None:
            state[param_name] = measurement
            covariance[param_name] = self.measurement_noise * measurement_scale
            return measurement
        
        # Prediction step (state stays same, covariance grows)
        prior_state = state[param_name]
        prior_cov = covariance[param_name] + (self.process_noise * process_scale)
        
        # Update step (Kalman gain & smoothed estimate)
        kalman_gain = prior_cov / (
            prior_cov + max(self.measurement_noise * measurement_scale, 1e-6)
        )
        smoothed_state = prior_state + kalman_gain * (measurement - prior_state)
        posterior_cov = (1.0 - kalman_gain) * prior_cov
        
        # Store updated state & covariance
        state[param_name] = smoothed_state
        covariance[param_name] = max(posterior_cov, 0.01)  # Avoid numerical issues
        
        return smoothed_state

    def _motion_profile(self, prefix: str, raw_dict: Dict[str, Any]) -> Dict[str, float]:
        x_key = f'{prefix}_x'
        y_key = f'{prefix}_y'
        cx = raw_dict.get(x_key)
        cy = raw_dict.get(y_key)
        if cx is None or cy is None:
            return {
                'mode': self._last_motion_mode.get(prefix, 'unknown'),
                'process_scale': 1.0,
                'measurement_scale': 1.0,
            }
        current = (float(cx), float(cy))
        prev = self._last_measured_centers.get(prefix)
        self._last_measured_centers[prefix] = current
        if prev is None:
            self._last_motion_mode[prefix] = 'acquire'
            return {'mode': 'acquire', 'process_scale': 1.2, 'measurement_scale': 1.0}
        speed = float(np.hypot(current[0] - prev[0], current[1] - prev[1]))
        if speed > 50.0:
            raw_dict[x_key] = prev[0]
            raw_dict[y_key] = prev[1]
            self._last_motion_mode[prefix] = 'rejected_jump'
            return {
                'mode': 'rejected_jump',
                'process_scale': 0.4,
                'measurement_scale': 2.4,
            }
        if speed <= 1.5:
            mode = 'fixation'
            profile = {'mode': mode, 'process_scale': 0.30, 'measurement_scale': 1.8}
        elif speed <= 8.0:
            mode = 'smooth_pursuit'
            profile = {'mode': mode, 'process_scale': 1.0, 'measurement_scale': 1.0}
        else:
            mode = 'saccade'
            profile = {'mode': mode, 'process_scale': 3.6, 'measurement_scale': 0.65}
        self._last_motion_mode[prefix] = mode
        return profile

    def reset(self) -> None:
        """Reset all smoothing state."""
        for key in self.pupil_state.keys():
            self.pupil_state[key] = None
            self.pupil_cov[key] = 100.0
        
        for key in self.limbus_state.keys():
            self.limbus_state[key] = None
            self.limbus_cov[key] = 100.0

        for key in self.ring_state.keys():
            self.ring_state[key] = None
            self.ring_cov[key] = 100.0
        
        self.pupil_detected = False
        self.limbus_detected = False
        self.ring_detected = False
        self.ring_detect_streak = 0
        self.ring_miss_streak = 0
        self.ring_dot_count = 0
        self._last_measured_centers = {
            'pupil': None,
            'limbus': None,
        }
        self._last_motion_mode = {
            'pupil': 'unknown',
            'limbus': 'unknown',
        }
        self._pupil_miss_count = 0
        self._limbus_miss_count = 0
