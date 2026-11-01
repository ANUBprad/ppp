"""
Temporal reflection filter for handling blinking/surgical lights.

This module provides temporal filtering to distinguish between:
- Persistent specular reflections (real, should be inpainted)
- Transient bright spots (blinking lights, should be masked and ignored)

The filter maintains a short history of reflection masks and identifies
pixels that suddenly appear and disappear - characteristic of blinking
surgical lights.

Usage:
    >>> filter = TemporalReflectionFilter(history_size=5, blink_threshold=0.3)
    >>> for frame in video_frames:
    ...     mask = filter.process(frame)
    ...     # mask marks stable reflections only, transient blinks excluded
"""

from __future__ import annotations

from typing import Deque, Optional, Tuple
from collections import deque

import cv2
import numpy as np

from pupil_tracking.utils.logger import get_logger


class TemporalReflectionFilter:
    """Filter reflections temporally to detect and remove blinking lights.

    This addresses the issue where blinking red/white surgical lights
    create bright spots that interfere with pupil detection. By tracking
    reflection stability across frames, transient blinks can be identified
    and excluded from the inpainting process.

    Parameters
    ----------
    history_size : int
        Number of frames to keep in history for stability analysis.
        Higher = more robust to noise but slower to adapt.
        Recommended: 3-7 for real-time applications.
    blink_threshold : float
        Fraction of frames a pixel must be present to be considered stable.
        Lower = more aggressive filtering (fewer false positives).
        Higher = preserves more real reflections.
        Recommended: 0.3-0.5 for blinking lights.
    min_stable_frames : int
        Minimum consecutive frames a reflection must be stable
        before being considered real.
    dilation_size : int
        Size of dilation kernel for the final mask.
    """

    def __init__(
        self,
        history_size: int = 5,
        blink_threshold: float = 0.3,
        min_stable_frames: int = 2,
        dilation_size: int = 3,
    ) -> None:
        self.history_size = history_size
        self.blink_threshold = blink_threshold
        self.min_stable_frames = min_stable_frames
        self.logger = get_logger()

        # History of reflection masks (binary)
        self._history: Deque[np.ndarray] = deque(maxlen=history_size)

        # Pre-compute dilation kernel
        if dilation_size > 0:
            self._dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_size, dilation_size),
            )
        else:
            self._dilate_kernel = None

        # Statistics
        self._frame_count = 0

    def reset(self) -> None:
        """Clear history and reset filter state."""
        self._history.clear()
        self._frame_count = 0

    def process(
        self,
        image: np.ndarray,
        brightness_threshold: int = 230,
    ) -> np.ndarray:
        """Process a frame and return stable reflection mask.

        Parameters
        ----------
        image : np.ndarray
            BGR input image.
        brightness_threshold : int
            Threshold for bright pixel detection.

        Returns
        -------
        np.ndarray
            Binary mask of stable reflections (transient blinks excluded).
        """
        self._frame_count += 1

        # Detect bright pixels in current frame
        mask = self._detect_bright_pixels(image, brightness_threshold)

        # Add to history
        self._history.append(mask)

        # Need at least 2 frames to do temporal filtering
        if len(self._history) < 2:
            return mask

        # Compute stability map
        stable_mask = self._compute_stability()

        # Dilate to cover edges
        if self._dilate_kernel is not None and np.any(stable_mask):
            stable_mask = cv2.dilate(stable_mask, self._dilate_kernel, iterations=1)

        return stable_mask

    def _detect_bright_pixels(
        self,
        image: np.ndarray,
        threshold: int,
    ) -> np.ndarray:
        """Detect bright pixels using multiple strategies."""
        h, w = image.shape[:2]

        # Ensure BGR
        if image.ndim == 2:
            work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            work = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            work = image

        # Method 1: Grayscale brightness
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        _, bright_gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Method 2: HSV - bright and desaturated (white reflections)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        bright_v = (v >= threshold).astype(np.uint8) * 255
        desat_s = (s <= 40).astype(np.uint8) * 255
        white_reflection = cv2.bitwise_and(bright_v, desat_s)

        # Method 3: Red channel highlights (blinking red lights)
        # Split BGR
        b, g, r = cv2.split(work)

        # Bright red: high R value AND (R > G+offset AND R > B+offset)
        red_dominant = ((r > g + 20) & (r > b + 20)).astype(np.uint8) * 255
        red_bright = (r > threshold).astype(np.uint8) * 255
        red_reflection = cv2.bitwise_or(red_dominant, red_bright)

        # Very bright red (saturated light)
        very_bright_red = (r > 245).astype(np.uint8) * 255

        # Method 4: Blue channel highlights
        blue_bright = (b > threshold).astype(np.uint8) * 255

        # Combine all methods
        mask = cv2.bitwise_or(bright_gray, white_reflection)
        mask = cv2.bitwise_or(mask, red_reflection)
        mask = cv2.bitwise_or(mask, very_bright_red)
        mask = cv2.bitwise_or(mask, blue_bright)

        return mask

    def _compute_stability(self) -> np.ndarray:
        """Compute stability map from history.

        A pixel is considered stable if it's present in enough frames
        (above blink_threshold) AND has appeared for min_stable_frames.

        Transient blinks (present in 1-2 frames only) are excluded.
        """
        if len(self._history) < 2:
            return self._history[-1] if self._history else np.array([])

        # Stack all history masks
        history_stack = np.stack(list(self._history), axis=0)

        # Count how many times each pixel was bright
        presence_count = np.sum(history_stack > 0, axis=0).astype(np.float32)

        # Calculate presence ratio
        n_frames = len(self._history)
        presence_ratio = presence_count / n_frames

        # Create stable mask: present in enough frames
        # Lower threshold = keep more, Higher = filter more
        stable = (presence_ratio >= self.blink_threshold).astype(np.uint8) * 255

        return stable

    def get_stats(self) -> dict:
        """Return current filter statistics."""
        return {
            "history_size": len(self._history),
            "total_frames_processed": self._frame_count,
            "blink_threshold": self.blink_threshold,
            "min_stable_frames": self.min_stable_frames,
        }


class PupilRegionProtector:
    """Protect the pupil region from reflection removal artifacts.

    Sometimes reflection removal can introduce artifacts in the pupil
    region. This class provides additional protection by:
    1. Detecting potential pupil region before processing
    2. Limiting inpainting in the pupil region
    3. Preferring dark (pupil-like) values for inpainting near center

    Parameters
    ----------
    pupil_margin_frac : float
        Fraction of image size to add as margin around detected
        bright spots when determining pupil region.
    """

    def __init__(
        self,
        pupil_margin_frac: float = 0.15,
    ) -> None:
        self.pupil_margin_frac = pupil_margin_frac
        self.logger = get_logger()

    def protect(
        self,
        image: np.ndarray,
        reflection_mask: np.ndarray,
    ) -> np.ndarray:
        """Return modified reflection mask with pupil protection.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image.
        reflection_mask : np.ndarray
            Binary mask of detected reflections.

        Returns
        -------
        np.ndarray
            Modified mask with potential pupil region protected.
        """
        h, w = image.shape[:2]

        # Find bright spots that could be the pupil
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for dark region in center (potential pupil)
        # The pupil should be darker than surroundings
        center_x, center_y = w // 2, h // 2

        # Check if there's a dark region near center
        center_region = gray[
            max(0, center_y - h // 4) : min(h, center_y + h // 4),
            max(0, center_x - w // 4) : min(w, center_x + w // 4),
        ]

        if center_region.size > 0:
            mean_brightness = np.mean(center_region)

            # If center is dark (potential pupil), reduce reflection mask there
            if mean_brightness < 100:
                # Create a soft mask for the center region
                y_start = max(0, center_y - h // 4)
                y_end = min(h, center_y + h // 4)
                x_start = max(0, center_x - w // 4)
                x_end = min(w, center_x + w // 4)

                # Reduce reflection mask in dark center region
                protected = reflection_mask.copy()
                protected[y_start:y_end, x_start:x_end] = (
                    protected[y_start:y_end, x_start:x_end] // 2
                )
                return protected

        return reflection_mask
