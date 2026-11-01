"""
Exposure and contrast normalization for clinical eye images.

Handles varying illumination conditions across different clinical
cameras, lighting setups, and patient skin tones. Ensures the
segmentation model receives consistent input regardless of
acquisition conditions.

Methods:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - White balance correction
    - Brightness normalization to target range
    - Gamma correction

Plan alignment:
    - A6: Used by OptimizedVideoProcessor and SegmentationInference
      for consistent illumination normalisation.
    - Added fast_normalize() for video pipeline (skips heavy ops).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.logger import get_logger


class ImageNormalizer:
    """Normalize clinical eye images for consistent detection.

    Ensures the segmentation model receives input with consistent
    brightness, contrast, and colour balance regardless of the
    clinical camera or lighting setup.

    Usage
    -----
    >>> norm = ImageNormalizer()
    >>> normalized = norm.normalize(image_bgr)

    For video (faster):
    >>> normalized = norm.fast_normalize(image_bgr)
    """

    def __init__(
        self,
        target_brightness: float = 128.0,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        enable_clahe: bool = True,
        enable_brightness: bool = True,
        enable_white_balance: bool = True,
        enable_gamma: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target_brightness : float
            Target mean brightness for normalisation (0-255).
        clahe_clip : float
            CLAHE clip limit. Higher = more contrast enhancement.
        clahe_grid : int
            CLAHE tile grid size. Larger = more global effect.
        enable_clahe : bool
            Enable CLAHE contrast enhancement.
        enable_brightness : bool
            Enable brightness normalisation to target mean.
        enable_white_balance : bool
            Enable gray-world white balance correction.
        enable_gamma : bool
            Enable automatic gamma correction.
        """
        self.target_brightness = target_brightness
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.enable_clahe = enable_clahe
        self.enable_brightness = enable_brightness
        self.enable_white_balance = enable_white_balance
        self.enable_gamma = enable_gamma
        self.logger = get_logger()

        # Pre-create CLAHE object (reusable, thread-safe for reads)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )

        # Faster CLAHE for video mode (smaller grid, lower clip)
        self._clahe_fast = cv2.createCLAHE(
            clipLimit=min(clahe_clip, 1.5),
            tileGridSize=(4, 4),
        )

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply full normalization pipeline.

        Order: white_balance → brightness → CLAHE → gamma

        Parameters
        ----------
        image : np.ndarray  BGR uint8  (H, W, 3) or grayscale (H, W)

        Returns
        -------
        np.ndarray  same format as input, normalized
        """
        if image is None or image.size == 0:
            return image

        # Phase 1: Handle grayscale input
        is_grayscale = image.ndim == 2 or (
            image.ndim == 3 and image.shape[2] == 1
        )

        if is_grayscale:
            # For grayscale: skip white balance, apply CLAHE directly
            gray = image if image.ndim == 2 else image[:, :, 0]
            result = gray.copy()

            if self.enable_brightness:
                current_mean = float(result.mean())
                if current_mean > 1.0:
                    scale = self.target_brightness / current_mean
                    scale = max(0.5, min(2.0, scale))
                    if abs(scale - 1.0) >= 0.05:
                        result = cv2.convertScaleAbs(
                            result, alpha=scale, beta=0
                        )

            if self.enable_clahe:
                result = self._clahe.apply(result)

            return result

        result = image.copy()

        if self.enable_white_balance:
            result = self._white_balance(result)

        if self.enable_brightness:
            result = self._normalize_brightness(result)

        if self.enable_clahe:
            result = self._apply_clahe(result)

        if self.enable_gamma:
            result = self._auto_gamma(result)

        return result

    def fast_normalize(self, image: np.ndarray) -> np.ndarray:
        """Fast normalization for video frames.

        Applies only CLAHE and brightness — skips white balance
        and gamma correction for speed.

        Typical latency: ~0.3ms (vs ~0.8ms for full normalize)

        Parameters
        ----------
        image : np.ndarray  BGR uint8  (H, W, 3) or grayscale (H, W)

        Returns
        -------
        np.ndarray  same format as input, normalized
        """
        if image is None or image.size == 0:
            return image

        # Phase 1: Handle grayscale input
        is_grayscale = image.ndim == 2 or (
            image.ndim == 3 and image.shape[2] == 1
        )

        if is_grayscale:
            gray = image if image.ndim == 2 else image[:, :, 0]
            result = gray

            if self.enable_brightness:
                current_mean = float(result.mean())
                if current_mean > 1.0:
                    scale = self.target_brightness / current_mean
                    scale = max(0.6, min(1.8, scale))
                    if abs(scale - 1.0) > 0.08:
                        result = cv2.convertScaleAbs(
                            result, alpha=scale, beta=0
                        )

            if self.enable_clahe:
                result = self._clahe_fast.apply(result)

            return result

        result = image

        # Fast brightness check and adjust
        if self.enable_brightness:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            current_mean = float(gray.mean())
            if current_mean > 1.0:
                scale = self.target_brightness / current_mean
                scale = max(0.6, min(1.8, scale))
                if abs(scale - 1.0) > 0.08:
                    result = cv2.convertScaleAbs(
                        result, alpha=scale, beta=0
                    )

        # Fast CLAHE (smaller grid, lower clip)
        if self.enable_clahe:
            if len(result.shape) == 3 and result.shape[2] >= 3:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe_fast.apply(lab[:, :, 0])
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif len(result.shape) == 2:
                result = self._clahe_fast.apply(result)

        return result

    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance.

        Assumes the average colour should be gray. Scales each
        channel so that the per-channel means are equal.
        """
        b, g, r = cv2.split(image.astype(np.float32))

        avg_b = max(b.mean(), 1.0)
        avg_g = max(g.mean(), 1.0)
        avg_r = max(r.mean(), 1.0)
        avg_all = (avg_b + avg_g + avg_r) / 3.0

        b = np.clip(b * (avg_all / avg_b), 0, 255)
        g = np.clip(g * (avg_all / avg_g), 0, 255)
        r = np.clip(r * (avg_all / avg_r), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Scale brightness to target mean value."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_mean = float(gray.mean())

        if current_mean < 1.0:
            return image

        scale = self.target_brightness / current_mean
        # Clamp scale to avoid extreme adjustments
        scale = max(0.5, min(2.0, scale))

        if abs(scale - 1.0) < 0.05:
            return image

        result = cv2.convertScaleAbs(image, alpha=scale, beta=0)
        return result

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the luminance channel (LAB color space).

        This enhances local contrast without affecting color balance,
        making iris-sclera and pupil-iris boundaries more visible.
        """
        if len(image.shape) == 3 and image.shape[2] >= 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            l_enhanced = self._clahe.apply(l_channel)

            lab_enhanced = cv2.merge(
                [l_enhanced, a_channel, b_channel]
            )
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            return result
        elif len(image.shape) == 2:
            return self._clahe.apply(image)

        return image

    def _auto_gamma(self, image: np.ndarray) -> np.ndarray:
        """Automatic gamma correction based on image brightness.

        Dark images get gamma < 1 (brighten), bright images get
        gamma > 1 (darken).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())

        if mean_val < 1.0:
            return image

        gamma = np.log(self.target_brightness / 255.0) / np.log(
            mean_val / 255.0
        )
        gamma = max(0.3, min(3.0, gamma))

        if abs(gamma - 1.0) < 0.05:
            return image

        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, lut)

    def get_image_stats(
        self, image: np.ndarray
    ) -> dict:
        """Return diagnostic statistics about image quality.

        Useful for debugging illumination issues and verifying
        that normalisation is working correctly.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)

        return {
            "brightness_mean": float(gray.mean()),
            "brightness_std": float(gray.std()),
            "brightness_min": int(gray.min()),
            "brightness_max": int(gray.max()),
            "channel_means": {
                "blue": float(b.mean()),
                "green": float(g.mean()),
                "red": float(r.mean()),
            },
            "contrast": float(
                gray.std() / max(gray.mean(), 1)
            ),
            "dynamic_range": int(gray.max()) - int(gray.min()),
        }