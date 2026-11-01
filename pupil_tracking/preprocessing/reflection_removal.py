"""
Specular reflection removal for clinical eye images.

Clinical eye images frequently contain bright specular reflections
from the microscope illumination. These reflections can:
    - Fragment the pupil segmentation mask
    - Create false edges in the iris region
    - Distort ellipse fitting

This module detects and inpaints specular reflections while
preserving the underlying anatomy.

Two speed tiers:
    * fast  — simple threshold + small Telea inpaint (~0.3-0.8 ms
              on 320×320).  Good for per-frame video use.
    * full  — bilateral pre-smooth, blue-channel detection,
               larger inpaint radius (~2-4 ms on 640×480).

Plan-aligned changes:
    - Added use_bilateral_pre for full-quality path (A3)
    - Added blue-channel highlight detection for surgical LEDs
    - Added roi_mask parameter to restrict detection area
    - Preserved existing API for backward compatibility
    - Added red-light reflection detection for surgical illumination
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.logger import get_logger


class ReflectionRemover:
    """Detect and remove specular reflections from eye images.

    Usage
    -----
    >>> remover = ReflectionRemover()
    >>> clean, mask = remover.remove(image_bgr)

    # Fast mode for video (smaller params):
    >>> remover = ReflectionRemover(
    ...     brightness_threshold=225,
    ...     min_reflection_area=10,
    ...     inpaint_radius=3,
    ... )

    # Full quality for single images:
    >>> remover = ReflectionRemover(
    ...     brightness_threshold=220,
    ...     min_reflection_area=15,
    ...     inpaint_radius=5,
    ...     use_bilateral_pre=True,
    ... )
    """

    def __init__(
        self,
        brightness_threshold: int = 230,
        saturation_threshold: int = 40,
        min_reflection_area: int = 20,
        max_reflection_area_frac: float = 0.05,
        dilation_size: int = 5,
        inpaint_radius: int = 7,
        use_bilateral_pre: bool = False,
        detect_blue_highlights: bool = True,
        blue_threshold_offset: int = 10,
        detect_red_highlights: bool = True,
        red_threshold_offset: int = 15,
    ) -> None:
        """
        Parameters
        ----------
        brightness_threshold : int
            Pixels brighter than this in the V channel are candidates.
        saturation_threshold : int
            Reflections have low saturation (near white).
        min_reflection_area : int
            Minimum area in pixels for a reflection blob.
        max_reflection_area_frac : float
            Maximum fraction of image area for a single reflection.
        dilation_size : int
            Dilate the reflection mask to cover edges.
        inpaint_radius : int
            Radius for cv2.inpaint().
        use_bilateral_pre : bool
            Apply bilateral filter before detection for full-quality
            mode.  Reduces noise-induced false positives but costs
            ~1-2ms extra.
        detect_blue_highlights : bool
            Also detect blue-channel highlights (common under surgical
            LED illumination).
        blue_threshold_offset : int
            Blue channel threshold = brightness_threshold + this offset.
        detect_red_highlights : bool
            Detect red-channel highlights (common from surgical red
            illumination lights that can cause false pupil detection).
        red_threshold_offset : int
            Red channel threshold = brightness_threshold + this offset.
        """
        self.brightness_threshold = brightness_threshold
        self.saturation_threshold = saturation_threshold
        self.min_reflection_area = min_reflection_area
        self.max_reflection_area_frac = max_reflection_area_frac
        self.dilation_size = dilation_size
        self.inpaint_radius = inpaint_radius
        self.use_bilateral_pre = use_bilateral_pre
        self.detect_blue_highlights = detect_blue_highlights
        self.blue_threshold_offset = blue_threshold_offset
        self.detect_red_highlights = detect_red_highlights
        self.red_threshold_offset = red_threshold_offset
        self.logger = get_logger()

        # Pre-allocate dilation kernel
        if dilation_size > 0:
            self._dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_size, dilation_size),
            )
        else:
            self._dilate_kernel = None

    def remove(
        self,
        image: np.ndarray,
        *,
        roi_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and inpaint specular reflections.

        Parameters
        ----------
        image : np.ndarray  BGR uint8  (H, W, 3) or grayscale (H, W)
        roi_mask : np.ndarray | None
            Optional mask (255 = region of interest).  Reflections
            outside the ROI are ignored.  Useful to restrict detection
            to the cornea area.

        Returns
        -------
        (cleaned_image, reflection_mask)
            cleaned_image : same format as input, with reflections inpainted
            reflection_mask : binary uint8 (0/255) showing where
                reflections were found
        """
        if image is None or image.size == 0:
            h, w = image.shape[:2] if image is not None else (0, 0)
            return image, np.zeros((h, w), dtype=np.uint8)

        # Phase 1: Handle grayscale input
        is_grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
        if is_grayscale:
            if image.ndim == 2:
                work_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                work_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            work_image = image

        # Detect reflections (always on BGR)
        mask = self._detect_reflections(work_image, roi_mask=roi_mask)

        if not np.any(mask):
            return image, mask

        # Count reflection pixels for logging
        n_reflection = int(np.count_nonzero(mask))
        total = work_image.shape[0] * work_image.shape[1]
        pct = 100.0 * n_reflection / total

        self.logger.debug(
            "Reflections: %d pixels (%.1f%% of image)",
            n_reflection,
            pct,
        )

        # Inpaint on BGR image
        cleaned_bgr = cv2.inpaint(
            work_image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )

        # Phase 1: convert back to grayscale if input was grayscale
        if is_grayscale:
            cleaned = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        else:
            cleaned = cleaned_bgr

        return cleaned, mask

    def _detect_reflections(
        self,
        image: np.ndarray,
        *,
        roi_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect specular reflection regions.

        Strategy:
        1. (Optional) bilateral pre-filter for noise reduction
        2. Convert to HSV
        3. Find pixels that are very bright (high V) AND
           low saturation (near white)
        4. (Optional) Also detect blue-channel highlights from
            surgical LED illumination
        5. (Optional) Detect red-channel highlights from surgical
            red illumination lights that cause false pupil detection
        6. Combine candidates
        7. Restrict to ROI if provided
        8. Filter by connected component size
        9. Dilate to cover reflection edges
        """
        h, w = image.shape[:2]
        total_area = h * w
        max_blob_area = int(total_area * self.max_reflection_area_frac)

        # Optional bilateral pre-filter (full-quality mode)
        if self.use_bilateral_pre:
            work = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)
        else:
            work = image

        # --- HSV-based detection (bright + desaturated) ---
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        _, s_channel, v_channel = cv2.split(hsv)

        bright = v_channel >= self.brightness_threshold
        desat = s_channel <= self.saturation_threshold
        candidates = (bright & desat).astype(np.uint8) * 255

        # --- Blue-channel highlights (surgical LED) ---
        if self.detect_blue_highlights:
            blue = work[:, 0] if work.ndim == 2 else work[:, :, 0]
            blue_thresh = min(
                self.brightness_threshold + self.blue_threshold_offset,
                254,
            )
            _, blue_bright = cv2.threshold(blue, blue_thresh, 255, cv2.THRESH_BINARY)
            candidates = cv2.bitwise_or(candidates, blue_bright)

        # --- Red-channel highlights (surgical red illumination) ---
        # Red lights appear as bright red spots - detect them by:
        # 1. High red channel value
        # 2. Red is dominant (R > G and R > B) OR just very bright red
        # This prevents detecting normal reddish iris as reflection
        if self.detect_red_highlights and work.ndim == 3 and work.shape[2] == 3:
            b, g, r = cv2.split(work)

            # Method 1: Bright red channel with saturation check
            red_thresh = min(
                self.brightness_threshold + self.red_threshold_offset,
                254,
            )
            _, red_bright = cv2.threshold(r, red_thresh, 255, cv2.THRESH_BINARY)

            # Method 2: Red-dominant pixels that are also bright
            # R > G + offset AND R > B + offset (to avoid iris coloration)
            red_dominant = ((r > g + 15) & (r > b + 15)).astype(np.uint8) * 255

            # Combine: bright AND (very bright red OR red-dominant)
            hsv_work = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            _, sat, _ = cv2.split(hsv_work)

            # Low saturation red is likely a light, not iris
            red_saturated = (sat > 30).astype(np.uint8) * 255
            red_dominant_filtered = cv2.bitwise_and(
                red_dominant, cv2.bitwise_not(red_saturated)
            )

            # Combine red detection methods
            red_candidates = cv2.bitwise_or(red_bright, red_dominant_filtered)

            # Also detect red that is very bright regardless of dominance
            # (handles case where red light is extremely bright)
            very_bright_red = (r > 240).astype(np.uint8) * 255
            red_candidates = cv2.bitwise_or(red_candidates, very_bright_red)

            candidates = cv2.bitwise_or(candidates, red_candidates)

        # --- Also catch pure-white via grayscale ---
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        _, gray_bright = cv2.threshold(
            gray, self.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        candidates = cv2.bitwise_or(candidates, gray_bright)

        # --- ROI restriction ---
        if roi_mask is not None:
            candidates = cv2.bitwise_and(candidates, roi_mask)

        # --- Connected components: filter by size ---
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_reflection_area:
                continue
            if area > max_blob_area:
                continue
            mask[labels == i] = 255

        # --- Dilate to cover the halo around reflections ---
        if self._dilate_kernel is not None and np.any(mask):
            mask = cv2.dilate(mask, self._dilate_kernel, iterations=1)

        return mask

    def detect_only(self, image: np.ndarray) -> np.ndarray:
        """Return the reflection mask without inpainting."""
        return self._detect_reflections(image)

    def get_reflection_stats(self, image: np.ndarray) -> dict:
        """Return statistics about reflections in the image."""
        mask = self._detect_reflections(image)
        total = image.shape[0] * image.shape[1]
        n_reflection = int(np.count_nonzero(mask))

        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        blobs = []
        for i in range(1, n_labels):
            blobs.append(
                {
                    "area": int(stats[i, cv2.CC_STAT_AREA]),
                    "center": (
                        float(centroids[i, 0]),
                        float(centroids[i, 1]),
                    ),
                }
            )

        return {
            "total_reflection_pixels": n_reflection,
            "reflection_fraction": n_reflection / max(total, 1),
            "num_blobs": len(blobs),
            "blobs": blobs,
        }
