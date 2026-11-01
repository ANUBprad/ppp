"""
Suction ring marker detection and masking.

Surgical eye images from femtosecond laser systems (e.g., Ziemer Z8,
Alcon LenSx) contain a ring of red LED marker dots around the limbus.
These markers:
    - Sit ON the limbus boundary (typically 16-20 dots)
    - Are small, uniformly-sized, red/orange circles
    - Can contaminate iris segmentation if not masked

This module detects and inpaints these markers before segmentation.

Plan-aligned changes:
    - Ring geometry validation: after finding red blobs, fit a circle
      through their centres and verify residual_std / radius < threshold.
      This prevents false positives from random red artifacts.
    - SuctionRingResult dataclass for diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.logger import get_logger


# ────────────────────────────────────────────────────────────────
# Diagnostics dataclass
# ────────────────────────────────────────────────────────────────

@dataclass
class SuctionRingResult:
    """Diagnostic information about suction ring detection."""
    detected: bool = False
    dot_count: int = 0
    ring_centre: Optional[Tuple[float, float]] = None
    ring_radius: Optional[float] = None
    ring_inner_radius: Optional[float] = None
    ring_outer_radius: Optional[float] = None
    ring_residual_std: Optional[float] = None
    dot_radius_median: Optional[float] = None
    ring_outer_boundary_std: Optional[float] = None
    dot_centres: List[Tuple[float, float]] = field(default_factory=list)
    mask: Optional[np.ndarray] = None  # uint8 255 = marker region


# ────────────────────────────────────────────────────────────────
# Main class
# ────────────────────────────────────────────────────────────────

class SuctionRingMasker:
    """Detect and mask red suction ring markers in surgical eye images.

    Algorithm:
        1. Convert to HSV and threshold for red hue (two ranges
           because red wraps around H=0/180).
        2. Find connected blobs, filter by area and circularity.
        3. **Ring validation**: fit a least-squares circle through the
           blob centres.  Accept only if the residual standard
           deviation is small relative to the fitted radius.
        4. Dilate the blob mask and inpaint.

    Usage
    -----
    >>> masker = SuctionRingMasker()
    >>> cleaned, mask = masker.remove(image_bgr)
    """

    def __init__(
        self,
        hue_red_low1: int = 0,
        hue_red_high1: int = 12,
        hue_red_low2: int = 165,
        hue_red_high2: int = 180,
        saturation_min: int = 60,
        value_min: int = 60,
        min_marker_area: int = 30,
        max_marker_area: int = 800,
        min_circularity: float = 0.45,
        min_markers_for_ring: int = 5,
        ring_residual_ratio: float = 0.18,
        dilation_size: int = 5,
        inpaint_radius: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        hue_red_low1, hue_red_high1 : int
            First red hue range (0-12 in OpenCV HSV).
        hue_red_low2, hue_red_high2 : int
            Second red hue range (165-180 in OpenCV HSV).
        saturation_min : int
            Minimum saturation for red markers.
        value_min : int
            Minimum value (brightness) for red markers.
        min_marker_area, max_marker_area : int
            Size range for individual marker blobs in pixels.
            Scaled proportionally to image size at runtime.
        min_circularity : float
            Minimum circularity for a blob to be considered a marker.
        min_markers_for_ring : int
            Minimum number of detected markers to confirm a ring.
        ring_residual_ratio : float
            Maximum ratio of residual_std / ring_radius for the
            dot pattern to be accepted as a valid ring.  Higher
            values are more permissive.
        dilation_size : int
            Dilate detected markers to cover their edges.
        inpaint_radius : int
            Radius for cv2.inpaint().
        """
        self.hue_red_low1 = hue_red_low1
        self.hue_red_high1 = hue_red_high1
        self.hue_red_low2 = hue_red_low2
        self.hue_red_high2 = hue_red_high2
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.min_marker_area = min_marker_area
        self.max_marker_area = max_marker_area
        self.min_circularity = min_circularity
        self.min_markers_for_ring = min_markers_for_ring
        self.ring_residual_ratio = ring_residual_ratio
        self.dilation_size = dilation_size
        self.inpaint_radius = inpaint_radius
        self.logger = get_logger()

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def remove(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect and inpaint suction ring markers.

        Parameters
        ----------
        image : np.ndarray  BGR uint8  (H, W, 3) or grayscale (H, W)

        Returns
        -------
        (cleaned_image, marker_mask)
            cleaned_image : same format as input, with markers inpainted
            marker_mask : binary uint8 (0/255) showing marker locations
        """
        if image is None or image.size == 0:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty

        # Phase 1: Handle grayscale input — red markers are invisible
        # in grayscale, so skip detection entirely (no false positives)
        is_grayscale = image.ndim == 2 or (
            image.ndim == 3 and image.shape[2] == 1
        )
        if is_grayscale:
            h = image.shape[0]
            w = image.shape[1] if image.ndim == 2 else image.shape[1]
            empty = np.zeros((h, w), dtype=np.uint8)
            return image, empty

        mask, ring_result = self._detect_markers(image)

        if not ring_result.detected:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty

        self.logger.debug(
            "Suction ring detected: %d markers, centre=(%.0f,%.0f), "
            "r=%.0f, residual_ratio=%.3f",
            ring_result.dot_count,
            ring_result.ring_centre[0] if ring_result.ring_centre else 0,
            ring_result.ring_centre[1] if ring_result.ring_centre else 0,
            ring_result.ring_radius or 0,
            (ring_result.ring_residual_std / ring_result.ring_radius)
            if ring_result.ring_radius and ring_result.ring_radius > 0
            else 0,
        )

        cleaned = cv2.inpaint(
            image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )
        return cleaned, mask

    def remove_with_diagnostics(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, SuctionRingResult]:
        """Like remove() but also returns the full diagnostics object.

        Returns
        -------
        (cleaned_image, marker_mask, ring_result)
        """
        if image is None or image.size == 0:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty, SuctionRingResult()

        mask, ring_result = self._detect_markers(image)

        if not ring_result.detected:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return image, empty, ring_result

        cleaned = cv2.inpaint(
            image, mask, self.inpaint_radius, cv2.INPAINT_TELEA
        )
        ring_result.mask = mask
        return cleaned, mask, ring_result

    def detect_only(self, image: np.ndarray) -> np.ndarray:
        """Return the marker mask without inpainting."""
        mask, _ = self._detect_markers(image)
        return mask

    # ────────────────────────────────────────────────────────────
    # Internal detection
    # ────────────────────────────────────────────────────────────

    def _detect_markers(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, SuctionRingResult]:
        """Detect red marker blobs and validate ring geometry.

        Returns (marker_mask, SuctionRingResult).
        """
        h, w = image.shape[:2]
        result = SuctionRingResult()

        # Grayscale safety: red markers are invisible in grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return np.zeros((h, w), dtype=np.uint8), result

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # Red wraps around 0 in HSV, so we need two ranges
        red_mask1 = (
            (h_chan >= self.hue_red_low1)
            & (h_chan <= self.hue_red_high1)
            & (s_chan >= self.saturation_min)
            & (v_chan >= self.value_min)
        ).astype(np.uint8) * 255

        red_mask2 = (
            (h_chan >= self.hue_red_low2)
            & (h_chan <= self.hue_red_high2)
            & (s_chan >= self.saturation_min)
            & (v_chan >= self.value_min)
        ).astype(np.uint8) * 255

        red_combined = cv2.bitwise_or(red_mask1, red_mask2)

        # Clean up noise
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        red_combined = cv2.morphologyEx(
            red_combined, cv2.MORPH_CLOSE, close_kernel
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        red_combined = cv2.morphologyEx(
            red_combined, cv2.MORPH_OPEN, open_kernel
        )

        # Find connected components and filter by size + circularity
        contours, _ = cv2.findContours(
            red_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Scale area limits proportionally to image size (ref: 640×480)
        scale = (h * w) / (640.0 * 480.0)
        min_area = max(5, int(self.min_marker_area * scale))
        max_area = max(50, int(self.max_marker_area * scale))

        centres: List[Tuple[float, float]] = []
        valid_contours: List[np.ndarray] = []
        dot_radii: List[float] = []
        image_cx = w / 2.0
        image_cy = h / 2.0
        min_dim = float(min(h, w))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue

            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
            if aspect_ratio > 2.4:
                continue

            (_, _), encl_r = cv2.minEnclosingCircle(cnt)
            if encl_r <= 1.0:
                continue
            fill_ratio = float(area / max(np.pi * encl_r * encl_r, 1.0))
            if fill_ratio < 0.22:
                continue

            M = cv2.moments(cnt)
            if M["m00"] < 1e-6:
                continue

            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            radial_dist = math.hypot(cx - image_cx, cy - image_cy)
            if radial_dist < min_dim * 0.28 or radial_dist > min_dim * 0.60:
                continue
            centres.append((cx, cy))
            valid_contours.append(cnt)
            dot_radii.append(float(encl_r))

        empty_mask = np.zeros((h, w), dtype=np.uint8)

        # Need minimum number of blobs
        if len(centres) < self.min_markers_for_ring:
            return empty_mask, result

        # ── Ring geometry validation ────────────────────────────
        # Fit a least-squares circle through the blob centres.
        # A genuine suction ring will have all dots at approximately
        # the same radius from a common centre.
        pts = np.array(centres, dtype=np.float64)
        ring_fit = self._fit_circle_robust(pts)

        if ring_fit is None:
            return empty_mask, result

        fit_cx, fit_cy, fit_r = ring_fit

        if fit_r < 1.0:
            return empty_mask, result

        # Check how well the dots lie on the fitted circle
        dists = np.sqrt(
            (pts[:, 0] - fit_cx) ** 2 + (pts[:, 1] - fit_cy) ** 2
        )
        residual_std = float(np.std(dists))
        residual_ratio = residual_std / fit_r

        angles = (
            np.degrees(np.arctan2(pts[:, 1] - fit_cy, pts[:, 0] - fit_cx)) + 360.0
        ) % 360.0
        hist, _ = np.histogram(angles, bins=24, range=(0.0, 360.0))
        angular_coverage = float(np.count_nonzero(hist > 0) / 24.0)

        if residual_ratio > self.ring_residual_ratio or angular_coverage < 0.16:
            # Dots found but they don't form a nice ring
            self.logger.debug(
                "Red blobs found (%d) but ring validation failed: "
                "residual_ratio=%.3f, angular_coverage=%.3f",
                len(centres), residual_ratio, angular_coverage,
            )
            return empty_mask, result

        # ── Build output mask ───────────────────────────────────
        marker_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(marker_mask, valid_contours, -1, 255, cv2.FILLED)

        contour_pts = np.vstack(valid_contours).reshape(-1, 2).astype(np.float64)
        contour_dists = np.sqrt(
            (contour_pts[:, 0] - fit_cx) ** 2 + (contour_pts[:, 1] - fit_cy) ** 2
        )
        dot_radius_median = float(np.median(dot_radii)) if dot_radii else 0.0
        dot_radii_arr = np.asarray(dot_radii, dtype=np.float64) if dot_radii else np.zeros((0,), dtype=np.float64)
        if dot_radii_arr.size:
            dot_outer_samples = dists + dot_radii_arr * 0.90
            dot_inner_samples = dists - dot_radii_arr * 0.10
            outer_radius = float(np.median(dot_outer_samples))
            inner_radius = float(np.median(dot_inner_samples))
            outer_radius = min(outer_radius, float(np.percentile(contour_dists, 88)))
            inner_radius = max(inner_radius, float(np.percentile(contour_dists, 12)))
            outer_boundary_std = float(np.std(dot_outer_samples - outer_radius))
        else:
            inner_radius = float(np.percentile(contour_dists, 18))
            outer_radius = float(np.percentile(contour_dists, 82))
            outer_boundary_std = float(np.std(contour_dists - outer_radius))

        outer_radius = max(outer_radius, fit_r + dot_radius_median * 0.20)
        if dot_radius_median > 0.0:
            inner_radius = min(inner_radius, fit_r - dot_radius_median * 0.10)

        # Dilate to cover edges around markers
        if self.dilation_size > 0 and np.any(marker_mask):
            dil_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilation_size, self.dilation_size),
            )
            marker_mask = cv2.dilate(
                marker_mask, dil_kernel, iterations=1
            )

        # ── Populate result ─────────────────────────────────────
        result.detected = True
        result.dot_count = len(centres)
        result.ring_centre = (float(fit_cx), float(fit_cy))
        result.ring_radius = float(fit_r)
        result.ring_inner_radius = max(1.0, inner_radius)
        result.ring_outer_radius = max(float(fit_r), outer_radius)
        result.ring_residual_std = residual_std
        result.dot_radius_median = dot_radius_median
        result.ring_outer_boundary_std = outer_boundary_std
        result.dot_centres = [
            (float(x), float(y)) for x, y in centres
        ]
        result.mask = marker_mask

        return marker_mask, result

    # ────────────────────────────────────────────────────────────
    # Circle fit for ring validation
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _fit_circle_lsq(
        pts: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """Algebraic least-squares circle fit (Kåsa method).

        Parameters
        ----------
        pts : np.ndarray  (N, 2)

        Returns
        -------
        (cx, cy, radius) or None
        """
        if len(pts) < 3:
            return None

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)

        A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
        b = x ** 2 + y ** 2

        try:
            res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        cx, cy, c = res
        r_sq = c + cx ** 2 + cy ** 2

        if r_sq <= 0:
            return None

        return (float(cx), float(cy), float(math.sqrt(r_sq)))

    @staticmethod
    def _fit_circle_robust(
        pts: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        if len(pts) < 3:
            return None

        initial = SuctionRingMasker._fit_circle_lsq(pts)
        if initial is None:
            return None

        cx, cy, r = initial
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        resid = np.abs(dists - r)
        median_resid = float(np.median(resid))
        tol = max(4.0, median_resid * 2.5, r * 0.02)
        inliers = resid <= tol
        if np.count_nonzero(inliers) >= 4 and np.count_nonzero(inliers) < len(pts):
            refined = SuctionRingMasker._fit_circle_lsq(pts[inliers])
            if refined is not None:
                return refined

        return initial

    # ────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────

    def count_markers(self, image: np.ndarray) -> int:
        """Count the number of detected ring markers."""
        _, result = self._detect_markers(image)
        return result.dot_count
