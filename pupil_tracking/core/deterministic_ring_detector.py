"""
Deterministic suction-ring detection.

This module classifies an eye image as pre-docked vs docked and, when
possible, returns ring geometry for overlays, calibration, and offset
calculations without requiring model training.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.preprocessing.suction_ring_masker import SuctionRingMasker

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


class RingStatus(Enum):
    PRESENT = "ring_present"
    ABSENT = "ring_absent"
    PARTIAL = "ring_partial"
    UNCERTAIN = "ring_uncertain"


@dataclass
class RingDetectionResult:
    status: RingStatus
    confidence: float
    ring_contour: Optional[np.ndarray] = None
    ring_center: Optional[Tuple[float, float]] = None
    ring_radius: Optional[float] = None
    ring_inner_radius: Optional[float] = None
    ring_mask: Optional[np.ndarray] = None
    method: str = "unknown"
    dot_centers: List[Tuple[float, float]] = field(default_factory=list)
    dot_count: int = 0
    corneal_reference_source: str = "limbus"
    details: dict = field(default_factory=dict)


@dataclass
class _CircleCandidate:
    center: Tuple[float, float]
    radius: float
    score: float
    mask: Optional[np.ndarray] = None
    contour: Optional[np.ndarray] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _MarkerCandidate:
    center: Tuple[float, float]
    radius: float
    score: float
    mask: np.ndarray
    contour: Optional[np.ndarray]
    dot_centers: List[Tuple[float, float]]
    details: Dict[str, Any] = field(default_factory=dict)


class HeuristicRingDetector:
    """
    Deterministic detector for docked/post-op imagery.

    The detector intentionally avoids treating every large circular edge
    as a suction ring. It needs red annular evidence or a validated
    marker-dot ring before reporting a confident docked image.
    """

    def __init__(
        self,
        ring_radius_range: Tuple[float, float] = (0.26, 0.49),
        hough_dp: float = 1.2,
        hough_param1: int = 90,
        hough_param2: int = 42,
        canny_low: int = 35,
        canny_high: int = 110,
    ) -> None:
        self.ring_radius_range = ring_radius_range
        self.hough_dp = hough_dp
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.marker_detector = SuctionRingMasker(
            saturation_min=45,
            value_min=45,
            min_marker_area=16,
            max_marker_area=1400,
            min_circularity=0.18,
            min_markers_for_ring=5,
            ring_residual_ratio=0.26,
        )

    def detect(self, image: np.ndarray) -> RingDetectionResult:
        quick_gate = self._quick_red_gate(image)
        if quick_gate is None:
            return RingDetectionResult(
                status=RingStatus.ABSENT,
                confidence=0.995,
                method="quick_red_gate",
                details={"reason": "quick_red_gate_failed"},
            )

        markers = self._detect_marker_ring(image)
        if markers is not None and markers.score >= 0.48:
            return self._candidate_to_result(
                markers,
                status=RingStatus.PRESENT,
                method="marker_geometry",
                corneal_reference_source="suction_ring",
            )

        structural = self._detect_structural_ring(image)
        if (
            structural is not None
            and structural.score >= 0.84
            and float(quick_gate.get("coverage", 0.0)) >= 0.55
            and float(quick_gate.get("radial_pixels", 0.0)) >= 120.0
        ):
            return self._candidate_to_result(
                structural,
                status=RingStatus.PRESENT,
                method="structural_red_fallback",
                corneal_reference_source="suction_ring",
            )

        details: Dict[str, Any] = {
            "reason": "no_valid_marker_ring",
            "quick_red_gate": quick_gate,
        }
        if markers is not None:
            details["marker_score"] = markers.score
        if structural is not None:
            details["structural_score"] = structural.score

        return RingDetectionResult(
            status=RingStatus.ABSENT,
            confidence=0.95,
            method="heuristic",
            details=details,
        )

    def _quick_red_gate(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > 160:
            scale = 160.0 / float(max_dim)
            small = cv2.resize(
                image,
                (max(32, int(round(w * scale))), max(32, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        red_excess = small[:, :, 2].astype(np.int16) - np.maximum(
            small[:, :, 1].astype(np.int16),
            small[:, :, 0].astype(np.int16),
        )
        red_mask = (
            (((hue <= 10) | (hue >= 170)) & (sat >= 95) & (val >= 90) & (red_excess >= 24))
        ).astype(np.uint8)

        red_pixels = int(np.count_nonzero(red_mask))
        if red_pixels < 18:
            return None

        ys, xs = np.nonzero(red_mask)
        cx = small.shape[1] / 2.0
        cy = small.shape[0] / 2.0
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        min_dim = float(min(small.shape[:2]))
        radial_mask = (dist >= min_dim * 0.22) & (dist <= min_dim * 0.52)
        radial_pixels = int(np.count_nonzero(radial_mask))
        if radial_pixels < 12:
            return None

        angles = (np.degrees(np.arctan2(ys[radial_mask] - cy, xs[radial_mask] - cx)) + 360.0) % 360.0
        coverage = self._angular_coverage(angles, bins=36)
        if coverage < 0.14:
            return None

        return {
            "red_pixels": float(red_pixels),
            "radial_pixels": float(radial_pixels),
            "coverage": float(coverage),
        }

    def _candidate_to_result(
        self,
        candidate: _CircleCandidate | _MarkerCandidate,
        status: RingStatus,
        method: str,
        corneal_reference_source: str = "limbus",
    ) -> RingDetectionResult:
        dot_centers = (
            list(candidate.dot_centers)
            if isinstance(candidate, _MarkerCandidate)
            else list(candidate.details.get("dot_centers", []))
        )
        return RingDetectionResult(
            status=status,
            confidence=float(np.clip(candidate.score, 0.0, 1.0)),
            ring_contour=candidate.contour,
            ring_center=candidate.center,
            ring_radius=candidate.radius,
            ring_inner_radius=candidate.details.get("ring_inner_radius"),
            ring_mask=candidate.mask,
            method=method,
            dot_centers=dot_centers,
            dot_count=len(dot_centers),
            corneal_reference_source=corneal_reference_source,
            details=dict(candidate.details),
        )

    def _detect_red_annulus(
        self,
        image: np.ndarray,
        structural: Optional[_CircleCandidate],
    ) -> Optional[_CircleCandidate]:
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        h, w = image.shape[:2]
        min_dim = min(h, w)
        cx0 = float(structural.center[0]) if structural is not None else w / 2.0
        cy0 = float(structural.center[1]) if structural is not None else h / 2.0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        strong_red = (
            (((hue <= 10) | (hue >= 170)) & (sat >= 115) & (val >= 115))
        ).astype(np.uint8) * 255
        if int(np.count_nonzero(strong_red)) < 200:
            return None

        red_excess = image[:, :, 2].astype(np.int16) - np.maximum(
            image[:, :, 1].astype(np.int16),
            image[:, :, 0].astype(np.int16),
        )
        strong_red[red_excess < 28] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        strong_red = cv2.morphologyEx(strong_red, cv2.MORPH_CLOSE, kernel, iterations=1)
        strong_red = cv2.morphologyEx(strong_red, cv2.MORPH_OPEN, kernel, iterations=1)

        ys, xs = np.nonzero(strong_red)
        if xs.size < 200:
            return None

        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        fit = self._fit_circle_robust(pts)
        if fit is None:
            return None

        fit_cx, fit_cy, fit_r, residual = fit
        min_r = min_dim * self.ring_radius_range[0]
        max_r = min_dim * self.ring_radius_range[1]
        if not (min_r <= fit_r <= max_r):
            return None

        center_offset = math.hypot((fit_cx - cx0) / w, (fit_cy - cy0) / h)
        if center_offset > 0.16:
            return None

        yy, xx = np.indices((h, w))
        dist = np.sqrt((xx - fit_cx) ** 2 + (yy - fit_cy) ** 2)
        band_width = max(8, int(round(fit_r * 0.03)))
        ring_band = np.abs(dist - fit_r) <= band_width
        angle = (np.degrees(np.arctan2(yy - fit_cy, xx - fit_cx)) + 360.0) % 360.0
        coverage = self._angular_coverage(angle[strong_red > 0], bins=72)
        band_red_frac = float(
            np.count_nonzero(strong_red[ring_band] > 0) / max(np.count_nonzero(ring_band), 1)
        )
        residual_ratio = residual / max(fit_r, 1.0)

        if coverage < 0.18 or band_red_frac < 0.03:
            return None

        ring_mask = np.zeros((h, w), dtype=np.uint8)
        ring_mask[ring_band] = 255
        ring_mask = cv2.bitwise_and(ring_mask, strong_red)
        contour = self._contour_from_mask(ring_mask)

        score = (
            min(coverage / 0.55, 1.0) * 0.42
            + min(band_red_frac / 0.20, 1.0) * 0.33
            + max(0.0, 1.0 - residual_ratio / 0.06) * 0.18
            + max(0.0, 1.0 - center_offset / 0.16) * 0.07
        )

        if contour is None:
            contour = self._circle_contour((float(fit_cx), float(fit_cy)), float(fit_r))

        return _CircleCandidate(
            center=(float(fit_cx), float(fit_cy)),
            radius=float(fit_r),
            score=float(np.clip(score, 0.0, 1.0)),
            mask=ring_mask,
            contour=contour,
            details={
                "coverage": coverage,
                "band_red_fraction": band_red_frac,
                "residual_std": residual,
                "residual_ratio": residual_ratio,
                "center_offset": center_offset,
                "pixel_count": int(xs.size),
            },
        )

    def _detect_marker_ring(self, image: np.ndarray) -> Optional[_MarkerCandidate]:
        _, marker_mask, ring_diag = self.marker_detector.remove_with_diagnostics(image)
        if (
            not ring_diag.detected
            or ring_diag.ring_centre is None
            or ring_diag.ring_radius is None
        ):
            return None

        h, w = image.shape[:2]
        center_offset = math.hypot(
            (ring_diag.ring_centre[0] - w / 2.0) / w,
            (ring_diag.ring_centre[1] - h / 2.0) / h,
        )
        min_dim = float(min(h, w))
        min_r = min_dim * self.ring_radius_range[0]
        max_r = min_dim * self.ring_radius_range[1]
        if not (min_r <= float(ring_diag.ring_radius) <= max_r):
            return None

        residual_ratio = (
            (ring_diag.ring_residual_std or 0.0) / max(ring_diag.ring_radius, 1.0)
        )
        dot_count = int(ring_diag.dot_count)
        if dot_count < 4:
            return None

        dots = np.array(ring_diag.dot_centres, dtype=np.float64)
        angles = (
            np.degrees(
                np.arctan2(
                    dots[:, 1] - ring_diag.ring_centre[1],
                    dots[:, 0] - ring_diag.ring_centre[0],
                )
            )
            + 360.0
        ) % 360.0
        coverage = self._angular_coverage(angles, bins=24)
        if coverage < 0.12:
            return None

        outer_radius = float(
            ring_diag.ring_outer_radius
            if getattr(ring_diag, "ring_outer_radius", None) is not None
            else ring_diag.ring_radius
        )
        inner_radius = float(
            ring_diag.ring_inner_radius
            if getattr(ring_diag, "ring_inner_radius", None) is not None
            else max(1.0, ring_diag.ring_radius)
        )
        contour = self._circle_contour(ring_diag.ring_centre, outer_radius)
        outer_boundary_std = float(
            getattr(ring_diag, "ring_outer_boundary_std", 0.0) or 0.0
        )
        outer_boundary_ratio = outer_boundary_std / max(outer_radius, 1.0)
        score = (
            min(dot_count / 7.0, 1.0) * 0.35
            + min(coverage / 0.40, 1.0) * 0.25
            + max(0.0, 1.0 - residual_ratio / 0.20) * 0.25
            + max(0.0, 1.0 - center_offset / 0.18) * 0.10
            + min(np.count_nonzero(marker_mask) / 1500.0, 1.0) * 0.03
            + max(0.0, 1.0 - outer_boundary_ratio / 0.018) * 0.02
        )

        return _MarkerCandidate(
            center=ring_diag.ring_centre,
            radius=outer_radius,
            score=float(np.clip(score, 0.0, 1.0)),
            mask=marker_mask,
            contour=contour,
            dot_centers=list(ring_diag.dot_centres),
            details={
                "dot_count": dot_count,
                "angular_coverage": coverage,
                "residual_std": ring_diag.ring_residual_std,
                "residual_ratio": residual_ratio,
                "center_offset": center_offset,
                "ring_inner_radius": inner_radius,
                "ring_outer_radius": outer_radius,
                "dot_radius_median": getattr(ring_diag, "dot_radius_median", None),
                "outer_boundary_std": outer_boundary_std,
                "outer_boundary_ratio": outer_boundary_ratio,
            },
        )

    def _detect_structural_ring(self, image: np.ndarray) -> Optional[_CircleCandidate]:
        h, w = image.shape[:2]
        min_dim = min(h, w)
        min_r = int(min_dim * self.ring_radius_range[0])
        max_r = int(min_dim * self.ring_radius_range[1])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
        enhanced = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=max(80, min_dim // 5),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is None:
            return None

        best: Optional[_CircleCandidate] = None
        for cx, cy, r in np.round(circles[0]).astype(int):
            center_offset = math.hypot((cx - w / 2.0) / w, (cy - h / 2.0) / h)
            if center_offset > 0.18:
                continue

            edge_density = self._ring_edge_density(edges, cx, cy, r, band_width=max(8, int(r * 0.03)))
            contrast = self._ring_contrast(enhanced, cx, cy, r, band=max(6, int(r * 0.02)))
            annular_peak = self._annular_edge_peak(edges, cx, cy, min_r, max_r)
            score = (
                min(edge_density / 0.08, 1.0) * 0.42
                + min(contrast / 28.0, 1.0) * 0.23
                + min(annular_peak / 2.2, 1.0) * 0.20
                + max(0.0, 1.0 - center_offset / 0.18) * 0.15
            )

            mask = self._create_ring_mask(
                h, w, cx, cy, r, thickness=max(10, int(r * 0.04))
            )
            contour = self._contour_from_mask(mask)
            if contour is None:
                contour = self._circle_contour((float(cx), float(cy)), float(r))
            candidate = _CircleCandidate(
                center=(float(cx), float(cy)),
                radius=float(r),
                score=float(np.clip(score, 0.0, 1.0)),
                mask=mask,
                contour=contour,
                details={
                    "edge_density": edge_density,
                    "contrast": contrast,
                    "annular_peak": annular_peak,
                    "center_offset": center_offset,
                },
            )
            if best is None or candidate.score > best.score:
                best = candidate

        return best

    @staticmethod
    def _fit_circle_robust(
        pts: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float]]:
        if pts.shape[0] < 12:
            return None

        rng = np.random.default_rng(12345)
        subset = pts
        if subset.shape[0] > 2500:
            subset = subset[rng.choice(subset.shape[0], size=2500, replace=False)]

        best_inliers = None
        best_model = None
        iters = min(80, max(20, subset.shape[0] // 40))
        for _ in range(iters):
            sample = subset[rng.choice(subset.shape[0], size=3, replace=False)]
            model = HeuristicRingDetector._fit_circle_lsq(sample)
            if model is None:
                continue
            cx, cy, r = model
            residual = np.abs(np.sqrt((subset[:, 0] - cx) ** 2 + (subset[:, 1] - cy) ** 2) - r)
            tol = max(5.0, r * 0.03)
            inliers = residual < tol
            if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_inliers = inliers
                best_model = model

        if best_model is None:
            return None

        refined_pts = (
            subset[best_inliers]
            if best_inliers is not None and np.count_nonzero(best_inliers) >= 12
            else subset
        )
        refined = HeuristicRingDetector._fit_circle_lsq(refined_pts)
        if refined is None:
            return None

        cx, cy, r = refined
        d = np.sqrt((subset[:, 0] - cx) ** 2 + (subset[:, 1] - cy) ** 2)
        residual_std = float(np.std(d - r))
        return float(cx), float(cy), float(r), residual_std

    @staticmethod
    def _fit_circle_lsq(pts: np.ndarray) -> Optional[Tuple[float, float, float]]:
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
        return float(cx), float(cy), float(math.sqrt(r_sq))

    @staticmethod
    def _angular_coverage(angles_deg: np.ndarray, bins: int = 72) -> float:
        if angles_deg.size == 0:
            return 0.0
        hist, _ = np.histogram(angles_deg, bins=bins, range=(0.0, 360.0))
        return float(np.count_nonzero(hist > 0) / bins)

    @staticmethod
    def _contour_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None or np.count_nonzero(mask) == 0:
            return None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        pts = np.vstack(contours)
        return cv2.convexHull(pts) if pts.shape[0] >= 5 else None

    @staticmethod
    def _ring_edge_density(
        edges: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        band_width: int = 15,
    ) -> float:
        h, w = edges.shape[:2]
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer, (cx, cy), r + band_width, 255, -1)
        cv2.circle(inner, (cx, cy), max(1, r - band_width), 255, -1)
        band = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        band_area = np.count_nonzero(band)
        if band_area == 0:
            return 0.0
        return float(np.count_nonzero(cv2.bitwise_and(edges, band)) / band_area)

    @staticmethod
    def _ring_contrast(
        gray: np.ndarray,
        cx: int,
        cy: int,
        r: int,
        band: int = 10,
    ) -> float:
        h, w = gray.shape[:2]
        inner_full = np.zeros((h, w), dtype=np.uint8)
        inner_hole = np.zeros((h, w), dtype=np.uint8)
        outer_full = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_full, (cx, cy), r, 255, -1)
        cv2.circle(inner_hole, (cx, cy), max(1, r - band), 255, -1)
        cv2.circle(outer_full, (cx, cy), r + band, 255, -1)
        inner_band = cv2.bitwise_and(inner_full, cv2.bitwise_not(inner_hole))
        outer_band = cv2.bitwise_and(outer_full, cv2.bitwise_not(inner_full))
        inner_vals = gray[inner_band > 0]
        outer_vals = gray[outer_band > 0]
        if inner_vals.size == 0 or outer_vals.size == 0:
            return 0.0
        return float(abs(np.mean(inner_vals) - np.mean(outer_vals)))

    def _annular_edge_peak(
        self,
        edges: np.ndarray,
        cx: int,
        cy: int,
        min_r: int,
        max_r: int,
    ) -> float:
        radii = np.linspace(min_r, max_r, 32)
        densities = np.array(
            [self._ring_edge_density(edges, cx, cy, int(r), band_width=8) for r in radii],
            dtype=np.float32,
        )
        if densities.size == 0:
            return 0.0
        return float(np.max(densities) / max(float(np.mean(densities)), 1e-6))

    @staticmethod
    def _create_ring_mask(
        h: int,
        w: int,
        cx: int,
        cy: int,
        r: int,
        thickness: int = 20,
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r + thickness, 255, -1)
        cv2.circle(mask, (cx, cy), max(1, r - thickness), 0, -1)
        return mask

    @staticmethod
    def _circle_contour(
        center: Tuple[float, float],
        radius: float,
        samples: int = 180,
    ) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        cx, cy = center
        pts = np.stack(
            [
                cx + radius * np.cos(angles),
                cy + radius * np.sin(angles),
            ],
            axis=1,
        )
        return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


class RingDetector:
    """
    High-level ring detector.

    The deterministic heuristic detector is the primary path. An
    optional CNN classifier is treated only as an auxiliary signal.
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: str = "auto",
        use_heuristic_fallback: bool = True,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.use_heuristic_fallback = use_heuristic_fallback
        self.heuristic = HeuristicRingDetector()
        self.classifier = None
        self.device = None

        if _HAS_TORCH:
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
            if classifier_path is not None:
                self._try_load_classifier(classifier_path)
        elif classifier_path is not None:
            logger.info("PyTorch unavailable; CNN ring classifier disabled.")

    def _try_load_classifier(self, path: str) -> None:
        from pathlib import Path

        if not Path(path).exists():
            logger.info("Ring classifier not found at %s; heuristic-only mode.", path)
            return
        try:
            from pupil_tracking.ml.ring_classifier import RingClassifierNet

            self.classifier = RingClassifierNet.load(path, device=self.device)
            self.classifier.eval()
        except Exception as exc:
            logger.warning("Could not load ring classifier: %s", exc)
            self.classifier = None

    def detect(self, image: np.ndarray) -> RingDetectionResult:
        heuristic_result = self.heuristic.detect(image)
        if self.classifier is None or not _HAS_TORCH:
            return heuristic_result
        classifier_result = self._run_classifier(image)
        return self._merge_results(classifier_result, heuristic_result)

    def _run_classifier(self, image: np.ndarray) -> RingDetectionResult:
        if not _HAS_TORCH:
            return RingDetectionResult(
                status=RingStatus.UNCERTAIN,
                confidence=0.0,
                method="classifier",
                details={"error": "PyTorch unavailable"},
            )

        from pupil_tracking.ml.ring_classifier import preprocess_for_classifier

        tensor = preprocess_for_classifier(image, size=224).to(self.device)
        with torch.no_grad():
            logits = self.classifier(tensor.unsqueeze(0))
            prob = torch.softmax(logits, dim=1)
            ring_prob = float(prob[0, 1])

        status = RingStatus.PRESENT if ring_prob >= 0.50 else RingStatus.ABSENT
        return RingDetectionResult(
            status=status,
            confidence=float(max(ring_prob, 1.0 - ring_prob)),
            method="classifier",
            details={"ring_probability": ring_prob},
        )

    @staticmethod
    def _merge_results(
        classifier_res: RingDetectionResult,
        heuristic_res: RingDetectionResult,
    ) -> RingDetectionResult:
        if heuristic_res.status == RingStatus.PRESENT:
            heuristic_res.confidence = min(
                1.0,
                heuristic_res.confidence * 0.85 + classifier_res.confidence * 0.15,
            )
            heuristic_res.method = f"{heuristic_res.method}+classifier"
            heuristic_res.details["classifier"] = classifier_res.details
            return heuristic_res

        if (
            heuristic_res.status == RingStatus.ABSENT
            and classifier_res.status == RingStatus.ABSENT
        ):
            heuristic_res.confidence = min(
                1.0,
                heuristic_res.confidence * 0.7 + classifier_res.confidence * 0.3,
            )
            heuristic_res.method = "heuristic+classifier"
            return heuristic_res

        if classifier_res.confidence >= 0.92 and classifier_res.status == RingStatus.PRESENT:
            classifier_res.details["heuristic"] = heuristic_res.details
            return classifier_res

        heuristic_res.details["classifier"] = classifier_res.details
        return heuristic_res
