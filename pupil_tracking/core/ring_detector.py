"""
ring_detector.py — Suction Ring Detection Module

Detects whether a suction ring is present in an eye image using:
  1. A fast CNN classifier (primary method, ~3 ms on GPU)
  2. Traditional CV heuristics (fallback / geometry enrichment)

The ring detector is the **first stage** of the adaptive pipeline.
Its output determines which preprocessing path, contour-filtering
constraints, and confidence adjustments are applied downstream.

Public classes
--------------
RingStatus              Enum of possible ring states.
RingDetectionResult     Dataclass bundling the full detection output.
HeuristicRingDetector   Traditional-CV ring finder (Hough + contour).
RingDetector            High-level façade combining CNN + heuristic.

Usage
-----
>>> from pupil_tracking.core.ring_detector import RingDetector, RingStatus
>>> detector = RingDetector(classifier_path="models/ring_classifier.pth")
>>> result = detector.detect(image)
>>> if result.status == RingStatus.PRESENT:
...     print(f"Ring at {result.ring_center}, r={result.ring_radius}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# PyTorch is optional — only needed for CNN classifier
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════

class RingStatus(Enum):
    """Classification of suction ring presence."""

    PRESENT = "ring_present"
    """Suction ring is clearly visible (docked image)."""

    ABSENT = "ring_absent"
    """No ring detected (pre-docked / natural eye image)."""

    PARTIAL = "ring_partial"
    """Ring is partially visible or occluded (e.g. by eyelid)."""

    UNCERTAIN = "ring_uncertain"
    """Classifier confidence is too low to make a reliable call."""


@dataclass
class RingDetectionResult:
    """Complete result of ring detection analysis."""

    status: RingStatus
    """Overall ring status classification."""

    confidence: float
    """Confidence in the status classification (0–1)."""

    ring_contour: Optional[np.ndarray] = None
    """Detected ring contour array (N×1×2) if available."""

    ring_center: Optional[Tuple[float, float]] = None
    """(x, y) centre of the ring in pixel coordinates."""

    ring_radius: Optional[float] = None
    """Outer radius of the ring in pixels."""

    ring_inner_radius: Optional[float] = None
    """Inner opening radius of the ring in pixels."""

    ring_mask: Optional[np.ndarray] = None
    """Binary uint8 mask highlighting the ring annulus region."""

    method: str = "unknown"
    """Detection method that produced this result:
    ``"classifier"``, ``"heuristic"``, ``"combined"``, ``"forced"``."""

    details: dict = field(default_factory=dict)
    """Free-form metadata for debugging and logging."""


# ═══════════════════════════════════════════════════════════════════════
#  Heuristic Ring Detector
# ═══════════════════════════════════════════════════════════════════════

class HeuristicRingDetector:
    """
    Traditional CV-based suction ring detection.

    Exploits the distinctive visual characteristics of suction rings:

    * Strong circular (or near-circular) edge in the mid-to-outer region
      of the image.
    * High contrast boundary where metal / plastic ring meets sclera or
      surrounding tissue.
    * Ring radius is typically *larger* than the limbus.
    * Ring often creates a shadow band or partial occlusion.

    Three complementary methods are tried and their candidate lists are
    merged:

    1. **Hough circle transform** on the CLAHE-enhanced grayscale image.
    2. **Contour analysis** on a Canny edge map — looks for large,
       circular contours centred near the image centre.
    3. **Annular edge-density scan** — sweeps a thin annulus outward
       from the image centre and looks for a peak in edge density that
       would indicate a ring.

    Parameters
    ----------
    canny_low, canny_high : int
        Canny edge detector thresholds.
    hough_dp : float
        Inverse ratio of accumulator resolution for ``HoughCircles``.
    hough_min_dist : int
        Minimum distance between detected circle centres.
    hough_param1, hough_param2 : int
        Canny upper threshold and accumulator threshold used internally
        by ``HoughCircles``.
    ring_radius_range : tuple of float
        ``(min_frac, max_frac)`` — ring radius expressed as a fraction
        of the smaller image dimension.
    min_ring_circularity : float
        Minimum circularity ``4π·area/perimeter²`` for contour method.
    edge_density_threshold : float
        Edge-pixel fraction in the ring band above which the Hough
        candidate scores well.
    """

    def __init__(
        self,
        canny_low: int = 30,
        canny_high: int = 100,
        hough_dp: float = 1.2,
        hough_min_dist: int = 100,
        hough_param1: int = 80,
        hough_param2: int = 40,
        ring_radius_range: Tuple[float, float] = (0.25, 0.48),
        min_ring_circularity: float = 0.70,
        edge_density_threshold: float = 0.12,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.ring_radius_range = ring_radius_range
        self.min_ring_circularity = min_ring_circularity
        self.edge_density_threshold = edge_density_threshold

    # ── public ────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> RingDetectionResult:
        """
        Detect suction ring using traditional image processing.

        Parameters
        ----------
        image : np.ndarray
            BGR ``(H, W, 3)`` or grayscale ``(H, W)`` input image.

        Returns
        -------
        RingDetectionResult
            Heuristic-based ring assessment including geometry when
            a ring is found.
        """
        h, w = image.shape[:2]
        min_dim = min(h, w)

        # Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Edge map
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)

        # Radius bounds in pixels
        min_r = int(min_dim * self.ring_radius_range[0])
        max_r = int(min_dim * self.ring_radius_range[1])

        # Collect candidates from all three methods
        ring_candidates: List[Dict[str, Any]] = []

        # Method 1 — Hough circles
        hough_candidates = self._hough_circle_candidates(
            enhanced, edges, w, h, min_r, max_r,
        )
        ring_candidates.extend(hough_candidates)

        # Method 2 — Contour analysis
        contour_candidate = self._contour_candidate(
            edges, gray, min_r, max_r, w, h,
        )
        if contour_candidate is not None:
            ring_candidates.append(contour_candidate)

        # Method 3 — Annular edge-density scan
        annular_score = self._annular_edge_analysis(edges, w, h, min_r, max_r)

        # ── Decision ──────────────────────────────────────────────
        if not ring_candidates:
            return RingDetectionResult(
                status=RingStatus.ABSENT,
                confidence=max(0.5, 1.0 - annular_score),
                method="heuristic",
                details={"annular_score": float(annular_score)},
            )

        best = max(ring_candidates, key=lambda c: c["score"])

        ring_mask = self._create_ring_mask(
            h, w,
            int(best["center"][0]),
            int(best["center"][1]),
            int(best["radius"]),
        )

        if best["score"] >= 0.60:
            status = RingStatus.PRESENT
        elif best["score"] >= 0.35:
            status = RingStatus.PARTIAL
        else:
            status = RingStatus.UNCERTAIN

        return RingDetectionResult(
            status=status,
            confidence=float(best["score"]),
            ring_center=best["center"],
            ring_radius=best["radius"],
            ring_mask=ring_mask,
            method="heuristic",
            details={
                "candidates": ring_candidates,
                "annular_score": float(annular_score),
                "best_candidate": best,
            },
        )

    # ── Method 1: Hough circles ──────────────────────────────────

    def _hough_circle_candidates(
        self,
        enhanced: np.ndarray,
        edges: np.ndarray,
        w: int, h: int,
        min_r: int, max_r: int,
    ) -> List[Dict[str, Any]]:
        """Find ring candidates via Hough circle transform."""
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        candidates: List[Dict[str, Any]] = []
        if circles is None:
            return candidates

        circles = np.round(circles[0]).astype(int)

        for cx, cy, r in circles:
            # Ring centre should be roughly centred in the image
            center_offset = np.sqrt(
                ((cx - w / 2.0) / w) ** 2 + ((cy - h / 2.0) / h) ** 2
            )
            if center_offset > 0.30:
                continue

            # Edge density along the detected circle
            edge_density = self._ring_edge_density(edges, cx, cy, r, band_width=15)

            # Intensity contrast across the ring boundary
            contrast = self._ring_contrast(
                cv2.cvtColor(edges, cv2.COLOR_GRAY2GRAY)
                if len(edges.shape) == 3
                else edges,
                cx, cy, r,
            )
            # Use enhanced image for contrast instead
            contrast = self._ring_contrast(
                cv2.GaussianBlur(
                    cv2.cvtColor(edges, cv2.COLOR_GRAY2GRAY)
                    if len(edges.shape) == 3
                    else edges, (1, 1), 0,
                ),
                cx, cy, r,
            )

            score = 0.0
            score += min(edge_density / self.edge_density_threshold, 1.0) * 0.40
            score += min(contrast / 40.0, 1.0) * 0.30
            score += (1.0 - center_offset / 0.30) * 0.20
            score += (r / max(max_r, 1)) * 0.10

            candidates.append({
                "center": (float(cx), float(cy)),
                "radius": float(r),
                "edge_density": float(edge_density),
                "contrast": float(contrast),
                "center_offset": float(center_offset),
                "score": float(np.clip(score, 0.0, 1.0)),
                "method": "hough",
            })

        return candidates

    # ── Method 2: Contour analysis ───────────────────────────────

    def _contour_candidate(
        self,
        edges: np.ndarray,
        gray: np.ndarray,
        min_r: int, max_r: int,
        w: int, h: int,
    ) -> Optional[Dict[str, Any]]:
        """Find the single best ring candidate via contour analysis."""
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )

        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_ring_circularity:
                continue

            equiv_r = np.sqrt(area / np.pi)
            if not (min_r <= equiv_r <= max_r):
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            center_offset = np.sqrt(
                ((cx - w / 2.0) / w) ** 2 + ((cy - h / 2.0) / h) ** 2
            )
            if center_offset > 0.25:
                continue

            score = (
                circularity * 0.50
                + (1.0 - center_offset / 0.25) * 0.30
                + 0.20
            )

            if score > best_score:
                best_score = score
                best = {
                    "center": (float(cx), float(cy)),
                    "radius": float(equiv_r),
                    "edge_density": float(circularity),
                    "contrast": 0.0,
                    "center_offset": float(center_offset),
                    "score": float(np.clip(score, 0.0, 1.0)),
                    "method": "contour",
                    "contour": cnt,
                }

        return best

    # ── Method 3: Annular edge-density scan ──────────────────────

    def _annular_edge_analysis(
        self,
        edges: np.ndarray,
        w: int, h: int,
        min_r: int, max_r: int,
    ) -> float:
        """
        Sweep an annulus outward from the image centre and look for a
        peak in edge density.  A strong peak suggests a ring.

        Returns a normalised score in ``[0, 1]``.
        """
        cx, cy = w // 2, h // 2
        num_radii = 50
        radii = np.linspace(min_r, max_r, num_radii)

        densities = np.array([
            self._ring_edge_density(edges, cx, cy, int(r), band_width=8)
            for r in radii
        ])

        if densities.size == 0:
            return 0.0

        peak = float(np.max(densities))
        mean = float(np.mean(densities))
        if mean == 0:
            return 0.0

        # A ring produces a sharp peak relative to the mean
        return float(np.clip((peak / mean) / 5.0, 0.0, 1.0))

    # ── Shared helpers ───────────────────────────────────────────

    def _ring_edge_density(
        self,
        edges: np.ndarray,
        cx: int, cy: int, r: int,
        band_width: int = 15,
    ) -> float:
        """Fraction of edge pixels inside a thin annulus at radius *r*."""
        h, w = edges.shape[:2]

        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_outer, (cx, cy), r + band_width, 255, -1)
        cv2.circle(mask_inner, (cx, cy), max(1, r - band_width), 255, -1)
        band = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

        band_area = np.count_nonzero(band)
        if band_area == 0:
            return 0.0

        edge_pixels = np.count_nonzero(cv2.bitwise_and(edges, band))
        return float(edge_pixels / band_area)

    def _ring_contrast(
        self,
        gray: np.ndarray,
        cx: int, cy: int, r: int,
        band: int = 10,
    ) -> float:
        """Absolute intensity difference across the ring boundary."""
        h, w = gray.shape[:2]

        # Inner band: [r - band, r)
        inner_full = np.zeros((h, w), dtype=np.uint8)
        inner_hole = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner_full, (cx, cy), r, 255, -1)
        cv2.circle(inner_hole, (cx, cy), max(1, r - band), 255, -1)
        inner_band = cv2.bitwise_and(inner_full, cv2.bitwise_not(inner_hole))

        # Outer band: (r, r + band]
        outer_full = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer_full, (cx, cy), r + band, 255, -1)
        outer_band = cv2.bitwise_and(outer_full, cv2.bitwise_not(inner_full))

        inner_vals = gray[inner_band > 0]
        outer_vals = gray[outer_band > 0]
        if inner_vals.size == 0 or outer_vals.size == 0:
            return 0.0

        return float(abs(np.mean(inner_vals) - np.mean(outer_vals)))

    @staticmethod
    def _create_ring_mask(
        h: int, w: int,
        cx: int, cy: int, r: int,
        thickness: int = 20,
    ) -> np.ndarray:
        """Create a binary mask of the ring annulus region."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r + thickness, 255, -1)
        cv2.circle(mask, (cx, cy), max(1, r - thickness), 0, -1)
        return mask


# ═══════════════════════════════════════════════════════════════════════
#  Combined Ring Detector (CNN + Heuristic)
# ═══════════════════════════════════════════════════════════════════════

class RingDetector:
    """
    High-level ring detector that combines a fast CNN classifier with
    traditional CV heuristics.

    Detection strategy
    ------------------
    1. If a trained CNN classifier is available, run it first (~3 ms).
    2. If classifier confidence exceeds the threshold, accept the result
       immediately.  When the verdict is *PRESENT*, optionally run the
       heuristic detector to enrich the result with ring geometry
       (centre, radius, mask).
    3. If confidence is below the threshold **or** no classifier is
       loaded, fall back to the heuristic detector.
    4. When both signals are available, merge them with a weighted
       combination (classifier 65 %, heuristic 35 %) and apply an
       agreement bonus or disagreement penalty.

    Parameters
    ----------
    classifier_path : str or None
        Path to a trained ``RingClassifierNet`` checkpoint (``.pth``).
        If ``None`` or the file does not exist, heuristic-only mode
        is used.
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        Ignored when PyTorch is not installed.
    use_heuristic_fallback : bool
        When ``True`` (default), the heuristic detector is used as a
        fallback and for geometry enrichment.
    confidence_threshold : float
        Minimum CNN confidence to accept without heuristic validation.

    Usage
    -----
    >>> detector = RingDetector(classifier_path="models/ring_classifier.pth")
    >>> result = detector.detect(image)
    >>> print(result.status, result.confidence)
    """

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: str = "auto",
        use_heuristic_fallback: bool = True,
        confidence_threshold: float = 0.70,
    ):
        self.confidence_threshold = confidence_threshold
        self.use_heuristic_fallback = use_heuristic_fallback
        self.heuristic = HeuristicRingDetector()
        self.classifier = None
        self.device = None

        # ── Resolve device (only if PyTorch is available) ─────────
        if _HAS_TORCH:
            if device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ):
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)

            # ── Load CNN classifier (if available) ────────────────
            if classifier_path is not None:
                self._try_load_classifier(classifier_path)
        else:
            if classifier_path is not None:
                logger.info(
                    "PyTorch not installed — CNN ring classifier unavailable. "
                    "Using heuristic-only mode."
                )

    def _try_load_classifier(self, path: str) -> None:
        """Attempt to load the ring classifier; warn on failure."""
        if not _HAS_TORCH:
            logger.info(
                "PyTorch not installed — cannot load ring classifier."
            )
            return

        from pathlib import Path as _Path

        if not _Path(path).exists():
            logger.info(
                "Ring classifier not found at %s — using heuristic only.", path,
            )
            return

        try:
            from pupil_tracking.ml.ring_classifier import RingClassifierNet

            self.classifier = RingClassifierNet.load(path, device=self.device)
            self.classifier.eval()
            logger.info(
                "Ring classifier loaded from %s on %s", path, self.device,
            )
        except Exception as exc:
            logger.warning("Could not load ring classifier: %s", exc)
            self.classifier = None

    # ── public ────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> RingDetectionResult:
        """
        Detect suction ring presence in *image*.

        Parameters
        ----------
        image : np.ndarray
            BGR ``(H, W, 3)`` or grayscale ``(H, W)``.

        Returns
        -------
        RingDetectionResult
        """
        classifier_result: Optional[RingDetectionResult] = None
        heuristic_result: Optional[RingDetectionResult] = None

        # ── Stage 1: CNN classifier ──────────────────────────────
        if self.classifier is not None and _HAS_TORCH:
            classifier_result = self._run_classifier(image)

            # High confidence → accept (optionally enrich with geometry)
            if classifier_result.confidence >= self.confidence_threshold:
                if (
                    self.use_heuristic_fallback
                    and classifier_result.status == RingStatus.PRESENT
                ):
                    heuristic_result = self.heuristic.detect(image)
                    return self._merge_results(classifier_result, heuristic_result)
                return classifier_result

        # ── Stage 2: Heuristic fallback ──────────────────────────
        if self.use_heuristic_fallback:
            heuristic_result = self.heuristic.detect(image)

            if classifier_result is not None:
                return self._merge_results(classifier_result, heuristic_result)
            return heuristic_result

        # ── Stage 3: No detector available ───────────────────────
        if classifier_result is not None:
            return classifier_result

        logger.warning("No ring detection method available — assuming absent.")
        return RingDetectionResult(
            status=RingStatus.UNCERTAIN,
            confidence=0.0,
            method="none",
        )

    # ── CNN classifier helper ────────────────────────────────────

    def _run_classifier(self, image: np.ndarray) -> RingDetectionResult:
        """Run CNN classifier and return a preliminary result."""
        if not _HAS_TORCH:
            return RingDetectionResult(
                status=RingStatus.UNCERTAIN,
                confidence=0.0,
                method="classifier",
                details={"error": "PyTorch not available"},
            )

        from pupil_tracking.ml.ring_classifier import preprocess_for_classifier

        tensor = preprocess_for_classifier(image, size=224).to(self.device)

        with torch.no_grad():
            logits = self.classifier(tensor.unsqueeze(0))
            prob = torch.softmax(logits, dim=1)
            ring_prob = float(prob[0, 1])  # class 1 = ring present

        if ring_prob >= 0.50:
            status = RingStatus.PRESENT
        else:
            status = RingStatus.ABSENT

        return RingDetectionResult(
            status=status,
            confidence=float(max(ring_prob, 1.0 - ring_prob)),
            method="classifier",
            details={"ring_probability": ring_prob},
        )

    # ── Merge classifier + heuristic ─────────────────────────────

    @staticmethod
    def _merge_results(
        classifier_res: RingDetectionResult,
        heuristic_res: RingDetectionResult,
    ) -> RingDetectionResult:
        """
        Combine CNN classifier and heuristic results.

        * Weights: classifier 65 %, heuristic 35 %.
        * Agreement bonus: ×1.15.
        * Disagreement penalty: ×0.80.
        * Geometry (centre, radius, mask) comes from the heuristic.
        """
        w_cls, w_heur = 0.65, 0.35
        combined_conf = (
            w_cls * classifier_res.confidence
            + w_heur * heuristic_res.confidence
        )

        # Agreement / disagreement adjustment
        if classifier_res.status == heuristic_res.status:
            combined_conf = min(combined_conf * 1.15, 1.0)
        else:
            combined_conf *= 0.80

        # Status: trust classifier unless its confidence is very low
        if classifier_res.confidence >= 0.55:
            final_status = classifier_res.status
        else:
            final_status = heuristic_res.status

        if combined_conf < 0.50:
            final_status = RingStatus.UNCERTAIN

        return RingDetectionResult(
            status=final_status,
            confidence=float(combined_conf),
            ring_contour=heuristic_res.ring_contour,
            ring_center=heuristic_res.ring_center,
            ring_radius=heuristic_res.ring_radius,
            ring_inner_radius=heuristic_res.ring_inner_radius,
            ring_mask=heuristic_res.ring_mask,
            method="combined",
            details={
                "classifier": classifier_res.details,
                "heuristic": heuristic_res.details,
                "weight_classifier": w_cls,
                "weight_heuristic": w_heur,
            },
        )