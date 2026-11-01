"""
confidence.py — Ring-Aware Confidence Scoring

Computes detection confidence and quality metrics for pupil, limbus,
and ring detections.  The scoring system accounts for whether a
suction ring is present (docked) or absent (pre-docked), adjusting
expectations accordingly.

When a ring is detected:
  - Pupil confidence gets a bonus if it lies inside the ring opening.
  - Limbus confidence gets a bonus for concentricity with the ring.
  - Overall confidence factors in ring detection confidence.

When no ring is detected:
  - Standard scoring based on contour quality, circularity, and
    detection agreement.

Quality levels
--------------
    EXCELLENT   ≥ 0.90 overall, both pupil and limbus detected
    GOOD        ≥ 0.75 overall, pupil detected
    FAIR        ≥ 0.55 overall, pupil detected
    POOR        < 0.55 overall, pupil detected
    UNUSABLE    pupil not detected

Usage
-----
>>> from pupil_tracking.core.confidence import ConfidenceScorer, QualityLevel
>>> scorer = ConfidenceScorer()
>>> score = scorer.compute_pupil_confidence(contour, image_shape)
>>> quality = scorer.assess_quality(pupil_conf, limbus_conf, ring_conf)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Quality Levels
# ═══════════════════════════════════════════════════════════════════════

class QualityLevel(Enum):
    """Five-point quality classification for detection results."""

    EXCELLENT = "EXCELLENT"
    """High contrast, centred, clear boundaries, ≥ 0.90 confidence."""

    GOOD = "GOOD"
    """Minor reflections or shadows, slightly off-centre, ≥ 0.75."""

    FAIR = "FAIR"
    """Moderate artefacts, low contrast regions, ≥ 0.55."""

    POOR = "POOR"
    """Significant occlusion or heavy artefacts, < 0.55."""

    UNUSABLE = "UNUSABLE"
    """Cannot reliably detect pupil or iris."""


# ═══════════════════════════════════════════════════════════════════════
#  Confidence Scorer
# ═══════════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """
    Computes confidence scores for pupil, limbus, and ring detections.

    All scores are in the range ``[0, 1]`` where 1 is perfect.

    Parameters
    ----------
    circularity_weight : float
        Weight of circularity in the pupil confidence score.
    area_weight : float
        Weight of area proportion in the pupil confidence score.
    centrality_weight : float
        Weight of spatial centrality in the pupil confidence score.
    ring_bonus : float
        Confidence bonus when detection is consistent with ring
        geometry (applied multiplicatively: ``conf *= 1 + ring_bonus``).
    ring_penalty : float
        Confidence penalty when detection is inconsistent with ring
        geometry (applied multiplicatively: ``conf *= 1 - ring_penalty``).
    """

    def __init__(
        self,
        circularity_weight: float = 0.40,
        area_weight: float = 0.30,
        centrality_weight: float = 0.30,
        ring_bonus: float = 0.10,
        ring_penalty: float = 0.15,
    ):
        self.circularity_weight = circularity_weight
        self.area_weight = area_weight
        self.centrality_weight = centrality_weight
        self.ring_bonus = ring_bonus
        self.ring_penalty = ring_penalty

    # ── Pupil confidence ──────────────────────────────────────

    def compute_pupil_confidence(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, int],
        ring_center: Optional[Tuple[float, float]] = None,
        ring_radius: Optional[float] = None,
    ) -> float:
        """Compute confidence score for a pupil contour candidate.

        Parameters
        ----------
        contour : np.ndarray
            Pupil contour array ``(N, 1, 2)`` or ``(N, 2)``.
        image_shape : (height, width)
            Shape of the source image.
        ring_center : tuple or None
            ``(x, y)`` ring centre if ring is detected.
        ring_radius : float or None
            Ring radius in pixels.

        Returns
        -------
        float
            Confidence in ``[0, 1]``.
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0 or area < 10:
            return 0.0

        h, w = image_shape[:2]
        img_area = h * w

        # ── Circularity ──────────────────────────────────────
        circularity = 4.0 * np.pi * area / (perimeter ** 2)
        circ_score = min(circularity, 1.0)

        # ── Area proportion ──────────────────────────────────
        area_frac = area / img_area
        if 0.005 < area_frac < 0.15:
            area_score = 1.0
        elif 0.002 < area_frac < 0.25:
            area_score = 0.6
        elif 0.001 < area_frac < 0.35:
            area_score = 0.3
        else:
            area_score = 0.1

        # ── Centrality ───────────────────────────────────────
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        if ring_center is not None and ring_radius is not None:
            # Distance to ring centre (normalised by ring radius)
            ring_cx, ring_cy = ring_center
            dist = np.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)
            centrality = 1.0 - min(dist / max(ring_radius, 1.0), 1.0)
        else:
            # Distance to image centre (normalised)
            center_dist = np.sqrt(
                ((cx - w / 2.0) / w) ** 2
                + ((cy - h / 2.0) / h) ** 2
            )
            centrality = 1.0 - min(center_dist / 0.5, 1.0)

        cent_score = centrality

        # ── Weighted combination ─────────────────────────────
        score = (
            self.circularity_weight * circ_score
            + self.area_weight * area_score
            + self.centrality_weight * cent_score
        )

        # ── Ring consistency bonus / penalty ──────────────────
        if ring_center is not None and ring_radius is not None:
            equiv_radius = np.sqrt(area / np.pi)
            ring_cx, ring_cy = ring_center
            dist = np.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)

            # Bonus: pupil is well inside ring and appropriately sized
            if dist < ring_radius * 0.5 and equiv_radius < ring_radius * 0.4:
                score *= (1.0 + self.ring_bonus)
            # Penalty: pupil is near ring edge or too large
            elif dist > ring_radius * 0.75 or equiv_radius > ring_radius * 0.6:
                score *= (1.0 - self.ring_penalty)

        return float(np.clip(score, 0.0, 1.0))

    # ── Limbus confidence ─────────────────────────────────────

    def compute_limbus_confidence(
        self,
        center: Tuple[float, float],
        radius: float,
        image_shape: Tuple[int, int],
        pupil_center: Optional[Tuple[float, float]] = None,
        pupil_radius: Optional[float] = None,
        ring_center: Optional[Tuple[float, float]] = None,
        ring_radius: Optional[float] = None,
    ) -> float:
        """Compute confidence score for a limbus detection.

        Parameters
        ----------
        center : (x, y)
            Limbus centre coordinates.
        radius : float
            Limbus radius in pixels.
        image_shape : (height, width)
            Source image shape.
        pupil_center : tuple or None
            Pupil centre for concentricity check.
        pupil_radius : float or None
            Pupil radius for size-ratio check.
        ring_center : tuple or None
            Ring centre for containment check.
        ring_radius : float or None
            Ring radius.

        Returns
        -------
        float
            Confidence in ``[0, 1]``.
        """
        h, w = image_shape[:2]
        cx, cy = center
        score = 0.0

        # ── Concentricity with pupil ─────────────────────────
        if pupil_center is not None:
            px, py = pupil_center
            dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            concentricity = 1.0 - min(dist / max(radius, 1.0), 1.0)
            score += concentricity * 0.40
        else:
            # Centrality in image
            center_dist = np.sqrt(
                ((cx - w / 2.0) / w) ** 2
                + ((cy - h / 2.0) / h) ** 2
            )
            score += (1.0 - min(center_dist / 0.5, 1.0)) * 0.25

        # ── Size ratio with pupil ────────────────────────────
        if pupil_radius is not None and pupil_radius > 0:
            ratio = radius / pupil_radius
            if 2.0 < ratio < 3.5:
                score += 0.30
            elif 1.5 < ratio < 4.5:
                score += 0.15
            else:
                score += 0.05
        else:
            # Reasonable absolute size
            min_dim = min(h, w)
            size_frac = radius / min_dim
            if 0.08 < size_frac < 0.30:
                score += 0.20
            else:
                score += 0.05

        # ── Ring containment ─────────────────────────────────
        if ring_center is not None and ring_radius is not None:
            ring_cx, ring_cy = ring_center
            dist_to_ring = np.sqrt(
                (cx - ring_cx) ** 2 + (cy - ring_cy) ** 2
            )
            # Limbus must fit inside ring
            if dist_to_ring + radius <= ring_radius:
                score += 0.30
            elif dist_to_ring + radius <= ring_radius * 1.1:
                score += 0.15
            else:
                score += 0.0  # Extends outside ring
        else:
            score += 0.15  # No ring constraint

        return float(np.clip(score, 0.0, 1.0))

    # ── Ring confidence ───────────────────────────────────────

    def compute_ring_confidence(
        self,
        classifier_confidence: float = 0.0,
        heuristic_confidence: float = 0.0,
        segmentation_area_fraction: float = 0.0,
        has_classifier: bool = False,
        has_heuristic: bool = False,
        has_segmentation: bool = False,
    ) -> float:
        """Compute combined ring detection confidence.

        Merges signals from up to three sources: CNN classifier,
        heuristic detector, and segmentation mask.

        Parameters
        ----------
        classifier_confidence : float
            CNN classifier confidence (0–1).
        heuristic_confidence : float
            Heuristic detector confidence (0–1).
        segmentation_area_fraction : float
            Fraction of image classified as ring by segmentation.
        has_classifier : bool
            Whether classifier result is available.
        has_heuristic : bool
            Whether heuristic result is available.
        has_segmentation : bool
            Whether segmentation ring class is available.

        Returns
        -------
        float
            Combined ring confidence in ``[0, 1]``.
        """
        signals = []
        weights = []

        if has_classifier:
            signals.append(classifier_confidence)
            weights.append(0.50)

        if has_heuristic:
            signals.append(heuristic_confidence)
            weights.append(0.30)

        if has_segmentation:
            # Convert area fraction to a confidence-like score
            # Ring typically covers 5–15% of image
            if segmentation_area_fraction > 0.03:
                seg_conf = min(segmentation_area_fraction / 0.10, 1.0)
            else:
                seg_conf = 0.0
            signals.append(seg_conf)
            weights.append(0.20)

        if not signals:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        combined = sum(s * w for s, w in zip(signals, weights)) / total_weight

        # Agreement bonus: if all signals agree (all high or all low)
        if len(signals) >= 2:
            all_high = all(s > 0.5 for s in signals)
            all_low = all(s <= 0.5 for s in signals)
            if all_high or all_low:
                combined = min(combined * 1.10, 1.0)
            else:
                combined *= 0.85

        return float(np.clip(combined, 0.0, 1.0))

    # ── Overall confidence ────────────────────────────────────

    def compute_overall_confidence(
        self,
        pupil_confidence: float = 0.0,
        limbus_confidence: float = 0.0,
        ring_confidence: float = 0.0,
        pupil_detected: bool = False,
        limbus_detected: bool = False,
        ring_detected: bool = False,
    ) -> float:
        """Compute the overall detection confidence.

        Weighted mean of component confidences with detection-aware
        weighting.

        Parameters
        ----------
        pupil_confidence : float
            Pupil detection confidence.
        limbus_confidence : float
            Limbus detection confidence.
        ring_confidence : float
            Ring detection confidence.
        pupil_detected : bool
            Whether pupil was detected.
        limbus_detected : bool
            Whether limbus was detected.
        ring_detected : bool
            Whether ring was detected (or confidently absent).

        Returns
        -------
        float
            Overall confidence in ``[0, 1]``.
        """
        components = []
        weights = []

        if pupil_detected:
            components.append(pupil_confidence)
            weights.append(1.0)  # Pupil is most important

        if limbus_detected:
            components.append(limbus_confidence)
            weights.append(0.8)

        # Ring confidence always contributes (even if absent —
        # high confidence in "no ring" is still informative)
        components.append(ring_confidence)
        weights.append(0.3)

        if not components:
            return 0.0

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return float(sum(c * w for c, w in zip(components, weights)) / total_weight)

    # ── Quality assessment ────────────────────────────────────

    def assess_quality(
        self,
        overall_confidence: float,
        pupil_detected: bool = False,
        limbus_detected: bool = False,
    ) -> QualityLevel:
        """Map overall confidence to a quality level.

        Parameters
        ----------
        overall_confidence : float
            From ``compute_overall_confidence``.
        pupil_detected : bool
            Whether pupil was detected.
        limbus_detected : bool
            Whether limbus was detected.

        Returns
        -------
        QualityLevel
        """
        c = overall_confidence

        if c >= 0.90 and pupil_detected and limbus_detected:
            return QualityLevel.EXCELLENT
        elif c >= 0.75 and pupil_detected:
            return QualityLevel.GOOD
        elif c >= 0.55 and pupil_detected:
            return QualityLevel.FAIR
        elif pupil_detected:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE

    # ── Detailed report ───────────────────────────────────────

    def generate_confidence_report(
        self,
        pupil_confidence: float = 0.0,
        limbus_confidence: float = 0.0,
        ring_confidence: float = 0.0,
        pupil_detected: bool = False,
        limbus_detected: bool = False,
        ring_detected: bool = False,
        image_category: str = "unknown",
    ) -> Dict[str, Any]:
        """Generate a detailed confidence report.

        Parameters
        ----------
        pupil_confidence, limbus_confidence, ring_confidence : float
            Individual component confidences.
        pupil_detected, limbus_detected, ring_detected : bool
            Detection flags.
        image_category : str
            ``"docked"`` or ``"pre_docked"``.

        Returns
        -------
        dict
            Detailed report with all scores and quality assessment.
        """
        overall = self.compute_overall_confidence(
            pupil_confidence, limbus_confidence, ring_confidence,
            pupil_detected, limbus_detected, ring_detected,
        )
        quality = self.assess_quality(
            overall, pupil_detected, limbus_detected,
        )

        return {
            "pupil_confidence": round(pupil_confidence, 4),
            "pupil_detected": pupil_detected,
            "limbus_confidence": round(limbus_confidence, 4),
            "limbus_detected": limbus_detected,
            "ring_confidence": round(ring_confidence, 4),
            "ring_detected": ring_detected,
            "image_category": image_category,
            "overall_confidence": round(overall, 4),
            "quality_level": quality.value,
        }