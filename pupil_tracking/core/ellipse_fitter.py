"""
Surgical-grade ellipse fitting with RANSAC and a graduated fallback chain.

Fixes the critical ``valid``-key bug: every result is now a ``FitResult``
data-class that always exposes ``.valid``.

Fallback chain:
    1. RANSAC ellipse  (≥ 10 points)
    2. Direct ellipse  (≥ 5 points, cv2.fitEllipse)
    3. Huber-weighted iterative ellipse  (≥ 5 points)
    4. Algebraic circle (Kåsa)  (≥ 3 points)
    5. cv2.minEnclosingCircle  (≥ 1 point, always succeeds)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.types import FitResult
from pupil_tracking.utils.config import get_config


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

class EllipseFitter:
    """Stateless fitter — all methods are class-level or static."""

    @classmethod
    def fit(
        cls,
        contour: np.ndarray,
        *,
        prefer_ellipse: bool = True,
        config=None,
    ) -> FitResult:
        """Fit an ellipse (or circle) to *contour* using the fallback chain.

        Parameters
        ----------
        contour : np.ndarray
            Shape ``(N, 1, 2)`` or ``(N, 2)``, dtype any numeric.
        prefer_ellipse : bool
            If False, skip ellipse methods and go straight to circle.
        config : PupilTrackingConfig | None
            Optional configuration override.

        Returns
        -------
        FitResult
            Always returned (never None).  Check ``.valid``.
        """
        cfg = config or get_config()
        pts = _normalise(contour)
        if pts is None or len(pts) < 1:
            return FitResult(valid=False, method="none_empty_input")

        n = len(pts)
        fc = cfg.fitting

        if prefer_ellipse and n >= 10:
            result = cls._ransac_ellipse(
                pts,
                max_iters=fc.ransac_iterations,
                threshold=fc.ransac_threshold,
                min_inlier_ratio=fc.ransac_min_inlier_ratio,
            )
            if result.valid and result.fit_quality_score >= fc.min_fit_quality:
                return result

        if prefer_ellipse and n >= 5:
            result = cls._direct_ellipse(pts)
            if result.valid and result.fit_quality_score >= fc.min_fit_quality:
                return result

        if prefer_ellipse and n >= 5:
            result = cls._huber_ellipse(
                pts, delta=fc.huber_delta, max_iters=fc.huber_max_iters,
            )
            if result.valid and result.fit_quality_score >= fc.min_fit_quality:
                return result

        if n >= 3:
            result = cls._circle_fit(pts)
            if result.valid:
                return result

        return cls._min_enclosing(pts)

    # ──────────────────────────────────────────────────────────────
    # RANSAC ellipse
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _ransac_ellipse(
        cls,
        pts: np.ndarray,
        max_iters: int = 200,
        threshold: float = 2.0,
        min_inlier_ratio: float = 0.60,
    ) -> FitResult:
        n = len(pts)
        best_inliers: Optional[np.ndarray] = None
        best_count = 0
        rng = np.random.RandomState(42)

        sample_size = min(6, n)
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)

        for _ in range(max_iters):
            idx = rng.choice(n, sample_size, replace=False)
            sample = pts_cv[idx]
            try:
                eparams = cv2.fitEllipse(sample)
            except cv2.error:
                continue

            if not _cv_ellipse_sane(eparams):
                continue

            dists = _sampson_distances(pts, eparams)
            if dists is None:
                continue

            inliers = dists < threshold
            cnt = int(inliers.sum())
            if cnt > best_count:
                best_count = cnt
                best_inliers = inliers

        if best_inliers is None or best_count < max(5, int(n * min_inlier_ratio)):
            return FitResult(valid=False, method="ransac_no_consensus")

        inlier_pts = pts_cv[best_inliers]
        if len(inlier_pts) < 5:
            return FitResult(valid=False, method="ransac_too_few_inliers")

        try:
            final = cv2.fitEllipse(inlier_pts)
        except cv2.error:
            return FitResult(valid=False, method="ransac_refit_failed")

        if not _cv_ellipse_sane(final):
            return FitResult(valid=False, method="ransac_bad_refit")

        # Phase 4: Iterative refinement — tighten inlier set and refit
        refined_final = final
        refined_inliers = best_inliers
        for _ in range(2):
            dists = _sampson_distances(pts, refined_final)
            if dists is None:
                break
            tight_threshold = threshold * 0.7
            tight_mask = dists < tight_threshold
            n_tight = int(tight_mask.sum())
            if n_tight < max(5, int(n * min_inlier_ratio * 0.8)):
                break
            tight_pts = pts_cv[tight_mask]
            try:
                candidate = cv2.fitEllipse(tight_pts)
            except cv2.error:
                break
            if not _cv_ellipse_sane(candidate):
                break
            new_rms = _rms_distance(pts[tight_mask], candidate)
            old_rms = _rms_distance(pts[refined_inliers], refined_final)
            if new_rms <= old_rms:
                refined_final = candidate
                refined_inliers = tight_mask
            else:
                break

        rms = _rms_distance(pts[refined_inliers], refined_final)
        quality = _quality_score(
            int(refined_inliers.sum()) / n, rms, refined_final, n
        )

        return _build_result(
            refined_final, n, int(refined_inliers.sum()), rms, quality,
            "ransac_ellipse",
        )

    # ──────────────────────────────────────────────────────────────
    # Direct ellipse (cv2.fitEllipse)
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _direct_ellipse(cls, pts: np.ndarray) -> FitResult:
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
        try:
            eparams = cv2.fitEllipse(pts_cv)
        except cv2.error:
            return FitResult(valid=False, method="direct_failed")

        if not _cv_ellipse_sane(eparams):
            return FitResult(valid=False, method="direct_bad_shape")

        rms = _rms_distance(pts, eparams)
        quality = _quality_score(1.0, rms, eparams, len(pts))
        return _build_result(eparams, len(pts), len(pts), rms, quality, "direct_ellipse")

    # ──────────────────────────────────────────────────────────────
    # Huber-weighted iterative ellipse
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _huber_ellipse(
        cls, pts: np.ndarray, delta: float = 1.5, max_iters: int = 10,
    ) -> FitResult:
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
        n = len(pts)
        weights = np.ones(n, dtype=np.float64)

        current_eparams = None
        for _ in range(max_iters):
            # build weighted point cloud (replicate by weight)
            w_int = np.clip(np.round(weights * 10).astype(int), 1, 100)
            expanded = np.repeat(pts_cv, w_int, axis=0)
            if len(expanded) < 5:
                break
            try:
                current_eparams = cv2.fitEllipse(expanded)
            except cv2.error:
                break

            if not _cv_ellipse_sane(current_eparams):
                current_eparams = None
                break

            dists = _sampson_distances(pts, current_eparams)
            if dists is None:
                current_eparams = None
                break

            # Huber weights
            for i in range(n):
                d = dists[i]
                weights[i] = 1.0 if d <= delta else delta / max(d, 1e-9)

        if current_eparams is None:
            return FitResult(valid=False, method="huber_failed")

        rms = _rms_distance(pts, current_eparams)
        inlier_count = int((weights > 0.5).sum())
        quality = _quality_score(inlier_count / n, rms, current_eparams, n)
        return _build_result(
            current_eparams, n, inlier_count, rms, quality, "huber_ellipse",
        )

    # ──────────────────────────────────────────────────────────────
    # Algebraic circle fit (Kåsa method)
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _circle_fit(cls, pts: np.ndarray) -> FitResult:
        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        n = len(pts)

        A = np.column_stack([x, y, np.ones(n)])
        b = x ** 2 + y ** 2
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return FitResult(valid=False, method="circle_lstsq_failed")

        cx = result[0] / 2.0
        cy = result[1] / 2.0
        r = math.sqrt(max(result[2] + cx ** 2 + cy ** 2, 0.0))

        if r < 1.0 or not math.isfinite(cx) or not math.isfinite(cy):
            return FitResult(valid=False, method="circle_bad_params")

        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
        rms = float(np.sqrt(np.mean(dists ** 2)))

        uc = max(1.0, rms)
        ur = max(0.5, rms)
        quality = _circle_quality(rms, r, n)

        return FitResult(
            valid=True,
            center_x=float(cx),
            center_y=float(cy),
            semi_major=float(r),
            semi_minor=float(r),
            angle_deg=0.0,
            radius=float(r),
            eccentricity=0.0,
            fit_quality_score=quality,
            rms_residual=rms,
            num_inliers=n,
            num_points=n,
            method="circle_kasa",
            uncertainty_center=(uc, uc),
            uncertainty_radius=ur,
        )

    # ──────────────────────────────────────────────────────────────
    # Minimum enclosing circle (always succeeds)
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def _min_enclosing(cls, pts: np.ndarray) -> FitResult:
        pts_cv = pts.reshape(-1, 1, 2).astype(np.float32)
        (cx, cy), r = cv2.minEnclosingCircle(pts_cv)
        r = max(float(r), 1.0)
        return FitResult(
            valid=True,
            center_x=float(cx),
            center_y=float(cy),
            semi_major=r,
            semi_minor=r,
            angle_deg=0.0,
            radius=r,
            eccentricity=0.0,
            fit_quality_score=0.15,
            rms_residual=0.0,
            num_inliers=len(pts),
            num_points=len(pts),
            method="min_enclosing",
            uncertainty_center=(r * 0.1, r * 0.1),
            uncertainty_radius=r * 0.1,
        )


# ══════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════

def _normalise(contour: np.ndarray) -> Optional[np.ndarray]:
    """Reshape to (N, 2) float64 and discard non-finite rows."""
    try:
        pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        return pts if len(pts) >= 1 else None
    except (ValueError, TypeError):
        return None


def _cv_ellipse_sane(e: Tuple) -> bool:
    """Quick sanity check on a cv2 RotatedRect ellipse."""
    (cx, cy), (d1, d2), angle = e
    if d1 <= 0 or d2 <= 0:
        return False
    a, b = max(d1, d2) / 2.0, min(d1, d2) / 2.0
    if a < 1.0:
        return False
    if b / a < 0.1:      # extremely elongated = degenerate
        return False
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False
    return True


def _ellipse_to_conic(
    cx: float, cy: float, a: float, b: float, angle_deg: float,
) -> np.ndarray:
    """Convert ellipse parameters to general conic coefficients [A..F].

    Conic: A x² + B xy + C y² + D x + E y + F = 0
    """
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    a2 = a * a
    b2 = b * b

    A = cos_t ** 2 / a2 + sin_t ** 2 / b2
    B = 2.0 * cos_t * sin_t * (1.0 / a2 - 1.0 / b2)
    C = sin_t ** 2 / a2 + cos_t ** 2 / b2
    D = -2.0 * A * cx - B * cy
    E = -B * cx - 2.0 * C * cy
    F = A * cx * cx + B * cx * cy + C * cy * cy - 1.0
    return np.array([A, B, C, D, E, F], dtype=np.float64)


def _sampson_distances(
    pts: np.ndarray, eparams: Tuple,
) -> Optional[np.ndarray]:
    """Sampson distance from each point to the ellipse."""
    (cx, cy), (d1, d2), angle = eparams
    a = max(d1, d2) / 2.0
    b = min(d1, d2) / 2.0
    if d2 > d1:
        angle = angle + 90.0

    conic = _ellipse_to_conic(cx, cy, a, b, angle)
    A, B, C, D, E, F = conic

    x = pts[:, 0]
    y = pts[:, 1]

    val = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    grad_x = 2.0 * A * x + B * y + D
    grad_y = B * x + 2.0 * C * y + E
    grad_sq = grad_x ** 2 + grad_y ** 2

    safe = grad_sq > 1e-12
    if not safe.any():
        return None

    dist = np.full(len(pts), 1e6)
    dist[safe] = np.sqrt(np.abs(val[safe] ** 2 / grad_sq[safe]))
    return dist


def _rms_distance(pts: np.ndarray, eparams: Tuple) -> float:
    d = _sampson_distances(pts, eparams)
    if d is None:
        return 999.0
    return float(np.sqrt(np.mean(d ** 2)))


def _quality_score(
    inlier_ratio: float,
    rms: float,
    eparams: Tuple,
    n_points: int,
) -> float:
    """Combined quality score in [0, 1]."""
    (_, _), (d1, d2), _ = eparams
    a = max(d1, d2) / 2.0
    b = min(d1, d2) / 2.0
    mean_r = (a + b) / 2.0

    # normalise rms relative to mean radius
    rms_norm = rms / max(mean_r, 1.0)
    rms_score = max(0.0, 1.0 - rms_norm / 0.15)

    # inlier score
    inlier_score = min(1.0, inlier_ratio / 0.80)

    # point density score (adaptive to radius)
    expected_pts = max(20, 2.0 * math.pi * mean_r * 0.3)
    pt_score = min(1.0, n_points / expected_pts)

    # circularity (moderate ellipticity is OK)
    circ = b / a if a > 0 else 0.0
    circ_score = min(1.0, circ / 0.5) if circ < 0.5 else 1.0

    return float(
        0.35 * rms_score
        + 0.30 * inlier_score
        + 0.20 * pt_score
        + 0.15 * circ_score
    )


def _circle_quality(rms: float, r: float, n: int) -> float:
    rms_norm = rms / max(r, 1.0)
    rms_score = max(0.0, 1.0 - rms_norm / 0.15)
    pt_score = min(1.0, n / 30.0)
    return float(0.6 * rms_score + 0.4 * pt_score)


def _build_result(
    eparams: Tuple,
    n_total: int,
    n_inliers: int,
    rms: float,
    quality: float,
    method: str,
) -> FitResult:
    """Convert a cv2 RotatedRect to a ``FitResult``."""
    (cx, cy), (d1, d2), angle = eparams
    a = max(d1, d2) / 2.0
    b = min(d1, d2) / 2.0
    if d2 > d1:
        angle = (angle + 90.0) % 180.0

    ecc = math.sqrt(max(0.0, 1.0 - (b / a) ** 2)) if a > 0 else 0.0
    mean_r = (a + b) / 2.0

    uc = max(0.5, rms * 1.5)
    ur = max(0.3, rms)

    return FitResult(
        valid=True,
        center_x=float(cx),
        center_y=float(cy),
        semi_major=float(a),
        semi_minor=float(b),
        angle_deg=float(angle) % 180.0,
        radius=float(mean_r),
        eccentricity=float(ecc),
        fit_quality_score=float(quality),
        rms_residual=float(rms),
        num_inliers=n_inliers,
        num_points=n_total,
        method=method,
        uncertainty_center=(uc, uc),
        uncertainty_radius=ur,
    )