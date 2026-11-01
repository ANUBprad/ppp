"""Iris-relative (normalized) coordinate mapping.

Converts between raw image-pixel coordinates and an iris-relative coordinate
system (angle, normalized radial distance). The normalized radial coordinate
is invariant to the absolute pixel size of the iris, which is essential for
the future pre-dock / post-dock correspondence stage: two images of the same
eye that differ in scale still place an anatomical point at the same
(angle, radial) location (up to the eye's torsional rotation, which is
precisely what a later phase measures).

No matching, rotation estimation, or registration is performed here.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from pupil_tracking.iris.types import IrisROI


def _ellipse_radius_at_angle(
    center_x: float,
    center_y: float,
    semi_major: float,
    semi_minor: float,
    ell_angle_deg: float,
    angle_deg: float,
) -> float:
    """Distance from ``(center_x, center_y)`` to the ellipse boundary along a
    given ray angle in the image frame.

    Uses the standard polar form of an ellipse rotated by ``ell_angle_deg``.
    """
    a = semi_major
    b = semi_minor
    if a <= 0 or b <= 0:
        return 0.0
    ang = math.radians(angle_deg)
    phi = math.radians(ell_angle_deg)
    cost = math.cos(ang - phi)
    sint = math.sin(ang - phi)
    denom = (cost / a) ** 2 + (sint / b) ** 2
    if denom <= 1e-12:
        return max(a, b)
    return 1.0 / math.sqrt(denom)


class IrisNormalizer:
    """Maps between image pixels and iris-relative coordinates.

    Parameters
    ----------
    radial_epsilon : float
        Small value added to the denominator and kept on the boundaries so the
        normalized radial coordinate does not divide by zero on exactly the
        pupil ellipse.
    """

    def __init__(self, radial_epsilon: float = 0.0) -> None:
        self.radial_epsilon = float(radial_epsilon)

    def radial_bounds(
        self,
        roi: IrisROI,
        angle_deg: float,
    ) -> Tuple[float, float]:
        """Return ``(inner_radius, outer_radius)`` at a given angle, in px.

        The inner radius is the pupil ellipse radius along ``angle_deg``,
        expanded by the inner inset; the outer radius is the limbus ellipse
        radius along that angle, contracted by the outer inset.
        """
        inner = _ellipse_radius_at_angle(
            roi.center_x, roi.center_y,
            roi.pupil_semi_major, roi.pupil_semi_minor, roi.pupil_angle_deg,
            angle_deg,
        ) * (1.0 + roi.inner_inset_frac)

        outer = _ellipse_radius_at_angle(
            roi.center_x, roi.center_y,
            roi.limbus_semi_major, roi.limbus_semi_minor, roi.limbus_angle_deg,
            angle_deg,
        ) * (1.0 - roi.outer_inset_frac)

        return inner, outer

    def to_iris_relative(
        self,
        x: float,
        y: float,
        roi: IrisROI,
    ) -> Optional[Tuple[float, float]]:
        """Map a pixel point to ``(angle_deg, radial_norm)``.

        ``radial_norm`` is in ``(0, 1]``: 0 at the inner (pupil-side) boundary,
        1 at the outer (limbus-side) boundary. Returns None when the ROI is
        invalid or the point is exactly at the centre.
        """
        if not roi.valid:
            return None

        # Check if centers are identical (concentric fast-path)
        pcx, pcy = roi.p_center_x, roi.p_center_y
        lcx, lcy = roi.l_center_x, roi.l_center_y
        if abs(pcx - lcx) < 1e-6 and abs(pcy - lcy) < 1e-6:
            dx = x - pcx
            dy = y - pcy
            dist = math.hypot(dx, dy)
            if dist <= 1e-9:
                return None
            angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0
            inner, outer = self.radial_bounds(roi, angle_deg)
            span = outer - inner
            if span <= 1e-9:
                return None
            radial_norm = (dist - inner) / span + self.radial_epsilon
            radial_norm = min(max(radial_norm, self.radial_epsilon), 1.0)
            return angle_deg, radial_norm

        # Non-concentric: find angle and radial fraction
        dx = x - pcx
        dy = y - pcy
        if math.hypot(dx, dy) <= 1e-9:
            return None
        ang_deg = math.degrees(math.atan2(dy, dx)) % 360.0
        rn = 0.5

        for _ in range(3):
            inner, outer = self.radial_bounds(roi, ang_deg)
            ang_rad = math.radians(ang_deg)
            cos_a = math.cos(ang_rad)
            sin_a = math.sin(ang_rad)
            xp = pcx + inner * cos_a
            yp = pcy + inner * sin_a
            xl = lcx + outer * cos_a
            yl = lcy + outer * sin_a

            vx = xl - xp
            vy = yl - yp
            vlen_sq = vx * vx + vy * vy
            if vlen_sq <= 1e-9:
                return None
            rn = ((x - xp) * vx + (y - yp) * vy) / vlen_sq
            rn = min(max(rn, 0.0), 1.0)

            # Update center along interpolation and re-evaluate angle
            cx_interp = (1.0 - rn) * pcx + rn * lcx
            cy_interp = (1.0 - rn) * pcy + rn * lcy
            ang_deg = math.degrees(math.atan2(y - cy_interp, x - cx_interp)) % 360.0

        inner, outer = self.radial_bounds(roi, ang_deg)
        span = outer - inner
        if span <= 1e-9:
            return None
        radial_norm = rn + self.radial_epsilon
        radial_norm = min(max(radial_norm, self.radial_epsilon), 1.0)
        return ang_deg, radial_norm

    def from_iris_relative(
        self,
        angle_deg: float,
        radial_norm: float,
        roi: IrisROI,
    ) -> Optional[Tuple[float, float]]:
        """Inverse: map iris-relative coordinates back to a pixel point.

        ``radial_norm=0`` gives a point on the inner (pupil-side) boundary;
        ``radial_norm=1`` gives a point on the outer (limbus-side) boundary.
        """
        if not roi.valid:
            return None
        rn = min(max(float(radial_norm), 0.0), 1.0)
        inner, outer = self.radial_bounds(roi, float(angle_deg))
        ang = math.radians(float(angle_deg))
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        xp = roi.p_center_x + inner * cos_a
        yp = roi.p_center_y + inner * sin_a
        xl = roi.l_center_x + outer * cos_a
        yl = roi.l_center_y + outer * sin_a
        x = (1.0 - rn) * xp + rn * xl
        y = (1.0 - rn) * yp + rn * yl
        return x, y
