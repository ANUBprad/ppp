"""
Smoothed state writer for EyeDetectionResult.

Writes Kalman-filtered values back into typed detection results,
recalculating derived measurements (mm values, corneal centre)
to maintain consistency after smoothing.

Key correction: Corneal centre = limbus centre (anatomical definition),
NOT the midpoint of pupil and limbus centres.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pupil_tracking.utils.types import (
        EyeDetectionResult,
        CalibrationInfo,
        CornealCenterResult,
        PupilDetection,
        LimbusDetection,
        EllipseParams,
    )

# Import CornealCenterResult at runtime for the calculator class
from pupil_tracking.utils.types import CornealCenterResult


class SmoothedStateWriter:
    """Writes Kalman-smoothed state back into EyeDetectionResult objects.

    After the Kalman filter produces smoothed pixel coordinates and radii,
    this class updates the typed result in-place — recalculating all
    derived quantities (mm values, corneal centre, offset) so downstream
    consumers see a fully consistent result.

    Corneal Centre Definition
    -------------------------
    The corneal centre is the geometric centre of the limbus
    (iris-sclera junction).  The pupil-limbus offset is the
    displacement from corneal centre to pupil centre.

    - Normal offset  : 0.1 – 0.5 mm (nasal, slightly inferior)
    - Warning        : > WARN_OFFSET_MM
    - Critical       : > CRITICAL_OFFSET_MM  (> ~0.8 mm)
    """

    @staticmethod
    def apply_smoothed_dict(
        result: "EyeDetectionResult",
        smoothed: Dict[str, Any],
    ) -> "EyeDetectionResult":
        """Write Kalman-smoothed values back into a typed result.

        Recalculates mm measurements and corneal centre after updating
        pixel coordinates.  Modifies *result* in-place and returns it.

        Parameters
        ----------
        result : EyeDetectionResult
            The detection result to update with smoothed values.
        smoothed : Dict[str, Any]
            Dictionary from the Kalman filter containing smoothed state.
            Expected keys::

                pupil_detected  : bool
                pupil_x         : float   (centre x, pixels)
                pupil_y         : float   (centre y, pixels)
                pupil_r         : float   (radius, pixels)

                limbus_detected : bool
                limbus_x        : float
                limbus_y        : float
                limbus_r        : float

        Returns
        -------
        EyeDetectionResult
            The same *result* object, modified in-place.

        Notes
        -----
        *   ``radius`` on ``EllipseParams`` is a read-only property
            computed as ``(semi_major + semi_minor) / 2``.  To change
            the effective radius we scale both semi-axes proportionally.

        *   Corneal centre is defined as the **limbus centre**, not
            the midpoint of pupil and limbus.  This matches the
            ``CornealCenterCalculator.calculate()`` implementation
            and the anatomical definition used in ophthalmic surgery.

        *   Offset angles are **not** normalised to [0, 360) — they
            preserve the signed output of ``math.atan2`` converted to
            degrees, exactly matching the reference calculator.
        """

        # ────────────────────────────────────────────────────────
        # 1.  UPDATE PUPIL ELLIPSE
        # ────────────────────────────────────────────────────────
        if smoothed.get("pupil_detected") and result.pupil.ellipse is not None:
            ell = result.pupil.ellipse

            # Update centre position
            ell.center_x = float(smoothed.get("pupil_x", ell.center_x))
            ell.center_y = float(smoothed.get("pupil_y", ell.center_y))

            # Update radius via proportional semi-axis scaling
            # radius is read-only = (semi_major + semi_minor) / 2
            target_r = float(smoothed.get("pupil_r", ell.radius))
            current_r = ell.radius

            if current_r > 1e-6:
                scale = target_r / current_r
                ell.semi_major *= scale
                ell.semi_minor *= scale
            else:
                # Degenerate ellipse — set both axes to target
                ell.semi_major = target_r
                ell.semi_minor = target_r

        # ────────────────────────────────────────────────────────
        # 2.  UPDATE LIMBUS ELLIPSE
        # ────────────────────────────────────────────────────────
        if smoothed.get("limbus_detected") and result.limbus.ellipse is not None:
            ell = result.limbus.ellipse

            # Update centre position
            ell.center_x = float(smoothed.get("limbus_x", ell.center_x))
            ell.center_y = float(smoothed.get("limbus_y", ell.center_y))

            # Update radius via proportional semi-axis scaling
            target_r = float(smoothed.get("limbus_r", ell.radius))
            current_r = ell.radius

            if current_r > 1e-6:
                scale = target_r / current_r
                ell.semi_major *= scale
                ell.semi_minor *= scale
            else:
                ell.semi_major = target_r
                ell.semi_minor = target_r

        # ────────────────────────────────────────────────────────
        # 3.  RECALCULATE MM VALUES (requires calibration)
        # ────────────────────────────────────────────────────────
        cal = result.calibration

        if cal is not None and cal.calibrated:

            # ── pupil mm ────────────────────────────────────────
            if result.pupil.detected and result.pupil.ellipse is not None:
                p_ell = result.pupil.ellipse

                # Radius in mm (scalar conversion)
                result.pupil.radius_mm = cal.px_to_mm(p_ell.radius)

                # Centre in mm (point conversion — accounts for origin)
                result.pupil.center_mm = cal.point_px_to_mm(p_ell.center)

                # Diameter in mm (convenience field, if present)
                if hasattr(result.pupil, "diameter_mm"):
                    result.pupil.diameter_mm = result.pupil.radius_mm * 2.0

            # ── limbus mm ───────────────────────────────────────
            if result.limbus.detected and result.limbus.ellipse is not None:
                l_ell = result.limbus.ellipse

                # Radius in mm
                result.limbus.radius_mm = cal.px_to_mm(l_ell.radius)

                # Centre in mm
                result.limbus.center_mm = cal.point_px_to_mm(l_ell.center)

                # Diameter in mm
                if hasattr(result.limbus, "diameter_mm"):
                    result.limbus.diameter_mm = result.limbus.radius_mm * 2.0

        # ────────────────────────────────────────────────────────
        # 4.  RECALCULATE CORNEAL CENTRE AND OFFSET
        # ────────────────────────────────────────────────────────
        #
        # Anatomical definition (from CornealCenterCalculator):
        #   - Corneal centre  = limbus centre
        #   - Offset          = pupil centre − limbus centre
        #   - Offset angle    = atan2(dy, dx) in degrees (signed)
        #
        # The offset is clinically significant for:
        #   - IOL centration in cataract surgery
        #   - LASIK flap placement
        #   - Diagnosis of angle kappa abnormalities
        #
        if result.has_both:
            p = result.pupil.ellipse
            l = result.limbus.ellipse

            # ── corneal centre = limbus centre ──────────────────
            cx = l.center_x
            cy = l.center_y

            # ── offset = pupil − limbus ─────────────────────────
            ox = p.center_x - l.center_x
            oy = p.center_y - l.center_y

            # Offset magnitude (Euclidean distance in pixels)
            mag_px = math.sqrt(ox * ox + oy * oy)

            # Offset angle (signed degrees, no normalisation)
            ang_deg = math.degrees(math.atan2(oy, ox))

            # ── write to corneal centre result ──────────────────
            cc = result.corneal_center

            cc.valid = True
            cc.center_px = (cx, cy)
            cc.offset_px = (ox, oy)
            cc.offset_magnitude_px = mag_px
            cc.offset_angle_deg = ang_deg

            # ── confidence from constituent detections ──────────
            # Matches CornealCenterCalculator logic:
            #   base = min(pupil_conf, limbus_conf) * 0.8
            #   penalty if offset_ratio > 0.2
            if hasattr(cc, "confidence"):
                base_conf = min(
                    result.pupil.confidence,
                    result.limbus.confidence,
                ) * 0.8

                if l.radius > 1e-6:
                    offset_ratio = mag_px / l.radius
                    if offset_ratio > 0.2:
                        base_conf *= max(
                            0.3,
                            1.0 - (offset_ratio - 0.2) * 2.0,
                        )

                cc.confidence = float(np.clip(base_conf, 0.0, 1.0))

            # ── mm conversion (requires calibration) ────────────
            if cal is not None and cal.calibrated:

                # Centre in mm
                cc.center_mm = cal.point_px_to_mm((cx, cy))

                # Offset in mm (direct multiplication, NOT px_to_mm)
                # This matches CornealCenterCalculator:
                #   dx_mm = dx * cal.mm_per_px
                #   dy_mm = dy * cal.mm_per_px
                dx_mm = ox * cal.mm_per_px
                dy_mm = oy * cal.mm_per_px

                cc.offset_mm = (dx_mm, dy_mm)

                # Offset magnitude in mm (from mm components)
                cc.offset_magnitude_mm = math.sqrt(
                    dx_mm * dx_mm + dy_mm * dy_mm
                )

        return result


# ════════════════════════════════════════════════════════════════
# Convenience function (module-level access)
# ════════════════════════════════════════════════════════════════

def apply_smoothed_dict(
    result: "EyeDetectionResult",
    smoothed: Dict[str, Any],
) -> "EyeDetectionResult":
    """Module-level convenience wrapper.

    Delegates to ``SmoothedStateWriter.apply_smoothed_dict()``.

    Parameters
    ----------
    result : EyeDetectionResult
        Detection result to update in-place.
    smoothed : Dict[str, Any]
        Kalman-smoothed state dictionary.

    Returns
    -------
    EyeDetectionResult
        The modified result (same object as input).
    """
    return SmoothedStateWriter.apply_smoothed_dict(result, smoothed)


# ════════════════════════════════════════════════════════════════
# Corneal Center Calculator (minimal wrapper)
# ════════════════════════════════════════════════════════════════

class CornealCenterCalculator:
    """Minimal wrapper for corneal center calculation.
    
    Computes corneal centre as the limbus centre, and calculates
    the pupil-limbus offset. Works with EyeDetectionResult objects
    containing pupil and limbus detections.
    
    Parameters
    ----------
    config : Any, optional
        Configuration object (not currently used, for API compatibility).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def calculate(
        self,
        pupil: Any,  # PupilDetection
        limbus: Any,  # LimbusDetection
        calibration: "CalibrationInfo",
    ) -> "CornealCenterResult":
        """Calculate corneal centre and pupil-limbus offset.

        The corneal centre is defined as the limbus centre (anatomical
        definition). The offset is the displacement from corneal centre
        to pupil centre.

        Parameters
        ----------
        pupil : PupilDetection
            Pupil detection result with ellipse geometry.
        limbus : LimbusDetection
            Limbus detection result with ellipse geometry.
        calibration : CalibrationInfo
            Pixel-to-mm calibration information.

        Returns
        -------
        CornealCenterResult
            Computed corneal centre, offset, and confidence.
        """
        result = CornealCenterResult()

        # Check if both detections are present
        if not (pupil.detected and limbus.detected and 
                pupil.ellipse is not None and limbus.ellipse is not None):
            result.valid = False
            return result

        p_ell = pupil.ellipse
        l_ell = limbus.ellipse

        # Corneal centre = limbus centre (anatomical definition)
        cx = l_ell.center_x
        cy = l_ell.center_y
        result.center_px = (cx, cy)

        # Offset = pupil centre - limbus centre
        ox = p_ell.center_x - l_ell.center_x
        oy = p_ell.center_y - l_ell.center_y
        result.offset_px = (ox, oy)

        # Offset magnitude in pixels
        result.offset_magnitude_px = math.sqrt(ox * ox + oy * oy)

        # Offset angle in degrees (signed)
        result.offset_angle_deg = math.degrees(math.atan2(oy, ox))

        # Confidence based on constituent detections
        base_conf = min(pupil.confidence, limbus.confidence) * 0.8

        # Apply penalty if offset exceeds 20% of limbus radius
        if l_ell.radius > 1e-6:
            offset_ratio = result.offset_magnitude_px / l_ell.radius
            if offset_ratio > 0.2:
                base_conf *= max(
                    0.3,
                    1.0 - (offset_ratio - 0.2) * 2.0,
                )

        result.confidence = float(np.clip(base_conf, 0.0, 1.0))
        result.valid = True

        # Convert to mm if calibration is available
        if calibration is not None and calibration.calibrated:
            result.center_mm = calibration.point_px_to_mm((cx, cy))
            
            # Offset in mm (direct multiplication, not px_to_mm point conversion)
            dx_mm = ox * calibration.mm_per_px
            dy_mm = oy * calibration.mm_per_px
            result.offset_mm = (dx_mm, dy_mm)
            result.offset_magnitude_mm = math.sqrt(
                dx_mm * dx_mm + dy_mm * dy_mm
            )

        return result

    def reset(self) -> None:
        """Reset calculator state.
        
        This is a no-op in the current implementation but is kept
        for API compatibility with previous versions.
        """
        pass