"""pair_readout.py — display-only OD/OS cyclotorsion readout helper.

This is the thin orchestration layer behind the Add Patient dialog's Pentacam
preview readout.  Given a *current* image, a *reference* image for the **same
eye**, and the normalised iris-ROI **box** drawn in the dialog, it loads both
images, builds an (approximate) iris annulus from the box, unwraps each into a
polar strip and estimates the signed torsional angle between them.

Pipeline reused (all additive cyclotorsion modules — nothing else is touched)::

    pentacam_loader  -> load / auto-flip / box -> IrisROI
    polar_unwrap     -> iris annulus -> polar strip
    cyclotorsion_engine -> signed angle from strip correlation

Safety / honesty
----------------
This is a **display-only** convenience.  It never modifies the source images,
never runs the centration / pupil / limbus pipeline, and its output must not be
injected into surgical control without independent validation.  The estimated
angle is the rotation that aligns the *reference* strip with the *current*
strip (see :mod:`cyclotorsion_engine` for the exact sign convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from pupil_tracking.cyclotorsion.pentacam_loader import (
    DEFAULT_PUPIL_LIMBUS_RATIO,
    flip_pentacam,
    load_pentacam_image,
    roi_norm_to_iris_roi,
)


#: Strip size handed to the unwrap / correlation (matches the engine default).
DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (64, 512)


@dataclass
class PairReadout:
    """Result of one OD/OS cyclotorsion pair readout (display-only)."""

    valid: bool = False
    angle_deg: float = 0.0
    score: float = 0.0
    reason: str = ""
    n_valid: int = 0
    n_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "angle_deg": float(self.angle_deg),
            "score": float(self.score),
            "reason": self.reason,
            "n_valid": int(self.n_valid),
            "n_total": int(self.n_total),
        }


def estimate_pair_readout(
    current_path: str,
    reference_path: str,
    roi_norm: Optional[Tuple[float, float, float, float]],
    *,
    flip: bool = True,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    angle_offset_deg: float = 0.0,
    pupil_ratio: float = DEFAULT_PUPIL_LIMBUS_RATIO,
    **engine_kwargs: Any,
) -> PairReadout:
    """Estimate the cyclotorsion between a reference and a current image.

    Parameters
    ----------
    current_path, reference_path:
        Pentacam image paths (JPG/PNG/PDF/DICOM) for the **same** eye.
    roi_norm:
        Normalised iris-ROI box ``(x0, y0, x1, y1)`` in flipped-image space,
        as drawn in the Add PX dialog.  It is reused for both images.
    flip:
        Apply the Pentacam auto-flip to both images (default True).
    output_size:
        ``(height, width)`` of the polar strips.
    angle_offset_deg:
        Angular offset of the unwrap grid (forwarded to the unwrap).
    pupil_ratio:
        Assumed pupil/limbus ratio for the approximate box ROI.
    **engine_kwargs:
        Forwarded to :func:`cyclotorsion_engine.estimate_cyclotorsion`
        (e.g. ``angle_range_deg``, ``min_score``).

    Returns
    -------
    PairReadout
        ``valid=False`` with a ``reason`` when any step fails; ``angle_deg`` is
        still populated when the engine produced an estimate.
    """
    out = PairReadout()
    if not current_path or not reference_path:
        out.reason = "missing image path"
        return out
    if roi_norm is None:
        out.reason = "no iris ROI"
        return out

    # Heavy imports are kept local so importing this module for typing stays
    # cheap and the GUI only pays for them when a readout is requested.
    from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_iris
    from pupil_tracking.cyclotorsion.cyclotorsion_engine import (
        estimate_cyclotorsion,
    )

    img_cur = load_pentacam_image(current_path)
    img_ref = load_pentacam_image(reference_path)
    if img_cur is None or img_ref is None:
        out.reason = "unreadable image"
        return out
    if flip:
        img_cur = flip_pentacam(img_cur)
        img_ref = flip_pentacam(img_ref)

    if img_cur.ndim < 2 or img_ref.ndim < 2:
        out.reason = "invalid image"
        return out

    roi_cur = roi_norm_to_iris_roi(
        roi_norm, img_cur.shape[1], img_cur.shape[0], pupil_ratio=pupil_ratio
    )
    roi_ref = roi_norm_to_iris_roi(
        roi_norm, img_ref.shape[1], img_ref.shape[0], pupil_ratio=pupil_ratio
    )
    if not roi_cur.valid or not roi_ref.valid:
        out.reason = "invalid ROI box"
        return out

    strip_cur = unwrap_iris(img_cur, roi_cur, output_size, angle_offset_deg)
    strip_ref = unwrap_iris(img_ref, roi_ref, output_size, angle_offset_deg)
    if strip_cur is None or strip_ref is None:
        out.reason = "iris unwrap failed"
        return out

    result = estimate_cyclotorsion(strip_ref, strip_cur, **engine_kwargs)
    out.valid = bool(result.valid)
    out.angle_deg = float(result.angle_deg)
    out.score = float(result.score)
    out.n_valid = int(result.n_valid)
    out.n_total = int(result.n_total)
    out.reason = result.reason or ("ok" if result.valid else "no estimate")
    return out


__all__ = [
    "DEFAULT_OUTPUT_SIZE",
    "PairReadout",
    "estimate_pair_readout",
]
