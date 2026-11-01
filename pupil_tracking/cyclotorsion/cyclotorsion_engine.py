"""cyclotorsion_engine.py — torsional (cyclotorsion) angle estimation.

Module 4 of the cyclotorsion feature.  Given two polar iris strips of the
*same* eye (typically produced by :mod:`pupil_tracking.cyclotorsion.polar_unwrap`),
this module estimates the signed angular displacement between them by
maximising the normalised cross-correlation (NCC) over integer column shifts,
then refining the peak to sub-pixel accuracy with a parabolic fit.

Sign convention (important)
---------------------------
The angle is defined on the **strip's angular axis**:

    ``angle_deg`` is the number of degrees the *reference* strip must be
    rolled toward increasing strip angle (:func:`polar_unwrap.roll_strip`)
    so that it aligns with the *target* strip::

        roll_strip(reference, angle_deg / 360 * width)  ~=  target

    Positive ``angle_deg`` therefore means the target's content sits at a
    larger strip angle than the reference's.

Because :func:`polar_unwrap.column_to_angle` measures image angle as
``atan2(dy, dx)`` (``+x`` axis, increasing downward in image coordinates), an
image rotated by ``+theta`` with ``cv2.getRotationMatrix2D`` produces a strip
that this engine reports as ``-theta``.  The raw column shift is also returned
(:attr:`CyclotorsionResult.shift_cols`) so callers can re-interpret the sign
for their own clinical convention.

Scope / honesty
---------------
This is a classical, deterministic estimator.  It makes **no clinical
accuracy claim**: ``min_score`` and ``min_valid_fraction`` are conservative
placeholders to be validated against real paired data.  It is *additive* and
is never wired into the centration / pupil / limbus pipeline; a returned angle
must not be injected into surgical control without independent validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.cyclotorsion import polar_unwrap as _pu


# Conservative default gates (placeholders, to be validated on real pairs).
DEFAULT_MIN_SCORE = 0.25
DEFAULT_MIN_VALID_FRACTION = 0.20
DEFAULT_STRIP_SIZE: Tuple[int, int] = (_pu.DEFAULT_STRIP_HEIGHT, _pu.DEFAULT_STRIP_WIDTH)


@dataclass
class CyclotorsionResult:
    """Outcome of a cyclotorsion estimate.

    Attributes
    ----------
    angle_deg:
        Signed torsional angle (degrees) per the module sign convention.
    shift_cols:
        Sub-pixel column shift applied to the reference to align it with the
        target (``shift_cols == angle_deg / 360 * width``).
    score:
        NCC at the best alignment, in ``[-1, 1]``.
    valid:
        Whether the estimate passed the validity gates.
    reason:
        Human-readable status: ``"ok"`` or the first failing gate.
    n_valid, n_total:
        Number of comparable pixels at the best alignment and total pixels.
    search_range_deg:
        Half-width of the searched angular window (degrees).
    """

    angle_deg: float = 0.0
    shift_cols: float = 0.0
    score: float = 0.0
    valid: bool = False
    reason: str = ""
    n_valid: int = 0
    n_total: int = 0
    search_range_deg: float = 180.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "angle_deg": float(self.angle_deg),
            "shift_cols": float(self.shift_cols),
            "score": float(self.score),
            "valid": bool(self.valid),
            "reason": self.reason,
            "n_valid": int(self.n_valid),
            "n_total": int(self.n_total),
            "search_range_deg": float(self.search_range_deg),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_luminance(strip: np.ndarray) -> np.ndarray:
    """Coerce a gray / BGR / BGRA strip to a ``float32`` luminance image."""
    a = np.asarray(strip)
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        channels = a.shape[2]
        if channels == 3:
            return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if channels == 4:
            return cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        # Unknown channel count: average the first three / all channels.
        return a[..., :3].mean(axis=2).astype(np.float32) if channels >= 3 \
            else a.mean(axis=2).astype(np.float32)
    raise ValueError(f"strip must be 2-D or 3-D, got ndim={a.ndim}")


def _coerce_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return a boolean mask of ``shape``; None stays None."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    return m > 0


def _preprocess(
    strip: np.ndarray,
    mask: Optional[np.ndarray],
    blur_sigma: float,
    radial_normalize: bool,
) -> np.ndarray:
    """Luminance + optional blur + rotation-invariant radial normalisation.

    Subtracting each row's (valid) mean removes the shared radial illumination
    profile without changing the strip's angular content, so the subsequent
    NCC is dominated by angular structure rather than brightness gradient.
    """
    lum = _to_luminance(strip)
    if blur_sigma and blur_sigma > 0.0:
        lum = cv2.GaussianBlur(lum, (0, 0), float(blur_sigma),
                               borderType=cv2.BORDER_REPLICATE)
    if radial_normalize:
        if mask is not None:
            m = mask.astype(np.float32)
            denom = m.sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            row_mean = (lum * m).sum(axis=1, keepdims=True) / denom
        else:
            row_mean = lum.mean(axis=1, keepdims=True)
        lum = lum - row_mean
    return lum


def _masked_pearson(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, int]:
    """Pearson correlation of ``a`` and ``b`` over the ``valid`` pixels."""
    n = int(valid.sum())
    if n < 2:
        return -1.0, n
    av = a[valid]
    bv = b[valid]
    av = av - av.mean()
    bv = bv - bv.mean()
    da = float(np.dot(av, av))
    db = float(np.dot(bv, bv))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0, n
    return float(np.dot(av, bv) / math.sqrt(da * db)), n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_cyclotorsion(
    reference_strip: np.ndarray,
    target_strip: np.ndarray,
    *,
    reference_mask: Optional[np.ndarray] = None,
    target_mask: Optional[np.ndarray] = None,
    angle_range_deg: float = 180.0,
    blur_sigma: float = 0.0,
    radial_normalize: bool = True,
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    min_score: float = DEFAULT_MIN_SCORE,
) -> CyclotorsionResult:
    """Estimate the signed torsional angle between two iris strips.

    Parameters
    ----------
    reference_strip, target_strip:
        Polar strips of equal ``(height, width)``.  Gray, BGR or BGRA accepted.
    reference_mask, target_mask:
        Optional bool/0-1 masks (same shape) marking *valid* pixels.  Occluded
        pixels (eyelid, lash, glare) should be False so they do not bias the
        correlation.  The two masks may differ.
    angle_range_deg:
        Half-width of the searched window.  ``180`` (default) searches the full
        circle; a smaller value (e.g. ``20``) restricts the search, which is
        faster and avoids locking onto a periodic alias.
    blur_sigma:
        Gaussian pre-blur sigma (0 = none) to suppress sensor noise.
    radial_normalize:
        Subtract each row's mean before correlating (see :func:`_preprocess`).
    min_valid_fraction, min_score:
        Validity gates.  A result that fails a gate is returned with
        ``valid=False`` and the reason set; ``angle_deg`` is still populated so
        callers can inspect it.

    Returns
    -------
    CyclotorsionResult
    """
    res = CyclotorsionResult(search_range_deg=float(abs(angle_range_deg)))
    if reference_strip is None or target_strip is None:
        res.reason = "missing strip"
        return res

    ra = np.asarray(reference_strip)
    ta = np.asarray(target_strip)
    if ra.ndim not in (2, 3) or ta.ndim not in (2, 3):
        res.reason = "strip must be 2-D or 3-D"
        return res
    if ra.shape[:2] != ta.shape[:2]:
        res.reason = f"strip size mismatch {ra.shape[:2]} vs {ta.shape[:2]}"
        return res

    height, width = ra.shape[:2]
    res.n_total = int(height * width)
    if width < 8 or height < 2:
        res.reason = "strip too small"
        return res

    rmask = _coerce_mask(reference_mask, (height, width))
    tmask = _coerce_mask(target_mask, (height, width))
    if rmask is not None and rmask.shape[:2] != (height, width):
        res.reason = "reference mask size mismatch"
        return res
    if tmask is not None and tmask.shape[:2] != (height, width):
        res.reason = "target mask size mismatch"
        return res
    if rmask is None:
        rmask = np.ones((height, width), dtype=bool)
    if tmask is None:
        tmask = np.ones((height, width), dtype=bool)

    a = _preprocess(ra, rmask, blur_sigma, radial_normalize)
    b = _preprocess(ta, tmask, blur_sigma, radial_normalize)

    # Search window of (signed) integer column shifts.
    if angle_range_deg is None or abs(angle_range_deg) >= 180.0:
        max_shift = width // 2
        res.search_range_deg = 180.0
    else:
        max_shift = int(round(abs(float(angle_range_deg)) / 360.0 * width))
        max_shift = max(1, min(max_shift, width // 2))

    corr: Dict[int, float] = {}
    best_shift = 0
    best_corr = -2.0
    best_valid_n = 0

    for s in range(-max_shift, max_shift + 1):
        rolled_b = np.roll(b, s, axis=1)
        valid = rmask & np.roll(tmask, s, axis=1)
        c, n = _masked_pearson(a, rolled_b, valid)
        corr[s] = c
        if c > best_corr:
            best_corr = c
            best_shift = s
            best_valid_n = n

    # Sub-pixel refinement with a parabolic fit around the NCC peak.
    shift_sub = float(best_shift)
    y0 = corr.get(best_shift - 1)
    y2 = corr.get(best_shift + 1)
    if y0 is not None and y2 is not None:
        denom = y0 - 2.0 * best_corr + y2
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-0.5, min(0.5, delta))
            shift_sub = best_shift + delta

    # Convert the (roll-target-to-match-reference) shift into the signed angle
    # applied to the reference (see module docstring).
    res.shift_cols = -shift_sub
    angle = -shift_sub / width * 360.0
    res.angle_deg = ((angle + 180.0) % 360.0) - 180.0
    res.score = best_corr
    res.n_valid = int(best_valid_n)

    if best_valid_n < 2:
        res.reason = "insufficient valid pixels"
        return res
    valid_fraction = best_valid_n / float(res.n_total)
    if valid_fraction < min_valid_fraction:
        res.reason = f"valid fraction {valid_fraction:.2f} < {min_valid_fraction:.2f}"
        return res
    if best_corr < min_score:
        res.reason = f"low correlation ({best_corr:.2f} < {min_score:.2f})"
        return res

    res.valid = True
    res.reason = "ok"
    return res


def align_strip(strip: np.ndarray, angle_deg: float) -> np.ndarray:
    """Roll ``strip`` by ``angle_deg`` on its angular axis (cyclic).

    This is the operation described in the module sign convention, i.e.
    ``align_strip(reference, result.angle_deg)`` approximately returns the
    target strip.
    """
    width = strip.shape[1]
    return _pu.roll_strip(strip, angle_deg / 360.0 * width)


def estimate_from_images(
    reference_image: np.ndarray,
    target_image: np.ndarray,
    roi,
    *,
    output_size: Tuple[int, int] = DEFAULT_STRIP_SIZE,
    angle_offset_deg: float = 0.0,
    **kwargs,
) -> CyclotorsionResult:
    """End-to-end helper: unwrap two images with a shared ROI, then estimate.

    Returns an invalid result (with ``reason``) when either unwrap fails.  Any
    extra keyword arguments are forwarded to :func:`estimate_cyclotorsion`.
    """
    ref_strip = _pu.unwrap_iris(reference_image, roi, output_size, angle_offset_deg)
    tgt_strip = _pu.unwrap_iris(target_image, roi, output_size, angle_offset_deg)
    if ref_strip is None or tgt_strip is None:
        res = CyclotorsionResult()
        res.reason = "iris unwrap failed (invalid ROI)"
        return res
    return estimate_cyclotorsion(ref_strip, tgt_strip, **kwargs)
