"""iris_features.py — iris features on the unwrapped iris strip.

Module 3 of the cyclotorsion feature.  It consumes a *polar strip* (the
rubber-sheet iris image produced by :mod:`pupil_tracking.cyclotorsion.polar_unwrap`)
and produces two complementary, rotation-aware representations:

1. an **angular profile / signature** — a 1-D signal along the strip's angle
   axis, obtained by reducing the radial axis.  Rolling the strip cyclically
   rolls the profile by the same amount, which makes it a cheap primitive for
   angular (torsion) comparison; and
2. a set of **strip features** — locally-distinctive iris landmarks with an
   angular position, a normalised radial position, a response/confidence, and
   a deterministic local descriptor.

Everything here is defined on the *strip's own angular axis*: column ``c`` of a
strip of width ``W`` corresponds to angle ``c / W * 360`` degrees.  Callers that
unwrapped with a non-zero ``angle_offset_deg`` should add that offset back (the
:func:`extract_features_from_image` helper does this for them).

Rotation behaviour (the whole point)
------------------------------------
Both the profile/signature and the *descriptors* are rotation-equivariant /
invariant in the strip's cyclic angle axis:

* ``angular_profile(np.roll(strip, k, axis=1)) == np.roll(angular_profile(strip), k)``
  (exactly, for the unmasked mean reducer), and
* features move by ``k`` columns while their local descriptor is unchanged,
  because each descriptor is built from a column-cyclic patch surrounding the
  feature.

Scope / honesty
---------------
This is a classical, deterministic extractor.  It makes **no clinical accuracy
claim**: the response/contrast thresholds are conservative placeholders to be
validated against real paired data.  It is *additive*: it never modifies the
source image or geometry and is never wired into the centration / pupil /
limbus pipeline.  A rotation estimate derived from these features must not be
injected into surgical control without independent validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants (placeholders to be validated on real data)
# ---------------------------------------------------------------------------

#: Number of intensity-histogram bins in a feature descriptor.
DEFAULT_DESCRIPTOR_BINS = 16

#: Descriptor patch half-window: rows (radial) and columns (angular).
DEFAULT_PATCH_ROWS = 3
DEFAULT_PATCH_COLS = 6

#: Centre-surround blur sigma (strip pixels) used to build the response map.
DEFAULT_RESPONSE_SIGMA = 1.5

#: Minimum response to consider a pixel a candidate, as a fraction of the
#: strip's dynamic range (so the gate is invariant to 0-1 vs 0-255 inputs).
DEFAULT_MIN_RESPONSE_FRAC = 0.05

#: Angular suppression for feature selection (degrees, circular).
DEFAULT_MIN_ANGULAR_SEP_DEG = 6.0

#: Radial suppression for feature selection (normalised radial units, 0..1).
DEFAULT_MIN_RADIAL_SEP = 0.25

#: Status strings (mirrors the iris package's simple status vocabulary).
STATUS_OK = "OK"
STATUS_NO_FEATURES = "NO_FEATURES"
STATUS_EMPTY = "EMPTY"


class StripFeatureType(Enum):
    """Coarse anatomical category of a strip feature (heuristic)."""
    TEXTURE = "TEXTURE"   # generic local texture
    CRYPT = "CRYPT"       # dark, locally-contrasty pit
    FURROW = "FURROW"     # elongated (more angular than radial) structure
    UNKNOWN = "UNKNOWN"


@dataclass
class StripFeature:
    """One detected iris feature on the unwrapped strip.

    ``column`` / ``row`` are strip coordinates; ``angle_deg`` is the strip's
    angular coordinate (``column / width * 360``) and ``radial_norm`` is the
    strip's normalised radial coordinate (``row / (height - 1)``: 0 = pupil
    boundary, 1 = limbus boundary).
    """
    id: int = -1
    column: int = 0
    row: int = 0
    angle_deg: float = 0.0
    radial_norm: float = 0.0

    response: float = 0.0
    local_contrast: float = 0.0
    confidence: float = 0.0
    feature_type: StripFeatureType = StripFeatureType.TEXTURE

    descriptor: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "column": int(self.column),
            "row": int(self.row),
            "angle_deg": float(self.angle_deg),
            "radial_norm": float(self.radial_norm),
            "response": float(self.response),
            "local_contrast": float(self.local_contrast),
            "confidence": float(self.confidence),
            "feature_type": self.feature_type.value,
            "descriptor_len": (
                int(len(self.descriptor)) if self.descriptor is not None else 0
            ),
        }


@dataclass
class IrisStripFeatureSet:
    """All features extracted from one strip, plus the angular profile."""
    height: int = 0
    width: int = 0

    features: List[StripFeature] = field(default_factory=list)
    angular_profile: Optional[np.ndarray] = None

    num_candidates: int = 0
    num_accepted: int = 0
    status: str = STATUS_EMPTY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height": int(self.height),
            "width": int(self.width),
            "num_candidates": int(self.num_candidates),
            "num_accepted": int(self.num_accepted),
            "status": self.status,
            "features": [f.to_dict() for f in self.features],
            "has_profile": self.angular_profile is not None,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_luminance(strip: np.ndarray) -> np.ndarray:
    """Coerce a gray / BGR / BGRA strip to a 2-D ``float32`` luminance image."""
    a = np.asarray(strip)
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        channels = a.shape[2]
        if channels == 3:
            return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if channels == 4:
            return cv2.cvtColor(a, cv2.COLOR_BGRA2GRAY).astype(np.float32)
        if channels >= 3:
            return a[..., :3].mean(axis=2).astype(np.float32)
        return a.mean(axis=2).astype(np.float32)
    raise ValueError(f"strip must be 2-D or 3-D, got ndim={a.ndim}")


def _coerce_mask(mask: Optional[np.ndarray], shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return a boolean mask of ``shape``; ``None`` stays ``None``."""
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2:
        return None
    if m.shape[:2] != shape:
        return None
    return m.astype(bool) if m.dtype != bool else m


def _cyclic_blur_columns(lum: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that wraps across the strip's angular (column) seam.

    OpenCV's ``BORDER_WRAP`` is unavailable for this path, so the columns are
    padded cyclically by hand before blurring and cropped back afterwards.  This
    keeps the operation exactly equivariant to a cyclic column roll.
    """
    if sigma is None or sigma <= 0.0:
        return lum
    k = max(1, int(math.ceil(3.0 * float(sigma))))
    w = lum.shape[1]
    if w <= 1:
        return lum
    k = min(k, w - 1)
    padded = np.concatenate([lum[:, w - k:], lum, lum[:, :k]], axis=1)
    blurred = cv2.GaussianBlur(padded, (0, 0), float(sigma),
                               borderType=cv2.BORDER_REPLICATE)
    return blurred[:, k:k + w]


def _row_mean_subtract(lum: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Subtract each row's (valid) mean, removing the shared radial profile."""
    if mask is None:
        return lum - lum.mean(axis=1, keepdims=True)
    m = mask.astype(np.float32)
    denom = m.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return lum - (lum * m).sum(axis=1, keepdims=True) / denom


def _circular_angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    d = abs((float(a) - float(b)) % 360.0)
    return min(d, 360.0 - d)


def _patch(lum: np.ndarray, row: int, col: int, rr: int, rc: int) -> np.ndarray:
    """Column-cyclic, row-clamped patch centred on ``(row, col)``."""
    h, w = lum.shape
    r0 = max(0, row - rr)
    r1 = min(h, row + rr + 1)
    cols = (np.arange(col - rc, col + rc + 1)) % w
    return lum[r0:r1][:, cols]


def _descriptor(patch: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    """Unit-L2 intensity histogram of a patch (roll-invariant)."""
    vals = np.asarray(patch).reshape(-1)
    if vals.size == 0:
        return np.zeros(int(bins), dtype=np.float32)
    rng = float(hi) - float(lo)
    if rng <= 1e-9:
        # Constant patch: a well-defined (uniform) direction rather than zeros.
        v = np.ones(int(bins), dtype=np.float32)
        return v / float(np.linalg.norm(v))
    hist, _ = np.histogram(vals, bins=int(bins), range=(float(lo), float(hi)))
    h = hist.astype(np.float32)
    n = float(np.linalg.norm(h))
    if n <= 1e-12:
        return np.zeros(int(bins), dtype=np.float32)
    return h / n


def _response_map(lum: np.ndarray, sigma: float) -> np.ndarray:
    """Centre-surround response magnitude (column-cyclic)."""
    sigma = float(sigma) if sigma and sigma > 0.0 else DEFAULT_RESPONSE_SIGMA
    surround = _cyclic_blur_columns(lum, sigma)
    return np.abs(lum - surround)


def _classify(center_val: float, patch: np.ndarray) -> StripFeatureType:
    """Light heuristic labelling of a feature."""
    if patch.size == 0:
        return StripFeatureType.UNKNOWN
    pmean = float(patch.mean())
    if float(center_val) < pmean - 3.0:
        return StripFeatureType.CRYPT
    col_profile = patch.mean(axis=0)
    row_profile = patch.mean(axis=1)
    if col_profile.size > 1 and row_profile.size > 1:
        if float(col_profile.var()) > 2.0 * float(row_profile.var()):
            return StripFeatureType.FURROW
    return StripFeatureType.TEXTURE


# ---------------------------------------------------------------------------
# Angular profile / signature
# ---------------------------------------------------------------------------

def angular_profile(
    strip: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    radial_normalize: bool = True,
    blur_sigma: float = 0.0,
    reducer: str = "mean",
) -> Optional[np.ndarray]:
    """Reduce the strip's radial axis to a 1-D angular profile.

    Parameters
    ----------
    strip:
        Polar strip (gray / BGR / BGRA); ``(height, width)``.
    mask:
        Optional bool/0-1 mask (same shape) marking *valid* pixels.  Invalid
        pixels never contribute to the reduction.
    radial_normalize:
        Subtract each row's mean before reducing, so the profile reflects
        angular structure rather than the radial brightness gradient.
    blur_sigma:
        Optional Gaussian pre-blur sigma (strip pixels, column-cyclic).
    reducer:
        ``"mean"`` (default) or ``"median"``.

    Returns
    -------
    np.ndarray or None
        ``float32`` array of length ``width``, or ``None`` when the strip is
        invalid (``None``, wrong ndim, empty).

    Notes
    -----
    For an unmasked ``"mean"`` reduction this is *exactly* cyclic: rolling the
    strip by ``k`` columns rolls the returned profile by ``k``.
    """
    if strip is None:
        return None
    try:
        lum = _to_luminance(strip)
    except (ValueError, TypeError):
        return None
    if lum.ndim != 2 or lum.shape[0] < 1 or lum.shape[1] < 1:
        return None

    h, w = lum.shape
    m = _coerce_mask(mask, (h, w))
    if mask is not None and m is None:
        # A mask was supplied but is unusable; treat the strip as invalid.
        return None

    if blur_sigma and blur_sigma > 0.0:
        lum = _cyclic_blur_columns(lum, float(blur_sigma))
    if radial_normalize:
        lum = _row_mean_subtract(lum, m)

    if m is None:
        if reducer == "median":
            return np.median(lum, axis=0).astype(np.float32)
        return lum.mean(axis=0).astype(np.float32)

    if reducer == "median":
        out = np.empty(w, dtype=np.float32)
        for c in range(w):
            vals = lum[:, c][m[:, c]]
            out[c] = float(np.median(vals)) if vals.size else 0.0
        return out

    mm = m.astype(np.float32)
    counts = mm.sum(axis=0)
    counts[counts == 0.0] = 1.0
    return ((lum * mm).sum(axis=0) / counts).astype(np.float32)


def angular_signature(strip: np.ndarray, **kwargs) -> Optional[np.ndarray]:
    """Zero-mean, unit-L2 form of :func:`angular_profile`.

    Returns ``None`` for an invalid strip; returns an all-zero vector when the
    profile is constant (no angular structure).
    """
    profile = angular_profile(strip, **kwargs)
    if profile is None:
        return None
    profile = profile - float(profile.mean())
    norm = float(np.linalg.norm(profile))
    if norm <= 1e-9:
        return np.zeros_like(profile, dtype=np.float32)
    return (profile / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_strip_features(
    strip: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    min_response_frac: float = DEFAULT_MIN_RESPONSE_FRAC,
    response_sigma: float = DEFAULT_RESPONSE_SIGMA,
    min_angular_sep_deg: float = DEFAULT_MIN_ANGULAR_SEP_DEG,
    min_radial_sep: float = DEFAULT_MIN_RADIAL_SEP,
    patch_rows: int = DEFAULT_PATCH_ROWS,
    patch_cols: int = DEFAULT_PATCH_COLS,
    descriptor_bins: int = DEFAULT_DESCRIPTOR_BINS,
    radial_normalize: bool = True,
    blur_sigma: float = 0.0,
) -> IrisStripFeatureSet:
    """Detect locally-distinctive features on a polar iris strip.

    Candidates are pixels whose centre-surround response exceeds
    ``min_response_frac`` of the strip's dynamic range (and which are inside
    ``mask``); they are accepted greedily in descending response order subject
    to angular (circular) and radial separation so features spread out instead
    of clumping.  Each accepted feature carries a column-cyclic patch
    descriptor, so descriptors are unchanged by a cyclic strip roll.

    Returns an :class:`IrisStripFeatureSet`; ``status`` is ``EMPTY`` for an
    invalid strip, ``NO_FEATURES`` when no feature passed the gates, and ``OK``
    otherwise.  Never raises for malformed input.
    """
    result = IrisStripFeatureSet()
    if strip is None:
        result.status = STATUS_EMPTY
        return result
    try:
        lum = _to_luminance(strip)
    except (ValueError, TypeError):
        result.status = STATUS_EMPTY
        return result

    if lum.ndim != 2 or lum.shape[0] < 1 or lum.shape[1] < 2:
        result.status = STATUS_EMPTY
        return result

    h, w = lum.shape
    result.height, result.width = int(h), int(w)

    m = _coerce_mask(mask, (h, w))
    if mask is not None and m is None:
        result.status = STATUS_EMPTY
        return result

    profile = angular_profile(
        lum, m, radial_normalize=radial_normalize, blur_sigma=blur_sigma
    )
    result.angular_profile = profile

    lo, hi = float(lum.min()), float(lum.max())
    rng = hi - lo
    if rng <= 1e-9:
        # Flat strip: nothing to detect (honest refusal).
        result.status = STATUS_NO_FEATURES
        return result

    thr = max(0.0, float(min_response_frac)) * rng

    resp = _response_map(lum, response_sigma)
    if m is not None:
        resp = np.where(m, resp, 0.0)

    ys, xs = np.where(resp >= thr)
    result.num_candidates = int(ys.size)
    if ys.size == 0:
        result.status = STATUS_NO_FEATURES
        return result

    order = np.argsort(-resp[ys, xs], kind="stable")
    rr = max(0, int(patch_rows))
    rc = max(0, int(patch_cols))
    rad_denom = float(h - 1) if h > 1 else 1.0

    accepted: List[StripFeature] = []
    for idx in order:
        r = int(ys[idx])
        c = int(xs[idx])
        rn = r / rad_denom
        ang = c / w * 360.0

        suppressed = False
        for f in accepted:
            if (
                _circular_angle_diff(ang, f.angle_deg) < float(min_angular_sep_deg)
                and abs(rn - f.radial_norm) < float(min_radial_sep)
            ):
                suppressed = True
                break
        if suppressed:
            continue

        patch = _patch(lum, r, c, rr, rc)
        desc = _descriptor(patch, descriptor_bins, lo, hi)
        contrast = float(np.clip(patch.std() / rng, 0.0, 1.0)) if patch.size else 0.0
        resp_val = float(resp[r, c])
        resp_norm = resp_val / (resp_val + thr + 1e-9)
        confidence = float(np.clip(0.5 * resp_norm + 0.5 * contrast, 0.0, 1.0))

        accepted.append(
            StripFeature(
                id=len(accepted),
                column=c,
                row=r,
                angle_deg=ang,
                radial_norm=float(rn),
                response=resp_val,
                local_contrast=contrast,
                confidence=confidence,
                feature_type=_classify(float(lum[r, c]), patch),
                descriptor=desc,
            )
        )

    result.features = accepted
    result.num_accepted = len(accepted)
    result.status = STATUS_OK if accepted else STATUS_NO_FEATURES
    return result


def extract_features_from_image(
    image: np.ndarray,
    roi,
    *,
    output_size: Tuple[int, int] = (64, 512),
    angle_offset_deg: float = 0.0,
    **kwargs,
) -> IrisStripFeatureSet:
    """Convenience: unwrap ``image`` with ``roi`` then extract strip features.

    Feature ``angle_deg`` values are reported in the *image* angular frame
    (``angle_offset_deg`` is added back).  Returns a set with ``status=EMPTY``
    when the ROI is invalid or the unwrap fails.  ``polar_unwrap`` is imported
    lazily so importing this module stays lightweight.
    """
    from pupil_tracking.cyclotorsion import polar_unwrap as _pu

    strip = _pu.unwrap_iris(image, roi, output_size, angle_offset_deg)
    if strip is None:
        result = IrisStripFeatureSet()
        result.status = STATUS_EMPTY
        return result

    result = extract_strip_features(strip, **kwargs)
    offset = float(angle_offset_deg) % 360.0
    if offset and result.features:
        for f in result.features:
            f.angle_deg = (f.angle_deg + offset) % 360.0
    return result


__all__ = [
    "DEFAULT_DESCRIPTOR_BINS",
    "DEFAULT_PATCH_ROWS",
    "DEFAULT_PATCH_COLS",
    "DEFAULT_RESPONSE_SIGMA",
    "DEFAULT_MIN_RESPONSE_FRAC",
    "DEFAULT_MIN_ANGULAR_SEP_DEG",
    "DEFAULT_MIN_RADIAL_SEP",
    "STATUS_OK",
    "STATUS_NO_FEATURES",
    "STATUS_EMPTY",
    "StripFeatureType",
    "StripFeature",
    "IrisStripFeatureSet",
    "angular_profile",
    "angular_signature",
    "extract_strip_features",
    "extract_features_from_image",
]
