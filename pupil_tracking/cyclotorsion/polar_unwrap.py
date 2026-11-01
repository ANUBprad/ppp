"""polar_unwrap.py — fast vectorised iris polar unwrapping.

Module 2 of the cyclotorsion feature.  Converts the annular iris region
(between the pupil and limbus boundaries) into a rectangular "rubber-sheet"
strip using Daugman normalisation:

    row    -> normalised radial position: 0 = pupil boundary,
              1 = limbus boundary
    column -> polar angle, 0 -> 360 degrees, measured in the image plane
              from the positive x-axis (matching ``IrisNormalizer``)

The unwrap is geometry-driven and read-only.  Given an :class:`IrisROI`
(reused from ``pupil_tracking.iris``) it builds a pair of ``cv2.remap``
coordinate maps and samples the image with bilinear interpolation.  Building
the maps once and re-applying them is the fast path used when many frames
share the same geometry (e.g. an image pair in the cyclotorsion stage).

The default strip size (64 x 512) matches the CNN iris encoder's expected
input, so strips can be fed straight to ``IrisEncoderNet.encode_strip``.

This module is *additive*: it never modifies the source image or the
geometry it is given, and it is not wired into the centration / pupil /
limbus pipeline.  It only produces strips for the (later) correspondence /
rotation-recovery stage and for texture encoding.

:func:`unwrap_disc` complements the annulus unwrap: it flattens the full
interior of a single ellipse (the region the GUI Snapshot crops inside the
blue limbus ring) rather than the pupil-to-limbus annulus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import IrisROI


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default strip size (height=radial samples, width=angular samples).  Matches
#: ``pupil_tracking.ml.iris_encoder``'s input, so strips are drop-in encodable.
DEFAULT_STRIP_HEIGHT = 64
DEFAULT_STRIP_WIDTH = 512

#: Border handling for samples that fall outside the source image.  Replicate
#: avoids black bleeding at the pupil/limbus edges of the strip.
DEFAULT_BORDER = cv2.BORDER_REPLICATE


# ---------------------------------------------------------------------------
# Coordinate maps
# ---------------------------------------------------------------------------

@dataclass
class UnwrapMaps:
    """Precomputed ``cv2.remap`` coordinate maps for one iris geometry.

    ``map_x`` / ``map_y`` are ``float32`` arrays of shape ``(height, width)``
    that map each output strip pixel back to a source-image pixel.
    """

    map_x: np.ndarray
    map_y: np.ndarray
    output_size: Tuple[int, int]      # (height, width)
    angle_offset_deg: float = 0.0

    @property
    def height(self) -> int:
        return self.output_size[0]

    @property
    def width(self) -> int:
        return self.output_size[1]


def build_unwrap_maps(
    roi: IrisROI,
    output_size: Tuple[int, int] = (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH),
    angle_offset_deg: float = 0.0,
    normalizer: Optional[IrisNormalizer] = None,
    eye: str = "OD",
) -> Optional[UnwrapMaps]:
    """Build ``cv2.remap`` maps that unwrap ``roi``'s iris annulus.

    Parameters
    ----------
    roi:
        Valid iris ROI (annulus between pupil and limbus ellipses).  Returns
        None when the ROI is missing or invalid.
    output_size:
        ``(height, width)`` of the produced strip.
    angle_offset_deg:
        Angular rotation of the sampling grid, in degrees.  A non-zero offset
        rotates the strip (useful for cyclotorsion alignment); the column that
        would be angle 0 is instead angle ``angle_offset_deg``.
    normalizer:
        Optional :class:`IrisNormalizer` (for a custom ``radial_epsilon``).
        Defaults to a fresh instance.
    eye:
        "OD" (right eye) or "OS" (left eye).
        Clinical anatomical convention (patient's nose is the central reference):
        - Nasal is 0°, Inferior is 90°, Temporal is 180°, Superior is 270°.
        - For OD (Right Eye): Nasal (0°) is Screen Right (+x), Temporal (180°) is Screen Left (-x).
        - For OS (Left Eye): Nasal (0°) is Screen Left (-x), Temporal (180°) is Screen Right (+x).
        Both eyes align to the common 0° → 360° ruler.

    Returns
    -------
    UnwrapMaps or None
        None when the ROI is invalid or the annulus has no usable span.
    """
    if roi is None or not roi.valid:
        return None

    out_h = int(output_size[0])
    out_w = int(output_size[1])
    if out_h < 1 or out_w < 1:
        return None

    normalizer = normalizer or IrisNormalizer()

    radial_frac = np.linspace(0.0, 1.0, out_h, dtype=np.float64)  # pupil -> limbus
    columns = np.arange(out_w, dtype=np.float64)
    angles_deg = (columns / out_w) * 360.0 + float(angle_offset_deg)

    map_x = np.empty((out_h, out_w), dtype=np.float32)
    map_y = np.empty((out_h, out_w), dtype=np.float32)

    px_c = float(getattr(roi, "p_center_x", getattr(roi, "pupil_center_x", roi.center_x)))
    py_c = float(getattr(roi, "p_center_y", getattr(roi, "pupil_center_y", roi.center_y)))
    lx_c = float(getattr(roi, "l_center_x", getattr(roi, "limbus_center_x", roi.center_x)))
    ly_c = float(getattr(roi, "l_center_y", getattr(roi, "limbus_center_y", roi.center_y)))

    is_os = (str(eye) if eye else "OD").strip().upper() == "OS"

    any_span = False
    for col in range(out_w):
        ang = float(angles_deg[col])
        if is_os:
            # OS (Left Eye): Nasal is Screen Left (-x, 180° in polar);
            # Inferior is Screen Down (+y, 90°); Temporal is Screen Right (+x, 0°).
            ray_angle = (180.0 - ang) % 360.0
            cos_ang = -np.cos(np.deg2rad(ang))
            sin_ang = np.sin(np.deg2rad(ang))
        else:
            # OD (Right Eye): Nasal is Screen Right (+x, 0° in polar);
            # Inferior is Screen Down (+y, 90°); Temporal is Screen Left (-x, 180°).
            ray_angle = ang % 360.0
            cos_ang = np.cos(np.deg2rad(ang))
            sin_ang = np.sin(np.deg2rad(ang))


        inner, outer = normalizer.radial_bounds(roi, float(ray_angle))
        span = outer - inner
        if span <= 0.0:
            # Degenerate angle: collapse safely onto the inner boundary.
            outer = inner
        else:
            any_span = True

        # Daugman rubber-sheet model: interpolate between pupil and limbus boundary points
        xp = px_c + inner * cos_ang
        yp = py_c + inner * sin_ang
        xl = lx_c + outer * cos_ang
        yl = ly_c + outer * sin_ang

        map_x[:, col] = (1.0 - radial_frac) * xp + radial_frac * xl
        map_y[:, col] = (1.0 - radial_frac) * yp + radial_frac * yl

    if not any_span:
        return None

    return UnwrapMaps(
        map_x=map_x,
        map_y=map_y,
        output_size=(out_h, out_w),
        angle_offset_deg=float(angle_offset_deg),
    )


def apply_unwrap_maps(
    image: np.ndarray,
    maps: UnwrapMaps,
    border_mode: int = DEFAULT_BORDER,
) -> np.ndarray:
    """Sample ``image`` through precomputed ``maps`` (bilinear interpolation).

    Works for both single-channel and multi-channel images; the output keeps
    the same number of channels with shape ``(*maps.output_size, C)``.
    """
    return cv2.remap(
        image,
        maps.map_x,
        maps.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------

def unwrap_iris(
    image: np.ndarray,
    roi: IrisROI,
    output_size: Tuple[int, int] = (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH),
    angle_offset_deg: float = 0.0,
    border_mode: int = DEFAULT_BORDER,
    eye: str = "OD",
) -> Optional[np.ndarray]:
    """Unwrap the iris annulus into a polar strip.

    Returns the strip (same dtype/channels as ``image``) or None when the ROI
    is invalid.  Row 0 is the pupil-side boundary, row ``height-1`` the
    limbus-side boundary; column 0 is angle 0 (Nasal: Screen Right for OD, Screen
    Left for OS) unless ``angle_offset_deg`` shifts the grid.
    """
    maps = build_unwrap_maps(roi, output_size, angle_offset_deg, eye=eye)
    if maps is None:
        return None
    return apply_unwrap_maps(image, maps, border_mode)


def unwrap_from_ellipses(
    image: np.ndarray,
    pupil,
    limbus,
    output_size: Tuple[int, int] = (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH),
    angle_offset_deg: float = 0.0,
    inner_inset_frac: float = 0.10,
    outer_inset_frac: float = 0.10,
    border_mode: int = DEFAULT_BORDER,
    eye: str = "OD",
) -> Optional[np.ndarray]:
    """Convenience wrapper: build an ``IrisROI`` from pupil/limbus ellipses,
    then unwrap.

    ``pupil`` / ``limbus`` are :class:`~pupil_tracking.utils.types.EllipseParams`
    (or anything accepted by :class:`IrisROIExtractor`).  Returns None when the
    geometry is missing or implausible.
    """
    from pupil_tracking.iris.roi import IrisROIExtractor

    roi = IrisROIExtractor(
        inner_inset_frac=inner_inset_frac,
        outer_inset_frac=outer_inset_frac,
    ).build(pupil, limbus)
    return unwrap_iris(image, roi, output_size, angle_offset_deg, border_mode, eye=eye)


def unwrap_disc(
    image: np.ndarray,
    center: Tuple[float, float],
    semi_major: float,
    semi_minor: float,
    angle_deg: float = 0.0,
    output_size: Tuple[int, int] = (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH),
    angle_offset_deg: float = 180.0,
    border_mode: int = DEFAULT_BORDER,
) -> Optional[np.ndarray]:
    """Flatten the full disc inside a single ellipse into a polar strip.

    Unlike :func:`unwrap_iris` (which unwraps the *annulus* between the pupil
    and limbus) this unwraps the whole interior of one ellipse — the region
    captured by the GUI Snapshot's circular limbus crop.

    Row 0 is the ellipse **centre**; row ``height-1`` is the ellipse
    **boundary**.  Column 0 is polar angle ``angle_offset_deg`` (default 180,
    the left edge of the strip); the strip spans a full 360 degrees, so the
    centre column is angle ``angle_offset_deg + 180`` (0/360 by default) and
    the right edge is 180 again.  Angles are measured in the image plane from
    the positive x-axis, increasing toward +y (matching :func:`unwrap_iris`).

    Returns ``None`` when the geometry is missing or degenerate.
    """
    if image is None:
        return None

    out_h = int(output_size[0])
    out_w = int(output_size[1])
    if out_h < 1 or out_w < 1:
        return None

    cx, cy = float(center[0]), float(center[1])
    a = float(max(semi_major, semi_minor))
    b = float(min(semi_major, semi_minor))
    if not all(np.isfinite(v) for v in (cx, cy, a, b)):
        return None
    if a <= 0.0 or b <= 0.0:
        return None

    radial = np.linspace(0.0, 1.0, out_h, dtype=np.float64)          # (H,)
    columns = np.arange(out_w, dtype=np.float64)
    angles_deg = columns / out_w * 360.0 + float(angle_offset_deg)
    theta = np.deg2rad(angles_deg)                                    # (W,)
    phi = np.deg2rad(float(angle_deg))

    # Ellipse boundary radius as a function of the polar angle.
    ct = np.cos(theta - phi)
    st = np.sin(theta - phi)
    r_boundary = (a * b) / np.sqrt((b * ct) ** 2 + (a * st) ** 2)     # (W,)

    radii = radial[:, None] * r_boundary[None, :]                     # (H, W)
    map_x = (cx + radii * np.cos(theta)[None, :]).astype(np.float32)
    map_y = (cy + radii * np.sin(theta)[None, :]).astype(np.float32)

    maps = UnwrapMaps(
        map_x=map_x,
        map_y=map_y,
        output_size=(out_h, out_w),
        angle_offset_deg=float(angle_offset_deg),
    )
    return apply_unwrap_maps(image, maps, border_mode)


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------

def column_to_angle(column: int, width: int = DEFAULT_STRIP_WIDTH) -> float:
    """Return the polar angle (degrees, ``[0, 360)``) for a strip column."""
    if width <= 0:
        return 0.0
    return (int(column) % width) / width * 360.0


def angle_to_column(angle_deg: float, width: int = DEFAULT_STRIP_WIDTH) -> int:
    """Return the nearest strip column for a polar angle (degrees)."""
    if width <= 0:
        return 0
    return int(round((float(angle_deg) % 360.0) / 360.0 * width)) % width


def roll_strip(strip: np.ndarray, shift_cols: int) -> np.ndarray:
    """Cyclically shift a strip by ``shift_cols`` columns along the angle axis.

    Positive ``shift_cols`` moves content toward higher columns (increasing
    angle).  Because angle is periodic, the strip wraps around.
    """
    return np.roll(strip, int(round(shift_cols)), axis=1)
