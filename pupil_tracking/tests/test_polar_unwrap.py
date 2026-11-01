"""Tests for Module 2 — fast vectorised iris polar unwrapping.

These tests are deterministic and synthetic: no real clinical imagery is used.
The key property exercised here is *rotation equivariance* — rotating the
source image about the iris centre must cyclically shift the unwrapped strip
by the corresponding number of columns.  That property is the foundation the
cyclotorsion stage relies on, so it is asserted explicitly.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.polar_unwrap import (
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    UnwrapMaps,
    angle_to_column,
    apply_unwrap_maps,
    build_unwrap_maps,
    column_to_angle,
    roll_strip,
    unwrap_from_ellipses,
    unwrap_iris,
)
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.utils.types import EllipseParams


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CX = CY = 100.0
PUPIL_R = 30.0
LIMBUS_R = 80.0
IMG_SIZE = 200


def _circular_roi(
    pupil_r: float = PUPIL_R,
    limbus_r: float = LIMBUS_R,
    inner_inset: float = 0.0,
    outer_inset: float = 0.0,
) -> IrisROI:
    """Build a valid annular IrisROI with circular pupil/limbus boundaries."""
    return IrisROI(
        valid=True,
        reason="ok",
        center_x=CX,
        center_y=CY,
        pupil_semi_major=pupil_r,
        pupil_semi_minor=pupil_r,
        pupil_angle_deg=0.0,
        limbus_semi_major=limbus_r,
        limbus_semi_minor=limbus_r,
        limbus_angle_deg=0.0,
        inner_inset_frac=inner_inset,
        outer_inset_frac=outer_inset,
        pupil_radius_px=pupil_r,
        limbus_radius_px=limbus_r,
    )


def _spoke_image(angle_deg: float = 0.0, width: int = 5) -> np.ndarray:
    """A dark image with one bright radial spoke through the iris annulus."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    a = math.radians(angle_deg)
    p0 = (int(round(CX + (PUPIL_R - 2) * math.cos(a))),
          int(round(CY + (PUPIL_R - 2) * math.sin(a))))
    p1 = (int(round(CX + (LIMBUS_R - 5) * math.cos(a))),
          int(round(CY + (LIMBUS_R - 5) * math.sin(a))))
    cv2.line(img, p0, p1, 255, width)
    return img


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    m = cv2.getRotationMatrix2D((CX, CY), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)


def _peak_column(strip: np.ndarray) -> int:
    """Column with the highest mean intensity (the spoke's angular position)."""
    profile = strip.reshape(-1, strip.shape[1]).mean(axis=0)
    return int(np.argmax(profile))


def _circular_distance(a: int, b: int, width: int) -> int:
    d = (a - b) % width
    return min(d, width - d)


# ---------------------------------------------------------------------------
# Basic geometry / shape
# ---------------------------------------------------------------------------

def test_default_strip_size_matches_cnn_encoder():
    assert (DEFAULT_STRIP_HEIGHT, DEFAULT_STRIP_WIDTH) == (64, 512)


def test_unwrap_gray_shape_and_dtype():
    strip = unwrap_iris(_spoke_image(), _circular_roi(), (64, 512))
    assert strip is not None
    assert strip.shape == (64, 512)
    assert strip.dtype == np.uint8


def test_unwrap_preserves_channels():
    gray = _spoke_image()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    strip = unwrap_iris(bgr, _circular_roi(), (48, 256))
    assert strip is not None
    assert strip.shape == (48, 256, 3)


def test_build_maps_returns_float32_maps():
    maps = build_unwrap_maps(_circular_roi(), (32, 128))
    assert isinstance(maps, UnwrapMaps)
    assert maps.map_x.shape == (32, 128)
    assert maps.map_y.shape == (32, 128)
    assert maps.map_x.dtype == np.float32
    assert maps.height == 32 and maps.width == 128


def test_apply_maps_matches_unwrap_iris():
    img = _spoke_image()
    roi = _circular_roi()
    maps = build_unwrap_maps(roi, (40, 200), angle_offset_deg=17.0)
    a = apply_unwrap_maps(img, maps)
    b = unwrap_iris(img, roi, (40, 200), angle_offset_deg=17.0)
    assert np.array_equal(a, b)


def test_invalid_roi_returns_none():
    invalid = IrisROI(valid=False)
    assert unwrap_iris(_spoke_image(), invalid) is None
    assert build_unwrap_maps(invalid) is None
    assert unwrap_iris(_spoke_image(), None) is None


def test_degenerate_output_size_returns_none():
    assert build_unwrap_maps(_circular_roi(), (0, 100)) is None
    assert build_unwrap_maps(_circular_roi(), (100, 0)) is None


# ---------------------------------------------------------------------------
# Angular helpers
# ---------------------------------------------------------------------------

def test_column_angle_roundtrip():
    for width in (360, 512):
        for col in (0, 1, width // 2, width - 1):
            angle = column_to_angle(col, width)
            assert 0.0 <= angle < 360.0
            assert _circular_distance(angle_to_column(angle, width), col, width) <= 1


def test_angle_to_column_wraps():
    assert angle_to_column(0.0, 360) == 0
    assert angle_to_column(360.0, 360) == 0
    assert angle_to_column(90.0, 360) == 90
    assert angle_to_column(-90.0, 360) == 270


def test_roll_strip_is_cyclic():
    strip = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    rolled = roll_strip(strip, 2)
    assert np.array_equal(rolled[:, 2:], strip[:, :-2])
    assert np.array_equal(rolled[:, :2], strip[:, -2:])
    assert np.array_equal(roll_strip(strip, 0), strip)


# ---------------------------------------------------------------------------
# Geometry correctness
# ---------------------------------------------------------------------------

def test_spoke_at_zero_angle_sits_at_column_zero():
    strip = unwrap_iris(_spoke_image(angle_deg=0.0), _circular_roi(), (64, 512))
    peak = _peak_column(strip)
    # Column 0 and column 511 are adjacent around the seam.
    assert _circular_distance(peak, 0, 512) <= 1


def test_spoke_at_ninety_degrees_sits_at_quarter_width():
    strip = unwrap_iris(_spoke_image(angle_deg=90.0), _circular_roi(), (64, 512))
    peak = _peak_column(strip)
    assert _circular_distance(peak, 512 // 4, 512) <= 1


def test_inner_rows_map_to_pupil_side_and_outer_rows_to_limbus_side():
    # An annulus that is bright from the pupil boundary out to the mid radius
    # and dark from the mid radius out to the limbus must produce a strip whose
    # pupil-side (inner) rows are bright and whose limbus-side (outer) rows are
    # dark.  This pins down the radial orientation of the strip.
    mid = int((PUPIL_R + LIMBUS_R) / 2)
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(img, (int(CX), int(CY)), mid, 255, -1)
    cv2.circle(img, (int(CX), int(CY)), int(PUPIL_R), 0, -1)

    strip = unwrap_iris(img, _circular_roi(), (64, 512))
    top = float(strip[:24].mean())      # near the pupil boundary
    bottom = float(strip[40:].mean())   # near the limbus boundary
    assert top > bottom
    assert top > 200.0
    assert bottom < 50.0


# ---------------------------------------------------------------------------
# Cyclotorsion-critical property: rotation == cyclic column shift
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [15.0, 30.0, 45.0, 90.0])
def test_rotation_shifts_strip_columns(angle_deg):
    width = 512
    base = _spoke_image(angle_deg=0.0)
    rotated = _rotate(base, angle_deg)
    roi = _circular_roi()

    strip_base = unwrap_iris(base, roi, (64, width))
    strip_rot = unwrap_iris(rotated, roi, (64, width))

    shift = _circular_distance(_peak_column(strip_rot), _peak_column(strip_base), width)
    expected = angle_deg / 360.0 * width
    assert abs(shift - expected) <= 2.0


def test_rotated_strip_is_cyclic_shift_of_original():
    width = 256
    base = _spoke_image(angle_deg=0.0)
    rotated = _rotate(base, 40.0)
    roi = _circular_roi()

    strip_base = unwrap_iris(base, roi, (64, width)).astype(np.float32)
    strip_rot = unwrap_iris(rotated, roi, (64, width)).astype(np.float32)

    # Find the integer column shift that best aligns the two strips.
    profile_base = strip_base.mean(axis=0)
    best_shift, best_err = 0, None
    for s in range(width):
        err = float(np.sum((profile_base - np.roll(strip_rot.mean(axis=0), s)) ** 2))
        if best_err is None or err < best_err:
            best_err, best_shift = err, s

    circ = _circular_distance(best_shift, 0, width)
    expected = 40.0 / 360.0 * width
    assert abs(circ - expected) <= 2.0


def test_angle_offset_shifts_strip():
    width = 512
    img = _spoke_image(angle_deg=0.0)
    roi = _circular_roi()

    base = unwrap_iris(img, roi, (64, width), angle_offset_deg=0.0)
    offset = unwrap_iris(img, roi, (64, width), angle_offset_deg=90.0)

    peak_base = _peak_column(base)
    peak_offset = _peak_column(offset)
    # A +90 deg offset moves the angle-0 content to the 3/4 column.
    assert _circular_distance(peak_offset, (peak_base + 3 * width // 4) % width, width) <= 1


def test_angle_offset_equals_roll():
    width = 256
    img = _spoke_image(angle_deg=0.0)
    roi = _circular_roi()

    base = unwrap_iris(img, roi, (64, width), angle_offset_deg=0.0)
    offset = unwrap_iris(img, roi, (64, width), angle_offset_deg=90.0)

    assert np.array_equal(offset, np.roll(base, -width // 4, axis=1))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def test_unwrap_from_ellipses():
    img = _spoke_image(angle_deg=0.0)
    pupil = EllipseParams(center_x=CX, center_y=CY, semi_major=PUPIL_R,
                          semi_minor=PUPIL_R, angle_deg=0.0)
    limbus = EllipseParams(center_x=CX, center_y=CY, semi_major=LIMBUS_R,
                           semi_minor=LIMBUS_R, angle_deg=0.0)
    strip = unwrap_from_ellipses(
        img, pupil, limbus, (48, 256),
        inner_inset_frac=0.0, outer_inset_frac=0.0,
    )
    assert strip is not None
    assert strip.shape == (48, 256)
    assert _circular_distance(_peak_column(strip), 0, 256) <= 1


def test_unwrap_from_ellipses_invalid_geometry_returns_none():
    img = _spoke_image()
    # Pupil larger than limbus -> implausible geometry -> invalid ROI.
    pupil = EllipseParams(center_x=CX, center_y=CY, semi_major=90.0,
                          semi_minor=90.0, angle_deg=0.0)
    limbus = EllipseParams(center_x=CX, center_y=CY, semi_major=80.0,
                           semi_minor=80.0, angle_deg=0.0)
    assert unwrap_from_ellipses(img, pupil, limbus) is None
    assert unwrap_from_ellipses(img, None, limbus) is None


# ---------------------------------------------------------------------------
# Ellipse polar radius & decentered Daugman rubber-sheet tests
# ---------------------------------------------------------------------------

def test_ellipse_polar_radius_axes():
    """Verify that ellipse radius equation correctly aligns major axis with a and minor axis with b."""
    from pupil_tracking.iris.normalization import _ellipse_radius_at_angle

    # Horizontal ellipse: semi_major a=100, semi_minor b=50, rotation 0 deg
    r_major = _ellipse_radius_at_angle(0.0, 0.0, 100.0, 50.0, 0.0, 0.0)
    r_minor = _ellipse_radius_at_angle(0.0, 0.0, 100.0, 50.0, 0.0, 90.0)
    assert abs(r_major - 100.0) < 1e-4, f"Expected 100.0 along major axis, got {r_major}"
    assert abs(r_minor - 50.0) < 1e-4, f"Expected 50.0 along minor axis, got {r_minor}"

    # Rotated ellipse: rotation 30 deg
    r_major_rot = _ellipse_radius_at_angle(0.0, 0.0, 100.0, 50.0, 30.0, 30.0)
    r_minor_rot = _ellipse_radius_at_angle(0.0, 0.0, 100.0, 50.0, 30.0, 120.0)
    assert abs(r_major_rot - 100.0) < 1e-4, f"Expected 100.0 along rotated major axis, got {r_major_rot}"
    assert abs(r_minor_rot - 50.0) < 1e-4, f"Expected 50.0 along rotated minor axis, got {r_minor_rot}"


def test_decentered_daugman_rubber_sheet():
    """Verify that decentered pupil and limbus interpolate correctly without clipping or intrusion."""
    # Create image with decentered pupil (center at 115, 100) and limbus at (100, 100)
    img_size = 240
    pupil_center = (115.0, 100.0)
    limbus_center = (100.0, 100.0)
    pupil_r = 25.0
    limbus_r = 75.0

    # Draw pupil disk as 10 (dark) and limbus boundary ring as 250 (bright)
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.circle(img, (int(pupil_center[0]), int(pupil_center[1])), int(pupil_r), 10, -1)
    cv2.circle(img, (int(limbus_center[0]), int(limbus_center[1])), int(limbus_r), 250, 2)

    pupil = EllipseParams(center_x=pupil_center[0], center_y=pupil_center[1],
                          semi_major=pupil_r, semi_minor=pupil_r, angle_deg=0.0)
    limbus = EllipseParams(center_x=limbus_center[0], center_y=limbus_center[1],
                           semi_major=limbus_r, semi_minor=limbus_r, angle_deg=0.0)

    # Unwrap with 0 insets so row 0 is exactly the pupil boundary and row -1 is limbus
    strip = unwrap_from_ellipses(
        img, pupil, limbus, output_size=(64, 256),
        inner_inset_frac=0.0, outer_inset_frac=0.0,
    )
    assert strip is not None
    assert strip.shape == (64, 256)

    # Row 0 (pupil boundary) should be right on the edge of the pupil circle
    # Check that row 0 does NOT cut deep into pupil (intensity 10) on one side while missing on other
    from pupil_tracking.iris.roi import IrisROIExtractor
    roi_decentered = IrisROIExtractor(0.0, 0.0).build(pupil, limbus)
    maps = build_unwrap_maps(
        roi_decentered,
        output_size=(64, 256)
    )
    assert maps is not None
    # Angle 0 (col 0): xp = 115 + 25 = 140
    assert abs(maps.map_x[0, 0] - 140.0) < 1e-2
    # Angle 180 (col 128): xp = 115 - 25 = 90
    assert abs(maps.map_x[0, 128] - 90.0) < 1e-2
    # Limbus at row -1: Angle 0: xl = 100 + 75 = 175
    assert abs(maps.map_x[-1, 0] - 175.0) < 1e-2
    # Limbus at row -1: Angle 180: xl = 100 - 75 = 25
    assert abs(maps.map_x[-1, 128] - 25.0) < 1e-2


def test_known_landmarks_and_rotation_displacement():
    """Verify that landmarks at known angles map to expected columns and rotate predictably."""
    width = 360
    height = 32
    roi = _circular_roi(pupil_r=30.0, limbus_r=70.0)

    # Place landmarks at 0°, 90°, 180°, 270° in the mid-iris (r = 50.0)
    for target_deg in [0.0, 90.0, 180.0, 270.0]:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        rad = math.radians(target_deg)
        lx = int(round(CX + 50.0 * math.cos(rad)))
        ly = int(round(CY + 50.0 * math.sin(rad)))
        cv2.circle(img, (lx, ly), 3, 255, -1)

        strip = unwrap_iris(img, roi, (height, width), angle_offset_deg=0.0)
        assert strip is not None
        col = _peak_column(strip)
        expected_col = int(round(target_deg / 360.0 * width)) % width
        assert _circular_distance(col, expected_col, width) <= 2, (
            f"Landmark at {target_deg}° expected at col {expected_col}, got {col}"
        )

    # Test known rotation displacement (+30 degrees)
    img_spoke = _spoke_image(angle_deg=45.0, width=3)
    strip_0 = unwrap_iris(img_spoke, roi, (height, width), angle_offset_deg=0.0)
    img_spoke_rot = _rotate(img_spoke, 30.0)
    strip_rot = unwrap_iris(img_spoke_rot, roi, (height, width), angle_offset_deg=0.0)

    col_0 = _peak_column(strip_0)
    col_rot = _peak_column(strip_rot)
    diff = _circular_distance(col_rot, (col_0 - 30) % width, width)
    assert diff <= 2, f"Expected 30 col shift, got difference of {diff}"

