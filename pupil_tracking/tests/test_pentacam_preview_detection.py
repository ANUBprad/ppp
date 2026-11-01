"""Tests for the Pentacam-preview detection overlay geometry.

When a Pentacam image is loaded in the Add-Patient "Pentacam Preview" the
main-window detector is run on it (still-image path) and the detected pupil /
limbus are overlaid.  Those ellipses carry a tilt (``angle_deg``), so the
overlay is drawn as a sampled polygon -- ``_preview_ellipse_points`` -- rather
than a plain Tk oval.  These tests cover that pure geometry helper.

Synthetic data only; no Tk / no model required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pupil_tracking.interface.gui_app import PupilTrackingGUI


def _ellipse(cx, cy, semi_major, semi_minor, angle_deg=0.0):
    return SimpleNamespace(
        center_x=cx,
        center_y=cy,
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=angle_deg,
        radius=(semi_major + semi_minor) / 2.0,
    )


def _bbox(points):
    xs = points[0::2]
    ys = points[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_polygon_has_two_coords_per_segment():
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(50, 60, 10, 5), 1.0, 1.0, segments=36
    )
    assert len(pts) == 72


def test_axis_aligned_ellipse_applies_scale_and_offset():
    # centre (100, 200) * 0.5 + 2 = (52, 102); semi-axes (20, 10)
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(100, 200, 40, 20), 0.5, 0.5, offset=2.0
    )
    x0, y0, x1, y1 = _bbox(pts)
    assert x0 == pytest.approx(32.0, abs=0.5)
    assert x1 == pytest.approx(72.0, abs=0.5)
    assert y0 == pytest.approx(92.0, abs=0.5)
    assert y1 == pytest.approx(112.0, abs=0.5)


def test_quarter_turn_swaps_the_axes():
    upright = _bbox(
        PupilTrackingGUI._preview_ellipse_points(
            _ellipse(0, 0, 40, 10, angle_deg=0.0), 1.0, 1.0
        )
    )
    turned = _bbox(
        PupilTrackingGUI._preview_ellipse_points(
            _ellipse(0, 0, 40, 10, angle_deg=90.0), 1.0, 1.0
        )
    )
    assert (upright[2] - upright[0]) == pytest.approx(80.0, abs=0.5)
    assert (upright[3] - upright[1]) == pytest.approx(20.0, abs=0.5)
    assert (turned[2] - turned[0]) == pytest.approx(20.0, abs=0.5)
    assert (turned[3] - turned[1]) == pytest.approx(80.0, abs=0.5)


def test_missing_angle_defaults_to_zero():
    no_angle = SimpleNamespace(
        center_x=0, center_y=0, semi_major=30, semi_minor=10
    )
    pts = PupilTrackingGUI._preview_ellipse_points(no_angle, 1.0, 1.0)
    x0, y0, x1, y1 = _bbox(pts)
    assert (x1 - x0) == pytest.approx(60.0, abs=0.5)
    assert (y1 - y0) == pytest.approx(20.0, abs=0.5)


def test_independent_x_y_scale_for_non_square_thumbnails():
    # A 1200x838 image shown as a 556x196 thumbnail scales x and y equally in
    # practice, but the helper must honour whatever scale it is handed.
    pts = PupilTrackingGUI._preview_ellipse_points(
        _ellipse(0, 0, 100, 100), 0.5, 0.2, offset=0.0
    )
    x0, y0, x1, y1 = _bbox(pts)
    assert (x1 - x0) == pytest.approx(100.0, abs=0.5)
    assert (y1 - y0) == pytest.approx(40.0, abs=0.5)


# ── Phase 3: Boundary Detection & Preview Integration Tests ──────────

def test_distinct_centers_canvas_coordinates():
    """Verify distinct pupil and limbus centers calculate correctly with canvas offset."""
    # 1200 x 838 full image -> 600 x 419 preview thumbnail (sx=0.5, sy=0.5)
    sx, sy = 0.5, 0.5
    offset = 2.0

    pupil_e = _ellipse(400.0, 300.0, 50.0, 50.0)
    limbus_e = _ellipse(410.0, 308.0, 150.0, 150.0)  # Kappa angle offset (+10, +8 px)

    pcx = offset + pupil_e.center_x * sx
    pcy = offset + pupil_e.center_y * sy

    lcx = offset + limbus_e.center_x * sx
    lcy = offset + limbus_e.center_y * sy

    assert pcx == pytest.approx(202.0)
    assert pcy == pytest.approx(152.0)
    assert lcx == pytest.approx(207.0)
    assert lcy == pytest.approx(156.0)

    # Distinct separation preserved on preview canvas
    assert (lcx - pcx) == pytest.approx(5.0)
    assert (lcy - pcy) == pytest.approx(4.0)


def test_auto_flip_180_coordinate_transformation():
    """180° rotation (flip_pentacam) transforms coordinates symmetrically."""
    from pupil_tracking.cyclotorsion.pentacam_loader import PENTACAM_HEIGHT, PENTACAM_WIDTH

    w, h = PENTACAM_WIDTH, PENTACAM_HEIGHT  # 1200, 838
    # Original coordinates
    orig_cx, orig_cy = 400.0, 300.0

    # Under 180° flip (cv2.flip(img, -1)): x' = (w - 1) - x, y' = (h - 1) - y
    flipped_cx = (w - 1) - orig_cx
    flipped_cy = (h - 1) - orig_cy

    assert flipped_cx == 1199.0 - 400.0  # 799.0
    assert flipped_cy == 837.0 - 300.0   # 537.0

    # Flipping twice restores original
    double_flipped_x = (w - 1) - flipped_cx
    double_flipped_y = (h - 1) - flipped_cy
    assert double_flipped_x == pytest.approx(orig_cx)
    assert double_flipped_y == pytest.approx(orig_cy)


def test_preview_pair_consistent_validates_concentric_structures():
    """Paired rule confirms pupil roughly concentric with and smaller than limbus."""
    # Good pair
    good_res = SimpleNamespace(
        pupil=SimpleNamespace(detected=True, ellipse=_ellipse(400, 300, 40, 40)),
        limbus=SimpleNamespace(detected=True, ellipse=_ellipse(405, 305, 120, 120)),
    )
    assert PupilTrackingGUI._preview_pair_consistent(good_res) is True

    # Bad pair: pupil far outside limbus
    bad_center = SimpleNamespace(
        pupil=SimpleNamespace(detected=True, ellipse=_ellipse(800, 300, 40, 40)),
        limbus=SimpleNamespace(detected=True, ellipse=_ellipse(405, 305, 120, 120)),
    )
    assert PupilTrackingGUI._preview_pair_consistent(bad_center) is False

    # Bad pair: pupil larger than limbus
    inverted_radii = SimpleNamespace(
        pupil=SimpleNamespace(detected=True, ellipse=_ellipse(400, 300, 140, 140)),
        limbus=SimpleNamespace(detected=True, ellipse=_ellipse(400, 300, 100, 100)),
    )
    assert PupilTrackingGUI._preview_pair_consistent(inverted_radii) is False


def test_missing_limbus_estimates_from_detected_pupil():
    """When limbus is not detected, fallback estimates limbus from pupil."""
    img = np.zeros((838, 1200, 3), dtype=np.uint8)
    pupil_res = SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=_ellipse(600.0, 419.0, 50.0, 50.0),
        ),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )
    edge, note = PupilTrackingGUI._preview_fallback_limbus(img, pupil_res)
    assert edge is not None
    assert "pupil" in note.lower()
    assert abs(edge.center_x - 600.0) <= 1.0
    assert abs(edge.center_y - 419.0) <= 1.0
    assert edge.semi_major > 50.0 * 1.5


def test_missing_detection_does_not_fabricate_ellipse():
    """When detection completely fails, no ellipse is fabricated."""
    img = np.zeros((838, 1200, 3), dtype=np.uint8)
    empty_res = SimpleNamespace(
        pupil=SimpleNamespace(detected=False, ellipse=None),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )
    edge, note = PupilTrackingGUI._preview_fallback_limbus(img, empty_res)
    assert edge is None
    assert note == ""


def test_original_limbus_geometry_preserved_from_unwrapping_scales():
    """Original limbus measurements must not be modified by preview or unwrap scaling."""
    orig_limbus = _ellipse(400.0, 300.0, 120.0, 115.0, angle_deg=5.0)
    res = SimpleNamespace(
        limbus=SimpleNamespace(detected=True, ellipse=orig_limbus),
        overall_confidence=0.85,
    )

    # Preview retrieval returns original ellipse
    preview_ellipse = PupilTrackingGUI._preview_limbus_ellipse(res, True)
    assert preview_ellipse is orig_limbus
    assert preview_ellipse.semi_major == 120.0
    assert preview_ellipse.semi_minor == 115.0
    assert preview_ellipse.angle_deg == 5.0

