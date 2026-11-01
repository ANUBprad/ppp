"""Tests for the GUI unwrap strips (iris annulus only).

The main-window panel shows two strips that share one 0° → 360° ruler: column 0
is 0° (screen right) and column ``width`` is 360°.  Both strips show only the
**iris annulus** — the area between the green pupil ring and the blue limbus
ring — so the dark pupil is never included: the detected pupil and limbus
ellipses are unwrapped together, and a missing boundary leaves the strip empty.

The lower-level ``unwrap_disc`` primitive (whole-ellipse unwrap) is still
exercised below, but the panels no longer use it as a fallback.
"""
from types import SimpleNamespace

import cv2
import inspect
import math

import numpy as np
import pytest

import pupil_tracking.interface.gui_app as gui_app
from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_disc
from pupil_tracking.interface.gui_app import (
    PupilTrackingGUI,
    _UNWRAP_STRIP_SIZE,
    _UNWRAP_ZONE_SECTORS,
    _UNWRAP_ZONE_QUADRANTS,
    _UNWRAP_ZONE_BANDS,
    _ELITA_SHIFT_STEP_DEG,
    _PENTACAM_MIN_TRUSTED_CONFIDENCE,
    _PENTACAM_MIN_PAIRED_CONFIDENCE,
    _PENTACAM_LIMBUS_PUPIL_RATIO,
    _scale_ring_radius,
)
from pupil_tracking.utils.types import EllipseParams

_STRIP_SHAPE = _UNWRAP_STRIP_SIZE  # (rows, cols)
_MID_ROW = _UNWRAP_STRIP_SIZE[0] // 2


def _angle_map(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Float image whose value at each pixel is its angle (deg) about (cx, cy)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return ((np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0)).astype(np.float32)


def _radius_map(h: int, w: int, cx: float, cy: float) -> np.ndarray:
    """Float image whose value at each pixel is its distance from (cx, cy)."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ── unwrap_disc primitives (unchanged) ────────────────────────────────────


def test_left_centre_right_angle_mapping():
    h = w = 400
    cx = cy = 200.0
    img = _angle_map(h, w, cx, cy)
    r = 120.0

    strip = unwrap_disc(
        img, (cx, cy), r, r, 0.0,
        output_size=(16, 720), angle_offset_deg=180.0,
    )
    assert strip is not None
    assert strip.shape == (16, 720)

    row = strip[8]
    # Left edge = 180°, centre = 0°/360°, and the angle increases to the right.
    assert _angle_diff(float(row[0]), 180.0) < 3.0
    assert _angle_diff(float(row[360]), 0.0) < 3.0
    assert _angle_diff(float(row[540]), 90.0) < 3.0


def test_row_zero_is_the_centre():
    h = w = 300
    cx = cy = 150.0
    img = _angle_map(h, w, cx, cy)
    strip = unwrap_disc(
        img, (cx, cy), 100.0, 100.0, output_size=(8, 360), angle_offset_deg=180.0
    )
    # Row 0 is radius 0 -> every column samples the (single) centre pixel.
    assert np.allclose(strip[0], strip[0, 0])
    # The centre pixel's own angle is 0 (arctan2(0, 0) -> 0).
    assert _angle_diff(float(strip[0, 0]), 0.0) < 1.0


def test_radius_grows_from_centre_to_boundary():
    h = w = 200
    cx = cy = 100.0
    img = _angle_map(h, w, cx, cy)  # angle is independent of radius
    strip = unwrap_disc(
        img, (cx, cy), 50.0, 50.0, output_size=(30, 90), angle_offset_deg=180.0
    )
    # Column at the centre angle (0°) should be flat down the strip.
    col = 45  # ~ 0/360
    assert np.allclose(strip[:, col], strip[0, col], atol=3.0)


def test_degenerate_geometry_returns_none():
    img = np.zeros((40, 40, 3), np.uint8)
    assert unwrap_disc(img, (20.0, 20.0), 0.0, 0.0) is None
    assert unwrap_disc(img, (20.0, 20.0), np.nan, 10.0) is None
    assert unwrap_disc(img, (20.0, 20.0), 10.0, 10.0, output_size=(0, 10)) is None
    assert unwrap_disc(None, (20.0, 20.0), 10.0, 10.0) is None


# ── GUI panel helpers ─────────────────────────────────────────────────────


def _make_app() -> PupilTrackingGUI:
    return object.__new__(PupilTrackingGUI)


def test_unwrap_pentacam_without_image_is_none():
    app = _make_app()
    img = np.zeros((40, 40, 3), np.uint8)
    assert app._unwrap_pentacam(None) is None
    # A missing image wins even when boundaries are supplied ...
    assert app._unwrap_pentacam(
        None, limbus_ellipse=_limbus(20.0, 20.0, 10.0, 10.0),
        pupil_ellipse=_ellipse(20.0, 20.0, 5.0),
    ) is None
    # ... and with no boundaries there is nothing to unwrap either.
    assert app._unwrap_pentacam(img) is None


def test_set_unwrap_strip_stores_and_redraws():
    app = _make_app()
    app._unwrap_canvas = None  # skip Tk redraw
    strip = np.zeros((10, 20, 3), np.uint8)

    app._set_unwrap_strip(strip)
    assert app._unwrap_strip is strip

    app._set_unwrap_strip(None)
    assert app._unwrap_strip is None


def _limbus(cx, cy, semi_major, semi_minor, angle_deg=0.0):
    """A limbus ellipse the annulus path can use (needs ``is_valid``)."""
    return EllipseParams(
        center_x=cx,
        center_y=cy,
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=angle_deg,
    )


def test_unwrap_pentacam_needs_both_pupil_and_limbus():
    """Only the iris annulus is unwrapped — a lone boundary yields nothing."""
    app = _make_app()
    h = w = 400
    cx = cy = 200.0
    img = _angle_map(h, w, cx, cy)
    limbus = _limbus(cx, cy, 100.0, 100.0)
    pupil = _ellipse(cx, cy, 60.0)

    # A limbus on its own is not enough (there is no disc fallback) ...
    assert app._unwrap_pentacam(img, limbus_ellipse=limbus) is None
    # ... and neither is a pupil on its own.
    assert app._unwrap_pentacam(img, pupil_ellipse=pupil) is None

    # Both boundaries together produce the annulus strip, 0° at column 0.
    strip = app._unwrap_pentacam(img, limbus_ellipse=limbus, pupil_ellipse=pupil)

    assert strip is not None
    assert strip.shape == _STRIP_SHAPE
    row = strip[_MID_ROW]
    assert _angle_diff(float(row[0]), 0.0) < 4.0
    assert _angle_diff(float(row[360]), 180.0) < 4.0


def test_unwrap_pentacam_degenerate_limbus_is_none():
    app = _make_app()
    h = w = 400
    img = _angle_map(h, w, 200.0, 200.0)
    degenerate = _limbus(200.0, 200.0, 0.0, 0.0)  # unusable detection
    pupil = _ellipse(200.0, 200.0, 60.0)

    assert (
        app._unwrap_pentacam(img, limbus_ellipse=degenerate, pupil_ellipse=pupil)
        is None
    )


# ── iris annulus (preferred content) ──────────────────────────────────────


def _ellipse(cx, cy, r):
    return EllipseParams(center_x=cx, center_y=cy, semi_major=r, semi_minor=r)


def test_unwrap_pentacam_uses_iris_annulus_when_pupil_present():
    app = _make_app()
    h = w = 400
    cx = cy = 200.0
    img = _radius_map(h, w, cx, cy)  # value == distance from the centre
    pupil = _ellipse(cx, cy, 60.0)
    limbus = _ellipse(cx, cy, 120.0)

    strip = app._unwrap_pentacam(
        img, limbus_ellipse=limbus, pupil_ellipse=pupil
    )

    assert strip is not None
    assert strip.shape == _STRIP_SHAPE
    # Row 0 must be the pupil boundary (inset into the iris), NOT the centre —
    # that is what distinguishes the annulus from the disc unwrap.
    assert strip[0].mean() > 45.0
    # And the last row must stay inside the limbus (outer inset applied).
    assert strip[-1].mean() < 120.0
    assert strip[0].mean() < strip[-1].mean()


def test_unwrap_pentacam_annulus_angle_mapping_starts_at_zero():
    app = _make_app()
    h = w = 400
    cx = cy = 200.0
    img = _angle_map(h, w, cx, cy)
    pupil = _ellipse(cx, cy, 60.0)
    limbus = _ellipse(cx, cy, 120.0)

    strip = app._unwrap_pentacam(
        img, limbus_ellipse=limbus, pupil_ellipse=pupil
    )

    assert strip is not None
    row = strip[_MID_ROW]
    assert _angle_diff(float(row[0]), 0.0) < 5.0
    assert _angle_diff(float(row[360]), 180.0) < 5.0
    assert _angle_diff(float(row[180]), 90.0) < 5.0
    assert _angle_diff(float(row[540]), 270.0) < 5.0


def _live_ellipse(cx, cy, r):
    """The ellipse shape the live/optimised pipeline actually produces.

    ``_adapt_frame_result._make_ellipse`` returns a ``SimpleNamespace`` with a
    centre and half-axes but **no** ``is_valid``; that is exactly what reaches
    the unwrap helpers during live/video playback.
    """
    return SimpleNamespace(
        center_x=cx,
        center_y=cy,
        radius=r,
        semi_major=r,
        semi_minor=r,
        angle_deg=0.0,
        eccentricity=0.0,
        circularity=1.0,
        fit_quality=0.5,
        fit_rms_residual=0.0,
        num_contour_points=0,
        uncertainty_center_x=1.0,
        uncertainty_center_y=1.0,
        fit_type=None,
    )


def test_update_elita_unwrap_uses_annulus():
    app = _make_app()
    app._unwrap_canvas = None  # skip Tk redraw
    app._elita_unwrap_last = 0.0
    app._elita_strip = None
    h = w = 400
    cx = cy = 200.0
    app._current_image = _radius_map(h, w, cx, cy)
    # Live-shaped geometry (SimpleNamespace, no ``is_valid``): regression guard
    # for the live strip silently failing to build during playback.
    app._current_result = SimpleNamespace(
        pupil=SimpleNamespace(detected=True, ellipse=_live_ellipse(cx, cy, 60.0)),
        limbus=SimpleNamespace(detected=True, ellipse=_live_ellipse(cx, cy, 120.0)),
    )

    app._update_elita_unwrap()

    assert app._elita_strip is not None
    assert app._elita_strip.shape == _STRIP_SHAPE
    assert app._elita_strip[0].mean() > 45.0  # annulus, not the disc centre


def test_update_elita_unwrap_leaves_strip_empty_without_pupil():
    """No pupil -> no annulus, and the pupil interior is never used as a disc."""
    app = _make_app()
    app._unwrap_canvas = None
    app._elita_unwrap_last = 0.0
    app._elita_strip = None
    h = w = 400
    cx = cy = 200.0
    app._current_image = _radius_map(h, w, cx, cy)
    app._current_result = SimpleNamespace(
        pupil=SimpleNamespace(detected=False, ellipse=None),
        limbus=SimpleNamespace(detected=True, ellipse=_ellipse(cx, cy, 120.0)),
    )

    app._update_elita_unwrap()

    assert app._elita_strip is None


# ---- ellipse-geometry coercion (live strip regression) --------------------


def _ref_strip(img, cx, cy, pr, lr):
    return PupilTrackingGUI._annulus_strip(
        img,
        EllipseParams(center_x=cx, center_y=cy, semi_major=pr, semi_minor=pr),
        EllipseParams(center_x=cx, center_y=cy, semi_major=lr, semi_minor=lr),
    )


def test_annulus_strip_accepts_live_namespace_geometry():
    """Live ``SimpleNamespace`` ellipses (no ``is_valid``) must build a strip.

    ``IrisROIExtractor`` reads ``EllipseParams.is_valid``, so the live path used
    to raise, get swallowed, and leave the ELITA row permanently empty.
    """
    img = _radius_map(400, 400, 200.0, 200.0)
    pupil = _live_ellipse(200.0, 200.0, 60.0)
    limbus = _live_ellipse(200.0, 200.0, 120.0)

    assert not hasattr(pupil, "is_valid")
    assert not hasattr(limbus, "is_valid")

    strip = PupilTrackingGUI._annulus_strip(img, pupil, limbus)

    assert strip is not None
    assert strip.shape == _STRIP_SHAPE
    # Identical to the fully validated EllipseParams path.
    assert np.array_equal(strip, _ref_strip(img, 200.0, 200.0, 60.0, 120.0))


def test_annulus_strip_accepts_hand_set_circle():
    """``_make_circle`` (centre + radius only) also builds a strip."""
    img = _radius_map(400, 400, 200.0, 200.0)
    circle = SimpleNamespace(
        center_x=200.0,
        center_y=200.0,
        radius=60.0,
        semi_major=60.0,
        semi_minor=60.0,
        angle_deg=0.0,
    )

    strip = PupilTrackingGUI._annulus_strip(img, circle, _live_ellipse(200.0, 200.0, 120.0))

    assert strip is not None
    assert strip.shape == _STRIP_SHAPE


def test_annulus_strip_accepts_estimated_limbus_namespace():
    """The edge-line / estimate limbus fallback also builds a strip."""
    img = _radius_map(400, 400, 200.0, 200.0)
    estimated = SimpleNamespace(
        center_x=200.0,
        center_y=200.0,
        radius=120.0,
        semi_major=120.0,
        semi_minor=110.0,
        angle_deg=0.0,
    )

    strip = PupilTrackingGUI._annulus_strip(
        img, _live_ellipse(200.0, 200.0, 60.0), estimated
    )

    assert strip is not None
    assert strip.shape == _STRIP_SHAPE


def test_as_ellipse_params_passthrough_and_normalisation():
    keep = EllipseParams(center_x=10.0, center_y=20.0, semi_major=30.0, semi_minor=30.0)
    assert PupilTrackingGUI._as_ellipse_params(keep) is keep

    # radius-only geometry becomes a circle
    radius_only = PupilTrackingGUI._as_ellipse_params(
        SimpleNamespace(center_x=5.0, center_y=6.0, radius=7.0)
    )
    assert radius_only.is_valid
    assert radius_only.semi_major == radius_only.semi_minor == 7.0

    # swapped half-axes are reordered so ``semi_major >= semi_minor``
    swapped = PupilTrackingGUI._as_ellipse_params(
        SimpleNamespace(center_x=0.0, center_y=0.0, semi_major=10.0, semi_minor=30.0)
    )
    assert (swapped.semi_major, swapped.semi_minor) == (30.0, 10.0)

    # a ``center`` tuple is accepted, and the angle is carried over
    from_center = PupilTrackingGUI._as_ellipse_params(
        SimpleNamespace(center=(1.0, 2.0), semi_major=8.0, semi_minor=8.0, angle_deg=15.0)
    )
    assert (from_center.center_x, from_center.center_y) == (1.0, 2.0)
    assert from_center.angle_deg == 15.0


def test_as_ellipse_params_rejects_unusable_geometry():
    assert PupilTrackingGUI._as_ellipse_params(None) is None
    assert PupilTrackingGUI._as_ellipse_params(SimpleNamespace()) is None
    assert (
        PupilTrackingGUI._as_ellipse_params(
            SimpleNamespace(center_x=float("nan"), center_y=0.0, radius=5.0)
        )
        is None
    )
    assert (
        PupilTrackingGUI._as_ellipse_params(
            SimpleNamespace(center_x=0.0, center_y=0.0, radius=0.0)
        )
        is None
    )


# ---- iris zone grid (display-only overlay) --------------------------------


def test_zone_lines_default_is_twelve_sectors_over_four_quadrants():
    lines = PupilTrackingGUI._unwrap_zone_lines()
    assert _UNWRAP_ZONE_SECTORS == 12
    # 12 sectors -> 13 boundaries, 0.0 (0 deg) .. 1.0 (360 deg).
    assert len(lines) == 13
    assert lines[0][0] == 0.0
    assert lines[-1][0] == 1.0
    # 30 deg spacing: boundary k sits at k/12.
    for k, (frac, _is_quad) in enumerate(lines):
        assert abs(frac - k / 12.0) < 1e-12
    # Quadrant boundaries (every 3rd) are flagged: 0/90/180/270/360 deg.
    quad_idx = [k for k, (_f, q) in enumerate(lines) if q]
    assert quad_idx == [0, 3, 6, 9, 12]


def test_zone_lines_eight_sectors_over_four_quadrants():
    lines = PupilTrackingGUI._unwrap_zone_lines(sectors=8, quadrants=4)
    assert len(lines) == 9
    # 45 deg spacing, quadrant boundaries every 2nd boundary.
    for k, (frac, _is_quad) in enumerate(lines):
        assert abs(frac - k / 8.0) < 1e-12
    quad_idx = [k for k, (_f, q) in enumerate(lines) if q]
    assert quad_idx == [0, 2, 4, 6, 8]


def test_zone_lines_requires_at_least_one_sector_and_quadrant():
    # Degenerate inputs are coerced to a single sector = two boundaries.
    assert PupilTrackingGUI._unwrap_zone_lines(sectors=0, quadrants=0) == [
        (0.0, True),
        (1.0, True),
    ]


def test_band_fracs_default_are_three_interior_boundaries():
    assert _UNWRAP_ZONE_BANDS == 4
    assert PupilTrackingGUI._unwrap_band_fracs() == [0.25, 0.5, 0.75]


def test_band_fracs_has_bands_minus_one_boundaries():
    assert PupilTrackingGUI._unwrap_band_fracs(2) == [0.5]
    assert PupilTrackingGUI._unwrap_band_fracs(1) == []


# ---- live (ELITA) strip left/right nudge (display-only) -------------------


def test_roll_strip_wraps_horizontally():
    strip = np.arange(12).reshape(3, 4)
    rolled = PupilTrackingGUI._roll_strip(strip, 1)
    assert rolled.tolist() == [[3, 0, 1, 2], [7, 4, 5, 6], [11, 8, 9, 10]]
    # Negative shift moves content left.
    assert PupilTrackingGUI._roll_strip(strip, -1).tolist() == [
        [1, 2, 3, 0], [5, 6, 7, 4], [9, 10, 11, 8],
    ]
    # Shape is preserved (nothing cropped) and zero/None are pass-through.
    assert rolled.shape == strip.shape
    assert PupilTrackingGUI._roll_strip(strip, 0) is strip
    assert PupilTrackingGUI._roll_strip(None, 5) is None


def test_nudge_elita_shift_step_is_one_degree():
    app = _make_app()
    app._unwrap_canvas = None
    app._elita_unwrap_shift_cols = 0
    # 1 deg of a 360 deg / 720 column strip = 2 columns.
    cols_per_step = int(round(_ELITA_SHIFT_STEP_DEG / 360.0 * _UNWRAP_STRIP_SIZE[1]))
    assert cols_per_step == 2

    app._nudge_elita_shift(+1)
    assert app._elita_unwrap_shift_cols == cols_per_step
    app._nudge_elita_shift(+1)
    assert app._elita_unwrap_shift_cols == 2 * cols_per_step
    app._nudge_elita_shift(-1)
    assert app._elita_unwrap_shift_cols == cols_per_step


def test_nudge_elita_shift_clamps_to_half_turn():
    app = _make_app()
    app._unwrap_canvas = None
    app._elita_unwrap_shift_cols = 0
    half = _UNWRAP_STRIP_SIZE[1] // 2

    for _ in range(3 * half):
        app._nudge_elita_shift(+1)
    assert app._elita_unwrap_shift_cols == half

    for _ in range(3 * half):
        app._nudge_elita_shift(-1)
    assert app._elita_unwrap_shift_cols == -half


class _StubCanvas:
    """Minimal Canvas stand-in so _redraw_unwrap can run without Tk."""

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height

    def winfo_width(self) -> int:
        return self._w

    def winfo_height(self) -> int:
        return self._h

    def delete(self, *a, **k) -> None:
        pass

    def create_text(self, *a, **k) -> None:
        pass

    def create_line(self, *a, **k) -> None:
        pass

    def create_image(self, *a, **k) -> None:
        pass


def test_redraw_applies_nudge_to_live_strip_only(monkeypatch):
    """The "+"/"-" nudge moves the ELITA strip but never the Pentacam one."""
    app = _make_app()
    app._unwrap_canvas = _StubCanvas(869, 200)
    app._colors = SimpleNamespace(FG_TERTIARY="#606070")
    app._elita_unwrap_shift_cols = 0

    # A column-ramp makes any horizontal shift plainly visible.
    base = np.tile(
        np.arange(_UNWRAP_STRIP_SIZE[1], dtype=np.uint8), (_UNWRAP_STRIP_SIZE[0], 1)
    )
    app._unwrap_strip = base.copy()
    app._elita_strip = base.copy()

    captured = []

    class _Photo:
        def __init__(self, img):
            self.img = img
            captured.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_Photo))

    app._redraw_unwrap()
    pent0, live0 = np.asarray(captured[0]), np.asarray(captured[1])
    assert np.array_equal(pent0, live0)  # no nudge yet -> strips identical

    captured.clear()
    app._elita_unwrap_shift_cols = 2  # +1 deg
    app._redraw_unwrap()
    pent1, live1 = np.asarray(captured[0]), np.asarray(captured[1])
    assert np.array_equal(pent0, pent1)  # Pentacam reference never moves
    assert not np.array_equal(live0, live1)  # live strip did move


# ── Pentacam preview reliability gate (display-only) ──────────────────────


def _det_result(confidence, pupil_detected=True, limbus_detected=True):
    """Minimal stand-in for a detector result (only the gate's inputs)."""
    return SimpleNamespace(
        overall_confidence=confidence,
        pupil=SimpleNamespace(detected=pupil_detected, ellipse=None),
        limbus=SimpleNamespace(detected=limbus_detected, ellipse=None),
    )


def test_reliability_rejects_missing_result():
    assert PupilTrackingGUI._pentacam_detection_reliable(None) is False


def test_reliability_requires_a_detected_limbus():
    # No limbus ellipse -> nothing to trust, whatever the confidence.
    assert PupilTrackingGUI._pentacam_detection_reliable(
        _det_result(0.99, limbus_detected=False)
    ) is False


def test_reliability_accepts_a_confident_limbus():
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result(0.63)) is True


def test_reliability_rejects_the_report_screenshot_case():
    """Regression: a *lone* low-confidence "limbus" on a report screen.

    Here the pupil carries no ellipse, so the paired rule cannot vouch for the
    limbus and it stays untrusted (a full pupil+limbus pair is a different,
    trusted case — see test_reliability_accepts_a_coherent_pupil_limbus_pair).
    """
    result = _det_result(0.45)  # pupil and limbus both "detected"
    assert PupilTrackingGUI._pentacam_detection_reliable(result) is False


def test_reliability_threshold_boundary():
    at = _PENTACAM_MIN_TRUSTED_CONFIDENCE
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result(at)) is True
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result(at - 0.001)) is False


def test_reliability_rejects_missing_or_bad_confidence():
    assert PupilTrackingGUI._pentacam_detection_reliable(
        SimpleNamespace(pupil=None, limbus=SimpleNamespace(detected=True, ellipse=None))
    ) is False
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result(None)) is False
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result(float("nan"))) is False
    assert PupilTrackingGUI._pentacam_detection_reliable(_det_result("high")) is False


def _ellipse(cx, cy, a, b=None):
    return EllipseParams(
        center_x=cx, center_y=cy, semi_major=a, semi_minor=a if b is None else b,
    )


def _pair_result(confidence, pupil_e, limbus_e):
    return SimpleNamespace(
        overall_confidence=confidence,
        pupil=SimpleNamespace(detected=pupil_e is not None, ellipse=pupil_e),
        limbus=SimpleNamespace(detected=limbus_e is not None, ellipse=limbus_e),
    )


def test_reliability_accepts_a_coherent_pupil_limbus_pair():
    """A full pupil+limbus pair is a real eye even below the confident floor."""
    result = _pair_result(
        0.45, _ellipse(422.0, 485.0, 75.0), _ellipse(419.0, 494.0, 228.0)
    )
    assert PupilTrackingGUI._preview_pair_consistent(result) is True
    assert PupilTrackingGUI._pentacam_detection_reliable(result) is True


def test_reliability_rejects_a_pupil_outside_the_limbus():
    result = _pair_result(
        0.45, _ellipse(900.0, 800.0, 75.0), _ellipse(419.0, 494.0, 228.0)
    )
    assert PupilTrackingGUI._preview_pair_consistent(result) is False
    assert PupilTrackingGUI._pentacam_detection_reliable(result) is False


def test_reliability_rejects_a_pupil_larger_than_the_limbus():
    result = _pair_result(
        0.45, _ellipse(419.0, 494.0, 300.0), _ellipse(419.0, 494.0, 228.0)
    )
    assert PupilTrackingGUI._pentacam_detection_reliable(result) is False


def test_reliability_rejects_a_pair_below_the_paired_floor():
    below = _PENTACAM_MIN_PAIRED_CONFIDENCE - 0.01
    result = _pair_result(
        below, _ellipse(422.0, 485.0, 75.0), _ellipse(419.0, 494.0, 228.0)
    )
    assert PupilTrackingGUI._pentacam_detection_reliable(result) is False


def test_preview_pair_consistent_needs_both_ellipses():
    result = SimpleNamespace(
        pupil=SimpleNamespace(detected=True, ellipse=None),
        limbus=SimpleNamespace(detected=True, ellipse=None),
    )
    assert PupilTrackingGUI._preview_pair_consistent(result) is False


def test_unwrap_pentacam_is_none_when_limbus_is_dropped():
    """`_refresh_unwrap` passes both boundaries as None when the detection is
    unconfirmed; the annulus-only strip must then stay empty rather than
    flattening a bogus region.
    """
    app = _make_app()
    h = w = 400
    img = _angle_map(h, w, 200.0, 200.0)

    strip = app._unwrap_pentacam(
        img, limbus_ellipse=None, pupil_ellipse=None
    )

    assert strip is None


# ── Edge-line limbus reference (display-only fallback) ────────────────────


def _eye_with_edge_lines(h=240, w=240, cx=120.0, cy=120.0, r=60.0) -> np.ndarray:
    """A bright field with a dark disc ringed by a black and a white line.

    This is the "two black/white lines on the left and right curve of the eye"
    the detector is asked to use as a limbus reference.
    """
    img = np.full((h, w, 3), 235, np.uint8)  # bright sclera / surround
    cv2.circle(img, (int(cx), int(cy)), int(r), (60, 60, 60), -1)  # iris
    cv2.circle(img, (int(cx), int(cy)), int(r), (0, 0, 0), 3)  # black line
    cv2.circle(img, (int(cx), int(cy)), int(r) + 4, (255, 255, 255), 3)  # white line
    return img


def test_edge_line_limbus_recovers_the_eye_circle():
    img = _eye_with_edge_lines()
    edge = PupilTrackingGUI._edge_line_limbus(img, (120.0, 120.0), 30.0, 130.0)
    assert edge is not None
    assert abs(edge.center_x - 120.0) <= 4.0
    assert abs(edge.center_y - 120.0) <= 4.0
    assert abs(edge.semi_major - 60.0) <= 8.0
    assert edge.semi_major == edge.semi_minor  # edge reference is a circle


def test_edge_line_limbus_rejects_a_smooth_gradient():
    """A report-screen gradient has no black/white eye line pair."""
    img = np.tile(np.linspace(0, 255, 240, dtype=np.uint8), (240, 1))
    assert PupilTrackingGUI._edge_line_limbus(img, (120.0, 120.0), 20.0, 100.0) is None


def test_edge_line_limbus_needs_lines_on_both_sides():
    """One line alone (e.g. a table border) is not an eye."""
    img = np.full((240, 240, 3), 235, np.uint8)
    img[:, 40:60] = 0  # a single dark vertical line on the left
    assert PupilTrackingGUI._edge_line_limbus(img, (140.0, 120.0), 20.0, 90.0) is None


def test_preview_edge_limbus_without_anchors_returns_none():
    img = _eye_with_edge_lines()
    assert PupilTrackingGUI._preview_edge_limbus(img, None, None) is None


def test_preview_edge_limbus_uses_the_pupil_anchor():
    img = _eye_with_edge_lines()
    result = SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=EllipseParams(
                center_x=120.0, center_y=120.0, semi_major=30.0, semi_minor=30.0,
            ),
        ),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )
    edge = PupilTrackingGUI._preview_edge_limbus(img, result)
    assert edge is not None
    assert abs(edge.center_x - 120.0) <= 4.0
    assert abs(edge.semi_major - 60.0) <= 8.0


def test_unwrap_limbus_choice_trusts_a_reliable_detection():
    limbus_e = EllipseParams(center_x=1.0, center_y=2.0, semi_major=3.0, semi_minor=3.0)
    pupil_e = EllipseParams(center_x=1.0, center_y=2.0, semi_major=1.0, semi_minor=1.0)
    result = SimpleNamespace(
        limbus=SimpleNamespace(detected=True, ellipse=limbus_e),
        pupil=SimpleNamespace(detected=True, ellipse=pupil_e),
    )
    assert PupilTrackingGUI._unwrap_limbus_choice(result, True) == (limbus_e, pupil_e)


def test_unwrap_limbus_choice_drops_a_bogus_detection():
    result = SimpleNamespace(
        limbus=SimpleNamespace(detected=True, ellipse=object()),
        pupil=SimpleNamespace(detected=True, ellipse=object()),
    )
    assert PupilTrackingGUI._unwrap_limbus_choice(result, False) == (None, None)


def test_unwrap_limbus_choice_uses_the_edge_reference():
    edge = SimpleNamespace(
        center_x=5.0, center_y=6.0, radius=7.0,
        semi_major=7.0, semi_minor=7.0, angle_deg=0.0,
    )
    result = SimpleNamespace(
        limbus=SimpleNamespace(detected=True, ellipse=object()),
        pupil=SimpleNamespace(detected=True, ellipse=object()),
    )
    assert PupilTrackingGUI._unwrap_limbus_choice(result, False, edge) == (edge, None)


def test_preview_limbus_ellipse_trusts_a_reliable_detection():
    limbus_e = EllipseParams(center_x=1.0, center_y=2.0, semi_major=3.0, semi_minor=3.0)
    result = SimpleNamespace(limbus=SimpleNamespace(detected=True, ellipse=limbus_e))
    assert PupilTrackingGUI._preview_limbus_ellipse(result, True) is limbus_e


def test_preview_limbus_ellipse_ignores_an_undetected_limbus():
    result = SimpleNamespace(limbus=SimpleNamespace(detected=False, ellipse=object()))
    assert PupilTrackingGUI._preview_limbus_ellipse(result, True) is None


def test_preview_limbus_ellipse_falls_back_only_when_unreliable():
    edge = SimpleNamespace(
        center_x=5.0, center_y=6.0, radius=7.0,
        semi_major=7.0, semi_minor=7.0, angle_deg=0.0,
    )
    # A bogus (unreliable) detection is dropped in favour of the estimate ...
    bogus = SimpleNamespace(limbus=SimpleNamespace(detected=True, ellipse=object()))
    assert PupilTrackingGUI._preview_limbus_ellipse(bogus, False, edge) is edge
    # ... but a reliable detection never lets the estimate override it.
    good = SimpleNamespace(limbus=SimpleNamespace(detected=True, ellipse=object()))
    assert PupilTrackingGUI._preview_limbus_ellipse(good, True, edge) is good.limbus.ellipse


def test_preview_limbus_ellipse_none_without_any_source():
    assert PupilTrackingGUI._preview_limbus_ellipse(None, True) is None
    assert PupilTrackingGUI._preview_limbus_ellipse(None, False) is None


# -- widen a trusted Pentacam ring to the eye's left/right edge lines ---------
# A Pentacam *report screenshot* is flat and edge-less, so the detector can
# return a limbus that is too small ("underestimated on the left and right").
# The eye's left/right edges are the two strong dark/bright vertical lines that
# bracket it, so a trusted ring is widened horizontally to reach them -- but
# only when BOTH lines lie clear of the ring, so a correct ring on a real eye
# (which re-finds its own limbus edge) is never stretched.


def _eye_with_side_lines(
    x_left: int, x_right: int, w: int = 400, h: int = 300, thickness: int = 4
) -> np.ndarray:
    """Bright eye between two dark vertical border lines."""
    img = np.full((h, w, 3), 200, np.uint8)
    for x in (x_left, x_right):
        img[:, max(0, x):x + thickness] = 0
    return img


def test_eye_edge_lines_finds_the_two_bright_vertical_borders():
    img = _eye_with_side_lines(100, 300)
    left, right = PupilTrackingGUI._eye_edge_lines(img, 200.0, 150.0, 80.0)
    assert left is not None and right is not None
    assert abs(left - 100) <= 6
    assert abs(right - 300) <= 6


def test_eye_edge_lines_rejects_a_flat_gradient():
    gray = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (300, 1))
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert PupilTrackingGUI._eye_edge_lines(img, 200.0, 150.0, 80.0) == (None, None)


def test_eye_edge_lines_ignores_the_image_frame():
    img = np.full((300, 400, 3), 200, np.uint8)
    img[:, 0:3] = 0          # the frame is a perfect vertical edge, not an eye
    img[:, 397:400] = 0
    assert PupilTrackingGUI._eye_edge_lines(img, 200.0, 150.0, 80.0) == (None, None)


def test_widen_limbus_horizontally_stretches_to_the_edge_lines():
    img = _eye_with_side_lines(100, 300)
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=80.0
    )
    widened = PupilTrackingGUI._widen_limbus_horizontally(img, ring)
    assert widened is not ring
    assert abs(widened.center_x - 200.0) <= 6
    assert 92.0 <= widened.semi_major <= 108.0
    # Only the left and right grow: the vertical extent is preserved.
    assert abs(widened.semi_minor - 80.0) < 1e-6


def test_widen_limbus_horizontally_leaves_a_correct_ring_alone():
    # The "lines" here are the ring's own limbus edge, so it must not stretch.
    img = _eye_with_side_lines(120, 280)
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=80.0
    )
    assert PupilTrackingGUI._widen_limbus_horizontally(img, ring) is ring


def test_widen_limbus_horizontally_never_shrinks():
    img = _eye_with_side_lines(126, 274)   # lines inside the ring
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=80.0
    )
    assert PupilTrackingGUI._widen_limbus_horizontally(img, ring) is ring


def test_widen_limbus_horizontally_skips_a_tilted_ellipse():
    img = _eye_with_side_lines(100, 300)
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=60.0,
        angle_deg=30.0,
    )
    assert PupilTrackingGUI._widen_limbus_horizontally(img, ring) is ring


def test_preview_limbus_ellipse_uses_the_raw_trusted_ring():
    # By default the detector's own ring is shown verbatim -- exactly what the
    # main window draws -- even when widening is available.
    img = _eye_with_side_lines(100, 300)
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=80.0
    )
    result = SimpleNamespace(limbus=SimpleNamespace(detected=True, ellipse=ring))
    assert PupilTrackingGUI._preview_limbus_ellipse(result, True, None, img) is ring
    # An unconfirmed detection's fallback is already derived from the eye's own
    # curves, so it is not stretched either.
    edge = SimpleNamespace(
        center_x=1.0, center_y=2.0, radius=3.0,
        semi_major=3.0, semi_minor=3.0, angle_deg=0.0,
    )
    assert PupilTrackingGUI._preview_limbus_ellipse(result, False, edge, img) is edge


def test_preview_limbus_ellipse_widens_only_a_trusted_ring(monkeypatch):
    monkeypatch.setattr(PupilTrackingGUI, "_PENTACAM_WIDEN_TRUSTED_LIMBUS", True)
    img = _eye_with_side_lines(100, 300)
    ring = EllipseParams(
        center_x=200.0, center_y=150.0, semi_major=80.0, semi_minor=80.0
    )
    result = SimpleNamespace(limbus=SimpleNamespace(detected=True, ellipse=ring))
    widened = PupilTrackingGUI._preview_limbus_ellipse(result, True, None, img)
    assert widened is not None
    assert widened.semi_major > 80.0
    # Even with widening on, an unconfirmed detection's fallback is already
    # derived from the eye's own curves, so it is not stretched a second time.
    edge = SimpleNamespace(
        center_x=1.0, center_y=2.0, radius=3.0,
        semi_major=3.0, semi_minor=3.0, angle_deg=0.0,
    )
    assert PupilTrackingGUI._preview_limbus_ellipse(result, False, edge, img) is edge


# ── Fallback limbus estimate (never leave the blue ring missing) ──────────


def _pupil_result(center_x, center_y, semi_major, semi_minor=None):
    """A detector result with a pupil and **no** limbus (the low-confidence case)."""
    if semi_minor is None:
        semi_minor = semi_major
    return SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=EllipseParams(
                center_x=center_x, center_y=center_y,
                semi_major=semi_major, semi_minor=semi_minor,
            ),
        ),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )


def test_estimate_limbus_scales_from_the_pupil():
    """No limbus detected -> the ring is placed from the pupil, not dropped."""
    img = np.zeros((400, 400, 3), np.uint8)
    est, note = PupilTrackingGUI._preview_estimate_limbus(
        img, _pupil_result(200.0, 200.0, 40.0)
    )
    assert est is not None
    assert note == "estimated from the pupil"
    assert abs(est.center_x - 200.0) < 1e-6 and abs(est.center_y - 200.0) < 1e-6
    assert abs(est.semi_major - _PENTACAM_LIMBUS_PUPIL_RATIO * 40.0) < 1e-6


def test_estimate_limbus_keeps_the_pupil_axis_ratio():
    img = np.zeros((400, 400, 3), np.uint8)
    est, _ = PupilTrackingGUI._preview_estimate_limbus(
        img, _pupil_result(200.0, 200.0, 40.0, 30.0)
    )
    assert est is not None
    assert abs(est.semi_minor / est.semi_major - 0.75) < 1e-6


def test_estimate_limbus_clamps_to_the_image():
    """A pupil near the border must not push the ring off screen."""
    img = np.zeros((400, 400, 3), np.uint8)
    est, note = PupilTrackingGUI._preview_estimate_limbus(
        img, _pupil_result(80.0, 200.0, 40.0)
    )
    assert est is not None and note == "estimated from the pupil"
    # 0.92 * min(80, 200, 320, 200) = 73.6, i.e. clamped below 2.6*40 = 104.
    assert est.semi_major < _PENTACAM_LIMBUS_PUPIL_RATIO * 40.0
    assert est.center_x - est.semi_major >= -1.0


def test_estimate_limbus_rejects_a_pupil_that_leaves_no_room():
    """If clamping would make the ring no bigger than the pupil, give up."""
    img = np.zeros((400, 400, 3), np.uint8)
    est, note = PupilTrackingGUI._preview_estimate_limbus(
        img, _pupil_result(50.0, 200.0, 40.0)
    )
    # 0.92 * 50 = 46 < 1.25 * 40 = 50 -> nothing sensible to draw.
    assert est is None and note == ""


def test_preview_roi_limbus_inscribes_the_box():
    img = np.zeros((400, 400, 3), np.uint8)
    roi = PupilTrackingGUI._preview_roi_limbus(img, (0.25, 0.25, 0.75, 0.75))
    assert roi is not None
    assert abs(roi.center_x - 200.0) < 1e-6 and abs(roi.center_y - 200.0) < 1e-6
    assert abs(roi.semi_major - 100.0) < 1e-6 and abs(roi.semi_minor - 100.0) < 1e-6
    # Degenerate boxes are ignored.
    assert PupilTrackingGUI._preview_roi_limbus(img, (0.5, 0.5, 0.5001, 0.9)) is None
    assert PupilTrackingGUI._preview_roi_limbus(img, None) is None


def test_estimate_limbus_without_pupil_returns_none():
    img = np.zeros((400, 400, 3), np.uint8)
    assert PupilTrackingGUI._preview_estimate_limbus(img, None) == (None, "")
    assert PupilTrackingGUI._preview_estimate_limbus(img, _no_limbus(0.0, 0.0, 0.0)) == (
        None, "",
    )


def _no_limbus(cx, cy, r):
    return SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=EllipseParams(
                center_x=cx, center_y=cy, semi_major=r, semi_minor=r,
            ),
        ),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )


def test_fallback_limbus_uses_the_iris_roi_box_when_drawn():
    """An explicit ROI box beats the pupil-based guess."""
    img = np.full((400, 400, 3), 128, np.uint8)
    est, note = PupilTrackingGUI._preview_fallback_limbus(
        img, _pupil_result(200.0, 200.0, 40.0), (0.25, 0.25, 0.75, 0.75)
    )
    assert est is not None and note == "from the iris ROI box"
    assert abs(est.semi_major - 100.0) < 1e-6


def test_fallback_limbus_uses_the_roi_box_without_a_pupil():
    img = np.full((400, 400, 3), 128, np.uint8)
    est, note = PupilTrackingGUI._preview_fallback_limbus(
        img, None, (0.25, 0.25, 0.75, 0.75)
    )
    assert est is not None and note == "from the iris ROI box"


def test_fallback_limbus_prefers_the_eye_edge_lines():
    """With the two dark/bright eye lines present, those win over the estimate."""
    img = _eye_with_edge_lines(h=240, w=240, cx=120.0, cy=120.0, r=60.0)
    result = SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=EllipseParams(
                center_x=120.0, center_y=120.0, semi_major=30.0, semi_minor=30.0,
            ),
        ),
        limbus=SimpleNamespace(detected=False, ellipse=None),
    )
    est, note = PupilTrackingGUI._preview_fallback_limbus(img, result)
    assert est is not None and note == "from the eye edge lines"
    assert abs(est.semi_major - 60.0) <= 8.0


def test_fallback_limbus_estimates_from_the_pupil_without_edge_lines():
    """A plain image has no eye lines, so the pupil-scaled estimate is used."""
    img = np.full((400, 400, 3), 128, np.uint8)
    est, note = PupilTrackingGUI._preview_fallback_limbus(
        img, _pupil_result(200.0, 200.0, 40.0)
    )
    assert est is not None and note == "estimated from the pupil"


def test_fallback_limbus_without_anything_returns_none():
    img = np.zeros((400, 400, 3), np.uint8)
    assert PupilTrackingGUI._preview_fallback_limbus(img, None, None) == (None, "")


# ── Manual ("- Ring" / "+ Ring") sizing of the detection ring ────────────────
# The Add Patient preview can size the blue limbus ring by hand.  The buttons
# share the mouse wheel's clamping helper; neither touches the pipeline.


def test_ring_step_constants_are_positive_and_buttons_are_coarser():
    """The button step is deliberately coarser than one wheel notch."""
    assert gui_app._RING_WHEEL_STEP > 0.0
    assert gui_app._RING_BTN_STEP > gui_app._RING_WHEEL_STEP


def test_scale_ring_radius_grows_and_shrinks():
    step = gui_app._RING_BTN_STEP
    grew = _scale_ring_radius(100.0, 1.0 + step, 200.0, 200.0)
    shrank = _scale_ring_radius(100.0, 1.0 - step, 200.0, 200.0)
    assert grew == pytest.approx(100.0 * (1.0 + step))
    assert shrank == pytest.approx(100.0 * (1.0 - step))
    assert grew > 100.0 > shrank


def test_scale_ring_radius_clamps_to_the_image_and_a_floor():
    # Never past the frame diagonal: hypot(200, 200) is the ceiling here.
    ceiling = math.hypot(200.0, 200.0)
    assert _scale_ring_radius(1.0e6, 2.0, 200.0, 200.0) == pytest.approx(ceiling)
    # Never below a few pixels, however hard the user shrinks.
    assert _scale_ring_radius(10.0, 1.0e-6, 200.0, 200.0) == 8.0


# ── pulling part of a ring out (per-angle profile) ───────────────────────────


def test_ring_radius_without_a_profile_is_a_plain_circle():
    """No profile: every angle reports the nominal radius."""
    for angle in (0.0, 37.5, 90.0, 180.0, 271.3, 359.9):
        assert gui_app._ring_radius_at(120.0, None, angle) == pytest.approx(120.0)
    arr = gui_app._ring_radius_at(120.0, None, np.array([0.0, 90.0, 180.0]))
    assert np.allclose(arr, 120.0)


def test_ring_bulge_reaches_the_pointer_at_the_grab_and_leaves_the_rest_alone():
    """A drag only moves the arc under the pointer, not the whole circle."""
    r = 100.0
    prof = gui_app._ring_bulge_profile(None, r, 90.0, 200.0, 8.0, 5000.0)
    # exactly on target at the grab angle ...
    assert gui_app._ring_radius_at(r, prof, 90.0) == pytest.approx(200.0)
    # ... and untouched on the far side of the ring
    for angle in (0.0, 180.0, 270.0):
        assert gui_app._ring_radius_at(r, prof, angle) == pytest.approx(r)
    # the bulge fades out at the edge of its span
    span = gui_app._RING_BULGE_SPAN_DEG
    assert gui_app._ring_radius_at(r, prof, 90.0 - span) == pytest.approx(r)
    assert gui_app._ring_radius_at(r, prof, 90.0 + span) == pytest.approx(r)
    # inside the span it has moved but not yet fully
    inside = gui_app._ring_radius_at(r, prof, 90.0 - span / 2.0)
    assert r < inside < 200.0


def test_ring_bulge_can_push_inwards_too():
    r = 100.0
    out = gui_app._ring_bulge_profile(None, r, 0.0, 180.0, 8.0, 5000.0)
    back = gui_app._ring_bulge_profile(out, r, 0.0, 60.0, 8.0, 5000.0)
    assert gui_app._ring_radius_at(r, back, 0.0) == pytest.approx(60.0)
    assert gui_app._ring_radius_at(r, back, 180.0) == pytest.approx(r)


def test_ring_bulge_is_clamped_like_the_uniform_sizing():
    r, lo, hi = 100.0, 8.0, 500.0
    big = gui_app._ring_bulge_profile(None, r, 0.0, 1.0e6, lo, hi)
    small = gui_app._ring_bulge_profile(None, r, 0.0, 1.0, lo, hi)
    assert gui_app._ring_radius_at(r, big, 0.0) == pytest.approx(hi)
    assert gui_app._ring_radius_at(r, small, 0.0) == pytest.approx(lo)


def test_ring_profile_wraps_smoothly_across_zero_degrees():
    """The profile is periodic, so there is no seam at 0°/360°."""
    prof = gui_app._ring_bulge_profile(None, 100.0, 5.0, 200.0, 8.0, 5000.0)
    # just either side of the wrap the two radii agree
    before = gui_app._ring_radius_at(100.0, prof, 359.5)
    after = gui_app._ring_radius_at(100.0, prof, 0.5)
    assert before == pytest.approx(after, abs=2.0)
    # and the wrap-around interpolation is finite everywhere
    for angle in np.linspace(0.0, 360.0, 181):
        assert np.isfinite(gui_app._ring_radius_at(100.0, prof, float(angle)))


def test_ring_profile_of_the_wrong_length_is_resampled():
    prof = np.ones(8, dtype=float)
    prof[2] = 2.0
    out = gui_app._ring_profile(prof, gui_app._RING_SAMPLES)
    assert out.size == gui_app._RING_SAMPLES
    assert out.max() == pytest.approx(2.0)
    assert out.min() == pytest.approx(1.0)


def test_ring_polygon_points_follow_the_profile():
    """The overlay polygon is the pulled shape, not a circle through it."""
    ring = {"cx": 200.0, "cy": 200.0, "r": 100.0}
    prof = gui_app._ring_bulge_profile(None, 100.0, 90.0, 200.0, 8.0, 5000.0)
    ring["profile"] = prof
    pts = gui_app._ring_polygon_points(ring, 1.0, 1.0)
    assert len(pts) == 2 * gui_app._RING_SAMPLES
    # sample index for 90 degrees (the grab), relative to the (2, 2) margin
    idx = int(round(90.0 / 360.0 * gui_app._RING_SAMPLES)) % gui_app._RING_SAMPLES
    x, y = pts[2 * idx], pts[2 * idx + 1]
    assert math.hypot(x - 202.0, y - 202.0) == pytest.approx(200.0)
    # a circle (no profile) still draws at the nominal radius
    circle = gui_app._ring_polygon_points(
        {"cx": 200.0, "cy": 200.0, "r": 100.0}, 1.0, 1.0
    )
    assert math.hypot(circle[0] - 202.0, circle[1] - 202.0) == pytest.approx(100.0)


def test_annulus_strip_follows_a_pulled_limbus_boundary():
    """A per-angle outer boundary yields a strip instead of a circle fit."""
    image = np.full((240, 240, 3), 40, dtype=np.uint8)
    pupil = SimpleNamespace(
        center_x=120.0, center_y=120.0, radius=30.0,
        semi_major=30.0, semi_minor=30.0, angle_deg=0.0,
    )
    limbus = SimpleNamespace(
        center_x=120.0, center_y=120.0, radius=90.0,
        semi_major=90.0, semi_minor=90.0, angle_deg=0.0,
    )
    plain = PupilTrackingGUI._annulus_strip(image, pupil, limbus)
    assert plain is not None
    # the same geometry with a pulled outer boundary still unwraps
    pulled = PupilTrackingGUI._annulus_strip(
        image, pupil, limbus,
        limbus_radius_fn=lambda a: 90.0 + 30.0 * math.cos(math.radians(a)),
    )
    assert pulled is not None
    assert pulled.shape == plain.shape


def test_source_has_ring_sizing_buttons_wired_to_the_nudge_handler():
    """The "+ Ring" button grows the ring, "- Ring" shrinks it (display-only)."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert 'text="+ Ring"' in src
    assert 'text="- Ring"' in src
    assert "ring_plus_btn.config(command=lambda: _nudge_ring(+1))" in src
    assert "ring_minus_btn.config(command=lambda: _nudge_ring(-1))" in src
    # The handler only re-sizes / redraws; it never invokes detection.
    body = src.split("def _nudge_ring(")[1].split("\n        def ")[0]
    assert "_scale_ring" in body
    assert "detect" not in body


# ── Pentacam preview window sized like the Add Patient window ────────────────


def test_preview_geometry_matches_the_dialog_size():
    """The preview opens as big as the Add Patient window (was 560x300)."""
    assert gui_app._preview_geometry(640, 510, 420, 360, 1920, 8) == (640, 510)


def test_preview_geometry_never_shrinks_below_its_contents():
    """If the panel needs more room than the dialog, its own size wins."""
    w, h = gui_app._preview_geometry(640, 510, 900, 700, 1920, 8)
    assert (w, h) == (900, 700)


def test_preview_geometry_clamps_so_the_pair_fits_the_screen():
    """The docked pair is never wider than the screen (1200px here)."""
    screen_w, dw, bx = 1200, 640, 8
    pw, _ph = gui_app._preview_geometry(dw, 510, 300, 300, screen_w, bx)
    assert dw + 2 * bx + pw <= screen_w
    assert pw >= 300  # ...but still shows its contents


def test_preview_window_is_sized_and_resynced_with_the_dialog():
    """The preview gets the dialog size on open and whenever it is resized."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert 'preview_win.geometry(f"{W}x{H}")' in src
    # _dock_preview must pass a full size+position, not just a position.
    dock = src.split("def _dock_preview(")[1].split("\n        def ")[0]
    assert 'f"{pw}x{ph}+{px}+{py}"' in dock


def test_preview_thumb_limits_fit_the_canvas():
    """The picture fills the canvas, so it grows with the bigger window."""
    assert gui_app._preview_thumb_limits(600, 400) == (592, 392)


def test_preview_thumb_limits_fall_back_to_the_old_fixed_size():
    """Before the canvas is laid out the old 556x196 limits keep it safe."""
    assert gui_app._preview_thumb_limits(0, 0) == (556, 196)
    # A degenerate canvas (zero height mid-layout) also falls back.
    assert gui_app._preview_thumb_limits(600, 4) == (556, 196)


def test_preview_image_refits_when_the_canvas_is_resized():
    """The image is re-fitted on <Configure> (never re-running detection)."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert 'preview_canvas.bind("<Configure>", _fit_preview_to_canvas, add="+")' in src
    fit = src.split("def _fit_preview_to_canvas(")[1].split("\n        def ")[0]
    assert "_render_preview_image()" in fit
    # Display-only: a resize redraws the picture, it never re-runs detection.
    assert "detector" not in fit
    assert "detect(" not in fit.replace("_draw_detection_overlay()", "")


# ── hand-adjusting BOTH rings (blue limbus + green pupil) ────────────────────
#
# The unwrap strip is the annulus *between* the green pupil ring and the blue
# limbus ring, so both boundaries have to be placeable by hand -- not just the
# limbus.  The controls act on whichever ring the selector points at.


def test_source_has_a_limbus_pupil_ring_selector():
    """The sizing buttons act on a *chosen* ring, not only the blue limbus."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert '(("limbus", "Limbus"), ("pupil", "Pupil"))' in src
    assert 'text="Adjust:"' in src
    assert "def _select_ring_target(" in src
    assert 'ring_target_var.set("pupil" if target == "pupil" else "limbus")' in src


def test_source_sizing_and_dragging_follow_the_selected_ring():
    """- Ring / + Ring, the wheel and the right-drag all target that ring."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)

    scale = src.split("def _scale_ring(")[1].split("\n        def ")[0]
    assert "_override_of(target)" in scale

    press = src.split("def _on_ring_press(")[1].split("\n        def ")[0]
    assert "target" in press
    # the drag remembers which ring it started on
    assert "tw, th, eye, target," in press

    drag = src.split("def _on_ring_drag(")[1].split("\n        def ")[0]
    assert "_override_of(target)" in drag


def test_source_has_a_separate_pupil_override():
    """The green pupil ring gets its own hand-set override, like the limbus."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert 'pupil_override: Dict[str, Optional[Dict[str, float]]] = {"OD": None, "OS": None}' in src

    overlay = src.split("def _draw_detection_overlay(")[1].split("\n        def ")[0]
    # the green ring is drawn from the override when one exists
    assert "manual_pupil" in overlay
    assert "if manual_pupil is not None:" in overlay
    # either hand-set ring is honoured, even when the detection is unconfirmed
    assert "if manual is not None or manual_pupil is not None:" in overlay


def test_source_unwrap_uses_the_hand_set_pupil_ring():
    """The annulus inner boundary follows the hand-set pupil ring too."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    unwrap = src.split("def _refresh_unwrap(")[1].split("\n        def ")[0]
    assert "manual_pupil" in unwrap
    assert "pupil_ellipse = _make_circle(" in unwrap
    # a *pulled* ring drives the strip through a per-angle radius function
    assert "pupil_radius_fn" in unwrap
    assert "limbus_radius_fn" in unwrap
    assert "_ring_radius_at(" in unwrap


def test_source_reset_ring_clears_only_the_selected_ring():
    """Reset hands back the ring the selector points at -- not both."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    reset = src.split("def _reset_ring(")[1].split("\n        def ")[0]
    assert "_override_of(_ring_target())[eye] = None" in reset
    # it must not clear the limbus override directly any more
    assert "ring_override[eye] = None" not in reset
    assert "pupil_override[eye] = None" not in reset


def test_source_detected_ring_knows_both_rings():
    """_detected_ring seeds either ring from the matching detection."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    detected = src.split("def _detected_ring(")[1].split("\n        def ")[0]
    assert 'if target == "pupil":' in detected
    assert 'getattr(pupil, "detected", False)' in detected


def test_source_seed_ring_falls_back_inside_the_limbus_for_the_pupil():
    """With no pupil detected the seed sits inside the blue ring, concentric."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    seed = src.split("def _seed_ring(")[1].split("\n        def ")[0]
    assert 'if target == "pupil":' in seed
    assert "0.4 * float(limbus[\"r\"])" in seed


# ── left-drag on the ring outline pulls that part of it out ──────────────────


def test_ring_grab_tolerance_is_a_narrow_band():
    """The grab band is a few pixels wide, so ROI drawing still has room."""
    assert 0.0 < gui_app._RING_GRAB_PX <= 20.0


def test_source_left_drag_on_the_outline_pulls_the_ring():
    """A left-press on the outline pulls it; the ROI handlers defer to it."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)

    # the ROI press/drag/release all hand the gesture to the ring first
    press = src.split("def _on_roi_press(")[1].split("\n        def ")[0]
    assert "_start_ring_extend(event.x, event.y)" in press
    drag = src.split("def _on_roi_drag(")[1].split("\n        def ")[0]
    assert "_ring_extend_drag(event.x, event.y)" in drag
    release = src.split("def _on_roi_release(")[1].split("\n        def ")[0]
    assert "_ring_extend_end()" in release

    # the press must land ON the outline, not merely inside the ring, and the
    # outline it tests is the ring *as drawn* (a pulled ring is not a circle)
    outline = src.split("def _on_ring_outline(")[1].split("\n        def ")[0]
    assert "_ring_canvas_radius(ring, sx, sy, angle)" in outline
    assert "_RING_GRAB_PX" in outline


def test_source_ring_extend_pulls_only_the_grabbed_arc():
    """A drag bulges the arc under the pointer; the rest of the ring stays."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    drag = src.split("def _ring_extend_drag(")[1].split("\n        def ")[0]
    # the grabbed angle is remembered and drives a per-angle profile
    assert "grab" in drag
    assert "_ring_bulge_profile(" in drag
    # the centre and nominal radius are preserved (only the arc moves)
    assert "_ring_dict(cx, cy, ring[\"r\"], profile)" in drag
    # display-only: it redraws the strip/overlay, it never re-runs detection
    assert "detect(" not in drag
    assert "_draw_detection_overlay()" in drag
    assert "_refresh_unwrap()" in drag
    assert "_scale_ring_radius(" not in drag


def test_source_ring_screen_geometry_grabs_what_is_drawn():
    """The hit-test follows the hand-set ring, else the drawn detection."""
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    geom = src.split("def _ring_screen_geometry(")[1].split("\n        def ")[0]
    assert "_override_of(target).get(eye) or _detected_ring(target)" in geom
    # the preview draws the image with display offset (ox, oy), so the centre is offset too
    assert 'float(ring["cx"]) * sx' in geom
    # ... and the pointer angle / drawn radius are handed back with it
    assert "ring," in geom
    assert "sx," in geom


def test_source_ring_drag_state_is_declared_and_hover_shows_a_cursor():
    src = inspect.getsource(gui_app.PupilTrackingGUI)
    assert '_ring_extend: Dict[str, Any] = {"origin": None}' in src
    assert 'preview_canvas.bind("<Motion>", _on_ring_hover)' in src
    hover = src.split("def _on_ring_hover(")[1].split("\n        def ")[0]
    assert '"sizing"' in hover


# ── the same two rings, driven through the real dialog ───────────────────────


def _tk_usable() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def _find_button(widget, text):
    """The first Button labelled exactly *text*, searched depth-first."""
    import tkinter as tk

    for child in widget.winfo_children():
        if isinstance(child, tk.Button) and child.cget("text") == text:
            return child
        hit = _find_button(child, text)
        if hit is not None:
            return hit
    return None


def _preview_overlay(canvas):
    """(blue rings, green rings, badge texts) drawn on a preview canvas."""
    blue = green = 0
    texts = []
    for item in canvas.find_withtag("detection"):
        kind = canvas.type(item)
        if kind == "polygon":
            outline = canvas.itemcget(item, "outline")
            blue += outline == "#5c8aff"
            green += outline == "#00e676"
        elif kind == "text":
            texts.append(canvas.itemcget(item, "text"))
    return blue, green, texts


class _StubDetector:
    """A detector that finds nothing: the rings must then be hand-placed."""

    def detect(self, img, frame_number=-1, source=""):
        return None


@pytest.mark.skipif(not _tk_usable(), reason="Tk unavailable in this environment")
def test_both_rings_can_be_hand_adjusted_in_the_preview(tmp_path):
    """Driving the real dialog: Pupil adjusts the green ring, Limbus the blue.

    Each is independent: "Reset ring" hands back only the selected one.
    Display-only -- the detection pipeline is never touched.
    """
    import logging
    import time
    import tkinter as tk

    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.circle(img, (160, 120), 90, (40, 40, 40), -1)
    path = tmp_path / "ring_eye.png"
    cv2.imwrite(str(path), img)

    class _FakeApp:
        # Re-use the real, pure helpers; stub everything needing a live app.
        _preview_ellipse_points = staticmethod(
            gui_app.PupilTrackingGUI._preview_ellipse_points
        )
        _preview_limbus_ellipse = staticmethod(
            gui_app.PupilTrackingGUI._preview_limbus_ellipse
        )
        _preview_fallback_limbus = staticmethod(
            lambda img, res, roi=None: (None, "")
        )
        _pentacam_detection_reliable = staticmethod(lambda res: False)
        _unwrap_limbus_choice = staticmethod(lambda *a, **k: (None, None))
        _unwrap_pentacam = staticmethod(lambda *a, **k: None)
        _set_unwrap_strip = staticmethod(lambda *a, **k: None)
        _sync_calibration_to_detector = staticmethod(lambda: None)
        _update_patient_controls = staticmethod(lambda: None)
        _safe_folder_name = staticmethod(lambda s: s or "x")
        _save_storage_settings = staticmethod(lambda: True)

        def __init__(self, root):
            self.root = root
            self._recorder = SimpleNamespace(is_recording=False)
            self._colors = None
            self._add_px_dialog_open = False
            self.logger = logging.getLogger("test_rings")
            self._detector = _StubDetector()
            self._storage_root_var = tk.StringVar(value="")

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            root.update()
            time.sleep(0.02)

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover - no usable Tk in this environment
        pytest.skip("Tk could not be initialised in this environment")
    root.withdraw()
    try:
        app = _FakeApp(root)
        gui_app.PupilTrackingGUI._open_patient_dialog(app)
        pump(0.4)

        tops = {
            w.title(): w
            for w in root.winfo_children()
            if isinstance(w, tk.Toplevel)
        }
        dialog = tops["Add Patient"]
        preview = tops.get("Pentacam Preview", dialog)

        # load a Pentacam image so the preview (and the rings) have something
        entries = [
            e for e in _find_widgets(dialog, tk.Entry)
            if str(e.grid_info().get("row")) == "4"
        ]
        assert entries, "the Pentacam OD field should exist"
        entries[0].delete(0, "end")
        entries[0].insert(0, str(path))
        pump(1.0)

        canvas = _find_widgets(preview, tk.Canvas)[0]
        pupil_btn = _find_button(root, "Pupil")
        limbus_btn = _find_button(root, "Limbus")
        plus_btn = _find_button(root, "+ Ring")
        reset_btn = _find_button(root, "Reset ring")
        assert None not in (pupil_btn, limbus_btn, plus_btn, reset_btn)

        # 1. the GREEN ring can be placed by hand
        pupil_btn.invoke()
        for _ in range(3):
            plus_btn.invoke()
        root.update()
        blue1, green1, texts1 = _preview_overlay(canvas)
        assert (blue1, green1) == (0, 1)
        assert any("pupil r=" in t for t in texts1)

        # 2. ...and so can the BLUE one, without disturbing the green
        limbus_btn.invoke()
        for _ in range(2):
            plus_btn.invoke()
        root.update()
        blue2, green2, texts2 = _preview_overlay(canvas)
        assert (blue2, green2) == (1, 1)
        assert any("limbus r=" in t and "pupil r=" in t for t in texts2)

        # 3. "Reset ring" hands back ONLY the selected (limbus) ring
        reset_btn.invoke()
        root.update()
        blue3, green3, texts3 = _preview_overlay(canvas)
        assert (blue3, green3) == (0, 1)
        assert any("pupil r=" in t and "limbus" not in t for t in texts3)

        # 4. resetting the pupil ring as well leaves nothing hand-set
        pupil_btn.invoke()
        reset_btn.invoke()
        root.update()
        blue4, green4, _texts4 = _preview_overlay(canvas)
        assert (blue4, green4) == (0, 0)

        # 5. a LEFT-DRAG that starts on the outline pulls THAT PART of the ring
        #    outwards -- the far side of the ring must not move.
        limbus_btn.invoke()
        plus_btn.invoke()
        root.update()
        before = _drawn_ring_vertices(canvas, "#5c8aff")
        n = len(before)
        vx, vy = before[0]          # the 0-degree vertex sits on the outline

        _drag(canvas, root, vx, vy, vx + 40, vy)
        after = _drawn_ring_vertices(canvas, "#5c8aff")
        assert _vertex_move(before, after, 0) > 25, (before[0], after[0])
        # ... while the opposite arc is untouched: this is a local pull, not a
        # whole-circle resize (which is what - Ring / + Ring are for).
        assert _vertex_move(before, after, n // 2) < 4
        assert _vertex_move(before, after, n // 4) < 4

        # 6. dragging that same arc back inwards pushes it in again
        vx, vy = after[0]
        _drag(canvas, root, vx, vy, vx - 30, vy)
        final = _drawn_ring_vertices(canvas, "#5c8aff")
        assert _vertex_move(after, final, 0) > 15
        assert final[0][0] < after[0][0]
        # the far arc stayed put throughout
        assert _vertex_move(before, final, n // 2) < 4

        # 6b. "- Ring" / "+ Ring" still grow the circle uniformly, but they
        #     keep the shape the user pulled out (the bulge scales with it).
        _, _, texts_before = _preview_overlay(canvas)
        assert any("(pulled)" in t for t in texts_before)
        plus_btn.invoke()
        root.update()
        grown = _drawn_ring_vertices(canvas, "#5c8aff")
        assert _vertex_move(final, grown, n // 2) > 3
        _, _, texts_after = _preview_overlay(canvas)
        assert any("(pulled)" in t for t in texts_after)

        # 7. a drag that starts AWAY from the outline still draws an ROI box
        canvas.delete("roi")
        roi_btn = _find_button(root, "ROI")
        if roi_btn is not None and getattr(roi_btn, "_roi_mode_var", None) and not roi_btn._roi_mode_var.get():
            roi_btn.invoke()
        _drag(canvas, root, 20, 20, 140, 140)
        assert canvas.find_withtag("roi"), "the iris ROI box must still be drawable"
    finally:
        root.destroy()


def _drawn_ring_vertices(canvas, outline):
    """The ``[(x, y), ...]`` vertices of the first ring drawn in *outline*.

    The ring is drawn as one polygon with a fixed number of vertices, the first
    at angle 0°; a *pulled* ring keeps that order, so vertex *i* before and
    after a drag differ by that arc's radial change.
    """
    items = [
        i for i in canvas.find_withtag("detection")
        if canvas.type(i) == "polygon" and canvas.itemcget(i, "outline") == outline
    ]
    assert items, f"no ring drawn with outline {outline}"
    pts = canvas.coords(items[0])
    return list(zip(pts[0::2], pts[1::2]))


def _vertex_move(before, after, index):
    """How far vertex *index* moved (px) between two drawn rings."""
    return math.hypot(
        after[index][0] - before[index][0], after[index][1] - before[index][1]
    )


def _drag(canvas, root, x0, y0, x1, y1):
    """Press at (x0, y0), drag to (x1, y1), release -- a real left-drag."""
    canvas.event_generate("<ButtonPress-1>", x=int(x0), y=int(y0))
    root.update()
    canvas.event_generate("<B1-Motion>", x=int(x1), y=int(y1))
    root.update()
    canvas.event_generate("<ButtonRelease-1>", x=int(x1), y=int(y1))
    root.update()


def _find_widgets(widget, cls):
    """Every descendant of *widget* of type *cls*, depth-first."""
    out = []
    for child in widget.winfo_children():
        if isinstance(child, cls):
            out.append(child)
        out.extend(_find_widgets(child, cls))
    return out

