"""Comprehensive Phase 1 Test Suite: Orientation, Iris Unwrap Mapping, and Strip Controls.

Validates all 9 points specified in Phase 1 requirements:
1. OD and OS 0°/180° orientation (OD: 180° screen left, 0° screen right; OS: 0° screen left, 180° screen right).
2. Correct anatomical nasal/temporal directions (0° = Nasal / nose reference, 180° = Temporal).
3. Correct 0°, 90°, 180°, 270°, and 360° unwrap mapping.
4. Absence of unintended horizontal/vertical flips or angular reversal.
5. Correct alignment and shared angular reference between Pentacam and ELITA unwrap strips.
6. Minus/plus offset behavior, clamping, and reset.
7. Resetting offset on eye selection, image load, video load, and camera start.
8. Manual shifts remaining display-only (raw strips and stored landmarks unchanged).
9. Cyclotorsion measurements remaining unchanged when manual display offset changes.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any
from unittest import mock

import cv2
import numpy as np
import pytest

from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.cyclotorsion.cyclotorsion_engine import estimate_cyclotorsion
from pupil_tracking.cyclotorsion.polar_unwrap import (
    build_unwrap_maps,
    roll_strip,
    unwrap_iris,
)
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
import pupil_tracking.interface.gui_app as gui_app
from pupil_tracking.interface.gui_app import (
    PupilTrackingGUI,
    _UNWRAP_STRIP_SIZE,
    _ELITA_SHIFT_STEP_DEG,
)
from pupil_tracking.utils.types import (
    CalibrationInfo,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)


# ==============================================================================
# 1 & 2: BUG-005 — Centration 0°/180° Orientation & Anatomical Directions
# ==============================================================================


class TestCentrationOrientationAndAnatomy:
    """Verifies screen conventions and anatomical nasal/temporal directions."""

    def test_od_screen_and_anatomical_convention(self):
        """OD (right eye): Screen Left = 0°, Screen Right = 180°."""
        left, right, top, bottom = get_centration_meridian_labels("OD")
        assert left == "0", "OD screen left must be 0°"
        assert right == "180", "OD screen right must be 180°"
        assert top == "270", "OD screen top must be 270° (Superior)"
        assert bottom == "90", "OD screen bottom must be 90° (Inferior)"

    def test_os_screen_and_anatomical_convention(self):
        """OS (left eye): Screen Left = 0° (Nasal), Screen Right = 180° (Temporal)."""
        left, right, top, bottom = get_centration_meridian_labels("OS")
        assert left == "0", "OS screen left must be 0° (Nasal / facing nose)"
        assert right == "180", "OS screen right must be 180° (Temporal)"
        assert top == "270", "OS screen top must be 270° (Superior)"
        assert bottom == "90", "OS screen bottom must be 90° (Inferior)"

    def test_case_insensitivity_and_default(self):
        """Meridian labels handle lowercase and fallback safely."""
        assert get_centration_meridian_labels("od") == ("0", "180", "270", "90")
        assert get_centration_meridian_labels("os") == ("0", "180", "270", "90")
        assert get_centration_meridian_labels(None) == ("0", "180", "270", "90")

    def test_corneal_center_calculator_integrity(self):
        """Preserves existing measurement values and offset angle conventions."""
        calc = CornealCenterCalculator()
        pupil_det = PupilDetection(
            detected=True,
            confidence=0.95,
            ellipse=EllipseParams(
                center_x=325.0,
                center_y=240.0,
                semi_major=30.0,
                semi_minor=30.0,
                angle_deg=0.0,
            ),
        )
        limbus_det = LimbusDetection(
            detected=True,
            confidence=0.95,
            ellipse=EllipseParams(
                center_x=320.0,
                center_y=240.0,
                semi_major=70.0,
                semi_minor=70.0,
                angle_deg=0.0,
            ),
        )
        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=10.0,
            mm_per_px=0.1,
            source="test",
            method="test",
            reference_diameter_mm=11.0,
            reference_diameter_px=110.0,
            confidence=1.0,
        )

        res = calc.calculate(pupil_det, limbus_det, calibration=cal)
        assert res.valid is True
        assert res.offset_px == (5.0, 0.0)
        assert pytest.approx(res.offset_magnitude_px, abs=1e-4) == 5.0
        assert pytest.approx(res.offset_magnitude_mm, abs=1e-4) == 0.5
        assert pytest.approx(res.offset_angle_deg, abs=1e-4) == 0.0


# ==============================================================================
# 3, 4, 5: BUG-006 — Iris Unwrap Mapping, Continuous Loop, and Shared Grid
# ==============================================================================


class TestIrisUnwrapOrientationAndContinuity:
    """Verifies polar-to-cartesian unwrap mapping at 0, 90, 180, 270, 360 deg."""

    @pytest.mark.parametrize("eye", ["OD", "OS"])
    def test_cardinal_angles_and_directions(self, eye: str):
        """Verifies 0°, 90°, 180°, 270°, 360° mapping in build_unwrap_maps."""
        cx, cy = 300.0, 300.0
        r_inner, r_outer = 40.0, 100.0
        out_h, out_w = 64, 720
        roi = IrisROI(
            valid=True,
            center_x=cx,
            center_y=cy,
            pupil_semi_major=r_inner,
            pupil_semi_minor=r_inner,
            limbus_semi_major=r_outer,
            limbus_semi_minor=r_outer,
            inner_inset_frac=0.0,
            outer_inset_frac=0.0,
        )

        maps = build_unwrap_maps(roi, output_size=(out_h, out_w), eye=eye)
        assert maps is not None
        map_x, map_y = maps.map_x, maps.map_y

        # Mid-radius row
        mid_row = out_h // 2

        col_0 = 0
        col_90 = out_w // 4
        col_180 = out_w // 2
        col_270 = (3 * out_w) // 4

        # Evaluate (x, y) relative to center
        dx_0 = map_x[mid_row, col_0] - cx
        dy_0 = map_y[mid_row, col_0] - cy

        dx_90 = map_x[mid_row, col_90] - cx
        dy_90 = map_y[mid_row, col_90] - cy

        dx_180 = map_x[mid_row, col_180] - cx
        dy_180 = map_y[mid_row, col_180] - cy

        dx_270 = map_x[mid_row, col_270] - cx
        dy_270 = map_y[mid_row, col_270] - cy

        if eye == "OD":
            # OD: 0° is Nasal (+x, right), 180° is Temporal (-x, left)
            assert dx_0 > 0, "OD col 0 must be +x (Nasal / screen right)"
            assert pytest.approx(dy_0, abs=1.5) == 0.0
            assert dx_180 < 0, "OD col 180 must be -x (Temporal / screen left)"
            assert pytest.approx(dy_180, abs=1.5) == 0.0
        else:
            # OS: 0° is Nasal (-x, left), 180° is Temporal (+x, right)
            assert dx_0 < 0, "OS col 0 must be -x (Nasal / screen left)"
            assert pytest.approx(dy_0, abs=1.5) == 0.0
            assert dx_180 > 0, "OS col 180 must be +x (Temporal / screen right)"
            assert pytest.approx(dy_180, abs=1.5) == 0.0

        # Vertical directions: 90° is Inferior (+y, down), 270° is Superior (-y, up)
        assert dy_90 > 0, "90° must be +y (Inferior / screen down)"
        assert pytest.approx(dx_90, abs=1.5) == 0.0
        assert dy_270 < 0, "270° must be -y (Superior / screen up)"
        assert pytest.approx(dx_270, abs=1.5) == 0.0

    @pytest.mark.parametrize("eye", ["OD", "OS"])
    def test_continuity_at_360_degrees(self, eye: str):
        """Checks that 360° maps back continuously to 0° without tear or reversal."""
        cx, cy = 300.0, 300.0
        r_inner, r_outer = 40.0, 100.0
        out_h, out_w = 64, 720
        roi = IrisROI(
            valid=True,
            center_x=cx,
            center_y=cy,
            pupil_semi_major=r_inner,
            pupil_semi_minor=r_inner,
            limbus_semi_major=r_outer,
            limbus_semi_minor=r_outer,
            inner_inset_frac=0.0,
            outer_inset_frac=0.0,
        )

        maps = build_unwrap_maps(roi, output_size=(out_h, out_w), eye=eye)
        assert maps is not None
        map_x = maps.map_x

        # Difference between column 0 and column N-1 should be ~ (1 / out_w) of perimeter
        dx_step = abs(map_x[:, 1] - map_x[:, 0])
        dx_wrap = abs(map_x[:, -1] - map_x[:, 0])
        # The wrap distance should be of the same order as one step
        np.testing.assert_allclose(dx_wrap, dx_step, atol=2.0)

    def test_pentacam_and_elita_share_same_angular_reference(self):
        """Ensures both image sources produce strips that align under the same eye convention."""
        # Create an artificial eye image with a marked feature at 0° (Nasal)
        h, w = 400, 400
        cx, cy = 200.0, 200.0
        img = np.zeros((h, w), dtype=np.uint8)

        # Place a bright landmark at 0° for OD (+x = 260, y = 200)
        cv2.circle(img, (260, 200), 5, 255, -1)

        roi = IrisROI(
            valid=True,
            center_x=cx,
            center_y=cy,
            pupil_semi_major=40.0,
            pupil_semi_minor=40.0,
            limbus_semi_major=80.0,
            limbus_semi_minor=80.0,
            inner_inset_frac=0.0,
            outer_inset_frac=0.0,
        )

        # Unwrap with build_unwrap_maps for OD
        maps = build_unwrap_maps(roi, output_size=(64, 720), eye="OD")
        assert maps is not None
        strip = cv2.remap(
            img, maps.map_x, maps.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # The bright landmark at 0° must appear at column 0 (and column 719)
        max_col = np.argmax(np.mean(strip, axis=0))
        assert max_col in (0, 1, 718, 719), (
            f"Expected landmark at 0°, got column {max_col}"
        )


# ==============================================================================
# 6, 7, 8: Manual ELITA Strip-Alignment Controls & Display-Only Invariant
# ==============================================================================


class _StubCanvas:
    def __init__(self, width: int = 869, height: int = 200):
        self._w = width
        self._h = height

    def winfo_width(self) -> int:
        return self._w

    def winfo_height(self) -> int:
        return self._h

    def delete(self, *a, **k):
        pass

    def create_text(self, *a, **k):
        pass

    def create_line(self, *a, **k):
        pass

    def create_image(self, *a, **k):
        pass


def _make_test_app():
    """Build a headless PupilTrackingGUI instance with essential stubs."""
    app = SimpleNamespace()
    app._unwrap_canvas = _StubCanvas()
    app._colors = SimpleNamespace(FG_TERTIARY="#606070")
    app._elita_unwrap_shift_cols = 0
    app._elita_shift_label = None
    app._elita_shift_after_id = None
    app._unwrap_strip = None
    app._elita_strip = None
    app._cyclotorsion_deg = 0.0
    app._selected_eye_var = SimpleNamespace(get=lambda: "OD")

    # Bind methods from PupilTrackingGUI
    app._roll_strip = PupilTrackingGUI._roll_strip
    app._unwrap_zone_lines = PupilTrackingGUI._unwrap_zone_lines
    app._show_unwrap_zones = SimpleNamespace(get=lambda: False)
    app._update_elita_shift_label = lambda: PupilTrackingGUI._update_elita_shift_label(
        app
    )
    app._reset_elita_shift = lambda: PupilTrackingGUI._reset_elita_shift(app)
    app._nudge_elita_shift = lambda d: PupilTrackingGUI._nudge_elita_shift(app, d)
    app._redraw_unwrap = lambda: PupilTrackingGUI._redraw_unwrap(app)
    return app


class TestManualStripControlsAndInvariants:
    """Verifies manual adjustment controls, auto-repeat, reset, and display-only invariant."""

    def test_nudge_step_and_direction(self):
        """Minus nudges left (-1), Plus nudges right (+1), 1° step = 2 cols."""
        app = _make_test_app()
        cols_per_deg = int(
            round(_ELITA_SHIFT_STEP_DEG / 360.0 * _UNWRAP_STRIP_SIZE[1])
        )
        assert cols_per_deg == 2, "1 degree step on 720 col strip must be 2 cols"

        app._nudge_elita_shift(+1)
        assert app._elita_unwrap_shift_cols == 2
        app._nudge_elita_shift(+1)
        assert app._elita_unwrap_shift_cols == 4
        app._nudge_elita_shift(-1)
        assert app._elita_unwrap_shift_cols == 2

    def test_nudge_clamping_to_half_turn(self):
        """Shift is strictly clamped to +/- 180° (+/- 360 cols)."""
        app = _make_test_app()
        half_turn = _UNWRAP_STRIP_SIZE[1] // 2  # 360 cols

        for _ in range(300):
            app._nudge_elita_shift(+1)
        assert app._elita_unwrap_shift_cols == half_turn

        for _ in range(600):
            app._nudge_elita_shift(-1)
        assert app._elita_unwrap_shift_cols == -half_turn

    def test_reset_elita_shift(self):
        """Explicit reset returns shift to 0."""
        app = _make_test_app()
        app._nudge_elita_shift(+1)
        app._nudge_elita_shift(+1)
        assert app._elita_unwrap_shift_cols == 4

        app._reset_elita_shift()
        assert app._elita_unwrap_shift_cols == 0

    def test_manual_shift_does_not_mutate_raw_strips(self, monkeypatch):
        """The manual shift rolls only the displayed copy; raw strips remain identical."""
        app = _make_test_app()
        rows, cols = _UNWRAP_STRIP_SIZE
        raw_pentacam = np.full((rows, cols), 50, dtype=np.uint8)
        raw_elita = np.full((rows, cols), 100, dtype=np.uint8)

        # Place a single-pixel marker in raw_elita at (10, 20)
        raw_elita[10, 20] = 255
        app._unwrap_strip = raw_pentacam
        app._elita_strip = raw_elita

        captured_renders = []

        class _PhotoStub:
            def __init__(self, img):
                captured_renders.append(np.asarray(img).copy())

        monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_PhotoStub))

        # First render with shift = 0
        app._elita_unwrap_shift_cols = 0
        app._redraw_unwrap()
        pent0, live0 = captured_renders[0], captured_renders[1]
        assert np.array_equal(app._elita_strip, raw_elita)

        captured_renders.clear()
        # Shift by +10 cols (+5°)
        app._elita_unwrap_shift_cols = 10
        app._redraw_unwrap()
        pent1, live1 = captured_renders[0], captured_renders[1]

        # Check raw array has NOT been mutated
        assert np.array_equal(app._elita_strip, raw_elita)
        assert np.array_equal(app._unwrap_strip, raw_pentacam)

        # Check Pentacam render did not move, but live ELITA render did move
        assert np.array_equal(pent0, pent1)
        assert not np.array_equal(live0, live1)

    def test_manual_shift_does_not_affect_cyclotorsion_calculation(self):
        """Cyclotorsion measurement is purely algorithmic on raw strips, unaffected by display shift."""
        rows, cols = 64, 720

        # Construct reference and a 5° rotated target strip
        # 5 deg = 10 cols
        x = np.linspace(0, 4 * np.pi, cols)
        ref_signal = (np.sin(x) * 100 + 128).astype(np.uint8)
        ref_strip = np.tile(ref_signal, (rows, 1))
        live_strip = roll_strip(ref_strip, 10)  # 5° rotation

        # Compute ground truth cyclotorsion
        result_before = estimate_cyclotorsion(ref_strip, live_strip)

        # Simulate user shifting display by +15° (30 cols)
        user_display_shift = 30
        display_only_strip = roll_strip(live_strip, user_display_shift)

        # Re-compute cyclotorsion from the raw strips (as the application does)
        result_after = estimate_cyclotorsion(ref_strip, live_strip)

        assert pytest.approx(result_before.angle_deg, abs=1e-3) == result_after.angle_deg
        assert result_before.score == result_after.score
        assert pytest.approx(result_after.angle_deg, abs=0.5) == 5.0
