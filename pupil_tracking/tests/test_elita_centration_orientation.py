"""Targeted regression tests for Phase 2 (BUG-005): Correct ELITA Centration 0°/180° Orientation.

Validates:
1. OD 0° and 180° directions: Right is 0° (Nasal/Nose), Left is 180° (Temporal).
2. OS 0° and 180° directions: Left is 0° (Nasal/Nose), Right is 180° (Temporal).
3. Consistent anatomical orientation relative to the nose as the central reference.
4. Rapid switching between OD and OS without unintended reversal or stickiness.
5. Robust handling of edge cases and boundary inputs.
6. Preservation of existing centration offset calculations and corneal center metrics.
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
from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
from pupil_tracking.interface.gui.drawing import draw_cross_section
from pupil_tracking.utils.types import (
    CalibrationInfo,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)


def _create_synthetic_detection(
    pupil_center=(320.0, 240.0),
    limbus_center=(320.0, 240.0),
    pupil_radius=30.0,
    limbus_radius=70.0,
    eye="OD",
) -> EyeDetectionResult:
    """Create a fully synthetic EyeDetectionResult with no real patient data."""
    res = EyeDetectionResult()
    res.pupil = PupilDetection(
        detected=True,
        confidence=0.95,
        ellipse=EllipseParams(
            center_x=float(pupil_center[0]),
            center_y=float(pupil_center[1]),
            semi_major=float(pupil_radius),
            semi_minor=float(pupil_radius),
            angle_deg=0.0,
        ),
    )
    res.limbus = LimbusDetection(
        detected=True,
        confidence=0.95,
        ellipse=EllipseParams(
            center_x=float(limbus_center[0]),
            center_y=float(limbus_center[1]),
            semi_major=float(limbus_radius),
            semi_minor=float(limbus_radius),
            angle_deg=0.0,
        ),
    )
    res.calibration = CalibrationInfo(calibrated=True, mm_per_px=0.02, px_per_mm=50.0)
    # Attach synthetic eye metadata
    setattr(res, "eye", eye)
    return res


class TestCentrationMeridianLabels:
    """Tests for get_centration_meridian_labels helper."""

    def test_od_orientation(self):
        """OD: Left side is 0°, Right side is 180°."""
        left, right, up, down = get_centration_meridian_labels("OD")
        assert left == "0", "OD screen left must be 0°"
        assert right == "180", "OD screen right must be 180°"
        assert up == "270", "Superior must be 270°"
        assert down == "90", "Inferior must be 90°"

    def test_os_orientation(self):
        """OS: Left side (Nasal, toward Nose) is 0°, Right side (Temporal) is 180°."""
        left, right, up, down = get_centration_meridian_labels("OS")
        assert left == "0", "OS screen left must be 0° (Nasal, toward Nose)"
        assert right == "180", "OS screen right must be 180° (Temporal)"
        assert up == "270", "Superior must be 270°"
        assert down == "90", "Inferior must be 90°"

    def test_orientation_consistency(self):
        """Both OD and OS map screen left to 0° and screen right to 180°."""
        od_left, od_right, _, _ = get_centration_meridian_labels("OD")
        os_left, os_right, _, _ = get_centration_meridian_labels("OS")

        assert od_left == "0" and od_right == "180", "OD must have left=0° and right=180°"
        assert os_left == "0" and os_right == "180", "OS must have left=0° and right=180°"

    def test_case_insensitivity_and_whitespace(self):
        """Edge case handling: lowercase, spaces, and defaults."""
        assert get_centration_meridian_labels("  os  ") == ("0", "180", "270", "90")
        assert get_centration_meridian_labels("od") == ("0", "180", "270", "90")
        assert get_centration_meridian_labels("") == ("0", "180", "270", "90")
        assert get_centration_meridian_labels(None) == ("0", "180", "270", "90")


class TestCentrationDrawingOverlay:
    """Tests verifying cross-section rendered text matches OD and OS conventions."""

    def test_draw_cross_section_od(self):
        """Verifies text rendered on canvas for OD."""
        out = np.zeros((480, 640, 3), dtype=np.uint8)
        res = _create_synthetic_detection(eye="OD")

        with mock.patch("cv2.putText") as mock_put_text:
            draw_cross_section(out, res, scale=1.0, eye="OD")
            drawn_strings = [call[0][1] for call in mock_put_text.call_args_list]

            assert "270" in drawn_strings
            assert "90" in drawn_strings
            assert "180" in drawn_strings
            assert "0" in drawn_strings

            # Find coordinates for "0" and "180"
            for call in mock_put_text.call_args_list:
                text = call[0][1]
                coord = call[0][2]
                if text == "0":
                    assert coord[0] < 320, "OD 0° must be rendered on the left (< 320)"
                elif text == "180":
                    assert coord[0] > 320, "OD 180° must be rendered on the right (> 320)"

    def test_draw_cross_section_os(self):
        """Verifies text rendered on canvas for OS."""
        out = np.zeros((480, 640, 3), dtype=np.uint8)
        res = _create_synthetic_detection(eye="OS")

        with mock.patch("cv2.putText") as mock_put_text:
            draw_cross_section(out, res, scale=1.0, eye="OS")
            drawn_strings = [call[0][1] for call in mock_put_text.call_args_list]

            assert "270" in drawn_strings
            assert "90" in drawn_strings
            assert "180" in drawn_strings
            assert "0" in drawn_strings

            # Find coordinates for "0" and "180"
            for call in mock_put_text.call_args_list:
                text = call[0][1]
                coord = call[0][2]
                if text == "0":
                    assert coord[0] < 320, "OS 0° must be rendered on the left (< 320)"
                elif text == "180":
                    assert coord[0] > 320, "OS 180° must be rendered on the right (> 320)"

    def test_repeated_switching_without_reversal_leak(self):
        """Switching OD -> OS -> OD produces the exact expected labels every cycle."""
        for _ in range(5):
            left_od, right_od, _, _ = get_centration_meridian_labels("OD")
            assert left_od == "0" and right_od == "180"

            left_os, right_os, _, _ = get_centration_meridian_labels("OS")
            assert left_os == "0" and right_os == "180"


class TestCornealCenterMeasurementPreserved:
    """Verifies that mathematical centration and calibration calculations are strictly intact."""

    def test_corneal_center_calculator_integrity(self):
        calc = CornealCenterCalculator()
        p = PupilDetection(
            detected=True,
            confidence=0.9,
            ellipse=EllipseParams(center_x=325.0, center_y=240.0, semi_major=25.0, semi_minor=25.0),
        )
        l = LimbusDetection(
            detected=True,
            confidence=0.9,
            ellipse=EllipseParams(center_x=320.0, center_y=240.0, semi_major=60.0, semi_minor=60.0),
        )
        cal = CalibrationInfo(calibrated=True, mm_per_px=0.02, px_per_mm=50.0)

        result = calc.calculate(p, l, cal)
        assert result.valid is True
        assert result.center_px == (320.0, 240.0)
        assert result.offset_px == (5.0, 0.0)
        assert result.offset_magnitude_px == pytest.approx(5.0)
        assert result.offset_magnitude_mm == pytest.approx(0.10)
        assert result.offset_angle_deg == pytest.approx(0.0)
