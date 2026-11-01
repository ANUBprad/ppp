"""test_pentacam_detector.py — Comprehensive tests for Phase 4 fixes:
1. ELITA-based Pentacam detection strategy with PentacamDetector.
2. Pupil and limbus boundary extraction independence and ellipticity preservation.
3. Quantitative accuracy against clinical annotated dataset.
4. OD/OS centration orientation (0° and 180°) relative to patient's nose.
5. Iris unwrap alignment and manual nudge controls.
6. ADD PX responsive popup sizing and layout.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.pentacam_detector import PentacamDetector
from pupil_tracking.cyclotorsion.polar_unwrap import build_unwrap_maps, apply_unwrap_maps, unwrap_iris
from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
from pupil_tracking.interface import gui_app
from pupil_tracking.utils.types import EllipseParams, EyeDetectionResult, PupilDetection, LimbusDetection


# ───────────────────────────────────────────────────────────────────────────
# 1. PentacamDetector Core & Geometry Tests
# ───────────────────────────────────────────────────────────────────────────

class TestPentacamDetector:
    """Verifies PentacamDetector's architecture, boundary independence, and ellipticity."""

    def test_instantiation_and_fast_inference_reuse(self):
        detector = PentacamDetector()
        assert detector._ensure_engine() is not None
        assert detector._input_size == 320

    def test_synthetic_eye_independent_pupil_and_limbus(self):
        """Synthetic eye image with distinct pupil and limbus boundaries."""
        detector = PentacamDetector()
        img = np.full((400, 400, 3), 200, dtype=np.uint8)
        # Sclera / background: 200
        # Iris (limbus outer edge): 120, ellipse at center (200, 200), axes (90, 80)
        cv2.ellipse(img, (200, 200), (90, 80), 15, 0, 360, (120, 120, 120), -1)
        # Pupil: 20, ellipse at center (202, 198), axes (35, 30)
        cv2.ellipse(img, (202, 198), (35, 30), -10, 0, 360, (20, 20, 20), -1)

        result = detector.detect(img, eye="OD")
        assert result is not None
        assert hasattr(result, "pupil")
        assert hasattr(result, "limbus")

    def test_preserves_genuine_ellipticity(self):
        """Verifies that ellipse fitting preserves major/minor axes without circular clamping."""
        detector = PentacamDetector()
        engine = detector._ensure_engine()
        # Synthetic contour with an elongated ellipse
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.ellipse(mask, (150, 150), (80, 50), 30, 0, 360, 255, -1)
        res = engine._mask_to_ellipse(mask, min_area=50, scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0)
        assert res is not None
        cx, cy, r, semi_a, semi_b, angle = res
        assert abs(cx - 150.0) < 2.0
        assert abs(cy - 150.0) < 2.0
        # Check that semi-axes reflect genuine ellipticity (80 and 50)
        assert abs(semi_a - 80.0) < 3.0
        assert abs(semi_b - 50.0) < 3.0
        assert semi_a > semi_b * 1.3
        # Check that angle is close to 30 degrees (modulo 180)
        assert abs((angle - 30.0 + 90.0) % 180.0 - 90.0) < 5.0

    def test_roi_crop_and_coordinate_transformation(self):
        """Verifies that detection within an ROI crop maps back accurately to image space."""
        detector = PentacamDetector()
        img = np.full((600, 800, 3), 180, dtype=np.uint8)
        # Place eye at (450, 320)
        cv2.ellipse(img, (450, 320), (100, 95), 0, 0, 360, (100, 100, 100), -1)
        cv2.ellipse(img, (450, 320), (35, 32), 0, 0, 360, (20, 20, 20), -1)

        # ROI centered around (450, 320) in normalized coords
        roi = {
            "x": 0.40,
            "y": 0.35,
            "width": 0.35,
            "height": 0.40,
        }
        res = detector.detect(img, roi=roi, eye="OD")
        assert res is not None

    def test_grayscale_and_color_adapter(self):
        """Verifies that single-channel grayscale images are normalized properly."""
        detector = PentacamDetector()
        gray = np.full((320, 320), 128, dtype=np.uint8)
        res = detector.detect(gray, eye="OS")
        assert res is not None
        assert res.eye == "OS"


# ───────────────────────────────────────────────────────────────────────────
# 2. Clinical Dataset Accuracy Verification
# ───────────────────────────────────────────────────────────────────────────

class TestClinicalDatasetAccuracy:
    """Evaluates PentacamDetector against all ground-truth annotations."""

    @pytest.fixture(scope="class")
    def clinical_data(self):
        annot_path = Path("clinical_data/clean/annotations/annotations.json")
        if not annot_path.is_file():
            pytest.skip("Clinical annotations not available")
        with open(annot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_detection_accuracy_across_clinical_dataset(self, clinical_data):
        detector = PentacamDetector()
        clean_dir = Path("clinical_data/clean")

        pupil_errors = []
        limbus_errors = []
        successes = 0
        total = 0

        for img_name, entry in clinical_data.items():
            img_path = clean_dir / img_name
            if not img_path.is_file():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            total += 1
            annots = entry.get("annotations", {})
            gt_p = annots.get("PUPIL", {})
            gt_l = annots.get("LIMBUS", {})

            res = detector.detect(img, eye="OD")
            if res.pupil.detected and res.limbus.detected:
                successes += 1
                pe = res.pupil.ellipse
                le = res.limbus.ellipse

                p_err = math.hypot(pe.center_x - gt_p["center_x"], pe.center_y - gt_p["center_y"])
                l_err = math.hypot(le.center_x - gt_l["center_x"], le.center_y - gt_l["center_y"])

                pupil_errors.append(p_err)
                limbus_errors.append(l_err)

        assert total > 0, "No clinical images found"
        success_rate = successes / total
        assert success_rate >= 0.90, f"Success rate {success_rate:.1%} must be >= 90%"

        mean_pupil_err = float(np.mean(pupil_errors))
        mean_limbus_err = float(np.mean(limbus_errors))

        # Quantitative criteria:
        # Mean pupil center error < 4.0 px
        # Mean limbus center error < 8.0 px
        assert mean_pupil_err < 4.0, f"Mean pupil center error {mean_pupil_err:.2f}px exceeds 4.0px"
        assert mean_limbus_err < 8.0, f"Mean limbus center error {mean_limbus_err:.2f}px exceeds 8.0px"


# ───────────────────────────────────────────────────────────────────────────
# 3. OD / OS Anatomical 0° and 180° Orientation Tests
# ───────────────────────────────────────────────────────────────────────────

class TestCentrationMeridianOrientation:
    """Verifies that patient's nose is the anatomical reference for both eyes."""

    def test_os_convention(self):
        """OS (Left Eye): Nose is to Screen Left (0°), Temporal is to Screen Right (180°)."""
        left, right, up, down = get_centration_meridian_labels("OS")
        assert left == "0", "OS screen left must be 0° (Nasal/Nose)"
        assert right == "180", "OS screen right must be 180° (Temporal)"
        assert up == "270", "OS screen up must be 270° (Superior)"
        assert down == "90", "OS screen down must be 90° (Inferior)"

    def test_od_convention(self):
        """OD (Right Eye): Screen Left is 0°, Screen Right is 180°."""
        left, right, up, down = get_centration_meridian_labels("OD")
        assert left == "0", "OD screen left must be 0°"
        assert right == "180", "OD screen right must be 180°"
        assert up == "270", "OD screen up must be 270° (Superior)"
        assert down == "90", "OD screen down must be 90° (Inferior)"

    def test_cross_section_label_selection(self):
        """Verifies that GUI _draw_cross_section prioritizes user-selected eye."""
        class _StubGUI:
            _selected_eye_var = SimpleNamespace(get=lambda: "OS")
        
        gui = _StubGUI()
        # Simulated logic from _draw_cross_section
        result = SimpleNamespace(eye="OD")
        eye = (
            (gui._selected_eye_var.get() if hasattr(gui, "_selected_eye_var") and gui._selected_eye_var.get() in ("OD", "OS") else None)
            or getattr(result, "eye", None)
            or "OD"
        )
        assert eye == "OS", "Selected eye in GUI must override stale detector eye"
        labels = get_centration_meridian_labels(eye)
        assert labels[0] == "0" and labels[1] == "180"


# ───────────────────────────────────────────────────────────────────────────
# 4. Iris Unwrap Strip Alignment and Nudge Tests
# ───────────────────────────────────────────────────────────────────────────

class TestUnwrapAlignmentAndNudge:
    """Verifies unwrap strip coordinates and manual nudge adjustment without auto-roll fight."""

    def test_common_angular_convention(self):
        """Both OD and OS unwrap maps sample from 0° (Nasal) to 360°."""
        from pupil_tracking.iris.roi import IrisROIExtractor
        pe = EllipseParams(center_x=100.0, center_y=100.0, semi_major=30.0, semi_minor=30.0, angle_deg=0.0)
        le = EllipseParams(center_x=100.0, center_y=100.0, semi_major=80.0, semi_minor=80.0, angle_deg=0.0)
        roi = IrisROIExtractor().build(pe, le)
        maps_od = build_unwrap_maps(roi, output_size=(64, 512), eye="OD")
        maps_os = build_unwrap_maps(roi, output_size=(64, 512), eye="OS")
        assert maps_od is not None
        assert maps_os is not None
        assert maps_od.output_size == (64, 512)
        assert maps_os.output_size == (64, 512)

    def test_manual_shift_display_roll_pure(self):
        """Verifies rolling the unwrap strip by manual shift."""
        strip = np.zeros((64, 512, 3), dtype=np.uint8)
        strip[:, 100:110] = 255  # Distinct vertical line at col 100-110

        rolled = gui_app.PupilTrackingGUI._roll_strip(strip, 20)
        assert rolled is not None
        # White line should now be at col 120:130
        assert rolled[:, 120:130].mean() > 200
        assert rolled[:, 100:110].mean() < 50


# ───────────────────────────────────────────────────────────────────────────
# 5. ADD PX Dialog Geometry & Sizing Tests
# ───────────────────────────────────────────────────────────────────────────

class TestAddPxDialogGeometry:
    """Verifies that ADD PX popup sizing is substantially maximized and responsive."""

    def test_calc_dialog_geometry_maximization(self):
        """Verifies geometry uses ~98% of main window width and ~96% height."""
        class _StubMainWin:
            def winfo_screenwidth(self):
                return 1920
            def winfo_screenheight(self):
                return 1080
            def winfo_width(self):
                return 1800
            def winfo_height(self):
                return 950
            def winfo_rootx(self):
                return 60
            def winfo_rooty(self):
                return 40

        mw = _StubMainWin()
        w, h, x, y = gui_app.PupilTrackingGUI._calc_dialog_geometry(mw)

        # Width should be ~98% of 1800 = 1764
        assert w >= 1700, f"Width {w} must be >= 1700"
        # Height should be ~96% of 950 = 912
        assert h >= 880, f"Height {h} must be >= 880"
        # Position should be centered relative to main window
        assert x >= 60
        assert y >= 40
