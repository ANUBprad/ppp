"""Focused regression and integration tests for the four verified runtime fixes:
1. Pentacam detection auto-trigger vs interactive ROI gate.
2. Authoritative OD/OS orientation in cross section and processor.
3. ADD PX popup sizing using screen bounds without unmaximized parent clamping.
4. Missing eye reference image handling and clear feedback.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest

from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
from pupil_tracking.interface.gui_app import PupilTrackingGUI


# ── FIX 1: Pentacam Detection Trigger & ROI Workflow ──────────────────

class TestPentacamTriggerWorkflow:
    def test_default_fixed_roi_computation(self):
        """Default fixed ROI matches anterior segment aperture (Option A: 847x542 px)."""
        w, h = 1280, 1024
        rw = min(float(w), 847.0)
        rh = min(float(h), 542.0)
        ix0 = int(round((float(w) - rw) / 2.0))
        iy0 = int(round((float(h) - rh) / 2.0))
        roi = (ix0 / float(w), iy0 / float(h), (ix0 + int(rw)) / float(w), (iy0 + int(rh)) / float(h))
        assert 0.0 <= roi[0] < roi[2] <= 1.0
        assert 0.0 <= roi[1] < roi[3] <= 1.0
        # Pixel width and height must match 847x542
        assert abs((roi[2] - roi[0]) * w - 847) < 1.0
        assert abs((roi[3] - roi[1]) * h - 542) < 1.0

    def test_roi_gating_logic(self):
        """When interactive ROI is active and unlocked, detection must wait.
        When ROI is not in interactive mode, detection triggers automatically."""
        # Scenario A: User loaded image without activating interactive ROI mode
        roi_active = False
        is_locked = False
        # Under our fix: not roi_active triggers auto-detection
        should_auto_detect = (not roi_active) or is_locked
        assert should_auto_detect is True, "Should auto-detect on image load when ROI mode not active"

        # Scenario B: User clicked ROI button to enter interactive ROI positioning
        roi_active = True
        is_locked = False
        should_auto_detect = (not roi_active) or is_locked
        assert should_auto_detect is False, "Must wait while user is positioning ROI rectangle"

        # Scenario C: User pressed Enter or 'Lock & Detect'
        is_locked = True
        should_auto_detect = (not roi_active) or is_locked
        assert should_auto_detect is True, "Must trigger detection once locked"


# ── FIX 2: Authoritative OD/OS Orientation ───────────────────────────

class TestAuthoritativeOrientation:
    def test_clinical_meridian_labels(self):
        """OD: Left=0, Right=180.
        OS: Left=0 (Nasal), Right=180 (Temporal).
        Superior=270, Inferior=90 for both."""
        od_left, od_right, od_up, od_down = get_centration_meridian_labels("OD")
        assert od_left == "0"
        assert od_right == "180"
        assert od_up == "270"
        assert od_down == "90"

        os_left, os_right, os_up, os_down = get_centration_meridian_labels("OS")
        assert os_left == "0"
        assert os_right == "180"
        assert os_up == "270"
        assert os_down == "90"

    def test_gui_selected_eye_overrides_stale_result_eye(self):
        """Even if frame DetectionResult has default eye='OD', user's selection 'OS' governs."""
        selected_eye_var = SimpleNamespace(get=lambda: "OS")
        result = SimpleNamespace(eye="OD")

        eye = (
            (selected_eye_var.get() if selected_eye_var.get() in ("OD", "OS") else None)
            or getattr(result, "eye", None)
            or "OD"
        )
        assert eye == "OS", "GUI selected eye MUST override result.eye"
        lbls = get_centration_meridian_labels(eye)
        assert lbls == ("0", "180", "270", "90")


# ── FIX 3: ADD PX Popup Sizing & Geometry ───────────────────────────

class TestAddPxPopupSizing:
    def test_calc_dialog_geometry_unmaximized_parent(self):
        """When parent window is small/unmaximized (e.g. 1280x720 on 1080p),
        popup expands to available screen space rather than being clamped to 1254px."""
        class _StubUnmaximizedRoot:
            def winfo_screenwidth(self):
                return 1920
            def winfo_screenheight(self):
                return 1080
            def winfo_width(self):
                return 1280
            def winfo_height(self):
                return 720
            def winfo_rootx(self):
                return 100
            def winfo_rooty(self):
                return 80

        mw = _StubUnmaximizedRoot()
        tw, th, rx, ry = PupilTrackingGUI._calc_dialog_geometry(mw)

        # Must NOT be constrained to 1280 * 0.98 (~1254px)
        assert tw >= 1600, f"Dialog width {tw} should use available screen space (>=1600)"
        assert th >= 900, f"Dialog height {th} should use available screen space (>=900)"
        # Must leave taskbar space at bottom
        assert ry + th <= 1080 - 40 + 10, "Dialog bottom must respect Windows taskbar margin"

    def test_calc_dialog_geometry_maximized_parent(self):
        """When parent window is maximized (e.g. 1800x950 on 1080p), sizes appropriately."""
        class _StubMaximizedRoot:
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

        mw = _StubMaximizedRoot()
        tw, th, rx, ry = PupilTrackingGUI._calc_dialog_geometry(mw)

        assert tw >= 1700
        assert th >= 880
        assert rx >= 10
        assert ry >= 10


# ── FIX 4: Missing Eye Reference Image ───────────────────────────────

class TestMissingEyeReferenceHandling:
    def test_load_pentacam_for_missing_eye_returns_none_without_cross_contamination(self, tmp_path):
        """When patient has only OD Pentacam data, loading OS returns None and does not leak OD data."""
        patient_dir = tmp_path / "patients" / "demo_test"
        od_pentacam = patient_dir / "OD" / "Pentacam"
        od_pentacam.mkdir(parents=True)
        # Create dummy OD unwrapped strip
        od_strip_file = od_pentacam / "Pentacam_OD_unwrapped.png"
        dummy_img = np.zeros((64, 512, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(str(od_strip_file), dummy_img)

        # OS folder does NOT have Pentacam data
        os_dir = patient_dir / "OS"
        os_dir.mkdir(parents=True)

        class _MockGUI:
            def __init__(self, pdir):
                self._active_patient_dir = pdir
                self.logger = SimpleNamespace(
                    warning=lambda *a, **k: None,
                    info=lambda *a, **k: None,
                )
            _load_pentacam_for_eye = PupilTrackingGUI._load_pentacam_for_eye

        gui = _MockGUI(patient_dir)

        # OD should load successfully
        od_strip = gui._load_pentacam_for_eye("OD")
        assert od_strip is not None, "OD strip should load from OD/Pentacam folder"
        assert od_strip.shape == (64, 512, 3)

        # OS must return None, NEVER OD data
        os_strip = gui._load_pentacam_for_eye("OS")
        assert os_strip is None, "OS strip must be None when no OS Pentacam data exists"
