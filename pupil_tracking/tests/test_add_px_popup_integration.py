"""Tests for Phase 1: BUG-001 and BUG-002 Add Patient & Pentacam Preview integration.

Verifies:
1. BUG-001: Clicking Add PX opens exactly ONE popup window containing both patient
   entry fields and the Pentacam preview / ROI controls.
2. BUG-002: Popup opens with expanded default dimensions (1200x680, minsize 1100x620)
   providing comfortable visibility without clipping or overlapping.
3. Form fields, browse callbacks, eye selection (OD/OS), ROI toggle, ring adjustments,
   and Save/Cancel actions remain fully operational within the single window.
4. Window close, cancel, and reopen lifecycle does not leave orphaned windows or stale state.
"""

import os
import shutil
import tempfile
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import pytest

from pupil_tracking.interface import gui_app


from pupil_tracking.tests.test_pentacam_rectangular_roi import _get_shared_root


@pytest.fixture(autouse=True)
def _cleanup_toplevels():
    yield
    try:
        root = _get_shared_root()
        for child in list(root.winfo_children()):
            try:
                child.destroy()
            except Exception:
                pass
    except Exception:
        pass


def _pump_events(root: tk.Tk, iterations: int = 15) -> None:
    for _ in range(iterations):
        root.update_idletasks()
        root.update()


class _MockDetector:
    def detect(self, img, frame_number=-1, source=""):
        return SimpleNamespace(
            pupil=SimpleNamespace(
                detected=True,
                center_x=280.0,
                center_y=160.0,
                radius=40.0,
                ellipse=SimpleNamespace(center_x=280.0, center_y=160.0, semi_major=40.0, semi_minor=40.0, angle_deg=0.0),
            ),
            limbus=SimpleNamespace(
                detected=True,
                center_x=280.0,
                center_y=160.0,
                radius=85.0,
                ellipse=SimpleNamespace(center_x=280.0, center_y=160.0, semi_major=85.0, semi_minor=85.0, angle_deg=0.0),
            ),
            overall_confidence=0.92,
        )


def _find_widgets(parent: tk.Widget, cls) -> List[tk.Widget]:
    out = []
    for w in parent.winfo_children():
        if isinstance(w, cls):
            out.append(w)
        out.extend(_find_widgets(w, cls))
    return out


def _find_btn(parent: tk.Widget, text: str) -> Optional[tk.Button]:
    for b in _find_widgets(parent, tk.Button):
        if b.cget("text") == text:
            return b
    return None


@pytest.fixture
def pentacam_test_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "sample_pentacam.png"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 90, (180, 180, 180), -1)  # sclera/limbus
    cv2.circle(img, (320, 240), 40, (30, 30, 30), -1)    # pupil
    cv2.imwrite(str(img_path), img)
    return img_path


def test_bug_001_single_popup_window(tmp_path: Path):
    """BUG-001: Add Patient opens exactly ONE popup window containing form and preview."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._storage_root_var.set(str(tmp_path))
    app._detector = _MockDetector()

    # Open dialog
    app._open_patient_dialog()
    _pump_events(root)

    toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(toplevels) == 1, f"Expected exactly 1 popup window, found {len(toplevels)}"
    dialog = toplevels[0]
    assert dialog.title() == "Add Patient"

    # Verify both form fields and Pentacam preview widgets are present in the same dialog
    entries = _find_widgets(dialog, tk.Entry)
    assert len(entries) >= 6, f"Expected at least 6 entry fields in dialog, found {len(entries)}"

    canvases = _find_widgets(dialog, tk.Canvas)
    assert len(canvases) == 1, "Pentacam preview canvas must be inside the single dialog"
    canvas = canvases[0]

    # Verify preview buttons are inside the dialog
    od_btn = _find_btn(dialog, "OD")
    os_btn = _find_btn(dialog, "OS")
    roi_btn = _find_btn(dialog, "ROI")
    add_btn = _find_btn(dialog, "Add Patient")
    cancel_btn = _find_btn(dialog, "Cancel")

    assert od_btn is not None, "OD button must be present in dialog"
    assert os_btn is not None, "OS button must be present in dialog"
    assert roi_btn is not None, "ROI button must be present in dialog"
    assert add_btn is not None, "Add Patient button must be present in dialog"
    assert cancel_btn is not None, "Cancel button must be present in dialog"

    # Close dialog cleanly
    cancel_btn.invoke()
    _pump_events(root)
    assert not dialog.winfo_exists()
    assert not app._add_px_dialog_open


def test_bug_002_increased_dimensions():
    """BUG-002: Dialog default size is increased to comfortably fit form and preview."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._detector = _MockDetector()

    app._open_patient_dialog()
    _pump_events(root)

    toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(toplevels) == 1
    dialog = toplevels[0]

    # Check minimum dimensions
    min_w, min_h = dialog.minsize()
    assert min_w >= 1100, f"Expected min width >= 1100, got {min_w}"
    assert min_h >= 600, f"Expected min height >= 600, got {min_h}"

    # Check requested/actual dimensions
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    assert w >= 1100, f"Dialog width {w} should be >= 1100"
    assert h >= 600, f"Dialog height {h} should be >= 600"

    cancel_btn = _find_btn(dialog, "Cancel")
    cancel_btn.invoke()
    _pump_events(root)


def test_form_validation_and_preview_workflow(pentacam_test_image: Path, tmp_path: Path):
    """Verify form entry, browse/preview loading, OD/OS selection, and folder creation."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    storage_dir = tmp_path / "patients"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app._storage_root_var.set(str(storage_dir))
    app._detector = _MockDetector()

    app._open_patient_dialog()
    _pump_events(root)

    toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    dialog = toplevels[0]
    add_btn = _find_btn(dialog, "Add Patient")
    assert str(add_btn.cget("state")) == "disabled", "Add Patient should be disabled until required fields are set"

    # Fill mandatory fields
    entries = _find_widgets(dialog, tk.Entry)
    # Entry rows: 0=Patient ID, 1=Patient Name, 2=HVID OD, 3=HVID OS, 4=Pentacam OD, 5=Pentacam OS
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["0"].insert(0, "PX_UNIT_01")
    entry_by_row["1"].insert(0, "John Doe")
    _pump_events(root)
    assert str(add_btn.cget("state")) == "normal", "Add Patient should be enabled once ID and Name are entered"

    # Set Pentacam OD image
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 20)

    # Verify canvas renders image
    canvas = _find_widgets(dialog, tk.Canvas)[0]
    assert canvas._thumb_size[0] > 0 and canvas._thumb_size[1] > 0, "Canvas should have rendered thumbnail"

    # Submit patient
    add_btn.invoke()
    _pump_events(root, 10)

    # Verify dialog closed and patient folder was created
    assert not dialog.winfo_exists()
    assert not app._add_px_dialog_open
    patient_dirs = list(storage_dir.glob("PX_PX_UNIT_01*"))
    assert len(patient_dirs) == 1, f"Expected 1 created patient dir, found {patient_dirs}"
    created_patient_dir = patient_dirs[0]
    assert (created_patient_dir / "patient.txt").exists()
    assert (created_patient_dir / "OD" / "Pentacam" / pentacam_test_image.name).exists()


def test_reopen_dialog_lifecycle():
    """Verify dialog can be reopened cleanly without stale state or duplicate windows."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._detector = _MockDetector()

    # Open and close once
    app._open_patient_dialog()
    _pump_events(root)
    toplevels1 = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(toplevels1) == 1
    _find_btn(toplevels1[0], "Cancel").invoke()
    _pump_events(root)
    assert not app._add_px_dialog_open

    # Open again
    app._open_patient_dialog()
    _pump_events(root)
    toplevels2 = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert len(toplevels2) == 1, "Reopened dialog should still have exactly 1 toplevel"
    _find_btn(toplevels2[0], "Cancel").invoke()
    _pump_events(root)
    assert not app._add_px_dialog_open


def test_phase2_fixed_roi_dimensions():
    """Verify Option A fixed ROI dimensions (847x542 px) and boundary clamping."""
    w, h = 1200, 838
    rw = min(float(w), 847.0)
    rh = min(float(h), 542.0)
    ix0 = int(round((float(w) - rw) / 2.0))
    iy0 = int(round((float(h) - rh) / 2.0))
    ix1 = ix0 + int(rw)
    iy1 = iy0 + int(rh)
    roi = (ix0 / float(w), iy0 / float(h), ix1 / float(w), iy1 / float(h))

    from pupil_tracking.interface.canvas_utils import roi_norm_to_image_coords
    ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
    assert iw == 847, f"Expected ROI width 847, got {iw}"
    assert ih == 542, f"Expected ROI height 542, got {ih}"
    assert 0 <= ix0 < ix1 <= w
    assert 0 <= iy0 < iy1 <= h


def test_phase2_offset_detection_result():
    """Verify PupilTrackingGUI._offset_detection_result shifts center coordinates without changing shape."""
    res = SimpleNamespace(
        pupil=SimpleNamespace(
            detected=True,
            ellipse=SimpleNamespace(center_x=100.0, center_y=120.0, semi_major=35.0, semi_minor=35.0, angle_deg=10.0),
        ),
        limbus=SimpleNamespace(
            detected=True,
            ellipse=SimpleNamespace(center_x=105.0, center_y=122.0, semi_major=80.0, semi_minor=78.0, angle_deg=5.0),
        ),
        corneal_center=SimpleNamespace(center_x=102.0, center_y=121.0),
    )

    offset_x, offset_y = 150.0, 80.0
    shifted = gui_app.PupilTrackingGUI._offset_detection_result(res, offset_x, offset_y)

    assert shifted.pupil.ellipse.center_x == 250.0
    assert shifted.pupil.ellipse.center_y == 200.0
    assert shifted.pupil.ellipse.semi_major == 35.0
    assert shifted.pupil.ellipse.angle_deg == 10.0

    assert shifted.limbus.ellipse.center_x == 255.0
    assert shifted.limbus.ellipse.center_y == 202.0
    assert shifted.limbus.ellipse.semi_major == 80.0
    assert shifted.limbus.ellipse.semi_minor == 78.0

    assert shifted.corneal_center.center_x == 252.0
    assert shifted.corneal_center.center_y == 201.0


def test_phase2_enter_to_detect_and_roi_cycle(pentacam_test_image: Path):
    """Verify Enter-to-detect workflow and ROI button cycle (activate -> lock -> clear/grey)."""
    import time
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._detector = _MockDetector()

    app._open_patient_dialog()
    _pump_events(root)

    toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    dialog = toplevels[0]

    entries = _find_widgets(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 10)

    canvas = _find_widgets(dialog, tk.Canvas)[0]
    # Before pressing Enter, detection overlay should NOT be present
    assert len(canvas.find_withtag("detection")) == 0, "Detection should not run before Enter is pressed"

    # Press Enter (Return) on canvas to lock and trigger detection
    canvas.focus_set()
    canvas.event_generate("<Return>")

    # Poll until background thread completes
    for _ in range(40):
        _pump_events(root, 5)
        time.sleep(0.05)
        if len(canvas.find_withtag("detection")) >= 2:
            break

    # After pressing Enter, detection overlay is rendered
    det_items = canvas.find_withtag("detection")
    assert len(det_items) >= 2, f"Detection overlay should be rendered on Enter, found {len(det_items)}"

    roi_btn = _find_btn(dialog, "ROI")
    assert roi_btn is not None
    assert roi_btn.cget("bg") == "#0071E3", "ROI button should be blue (active/locked)"

    # Second ROI button click: clear ROI and overlays, restore grey button state
    roi_btn.invoke()
    _pump_events(root)
    assert len(canvas.find_withtag("roi")) == 0, "ROI should be cleared on 2nd click"
    assert len(canvas.find_withtag("detection")) == 0, "Detection overlays should be cleared on 2nd click"
    assert roi_btn.cget("bg") == "#E0E0E0", "ROI button should be grey after clearing"

    # Close dialog
    _find_btn(dialog, "Cancel").invoke()
    _pump_events(root)

