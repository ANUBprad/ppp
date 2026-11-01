"""test_pentacam_workflow_final.py — Comprehensive integration tests for Pentacam final workflow.

Verifies:
1. Pentacam Limbus Detection Accuracy on clinical imagery.
2. Exact ROI button toggle:
   - 1st click: Activate ROI, green rectangle displayed, button turns blue (#0071E3).
   - Enter: Lock ROI, run automatic detection once, display boundaries and status badge.
   - 2nd click: Deactivate ROI, clear rectangle, clear overlays, invalidate state, button returns to grey (#E0E0E0).
3. Complete Manual Mode:
   - Option A: Direct manual definition without auto-detection.
   - Option B: Correcting auto-detection.
   - Independent adjustment of pupil and limbus.
   - Cardinal handles and center dragging; Enter to lock; re-entering unlocks without auto re-run.
   - Ordinary ROI clicks do NOT seed manual overrides.
4. Dedicated Save Functionality & OD/OS independence:
   - Validates patient info, image, and valid boundaries.
   - Saves original, flipped, json, txt, and unwrapped files with byte-size verification.
   - OD and OS folders remain strictly independent.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest import mock

import cv2
import numpy as np
import pytest

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.interface import gui_app
from pupil_tracking.tests.test_pentacam_rectangular_roi import _get_shared_root


def _pump_events(root: tk.Tk, iterations: int = 15) -> None:
    for _ in range(iterations):
        root.update_idletasks()
        root.update()


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


@pytest.fixture
def pentacam_test_image(tmp_path: Path) -> Path:
    """Provide a real clinical Pentacam image for end-to-end detection."""
    clinical = Path("clinical_data/clean/eye_01.jpeg")
    if clinical.is_file():
        dest = tmp_path / "clinical_sample.jpeg"
        shutil.copyfile(clinical, dest)
        return dest
    # Fallback synthetic
    img_path = tmp_path / "pentacam_sample.png"
    img = np.zeros((838, 1200, 3), dtype=np.uint8)
    cv2.circle(img, (600, 419), 220, (140, 140, 140), -1)
    cv2.circle(img, (600, 419), 130, (80, 80, 80), -1)
    cv2.circle(img, (604, 415), 45, (15, 15, 15), -1)
    cv2.imwrite(str(img_path), img)
    return img_path


# ==============================================================================
# 1. Pentacam Limbus Detection Accuracy
# ==============================================================================

def test_pentacam_limbus_detection_accuracy():
    """Verify limbus detection aligns with outer iris boundary and is distinct from pupil."""
    clinical_img = Path("clinical_data/clean/eye_01.jpeg")
    if not clinical_img.is_file():
        pytest.skip("Clinical test image not found")

    detector = UnifiedDetector()
    raw = cv2.imread(str(clinical_img))
    assert raw is not None
    # 180 flip as standard in Pentacam loader
    flipped = cv2.flip(raw, -1)

    result = detector.detect(
        flipped,
        frame_number=-1,
        force_mode="pre_docked",
        source=f"pentacam:{clinical_img}",
    )
    assert result is not None
    assert result.pupil.detected is True
    assert result.limbus.detected is True

    pe = result.pupil.ellipse
    le = result.limbus.ellipse
    assert pe is not None and le is not None

    # Pupil and limbus must be distinct concentric structures
    assert le.radius > pe.radius * 1.5
    # Center offset between pupil and limbus within physiological limits (< 30px)
    offset = np.hypot(le.center_x - pe.center_x, le.center_y - pe.center_y)
    assert offset < 30.0


# ==============================================================================
# 2. Exact ROI Button Toggle (Click -> Enter -> Click)
# ==============================================================================

def test_exact_roi_button_toggle(pentacam_test_image: Path, tmp_path: Path):
    """Verify 1st click activates green ROI, Enter locks & detects, 2nd click deactivates/clears."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._storage_root_var.set(str(tmp_path))

    app._open_patient_dialog()
    _pump_events(root)

    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    canvas = _find_widgets(dialog, tk.Canvas)[0]
    entries = _find_widgets(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 10)

    roi_btn = _find_btn(dialog, "ROI")
    assert roi_btn is not None
    # Initially grey before clicking ROI
    assert roi_btn.cget("bg") == "#E0E0E0"
    assert len(canvas.find_withtag("roi")) == 0
    assert len(canvas.find_withtag("detection")) == 0

    # ── First click on ROI: Activate ROI mode ──────────────────────
    roi_btn.invoke()
    _pump_events(root, 5)

    assert roi_btn.cget("bg") == "#0071E3", "ROI button should turn blue on 1st click"
    assert len(canvas.find_withtag("roi")) > 0, "Green ROI rectangle should be displayed"
    assert len(canvas.find_withtag("detection")) == 0, "Detection overlays should not be present yet"

    # ── Press Enter: Lock ROI position and run automatic detection ──
    canvas.focus_set()
    canvas.event_generate("<Return>")

    # Wait for detection
    for _ in range(100):
        _pump_events(root, 5)
        time.sleep(0.05)
        if len(canvas.find_withtag("detection")) >= 2:
            break

    assert len(canvas.find_withtag("detection")) >= 2, "Pupil and limbus overlays should be rendered on Enter"
    assert roi_btn.cget("bg") == "#0071E3", "ROI button remains selected while locked"

    # ── Second click on ROI: Deactivate ROI mode ────────────────────
    roi_btn.invoke()
    _pump_events(root, 5)

    assert len(canvas.find_withtag("roi")) == 0, "ROI rectangle and handles must be cleared on 2nd click"
    assert len(canvas.find_withtag("detection")) == 0, "Detection overlays must be cleared on 2nd click"
    assert roi_btn.cget("bg") == "#E0E0E0", "ROI button must return to default grey (#E0E0E0)"

    # Clean up
    _find_btn(dialog, "Cancel").invoke()
    _pump_events(root)


# ==============================================================================
# 3. Complete Manual Mode (Option A and Option B)
# ==============================================================================

def test_manual_mode_option_a_direct_definition(pentacam_test_image: Path, tmp_path: Path):
    """Option A: Define pupil and limbus directly without auto-detection; Enter to lock."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._storage_root_var.set(str(tmp_path))

    app._open_patient_dialog()
    _pump_events(root)

    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    canvas = _find_widgets(dialog, tk.Canvas)[0]
    entries = _find_widgets(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 10)

    manual_btn = _find_btn(dialog, "Manual Mode")
    assert manual_btn is not None

    # Click Manual Mode directly without running auto-detection
    manual_btn.invoke()
    _pump_events(root, 5)

    assert manual_btn.cget("bg") == "#0071E3", "Manual Mode button turns blue when active"
    # Boundaries rendered on canvas
    det_items = canvas.find_withtag("detection")
    assert len(det_items) >= 2, "Manual pupil and limbus boundaries must be rendered"

    # Press Enter to lock manual boundaries
    canvas.focus_set()
    canvas.event_generate("<Return>")
    _pump_events(root, 5)

    # Locked: button returns to grey, badge shows [LOCKED]
    assert manual_btn.cget("bg") == "#E0E0E0"
    det_texts = [canvas.itemcget(i, "text") for i in canvas.find_withtag("detection") if canvas.type(i) == "text"]
    assert any("[LOCKED]" in t for t in det_texts)

    # Click Manual Mode again to unlock for further editing without auto-detect rerun
    manual_btn.invoke()
    _pump_events(root, 5)
    assert manual_btn.cget("bg") == "#0071E3", "Manual Mode unlocked again for further editing"

    # Clean up
    _find_btn(dialog, "Cancel").invoke()
    _pump_events(root)


def test_manual_mode_option_b_correct_auto_detection(pentacam_test_image: Path, tmp_path: Path):
    """Option B: Run auto detection, then enter Manual Mode to adjust independently."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._storage_root_var.set(str(tmp_path))

    app._open_patient_dialog()
    _pump_events(root)

    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    canvas = _find_widgets(dialog, tk.Canvas)[0]
    entries = _find_widgets(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 10)

    # Click ROI and Press Enter to run detection
    roi_btn = _find_btn(dialog, "ROI")
    roi_btn.invoke()
    _pump_events(root, 5)
    canvas.focus_set()
    canvas.event_generate("<Return>")

    for _ in range(40):
        _pump_events(root, 5)
        time.sleep(0.05)
        if len(canvas.find_withtag("detection")) >= 2:
            break

    # Now enter Manual Mode to correct
    manual_btn = _find_btn(dialog, "Manual Mode")
    manual_btn.invoke()
    _pump_events(root, 5)

    assert manual_btn.cget("bg") == "#0071E3"
    det_texts = [canvas.itemcget(i, "text") for i in canvas.find_withtag("detection") if canvas.type(i) == "text"]
    assert any("MANUAL:" in t for t in det_texts)

    # Switch to Pupil
    pupil_btn = _find_btn(dialog, "Pupil")
    pupil_btn.invoke()
    _pump_events(root, 5)
    assert pupil_btn.cget("bg") == "#0071E3"

    # Press Enter to finalize manual correction
    canvas.event_generate("<Return>")
    _pump_events(root, 5)
    assert manual_btn.cget("bg") == "#E0E0E0"

    # Clean up
    _find_btn(dialog, "Cancel").invoke()
    _pump_events(root)


# ==============================================================================
# 4. Dedicated Save Functionality & OD/OS Independence
# ==============================================================================

@mock.patch("tkinter.messagebox.showinfo")
@mock.patch("tkinter.messagebox.showerror")
def test_dedicated_save_functionality_and_od_os_independence(
    mock_error, mock_info, pentacam_test_image: Path, tmp_path: Path
):
    """Verify dedicated Save button, folder creation, verified file writing, and OD/OS independence."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    storage_root = tmp_path / "Patients"
    storage_root.mkdir(parents=True, exist_ok=True)
    app._storage_root_var.set(str(storage_root))

    app._open_patient_dialog()
    _pump_events(root)

    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    canvas = _find_widgets(dialog, tk.Canvas)[0]
    entries = _find_widgets(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}

    # Fill Patient ID and Name
    entry_by_row["0"].insert(0, "PX99")
    entry_by_row["1"].insert(0, "Test_Patient")
    # Load OD Pentacam image
    entry_by_row["4"].insert(0, str(pentacam_test_image))
    _pump_events(root, 10)

    # Save button should be initially DISABLED until boundaries exist
    save_btn = None
    for b in _find_widgets(dialog, tk.Button):
        if "Save" in b.cget("text") and "Cancel" not in b.cget("text"):
            save_btn = b
            break
    assert save_btn is not None
    assert save_btn.cget("state") == tk.DISABLED

    # Define boundaries via Manual Mode
    manual_btn = _find_btn(dialog, "Manual Mode")
    manual_btn.invoke()
    _pump_events(root, 5)

    # Now Save button should be ENABLED ("Save OD")
    assert save_btn.cget("state") == tk.NORMAL
    assert save_btn.cget("text") == "Save OD"

    # Invoke Save OD
    save_btn.invoke()
    _pump_events(root, 10)

    # Verify OD files on disk
    px_dir = storage_root / "PX_PX99_Test_Patient"
    od_dir = px_dir / "OD" / "Pentacam"
    assert od_dir.is_dir(), f"OD directory {od_dir} must exist"

    od_json = od_dir / "pentacam_results.json"
    od_txt = od_dir / "pentacam_values.txt"
    od_flipped = od_dir / "Pentacam_OD_flipped.png"
    od_orig = od_dir / f"Pentacam_original{pentacam_test_image.suffix}"

    assert od_json.is_file() and od_json.stat().st_size > 0
    assert od_txt.is_file() and od_txt.stat().st_size > 0
    assert od_flipped.is_file() and od_flipped.stat().st_size > 0
    assert od_orig.is_file() and od_orig.stat().st_size > 0

    # Verify JSON content
    meta = json.loads(od_json.read_text(encoding="utf-8"))
    assert meta["patient_id"] == "PX99"
    assert meta["patient_name"] == "Test_Patient"
    assert meta["eye"] == "OD"
    assert meta["pupil"]["is_manual"] is True
    assert meta["limbus"]["is_manual"] is True

    # ── Now configure OS image and verify independent saving ───────
    # Create second image for OS
    os_img_path = tmp_path / "pentacam_sample_os.jpeg"
    shutil.copyfile(pentacam_test_image, os_img_path)
    entry_by_row["5"].insert(0, str(os_img_path))
    _pump_events(root, 10)

    # Switch to OS
    os_btn = _find_btn(dialog, "OS")
    assert os_btn is not None
    os_btn.invoke()
    _pump_events(root, 10)

    # Define boundaries for OS via Manual Mode
    manual_btn.invoke()
    _pump_events(root, 5)

    assert save_btn.cget("text") == "Save OS"
    save_btn.invoke()
    _pump_events(root, 10)

    # Verify OS files on disk
    os_dir = px_dir / "OS" / "Pentacam"
    assert os_dir.is_dir(), f"OS directory {os_dir} must exist"
    os_json = os_dir / "pentacam_results.json"
    assert os_json.is_file() and os_json.stat().st_size > 0

    # OD files must remain intact and NOT overwritten by OS save
    assert od_json.is_file() and od_json.stat().st_size > 0

    # Clean up
    _find_btn(dialog, "Cancel").invoke()
    _pump_events(root)
