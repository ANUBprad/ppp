"""test_od_os_pentacam_mapping.py — Targeted regression tests for BUG-004: OD/OS Pentacam Image Mapping.

Verifies:
1. Saving a patient with both OD and OS Pentacam images.
2. Selecting OD and verifying the bottom window loads the OD image.
3. Selecting OS and verifying the bottom window replaces it with the OS image.
4. Repeatedly switching OD → OS → OD without stale images.
5. Keeping the selected patient's images separate from another patient's images.
6. Handling a missing OD or OS image without showing the other eye's image.
7. Completing OD recording, switching to OS, and completing OS recording with correct folder and metadata associations.
8. Safety lock: preventing eye switching while recording is active.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest import mock

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


def _create_distinct_pentacam_image(path: Path, eye_marker: str, color_val: int) -> Path:
    """Create a synthetic Pentacam image with unique visual characteristics for testing."""
    img = np.full((838, 1200, 3), 40, dtype=np.uint8)
    # Limbus
    cv2.circle(img, (600, 419), 220, (color_val, color_val, color_val), -1)
    # Iris pattern with eye marker
    cv2.putText(
        img,
        f"EYE_{eye_marker}",
        (500, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
    )
    # Pupil
    cv2.circle(img, (600, 419), 70, (10, 10, 10), -1)
    cv2.imwrite(str(path), img)
    return path


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
def test_setup(tmp_path: Path):
    od_img = _create_distinct_pentacam_image(tmp_path / "sample_OD.png", "OD", 150)
    os_img = _create_distinct_pentacam_image(tmp_path / "sample_OS.png", "OS", 200)

    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    app._storage_root_var.set(str(tmp_path))

    return SimpleNamespace(
        tmp_path=tmp_path,
        od_img=od_img,
        os_img=os_img,
        root=root,
        app=app,
    )


# ==============================================================================
# 1. Saving a patient with both OD and OS Pentacam images through ADD PX
# ==============================================================================

def test_1_save_patient_with_both_od_and_os(test_setup):
    """Test creating a patient with both OD and OS Pentacam images."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    app._open_patient_dialog()
    _pump_events(root)

    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    entries = _find_widgets(dialog, tk.Entry)

    # Set patient ID and Name
    entries[0].delete(0, "end")
    entries[0].insert(0, "SYNTH_001")
    entries[1].delete(0, "end")
    entries[1].insert(0, "TEST_SUBJECT")

    # Set OD and OS images
    entries[4].delete(0, "end")
    entries[4].insert(0, str(test_setup.od_img))
    entries[5].delete(0, "end")
    entries[5].insert(0, str(test_setup.os_img))
    _pump_events(root)

    add_btn = _find_btn(dialog, "Add Patient")
    assert add_btn is not None
    assert add_btn.cget("state") == tk.NORMAL
    add_btn.invoke()
    _pump_events(root)

    # Patient folder created
    patient_dir = tmp_path / "PX_SYNTH_001_TEST_SUBJECT"
    assert patient_dir.is_dir()
    assert (patient_dir / "OD" / "Pentacam").is_dir()
    assert (patient_dir / "OS" / "Pentacam").is_dir()

    # Images copied to each eye
    assert list((patient_dir / "OD" / "Pentacam").glob("*.png"))
    assert list((patient_dir / "OS" / "Pentacam").glob("*.png"))
    assert (patient_dir / "OD" / "Pentacam" / "pentacam_values.txt").is_file()
    assert (patient_dir / "OS" / "Pentacam" / "pentacam_values.txt").is_file()


# ==============================================================================
# 2. Selecting OD loads OD image, Selecting OS replaces with OS image
# ==============================================================================

def test_2_and_3_od_os_image_loading_and_replacement(test_setup):
    """Test that selecting OD loads OD image and selecting OS immediately replaces it."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    # Setup patient directly
    patient_dir = tmp_path / "PX_SYNTH_002_EYE_TEST"
    od_pentacam = patient_dir / "OD" / "Pentacam"
    os_pentacam = patient_dir / "OS" / "Pentacam"
    od_pentacam.mkdir(parents=True, exist_ok=True)
    os_pentacam.mkdir(parents=True, exist_ok=True)

    # Save distinct unwrapped strips
    od_strip = np.full((72, 720, 3), 100, dtype=np.uint8)
    cv2.putText(od_strip, "OD_STRIP", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(od_pentacam / "Pentacam_OD_unwrapped.png"), od_strip)

    os_strip = np.full((72, 720, 3), 200, dtype=np.uint8)
    cv2.putText(os_strip, "OS_STRIP", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imwrite(str(os_pentacam / "Pentacam_OS_unwrapped.png"), os_strip)

    app._active_patient_dir = patient_dir
    app._patient_id_var.set("SYNTH_002")
    app._patient_name_var.set("EYE_TEST")
    app._update_patient_controls()

    # 1. Select OD
    app._select_eye("OD")
    _pump_events(root)

    assert app._selected_eye_var.get() == "OD"
    assert app._unwrap_strip is not None
    # Verify the unwrap strip contains OD content (mean intensity around 100)
    assert 90 <= np.mean(app._unwrap_strip) <= 110
    assert "OD" in app._unwrap_outer.cget("text")

    # 2. Select OS -> Must replace OD immediately with OS image
    app._select_eye("OS")
    _pump_events(root)

    assert app._selected_eye_var.get() == "OS"
    assert app._unwrap_strip is not None
    # Verify the unwrap strip contains OS content (mean intensity around 200, distinctly different)
    assert 190 <= np.mean(app._unwrap_strip) <= 210
    assert "OS" in app._unwrap_outer.cget("text")


# ==============================================================================
# 4. Repeatedly switching OD → OS → OD without stale images
# ==============================================================================

def test_4_repeated_switching_without_stale_images(test_setup):
    """Test switching OD -> OS -> OD multiple times preserves eye isolation and never leaves stale images."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    patient_dir = tmp_path / "PX_SYNTH_003_CYCLE"
    od_pentacam = patient_dir / "OD" / "Pentacam"
    os_pentacam = patient_dir / "OS" / "Pentacam"
    od_pentacam.mkdir(parents=True, exist_ok=True)
    os_pentacam.mkdir(parents=True, exist_ok=True)

    od_strip = np.full((72, 720, 3), 75, dtype=np.uint8)
    cv2.imwrite(str(od_pentacam / "Pentacam_OD_unwrapped.png"), od_strip)
    os_strip = np.full((72, 720, 3), 175, dtype=np.uint8)
    cv2.imwrite(str(os_pentacam / "Pentacam_OS_unwrapped.png"), os_strip)

    app._active_patient_dir = patient_dir
    app._patient_id_var.set("SYNTH_003")
    app._patient_name_var.set("CYCLE")
    app._update_patient_controls()

    for cycle in range(3):
        # Switch to OD
        app._select_eye("OD")
        _pump_events(root)
        assert app._selected_eye_var.get() == "OD"
        assert np.mean(app._unwrap_strip) < 100
        assert "OD" in app._unwrap_outer.cget("text")

        # Switch to OS
        app._select_eye("OS")
        _pump_events(root)
        assert app._selected_eye_var.get() == "OS"
        assert np.mean(app._unwrap_strip) > 150
        assert "OS" in app._unwrap_outer.cget("text")


# ==============================================================================
# 5. Keeping selected patient's images separate from another patient's images
# ==============================================================================

def test_5_patient_isolation(test_setup):
    """Test that switching patients completely reloads from the new patient's folder without cross-contamination."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    # Patient A
    px_a_dir = tmp_path / "PX_PATIENT_A_JANE"
    (px_a_dir / "OD" / "Pentacam").mkdir(parents=True, exist_ok=True)
    strip_a = np.full((72, 720, 3), 50, dtype=np.uint8)
    cv2.imwrite(str(px_a_dir / "OD" / "Pentacam" / "Pentacam_OD_unwrapped.png"), strip_a)

    # Patient B
    px_b_dir = tmp_path / "PX_PATIENT_B_JOHN"
    (px_b_dir / "OD" / "Pentacam").mkdir(parents=True, exist_ok=True)
    strip_b = np.full((72, 720, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(px_b_dir / "OD" / "Pentacam" / "Pentacam_OD_unwrapped.png"), strip_b)

    # Activate Patient A
    app._active_patient_dir = px_a_dir
    app._patient_id_var.set("PATIENT_A")
    app._patient_name_var.set("JANE")
    app._select_eye("OD")
    _pump_events(root)
    assert 45 <= np.mean(app._unwrap_strip) <= 55

    # Switch to Patient B
    app._active_patient_dir = px_b_dir
    app._patient_id_var.set("PATIENT_B")
    app._patient_name_var.set("JOHN")
    app._select_eye("OD")
    _pump_events(root)
    assert 145 <= np.mean(app._unwrap_strip) <= 155


# ==============================================================================
# 6. Handling missing OD or OS image without showing the other eye's image
# ==============================================================================

def test_6_missing_eye_image_clears_window(test_setup):
    """Test that selecting an eye with a missing image immediately clears the window and shows a clear message."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    patient_dir = tmp_path / "PX_SYNTH_006_PARTIAL"
    (patient_dir / "OD" / "Pentacam").mkdir(parents=True, exist_ok=True)
    (patient_dir / "OS" / "Pentacam").mkdir(parents=True, exist_ok=True)

    # Only OD has an image; OS is empty
    od_strip = np.full((72, 720, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(patient_dir / "OD" / "Pentacam" / "Pentacam_OD_unwrapped.png"), od_strip)

    app._active_patient_dir = patient_dir
    app._patient_id_var.set("SYNTH_006")
    app._patient_name_var.set("PARTIAL")
    app._update_patient_controls()

    # Select OD first
    app._select_eye("OD")
    _pump_events(root)
    assert app._unwrap_strip is not None

    # Select OS (which has no image)
    app._select_eye("OS")
    _pump_events(root)

    # Window must be cleared, not showing OD's image!
    assert app._unwrap_strip is None
    assert "No Pentacam image saved" in app._unwrap_outer.cget("text")
    assert "No Pentacam image saved for OS — bottom window cleared" in app._status_var.get()


# ==============================================================================
# 7. Completing OD recording, switching to OS, and completing OS recording
# ==============================================================================

def test_7_recording_workflow_with_eye_switching(test_setup):
    """Test OD recording -> switch to OS -> OS recording keeps paths and metadata independent."""
    app = test_setup.app
    root = test_setup.root
    tmp_path = test_setup.tmp_path

    patient_dir = tmp_path / "PX_SYNTH_007_REC"
    (patient_dir / "OD" / "Pentacam").mkdir(parents=True, exist_ok=True)
    (patient_dir / "OS" / "Pentacam").mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(patient_dir / "OD" / "Pentacam" / "Pentacam_OD_unwrapped.png"), np.full((72, 720, 3), 100, dtype=np.uint8))
    cv2.imwrite(str(patient_dir / "OS" / "Pentacam" / "Pentacam_OS_unwrapped.png"), np.full((72, 720, 3), 200, dtype=np.uint8))

    app._active_patient_dir = patient_dir
    app._patient_id_var.set("SYNTH_007")
    app._patient_name_var.set("REC")
    app._current_image = np.zeros((480, 640, 3), dtype=np.uint8)
    app._update_patient_controls()

    # 1. Select OD
    app._select_eye("OD")
    _pump_events(root)
    od_rec_path = app._patient_recording_path()
    assert str(patient_dir / "OD") in od_rec_path

    recording_state = {"active": False}
    def mock_start(*args, **kwargs):
        recording_state["active"] = True
        return True
    def mock_stop(ret_path):
        def _stop(*args, **kwargs):
            recording_state["active"] = False
            return ret_path
        return _stop

    with mock.patch("tkinter.messagebox.showinfo"), \
         mock.patch.object(type(app._recorder), "is_recording", new_callable=mock.PropertyMock, side_effect=lambda: recording_state["active"]):
        with mock.patch.object(app._recorder, "start", side_effect=mock_start) as m_start_od, \
             mock.patch.object(app._recorder, "stop", side_effect=mock_stop(od_rec_path)) as m_stop_od:
            
            # Start OD recording
            app._start_recording()
            assert m_start_od.called
            assert app._recorder.is_recording is True

            # 2. Safety check: attempt to switch eye while recording is active
            with mock.patch("tkinter.messagebox.showwarning") as mock_warn:
                app._select_eye("OS")
                assert mock_warn.called
                assert app._selected_eye_var.get() == "OD", "Eye must NOT switch while recording is active"

            # 3. Stop OD recording
            app._stop_recording()
            assert m_stop_od.called
            assert app._recorder.is_recording is False

        # 4. Switch to OS
        app._select_eye("OS")
        _pump_events(root)
        assert app._selected_eye_var.get() == "OS"
        os_rec_path = app._patient_recording_path()
        assert str(patient_dir / "OS") in os_rec_path

        # Start and stop OS recording
        with mock.patch.object(app._recorder, "start", side_effect=mock_start) as m_start_os, \
             mock.patch.object(app._recorder, "stop", side_effect=mock_stop(os_rec_path)) as m_stop_os:
            app._start_recording()
            assert m_start_os.called
            assert app._recorder.is_recording is True
            app._stop_recording()
            assert m_stop_os.called
            assert app._recorder.is_recording is False

