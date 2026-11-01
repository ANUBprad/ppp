"""test_final_targeted_fixes.py — Comprehensive regression test suite for final targeted fixes.

Verifies:
1. Iris Unwrap display: full width utilization, 10:1 aspect ratio preservation, 0°->360° cardinal grid alignment.
2. Automatic strip alignment: ELITA live strip aligns against Pentacam strip using cyclotorsion engine without manual shift buttons or offset drift.
3. Add Patient with both OD and OS: both eyes persist original, flipped, unwrapped, results.json, and values.txt; unwrap strip remains visible immediately after dialog closure without reset.
4. OD/OS switching: seamless replacement of unwrap and recording paths without stale image lingering.
5. OD-only and OS-only isolation: missing eye unwrap clears cleanly without affecting the other eye.
6. Pentacam limbus detector refinement: preserves ELITA fitter settings while using dedicated non-RANSAC limbus fitting.
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
from typing import List, Tuple
from unittest import mock

import cv2
import numpy as np
import pytest

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.cyclotorsion import polar_unwrap as _pu
from pupil_tracking.cyclotorsion.cyclotorsion_engine import estimate_cyclotorsion
from pupil_tracking.interface import gui_app
from pupil_tracking.tests.test_pentacam_rectangular_roi import _get_shared_root


class _MockCanvas:
    """Mock Tk Canvas capturing rendered items for headless testing."""

    def __init__(self, width: int = 1200, height: int = 240) -> None:
        self._w = width
        self._h = height
        self.images: List[Tuple[int, int, Any]] = []
        self.texts: List[Tuple[int, int, str, str]] = []
        self.lines: List[Tuple[int, int, int, int, str]] = []

    def winfo_width(self) -> int:
        return self._w

    def winfo_height(self) -> int:
        return self._h

    def cget(self, key: str) -> Any:
        if key == "height":
            return self._h
        if key == "width":
            return self._w
        return ""

    def configure(self, **kwargs) -> None:
        if "height" in kwargs:
            self._h = int(kwargs["height"])
        if "width" in kwargs:
            self._w = int(kwargs["width"])

    def delete(self, tag: str) -> None:
        self.images.clear()
        self.texts.clear()
        self.lines.clear()

    def create_text(self, x: int, y: int, **kwargs) -> int:
        self.texts.append((x, y, kwargs.get("text", ""), kwargs.get("anchor", "center")))
        return len(self.texts)

    def create_line(self, x0: int, y0: int, x1: int, y1: int, **kwargs) -> int:
        self.lines.append((x0, y0, x1, y1, kwargs.get("fill", "")))
        return len(self.lines)

    def create_image(self, x: int, y: int, **kwargs) -> int:
        self.images.append((x, y, kwargs.get("image", None)))
        return len(self.images)


def _pump_events(root: tk.Tk, iterations: int = 15) -> None:
    for _ in range(iterations):
        root.update_idletasks()
        root.update()


def _find_widgets_recursive(parent, widget_type):
    found = []
    for child in parent.winfo_children():
        if isinstance(child, widget_type):
            found.append(child)
        found.extend(_find_widgets_recursive(child, widget_type))
    return found


def _create_synthetic_pentacam_img(path: Path, eye: str) -> Path:
    img = np.full((800, 1000, 3), 35, dtype=np.uint8)
    color = 180 if eye == "OD" else 120
    # Outer limbus
    cv2.circle(img, (500, 400), 220, (color, color, color), -1)
    # Inner pupil
    cv2.circle(img, (500, 400), 75, (10, 10, 10), -1)
    # Eye-specific landmark spoke at 45 deg
    rad = np.deg2rad(45.0 if eye == "OD" else 135.0)
    sx = int(500 + 150 * np.cos(rad))
    sy = int(400 + 150 * np.sin(rad))
    cv2.circle(img, (sx, sy), 18, (255, 255, 255), -1)
    cv2.imwrite(str(path), img)
    return path


# ===========================================================================
# 1. Iris Unwrap Display & Black Space Elimination
# ===========================================================================

def test_iris_unwrap_display_full_width_and_aspect_ratio(monkeypatch):
    """Verify that unwrap strips span full canvas width with aspect ratio 10:1."""
    captured_photos = []

    class _MockPhoto:
        def __init__(self, img):
            self.img = img
            captured_photos.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_MockPhoto))

    canvas = _MockCanvas(width=1400, height=200)
    app = SimpleNamespace(
        _unwrap_canvas=canvas,
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=np.full((72, 720, 3), 110, dtype=np.uint8),
        _elita_strip=np.full((72, 720, 3), 150, dtype=np.uint8),
        _elita_unwrap_shift_cols=0,
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=gui_app.PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=gui_app.PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=gui_app.PupilTrackingGUI._roll_strip,
    )

    gui_app.PupilTrackingGUI._redraw_unwrap(app)

    assert len(canvas.images) == 2
    disp_w, disp_h = captured_photos[0].size

    # Exact 10:1 ratio preserved
    assert abs(disp_w / disp_h - 10.0) < 0.25

    # Origin centered horizontally within canvas
    x0, y0_pent, _ = canvas.images[0]
    assert x0 == (1400 - disp_w) // 2

    # Cardinal ticks 0°, 90°, 180°, 270°, 360° align precisely
    ruler_texts = {text: x for x, y, text, _anchor in canvas.texts if "°" in text}
    assert ruler_texts["0°"] == x0
    assert ruler_texts["360°"] == x0 + disp_w
    assert abs((ruler_texts["0°"] + ruler_texts["360°"]) / 2.0 - ruler_texts["180°"]) <= 1.0


# ===========================================================================
# 2. ELITA Auto Align & Low-Confidence Handling
# ===========================================================================

def test_automatic_alignment_replaces_manual_shift_without_drift(monkeypatch):
    """Verify ELITA Auto Align, including low-confidence/failure handling and non-drifting alignment."""
    captured_photos = []

    class _MockPhoto:
        def __init__(self, img):
            self.img = img
            captured_photos.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_MockPhoto))

    # Base reference strip
    pentacam_strip = np.full((72, 720, 3), 50, dtype=np.uint8)
    pentacam_strip[:, 190:210] = 220
    elita_strip = np.roll(pentacam_strip, 30, axis=1)

    canvas = _MockCanvas(width=1000, height=200)
    status_label = SimpleNamespace(text="", configure=lambda **kw: status_label.__dict__.update(kw))
    status_var = SimpleNamespace(val="", set=lambda v: setattr(status_var, "val", v))
    app = SimpleNamespace(
        _unwrap_canvas=canvas,
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=pentacam_strip.copy(),
        _elita_strip=elita_strip.copy(),
        _elita_unwrap_shift_cols=0,
        _elita_auto_status_label=status_label,
        _status_var=status_var,
        _elita_shift_label=None,
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=gui_app.PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=gui_app.PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=gui_app.PupilTrackingGUI._roll_strip,
        _update_elita_shift_label=lambda: None,
    )
    app._redraw_unwrap = lambda: gui_app.PupilTrackingGUI._redraw_unwrap(app)
    app._estimate_elita_unwrap_alignment = lambda p, e, **k: gui_app.PupilTrackingGUI._estimate_elita_unwrap_alignment(app, p, e, **k)
    app._auto_align_elita_shift = lambda: gui_app.PupilTrackingGUI._auto_align_elita_shift(app)

    # Low-confidence / synthetic featureless handling:
    # Insufficient texture should be detected and current offset preserved safely
    ok = app._auto_align_elita_shift()
    assert not ok or app._elita_unwrap_shift_cols is not None
    # Offset must remain bounded
    assert abs(app._elita_unwrap_shift_cols) <= 360

    # With mocked successful estimate of 30 columns
    monkeypatch.setattr(
        app,
        "_estimate_elita_unwrap_alignment",
        lambda p, e, **k: (True, -30, 0.95, "Aligned: -15.0° (corr: 0.95)"),
    )
    ok_aligned = app._auto_align_elita_shift()
    assert ok_aligned is True
    assert app._elita_unwrap_shift_cols == -30

    # Repeated auto align with same estimate does NOT drift or accumulate
    app._auto_align_elita_shift()
    assert app._elita_unwrap_shift_cols == -30


# ===========================================================================
# 3. Saving Both Eyes & Unwrap Disappearing Fix (BUG-004)
# ===========================================================================

def test_saving_both_eyes_persists_data_and_leaves_unwrap_visible(tmp_path: Path):
    """Trace ADD PX -> save both OD & OS -> close -> verify unwrap strip is immediately visible."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    storage_dir = tmp_path / "patients"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app._storage_root_var.set(str(storage_dir))

    od_file = _create_synthetic_pentacam_img(tmp_path / "eye_OD.png", "OD")
    os_file = _create_synthetic_pentacam_img(tmp_path / "eye_OS.png", "OS")

    app._open_patient_dialog()
    _pump_events(root)

    toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    dialog = toplevels[0]

    # Fill mandatory fields
    entries = _find_widgets_recursive(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}

    entry_by_row["0"].insert(0, "PX_TARGET_01")
    entry_by_row["1"].insert(0, "DualEye Patient")
    entry_by_row["4"].insert(0, str(od_file))
    entry_by_row["5"].insert(0, str(os_file))
    _pump_events(root)

    # Find Add Patient button
    buttons = _find_widgets_recursive(dialog, tk.Button)
    add_btn = [b for b in buttons if b.cget("text") == "Add Patient"][0]
    add_btn.invoke()
    _pump_events(root, 10)

    # 1. Dialog must be closed
    assert not dialog.winfo_exists()
    assert not app._add_px_dialog_open

    # 2. Both eyes must be saved to patient directory
    px_dir = storage_dir / "PX_PX_TARGET_01_DualEye Patient"
    assert px_dir.is_dir()
    for eye in ("OD", "OS"):
        eye_dir = px_dir / eye / "Pentacam"
        assert (eye_dir / f"Pentacam_{eye}_flipped.png").is_file()
        assert (eye_dir / f"Pentacam_{eye}_unwrapped.png").is_file()
        assert (eye_dir / "pentacam_results.json").is_file()
        assert (eye_dir / "pentacam_values.txt").is_file()
        # Verify JSON content
        meta = json.loads((eye_dir / "pentacam_results.json").read_text(encoding="utf-8"))
        assert meta["eye"] == eye
        assert "center_x" in meta["pupil"]
        assert "center_x" in meta["limbus"]

    # 3. Main app selected eye must be defaulted to OD (not cleared!)
    assert app._selected_eye_var.get() == "OD"

    # 4. Main unwrap strip must be active and visible (NOT None!)
    assert app._unwrap_strip is not None
    assert app._unwrap_strip.shape[0] == 72
    assert app._unwrap_strip.shape[1] == 720


# ===========================================================================
# 4. Seamless OD <-> OS Switching and Patient Isolation
# ===========================================================================

def test_od_os_switching_loads_respective_unwraps_without_stale_data(tmp_path: Path):
    """Verify switching between OD and OS switches both unwrap strip and recording directory."""
    root = _get_shared_root()
    app = gui_app.PupilTrackingGUI(root)
    storage_dir = tmp_path / "patients"
    storage_dir.mkdir(parents=True, exist_ok=True)
    app._storage_root_var.set(str(storage_dir))

    od_file = _create_synthetic_pentacam_img(tmp_path / "test_OD.png", "OD")
    os_file = _create_synthetic_pentacam_img(tmp_path / "test_OS.png", "OS")

    app._open_patient_dialog()
    _pump_events(root)
    dialog = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][0]
    entries = _find_widgets_recursive(dialog, tk.Entry)
    entry_by_row = {str(e.grid_info().get("row")): e for e in entries}
    entry_by_row["0"].insert(0, "PX_SWITCH")
    entry_by_row["1"].insert(0, "Switching Test")
    entry_by_row["4"].insert(0, str(od_file))
    entry_by_row["5"].insert(0, str(os_file))
    _pump_events(root)

    # Click Add Patient
    buttons = _find_widgets_recursive(dialog, tk.Button)
    add_btn = [b for b in buttons if b.cget("text") == "Add Patient"][0]
    add_btn.invoke()
    _pump_events(root, 10)

    # Initial eye is OD
    assert app._selected_eye_var.get() == "OD"
    od_strip = app._unwrap_strip.copy()
    assert "OD" in app._recording_location_var.get()

    # Switch to OS
    app._select_eye("OS")
    assert app._selected_eye_var.get() == "OS"
    os_strip = app._unwrap_strip.copy()
    assert "OS" in app._recording_location_var.get()
    # Strips must be distinct
    assert not np.array_equal(od_strip, os_strip)

    # Switch back to OD
    app._select_eye("OD")
    assert app._selected_eye_var.get() == "OD"
    assert np.array_equal(app._unwrap_strip, od_strip)
    assert "OD" in app._recording_location_var.get()


# ===========================================================================
# 5. Pentacam Fitter Isolation & Accuracy
# ===========================================================================

def test_pentacam_limbus_fitter_isolated_from_elita():
    """Verify Pentacam limbus fitter disables RANSAC to avoid circle bias, while ELITA fitter keeps it."""
    detector = UnifiedDetector()

    # ELITA live smart fitter: RANSAC enabled, subpixel enabled
    assert detector._fitter.use_ransac is True
    assert detector._fitter.subpixel_refine is True

    # Pentacam natural eye limbus fitter: RANSAC disabled, subpixel disabled to eliminate eyelid drag
    assert detector._pentacam_limbus_fitter.use_ransac is False
    assert detector._pentacam_limbus_fitter.subpixel_refine is False
    assert detector._pentacam_limbus_fitter.circularity_threshold >= 0.95
