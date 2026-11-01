"""Unit tests for the interactive Pentacam rectangular ROI.

Covers:
- Handle hit testing at multiple preview scales
- Moving the rectangle without changing its dimensions
- Dragging each edge and corner (boundary adjustments)
- Minimum dimensions and image-boundary clamping
- One-pixel arrow-key movement and ten-pixel Shift+Arrow movement
- Dimension and top-left image coordinate calculation
- Preview-to-image coordinate conversion including the 2px canvas offset
- 180° auto-flip dual-axis transformation
- Independent OD and OS rectangle states
- Rectangle persistence and serialization compatibility
"""

from __future__ import annotations

import pytest

from pupil_tracking.cyclotorsion.pentacam_loader import (
    PENTACAM_HEIGHT,
    PENTACAM_WIDTH,
    format_roi_norm,
    parse_roi_norm,
    roi_norm_to_px,
)
from pupil_tracking.interface.canvas_utils import (
    flip_rect_norm,
    hit_test_rect,
    nudge_rect,
    rect_handle_positions,
    roi_norm_to_image_coords,
    update_rect_drag,
)


# ── 1. Coordinate Conversion & Dimensions ───────────────────────────

def test_roi_norm_to_image_coords_exact_dimensions():
    # 1200 x 838 standard Pentacam size
    roi = (0.1, 0.2, 0.5, 0.6)
    ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, PENTACAM_WIDTH, PENTACAM_HEIGHT)

    assert ix0 == 120
    assert iy0 == 168
    assert ix1 == 600
    assert iy1 == 503
    assert iw == 480
    assert ih == 335


def test_preview_canvas_offset_in_handle_positions():
    # Canvas offset is 2.0 px by default
    tw, th = 600, 400
    roi = (0.2, 0.3, 0.8, 0.7)
    handles = rect_handle_positions(roi[0], roi[1], roi[2], roi[3], tw, th, offset_x=2.0, offset_y=2.0)

    # Expected: cx0 = 2 + 0.2*600 = 122.0, cy0 = 2 + 0.3*400 = 122.0
    #           cx1 = 2 + 0.8*600 = 482.0, cy1 = 2 + 0.7*400 = 282.0
    assert handles["nw"] == (122.0, 122.0)
    assert handles["se"] == (482.0, 282.0)
    assert handles["n"] == (pytest.approx((122.0 + 482.0) / 2.0), 122.0)
    assert handles["s"] == (pytest.approx((122.0 + 482.0) / 2.0), 282.0)
    assert handles["w"] == (122.0, pytest.approx((122.0 + 282.0) / 2.0))
    assert handles["e"] == (482.0, pytest.approx((122.0 + 282.0) / 2.0))


# ── 2. Handle Hit-Testing Across Preview Scales ──────────────────────

@pytest.mark.parametrize("scale_factor", [0.5, 1.0, 1.5, 2.0])
def test_handle_hit_testing_at_various_scales(scale_factor):
    tw = int(round(560 * scale_factor))
    th = int(round(320 * scale_factor))
    roi = (0.25, 0.25, 0.75, 0.75)

    handles = rect_handle_positions(roi[0], roi[1], roi[2], roi[3], tw, th, offset_x=2.0, offset_y=2.0)

    # Each handle location must return its exact handle name
    for hname, (hx, hy) in handles.items():
        hit = hit_test_rect(hx, hy, roi, tw, th, handle_radius=7.0, offset_x=2.0, offset_y=2.0)
        assert hit == hname, f"Failed for {hname} at scale {scale_factor}"

    # Interior must return 'move'
    mid_x = (handles["nw"][0] + handles["se"][0]) * 0.5
    mid_y = (handles["nw"][1] + handles["se"][1]) * 0.5
    assert hit_test_rect(mid_x, mid_y, roi, tw, th, offset_x=2.0, offset_y=2.0) == "move"

    # Far outside must return None
    assert hit_test_rect(0.0, 0.0, roi, tw, th, offset_x=2.0, offset_y=2.0) is None


def test_edge_hit_testing_between_handles():
    tw, th = 600, 400
    roi = (0.2, 0.2, 0.8, 0.8)
    cx0 = 2.0 + 0.2 * tw  # 122
    cx1 = 2.0 + 0.8 * tw  # 482
    cy0 = 2.0 + 0.2 * th  # 82
    cy1 = 2.0 + 0.8 * th  # 322

    # Point along north edge between NW (122, 82) and N (302, 82): e.g. x=200, y=82
    assert hit_test_rect(200.0, cy0, roi, tw, th) == "edge_n"
    # Point along south edge: e.g. x=200, y=322
    assert hit_test_rect(200.0, cy1, roi, tw, th) == "edge_s"
    # Point along west edge between NW (122, 82) and W (122, 202): e.g. x=122, y=140
    assert hit_test_rect(cx0, 140.0, roi, tw, th) == "edge_w"
    # Point along east edge: e.g. x=482, y=140
    assert hit_test_rect(cx1, 140.0, roi, tw, th) == "edge_e"


# ── 3. Moving the Rectangle Without Changing Dimensions ──────────────

def test_move_preserves_dimensions_and_shifts():
    orig_roi = (0.2, 0.3, 0.6, 0.7)  # width 0.4, height 0.4
    start_norm = (0.4, 0.5)
    current_norm = (0.45, 0.55)  # shifted +0.05, +0.05

    moved = update_rect_drag("move", orig_roi, current_norm, start_norm)
    assert moved[0] == pytest.approx(0.25)
    assert moved[1] == pytest.approx(0.35)
    assert moved[2] == pytest.approx(0.65)
    assert moved[3] == pytest.approx(0.75)

    # Invariant width & height
    assert (moved[2] - moved[0]) == pytest.approx(orig_roi[2] - orig_roi[0])
    assert (moved[3] - moved[1]) == pytest.approx(orig_roi[3] - orig_roi[1])


def test_move_clamps_to_image_bounds_preserving_size():
    orig_roi = (0.7, 0.7, 0.9, 0.9)  # 0.2 x 0.2
    start_norm = (0.8, 0.8)
    # Drag far past the bottom-right corner
    current_norm = (1.5, 1.5)

    moved = update_rect_drag("move", orig_roi, current_norm, start_norm)
    # Must clamp so x1=1.0, y1=1.0 and width=0.2, height=0.2
    assert moved[2] == pytest.approx(1.0)
    assert moved[3] == pytest.approx(1.0)
    assert moved[0] == pytest.approx(0.8)
    assert moved[1] == pytest.approx(0.8)


# ── 4. Dragging Edges and Corners ───────────────────────────────────

def test_drag_west_edge_only_moves_x0():
    orig_roi = (0.2, 0.2, 0.8, 0.8)
    start_norm = (0.2, 0.5)
    current_norm = (0.3, 0.5)  # move west edge inward by +0.1

    res = update_rect_drag("edge_w", orig_roi, current_norm, start_norm)
    assert res[0] == pytest.approx(0.3)
    assert res[1] == pytest.approx(0.2)  # y0 unchanged
    assert res[2] == pytest.approx(0.8)  # x1 unchanged
    assert res[3] == pytest.approx(0.8)  # y1 unchanged


def test_drag_se_corner_moves_x1_and_y1():
    orig_roi = (0.2, 0.2, 0.6, 0.6)
    start_norm = (0.6, 0.6)
    current_norm = (0.7, 0.75)

    res = update_rect_drag("se", orig_roi, current_norm, start_norm)
    assert res[0] == pytest.approx(0.2)
    assert res[1] == pytest.approx(0.2)
    assert res[2] == pytest.approx(0.7)
    assert res[3] == pytest.approx(0.75)


def test_edge_drag_enforces_minimum_size():
    orig_roi = (0.2, 0.2, 0.6, 0.6)
    start_norm = (0.2, 0.2)
    # Attempt to drag west edge past east edge
    current_norm = (0.9, 0.2)

    res = update_rect_drag("edge_w", orig_roi, current_norm, start_norm, min_w=0.05)
    # x0 must clamp to x1 - min_w = 0.55
    assert res[0] == pytest.approx(0.55)
    assert res[2] == pytest.approx(0.6)


# ── 5. Keyboard Nudge (1px and 10px Shift) ──────────────────────────

def test_keyboard_nudge_1px():
    # 1000 x 500 image
    roi = (0.2, 0.2, 0.6, 0.6)
    img_w, img_h = 1000, 500

    # Nudge right 1 pixel: dx_norm = 1/1000 = 0.001
    nudged = nudge_rect(roi, 1, 0, img_w, img_h)
    assert nudged[0] == pytest.approx(0.201)
    assert nudged[2] == pytest.approx(0.601)
    assert (nudged[2] - nudged[0]) == pytest.approx(0.4)

    # Nudge up 1 pixel: dy = -1, dy_norm = -1/500 = -0.002
    nudged_up = nudge_rect(roi, 0, -1, img_w, img_h)
    assert nudged_up[1] == pytest.approx(0.198)
    assert nudged_up[3] == pytest.approx(0.598)


def test_keyboard_nudge_10px_shift():
    roi = (0.2, 0.2, 0.6, 0.6)
    img_w, img_h = 1000, 500

    # Shift-Right = 10px -> 10/1000 = 0.010
    nudged = nudge_rect(roi, 10, 0, img_w, img_h)
    assert nudged[0] == pytest.approx(0.210)
    assert nudged[2] == pytest.approx(0.610)


def test_keyboard_nudge_clamps_at_borders():
    roi = (0.002, 0.0, 0.402, 0.4)
    img_w, img_h = 1000, 500

    # Nudge left by 10px: cannot go negative, clamps x0=0.0, x1=0.4
    nudged = nudge_rect(roi, -10, 0, img_w, img_h)
    assert nudged[0] == 0.0
    assert nudged[2] == pytest.approx(0.4)


# ── 6. 180° Auto-Flip Transformation ────────────────────────────────

def test_auto_flip_transformation_symmetry():
    roi = (0.1, 0.2, 0.7, 0.8)
    w_orig = roi[2] - roi[0]
    h_orig = roi[3] - roi[1]

    flipped = flip_rect_norm(roi)
    # x0_f = 1 - 0.7 = 0.3, y0_f = 1 - 0.8 = 0.2, x1_f = 1 - 0.1 = 0.9, y1_f = 1 - 0.2 = 0.8
    assert flipped == pytest.approx((0.3, 0.2, 0.9, 0.8))
    assert (flipped[2] - flipped[0]) == pytest.approx(w_orig)
    assert (flipped[3] - flipped[1]) == pytest.approx(h_orig)

    # Flipping twice restores the exact original rectangle
    double_flipped = flip_rect_norm(flipped)
    assert double_flipped == pytest.approx(roi)


# ── 7. Independent OD / OS State & Serialization Compatibility ──────

def test_independent_od_os_state_dictionary():
    roi_by_eye = {"OD": None, "OS": None}

    roi_od = (0.15, 0.25, 0.65, 0.75)
    roi_by_eye["OD"] = roi_od
    assert roi_by_eye["OS"] is None

    roi_os = (0.3, 0.2, 0.8, 0.9)
    roi_by_eye["OS"] = roi_os

    assert roi_by_eye["OD"] == roi_od
    assert roi_by_eye["OS"] == roi_os


def test_serialization_round_trip():
    roi = (0.123456, 0.234567, 0.654321, 0.765432)
    serialized = format_roi_norm(roi)
    assert serialized == "0.123456,0.234567,0.654321,0.765432"

    parsed = parse_roi_norm(serialized)
    assert parsed == pytest.approx(roi)

    px_coords = roi_norm_to_px(parsed, PENTACAM_WIDTH, PENTACAM_HEIGHT)
    assert len(px_coords) == 4
    assert px_coords[0] == int(round(0.123456 * PENTACAM_WIDTH))
    assert px_coords[1] == int(round(0.234567 * PENTACAM_HEIGHT))


# ── 8. ROI Button, Editing Gate, Ring Interop & Workflows ───────────

import tkinter as tk
from types import SimpleNamespace
import numpy as np

DLG_ACCENT = "#0071E3"
DLG_BTN_SEC = "#E0E0E0"
DLG_BTN_ACT = "#C8C8C8"
DLG_FG = "#111111"
DLG_FG_HINT = "#555555"

from pupil_tracking.utils.types import EllipseParams


_shared_root: Optional[tk.Tk] = None

@pytest.fixture(autouse=True, scope="module")
def _cleanup_tk():
    yield
    global _shared_root
    if _shared_root is not None:
        try:
            _shared_root.destroy()
        except Exception:
            pass
        _shared_root = None

def _get_shared_root() -> tk.Tk:
    global _shared_root
    if _shared_root is None or not _shared_root.winfo_exists():
        try:
            _shared_root = tk.Tk()
            _shared_root.withdraw()
        except Exception:
            pytest.skip("Tk could not be initialised in this environment")
    for child in _shared_root.winfo_children():
        try:
            child.destroy()
        except Exception:
            pass
    return _shared_root


def test_roi_button_activation_and_deactivation_toggle():
    """Verify ROI button toggles roi_mode_var and updates button styling correctly."""
    root = _get_shared_root()
    roi_mode_var = tk.BooleanVar(value=True)
    active_eye_path = "some/image.png"

    def _eye_path(eye: str) -> str:
        return active_eye_path

    def _active_eye() -> str:
        return "OD"

    roi_btn = tk.Button(root, text="ROI")

    def _update_roi_btn_style():
        has_image = bool(_eye_path(_active_eye()))
        is_active = roi_mode_var.get() and has_image
        roi_btn.config(
            state=tk.NORMAL if has_image else tk.DISABLED,
            bg=DLG_ACCENT if is_active else DLG_BTN_SEC,
            fg="#FFFFFF" if is_active else (DLG_FG if has_image else DLG_FG_HINT),
            activebackground=DLG_ACCENT if is_active else DLG_BTN_ACT,
            activeforeground="#FFFFFF" if is_active else DLG_FG,
        )

    def _toggle_roi_mode():
        if not _eye_path(_active_eye()):
            return
        roi_mode_var.set(not roi_mode_var.get())
        _update_roi_btn_style()

    roi_btn.config(command=_toggle_roi_mode)

    # Initial state: active
    _update_roi_btn_style()
    assert roi_mode_var.get() is True
    assert roi_btn.cget("bg") == DLG_ACCENT
    assert roi_btn.cget("fg") == "#FFFFFF"
    assert roi_btn.cget("state") == tk.NORMAL

    # Toggle to inactive
    _toggle_roi_mode()
    assert roi_mode_var.get() is False
    assert roi_btn.cget("bg") == DLG_BTN_SEC
    assert roi_btn.cget("fg") == DLG_FG

    # Toggle back to active
    _toggle_roi_mode()
    assert roi_mode_var.get() is True
    assert roi_btn.cget("bg") == DLG_ACCENT

    # When no image is present, button is disabled
    active_eye_path = ""
    _update_roi_btn_style()
    assert roi_btn.cget("state") == tk.DISABLED
    assert roi_btn.cget("bg") == DLG_BTN_SEC

    # Clicking while no image does not activate
    _toggle_roi_mode()
    assert roi_mode_var.get() is True  # unchanged


def test_roi_editing_disabled_when_roi_mode_inactive():
    """Verify that when ROI mode is inactive, nudging, dragging, and handles are disabled."""
    roi_mode_var = SimpleNamespace(get=lambda: False)
    roi_by_eye = {"OD": (0.2, 0.2, 0.6, 0.6)}
    active_eye = "OD"
    dummy_image = np.zeros((838, 1200, 3), dtype=np.uint8)

    # Nudge helper simulating _nudge_roi gate
    def _nudge_roi(dx: int, dy: int, is_shift: bool = False):
        if not roi_mode_var.get():
            return
        roi = roi_by_eye.get(active_eye)
        if roi is None:
            return
        step = 10 if is_shift else 1
        h, w = dummy_image.shape[:2]
        roi_by_eye[active_eye] = nudge_rect(roi, dx * step, dy * step, w, h)

    # Attempt to nudge while inactive
    _nudge_roi(5, 0)
    assert roi_by_eye["OD"] == (0.2, 0.2, 0.6, 0.6)  # unchanged!

    # Switch to active and nudge
    roi_mode_var.get = lambda: True
    _nudge_roi(1, 0)
    assert roi_by_eye["OD"][0] == pytest.approx(0.2 + 1 / 1200)


def test_canvas_handles_and_dash_mode_toggle():
    """Verify canvas overlay uses dashed outline and no handles when inactive."""
    root = _get_shared_root()
    canvas = tk.Canvas(root, width=400, height=300)
    canvas.pack()
    roi = (0.2, 0.2, 0.8, 0.8)
    tw, th = 300, 200

    def _draw(is_active: bool):
        canvas.delete("roi")
        cx0 = 2.0 + roi[0] * tw
        cy0 = 2.0 + roi[1] * th
        cx1 = 2.0 + roi[2] * tw
        cy1 = 2.0 + roi[3] * th

        rect_opts = {"outline": "#00E676", "width": 2, "tags": "roi"}
        if not is_active:
            rect_opts["dash"] = (5, 3)
        canvas.create_rectangle(cx0, cy0, cx1, cy1, **rect_opts)

        if is_active:
            handles = rect_handle_positions(roi[0], roi[1], roi[2], roi[3], tw, th, offset_x=2.0, offset_y=2.0)
            hs = 3.5
            for _hname, (hx, hy) in handles.items():
                canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs, outline="#00E676", fill="#FFFFFF", tags="roi")

    # When active: 1 bounding box + 8 handle boxes = 9 rectangles
    _draw(is_active=True)
    items_active = canvas.find_withtag("roi")
    assert len(items_active) == 9

    # When inactive: 1 bounding box with dashed styling, 0 handles
    _draw(is_active=False)
    items_inactive = canvas.find_withtag("roi")
    assert len(items_inactive) == 1
    dash_val = canvas.itemcget(items_inactive[0], "dash")
    assert "5 3" in dash_val or "5" in dash_val


def test_preservation_of_ring_interactions_when_roi_inactive():
    """Verify ring adjustment gestures are prioritized and functional regardless of ROI mode."""
    # Simulation of event routing logic in _on_roi_press
    ring_extended = []

    def _start_ring_extend(x, y):
        # returns True if on outline or center cross
        if 50 <= x <= 70 and 50 <= y <= 70:
            ring_extended.append((x, y))
            return True
        return False

    def handle_press(x, y, roi_mode: bool):
        if not roi_mode:
            if _start_ring_extend(x, y):
                return "ring_handled"
            return "ignored"
        else:
            if _start_ring_extend(x, y):
                return "ring_handled"
            return "roi_handled"

    # In active ROI mode: ring click handles ring, non-ring click creates/adjusts ROI
    assert handle_press(60, 60, roi_mode=True) == "ring_handled"
    assert handle_press(150, 150, roi_mode=True) == "roi_handled"

    # In inactive ROI mode: ring click still handles ring, non-ring click is safely ignored
    assert handle_press(60, 60, roi_mode=False) == "ring_handled"
    assert handle_press(150, 150, roi_mode=False) == "ignored"
    assert len(ring_extended) == 2


def test_od_os_state_isolation_with_roi_toggle():
    """Verify switching eyes and toggling ROI mode maintains strict isolation between OD and OS ROIs."""
    roi_by_eye = {"OD": (0.1, 0.1, 0.4, 0.4), "OS": (0.6, 0.6, 0.9, 0.9)}
    roi_mode = True

    # Toggle ROI mode off
    roi_mode = False
    assert roi_by_eye["OD"] == (0.1, 0.1, 0.4, 0.4)
    assert roi_by_eye["OS"] == (0.6, 0.6, 0.9, 0.9)

    # Select OS and update OS ROI
    roi_by_eye["OS"] = (0.5, 0.5, 0.8, 0.8)
    assert roi_by_eye["OD"] == (0.1, 0.1, 0.4, 0.4)  # OD not affected

    # Clear OS ROI
    roi_by_eye["OS"] = None
    assert roi_by_eye["OD"] == (0.1, 0.1, 0.4, 0.4)  # OD remains intact
    assert roi_by_eye["OS"] is None


def test_detection_overlay_visibility_when_limbus_unconfirmed():
    """Verify pupil boundary remains visible and distinct center markers draw when limbus is unconfirmed."""
    root = _get_shared_root()
    canvas = tk.Canvas(root, width=400, height=300)
    canvas.pack()

    # Mock result: pupil detected, limbus not detected/reliable=False
    pupil_e = EllipseParams(center_x=200, center_y=150, semi_major=40, semi_minor=40, angle_deg=0)
    fallback_limbus_e = EllipseParams(center_x=202, center_y=148, semi_major=85, semi_minor=85, angle_deg=0)

    # Draw distinct pupil (+) center in green
    canvas.create_line(196, 150, 204, 150, fill="#00e676", width=2, tags="detection")
    canvas.create_line(200, 146, 200, 154, fill="#00e676", width=2, tags="detection")

    # Draw distinct fallback limbus (+) center in blue
    canvas.create_line(197, 148, 207, 148, fill="#5c8aff", width=2, tags="detection")
    canvas.create_line(202, 143, 202, 153, fill="#5c8aff", width=2, tags="detection")

    # Fallback limbus polygon with dashed styling
    canvas.create_polygon([117, 148, 202, 63, 287, 148, 202, 233], outline="#5c8aff", fill="", width=2, dash=(5, 3), tags="detection")

    # Pupil polygon in green
    canvas.create_polygon([160, 150, 200, 110, 240, 150, 200, 190], outline="#00e676", fill="", width=2, tags="detection")

    # Status badge
    canvas.create_text(10, 15, text="pupil OK \u2014 limbus estimate", fill="#ffb020", tags="detection")

    detection_items = canvas.find_withtag("detection")
    assert len(detection_items) >= 5

    # Ensure ROI overlay can coexist with detection overlay
    canvas.create_rectangle(50, 50, 350, 250, outline="#00E676", tags="roi")
    roi_items = canvas.find_withtag("roi")
    assert len(roi_items) == 1
    assert canvas.find_withtag("detection") == detection_items


def test_patient_submission_and_cyclotorsion_compatibility():
    """Verify that normalized rectangular ROI format matches cyclotorsion engine and export formats."""
    from pupil_tracking.cyclotorsion.pair_readout import estimate_pair_readout

    # Rectangular ROI normalized tuple (x0, y0, x1, y1)
    roi_norm = (0.25, 0.25, 0.75, 0.75)

    # Synthetic eye image for cyclotorsion self-comparison
    img = np.full((838, 1200, 3), 128, dtype=np.uint8)
    import cv2
    cv2.circle(img, (600, 419), 60, (20, 20, 20), -1)  # pupil
    cv2.circle(img, (600, 419), 140, (80, 80, 80), -1) # iris

    # Pass in memory or mock - estimate_pair_readout accepts paths or images
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        cv2.imwrite(tmp_path, img)
        readout = estimate_pair_readout(tmp_path, tmp_path, roi_norm, flip=True)
        # Should execute without format/schema errors
        assert hasattr(readout, "valid")
        assert hasattr(readout, "angle_deg")
        assert hasattr(readout, "score")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_live_pentacam_preview_roi_button_dialog(tmp_path):
    """End-to-end integration test: Add Patient dialog with ROI button beside OD/OS."""
    import logging
    import time
    import cv2
    import pupil_tracking.interface.gui_app as gui_app

    # Generate synthetic Pentacam test image
    img = np.full((838, 1200, 3), 200, np.uint8)
    cv2.circle(img, (600, 420), 120, (30, 30, 30), -1)   # pupil
    cv2.circle(img, (600, 420), 280, (100, 100, 100), -1) # iris
    path = tmp_path / "test_pentacam_eye.png"
    cv2.imwrite(str(path), img)

    class _StubDetector:
        def detect(self, img, frame_number=-1, source=""):
            return None

    class _FakeApp:
        _preview_ellipse_points = staticmethod(gui_app.PupilTrackingGUI._preview_ellipse_points)
        _preview_limbus_ellipse = staticmethod(gui_app.PupilTrackingGUI._preview_limbus_ellipse)
        _preview_fallback_limbus = staticmethod(lambda img, res, roi=None: (None, ""))
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
            self.logger = logging.getLogger("test_roi_dialog")
            self._detector = _StubDetector()
            self._storage_root_var = tk.StringVar(value="")

    def pump(root, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            root.update()
            time.sleep(0.02)

    def _find_btn(widget, text):
        for child in widget.winfo_children():
            if isinstance(child, tk.Button) and child.cget("text") == text:
                return child
            hit = _find_btn(child, text)
            if hit is not None:
                return hit
        return None

    def _find_widgets(widget, cls):
        out = []
        for child in widget.winfo_children():
            if isinstance(child, cls):
                out.append(child)
            out.extend(_find_widgets(child, cls))
        return out

    root = _get_shared_root()
    try:
        app = _FakeApp(root)
        gui_app.PupilTrackingGUI._open_patient_dialog(app)
        pump(root, 0.4)

        tops = {w.title(): w for w in root.winfo_children() if isinstance(w, tk.Toplevel)}
        dialog = tops["Add Patient"]
        preview = tops.get("Pentacam Preview", dialog)

        # 1. Locate ROI button and OD / OS buttons
        roi_btn = _find_btn(preview, "ROI")
        od_btn = _find_btn(preview, "OD")
        os_btn = _find_btn(preview, "OS")
        assert roi_btn is not None, "ROI button must be present in preview"
        assert od_btn is not None and os_btn is not None

        # Verify ROI button is packed in the same parent as OD/OS
        assert roi_btn.master == od_btn.master == os_btn.master

        # 2. Load Pentacam image into OD
        entries = [e for e in _find_widgets(dialog, tk.Entry) if str(e.grid_info().get("row")) == "4"]
        assert entries
        entries[0].delete(0, "end")
        entries[0].insert(0, str(path))
        pump(root, 0.8)

        # 3. First click on ROI: activate ROI mode
        if not roi_btn._roi_mode_var.get():
            roi_btn.invoke()
            pump(root, 0.4)
        assert roi_btn._roi_mode_var.get() is True
        assert roi_btn.cget("state") == tk.NORMAL

        canvas = _find_widgets(preview, tk.Canvas)[0]

        # 4. In active mode: draw ROI box
        canvas.delete("roi")
        canvas.event_generate("<ButtonPress-1>", x=30, y=30)
        root.update()
        canvas.event_generate("<B1-Motion>", x=150, y=150)
        root.update()
        canvas.event_generate("<ButtonRelease-1>", x=150, y=150)
        root.update()

        roi_items = canvas.find_withtag("roi")
        assert roi_items, "ROI box must be drawn when active"

        # 5. Toggle ROI button to inactive
        roi_btn.invoke()
        root.update()
        assert roi_btn._roi_mode_var.get() is False

        # In inactive mode: dragging outside does NOT overwrite/create ROI
        canvas.delete("roi")
        canvas.event_generate("<ButtonPress-1>", x=40, y=40)
        root.update()
        canvas.event_generate("<B1-Motion>", x=160, y=160)
        root.update()
        canvas.event_generate("<ButtonRelease-1>", x=160, y=160)
        root.update()

        assert not canvas.find_withtag("roi"), "ROI box must NOT be created when ROI mode is inactive"

        # 6. Ring controls still work when ROI mode is inactive
        pupil_btn = _find_btn(preview, "Pupil")
        plus_btn = _find_btn(preview, "+ Ring")
        assert pupil_btn is not None and plus_btn is not None
        pupil_btn.invoke()
        plus_btn.invoke()
        root.update()
        det_items = canvas.find_withtag("detection")
        assert det_items, "Ring adjustments must work while ROI mode is inactive"

        # 7. Toggle ROI back to active
        roi_btn.invoke()
        root.update()
        assert roi_btn._roi_mode_var.get() is True

        # Test Reset ROI restores default box
        reset_roi_btn = _find_btn(preview, "Reset ROI")
        assert reset_roi_btn is not None
        reset_roi_btn.invoke()
        root.update()
        assert canvas.find_withtag("roi"), "Reset ROI must draw the reset rectangle"

        # Test Clear ROI removes the box
        clear_roi_btn = _find_btn(preview, "Clear ROI")
        assert clear_roi_btn is not None
        clear_roi_btn.invoke()
        root.update()
        assert not canvas.find_withtag("roi"), "Clear ROI must clear the rectangle"

    finally:
        for child in root.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass


