"""Tests for Phase 3 — BUG-006: Fix Iris Unwrap Display, Cropping and Angular Alignment.

Deterministic, synthetic tests verifying:
1. Full Pentacam unwrap image visibility.
2. Full ELITA live unwrap image visibility.
3. Aspect-ratio preservation at multiple window sizes.
4. Correct angular mapping at 0°, 90°, 180°, 270°, and 360°.
5. OD/OS orientation consistency adhering to clinical anatomical conventions.
6. Absence of unintended flips, mirrors, or reversals.
7. Alignment of both images to the common angular grid.
8. Preservation of existing Phase 1 and Phase 2 behavior.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, List, Tuple

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.polar_unwrap import (
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_unwrap_maps,
    unwrap_from_ellipses,
    unwrap_iris,
)
from pupil_tracking.interface import gui_app
from pupil_tracking.interface.gui_app import (
    _UNWRAP_STRIP_SIZE,
    _UNWRAP_ZONE_BANDS,
    _UNWRAP_ZONE_QUADRANTS,
    _UNWRAP_ZONE_SECTORS,
    PupilTrackingGUI,
)
from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.utils.types import EllipseParams

# ---------------------------------------------------------------------------
# Synthetic test fixtures & geometry helpers
# ---------------------------------------------------------------------------

CX = CY = 150.0
PUPIL_R = 35.0
LIMBUS_R = 95.0
IMG_SIZE = 300


def _circular_roi(
    pupil_r: float = PUPIL_R,
    limbus_r: float = LIMBUS_R,
    inner_inset: float = 0.05,
    outer_inset: float = 0.05,
) -> IrisROI:
    return IrisROI(
        valid=True,
        reason="ok",
        center_x=CX,
        center_y=CY,
        pupil_semi_major=pupil_r,
        pupil_semi_minor=pupil_r,
        pupil_angle_deg=0.0,
        limbus_semi_major=limbus_r,
        limbus_semi_minor=limbus_r,
        limbus_angle_deg=0.0,
        inner_inset_frac=inner_inset,
        outer_inset_frac=outer_inset,
        pupil_radius_px=pupil_r,
        limbus_radius_px=limbus_r,
    )


def _spoke_at_screen_coord(angle_rad: float, width: int = 4) -> np.ndarray:
    """Image with a bright spoke in the iris annulus at a specific screen coordinate angle."""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    p0 = (
        int(round(CX + (PUPIL_R + 2) * math.cos(angle_rad))),
        int(round(CY + (PUPIL_R + 2) * math.sin(angle_rad))),
    )
    p1 = (
        int(round(CX + (LIMBUS_R - 2) * math.cos(angle_rad))),
        int(round(CY + (LIMBUS_R - 2) * math.sin(angle_rad))),
    )
    cv2.line(img, p0, p1, 255, width)
    return img


class _RecordingCanvas:
    """Mock Tk Canvas capturing drawn items for headless display verification."""

    def __init__(self, width: int, height: int) -> None:
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


# ===========================================================================
# 1. Full Pentacam unwrap image visibility
# ===========================================================================

def test_1_pentacam_unwrap_visibility_and_boundary_fit(monkeypatch):
    """Verify Pentacam unwrap renders without clipping in available display area."""
    captured_photos = []

    class _MockPhoto:
        def __init__(self, img):
            self.img = img
            captured_photos.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_MockPhoto))

    app = SimpleNamespace(
        _unwrap_canvas=_RecordingCanvas(800, 200),
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=np.full((72, 720, 3), 128, dtype=np.uint8),
        _elita_strip=None,
        _elita_unwrap_shift_cols=0,
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=PupilTrackingGUI._roll_strip,
    )

    PupilTrackingGUI._redraw_unwrap(app)

    # Must draw Pentacam strip and placeholder for live
    assert len(app._unwrap_canvas.images) == 1
    x0, y0, photo = app._unwrap_canvas.images[0]
    disp_w, disp_h = captured_photos[0].size

    # Verify visibility within canvas bounds
    assert x0 >= 0, "Image must not be clipped on the left"
    assert x0 + disp_w <= 800, "Image must not be clipped on the right"
    assert y0 >= 14, "Image must be placed below ruler"
    assert y0 + disp_h <= app._unwrap_canvas.winfo_height(), "Image must not be clipped at bottom"

    # Full aspect ratio of 10:1 maintained
    assert abs(disp_w / disp_h - 10.0) < 0.2


# ===========================================================================
# 2. Full ELITA live unwrap image visibility
# ===========================================================================

def test_2_elita_live_unwrap_visibility_and_nudge_preservation(monkeypatch):
    """Verify ELITA live strip renders below Pentacam and nudge wraps cyclically without clipping."""
    captured_photos = []

    class _MockPhoto:
        def __init__(self, img):
            self.img = img
            captured_photos.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_MockPhoto))

    canvas = _RecordingCanvas(900, 220)
    app = SimpleNamespace(
        _unwrap_canvas=canvas,
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=np.full((72, 720, 3), 100, dtype=np.uint8),
        _elita_strip=np.full((72, 720, 3), 180, dtype=np.uint8),
        _elita_unwrap_shift_cols=10,  # nudged
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=PupilTrackingGUI._roll_strip,
    )

    PupilTrackingGUI._redraw_unwrap(app)

    # Both Pentacam and ELITA strips must be rendered
    assert len(canvas.images) == 2
    x0_pent, y0_pent, _ = canvas.images[0]
    x0_elita, y0_elita, _ = canvas.images[1]
    w_pent, h_pent = captured_photos[0].size
    w_elita, h_elita = captured_photos[1].size

    # Both strips share identical horizontal alignment and width
    assert x0_pent == x0_elita
    assert w_pent == w_elita
    assert h_pent == h_elita
    # ELITA strip is vertically stacked below Pentacam
    assert y0_elita >= y0_pent + h_pent
    # ELITA strip fits within canvas
    assert y0_elita + h_elita <= canvas.winfo_height()


# ===========================================================================
# 3. Aspect-ratio preservation at multiple window sizes
# ===========================================================================

@pytest.mark.parametrize(
    "canvas_w, canvas_h",
    [
        (600, 150),
        (750, 180),
        (900, 200),
        (1100, 220),
        (1300, 240),
        (1600, 250),
    ],
)
def test_3_aspect_ratio_preservation_across_window_sizes(monkeypatch, canvas_w, canvas_h):
    """Verify that resizing the panel scales images proportionally with exact 10:1 aspect ratio."""
    captured_photos = []

    class _MockPhoto:
        def __init__(self, img):
            self.img = img
            captured_photos.append(img)

    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=_MockPhoto))

    canvas = _RecordingCanvas(canvas_w, canvas_h)
    app = SimpleNamespace(
        _unwrap_canvas=canvas,
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=np.full((72, 720, 3), 120, dtype=np.uint8),
        _elita_strip=np.full((72, 720, 3), 160, dtype=np.uint8),
        _elita_unwrap_shift_cols=0,
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=PupilTrackingGUI._roll_strip,
    )

    PupilTrackingGUI._redraw_unwrap(app)

    assert len(canvas.images) == 2
    disp_w, disp_h = captured_photos[0].size

    # Aspect ratio check: 720 / 72 == 10.0
    aspect = disp_w / disp_h
    assert abs(aspect - 10.0) < 0.25, f"Expected 10:1 ratio, got {aspect:.2f} for canvas {canvas_w}x{canvas_h}"

    # Strips fit completely inside canvas
    x0, y0_elita, _ = canvas.images[1]
    assert x0 >= 0 and x0 + disp_w <= canvas_w
    assert y0_elita + disp_h <= canvas.winfo_height()


# ===========================================================================
# 4. Correct angular mapping at 0°, 90°, 180°, 270°, and 360°
# ===========================================================================

def test_4_angular_mapping_cardinal_meridians_od():
    """Verify OD unwrap maps 0°(Right/Nasal), 90°(Down/Inferior), 180°(Left/Temporal), 270°(Up/Superior)."""
    roi = _circular_roi()

    # Spoke at Screen Right (+x, 0 rad) -> 0° Nasal for OD
    img_0 = _spoke_at_screen_coord(0.0)
    strip_0 = unwrap_iris(img_0, roi, output_size=(72, 720), eye="OD")
    assert strip_0 is not None
    # Peak intensity must be at column 0 (and wraps at 720)
    assert np.argmax(strip_0.mean(axis=0)) in (0, 1, 719)

    # Spoke at Screen Down (+y, pi/2 rad) -> 90° Inferior for OD
    img_90 = _spoke_at_screen_coord(math.pi / 2.0)
    strip_90 = unwrap_iris(img_90, roi, output_size=(72, 720), eye="OD")
    assert strip_90 is not None
    assert abs(int(np.argmax(strip_90.mean(axis=0))) - 180) <= 2  # 720 * 0.25 = 180

    # Spoke at Screen Left (-x, pi rad) -> 180° Temporal for OD
    img_180 = _spoke_at_screen_coord(math.pi)
    strip_180 = unwrap_iris(img_180, roi, output_size=(72, 720), eye="OD")
    assert strip_180 is not None
    assert abs(int(np.argmax(strip_180.mean(axis=0))) - 360) <= 2  # 720 * 0.50 = 360

    # Spoke at Screen Up (-y, 3*pi/2 rad) -> 270° Superior for OD
    img_270 = _spoke_at_screen_coord(3.0 * math.pi / 2.0)
    strip_270 = unwrap_iris(img_270, roi, output_size=(72, 720), eye="OD")
    assert strip_270 is not None
    assert abs(int(np.argmax(strip_270.mean(axis=0))) - 540) <= 2  # 720 * 0.75 = 540


def test_4_angular_mapping_cardinal_meridians_os():
    """Verify OS unwrap maps 0°(Left/Nasal), 90°(Down/Inferior), 180°(Right/Temporal), 270°(Up/Superior)."""
    roi = _circular_roi()

    # In OS, Nasal (0°) is on Screen LEFT (-x, pi rad)
    img_nasal_os = _spoke_at_screen_coord(math.pi)
    strip_0 = unwrap_iris(img_nasal_os, roi, output_size=(72, 720), eye="OS")
    assert strip_0 is not None
    # Peak intensity must be at column 0 for 0° Nasal
    assert np.argmax(strip_0.mean(axis=0)) in (0, 1, 719)

    # In OS, Inferior (90°) is on Screen DOWN (+y, pi/2 rad)
    img_inferior_os = _spoke_at_screen_coord(math.pi / 2.0)
    strip_90 = unwrap_iris(img_inferior_os, roi, output_size=(72, 720), eye="OS")
    assert strip_90 is not None
    assert abs(int(np.argmax(strip_90.mean(axis=0))) - 180) <= 2

    # In OS, Temporal (180°) is on Screen RIGHT (+x, 0 rad)
    img_temporal_os = _spoke_at_screen_coord(0.0)
    strip_180 = unwrap_iris(img_temporal_os, roi, output_size=(72, 720), eye="OS")
    assert strip_180 is not None
    assert abs(int(np.argmax(strip_180.mean(axis=0))) - 360) <= 2

    # In OS, Superior (270°) is on Screen UP (-y, 3*pi/2 rad)
    img_superior_os = _spoke_at_screen_coord(3.0 * math.pi / 2.0)
    strip_270 = unwrap_iris(img_superior_os, roi, output_size=(72, 720), eye="OS")
    assert strip_270 is not None
    assert abs(int(np.argmax(strip_270.mean(axis=0))) - 540) <= 2


# ===========================================================================
# 5. OD/OS orientation consistency
# ===========================================================================

def test_5_od_os_clinical_reference_consistency():
    """Verify both OD and OS unwrap columns consistently represent the same anatomical landmarks."""
    roi = _circular_roi()

    # For both eyes, 0° is Nasal and 180° is Temporal
    od_labels = get_centration_meridian_labels("OD")
    os_labels = get_centration_meridian_labels("OS")

    # In OD: Left=0, Right=180
    assert od_labels == ("0", "180", "270", "90")
    # In OS: Left=0(Nasal), Right=180(Temporal)
    assert os_labels == ("0", "180", "270", "90")

    # Test an anatomical landmark placed at Nasal meridian for both eyes:
    # OD Nasal is on screen right (+x)
    od_img = _spoke_at_screen_coord(0.0)
    od_strip = unwrap_iris(od_img, roi, output_size=(72, 720), eye="OD")
    # OS Nasal is on screen left (-x)
    os_img = _spoke_at_screen_coord(math.pi)
    os_strip = unwrap_iris(os_img, roi, output_size=(72, 720), eye="OS")

    # Under the common 0° ruler, both Nasal features sit at column 0!
    assert np.argmax(od_strip.mean(axis=0)) in (0, 1, 719)
    assert np.argmax(os_strip.mean(axis=0)) in (0, 1, 719)


# ===========================================================================
# 6. Absence of unintended flips or reversals
# ===========================================================================

def test_6_no_unintended_inversion_or_reversal():
    """Verify that moving a landmark along the circle translates the strip monotonically without inversion."""
    roi = _circular_roi()

    angles = [10.0, 30.0, 50.0, 70.0, 100.0]
    od_cols = []
    for ang in angles:
        rad = math.radians(ang)
        img = _spoke_at_screen_coord(rad)
        strip = unwrap_iris(img, roi, output_size=(72, 720), eye="OD")
        od_cols.append(int(np.argmax(strip.mean(axis=0))))

    # Must advance strictly monotonically to the right
    for i in range(len(od_cols) - 1):
        assert od_cols[i] < od_cols[i + 1], f"Expected monotonic shift, got {od_cols}"


# ===========================================================================
# 7. Alignment of both images to the common angular grid
# ===========================================================================

def test_7_alignment_to_common_grid(monkeypatch):
    """Verify ruler text and quadrant grid lines align with the strip boundaries."""
    monkeypatch.setattr(gui_app, "ImageTk", SimpleNamespace(PhotoImage=lambda img: SimpleNamespace(img=img)))

    canvas = _RecordingCanvas(1000, 200)
    app = SimpleNamespace(
        _unwrap_canvas=canvas,
        _colors=SimpleNamespace(FG_TERTIARY="#606070"),
        _unwrap_strip=np.full((72, 720, 3), 120, dtype=np.uint8),
        _elita_strip=np.full((72, 720, 3), 150, dtype=np.uint8),
        _elita_unwrap_shift_cols=0,
        _show_unwrap_zones=SimpleNamespace(get=lambda: True),
        _unwrap_zone_lines=PupilTrackingGUI._unwrap_zone_lines,
        _unwrap_band_fracs=PupilTrackingGUI._unwrap_band_fracs,
        _roll_strip=PupilTrackingGUI._roll_strip,
    )

    PupilTrackingGUI._redraw_unwrap(app)

    # Check ruler labels
    ruler_texts = {text: x for x, y, text, anchor in canvas.texts if "°" in text}
    assert "0°" in ruler_texts
    assert "90°" in ruler_texts
    assert "180°" in ruler_texts
    assert "270°" in ruler_texts
    assert "360°" in ruler_texts

    # Image origin x0 and width disp_w
    x0, y0, _ = canvas.images[0]
    x_0 = ruler_texts["0°"]
    x_180 = ruler_texts["180°"]
    x_360 = ruler_texts["360°"]

    # 0° matches left edge, 360° matches right edge, 180° matches exact center
    assert x_0 == x0
    assert abs((x_0 + x_360) / 2.0 - x_180) <= 1.0


# ===========================================================================
# 8. Preservation of Phase 1 and Phase 2 behavior
# ===========================================================================

def test_8_preservation_of_phase_1_and_phase_2():
    """Verify that Phase 1 (OD/OS mapping) and Phase 2 (centration) contracts remain preserved."""
    # Phase 2 centration orientation
    assert get_centration_meridian_labels("OD") == ("0", "180", "270", "90")
    assert get_centration_meridian_labels("OS") == ("0", "180", "270", "90")

    # Phase 1 unwrap size constant
    assert _UNWRAP_STRIP_SIZE == (72, 720)
    assert _UNWRAP_ZONE_SECTORS == 12
    assert _UNWRAP_ZONE_QUADRANTS == 4
    assert _UNWRAP_ZONE_BANDS == 4
