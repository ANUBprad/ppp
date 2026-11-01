"""Tests for the circular limbus crop used by the GUI Snapshot button.

The GUI method :meth:`PupilTrackingGUI._circular_crop_to_limbus` is a pure
function of its arguments (it never touches ``self``), so it is exercised
here with an uninitialised GUI instance built via ``object.__new__`` — the
same pattern used by ``test_gui_detection_confidence.py`` — avoiding a Tk
root.
"""
from types import SimpleNamespace

import numpy as np

from pupil_tracking.interface.gui_app import PupilTrackingGUI
from pupil_tracking.utils.types import EllipseParams


def _make_app() -> PupilTrackingGUI:
    return object.__new__(PupilTrackingGUI)


def _make_result(ellipse, detected: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        limbus=SimpleNamespace(detected=detected, ellipse=ellipse)
    )


def _circle_ellipse(cx: float, cy: float, r: float) -> EllipseParams:
    return EllipseParams(
        center_x=cx, center_y=cy, semi_major=r, semi_minor=r, angle_deg=0.0
    )


def test_no_result_returns_image_unchanged():
    app = _make_app()
    image = np.full((60, 80, 3), 200, dtype=np.uint8)

    out, cropped = app._circular_crop_to_limbus(image, None)

    assert cropped is False
    assert out is image


def test_no_limbus_returns_image_unchanged():
    app = _make_app()
    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    result = SimpleNamespace(
        limbus=SimpleNamespace(detected=False, ellipse=None)
    )

    out, cropped = app._circular_crop_to_limbus(image, result)

    assert cropped is False
    assert out is image


def test_detected_limbus_crops_and_masks_transparent():
    app = _make_app()
    image = np.full((100, 120, 3), 30, dtype=np.uint8)
    result = _make_result(_circle_ellipse(60.0, 50.0, 20.0))

    out, cropped = app._circular_crop_to_limbus(image, result, transparent=True)

    assert cropped is True
    assert out.shape[2] == 4  # BGRA
    # A square crop roughly one diameter (plus a 1px guard) on a side.
    assert abs(out.shape[0] - out.shape[1]) <= 2

    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    assert out[cy, cx, 3] == 255  # centre inside the ring -> opaque
    assert out[0, 0, 3] == 0  # corner outside the ring -> transparent


def test_detected_limbus_masks_black_when_not_transparent():
    app = _make_app()
    image = np.full((100, 120, 3), 30, dtype=np.uint8)
    result = _make_result(_circle_ellipse(60.0, 50.0, 20.0))

    out, cropped = app._circular_crop_to_limbus(image, result, transparent=False)

    assert cropped is True
    assert out.shape[2] == 3

    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    assert tuple(out[cy, cx]) == (30, 30, 30)  # centre kept
    assert tuple(out[0, 0]) == (0, 0, 0)  # corner blacked out


def test_limbus_touching_border_is_clamped():
    app = _make_app()
    image = np.full((80, 90, 3), 50, dtype=np.uint8)
    result = _make_result(_circle_ellipse(5.0, 5.0, 40.0))

    out, cropped = app._circular_crop_to_limbus(image, result, transparent=True)

    assert cropped is True
    # Crop is clamped to the source image bounds.
    assert out.shape[0] <= 80
    assert out.shape[1] <= 90
    # The centre (image origin, distance 0 from the limbus centre) stays.
    assert out[0, 0, 3] == 255


def test_gui_adapted_simple_namespace_ellipse_is_cropped():
    """Regression: GUI-adapted results expose ellipses as ``SimpleNamespace``
    without an ``is_valid`` attribute.  The crop must still happen (previously
    it silently fell back to the full frame)."""
    app = _make_app()
    image = np.full((100, 120, 3), 30, dtype=np.uint8)
    adapted = SimpleNamespace(
        center_x=60.0,
        center_y=50.0,
        radius=20.0,
        semi_major=20.0,
        semi_minor=20.0,
        angle_deg=0.0,
    )
    assert not hasattr(adapted, "is_valid")  # the shape that broke it
    result = _make_result(adapted)

    out, cropped = app._circular_crop_to_limbus(image, result, transparent=True)

    assert cropped is True
    assert out.shape[2] == 4
    # Inside the ring stays, outside becomes transparent.
    assert out[out.shape[0] // 2, out.shape[1] // 2, 3] == 255
    assert out[0, 0, 3] == 0


def test_degenerate_ellipse_falls_back_to_full_frame():
    app = _make_app()
    image = np.full((60, 80, 3), 200, dtype=np.uint8)
    result = _make_result(_circle_ellipse(40.0, 30.0, 0.0))

    out, cropped = app._circular_crop_to_limbus(image, result)

    assert cropped is False
    assert out is image


def test_prepare_recording_frame_clean_skips_overlay_drawing():
    """The Snapshot uses ``draw_overlay=False`` so the cropped circle has no
    annotations.  Any attempt to draw overlays here would raise (the drawing
    helpers are not stubbed), proving they are skipped."""
    app = _make_app()
    app._grayscale_mode_var = SimpleNamespace(get=lambda: "off")
    frame = np.full((40, 50, 3), 123, dtype=np.uint8)

    out = app._prepare_recording_frame(
        frame, SimpleNamespace(), draw_overlay=False
    )

    assert out is not frame  # returns a copy
    assert np.array_equal(out, frame)  # untouched by overlays
