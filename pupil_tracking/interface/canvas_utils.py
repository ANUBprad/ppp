"""Pure geometry and state helpers for canvas/event handling.

Extracted from gui_app.py.  These functions have NO GUI dependencies
and NO clinical logic — they operate on plain data (coordinates, ROIs,
image shapes).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def canvas_to_image_point(
    canvas_x: float,
    canvas_y: float,
    display_origin: Tuple[int, int],
    display_size: Tuple[int, int],
    display_scale: float,
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[float, float]]:
    """Convert canvas pixel coordinates to original image coordinates.

    Returns ``(x, y)`` in image space, or ``None`` if the point is
    outside the displayed image region.
    """
    ox, oy = display_origin
    dw, dh = display_size
    if dw <= 0 or dh <= 0:
        return None
    if not (ox <= canvas_x <= ox + dw and oy <= canvas_y <= oy + dh):
        return None
    x = (canvas_x - ox) / max(display_scale, 1e-6)
    y = (canvas_y - oy) / max(display_scale, 1e-6)
    h, w = image_shape[:2]
    return (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))


def active_manual_roi(
    manual_roi: Optional[Dict[str, float]],
    image_shape: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, float]]:
    """Return *manual_roi* if it matches the current image dimensions."""
    if manual_roi is None or image_shape is None:
        return None
    h, w = image_shape[:2]
    if (
        int(round(manual_roi.get("frame_width", w))) != w
        or int(round(manual_roi.get("frame_height", h))) != h
    ):
        return None
    return manual_roi


def active_manual_ring(
    manual_ring: Optional[Dict[str, float]],
    image_shape: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, float]]:
    """Return *manual_ring* if it matches the current image dimensions."""
    if manual_ring is None or image_shape is None:
        return None
    h, w = image_shape[:2]
    if (
        int(round(manual_ring.get("frame_width", w))) != w
        or int(round(manual_ring.get("frame_height", h))) != h
    ):
        return None
    return manual_ring


def manual_roi_crop(
    frame: np.ndarray,
    roi: Dict[str, float],
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Crop *frame* to the bounding box of a circular ROI.

    Returns ``(cropped, x_offset, y_offset)`` or ``None`` if the crop
    is empty.
    """
    h, w = frame.shape[:2]
    cx = float(np.clip(roi["center_x"], 0, w - 1))
    cy = float(np.clip(roi["center_y"], 0, h - 1))
    radius = max(1.0, float(roi["radius"]))
    x0 = max(0, int(np.floor(cx - radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    x1 = min(w, int(np.ceil(cx + radius)))
    y1 = min(h, int(np.ceil(cy + radius)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop, float(x0), float(y0)


def clamp_circle_center(
    cx: float,
    cy: float,
    radius: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float]:
    """Clamp a circle centre so the full circle stays inside the image."""
    return (
        float(np.clip(cx, radius, image_w - radius)),
        float(np.clip(cy, radius, image_h - radius)),
    )


def clamp_circle_radius(
    cx: float,
    cy: float,
    radius: float,
    image_w: int,
    image_h: int,
    min_radius: float = 8.0,
) -> float:
    """Clamp a circle radius so it stays inside the image bounds."""
    max_radius = min(cx, cy, image_w - cx, image_h - cy)
    return float(max(min_radius, min(radius, max_radius)))


def hit_test_circle(
    point_x: float,
    point_y: float,
    circle_cx: float,
    circle_cy: float,
    circle_radius: float,
    rim_fraction: float = 0.18,
) -> str:
    """Classify a click relative to a circle.

    Returns ``"move"``, ``"resize"``, or ``"outside"``.
    """
    import math
    distance = math.hypot(point_x - circle_cx, point_y - circle_cy)
    rim_threshold = max(10.0, circle_radius * rim_fraction)
    if abs(distance - circle_radius) <= rim_threshold:
        return "resize"
    elif distance < circle_radius:
        return "move"
    return "outside"


def roi_norm_to_image_coords(
    roi_norm: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[int, int, int, int, int, int]:
    """Convert normalized ROI (x0, y0, x1, y1) to integer image coordinates.

    Returns ``(ix0, iy0, ix1, iy1, width_px, height_px)``.
    """
    x0, y0, x1, y1 = roi_norm
    ix0 = int(round(x0 * width))
    iy0 = int(round(y0 * height))
    ix1 = int(round(x1 * width))
    iy1 = int(round(y1 * height))
    return ix0, iy0, ix1, iy1, max(0, ix1 - ix0), max(0, iy1 - iy0)


def rect_handle_positions(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    tw: int,
    th: int,
    offset_x: float = 2.0,
    offset_y: float = 2.0,
) -> Dict[str, Tuple[float, float]]:
    """Return canvas pixel coordinates for 8 rectangular ROI handles.

    Handles returned: nw, n, ne, e, se, s, sw, w.
    """
    cx0 = offset_x + x0 * tw
    cy0 = offset_y + y0 * th
    cx1 = offset_x + x1 * tw
    cy1 = offset_y + y1 * th
    mid_x = (cx0 + cx1) * 0.5
    mid_y = (cy0 + cy1) * 0.5

    return {
        "nw": (cx0, cy0),
        "n": (mid_x, cy0),
        "ne": (cx1, cy0),
        "e": (cx1, mid_y),
        "se": (cx1, cy1),
        "s": (mid_x, cy1),
        "sw": (cx0, cy1),
        "w": (cx0, mid_y),
    }


def hit_test_rect(
    canvas_x: float,
    canvas_y: float,
    roi_norm: Tuple[float, float, float, float],
    tw: int,
    th: int,
    handle_radius: float = 7.0,
    edge_tol: float = 6.0,
    offset_x: float = 2.0,
    offset_y: float = 2.0,
) -> Optional[str]:
    """Classify pointer position relative to a normalized rectangular ROI.

    Returns:
    - Handle name ('nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w') if near a handle.
    - Edge name ('edge_n', 'edge_s', 'edge_w', 'edge_e') if near an edge.
    - 'move' if inside the rectangle.
    - None if outside.
    """
    import math

    x0, y0, x1, y1 = roi_norm
    handles = rect_handle_positions(x0, y0, x1, y1, tw, th, offset_x, offset_y)

    # 1. Check handles first (highest priority)
    for name, (hx, hy) in handles.items():
        if math.hypot(canvas_x - hx, canvas_y - hy) <= handle_radius:
            return name

    cx0 = offset_x + x0 * tw
    cy0 = offset_y + y0 * th
    cx1 = offset_x + x1 * tw
    cy1 = offset_y + y1 * th

    # 2. Check edges
    on_horiz_span = (cx0 - edge_tol <= canvas_x <= cx1 + edge_tol)
    on_vert_span = (cy0 - edge_tol <= canvas_y <= cy1 + edge_tol)

    if on_horiz_span:
        if abs(canvas_y - cy0) <= edge_tol:
            return "edge_n"
        if abs(canvas_y - cy1) <= edge_tol:
            return "edge_s"

    if on_vert_span:
        if abs(canvas_x - cx0) <= edge_tol:
            return "edge_w"
        if abs(canvas_x - cx1) <= edge_tol:
            return "edge_e"

    # 3. Check interior
    if cx0 < canvas_x < cx1 and cy0 < canvas_y < cy1:
        return "move"

    return None


def update_rect_drag(
    drag_mode: str,
    orig_roi: Tuple[float, float, float, float],
    current_norm: Tuple[float, float],
    start_norm: Tuple[float, float],
    min_w: float = 0.02,
    min_h: float = 0.02,
) -> Tuple[float, float, float, float]:
    """Compute updated normalized coordinates during dragging.

    Preserves dimensions during 'move', enforces min_w/min_h and clamps to [0..1].
    """
    ox0, oy0, ox1, oy1 = orig_roi
    dx = current_norm[0] - start_norm[0]
    dy = current_norm[1] - start_norm[1]

    if drag_mode == "move":
        w = ox1 - ox0
        h = oy1 - oy0
        nx0 = max(0.0, min(ox0 + dx, 1.0 - w))
        ny0 = max(0.0, min(oy0 + dy, 1.0 - h))
        return (nx0, ny0, nx0 + w, ny0 + h)

    x0, y0, x1, y1 = ox0, oy0, ox1, oy1

    touches_west = drag_mode in ("w", "nw", "sw", "edge_w")
    touches_east = drag_mode in ("e", "ne", "se", "edge_e")
    touches_north = drag_mode in ("n", "nw", "ne", "edge_n")
    touches_south = drag_mode in ("s", "sw", "se", "edge_s")

    if touches_west:
        x0 = min(max(ox0 + dx, 0.0), ox1 - min_w)
    elif touches_east:
        x1 = max(min(ox1 + dx, 1.0), ox0 + min_w)

    if touches_north:
        y0 = min(max(oy0 + dy, 0.0), oy1 - min_h)
    elif touches_south:
        y1 = max(min(oy1 + dy, 1.0), oy0 + min_h)

    return (x0, y0, x1, y1)


def nudge_rect(
    roi_norm: Tuple[float, float, float, float],
    dx_px: int,
    dy_px: int,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Nudge rectangle by integer image pixels while preserving dimensions."""
    if img_w <= 0 or img_h <= 0:
        return roi_norm
    x0, y0, x1, y1 = roi_norm
    w = x1 - x0
    h = y1 - y0

    dx_norm = dx_px / float(img_w)
    dy_norm = dy_px / float(img_h)

    nx0 = max(0.0, min(x0 + dx_norm, 1.0 - w))
    ny0 = max(0.0, min(y0 + dy_norm, 1.0 - h))

    return (nx0, ny0, nx0 + w, ny0 + h)


def flip_rect_norm(
    roi_norm: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """180° dual-axis inversion matching Pentacam flip."""
    x0, y0, x1, y1 = roi_norm
    return (1.0 - x1, 1.0 - y1, 1.0 - x0, 1.0 - y0)
