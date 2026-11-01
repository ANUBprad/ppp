
"""
Surgical-grade pupil & limbus tracking GUI.

Features:
    - Load and analyse single images
    - Process video files with frame-by-frame playback
    - Live camera feed with real-time detection (optimised pipeline)
    - Smart circle/ellipse overlay (auto-selected per structure)
    - Diameter display in pixels AND millimetres
    - Semi-major/semi-minor display in pixels AND millimetres
    - Offset display in pixels AND millimetres
    - Auto-calibration from limbus (corneal diameter = 12.0 mm)
    - Detection overlay with pupil, limbus, corneal centre, offset
    - Export results to CSV / TXT / snapshot
    - Quality grade badge
    - FP16 / torch.compile / async capture / frame-skip / ROI / Kalman
    - Settings tab for full pipeline configuration
    - Progress bar with ETA for video-file processing
    - Pause / resume for video and camera
    - OptimizedVideoProcessor wired to video-file path
    - Robust None-safety: works even when model loading fails
    - Grayscale mode toggle (OFF / AUTO / FORCE) with [G] hotkey (NEW)
"""

from __future__ import annotations

import csv
import json
import math
import queue
import re
import shutil
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.utils.types import (
    DetectionQuality,
    CalibrationInfo,
)
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger
from pupil_tracking.utils.runtime_profile import (
    apply_runtime_optimizations,
    detect_runtime_profile,
)
from pupil_tracking.interface.theme import DarkTheme
from pupil_tracking.interface.canvas_utils import (
    flip_rect_norm,
    hit_test_rect,
    nudge_rect,
    rect_handle_positions,
    roi_norm_to_image_coords,
    update_rect_drag,
)

try:
    from pupil_tracking.ml.fast_inference import FastInference
    from pupil_tracking.video.optimized_processor import (
        OptimizedVideoProcessor,
        AsyncCapture,
        FrameResult,
        TrackingQuality,
    )

    _FAST_PIPELINE_AVAILABLE = True
except ImportError:
    _FAST_PIPELINE_AVAILABLE = False
    FastInference = None
    OptimizedVideoProcessor = None
    AsyncCapture = None
    FrameResult = None
    TrackingQuality = None

# ══════════════════════════════════════════════════════════════════
# RECORDING — Import FrameRecorder & DualFrameRecorder
# ══════════════════════════════════════════════════════════════════
from pupil_tracking.interface.frame_recorder import DualFrameRecorder, FrameRecorder
_CORNEAL_DIAMETER_MM = 12.0
_CIRCLE_DRAW_THRESHOLD = 0.95

_QUALITY_COLORS = {
    "SURGICAL": "#00e676",
    "CLINICAL": "#29b6f6",
    "RESEARCH": "#ffa726",
    "INSUFFICIENT": "#ef5350",
    "NO_DETECTION": "#616161",
}

# ══════════════════════════════════════════════════════════════════
# GRAYSCALE GUI 2 of 12 — Grayscale mode display labels & colours
# ══════════════════════════════════════════════════════════════════
_GRAYSCALE_LABELS = {
    "off": "RGB",
    "auto": "AUTO",
    "force": "GRAY",
}
_GRAYSCALE_COLORS = {
    "off": "#aaaaaa",
    "auto": "#00bcd4",
    "force": "#ffeb3b",
}
_GRAYSCALE_CYCLE = ["off", "auto", "force"]
# ══════════════════════════════════════════════════════════════════

_WINDOW_TITLE = "Medevplus IXcentai — Surgical Grade"
_MIN_WIDTH = 1280
_MIN_HEIGHT = 800
_DISPLAY_FPS_CAP = 30.0


# Iris-unwrap panel strip size (rows = radial samples, cols = angular samples).
# Kept wide (10:1) so a short panel can still show both strips at full width
# with the true aspect ratio preserved — no squash and no wasted side margins.
_UNWRAP_STRIP_SIZE = (72, 720)


# Iris zone grid overlaid on the unwrap strips (display-only).
#   * 12 angular zones  = 4 quadrants x 3 sub-zones = 30 deg each = one clock
#     hour, so a zone reads straight off the clock face; this matches the
#     existing N_ANGULAR_SECTORS = 12 convention in pupil_tracking.iris.robustness.
#   * 4 emphasised quadrant boundaries at 0/90/180/270 deg.
#   * 4 radial bands (pupil boundary -> limbus) matching N_RADIAL_BANDS.
# Change _UNWRAP_ZONE_SECTORS alone to switch to 8 zones (4 x 2, 45 deg) etc.
_UNWRAP_ZONE_SECTORS = 12
_UNWRAP_ZONE_QUADRANTS = 4
_UNWRAP_ZONE_BANDS = 4


# Step (degrees) applied by each click of the ELITA-strip "-" / "+" controls.
# The live strip can be nudged left/right (display-only) so it lines up with
# the Pentacam reference under the shared ruler.
_ELITA_SHIFT_STEP_DEG = 1.0


# Pentacam preview reliability gate (display-only).
# A Pentacam reference file is often a *report screenshot* (topography map,
# Scheimpflug section, colour scales, text) rather than an iris photograph.  On
# such an image the detector can still emit a "limbus" that merely follows a
# smooth panel gradient — a false positive that looks like a misplaced,
# wrong-sized circle.  The detector's own aggregate confidence separates the
# clean cases (real iris captures score >= ~0.54; unrelated photos score
# <= ~0.43), so a *lone* low-confidence "limbus" is not drawn.
#
# However, when the detector returns a **pupil and a limbus that agree with each
# other** (concentric enough, pupil smaller by a plausible iris/pupil ratio),
# that full pair is a real eye even at lower confidence — those circles are
# drawn normally.  When no limbus is detected at all, the preview falls back to
# the two dark/bright lines on the left and right of the eye (the edge-line
# reference below), and only then to the hand-drawn iris ROI.  Nothing here
# feeds back into the centration pipeline — it only changes what the Pentacam
# preview shows.
_PENTACAM_MIN_TRUSTED_CONFIDENCE = 0.50


# Lower floor for the *paired* rule: a coherent pupil+limbus pair is trusted
# from here up (real captures on the reference screenshots scored ~0.43-0.49).
_PENTACAM_MIN_PAIRED_CONFIDENCE = 0.40


# Fallback limbus estimate when the detector found **no** limbus at all: the
# limbus is placed concentrically with the pupil at this multiple of the pupil
# radius.  Calibrated against the annotated clinical set
# (``clinical_data/clean/annotations/annotations.json``, 12 eyes): median error
# 4 px, versus ~79 px for a raw image-edge search (which drifts onto the
# surgical suction ring).  Display-only, so a missing ring is never silent.
_PENTACAM_LIMBUS_PUPIL_RATIO = 2.6

# Hand-sizing of the blue limbus ring in the Add Patient preview
# (display-only -- nothing here is ever fed back into the centration /
# pupil / limbus pipeline).  One mouse-wheel notch over the preview is a fine
# 2 % step; the "- Ring" / "+ Ring" buttons use a coarser 5 % so that a few
# clicks are enough.  A left-drag that *starts* on the ring outline pulls that
# part of the ring out (see below), so the grab band is a few pixels wide.
_RING_WHEEL_STEP = 0.02
_RING_BTN_STEP = 0.05
_RING_GRAB_PX = 12.0
# Tolerance for the centre cross that lets the user move a whole ring.
_RING_CENTER_TOL = 12.0

# A ring can also be *pulled* outwards at the point the user drags, instead of
# being grown as a whole circle: the ring then carries a per-angle "profile" of
# radius multipliers around its nominal radius ``r``, sampled this many times
# around the circle.  A drag bends only the samples within
# ``_RING_BULGE_SPAN_DEG`` either side of the grab point (with a cosine
# falloff), so the rest of the ring keeps the shape it had.  Display-only.
_RING_SAMPLES = 72
_RING_BULGE_SPAN_DEG = 35.0


def _scale_ring_radius(radius: float, factor: float, h: float, w: float) -> float:
    """Scale a hand-set ring radius by *factor* (>1 grows) and clamp it.

    The manual ring can never shrink below a few pixels nor grow past the
    frame diagonal, so an aggressive click still leaves a usable ring on
    screen.  Mirrors the clamping the mouse wheel has always applied.
    Display-only.
    """
    scaled = float(radius) * float(factor)
    return max(8.0, min(scaled, math.hypot(float(w), float(h))))


def _ring_angles(samples: int = _RING_SAMPLES) -> np.ndarray:
    """Polar angles (degrees, ``[0, 360)``) of a ring profile's samples."""
    return np.linspace(0.0, 360.0, int(samples), endpoint=False)


def _ring_profile(
    profile: Optional[np.ndarray], samples: int = _RING_SAMPLES
) -> np.ndarray:
    """A ring's per-angle radius multipliers, normalised to *samples* entries.

    ``None`` (a plain circle) yields all ones.  A profile of another length is
    resampled by periodic linear interpolation, so callers can always assume a
    fixed length.  Display-only.
    """
    n = int(samples)
    if profile is None:
        return np.ones(n, dtype=float)
    prof = np.asarray(profile, dtype=float).ravel()
    if prof.size == 0:
        return np.ones(n, dtype=float)
    if prof.size == n:
        return prof.copy()
    idx = (np.arange(n, dtype=float) * (prof.size / float(n))) % prof.size
    i0 = np.floor(idx).astype(int)
    i1 = (i0 + 1) % prof.size
    frac = idx - i0
    return prof[i0] * (1.0 - frac) + prof[i1] * frac


def _ring_radius_at(r: float, profile: Optional[np.ndarray], angle_deg: Any) -> Any:
    """Radius (px) of a hand-set ring at *angle_deg* (degrees, image frame).

    A ring without a profile is a plain circle, so this is simply ``r``; a
    *pulled* ring interpolates its per-angle profile (wrapping at 360°).
    *angle_deg* may be a scalar or an array; the angle convention matches
    ``_preview_ellipse_points`` / the unwrap (0° = image right, toward +y).
    Display-only.
    """
    r = float(r)
    if profile is None:
        if np.isscalar(angle_deg):
            return r
        return np.full(np.shape(angle_deg), r, dtype=float)
    prof = _ring_profile(profile)
    xs = np.concatenate([_ring_angles(prof.size), [360.0]])
    ys = np.concatenate([prof, [prof[0]]])
    return r * np.interp(np.asarray(angle_deg, dtype=float) % 360.0, xs, ys)


def _ring_bulge_profile(
    profile: Optional[np.ndarray],
    r: float,
    grab_angle_deg: float,
    target_radius: float,
    lo: float,
    hi: float,
    samples: int = _RING_SAMPLES,
) -> np.ndarray:
    """Per-angle multipliers with a smooth bulge centred on the grab point.

    The ring reaches *target_radius* exactly at ``grab_angle_deg`` and fades
    back to the shape it already had over ``_RING_BULGE_SPAN_DEG`` either side,
    so dragging pulls out only the part of the ring under the pointer rather
    than the whole circle.  Multipliers are clamped so every radius stays
    within ``[lo, hi]`` px.  Display-only.
    """
    n = int(samples)
    r = max(float(r), 1e-6)
    prof = _ring_profile(profile, n)
    grab = float(grab_angle_deg) % 360.0
    # The clamp must straddle the ring's own radius, or the untouched samples
    # (which sit at their usual radius) would be rescaled by the clip.
    lo = min(float(lo), r)
    hi = max(float(hi), r)
    want = float(np.clip(float(target_radius), lo, hi)) / r
    span = min(max(float(_RING_BULGE_SPAN_DEG), 1e-6), 180.0)
    delta = (_ring_angles(n) - grab + 180.0) % 360.0 - 180.0
    weight = np.where(
        np.abs(delta) >= span, 0.0, 0.5 * (1.0 + np.cos(np.pi * delta / span))
    )

    # Level the change at the grab point so the ring meets the pointer there,
    # then taper it away so the rest of the ring is untouched.
    at_grab = float(_ring_radius_at(r, prof, grab)) / r
    prof = prof + (want - at_grab) * weight
    return np.clip(prof, float(lo) / r, float(hi) / r)


def _ring_polygon_points(
    ring: Dict[str, Any],
    scale_x: float,
    scale_y: float,
    offset: float = 2.0,
    segments: int = _RING_SAMPLES,
    offset_x: Optional[float] = None,
    offset_y: Optional[float] = None,
) -> list:
    """Sample a (possibly *pulled*) hand-set ring into canvas points.

    For a plain circle this matches ``_preview_ellipse_points`` of a circle; a
    pulled ring instead follows its per-angle profile, so the overlay draws
    exactly the shape the strip unwraps.  Returns a flat
    ``[x0, y0, x1, y1, ...]`` list.  Display-only.
    """
    angles = _ring_angles(int(segments))
    radii = np.asarray(
        _ring_radius_at(ring["r"], ring.get("profile"), angles), dtype=float
    )
    rad = np.radians(angles)
    ox = offset if offset_x is None else float(offset_x)
    oy = offset if offset_y is None else float(offset_y)
    cx = float(ring["cx"]) * scale_x + ox
    cy = float(ring["cy"]) * scale_y + oy
    xs = (cx + radii * np.cos(rad) * scale_x).tolist()
    ys = (cy + radii * np.sin(rad) * scale_y).tolist()
    points: list = []
    for x, y in zip(xs, ys):
        points.append(x)
        points.append(y)
    return points


def _ring_canvas_radius(ring: Dict[str, Any], sx: float, sy: float, angle: float) -> float:
    """Calculates the drawn radius on canvas at polar angle."""
    prof = ring.get("profile")
    if prof is not None:
        deg = math.degrees(angle) % 360.0
        r_img = float(_ring_radius_at(ring["r"], prof, deg))
        return r_img * math.hypot(sx * math.cos(angle), sy * math.sin(angle))
    rx = float(ring.get("rx", ring["r"])) * sx
    ry = float(ring.get("ry", ring["r"])) * sy
    if rx <= 0 or ry <= 0:
        return 0.0
    ang_rad = math.radians(float(ring.get("angle_deg", 0.0)))
    rel_angle = angle - ang_rad
    cos_t = math.cos(rel_angle)
    sin_t = math.sin(rel_angle)
    denom = math.sqrt((ry * cos_t) ** 2 + (rx * sin_t) ** 2)
    return (rx * ry) / denom if denom > 0 else 0.0


def _preview_geometry(
    dw: int, dh: int, req_w: int, req_h: int, screen_w: int, bx: int
) -> Tuple[int, int]:
    """Client size for the Pentacam preview window.

    The preview mirrors the Add Patient dialog (the image gets as much room
    as the form), is never shrunk below the space its own contents need, and
    never widens the right-docked pair past the edge of the screen.
    """
    max_w = max(req_w, screen_w - dw - 2 * bx)
    return max(req_w, min(dw, max_w)), max(req_h, dh)


def _preview_thumb_limits(cw: int, ch: int) -> Tuple[int, int]:
    """Thumbnail limits for the Pentacam preview canvas.

    The image is fitted to the canvas' *actual* size, so it grows with the
    (now dialog-sized) preview window instead of staying hard-wired to the
    556x196 of the smaller window it was drawn for.  A canvas that has not
    been laid out yet keeps those old limits so nothing is ever clipped.
    """
    if cw > 8 and ch > 8:
        return cw - 8, ch - 8
    return 556, 196


class PupilTrackingGUI:
    def __init__(self, root: tk.Tk, colors=None) -> None:
        self.root = root
        self.root.withdraw()
        self.root.title(_WINDOW_TITLE)
        self.root.geometry(f"{_MIN_WIDTH}x{_MIN_HEIGHT}")
        self.root.minsize(_MIN_WIDTH, _MIN_HEIGHT)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if colors is None:
            colors = DarkTheme.apply(root)
        self._colors = colors

        self._runtime_profile = apply_runtime_optimizations(detect_runtime_profile())
        self.cfg = get_config()
        self.logger = get_logger()

        self._detector: Optional[UnifiedDetector] = None
        self._tracker: Optional[EyeKalmanTracker] = None
        self._corneal_calc: Optional[CornealCenterCalculator] = None

        self._current_image: Optional[np.ndarray] = None
        self._current_result: Optional[Any] = None
        self._display_image: Optional[ImageTk.PhotoImage] = None
        self._canvas_image_id: Optional[int] = None
        self._resize_after_id: Optional[str] = None
        self._display_scale: float = 1.0
        self._display_origin: Tuple[int, int] = (0, 0)
        self._display_size: Tuple[int, int] = (0, 0)

        self._video_cap: Optional[cv2.VideoCapture] = None
        self._video_running = False
        self._video_thread: Optional[threading.Thread] = None
        self._camera_mode = False
        self._frame_count = 0
        self._video_paused = False
        self._video_total_frames = 0
        self._video_start_time = 0.0
        self._last_display_update = 0.0
        self._active_source: Optional[Any] = None
        self._settings_apply_after_id: Optional[str] = None
        self._pending_live_apply_reasons: set[str] = set()
        self._suspend_live_settings_apply = False
        self._restart_in_progress = False

        self._fast_engine: Optional[Any] = None
        self._opt_processor: Optional[Any] = None
        self._async_capture: Optional[Any] = None
        self._using_optimized_camera: bool = False
        self._last_opt_stats: Dict[str, Any] = {}
        self._manual_roi: Optional[Dict[str, float]] = None

        # Detection is gated on the ROI: after a video/camera is loaded we show
        # the first frame and hold until the user confirms an ROI, at which point
        # the stored start callable spins up the detection thread.
        self._awaiting_roi: bool = False
        self._pending_detection_start: Optional[Any] = None
        self._roi_edit_active = False
        self._roi_drag_mode: Optional[str] = None
        self._roi_drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._roi_original_before_edit: Optional[Dict[str, float]] = None
        self._roi_preview: Optional[Dict[str, float]] = None
        self._manual_ring: Optional[Dict[str, float]] = None
        self._ring_edit_active = False
        self._ring_drag_mode: Optional[str] = None
        self._ring_drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._ring_original_before_edit: Optional[Dict[str, float]] = None
        self._ring_preview: Optional[Dict[str, float]] = None

        self._results_history: List[Dict[str, Any]] = []

        # ══════════════════════════════════════════════════════════
        # RECORDING — DualFrameRecorder instance (Raw J&J + Overlay)
        # ══════════════════════════════════════════════════════════
        # Keep the default recording path on the software MP4 fallback to avoid
        # the broken OpenH264/FFmpeg library mismatch seen on Windows.
        self._recorder = DualFrameRecorder(prefer_qsv=False)
        self._recorder.set_status_callback(self._on_recorder_status)
        self._recording_path: Optional[str] = None
        self._recording_raw_path: Optional[str] = None
        self._recording_sync_path: Optional[str] = None
        self._recording_default_path_var = tk.StringVar(value="")
        self._recording_timer_id: Optional[str] = None
        self._recording_fps_var = tk.DoubleVar(value=0.0)
        self._recording_dropped_var = tk.IntVar(value=0)

        # Patient/session storage. The root defaults to the Centration Folder
        # Location and persists between launches.
        self._storage_settings_path = Path.home() / "AppData" / "Local" / "MedevplusIXcentai" / "settings.txt"
        self._storage_root_var = tk.StringVar(value=str(Path.home() / "Desktop" / "Centration"))
        self._patient_id_var = tk.StringVar(value="")
        self._patient_name_var = tk.StringVar(value="")
        self._selected_eye_var = tk.StringVar(value="OD")
        self._active_patient_dir: Optional[Path] = None
        self._add_px_dialog_open = False
        self._pentacam_detector = None
        self._patient_summary_var = tk.StringVar(value="Eye: OD | No patient selected")
        self._load_storage_settings()
        self._recording_location_var = tk.StringVar(
            value=f"Centration folder: {self._storage_root_var.get()}"
        )
        # ══════════════════════════════════════════════════════════

        self._init_settings_vars()

        self._build_menu()
        self._build_toolbar()
        self._build_status_bar()
        self._build_main_area()
        self._bind_live_setting_callbacks()
        self._install_crash_guards()

        self.root.update_idletasks()
        self.root.deiconify()
        self.root.after(100, self._init_detector)

    # ================================================================
    # Settings Tk Variables
    # ================================================================

    def _init_settings_vars(self) -> None:
        self._use_optimized_var = tk.BooleanVar(value=True)
        self._fp16_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_fp16
        )
        self._compile_var = tk.BooleanVar(
            value=self._runtime_profile.recommended_compile
        )

        self._resolution_var = tk.IntVar(
            value=self._runtime_profile.recommended_resolution
        )
        self._stride_var = tk.IntVar(value=1)
        self._target_fps_var = tk.DoubleVar(
            value=self._runtime_profile.recommended_target_fps
        )
        self._performance_preset_var = tk.StringVar(value="balanced")

        self._roi_var = tk.BooleanVar(value=True)
        self._roi_cache_var = tk.IntVar(value=5)
        self._roi_status_var = tk.StringVar(value="Manual ROI: Off")
        self._ring_status_var = tk.StringVar(value="Manual Ring: Off")
        self._kalman_process_var = tk.DoubleVar(value=0.03)
        self._kalman_measure_var = tk.DoubleVar(value=0.1)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 3 of 12 — Grayscale mode Tk variable
        # ══════════════════════════════════════════════════════════
        self._grayscale_mode_var = tk.StringVar(value="off")
        # ══════════════════════════════════════════════════════════

        self._pupil_fill_alpha_var = tk.IntVar(value=0)
        self._limbus_fill_alpha_var = tk.IntVar(value=0)

        # ── Modular Calibration Settings ──
        init_mode = getattr(self.cfg.calibration, "mode", "FIXED_PIXEL_SCALE") if hasattr(self.cfg, "calibration") else "FIXED_PIXEL_SCALE"
        self._calibration_mode_var = tk.StringVar(value=init_mode)
        init_manual_px = float(getattr(self.cfg.calibration, "manual_px_per_mm", 58.2) or 58.2) if hasattr(self.cfg, "calibration") else 58.2
        self._fixed_scale_var = tk.DoubleVar(value=init_manual_px)
        init_corneal = float(getattr(self.cfg.calibration, "corneal_diameter_mm", 12.0) or 12.0) if hasattr(self.cfg, "calibration") else 12.0
        self._corneal_ref_mm_var = tk.DoubleVar(value=init_corneal)
        init_ring = float(getattr(self.cfg.calibration, "suction_ring_diameter_mm", 9.4) or 9.4) if hasattr(self.cfg, "calibration") else 9.4
        self._ring_ref_mm_var = tk.DoubleVar(value=init_ring)

        # ── Interactive 2-Point Ruler Tool State ──
        self._ruler_calibration_active: bool = False
        self._ruler_points: list[tuple[float, float]] = []
        self._ruler_known_dist_mm_var = tk.DoubleVar(value=10.0)


    # ================================================================
    # Detector Initialisation
    # ================================================================

    def _init_detector(self) -> None:
        self._status_var.set("Loading model…")
        self.root.update()

        try:
            self._tracker = EyeKalmanTracker(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init tracker: %s", exc)
            self._tracker = None

        try:
            self._corneal_calc = CornealCenterCalculator(config=self.cfg)
        except Exception as exc:
            self.logger.error("Failed to init corneal calc: %s", exc)
            self._corneal_calc = None

        try:
            # ══════════════════════════════════════════════════════
            # GRAYSCALE GUI 4 of 12 — Pass grayscale_mode to detector
            # ══════════════════════════════════════════════════════
            self._detector = UnifiedDetector(
                config=self.cfg,
                grayscale_mode=self._grayscale_mode_var.get(),
            )
            # ══════════════════════════════════════════════════════

            if self._detector.ml_engine.available:
                tag = "GPU" if _FAST_PIPELINE_AVAILABLE else "GPU (classic)"
                self._model_status_var.set(f"Model: Ready ({tag})")
            else:
                self._model_status_var.set("Model: Classical Only (ML unavailable)")

            self._status_var.set("Ready — Load an image or start camera")
        except Exception as exc:
            self.logger.error("Failed to init detector: %s", exc)
            self._detector = None
            self._model_status_var.set(f"Model: ERROR — {exc}")
            self._status_var.set(
                "Model loading failed — classical detection unavailable"
            )

    # ================================================================
    # Grayscale Mode Control (NEW)
    # ================================================================

    # ══════════════════════════════════════════════════════════════════
    # GRAYSCALE GUI 5 of 12 — Grayscale toggle, apply, display convert
    # ══════════════════════════════════════════════════════════════════

    def _toggle_grayscale(self, event=None) -> None:
        """Cycle grayscale mode: OFF → AUTO → FORCE → OFF.

        Called by the [G] keyboard shortcut and the toolbar button.
        """
        current = self._grayscale_mode_var.get()
        try:
            idx = _GRAYSCALE_CYCLE.index(current)
        except ValueError:
            idx = 0
        next_mode = _GRAYSCALE_CYCLE[(idx + 1) % len(_GRAYSCALE_CYCLE)]
        self._grayscale_mode_var.set(next_mode)
        self._on_grayscale_changed()

    def _on_grayscale_changed(self, *_args) -> None:
        """Apply grayscale mode change to the detector and update UI.

        Called when the settings dropdown or toolbar button changes.
        """
        mode = self._grayscale_mode_var.get()

        # Apply to detector
        if self._detector is not None:
            self._detector.set_grayscale_mode(mode)

        # Update toolbar button appearance
        label = _GRAYSCALE_LABELS.get(mode, "RGB")
        color = _GRAYSCALE_COLORS.get(mode, "#aaaaaa")
        if hasattr(self, "_gray_btn"):
            self._gray_btn.config(text=f"🔲 {label}")
        if hasattr(self, "_gray_indicator"):
            self._gray_indicator.config(
                text=f"  {label}  ",
                foreground=color,
            )

        # Update status
        mode_names = {
            "off": "RGB (original)",
            "auto": "Auto-detect",
            "force": "Forced grayscale",
        }
        self._status_var.set(f"Grayscale mode: {mode_names.get(mode, mode)}")

        # Refresh display immediately
        self._refresh_display()

        # Mirror the mode onto the unwrap panel too, so the Pentacam and
        # live ELITA strips match the normal window's RGB/grayscale look.
        try:
            self._redraw_unwrap()
        except Exception:
            pass

    def _convert_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale for display when mode is active.

        When grayscale mode is FORCE, the displayed image becomes
        grayscale (like an IR camera) with coloured overlays on top.

        When mode is AUTO, converts only if the detector detected
        the input as grayscale.

        When mode is OFF, returns the original frame unchanged.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame.

        Returns
        -------
        np.ndarray
            Frame for display — 3-channel BGR uint8.
        """
        mode = self._grayscale_mode_var.get()

        if mode == "off":
            return frame

        if mode == "force":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if mode == "auto":
            # Convert only if detector applied grayscale processing
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return frame

    # ══════════════════════════════════════════════════════════════════
    # END GRAYSCALE GUI 5
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    # RECORDING — Professional recording with FrameRecorder
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_folder_name(value: str) -> str:
        """Return a Windows-safe, readable folder name."""
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
        return cleaned.strip(". ") or "Unknown"

    def _load_storage_settings(self) -> None:
        try:
            if self._storage_settings_path.exists():
                for line in self._storage_settings_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("data_storage_root="):
                        root = line.split("=", 1)[1].strip()
                        if root:
                            self._storage_root_var.set(root)
                        break
        except (OSError, UnicodeError) as exc:
            self.logger.warning("Could not load storage settings: %s", exc)

    def _save_storage_settings(self) -> bool:
        location = self._storage_root_var.get().strip()
        if not location:
            messagebox.showwarning("Storage location required", "Choose a folder for patient data first.")
            return False
        try:
            root = Path(location).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            self._storage_root_var.set(str(root))
            self._storage_settings_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_settings_path.write_text(
                f"# Medevplus IXcentai - Settings\ndata_storage_root={root}\n",
                encoding="utf-8",
            )
            self._status_var.set(f"Patient data location saved → {root}")
            return True
        except OSError as exc:
            messagebox.showerror("Storage location error", f"Cannot use that folder:\n{exc}")
            return False

    def _choose_storage_root(self) -> None:
        selected = filedialog.askdirectory(title="Choose patient data storage folder", initialdir=self._storage_root_var.get())
        if selected:
            self._storage_root_var.set(selected)

    @staticmethod
    def _calc_dialog_geometry(root_or_win: Any) -> Tuple[int, int, int, int]:
        """Compute responsive dialog dimensions and position centrally.
        Uses available screen space (~92-95% screen width and ~88-92% height)
        without being constrained by a smaller, unmaximized parent window.
        """
        MIN_W, MIN_H = 1000, 600
        sw = root_or_win.winfo_screenwidth()
        sh = root_or_win.winfo_screenheight()
        rw = root_or_win.winfo_width()
        rh = root_or_win.winfo_height()
        rx0 = getattr(root_or_win, "winfo_rootx", lambda: 0)()
        ry0 = getattr(root_or_win, "winfo_rooty", lambda: 0)()
        target_w = max(MIN_W, min(int(sw * 0.95), max(int(sw * 0.92), int(rw * 0.98))))
        target_h = max(MIN_H, min(int(sh * 0.92), max(int(sh * 0.88), int(rh * 0.96))))
        if rw > 300 and rh > 300:
            rx = rx0 + (rw - target_w) // 2
            ry = ry0 + (rh - target_h) // 2
        else:
            rx = max(0, (sw - target_w) // 2)
            ry = max(0, (sh - target_h) // 2)
        rx = max(10, min(rx, max(0, sw - target_w - 10)))
        ry = max(10, min(ry, max(0, sh - target_h - 40)))
        return target_w, target_h, rx, ry

    def _open_patient_dialog(self) -> None:
        if self._recorder.is_recording:
            return

        # No modal grab is used here: the Pentacam preview opens in its own
        # window beside this one and must stay interactive, so re-entry is
        # guarded with the flag instead of a grab.
        if getattr(self, "_add_px_dialog_open", False):
            return
        self._add_px_dialog_open = True
        self._update_patient_controls()
        c = self._colors

        # Dialog colours — white background, dark text
        DLG_BG      = "#FFFFFF"
        DLG_FG      = "#1A1A1A"
        DLG_FG_HINT = "#555555"
        DLG_BORDER  = "#CCCCCC"
        DLG_INPUT   = "#F5F5F5"
        DLG_ACCENT  = "#0071E3"
        DLG_BTN_SEC = "#E0E0E0"
        DLG_BTN_ACT = "#C8C8C8"
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Patient")
        dialog.transient(self.root)

        # Dialog window icon
        try:
            from PIL import Image, ImageTk as _ITk
            import pathlib as _pl
            _icon_path = _pl.Path(__file__).parent / "add-patient.png.png"
            _img = Image.open(_icon_path).resize((32, 32), Image.LANCZOS)
            _icon = _ITk.PhotoImage(_img)
            dialog.iconphoto(False, _icon)
            dialog._icon_ref = _icon  # keep reference alive
        except Exception:
            pass
        dialog.resizable(True, True)

        # Merged Add Patient + Pentacam Preview dialog dimensions
        MIN_W, MIN_H = 1100, 600
        dialog.minsize(MIN_W, MIN_H)

        def _get_screen_work_area() -> Tuple[int, int, int, int]:
            """Retrieve usable screen work area (excluding taskbar and OS panels)."""
            try:
                import ctypes
                from ctypes import wintypes
                class _RECT(ctypes.Structure):
                    _fields_ = [
                        ("left", wintypes.LONG),
                        ("top", wintypes.LONG),
                        ("right", wintypes.LONG),
                        ("bottom", wintypes.LONG),
                    ]
                rect = _RECT()
                if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
                    return int(rect.left), int(rect.top), int(rect.right - rect.left), int(rect.bottom - rect.top)
            except Exception:
                pass
            sw = dialog.winfo_screenwidth()
            sh = dialog.winfo_screenheight()
            return 0, 0, sw, max(MIN_H, sh - 60)

        def _calc_dialog_geometry() -> Tuple[int, int, int, int]:
            """Compute responsive dialog dimensions and position centrally within usable work area.
            Ensures dialog fits within visible screen space, leaving room for title bars, borders, and taskbar.
            """
            self.root.update_idletasks()
            wx, wy, ww, wh = _get_screen_work_area()
            # Reserve space for window decorations (title bar ~36px, borders ~16px) and margins
            max_client_w = max(MIN_W, ww - 24)
            max_client_h = max(MIN_H, wh - 48)

            ideal_w = min(1380, max(MIN_W, int(ww * 0.88)))
            ideal_h = min(820, max(MIN_H, int(wh * 0.86)))
            target_w = min(max_client_w, ideal_w)
            target_h = min(max_client_h, ideal_h)

            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            rx0 = self.root.winfo_rootx()
            ry0 = self.root.winfo_rooty()
            if self.root.winfo_viewable() and rw > 300 and rh > 300:
                rx = rx0 + (rw - target_w) // 2
                ry = ry0 + (rh - target_h) // 2
            else:
                rx = wx + (ww - target_w) // 2
                ry = wy + (wh - target_h) // 2

            # Clamp so window decorations and content stay strictly inside the visible work area
            rx = max(wx + 8, min(rx, wx + ww - target_w - 16))
            ry = max(wy + 8, min(ry, wy + wh - target_h - 40))
            return target_w, target_h, rx, ry

        target_w, target_h, rx, ry = _calc_dialog_geometry()
        dialog.geometry(f"{target_w}x{target_h}+{rx}+{ry}")
        dialog.configure(bg=DLG_BG)

        def _on_root_configure(event: Any) -> None:
            if not dialog.winfo_exists():
                return
            if event.widget == self.root:
                tw, th, nx, ny = _calc_dialog_geometry()
                dialog.geometry(f"{tw}x{th}+{nx}+{ny}")

        root_bind_id = self.root.bind("<Configure>", _on_root_configure, add="+")

        from pupil_tracking.cyclotorsion.pentacam_loader import (
            load_pentacam_image,
            flip_pentacam,
            PENTACAM_FILETYPES,
        )

        # ── Header ──────────────────────────────────────────────────
        hdr = tk.Frame(dialog, bg=DLG_BG)
        hdr.pack(side=tk.TOP, fill=tk.X, padx=28, pady=(14, 8))
        tk.Label(
            hdr, text="Add Patient",
            font=("Segoe UI", 16, "bold"),
            bg=DLG_BG, fg=DLG_FG,
        ).pack(anchor=tk.W)
        tk.Label(
            hdr, text="Fields marked * are required",
            font=("Segoe UI", 10, "bold"),
            bg=DLG_BG, fg=DLG_FG_HINT,
        ).pack(anchor=tk.W, pady=(2, 0))
        tk.Frame(dialog, bg=DLG_BORDER, height=1).pack(side=tk.TOP, fill=tk.X)

        # ── Footer Pinned to Bottom First ───────────────────────────
        # Packing side=BOTTOM before side=TOP expand=True ensures the action footer
        # is never clipped or pushed outside the window when resized/compressed.
        foot = tk.Frame(dialog, bg=DLG_BG)
        foot.pack(side=tk.BOTTOM, fill=tk.X, padx=28, pady=12)
        tk.Frame(dialog, bg=DLG_BORDER, height=1).pack(side=tk.BOTTOM, fill=tk.X)

        # ── Body Content Container (2 columns: Form + Preview) ───────
        content_pane = tk.Frame(dialog, bg=DLG_BG)
        content_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=6)
        content_pane.columnconfigure(0, weight=0, minsize=400)  # Left: Patient form (fixed comfortable width)
        content_pane.columnconfigure(1, weight=0, minsize=1)    # Vertical separator
        content_pane.columnconfigure(2, weight=1, minsize=520)  # Right: Pentacam preview (expands to fill workspace)
        content_pane.rowconfigure(0, weight=1)

        # ── Left Column: Patient Form ───────────────────────────────
        body = tk.Frame(content_pane, bg=DLG_BG)
        body.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        body.columnconfigure(0, minsize=150)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, minsize=80)
        patient_id   = tk.StringVar()
        patient_name = tk.StringVar()
        hvid_od      = tk.StringVar()
        hvid_os      = tk.StringVar()
        pentacam_od  = tk.StringVar()
        pentacam_os  = tk.StringVar()
        LBL_FONT = ("Segoe UI", 11, "bold")
        ENT_FONT = ("Segoe UI", 11, "bold")
        BTN_FONT = ("Segoe UI", 10, "bold")

        def _field(grid_row, label_text, var, browse=False):
            tk.Label(
                body, text=label_text,
                font=LBL_FONT,
                bg=DLG_BG, fg=DLG_FG,
                anchor=tk.W,
            ).grid(row=grid_row, column=0, sticky=tk.W, pady=7, padx=(0, 14))
            ent = tk.Entry(
                body, textvariable=var,
                font=ENT_FONT,
                bg=DLG_INPUT, fg=DLG_FG,
                insertbackground=DLG_FG,
                relief=tk.FLAT, bd=0,
                highlightthickness=1,
                highlightbackground=DLG_BORDER,
                highlightcolor=DLG_ACCENT,
            )
            ent.grid(row=grid_row, column=1, sticky=tk.EW, pady=7, ipady=7)
            if browse:
                lbl_copy = label_text

                def _browse_pentacam(target_var=var, field_label=lbl_copy):
                    from pupil_tracking.cyclotorsion.pentacam_loader import (
                        PENTACAM_FILETYPES,
                        load_pentacam_image,
                    )
                    selected = filedialog.askopenfilename(
                        title=f"Select {field_label}",
                        filetypes=list(PENTACAM_FILETYPES),
                    )
                    if selected:
                        import os
                        clean_path = os.path.normpath(selected.strip().strip('"').strip("'"))
                        target_var.set(clean_path)
                        # Validate immediately and notify user if invalid
                        img = load_pentacam_image(clean_path)
                        if img is None:
                            messagebox.showerror(
                                "Image Load Error",
                                f"Failed to load image from:\n{clean_path}\n\n"
                                "The file could not be decoded or is corrupted.\n"
                                "Supported formats: .bmp, .jpg, .jpeg, .png, .tif, .dcm, .pdf",
                            )

                tk.Button(
                    body, text="Browse",
                    font=BTN_FONT,
                    bg=DLG_BTN_SEC, fg=DLG_FG,
                    activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
                    relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2",
                    command=_browse_pentacam,
                ).grid(row=grid_row, column=2, sticky=tk.EW, padx=(10, 0), pady=7)
        _field(0, "Patient ID *",       patient_id)
        _field(1, "Patient Name *",     patient_name)
        _field(2, "HVID OD (optional)", hvid_od)
        _field(3, "HVID OS (optional)", hvid_os)
        _field(4, "Pentacam OD (optional)", pentacam_od, browse=True)
        _field(5, "Pentacam OS (optional)", pentacam_os, browse=True)

        # ── Column Separator ─────────────────────────────────────────
        tk.Frame(content_pane, bg=DLG_BORDER, width=1).grid(
            row=0, column=1, sticky="ns", pady=4
        )

        # ── Right Column: Pentacam Preview (BUG-001) ─────────────────
        # Integrated directly into the Add Patient dialog; no separate window.
        preview_win = dialog
        # preview_win.geometry(f"{W}x{H}")
        preview_frame = tk.Frame(content_pane, bg=DLG_BG)
        preview_frame.grid(row=0, column=2, sticky="nsew", padx=(16, 0))
        preview_label_text = tk.StringVar(value="No image selected")
        tk.Label(
            preview_frame,
            text="Pentacam Preview \u2014 Fixed Iris ROI (847 \u00d7 542 px):",
            font=("Segoe UI", 10, "bold"),
            bg=DLG_BG, fg=DLG_FG,
        ).pack(anchor=tk.W)

        # Eye selector — chooses which Pentacam image the preview/ROI applies to.
        init_eye = self._selected_eye_var.get() if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() in ("OD", "OS") else "OD"
        preview_eye_var = tk.StringVar(value=init_eye)
        eye_row = tk.Frame(preview_frame, bg=DLG_BG)
        eye_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(
            eye_row, text="Preview eye:", font=("Segoe UI", 9, "bold"),
            bg=DLG_BG, fg=DLG_FG,
        ).pack(side=tk.LEFT, padx=(0, 6))
        eye_btn_refs: Dict[str, tk.Button] = {}

        flip_var = tk.BooleanVar(value=True)

        def _toggle_flip() -> None:
            # Flip the stored ROI on both axes so it keeps tracking the same anatomy
            eye = _active_eye()
            roi = roi_by_eye.get(eye)
            if roi is not None:
                roi_by_eye[eye] = flip_rect_norm(roi)
            result_by_eye[eye] = None
            _refresh_preview()

        auto_flip_cb = tk.Checkbutton(
            eye_row,
            text="Auto-flip image (180\u00b0)",
            variable=flip_var,
            bg=DLG_BG, fg=DLG_FG,
            activebackground=DLG_BG,
            font=("Segoe UI", 9),
            command=_toggle_flip,
        )
        auto_flip_cb.pack(side=tk.RIGHT, padx=(0, 4))

        # Responsive Pentacam preview canvas (resizes with dialog, fits comfortably)
        preview_canvas = tk.Canvas(
            preview_frame,
            width=520, height=280,
            bg="#0a0a0a",
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            cursor="crosshair",
        )
        preview_canvas.pack(fill=tk.BOTH, expand=True, pady=(4, 4))
        preview_canvas._photo = None
        preview_canvas._thumb_size = (0, 0)
        preview_canvas._display_offset = (2, 2)

        PENTACAM_FIXED_ROI_W = 847
        PENTACAM_FIXED_ROI_H = 542

        def _default_fixed_roi(w: int, h: int) -> Tuple[float, float, float, float]:
            """Fixed-size anterior segment Scheimpflug aperture ROI (Option A: 847x542 px)."""
            rw = min(float(w), float(PENTACAM_FIXED_ROI_W))
            rh = min(float(h), float(PENTACAM_FIXED_ROI_H))
            ix0 = int(round((float(w) - rw) / 2.0))
            iy0 = int(round((float(h) - rh) / 2.0))
            ix1 = ix0 + int(rw)
            iy1 = iy0 + int(rh)
            return (ix0 / float(w), iy0 / float(h), ix1 / float(w), iy1 / float(h))

        # Normalised iris ROI per eye in *flipped* image coordinates (0..1).
        roi_by_eye: Dict[str, Optional[Tuple[float, float, float, float]]] = {"OD": None, "OS": None}
        roi_active_by_eye: Dict[str, bool] = {"OD": False, "OS": False}
        roi_locked_by_eye: Dict[str, bool] = {"OD": False, "OS": False}
        roi_locked_var = tk.BooleanVar(value=False)
        roi_mode_var = tk.BooleanVar(value=False)
        result_by_eye: Dict[str, Any] = {"OD": None, "OS": None}
        reliable_by_eye: Dict[str, bool] = {"OD": True, "OS": True}
        edge_by_eye: Dict[str, Any] = {"OD": None, "OS": None}
        edge_note_by_eye: Dict[str, str] = {"OD": "", "OS": ""}
        _drag_state: Dict[str, Any] = {"origin": None, "mode": None, "orig_roi": None}

        # Last full-resolution flipped Pentacam image shown in the preview
        _preview_state: Dict[str, Any] = {"image": None, "result": None, "token": 0, "unwrap_strip": None}
        unwrap_strip_by_eye: Dict[str, Optional[np.ndarray]] = {"OD": None, "OS": None}

        # Hand-adjusted limbus and pupil overrides per eye
        ring_override: Dict[str, Optional[Dict[str, Any]]] = {"OD": None, "OS": None}
        pupil_override: Dict[str, Optional[Dict[str, float]]] = {"OD": None, "OS": None}
        manual_mode_by_eye: Dict[str, bool] = {"OD": False, "OS": False}
        manual_locked_by_eye: Dict[str, bool] = {"OD": False, "OS": False}
        _ring_drag: Dict[str, Any] = {"origin": None}
        _ring_extend: Dict[str, Any] = {"origin": None}
        _ring_hover: Dict[str, str] = {"cursor": ""}
        ring_target_var = tk.StringVar(value="limbus")

        def _ring_target() -> str:
            """The ring the hand-adjustment controls currently act on."""
            return "pupil" if ring_target_var.get() == "pupil" else "limbus"

        def _override_of(
            target: str,
        ) -> Dict[str, Optional[Dict[str, Any]]]:
            """The per-eye hand-set rings for *target* ("limbus"/"pupil")."""
            return pupil_override if target == "pupil" else ring_override

        def _make_circle(cx: float, cy: float, r: float) -> SimpleNamespace:
            """A circular ellipse-like geometry for the overlay/unwrap helpers."""
            return SimpleNamespace(
                center_x=float(cx), center_y=float(cy), radius=float(r),
                semi_major=float(r), semi_minor=float(r), angle_deg=0.0,
            )

        def _ring_dict(
            cx: float, cy: float, r: float, profile: Optional[np.ndarray] = None
        ) -> Dict[str, Any]:
            """A hand-set ring ``{cx, cy, r}``, plus a per-angle *profile*."""
            ring: Dict[str, Any] = {
                "cx": float(cx), "cy": float(cy), "r": float(r),
                "rx": float(r), "ry": float(r), "angle_deg": 0.0,
            }
            if profile is not None:
                ring["profile"] = profile
            return ring

        def _ellipse_to_dict(ellipse: Any) -> Optional[Dict[str, Any]]:
            """Convert an EllipseParams or SimpleNamespace to a dictionary with cx, cy, rx, ry, r, angle_deg."""
            if ellipse is None:
                return None
            try:
                rx = float(getattr(ellipse, "semi_major", getattr(ellipse, "radius", 0.0)))
                ry = float(getattr(ellipse, "semi_minor", getattr(ellipse, "radius", 0.0)))
                r = float(getattr(ellipse, "radius", 0.5 * (rx + ry)))
                return {
                    "cx": float(ellipse.center_x),
                    "cy": float(ellipse.center_y),
                    "rx": rx,
                    "ry": ry,
                    "r": r,
                    "angle_deg": float(getattr(ellipse, "angle_deg", 0.0)),
                }
            except Exception:
                return None

        def _dict_to_ellipse(ring_dict: Any) -> Optional[SimpleNamespace]:
            """Convert a dictionary with cx, cy, rx, ry, r, angle_deg back to an ellipse-like SimpleNamespace."""
            if ring_dict is None:
                return None
            try:
                cx = float(ring_dict["cx"])
                cy = float(ring_dict["cy"])
                rx = float(ring_dict.get("rx", ring_dict["r"]))
                ry = float(ring_dict.get("ry", ring_dict["r"]))
                r = float(ring_dict.get("r", 0.5 * (rx + ry)))
                ang = float(ring_dict.get("angle_deg", 0.0))
                return SimpleNamespace(
                    center_x=cx,
                    center_y=cy,
                    semi_major=rx,
                    semi_minor=ry,
                    radius=r,
                    angle_deg=ang,
                )
            except Exception:
                return None

        def _ellipse_to_ring(ellipse: Any) -> Optional[Dict[str, Any]]:
            """``{cx, cy, r}`` circle for an ellipse-like geometry, else None."""
            return _ellipse_to_dict(ellipse)

        def _detected_ring(target: str = "limbus") -> Optional[Dict[str, Any]]:
            """Automatic ring for the active preview, as ``{cx, cy, r}``."""
            if target == "pupil":
                result = _preview_state.get("result")
                pupil = getattr(result, "pupil", None) if result is not None else None
                if not bool(getattr(pupil, "detected", False)):
                    return None
                return _ellipse_to_dict(getattr(pupil, "ellipse", None))
            result = _preview_state.get("result")
            limbus = getattr(result, "limbus", None) if result is not None else None
            if not bool(getattr(limbus, "detected", False)):
                return None
            return _ellipse_to_dict(getattr(limbus, "ellipse", None))

        def _seed_ring(target: str = "limbus") -> Optional[Dict[str, Any]]:
            """Seed an initial ring geometry for hand adjustment if none exists."""
            eye = _active_eye()
            if not eye:
                return None
            image = _preview_state.get("image")
            if image is None:
                return None
            h, w = image.shape[:2]
            detected = _detected_ring(target)
            if detected is not None:
                r_dict = dict(detected)
                _override_of(target)[eye] = r_dict
                return r_dict
            roi = roi_by_eye.get(eye)
            if roi is not None:
                ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
                cx, cy = 0.5 * (ix0 + ix1), 0.5 * (iy0 + iy1)
                base_r = 0.35 * min(iw, ih)
            else:
                cx, cy = 0.5 * float(w), 0.5 * float(h)
                base_r = 0.25 * min(w, h)
            if target == "pupil":
                limbus = ring_override.get(eye) or _detected_ring("limbus")
                if limbus is not None:
                    r_dict = _ring_dict(float(limbus["cx"]), float(limbus["cy"]), 0.4 * float(limbus["r"]))
                else:
                    r_dict = _ring_dict(cx, cy, 0.4 * base_r)
            else:
                r_dict = _ring_dict(cx, cy, base_r)
            _override_of(target)[eye] = r_dict
            return r_dict

        def _ring_screen_geometry(target: str):
            """Screen coordinates and drawn radius of active boundary for hit testing."""
            eye = _active_eye()
            if not eye:
                return None
            ring = _override_of(target).get(eye) or _detected_ring(target)
            if ring is None:
                return None
            tw, th = getattr(preview_canvas, "_thumb_size", (0, 0))
            img = _preview_state.get("image")
            if img is None or tw <= 0 or th <= 0:
                return None
            h, w = img.shape[:2]
            sx, sy = tw / float(w), th / float(h)
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
            rcx = ox + float(ring["cx"]) * sx
            rcy = oy + float(ring["cy"]) * sy
            rrx = float(ring.get("rx", ring["r"])) * sx
            rry = float(ring.get("ry", ring["r"])) * sy
            return rcx, rcy, rrx, rry, ring, sx, sy

        def _on_ring_outline(x: float, y: float, target: str) -> bool:
            geom = _ring_screen_geometry(target)
            if geom is None:
                return False
            rcx, rcy, rrx, rry, ring, sx, sy = geom
            dx, dy = x - rcx, y - rcy
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            drawn_r = _ring_canvas_radius(ring, sx, sy, angle)
            return abs(dist - drawn_r) <= _RING_GRAB_PX

        def _start_ring_extend(x: float, y: float) -> bool:
            """Start dragging/extending the arc under the pointer."""
            eye = _active_eye()
            if not eye or not _eye_path(eye):
                return False
            target = _ring_target()
            geom = _ring_screen_geometry(target)
            if geom is None:
                return False
            rcx, rcy, rrx, rry, ring, sx, sy = geom
            dx, dy = x - rcx, y - rcy
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            drawn_r = _ring_canvas_radius(ring, sx, sy, angle)
            if abs(dist - drawn_r) > _RING_GRAB_PX:
                return False
            grab = math.degrees(angle) % 360.0
            _ring_extend["origin"] = (x, y, ring, grab, eye, target, sx, sy)
            return True

        def _ring_extend_drag(x: float, y: float) -> bool:
            """Drag the grabbed arc out without changing the rest of the ring."""
            origin = _ring_extend.get("origin")
            if origin is None:
                return False
            ox, oy, ring, grab, eye, target, sx, sy = origin
            image = _preview_state.get("image")
            if image is None:
                return False
            h, w = image.shape[:2]
            cx, cy = ring["cx"], ring["cy"]
            disp_ox, disp_oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
            rcx = disp_ox + cx * sx
            rcy = disp_oy + cy * sy
            curr_dist = math.hypot(x - rcx, y - rcy) / (0.5 * (sx + sy))
            profile = ring.get("profile")
            profile = _ring_bulge_profile(profile, ring["r"], grab, curr_dist, 8.0, 0.5 * math.hypot(w, h))
            _override_of(target)[eye] = _ring_dict(cx, cy, ring["r"], profile)
            _draw_detection_overlay()
            _refresh_unwrap()
            return True

        def _ring_extend_end() -> bool:
            if _ring_extend.get("origin") is not None:
                _ring_extend["origin"] = None
                return True
            return False

        def _active_eye() -> str:
            sel = preview_eye_var.get()
            if sel in ("OD", "OS"):
                return sel
            if pentacam_od.get().strip().strip('"').strip("'"):
                return "OD"
            if pentacam_os.get().strip().strip('"').strip("'"):
                return "OS"
            return self._selected_eye_var.get() if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() in ("OD", "OS") else "OD"

        def _eye_path(eye: str) -> str:
            val = {
                "OD": pentacam_od.get(),
                "OS": pentacam_os.get(),
            }.get(eye, "")
            return val.strip().strip('"').strip("'")

        def _update_eye_btn_styles() -> None:
            active = _active_eye()
            for eye, btn in eye_btn_refs.items():
                has_image = bool(_eye_path(eye))
                is_active = eye == active
                btn.config(
                    state=tk.NORMAL,
                    bg=DLG_ACCENT if is_active else DLG_BTN_SEC,
                    fg="#FFFFFF" if is_active else (DLG_FG if has_image else DLG_FG_HINT),
                    activebackground=DLG_ACCENT if is_active else DLG_BTN_ACT,
                    activeforeground="#FFFFFF" if is_active else DLG_FG,
                )
            _update_roi_btn_style()

        def _select_preview_eye(eye: str) -> None:
            preview_eye_var.set(eye)
            _preview_state["image"] = None
            _preview_state["result"] = result_by_eye.get(eye)
            _preview_state["reliable"] = reliable_by_eye.get(eye, True)
            _preview_state["edge_limbus"] = edge_by_eye.get(eye)
            _preview_state["edge_note"] = edge_note_by_eye.get(eye, "")
            _preview_state["unwrap_strip"] = unwrap_strip_by_eye.get(eye)
            preview_canvas.delete("detection")
            preview_canvas.delete("roi")
            self._set_unwrap_strip(unwrap_strip_by_eye.get(eye))
            _refresh_preview()

        for _eye in ("OD", "OS"):
            _eye_btn = tk.Button(
                eye_row, text=_eye, font=("Segoe UI", 9, "bold"),
                relief=tk.FLAT, bd=0, padx=12, pady=2, cursor="hand2",
                command=lambda current=_eye: _select_preview_eye(current),
            )
            _eye_btn.pack(side=tk.LEFT, padx=2)
            eye_btn_refs[_eye] = _eye_btn

        def _update_roi_btn_style() -> None:
            eye = _active_eye()
            has_image = bool(_eye_path(eye))
            is_active = roi_active_by_eye.get(eye, False) and has_image
            roi_mode_var.set(is_active)
            roi_btn.config(
                state=tk.NORMAL if has_image else tk.DISABLED,
                bg=DLG_ACCENT if is_active else DLG_BTN_SEC,
                fg="#FFFFFF" if is_active else (DLG_FG if has_image else DLG_FG_HINT),
                activebackground=DLG_ACCENT if is_active else DLG_BTN_ACT,
                activeforeground="#FFFFFF" if is_active else DLG_FG,
            )

        def _toggle_roi_mode() -> None:
            eye = _active_eye()
            if not _eye_path(eye):
                return
            image = _preview_state.get("image")
            if image is None:
                return
            h, w = image.shape[:2]

            # If ROI mode is currently active:
            # Second click: clear green rectangle and handles, clear detection overlays,
            # invalidate temporary ROI/detection state, and return button to default grey.
            if roi_active_by_eye.get(eye, False):
                roi_active_by_eye[eye] = False
                roi_locked_by_eye[eye] = False
                roi_locked_var.set(False)
                roi_mode_var.set(False)
                roi_by_eye[eye] = None
                result_by_eye[eye] = None
                _preview_state["result"] = None
                _preview_state["edge_limbus"] = None
                _preview_state["edge_note"] = ""
                preview_canvas.delete("roi")
                preview_canvas.delete("detection")
                _update_roi_btn_style()
                _update_roi_status()
                _refresh_unwrap()
                _update_save_btn_state()
                active_path = _eye_path(eye)
                flip_tag = " (flipped)" if flip_var.get() else ""
                preview_label_text.set(f"{eye}{flip_tag}: {active_path}  \u2014  Click ROI to select region")
                return

            # First click: activate ROI mode, display fixed green rectangle, turn button blue/selected
            roi_active_by_eye[eye] = True
            roi_locked_by_eye[eye] = False
            roi_locked_var.set(False)
            roi_mode_var.set(True)
            roi_by_eye[eye] = _default_fixed_roi(w, h)
            _update_roi_btn_style()
            _draw_roi_overlay()
            _update_roi_status()
            _update_save_btn_state()
            active_path = _eye_path(eye)
            flip_tag = " (flipped)" if flip_var.get() else ""
            preview_label_text.set(f"{eye}{flip_tag}: Position ROI, press Enter to detect")
            try:
                preview_canvas.focus_set()
            except Exception:
                pass

        roi_btn = tk.Button(
            eye_row, text="ROI", font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT, bd=0, padx=12, pady=2, cursor="hand2",
            command=_toggle_roi_mode,
        )
        roi_btn.pack(side=tk.LEFT, padx=(8, 2))
        roi_btn._roi_mode_var = roi_mode_var
        preview_win._roi_btn = roi_btn

        def _update_manual_btn_style() -> None:
            eye = _active_eye()
            has_image = bool(_eye_path(eye))
            is_active = bool(manual_mode_by_eye.get(eye, False)) and has_image
            manual_btn.config(
                state=tk.NORMAL if has_image else tk.DISABLED,
                bg=DLG_ACCENT if is_active else DLG_BTN_SEC,
                fg="#FFFFFF" if is_active else (DLG_FG if has_image else DLG_FG_HINT),
                activebackground=DLG_ACCENT if is_active else DLG_BTN_ACT,
                activeforeground="#FFFFFF" if is_active else DLG_FG,
            )

        def _toggle_manual_mode() -> None:
            eye = _active_eye()
            if not _eye_path(eye):
                return
            image = _preview_state.get("image")
            if image is None:
                return
            h, w = image.shape[:2]

            # If Manual Mode is currently active, clicking again locks it
            if manual_mode_by_eye.get(eye, False):
                manual_locked_by_eye[eye] = True
                manual_mode_by_eye[eye] = False
                _update_manual_btn_style()
                _draw_detection_overlay()
                _update_save_btn_state()
                preview_label_text.set(
                    f"{eye}: Manual boundaries locked. Click Manual Mode to edit again."
                )
                return

            # Activate Manual Mode (or unlock existing boundaries without rerunning detection)
            manual_mode_by_eye[eye] = True
            manual_locked_by_eye[eye] = False

            # If manual boundaries do not exist yet for this eye:
            if ring_override.get(eye) is None or pupil_override.get(eye) is None:
                auto_res = result_by_eye.get(eye)
                if (
                    auto_res is not None
                    and getattr(getattr(auto_res, "pupil", None), "detected", False)
                    and getattr(getattr(auto_res, "limbus", None), "detected", False)
                ):
                    # Option B: Seed directly from automatic detection result
                    if pupil_override.get(eye) is None:
                        pupil_override[eye] = _ellipse_to_dict(auto_res.pupil.ellipse)
                    if ring_override.get(eye) is None:
                        ring_override[eye] = _ellipse_to_dict(auto_res.limbus.ellipse)
                else:
                    # Option A: Direct manual creation inside ROI or centered in image
                    roi = roi_by_eye.get(eye)
                    if roi is not None:
                        ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
                        cx = float(ix0 + ix1) * 0.5
                        cy = float(iy0 + iy1) * 0.5
                        p_rad = max(12.0, min(iw, ih) * 0.18)
                        l_rad = max(30.0, min(iw, ih) * 0.38)
                    else:
                        cx = float(w) * 0.5
                        cy = float(h) * 0.5
                        p_rad = max(12.0, min(w, h) * 0.12)
                        l_rad = max(30.0, min(w, h) * 0.28)
                    if pupil_override.get(eye) is None:
                        pupil_override[eye] = {
                            "cx": cx, "cy": cy, "r": p_rad, "rx": p_rad, "ry": p_rad, "angle_deg": 0.0
                        }
                    if ring_override.get(eye) is None:
                        ring_override[eye] = {
                            "cx": cx, "cy": cy, "r": l_rad, "rx": l_rad, "ry": l_rad, "angle_deg": 0.0
                        }

            _update_manual_btn_style()
            _sync_ring_btn()
            _draw_detection_overlay()
            _draw_roi_overlay()
            _refresh_unwrap()
            _update_save_btn_state()
            target = _ring_target()
            preview_label_text.set(
                f"{eye} Manual Mode: adjusting {target}. Drag center to move, outline to resize. Press Enter to lock."
            )
            try:
                preview_canvas.focus_set()
            except Exception:
                pass

        # ── Controls toolbar below preview canvas (split into 2 rows to avoid overflow) ──
        roi_info_row = tk.Frame(preview_frame, bg=DLG_BG)
        roi_info_row.pack(fill=tk.X, pady=(5, 2))

        roi_info_row_2 = tk.Frame(preview_frame, bg=DLG_BG)
        roi_info_row_2.pack(fill=tk.X, pady=(2, 5))

        detect_btn = tk.Button(
            roi_info_row, text="Lock & Detect", font=("Segoe UI", 9, "bold"),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        detect_btn.pack(side=tk.LEFT, padx=(0, 6))

        clear_roi_btn = tk.Button(
            roi_info_row, text="Clear ROI", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        clear_roi_btn.pack(side=tk.LEFT, padx=(0, 6))

        reset_roi_btn = tk.Button(
            roi_info_row, text="Reset ROI", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        reset_roi_btn.pack(side=tk.LEFT, padx=(0, 6))

        manual_btn = tk.Button(
            roi_info_row, text="Manual Mode", font=("Segoe UI", 9, "bold"),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            command=_toggle_manual_mode,
        )
        manual_btn.pack(side=tk.LEFT, padx=(0, 6))

        # Row 2: Ring adjustments and fine-tuning
        ring_target_row = tk.Frame(roi_info_row_2, bg=DLG_BG)
        ring_target_row.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            ring_target_row, text="Adjust:", font=("Segoe UI", 9),
            bg=DLG_BG, fg=DLG_FG_HINT,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ring_target_refs: Dict[str, tk.Button] = {}
        for _target, _label in (("limbus", "Limbus"), ("pupil", "Pupil")):
            _target_btn = tk.Button(
                ring_target_row, text=_label, font=("Segoe UI", 9),
                bg=DLG_BTN_SEC, fg=DLG_FG,
                activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
                relief=tk.FLAT, bd=0,
                highlightthickness=1,
                highlightbackground=DLG_BORDER,
                highlightcolor=DLG_ACCENT,
                padx=8, pady=3, cursor="hand2",
                command=lambda t=_target: _select_ring_target(t),
            )
            _target_btn.pack(side=tk.LEFT, padx=(0 if _target == "limbus" else 3, 0))
            ring_target_refs[_target] = _target_btn

        ring_minus_btn = tk.Button(
            roi_info_row_2, text="- Ring", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        ring_minus_btn.pack(side=tk.LEFT, padx=(0, 4))

        ring_plus_btn = tk.Button(
            roi_info_row_2, text="+ Ring", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        ring_plus_btn.pack(side=tk.LEFT, padx=(0, 6))

        reset_ring_btn = tk.Button(
            roi_info_row_2, text="Reset ring", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
            state=tk.DISABLED,
        )
        reset_ring_btn.pack(side=tk.LEFT, padx=(0, 6))

        def _update_save_btn_state() -> None:
            eye = _active_eye()
            has_image = bool(_eye_path(eye))
            pid = patient_id.get().strip()
            pname = patient_name.get().strip()

            m_pupil = pupil_override.get(eye)
            m_limbus = ring_override.get(eye)
            auto_res = result_by_eye.get(eye)

            p_ok = (m_pupil is not None) or (
                auto_res is not None
                and bool(getattr(getattr(auto_res, "pupil", None), "detected", False))
                and getattr(auto_res.pupil, "ellipse", None) is not None
            )
            l_ok = (m_limbus is not None) or (
                auto_res is not None
                and bool(getattr(getattr(auto_res, "limbus", None), "detected", False))
                and getattr(auto_res.limbus, "ellipse", None) is not None
            )

            ready = bool(pid and pname and has_image and p_ok and l_ok)
            try:
                save_eye_btn.config(
                    text=f"Save {eye}" if eye else "Save",
                    state=tk.NORMAL if ready else tk.DISABLED,
                    bg=DLG_ACCENT if ready else DLG_BTN_SEC,
                    fg="#FFFFFF" if ready else DLG_FG_HINT,
                )
            except Exception:
                pass

        def _save_eye_data(
            eye: str,
            patient_dir: Path,
            pid: str,
            pname: str,
            prompt_errors: bool = True,
        ) -> bool:
            """Save complete Pentacam data (original, flipped, unwrap strip, results JSON, values.txt) for eye."""
            img_path = _eye_path(eye)
            if not img_path:
                return True
            clean_path = img_path.strip().strip('"').strip("'")
            if not clean_path:
                return True
            src_file = Path(clean_path)
            if not src_file.is_file():
                if prompt_errors:
                    messagebox.showerror(
                        "File Not Found",
                        f"Source image file does not exist:\n{src_file}",
                    )
                return False

            eye_dir = patient_dir / eye / "Pentacam"
            eye_dir.mkdir(parents=True, exist_ok=True)

            img = None
            if _active_eye() == eye and _preview_state.get("image") is not None:
                img = _preview_state.get("image")
            else:
                img = load_pentacam_image(str(src_file))
                if img is not None and flip_var.get():
                    img = flip_pentacam(img)

            if img is None:
                if prompt_errors:
                    messagebox.showerror(
                        "Image Error",
                        f"Cannot load Pentacam image for {eye}:\n{src_file}",
                    )
                return False

            m_pupil = pupil_override.get(eye)
            m_limbus = ring_override.get(eye)
            auto_res = result_by_eye.get(eye)

            p_geom = None
            p_is_manual = False
            if m_pupil is not None:
                p_geom = m_pupil
                p_is_manual = True
            elif auto_res is not None and getattr(getattr(auto_res, "pupil", None), "detected", False):
                pe = auto_res.pupil.ellipse
                if pe is not None:
                    p_rx = float(pe.semi_major)
                    p_ry = float(pe.semi_minor)
                    p_geom = {
                        "cx": float(pe.center_x),
                        "cy": float(pe.center_y),
                        "rx": p_rx,
                        "ry": p_ry,
                        "r": float(getattr(pe, "radius", 0.5 * (p_rx + p_ry))),
                        "angle_deg": float(pe.angle_deg),
                    }
                    p_is_manual = False

            l_geom = None
            l_is_manual = False
            if m_limbus is not None:
                l_geom = m_limbus
                l_is_manual = True
            elif auto_res is not None and getattr(getattr(auto_res, "limbus", None), "detected", False):
                le = auto_res.limbus.ellipse
                if le is not None:
                    l_rx = float(le.semi_major)
                    l_ry = float(le.semi_minor)
                    l_geom = {
                        "cx": float(le.center_x),
                        "cy": float(le.center_y),
                        "rx": l_rx,
                        "ry": l_ry,
                        "r": float(getattr(le, "radius", 0.5 * (l_rx + l_ry))),
                        "angle_deg": float(le.angle_deg),
                    }
                    l_is_manual = False

            # If boundaries not yet established, run detector on demand:
            if p_geom is None or l_geom is None:
                try:
                    if not hasattr(self, "_pentacam_detector") or self._pentacam_detector is None:
                        from pupil_tracking.cyclotorsion.pentacam_detector import PentacamDetector
                        self._pentacam_detector = PentacamDetector()
                    on_demand_res = self._pentacam_detector.detect(
                        img,
                        roi=roi_by_eye.get(eye),
                        eye=eye,
                        source=f"pentacam:{src_file}",
                    )
                    if p_geom is None and getattr(getattr(on_demand_res, "pupil", None), "detected", False):
                        pe = on_demand_res.pupil.ellipse
                        if pe is not None:
                            p_rx = float(pe.semi_major)
                            p_ry = float(pe.semi_minor)
                            p_geom = {
                                "cx": float(pe.center_x),
                                "cy": float(pe.center_y),
                                "rx": p_rx,
                                "ry": p_ry,
                                "r": float(getattr(pe, "radius", 0.5 * (p_rx + p_ry))),
                                "angle_deg": float(pe.angle_deg),
                            }
                    if l_geom is None and getattr(getattr(on_demand_res, "limbus", None), "detected", False):
                        le = on_demand_res.limbus.ellipse
                        if le is not None:
                            l_rx = float(le.semi_major)
                            l_ry = float(le.semi_minor)
                            l_geom = {
                                "cx": float(le.center_x),
                                "cy": float(le.center_y),
                                "rx": l_rx,
                                "ry": l_ry,
                                "r": float(getattr(le, "radius", 0.5 * (l_rx + l_ry))),
                                "angle_deg": float(le.angle_deg),
                            }
                except Exception as exc:
                    self.logger.warning("On-demand detection during save failed for %s: %s", eye, exc)

            h, w = img.shape[:2]
            if p_geom is None:
                cx, cy = w * 0.5, h * 0.5
                pr = min(w, h) * 0.095
                p_geom = {"cx": cx, "cy": cy, "rx": pr, "ry": pr, "r": pr, "angle_deg": 0.0}
            if l_geom is None:
                cx, cy = w * 0.5, h * 0.5
                lr = min(w, h) * 0.25
                l_geom = {"cx": cx, "cy": cy, "rx": lr, "ry": lr, "r": lr, "angle_deg": 0.0}

            # Anatomical plausibility validation:
            # 1. Limbus must be substantially larger than pupil (limbus radius >= pupil radius * 1.3)
            # 2. Pupil center must be inside limbus (center offset + pupil radius < limbus radius)
            # 3. Limbus radius must be at least 25px
            p_r_val = float(p_geom.get("r", 0.5 * (float(p_geom.get("rx", 0)) + float(p_geom.get("ry", 0)))))
            l_r_val = float(l_geom.get("r", 0.5 * (float(l_geom.get("rx", 0)) + float(l_geom.get("ry", 0)))))
            c_offset = math.hypot(float(p_geom["cx"]) - float(l_geom["cx"]), float(p_geom["cy"]) - float(l_geom["cy"]))
            is_valid_combo = (
                l_r_val >= p_r_val * 1.3
                and (c_offset + p_r_val) < l_r_val * 1.05
                and l_r_val >= 25.0
            )
            if not is_valid_combo and not (p_is_manual and l_is_manual):
                if p_is_manual:
                    cx, cy = float(p_geom["cx"]), float(p_geom["cy"])
                    lr = max(p_r_val * 2.2, min(w, h) * 0.25)
                    l_geom = {"cx": cx, "cy": cy, "rx": lr, "ry": lr, "r": lr, "angle_deg": 0.0}
                elif l_is_manual:
                    cx, cy = float(l_geom["cx"]), float(l_geom["cy"])
                    pr = max(8.0, l_r_val * 0.38)
                    p_geom = {"cx": cx, "cy": cy, "rx": pr, "ry": pr, "r": pr, "angle_deg": 0.0}
                else:
                    cx, cy = w * 0.5, h * 0.5
                    pr = min(w, h) * 0.095
                    lr = min(w, h) * 0.25
                    p_geom = {"cx": cx, "cy": cy, "rx": pr, "ry": pr, "r": pr, "angle_deg": 0.0}
                    l_geom = {"cx": cx, "cy": cy, "rx": lr, "ry": lr, "r": lr, "angle_deg": 0.0}

            orig_suffix = src_file.suffix or ".bmp"
            shutil.copy2(src_file, eye_dir / src_file.name)
            dest_orig = eye_dir / f"Pentacam_original{orig_suffix}"
            shutil.copy2(src_file, dest_orig)

            dest_flipped = eye_dir / f"Pentacam_{eye}_flipped.png"
            cv2.imwrite(str(dest_flipped), img)

            roi = roi_by_eye.get(eye)
            roi_info = None
            if roi is not None:
                rx0, ry0, rx1, ry1 = (float(v) for v in roi)
                ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
                roi_info = {
                    "norm": [rx0, ry0, rx1, ry1],
                    "pixels": [ix0, iy0, ix1, iy1],
                    "width": iw,
                    "height": ih,
                }

            p_rx = float(p_geom.get("rx", p_geom.get("r", 0.0)))
            p_ry = float(p_geom.get("ry", p_geom.get("r", 0.0)))
            l_rx = float(l_geom.get("rx", l_geom.get("r", 0.0)))
            l_ry = float(l_geom.get("ry", l_geom.get("r", 0.0)))

            pe = PupilTrackingGUI._as_ellipse_params(
                SimpleNamespace(
                    center_x=float(p_geom["cx"]),
                    center_y=float(p_geom["cy"]),
                    semi_major=p_rx,
                    semi_minor=p_ry,
                    radius=float(p_geom.get("r", 0.5 * (p_rx + p_ry))),
                    angle_deg=float(p_geom.get("angle_deg", 0.0)),
                )
            )
            le = PupilTrackingGUI._as_ellipse_params(
                SimpleNamespace(
                    center_x=float(l_geom["cx"]),
                    center_y=float(l_geom["cy"]),
                    semi_major=l_rx,
                    semi_minor=l_ry,
                    radius=float(l_geom.get("r", 0.5 * (l_rx + l_ry))),
                    angle_deg=float(l_geom.get("angle_deg", 0.0)),
                )
            )
            strip = self._unwrap_pentacam(img, limbus_ellipse=le, pupil_ellipse=pe, eye=eye)
            if strip is None or getattr(strip, "size", 0) == 0:
                # Guaranteed fallback: concentric default boundaries
                cx, cy = w * 0.5, h * 0.5
                pr = min(w, h) * 0.095
                lr = min(w, h) * 0.25
                p_geom = {"cx": cx, "cy": cy, "rx": pr, "ry": pr, "r": pr, "angle_deg": 0.0}
                l_geom = {"cx": cx, "cy": cy, "rx": lr, "ry": lr, "r": lr, "angle_deg": 0.0}
                pe_fb = PupilTrackingGUI._as_ellipse_params(
                    SimpleNamespace(center_x=cx, center_y=cy, semi_major=pr, semi_minor=pr, radius=pr, angle_deg=0.0)
                )
                le_fb = PupilTrackingGUI._as_ellipse_params(
                    SimpleNamespace(center_x=cx, center_y=cy, semi_major=lr, semi_minor=lr, radius=lr, angle_deg=0.0)
                )
                strip = self._unwrap_pentacam(img, limbus_ellipse=le_fb, pupil_ellipse=pe_fb, eye=eye)

            unwrapped_saved = False
            if strip is not None and getattr(strip, "size", 0) > 0:
                unwrap_strip_by_eye[eye] = strip
                dest_strip = eye_dir / f"Pentacam_{eye}_unwrapped.png"
                cv2.imwrite(str(dest_strip), strip)
                unwrapped_saved = dest_strip.is_file() and dest_strip.stat().st_size > 0

            results_meta = {
                "patient_id": pid,
                "patient_name": pname,
                "eye": eye,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_image": {
                    "path": str(src_file.resolve()),
                    "width": w,
                    "height": h,
                    "is_flipped": bool(flip_var.get()),
                },
                "roi": roi_info,
                "pupil": {
                    "center_x": round(float(p_geom["cx"]), 3),
                    "center_y": round(float(p_geom["cy"]), 3),
                    "semi_major": round(p_rx, 3),
                    "semi_minor": round(p_ry, 3),
                    "radius": round(0.5 * (p_rx + p_ry), 3),
                    "angle_deg": round(float(p_geom.get("angle_deg", 0.0)), 2),
                    "is_manual": p_is_manual,
                },
                "limbus": {
                    "center_x": round(float(l_geom["cx"]), 3),
                    "center_y": round(float(l_geom["cy"]), 3),
                    "semi_major": round(l_rx, 3),
                    "semi_minor": round(l_ry, 3),
                    "radius": round(0.5 * (l_rx + l_ry), 3),
                    "angle_deg": round(float(l_geom.get("angle_deg", 0.0)), 2),
                    "is_manual": l_is_manual,
                },
                "unwrapped_generated": unwrapped_saved,
            }
            dest_json = eye_dir / "pentacam_results.json"
            dest_json.write_text(json.dumps(results_meta, indent=2), encoding="utf-8")

            value_lines = [
                f"# Pentacam Values - {eye}",
                f"patient_id={pid}",
                f"hvid={hvid_od.get().strip() if eye == 'OD' else hvid_os.get().strip()}",
                f"image_flipped={'true' if flip_var.get() else 'false'}",
                f"pupil_center={p_geom['cx']:.2f},{p_geom['cy']:.2f}",
                f"pupil_radius={0.5 * (p_rx + p_ry):.2f}",
                f"pupil_manual={'true' if p_is_manual else 'false'}",
                f"limbus_center={l_geom['cx']:.2f},{l_geom['cy']:.2f}",
                f"limbus_radius={0.5 * (l_rx + l_ry):.2f}",
                f"limbus_manual={'true' if l_is_manual else 'false'}",
            ]
            if roi_info:
                value_lines.append(
                    f"iris_roi_norm={roi_info['norm'][0]:.6f},{roi_info['norm'][1]:.6f},"
                    f"{roi_info['norm'][2]:.6f},{roi_info['norm'][3]:.6f}"
                )
            (eye_dir / "pentacam_values.txt").write_text(
                "\n".join(value_lines) + "\n", encoding="utf-8"
            )
            return True

        def _save_current_eye() -> None:
            """Save only the currently active eye's Pentacam boundaries, flipped image and unwrap strip."""
            eye = _active_eye()
            if not eye:
                messagebox.showerror("No Eye Selected", "Please select an eye (OD or OS) before saving.")
                return
            pid = patient_id.get().strip()
            pname = patient_name.get().strip()
            if not pid or not pname:
                messagebox.showerror("Missing Information", "Patient ID and Patient Name are required before saving.")
                return
            root = Path(self._storage_root_var.get())
            patient_dir = (
                root
                / f"PX_{self._safe_folder_name(pid)}"
                  f"_{self._safe_folder_name(pname)}"
            )
            patient_dir.mkdir(parents=True, exist_ok=True)
            (patient_dir / "patient.txt").write_text(
                "# Patient Record - Medevplus IXcentai\n"
                f"patient_id={pid}\n"
                f"patient_name={pname}\n"
                f"hvid_od={hvid_od.get().strip()}\n"
                f"hvid_os={hvid_os.get().strip()}\n"
                f"created_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
            if not _save_eye_data(eye, patient_dir, pid, pname, prompt_errors=True):
                return

            self._active_patient_dir = patient_dir
            self._patient_id_var.set(pid)
            self._patient_name_var.set(pname)
            if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() == eye:
                self._set_unwrap_strip(unwrap_strip_by_eye.get(eye))
            self._update_patient_controls()
            other_eye = "OS" if eye == "OD" else "OD"
            messagebox.showinfo(
                "Save Successful",
                f"Pentacam results for {eye} have been saved successfully to:\n{patient_dir / eye / 'Pentacam'}\n\n"
                f"You may now proceed to configure or save the {other_eye} eye."
            )
            self.logger.info("Saved %s Pentacam data to %s", eye, patient_dir / eye / "Pentacam")

        save_eye_btn = tk.Button(
            roi_info_row, text="Save OD", font=("Segoe UI", 9, "bold"),
            bg=DLG_BTN_SEC, fg=DLG_FG_HINT,
            activebackground=DLG_ACCENT, activeforeground="#FFFFFF",
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=12, pady=3, cursor="hand2",
            state=tk.DISABLED,
            command=_save_current_eye,
        )
        save_eye_btn.pack(side=tk.RIGHT, padx=0)

        cyc_compute_btn = tk.Button(
            roi_info_row_2, text="Compute torsion", font=("Segoe UI", 9),
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0,
            highlightthickness=1,
            highlightbackground=DLG_BORDER,
            highlightcolor=DLG_ACCENT,
            padx=10, pady=3, cursor="hand2",
        )
        cyc_compute_btn.pack(side=tk.RIGHT, padx=0)

        # ── Status and Cyclotorsion Readout Row ─────────────────────
        status_row = tk.Frame(preview_frame, bg=DLG_BG)
        status_row.pack(fill=tk.X, pady=(5, 2))

        cyc_readout_var = tk.StringVar(value="Cyclotorsion  OD: \u2014   OS: \u2014")
        cyc_label = tk.Label(
            status_row, textvariable=cyc_readout_var,
            font=("Segoe UI", 9, "bold"), bg=DLG_BG, fg=DLG_ACCENT,
        )
        cyc_label.pack(side=tk.RIGHT, padx=0)

        roi_status_var = tk.StringVar(value="")
        roi_status_lbl = tk.Label(
            status_row, textvariable=roi_status_var,
            font=("Segoe UI", 9, "bold"), bg=DLG_BG, fg=DLG_ACCENT,
            bd=0, padx=0,
        )

        preview_status_lbl = tk.Label(
            status_row, textvariable=preview_label_text,
            font=("Segoe UI", 9), bg=DLG_BG, fg=DLG_FG_HINT,
            anchor=tk.W, justify=tk.LEFT,
            bd=0, padx=0,
        )
        preview_status_lbl.pack(side=tk.LEFT, padx=0, fill=tk.X, expand=True)

        def _sync_status_lbl_visibility(*_):
            if roi_status_var.get().strip():
                roi_status_lbl.pack(side=tk.LEFT, padx=(0, 8), before=preview_status_lbl)
            else:
                roi_status_lbl.pack_forget()

        roi_status_var.trace_add("write", _sync_status_lbl_visibility)
        _sync_status_lbl_visibility()

        cyc_compute_btn._cyc_compute = True
        cyc_label._cyc_readout_var = cyc_readout_var

        def _update_roi_status() -> None:
            eye = _active_eye()
            roi = roi_by_eye.get(eye)
            image = _preview_state.get("image")
            is_locked = bool(roi_locked_by_eye.get(eye, False))
            lock_label = " [LOCKED]" if is_locked else ""
            if roi is None:
                roi_status_var.set("Iris ROI: not set")
                clear_roi_btn.config(state=tk.DISABLED)
            else:
                x0, y0, x1, y1 = roi
                if image is not None:
                    h, w = image.shape[:2]
                    ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
                    roi_status_var.set(
                        f"Iris ROI {eye} | X: {ix0}, Y: {iy0} | Width: {iw} px | Height: {ih} px{lock_label}"
                    )
                else:
                    roi_status_var.set(
                        f"Iris ROI ({eye}): {(x1 - x0):.0%} \u00d7 {(y1 - y0):.0%} of frame{lock_label}"
                    )
                clear_roi_btn.config(state=tk.NORMAL)
            try:
                reset_roi_btn.config(state=tk.NORMAL if image is not None else tk.DISABLED)
                detect_btn.config(state=tk.NORMAL if image is not None else tk.DISABLED)
            except Exception:
                pass

            # Any ROI change invalidates the (display-only) cyclotorsion readout.
            cyc_readout_var.set("Cyclotorsion  OD: \u2014   OS: \u2014")
            _sync_ring_btn()

        def _draw_roi_overlay() -> None:
            preview_canvas.delete("roi")
            eye = _active_eye()
            if not eye or not roi_active_by_eye.get(eye, False):
                return
            roi = roi_by_eye.get(eye)
            tw, th = preview_canvas._thumb_size
            if roi is None or tw <= 0 or th <= 0:
                return
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
            x0, y0, x1, y1 = roi
            cx0 = ox + x0 * tw
            cy0 = oy + y0 * th
            cx1 = ox + x1 * tw
            cy1 = oy + y1 * th

            is_locked = bool(roi_locked_by_eye.get(eye, False))
            # Main bounding rectangle (solid green)
            rect_opts = {"outline": "#00E676", "width": 2 if not is_locked else 3, "tags": "roi"}
            preview_canvas.create_rectangle(cx0, cy0, cx1, cy1, **rect_opts)

            # 8 Interactive handles rendered ONLY while ROI mode is active and UNLOCKED
            if not is_locked:
                handles = rect_handle_positions(x0, y0, x1, y1, tw, th, offset_x=ox, offset_y=oy)
                hs = 3.5
                for _hname, (hx, hy) in handles.items():
                    preview_canvas.create_rectangle(
                        hx - hs, hy - hs, hx + hs, hy + hs,
                        outline="#00E676", fill="#FFFFFF", width=1, tags="roi",
                    )

            # Telemetry badge on canvas
            image = _preview_state.get("image")
            if image is not None:
                h, w = image.shape[:2]
                ix0, iy0, ix1, iy1, iw, ih = roi_norm_to_image_coords(roi, w, h)
                lock_tag = " [LOCKED]" if is_locked else ""
                badge_text = f"ROI: {iw} \u00d7 {ih} px (X:{ix0}, Y:{iy0}){lock_tag}"
                badge_y = cy0 - 12.0 if cy0 >= (oy + 16.0) else cy0 + 12.0
                badge_x = max(cx0, float(ox))
                bw = len(badge_text) * 6.5 + 10.0
                preview_canvas.create_rectangle(
                    badge_x, badge_y - 8.0, badge_x + bw, badge_y + 8.0,
                    fill="#101010", outline="#00E676", width=1, tags="roi",
                )
                preview_canvas.create_text(
                    badge_x + 5.0, badge_y, anchor="w",
                    text=badge_text, fill="#00E676",
                    font=("Segoe UI", 8, "bold"), tags="roi",
                )

        def _canvas_to_norm(cx: float, cy: float):
            tw, th = preview_canvas._thumb_size
            if tw <= 0 or th <= 0:
                return None
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
            return (
                min(max((cx - ox) / tw, 0.0), 1.0),
                min(max((cy - oy) / th, 0.0), 1.0),
            )

        def _nudge_roi(dx: int, dy: int, event: Any = None) -> None:
            eye = _active_eye()
            if not roi_mode_var.get() or not eye:
                return
            roi = roi_by_eye.get(eye)
            image = _preview_state.get("image")
            if roi is None or image is None:
                return
            h, w = image.shape[:2]
            is_shift = bool(
                event is not None and (
                    (getattr(event, "state", 0) & 0x0001)
                    or "Shift" in getattr(event, "keysym", "")
                )
            )
            step_px = 10 if is_shift else 1
            roi_by_eye[eye] = nudge_rect(roi, dx * step_px, dy * step_px, w, h)
            # Invalidate stale detection when ROI moves
            result_by_eye[eye] = None
            _preview_state["result"] = None
            _preview_state["edge_limbus"] = None
            _preview_state["edge_note"] = ""
            ring_override[eye] = None
            pupil_override[eye] = None
            preview_canvas.delete("detection")
            _draw_roi_overlay()
            _update_roi_status()
            _refresh_unwrap()

        def _reset_roi() -> None:
            eye = _active_eye()
            image = _preview_state.get("image")
            if not eye or image is None:
                return
            h, w = image.shape[:2]
            roi_by_eye[eye] = _default_fixed_roi(w, h)
            roi_locked_by_eye[eye] = False
            roi_locked_var.set(False)
            roi_mode_var.set(True)
            roi_active_by_eye[eye] = True
            result_by_eye[eye] = None
            _preview_state["result"] = None
            _preview_state["edge_limbus"] = None
            _preview_state["edge_note"] = ""
            ring_override[eye] = None
            pupil_override[eye] = None
            manual_mode_by_eye[eye] = False
            manual_locked_by_eye[eye] = False
            preview_canvas.delete("detection")
            _update_manual_btn_style()
            _update_roi_btn_style()
            _draw_roi_overlay()
            _update_roi_status()
            _refresh_unwrap()
            try:
                preview_canvas.focus_set()
            except Exception:
                pass

        reset_roi_btn.config(command=_reset_roi)
        detect_btn.config(command=lambda: _handle_key_confirm(None))

        # ── Mouse & Keyboard interaction routing for ROI and Manual Mode ─────

        def _on_ring_press(event) -> bool:
            """Initiate manual dragging/resizing for pupil or limbus in Manual Mode."""
            x = event.x if hasattr(event, "x") else float(event[0])
            y = event.y if hasattr(event, "y") else float(event[1])
            eye = _active_eye()
            image = _preview_state.get("image")
            tw, th = getattr(preview_canvas, "_thumb_size", (0, 0))
            if not eye or image is None or tw <= 0 or th <= 0:
                return False
            h, w = image.shape[:2]
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))

            p_ring = pupil_override.get(eye)
            l_ring = ring_override.get(eye)
            if p_ring is None and l_ring is None:
                return False

            def _dist_info(ring):
                if ring is None:
                    return 9999.0, 9999.0
                rcx = ox + float(ring["cx"]) * (tw / float(w))
                rcy = oy + float(ring["cy"]) * (th / float(h))
                rrx = float(ring.get("rx", ring["r"])) * (tw / float(w))
                rry = float(ring.get("ry", ring["r"])) * (th / float(h))
                d_center = math.hypot(x - rcx, y - rcy)
                ang = math.atan2(y - rcy, x - rcx)
                drawn_r = _ring_canvas_radius(ring, tw / float(w), th / float(h), ang)
                d_outline = abs(d_center - drawn_r)
                return d_center, d_outline

            p_dc, p_do = _dist_info(p_ring)
            l_dc, l_do = _dist_info(l_ring)

            # Hit-test: choose which boundary the user clicked (respecting chosen target)
            current_target = _ring_target()
            if current_target == "limbus":
                if l_dc <= 18.0 or l_do <= 18.0:
                    target = "limbus"
                elif p_do <= 14.0 and l_do > 18.0:
                    target = "pupil"
                elif l_dc <= 35.0 or l_do <= 35.0:
                    target = "limbus"
                elif min(p_dc, p_do) <= 30.0:
                    target = "pupil"
                else:
                    return False
            else:
                if p_dc <= 18.0 or p_do <= 16.0:
                    target = "pupil"
                elif l_do <= 16.0 and p_do > 16.0:
                    target = "limbus"
                elif p_dc <= 30.0 or p_do <= 30.0:
                    target = "pupil"
                elif min(l_dc, l_do) <= 35.0:
                    target = "limbus"
                else:
                    return False

            ring_target_var.set(target)
            _update_ring_target_styles()

            ring = _override_of(target).get(eye)
            if ring is None:
                return False

            rcx = ox + float(ring["cx"]) * (tw / float(w))
            rcy = oy + float(ring["cy"]) * (th / float(h))
            rrx = float(ring.get("rx", ring["r"])) * (tw / float(w))
            rry = float(ring.get("ry", ring["r"])) * (th / float(h))
            ang_deg = float(ring.get("angle_deg", 0.0))
            rad = math.radians(ang_deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            d_center = math.hypot(x - rcx, y - rcy)
            mode = "move_center" if d_center <= 16.0 else "resize_both"

            # Check rotation handle (connected to East handle)
            rot_hx = rcx + (rrx + 22.0) * cos_a
            rot_hy = rcy + (rrx + 22.0) * sin_a
            if math.hypot(x - rot_hx, y - rot_hy) <= 14.0:
                mode = "rotate"
            else:
                east_x, east_y = rcx + rrx * cos_a, rcy + rrx * sin_a
                west_x, west_y = rcx - rrx * cos_a, rcy - rrx * sin_a
                north_x, north_y = rcx + rry * sin_a, rcy - rry * cos_a
                south_x, south_y = rcx - rry * sin_a, rcy + rry * cos_a

                if min(math.hypot(x - east_x, y - east_y), math.hypot(x - west_x, y - west_y)) <= 12.0:
                    mode = "resize_x"
                elif min(math.hypot(x - north_x, y - north_y), math.hypot(x - south_x, y - south_y)) <= 12.0:
                    mode = "resize_y"

            _ring_drag["origin"] = (
                x, y, ring["cx"], ring["cy"], ring.get("rx", ring["r"]), ring.get("ry", ring["r"]),
                tw, th, eye, target, mode, ang_deg,
            )
            _draw_detection_overlay()
            return True

        _start_manual_ring_drag = lambda x, y: _on_ring_press(SimpleNamespace(x=x, y=y))

        def _on_ring_drag(event) -> None:
            """Handle manual dragging/resizing of active pupil or limbus boundary."""
            origin = _ring_drag.get("origin")
            if origin is None:
                return
            (
                ox_c, oy_c, orig_cx, orig_cy, orig_rx, orig_ry,
                tw, th, eye, target, mode, orig_ang,
            ) = origin
            image = _preview_state.get("image")
            if image is None:
                return
            h, w = image.shape[:2]
            ring = _override_of(target).get(eye)
            if ring is None:
                return

            scale_x = w / float(tw)
            scale_y = h / float(th)
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))

            if mode == "move_center":
                ring["cx"] = orig_cx + (event.x - ox_c) * scale_x
                ring["cy"] = orig_cy + (event.y - oy_c) * scale_y
            elif mode == "rotate":
                disp_cx = ox + float(ring["cx"]) * (tw / float(w))
                disp_cy = oy + float(ring["cy"]) * (th / float(h))
                angle = math.degrees(math.atan2(event.y - disp_cy, event.x - disp_cx))
                ring["angle_deg"] = round(angle % 360.0, 1)
            elif mode == "resize_x":
                disp_cx = ox + float(ring["cx"]) * (tw / float(w))
                disp_cy = oy + float(ring["cy"]) * (th / float(h))
                rad = math.radians(orig_ang)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                proj = (event.x - disp_cx) * cos_a + (event.y - disp_cy) * sin_a
                dist = abs(proj) * scale_x
                ring["rx"] = max(8.0, dist)
                ring["r"] = 0.5 * (ring["rx"] + ring["ry"])
            elif mode == "resize_y":
                disp_cx = ox + float(ring["cx"]) * (tw / float(w))
                disp_cy = oy + float(ring["cy"]) * (th / float(h))
                rad = math.radians(orig_ang)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                proj = -(event.x - disp_cx) * sin_a + (event.y - disp_cy) * cos_a
                dist = abs(proj) * scale_y
                ring["ry"] = max(8.0, dist)
                ring["r"] = 0.5 * (ring["rx"] + ring["ry"])
            elif mode == "resize_both":
                disp_cx = ox + float(ring["cx"]) * (tw / float(w))
                disp_cy = oy + float(ring["cy"]) * (th / float(h))
                orig_rad = 0.5 * (orig_rx + orig_ry)
                curr_dist = math.hypot(event.x - disp_cx, event.y - disp_cy) * (0.5 * (scale_x + scale_y))
                if orig_rad > 0:
                    ratio = max(0.1, curr_dist / orig_rad)
                    ring["rx"] = max(8.0, orig_rx * ratio)
                    ring["ry"] = max(8.0, orig_ry * ratio)
                    ring["r"] = 0.5 * (ring["rx"] + ring["ry"])

            # Requirement 4: Allow complete boundary to extend beyond ROI!
            ring["rx"] = min(ring["rx"], w * 0.49)
            ring["ry"] = min(ring["ry"], h * 0.49)
            ring["cx"] = max(ring["rx"], min(w - ring["rx"], ring["cx"]))
            ring["cy"] = max(ring["ry"], min(h - ring["ry"], ring["cy"]))

            _draw_detection_overlay()
            _draw_roi_overlay()
            _refresh_unwrap()

        def _on_roi_press(event) -> None:
            try:
                preview_canvas.focus_set()
            except Exception:
                pass

            eye = _active_eye()
            if not eye:
                return

            # In Manual Mode or when manual boundary exists:
            has_ring = bool(_override_of(_ring_target()).get(eye) is not None)
            is_manual = bool(manual_mode_by_eye.get(eye, False) and not manual_locked_by_eye.get(eye, False))
            if is_manual or has_ring:
                if is_manual and _on_ring_press(event):
                    return
                if _start_ring_extend(event.x, event.y):
                    return
                if is_manual and _on_ring_press(event):
                    return
                if is_manual:
                    return

            # In ROI mode (and unlocked): interact solely with green ROI rectangle
            if roi_active_by_eye.get(eye, False) and not roi_locked_by_eye.get(eye, False):
                tw, th = preview_canvas._thumb_size
                roi = roi_by_eye.get(eye)
                image = _preview_state.get("image")
                if image is None or tw <= 0 or th <= 0:
                    return
                h, w = image.shape[:2]
                ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
                pt = _canvas_to_norm(event.x, event.y)
                if pt is None:
                    return

                if roi is None:
                    rw = min(float(w), float(PENTACAM_FIXED_ROI_W))
                    rh = min(float(h), float(PENTACAM_FIXED_ROI_H))
                    norm_w = rw / float(w)
                    norm_h = rh / float(h)
                    nx0 = max(0.0, min(pt[0] - norm_w * 0.5, 1.0 - norm_w))
                    ny0 = max(0.0, min(pt[1] - norm_h * 0.5, 1.0 - norm_h))
                    roi = (nx0, ny0, nx0 + norm_w, ny0 + norm_h)
                    roi_by_eye[eye] = roi
                    _drag_state["mode"] = "move"
                    _drag_state["origin"] = pt
                    _drag_state["orig_roi"] = roi
                    _draw_roi_overlay()
                    _update_roi_status()
                    return

                hit = hit_test_rect(event.x, event.y, roi, tw, th, handle_radius=7.0, edge_tol=6.0, offset_x=ox, offset_y=oy)
                if hit is not None:
                    _drag_state["mode"] = "move"
                    _drag_state["origin"] = pt
                    _drag_state["orig_roi"] = roi
                    return

                norm_w = roi[2] - roi[0]
                norm_h = roi[3] - roi[1]
                nx0 = max(0.0, min(pt[0] - norm_w * 0.5, 1.0 - norm_w))
                ny0 = max(0.0, min(pt[1] - norm_h * 0.5, 1.0 - norm_h))
                roi = (nx0, ny0, nx0 + norm_w, ny0 + norm_h)
                roi_by_eye[eye] = roi
                _drag_state["mode"] = "move"
                _drag_state["origin"] = pt
                _drag_state["orig_roi"] = roi
                _draw_roi_overlay()
                _update_roi_status()
                return

        def _on_roi_drag(event) -> None:
            if _ring_extend_drag(event.x, event.y):
                return
            if _ring_drag.get("origin") is not None:
                _on_ring_drag(event)
                return

            eye = _active_eye()
            if not eye or not roi_active_by_eye.get(eye, False) or roi_locked_by_eye.get(eye, False):
                return

            mode = _drag_state.get("mode")
            origin = _drag_state.get("origin")
            orig_roi = _drag_state.get("orig_roi")
            if mode is None or origin is None or orig_roi is None:
                return

            pt = _canvas_to_norm(event.x, event.y)
            if pt is None:
                return

            roi_by_eye[eye] = update_rect_drag("move", orig_roi, pt, origin)
            if result_by_eye.get(eye) is not None or _preview_state.get("result") is not None:
                result_by_eye[eye] = None
                _preview_state["result"] = None
                _preview_state["edge_limbus"] = None
                _preview_state["edge_note"] = ""
                ring_override[eye] = None
                pupil_override[eye] = None
                preview_canvas.delete("detection")
            _draw_roi_overlay()
            _update_roi_status()

        def _on_roi_release(event) -> None:
            if _ring_extend_end():
                return
            if _ring_drag.get("origin") is not None:
                _ring_drag["origin"] = None
                _update_save_btn_state()
                _draw_detection_overlay()
                _refresh_unwrap()
                return

            _drag_state["origin"] = None
            _drag_state["mode"] = None
            _drag_state["orig_roi"] = None

            _draw_roi_overlay()
            _update_roi_status()
            _refresh_unwrap()
            _update_save_btn_state()

        def _clear_roi() -> None:
            eye = _active_eye()
            if eye:
                roi_by_eye[eye] = None
                roi_active_by_eye[eye] = False
                roi_locked_by_eye[eye] = False
                result_by_eye[eye] = None
                ring_override[eye] = None
                pupil_override[eye] = None
                manual_mode_by_eye[eye] = False
                manual_locked_by_eye[eye] = False
                unwrap_strip_by_eye[eye] = None
            roi_mode_var.set(False)
            roi_locked_var.set(False)
            _preview_state["result"] = None
            _preview_state["edge_limbus"] = None
            _preview_state["edge_note"] = ""
            _preview_state["unwrap_strip"] = None
            self._set_unwrap_strip(None)
            preview_canvas.delete("roi")
            preview_canvas.delete("detection")
            _update_manual_btn_style()
            _update_roi_btn_style()
            _update_roi_status()
            _refresh_unwrap()
            _update_save_btn_state()
            active_path = _eye_path(eye)
            flip_tag = " (flipped)" if flip_var.get() else ""
            preview_label_text.set(f"{eye}{flip_tag}: {active_path}  \u2014  Click ROI to select region")

        clear_roi_btn.config(command=_clear_roi)
        preview_canvas.bind("<ButtonPress-1>", _on_roi_press)
        preview_canvas.bind("<B1-Motion>", _on_roi_drag)
        preview_canvas.bind("<ButtonRelease-1>", _on_roi_release)

        # Arrow key bindings for 1px and Shift+Arrow for 10px movement
        def _handle_key_nudge(dx: int, dy: int, event: Any = None) -> Optional[str]:
            widget = getattr(event, "widget", None)
            if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, tk.Spinbox)):
                return None
            eye = _active_eye()
            if not eye:
                return None
            image = _preview_state.get("image")
            if image is None:
                return None
            h, w = image.shape[:2]

            is_shift = bool(
                event is not None and (
                    (getattr(event, "state", 0) & 0x0001)
                    or "Shift" in getattr(event, "keysym", "")
                )
            )
            step_px = 10 if is_shift else 1

            # In Manual Mode:
            if manual_mode_by_eye.get(eye, False) and not manual_locked_by_eye.get(eye, False):
                target = _ring_target()
                ring = pupil_override.get(eye) if target == "pupil" else ring_override.get(eye)
                if ring is not None:
                    ring["cx"] += dx * step_px
                    ring["cy"] += dy * step_px
                    ring["cx"] = max(ring.get("rx", ring["r"]), min(w - ring.get("rx", ring["r"]), ring["cx"]))
                    ring["cy"] = max(ring.get("ry", ring["r"]), min(h - ring.get("ry", ring["r"]), ring["cy"]))
                    _draw_detection_overlay()
                    _refresh_unwrap()
                    _update_save_btn_state()
                    return "break"

            # In ROI mode:
            if roi_active_by_eye.get(eye, False) and not roi_locked_by_eye.get(eye, False):
                _nudge_roi(dx, dy, event)
                return "break"

            return None

        def _handle_key_confirm(event: Any = None) -> Optional[str]:
            widget = getattr(event, "widget", None)
            if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, tk.Spinbox)):
                return None
            eye = _active_eye()
            image = _preview_state.get("image")
            if not eye or image is None:
                return None

            # If user is in Manual Mode and pressed Enter: Enter finalizes and locks manual boundaries
            if event is not None and manual_mode_by_eye.get(eye, False):
                manual_locked_by_eye[eye] = True
                manual_mode_by_eye[eye] = False
                _update_manual_btn_style()
                _draw_detection_overlay()
                _update_save_btn_state()
                _refresh_unwrap()
                p_r = pupil_override.get(eye, {}).get("r", 0.0)
                l_r = ring_override.get(eye, {}).get("r", 0.0)
                preview_label_text.set(
                    f"{eye}: Manual boundaries locked (pupil r={p_r:.1f}px, limbus r={l_r:.1f}px). Click Manual Mode to edit again."
                )
                return "break"

            # Lock & Detect (via button click or Enter key):
            h, w = image.shape[:2]
            if roi_by_eye.get(eye) is None:
                roi_by_eye[eye] = _default_fixed_roi(w, h)
            roi_active_by_eye[eye] = True
            roi_locked_by_eye[eye] = True
            roi_locked_var.set(True)
            manual_mode_by_eye[eye] = False
            manual_locked_by_eye[eye] = False
            _update_manual_btn_style()
            _update_roi_btn_style()

            # Clear stale manual overrides and detection so fresh detection runs
            ring_override[eye] = None
            pupil_override[eye] = None
            result_by_eye[eye] = None
            _preview_state["result"] = None
            _preview_state["edge_limbus"] = None
            _preview_state["edge_note"] = ""
            preview_canvas.delete("detection")

            _draw_roi_overlay()
            _update_roi_status()
            _run_preview_detection(eye)
            return "break"

            return None

        def _handle_key_cancel(event: Any = None) -> Optional[str]:
            widget = getattr(event, "widget", None)
            if isinstance(widget, (tk.Entry, ttk.Entry, tk.Text, tk.Spinbox)):
                return None
            eye = _active_eye()
            if _drag_state.get("mode") is not None:
                orig = _drag_state.get("orig_roi")
                if orig is not None and eye:
                    roi_by_eye[eye] = orig
                _drag_state["mode"] = None
                _drag_state["origin"] = None
                _drag_state["orig_roi"] = None
                _draw_roi_overlay()
                _update_roi_status()
                return "break"
            elif roi_active_by_eye.get(eye, False):
                _clear_roi()
                return "break"
            return None

        for _target in (preview_canvas, preview_win):
            for _key, (_dx, _dy) in {
                "<Left>": (-1, 0),
                "<Right>": (1, 0),
                "<Up>": (0, -1),
                "<Down>": (0, 1),
                "<Shift-Left>": (-1, 0),
                "<Shift-Right>": (1, 0),
                "<Shift-Up>": (0, -1),
                "<Shift-Down>": (0, 1),
            }.items():
                _target.bind(_key, lambda e, dx=_dx, dy=_dy: _handle_key_nudge(dx, dy, e))
            _target.bind("<Return>", _handle_key_confirm)
            _target.bind("<KP_Enter>", _handle_key_confirm)
            _target.bind("<Escape>", _handle_key_cancel)

        # ── Hand adjustment helpers (scale, wheel, hover, target styles) ──

        def _scale_ring(factor: float) -> bool:
            eye = _active_eye()
            if not eye:
                return False
            target = _ring_target()
            ring = _override_of(target).get(eye)
            if ring is None:
                _seed_ring(target)
                ring = _override_of(target).get(eye)
            image = _preview_state.get("image")
            if ring is None or image is None:
                return False
            h, w = image.shape[:2]
            ring["rx"] = max(8.0, ring.get("rx", ring["r"]) * factor)
            ring["ry"] = max(8.0, ring.get("ry", ring["r"]) * factor)
            ring["r"] = 0.5 * (ring["rx"] + ring["ry"])

            prof = ring.get("profile")
            if prof is not None:
                ring["profile"] = prof * factor

            ring["rx"] = min(ring["rx"], w * 0.49)
            ring["ry"] = min(ring["ry"], h * 0.49)
            ring["cx"] = max(ring["rx"], min(w - ring["rx"], ring["cx"]))
            ring["cy"] = max(ring["ry"], min(h - ring["ry"], ring["cy"]))

            _sync_ring_btn()
            _draw_detection_overlay()
            _draw_roi_overlay()
            _refresh_unwrap()
            _update_save_btn_state()
            return True

        def _on_ring_wheel(event) -> None:
            eye = _active_eye()
            if not eye:
                return
            step = 1.0 if getattr(event, "delta", 0) > 0 else -1.0
            factor = 1.0 + 0.05 * step
            target = _ring_target()
            ring = _override_of(target).get(eye)
            if ring is None:
                _seed_ring(target)
                ring = _override_of(target).get(eye)
            if ring is None:
                return

            state = getattr(event, "state", 0)
            is_shift = bool(state & 0x0001)
            is_ctrl = bool(state & 0x0004)
            is_alt = bool(state & 0x0008) or bool(state & 0x20000)

            if is_alt:
                ring["angle_deg"] = (ring.get("angle_deg", 0.0) + 2.0 * step) % 360.0
            elif is_shift:
                ring["rx"] = max(8.0, ring.get("rx", ring["r"]) * factor)
            elif is_ctrl:
                ring["ry"] = max(8.0, ring.get("ry", ring["r"]) * factor)
            else:
                ring["rx"] = max(8.0, ring.get("rx", ring["r"]) * factor)
                ring["ry"] = max(8.0, ring.get("ry", ring["r"]) * factor)
            ring["r"] = 0.5 * (ring["rx"] + ring["ry"])

            prof = ring.get("profile")
            if prof is not None and not is_alt:
                ring["profile"] = prof * factor

            image = _preview_state.get("image")
            if image is not None:
                h, w = image.shape[:2]
                ring["rx"] = min(ring["rx"], w * 0.49)
                ring["ry"] = min(ring["ry"], h * 0.49)
                ring["cx"] = max(ring["rx"], min(w - ring["rx"], ring["cx"]))
                ring["cy"] = max(ring["ry"], min(h - ring["ry"], ring["cy"]))

            _draw_detection_overlay()
            _draw_roi_overlay()
            _refresh_unwrap()
            _update_save_btn_state()

        def _nudge_ring(direction: int) -> None:
            """Grow (+1) or shrink (-1) the selected manual boundary."""
            _scale_ring(1.0 + 0.05 * int(direction))

        def _on_ring_hover(event) -> None:
            """Show appropriate cursor over rectangle handles, interior, and ring."""
            eye = _active_eye()
            if not eye:
                return
            tw, th = getattr(preview_canvas, "_thumb_size", (0, 0))
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))
            want = ""

            # In Manual Mode (and unlocked)
            if manual_mode_by_eye.get(eye, False) and not manual_locked_by_eye.get(eye, False):
                target = _ring_target()
                ring = pupil_override.get(eye) if target == "pupil" else ring_override.get(eye)
                image = _preview_state.get("image")
                if ring is not None and image is not None and tw > 0 and th > 0:
                    h, w = image.shape[:2]
                    rcx = ox + float(ring["cx"]) * (tw / float(w))
                    rcy = oy + float(ring["cy"]) * (th / float(h))
                    rrx = float(ring.get("rx", ring["r"])) * (tw / float(w))
                    rry = float(ring.get("ry", ring["r"])) * (th / float(h))
                    d_center = math.hypot(event.x - rcx, event.y - rcy)
                    if d_center <= 14.0:
                        want = "fleur"
                    elif abs(event.y - rcy) <= 12.0 and abs(abs(event.x - rcx) - rrx) <= 14.0:
                        want = "sb_h_double_arrow"
                    elif abs(event.x - rcx) <= 12.0 and abs(abs(event.y - rcy) - rry) <= 14.0:
                        want = "sb_v_double_arrow"
                    else:
                        ang = math.atan2(event.y - rcy, event.x - rcx)
                        ox_pt = rcx + rrx * math.cos(ang)
                        oy_pt = rcy + rry * math.sin(ang)
                        if math.hypot(event.x - ox_pt, event.y - oy_pt) <= 14.0:
                            want = "sizing"

            # In ROI mode (and unlocked)
            elif roi_active_by_eye.get(eye, False) and not roi_locked_by_eye.get(eye, False):
                roi = roi_by_eye.get(eye)
                if roi is not None and tw > 0 and th > 0:
                    hit = hit_test_rect(event.x, event.y, roi, tw, th, handle_radius=7.0, edge_tol=6.0, offset_x=ox, offset_y=oy)
                    if hit in ("nw", "se"):
                        want = "size_nw_se"
                    elif hit in ("ne", "sw"):
                        want = "size_ne_sw"
                    elif hit in ("edge_w", "edge_e"):
                        want = "size_we"
                    elif hit in ("edge_n", "edge_s"):
                        want = "size_ns"
                    elif hit == "move":
                        want = "fleur"

            if want != _ring_hover.get("cursor"):
                _ring_hover["cursor"] = want
                try:
                    preview_canvas.config(cursor=want if want else "crosshair")
                except tk.TclError:
                    pass

        def _reset_ring() -> None:
            """Clear manual override for active target."""
            eye = _active_eye()
            if eye:
                _override_of(_ring_target())[eye] = None
            _sync_ring_btn()
            _draw_detection_overlay()
            _draw_roi_overlay()
            _refresh_unwrap()
            _update_save_btn_state()

        def _update_ring_target_styles() -> None:
            """Highlight the ring target button."""
            active = _ring_target()
            for target, btn in ring_target_refs.items():
                is_active = target == active
                btn.config(
                    bg=DLG_ACCENT if is_active else DLG_BTN_SEC,
                    fg="#FFFFFF" if is_active else DLG_FG,
                    activebackground=DLG_ACCENT if is_active else DLG_BTN_ACT,
                    activeforeground="#FFFFFF" if is_active else DLG_FG,
                )

        def _select_ring_target(target: str) -> None:
            """Choose active manual boundary (pupil vs limbus)."""
            ring_target_var.set("pupil" if target == "pupil" else "limbus")
            _update_ring_target_styles()
            _sync_ring_btn()
            _draw_detection_overlay()
            _draw_roi_overlay()

        def _sync_ring_btn() -> None:
            """Enable ring controls when image and manual overrides exist."""
            eye = _active_eye()
            has_override = bool(_override_of(_ring_target()).get(eye) is not None)
            has_image = bool(_eye_path(eye))
            try:
                reset_ring_btn.config(state=tk.NORMAL if has_override else tk.DISABLED)
                sizing_state = tk.NORMAL if has_image else tk.DISABLED
                ring_minus_btn.config(state=sizing_state)
                ring_plus_btn.config(state=sizing_state)
            except tk.TclError:
                pass

        reset_ring_btn.config(command=_reset_ring)
        ring_minus_btn.config(command=lambda: _nudge_ring(-1))
        ring_plus_btn.config(command=lambda: _nudge_ring(+1))
        _update_ring_target_styles()
        preview_canvas.bind("<MouseWheel>", _on_ring_wheel)
        preview_canvas.bind("<Motion>", _on_ring_hover)

        def _draw_detection_overlay() -> None:
            """Draw the pupil / limbus overlays (manual or automatic) on the Pentacam preview."""
            preview_canvas.delete("detection")
            eye = _active_eye()
            if not eye:
                return
            img = _preview_state.get("image")
            tw, th = getattr(preview_canvas, "_thumb_size", (0, 0))
            if img is None or tw <= 0 or th <= 0:
                return
            h, w = img.shape[:2]
            if w <= 0 or h <= 0:
                return
            sx, sy = tw / float(w), th / float(h)
            ox, oy = getattr(preview_canvas, "_display_offset", (2.0, 2.0))

            manual = ring_override.get(eye)
            manual_pupil = pupil_override.get(eye)
            is_manual_active = bool(manual_mode_by_eye.get(eye, False) and not manual_locked_by_eye.get(eye, False))
            active_target = _ring_target()

            def _draw_cross(cx: float, cy: float, color: str, size: int = 6, width: int = 2) -> None:
                preview_canvas.create_line(
                    cx - size, cy, cx + size, cy,
                    fill=color, width=width, tags="detection",
                )
                preview_canvas.create_line(
                    cx, cy - size, cx, cy + size,
                    fill=color, width=width, tags="detection",
                )

            def _draw_handles(cx: float, cy: float, rx: float, ry: float, ang_deg: float, color: str) -> None:
                hs = 4.0
                rad = math.radians(ang_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                east_x, east_y = cx + rx * cos_a, cy + rx * sin_a
                west_x, west_y = cx - rx * cos_a, cy - rx * sin_a
                north_x, north_y = cx + ry * sin_a, cy - ry * cos_a
                south_x, south_y = cx - ry * sin_a, cy + ry * cos_a

                # Cardinal handles for major/minor axis resizing
                for hx, hy in [(east_x, east_y), (west_x, west_y), (north_x, north_y), (south_x, south_y)]:
                    preview_canvas.create_rectangle(
                        hx - hs, hy - hs, hx + hs, hy + hs,
                        outline=color, fill="#FFFFFF", width=1, tags="detection",
                    )

                # Rotation handle extending outward from East handle
                rot_len = 22.0
                rot_hx = cx + (rx + rot_len) * cos_a
                rot_hy = cy + (rx + rot_len) * sin_a
                preview_canvas.create_line(
                    east_x, east_y, rot_hx, rot_hy,
                    fill=color, width=1, dash=(2, 2), tags="detection",
                )
                rhs = 4.5
                preview_canvas.create_oval(
                    rot_hx - rhs, rot_hy - rhs, rot_hx + rhs, rot_hy + rhs,
                    outline=color, fill="#FFCC00", width=1.5, tags="detection",
                )

            def _draw_ring_dict(ring: Dict[str, Any], color: str, is_selected: bool) -> None:
                rcx = ox + float(ring["cx"]) * sx
                rcy = oy + float(ring["cy"]) * sy
                rrx = float(ring.get("rx", ring["r"])) * sx
                rry = float(ring.get("ry", ring["r"])) * sy

                ang_deg = float(ring.get("angle_deg", 0.0))
                rad = math.radians(ang_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                prof = ring.get("profile")
                if prof is not None:
                    pts = _ring_polygon_points(ring, sx, sy, offset_x=ox, offset_y=oy)
                elif abs(rrx - rry) > 1.0 or abs(ang_deg) > 0.1:
                    pts = []
                    for step in range(_RING_SAMPLES):
                        theta = 2.0 * math.pi * step / float(_RING_SAMPLES)
                        ex = rrx * math.cos(theta)
                        ey = rry * math.sin(theta)
                        rx_rot = ex * cos_a - ey * sin_a
                        ry_rot = ex * sin_a + ey * cos_a
                        pts.extend([rcx + rx_rot, rcy + ry_rot])
                else:
                    pts = _ring_polygon_points(ring, sx, sy, offset_x=ox, offset_y=oy)

                width = 3 if is_selected else 2
                preview_canvas.create_polygon(pts, outline=color, fill="", width=width, tags="detection")
                _draw_cross(rcx, rcy, "#FFFFFF" if is_selected else color, size=7 if is_selected else 5, width=2)
                if is_selected and is_manual_active:
                    _draw_handles(rcx, rcy, rrx, rry, ang_deg, color)

            def _draw_angle_markers(cx: float, cy: float, rx: float, ry: float, color: str = "#88b0ff") -> None:
                """Draw cardinal angle markers (0°, 90°, 180°, 270°) following clinical convention."""
                if rx <= 0.0 or ry <= 0.0:
                    return
                from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
                left_lbl, right_lbl, up_lbl, down_lbl = get_centration_meridian_labels(eye)

                # Light crosshair lines connecting opposite meridians
                preview_canvas.create_line(cx - rx, cy, cx + rx, cy, fill=color, dash=(2, 3), width=1, tags="detection")
                preview_canvas.create_line(cx, cy - ry, cx, cy + ry, fill=color, dash=(2, 3), width=1, tags="detection")

                # Degree labels (OD: left=0°, right=180°; OS: left=0°, right=180°)
                lbl_font = ("Segoe UI", 8, "bold")
                preview_canvas.create_text(cx + rx + 14, cy, text=f"{right_lbl}°", fill=color, font=lbl_font, anchor="w", tags="detection")
                preview_canvas.create_text(cx - rx - 14, cy, text=f"{left_lbl}°", fill=color, font=lbl_font, anchor="e", tags="detection")
                preview_canvas.create_text(cx, cy - ry - 10, text=f"{up_lbl}°", fill=color, font=lbl_font, anchor="s", tags="detection")
                preview_canvas.create_text(cx, cy + ry + 10, text=f"{down_lbl}°", fill=color, font=lbl_font, anchor="n", tags="detection")

            def _badge(text: str, outline: str, fill: str) -> None:
                width = max(290, 14 + int(6.8 * len(text)))
                bx = max(4, int(ox))
                by = max(4, int(oy))
                preview_canvas.create_rectangle(
                    bx, by, bx + width, by + 22, fill="#101010", outline=outline,
                    tags="detection",
                )
                preview_canvas.create_text(
                    bx + 6, by + 11, anchor="w", tags="detection",
                    text=text, fill=fill, font=("Segoe UI", 9, "bold"),
                )

            # --- Case 1: Manual Mode active or manual overrides present ---
            if manual is not None or manual_pupil is not None:
                if manual is not None:
                    _draw_ring_dict(manual, "#5c8aff", is_selected=(active_target == "limbus" and is_manual_active))
                    mcx = ox + float(manual["cx"]) * sx
                    mcy = oy + float(manual["cy"]) * sy
                    mrx = float(manual.get("rx", manual["r"])) * sx
                    mry = float(manual.get("ry", manual["r"])) * sy
                    _draw_angle_markers(mcx, mcy, mrx, mry, "#5c8aff")
                if manual_pupil is not None:
                    _draw_ring_dict(manual_pupil, "#00e676", is_selected=(active_target == "pupil" and is_manual_active))
                    if manual is None:
                        mcx = ox + float(manual_pupil["cx"]) * sx
                        mcy = oy + float(manual_pupil["cy"]) * sy
                        mrx = float(manual_pupil.get("rx", manual_pupil["r"])) * sx * 2.5
                        mry = float(manual_pupil.get("ry", manual_pupil["r"])) * sy * 2.5
                        _draw_angle_markers(mcx, mcy, mrx, mry, "#5c8aff")

                parts = []
                if manual is not None:
                    p_str = " (pulled)" if manual.get("profile") is not None else ""
                    parts.append(f"limbus r={float(manual['r']):.1f}px{p_str}")
                if manual_pupil is not None:
                    p_str = " (pulled)" if manual_pupil.get("profile") is not None else ""
                    parts.append(f"pupil r={float(manual_pupil['r']):.1f}px{p_str}")
                badge_text = ", ".join(parts)
                if is_manual_active:
                    badge_text = f"MANUAL: [{active_target.upper()}] " + badge_text
                else:
                    badge_text = "[LOCKED] " + badge_text
                _badge(badge_text, "#00e676" if not is_manual_active else "#5c8aff", "#00e676" if not is_manual_active else "#5c8aff")
                try:
                    preview_canvas.tag_raise("roi")
                except Exception:
                    pass
                return

            # --- Case 2: Automatic Detection Result ---
            result = _preview_state.get("result")
            if result is None:
                return

            is_reliable = bool(_preview_state.get("reliable", True))
            pupil = getattr(result, "pupil", None)
            limbus = getattr(result, "limbus", None)
            pupil_ok = bool(getattr(pupil, "detected", False) and getattr(pupil, "ellipse", None) is not None)
            limbus_ok = bool(getattr(limbus, "detected", False) and getattr(limbus, "ellipse", None) is not None)

            # Draw limbus if detected (blue "#5c8aff")
            if limbus_ok:
                le = limbus.ellipse
                flat_l = self._preview_ellipse_points(le, sx, sy, offset_x=ox, offset_y=oy)
                outline_color = "#5c8aff" if is_reliable else "#FFB020"
                preview_canvas.create_polygon(flat_l, outline=outline_color, fill="", width=2, tags="detection")
                lcx = ox + float(le.center_x) * sx
                lcy = oy + float(le.center_y) * sy
                lrx = float(getattr(le, "semi_major", getattr(le, "radius", 0.0))) * sx
                lry = float(getattr(le, "semi_minor", getattr(le, "radius", 0.0))) * sy
                _draw_cross(lcx, lcy, outline_color, size=7)
                _draw_angle_markers(lcx, lcy, lrx, lry, outline_color)

            # Draw pupil if detected (green "#00e676")
            if pupil_ok:
                pe = pupil.ellipse
                flat_p = self._preview_ellipse_points(pe, sx, sy, offset_x=ox, offset_y=oy)
                preview_canvas.create_polygon(flat_p, outline="#00e676", fill="", width=2, tags="detection")
                pcx = ox + float(pe.center_x) * sx
                pcy = oy + float(pe.center_y) * sy
                _draw_cross(pcx, pcy, "#00e676", size=6)
                if not limbus_ok:
                    prx = float(getattr(pe, "semi_major", getattr(pe, "radius", 0.0))) * sx * 2.5
                    pry = float(getattr(pe, "semi_minor", getattr(pe, "radius", 0.0))) * sy * 2.5
                    _draw_angle_markers(pcx, pcy, prx, pry, "#FFB020")

            # Status Badge
            if pupil_ok and limbus_ok and is_reliable:
                _badge("pupil OK, limbus OK", "#00e676", "#00e676")
            elif pupil_ok and not limbus_ok:
                _badge("pupil OK — limbus unconfirmed (use Manual Mode)", "#FFB020", "#FFB020")
            elif not pupil_ok and limbus_ok and is_reliable:
                _badge("pupil missing — limbus OK", "#FFB020", "#FFB020")
            else:
                _badge("detection unconfirmed — use Manual Mode", "#FF5252", "#FF5252")

            try:
                preview_canvas.tag_raise("roi")
            except Exception:
                pass

        # ── Render the preview image at the canvas' current size ──────────
        # The image fills the (now dialog-sized) preview window and is
        # re-fitted whenever that window changes size.  Purely display: the
        # detection result, the ROI and the hand ring keep their own
        # coordinates, so nothing here can touch the pipeline.
        _rendered_canvas: List[Tuple[int, int]] = [(0, 0)]

        def _render_preview_image() -> bool:
            """Draw ``_preview_state["image"]`` into the canvas at its size.

            The thumbnail is fitted to the canvas' actual size and centered,
            eliminating large unused black voids while preserving aspect ratio.
            Returns False when there is no image to draw.
            """
            img = _preview_state.get("image")
            if img is None:
                return False
            import cv2 as _cv2
            from PIL import Image as _PImg, ImageTk as _PITk
            from pupil_tracking.cyclotorsion.pentacam_loader import (
                resize_for_preview,
            )

            cw = preview_canvas.winfo_width()
            ch = preview_canvas.winfo_height()
            if cw <= 10 or ch <= 10:
                cw = 520
                ch = 280
            max_w, max_h = _preview_thumb_limits(cw, ch)
            thumb = resize_for_preview(img, max_width=max_w, max_height=max_h)
            rgb = _cv2.cvtColor(thumb, _cv2.COLOR_BGR2RGB)
            photo = _PITk.PhotoImage(_PImg.fromarray(rgb))
            preview_canvas.delete("all")
            tw, th = thumb.shape[1], thumb.shape[0]
            ox = max(2, (cw - tw) // 2)
            oy = max(2, (ch - th) // 2)
            preview_canvas.create_image(ox, oy, anchor="nw", image=photo, tags="image")
            preview_canvas._photo = photo
            preview_canvas._thumb_size = (tw, th)
            preview_canvas._display_offset = (ox, oy)
            _rendered_canvas[0] = (cw, ch)
            return True

        def _fit_preview_to_canvas(_event: Any = None) -> None:
            """Re-fit the shown image after the preview window changed size.

            ``<Configure>`` also fires for plain moves, so it does nothing
            unless the canvas really got a new size; detection is never
            re-run here -- only the picture and its overlays are redrawn.
            """
            if _preview_state.get("image") is None:
                return
            cw = preview_canvas.winfo_width()
            ch = preview_canvas.winfo_height()
            if (cw, ch) == _rendered_canvas[0]:
                return
            if not _render_preview_image():
                return
            _draw_detection_overlay()
            _draw_roi_overlay()

        preview_canvas.bind("<Configure>", _fit_preview_to_canvas, add="+")

        def _apply_detection(
            token: int,
            eye: str,
            result: Any,
            reliable: bool = True,
            edge: Any = None,
            note: str = "",
        ) -> None:
            try:
                if not preview_win.winfo_exists():
                    return
            except Exception:
                return
            if _preview_state.get("token") != token:
                return  # superseded by a newer run
            result_by_eye[eye] = result
            reliable_by_eye[eye] = bool(reliable)
            edge_by_eye[eye] = edge
            edge_note_by_eye[eye] = note

            if _active_eye() == eye:
                _preview_state["result"] = result
                _preview_state["reliable"] = bool(reliable)
                _preview_state["edge_limbus"] = edge
                _preview_state["edge_note"] = note
                _draw_detection_overlay()
                _draw_roi_overlay()  # keep user's ROI visible

                pupil_ok = bool(
                    getattr(getattr(result, "pupil", None), "detected", False)
                )
                limbus_ok = bool(
                    getattr(getattr(result, "limbus", None), "detected", False)
                )
                active_path = _eye_path(eye)
                flip_tag = " (flipped)" if flip_var.get() else ""
                if result is None:
                    det_tag = "  \u2014  detection failed (adjust ROI, press Enter)"
                elif not reliable and edge is not None:
                    p_prefix = "pupil OK, " if pupil_ok else ""
                    det_tag = f"  \u2014  {p_prefix}limbus {note or 'estimate'}"
                elif not reliable:
                    if pupil_ok:
                        det_tag = "  \u2014  pupil OK, limbus unconfirmed (adjust ROI, press Enter)"
                    else:
                        det_tag = (
                            "  \u2014  detection unconfirmed "
                            "(adjust ROI, press Enter)"
                        )
                else:
                    det_tag = (
                        "  \u2014  pupil " + ("OK" if pupil_ok else "\u2014")
                        + ", limbus " + ("OK" if limbus_ok else "\u2014")
                    )
                preview_label_text.set(f"{eye}{flip_tag}: {active_path}{det_tag}")
                _refresh_unwrap()
                _update_save_btn_state()

        def _run_preview_detection(eye: str) -> None:
            """Execute automatic pupil and limbus detection on Pentacam image using ELITA segmentation."""
            if not preview_win.winfo_exists():
                return
            active_path = _eye_path(eye)
            img = _preview_state.get("image")
            if not active_path or img is None:
                return

            flip_tag = " (flipped)" if flip_var.get() else ""
            if not hasattr(self, "_pentacam_detector") or self._pentacam_detector is None:
                from pupil_tracking.cyclotorsion.pentacam_detector import PentacamDetector
                self._pentacam_detector = PentacamDetector()

            h, w = img.shape[:2]
            roi = roi_by_eye.get(eye)
            if roi is None:
                roi = _default_fixed_roi(w, h)
                roi_by_eye[eye] = roi

            preview_label_text.set(
                f"{eye}{flip_tag}: {active_path}  —  detecting…"
            )

            _preview_state["token"] = _preview_state.get("token", 0) + 1
            token = _preview_state["token"]

            _detection_queue: queue.Queue = queue.Queue()

            def _worker() -> None:
                res = None
                reliable = False
                edge = None
                note = ""

                try:
                    res = self._pentacam_detector.detect(
                        img,
                        roi=roi,
                        eye=eye,
                        source=f"pentacam:{active_path}",
                    )
                    if res is not None:
                        pupil_ok = bool(getattr(getattr(res, "pupil", None), "detected", False) and getattr(res.pupil, "ellipse", None) is not None)
                        limbus_ok = bool(getattr(getattr(res, "limbus", None), "detected", False) and getattr(res.limbus, "ellipse", None) is not None)
                        if pupil_ok and limbus_ok:
                            reliable = True
                            note = "pupil OK, limbus OK"
                        elif pupil_ok:
                            reliable = False
                            note = "pupil OK — limbus unconfirmed"
                        elif limbus_ok:
                            reliable = False
                            note = "limbus OK — pupil unconfirmed"
                        else:
                            reliable = False
                            note = "unconfirmed"
                except Exception as exc:
                    self.logger.warning("Pentacam preview detection attempt failed: %s", exc)
                    reliable = False
                    note = f"error: {exc}"

                _detection_queue.put((token, eye, res, reliable, edge, note))
                try:
                    self.root.after(
                        0,
                        lambda: _apply_detection(token, eye, res, reliable, edge, note),
                    )
                except Exception:
                    pass

            def _poll_detection_queue() -> None:
                try:
                    if not preview_win.winfo_exists():
                        return
                except Exception:
                    return
                applied = False
                while not _detection_queue.empty():
                    try:
                        q_tok, q_eye, q_res, q_rel, q_edge, q_note = _detection_queue.get_nowait()
                        if _preview_state.get("token") == q_tok:
                            _apply_detection(q_tok, q_eye, q_res, q_rel, q_edge, q_note)
                            applied = True
                    except Exception:
                        break
                if not applied and _preview_state.get("token") == token:
                    try:
                        self.root.after(30, _poll_detection_queue)
                    except Exception:
                        pass

            threading.Thread(target=_worker, daemon=True).start()
            try:
                self.root.after(30, _poll_detection_queue)
            except Exception:
                pass

        def _refresh_preview():
            if not preview_win.winfo_exists():
                return
            from pupil_tracking.cyclotorsion.pentacam_loader import (
                load_pentacam_image, flip_pentacam,
            )

            _preview_state["token"] = _preview_state.get("token", 0) + 1
            token = _preview_state["token"]
            eye = _active_eye()
            active_path = _eye_path(eye)
            if not active_path:
                preview_label_text.set("No image selected")
                preview_canvas.delete("all")
                preview_canvas._thumb_size = (0, 0)
                _preview_state["image"] = None
                _preview_state["result"] = None
                _update_roi_status()
                _update_eye_btn_styles()
                _update_roi_btn_style()
                _update_manual_btn_style()
                _update_save_btn_state()
                _refresh_unwrap()
                return

            img = load_pentacam_image(active_path)
            if img is None:
                preview_label_text.set(f"Cannot read image: {active_path}")
                preview_canvas.delete("all")
                preview_canvas._thumb_size = (0, 0)
                _preview_state["image"] = None
                _preview_state["result"] = None
                cw = max(preview_canvas.winfo_width(), 640)
                ch = max(preview_canvas.winfo_height(), 380)
                preview_canvas.create_text(
                    cw // 2, ch // 2,
                    text=f"Failed to load image:\n{active_path}\n\nUnsupported format or corrupted file.",
                    fill="#ff5252",
                    font=("Segoe UI", 11, "bold"),
                    justify=tk.CENTER,
                    tags="error_msg",
                )
                _update_roi_status()
                _update_eye_btn_styles()
                _update_roi_btn_style()
                _update_manual_btn_style()
                _update_save_btn_state()
                _refresh_unwrap()
                return

            if flip_var.get():
                img = flip_pentacam(img)
            _preview_state["image"] = img

            _render_preview_image()
            _sync_ring_btn()

            flip_tag = " (flipped)" if flip_var.get() else ""
            h, w = img.shape[:2]

            # Independent per-eye state restoration:
            is_locked = bool(roi_locked_by_eye.get(eye, False))
            roi_locked_var.set(is_locked)
            has_manual = bool(ring_override.get(eye) is not None or pupil_override.get(eye) is not None)
            is_manual_active = bool(manual_mode_by_eye.get(eye, False))

            if has_manual:
                _draw_detection_overlay()
                if is_manual_active:
                    preview_label_text.set(
                        f"{eye} Manual Mode: adjusting {_ring_target()}. Drag center to move, outline to resize. Press Enter to lock."
                    )
                else:
                    p_r = pupil_override.get(eye, {}).get("r", 0.0)
                    l_r = ring_override.get(eye, {}).get("r", 0.0)
                    preview_label_text.set(
                        f"{eye}: Manual boundaries locked (pupil r={p_r:.1f}px, limbus r={l_r:.1f}px). Click Manual Mode to edit again."
                    )
            elif is_locked and result_by_eye.get(eye) is not None:
                _preview_state["result"] = result_by_eye[eye]
                _preview_state["reliable"] = reliable_by_eye.get(eye, True)
                _preview_state["edge_limbus"] = edge_by_eye.get(eye)
                _preview_state["edge_note"] = edge_note_by_eye.get(eye, "")
                _draw_detection_overlay()
                pupil_ok = bool(
                    getattr(getattr(_preview_state["result"], "pupil", None), "detected", False)
                )
                limbus_ok = bool(
                    getattr(getattr(_preview_state["result"], "limbus", None), "detected", False)
                )
                det_tag = (
                    "  \u2014  pupil " + ("OK" if pupil_ok else "\u2014")
                    + ", limbus " + ("OK" if limbus_ok else "\u2014")
                )
                preview_label_text.set(f"{eye}{flip_tag}: {active_path}{det_tag}")
            elif is_locked:
                _preview_state["result"] = None
                _run_preview_detection(eye)
            elif not roi_active_by_eye.get(eye, False):
                # When valid image is loaded and user is NOT in interactive ROI selection mode:
                # Automatically lock default ROI and trigger preview detection immediately
                if roi_by_eye.get(eye) is None:
                    roi_by_eye[eye] = _default_fixed_roi(w, h)
                roi_locked_by_eye[eye] = True
                roi_locked_var.set(True)
                _preview_state["result"] = None
                _run_preview_detection(eye)
            else:
                # Interactive ROI mode is active and unlocked:
                # Wait until the user positions and locks ROI (via Enter or 'Lock & Detect') before detecting
                _preview_state["result"] = None
                preview_canvas.delete("detection")
                if roi_by_eye.get(eye) is None:
                    roi_by_eye[eye] = _default_fixed_roi(w, h)
                preview_label_text.set(
                    f"{eye}{flip_tag}: {active_path}  \u2014  Position ROI, press Enter to detect"
                )

            if roi_active_by_eye.get(eye, False):
                _draw_roi_overlay()
            else:
                preview_canvas.delete("roi")

            _update_roi_status()
            _update_eye_btn_styles()
            _update_roi_btn_style()
            _update_manual_btn_style()
            _update_save_btn_state()
            _refresh_unwrap()

        def _refresh_unwrap() -> None:
            """Flatten the active eye's Pentacam iris into the main-window panel."""
            active = _active_eye()
            try:
                result = _preview_state.get("result")
                ellipse, pupil_ellipse = self._unwrap_limbus_choice(
                    result,
                    bool(_preview_state.get("reliable", True)),
                    _preview_state.get("edge_limbus"),
                    _preview_state.get("image"),
                )

                limbus_radius_fn = None
                pupil_radius_fn = None

                manual = ring_override.get(active)
                if manual is not None:
                    ellipse = _make_circle(manual["cx"], manual["cy"], manual["r"]) if manual.get("rx") == manual.get("ry") else _dict_to_ellipse(manual)
                    if manual.get("profile") is not None:
                        limbus_radius_fn = lambda a, m=manual: _ring_radius_at(m["r"], m.get("profile"), a)

                manual_pupil = pupil_override.get(active)
                if manual_pupil is not None:
                    pupil_ellipse = _make_circle(manual_pupil["cx"], manual_pupil["cy"], manual_pupil["r"]) if manual_pupil.get("rx") == manual_pupil.get("ry") else _dict_to_ellipse(manual_pupil)
                    if manual_pupil.get("profile") is not None:
                        pupil_radius_fn = lambda a, mp=manual_pupil: _ring_radius_at(mp["r"], mp.get("profile"), a)

                proc_img = _preview_state.get("image")
                if proc_img is not None and roi_active_by_eye.get(active, False) and roi_locked_by_eye.get(active, False):
                    roi = roi_by_eye.get(active)
                    if roi is not None:
                        h_img, w_img = proc_img.shape[:2]
                        ix0, iy0, ix1, iy1, _, _ = roi_norm_to_image_coords(roi, w_img, h_img)
                        mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        mask[iy0:iy1, ix0:ix1] = 255
                        proc_img = cv2.bitwise_and(proc_img, proc_img, mask=mask)

                strip = self._unwrap_pentacam(
                    proc_img,
                    limbus_ellipse=ellipse,
                    pupil_ellipse=pupil_ellipse,
                    limbus_radius_fn=limbus_radius_fn,
                    pupil_radius_fn=pupil_radius_fn,
                    eye=active,
                )
                if active:
                    unwrap_strip_by_eye[active] = strip
                _preview_state["unwrap_strip"] = strip
                self._set_unwrap_strip(strip)
            except Exception:
                if active:
                    unwrap_strip_by_eye[active] = None
                _preview_state["unwrap_strip"] = None
                self._set_unwrap_strip(None)

        def _on_image_path_changed(target_eye: str) -> None:
            preview_eye_var.set(target_eye)
            roi_by_eye[target_eye] = None
            roi_active_by_eye[target_eye] = False
            roi_locked_by_eye[target_eye] = False
            manual_mode_by_eye[target_eye] = False
            manual_locked_by_eye[target_eye] = False
            result_by_eye[target_eye] = None
            reliable_by_eye[target_eye] = True
            edge_by_eye[target_eye] = None
            edge_note_by_eye[target_eye] = ""
            ring_override[target_eye] = None
            pupil_override[target_eye] = None
            unwrap_strip_by_eye[target_eye] = None
            _preview_state["image"] = None
            _preview_state["result"] = None
            _preview_state["edge_limbus"] = None
            _preview_state["edge_note"] = ""
            _preview_state["unwrap_strip"] = None
            self._set_unwrap_strip(None)
            preview_canvas.delete("roi")
            preview_canvas.delete("detection")
            _refresh_preview()

        pentacam_od.trace_add("write", lambda *_: _on_image_path_changed("OD"))
        pentacam_os.trace_add("write", lambda *_: _on_image_path_changed("OS"))

        def _compute_cyclotorsion() -> None:
            """Display-only OD/OS cyclotorsion estimate for the preview.
            The per-eye reference fields have been removed, so each eye's own
            Pentacam image is used as its (self) reference baseline.  Purely a
            readout: nothing is fed back into the centration / pupil / limbus
            pipeline.
            """
            eyes = ("OD", "OS")
            idle = "Cyclotorsion  OD: \u2014   OS: \u2014"
            if not any(_eye_path(e) and roi_by_eye.get(e) for e in eyes):
                cyc_readout_var.set(idle)
                return
            try:
                from pupil_tracking.cyclotorsion.pair_readout import (
                    estimate_pair_readout,
                )
            except Exception:
                cyc_readout_var.set("Cyclotorsion: unavailable")
                return
            parts = []
            for eye in eyes:
                current = _eye_path(eye)
                roi = roi_by_eye.get(eye)
                if not current:
                    parts.append(f"{eye}: no image")
                    continue
                if roi is None:
                    parts.append(f"{eye}: no ROI")
                    continue
                try:

                    # Auto-reference: compare the image against itself.
                    readout = estimate_pair_readout(
                        current, current, roi, flip=flip_var.get()
                    )
                except Exception:
                    parts.append(f"{eye}: error")
                    continue
                if readout.valid:
                    parts.append(
                        f"{eye}: {readout.angle_deg:+.1f}\u00b0 "
                        f"(score {readout.score:.2f})"
                    )
                else:
                    parts.append(f"{eye}: n/a")
            cyc_readout_var.set("Cyclotorsion  " + "   ".join(parts))
        cyc_compute_btn.config(command=_compute_cyclotorsion)

        def _close_preview() -> None:
            """Clean up preview resources / state on dialog close."""
            pass

        def _frame_border_x() -> int:
            """Window-manager frame border width (client vs frame x-offset)."""
            return dialog.winfo_rootx() - dialog.winfo_x()

        def _preview_size() -> Tuple[int, int]:
            """Keep the Pentacam preview the same size as the Add Patient window.

            See :func:`_preview_geometry` for the clamping rules.
            """
            return _preview_geometry(
                dialog.winfo_width(),
                dialog.winfo_height(),
                dialog.winfo_reqwidth(),
                dialog.winfo_reqheight(),
                dialog.winfo_screenwidth(),
                _frame_border_x(),
            )

        def _dock_preview() -> None:
            """Docking helper retained for backwards compatibility."""
            try:
                pw, ph = _preview_size()
                px, py = dialog.winfo_x(), dialog.winfo_y()
                _dummy = f"{pw}x{ph}+{px}+{py}"
            except Exception:
                pass

        def _position_preview() -> None:
            """Positioning helper retained for backwards compatibility."""
            pass

        # ── end preview frame ─────────────────────────────────────────
        # Buttons are packed into the pre-pinned 'foot' frame at the bottom of dialog
        tk.Button(
            foot, text="Cancel",
            font=BTN_FONT,
            bg=DLG_BTN_SEC, fg=DLG_FG,
            activebackground=DLG_BTN_ACT, activeforeground=DLG_FG,
            relief=tk.FLAT, bd=0, padx=22, pady=8, cursor="hand2",
            command=lambda: close_dialog(),
        ).pack(side=tk.RIGHT, padx=(10, 0))
        add_btn = tk.Button(
            foot, text="Add Patient",
            font=("Segoe UI", 10, "bold"),
            bg=DLG_BTN_SEC, fg=DLG_FG_HINT,
            activebackground=DLG_ACCENT, activeforeground="#FFFFFF",
            relief=tk.FLAT, bd=0, padx=22, pady=8, cursor="hand2",
            state=tk.DISABLED,
        )
        add_btn.pack(side=tk.RIGHT)

        def _update_enabled(*_):
            ok = bool(patient_id.get().strip() and patient_name.get().strip())
            add_btn.config(
                state=tk.NORMAL if ok else tk.DISABLED,
                bg=DLG_ACCENT      if ok else DLG_BTN_SEC,
                fg="#FFFFFF"       if ok else DLG_FG_HINT,
            )
            try:
                _update_save_btn_state()
            except Exception:
                pass
        patient_id.trace_add("write", _update_enabled)
        patient_name.trace_add("write", _update_enabled)

        def create_patient():
            if not self._save_storage_settings():
                return
            pid = patient_id.get().strip()
            pname = patient_name.get().strip()
            if not pid or not pname:
                return
            try:
                root = Path(self._storage_root_var.get())
                patient_dir = (
                    root
                    / f"PX_{self._safe_folder_name(pid)}"
                      f"_{self._safe_folder_name(pname)}"
                )
                patient_dir.mkdir(parents=True, exist_ok=True)
                for eye in ("OD", "OS"):
                    (patient_dir / eye / "Pentacam").mkdir(parents=True, exist_ok=True)
                metadata = {
                    "patient_id":   pid,
                    "patient_name": pname,
                    "hvid":         {"OD": hvid_od.get().strip(), "OS": hvid_os.get().strip()},
                    "created_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                (patient_dir / "patient.txt").write_text(
                    "# Patient Record - Medevplus IXcentai\n"
                    f"patient_id={metadata['patient_id']}\n"
                    f"patient_name={metadata['patient_name']}\n"
                    f"hvid_od={metadata['hvid']['OD']}\n"
                    f"hvid_os={metadata['hvid']['OS']}\n"
                    f"created_at={metadata['created_at']}\n",
                    encoding="utf-8",
                )

                # Save both eyes completely and reliably
                saved_od = _save_eye_data("OD", patient_dir, pid, pname, prompt_errors=False)
                saved_os = _save_eye_data("OS", patient_dir, pid, pname, prompt_errors=False)

                self._active_patient_dir = patient_dir
                self._patient_id_var.set(metadata["patient_id"])
                self._patient_name_var.set(metadata["patient_name"])

                # Determine active eye automatically: prioritize OD if available, else OS
                has_od = bool(_eye_path("OD"))
                has_os = bool(_eye_path("OS"))
                default_eye = "OD" if has_od else ("OS" if has_os else "")

                if default_eye:
                    self._select_eye(default_eye)
                else:
                    self._selected_eye_var.set("")
                    self._set_unwrap_strip(None)

                self._add_px_dialog_open = False
                self._update_patient_controls()
                self._status_var.set(f"Patient folder created \u2192 {patient_dir}")
                try:
                    self.root.unbind("<Configure>", root_bind_id)
                except Exception:
                    pass
                _close_preview()
                dialog.destroy()
            except OSError as exc:
                messagebox.showerror(
                    "Patient creation error",
                    f"Could not create patient folder:\n{exc}",
                )

        def close_dialog() -> None:
            try:
                self.root.unbind("<Configure>", root_bind_id)
            except Exception:
                pass
            self._add_px_dialog_open = False
            self._update_patient_controls()
            if self._active_patient_dir is not None and not self._selected_eye_var.get():
                p_dir = Path(self._active_patient_dir)
                has_od = bool(_eye_path("OD")) or (
                    (p_dir / "OD" / "Pentacam").is_dir()
                    and bool(list((p_dir / "OD" / "Pentacam").glob("*.png")))
                )
                has_os = bool(_eye_path("OS")) or (
                    (p_dir / "OS" / "Pentacam").is_dir()
                    and bool(list((p_dir / "OS" / "Pentacam").glob("*.png")))
                )
                def_eye = "OD" if has_od else ("OS" if has_os else "")
                if def_eye:
                    self._select_eye(def_eye)
            _close_preview()
            dialog.destroy()
        add_btn.config(command=create_patient)
        def _on_dialog_return(event: Any) -> Optional[str]:
            res = _handle_key_confirm(event)
            if res == "break":
                return "break"
            if add_btn.cget("state") == tk.NORMAL:
                create_patient()
            return None

        dialog.bind("<Return>", _on_dialog_return)
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        dialog.update_idletasks()
        try:
            dialog.focus_set()
        except Exception:
            pass

    def _open_static_mode(self) -> None:
        """Open the Static Mode (image-based detection) window.
        ADDITIVE + display-only: a separate window where the Pentacam and
        ELITA stills are unwrapped into straight iris strips and the
        cyclotorsion between them is reported with features, confidence and a
        validation checklist.  It only *calls* the existing detector — the
        live centration / pupil / limbus pipeline is untouched and nothing
        here is fed to surgical control.
        """
        try:
            from pupil_tracking.interface.static_mode_window import open_static_mode
        except Exception as exc:  # pragma: no cover - import failure
            messagebox.showerror(
                "Static Mode",
                f"Could not open Static Mode:\n{exc}",
            )
            return
        open_static_mode(self)

    def _open_dynamic_mode(self) -> None:
        """Open the Dynamic (Live) Mode cyclotorsion monitor.
        ADDITIVE + display-only: a separate window that continuously unwraps
        the current live frame's iris and compares it with the patient's
        Pentacam baseline, reporting the ongoing cyclotorsion angle and
        detection confidence.  It only *reads* the live detection geometry
        (``_current_image`` / ``_current_result``); the centration / pupil /
        limbus pipeline is untouched and nothing here is fed to surgical
        control.
        """
        try:
            from pupil_tracking.interface.dynamic_mode_window import (
                open_dynamic_mode,
            )
        except Exception as exc:  # pragma: no cover - import failure
            messagebox.showerror(
                "Dynamic Mode",
                f"Could not open Dynamic Mode:\n{exc}",
            )
            return
        open_dynamic_mode(self)

    def _set_patient_btn_colors(self) -> None:
        active_eye = self._selected_eye_var.get()
        has_patient = self._active_patient_dir is not None
        add_px_active = bool(getattr(self, "_add_px_dialog_open", False))
        if hasattr(self, "_add_px_btn"):
            if isinstance(self._add_px_btn, tk.Button):
                self._add_px_btn.config(
                    state=tk.DISABLED if self._recorder.is_recording else tk.NORMAL,
                    bg=self._colors.BTN_ACTION if add_px_active else self._colors.BTN_BG,
                    fg=self._colors.BTN_PRIMARY if add_px_active else self._colors.FG_PRIMARY,
                    activebackground=self._colors.BTN_ACTION if add_px_active else self._colors.BTN_HOVER,
                    activeforeground=self._colors.BTN_PRIMARY,
                )
            else:
                self._add_px_btn.configure(
                    style="ToolbarActive.TButton" if add_px_active else "TButton",
                )
        for button in getattr(self, "_eye_buttons", []):
            text = str(button.cget("text"))
            is_active = active_eye == text
            button.configure(style="ToolbarActive.TButton" if is_active else "TButton")
            button.config(state=tk.DISABLED if self._recorder.is_recording else tk.NORMAL)

    def _load_pentacam_for_eye(self, eye: str) -> Optional[np.ndarray]:
        """Resolve and load the saved Pentacam unwrap strip for *eye*.

        Follows existing storage conventions:
        1. Checks for 'Pentacam_<eye>_unwrapped.png' in <patient_dir>/<eye>/Pentacam/
        2. If not found, locates the saved flipped/original Pentacam image and
           reconstructs the unwrap strip using saved pupil/limbus geometry from
           'pentacam_results.json' or 'pentacam_values.txt'.
        3. Returns None if no valid Pentacam image or data exists for *eye*.
        """
        if self._active_patient_dir is None:
            return None
        eye_dir = Path(self._active_patient_dir) / eye / "Pentacam"
        if not eye_dir.is_dir():
            return None

        # 1. Direct saved unwrapped strip
        strip_path = eye_dir / f"Pentacam_{eye}_unwrapped.png"
        if strip_path.is_file() and strip_path.stat().st_size > 0:
            try:
                from pupil_tracking.cyclotorsion.pentacam_loader import _read_image_robust
                strip = _read_image_robust(strip_path)
                if strip is not None and getattr(strip, "size", 0) > 0 and getattr(strip, "ndim", 0) >= 2:
                    return strip
            except Exception as exc:
                self.logger.warning("Failed loading saved unwrapped strip %s: %s", strip_path, exc)

        # 2. Locate saved Pentacam image in eye_dir
        from pupil_tracking.cyclotorsion.pentacam_loader import (
            load_pentacam_image, flip_pentacam, _SUPPORTED_EXTS,
        )
        img_file: Optional[Path] = None
        is_already_flipped = False

        flipped_path = eye_dir / f"Pentacam_{eye}_flipped.png"
        if flipped_path.is_file() and flipped_path.stat().st_size > 0:
            img_file = flipped_path
            is_already_flipped = True
        else:
            orig_matches = list(eye_dir.glob("Pentacam_original.*"))
            if orig_matches and orig_matches[0].is_file() and orig_matches[0].stat().st_size > 0:
                img_file = orig_matches[0]
            else:
                from pupil_tracking.cyclotorsion.static_mode import find_reference_image
                ref = find_reference_image(self._active_patient_dir, eye)
                if ref is not None and ref.is_file() and ref.stat().st_size > 0:
                    img_file = ref
                else:
                    for cand in eye_dir.iterdir():
                        if (
                            cand.is_file()
                            and cand.stat().st_size > 0
                            and cand.suffix.lower() in _SUPPORTED_EXTS
                            and not cand.name.lower().endswith("_unwrapped.png")
                        ):
                            img_file = cand
                            break

        if img_file is None:
            return None

        try:
            img = load_pentacam_image(img_file)
            if img is None:
                return None
            if not is_already_flipped:
                flip_flag = True
                val_txt = eye_dir / "pentacam_values.txt"
                if val_txt.is_file():
                    for line in val_txt.read_text(encoding="utf-8").splitlines():
                        if line.strip().startswith("image_flipped="):
                            flip_flag = line.strip().split("=")[1].lower() == "true"
                            break
                if flip_flag:
                    img = flip_pentacam(img)

            # Retrieve boundary ellipses from pentacam_results.json or pentacam_values.txt
            p_ellipse = None
            l_ellipse = None
            res_json = eye_dir / "pentacam_results.json"
            if res_json.is_file() and res_json.stat().st_size > 0:
                try:
                    meta = json.loads(res_json.read_text(encoding="utf-8"))
                    p_meta = meta.get("pupil")
                    l_meta = meta.get("limbus")
                    if p_meta and "center_x" in p_meta:
                        p_rx = float(p_meta.get("semi_major", p_meta.get("radius", 40.0)))
                        p_ry = float(p_meta.get("semi_minor", p_meta.get("radius", 40.0)))
                        p_ellipse = SimpleNamespace(
                            center_x=float(p_meta["center_x"]),
                            center_y=float(p_meta["center_y"]),
                            semi_major=p_rx,
                            semi_minor=p_ry,
                            radius=float(p_meta.get("radius", 0.5 * (p_rx + p_ry))),
                            angle_deg=float(p_meta.get("angle_deg", 0.0)),
                        )
                    if l_meta and "center_x" in l_meta:
                        l_rx = float(l_meta.get("semi_major", l_meta.get("radius", 120.0)))
                        l_ry = float(l_meta.get("semi_minor", l_meta.get("radius", 120.0)))
                        l_ellipse = SimpleNamespace(
                            center_x=float(l_meta["center_x"]),
                            center_y=float(l_meta["center_y"]),
                            semi_major=l_rx,
                            semi_minor=l_ry,
                            radius=float(l_meta.get("radius", 0.5 * (l_rx + l_ry))),
                            angle_deg=float(l_meta.get("angle_deg", 0.0)),
                        )
                except Exception as exc:
                    self.logger.warning("Error reading %s: %s", res_json, exc)

            if p_ellipse is None or l_ellipse is None:
                val_txt = eye_dir / "pentacam_values.txt"
                if val_txt.is_file():
                    try:
                        p_cx, p_cy, p_r = None, None, None
                        l_cx, l_cy, l_r = None, None, None
                        for line in val_txt.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if line.startswith("pupil_center="):
                                parts = line.split("=")[1].split(",")
                                p_cx, p_cy = float(parts[0]), float(parts[1])
                            elif line.startswith("pupil_radius="):
                                p_r = float(line.split("=")[1])
                            elif line.startswith("limbus_center="):
                                parts = line.split("=")[1].split(",")
                                l_cx, l_cy = float(parts[0]), float(parts[1])
                            elif line.startswith("limbus_radius="):
                                l_r = float(line.split("=")[1])
                        if p_cx is not None and p_cy is not None and p_r is not None:
                            p_ellipse = SimpleNamespace(
                                center_x=p_cx, center_y=p_cy, semi_major=p_r, semi_minor=p_r, radius=p_r, angle_deg=0.0
                            )
                        if l_cx is not None and l_cy is not None and l_r is not None:
                            l_ellipse = SimpleNamespace(
                                center_x=l_cx, center_y=l_cy, semi_major=l_r, semi_minor=l_r, radius=l_r, angle_deg=0.0
                            )
                    except Exception as exc:
                        self.logger.warning("Error reading %s: %s", val_txt, exc)

            if p_ellipse is None or l_ellipse is None:
                roi_norm = None
                val_txt = eye_dir / "pentacam_values.txt"
                if val_txt.is_file():
                    try:
                        for line in val_txt.read_text(encoding="utf-8").splitlines():
                            if line.strip().startswith("iris_roi_norm="):
                                from pupil_tracking.cyclotorsion.pentacam_loader import parse_roi_norm
                                roi_norm = parse_roi_norm(line.strip().split("=")[1])
                                break
                    except Exception:
                        pass
                if roi_norm is not None:
                    from pupil_tracking.cyclotorsion.pentacam_loader import roi_norm_to_iris_roi
                    ih, iw = img.shape[:2]
                    iris_roi = roi_norm_to_iris_roi(roi_norm, width=iw, height=ih)
                    if getattr(iris_roi, "valid", False):
                        p_ellipse = SimpleNamespace(
                            center_x=iris_roi.center_x, center_y=iris_roi.center_y,
                            semi_major=iris_roi.pupil_semi_major, semi_minor=iris_roi.pupil_semi_minor,
                            radius=iris_roi.pupil_radius_px, angle_deg=iris_roi.pupil_angle_deg,
                        )
                        l_ellipse = SimpleNamespace(
                            center_x=iris_roi.center_x, center_y=iris_roi.center_y,
                            semi_major=iris_roi.limbus_semi_major, semi_minor=iris_roi.limbus_semi_minor,
                            radius=iris_roi.limbus_radius_px, angle_deg=iris_roi.limbus_angle_deg,
                        )

            if p_ellipse is None or l_ellipse is None:
                try:
                    if not hasattr(self, "_pentacam_detector") or self._pentacam_detector is None:
                        from pupil_tracking.cyclotorsion.pentacam_detector import PentacamDetector
                        self._pentacam_detector = PentacamDetector()
                    res = self._pentacam_detector.detect(img, eye=eye, source=f"pentacam:{img_file}")
                    if res is not None:
                        if getattr(getattr(res, "pupil", None), "detected", False):
                            p_ellipse = res.pupil.ellipse
                        if getattr(getattr(res, "limbus", None), "detected", False):
                            l_ellipse = res.limbus.ellipse
                except Exception as exc:
                    self.logger.warning("Pentacam load detection failed for %s: %s", eye, exc)

            if p_ellipse is None or l_ellipse is None:
                ih, iw = img.shape[:2]
                cx, cy = iw * 0.5, ih * 0.5
                lr = min(iw, ih) * 0.25
                pr = lr * 0.38
                p_ellipse = SimpleNamespace(
                    center_x=cx, center_y=cy, semi_major=pr, semi_minor=pr, radius=pr, angle_deg=0.0
                )
                l_ellipse = SimpleNamespace(
                    center_x=cx, center_y=cy, semi_major=lr, semi_minor=lr, radius=lr, angle_deg=0.0
                )

            strip = self._unwrap_pentacam(img, limbus_ellipse=l_ellipse, pupil_ellipse=p_ellipse, eye=eye)
            if strip is not None and getattr(strip, "size", 0) > 0:
                try:
                    cv2.imwrite(str(strip_path), strip)
                except Exception:
                    pass
                return strip
        except Exception as exc:
            self.logger.warning("Failed loading/reconstructing Pentacam unwrap for %s: %s", eye, exc)

        return None

    def _select_eye(self, eye: str) -> None:
        if self._recorder.is_recording:
            messagebox.showwarning(
                "Recording Active",
                f"Cannot switch eye while recording is active.\n"
                f"Stop current {self._selected_eye_var.get()} recording before switching to {eye}.",
            )
            return

        # 1. Update selected eye variable
        self._selected_eye_var.set(eye)

        # 2. Update current result eye if result exists
        if hasattr(self, "_current_result") and self._current_result is not None:
            self._current_result.eye = eye
        if hasattr(self, "_opt_processor") and self._opt_processor is not None:
            setattr(self._opt_processor, "eye", eye)

        # 3. Handle patient folder if active
        if self._active_patient_dir is not None:
            self._set_unwrap_strip(None)
            self._reset_elita_shift()
            target = self._active_patient_dir / eye
            self._recording_location_var.set(f"Recording folder: {target}")
            self._patient_summary_var.set(f"PX: {self._patient_id_var.get()} — {self._patient_name_var.get()} | Eye: {eye}")
            strip = self._load_pentacam_for_eye(eye)
            self._set_unwrap_strip(strip)
            if strip is not None:
                self._status_var.set(f"Loaded {eye} Pentacam image for PX: {self._patient_id_var.get()}")
            else:
                pid_txt = self._patient_id_var.get()
                msg = f"No Pentacam image saved for {eye} — bottom window cleared (Patient: {pid_txt})"
                self._status_var.set(msg)
                self.logger.info(f"Patient {pid_txt} ({self._patient_name_var.get()}): {msg}")
        else:
            self._patient_summary_var.set(f"Eye: {eye} | No patient selected")
            self._status_var.set(f"Selected eye: {eye}")

        self._update_patient_controls()

        if hasattr(self, "_unwrap_outer"):
            try:
                strip = getattr(self, "_unwrap_strip", None)
                if strip is not None:
                    self._unwrap_outer.configure(
                        text=f"Iris Unwrap ({eye}) — Pentacam (top) / ELITA live (bottom) · 0° left → 360° right"
                    )
                else:
                    self._unwrap_outer.configure(
                        text=f"Iris Unwrap ({eye}) — [No Pentacam image saved for {eye}]"
                    )
            except Exception:
                pass

        # Immediate display refresh to update centration labels (0° and 180°)
        self._refresh_display()

        # Sync Dynamic Mode window if open
        dyn_win = getattr(self, "_dynamic_mode_window", None)
        if dyn_win is not None and getattr(dyn_win, "_win", None) is not None:
            try:
                dyn_win._set_eye(eye)
            except Exception:
                pass

    def _update_patient_controls(self) -> None:
        locked = self._recorder.is_recording
        if hasattr(self, "_add_px_btn"):
            self._add_px_btn.config(state=tk.DISABLED if locked else tk.NORMAL)
        for button in getattr(self, "_eye_buttons", []):
            button.config(state=tk.DISABLED if locked else tk.NORMAL)
        self._set_patient_btn_colors()

    def _patient_recording_path(self) -> Optional[str]:
        """Resolve output path inside the current Centration Folder Location."""
        if (
            self._active_patient_dir is not None
            and self._selected_eye_var.get() in ("OD", "OS")
        ):
            target_dir = self._active_patient_dir / self._selected_eye_var.get()
        elif self._active_patient_dir is not None:
            target_dir = self._active_patient_dir
        else:
            raw_root = self._storage_root_var.get().strip()
            target_dir = (
                Path(raw_root).expanduser()
                if raw_root
                else (Path.home() / "Desktop" / "Centration")
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        return str(target_dir / f"detection_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    def _choose_recording_path(self) -> Optional[str]:
        """Ask user where to save the recording."""
        default_name = self._recording_default_path_var.get()
        if not default_name:
            default_name = f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path = filedialog.asksaveasfilename(
            title="Save Recording As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video (H.264)", "*.mp4"),
                ("AVI Video", "*.avi"),
                ("All files", "*.*"),
            ],
            initialfile=default_name,
        )
        return path if path else None

    def _start_recording(self) -> None:
        """Start dual recording: raw J&J surgical feed + centration overlay feed."""
        if self._recorder.is_recording:
            return

        if self._current_image is None:
            messagebox.showinfo(
                "No Source",
                "Start a video or camera before recording.",
            )
            return
        path = self._patient_recording_path()
        if not path:
            return
        overlay_path, raw_path, sync_path = self._recorder.derive_companion_paths(path)

        # The companion CSV should describe this recording only, never frames
        # viewed while preparing the camera or a prior patient session.
        self._results_history.clear()
        raw_h, raw_w = self._current_image.shape[:2]

        # Prepare a frame that includes the measurements panel so the
        # overlay stream captures the full on-screen layout (image + table).
        display_image = self._prepare_recording_frame(
            self._current_image,
            self._current_result,
        )
        composite = self._compose_capture_frame(display_image, self._current_result)
        h, w = composite.shape[:2]

        target_fps = 30.0
        if self._video_cap is not None:
            fps = self._video_cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps <= 120:
                target_fps = fps
        if not self._recorder.start(
            overlay_path,
            w,
            h,
            target_fps,
            raw_path=raw_path,
            raw_width=raw_w,
            raw_height=raw_h,
            sync_path=sync_path,
            overlay_composer=self._compose_capture_frame,
        ):
            messagebox.showerror(
                "Recording Error",
                f"Cannot start dual recording. Check codec support.\nPath: {overlay_path}",
            )
            return
        self._recording_path = overlay_path
        self._recording_raw_path = raw_path
        self._recording_sync_path = sync_path
        self._recording_default_path_var.set(
            f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        )

        self._update_recording_ui(started=True)
        self._update_patient_controls()
        hw_info = getattr(self._recorder, "last_hw_backend", "QSV/H.264")
        self._recording_location_var.set(
            f"Dual Rec ({hw_info}) → Raw: {Path(raw_path).name} | Overlay: {Path(overlay_path).name}"
        )
        self._status_var.set(
            f"Dual recording started ({hw_info}) → Raw: {Path(raw_path).name} + Overlay: {Path(overlay_path).name}"
        )
        self._start_recording_timer()

    def _stop_recording(self) -> None:
        """Stop dual recording and save raw surgical video, overlay video, and sync CSV."""
        if not self._recorder.is_recording:
            return

        path = self._recorder.stop()
        self._stop_recording_timer()
        raw_path = getattr(self._recorder, "last_raw_path", self._recording_raw_path)
        sync_path = getattr(self._recorder, "last_sync_path", self._recording_sync_path)

        elapsed = (
            self._recorder.elapsed_time
            if hasattr(self._recorder, "elapsed_time")
            else 0
        )
        frame_count = (
            self._recorder.frame_count if hasattr(self._recorder, "frame_count") else 0
        )

        self._update_recording_ui(started=False)
        csv_path = None
        if path and self._results_history:
            csv_path = str(Path(path).with_suffix(".csv"))
            self._export_csv(csv_path, show_success=False)
        self._update_patient_controls()
        folder_str = str(Path(path).parent) if path else "unknown"
        raw_name = Path(raw_path).name if raw_path else "N/A"
        overlay_name = Path(path).name if path else "N/A"
        sync_name = Path(sync_path).name if sync_path else "N/A"
        saved_message = (
            f"Dual recording saved ({frame_count} synced frames in {elapsed:.1f}s) "
            f"in {folder_str} → Raw: {raw_name} | Overlay: {overlay_name} | Sync: {sync_name}"
        )
        if csv_path:
            saved_message += f" | CSV: {Path(csv_path).name}"
        self._recording_location_var.set(saved_message)
        self._status_var.set(saved_message)
        if path:
            messagebox.showinfo("Dual Recording Saved", saved_message)

    def _toggle_recording(self) -> None:
        """Toggle recording on/off."""
        if self._recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _on_recorder_status(self, status: dict) -> None:
        """Callback from DualFrameRecorder for status updates."""
        if status.get("is_recording"):
            elapsed = status.get("elapsed_time", 0)
            fps = status.get("fps", 0)
            dropped = status.get("dropped_frames", 0)
            mode_tag = "DUAL" if status.get("is_dual") else "REC"

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  {mode_tag} {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_fps_var.set(fps)
            self._recording_dropped_var.set(dropped)
        else:
            self._recording_indicator.config(text="  --:--  ", foreground="#616161")

    def _start_recording_timer(self) -> None:
        """Start the recording indicator update timer."""
        self._stop_recording_timer()
        self._update_recording_timer()

    def _update_recording_timer(self) -> None:
        """Periodically update the recording indicator."""
        if self._recorder.is_recording:
            elapsed = self._recorder.elapsed_time
            fps = self._recorder.frame_count / elapsed if elapsed > 0 else 0
            mode_tag = "DUAL" if getattr(self._recorder, "is_dual_recording", False) else "REC"

            mins, secs = divmod(int(elapsed), 60)
            self._recording_indicator.config(
                text=f"  {mode_tag} {mins:02d}:{secs:02d} ({fps:.0f}fps)  ",
                foreground="#ff1744",
            )
            self._recording_timer_id = self.root.after(
                500, self._update_recording_timer
            )
        else:
            self._stop_recording_timer()

    def _stop_recording_timer(self) -> None:
        """Stop the recording indicator update timer."""
        if self._recording_timer_id is not None:
            self.root.after_cancel(self._recording_timer_id)
            self._recording_timer_id = None

    def _update_recording_ui(self, started: bool = False) -> None:
        """Update dual recording button state in the frontend toolbar."""
        if hasattr(self, "_rec_btn"):
            if started or self._recorder.is_recording:
                self._rec_btn.config(text="⏹ Stop Dual Rec")
            else:
                self._rec_btn.config(text="⏺ Start Dual Rec")

    def _write_frame_to_recorder(
        self,
        frame: np.ndarray,
        *,
        raw_frame: Optional[np.ndarray] = None,
        result: Optional[Any] = None,
        frame_index: Optional[int] = None,
    ) -> None:
        """Write synchronised raw surgical frame + overlay frame to recorder (non-blocking).
        Offloads measurements panel composition (`_compose_capture_frame`) and dual
        H.264/H.265/QSV encoding to the background recorder threads so detection
        latency stays well below <150 ms.
        """
        if not self._recorder.is_recording:
            return
        raw_src = raw_frame if raw_frame is not None else self._current_image
        if raw_src is None:
            raw_src = frame
        res_src = result if result is not None else self._current_result
        idx_src = frame_index if frame_index is not None else self._frame_count

        # Determine whether `frame` still needs the side measurements panel composed
        needs_compose = bool(
            raw_src is not None and frame.shape[:2] == raw_src.shape[:2]
        )
        self._recorder.write(
            frame,
            raw_frame=raw_src,
            frame_index=idx_src,
            result=res_src,
            compose_in_background=needs_compose,
        )

    def _prepare_recording_frame(
        self,
        frame: np.ndarray,
        result: Any,
        *,
        draw_overlay: bool = True,
    ) -> np.ndarray:
        """Prepare a frame that mirrors the current on-screen display for recording.
        Pass ``draw_overlay=False`` to get just the (grayscale-converted)
        image with no annotations — used by the Snapshot button so the cropped
        region inside the blue ring contains only clean image pixels.
        """
        mode = self._grayscale_mode_var.get()
        if mode == "off":
            image = frame.copy()
        else:
            image = self._convert_display_frame(frame.copy())
        if draw_overlay:
            if result is not None and self._show_overlay.get():
                self._draw_overlay_scaled(image, result, 1.0)
            self._draw_manual_roi_overlay(image, 1.0)
            self._draw_manual_ring_overlay(image, 1.0)
            if self._show_debug_overlay.get():
                self._draw_debug_overlay(image, 1.0)

        return image

    @staticmethod
    def _hex_to_bgr(value: str) -> Tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            return (200, 200, 200)
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return (b, g, r)

    def _measurement_capture_sections(self) -> List[Tuple[str, Tuple[int, int, int], List[Tuple[str, str]]]]:
        return [
            (
                "PUPIL",
                self._hex_to_bgr(self._colors.PUPIL),
                [
                    ("Center", self._pv["center"].get()),
                    ("Diameter (px)", self._pv["diameter_px"].get()),
                    ("Diameter (mm)", self._pv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._pv["semi_major"].get()),
                    ("Semi-Major (mm)", self._pv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._pv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._pv["semi_minor_mm"].get()),
                    ("Angle", self._pv["angle"].get()),
                    ("Fit Type", self._pv["fit_type"].get()),
                    ("Confidence", self._pv["confidence"].get()),
                    ("Quality", self._pv["quality"].get()),
                ],
            ),
            (
                "LIMBUS",
                self._hex_to_bgr(self._colors.LIMBUS),
                [
                    ("Center", self._lv["center"].get()),
                    ("Diameter (px)", self._lv["diameter_px"].get()),
                    ("Diameter (mm)", self._lv["diameter_mm"].get()),
                    ("Semi-Major (px)", self._lv["semi_major"].get()),
                    ("Semi-Major (mm)", self._lv["semi_major_mm"].get()),
                    ("Semi-Minor (px)", self._lv["semi_minor"].get()),
                    ("Semi-Minor (mm)", self._lv["semi_minor_mm"].get()),
                    ("Angle", self._lv["angle"].get()),
                    ("Fit Type", self._lv["fit_type"].get()),
                    ("Confidence", self._lv["confidence"].get()),
                    ("Quality", self._lv["quality"].get()),
                ],
            ),
            (
                "CORNEAL OFFSET",
                self._hex_to_bgr(self._colors.OFFSET),
                [
                    ("Corneal Centre", self._ov["corneal_center"].get()),
                    ("Offset (px)", self._ov["offset_px"].get()),
                    ("Offset (mm)", self._ov["offset_mm"].get()),
                    ("Offset dX,dY px", self._ov["offset_vec_px"].get()),
                    ("Offset dX,dY mm", self._ov["offset_vec_mm"].get()),
                    ("Offset Angle", self._ov["offset_angle"].get()),
                    ("Pupil/Limbus", self._ov["pupil_limbus_ratio"].get()),
                ],
            ),
            (
                "CORNEAL DIMENSIONS (WTW)",
                self._hex_to_bgr(self._colors.LIMBUS),
                [
                    ("Horizontal", self._wtw_vars["horizontal"].get()),
                    ("Vertical", self._wtw_vars["vertical"].get()),
                    ("Mean", self._wtw_vars["mean"].get()),
                ],
            ),
            (
                "PROCESSING",
                self._hex_to_bgr(self._colors.PROCESSING),
                [
                    ("Proc. Time", self._proc_time_var.get()),
                    ("Latency", self._latency_var.get()),
                    ("Latency Avg", self._latency_avg_var.get()),
                    ("Dropped/Stale", self._drop_var.get()),
                    ("Tracking", self._tracking_state_var.get()),
                    ("FPS", self._fps_var.get()),
                    ("Frame", self._frame_var.get()),
                    ("Image Size", self._image_size_var.get()),
                    ("Pipeline", self._pipeline_var.get()),
                    ("Grayscale", self._gray_mode_var_display.get()),
                ],
            ),
        ]

    # Map of non-ASCII glyphs used in the live Tk UI to ASCII equivalents
    # that OpenCV's cv2.putText (Hershey fonts) CAN render. Without this,
    # cv2.putText draws '?' for any codepoint outside its ASCII glyph table,
    # which is why the recording showed "155.5??" etc. The live UI is
    # unaffected — this only touches the composited/recorded panel.
    _CAPTURE_GLYPH_MAP = {
        "°": " deg",   # ° degree
        "×": "x",      # × multiplication
        "—": "-",      # — em dash
        "–": "-",      # – en dash
        "→": "->",     # → arrow
        "µ": "u",      # µ micro
        "≥": ">=",     # ≥
        "≤": "<=",     # ≤
        "±": "+/-",    # ±
        "•": "*",      # • bullet
        "⚠": "!",      # ⚠ warning
        "⚡": "",       # ⚡ lightning
        "✓": "OK",     # ✓ check
        "✗": "X",      # ✗ cross
    }

    def _ascii_for_capture(self, text: str) -> str:
        """Sanitize a UI string so cv2.putText renders it without '?' glyphs.

        Replaces known symbols with ASCII equivalents, then drops any
        remaining non-ASCII codepoint so nothing renders as a question mark.
        """
        if not text:
            return text
        for uni, ascii_rep in self._CAPTURE_GLYPH_MAP.items():
            if uni in text:
                text = text.replace(uni, ascii_rep)
        # Drop any leftover non-ASCII so OpenCV never emits '?'
        if any(ord(ch) > 127 for ch in text):
            text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def _render_measurements_capture(self, height: int, width: int) -> np.ndarray:
        panel = np.full((height, width, 3), self._hex_to_bgr(self._colors.BG_SECONDARY), dtype=np.uint8)
        cv2.rectangle(
            panel,
            (0, 0),
            (width - 1, height - 1),
            self._hex_to_bgr(self._colors.BORDER),
            1,
        )
        pad = max(10, height // 72)
        gutter = max(10, width // 42)
        inner_w = width - pad * 2
        col_w = max(220, (inner_w - gutter) // 2)
        title_font = max(0.42, min(0.72, height / 900.0))
        body_font = max(0.34, min(0.54, height / 1080.0))
        line_h = max(16, int(height / 36))
        row_gap = max(3, int(line_h * 0.2))
        section_gap = max(8, int(line_h * 0.55))
        summary_box_h = max(62, int(height * 0.09))
        summary_gap = max(8, gutter // 2)
        summary_w = max(120, (inner_w - summary_gap) // 2)
        fg_primary = self._hex_to_bgr(self._colors.FG_PRIMARY)
        fg_secondary = self._hex_to_bgr(self._colors.FG_SECONDARY)
        card_bg = self._hex_to_bgr(self._colors.BG_TERTIARY)
        quality_color = self._hex_to_bgr(
            _QUALITY_COLORS.get(self._summary_quality_var.get(), self._colors.FG_PRIMARY)
        )
        tracking_color = self._hex_to_bgr(
            {
                "Tracking Stable": self._colors.SURGICAL,
                "Tracking Acquiring": self._colors.CLINICAL,
                "Tracking Degraded": self._colors.RESEARCH,
                "No Detection": self._colors.INSUFFICIENT,
                "Ready": self._colors.ACCENT,
                "Waiting": self._colors.FG_SECONDARY,
            }.get(self._summary_tracking_var.get(), self._colors.FG_PRIMARY)
        )

        summaries = [
            ("QUALITY", self._summary_quality_var.get(), quality_color),
            ("TRACKING", self._summary_tracking_var.get(), tracking_color),
            ("LATENCY", self._summary_latency_var.get(), fg_primary),
            ("PIPELINE", self._summary_pipeline_var.get(), fg_primary),
        ]
        for idx, (label, value, color) in enumerate(summaries):
            row = idx // 2
            col = idx % 2
            x0 = pad + col * (summary_w + summary_gap)
            y0 = pad + row * (summary_box_h + summary_gap)
            x1 = min(width - pad, x0 + summary_w)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), card_bg, -1)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), self._hex_to_bgr(self._colors.BORDER), 1)
            cv2.putText(panel, self._ascii_for_capture(label), (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg_secondary, 1, cv2.LINE_AA)
            cv2.putText(panel, self._ascii_for_capture(value or "---"), (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

        sections = self._measurement_capture_sections()
        left_sections = sections[:2]
        right_sections = sections[2:]
        start_y = pad * 2 + summary_box_h * 2 + summary_gap

        def draw_section_column(items, x_start):
            y = start_y
            for title, accent, rows in items:
                box_h = max(60, 34 + len(rows) * (line_h + row_gap))
                if y + box_h > height - pad:
                    box_h = max(40, height - pad - y)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), card_bg, -1)
                cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), self._hex_to_bgr(self._colors.BORDER), 1)
                cv2.putText(panel, self._ascii_for_capture(title), (x_start + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, title_font, accent, 2, cv2.LINE_AA)
                row_y = y + 48
                for label, value in rows:
                    if row_y > y + box_h - 8:
                        break
                    clean_label = self._ascii_for_capture(label.replace("_", " ").title())
                    clean_value = self._ascii_for_capture((value or "---").replace("\n", " "))
                    if len(clean_value) > 32:
                        clean_value = clean_value[:29] + "..."
                    cv2.putText(panel, clean_label, (x_start + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_secondary, 1, cv2.LINE_AA)
                    text_size = cv2.getTextSize(clean_value, cv2.FONT_HERSHEY_SIMPLEX, body_font, 1)[0]
                    value_x = max(x_start + 140, x_start + col_w - 10 - text_size[0])
                    cv2.putText(panel, clean_value, (value_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_primary, 1, cv2.LINE_AA)
                    row_y += line_h + row_gap
                y += box_h + section_gap

        draw_section_column(left_sections, pad)
        draw_section_column(right_sections, pad + col_w + gutter)
        return panel

    def _compose_capture_frame(self, image: np.ndarray, result: Any) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        panel_w = max(700, int(img_w * 0.62))
        panel = self._render_measurements_capture(img_h, panel_w)
        divider = np.full((img_h, 3, 3), self._hex_to_bgr(self._colors.BORDER), dtype=np.uint8)
        return np.concatenate([image, divider, panel], axis=1)

    # ══════════════════════════════════════════════════════════════════
    # END RECORDING
    # ══════════════════════════════════════════════════════════════════

    # ================================================================
    # Menu
    # ================================================================

    def _build_menu(self) -> None:
        c = self._colors
        menu_cfg = dict(
            bg=c.BG_SECONDARY,
            fg=c.FG_PRIMARY,
            activebackground=c.ACCENT_DIM,
            activeforeground=c.FG_PRIMARY,
            relief="flat",
            borderwidth=0,
        )
        menubar = tk.Menu(self.root, **menu_cfg)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Image…",
            command=self._open_image,
            accelerator="Ctrl+O",
        )
        file_menu.add_command(
            label="Open Video…",
            command=self._open_video,
            accelerator="Ctrl+V",
        )
        file_menu.add_separator()
        file_menu.add_command(label="Open Folder…", command=self._open_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results CSV…", command=self._export_csv)
        file_menu.add_command(label="Export Results TXT…", command=self._export_txt)
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Save Snapshot…", command=self._export_snapshot)
        file_menu.add_separator()
        file_menu.add_command(
            label="Static Mode (Image-Based Detection)…",
            command=self._open_static_mode,
            accelerator="Ctrl+M",
        )
        file_menu.add_command(
            label="Dynamic Mode (Live Cyclotorsion)…",
            command=self._open_dynamic_mode,
            accelerator="Ctrl+Shift+M",
        )
        # ══════════════════════════════════════════════════════════
        # RECORDING 3 of 8 — Dual Recording menu options
        # ══════════════════════════════════════════════════════════
        file_menu.add_separator()
        self._recording_menu_var = tk.StringVar(value="Start Dual Recording")
        file_menu.add_command(
            label="Start Dual Recording (Raw + Overlay)…",
            command=self._start_recording,
            accelerator="Ctrl+R",
        )
        file_menu.add_command(
            label="Stop Dual Recording",
            command=self._stop_recording,
            accelerator="Ctrl+Shift+R",
        )
        file_menu.add_separator()
        # ══════════════════════════════════════════════════════════
        file_menu.add_command(
            label="Exit", command=self._on_close, accelerator="Ctrl+Q"
        )

        camera_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Start Camera", command=self._start_camera)
        camera_menu.add_command(label="Stop Camera", command=self._stop_video)

        view_menu = tk.Menu(menubar, tearoff=0, **menu_cfg)
        menubar.add_cascade(label="View", menu=view_menu)
        self._show_overlay = tk.BooleanVar(value=True)
        self._show_pupil = tk.BooleanVar(value=True)
        self._show_limbus = tk.BooleanVar(value=True)
        self._show_offset = tk.BooleanVar(value=True)
        self._show_centers = tk.BooleanVar(value=True)
        self._show_ring_center = tk.BooleanVar(value=False)
        self._show_measurements = tk.BooleanVar(value=False)
        self._show_debug_overlay = tk.BooleanVar(value=False)

        view_menu.add_checkbutton(
            label="Show Overlay",
            variable=self._show_overlay,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Pupil",
            variable=self._show_pupil,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Limbus",
            variable=self._show_limbus,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Offset Line",
            variable=self._show_offset,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Centers",
            variable=self._show_centers,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Ring Center",
            variable=self._show_ring_center,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show On-Image Measurements",
            variable=self._show_measurements,
            command=self._refresh_display,
        )
        view_menu.add_checkbutton(
            label="Show Debug Overlay",
            variable=self._show_debug_overlay,
            command=self._refresh_display,
        )

        # ADDITIVE display-only: iris zone grid (12 angular sectors x 30 deg =
        # clock hours) drawn over the unwrap strips.  Moves nothing.
        self._show_unwrap_zones = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(
            label="Show Iris Zone Grid",
            variable=self._show_unwrap_zones,
            command=self._redraw_unwrap,
        )

        self.root.bind("<Control-o>", lambda _e: self._open_image())
        self.root.bind("<Control-v>", lambda _e: self._open_video())
        self.root.bind("<Control-m>", lambda _e: self._open_static_mode())
        self.root.bind("<Control-Shift-M>", lambda _e: self._open_dynamic_mode())
        self.root.bind("<Control-q>", lambda _e: self._on_close())
        self.root.bind("<space>", lambda _e: self._toggle_pause())
        # ══════════════════════════════════════════════════════════
        # RECORDING 4 of 8 — Recording keyboard shortcuts
        # ══════════════════════════════════════════════════════════
        self.root.bind("<Control-r>", lambda _e: self._start_recording())
        self.root.bind("<Control-R>", lambda _e: self._toggle_recording())
        # ══════════════════════════════════════════════════════════

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 6 of 12 — [G] keyboard shortcut
        # ══════════════════════════════════════════════════════════
        self.root.bind("<g>", self._toggle_grayscale)
        self.root.bind("<G>", self._toggle_grayscale)
        self.root.bind("<Return>", self._confirm_active_selection)
        self.root.bind("<Escape>", self._cancel_active_selection)
        self.root.bind("<Left>", lambda e: self._nudge_roi(-1, 0, e))
        self.root.bind("<Right>", lambda e: self._nudge_roi(1, 0, e))
        self.root.bind("<Up>", lambda e: self._nudge_roi(0, -1, e))
        self.root.bind("<Down>", lambda e: self._nudge_roi(0, 1, e))
        # ══════════════════════════════════════════════════════════

    # ================================================================
    # Toolbar
    # ================================================================

    def _toolbar_set_active(self, active_btn) -> None:
        """Highlight active_btn blue; reset all other tracked toolbar buttons."""
        for btn in getattr(self, "_toolbar_tracked_btns", []):
            try:
                is_active = btn is active_btn
                if isinstance(btn, tk.Button):
                    btn.config(
                        bg=self._colors.BTN_ACTION if is_active else self._colors.BTN_BG,
                        fg=self._colors.BTN_PRIMARY if is_active else self._colors.FG_PRIMARY,
                    )
                else:
                    btn.configure(style="ToolbarActive.TButton" if is_active else "TButton")
            except Exception:
                pass

    def _build_toolbar(self) -> None:
        c = self._colors

        # Outer bar spans the window: it holds the horizontally scrollable
        # ribbon on the left and a pinned, always-visible quality badge on
        # the right (the badge must never scroll out of view).
        bar = ttk.Frame(self.root, style="Primary.TFrame")
        bar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(0, 1))
        self._quality_label = ttk.Label(
            bar,
            text="  NO IMAGE  ",
            style="Quality.TLabel",
            anchor="center",
        )
        self._quality_label.pack(side=tk.RIGHT, padx=10)

        # Scrollable ribbon: a Canvas hosting an inner frame, with a slider
        # (horizontal scrollbar) that appears only when the ribbon overflows
        # the available width so every button stays reachable.
        scroll_wrap = ttk.Frame(bar, style="Primary.TFrame")
        scroll_wrap.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._toolbar_canvas = tk.Canvas(
            scroll_wrap,
            bg=c.BG_PRIMARY,
            highlightthickness=0,
            borderwidth=0,
            height=1,
        )
        self._toolbar_scroll = ttk.Scrollbar(
            scroll_wrap,
            orient=tk.HORIZONTAL,
            command=self._toolbar_canvas.xview,
        )
        self._toolbar_canvas.configure(xscrollcommand=self._toolbar_scroll.set)
        self._toolbar_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        # The scrollbar is packed/unpacked on demand by _sync_toolbar().
        self._toolbar_scroll_shown = False

        # Inner frame kept as ``toolbar`` so all button rows below are hosted
        # inside the scrollable area unchanged.
        toolbar = ttk.Frame(self._toolbar_canvas, style="Primary.TFrame")
        self._toolbar_inner = toolbar
        self._toolbar_window = self._toolbar_canvas.create_window(
            (0, 0), window=toolbar, anchor="nw"
        )
        self._toolbar_tracked_btns: list = []
        _vid_btn = ttk.Button(toolbar, text="🏞️ Video", command=lambda: [self._open_video(), self._toolbar_set_active(_vid_btn)])
        _vid_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_vid_btn)
        _cam_btn = ttk.Button(toolbar, text="📷 Camera", command=lambda: [self._start_camera(), self._toolbar_set_active(_cam_btn)])
        _cam_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_cam_btn)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self._pause_btn = ttk.Button(
            toolbar,
            text="⏸ Pause",
            command=self._toggle_pause,
            state=tk.DISABLED,
        )
        self._pause_btn.pack(side=tk.LEFT, padx=2)
        self._pause_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(self._pause_btn)
        _stop_btn = ttk.Button(toolbar, text="⏹ Stop", command=lambda: [self._stop_video(), self._toolbar_set_active(None)])
        _stop_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_stop_btn)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Load Add PX icon for toolbar button
        try:
            from PIL import Image as _PILImg, ImageTk as _PILITk
            import pathlib as _pl
            _apx_icon_path = _pl.Path(__file__).parent / "pupil_tracking" / "interface" / "add-patient.png.png"
            if not _apx_icon_path.exists():
                _apx_icon_path = _pl.Path("pupil_tracking") / "interface" / "add-patient.png.png"
            _apx_img = _PILImg.open(_apx_icon_path).resize((18, 18), _PILImg.LANCZOS)
            self._add_px_icon = _PILITk.PhotoImage(_apx_img)
        except Exception:
            self._add_px_icon = None
        if self._add_px_icon:
            self._add_px_btn = tk.Button(
                toolbar,
                text=" Add PX",
                image=self._add_px_icon,
                compound=tk.LEFT,
                font=("Segoe UI", 10),
                bg=self._colors.BTN_BG,
                fg=self._colors.FG_PRIMARY,
                activebackground=self._colors.BTN_ACTIVE,
                activeforeground=self._colors.FG_PRIMARY,
                relief=tk.FLAT, bd=0, padx=6, pady=3, cursor="hand2",
                command=lambda: [self._open_patient_dialog(), self._toolbar_set_active(self._add_px_btn)],
            )
        else:
            self._add_px_btn = ttk.Button(
                toolbar,
                text="⊕ Add PX",
                command=lambda: [self._open_patient_dialog(), self._toolbar_set_active(self._add_px_btn)],
            )
        self._toolbar_tracked_btns.append(self._add_px_btn)
        self._add_px_btn.pack(side=tk.LEFT, padx=2)
        self._eye_buttons: List[ttk.Button] = []
        for eye in ("OD", "OS"):
            button = ttk.Button(toolbar, text=eye, command=lambda current_eye=eye: self._select_eye(current_eye), state=tk.DISABLED)
            button.pack(side=tk.LEFT, padx=2)
            self._eye_buttons.append(button)

        # ══════════════════════════════════════════════════════════
        # RECORDING — Dual Recording toolbar button
        # ══════════════════════════════════════════════════════════
        self._rec_btn = ttk.Button(
            toolbar,
            text="⏺ Start Dual Rec",
            command=self._toggle_recording,
        )
        self._rec_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(self._rec_btn)

        self._recording_indicator = ttk.Label(
            toolbar,
            text="  --:--  ",
            style="Quality.TLabel",
            foreground="#616161",
        )
        self._recording_indicator.pack(side=tk.LEFT, padx=(0, 4))
        # ══════════════════════════════════════════════════════════

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 7 of 12 — Grayscale toggle button in toolbar
        #
        # Button cycles: RGB → AUTO → GRAY → RGB
        # Indicator label shows current mode with colour coding
        # ══════════════════════════════════════════════════════════
        self._gray_btn = ttk.Button(
            toolbar,
            text="🔲 RGB",
            command=self._toggle_grayscale,
        )
        self._gray_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self._roi_btn = ttk.Button(
            toolbar,
            text="ROI",
            command=self._toggle_roi,
        )
        self._roi_btn.pack(side=tk.LEFT, padx=2)
        self._ring_btn = ttk.Button(
            toolbar,
            text="Ring",
            command=self._toggle_ring,
        )
        self._ring_btn.pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        _csv_btn = ttk.Button(toolbar, text="Export CSV", command=lambda: [self._export_csv(), self._toolbar_set_active(_csv_btn)])
        _csv_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_csv_btn)
        _snap_btn = ttk.Button(toolbar, text="Snapshot", command=lambda: [self._export_snapshot(), self._toolbar_set_active(_snap_btn)])
        _snap_btn.pack(side=tk.LEFT, padx=2)
        self._toolbar_tracked_btns.append(_snap_btn)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Static Mode opens a separate window, so it is deliberately NOT added
        # to _toolbar_tracked_btns (it is not a source/mode of this view).
        ttk.Button(
            toolbar,
            text="📐 Static Mode",
            command=self._open_static_mode,
        ).pack(side=tk.LEFT, padx=2)

        # Dynamic Mode likewise opens its own window.
        ttk.Button(
            toolbar,
            text="🔴 Dynamic Mode",
            command=self._open_dynamic_mode,
        ).pack(side=tk.LEFT, padx=2)

        # ── Horizontal scroll plumbing ──────────────────────────────
        # Keeps the canvas exactly one ribbon-row tall, updates the
        # scrollable region, and shows the slider only when the ribbon is
        # wider than the visible area.

        def _sync_toolbar(_event=None) -> None:
            canvas = self._toolbar_canvas
            inner = self._toolbar_inner
            req_w = inner.winfo_reqwidth()
            req_h = inner.winfo_reqheight()

            # Match the canvas height to the ribbon only when it actually
            # changes, so we never fight the canvas <Configure> event.
            if req_h != getattr(self, "_toolbar_req_h", None):
                self._toolbar_req_h = req_h
                canvas.configure(height=req_h)
            canvas.configure(scrollregion=(0, 0, req_w, req_h))
            need = req_w > canvas.winfo_width() + 1
            if need and not self._toolbar_scroll_shown:
                self._toolbar_scroll.pack(side=tk.BOTTOM, fill=tk.X)
                self._toolbar_scroll_shown = True
            elif not need and self._toolbar_scroll_shown:
                self._toolbar_scroll.pack_forget()
                self._toolbar_scroll_shown = False
                canvas.xview_moveto(0.0)

        def _on_toolbar_wheel(event) -> str:

            # Wheel (and Shift+wheel) over the ribbon scrolls it sideways.
            if self._toolbar_scroll_shown and event.delta:
                self._toolbar_canvas.xview_scroll(
                    int(-event.delta / 120), "units"
                )
            return "break"
        self._toolbar_inner.bind("<Configure>", _sync_toolbar)
        self._toolbar_canvas.bind("<Configure>", _sync_toolbar)
        self._toolbar_canvas.bind("<MouseWheel>", _on_toolbar_wheel)
        self._toolbar_canvas.bind("<Shift-MouseWheel>", _on_toolbar_wheel)
        self._toolbar_canvas.after_idle(_sync_toolbar)

    # ================================================================
    # Main Area
    # ================================================================

    def _build_main_area(self) -> None:
        c = self._colors
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Default split ratio: 7:3 (image/loading on the left, panels on the right)
        left_frame = ttk.Frame(main, style="Primary.TFrame")
        main.add(left_frame, weight=65)
        self._build_unwrap_frame(left_frame)

        self._canvas = tk.Canvas(
            left_frame,
            bg=c.CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self._canvas.bind("<MouseWheel>", self._on_canvas_wheel)

        right_frame = ttk.Frame(main, width=500)
        main.add(right_frame, weight=35)

        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        meas_frame = ttk.Frame(notebook, padding=8)
        notebook.add(meas_frame, text="Measurements")
        self._build_measurements_panel(meas_frame)

        detail_frame = ttk.Frame(notebook, padding=8)
        notebook.add(detail_frame, text="Details")
        self._build_details_panel(detail_frame)

        settings_frame = ttk.Frame(notebook, padding=8)
        notebook.add(settings_frame, text="⚙ Settings")
        self._build_settings_panel(settings_frame)

        # Ensure the PanedWindow sash initial position is set once layout is realized.
        # Without this, some Tk themes/OS window managers keep the sash at a default
        # position until the user resizes the window.
        def _set_initial_sash() -> None:
            w = main.winfo_width()
            if w is None or w <= 1:
                self.root.after(150, _set_initial_sash)
                return
            main.sashpos(0, int(w * 0.70))

        self.root.after(0, _set_initial_sash)

    def _build_unwrap_frame(self, parent: ttk.Frame) -> None:
        """Build the Iris-Unwrap panel in place of the old Video-Progress panel.
        The area inside the blue ring (limbus) is flattened into a polar strip
        and shown for **two** sources in the same bounded panel, using one
        shared angular ruler:
        * **Pentacam** — the recorded reference still (from Add Patient), and
        * **ELITA live** — the current camera/video frame, redrawn as the
          frames progress.
        Both strips are the **iris annulus** (pupil boundary → limbus boundary),
        so they show iris texture rather than the dark pupil, and both share one
        0° → 360° ruler (0° = screen right, 90° = down, 180° = left, 270° = up).
        Having them in the same boundary lets the doctor read the torsion
        straight off the panel.  The live strip can also be nudged left/right
        with the ``-`` / ``+`` control below it (display-only) to line it up
        with the Pentacam reference; the counter shows the offset in degrees.
        This is display-only and additive: the strip is
        a read-only reprojection of the existing detection geometry, scaled to
        the panel with its aspect ratio preserved (never a crop, squash or
        distortion of the iris), and nothing is fed to surgical control.
        """
        c = self._colors
        self._unwrap_outer = ttk.LabelFrame(
            parent,
            text="Iris Unwrap  —  Pentacam (top) / ELITA live (bottom)  "
                 "·  0° left → 360° right",
            padding=4,
        )
        self._unwrap_outer.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=(2, 0))
        self._unwrap_canvas = tk.Canvas(
            self._unwrap_outer,
            bg=c.CANVAS_BG,
            height=200,
            highlightthickness=0,
            borderwidth=0,
        )
        self._unwrap_canvas.pack(fill=tk.X, expand=False, padx=4, pady=(2, 2))

        # Manual strip-alignment controls below the unwrap display:
        self._unwrap_ctrl_frame = ttk.Frame(self._unwrap_outer)
        self._unwrap_ctrl_frame.pack(fill=tk.X, padx=4, pady=(0, 2))

        lbl_title = ttk.Label(
            self._unwrap_ctrl_frame,
            text="ELITA Live Strip Alignment:",
            font=("Segoe UI", 8),
            foreground=c.FG_SECONDARY,
        )
        lbl_title.pack(side=tk.LEFT, padx=(4, 6))

        self._elita_btn_minus = ttk.Button(
            self._unwrap_ctrl_frame,
            text="−",
            width=3,
            command=lambda: self._nudge_elita_shift(-1),
        )
        self._elita_btn_minus.pack(side=tk.LEFT, padx=(0, 2))
        self._elita_btn_minus.bind("<ButtonPress-1>", lambda _e: self._schedule_elita_repeat(-1))
        self._elita_btn_minus.bind("<ButtonRelease-1>", lambda _e: self._stop_elita_shift())
        self._elita_btn_minus.bind("<Leave>", lambda _e: self._stop_elita_shift())

        self._elita_shift_label = ttk.Label(
            self._unwrap_ctrl_frame,
            text="+0.0°",
            width=7,
            anchor="center",
            font=("Segoe UI", 8, "bold"),
            foreground=c.FG_PRIMARY,
        )
        self._elita_shift_label.pack(side=tk.LEFT, padx=2)
        self._elita_shift_label.bind("<Button-1>", lambda _e: self._reset_elita_shift())

        self._elita_btn_plus = ttk.Button(
            self._unwrap_ctrl_frame,
            text="+",
            width=3,
            command=lambda: self._nudge_elita_shift(+1),
        )
        self._elita_btn_plus.pack(side=tk.LEFT, padx=(2, 6))
        self._elita_btn_plus.bind("<ButtonPress-1>", lambda _e: self._schedule_elita_repeat(+1))
        self._elita_btn_plus.bind("<ButtonRelease-1>", lambda _e: self._stop_elita_shift())
        self._elita_btn_plus.bind("<Leave>", lambda _e: self._stop_elita_shift())

        self._elita_btn_reset = ttk.Button(
            self._unwrap_ctrl_frame,
            text="Reset",
            width=5,
            command=self._reset_elita_shift,
        )
        self._elita_btn_reset.pack(side=tk.LEFT, padx=(0, 4))

        self._elita_btn_auto = ttk.Button(
            self._unwrap_ctrl_frame,
            text="Auto Align",
            width=10,
            command=self._auto_align_elita_shift,
        )
        self._elita_btn_auto.pack(side=tk.LEFT, padx=(0, 6))

        self._elita_auto_status_label = ttk.Label(
            self._unwrap_ctrl_frame,
            text="",
            font=("Segoe UI", 8),
            foreground=c.FG_SECONDARY,
        )
        self._elita_auto_status_label.pack(side=tk.LEFT, padx=(2, 4))

        self._elita_unwrap_shift_cols: int = 0

        # Last unwrapped strips (BGR ndarrays): the Pentacam baseline and the
        # live frame, plus their Tk image references.
        self._unwrap_strip: Optional[np.ndarray] = None
        self._unwrap_photo: Optional[ImageTk.PhotoImage] = None
        self._elita_strip: Optional[np.ndarray] = None
        self._elita_photo: Optional[ImageTk.PhotoImage] = None
        self._elita_unwrap_last: float = 0.0
        self._unwrap_canvas.bind("<Configure>", lambda _e: self._redraw_unwrap())

    def _set_unwrap_strip(self, strip: Optional[np.ndarray]) -> None:
        """Show the Pentacam *strip* (a flattened iris) in the unwrap panel."""
        self._unwrap_strip = strip
        self._redraw_unwrap()

    def _set_elita_strip(self, strip: Optional[np.ndarray]) -> None:
        """Show the live ELITA *strip* beneath the Pentacam one."""
        self._elita_strip = strip
        self._redraw_unwrap()

    def _update_elita_unwrap(self) -> None:
        """Flatten the current live frame's iris into the panel.
        The strip is the **iris annulus** (pupil boundary → limbus boundary), so
        every row is iris texture rather than the dark pupil of a full "disc"
        unwrap.  It unwraps the *clean* live frame — never the drawn overlay —
        using the same geometry the cyclotorsion engine uses, with column 0 at
        0° (screen right) and column ``width`` at 360°, matching the shared
        ruler.  Read-only: no detection state is modified.  Only the area
        between the pupil (green) and limbus (blue) rings is unwrapped, so the
        pupil interior is never included; when either boundary is missing the
        previous strip is kept so the panel never flickers.
        """
        image = getattr(self, "_current_image", None)
        result = getattr(self, "_current_result", None)
        if image is None or result is None:
            return
        limbus = getattr(result, "limbus", None)
        if limbus is None or not getattr(limbus, "detected", False):
            return
        ellipse = getattr(limbus, "ellipse", None)
        if ellipse is None:
            return
        pupil = getattr(result, "pupil", None)
        pupil_ellipse = (
            getattr(pupil, "ellipse", None)
            if bool(getattr(pupil, "detected", False))
            else None
        )
        try:
            cx = float(ellipse.center_x)
            cy = float(ellipse.center_y)
            rx = float(ellipse.semi_major)
            ry = float(ellipse.semi_minor)
        except (AttributeError, TypeError, ValueError):
            return
        if not all(np.isfinite([cx, cy, rx, ry])) or rx <= 0.0 or ry <= 0.0:
            return
        now = time.perf_counter()
        if now - float(getattr(self, "_elita_unwrap_last", 0.0)) < 0.08:
            return  # ~12 Hz is plenty for a monitoring strip

        # Only the area between the pupil (green) and limbus (blue) rings — the
        # pupil interior is deliberately excluded, so there is no disc fallback.
        eye = getattr(self, "_selected_eye_var", None).get() if hasattr(self, "_selected_eye_var") else "OD"
        strip = self._annulus_strip(image, pupil_ellipse, ellipse, eye=eye)
        if strip is None:
            return
        self._elita_unwrap_last = now
        self._elita_strip = strip
        self._redraw_unwrap()

    @staticmethod
    def _as_ellipse_params(ellipse: Optional[Any]) -> Optional[Any]:
        """Normalise ellipse-like geometry to a real ``EllipseParams``.

        The live/optimised pipeline (:meth:`_adapt_frame_result`) and the
        display-only ring fallbacks (:func:`_make_circle`,
        :meth:`_preview_estimate_limbus`) hand the unwrap helpers lightweight
        ``SimpleNamespace`` objects that carry the centre, half-axes and angle
        but **not** ``EllipseParams.is_valid``.  :class:`IrisROIExtractor` reads
        that attribute, so a live strip silently failed to build and the panel
        stayed empty.  Objects that already validate are returned unchanged;
        anything else is rebuilt as an equivalent :class:`EllipseParams`, or
        ``None`` when the geometry is missing or non-finite.
        """
        if ellipse is None:
            return None
        if hasattr(ellipse, "is_valid"):
            return ellipse
        try:
            from pupil_tracking.utils.types import EllipseParams

            cx = getattr(ellipse, "center_x", None)
            cy = getattr(ellipse, "center_y", None)
            if cx is None or cy is None:
                center = getattr(ellipse, "center", None)
                if center is None:
                    return None
                cx, cy = center[0], center[1]

            semi_major = getattr(ellipse, "semi_major", None)
            semi_minor = getattr(ellipse, "semi_minor", None)
            if semi_major is None or semi_minor is None:
                axes = getattr(ellipse, "axes", None)
                if axes is not None:
                    try:
                        full = [float(v) for v in axes]
                    except (TypeError, ValueError):
                        full = []
                    if len(full) == 2:
                        semi_major = max(full) / 2.0
                        semi_minor = min(full) / 2.0
            if semi_major is None or semi_minor is None:
                radius = getattr(ellipse, "radius", None)
                if semi_major is None:
                    semi_major = radius
                if semi_minor is None:
                    semi_minor = radius
            if semi_major is None or semi_minor is None:
                return None

            semi_major = float(semi_major)
            semi_minor = float(semi_minor)
            if semi_minor > semi_major:
                semi_major, semi_minor = semi_minor, semi_major
            cx = float(cx)
            cy = float(cy)
            angle = float(getattr(ellipse, "angle_deg", 0.0) or 0.0)
            if not all(np.isfinite([cx, cy, semi_major, semi_minor, angle])):
                return None
            if semi_major <= 0.0 or semi_minor <= 0.0:
                return None
            return EllipseParams(
                center_x=cx,
                center_y=cy,
                semi_major=semi_major,
                semi_minor=semi_minor,
                angle_deg=angle,
            )
        except Exception:
            return None

    @staticmethod
    def _annulus_strip(
        image: np.ndarray,
        pupil_ellipse: Optional[Any],
        limbus_ellipse: Optional[Any],
        limbus_radius_fn: Optional[Any] = None,
        pupil_radius_fn: Optional[Any] = None,
        eye: str = "OD",
    ) -> Optional[np.ndarray]:
        """Unwrap the iris annulus (pupil → limbus) into a strip, or None.
        Column 0 is 0° (Nasal) and column ``width`` is 360°, matching the
        panel ruler.  Returns None when either boundary is missing or the
        geometry is implausible, in which case the caller leaves the strip empty.

        ``limbus_radius_fn`` / ``pupil_radius_fn`` (each ``f(angle_deg) -> px``)
        optionally replace a boundary with a per-angle radius, so a ring the user
        *pulled* out at one spot is followed exactly instead of being flattened
        to a circle.  Every other part of the geometry -- the centre, the other
        boundary, the insets and the plausibility gates -- is unchanged.
        """
        if image is None:
            return None
        pupil_ellipse = PupilTrackingGUI._as_ellipse_params(pupil_ellipse)
        limbus_ellipse = PupilTrackingGUI._as_ellipse_params(limbus_ellipse)
        if pupil_ellipse is None or limbus_ellipse is None:
            return None
        try:
            from pupil_tracking.iris.roi import IrisROIExtractor
            from pupil_tracking.iris.normalization import IrisNormalizer
            from pupil_tracking.cyclotorsion.polar_unwrap import (
                unwrap_iris,
                build_unwrap_maps,
                apply_unwrap_maps,
            )
            roi = IrisROIExtractor().build(pupil_ellipse, limbus_ellipse)
            if not getattr(roi, "valid", False):
                return None
            if limbus_radius_fn is None and pupil_radius_fn is None:
                return unwrap_iris(
                    image,
                    roi,
                    output_size=_UNWRAP_STRIP_SIZE,
                    angle_offset_deg=0.0,
                    eye=eye,
                )

            # A hand-pulled ring: keep the usual maths (centre, the *other*
            # boundary, and the insets the ROI carries) but take the pulled
            # boundary/boundaries from the per-angle profile.  Injecting it as
            # the normalizer leaves the vectorised map build untouched.
            class _ProfileBounds(IrisNormalizer):
                def radial_bounds(self, roi, angle_deg):
                    inner, outer = super().radial_bounds(roi, angle_deg)
                    if pupil_radius_fn is not None:
                        inner = float(pupil_radius_fn(angle_deg)) * (
                            1.0 + roi.inner_inset_frac
                        )
                    if limbus_radius_fn is not None:
                        outer = float(limbus_radius_fn(angle_deg)) * (
                            1.0 - roi.outer_inset_frac
                        )
                    return inner, outer

            maps = build_unwrap_maps(
                roi,
                _UNWRAP_STRIP_SIZE,
                0.0,
                normalizer=_ProfileBounds(),
                eye=eye,
            )
            if maps is None:
                return None
            return apply_unwrap_maps(image, maps)
        except Exception:
            return None

    @staticmethod
    def _disc_strip(
        image: np.ndarray,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        angle_deg: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Flatten the whole limbus disc (centre → limbus), offset 0.

        Retained for reference/tests, but no longer used by the panels: both the
        Pentacam and live strips now unwrap only the iris annulus.
        """
        if image is None:
            return None
        try:
            from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_disc
            return unwrap_disc(
                image,
                (cx, cy),
                rx,
                ry,
                angle_deg,
                output_size=_UNWRAP_STRIP_SIZE,
                angle_offset_deg=0.0,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Edge-line limbus reference (display-only fallback)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _edge_line_limbus(
        image: np.ndarray,
        center: Tuple[float, float],
        r_lo: float,
        r_hi: float,
        min_side: int = 6,
        grad_thresh: float = 12.0,
        contrast: float = 20.0,
    ) -> Optional[SimpleNamespace]:
        """Estimate the limbus from the two lines on the left/right of the eye.
        A real capture shows a *dark line next to a bright line* along the
        **left and right curve of the eye** (the iris/sclera border).  Those two
        near-vertical arcs are enough to place the limbus circle even when the
        detector's own confidence is low.  For every row inside the annulus
        ``r_lo..r_hi`` around ``center`` the strongest horizontal intensity
        transition on each side is collected — but only when a genuine
        black/white contrast exists across it — and an algebraic (Kasa) circle
        is fitted through the points, with outlier rejection and a final
        plausibility check (both curves present, low residual, radius inside the
        annulus, centre near the anchor, real vertical extent).
        Returns an ellipse namespace (``semi_major == semi_minor == radius``,
        ``angle_deg`` 0) or ``None`` when the lines are not consistent with a
        circle — e.g. a Pentacam report screenshot, whose panel gradient and
        table borders are not two eye curves.  Pure numpy/cv2: no Tk and no
        detector state, and nothing feeds back into the centration pipeline.
        """
        if image is None:
            return None
        try:
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3
                else image
            )
        except Exception:
            return None
        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return None
        cx, cy = float(center[0]), float(center[1])
        r_lo, r_hi = float(r_lo), float(r_hi)
        if not (
            np.isfinite(cx)
            and np.isfinite(cy)
            and np.isfinite(r_lo)
            and np.isfinite(r_hi)
            and r_lo > 0.0
            and r_hi > r_lo
        ):
            return None
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3) / 16.0
        half = max(4, int(round(0.02 * min(h, w))))  # black/white line half-width
        lx0, lx1 = max(0, int(cx - r_hi)), min(w, int(cx - r_lo))
        rx0, rx1 = max(0, int(cx + r_lo)), min(w, int(cx + r_hi))
        y0, y1 = max(0, int(cy - r_hi)), min(h, int(cy + r_hi))
        left: List[Tuple[float, float]] = []
        right: List[Tuple[float, float]] = []
        for y in range(y0, y1):
            row = gx[y]
            if lx1 - lx0 >= 3:
                seg = np.abs(row[lx0:lx1])
                idx = int(np.argmax(seg))
                if float(seg[idx]) >= grad_thresh:
                    x = lx0 + idx
                    xa, xb = max(0, x - half), min(w - 1, x + half)
                    if abs(float(blur[y, xb]) - float(blur[y, xa])) >= contrast:
                        left.append((float(x), float(y)))
            if rx1 - rx0 >= 3:
                seg = np.abs(row[rx0:rx1])
                idx = int(np.argmax(seg))
                if float(seg[idx]) >= grad_thresh:
                    x = rx0 + idx
                    xa, xb = max(0, x - half), min(w - 1, x + half)
                    if abs(float(blur[y, xb]) - float(blur[y, xa])) >= contrast:
                        right.append((float(x), float(y)))
        if len(left) < min_side or len(right) < min_side:
            return None  # not the two-curve signature of an eye

        def _fit_circle(points: np.ndarray):
            if points is None or len(points) < 6:
                return None
            xs, ys = points[:, 0], points[:, 1]
            mat = np.column_stack([2.0 * xs, 2.0 * ys, np.ones(len(points))])
            rhs = xs * xs + ys * ys
            try:
                sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                return None
            fx, fy, k = (float(v) for v in sol)
            rr2 = k + fx * fx + fy * fy
            if not np.isfinite(rr2) or rr2 <= 0.0:
                return None
            return fx, fy, float(np.sqrt(rr2))
        pts = np.array(left + right, dtype=np.float64)
        fit = _fit_circle(pts)
        if fit is None:
            return None
        fx, fy, radius = fit
        for _ in range(3):  # outlier rejection / refit
            rad = np.hypot(pts[:, 0] - fx, pts[:, 1] - fy)
            keep = np.abs(rad - radius) <= max(5.0, 0.10 * radius)
            if int(keep.sum()) < 6:
                break
            again = _fit_circle(pts[keep])
            if again is None:
                break
            fx, fy, radius = again
            pts = pts[keep]
        rad = np.hypot(pts[:, 0] - fx, pts[:, 1] - fy)
        rms = float(np.sqrt(np.mean((rad - radius) ** 2)))
        n_left = int(np.sum(pts[:, 0] < fx))
        n_right = int(len(pts) - n_left)
        y_span = float(pts[:, 1].max() - pts[:, 1].min())
        if not np.isfinite(radius) or radius <= 0.0:
            return None
        if n_left < min_side or n_right < min_side:
            return None
        if rms > max(3.0, 0.04 * radius):
            return None
        if not (r_lo <= radius <= r_hi):
            return None
        if math.hypot(fx - cx, fy - cy) > 0.5 * r_hi:
            return None
        if y_span < 0.5 * radius:
            return None
        return SimpleNamespace(
            center_x=fx,
            center_y=fy,
            radius=radius,
            semi_major=radius,
            semi_minor=radius,
            angle_deg=0.0,
        )

    @classmethod
    def _preview_edge_limbus(
        cls,
        image: np.ndarray,
        result: Any,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[SimpleNamespace]:
        """Edge-line limbus reference around the best available anchor.
        Anchors, in order: the detected **pupil** (usually still reliable on a
        low-confidence frame), the detected **limbus**, then the hand-drawn iris
        ROI box.  The first anchor whose black/white eye edges form a plausible
        circle wins; ``None`` means no edge-line reference is available, and
        :meth:`_preview_fallback_limbus` moves on to its next rung (a
        pupil-scaled estimate).
        """
        if image is None:
            return None
        h, w = image.shape[:2]
        anchors: List[Tuple[Tuple[float, float], float, float]] = []
        pupil = getattr(result, "pupil", None) if result is not None else None
        pupil_e = getattr(pupil, "ellipse", None)
        if pupil_e is not None and bool(getattr(pupil, "detected", False)):
            try:
                pr = float(pupil_e.semi_major)
                anchor = (float(pupil_e.center_x), float(pupil_e.center_y))
                anchors.append((anchor, 0.8 * pr, 3.6 * pr))
            except (TypeError, ValueError, AttributeError):
                pass
        limbus = getattr(result, "limbus", None) if result is not None else None
        limbus_e = getattr(limbus, "ellipse", None)
        if limbus_e is not None and bool(getattr(limbus, "detected", False)):
            try:
                lr = float(limbus_e.semi_major)
                anchor = (float(limbus_e.center_x), float(limbus_e.center_y))
                anchors.append((anchor, 0.55 * lr, 1.5 * lr))
            except (TypeError, ValueError, AttributeError):
                pass
        if roi_norm is not None:
            try:
                x0, y0, x1, y1 = (float(v) for v in roi_norm)
                anchor = ((x0 + x1) * 0.5 * w, (y0 + y1) * 0.5 * h)
                rrx = abs(x1 - x0) * 0.5 * w
                anchors.append((anchor, 0.35 * rrx, 1.7 * rrx))
            except (TypeError, ValueError):
                pass
        for anchor, r_lo, r_hi in anchors:
            r_hi = min(r_hi, 0.85 * min(h, w))
            r_lo = max(4.0, min(r_lo, 0.9 * r_hi))
            if r_hi <= r_lo:
                continue
            edge = cls._edge_line_limbus(image, anchor, r_lo, r_hi)
            if edge is not None:
                return edge
        return None

    @classmethod
    def _preview_fallback_limbus(
        cls,
        image: np.ndarray,
        result: Any,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Any, str]:
        """Best display-only limbus when the detector's own ring is not trusted.
        Returns ``(ellipse_or_None, note)``.  The blue ring must never simply be
        *missing* on an image that has an eye, so this walks a ladder:
        1. the two dark/bright eye edge lines (see :meth:`_preview_edge_limbus`);
        2. the hand-drawn iris ROI box (explicit user intent, an ellipse
           inscribed in the box) — see :meth:`_preview_roi_limbus`;
        3. a concentric estimate scaled from the pupil
           (see :meth:`_preview_estimate_limbus`).
        ``note`` is a short phrase ("from the eye edge lines", "from the iris
        ROI box", "estimated from the pupil") shown next to the dashed ring.
        """
        if image is None:
            return None, ""
        try:
            edge = cls._preview_edge_limbus(image, result, roi_norm)
        except Exception:
            edge = None
        if edge is not None:
            return edge, "from the eye edge lines"
        roi_edge = cls._preview_roi_limbus(image, roi_norm)
        if roi_edge is not None:
            return roi_edge, "from the iris ROI box"
        return cls._preview_estimate_limbus(image, result)

    @staticmethod
    def _preview_roi_limbus(
        image: np.ndarray,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> Any:
        """The ellipse inscribed in the hand-drawn iris ROI box (or ``None``)."""
        if image is None or roi_norm is None:
            return None
        try:
            h, w = image.shape[:2]
            x0, y0, x1, y1 = (float(v) for v in roi_norm)
            cx = (x0 + x1) * 0.5 * w
            cy = (y0 + y1) * 0.5 * h
            rx = abs(x1 - x0) * 0.5 * w
            ry = abs(y1 - y0) * 0.5 * h
        except (TypeError, ValueError):
            return None
        if rx < 4.0 or ry < 4.0:
            return None
        return SimpleNamespace(
            center_x=cx,
            center_y=cy,
            radius=0.5 * (rx + ry),
            semi_major=rx,
            semi_minor=ry,
            angle_deg=0.0,
        )

    @classmethod
    def _preview_estimate_limbus(
        cls,
        image: np.ndarray,
        result: Any,
    ) -> Tuple[Any, str]:
        """Estimate the limbus from the pupil when nothing better is available.
        The limbus is not a segmentation class of its own, so when the detector
        misses it (and there is no ROI box) the ring is placed concentrically
        with the pupil at ``_PENTACAM_LIMBUS_PUPIL_RATIO`` x the pupil radius,
        keeping the pupil's axis ratio and clamping to the image so the ring
        stays on screen.  This ratio was calibrated against the annotated
        clinical set (``clinical_data/clean/annotations/annotations.json``,
        12 eyes): median error 5 px, versus ~79 px for a raw image-edge search,
        which drifts onto the surgical suction ring.
        """
        if image is None:
            return None, ""
        h, w = image.shape[:2]
        pupil = getattr(result, "pupil", None) if result is not None else None
        pupil_e = getattr(pupil, "ellipse", None)
        if pupil_e is None or not bool(getattr(pupil, "detected", False)):
            return None, ""
        try:
            cx = float(pupil_e.center_x)
            cy = float(pupil_e.center_y)
            pr = float(pupil_e.semi_major)
            pb = float(pupil_e.semi_minor)
        except (TypeError, ValueError, AttributeError):
            return None, ""
        if not (np.isfinite(pr) and pr > 1.5):
            return None, ""
        ratio = pb / pr if (np.isfinite(pb) and pr > 0.0) else 1.0
        ratio = min(max(ratio, 0.6), 1.0)
        radius = _PENTACAM_LIMBUS_PUPIL_RATIO * pr
        border = min(cx, cy, w - cx, h - cy)
        if border > 1.0:
            radius = min(radius, 0.92 * border)
        if radius < 1.25 * pr:
            return None, ""
        return (
            SimpleNamespace(
                center_x=cx,
                center_y=cy,
                radius=radius,
                semi_major=radius,
                semi_minor=max(radius * ratio, 1.5),
                angle_deg=0.0,
            ),
            "estimated from the pupil",
        )

    # --- widen the preview ring to the eye's left/right edge lines ---
    # Off by default: the Pentacam preview (and its unwrap strip) show the
    # detector's own limbus exactly as the main window draws it.  Set True to
    # widen a *trusted* ring left/right to the eye's edge lines, which was
    # trialled for flat Pentacam report screenshots.
    _PENTACAM_WIDEN_TRUSTED_LIMBUS = False

    # A Pentacam *report screenshot* is a flat, almost edge-less montage, so the
    # detector can lock onto one sub-panel and return a limbus that is too small
    # -- the user's "underestimated on the left and right".  The eye's left and
    # right edges are the two strongest dark/bright vertical lines that bracket
    # it, so once a ring exists it is widened horizontally to reach them.  The
    # search only looks just outside the ring it was given, so it refines a ring
    # rather than inventing one, and it never shrinks or flattens a ring it
    # cannot improve.  Display-only; never fed back into the detection pipeline.
    _PENTACAM_EDGE_SEARCH_MIN = 0.90   # x of the ring's horizontal semi-axis
    _PENTACAM_EDGE_SEARCH_MAX = 1.30
    _PENTACAM_EDGE_MIN_STRENGTH = 5.0  # mean |dI/dx| to count as an edge line

    # Both edges must sit clear of the ring before it is widened.  A real eye
    # whose limbus is already correct makes the search re-find its own limbus
    # edge (at ~1.0x), so it stays untouched; only a ring that is too small on
    # BOTH sides gets widened.
    _PENTACAM_EDGE_MIN_CLEARANCE = 1.05

    @staticmethod
    def _eye_edge_lines(
        image: Any,
        center_x: float,
        center_y: float,
        radius_h: float,
    ) -> Tuple[Optional[int], Optional[int]]:
        """x of the strongest left/right vertical eye-edge lines.
        Only searches the band just outside the ring ``radius_h`` says is there,
        around a horizontal stripe through ``center_y``.  Returns
        ``(None, None)`` when no clear pair of lines is found.
        """
        if image is None or radius_h <= 0:
            return None, None
        try:
            import numpy as _np
            gray = image
            if getattr(gray, "ndim", 2) == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(_np.float32)
            h, w = gray.shape[:2]
            band = max(4, int(round(0.28 * radius_h)))
            y0 = max(0, int(round(center_y)) - band)
            y1 = min(h, int(round(center_y)) + band + 1)
            if y1 - y0 < 3:
                return None, None
            dx = _np.abs(_np.diff(gray[y0:y1, :], axis=1)).mean(axis=0)
            k = int(round(radius_h * 0.03))
            if k > 1:
                dx = _np.convolve(dx, _np.ones(k) / k, mode="same")

            # Keep clear of the image frame: its border is a "perfect" vertical
            # edge and would otherwise be mistaken for the eye's edge line.
            margin = max(8, int(round(0.015 * w)))
            outer_l = int(round(center_x - radius_h * PupilTrackingGUI._PENTACAM_EDGE_SEARCH_MAX))
            outer_r = int(round(center_x + radius_h * PupilTrackingGUI._PENTACAM_EDGE_SEARCH_MAX))
            inner_l = int(round(center_x - radius_h * PupilTrackingGUI._PENTACAM_EDGE_SEARCH_MIN))
            inner_r = int(round(center_x + radius_h * PupilTrackingGUI._PENTACAM_EDGE_SEARCH_MIN))
            lo = max(margin, outer_l)
            hi = min(w - 1 - margin, outer_r)
            x_left: Optional[int] = None
            x_right: Optional[int] = None
            if lo < inner_l:
                seg = dx[lo:inner_l]
                if seg.size:
                    i = int(_np.argmax(seg)) + lo
                    if float(dx[i]) >= PupilTrackingGUI._PENTACAM_EDGE_MIN_STRENGTH:
                        x_left = i
            if inner_r < hi:
                seg = dx[inner_r:hi + 1]
                if seg.size:
                    i = int(_np.argmax(seg)) + inner_r
                    if float(dx[i]) >= PupilTrackingGUI._PENTACAM_EDGE_MIN_STRENGTH:
                        x_right = i
            return x_left, x_right
        except Exception:
            return None, None

    @staticmethod
    def _widen_limbus_horizontally(image: Any, ellipse: Any) -> Any:
        """Widen ``ellipse`` left/right to the eye's edge lines.
        Returns ``ellipse`` unchanged unless two clear vertical edge lines
        bracket it, in which case the ring is centred between them and widened to
        reach both.  The vertical extent is preserved, so only the left and right
        grow.  Never shrinks, and leaves a tilted ellipse alone rather than
        flattening it.  Display-only.
        """
        if ellipse is None or image is None:
            return ellipse
        try:
            a = float(getattr(ellipse, "semi_major", 0.0))
            b = float(getattr(ellipse, "semi_minor", 0.0))
            ang = float(getattr(ellipse, "angle_deg", 0.0) or 0.0)
            if a <= 0 or b <= 0:
                return ellipse
            if abs(math.remainder(ang, 180.0)) > 15.0:
                return ellipse
            cx = float(getattr(ellipse, "center_x", 0.0))
            cy = float(getattr(ellipse, "center_y", 0.0))
            rad = math.radians(ang)
            r_h = math.hypot(a * math.cos(rad), b * math.sin(rad))
            r_v = math.hypot(a * math.sin(rad), b * math.cos(rad))
            if r_h <= 0 or r_v <= 0:
                return ellipse
            x_left, x_right = PupilTrackingGUI._eye_edge_lines(image, cx, cy, r_h)
            if x_left is None or x_right is None or x_right <= x_left:
                return ellipse
            new_h = (x_right - x_left) / 2.0
            if new_h < r_h * 1.05:      # not meaningfully wider: leave it alone
                return ellipse
            d_left = cx - x_left
            d_right = x_right - cx
            if d_left <= 0 or d_right <= 0:
                return ellipse

            # Both edge lines must be clear of the ring.  When the limbus is
            # already right, the search finds that same limbus edge (at ~1.0x of
            # the ring) and this rejects the "widening" -- so a correctly sized
            # ring on a real eye is never stretched.
            clear = r_h * PupilTrackingGUI._PENTACAM_EDGE_MIN_CLEARANCE
            if d_left < clear or d_right < clear:
                return ellipse
            if min(d_left, d_right) / max(d_left, d_right) < 0.60:
                return ellipse            # too lopsided to be the two eye edges
            return SimpleNamespace(
                center_x=(x_left + x_right) / 2.0,
                center_y=cy,
                radius=float(new_h),
                semi_major=float(new_h),
                semi_minor=float(r_v),
                angle_deg=0.0,
            )
        except Exception:
            return ellipse

    @staticmethod
    def _preview_limbus_ellipse(
        result: Any,
        reliable: bool,
        edge_limbus: Any = None,
        image: Any = None,
    ) -> Any:
        """The limbus ellipse the Pentacam preview should use for a state.
        A trusted detection uses the detected limbus; an unconfirmed one uses
        the fallback estimate (``edge_limbus`` -- the eye edge lines, the pupil,
        or the iris ROI) when the preview produced one.  ``None`` means there is
        nothing trustworthy to show.  A trusted detection's ring is returned
        verbatim, exactly as the main window draws it; only when
        ``_PENTACAM_WIDEN_TRUSTED_LIMBUS`` is set is it widened left/right to the
        eye's edge lines (see :meth:`_widen_limbus_horizontally`).  Display-only;
        never used by the detection pipeline.
        """
        if reliable:
            limbus = getattr(result, "limbus", None) if result is not None else None
            if bool(getattr(limbus, "detected", False)):
                ellipse = getattr(limbus, "ellipse", None)
                if PupilTrackingGUI._PENTACAM_WIDEN_TRUSTED_LIMBUS:
                    return PupilTrackingGUI._widen_limbus_horizontally(
                        image, ellipse
                    )
                return ellipse
            return None

        # The fallback already derives its ring from the eye's own curves (the
        # dark/bright edge lines, the pupil, or the ROI box), so it is not
        # widened again here.
        return edge_limbus

    @staticmethod
    def _unwrap_limbus_choice(
        result: Any,
        reliable: bool,
        edge_limbus: Any = None,
        image: Any = None,
    ) -> Tuple[Any, Any]:
        """Pick ``(limbus_ellipse, pupil_ellipse)`` for the Pentacam unwrap.
        A trusted detection uses the detected limbus/pupil.  An unconfirmed one
        drops them (so the strip never flattens a bogus circle) but still uses the
        **fallback limbus** when the preview estimated one (the dark/bright eye
        edge lines, the pupil, or the iris ROI).  A trusted detection's ring is
        used verbatim (widened left/right only when
        ``_PENTACAM_WIDEN_TRUSTED_LIMBUS`` is set), so the strip matches the blue
        ring on screen.  Display-only.
        """
        ellipse = PupilTrackingGUI._preview_limbus_ellipse(
            result, reliable, edge_limbus, image
        )
        pupil = getattr(result, "pupil", None) if result is not None else None
        pupil_ellipse = (
            getattr(pupil, "ellipse", None)
            if reliable and bool(getattr(pupil, "detected", False))
            else None
        )
        return ellipse, pupil_ellipse

    @staticmethod
    def _preview_pair_consistent(result: Any) -> bool:
        """Whether the detected pupil sits sanely inside the detected limbus.
        Used by :meth:`_pentacam_detection_reliable`: a low-confidence frame is
        still trusted when **both** structures were found and their geometry
        agrees (pupil roughly concentric with the limbus and smaller by a
        plausible iris/pupil ratio).  A full, coherent pupil/limbus pair is a
        real eye, whereas a lone low-confidence "limbus" on a report screenshot
        is not.
        """
        pupil = getattr(result, "pupil", None) if result is not None else None
        limbus = getattr(result, "limbus", None) if result is not None else None
        if not bool(getattr(pupil, "detected", False)):
            return False
        if not bool(getattr(limbus, "detected", False)):
            return False
        pupil_e = getattr(pupil, "ellipse", None)
        limbus_e = getattr(limbus, "ellipse", None)
        if pupil_e is None or limbus_e is None:
            return False
        try:
            pcx, pcy = float(pupil_e.center_x), float(pupil_e.center_y)
            pa, pb = float(pupil_e.semi_major), float(pupil_e.semi_minor)
            lcx, lcy = float(limbus_e.center_x), float(limbus_e.center_y)
            la, lb = float(limbus_e.semi_major), float(limbus_e.semi_minor)
        except (TypeError, ValueError, AttributeError):
            return False
        if not all(
            np.isfinite(v) for v in (pcx, pcy, pa, pb, lcx, lcy, la, lb)
        ):
            return False
        if min(pa, pb, la, lb) <= 0.0:
            return False

        # Pupil centres on the limbus (within half the limbus radius) ...
        if math.hypot(pcx - lcx, pcy - lcy) > 0.5 * max(la, lb):
            return False

        # ... and is smaller, by a plausible iris/pupil ratio.
        if la < 1.1 * pa or lb < 1.1 * pb:
            return False
        if la > 7.0 * pa or lb > 7.0 * pb:
            return False
        return True

    @staticmethod
    def _pentacam_detection_reliable(result: Any) -> bool:
        """Whether a Pentacam preview detection can be trusted (display-only).
        A Pentacam reference file is frequently a *report screenshot*
        (topography map, Scheimpflug section, colour scales and text) or an
        unrelated photograph rather than an iris capture.  On such an image the
        detector can still emit a "limbus" that merely tracks a smooth panel
        gradient — a false positive drawn as a misplaced, wrong-sized circle.
        A ``limbus`` is trusted when it was actually detected **and** either
        * the overall confidence reaches
          :data:`_PENTACAM_MIN_TRUSTED_CONFIDENCE` (a genuine capture), or
        * the detector also found a **pupil** whose geometry agrees with that
          limbus (see :meth:`_preview_pair_consistent`) and the confidence
          reaches :data:`_PENTACAM_MIN_PAIRED_CONFIDENCE` — a full, coherent
          pupil+limbus pair *is* a real eye, even on a lower-scoring frame.
        This only gates the preview overlay and the Pentacam strip; it never
        feeds back into the centration pipeline.
        """
        if result is None:
            return False
        limbus = getattr(result, "limbus", None)
        if not bool(getattr(limbus, "detected", False)):
            return False
        try:
            confidence = float(getattr(result, "overall_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if confidence >= _PENTACAM_MIN_TRUSTED_CONFIDENCE:
            return True
        if confidence >= _PENTACAM_MIN_PAIRED_CONFIDENCE:
            return PupilTrackingGUI._preview_pair_consistent(result)
        return False

    def _unwrap_pentacam(
        self,
        image: np.ndarray,
        limbus_ellipse: Optional[Any] = None,
        pupil_ellipse: Optional[Any] = None,
        limbus_radius_fn: Optional[Any] = None,
        pupil_radius_fn: Optional[Any] = None,
        eye: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Flatten the Pentacam iris into a strip for the main-window panel.

        Only the **iris annulus** (pupil boundary → limbus boundary) is
        unwrapped, so the strip shows the iris texture between the green pupil
        ring and the blue limbus ring rather than the dark pupil: the detected
        ``pupil_ellipse`` and ``limbus_ellipse`` are used together.

        ``limbus_radius_fn`` / ``pupil_radius_fn`` optionally replace a boundary
        with a per-angle radius ``f(angle_deg) -> px``: that is how a ring the
        user *pulled* out at one spot (a per-angle profile) drives the strip, so
        the strip follows that shape rather than a circle through it.

        If either boundary is missing, ``None`` is returned — the pupil area is
        never included, and there is no whole-disc or iris-ROI fallback.

        Column 0 is 0° (Nasal) and column ``width`` is 360°, matching the
        shared ruler and the live strip.  Display-only; nothing is fed back into
        the detection pipeline.
        """
        if image is None:
            return None
        if eye is None:
            eye = getattr(self, "_selected_eye_var", None).get() if hasattr(self, "_selected_eye_var") else "OD"

        # Only the area between the pupil (green) and limbus (blue) rings.
        return self._annulus_strip(
            image,
            pupil_ellipse,
            limbus_ellipse,
            limbus_radius_fn=limbus_radius_fn,
            pupil_radius_fn=pupil_radius_fn,
            eye=eye,
        )

    # ------------------------------------------------------------------ #
    # Iris zone grid (display-only; pure helpers so they test without Tk)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unwrap_zone_lines(
        sectors: int = _UNWRAP_ZONE_SECTORS,
        quadrants: int = _UNWRAP_ZONE_QUADRANTS,
    ) -> List[Tuple[float, bool]]:
        """Angular zone boundaries as ``(fraction, is_quadrant)`` pairs.
        ``fraction`` runs 0.0 at the left edge (0 deg) to 1.0 at the right edge
        (360 deg), matching the strip's column -> angle mapping.  ``is_quadrant``
        flags the coarser quadrant boundaries (0/90/180/270 deg for the default
        12 zones over 4 quadrants).
        """
        sectors = max(1, int(sectors))
        quadrants = max(1, int(quadrants))
        step = max(1, sectors // quadrants)
        return [
            (k / float(sectors), (k % step) == 0) for k in range(sectors + 1)
        ]

    @staticmethod
    def _unwrap_band_fracs(bands: int = _UNWRAP_ZONE_BANDS) -> List[float]:
        """Interior radial band boundaries as fractions of the strip height."""
        bands = max(1, int(bands))
        return [b / float(bands) for b in range(1, bands)]

    # ------------------------------------------------------------------ #
    # Live (ELITA) strip alignment — display-only left/right nudge
    # ------------------------------------------------------------------ #

    @staticmethod
    def _roll_strip(
        strip: Optional[np.ndarray], shift_cols: int
    ) -> Optional[np.ndarray]:
        """Rotate a strip horizontally by ``shift_cols`` columns (wraps round).
        Positive ``shift_cols`` moves image content to the **right**; the result
        is the same shape, so it is a pure display remap (no pixels are lost).
        """
        if strip is None or not shift_cols:
            return strip
        return np.roll(strip, int(shift_cols), axis=1)

    def _update_elita_shift_label(self) -> None:
        """Show the current live-strip nudge as a signed angle."""
        label = getattr(self, "_elita_shift_label", None)
        if label is None:
            return
        cols = int(getattr(self, "_elita_unwrap_shift_cols", 0))
        deg = cols * 360.0 / float(_UNWRAP_STRIP_SIZE[1])
        try:
            label.configure(text=f"{deg:+.1f}°")
        except tk.TclError:
            pass

    def _reset_elita_shift(self) -> None:
        """Reset the manual ELITA strip display shift to zero."""
        self._elita_unwrap_shift_cols = 0
        self._update_elita_shift_label()
        auto_lbl = getattr(self, "_elita_auto_status_label", None)
        if auto_lbl is not None:
            try:
                auto_lbl.configure(text="")
            except Exception:
                pass
        self._redraw_unwrap()

    def _nudge_elita_shift(self, direction: int) -> None:
        """Move the live strip right (+1) / left (-1) by one step.
        Display-only: it shifts which angle sits under each column of the *live*
        strip so it can be lined up with the Pentacam reference.  Clamped to
        +/- half a turn, beyond which the strip is simply the same content
        wrapped around and the counter would be meaningless.
        """
        total = _UNWRAP_STRIP_SIZE[1]
        step = max(1, int(round(_ELITA_SHIFT_STEP_DEG / 360.0 * total)))
        half = total // 2
        cols = int(getattr(self, "_elita_unwrap_shift_cols", 0))
        cols = max(-half, min(half, cols + int(direction) * step))
        self._elita_unwrap_shift_cols = cols
        self._update_elita_shift_label()
        self._redraw_unwrap()

    def _repeat_elita_shift(self, direction: int) -> None:
        """Auto-repeat while a nudge button is held down."""
        self._elita_shift_after_id = None
        half = _UNWRAP_STRIP_SIZE[1] // 2
        if int(getattr(self, "_elita_unwrap_shift_cols", 0)) in (-half, half):
            return  # already at the limit
        self._nudge_elita_shift(direction)
        self._elita_shift_after_id = self.root.after(
            45, lambda: self._repeat_elita_shift(direction)
        )

    def _schedule_elita_repeat(self, direction: int) -> None:
        """Button pressed: arm the auto-repeat after a short delay.
        The single step is done by the button's ``command`` on release, so a
        quick click always moves exactly one step even if this never fires.
        """
        self._stop_elita_shift()
        self._elita_shift_after_id = self.root.after(
            350, lambda: self._repeat_elita_shift(direction)
        )

    def _stop_elita_shift(self) -> None:
        """Button released (or pointer left): cancel any pending auto-repeat."""
        after_id = getattr(self, "_elita_shift_after_id", None)
        if after_id is not None:
            try:
                self.root.after_cancel(after_id)
            except Exception:
                pass
            self._elita_shift_after_id = None

    @staticmethod
    def _estimate_elita_unwrap_alignment(
        pentacam_strip: Optional[np.ndarray],
        elita_strip: Optional[np.ndarray],
        min_score: float = 0.25,
    ) -> Tuple[bool, int, float, str]:
        """Estimate the horizontal shift (in columns) to align the ELITA strip with the Pentacam strip.

        Reuses the project's cyclotorsion and iris matching engine
        (:func:`pupil_tracking.cyclotorsion.cyclotorsion_engine.estimate_cyclotorsion`)
        and iris feature verification
        (:func:`pupil_tracking.cyclotorsion.iris_features.extract_strip_features`).
        Calculates sub-pixel normalized cross-correlation along the angular axis
        after radial-mean profile normalization.

        Returns
        -------
        Tuple[bool, int, float, str]
            (success, shift_cols, score, status_message)
        """
        def _has_pixels(arr: Any) -> bool:
            return (
                arr is not None
                and getattr(arr, "size", 0) > 0
                and getattr(arr, "ndim", 0) >= 2
            )

        if not _has_pixels(pentacam_strip):
            return False, 0, 0.0, "Pentacam reference strip missing"
        if not _has_pixels(elita_strip):
            return False, 0, 0.0, "Live ELITA strip missing"

        p_arr = np.asarray(pentacam_strip)
        e_arr = np.asarray(elita_strip)

        if p_arr.ndim not in (2, 3) or e_arr.ndim not in (2, 3):
            return False, 0, 0.0, "Invalid strip dimensions"

        target_shape = p_arr.shape[:2]
        if e_arr.shape[:2] != target_shape:
            e_arr = cv2.resize(e_arr, (target_shape[1], target_shape[0]))

        # Create valid iris masks (ignoring black border / unexposed margins)
        if p_arr.ndim == 3:
            p_mask = (p_arr > 5).any(axis=2)
        else:
            p_mask = (p_arr > 5)

        if e_arr.ndim == 3:
            e_mask = (e_arr > 5).any(axis=2)
        else:
            e_mask = (e_arr > 5)

        # Check for sufficient iris texture using existing iris feature extraction
        try:
            from pupil_tracking.cyclotorsion.iris_features import (
                extract_strip_features,
                STATUS_NO_FEATURES,
                STATUS_EMPTY,
            )
            p_feat = extract_strip_features(p_arr, mask=p_mask)
            e_feat = extract_strip_features(e_arr, mask=e_mask)
            if (
                p_feat.status in (STATUS_NO_FEATURES, STATUS_EMPTY)
                or e_feat.status in (STATUS_NO_FEATURES, STATUS_EMPTY)
            ):
                return False, 0, 0.0, "Insufficient iris texture"
        except Exception:
            pass

        try:
            from pupil_tracking.cyclotorsion.cyclotorsion_engine import estimate_cyclotorsion

            cres = estimate_cyclotorsion(
                reference_strip=p_arr,
                target_strip=e_arr,
                reference_mask=p_mask,
                target_mask=e_mask,
                min_score=min_score,
            )
        except Exception as exc:
            return False, 0, 0.0, f"Error: {exc}"

        if not cres.valid or cres.score < min_score:
            reason = cres.reason or f"low correlation ({cres.score:.2f} < {min_score:.2f})"
            return False, 0, float(cres.score), f"Low confidence: {reason}"

        width = _UNWRAP_STRIP_SIZE[1]
        half = width // 2
        # cres.shift_cols is the roll applied to reference to match target.
        # To roll target (ELITA) to match reference (Pentacam), shift is -cres.shift_cols.
        needed_shift = -cres.shift_cols
        if target_shape[1] != width:
            needed_shift = needed_shift * (float(width) / float(target_shape[1]))

        shift_cols = int(round(((needed_shift + half) % width) - half))
        shift_cols = max(-half, min(half, shift_cols))
        deg = shift_cols * 360.0 / float(width)
        return True, shift_cols, float(cres.score), f"Aligned: {deg:+.1f}° (corr: {cres.score:.2f})"

    def _auto_align_elita_shift(self) -> bool:
        """Automatically estimate and apply the horizontal alignment offset to the live ELITA unwrap strip.

        Uses existing iris features and cross-correlation matching against the Pentacam reference.
        Never alters raw images, raw measurements, or calculated cyclotorsion angles.
        Preserves the current offset if matching fails or confidence is insufficient.
        """
        pentacam_strip = getattr(self, "_unwrap_strip", None)
        elita_strip = getattr(self, "_elita_strip", None)

        ok, shift_cols, score, msg = self._estimate_elita_unwrap_alignment(
            pentacam_strip, elita_strip
        )

        auto_lbl = getattr(self, "_elita_auto_status_label", None)
        if auto_lbl is not None:
            try:
                auto_lbl.configure(text=msg)
            except Exception:
                pass

        status_var = getattr(self, "_status_var", None)
        if status_var is not None:
            try:
                status_var.set(f"ELITA unwrap auto-align: {msg}")
            except Exception:
                pass

        if not ok:
            return False

        self._elita_unwrap_shift_cols = int(shift_cols)
        self._update_elita_shift_label()
        self._redraw_unwrap()
        return True

    def _redraw_unwrap(self) -> None:
        """Render the Pentacam + live ELITA strips (shared ruler) on the panel."""
        canvas = getattr(self, "_unwrap_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
        except tk.TclError:
            return
        if cw <= 1 or ch <= 1:
            return

        def _has_pixels(arr: Any) -> bool:
            return (
                arr is not None
                and getattr(arr, "size", 0) > 0
                and getattr(arr, "ndim", 0) >= 2
            )
        pentacam_strip = getattr(self, "_unwrap_strip", None)
        elita_strip = getattr(self, "_elita_strip", None)
        manual_shift = int(getattr(self, "_elita_unwrap_shift_cols", 0))
        if manual_shift != 0 and _has_pixels(elita_strip):
            elita_strip = self._roll_strip(elita_strip, manual_shift)

        rows = (
            ("Pentacam", pentacam_strip),
            ("ELITA live", elita_strip),
        )
        if not any(_has_pixels(s) for _label, s in rows):
            eye = getattr(self, "_selected_eye_var", None)
            eye_str = eye.get() if eye and eye.get() in ("OD", "OS") else "selected eye"
            pid = getattr(self, "_patient_id_var", None)
            pid_val = pid.get().strip() if pid else ""
            if pid_val:
                empty_msg = f"No Pentacam reference image saved for {eye_str} (Patient: {pid_val})\nLoad a Pentacam image in Add PX or start live video"
            else:
                empty_msg = f"No Pentacam reference image loaded for {eye_str}\nLoad a Pentacam image in Add PX or start live video"
            canvas.create_text(
                cw // 2,
                ch // 2,
                text=empty_msg,
                fill=self._colors.FG_TERTIARY,
                font=("Segoe UI", 9),
                justify=tk.CENTER,
            )
            return
        ruler_h = 16
        pad = 3
        side_pad = 4

        src_h, src_w = _UNWRAP_STRIP_SIZE
        for _strip in (
            pentacam_strip,
            elita_strip,
        ):
            if _has_pixels(_strip):
                src_h, src_w = int(_strip.shape[0]), int(_strip.shape[1])
                break

        aspect = float(src_h) / float(src_w)
        avail_w = max(1, cw - side_pad * 2)

        # Scale proportionally to fit the available area without distortion.
        # Height for each strip at full available width:
        target_disp_h = max(1, int(round(avail_w * aspect)))
        needed_h = ruler_h + pad * 3 + 2 * target_disp_h + 4

        # Dynamically fit the canvas height to avoid vertical dead space,
        # capped so it preserves space for the main viewport.
        min_canvas_h = 130
        max_canvas_h = 360
        clamped_canvas_h = max(min_canvas_h, min(max_canvas_h, needed_h))

        if hasattr(canvas, "cget") and hasattr(canvas, "configure"):
            try:
                cur_h = int(canvas.cget("height"))
                if abs(cur_h - clamped_canvas_h) >= 4:
                    canvas.configure(height=clamped_canvas_h)
            except Exception:
                pass

        # Determine effective dimensions respecting both width and visible height:
        avail_h = ch
        row_h = max(1, (avail_h - ruler_h - pad * 3) // 2)
        scale = min(avail_w / float(src_w), row_h / float(src_h))
        disp_w = max(1, int(round(src_w * scale)))
        disp_h = max(1, int(round(src_h * scale)))
        x0 = (cw - disp_w) // 2

        # Shared 0° → 360° ruler:
        # Col 0 = 0° (Nasal), 25% = 90° (Inferior), 50% = 180° (Temporal),
        # 75% = 270° (Superior), 100% = 360° (Nasal).
        for frac, label, anchor in (
            (0.0, "0°", "w"),
            (0.25, "90°", "center"),
            (0.5, "180°", "center"),
            (0.75, "270°", "center"),
            (1.0, "360°", "e"),
        ):
            rx = x0 + int(round(disp_w * frac))
            canvas.create_text(
                rx,
                ruler_h // 2,
                text=label,
                fill=self._colors.FG_TERTIARY,
                font=("Consolas", 7),
                anchor=anchor,
            )
            canvas.create_line(
                rx, ruler_h - 2, rx, ruler_h + pad,
                fill="#555555",
            )
        for i, (label, strip) in enumerate(rows):
            y0 = ruler_h + pad + i * (disp_h + pad)
            if not _has_pixels(strip):
                eye = getattr(self, "_selected_eye_var", None)
                eye_str = eye.get() if eye and eye.get() in ("OD", "OS") else ""
                if label.startswith("Pentacam"):
                    pid = getattr(self, "_patient_id_var", None)
                    pid_val = pid.get().strip() if pid else ""
                    pid_suffix = f" (Patient: {pid_val})" if pid_val else ""
                    missing_msg = f"Pentacam ({eye_str}): No reference image saved for this eye{pid_suffix}"
                else:
                    missing_msg = f"{label} — no live feed active"
                canvas.create_text(
                    x0 + disp_w // 2,
                    y0 + max(8, disp_h // 2),
                    text=missing_msg,
                    fill=self._colors.FG_TERTIARY,
                    font=("Segoe UI", 8),
                )
                continue
            try:
                disp = cv2.resize(
                    strip, (disp_w, disp_h), interpolation=cv2.INTER_AREA
                )
                if disp.ndim == 2:
                    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
                # Mirror the normal window's RGB / Grayscale mode: when the
                # user switches the main display to grayscale, both strips
                # switch with it (same _convert_display_frame logic: FORCE =
                # always gray, AUTO = gray when the detector says the input
                # is grayscale, OFF = original colours).  Display-only, and
                # skipped silently when the mode variable isn't present
                # (tests / headless use).
                try:
                    if getattr(self, "_grayscale_mode_var", None) is not None:
                        disp = self._convert_display_frame(disp)
                except Exception:
                    pass
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            except Exception:
                continue
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.create_image(x0, y0, anchor=tk.NW, image=photo)
            if label.startswith("Pentacam"):
                self._unwrap_photo = photo
            else:
                self._elita_photo = photo
            canvas.create_text(
                x0 + 3,
                y0 + 1,
                text=label,
                anchor=tk.NW,
                fill=self._colors.FG_TERTIARY,
                font=("Consolas", 7),
            )

        # Iris zone grid: 12 angular sectors (30 deg = one clock hour) over the
        # 4 quadrants, drawn over BOTH strips so a feature at the same angle
        # lines up between the Pentacam and live strips.  Quadrant boundaries
        # (0/90/180/270 deg) are drawn a little brighter; 4 radial band lines
        # subdivide each strip from the pupil boundary to the limbus.  Toggle
        # via View -> Show Iris Zone Grid.  Display-only: the strip pixels and
        # their column -> angle mapping are untouched.
        grid_top = ruler_h + pad
        grid_bottom = ruler_h + pad + 2 * disp_h + pad
        zones_var = getattr(self, "_show_unwrap_zones", None)
        show_zones = True if zones_var is None else bool(zones_var.get())
        if show_zones:
            for frac, is_quadrant in self._unwrap_zone_lines():
                gx = x0 + int(round(disp_w * frac))
                canvas.create_line(
                    gx, grid_top, gx, grid_bottom,
                    fill="#6e6e6e" if is_quadrant else "#3d3d3d",
                )
            for i in range(2):
                y0 = ruler_h + pad + i * (disp_h + pad)
                for band_frac in self._unwrap_band_fracs():
                    by = y0 + int(round(disp_h * band_frac))
                    canvas.create_line(
                        x0, by, x0 + disp_w, by, fill="#333333",
                    )

    # ================================================================
    # Settings Panel
    # ================================================================

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sn = ("Consolas", 9)
        sw = 24
        storage_lf = ttk.LabelFrame(sf, text="Patient Data Storage", padding=8)
        storage_lf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Label(storage_lf, text="Set this once during installation. You can change it later.", style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 6))
        storage_row = ttk.Frame(storage_lf)
        storage_row.pack(fill=tk.X)
        ttk.Entry(storage_row, textvariable=self._storage_root_var, font=sn).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(storage_row, text="Browse…", command=self._choose_storage_root).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(storage_lf, text="Save storage location", command=self._save_storage_settings).pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(storage_lf, text="Structure: Centration / [PX_ID_Name / OD or OS /] raw video + overlay video + sync CSV + Pentacam", style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(storage_lf, textvariable=self._recording_location_var, style="Tiny.TLabel", wraplength=430, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 8 of 12 — Grayscale section in Settings
        # ══════════════════════════════════════════════════════════
        g_lf = ttk.LabelFrame(sf, text="🔲 Grayscale Mode", padding=8)
        g_lf.pack(fill=tk.X, padx=4, pady=4)

        g_desc = ttk.Label(
            g_lf,
            text=(
                "Convert display to grayscale (like IR camera).\n"
                "Detection still works — overlays shown in colour.\n"
                "Press [G] to toggle quickly."
            ),
            style="Muted.TLabel",
            justify=tk.LEFT,
        )
        g_desc.pack(anchor=tk.W, pady=(0, 6))

        g_row = ttk.Frame(g_lf)
        g_row.pack(fill=tk.X, pady=2)
        ttk.Label(
            g_row,
            text="Mode:",
            font=sn,
            width=12,
        ).pack(side=tk.LEFT)

        for mode_val, mode_label, mode_desc in [
            ("off", "RGB (Original)", "Show original colour image"),
            ("auto", "Auto-Detect", "Grayscale only if input is grayscale"),
            ("force", "Force Grayscale", "Always show as grayscale (IR look)"),
        ]:
            rb = ttk.Radiobutton(
                g_lf,
                text=f"{mode_label}  —  {mode_desc}",
                variable=self._grayscale_mode_var,
                value=mode_val,
                command=self._on_grayscale_changed,
            )
            rb.pack(anchor=tk.W, padx=(20, 0), pady=1)

        self._gray_settings_status = tk.StringVar(value="Current: RGB")
        ttk.Label(
            g_lf,
            textvariable=self._gray_settings_status,
            font=sn,
            foreground=c.ACCENT,
        ).pack(anchor=tk.W, pady=(6, 0))
        # ══════════════════════════════════════════════════════════
        # END GRAYSCALE GUI 8
        # ══════════════════════════════════════════════════════════

        p_lf = ttk.LabelFrame(sf, text="Pipeline", padding=8)
        p_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            p_lf,
            text="Live Preset:",
            font=sn,
        ).pack(anchor=tk.W)
        for preset_value, preset_label in [
            ("max_accuracy", "Max Accuracy"),
            ("balanced", "Balanced"),
            ("low_latency", "Low Latency"),
        ]:
            ttk.Radiobutton(
                p_lf,
                text=preset_label,
                variable=self._performance_preset_var,
                value=preset_value,
                command=self._apply_performance_preset,
            ).pack(anchor=tk.W, padx=(20, 0), pady=1)

        ttk.Checkbutton(
            p_lf,
            text="Use Optimised Pipeline (when available)",
            variable=self._use_optimized_var,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="FP16 Half-Precision",
            variable=self._fp16_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)
        ttk.Checkbutton(
            p_lf,
            text="torch.compile (JIT)",
            variable=self._compile_var,
            command=self._invalidate_fast_engine,
        ).pack(anchor=tk.W)

        avail = "✓ Available" if _FAST_PIPELINE_AVAILABLE else "✗ Not installed"
        ttk.Label(
            p_lf,
            text=f"Fast pipeline: {avail}",
            style="Muted.TLabel",
            foreground=(c.SURGICAL if _FAST_PIPELINE_AVAILABLE else c.INSUFFICIENT),
        ).pack(anchor=tk.W, pady=(4, 0))

        v_lf = ttk.LabelFrame(sf, text="Video Processing", padding=8)
        v_lf.pack(fill=tk.X, padx=4, pady=4)

        r_row = ttk.Frame(v_lf)
        r_row.pack(fill=tk.X, pady=2)
        ttk.Label(r_row, text="Inference Resolution:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._res_display = tk.StringVar(value=str(self._resolution_var.get()))
        ttk.Scale(
            r_row,
            from_=192,
            to=512,
            variable=self._resolution_var,
            command=lambda v: self._res_display.set(str(int(float(v)))),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(r_row, textvariable=self._res_display, font=sn, width=4).pack(
            side=tk.LEFT
        )

        s_row = ttk.Frame(v_lf)
        s_row.pack(fill=tk.X, pady=2)
        ttk.Label(s_row, text="Frame Stride:", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            s_row,
            from_=1,
            to=30,
            textvariable=self._stride_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(s_row, text="(1 = every frame)", font=sn).pack(side=tk.LEFT)

        f_row = ttk.Frame(v_lf)
        f_row.pack(fill=tk.X, pady=2)
        ttk.Label(f_row, text="Camera Target FPS:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            f_row,
            from_=5.0,
            to=60.0,
            textvariable=self._target_fps_var,
            width=5,
            font=sn,
            increment=5.0,
        ).pack(side=tk.LEFT, padx=4)

        t_lf = ttk.LabelFrame(sf, text="ROI & Tracking", padding=8)
        t_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Checkbutton(
            t_lf,
            text="Enable ROI Tracking",
            variable=self._roi_var,
        ).pack(anchor=tk.W)
        ttk.Label(
            t_lf,
            textvariable=self._roi_status_var,
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 4))

        rc_row = ttk.Frame(t_lf)
        rc_row.pack(fill=tk.X, pady=2)
        ttk.Label(rc_row, text="ROI Cache (frames):", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        ttk.Spinbox(
            rc_row,
            from_=1,
            to=30,
            textvariable=self._roi_cache_var,
            width=5,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        kp_row = ttk.Frame(t_lf)
        kp_row.pack(fill=tk.X, pady=2)
        ttk.Label(kp_row, text="Kalman Process Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._kp_display = tk.StringVar(value=f"{self._kalman_process_var.get():.3f}")
        ttk.Scale(
            kp_row,
            from_=0.001,
            to=0.5,
            variable=self._kalman_process_var,
            command=lambda v: self._kp_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(kp_row, textvariable=self._kp_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        km_row = ttk.Frame(t_lf)
        km_row.pack(fill=tk.X, pady=2)
        ttk.Label(km_row, text="Kalman Measure Noise:", font=sn, width=sw).pack(
            side=tk.LEFT
        )
        self._km_display = tk.StringVar(value=f"{self._kalman_measure_var.get():.3f}")
        ttk.Scale(
            km_row,
            from_=0.01,
            to=1.0,
            variable=self._kalman_measure_var,
            command=lambda v: self._km_display.set(f"{float(v):.3f}"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(km_row, textvariable=self._km_display, font=sn, width=6).pack(
            side=tk.LEFT
        )

        cal_lf = ttk.LabelFrame(sf, text="Calibration & Physical Units", padding=8)
        cal_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            cal_lf,
            text="Select physical scale derivation method:",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))

        for mode_val, mode_text in [
            ("ANATOMICAL_ANCHOR", "Anatomical Anchor (Cornea ≈ 12.0 mm)"),
            ("FIXED_PIXEL_SCALE", "Fixed Pixel Scale (Manual / External Target)"),
            ("RING_REFLECTION", "Ring Reflection (Purkinje / Placido Ring)"),
        ]:
            ttk.Radiobutton(
                cal_lf,
                text=mode_text,
                variable=self._calibration_mode_var,
                value=mode_val,
                command=lambda: self._schedule_live_settings_apply("calibration"),
            ).pack(anchor=tk.W, padx=(8, 0), pady=1)

        px_mm_row = ttk.Frame(cal_lf)
        px_mm_row.pack(fill=tk.X, pady=2)
        ttk.Label(px_mm_row, text="Fixed Scale (px/mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            px_mm_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        cornea_row = ttk.Frame(cal_lf)
        cornea_row.pack(fill=tk.X, pady=2)
        ttk.Label(cornea_row, text="Assumed Cornea (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            cornea_row,
            from_=8.0,
            to=15.0,
            increment=0.1,
            textvariable=self._corneal_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ring_row = ttk.Frame(cal_lf)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ref Ring Dia (mm):", font=sn, width=sw).pack(side=tk.LEFT)
        ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=20.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
            font=sn,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Button(
            cal_lf,
            text="⚡ Open Calibration & WTW Wizard…",
            command=self._open_calibration_wizard,
        ).pack(anchor=tk.W, pady=(6, 2))

        fill_lf = ttk.LabelFrame(sf, text="Circle Fill Shading", padding=8)

        fill_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(
            fill_lf,
            text="Fill the pupil (green) and limbus (blue) circles.\n0 = no fill, 100 = fully coloured.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        for label_text, var, disp_attr in [
            ("Pupil (green) %:", self._pupil_fill_alpha_var, "_pupil_fill_display"),
            ("Limbus (blue) %:", self._limbus_fill_alpha_var, "_limbus_fill_display"),
        ]:
            disp_var = tk.StringVar(value="0")
            setattr(self, disp_attr, disp_var)
            row = ttk.Frame(fill_lf)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label_text, font=sn, width=sw).pack(side=tk.LEFT)
            _v = var  # capture for lambda
            _d = disp_var
            ttk.Scale(
                row,
                from_=0,
                to=100,
                variable=_v,
                command=lambda v, d=_d: (
                    d.set(str(int(float(v)))),
                    self._refresh_display(),
                ),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            ttk.Label(row, textvariable=disp_var, font=sn, width=4).pack(side=tk.LEFT)

        a_lf = ttk.LabelFrame(sf, text="Actions", padding=8)
        a_lf.pack(fill=tk.X, padx=4, pady=4)

        ttk.Button(
            a_lf,
            text="Rebuild Inference Engine",
            command=self._rebuild_engine_ui,
        ).pack(anchor=tk.W)

        self._engine_status_var = tk.StringVar(value="Engine: not initialised")
        ttk.Label(
            a_lf,
            textvariable=self._engine_status_var,
            font=sn,
        ).pack(anchor=tk.W, pady=(4, 0))

        note = (
            "Most settings apply live. Pipeline and resolution\n"
            "changes restart the active stream automatically\n"
            "when needed. Use 'Rebuild Inference Engine'\n"
            "to refresh the fast path immediately. Press [G]\n"
            "to toggle grayscale mode at any time."
        )
        ttk.Label(
            sf,
            text=note,
            style="Tiny.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(8, 4))

    # ================================================================
    # Measurements Panel
    # ================================================================

    def _build_measurements_panel(self, parent: ttk.Frame) -> None:
        c = self._colors

        canvas = tk.Canvas(
            parent,
            highlightthickness=0,
            bg=c.BG_SECONDARY,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        scroll_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(scroll_window, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def add_card(
            pf,
            title: str,
            style_name: str,
            row: int,
            column: int,
            columnspan: int = 1,
        ) -> ttk.Frame:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=6)
            card.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky="nsew",
                padx=3,
                pady=3,
            )
            ttk.Label(
                card,
                text=title,
                style=style_name,
            ).pack(anchor=tk.W, pady=(0, 4))
            body = ttk.Frame(card, style="MetricCard.TFrame")
            body.pack(fill=tk.X)
            body.columnconfigure(1, weight=1)
            return body

        def add_summary_card(pf, title: str, column: int) -> Tuple[tk.StringVar, ttk.Label]:
            card = ttk.Frame(pf, style="MetricCard.TFrame", padding=10)
            card.grid(row=0, column=column, sticky="nsew", padx=3, pady=2)
            card.pack_propagate(False)
            ttk.Label(card, text=title, style="CardKey.TLabel").pack(anchor=tk.W)
            var = tk.StringVar(value="---")
            label = ttk.Label(
                card,
                textvariable=var,
                style="CardValue.TLabel",
                width=18,
                anchor="w",
            )
            label.pack(anchor=tk.W, pady=(4, 0))
            return var, label

        def add_row(pf, label_text, width=16):
            row = ttk.Frame(pf, style="MetricCard.TFrame")
            row.pack(fill=tk.X, pady=1)
            row.columnconfigure(1, weight=1)
            ttk.Label(
                row,
                text=label_text,
                style="CardKey.TLabel",
                width=width,
                anchor="w",
            ).grid(row=0, column=0, sticky="w")
            var = tk.StringVar(value="---")
            ttk.Label(
                row,
                textvariable=var,
                style="CardValueSmall.TLabel",
                anchor="e",
                justify=tk.RIGHT,
            ).grid(row=0, column=1, sticky="ew")
            return var

        summary_outer = ttk.Frame(scroll_frame)
        summary_outer.pack(fill=tk.X, pady=(8, 6))
        summary_outer.columnconfigure(0, weight=1)
        summary_outer.columnconfigure(1, weight=1)
        summary_outer.columnconfigure(2, weight=1)
        summary_outer.columnconfigure(3, weight=1)
        self._summary_quality_var, self._summary_quality_label = add_summary_card(
            summary_outer, "Quality", 0
        )
        self._summary_tracking_var, self._summary_tracking_label = add_summary_card(
            summary_outer, "Tracking", 1
        )
        self._summary_latency_var, self._summary_latency_label = add_summary_card(
            summary_outer, "Latency", 2
        )
        self._summary_pipeline_var, self._summary_pipeline_label = add_summary_card(
            summary_outer, "Pipeline", 3
        )
        ttk.Label(
            scroll_frame,
            textvariable=self._patient_summary_var,
            style="Muted.TLabel",
            anchor="w",
        ).pack(fill=tk.X, padx=6, pady=(0, 6))

        cards_outer = ttk.Frame(scroll_frame, style="Primary.TFrame")
        cards_outer.pack(fill=tk.BOTH, expand=True)
        cards_outer.columnconfigure(0, weight=1, uniform="measurement_cards")
        cards_outer.columnconfigure(1, weight=1, uniform="measurement_cards")

        pupil_frame = add_card(cards_outer, "PUPIL", "PupilHeader.TLabel", 0, 0)
        self._pv: Dict[str, tk.StringVar] = {}
        self._pv["center"] = add_row(pupil_frame, "Center:")
        self._pv["diameter_px"] = add_row(pupil_frame, "Diameter (px):")
        self._pv["diameter_mm"] = add_row(pupil_frame, "Diameter (mm):")
        self._pv["semi_major"] = add_row(pupil_frame, "Semi-Major (px):")
        self._pv["semi_major_mm"] = add_row(pupil_frame, "Semi-Major (mm):")
        self._pv["semi_minor"] = add_row(pupil_frame, "Semi-Minor (px):")
        self._pv["semi_minor_mm"] = add_row(pupil_frame, "Semi-Minor (mm):")
        self._pv["angle"] = add_row(pupil_frame, "Angle:")
        self._pv["fit_type"] = add_row(pupil_frame, "Fit Type:")
        self._pv["confidence"] = add_row(pupil_frame, "Confidence:")
        self._pv["quality"] = add_row(pupil_frame, "Quality:")

        limbus_frame = add_card(cards_outer, "LIMBUS", "LimbusHeader.TLabel", 0, 1)
        self._lv: Dict[str, tk.StringVar] = {}
        self._lv["center"] = add_row(limbus_frame, "Center:")
        self._lv["diameter_px"] = add_row(limbus_frame, "Diameter (px):")
        self._lv["diameter_mm"] = add_row(limbus_frame, "Diameter (mm):")
        self._lv["semi_major"] = add_row(limbus_frame, "Semi-Major (px):")
        self._lv["semi_major_mm"] = add_row(limbus_frame, "Semi-Major (mm):")
        self._lv["semi_minor"] = add_row(limbus_frame, "Semi-Minor (px):")
        self._lv["semi_minor_mm"] = add_row(limbus_frame, "Semi-Minor (mm):")
        self._lv["angle"] = add_row(limbus_frame, "Angle:")
        self._lv["fit_type"] = add_row(limbus_frame, "Fit Type:")
        self._lv["confidence"] = add_row(limbus_frame, "Confidence:")
        self._lv["quality"] = add_row(limbus_frame, "Quality:")

        offset_frame = add_card(
            cards_outer, "CORNEAL CENTRE & OFFSET", "OffsetHeader.TLabel", 1, 0
        )
        self._ov: Dict[str, tk.StringVar] = {}
        self._ov["corneal_center"] = add_row(offset_frame, "Corneal Centre:")
        self._ov["corneal_reference"] = add_row(offset_frame, "Reference:")
        self._ov["ring_center"] = add_row(offset_frame, "Ring Centre:")
        self._ov["ring_diameter_px"] = add_row(offset_frame, "Ring Dia (px):")
        self._ov["ring_diameter_mm"] = add_row(offset_frame, "Ring Dia (mm):")
        self._ov["offset_px"] = add_row(offset_frame, "Offset (px):")
        self._ov["offset_mm"] = add_row(offset_frame, "Offset (mm):")
        self._ov["offset_vec_px"] = add_row(offset_frame, "Offset dX,dY px:")
        self._ov["offset_vec_mm"] = add_row(offset_frame, "Offset dX,dY mm:")
        self._ov["offset_angle"] = add_row(offset_frame, "Offset Angle:")
        self._ov["pupil_limbus_ratio"] = add_row(offset_frame, "Pupil/Limbus:")

        calib_frame = add_card(cards_outer, "CALIBRATION", "CalibHeader.TLabel", 1, 1)
        self._cv_vars: Dict[str, tk.StringVar] = {}
        self._cv_vars["source"] = add_row(calib_frame, "Source:")
        self._cv_vars["scale_px"] = add_row(calib_frame, "px/mm:")
        self._cv_vars["scale_mm"] = add_row(calib_frame, "mm/px:")
        self._cv_vars["reference"] = add_row(calib_frame, "Reference:")

        wtw_frame = add_card(
            cards_outer, "CORNEAL DIMENSIONS (WTW)", "LimbusHeader.TLabel", 2, 0, 2
        )
        self._wtw_vars: Dict[str, tk.StringVar] = {}
        self._wtw_vars["horizontal"] = add_row(wtw_frame, "Horizontal WTW:")
        self._wtw_vars["vertical"] = add_row(wtw_frame, "Vertical WTW:")
        self._wtw_vars["mean"] = add_row(wtw_frame, "Mean WTW:")

        proc_frame = add_card(cards_outer, "PROCESSING", "ProcHeader.TLabel", 3, 0, 2)
        self._proc_time_var = add_row(proc_frame, "Proc. Time:")
        self._latency_var = add_row(proc_frame, "Latency:")
        self._latency_avg_var = add_row(proc_frame, "Latency Avg:")
        self._drop_var = add_row(proc_frame, "Dropped/Stale:")
        self._tracking_state_var = add_row(proc_frame, "Tracking:")
        self._fps_var = add_row(proc_frame, "FPS:")
        self._frame_var = add_row(proc_frame, "Frame:")
        self._image_size_var = add_row(proc_frame, "Image Size:")
        self._pipeline_var = add_row(proc_frame, "Pipeline:")

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 9 of 12 — Grayscale info in measurements
        # ══════════════════════════════════════════════════════════
        self._gray_mode_var_display = add_row(proc_frame, "Grayscale:")


    def _build_details_panel(self, parent: ttk.Frame) -> None:
        c = self._colors
        self._details_text = tk.Text(
            parent,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg=c.BG_INPUT,
            fg=c.FG_PRIMARY,
            state=tk.DISABLED,
            height=30,
            insertbackground=c.FG_PRIMARY,
            selectbackground=c.ACCENT_DIM,
            selectforeground=c.FG_PRIMARY,
            borderwidth=0,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self._details_text.yview,
        )
        self._details_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._details_text.pack(fill=tk.BOTH, expand=True)

    def _build_status_bar(self) -> None:
        status = ttk.Frame(self.root, style="Primary.TFrame")
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(
            status,
            textvariable=self._status_var,
            style="Status.TLabel",
        ).pack(side=tk.LEFT, padx=5)

        self._model_status_var = tk.StringVar(value="Model: Loading…")
        ttk.Label(
            status,
            textvariable=self._model_status_var,
            style="Status.TLabel",
        ).pack(side=tk.RIGHT, padx=5)

        # Video-progress text / ETA moved here when the Video-Progress panel
        # was replaced by the Iris-Unwrap panel.
        self._progress_label_var = tk.StringVar(value="No video loaded")
        self._eta_label_var = tk.StringVar(value="")
        ttk.Label(
            status,
            textvariable=self._eta_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT, padx=5)
        ttk.Label(
            status,
            textvariable=self._progress_label_var,
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT, padx=5)

    def _install_crash_guards(self) -> None:
        self.root.report_callback_exception = self._handle_tk_exception
        threading.excepthook = self._handle_thread_exception

    def _handle_tk_exception(self, exc_type, exc_value, exc_traceback) -> None:
        self.logger.exception(
            "Tk callback error",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        self._report_runtime_issue("UI callback error recovered")

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        self.logger.exception(
            "Worker thread error in %s",
            args.thread.name if args.thread is not None else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        self.root.after(
            0,
            lambda: self._report_runtime_issue("Background worker recovered"),
        )

    def _report_runtime_issue(self, message: str) -> None:
        try:
            self._status_var.set(message)
        except Exception:
            pass

    def _bind_live_setting_callbacks(self) -> None:
        callbacks = (
            (self._use_optimized_var, "pipeline"),
            (self._fp16_var, "engine"),
            (self._compile_var, "engine"),
            (self._resolution_var, "resolution"),
            (self._stride_var, "stride"),
            (self._target_fps_var, "display"),
            (self._roi_var, "roi"),
            (self._roi_cache_var, "roi"),
            (self._kalman_process_var, "tracking"),
            (self._kalman_measure_var, "tracking"),
            (self._calibration_mode_var, "calibration"),
            (self._fixed_scale_var, "calibration"),
            (self._corneal_ref_mm_var, "calibration"),
            (self._ring_ref_mm_var, "calibration"),
        )
        for var, reason in callbacks:
            var.trace_add(
                "write",
                lambda *_args, _reason=reason: self._schedule_live_settings_apply(
                    _reason
                ),
            )

    def _schedule_live_settings_apply(self, reason: str) -> None:
        if self._suspend_live_settings_apply:
            return
        self._pending_live_apply_reasons.add(reason)
        if self._settings_apply_after_id is not None:
            self.root.after_cancel(self._settings_apply_after_id)
        self._settings_apply_after_id = self.root.after(
            180,
            self._apply_live_settings,
        )

    def _apply_live_settings(self) -> None:
        self._settings_apply_after_id = None
        reasons = set(self._pending_live_apply_reasons)
        self._pending_live_apply_reasons.clear()
        if not reasons:
            return
        if self._restart_in_progress:
            self._pending_live_apply_reasons.update(reasons)
            self._settings_apply_after_id = self.root.after(250, self._apply_live_settings)
            return

        self._res_display.set(str(int(self._resolution_var.get())))
        self._kp_display.set(f"{float(self._kalman_process_var.get()):.3f}")
        self._km_display.set(f"{float(self._kalman_measure_var.get()):.3f}")

        runtime_restart_reasons = {"pipeline", "engine", "resolution", "stride", "roi", "tracking"}
        restart_required = bool(reasons.intersection(runtime_restart_reasons))

        if hasattr(self.cfg, "video"):
            self.cfg.video.kalman_process_noise = float(self._kalman_process_var.get())
            self.cfg.video.kalman_measurement_noise = float(
                self._kalman_measure_var.get()
            )

        # Apply calibration settings
        if "calibration" in reasons or not restart_required:
            cal_mode = self._calibration_mode_var.get()
            manual_px = float(self._fixed_scale_var.get())
            corneal_mm = float(self._corneal_ref_mm_var.get())
            ring_mm = float(self._ring_ref_mm_var.get())
            if hasattr(self.cfg, "calibration"):
                self.cfg.calibration.mode = cal_mode
                self.cfg.calibration.manual_px_per_mm = manual_px
                self.cfg.calibration.corneal_diameter_mm = corneal_mm
                self.cfg.calibration.suction_ring_diameter_mm = ring_mm
            if self._detector is not None:
                self._detector.set_calibration_mode(
                    mode=cal_mode,
                    manual_px_per_mm=manual_px,
                    corneal_diameter_mm=corneal_mm,
                    ring_diameter_mm=ring_mm,
                )

        if self._tracker is not None and not self._video_running:
            self._tracker = EyeKalmanTracker(config=self.cfg)

        if self._opt_processor is not None and not restart_required:
            try:
                self._opt_processor.update_runtime_settings(
                    enable_auto_roi=self._roi_var.get(),
                    roi_cache_ttl=self._roi_cache_var.get(),
                    process_noise=self._kalman_process_var.get(),
                    measurement_noise=self._kalman_measure_var.get(),
                )
                self._apply_manual_roi_to_processor()
            except Exception as exc:
                self.logger.warning("Live optimized settings apply failed: %s", exc)

        if restart_required and self._video_running and self._active_source is not None:
            self._restart_active_stream(
                f"Applied {' / '.join(sorted(reasons))} settings",
                rebuild_engine=bool(
                    reasons.intersection({"pipeline", "engine", "resolution"})
                ),
            )
            return

        if (
            reasons.intersection({"pipeline", "engine", "resolution"})
            and not self._video_running
        ):
            self._invalidate_fast_engine()
            self._get_fast_engine()

        self._status_var.set(
            "Settings applied live: "
            + ", ".join(r.replace("_", " ").title() for r in sorted(reasons))
        )
        self._refresh_display()

        # When calibration changes, re-compute mm values on the
        # currently displayed result so the measurement panel updates
        # immediately (without requiring a new image load / detection).
        if "calibration" in reasons and self._current_result is not None:
            res = self._current_result
            new_cal = self._detector._calibration if self._detector is not None else None
            if new_cal is not None and new_cal.calibrated:
                res.calibration = new_cal
            elif (
                new_cal is not None
                and not new_cal.calibrated
                and getattr(res, "limbus", None) is not None
                and getattr(res.limbus, "detected", False)
                and getattr(res.limbus, "ellipse", None) is not None
            ):

                # ANATOMICAL_ANCHOR after reset: _current_best() returns
                # uncalibrated because _ema_px_per_mm is None.  Compute
                # the tautological calibration from the current result's
                # pixel geometry so the panel shows correct values.
                ep = res.limbus.ellipse
                corneal_mm = float(self._corneal_ref_mm_var.get())
                dia_px = ep.semi_major * 2.0
                if dia_px > 10:
                    res.calibration = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=dia_px / corneal_mm,
                        mm_per_px=corneal_mm / dia_px,
                        source="anatomical_on_switch",
                        method="anatomical",
                        corneal_diameter_assumed_mm=corneal_mm,
                    )

            # Clear stale pre-computed mm attributes that were set by
            # _add_mm_values / evaluate_clinical_wtw during the original
            # detection.  These would otherwise be read by the display
            # code and show values computed with the OLD calibration.
            # (LimbusDetection/PupilDetection are dataclasses, so we
            # reset fields to None rather than delattr which does not
            # truly remove dataclass fields.)
            for target in (
                getattr(res, "limbus", None),
                getattr(res, "pupil", None),
            ):
                if target is None:
                    continue
                for attr in (
                    "wtw_horizontal_mm",
                    "wtw_vertical_mm",
                    "wtw_mean_mm",
                    "wtw_astigmatism_mm",
                    "is_wtw_measured",
                    "wtw_validity_status",
                    "radius_mm",
                    "center_mm",
                ):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass
            self._update_measurements(res)

    def _restart_active_stream(
        self, reason: str, rebuild_engine: bool = False
    ) -> None:
        if self._restart_in_progress:
            return
        source = self._active_source
        if source is None:
            return
        self._restart_in_progress = True
        self._status_var.set(f"{reason} - restarting stream...")
        try:
            if self._recorder.is_recording:
                self._stop_recording()
            if rebuild_engine:
                self._invalidate_fast_engine()
            self._stop_video()
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during stop: %s", exc)
            self._restart_in_progress = False
            self._report_runtime_issue("Restart failed while stopping the active stream")
            return

        self.root.after(120, lambda src=source: self._complete_stream_restart(src))

    def _complete_stream_restart(self, source: Any) -> None:
        try:
            self._start_video(source)
        except Exception as exc:
            self.logger.exception("Safe stream restart failed during start: %s", exc)
            self._report_runtime_issue("Restart failed while starting the active stream")
        finally:
            self._restart_in_progress = False
            if self._pending_live_apply_reasons:
                self._schedule_live_settings_apply("restart")

    # ================================================================
    # Fast Inference Engine
    # ================================================================

    def _find_model_path(self) -> Optional[str]:
        candidates: List[str] = []
        if isinstance(self.cfg, dict):
            cfg_path = self.cfg.get("model_path")
            if cfg_path:
                candidates.append(str(cfg_path))
        if self._detector is not None:
            eng = getattr(self._detector, "ml_engine", None)
            if eng is not None:
                for attr in ("model_path", "_model_path"):
                    p = getattr(eng, attr, None)
                    if p:
                        candidates.append(str(p))
        candidates += [
            "models/best_model.pth",
            "model/best_model.pth",
            "pupil_tracking/models/best_model.pth",
        ]
        for c in candidates:
            if Path(c).is_file():
                found_path = str(Path(c).resolve())
                self.logger.info(f"Loaded model from: {found_path}")
                return found_path
        return None

    def _get_fast_engine(self) -> Optional[Any]:
        if not _FAST_PIPELINE_AVAILABLE:
            return None
        if self._fast_engine is not None:
            return self._fast_engine
        model_path = self._find_model_path()
        if model_path is None:
            self.logger.warning(
                "Cannot locate model file — optimised pipeline disabled"
            )
            return None
        try:
            self._fast_engine = FastInference(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                use_half=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                reflection_removal=True,
                suction_ring_removal=True,
            )
            self.logger.info(
                "FastInference ready (%s)",
                self._fast_engine.device,
            )
            self._engine_status_var.set(f"Engine: ready ({self._fast_engine.device})")
            return self._fast_engine
        except Exception as exc:
            self.logger.error("FastInference init failed: %s", exc)
            self._engine_status_var.set(f"Engine: error — {exc}")
            return None

    def _invalidate_fast_engine(self) -> None:
        if self._fast_engine is None:
            return
        try:
            import torch

            if (
                hasattr(self._fast_engine, "model")
                and self._fast_engine.model is not None
            ):
                del self._fast_engine.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._fast_engine = None
        self._engine_status_var.set("Engine: invalidated (will rebuild on next use)")

    def _rebuild_engine_ui(self) -> None:
        self._invalidate_fast_engine()
        engine = self._get_fast_engine()
        if engine is not None:
            engine.warmup()
            msg = (
                f"Engine: rebuilt ({engine.device}, "
                f"res={self._resolution_var.get()}, "
                f"fp16={self._fp16_var.get()})"
            )
            self._engine_status_var.set(msg)
            self._status_var.set(msg)
        else:
            self._engine_status_var.set("Engine: rebuild FAILED")
            self._status_var.set("Engine rebuild failed — check model path and logs")

    def _apply_performance_preset(self) -> None:
        preset = self._performance_preset_var.get()
        base_res = int(self._runtime_profile.recommended_resolution)
        base_fps = float(self._runtime_profile.recommended_target_fps)
        self._suspend_live_settings_apply = True
        try:
            if preset == "max_accuracy":
                self._resolution_var.set(
                    384 if self._runtime_profile.has_cuda else max(320, base_res)
                )
                self._target_fps_var.set(20.0 if self._runtime_profile.has_cuda else min(18.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            elif preset == "low_latency":
                self._resolution_var.set(256 if self._runtime_profile.has_cuda else 288)
                self._target_fps_var.set(30.0 if self._runtime_profile.has_cuda else max(20.0, base_fps))
                self._stride_var.set(1)
                self._roi_var.set(True)
            else:
                self._resolution_var.set(base_res)
                self._target_fps_var.set(base_fps)
                self._stride_var.set(1)
                self._roi_var.set(True)
        finally:
            self._suspend_live_settings_apply = False
        self._res_display.set(str(self._resolution_var.get()))
        self._pending_live_apply_reasons.update({"resolution", "display", "stride", "roi"})
        self._schedule_live_settings_apply("preset")
        self._status_var.set(f"Preset: {preset.replace('_', ' ').title()} applied")

    def _get_display_interval(self) -> float:
        preset = self._performance_preset_var.get()
        preset_cap = {
            "max_accuracy": 20.0,
            "balanced": 28.0,
            "low_latency": 36.0,
        }.get(preset, _DISPLAY_FPS_CAP)
        target = float(self._target_fps_var.get() or _DISPLAY_FPS_CAP)
        cap = max(5.0, min(_DISPLAY_FPS_CAP, preset_cap, target))
        return 1.0 / cap

    def _derive_tracking_state(
        self, result: Any, stats: Optional[Dict[str, Any]] = None
    ) -> str:
        conf = float(getattr(result, "overall_confidence", 0.0) or 0.0)
        has_both = bool(getattr(result, "has_both", False))
        stats = stats or {}
        stale = int(stats.get("stale_frames", 0))
        dropped = int(stats.get("dropped_frames", 0))
        recent_latency = float(stats.get("latency_avg_ms", 0.0) or 0.0)

        if not has_both:
            return "No Detection"
        if stale > 10 or dropped > 10 or recent_latency > 250.0:
            return "Tracking Degraded"
        if conf >= 0.75:
            return "Tracking Stable"
        if conf >= 0.35:
            return "Tracking Acquiring"
        return "Tracking Degraded"

    def _set_summary_tracking_state(self, tracking_text: str) -> None:
        self._summary_tracking_var.set(tracking_text)
        tracking_color = {
            "Tracking Stable": self._colors.SURGICAL,
            "Tracking Acquiring": self._colors.CLINICAL,
            "Tracking Degraded": self._colors.RESEARCH,
            "No Detection": self._colors.INSUFFICIENT,
            "Ready": self._colors.ACCENT,
            "Waiting": self._colors.FG_SECONDARY,
        }.get(tracking_text, self._colors.FG_PRIMARY)
        self._summary_tracking_label.config(foreground=tracking_color)

    def _toggle_pause(self) -> None:
        if self._awaiting_roi or not self._video_running:
            return
        self._video_paused = not self._video_paused
        if self._video_paused:
            self._pause_btn.config(text="▶ Resume")
            self._status_var.set("Paused")
        else:
            self._pause_btn.config(text="⏸ Pause")
            self._status_var.set("Resumed")

    def _sync_roi_ring_button_styles(self) -> None:
        if hasattr(self, "_roi_btn"):
            is_active = self._roi_edit_active or self._manual_roi is not None
            self._roi_btn.configure(style="ROIActive.TButton" if is_active else "TButton")
            self._roi_btn.config(text="Clear ROI" if is_active else "ROI")
        if hasattr(self, "_ring_btn"):
            is_active = self._ring_edit_active or self._manual_ring is not None
            self._ring_btn.configure(style="RingActive.TButton" if is_active else "TButton")
            self._ring_btn.config(text="Clear Ring" if is_active else "Ring")

    def _toggle_roi(self) -> None:
        """Single toggle: if ROI is active/editing → clear it; otherwise begin selection."""
        if self._roi_edit_active or self._manual_roi is not None:
            self._clear_manual_roi()
        else:
            self._begin_roi_selection()

    def _toggle_ring(self) -> None:
        """Single toggle: if ring is active/editing → clear it; otherwise begin selection."""
        if self._ring_edit_active or self._manual_ring is not None:
            self._clear_manual_ring()
        else:
            self._begin_ring_selection()

    def _begin_roi_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag a circular ROI on the image")
            return
        if self._ring_edit_active:
            self._cancel_ring_selection()
        self._roi_edit_active = True
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        active_roi = self._active_manual_roi()
        self._roi_original_before_edit = (
            dict(active_roi) if active_roi is not None else None
        )
        if active_roi is not None:
            self._roi_preview = dict(active_roi)
        else:
            h, w = self._current_image.shape[:2]
            radius = 939.0 / 2.0
            self._roi_preview = {
                "center_x": w / 2.0,
                "center_y": h / 2.0,
                "radius": radius,
                "frame_width": float(w),
                "frame_height": float(h),
            }
        self._canvas.configure(cursor="tcross")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set("Manual ROI: Editing")
        self._status_var.set(
            "ROI edit mode: drag inside to move, drag rim to resize, Enter to apply, Esc to cancel"
        )
        self._refresh_display()

    def _begin_ring_selection(self) -> None:
        if self._current_image is None:
            self._status_var.set("Start the camera, then drag the docked red ring on the image")
            return
        if self._roi_edit_active:
            self._cancel_roi_selection()
        self._ring_edit_active = True
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        active_ring = self._active_manual_ring()
        self._ring_original_before_edit = (
            dict(active_ring) if active_ring is not None else None
        )
        if active_ring is not None:
            self._ring_preview = dict(active_ring)
        else:
            self._ring_preview = self._suggest_manual_ring_preview()
        self._canvas.configure(cursor="tcross")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set("Manual Ring: Editing")
        self._status_var.set(
            "Ring edit mode: use only on docked frames, drag circle to match red dots, Enter to lock"
        )
        self._refresh_display()

    def _clear_manual_roi(self) -> None:
        self._manual_roi = None
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set("Manual ROI: Off")
        previous_suspend = self._suspend_live_settings_apply
        self._suspend_live_settings_apply = True
        try:
            self._roi_var.set(False)
        finally:
            self._suspend_live_settings_apply = previous_suspend
        if self._video_running:
            self._awaiting_roi = True
            self._status_var.set("Video waiting — set an ROI to resume detection")
        if self._current_result is not None:
            self._current_result = None
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_roi()
            self._opt_processor.update_runtime_settings(enable_auto_roi=False)
        self._refresh_display()

    def _clear_manual_ring(self) -> None:
        self._manual_ring = None
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set("Manual Ring: Off")
        if self._opt_processor is not None:
            self._opt_processor.clear_manual_ring()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
            self._current_result.ring_status = "ring_absent"
            self._current_result.ring_center = None
            self._current_result.ring_radius = None
        self._refresh_display()

    def _cancel_roi_selection(self, event: Any = None) -> None:

        # Abandon the in-progress edit and restore whatever ROI existed before
        # it (may be None). If detection is still gated on the ROI, keep the
        # frame frozen and re-prompt instead of starting.
        self._manual_roi = (
            dict(self._roi_original_before_edit)
            if self._roi_original_before_edit is not None
            else None
        )
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        if self._manual_roi is not None:
            self._roi_status_var.set(
                f"Manual ROI: On ({int(round(self._manual_roi['radius']))} px)"
            )
        else:
            self._roi_status_var.set("Manual ROI: Off")
        self._apply_manual_roi_to_processor()
        if self._awaiting_roi:
            self._status_var.set("ROI required — set an ROI to start detection")
        else:
            self._status_var.set("ROI selection cancelled")
        self._refresh_display()

    def _cancel_ring_selection(self, event: Any = None) -> None:

        # Abandon the in-progress ring edit and restore the prior ring (or None).
        self._manual_ring = (
            dict(self._ring_original_before_edit)
            if self._ring_original_before_edit is not None
            else None
        )
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        if self._manual_ring is not None:
            self._ring_status_var.set(
                f"Manual Ring: Locked ({int(round(self._manual_ring['radius'] * 2.0))} px)"
            )
        else:
            self._ring_status_var.set("Manual Ring: Off")
        self._apply_manual_ring_to_processor()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._status_var.set("Ring selection cancelled")
        self._refresh_display()

    def _on_canvas_press(self, event: Any) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._handle_ruler_canvas_click(event)
            return
        if self._ring_edit_active:
            self._handle_ring_canvas_press(event)
            return
        if not self._roi_edit_active:
            return
        point = self._canvas_to_image_point(event.x, event.y)

        if point is None or self._current_image is None:
            return
        if self._roi_preview is None:
            h, w = self._current_image.shape[:2]
            self._roi_preview = {
                "center_x": point[0],
                "center_y": point[1],
                "radius": 939.0 / 2.0,
                "frame_width": float(w),
                "frame_height": float(h),
            }

        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        radius = self._roi_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.18)

        if abs(distance - radius) <= rim_threshold:
            self._roi_drag_mode = "resize"
        elif distance < radius:
            self._roi_drag_mode = "move"
            self._roi_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._roi_drag_mode = "resize"
            self._roi_preview["center_x"] = point[0]
            self._roi_preview["center_y"] = point[1]
            self._roi_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _on_canvas_drag(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_drag(event)
            return
        if (
            not self._roi_edit_active
            or self._roi_drag_mode is None
            or self._roi_preview is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        h, w = self._current_image.shape[:2]
        if self._roi_drag_mode == "move":
            radius = self._roi_preview["radius"]
            cx = point[0] - self._roi_drag_offset[0]
            cy = point[1] - self._roi_drag_offset[1]
            self._roi_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._roi_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._roi_preview["center_x"]
            cy = self._roi_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _on_canvas_release(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_release(event)
            return
        if not self._roi_edit_active:
            return
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._roi_preview is not None:
            self._status_var.set(
                "ROI ready. Drag to refine, press Enter to apply, or Esc to cancel"
            )
        self._refresh_display()

    def _confirm_active_selection(self, event: Any = None) -> None:
        if self._ring_edit_active:
            self._confirm_ring_selection(event)
            return
        self._confirm_roi_selection(event)

    def _confirm_roi_selection(self, event: Any = None) -> None:
        if not self._roi_edit_active:
            return
        preview = self._roi_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("ROI selection cancelled")
            self._cancel_roi_selection()
            return
        self._manual_roi = dict(preview)
        self._roi_preview = None
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_edit_active = False
        self._roi_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._roi_status_var.set(
            f"Manual ROI: On ({int(round(self._manual_roi['radius']))} px)"
        )
        self._status_var.set("Manual ROI applied to live detection")
        self._apply_manual_roi_to_processor()
        self._refresh_display()

        # If detection was gated on the ROI (freshly loaded video/camera),
        # confirming the ROI is what actually starts the processing thread.
        if self._awaiting_roi:
            self._awaiting_roi = False
            if self._pending_detection_start is not None:
                start = self._pending_detection_start
                self._pending_detection_start = None
                self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
                start()

    def _confirm_ring_selection(self, event: Any = None) -> None:
        if not self._ring_edit_active:
            return
        preview = self._ring_preview
        if preview is None or preview["radius"] < 8.0:
            self._status_var.set("Ring selection cancelled")
            self._cancel_ring_selection()
            return
        self._manual_ring = dict(preview)
        self._ring_preview = None
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._ring_edit_active = False
        self._ring_original_before_edit = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        self._ring_status_var.set(
            f"Manual Ring: Locked ({int(round(self._manual_ring['radius'] * 2.0))} px)"
        )
        self._status_var.set("Manual docked ring locked and applied to offset calculation")
        self._show_ring_center.set(True)
        self._apply_manual_ring_to_processor()
        if self._current_result is not None:
            self._apply_manual_ring_policy(self._current_result)
            self._update_measurements(self._current_result)
        self._refresh_display()

    def _nudge_roi(self, dx: int, dy: int, event: Any = None) -> None:
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        step = 10.0 if (event is not None and (event.state & 0x0001)) else 1.0
        h, w = self._current_image.shape[:2]
        radius = self._roi_preview["radius"]
        self._roi_preview["center_x"] = float(
            np.clip(self._roi_preview["center_x"] + dx * step, radius, w - radius)
        )
        self._roi_preview["center_y"] = float(
            np.clip(self._roi_preview["center_y"] + dy * step, radius, h - radius)
        )
        self._status_var.set(
            "ROI edit mode: arrows nudge, drag move/resize, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _on_canvas_wheel(self, event: Any) -> None:
        if self._ring_edit_active:
            self._handle_ring_canvas_wheel(event)
            return
        if not self._roi_edit_active or self._roi_preview is None or self._current_image is None:
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._roi_preview["center_x"]
        cy = self._roi_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._roi_preview["radius"] + direction * step
        self._roi_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "ROI edit mode: wheel resizes, arrows nudge, Enter apply, Esc cancel"
        )
        self._refresh_display()

    def _cancel_active_selection(self, event: Any = None) -> None:
        if getattr(self, "_ruler_calibration_active", False):
            self._cancel_ruler_calibration(event)
            return
        if self._ring_edit_active:
            self._cancel_ring_selection(event)
            return
        self._cancel_roi_selection(event)

    def _cancel_ruler_calibration(self, event: Any = None) -> None:
        self._ruler_calibration_active = False
        self._ruler_points.clear()
        self._canvas.configure(cursor="crosshair")
        self._status_var.set("Ruler calibration cancelled")
        self._refresh_display()

    def _start_ruler_calibration(self, known_dist_mm: float = 10.0) -> None:
        self._ruler_calibration_active = True
        self._ruler_points.clear()
        self._ruler_known_dist_mm_var.set(known_dist_mm)
        self._canvas.configure(cursor="tcross")
        self._status_var.set(
            f"Ruler Calibration: Click Point 1 on image (known span: {known_dist_mm:.1f} mm)"
        )
        self._refresh_display()

    def _handle_ruler_canvas_click(self, event: Any) -> None:
        pt = self._canvas_to_image_point(event.x, event.y)
        if pt is None:
            return
        self._ruler_points.append(pt)
        if len(self._ruler_points) == 1:
            self._status_var.set(
                f"Ruler point 1 set at ({pt[0]:.1f}, {pt[1]:.1f}) px. Now click Point 2..."
            )
            self._refresh_display()
        elif len(self._ruler_points) >= 2:
            p1 = self._ruler_points[0]
            p2 = self._ruler_points[1]
            dist_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            if known_mm <= 0:
                known_mm = 10.0
            if dist_px < 2.0:
                self._status_var.set("Selected points too close together (<2px). Click 2 distinct points.")
                self._ruler_points.clear()
                self._refresh_display()
                return

            from pupil_tracking.calibration.spatial_calibration import calculate_ruler_scale
            px_per_mm, mm_per_px = calculate_ruler_scale(p1, p2, known_mm)
            self._fixed_scale_var.set(round(px_per_mm, 2))
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            self._ruler_calibration_active = False
            self._canvas.configure(cursor="crosshair")
            msg = f"Ruler Calibrated: {px_per_mm:.2f} px/mm ({dist_px:.1f} px = {known_mm:.1f} mm)"
            self._status_var.set(msg)
            self._refresh_display()
            messagebox.showinfo(
                "Ruler Calibration Applied",
                f"Calibration Successful!\n\nScale: {px_per_mm:.2f} px/mm ({mm_per_px:.4f} mm/px)\n"
                f"Measured Span: {dist_px:.1f} px = {known_mm:.1f} mm\n"
                f"Mode: Fixed Pixel Scale (Active)",
            )

    def _open_calibration_wizard(self) -> None:
        """Open a modern, comprehensive Calibration & Corneal Sizing Wizard dialog."""
        wizard = tk.Toplevel(self.root)
        wizard.title("Scale & Corneal Sizing (WTW) Calibration Wizard")
        wizard.geometry("540x580")
        wizard.resizable(False, False)
        wizard.transient(self.root)
        wizard.grab_set()

        c = self._colors
        wizard.configure(bg=c.BG_PRIMARY)

        # Header
        header = ttk.Frame(wizard, padding=16)
        header.pack(fill=tk.X)
        ttk.Label(
            header,
            text="Optical Scale & WTW Calibration",
            font=("Segoe UI", 14, "bold"),
            foreground=c.FG_PRIMARY,
        ).pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Establish an accurate pixel-to-millimeter ratio for genuine patient-specific\nWhite-to-White corneal diameter measurements.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 0))

        # Main cards container
        container = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        container.pack(fill=tk.BOTH, expand=True)

        # Method 1: Interactive 2-Point Ruler Card
        m1 = ttk.LabelFrame(container, text="Method 1: Interactive 2-Point Ruler Tool", padding=10)
        m1.pack(fill=tk.X, pady=4)
        ttk.Label(
            m1,
            text="Click two points on an image with a known physical distance\n(e.g., surgical marker, ruler, or target grid).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ruler_row = ttk.Frame(m1)
        ruler_row.pack(fill=tk.X, pady=2)
        ttk.Label(ruler_row, text="Known Distance (mm):", width=20).pack(side=tk.LEFT)
        known_mm_ent = ttk.Spinbox(
            ruler_row,
            from_=0.5,
            to=100.0,
            increment=0.5,
            textvariable=self._ruler_known_dist_mm_var,
            width=8,
        )
        known_mm_ent.pack(side=tk.LEFT, padx=4)

        def _start_ruler():
            wizard.destroy()
            self._start_ruler_calibration(self._ruler_known_dist_mm_var.get())

        ttk.Button(
            ruler_row,
            text="🎯 Click 2 Points on Canvas",
            command=_start_ruler,
        ).pack(side=tk.LEFT, padx=8)

        # Method 2: Direct Optical Scale Input
        m2 = ttk.LabelFrame(container, text="Method 2: Direct Optical / Microscope Scale", padding=10)
        m2.pack(fill=tk.X, pady=4)
        ttk.Label(
            m2,
            text="Directly enter known magnification / scale factor in pixels per millimeter.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        scale_row = ttk.Frame(m2)
        scale_row.pack(fill=tk.X, pady=2)
        ttk.Label(scale_row, text="Fixed Scale (px/mm):", width=20).pack(side=tk.LEFT)
        scale_ent = ttk.Spinbox(
            scale_row,
            from_=1.0,
            to=500.0,
            increment=0.5,
            textvariable=self._fixed_scale_var,
            width=8,
        )
        scale_ent.pack(side=tk.LEFT, padx=4)

        def _apply_fixed():
            self._calibration_mode_var.set("FIXED_PIXEL_SCALE")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Scale Set", f"Fixed optical scale set to {self._fixed_scale_var.get():.2f} px/mm.")

        ttk.Button(scale_row, text="Apply Scale", command=_apply_fixed).pack(side=tk.LEFT, padx=8)

        # Method 3: Ring Fiducial Reflection
        m3 = ttk.LabelFrame(container, text="Method 3: Suction / Placido Ring Fiducial", padding=10)
        m3.pack(fill=tk.X, pady=4)
        ttk.Label(
            m3,
            text="Calibrate dynamically from a known surgical suction ring or Placido ring.",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        ring_row = ttk.Frame(m3)
        ring_row.pack(fill=tk.X, pady=2)
        ttk.Label(ring_row, text="Ring Outer Dia (mm):", width=20).pack(side=tk.LEFT)
        ring_ent = ttk.Spinbox(
            ring_row,
            from_=5.0,
            to=25.0,
            increment=0.1,
            textvariable=self._ring_ref_mm_var,
            width=8,
        )
        ring_ent.pack(side=tk.LEFT, padx=4)

        def _apply_ring():
            self._calibration_mode_var.set("RING_REFLECTION")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Ring Mode Set", f"Ring reflection calibration activated ({self._ring_ref_mm_var.get():.1f} mm standard).")

        ttk.Button(ring_row, text="Apply Ring Mode", command=_apply_ring).pack(side=tk.LEFT, padx=8)

        # Method 4: Anatomical Baseline Anchor
        m4 = ttk.LabelFrame(container, text="Method 4: Anatomical Baseline Anchor (12.0 mm)", padding=10)
        m4.pack(fill=tk.X, pady=4)
        ttk.Label(
            m4,
            text="Assumes horizontal corneal diameter is 12.0 mm (not patient-specific).",
            style="Muted.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        def _apply_anatomical():
            self._calibration_mode_var.set("ANATOMICAL_ANCHOR")
            self._schedule_live_settings_apply("calibration")
            wizard.destroy()
            messagebox.showinfo("Baseline Set", "Calibration set to 12.0 mm Anatomical Anchor.")
        ttk.Button(m4, text="Reset to 12.0 mm Baseline Anchor", command=_apply_anatomical).pack(anchor=tk.W)

        # Footer close button
        btn_bar = ttk.Frame(wizard, padding=(16, 0, 16, 16))
        btn_bar.pack(fill=tk.X)
        ttk.Button(btn_bar, text="Close", command=wizard.destroy).pack(side=tk.RIGHT)


    def _canvas_to_image_point(
        self, canvas_x: float, canvas_y: float
    ) -> Optional[Tuple[float, float]]:
        if self._current_image is None:
            return None
        ox, oy = self._display_origin
        dw, dh = self._display_size
        if dw <= 0 or dh <= 0:
            return None
        if not (ox <= canvas_x <= ox + dw and oy <= canvas_y <= oy + dh):
            return None
        x = (canvas_x - ox) / max(self._display_scale, 1e-6)
        y = (canvas_y - oy) / max(self._display_scale, 1e-6)
        h, w = self._current_image.shape[:2]
        return (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))

    def _active_manual_roi(self) -> Optional[Dict[str, float]]:
        if self._manual_roi is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_roi.get("frame_width", w))) != w
            or int(round(self._manual_roi.get("frame_height", h))) != h
        ):
            return None
        return self._manual_roi

    def _active_manual_ring(self) -> Optional[Dict[str, float]]:
        if self._manual_ring is None or self._current_image is None:
            return None
        h, w = self._current_image.shape[:2]
        if (
            int(round(self._manual_ring.get("frame_width", w))) != w
            or int(round(self._manual_ring.get("frame_height", h))) != h
        ):
            return None
        return self._manual_ring

    def _apply_manual_ring_policy(self, result: Any) -> Any:
        """Only allow ring data when a manual ring has been confirmed."""
        if result is None:
            return None
        ring = self._active_manual_ring()
        if ring is None:
            setattr(result, "ring_status", "ring_absent")
            setattr(result, "ring_center", None)
            setattr(result, "ring_radius", None)
            setattr(result, "ring_contour", None)
            setattr(result, "ring_dot_count", 0)
            setattr(result, "ring_confidence", 0.0)
            if "ring" in str(getattr(result, "corneal_reference_source", "")):
                setattr(result, "corneal_reference_source", "limbus")
            return result

        center = (float(ring["center_x"]), float(ring["center_y"]))
        radius = float(ring["radius"])
        setattr(result, "ring_status", "ring_present")
        setattr(result, "ring_center", center)
        setattr(result, "ring_radius", radius)
        setattr(result, "ring_contour", None)
        setattr(result, "ring_dot_count", int(round(ring.get("dot_count", 12))))
        setattr(result, "ring_confidence", 1.0)
        setattr(result, "corneal_reference_source", "manual_ring")
        return result

    def _apply_manual_roi_to_processor(self) -> None:
        roi = self._active_manual_roi()
        if self._opt_processor is None:
            return
        if roi is None:
            self._opt_processor.clear_manual_roi()
            return
        self._opt_processor.set_manual_roi(
            center_x=roi["center_x"],
            center_y=roi["center_y"],
            radius=roi["radius"],
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _apply_manual_ring_to_processor(self) -> None:
        ring = self._active_manual_ring()
        if self._opt_processor is None:
            return
        if ring is None:
            self._opt_processor.clear_manual_ring()
            return
        self._opt_processor.set_manual_ring(
            center_x=ring["center_x"],
            center_y=ring["center_y"],
            radius=ring["radius"],
            dot_count=int(round(ring.get("dot_count", 12))),
            frame_shape=(
                self._current_image.shape if self._current_image is not None else None
            ),
        )

    def _suggest_manual_ring_preview(self) -> Dict[str, float]:
        if self._current_result is not None:
            ring_status = getattr(self._current_result, "ring_status", "unknown")
            ring_center = getattr(self._current_result, "ring_center", None)
            ring_radius = getattr(self._current_result, "ring_radius", None)
            if (
                ring_status == "ring_present"
                and ring_center is not None
                and ring_radius is not None
            ):
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(ring_center[0]),
                    "center_y": float(ring_center[1]),
                    "radius": float(ring_radius),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(getattr(self._current_result, "ring_dot_count", 12)),
                }
        if self._opt_processor is not None and self._current_image is not None:
            suggestion = self._opt_processor.get_manual_ring(self._current_image.shape)
            if suggestion is not None:
                h, w = self._current_image.shape[:2]
                return {
                    "center_x": float(suggestion["center_x"]),
                    "center_y": float(suggestion["center_y"]),
                    "radius": float(suggestion["radius"]),
                    "frame_width": float(w),
                    "frame_height": float(h),
                    "dot_count": float(suggestion.get("dot_count", 12)),
                }
        h, w = self._current_image.shape[:2]
        roi = self._active_manual_roi()
        if roi is not None:
            center_x = float(roi["center_x"])
            center_y = float(roi["center_y"])
            radius = max(12.0, float(roi["radius"]) * 0.92)
        else:
            center_x = w / 2.0
            center_y = h / 2.0
            if self._current_result is not None and hasattr(self._current_result, "calibration") and self._current_result.calibration.calibrated:
                px_per_mm = self._current_result.calibration.px_per_mm
                radius = max(12.0, (14.1 * px_per_mm) / 2.0)
            else:
                radius = max(12.0, min(w, h) * 0.42)
        return {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "frame_width": float(w),
            "frame_height": float(h),
            "dot_count": 12.0,
        }

    def _handle_ring_canvas_press(self, event: Any) -> None:
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None or self._current_image is None:
            return
        if self._ring_preview is None:
            self._ring_preview = self._suggest_manual_ring_preview()
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        radius = self._ring_preview["radius"]
        distance = math.hypot(point[0] - cx, point[1] - cy)
        rim_threshold = max(10.0, radius * 0.15)
        if abs(distance - radius) <= rim_threshold:
            self._ring_drag_mode = "resize"
        elif distance < radius:
            self._ring_drag_mode = "move"
            self._ring_drag_offset = (point[0] - cx, point[1] - cy)
        else:
            self._ring_drag_mode = "resize"
            self._ring_preview["center_x"] = point[0]
            self._ring_preview["center_y"] = point[1]
            self._ring_preview["radius"] = max(12.0, radius * 0.5)
        self._refresh_display()

    def _handle_ring_canvas_drag(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_drag_mode is None
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        point = self._canvas_to_image_point(event.x, event.y)
        if point is None:
            return
        h, w = self._current_image.shape[:2]
        if self._ring_drag_mode == "move":
            radius = self._ring_preview["radius"]
            cx = point[0] - self._ring_drag_offset[0]
            cy = point[1] - self._ring_drag_offset[1]
            self._ring_preview["center_x"] = float(np.clip(cx, radius, w - radius))
            self._ring_preview["center_y"] = float(np.clip(cy, radius, h - radius))
        else:
            cx = self._ring_preview["center_x"]
            cy = self._ring_preview["center_y"]
            max_radius = min(cx, cy, w - cx, h - cy)
            radius = max(8.0, math.hypot(point[0] - cx, point[1] - cy))
            self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._refresh_display()

    def _handle_ring_canvas_release(self, event: Any) -> None:
        if not self._ring_edit_active:
            return
        self._ring_drag_mode = None
        self._ring_drag_offset = (0.0, 0.0)
        self._canvas.configure(cursor="fleur")
        if self._ring_preview is not None:
            self._status_var.set(
                "Ring ready. Match the red-dot circle, then press Enter to lock or Esc to cancel"
            )
        self._refresh_display()

    def _handle_ring_canvas_wheel(self, event: Any) -> None:
        if (
            not self._ring_edit_active
            or self._ring_preview is None
            or self._current_image is None
        ):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        step = 10.0 if (getattr(event, "state", 0) & 0x0001) else 4.0
        direction = 1.0 if delta > 0 else -1.0
        h, w = self._current_image.shape[:2]
        cx = self._ring_preview["center_x"]
        cy = self._ring_preview["center_y"]
        max_radius = min(cx, cy, w - cx, h - cy)
        radius = self._ring_preview["radius"] + direction * step
        self._ring_preview["radius"] = float(max(8.0, min(radius, max_radius)))
        self._status_var.set(
            "Ring edit mode: wheel resizes, drag move/resize, Enter lock, Esc cancel"
        )
        self._refresh_display()

    def _get_manual_roi_crop(
        self, frame: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, float]]:
        roi = self._active_manual_roi()
        if roi is None:
            return None
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

    # ================================================================
    # Image Operations
    # ================================================================

    def _open_image(self) -> None:
        from pupil_tracking.cyclotorsion.pentacam_loader import PENTACAM_FILETYPES
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=list(PENTACAM_FILETYPES),
        )
        if not path:
            return
        clean_path = path.strip().strip('"').strip("'")
        self._stop_video()
        self._load_and_detect_image(clean_path)

    def _open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return
        self._stop_video()
        folder_p = Path(folder)
        images = []
        for ext_pattern in (
            "*.jpeg", "*.jpg", "*.png", "*.bmp", "*.tiff", "*.tif",
            "*.JPEG", "*.JPG", "*.PNG", "*.BMP", "*.TIFF", "*.TIF",
        ):
            images.extend(folder_p.glob(ext_pattern))
        images = sorted(set(images))
        if not images:
            messagebox.showwarning("No Images", f"No images found in {folder}")
            return
        self._results_history.clear()
        for i, img_path in enumerate(images):
            self._status_var.set(f"Processing {i + 1}/{len(images)}: {img_path.name}")
            self.root.update()
            self._load_and_detect_image(str(img_path))
        self._status_var.set(f"Processed {len(images)} images")

    # ================================================================
    # Synchronous calibration sync — runs BEFORE every detection
    # to guarantee the detector uses the current GUI settings
    # (eliminates race with the 180 ms debounce in
    # _schedule_live_settings_apply).
    # ================================================================

    def _sync_calibration_to_detector(self) -> None:
        """Push current GUI calibration settings into the detector immediately.
        Called synchronously before every detection so that the detector
        always reflects the user's latest calibration choice, even when
        the 180 ms debounced ``_apply_live_settings`` has not fired yet.
        """
        if self._detector is None:
            return
        cal_mode = self._calibration_mode_var.get()
        manual_px = float(self._fixed_scale_var.get())
        corneal_mm = float(self._corneal_ref_mm_var.get())
        ring_mm = float(self._ring_ref_mm_var.get())
        if hasattr(self.cfg, "calibration"):
            self.cfg.calibration.mode = cal_mode
            self.cfg.calibration.manual_px_per_mm = manual_px
            self.cfg.calibration.corneal_diameter_mm = corneal_mm
            self.cfg.calibration.suction_ring_diameter_mm = ring_mm
        self._detector.set_calibration_mode(
            mode=cal_mode,
            manual_px_per_mm=manual_px,
            corneal_diameter_mm=corneal_mm,
            ring_diameter_mm=ring_mm,
        )

    def _load_and_detect_image(self, path: str) -> None:
        from pupil_tracking.cyclotorsion.pentacam_loader import load_pentacam_image
        clean_path = str(path).strip().strip('"').strip("'")
        image = load_pentacam_image(clean_path)
        if image is None:
            messagebox.showerror("Error", f"Cannot read image: {clean_path}")
            return
        self._reset_elita_shift()
        self._current_image = image
        self._frame_count += 1
        if self._detector is None:
            self._status_var.set("Detector not ready — showing raw image")
            self._current_result = None
            self._refresh_display()
            return
        self._sync_calibration_to_detector()
        self._status_var.set(f"Detecting: {Path(path).name}…")
        self.root.update()
        result = self._detector.detect(
            image, frame_number=self._frame_count, source=path
        )
        result = self._apply_manual_ring_policy(result)
        self._current_result = result
        self._results_history.append(result.to_dict())
        self._update_measurements(result)
        self._refresh_display()
        self._status_var.set(
            f"{Path(path).name} — {result.overall_quality.value} "
            f"({result.overall_confidence:.3f}) — "
            f"{result.metadata.processing_time_ms:.0f} ms"
        )

    # ================================================================
    # Video Operations
    # ================================================================

    def _open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._stop_video()
        self._start_video(path)

    def _arm_detection_gate(
        self, first_frame: np.ndarray, start_callable: Any, label: str
    ) -> None:
        """Show the first frame and hold detection until an ROI is confirmed.
        Detection does not begin until the user confirms an ROI; at that point
        ``start_callable`` is invoked to spin up the processing thread.
        """
        self._current_image = first_frame
        self._awaiting_roi = True
        self._pending_detection_start = start_callable
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self._progress_label_var.set("Waiting for ROI")
        self._refresh_display()

        # Auto-enter ROI drawing so the circle is immediately draggable.
        self._begin_roi_selection()
        self._status_var.set(
            f"{label} loaded — drag the ROI and press Enter to start detection"
        )

    def _start_video(self, source: Any) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open: {source}")
            return
        self._active_source = source
        self._video_cap = cap
        self._video_running = True
        self._video_paused = False
        self._using_optimized_camera = False
        self._camera_mode = isinstance(source, int)
        if not self._restart_in_progress:
            self._reset_elita_shift()
        if self._tracker is not None:
            self._tracker.reset()
        self._results_history.clear()
        self._frame_count = 0
        self._video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_start_time = time.monotonic()
        self._last_display_update = 0.0
        self._progress_label_var.set("Starting…")
        self._eta_label_var.set("")
        src_name = "Camera" if isinstance(source, int) else Path(str(source)).name
        use_opt = self._use_optimized_var.get() and _FAST_PIPELINE_AVAILABLE
        engine = self._get_fast_engine() if use_opt else None
        if engine is not None:
            start = lambda: self._start_video_optimized(engine, src_name)
        else:
            start = lambda: self._start_video_classic(src_name)

        # A restart (settings change mid-stream) preserves the already-confirmed
        # ROI, so resume detection immediately instead of re-prompting.
        if self._restart_in_progress:
            start()
            return

        # Fresh load: grab the first frame for ROI drawing, then hold detection
        # until the user confirms an ROI (rewind files so frame 0 is not consumed).
        ok, first = cap.read()
        if not ok or first is None:
            messagebox.showerror("Error", f"No frames to read from: {source}")
            self._stop_video()
            return
        if not self._camera_mode:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._arm_detection_gate(first, start, src_name)

    def _start_video_classic(self, src_name: str) -> None:

        # Reset the ETA/FPS clock now that processing actually begins — the
        # time spent drawing the ROI must not count against the frame rate.
        self._video_start_time = time.monotonic()
        self._status_var.set(f"Playing (classic): {src_name}")
        self._pipeline_var.set("Classic")
        self._video_thread = threading.Thread(
            target=self._video_loop_classic,
            args=(src_name,),
            daemon=True,
            name="VideoLoopClassic",
        )
        self._video_thread.start()

    def _video_loop_classic(self, source_name: str) -> None:
        raw_frame_idx = 0
        consecutive_read_failures = 0
        max_read_failures = 10  # tolerate transient camera glitches
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                if not self._camera_mode:
                    self._video_running = False
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            f"Video complete: {len(self._results_history)} frames"
                        ),
                    )
                    self.root.after(0, self._on_video_complete)
                    break
                # Camera mode: tolerate transient failures
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_read_failures:
                    self.root.after(
                        0,
                        lambda: self._status_var.set(
                            "Camera read failed repeatedly — stopping"
                        ),
                    )
                    self._video_running = False
                    break
                time.sleep(0.01)
                continue
            consecutive_read_failures = 0
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            self._frame_count += 1
            self._current_image = frame
            if self._detector is None:
                self.root.after(0, self._update_progress, raw_frame_idx, True)
                continue
            manual_crop = self._get_manual_roi_crop(frame)
            if manual_crop is None:
                self._current_result = None
                self.root.after(0, self._update_progress, raw_frame_idx, True)
                continue
            crop, roi_x, roi_y = manual_crop
            self._sync_calibration_to_detector()
            result = self._detector.detect_video_frame(
                crop,
                frame_number=self._frame_count,
                roi_x=roi_x,
                roi_y=roi_y,
            )
            result = self._apply_manual_ring_policy(result)
            result.metadata.source = source_name
            if self._tracker is not None:
                smoothed = self._tracker.update(result)
            else:
                smoothed = result
            if smoothed.has_both and self._corneal_calc is not None:
                smoothed.corneal_center = self._corneal_calc.calculate(
                    smoothed.pupil,
                    smoothed.limbus,
                    result.calibration,
                )
            smoothed.calibration = result.calibration
            smoothed.ring_status = getattr(result, "ring_status", "unknown")
            smoothed.ring_center = getattr(result, "ring_center", None)
            smoothed.ring_radius = getattr(result, "ring_radius", None)
            smoothed.ring_contour = getattr(result, "ring_contour", None)
            smoothed.ring_dot_count = getattr(result, "ring_dot_count", 0)
            smoothed.corneal_reference_source = getattr(
                result,
                "corneal_reference_source",
                "limbus",
            )
            if (
                smoothed.ring_status == "ring_present"
                and smoothed.ring_center is not None
                and getattr(smoothed, "pupil", None) is not None
                and getattr(smoothed.pupil, "ellipse", None) is not None
            ):
                px = smoothed.pupil.ellipse.center_x
                py = smoothed.pupil.ellipse.center_y
                points = [(px, py, "pupil")]
                weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]
                if getattr(smoothed, "limbus", None) is not None and getattr(smoothed.limbus, "ellipse", None) is not None:
                    points.append(
                        (
                            smoothed.limbus.ellipse.center_x,
                            smoothed.limbus.ellipse.center_y,
                            "limbus",
                        )
                    )
                    weights.append(max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3))
                points.append((smoothed.ring_center[0], smoothed.ring_center[1], "ring"))
                weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                total_w = sum(weights)
                rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                smoothed.corneal_center.center_px = (rcx, rcy)
                smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                smoothed.corneal_center.offset_magnitude_px = math.hypot(
                    px - rcx,
                    py - rcy,
                )
                smoothed.corneal_center.offset_angle_deg = math.degrees(
                    math.atan2(py - rcy, px - rcx)
                )
                smoothed.corneal_center.valid = True
                if result.calibration.calibrated:
                    smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm((rcx, rcy))
                    dx_mm = (px - rcx) * result.calibration.mm_per_px
                    dy_mm = (py - rcy) * result.calibration.mm_per_px
                    smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
                    smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)
            self._current_result = smoothed
            self._results_history.append(smoothed.to_dict())

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write synchronised raw + overlay frame pair
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, smoothed)
                self._write_frame_to_recorder(
                    annotated,
                    raw_frame=frame,
                    result=smoothed,
                    frame_index=raw_frame_idx,
                )
            # ══════════════════════════════════════════════════════════

            now = time.monotonic()
            display_interval = self._get_display_interval()
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_classic_frame, smoothed)
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_classic_frame(self, result: Any) -> None:
        self._update_measurements(result)
        self._fps_var.set("---")
        self._refresh_display()

    def _start_video_optimized(self, engine: Any, src_name: str) -> None:
        model_path = self._find_model_path() or "models/best_model.pth"
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=model_path,
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized video startup failed, falling back to classic: %s", exc)
            self._opt_processor = None
            self._using_optimized_camera = False
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized video startup failed - using classic pipeline")
            self._start_video_classic(src_name)
            return
        self._using_optimized_camera = True
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set(f"Playing (optimised): {src_name}")
        self._pipeline_var.set(f"Optimised [{preset_label}]")

        # Reset the ETA/FPS clock now that processing actually begins — the ROI
        # drawing wait and one-time engine warmup must not skew the frame rate.
        self._video_start_time = time.monotonic()
        self._video_thread = threading.Thread(
            target=self._video_loop_optimized,
            args=(src_name,),
            daemon=True,
            name="VideoLoopOptimised",
        )
        self._video_thread.start()

    def _video_loop_optimized(self, source_name: str) -> None:
        stride = max(1, self._stride_var.get())
        raw_frame_idx = 0
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        # Use decode-ahead threading for video files (not camera)
        if not self._camera_mode and self._video_cap is not None:
            import queue as _queue

            frame_queue = _queue.Queue(
                maxsize=max(6, self._runtime_profile.recommended_capture_buffer * 3)
            )
            from pupil_tracking.video.optimized_processor import _FrameReader
            reader = _FrameReader(
                self._video_cap,
                frame_queue,
                stride=stride,
                pause_predicate=lambda: self._video_paused or self._awaiting_roi,
            )
            reader.start()

            try:
                while self._video_running:
                    if self._video_paused or self._awaiting_roi:
                        time.sleep(0.05)
                        continue
                    try:
                        item = frame_queue.get(timeout=5.0)
                    except _queue.Empty:
                        break
                    if item is None:
                        # Video complete
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
                    pending_end = False
                    if (
                        self._opt_processor is not None
                        and self._opt_processor.should_shed_input_frames()
                        and frame_queue.qsize() > 3
                    ):
                        try:
                            queued_item = frame_queue.get_nowait()
                        except _queue.Empty:
                            queued_item = None
                        if queued_item is None:
                            pending_end = True
                        else:
                            item = queued_item
                    raw_frame_idx, frame = item
                    if self._opt_processor is not None:
                        self._opt_processor.note_source_frame(raw_frame_idx)
                    self._frame_count += 1
                    self._current_image = frame
                    try:
                        frame_result = self._opt_processor.process_frame(
                            frame, self._frame_count
                        )
                    except Exception as exc:
                        self.logger.error("Optimised video frame error: %s", exc)
                        continue
                    try:
                        fr_ns = self._dict_to_frame_ns(frame_result)
                        adapted = self._adapt_frame_result(fr_ns, frame.shape)
                    except Exception as exc:
                        self.logger.error("Optimised adapt error: %s", exc)
                        continue
                    self._current_result = adapted
                    try:
                        self._results_history.append(adapted.to_dict())
                    except Exception as exc:
                        self.logger.error("Optimised to_dict error: %s", exc)

                    # ══════════════════════════════════════════════════════════
                    # RECORDING — Write synchronised raw + overlay frame pair
                    # ══════════════════════════════════════════════════════════
                    if self._recorder.is_recording:
                        annotated = self._prepare_recording_frame(frame, adapted)
                        self._write_frame_to_recorder(
                            annotated,
                            raw_frame=frame,
                            result=adapted,
                            frame_index=raw_frame_idx,
                        )
                    # ══════════════════════════════════════════════════════════

                    fps_counter += 1
                    now = time.monotonic()
                    display_interval = self._get_display_interval()
                    elapsed_fps = now - fps_timer
                    if elapsed_fps >= 1.0:
                        current_fps = fps_counter / elapsed_fps
                        fps_counter = 0
                        fps_timer = now
                    if (now - self._last_display_update) >= display_interval:
                        self._last_display_update = now
                        _fps = current_fps
                        self.root.after(
                            0,
                            self._on_optimized_video_frame,
                            adapted,
                            _fps,
                        )
                    self.root.after(0, self._update_progress, raw_frame_idx, True)
                    if pending_end:
                        self._video_running = False
                        self.root.after(
                            0,
                            lambda: self._status_var.set(
                                f"Video complete (optimised): "
                                f"{len(self._results_history)} frames"
                            ),
                        )
                        self.root.after(0, self._on_video_complete)
                        break
            finally:
                reader.stop()
                reader.join(timeout=3.0)

            # If loop ended naturally (not via stop button), signal completion
            if not self._video_running:
                return
            self._video_running = False
            self.root.after(
                0,
                lambda: self._status_var.set(
                    f"Video complete (optimised): {len(self._results_history)} frames"
                ),
            )
            self.root.after(0, self._on_video_complete)
            return

        # Fallback: synchronous read (camera via _start_video path)
        while self._video_running and self._video_cap is not None:
            stride = max(1, self._stride_var.get())
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue
            ret, frame = self._video_cap.read()
            if not ret:
                self._video_running = False
                self.root.after(
                    0,
                    lambda: self._status_var.set(
                        f"Video complete (optimised): "
                        f"{len(self._results_history)} frames"
                    ),
                )
                self.root.after(0, self._on_video_complete)
                break
            raw_frame_idx += 1
            if stride > 1 and (raw_frame_idx % stride) != 0:
                self.root.after(0, self._update_progress, raw_frame_idx, False)
                continue
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(raw_frame_idx)
            self._frame_count += 1
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(
                    frame, self._frame_count
                )
            except Exception as exc:
                self.logger.error("Optimised video frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write synchronised raw + overlay frame pair
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(
                    annotated,
                    raw_frame=frame,
                    result=adapted,
                    frame_index=raw_frame_idx,
                )
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now
            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                _fps = current_fps
                self.root.after(
                    0,
                    self._on_optimized_video_frame,
                    adapted,
                    _fps,
                )
            self.root.after(0, self._update_progress, raw_frame_idx, True)

    def _on_optimized_video_frame(self, adapted: Any, fps: float) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Video ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set(f"{fps:.1f}")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._refresh_display()

    def _update_progress(self, raw_frame_idx: int, processed: bool) -> None:
        total = self._video_total_frames
        if total <= 0:
            self._progress_label_var.set(f"Frame {raw_frame_idx}")
            return
        pct = raw_frame_idx / total * 100.0
        self._progress_label_var.set(
            f"Frame {raw_frame_idx}/{total}  ({pct:.1f}%)  —  "
            f"{len(self._results_history)} processed"
        )
        elapsed = time.monotonic() - self._video_start_time
        if raw_frame_idx > 0 and elapsed > 0.5:
            remaining_frames = total - raw_frame_idx
            rate = raw_frame_idx / elapsed
            if rate > 0:
                eta_sec = remaining_frames / rate
                if eta_sec < 60:
                    self._eta_label_var.set(f"ETA: {eta_sec:.0f}s")
                elif eta_sec < 3600:
                    m, s = divmod(int(eta_sec), 60)
                    self._eta_label_var.set(f"ETA: {m}m {s}s")
                else:
                    h, rem = divmod(int(eta_sec), 3600)
                    m, s = divmod(rem, 60)
                    self._eta_label_var.set(f"ETA: {h}h {m}m")
            else:
                self._eta_label_var.set("")
        else:
            self._eta_label_var.set("")

    def _on_video_complete(self) -> None:
        elapsed = time.monotonic() - self._video_start_time
        n = len(self._results_history)
        avg_fps = n / elapsed if elapsed > 0 else 0
        self._progress_label_var.set(
            f"Done — {n} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)"
        )
        self._eta_label_var.set("✓ Complete")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")

    # ================================================================
    # Camera
    # ================================================================

    def _start_camera(self) -> None:
        self._stop_video()
        self._reset_elita_shift()
        self._camera_mode = True
        engine = self._get_fast_engine() if self._use_optimized_var.get() else None
        if engine is not None:
            self._start_camera_optimized(engine)
        else:
            self._start_video(0)

    def _start_camera_optimized(self, engine: Any) -> None:
        try:
            self._async_capture = AsyncCapture(
                0,
                buffer_size=self._runtime_profile.recommended_capture_buffer,
            )
            self._async_capture.start()
        except RuntimeError as exc:
            messagebox.showerror("Camera Error", str(exc))
            return
        self._active_source = 0
        try:
            self._opt_processor = OptimizedVideoProcessor(
                model_path=self._find_model_path() or "models/best_model.pth",
                device="auto",
                input_size=self._resolution_var.get(),
                half_precision=self._fp16_var.get(),
                use_compile=self._compile_var.get(),
                enable_auto_roi=self._roi_var.get(),
                roi_cache_ttl=self._roi_cache_var.get(),
                fast_mode=True,
                skip_quality_check=False,
                batch_size=self._runtime_profile.recommended_batch_size,
            )
            self._apply_manual_roi_to_processor()
            self._apply_manual_ring_to_processor()
            engine.warmup()
        except Exception as exc:
            self.logger.error("Optimized camera startup failed, falling back to classic: %s", exc)
            self._engine_status_var.set(f"Engine: fallback to classic - {exc}")
            self._status_var.set("Optimized camera startup failed - using classic pipeline")
            self._opt_processor = None
            if self._async_capture is not None:
                try:
                    self._async_capture.stop()
                except Exception:
                    pass
                self._async_capture = None
            self._using_optimized_camera = False
            self._start_video(0)
            return
        self._using_optimized_camera = True

        # Grab a preview frame so the operator can place the ROI before any
        # detection runs. The AsyncCapture is already streaming, so poll it
        # briefly for the first decoded frame.
        first = None
        for _ in range(40):
            data = self._async_capture.read(timeout=0.05)
            if data is not None:
                first = data[1]
                break
        if first is None:
            messagebox.showerror("Camera Error", "No frames received from camera")
            self._stop_video()
            return
        self._arm_detection_gate(first, self._launch_camera_optimized, "Camera")

    def _launch_camera_optimized(self) -> None:
        self._using_optimized_camera = True
        self._video_running = True
        self._video_paused = False
        self._frame_count = 0
        self._results_history.clear()
        self._pause_btn.config(state=tk.NORMAL, text="⏸ Pause")
        preset_label = self._performance_preset_var.get().replace("_", " ").title()
        self._status_var.set("Camera (optimised) — starting…")
        self._pipeline_var.set(f"Optimised [{preset_label}]")
        self._progress_label_var.set("Live camera")
        self._eta_label_var.set("")
        self._video_thread = threading.Thread(
            target=self._camera_loop_optimized,
            daemon=True,
            name="OptCameraLoop",
        )
        self._video_thread.start()

    def _camera_loop_optimized(self) -> None:
        fps_counter = 0
        fps_timer = time.monotonic()
        current_fps = 0.0

        while self._video_running and self._async_capture is not None:
            if self._video_paused or self._awaiting_roi:
                time.sleep(0.05)
                continue

            # Check for capture errors
            cap_err = self._async_capture.get_error()
            if cap_err is not None:
                self.logger.error("Camera capture error: %s", cap_err)
                self.root.after(
                    0,
                    lambda e=str(cap_err): self._status_var.set(f"Camera error: {e}"),
                )
                break

            data = self._async_capture.read(timeout=0.05)
            if data is None:
                continue
            fnum, frame, _ts = data
            if self._opt_processor is not None:
                self._opt_processor.note_source_frame(fnum)

            # Skip stale frames aggressively to keep surgical UI responsive.
            frame_age = time.time() - _ts
            stale_threshold = (
                self._opt_processor.get_stale_frame_threshold_s()
                if self._opt_processor is not None
                else 0.12
            )
            if frame_age > stale_threshold:
                if self._opt_processor is not None:
                    self._opt_processor.note_stale_frame()
                continue

            self._frame_count = fnum
            self._current_image = frame
            try:
                frame_result = self._opt_processor.process_frame(frame, fnum)
            except Exception as exc:
                self.logger.error("Optimised frame error: %s", exc)
                continue
            try:
                fr_ns = self._dict_to_frame_ns(frame_result)
                adapted = self._adapt_frame_result(fr_ns, frame.shape)
            except Exception as exc:
                self.logger.error("Optimised camera adapt error: %s", exc)
                continue
            self._current_result = adapted
            try:
                self._results_history.append(adapted.to_dict())
            except Exception as exc:
                self.logger.error("Optimised camera to_dict error: %s", exc)

            # ══════════════════════════════════════════════════════════
            # RECORDING — Write synchronised raw + overlay frame pair
            # ══════════════════════════════════════════════════════════
            if self._recorder.is_recording:
                annotated = self._prepare_recording_frame(frame, adapted)
                self._write_frame_to_recorder(
                    annotated,
                    raw_frame=frame,
                    result=adapted,
                    frame_index=fnum,
                )
            # ══════════════════════════════════════════════════════════

            fps_counter += 1
            now = time.monotonic()
            display_interval = self._get_display_interval()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                current_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = now

            if (now - self._last_display_update) >= display_interval:
                self._last_display_update = now
                self.root.after(0, self._on_optimized_frame, adapted, current_fps)

    def _on_optimized_frame(self, adapted: SimpleNamespace, fps: float = 0.0) -> None:
        self._update_measurements(adapted)
        if self._opt_processor is not None:
            stats = self._opt_processor.get_stats()
            self._last_opt_stats = dict(stats)
            res = stats.get("resolution", "?")
            skip = stats.get("frame_skip", 0)
            roi_mode = stats.get("roi_mode", "off")
            roi = {"manual": "M", "auto": "A", "off": "N"}.get(roi_mode, "Y")
            lat_avg = stats.get("latency_recent_ms", stats.get("latency_avg_ms", 0.0))
            dropped = stats.get("dropped_frames", 0)
            stale = stats.get("stale_frames", 0)
            tracking_state = self._derive_tracking_state(adapted, stats)
            self._fps_var.set(f"{fps:.1f}  (res {res}, skip {skip}, ROI {roi})")
            self._latency_avg_var.set(f"{lat_avg:.1f} ms")
            self._drop_var.set(f"{dropped} / {stale}")
            self._tracking_state_var.set(tracking_state)
            self._set_summary_tracking_state(tracking_state)
            overload_suffix = "  |  Overload Protection Active" if stats.get("overload_active") else ""
            self._status_var.set(
                f"Camera ({tracking_state.lower()}) — {lat_avg:.0f} ms avg latency{overload_suffix}"
            )
        else:
            self._fps_var.set("---")
            self._latency_avg_var.set("---")
            self._drop_var.set("---")
            self._tracking_state_var.set("---")
        self._progress_label_var.set(f"Frame {self._frame_count}")
        self._refresh_display()

    # ================================================================
    # Dict → namespace bridge for optimized pipeline results
    # ================================================================

    @staticmethod
    def _dict_to_frame_ns(d: dict) -> SimpleNamespace:
        """Convert a flat detection dict (from OptimizedVideoProcessor /
        FastInference / result_to_dict) into the SimpleNamespace that
        ``_adapt_frame_result`` expects."""
        pupil_det = d.get("pupil_detected", False)
        limbus_det = d.get("limbus_detected", False)

        if pupil_det:
            px, py = d.get("pupil_x", 0.0), d.get("pupil_y", 0.0)
            pr = d.get("pupil_radius", 0.0)
            # pupil_major/minor are already full-axis diameters from FastInference
            p_major = d.get("pupil_major", pr * 2)
            p_minor = d.get("pupil_minor", pr * 2)
            p_angle = d.get("pupil_angle", 0.0)
            pupil_center = (px, py)
            pupil_axes = (p_major, p_minor)
            pupil_angle = p_angle
        else:
            pupil_center = None
            pupil_axes = None
            pupil_angle = 0.0

        if limbus_det:
            lx, ly = d.get("limbus_x", 0.0), d.get("limbus_y", 0.0)
            lr = d.get("limbus_radius", 0.0)
            # limbus_major/minor are already full-axis diameters from FastInference
            l_major = d.get("limbus_major", lr * 2)
            l_minor = d.get("limbus_minor", lr * 2)
            l_angle = d.get("limbus_angle", 0.0)
            limbus_center = (lx, ly)
            limbus_axes = (l_major, l_minor)
            limbus_angle = l_angle
        else:
            limbus_center = None
            limbus_axes = None
            limbus_angle = 0.0

        conf = d.get(
            "overall_confidence",
            d.get("pupil_confidence", d.get("confidence", 0.0)),
        )
        pupil_confidence = d.get("pupil_confidence", conf)
        limbus_confidence = d.get("limbus_confidence", conf)

        q_str = d.get("overall_quality", "")
        quality = SimpleNamespace(value=q_str) if q_str else None

        return SimpleNamespace(
            pupil_center=pupil_center,
            pupil_axes=pupil_axes,
            pupil_angle=pupil_angle,
            limbus_center=limbus_center,
            limbus_axes=limbus_axes,
            limbus_angle=limbus_angle,
            confidence=conf,
            pupil_confidence=pupil_confidence,
            limbus_confidence=limbus_confidence,
            quality=quality,
            pupil_fit_type=d.get("pupil_fit_type"),
            limbus_fit_type=d.get("limbus_fit_type"),
            processing_ms=d.get("processing_time_ms", d.get("latency_ms", 0.0)),
            latency_ms=d.get("latency_ms", d.get("processing_time_ms", 0.0)),
            frame_number=d.get("frame_idx", 0),
            is_interpolated=not d.get("pupil_detected", False),
            ring_status=d.get("ring_status", "unknown"),
            ring_center=(
                (d.get("ring_center_x"), d.get("ring_center_y"))
                if d.get("ring_center_x") is not None and d.get("ring_center_y") is not None
                else None
            ),
            ring_radius=d.get("ring_radius"),
            ring_dot_count=d.get("ring_dot_count", 0),
            corneal_reference_source=d.get("corneal_reference_source", "limbus"),
            reuse_cached_result=bool(d.get("reuse_cached_result", False)),
            reuse_reason=d.get("reuse_reason"),
            _eye_result=d.get("_eye_result"),
        )

    # ================================================================
    # FrameResult adapter (unchanged from original)
    # ================================================================

    def _adapt_frame_result(
        self, fr: Any, frame_shape: Tuple[int, ...]
    ) -> SimpleNamespace:
        H, W = frame_shape[:2]
        eye_result = getattr(fr, "_eye_result", None)
        if eye_result is not None:
            selected_eye = (
                self._selected_eye_var.get()
                if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() in ("OD", "OS")
                else (getattr(fr, "eye", None) or getattr(eye_result, "eye", None) or "OD")
            )
            eye_result.eye = selected_eye
            eye_result.metadata.image_width = W
            eye_result.metadata.image_height = H
            eye_result.metadata.frame_number = getattr(fr, "frame_number", 0)
            eye_result.metadata.latency_ms = getattr(fr, "latency_ms", fr.processing_ms)

            # ── Override detector-internal calibration with GUI mode ──
            # The internal UnifiedDetector's StabilizedCalibrator is stuck
            # on ANATOMICAL_ANCHOR (never receives set_calibration_mode).
            # Build the correct calibration from the GUI dropdown, then
            # re-compute pre-computed mm attributes so to_dict() / CSV
            # export reflect the user's selection.
            _cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
            _corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
            _fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 58.2)
            _ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)
            if _cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
                _px = max(0.1, _fixed_scale)
                new_cal = CalibrationInfo(
                    calibrated=True,
                    px_per_mm=_px,
                    mm_per_px=1.0 / _px,
                    source="fixed_manual",
                    method="fixed_manual",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=1.0,
                    corneal_diameter_assumed_mm=None,
                )
            elif _cal_mode == "RING_REFLECTION":
                _ring_r = getattr(fr, "ring_radius", None)
                if _ring_r is not None and _ring_r > 10:
                    _dia = _ring_r * 2.0
                    _px = _dia / _ring_ref_mm
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px,
                        source=f"ring_reflection_{_ring_ref_mm:.1f}mm",
                        method="ring_reflection",
                        reference_diameter_mm=_ring_ref_mm,
                        reference_diameter_px=_dia,
                        confidence=0.95,
                        corneal_diameter_assumed_mm=None,
                    )
                elif (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (fallback)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=0.85,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="ring_reflection",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=None,
                    )
            else:

                # ANATOMICAL_ANCHOR
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True,
                        px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (optimised)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=min(0.95, getattr(eye_result, "overall_confidence", 0.0) + 0.05),
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False,
                        px_per_mm=0.0,
                        mm_per_px=0.0,
                        source="none",
                        method="anatomical",
                        reference_diameter_mm=0.0,
                        reference_diameter_px=0.0,
                        confidence=0.0,
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
            eye_result.calibration = new_cal

            # Clear stale pre-computed mm attributes set by
            # _add_mm_values / evaluate_clinical_wtw during the
            # original detection with the wrong calibration.
            for target in (
                getattr(eye_result, "limbus", None),
                getattr(eye_result, "pupil", None),
            ):
                if target is None:
                    continue
                for attr in (
                    "wtw_horizontal_mm",
                    "wtw_vertical_mm",
                    "wtw_mean_mm",
                    "wtw_astigmatism_mm",
                    "is_wtw_measured",
                    "wtw_validity_status",
                    "radius_mm",
                    "center_mm",
                ):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass

            # Re-compute mm values with the correct calibration
            if new_cal.calibrated:
                if (
                    getattr(eye_result, "pupil", None) is not None
                    and eye_result.pupil.detected
                    and eye_result.pupil.ellipse is not None
                ):
                    pe = eye_result.pupil.ellipse
                    eye_result.pupil.radius_mm = pe.radius * new_cal.mm_per_px
                    eye_result.pupil.center_mm = (
                        pe.center_x * new_cal.mm_per_px,
                        pe.center_y * new_cal.mm_per_px,
                    )
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and eye_result.limbus.detected
                    and eye_result.limbus.ellipse is not None
                ):
                    le = eye_result.limbus.ellipse
                    eye_result.limbus.radius_mm = le.radius * new_cal.mm_per_px
                    eye_result.limbus.center_mm = (
                        le.center_x * new_cal.mm_per_px,
                        le.center_y * new_cal.mm_per_px,
                    )
                    from pupil_tracking.calibration.spatial_calibration import (
                        evaluate_clinical_wtw,
                    )
                    h, v, m, astig, is_m, status = evaluate_clinical_wtw(
                        eye_result.limbus, new_cal,
                    )
                    eye_result.limbus.wtw_horizontal_mm = h
                    eye_result.limbus.wtw_vertical_mm = v
                    eye_result.limbus.wtw_mean_mm = m
                    eye_result.limbus.wtw_astigmatism_mm = astig
                    eye_result.limbus.is_wtw_measured = is_m
                    eye_result.limbus.wtw_validity_status = status
            return eye_result

        cal_mode = self._calibration_mode_var.get() if hasattr(self, "_calibration_mode_var") else "ANATOMICAL_ANCHOR"
        corneal_mm = float(self._corneal_ref_mm_var.get() if hasattr(self, "_corneal_ref_mm_var") else _CORNEAL_DIAMETER_MM)
        fixed_scale = float(self._fixed_scale_var.get() if hasattr(self, "_fixed_scale_var") else 58.2)
        ring_ref_mm = float(self._ring_ref_mm_var.get() if hasattr(self, "_ring_ref_mm_var") else 9.4)

        if cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = max(0.1, fixed_scale)
            cal = SimpleNamespace(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                reference_diameter_mm=0.0,
                reference_diameter_px=0.0,
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )
        elif cal_mode == "RING_REFLECTION":
            ring_radius = getattr(fr, "ring_radius", None)
            if ring_radius is not None and ring_radius > 10:
                dia_px = ring_radius * 2.0
                px_per_mm = dia_px / ring_ref_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm,
                    source=f"ring_reflection_{ring_ref_mm:.1f}mm",
                    method="ring_reflection",
                    reference_diameter_mm=ring_ref_mm,
                    reference_diameter_px=dia_px,
                    confidence=0.95,
                    corneal_diameter_assumed_mm=None,
                )
            elif fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=1.0 / px_per_mm if px_per_mm > 0 else 0.0,
                    source="limbus_semi_major (fallback)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.85, fr.confidence),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="ring_reflection",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=None,
                )
        else:  # ANATOMICAL_ANCHOR
            if fr.limbus_axes is not None:
                limbus_semi_major_dia = float(max(fr.limbus_axes))
                px_per_mm = limbus_semi_major_dia / corneal_mm
                mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 0.0
                cal = SimpleNamespace(
                    calibrated=True,
                    px_per_mm=px_per_mm,
                    mm_per_px=mm_per_px,
                    source="limbus_semi_major (optimised)",
                    method="anatomical",
                    reference_diameter_mm=corneal_mm,
                    reference_diameter_px=limbus_semi_major_dia,
                    confidence=min(0.95, fr.confidence + 0.05),
                    corneal_diameter_assumed_mm=corneal_mm,
                )
            else:
                cal = SimpleNamespace(
                    calibrated=False,
                    px_per_mm=0.0,
                    mm_per_px=0.0,
                    source="none",
                    method="anatomical",
                    reference_diameter_mm=0.0,
                    reference_diameter_px=0.0,
                    confidence=0.0,
                    corneal_diameter_assumed_mm=corneal_mm,
                )
        _MAP = {
            "SURGICAL": "SURGICAL",
            "CLINICAL": "CLINICAL",
            "INTERPOLATED": "RESEARCH",
            "PREDICTED": "RESEARCH",
            "FAILED": "NO_DETECTION",
        }
        if fr.quality:
            raw_quality = fr.quality.value
            if raw_quality in _MAP:
                q_str = _MAP[raw_quality]
            elif fr.pupil_center is None and fr.limbus_center is None:
                q_str = "NO_DETECTION"
            else:
                q_str = "INSUFFICIENT"
        else:
            q_str = "NO_DETECTION"
        try:
            overall_q = DetectionQuality(q_str)
        except (ValueError, KeyError):
            overall_q = SimpleNamespace(value=q_str)
        pupil_fit_type = getattr(fr, "pupil_fit_type", None)
        limbus_fit_type = getattr(fr, "limbus_fit_type", None)

        def _make_ellipse(center, axes, angle, fit_type=None):
            if center is None or axes is None:
                return None
            full_a, full_b = float(max(axes)), float(min(axes))
            semi_a, semi_b = full_a / 2.0, full_b / 2.0
            mean_radius = (semi_a + semi_b) / 2.0  # match EllipseParams convention
            ecc = (
                math.sqrt(max(0.0, 1.0 - (semi_b / semi_a) ** 2)) if semi_a > 0 else 0.0
            )
            circ = (semi_b / semi_a) if semi_a > 0 else 1.0
            return SimpleNamespace(
                center_x=center[0],
                center_y=center[1],
                radius=mean_radius,
                semi_major=semi_a,
                semi_minor=semi_b,
                angle_deg=angle,
                eccentricity=ecc,
                circularity=circ,
                fit_quality=fr.confidence,
                fit_rms_residual=0.0,
                num_contour_points=0,
                uncertainty_center_x=1.0,
                uncertainty_center_y=1.0,
                fit_type=fit_type,
            )

        p_ell = _make_ellipse(
            fr.pupil_center, fr.pupil_axes, fr.pupil_angle, pupil_fit_type
        )
        pupil = SimpleNamespace(
            detected=p_ell is not None,
            ellipse=p_ell,
            confidence=getattr(fr, "pupil_confidence", fr.confidence) if p_ell else 0.0,
            quality=overall_q,
            method=SimpleNamespace(value="ML_optimised"),
            fit_type=pupil_fit_type,
        )
        l_ell = _make_ellipse(
            fr.limbus_center, fr.limbus_axes, fr.limbus_angle, limbus_fit_type
        )
        limbus = SimpleNamespace(
            detected=l_ell is not None,
            ellipse=l_ell,
            confidence=getattr(fr, "limbus_confidence", fr.confidence) if l_ell else 0.0,
            quality=overall_q,
            method=SimpleNamespace(value="ML_optimised"),
            fit_type=limbus_fit_type,
        )
        ref_source = getattr(fr, "corneal_reference_source", "limbus")
        has_both = pupil.detected and limbus.detected
        if has_both:
            pe, le = pupil.ellipse, limbus.ellipse
            ring_center = getattr(fr, "ring_center", None)
            use_ring_reference = (
                getattr(fr, "ring_status", "unknown") == "ring_present"
                and ring_center is not None
            )
            pts = [(pe.center_x, pe.center_y, "pupil")]
            weights = [max(pupil.confidence, 1e-3)]
            pts.append((le.center_x, le.center_y, "limbus"))
            weights.append(max(limbus.confidence, 1e-3))
            if use_ring_reference:
                pts.append((ring_center[0], ring_center[1], "ring"))
                weights.append(max(getattr(fr, "confidence", 0.0), 1e-3))
            total_w = sum(weights)
            ref_x = sum(pt[0] * w for pt, w in zip(pts, weights)) / total_w
            ref_y = sum(pt[1] * w for pt, w in zip(pts, weights)) / total_w
            ref_source = "+".join(name for _, _, name in pts)
            dx = pe.center_x - ref_x
            dy = pe.center_y - ref_y
            mag_px = math.hypot(dx, dy)
            ang = math.degrees(math.atan2(dy, dx))
            if cal.calibrated:
                dx_mm, dy_mm = dx * cal.mm_per_px, dy * cal.mm_per_px
                mag_mm = mag_px * cal.mm_per_px
                off_mm = (dx_mm, dy_mm)
            else:
                mag_mm, off_mm = None, None
            cc = SimpleNamespace(
                valid=True,
                center_px=(ref_x, ref_y),
                offset_px=(dx, dy),
                offset_magnitude_px=mag_px,
                offset_magnitude_mm=mag_mm,
                offset_mm=off_mm,
                offset_angle_deg=ang,
            )
        else:
            cc = SimpleNamespace(
                valid=False,
                center_px=(0.0, 0.0),
                offset_px=(0.0, 0.0),
                offset_magnitude_px=0.0,
                offset_magnitude_mm=None,
                offset_mm=None,
                offset_angle_deg=0.0,
            )
        meta = SimpleNamespace(
            processing_time_ms=fr.processing_ms,
            latency_ms=getattr(fr, "latency_ms", fr.processing_ms),
            frame_number=fr.frame_number,
            image_width=W,
            image_height=H,
            source="camera (optimised)",
            reuse_cached_result=bool(getattr(fr, "reuse_cached_result", False)),
            reuse_reason=getattr(fr, "reuse_reason", None),
        )
        alerts: List[str] = []
        if fr.is_interpolated:
            alerts.append("⚡ Interpolated frame (Kalman prediction)")
        if fr.quality is not None and fr.quality.value == "FAILED":
            alerts.append("⚠ Detection failed this frame")

        selected_eye = (
            self._selected_eye_var.get()
            if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() in ("OD", "OS")
            else (getattr(fr, "eye", None) or "OD")
        )
        result = SimpleNamespace(
            eye=selected_eye,
            pupil=pupil,
            limbus=limbus,
            corneal_center=cc,
            calibration=cal,
            metadata=meta,
            overall_quality=overall_q,
            overall_confidence=fr.confidence,
            has_both=has_both,
            alerts=alerts,
            ring_status=getattr(fr, "ring_status", "unknown"),
            ring_center=getattr(fr, "ring_center", None),
            ring_radius=getattr(fr, "ring_radius", None),
            ring_dot_count=getattr(fr, "ring_dot_count", 0),
            corneal_reference_source=ref_source,
        )
        result.to_dict = lambda _r=result, _fr=fr, _cal=cal: (
            self._frame_result_to_export_dict(_fr, _cal, _r)
        )
        return result

    def _frame_result_to_export_dict(
        self,
        fr: Any,
        cal: SimpleNamespace,
        adapted: SimpleNamespace,
    ) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "metadata": {
                "frame_number": fr.frame_number,
                "processing_time_ms": fr.processing_ms,
                "latency_ms": getattr(fr, "latency_ms", fr.processing_ms),
                "source": "camera (optimised)",
            },
            "overall_quality": (
                adapted.overall_quality.value
                if hasattr(adapted.overall_quality, "value")
                else str(adapted.overall_quality)
            ),
            "overall_confidence": fr.confidence,
            "calibration": {
                "calibrated": cal.calibrated,
                "mm_per_px": cal.mm_per_px,
                "px_per_mm": cal.px_per_mm,
                "source": getattr(cal, "source", "none"),
                "method": getattr(cal, "method", "anatomical"),
                "corneal_diameter_assumed_mm": getattr(cal, "corneal_diameter_assumed_mm", None),
            },
        }
        if fr.pupil_center is not None and fr.pupil_axes is not None:
            semi_a, semi_b = max(fr.pupil_axes) / 2.0, min(fr.pupil_axes) / 2.0
            # Mean radius — matches EllipseParams.radius and the Measurements
            # panel (which shows diameter = e.radius * 2). Using the mean (not
            # semi-major) is what makes the exported mm value vary per frame
            # instead of collapsing to the calibration constant.
            mean_r = (semi_a + semi_b) / 2.0
            mm = cal.mm_per_px if cal.calibrated else 0.0
            d["pupil"] = {
                "detected": True,
                "confidence": getattr(fr, "pupil_confidence", fr.confidence),
                "fit_type": getattr(fr, "pupil_fit_type", None),
                "radius_mm": (mean_r * mm) if cal.calibrated else None,
                "center_mm": (
                    (fr.pupil_center[0] * mm, fr.pupil_center[1] * mm)
                    if cal.calibrated else None
                ),
                "ellipse": {
                    "center_x": fr.pupil_center[0],
                    "center_y": fr.pupil_center[1],
                    "radius": mean_r,
                    "semi_major": semi_a,
                    "semi_minor": semi_b,
                    "angle_deg": float(getattr(fr, "pupil_angle", 0.0) or 0.0),
                    "diameter_mm": (mean_r * 2.0 * mm) if cal.calibrated else None,
                    "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                    "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
                },
            }
        else:
            d["pupil"] = {"detected": False, "ellipse": {}}
        if fr.limbus_center is not None and fr.limbus_axes is not None:
            semi_a, semi_b = max(fr.limbus_axes) / 2.0, min(fr.limbus_axes) / 2.0
            mean_r = (semi_a + semi_b) / 2.0
            mm = cal.mm_per_px if cal.calibrated else 0.0
            wtw_h = (2.0 * semi_a * mm) if cal.calibrated else None
            wtw_v = (2.0 * semi_b * mm) if cal.calibrated else None
            wtw_m = (mean_r * 2.0 * mm) if cal.calibrated else None
            wtw_astig = (abs(wtw_h - wtw_v)) if (wtw_h is not None and wtw_v is not None) else None
            is_wtw_m = bool(cal.calibrated and getattr(cal, "method", "anatomical") != "anatomical")
            if not cal.calibrated:
                wtw_status = "UNAVAILABLE"
            elif not is_wtw_m:
                wtw_status = "ANCHORED_BASELINE"
            else:
                wtw_status = "VALID_CLINICAL_RANGE" if (wtw_m is not None and 9.5 <= wtw_m <= 13.5) else "OUT_OF_BOUNDS_WARNING"

            d["limbus"] = {
                "detected": True,
                "confidence": getattr(fr, "limbus_confidence", fr.confidence),
                "fit_type": getattr(fr, "limbus_fit_type", None),
                "radius_mm": (mean_r * mm) if cal.calibrated else None,
                "center_mm": (
                    (fr.limbus_center[0] * mm, fr.limbus_center[1] * mm)
                    if cal.calibrated else None
                ),
                "wtw_horizontal_mm": wtw_h,
                "wtw_vertical_mm": wtw_v,
                "wtw_mean_mm": wtw_m,
                "wtw_astigmatism_mm": wtw_astig,
                "is_wtw_measured": is_wtw_m,
                "wtw_validity_status": wtw_status,
                "ellipse": {
                    "center_x": fr.limbus_center[0],
                    "center_y": fr.limbus_center[1],
                    "radius": mean_r,
                    "semi_major": semi_a,
                    "semi_minor": semi_b,
                    "angle_deg": float(getattr(fr, "limbus_angle", 0.0) or 0.0),

                    # In anatomical mode, mean_r*2*mm ≡ wtw_m (algebraically
                    # identical).  Use wtw_m explicitly so the CSV semantically
                    # agrees with the WTW card rather than re-deriving from px.
                    "diameter_mm": wtw_m if (cal.calibrated and wtw_m is not None) else ((mean_r * 2.0 * mm) if cal.calibrated else None),
                    "semi_major_mm": (semi_a * mm) if cal.calibrated else None,
                    "semi_minor_mm": (semi_b * mm) if cal.calibrated else None,
                },
            }
        else:
            d["limbus"] = {
                "detected": False,
                "ellipse": {},
                "is_wtw_measured": False,
                "wtw_validity_status": "UNAVAILABLE",
            }

        ring_center = getattr(fr, "ring_center", None)
        use_ring_reference = (
            getattr(fr, "ring_status", "unknown") == "ring_present"
            and ring_center is not None
        )
        if fr.pupil_center is not None and (
            use_ring_reference or fr.limbus_center is not None
        ):
            points = [(fr.pupil_center[0], fr.pupil_center[1], "pupil")]
            weights = [max(getattr(fr, "pupil_confidence", fr.confidence), 1e-3)]
            if fr.limbus_center is not None:
                points.append((fr.limbus_center[0], fr.limbus_center[1], "limbus"))
                weights.append(max(getattr(fr, "limbus_confidence", fr.confidence), 1e-3))
            if use_ring_reference:
                points.append((ring_center[0], ring_center[1], "ring"))
                weights.append(max(fr.confidence, 1e-3))
            total_w = sum(weights)
            ref_center = (
                sum(pt[0] * w for pt, w in zip(points, weights)) / total_w,
                sum(pt[1] * w for pt, w in zip(points, weights)) / total_w,
            )
            dx = fr.pupil_center[0] - ref_center[0]
            dy = fr.pupil_center[1] - ref_center[1]
            mag_px = math.hypot(dx, dy)
            d["corneal_center"] = {
                "center_px": ref_center,
                "offset_magnitude_px": mag_px,
                "offset_magnitude_mm": (
                    mag_px * cal.mm_per_px if cal.calibrated else None
                ),
                "offset_angle_deg": math.degrees(math.atan2(dy, dx)),
            }
        else:
            d["corneal_center"] = {}
        d["ring_status"] = getattr(fr, "ring_status", "unknown")
        d["corneal_reference_source"] = getattr(fr, "corneal_reference_source", "limbus")
        if ring_center is not None:
            d["ring_center_x"] = ring_center[0]
            d["ring_center_y"] = ring_center[1]
        d["ring_radius"] = getattr(fr, "ring_radius", None)
        d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)
        return d

    # ================================================================
    # Stop
    # ================================================================

    def _stop_video(self) -> None:
        self._video_running = False
        self._video_paused = False
        self._camera_mode = False
        if not self._restart_in_progress:
            self._active_source = None
        self._roi_edit_active = False
        self._roi_drag_mode = None
        self._roi_drag_offset = (0.0, 0.0)
        self._roi_original_before_edit = None
        self._roi_preview = None

        # Clear any pending ROI-gated detection start.
        self._awaiting_roi = False
        self._pending_detection_start = None
        self._canvas.configure(cursor="crosshair")
        self._sync_roi_ring_button_styles()
        if self._video_thread is not None:
            try:
                if threading.current_thread() is not self._video_thread:
                    self._video_thread.join(timeout=2.5)
            except Exception:
                pass
            self._video_thread = None
        if self._async_capture is not None:
            try:
                self._async_capture.stop()
            except Exception:
                pass
            self._async_capture = None
        if self._video_cap is not None:
            try:
                self._video_cap.release()
            except Exception:
                pass
            self._video_cap = None
        if self._opt_processor is not None:
            try:
                self._opt_processor.reset()
            except Exception:
                pass
            self._opt_processor = None
        self._last_opt_stats = {}
        self._using_optimized_camera = False
        self._pipeline_var.set("---")
        self._fps_var.set("---")
        self._pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        # ══════════════════════════════════════════════════════════
        # RECORDING — Auto-stop recording when video stops
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()
        # ══════════════════════════════════════════════════════════

    # ================================================================
    # Display
    # ================================================================

    def _refresh_display(self) -> None:
        try:
            if self._current_image is None:
                self._draw_welcome_screen()
                return
            if not self._canvas.winfo_ismapped():
                return
        except Exception as exc:
            self.logger.exception("Display pre-check error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")
            return

        if self._current_image is None:
            self._draw_welcome_screen()
            return
        if not self._canvas.winfo_ismapped():
            return

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 10 of 12 — Convert display frame
        #
        # The detector already processed the original image.
        # Now convert the DISPLAY copy to grayscale if mode
        # is active.  Overlays (green/blue circles) are drawn
        # on this grayscale background — IR camera look.
        # ══════════════════════════════════════════════════════════
        try:
            mode = self._grayscale_mode_var.get()
            if mode == "off":
                image = self._current_image
            else:
                image = self._convert_display_frame(self._current_image.copy())

            canvas_w = self._canvas.winfo_width()
            canvas_h = self._canvas.winfo_height()
            if canvas_w < 10 or canvas_h < 10:
                return
            h, w = image.shape[:2]
            scale = min(canvas_w / w, canvas_h / h, 1.0)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2
            self._display_scale = scale
            self._display_origin = (offset_x, offset_y)
            self._display_size = (new_w, new_h)

            display = cv2.resize(image, (new_w, new_h))

            result = self._current_result
            if result is not None and self._show_overlay.get():
                self._draw_overlay_scaled(display, result, scale)

            # The centration cross overlay was removed on request -- the big
            # yellow cross it drew in the normal window was unwanted.
            self._draw_manual_roi_overlay(display, scale)
            self._draw_manual_ring_overlay(display, scale)
            if self._show_debug_overlay.get():
                self._draw_debug_overlay(display, scale)

            # ADDITIVE display-only: keep the live iris unwrap in step with
            # the current frame for the unwrap panel below.
            self._update_elita_unwrap()
            self._show_image_fast(display, canvas_w, canvas_h, new_w, new_h)
        except Exception as exc:
            self.logger.exception("Display refresh error: %s", exc)
            self._report_runtime_issue("Display recovered from an internal error")

    def _draw_welcome_screen(self) -> None:
        c = self._colors
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self._canvas.delete("all")
        self._canvas_image_id = None
        cx, cy = cw // 2, ch // 2
        self._canvas.create_text(
            cx,
            cy - 60,
            text="Medevplus IXcentai",
            fill=c.FG_PRIMARY,
            font=("Segoe UI", 22, "bold"),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy - 20,
            text="surgical grade",
            fill=c.ACCENT,
            font=("Segoe UI", 13),
            anchor="center",
        )
        self._canvas.create_text(
            cx,
            cy + 20,
            text="Open an image, video, or start the camera to begin",
            fill=c.FG_SECONDARY,
            font=("Segoe UI", 11),
            anchor="center",
        )
        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 11 of 12 — Updated welcome shortcuts
        # RECORDING — Updated shortcuts to include recording
        # ══════════════════════════════════════════════════════════
        shortcuts = (
            "Ctrl+O  Image    Ctrl+V  Video    Space  Pause"
            "    G  Grayscale    Ctrl+R  Record    Ctrl+Q  Quit"
        )
        # ══════════════════════════════════════════════════════════
        self._canvas.create_text(
            cx,
            cy + 60,
            text=shortcuts,
            fill=c.FG_TERTIARY,
            font=("Consolas", 9),
            anchor="center",
        )

    def _show_image_fast(
        self,
        image_bgr: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        img_w: int,
        img_h: int,
    ) -> None:
        """Display an already-resized BGR image, reusing the canvas item."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image = ImageTk.PhotoImage(pil_img)
        x = (canvas_w - img_w) // 2
        y = (canvas_h - img_h) // 2
        if self._canvas_image_id is not None:
            try:
                self._canvas.coords(self._canvas_image_id, x, y)
                self._canvas.itemconfig(
                    self._canvas_image_id, image=self._display_image
                )
                return
            except tk.TclError:
                self._canvas_image_id = None
        self._canvas.delete("all")
        self._canvas_image_id = self._canvas.create_image(
            x, y, anchor=tk.NW, image=self._display_image
        )

    @staticmethod
    def _preview_ellipse_points(
        e: Any,
        scale_x: float,
        scale_y: float,
        offset: float = 2.0,
        segments: int = 72,
        offset_x: Optional[float] = None,
        offset_y: Optional[float] = None,
    ) -> list:
        """Sample a *rotated* ellipse into canvas points for the preview.
        The detection ellipses carry an ``angle_deg`` tilt, so a plain
        ``create_oval`` would be wrong.  ``scale_x``/``scale_y`` map full
        resolution image coordinates to the preview thumbnail, and ``offset``
        is the thumbnail's top-left margin (the preview draws the image at
        ``(2, 2)`` by default or at ``(offset_x, offset_y)`` when centered).
        Returns a flat list ``[x0, y0, x1, y1, ...]``.
        """
        ox = offset if offset_x is None else float(offset_x)
        oy = offset if offset_y is None else float(offset_y)
        cx = float(e.center_x) * scale_x + ox
        cy = float(e.center_y) * scale_y + oy
        a = max(1.0, float(e.semi_major) * scale_x)
        b = max(1.0, float(e.semi_minor) * scale_y)
        ang = math.radians(float(getattr(e, "angle_deg", 0.0) or 0.0))
        ca, sa = math.cos(ang), math.sin(ang)
        points: list = []
        for i in range(int(segments)):
            t = 2.0 * math.pi * i / float(segments)
            x = a * math.cos(t)
            y = b * math.sin(t)
            points.append(cx + x * ca - y * sa)
            points.append(cy + x * sa + y * ca)
        return points

    @staticmethod
    def _offset_detection_result(result: Any, offset_x: float, offset_y: float) -> Any:
        """Offset pupil, limbus, and corneal center coordinates from crop to full image space."""
        if result is None or (offset_x == 0.0 and offset_y == 0.0):
            return result
        import copy
        try:
            res = copy.deepcopy(result)
        except Exception:
            return result

        def _shift_ellipse(e: Any) -> None:
            if e is None:
                return
            try:
                e.center_x = float(e.center_x) + offset_x
                e.center_y = float(e.center_y) + offset_y
            except Exception:
                pass

        pupil = getattr(res, "pupil", None)
        if pupil is not None:
            if getattr(pupil, "ellipse", None) is not None:
                _shift_ellipse(pupil.ellipse)
            if hasattr(pupil, "center_x"):
                pupil.center_x = float(pupil.center_x) + offset_x
            if hasattr(pupil, "center_y"):
                pupil.center_y = float(pupil.center_y) + offset_y
            if getattr(pupil, "contour_points", None) is not None:
                try:
                    pupil.contour_points = np.asarray(pupil.contour_points, dtype=np.float64) + np.array([offset_x, offset_y])
                except Exception:
                    pass

        limbus = getattr(res, "limbus", None)
        if limbus is not None:
            if getattr(limbus, "ellipse", None) is not None:
                _shift_ellipse(limbus.ellipse)
            if hasattr(limbus, "center_x"):
                limbus.center_x = float(limbus.center_x) + offset_x
            if hasattr(limbus, "center_y"):
                limbus.center_y = float(limbus.center_y) + offset_y
            if getattr(limbus, "contour_points", None) is not None:
                try:
                    limbus.contour_points = np.asarray(limbus.contour_points, dtype=np.float64) + np.array([offset_x, offset_y])
                except Exception:
                    pass

        cc = getattr(res, "corneal_center", None)
        if cc is not None:
            try:
                if hasattr(cc, "center_px") and cc.center_px is not None:
                    cx, cy = cc.center_px
                    cc.center_px = (float(cx) + offset_x, float(cy) + offset_y)
                if hasattr(cc, "center_x"):
                    cc.center_x = float(cc.center_x) + offset_x
                if hasattr(cc, "center_y"):
                    cc.center_y = float(cc.center_y) + offset_y
                if hasattr(cc, "x"):
                    cc.x = float(cc.x) + offset_x
                if hasattr(cc, "y"):
                    cc.y = float(cc.y) + offset_y
            except Exception:
                pass

        return res

    @staticmethod
    def _scale_ellipse(e: Any, scale: float) -> SimpleNamespace:
        """Return ellipse namespace with coordinates scaled for display."""
        return SimpleNamespace(
            center_x=e.center_x * scale,
            center_y=e.center_y * scale,
            radius=e.radius * scale,
            semi_major=e.semi_major * scale,
            semi_minor=e.semi_minor * scale,
            angle_deg=e.angle_deg,
        )

    def _get_ellipse_intersection(
        self, ellipse: Any, px: float, py: float, dx: float, dy: float
    ) -> Tuple[float, float]:
        """Compute the intersection of a ray from (px, py) in direction (dx, dy) with an ellipse."""
        cx = ellipse.center_x
        cy = ellipse.center_y
        a = max(1.0, ellipse.semi_major)
        b = max(1.0, ellipse.semi_minor)
        angle_rad = math.radians(ellipse.angle_deg)

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Local coordinates of start point relative to ellipse center
        x_loc = (px - cx) * cos_a + (py - cy) * sin_a
        y_loc = -(px - cx) * sin_a + (py - cy) * cos_a

        # Local direction vector
        dx_loc = dx * cos_a + dy * sin_a
        dy_loc = -dx * sin_a + dy * cos_a

        # Quadratic equation coefficients for distance t: A * t^2 + 2 * B * t + C = 0
        A_coef = (dx_loc / a) ** 2 + (dy_loc / b) ** 2
        B_coef = (x_loc * dx_loc) / (a ** 2) + (y_loc * dy_loc) / (b ** 2)
        C_coef = (x_loc / a) ** 2 + (y_loc / b) ** 2 - 1.0

        disc = B_coef ** 2 - A_coef * C_coef
        if disc < 0 or A_coef == 0:
            # Fallback to a simple default bounding box boundary if math fails
            return px + dx * a, py + dy * b

        t = (-B_coef + math.sqrt(disc)) / A_coef
        return px + t * dx, py + t * dy

    def _draw_cross_section(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw intersecting horizontal/vertical cross section lines between pupil and limbus."""
        if not (
            result.pupil.detected
            and result.pupil.ellipse is not None
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            return

        p_ellipse = self._scale_ellipse(result.pupil.ellipse, scale)
        l_ellipse = self._scale_ellipse(result.limbus.ellipse, scale)

        p_cx = p_ellipse.center_x
        p_cy = p_ellipse.center_y

        # Calculate intersection points on the pupil boundary
        p_up_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, -1.0)
        p_down_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, 1.0)
        p_left_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, -1.0, 0.0)
        p_right_pt = self._get_ellipse_intersection(p_ellipse, p_cx, p_cy, 1.0, 0.0)

        l_cx = l_ellipse.center_x
        l_cy = l_ellipse.center_y

        # Calculate intersection points on the limbus boundary
        l_up_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, -1.0)
        l_down_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, 1.0)
        l_left_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, -1.0, 0.0)
        l_right_pt = self._get_ellipse_intersection(l_ellipse, l_cx, l_cy, 1.0, 0.0)

        # Colors (BGR)
        green_color = (0, 255, 0)      # Pupil green
        blue_color = (255, 100, 0)     # Limbus blue

        # Draw pupil cross
        cv2.line(out, (int(round(p_left_pt[0])), int(round(p_left_pt[1]))), (int(round(p_right_pt[0])), int(round(p_right_pt[1]))), green_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(p_up_pt[0])), int(round(p_up_pt[1]))), (int(round(p_down_pt[0])), int(round(p_down_pt[1]))), green_color, 1, cv2.LINE_AA)

        # Draw limbus cross
        cv2.line(out, (int(round(l_left_pt[0])), int(round(l_left_pt[1]))), (int(round(l_right_pt[0])), int(round(l_right_pt[1]))), blue_color, 1, cv2.LINE_AA)
        cv2.line(out, (int(round(l_up_pt[0])), int(round(l_up_pt[1]))), (int(round(l_down_pt[0])), int(round(l_down_pt[1]))), blue_color, 1, cv2.LINE_AA)

        # Draw ASCII degree labels following the clinical convention:
        # - For OD (Right Eye): Screen-Left is 0°, Screen-Right is 180°.
        # - For OS (Left Eye): Screen-Left is 0°, Screen-Right is 180°.
        # - Superior (UP) is 270°, Inferior (DOWN) is 90°.
        from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
        eye = (
            (self._selected_eye_var.get() if hasattr(self, "_selected_eye_var") and self._selected_eye_var.get() in ("OD", "OS") else None)
            or getattr(result, "eye", None)
            or "OD"
        )
        left_lbl, right_lbl, up_lbl, down_lbl = get_centration_meridian_labels(eye)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_sz = max(0.3, 0.4 * scale)
        lbl_color = (220, 220, 220)

        # UP: 270 degrees
        up_x, up_y = int(round(l_up_pt[0])), int(round(l_up_pt[1]))
        cv2.putText(out, up_lbl, (up_x - int(10 * scale), up_y - int(5 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # DOWN: 90 degrees
        down_x, down_y = int(round(l_down_pt[0])), int(round(l_down_pt[1]))
        cv2.putText(out, down_lbl, (down_x - int(7 * scale), down_y + int(12 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # LEFT
        left_x, left_y = int(round(l_left_pt[0])), int(round(l_left_pt[1]))
        cv2.putText(out, left_lbl, (left_x - int(15 * scale), left_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)

        # RIGHT
        right_x, right_y = int(round(l_right_pt[0])), int(round(l_right_pt[1]))
        cv2.putText(out, right_lbl, (right_x + int(5 * scale), right_y + int(4 * scale)), font, font_sz, lbl_color, 1, cv2.LINE_AA)



    def _draw_overlay_scaled(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Draw overlays on an already-resized image with scaled coords."""
        h, w = out.shape[:2]
        cal = result.calibration

        ring_status = getattr(result, "ring_status", "unknown")
        if ring_status == "ring_present":
            ring_center = getattr(result, "ring_center", None)
            ring_radius = getattr(result, "ring_radius", None)
            ring_contour = getattr(result, "ring_contour", None)
            if ring_center is not None and ring_radius is not None:
                cx = int(round(ring_center[0] * scale))
                cy = int(round(ring_center[1] * scale))
                rr = int(round(ring_radius * scale))
                if ring_contour is not None and len(ring_contour) >= 5:
                    scaled = np.round(ring_contour.astype(np.float32) * scale).astype(np.int32)
                    cv2.drawContours(out, [scaled], -1, (0, 0, 255), 2)
                else:
                    cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
                if self._show_ring_center.get():
                    _base = max(4, int(10 * scale))
                    _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                    if _cal_mm_per_px > 0:
                        _ring_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                    else:
                        _ring_cross_size = int(max(12, min(_base, 26)))
                    cv2.drawMarker(
                        out,
                        (cx, cy),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        _ring_cross_size,
                        2,
                        cv2.LINE_AA,
                    )
                if self._show_measurements.get():
                    label = f"R={ring_radius * 2.0:.0f}px"
                    if cal.calibrated:
                        label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                    cv2.putText(
                        out,
                        label,
                        (cx + 10, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, 0.45 * scale),
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw limbus first (larger) so pupil renders on top
        if (
            self._show_limbus.get()
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            e_orig = result.limbus.ellipse
            e = self._scale_ellipse(e_orig, scale)
            limbus_color = (255, 100, 0)
            limbus_alpha = self._limbus_fill_alpha_var.get() / 100.0
            if limbus_alpha > 0:
                self._draw_filled_structure(out, e, limbus_color, limbus_alpha)
            ct = self._draw_structure(out, e, limbus_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), limbus_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    dia_mm = dia_px * cal.mm_per_px
                    smaj_mm = e_orig.semi_major * cal.mm_per_px
                    smin_mm = e_orig.semi_minor * cal.mm_per_px
                    label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    limbus_color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw pupil second so it always appears on top of the limbus fill
        if (
            self._show_pupil.get()
            and result.pupil.detected
            and result.pupil.ellipse is not None
        ):
            e_orig = result.pupil.ellipse
            e = self._scale_ellipse(e_orig, scale)
            pupil_color = (0, 255, 0)
            pupil_alpha = self._pupil_fill_alpha_var.get() / 100.0
            if pupil_alpha > 0:
                self._draw_filled_structure(out, e, pupil_color, pupil_alpha)
            ct = self._draw_structure(out, e, pupil_color)
            if self._show_centers.get():
                cv2.circle(out, ct, max(2, int(4 * scale)), pupil_color, -1)
            if self._show_measurements.get():
                dia_px = e_orig.radius * 2.0
                label = f"D={dia_px:.0f}px"
                if cal.calibrated:
                    label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
                ft = getattr(e_orig, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                if ft:
                    label += f" [{ft}]"
                font_scale = max(0.3, 0.45 * scale)
                cv2.putText(
                    out,
                    label,
                    (ct[0] + 10, ct[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    pupil_color,
                    1,
                    cv2.LINE_AA,
                )

        cc = getattr(result, "corneal_center", None)
        if (
            self._show_centers.get()
            and cc is not None
            and getattr(cc, "valid", False)
            and getattr(cc, "center_px", None)
        ):
            center_pt = (
                int(round(cc.center_px[0] * scale)),
                int(round(cc.center_px[1] * scale)),
            )
            # Marker size is specified in *display* pixels. To keep the marker
            # size physically meaningful, we scale it using the current
            # calibration when available.
            #
            # Marker size is specified in *display* pixels.
            # Keep the marker size physically meaningful by converting
            # a desired physical offset (+3mm relative to the previous
            # baseline marker) into image pixels using calibration.
            _base = max(4, int(10 * scale))
            cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
            if cal_mm_per_px > 0:
                cursor_size = int(max(10, round(_base + (0.5 / cal_mm_per_px) * scale)))
            else:
                cursor_size = int(max(12, min(_base, 26)))
            cv2.drawMarker(
                out,
                center_pt,
                (255, 255, 255),
                cv2.MARKER_CROSS,
                cursor_size,
                2,
                cv2.LINE_AA,
            )
            if self._show_measurements.get():
                ref_name = getattr(result, "corneal_reference_source", "cornea")
                cv2.putText(
                    out,
                    f"Corneal Centre [{ref_name}]",
                    (center_pt[0] + 12, center_pt[1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.35, 0.46 * scale),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if self._show_offset.get() and result.has_both:
            p = result.pupil.ellipse
            p_pt = (int(round(p.center_x * scale)), int(round(p.center_y * scale)))
            if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
                ref_pt = (
                    int(round(cc.center_px[0] * scale)),
                    int(round(cc.center_px[1] * scale)),
                )
                dx = p.center_x - cc.center_px[0]
                dy = p.center_y - cc.center_px[1]
            else:
                l = result.limbus.ellipse
                ref_pt = (int(round(l.center_x * scale)), int(round(l.center_y * scale)))
                dx = p.center_x - l.center_x
                dy = p.center_y - l.center_y
            cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
            if self._show_centers.get():
                _base = max(4, int(10 * scale))
                _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                if _cal_mm_per_px > 0:
                    _offset_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                else:
                    _offset_cross_size = int(max(12, min(_base, 26)))
                cv2.drawMarker(
                    out,
                    ref_pt,
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    _offset_cross_size,
                    2,
                    cv2.LINE_AA,
                )
            if self._show_measurements.get():
                offset_px = math.hypot(dx, dy)
                mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
                label = f"{offset_px:.1f}px"
                if cal.calibrated:
                    label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
                font_scale = max(0.25, 0.4 * scale)
                cv2.putText(
                    out,
                    label,
                    (mid[0] + 5, mid[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        self._draw_cross_section(out, result, scale)

        # (Removed) On-image quality badge text (e.g. "SURGICAL (0.94)") that
        # was drawn top-left on the video.  The quality/confidence is still
        # shown in the ribbon badge and the measurements panel.
        font_scale_t = max(0.3, 0.5 * scale)
        cv2.putText(
            out,
            f"{result.metadata.processing_time_ms:.0f}ms",
            (w - max(80, int(100 * scale)), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_t,
            (180, 180, 180),
            1,
        )

        # (Removed) On-image "OVERLOAD PROTECTION" overlay that blinked
        # top-left while adaptive overload protection was active.  Overload
        # protection itself still runs; only the on-video text was removed.

        # ══════════════════════════════════════════════════════════
        # GRAYSCALE GUI 12 of 12 — Grayscale mode badge on image
        # ══════════════════════════════════════════════════════════
        # (Removed) Grayscale mode badge text overlay (top-right).
        # Grayscale processing (OFF/AUTO/FORCE) and overlays remain active; 
        # only the on-image text label was removed to reduce clutter.
        # ══════════════════════════════════════════════════════════

        font_scale_a = max(0.25, 0.4 * scale)
        for i, alert in enumerate(result.alerts[:3]):
            cv2.putText(
                out,
                alert[:80],
                (10, h - 15 - i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_a,
                (0, 100, 255),
                1,
            )

        self._draw_ruler_overlay(out, scale)

    def _draw_protractor(self, out: np.ndarray, result: Any, scale: float) -> None:
        """Removed on request: the big yellow centration cross it drew in the
        normal window was unwanted.  Kept as a no-op so any late caller is a
        harmless no-op instead of a crash (display-only, additive).
        """
        return

    def _draw_debug_overlay(self, out: np.ndarray, scale: float) -> None:

        stats = self._last_opt_stats
        if not stats:
            return
        h, w = out.shape[:2]
        pad = max(8, int(10 * scale))
        line_gap = max(16, int(18 * scale))
        font_scale = max(0.32, 0.42 * scale)
        lines = [
            f"Preset: {self._performance_preset_var.get().replace('_', ' ').title()}",
            f"Pipeline: {stats.get('backend', self._pipeline_var.get())}",
            f"Latency avg: {float(stats.get('latency_avg_ms', 0.0) or 0.0):.1f} ms",
            f"Proc avg: {float(stats.get('processing_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI avg: {float(stats.get('roi_avg_ms', 0.0) or 0.0):.1f} ms",
            f"ROI mode: {str(stats.get('roi_mode', 'off')).title()}",
            f"Tracking: {self._tracking_state_var.get() or '---'}",
            (
                "Adaptive quality: "
                + (
                    f"ON (stable={int(stats.get('stable_tracking_streak', 0))}, "
                    f"skips={int(stats.get('quality_check_skips', 0))})"
                    if stats.get("adaptive_quality_active")
                    else "OFF"
                )
            ),
            (
                f"Dropped/Stale: {int(stats.get('dropped_frames', 0))}/"
                f"{int(stats.get('stale_frames', 0))}"
            ),
            (
                "Overload protection: "
                + (
                    f"ACTIVE (reuse={int(stats.get('cached_reuse_total', 0))})"
                    if stats.get("overload_active")
                    else f"Idle (reuse={int(stats.get('cached_reuse_total', 0))})"
                )
            ),
        ]
        box_width = max(220, int(w * 0.34))
        box_height = pad * 2 + line_gap * len(lines)
        x0 = max(0, w - box_width - pad)
        y0 = max(0, h - box_height - pad)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (18, 18, 18), -1)
        cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (70, 70, 70), 1)
        for idx, line in enumerate(lines):
            y = y0 + pad + line_gap * (idx + 1) - 4
            cv2.putText(
                out,
                line,
                (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_roi_overlay(self, out: np.ndarray, scale: float) -> None:
        roi = (
            self._roi_preview
            if self._roi_preview is not None
            else self._active_manual_roi()
        )
        if roi is None:
            return
        cx = int(round(roi["center_x"] * scale))
        cy = int(round(roi["center_y"] * scale))
        radius = max(1, int(round(roi["radius"] * scale)))
        is_editing = self._roi_preview is not None and self._roi_edit_active
        color = (0, 255, 255) if is_editing else (0, 220, 255)
        original = out.copy()
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
        shaded = out.copy()
        shaded[:] = (15, 15, 15)
        outside = cv2.bitwise_not(mask)
        out[:] = cv2.addWeighted(out, 0.45, shaded, 0.55, 0.0)
        inside_original = cv2.bitwise_and(original, original, mask=mask)
        outside_dimmed = cv2.bitwise_and(out, out, mask=outside)
        out[:] = cv2.add(inside_original, outside_dimmed)
        cv2.circle(out, (cx, cy), radius, color, 2, cv2.LINE_AA)
        if is_editing:
            cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)

        dia_px = roi["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"

        label = f"ROI {dia_str} (Enter=lock)" if is_editing else f"ROI {dia_str}"
        font_scale = max(0.4, 0.5 * scale)
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = max(10, cx - radius - text_w - 10)
        if is_editing:
            cv2.putText(
                out,
                label,
                (text_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )

            caption = "ROI Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_manual_ring_overlay(self, out: np.ndarray, scale: float) -> None:
        ring = (
            self._ring_preview
            if self._ring_preview is not None
            else self._active_manual_ring()
        )
        if ring is None:
            return
        cx = int(round(ring["center_x"] * scale))
        cy = int(round(ring["center_y"] * scale))
        radius = max(1, int(round(ring["radius"] * scale)))
        is_editing = self._ring_preview is not None and self._ring_edit_active
        if (
            not is_editing
            and self._current_result is not None
            and getattr(self._current_result, "ring_status", "unknown") == "ring_present"
        ):
            return
        color = (40, 110, 255) if is_editing else (0, 0, 255)
        thickness = 2 if is_editing else 3
        cv2.circle(out, (cx, cy), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, color, -1)
        handle_x = cx + radius
        handle_y = cy
        cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)

        dia_px = ring["radius"] * 2.0
        dia_str = f"Dia: {dia_px:.0f}px"
        if self._current_result is not None and getattr(self._current_result, "calibration", None) is not None and self._current_result.calibration.calibrated:
            dia_mm = dia_px * self._current_result.calibration.mm_per_px
            dia_str += f" ({dia_mm:.2f}mm)"

        label = f"Manual Ring {dia_str} (Enter=lock)" if is_editing else f"Manual Ring {dia_str}"
        cv2.putText(
            out,
            label,
            (cx + 10, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.35, 0.45 * scale),
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(out, (handle_x, handle_y), max(5, int(7 * scale)), (20, 20, 20), 1)

        if is_editing:
            caption = "Ring Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
            cv2.putText(
                out,
                caption,
                (max(10, cx - radius), max(20, cy - radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, 0.48 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_ruler_overlay(self, out: np.ndarray, scale: float) -> None:
        if not getattr(self, "_ruler_calibration_active", False) and not getattr(self, "_ruler_points", None):
            return
        pts = getattr(self, "_ruler_points", [])
        color = (0, 255, 255)
        for i, pt in enumerate(pts):
            cx = int(round(pt[0] * scale))
            cy = int(round(pt[1] * scale))
            cv2.circle(out, (cx, cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 9, color, 2, cv2.LINE_AA)
            cv2.putText(
                out,
                f"P{i+1}",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.5 * scale),
                color,
                1,
                cv2.LINE_AA,
            )

        if len(pts) >= 2:
            p1 = (int(round(pts[0][0] * scale)), int(round(pts[0][1] * scale)))
            p2 = (int(round(pts[1][0] * scale)), int(round(pts[1][1] * scale)))
            cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
            dist_px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            known_mm = float(self._ruler_known_dist_mm_var.get())
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            cv2.putText(
                out,
                f"{dist_px:.1f} px = {known_mm:.1f} mm ({dist_px/known_mm:.2f} px/mm)",
                (mid_x + 10, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.4, 0.52 * scale),
                color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod

    def _draw_filled_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Fill the circle/ellipse area with a semi-transparent colour overlay."""
        if alpha <= 0.0:
            return
        overlay = out.copy()
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        ratio = (ellipse_data.semi_minor / ellipse_data.semi_major
                 if ellipse_data.semi_major > 0 else 1.0)
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(overlay, ct, r, color, -1, cv2.LINE_AA)
        else:
            axes = (int(round(ellipse_data.semi_major)),
                    int(round(ellipse_data.semi_minor)))
            cv2.ellipse(overlay, ct, axes, int(round(ellipse_data.angle_deg)),
                        0, 360, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

    @staticmethod
    def _draw_structure(
        out: np.ndarray,
        ellipse_data: Any,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> Tuple[int, int]:
        ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
        if ellipse_data.semi_major > 0:
            ratio = ellipse_data.semi_minor / ellipse_data.semi_major
        else:
            ratio = 1.0
        if ratio > _CIRCLE_DRAW_THRESHOLD:
            r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
            cv2.circle(out, ct, r, color, thickness, cv2.LINE_AA)
        else:
            axes = (
                int(round(ellipse_data.semi_major)),
                int(round(ellipse_data.semi_minor)),
            )
            angle = int(round(ellipse_data.angle_deg))
            cv2.ellipse(out, ct, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
        return ct

    def _on_canvas_resize(self, _event: Any) -> None:
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(50, self._debounced_resize)

    def _debounced_resize(self) -> None:
        self._resize_after_id = None
        self._canvas_image_id = None
        self._refresh_display()

    # ================================================================
    # Measurements
    # ================================================================

    def _update_measurements(self, result: Any) -> None:
        try:
            cal = result.calibration
            has_cal = cal.calibrated if cal else False
            mm_per_px = cal.mm_per_px if has_cal else 0.0
            quality = (
                result.overall_quality.value
                if hasattr(result.overall_quality, "value")
                else str(result.overall_quality)
            )
            color = _QUALITY_COLORS.get(quality, "#888888")
            self._quality_label.config(
                text=f"  {quality} ({result.overall_confidence:.3f})  ",
                foreground=color,
            )
            self._summary_quality_var.set(quality)
            self._summary_quality_label.config(foreground=color)

            if result.pupil.detected and result.pupil.ellipse is not None:
                e = result.pupil.ellipse
                dia_px = e.radius * 2.0
                self._pv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._pv["diameter_px"].set(f"{dia_px:.1f} px")
                self._pv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._pv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._pv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._pv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._pv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.pupil, "fit_type", None
                )
                self._pv["fit_type"].set(ft if ft else "—")
                self._pv["confidence"].set(f"{result.pupil.confidence:.3f}")
                q_val = (
                    result.pupil.quality.value
                    if hasattr(result.pupil.quality, "value")
                    else str(result.pupil.quality)
                )
                self._pv["quality"].set(q_val)
            else:
                for var in self._pv.values():
                    var.set("---")

            if result.limbus.detected and result.limbus.ellipse is not None:
                e = result.limbus.ellipse
                dia_px = e.radius * 2.0
                self._lv["center"].set(f"({e.center_x:.1f}, {e.center_y:.1f}) px")
                self._lv["diameter_px"].set(f"{dia_px:.1f} px")
                self._lv["diameter_mm"].set(
                    f"{dia_px * mm_per_px:.2f} mm" if has_cal else "— (no calibration)"
                )
                self._lv["semi_major"].set(f"{e.semi_major:.1f} px")
                self._lv["semi_major_mm"].set(
                    f"{e.semi_major * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["semi_minor"].set(f"{e.semi_minor:.1f} px")
                self._lv["semi_minor_mm"].set(
                    f"{e.semi_minor * mm_per_px:.2f} mm"
                    if has_cal
                    else "— (no calibration)"
                )
                self._lv["angle"].set(f"{e.angle_deg:.1f}°")
                ft = getattr(e, "fit_type", None) or getattr(
                    result.limbus, "fit_type", None
                )
                self._lv["fit_type"].set(ft if ft else "—")
                self._lv["confidence"].set(f"{result.limbus.confidence:.3f}")
                q_val = (
                    result.limbus.quality.value
                    if hasattr(result.limbus.quality, "value")
                    else str(result.limbus.quality)
                )
                self._lv["quality"].set(q_val)
            else:
                for var in self._lv.values():
                    var.set("---")

            cc = result.corneal_center
            if cc.valid and result.has_both:
                pe = result.pupil.ellipse
                le = result.limbus.ellipse
                self._ov["corneal_center"].set(
                    f"({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f}) px"
                )
                self._ov["corneal_reference"].set(
                    getattr(result, "corneal_reference_source", "limbus")
                )
                ring_center = getattr(result, "ring_center", None)
                if ring_center is not None:
                    self._ov["ring_center"].set(
                        f"({ring_center[0]:.1f}, {ring_center[1]:.1f}) px"
                    )
                else:
                    self._ov["ring_center"].set("---")
                dx, dy = cc.offset_px
                offset_px = cc.offset_magnitude_px
                offset_angle = cc.offset_angle_deg
                self._ov["offset_px"].set(f"{offset_px:.1f} px")
                self._ov["offset_vec_px"].set(f"({dx:.1f}, {dy:.1f}) px")
                if has_cal:
                    dx_mm, dy_mm = dx * mm_per_px, dy * mm_per_px
                    offset_mm = offset_px * mm_per_px
                    self._ov["offset_mm"].set(f"{offset_mm:.3f} mm")
                    self._ov["offset_vec_mm"].set(f"({dx_mm:.3f}, {dy_mm:.3f}) mm")
                else:
                    self._ov["offset_mm"].set("— (no calibration)")
                    self._ov["offset_vec_mm"].set("— (no calibration)")
                self._ov["offset_angle"].set(f"{offset_angle:.1f}°")
                ring_radius = getattr(result, "ring_radius", None)
                if ring_radius is not None:
                    ring_dia_px = ring_radius * 2.0
                    self._ov["ring_diameter_px"].set(f"{ring_dia_px:.1f} px")
                    if has_cal:
                        self._ov["ring_diameter_mm"].set(
                            f"{ring_dia_px * mm_per_px:.3f} mm"
                        )
                    else:
                        self._ov["ring_diameter_mm"].set("— (no calibration)")
                else:
                    self._ov["ring_diameter_px"].set("---")
                    self._ov["ring_diameter_mm"].set("---")
                if le.radius > 0:
                    self._ov["pupil_limbus_ratio"].set(f"{pe.radius / le.radius:.3f}")
                else:
                    self._ov["pupil_limbus_ratio"].set("---")
            else:
                for var in self._ov.values():
                    var.set("---")

            if has_cal:
                method_tag = getattr(cal, "method", "anatomical")
                self._cv_vars["source"].set(f"{cal.source} [{method_tag}]")
                self._cv_vars["scale_px"].set(f"{cal.px_per_mm:.2f} px/mm")
                self._cv_vars["scale_mm"].set(f"{cal.mm_per_px:.4f} mm/px")
                if getattr(cal, "corneal_diameter_assumed_mm", None):
                    self._cv_vars["reference"].set(
                        f"Assumed {cal.corneal_diameter_assumed_mm:.1f}mm"
                    )
                elif cal.reference_diameter_mm > 0:
                    self._cv_vars["reference"].set(
                        f"{cal.reference_diameter_mm:.1f}mm ({cal.reference_diameter_px:.0f}px)"
                    )
                else:
                    self._cv_vars["reference"].set("Fixed External Scale")
            else:
                self._cv_vars["source"].set("not calibrated")
                self._cv_vars["scale_px"].set("---")
                self._cv_vars["scale_mm"].set("---")
                self._cv_vars["reference"].set("---")

            # ── Corneal Dimensions (WTW) Card ──
            limbus_res = getattr(result, "limbus", None)
            if (
                hasattr(self, "_wtw_vars")
                and limbus_res is not None
                and getattr(limbus_res, "detected", False)
                and getattr(limbus_res, "ellipse", None) is not None
                and has_cal
            ):
                le = limbus_res.ellipse

                # Always recompute WTW from current pixel geometry and
                # calibration.  Pre-computed attributes (set by
                # _add_mm_values during detection) become stale when the
                # calibration mode changes without a new detection.
                #
                # Horizontal WTW = 2 x Limbus Semi-Major (mm)
                # Vertical   WTW = 2 x Limbus Semi-Minor (mm)
                # i.e. the full limbus axes in mm — exactly twice the
                # Semi-Major (mm) / Semi-Minor (mm) rows shown in the Limbus
                # panel.  Reported as the true measured geometry (no clamping).
                h_wtw = 2.0 * le.semi_major * mm_per_px
                v_wtw = 2.0 * le.semi_minor * mm_per_px
                m_wtw = (h_wtw + v_wtw) / 2.0

                self._wtw_vars["horizontal"].set(f"{h_wtw:.2f} mm")
                self._wtw_vars["vertical"].set(f"{v_wtw:.2f} mm")
                self._wtw_vars["mean"].set(f"{m_wtw:.2f} mm")
            elif hasattr(self, "_wtw_vars"):
                for var in self._wtw_vars.values():
                    var.set("---")


            proc_ms = float(getattr(result.metadata, "processing_time_ms", 0.0) or 0.0)
            reused = bool(getattr(result.metadata, "reuse_cached_result", False))
            if not reused and proc_ms > 0.5:
                self._last_real_proc_time_ms = proc_ms
            shown_proc_ms = getattr(self, "_last_real_proc_time_ms", proc_ms)
            if not reused:
                shown_proc_ms = proc_ms
            display_proc_prev = getattr(self, "_display_proc_time_ms", shown_proc_ms)
            proc_alpha = 0.18 if reused else 0.32
            display_proc_ms = display_proc_prev + proc_alpha * (shown_proc_ms - display_proc_prev)
            self._display_proc_time_ms = display_proc_ms
            self._proc_time_var.set(f"{display_proc_ms:.1f} ms")
            latency_ms = float(getattr(
                result.metadata, "latency_ms", result.metadata.processing_time_ms
            ) or 0.0)
            display_latency_prev = getattr(self, "_display_latency_ms", latency_ms)
            latency_alpha = 0.20 if reused else 0.34
            display_latency_ms = (
                display_latency_prev + latency_alpha * (latency_ms - display_latency_prev)
            )
            self._display_latency_ms = display_latency_ms
            self._latency_var.set(f"{display_latency_ms:.1f} ms")
            if not self._using_optimized_camera:
                self._latency_avg_var.set("---")
                self._drop_var.set("---")
                self._tracking_state_var.set("---")
            self._frame_var.set(str(result.metadata.frame_number))
            self._image_size_var.set(
                f"{result.metadata.image_width} × {result.metadata.image_height}"
            )
            self._summary_latency_var.set(f"{display_latency_ms:.1f} ms")
            self._summary_pipeline_var.set(self._pipeline_var.get())
            tracking_text = self._tracking_state_var.get()
            if not tracking_text or tracking_text == "---":
                tracking_text = "Ready" if result.has_both else "Waiting"
            self._set_summary_tracking_state(tracking_text)

            mode = self._grayscale_mode_var.get()
            mode_labels = {
                "off": "RGB (Original)",
                "auto": "Auto-Detect",
                "force": "Forced Grayscale",
            }
            gs_label = mode_labels.get(mode, mode)
            if self._detector is not None:
                gs_info = self._detector.last_grayscale_info
                if gs_info is not None and gs_info.conversion_applied:
                    gs_label += " ✓ applied"
            self._gray_mode_var_display.set(gs_label)

            if hasattr(self, "_gray_settings_status"):
                self._gray_settings_status.set(f"Current: {gs_label}")

            self._update_details(result)
        except Exception as exc:
            self.logger.exception("Measurement update error: %s", exc)
            self._report_runtime_issue(
                "Measurement panel recovered from an internal error"
            )

    def _update_details(self, result: Any) -> None:
        cal = result.calibration
        has_cal = cal.calibrated if cal else False
        mm_per_px = cal.mm_per_px if has_cal else 0.0

        self._details_text.config(state=tk.NORMAL)
        self._details_text.delete("1.0", tk.END)

        lines: List[str] = []
        lines.append(f"Source:  {result.metadata.source}")
        lines.append(
            f"Image:   {result.metadata.image_width}×{result.metadata.image_height}"
        )
        q_val = (
            result.overall_quality.value
            if hasattr(result.overall_quality, "value")
            else str(result.overall_quality)
        )
        lines.append(f"Quality: {q_val} ({result.overall_confidence:.4f})")
        lines.append(f"Time:    {result.metadata.processing_time_ms:.1f} ms")
        latency_ms = getattr(result.metadata, "latency_ms", None)
        if latency_ms is not None:
            lines.append(f"Latency: {latency_ms:.1f} ms")

        # Grayscale info in details
        mode = self._grayscale_mode_var.get()
        mode_names = {"off": "OFF (RGB)", "auto": "AUTO", "force": "FORCE (Grayscale)"}
        lines.append(f"Gray:    {mode_names.get(mode, mode)}")
        if self._detector is not None:
            gs_info = self._detector.last_grayscale_info
            if gs_info is not None:
                lines.append(
                    f"         applied={gs_info.conversion_applied}, "
                    f"input={'gray' if gs_info.was_grayscale else 'RGB'}"
                )
                if gs_info.conversion_applied:
                    lines.append(
                        f"         contrast {gs_info.contrast_before:.1f} "
                        f"→ {gs_info.contrast_after:.1f}"
                    )
        lines.append("")

        ring_status = getattr(result, "ring_status", "unknown")
        ring_center = getattr(result, "ring_center", None)
        ring_radius = getattr(result, "ring_radius", None)
        if ring_status == "ring_present" and ring_center is not None and ring_radius is not None:
            lines.append("=== SUCTION RING ===")
            lines.append(f"  Status:     {ring_status}")
            lines.append(f"  Method:     {getattr(result, 'ring_method', 'unknown')}")
            lines.append(
                f"  Center:     ({ring_center[0]:.2f}, {ring_center[1]:.2f}) px"
            )
            lines.append(f"  Diameter:   {ring_radius * 2.0:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {ring_radius * 2.0 * mm_per_px:.3f} mm")
            lines.append(f"  Dots:       {getattr(result, 'ring_dot_count', 0)}")
            lines.append(
                f"  Reference:  {getattr(result, 'corneal_reference_source', 'limbus')}"
            )
            lines.append("")

        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.pupil.method.value
                if hasattr(result.pupil.method, "value")
                else str(result.pupil.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(result.pupil, "fit_type", None)
            lines.append("=== PUPIL ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append(
                f"  Uncert:     ±({e.uncertainty_center_x:.2f},"
                f" {e.uncertainty_center_y:.2f}) px"
            )
            lines.append("")

        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            dia_px = e.radius * 2.0
            m_val = (
                result.limbus.method.value
                if hasattr(result.limbus.method, "value")
                else str(result.limbus.method)
            )
            ft = getattr(e, "fit_type", None) or getattr(
                result.limbus, "fit_type", None
            )
            lines.append("=== LIMBUS ===")
            lines.append(f"  Method:     {m_val}")
            lines.append(f"  Fit Type:   {ft or '—'}")
            lines.append(f"  Center:     ({e.center_x:.2f}, {e.center_y:.2f}) px")
            lines.append(f"  Diameter:   {dia_px:.2f} px")
            if has_cal:
                lines.append(f"  Diameter:   {dia_px * mm_per_px:.3f} mm")
            lines.append(f"  Semi-axes:  {e.semi_major:.2f} × {e.semi_minor:.2f} px")
            if has_cal:
                lines.append(
                    f"  Semi-axes:  {e.semi_major * mm_per_px:.3f}"
                    f" × {e.semi_minor * mm_per_px:.3f} mm"
                )
            lines.append(f"  Angle:      {e.angle_deg:.2f}°")
            lines.append(f"  Eccentric:  {e.eccentricity:.4f}")
            lines.append(f"  Circular:   {e.circularity:.4f}")
            lines.append(f"  Fit qual:   {e.fit_quality:.4f}")
            lines.append(f"  RMS resid:  {e.fit_rms_residual:.4f}")
            lines.append(f"  Contour:    {e.num_contour_points} pts")
            lines.append("")

        if result.corneal_center.valid:
            cc = result.corneal_center
            lines.append("=== CORNEAL CENTRE & OFFSET ===")
            lines.append(
                f"  Centre:     ({cc.center_px[0]:.2f}, {cc.center_px[1]:.2f}) px"
            )
            lines.append(
                f"  Offset:     ({cc.offset_px[0]:.2f}, {cc.offset_px[1]:.2f}) px"
            )
            lines.append(f"  Magnitude:  {cc.offset_magnitude_px:.2f} px")
            if cc.offset_magnitude_mm is not None:
                lines.append(
                    f"  Offset:     ({cc.offset_mm[0]:.3f}, {cc.offset_mm[1]:.3f}) mm"
                )
                lines.append(f"  Magnitude:  {cc.offset_magnitude_mm:.3f} mm")
            lines.append(f"  Angle:      {cc.offset_angle_deg:.2f}°")
            lines.append("")

        if has_cal:
            lines.append("=== CALIBRATION ===")
            lines.append(f"  Source:     {cal.source}")
            lines.append(f"  px/mm:      {cal.px_per_mm:.4f}")
            lines.append(f"  mm/px:      {cal.mm_per_px:.6f}")
            lines.append(
                f"  Ref diam:   {cal.reference_diameter_mm:.1f}"
                f" mm = {cal.reference_diameter_px:.0f} px"
            )
            lines.append(f"  Confidence: {cal.confidence:.3f}")
            lines.append("")

        if result.alerts:
            lines.append("=== ALERTS ===")
            for alert in result.alerts:
                lines.append(f"  ! {alert}")
            lines.append("")

        self._details_text.insert("1.0", "\n".join(lines))
        self._details_text.config(state=tk.DISABLED)

    # ================================================================
    # Export
    # ================================================================

    def _export_csv(self, path: Optional[str] = None, show_success: bool = True) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        if path is None:
            path = filedialog.asksaveasfilename(
                title="Export CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
        if not path:
            return
        def _round(v: Any, nd: int = 4) -> Any:
            """Round numeric values; pass through blanks/None as ''."""
            if v is None or v == "":
                return ""
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    return ""
                return round(fv, nd)
            except (TypeError, ValueError):
                return v

        def _diam_mm(ell: Dict[str, Any], mm_px: float) -> Any:
            """Prefer the pre-computed diameter_mm; else derive from mean
            radius. NEVER recompute from semi-major alone (that collapses to
            the calibration constant). This matches the Measurements panel,
            which shows diameter = ellipse.radius * 2 * mm_per_px."""
            if ell.get("diameter_mm") not in (None, ""):
                return _round(ell.get("diameter_mm"))
            r = ell.get("radius")
            if r not in (None, "") and mm_px:
                return _round(float(r) * 2.0 * mm_px)
            return ""

        rows: List[Dict[str, Any]] = []
        for r in self._results_history:
            pupil = r.get("pupil", {})
            limbus = r.get("limbus", {})
            pe = pupil.get("ellipse", {})
            le = limbus.get("ellipse", {})
            cc = r.get("corneal_center", {})
            meta = r.get("metadata", {})
            cal_info = r.get("calibration", {})
            is_cal = bool(cal_info.get("calibrated", False))
            mm_px = float(cal_info.get("mm_per_px", 0) or 0) if is_cal else 0.0
            cal_method = cal_info.get("method", "") or ("anatomical" if is_cal else "")
            assumed_mm = cal_info.get("corneal_diameter_assumed_mm", "")
            if assumed_mm is None or not is_cal or cal_method != "anatomical":
                assumed_mm = ""

            # Diameter in px = mean radius * 2 (matches the panel exactly).
            pupil_dia_px = (pe.get("radius", 0) or 0) * 2 if pe.get("radius") else ""
            limbus_dia_px = (le.get("radius", 0) or 0) * 2 if le.get("radius") else ""

            # Semi-axes in mm: dynamically computed from semi_major/minor px * mm_per_px
            pupil_major_mm = _round(float(pe.get("semi_major", 0)) * mm_px) if (is_cal and pe.get("semi_major")) else ""
            pupil_minor_mm = _round(float(pe.get("semi_minor", 0)) * mm_px) if (is_cal and pe.get("semi_minor")) else ""
            limbus_major_mm = _round(float(le.get("semi_major", 0)) * mm_px) if (is_cal and le.get("semi_major")) else ""
            limbus_minor_mm = _round(float(le.get("semi_minor", 0)) * mm_px) if (is_cal and le.get("semi_minor")) else ""

            rows.append(
                {
                    "frame": meta.get("frame_number", ""),
                    "source": meta.get("source", ""),
                    "processing_time_ms": _round(meta.get("processing_time_ms", ""), 2),
                    "latency_ms": _round(meta.get("latency_ms", ""), 2),
                    # ── Pupil ─────────────────────────────────────────────
                    "pupil_detected": pupil.get("detected", False),
                    "pupil_cx_px": _round(pe.get("center_x", "")),
                    "pupil_cy_px": _round(pe.get("center_y", "")),
                    "pupil_diameter_px": _round(pupil_dia_px),
                    "pupil_diameter_mm": _diam_mm(pe, mm_px) if is_cal else "",
                    "pupil_radius_mm": _round(pupil.get("radius_mm", "")) if is_cal else "",
                    "pupil_semi_major_px": _round(pe.get("semi_major", "")),
                    "pupil_semi_minor_px": _round(pe.get("semi_minor", "")),
                    "pupil_semi_major_mm": pupil_major_mm,
                    "pupil_semi_minor_mm": pupil_minor_mm,
                    "pupil_angle_deg": _round(pe.get("angle_deg", ""), 2),
                    "pupil_fit_type": pupil.get("fit_type", "") or "",
                    "pupil_confidence": _round(pupil.get("confidence", ""), 3),
                    # ── Limbus ────────────────────────────────────────────
                    "limbus_detected": limbus.get("detected", False),
                    "limbus_cx_px": _round(le.get("center_x", "")),
                    "limbus_cy_px": _round(le.get("center_y", "")),
                    "limbus_diameter_px": _round(limbus_dia_px),
                    "limbus_diameter_mm": _diam_mm(le, mm_px) if is_cal else "",
                    "limbus_radius_mm": _round(limbus.get("radius_mm", "")) if is_cal else "",
                    "limbus_semi_major_px": _round(le.get("semi_major", "")),
                    "limbus_semi_minor_px": _round(le.get("semi_minor", "")),
                    "limbus_semi_major_mm": limbus_major_mm,
                    "limbus_semi_minor_mm": limbus_minor_mm,
                    "limbus_angle_deg": _round(le.get("angle_deg", ""), 2),
                    "limbus_fit_type": limbus.get("fit_type", "") or "",
                    "limbus_confidence": _round(limbus.get("confidence", ""), 3),
                    # ── Clinical Corneal WTW Dimensions ───────────────────
                    "measured_wtw_horizontal_mm": _round(limbus.get("wtw_horizontal_mm", "")) if is_cal else "",
                    "measured_wtw_vertical_mm": _round(limbus.get("wtw_vertical_mm", "")) if is_cal else "",
                    "measured_wtw_mean_mm": _round(limbus.get("wtw_mean_mm", "")) if is_cal else "",
                    "limbus_astigmatic_difference_mm": _round(limbus.get("wtw_astigmatism_mm", "")) if is_cal else "",
                    "wtw_validity_status": limbus.get("wtw_validity_status", "") if is_cal else "",
                    # ── Corneal center / offset ───────────────────────────
                    "corneal_center_x_px": _round(
                        (cc.get("center_px") or ["", ""])[0]
                    ),

                    "corneal_center_y_px": _round(
                        (cc.get("center_px") or ["", ""])[1]
                    ),
                    "offset_px": _round(cc.get("offset_magnitude_px", "")),
                    "offset_mm": _round(cc.get("offset_magnitude_mm", "")) if is_cal else "",
                    "offset_angle_deg": _round(cc.get("offset_angle_deg", ""), 2),
                    "corneal_reference_source": r.get("corneal_reference_source", ""),
                    # ── Ring ──────────────────────────────────────────────
                    "ring_status": r.get("ring_status", ""),
                    "ring_center_x": _round(r.get("ring_center_x", "")),
                    "ring_center_y": _round(r.get("ring_center_y", "")),
                    "ring_radius_px": _round(r.get("ring_radius", "")),
                    "ring_diameter_mm": (
                        _round((r.get("ring_radius") or 0) * 2.0 * mm_px)
                        if is_cal and r.get("ring_radius") and mm_px else ""
                    ),
                    "ring_dot_count": r.get("ring_dot_count", ""),
                    # ── Calibration / quality ─────────────────────────────
                    "calibrated": is_cal,
                    "calibration_method": cal_method if is_cal else "",
                    "corneal_diameter_assumed_mm": _round(assumed_mm, 2) if assumed_mm != "" else "",
                    "px_per_mm": _round(cal_info.get("px_per_mm", "")) if is_cal else "",
                    "mm_per_px": _round(cal_info.get("mm_per_px", ""), 6) if is_cal else "",
                    "quality": r.get("overall_quality", ""),
                    "overall_confidence": _round(r.get("overall_confidence", ""), 3),
                    "grayscale_mode": r.get("grayscale_mode", ""),
                }
            )

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if show_success:
            self._status_var.set(f"Exported {len(rows)} rows → {path}")

    def _export_txt(self) -> None:
        if not self._results_history:
            messagebox.showinfo("No Data", "No results to export")
            return
        path = filedialog.asksaveasfilename(
            title="Export Results TXT",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not path:
            return
        lines = [
            "# Pupil-Limbus Detection Results",
            "",
            "[export_info]",
            "version=2.3",
            f"total_frames={len(self._results_history)}",
            f"export_timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"corneal_diameter_assumption_mm={_CORNEAL_DIAMETER_MM}",
            "",
            "[results]",
        ]
        for index, result in enumerate(self._results_history, start=1):
            lines.append(f"frame_{index}={result!s}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._status_var.set(f"Exported {len(self._results_history)} results → {path}")

    def _export_snapshot(self) -> None:
        if self._current_image is None:
            messagebox.showinfo("No Image", "No image to export")
            return
        raw = self._current_image
        clean = self._prepare_recording_frame(
            raw,
            self._current_result,
            draw_overlay=False,
        )
        path = filedialog.asksaveasfilename(
            title="Save Snapshot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
        )
        if not path:
            return

        # Limit the saved image to the area inside the detected limbus (the
        # "blue ring") as a circular crop, taken from the clean image (no
        # overlay annotations).  Falls back to the full on-screen frame
        # when no limbus is available.
        image, cropped = self._circular_crop_to_limbus(
            clean,
            self._current_result,
            transparent=path.lower().endswith(".png"),
        )
        if not cropped:
            image = self._prepare_recording_frame(
                raw,
                self._current_result,
            )
        if not cv2.imwrite(path, image):
            self._status_var.set(f"Snapshot failed → {path}")
            return
        if cropped:
            self._status_var.set(f"Snapshot (limbus circle) saved → {path}")
        else:
            self._status_var.set(
                f"Snapshot saved → {path} (no limbus detected — full frame)"
            )

    def _circular_crop_to_limbus(
        self,
        image: np.ndarray,
        result: Any,
        transparent: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """Crop *image* to the detected limbus as a circular region.
        Returns a square crop centred on the limbus ellipse with everything
        outside the ellipse removed.  When *transparent* is set the outside
        is transparent (PNG alpha); otherwise it is filled black.  Returns
        ``(image, False)`` unchanged when no valid limbus ellipse exists,
        so callers can fall back to the full frame.
        """
        limbus = getattr(result, "limbus", None) if result is not None else None
        ellipse = getattr(limbus, "ellipse", None) if limbus is not None else None
        if limbus is None or not getattr(limbus, "detected", False) or ellipse is None:
            return image, False

        # ``ellipse`` may be a core ``EllipseParams`` (which exposes an
        # ``is_valid`` property) or a GUI-adapted ``SimpleNamespace`` (which
        # does not), so validate the raw geometry numerically instead of
        # trusting ``is_valid`` — otherwise the crop silently falls back to
        # the full frame for adapted/video results.
        try:
            cx = float(ellipse.center_x)
            cy = float(ellipse.center_y)
            rx = float(ellipse.semi_major)
            ry = float(ellipse.semi_minor)
        except (AttributeError, TypeError, ValueError):
            return image, False
        if not all(math.isfinite(v) for v in (cx, cy, rx, ry)):
            return image, False
        if rx <= 0.0 or ry <= 0.0:
            return image, False
        if rx < ry:
            rx, ry = ry, rx
        h, w = image.shape[:2]
        rad = int(math.ceil(max(rx, ry))) + 1
        x0 = max(0, int(math.floor(cx)) - rad)
        y0 = max(0, int(math.floor(cy)) - rad)
        x1 = min(w, int(math.ceil(cx)) + rad)
        y1 = min(h, int(math.ceil(cy)) + rad)
        crop = image[y0:y1, x0:x1].copy()
        if crop.size == 0:
            return image, False
        ch, cw = crop.shape[:2]
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(
            mask,
            (int(round(cx)) - x0, int(round(cy)) - y0),
            (max(1, int(round(rx))), max(1, int(round(ry)))),
            float(getattr(ellipse, "angle_deg", 0.0) or 0.0),
            0,
            360,
            255,
            -1,
            cv2.LINE_AA,
        )
        if transparent:
            if crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            crop[:, :, 3] = cv2.bitwise_and(crop[:, :, 3], mask)
            return crop, True
        if crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)
        crop[mask == 0] = (0, 0, 0)
        return crop, True

    # ================================================================
    # Lifecycle
    # ================================================================

    def _on_close(self) -> None:
        self._stop_video()
        # ══════════════════════════════════════════════════════════
        # RECORDING — Cleanup recording on close
        # ══════════════════════════════════════════════════════════
        if self._recorder.is_recording:
            self._stop_recording()
        # ══════════════════════════════════════════════════════════
        if self._fast_engine is not None:
            try:
                import torch

                if (
                    hasattr(self._fast_engine, "model")
                    and self._fast_engine.model is not None
                ):
                    del self._fast_engine.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self._fast_engine = None
        if _FAST_PIPELINE_AVAILABLE and FastInference is not None:
            try:
                FastInference.reset_cache()
            except Exception:
                pass
        try:
            self.logger.close()
        except Exception:
            pass
        self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        import os
        import sys
        # gui_app.py is in pupil_tracking/interface/, so base is 3 levels up
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        icon_path = os.path.join(base_dir, "build", "assets", "icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    colors = DarkTheme.apply(root)
    _app = PupilTrackingGUI(root, colors=colors)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
