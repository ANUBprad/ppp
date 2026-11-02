#!/usr/bin/env python3
"""
╔======================================================================╗
║           Medevplus IXcentai surgical grade                        ║
╚======================================================================╝

USAGE EXAMPLES:
-----------------------------------------------------------------------
  python launch_gui.py                                # Launch GUI
  python launch_gui.py gui                            # Launch GUI
  python launch_gui.py gui --stride 2 --no-fp16       # GUI + overrides
  python launch_gui.py image -i eye.jpg               # Single image
  python launch_gui.py video -i vid.mp4               # Video file
  python launch_gui.py video -i vid.mp4 --stride 3    # Skip frames
  python launch_gui.py video -i vid.mp4 -o out.mp4    # Save output
  python launch_gui.py camera                         # Live webcam
  python launch_gui.py annotate                        # Open Pentacam annotation workspace
  python launch_gui.py annotate --dataset-dir "path\to\pentacam"

RING DETECTION & ADAPTIVE PIPELINE (NEW):
-----------------------------------------------------------------------
  --ring-mode MODE          auto / docked / pre_docked    [default: auto]
  --ring-classifier PATH    Path to ring classifier model
  --show-ring / --no-show-ring  Draw ring in output       [default: ON]

GRAYSCALE MODE (NEW):
-----------------------------------------------------------------------
  --grayscale MODE          off / auto / force            [default: off]
                            off   = RGB passthrough (original behaviour)
                            auto  = detect & enhance grayscale inputs
                            force = always convert to enhanced grayscale

  When grayscale is FORCE or AUTO (on grayscale input), the displayed
  image is converted to grayscale — like an IR camera feed — with
  coloured detection overlays drawn on top.

PIPELINE FLAGS (gui / video / camera):
-----------------------------------------------------------------------
  --stride N                Process every Nth frame       [default: 1]
  --resolution N            Inference resolution (px)     [default: profile]
  --target-fps F            Target processing FPS         [default: profile]
  --fp16 / --no-fp16        FP16 half-precision           [default: profile]
  --compile / --no-compile  torch.compile JIT             [default: profile]
  --roi / --no-roi          ROI tracking                  [default: ON]
  --optimized / --no-optimized  Use fast pipeline         [default: ON]
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import AuditLogger, set_logger
from pupil_tracking.utils.runtime_profile import (
    apply_runtime_optimizations,
    detect_runtime_profile,
)


_RUNTIME_PROFILE = apply_runtime_optimizations(detect_runtime_profile())


# -- Constants ---------------------------------------------------
_CORNEAL_DIAMETER_MM = 12.0

_BANNER = "Medevplus IXcentai - Surgical Grade"


# ================================================================
# Optional fast-pipeline imports (graceful fallback)
# ================================================================

try:
    from pupil_tracking.ml.fast_inference import FastInference
    from pupil_tracking.video.optimized_processor import (
        OptimizedVideoProcessor,
        AsyncCapture,
        FrameResult,
        TrackingQuality,
    )

    _FAST_PIPELINE_AVAILABLE = True
    FastInferenceEngine = FastInference
except ImportError:
    _FAST_PIPELINE_AVAILABLE = False
    FastInferenceEngine = None
    OptimizedVideoProcessor = None
    AsyncCapture = None
    FrameResult = None
    TrackingQuality = None


# ==================================================================
# DISPLAY PATCH 1 of 4 — Grayscale display conversion helper
#
# When grayscale mode is active, this converts the frame to a
# 3-channel grayscale image for DISPLAY purposes.  The detection
# pipeline receives the original frame — it handles its own
# grayscale conversion internally.  This function only affects
# what the user SEES on screen and in output video.
#
# Result: grayscale background with coloured detection circles
# drawn on top — like looking through an IR camera with tracking.
# ==================================================================


def _convert_frame_for_display(
    frame: np.ndarray,
    detector: Any,
) -> np.ndarray:
    """Convert a frame to grayscale for display when mode is active.

    When the detector's grayscale mode is FORCE, this returns a
    3-channel grayscale image (gray replicated to BGR) so that
    coloured overlays (green pupil, blue limbus) are visible on
    a gray background.

    When mode is AUTO, the frame is only converted if the detector
    actually applied grayscale processing (i.e. the input was
    detected as grayscale).

    When mode is OFF, returns the original frame unchanged.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame from camera/video/image.
    detector : UnifiedDetector
        Detector instance — checked for current grayscale mode
        and last processing info.

    Returns
    -------
    np.ndarray
        Frame for display — always 3-channel BGR uint8.
        Either the original RGB or a grayscale-replicated version.
    """
    from pupil_tracking.preprocessing.grayscale_handler import GrayscaleMode

    mode = detector.grayscale_mode

    if mode == GrayscaleMode.OFF:
        return frame

    if mode == GrayscaleMode.FORCE:
        # Always show grayscale display
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == GrayscaleMode.AUTO:
        # Show grayscale only if the detector actually converted
        gs_info = detector.last_grayscale_info
        if gs_info is not None and gs_info.conversion_applied:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return frame


# ==================================================================
# END DISPLAY PATCH 1
# ==================================================================


# ================================================================
# CLI — Image Processing
# ================================================================


def process_image(
    image_path: str,
    model_path: Optional[str] = None,
    device: str = "auto",
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process a single image and display results."""
    from pupil_tracking.core.detector import UnifiedDetector

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n  ✗ ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    cfg = get_config()

    # Ring configuration overrides
    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    if ring_classifier:
        cfg.ring.classifier.classifier_path = ring_classifier

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )

    result = detector.detect(image, source=image_path, force_mode=force_mode)

    # -- calibration from limbus / ring --------------------------
    cal = result.calibration
    has_cal = cal.calibrated if cal else False
    mm_per_px = cal.mm_per_px if has_cal else 0.0

    # -- corneal centre reference (use blended when available) -------
    cc_ref = getattr(result, 'corneal_reference_source', 'pupil')
    if hasattr(result, 'corneal_center'):
        cc = result.corneal_center
        if cc.valid and hasattr(cc, 'center_px'):
            cc_ref = f"blended ({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f})"

    print(f"\n{'=' * 64}")
    print(f"  IMAGE ANALYSIS RESULTS")
    print(f"{'=' * 64}")
    print(f"  File:       {image_path}")
    print(
        f"  Size:       {result.metadata.image_width}"
        f" × {result.metadata.image_height} px"
    )
    print(f"  Category:   {getattr(result, 'image_category', 'unknown').upper()}")
    print(f"  Corneal Ref:{cc_ref}")

    ring_status = getattr(result, "ring_status", "unknown")
    ring_conf = getattr(result, "ring_confidence", 0.0)
    print(f"  Ring:       {ring_status} (conf: {ring_conf:.3f})")

    print(
        f"  Quality:    {result.overall_quality.value} "
        f"({result.overall_confidence:.3f})"
    )

    gs_info = detector.last_grayscale_info
    gs_mode_display = detector.grayscale_mode.name
    if gs_info is not None:
        gs_applied = "YES" if gs_info.conversion_applied else "NO"
        gs_input = "grayscale" if gs_info.was_grayscale else "RGB"
        print(
            f"  Grayscale:  mode={gs_mode_display}, applied={gs_applied}, "
            f"input={gs_input}"
        )
        if gs_info.conversion_applied:
            print(
                f"              contrast {gs_info.contrast_before:.1f} "
                f"→ {gs_info.contrast_after:.1f}"
            )
    else:
        print(f"  Grayscale:  mode={gs_mode_display} (not applied)")

    print(f"{'-' * 64}")

    # Ring Geometry
    if ring_status == "ring_present":
        rc = getattr(result, "ring_center", None)
        rr = getattr(result, "ring_radius", None)
        if rc and rr:
            print(f"\n  * SUCTION RING")
            print(f"    Center:       ({rc[0]:.1f}, {rc[1]:.1f}) px")
            print(f"    Radius:       {rr:.1f} px")
            print(f"    Diameter:     {rr * 2.0:.1f} px")
            if has_cal:
                print(f"    Diameter:     {rr * 2.0 * mm_per_px:.3f} mm")
            print(
                f"    Dots:         {getattr(result, 'ring_dot_count', 0)}"
            )

    # Pupil Geometry
    if result.pupil.detected:
        p = result.pupil.ellipse
        dia_px = p.radius * 2.0
        print(f"\n  * PUPIL")
        print(f"    Center:       ({p.center_x:.1f}, {p.center_y:.1f}) px")
        print(f"    Diameter:     {dia_px:.1f} px")
        if has_cal:
            print(f"    Diameter:     {dia_px * mm_per_px:.3f} mm")
        print(f"    Semi-axes:    {p.semi_major:.1f} × {p.semi_minor:.1f} px")
        print(f"    Angle:        {p.angle_deg:.1f}°")
        ft = getattr(p, "fit_type", None) or getattr(result.pupil, "fit_type", None)
        if ft:
            print(f"    Fit type:     {ft}")
        print(f"    Confidence:   {result.pupil.confidence:.3f}")
        print(f"    Quality:      {result.pupil.quality.value}")
    else:
        print(f"\n  o PUPIL:  NOT DETECTED")

    # Limbus Geometry
    if result.limbus.detected:
        el = result.limbus.ellipse
        dia_px = el.radius * 2.0
        print(f"\n  * LIMBUS")
        print(f"    Center:       ({el.center_x:.1f}, {el.center_y:.1f}) px")
        print(f"    Diameter:     {dia_px:.1f} px")
        if has_cal:
            print(f"    Diameter:     {dia_px * mm_per_px:.3f} mm")
        print(f"    Semi-axes:    {el.semi_major:.1f} × {el.semi_minor:.1f} px")
        print(f"    Angle:        {el.angle_deg:.1f}°")
        ft = getattr(el, "fit_type", None) or getattr(result.limbus, "fit_type", None)
        if ft:
            print(f"    Fit type:     {ft}")
        print(f"    Confidence:   {result.limbus.confidence:.3f}")
        print(f"    Quality:      {result.limbus.quality.value}")
    else:
        print(f"\n  o LIMBUS:  NOT DETECTED")

    if result.corneal_center.valid:
        cc = result.corneal_center
        print(f"\n  * CORNEAL CENTRE & OFFSET")
        print(f"    Centre:       ({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f}) px")
        print(
            f"    Reference:    "
            f"{getattr(result, 'corneal_reference_source', 'limbus')}"
        )
        print(f"    Offset:       {cc.offset_magnitude_px:.1f} px")
        if cc.offset_magnitude_mm is not None:
            print(f"    Offset:       {cc.offset_magnitude_mm:.3f} mm")
        print(f"    Angle:        {cc.offset_angle_deg:.1f}°")
        for alert in getattr(cc, "alerts", []):
            print(f"    ! {alert}")

    if has_cal:
        print(f"\n  * CALIBRATION")
        print(f"    Source:       {cal.source}")
        print(f"    Scale:        {cal.px_per_mm:.2f} px/mm")
        print(f"    Scale:        {cal.mm_per_px:.4f} mm/px")

    print(f"\n{'-' * 64}")
    print(f"  Processing time: {result.metadata.processing_time_ms:.1f} ms")
    print(f"{'=' * 64}")

    # ==============================================================
    # DISPLAY PATCH 2 of 4 — Convert image for display
    #
    # The detector already processed the original image internally.
    # Now convert the DISPLAY image to grayscale if mode is active.
    # Detection overlays (green/blue circles) will be drawn on the
    # grayscale background, giving the IR camera look.
    # ==============================================================
    display_frame = _convert_frame_for_display(image, detector)
    annotated = _draw_cli_overlay(
        display_frame,
        result,
        show_ring=show_ring,
        grayscale_mode=gs_mode_display,
    )
    # ==============================================================
    # END DISPLAY PATCH 2
    # ==============================================================
    cv2.imshow("Detection Result", annotated)
    print("\n  Press any key to close the window…\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================================================================
# CLI — Video File Processing
# ================================================================


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process a video file with optional optimised pipeline."""
    from pupil_tracking.core.detector import UnifiedDetector
    from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
    from pupil_tracking.core.corneal_center import CornealCenterCalculator

    cfg = get_config()

    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )
    tracker = EyeKalmanTracker(config=cfg)
    corneal_calc = CornealCenterCalculator(config=cfg)

    # -- read CLI flags (with defaults) --------------------------
    stride = getattr(args, "stride", 1) or 1
    resolution = getattr(args, "resolution", 320) or 320
    use_fp16 = getattr(args, "fp16", True)
    use_compile = getattr(args, "compile", True)
    use_roi = getattr(args, "roi", True)
    target_fps = getattr(args, "target_fps", 20.0) or 20.0
    use_optimized = getattr(args, "optimized", True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n  ✗ ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'=' * 64}")
    print(f"  VIDEO PROCESSING")
    print(f"{'=' * 64}")
    print(f"  File:         {video_path}")
    print(f"  Resolution:   {width} × {height} @ {fps:.1f} FPS")
    print(f"  Total frames: {total_frames}")
    print(f"{'-' * 64}")
    print(f"  Ring Mode:    {ring_mode.upper()}")
    print(f"  Grayscale:    {grayscale_mode.upper()}")
    print(f"  Stride:       {stride}  (every {stride} frame(s))")
    print(f"  Inference:    {resolution} px")
    print(f"  FP16:         {'ON' if use_fp16 else 'OFF'}")
    print(f"  Compile:      {'ON' if use_compile else 'OFF'}")
    print(f"  ROI:          {'ON' if use_roi else 'OFF'}")
    print(f"  Target FPS:   {target_fps:.1f}")

    # -- output video writer -------------------------------------
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / stride
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        print(f"  Output:       {output_path}")

    # -- try optimised pipeline ----------------------------------
    fast_engine = None
    opt_processor = None

    if use_optimized and _FAST_PIPELINE_AVAILABLE:
        model_file = _find_model_path(detector, cfg)
        if model_file:
            try:
                opt_processor = OptimizedVideoProcessor(
                    model_path=model_file,
                    device="auto",
                    input_size=resolution,
                    half_precision=use_fp16,
                    use_compile=use_compile,
                    enable_auto_roi=use_roi,
                    roi_cache_ttl=getattr(args, "roi_cache", 5),
                    process_noise=getattr(args, "kalman_process_noise", 0.03),
                    measurement_noise=getattr(args, "kalman_measure_noise", 0.1),
                    batch_size=_RUNTIME_PROFILE.recommended_batch_size,
                )
                print(
                    f"  Pipeline:     ✓ OPTIMISED "
                    f"(FP16={use_fp16}, ROI={use_roi}, "
                    f"batch={opt_processor.get_stats().get('batch_size')})"
                )
            except Exception as exc:
                print(f"  Pipeline:     ! Optimised failed: {exc}")
                print(f"                  Falling back to classic")
                opt_processor = None

    if opt_processor is None:
        print(f"  Pipeline:     Classic (UnifiedDetector)")

    print(f"{'=' * 64}\n")

    # -- processing loop -----------------------------------------
    results_history: List[Dict[str, Any]] = []
    raw_frame_idx = 0
    processed_count = 0
    start_time = time.monotonic()

    gs_mode_display = grayscale_mode.upper()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame_idx += 1

            # stride
            if stride > 1 and (raw_frame_idx % stride) != 0:
                continue

            processed_count += 1

            # -- detect ------------------------------------------
            if opt_processor is not None:
                try:
                    frame_result = opt_processor.process_frame(frame, processed_count)
                    result_dict = _frame_result_to_dict(frame_result)
                except Exception:
                    result = detector.detect(
                        frame,
                        frame_number=processed_count,
                        source=video_path,
                        force_mode=force_mode,
                    )
                    result_dict = result.to_dict()
            else:
                result = detector.detect(
                    frame,
                    frame_number=processed_count,
                    source=video_path,
                    force_mode=force_mode,
                )
                smoothed = tracker.update(result)
                if smoothed.has_both:
                    smoothed.corneal_center = corneal_calc.calculate(
                        smoothed.pupil,
                        smoothed.limbus,
                        result.calibration,
                    )
                smoothed.calibration = result.calibration
                if (
                    getattr(result, "ring_status", "unknown") == "ring_present"
                    and getattr(result, "ring_center", None) is not None
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
                    points.append((result.ring_center[0], result.ring_center[1], "ring"))
                    weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                    total_w = sum(weights)
                    rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                    rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                    smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                    smoothed.corneal_center.center_px = (rcx, rcy)
                    smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                    smoothed.corneal_center.offset_magnitude_px = math.hypot(px - rcx, py - rcy)
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
                    smoothed.corneal_reference_source = getattr(
                        result,
                        "corneal_reference_source",
                        "suction_ring",
                    )
                result_dict = smoothed.to_dict()

            results_history.append(result_dict)

            # ======================================================
            # DISPLAY PATCH 3 of 4 — Convert video frame for display
            #
            # detector.detect(frame) already received the ORIGINAL
            # frame — the detector handles grayscale internally.
            # Now convert the display copy so the output video and
            # any preview shows grayscale with coloured overlays.
            # ======================================================
            if writer is not None:
                display_frame = _convert_frame_for_display(frame, detector)
                if opt_processor is not None:
                    annotated = _draw_cli_overlay_from_dict(
                        display_frame,
                        result_dict,
                        show_ring=show_ring,
                        grayscale_mode=gs_mode_display,
                    )
                else:
                    annotated = _draw_cli_overlay(
                        display_frame,
                        smoothed,
                        show_ring=show_ring,
                        grayscale_mode=gs_mode_display,
                    )
                writer.write(annotated)
            # ======================================================
            # END DISPLAY PATCH 3
            # ======================================================

            # -- progress bar ------------------------------------
            if total_frames > 0:
                pct = raw_frame_idx / total_frames * 100
                elapsed = time.monotonic() - start_time
                rate = raw_frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - raw_frame_idx) / rate if rate > 0 else 0

                bar_len = 35
                filled = int(bar_len * raw_frame_idx / total_frames)
                bar = "█" * filled + "░" * (bar_len - filled)

                sys.stdout.write(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"fr {raw_frame_idx}/{total_frames}  "
                    f"proc {processed_count}  "
                    f"ETA {eta:.0f}s  "
                    f"({rate:.1f} fr/s)    "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  ! Interrupted by user")

    # -- cleanup -------------------------------------------------
    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.monotonic() - start_time
    avg_fps = processed_count / elapsed if elapsed > 0 else 0

    print(f"\n\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"{'-' * 64}")
    print(f"  Processed:    {processed_count} frames")
    print(f"  Elapsed:      {elapsed:.1f} s")
    print(f"  Average FPS:  {avg_fps:.1f}")

    # -- auto-export CSV alongside output video ------------------
    if output_path:
        csv_path = str(Path(output_path).with_suffix(".csv"))
        _export_results_csv(results_history, csv_path)
        print(f"  Results CSV:  {csv_path}")
        print(f"  Output video: {output_path}")

    print(f"{'=' * 64}\n")


# ================================================================
# CLI — Camera Processing
# ================================================================


def process_camera(
    camera_id: int = 0,
    model_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process live camera feed."""
    from pupil_tracking.core.detector import UnifiedDetector
    from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
    from pupil_tracking.core.corneal_center import CornealCenterCalculator

    cfg = get_config()
    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )
    tracker = EyeKalmanTracker(config=cfg)
    corneal_calc = CornealCenterCalculator(config=cfg)

    # -- CLI flags -----------------------------------------------
    resolution = getattr(args, "resolution", 320) or 320
    use_fp16 = getattr(args, "fp16", True)
    use_compile = getattr(args, "compile", True)
    use_roi = getattr(args, "roi", True)
    target_fps = getattr(args, "target_fps", 20.0) or 20.0
    stride = getattr(args, "stride", 1) or 1
    use_optimized = getattr(args, "optimized", True)

    print(f"\n{'=' * 64}")
    print(f"{'=' * 64}")
    print(f"  IMAGE ANALYSIS RESULTS")
    print(f"{'=' * 64}")
    print(f"  Device:       camera {camera_id}")
    print(f"  Resolution:   {resolution} px inference")
    print(f"  Target FPS:   {target_fps}")
    print(f"  Stride:       {stride}")
    print(f"  Ring Mode:    {ring_mode.upper()}")
    print(f"  Grayscale:    {grayscale_mode.upper()}")
    print(f"  FP16:         {'ON' if use_fp16 else 'OFF'}")
    print(f"  Compile:      {'ON' if use_compile else 'OFF'}")
    print(f"  ROI:          {'ON' if use_roi else 'OFF'}")
    print(f"{'-' * 64}")
    print(f"  CONTROLS:")
    print(f"    [Q]       Quit")
    print(f"    [S]       Save snapshot")
    print(f"    [SPACE]   Pause / Resume")
    print(f"    [G]       Toggle grayscale mode (OFF → AUTO → FORCE)")
    print(f"{'=' * 64}\n")

    # -- try optimised pipeline ----------------------------------
    opt_processor = None
    async_capture = None

    if use_optimized and _FAST_PIPELINE_AVAILABLE:
        model_file = _find_model_path(detector, cfg)
        if model_file:
            try:
                opt_processor = OptimizedVideoProcessor(
                    model_path=model_file,
                    device="auto",
                    input_size=resolution,
                    half_precision=use_fp16,
                    use_compile=use_compile,
                    enable_auto_roi=use_roi,
                    roi_cache_ttl=getattr(args, "roi_cache", 5),
                    process_noise=getattr(args, "kalman_process_noise", 0.03),
                    measurement_noise=getattr(args, "kalman_measure_noise", 0.1),
                    batch_size=_RUNTIME_PROFILE.recommended_batch_size,
                )
                async_capture = AsyncCapture(
                    camera_id,
                    buffer_size=_RUNTIME_PROFILE.recommended_capture_buffer,
                )
                async_capture.start()
                print(
                    "  Pipeline: ✓ OPTIMISED "
                    f"(batch={opt_processor.get_stats().get('batch_size')})\n"
                )
            except Exception as exc:
                print(f"  Pipeline: ! Optimised failed: {exc}")
                print(f"            Falling back to classic\n")
                opt_processor = None
                if async_capture is not None:
                    try:
                        async_capture.stop()
                    except Exception:
                        pass
                    async_capture = None

    # -- classic fallback ----------------------------------------
    cap = None
    if opt_processor is None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"  ✗ ERROR: Cannot open camera {camera_id}")
            sys.exit(1)
        print("  Pipeline: Classic (UnifiedDetector)\n")

    # -- processing loop -----------------------------------------
    frame_count = 0
    paused = False
    snapshot_count = 0
    fps_counter = 0
    fps_timer = time.monotonic()
    current_fps = 0.0

    _GRAYSCALE_CYCLE = ["off", "auto", "force"]

    try:
        while True:
            # -- pause -------------------------------------------
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    paused = False
                    print("  ▶ Resumed")
                elif key == ord("q"):
                    break
                continue

            # -- read frame --------------------------------------
            frame = None
            if async_capture is not None:
                data = async_capture.read(timeout=0.1)
                if data is not None:
                    _, frame, _ = data
            elif cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break

            if frame is None:
                continue

            frame_count += 1

            # -- stride ------------------------------------------
            if stride > 1 and (frame_count % stride) != 0:
                continue

            # ======================================================
            # DISPLAY PATCH 4 of 4 — Convert camera frame for display
            #
            # KEY SEPARATION:
            #   - detector.detect(frame) gets the ORIGINAL frame
            #     (detector handles grayscale internally for model)
            #   - display_frame is the GRAYSCALE version shown to user
            #     (overlays are drawn on this — IR camera look)
            #
            # This conversion happens BEFORE detection so the display
            # frame is ready regardless of which detection path runs.
            # ======================================================
            display_frame = _convert_frame_for_display(frame, detector)
            # ======================================================
            # END DISPLAY PATCH 4 (frame conversion)
            # ======================================================

            # -- detect ------------------------------------------
            if opt_processor is not None:
                try:
                    frame_result = opt_processor.process_frame(frame, frame_count)
                    res_dict = _frame_result_to_dict(frame_result)
                    # Draw overlays on the DISPLAY frame (gray or RGB)
                    annotated = _draw_cli_overlay_from_dict(
                        display_frame,
                        res_dict,
                        show_ring=show_ring,
                        grayscale_mode=detector.grayscale_mode.name,
                    )
                except Exception:
                    annotated = display_frame.copy()
            else:
                result = detector.detect(
                    frame,
                    frame_number=frame_count,
                    source=f"camera:{camera_id}",
                    force_mode=force_mode,
                )
                smoothed = tracker.update(result)
                if smoothed.has_both:
                    smoothed.corneal_center = corneal_calc.calculate(
                        smoothed.pupil,
                        smoothed.limbus,
                        result.calibration,
                    )
                smoothed.calibration = result.calibration

                # Assign ring status manually from raw result to smoothed
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
                    smoothed.corneal_center.offset_magnitude_px = math.hypot(px - rcx, py - rcy)
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

                # Draw overlays on the DISPLAY frame (gray or RGB)
                annotated = _draw_cli_overlay(
                    display_frame,
                    smoothed,
                    show_ring=show_ring,
                    grayscale_mode=detector.grayscale_mode.name,
                )

            # -- FPS ---------------------------------------------
            fps_counter += 1
            now = time.monotonic()
            if (now - fps_timer) >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            h, w = annotated.shape[:2]
            cv2.putText(
                annotated,
                f"FPS: {current_fps:.1f}",
                (w - 160, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pupil Tracker — Camera", annotated)

            # -- keyboard ----------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
                print("  ⏸ Paused  (press SPACE to resume)")
            elif key == ord("s"):
                snapshot_count += 1
                snap_path = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(snap_path, annotated)
                print(f"  📸 Snapshot saved: {snap_path}")
            elif key == ord("g"):
                current_mode = detector.grayscale_mode.name.lower()
                try:
                    idx = _GRAYSCALE_CYCLE.index(current_mode)
                except ValueError:
                    idx = 0
                next_mode = _GRAYSCALE_CYCLE[(idx + 1) % len(_GRAYSCALE_CYCLE)]
                detector.set_grayscale_mode(next_mode)
                print(f"  🔲 Grayscale mode: {next_mode.upper()}")

    except KeyboardInterrupt:
        print("\n  ! Interrupted")

    # -- cleanup -------------------------------------------------
    if async_capture is not None:
        try:
            async_capture.stop()
        except Exception:
            pass
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done — {frame_count} frames captured\n")


# ================================================================
# CLI Overlay Drawing Helpers
# ================================================================


def _draw_cli_overlay(
    image: np.ndarray,
    result: Any,
    show_ring: bool = True,
    grayscale_mode: str = "",
) -> np.ndarray:
    """Draw detection overlay for CLI display.
    Works with both EyeDetectionResult and SimpleNamespace."""
    out = image.copy()
    h, w = out.shape[:2]

    cal = getattr(result, "calibration", None)
    has_cal = cal is not None and getattr(cal, "calibrated", False)
    mm_per_px = cal.mm_per_px if has_cal else 0.0

    # -- ring ----------------------------------------------------
    ring_status = getattr(result, "ring_status", "unknown")
    if show_ring and ring_status == "ring_present":
        rc = getattr(result, "ring_center", None)
        rr = getattr(result, "ring_radius", None)
        if rc and rr:
            cx, cy = int(round(rc[0])), int(round(rc[1]))
            r = int(round(rr))
            contour = getattr(result, "ring_contour", None)
            if contour is not None and len(contour) >= 5:
                cv2.drawContours(out, [contour.astype(np.int32)], -1, (0, 0, 255), 2)
            else:
                cv2.circle(out, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)
            label = f"R D={rr * 2.0:.0f}px"
            if has_cal:
                label += f" ({rr * 2.0 * mm_per_px:.2f}mm)"
            cv2.putText(
                out,
                label,
                (cx + 12, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # -- pupil ---------------------------------------------------
    if (
        getattr(result, "pupil", None)
        and getattr(result.pupil, "detected", False)
        and getattr(result.pupil, "ellipse", None) is not None
    ):
        e = result.pupil.ellipse
        ct = (int(round(e.center_x)), int(round(e.center_y)))
        r = int(round(e.radius))
        cv2.circle(out, ct, r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, ct, 4, (0, 255, 0), -1)

        dia_px = e.radius * 2.0
        label = f"P D={dia_px:.0f}px"
        if has_cal:
            label += f" ({dia_px * mm_per_px:.2f}mm)"
        cv2.putText(
            out,
            label,
            (ct[0] + 12, ct[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # -- limbus --------------------------------------------------
    if (
        getattr(result, "limbus", None)
        and getattr(result.limbus, "detected", False)
        and getattr(result.limbus, "ellipse", None) is not None
    ):
        e = result.limbus.ellipse
        ct = (int(round(e.center_x)), int(round(e.center_y)))
        r = int(round(e.radius))
        cv2.circle(out, ct, r, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.circle(out, ct, 4, (255, 100, 0), -1)

        dia_px = e.radius * 2.0
        label = f"L D={dia_px:.0f}px"
        if has_cal:
            label += f" ({dia_px * mm_per_px:.2f}mm)"
        cv2.putText(
            out,
            label,
            (ct[0] + 12, ct[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 100, 0),
            1,
            cv2.LINE_AA,
        )

    # -- offset line ---------------------------------------------
    if getattr(result, "has_both", False):
        pe = result.pupil.ellipse
        p_pt = (int(round(pe.center_x)), int(round(pe.center_y)))
        cc = getattr(result, "corneal_center", None)
        if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
            ref_pt = (int(round(cc.center_px[0])), int(round(cc.center_px[1])))
        else:
            le = result.limbus.ellipse
            ref_pt = (int(round(le.center_x)), int(round(le.center_y)))
        cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)

    # -- quality badge -------------------------------------------
    quality = "---"
    confidence = 0.0
    oq = getattr(result, "overall_quality", None)
    if oq is not None:
        quality = oq.value if hasattr(oq, "value") else str(oq)
    confidence = getattr(result, "overall_confidence", 0.0)

    color_map = {
        "SURGICAL": (0, 204, 0),
        "CLINICAL": (255, 204, 0),
        "RESEARCH": (0, 165, 255),
        "INSUFFICIENT": (0, 0, 255),
        "NO_DETECTION": (128, 128, 128),
        "EXCELLENT": (0, 204, 0),
        "GOOD": (255, 204, 0),
        "FAIR": (0, 165, 255),
        "POOR": (0, 0, 255),
        "UNUSABLE": (128, 128, 128),
    }
    badge_color = color_map.get(quality, (128, 128, 128))

    cv2.putText(
        out,
        f"{quality} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        badge_color,
        2,
    )

    if ring_status == "ring_present":
        cv2.putText(
            out,
            "RING DETECTED",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 200, 0),
            2,
        )

    # -- grayscale mode indicator --------------------------------
    if grayscale_mode and grayscale_mode.upper() != "OFF":
        gs_colors = {
            "AUTO": (255, 255, 0),
            "FORCE": (0, 255, 255),
        }
        gs_color = gs_colors.get(grayscale_mode.upper(), (200, 200, 200))
        gs_label = f"GRAY: {grayscale_mode.upper()}"
        cv2.putText(
            out,
            gs_label,
            (w - 180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gs_color,
            2,
            cv2.LINE_AA,
        )

    # -- processing time -----------------------------------------
    meta = getattr(result, "metadata", None)
    if meta is not None:
        proc_ms = getattr(meta, "processing_time_ms", 0)
        cv2.putText(
            out,
            f"{proc_ms:.0f}ms",
            (w - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

    return out


def _draw_cli_overlay_from_dict(
    image: np.ndarray,
    result_dict: Dict[str, Any],
    show_ring: bool = True,
    grayscale_mode: str = "",
) -> np.ndarray:
    """Draw overlay from a serialised result dict."""
    out = image.copy()
    h, w = out.shape[:2]

    ring_status = result_dict.get("ring_status", "unknown")
    if show_ring and ring_status == "ring_present":
        rx = result_dict.get("ring_center_x")
        ry = result_dict.get("ring_center_y")
        rr = result_dict.get("ring_radius")
        if rx is not None and ry is not None and rr is not None:
            cx, cy = int(round(rx)), int(round(ry))
            r = int(round(rr))
            cv2.circle(out, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    pupil = result_dict.get("pupil", {})
    pe = pupil.get("ellipse", {})
    if pupil.get("detected") and pe.get("center_x") is not None:
        ct = (int(round(pe["center_x"])), int(round(pe["center_y"])))
        r = int(round(pe.get("radius", 0)))
        if r > 0:
            cv2.circle(out, ct, r, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(out, ct, 4, (0, 255, 0), -1)

    limbus = result_dict.get("limbus", {})
    le = limbus.get("ellipse", {})
    if limbus.get("detected") and le.get("center_x") is not None:
        ct = (int(round(le["center_x"])), int(round(le["center_y"])))
        r = int(round(le.get("radius", 0)))
        if r > 0:
            cv2.circle(out, ct, r, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.circle(out, ct, 4, (255, 100, 0), -1)

    quality = result_dict.get("overall_quality", "---")
    conf = result_dict.get("overall_confidence", 0.0)
    cv2.putText(
        out,
        f"{quality} ({conf:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 204, 0),
        2,
    )

    if ring_status == "ring_present":
        cv2.putText(
            out,
            "RING DETECTED",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 200, 0),
            2,
        )

    # -- grayscale mode indicator --------------------------------
    if grayscale_mode and grayscale_mode.upper() != "OFF":
        gs_colors = {
            "AUTO": (255, 255, 0),
            "FORCE": (0, 255, 255),
        }
        gs_color = gs_colors.get(grayscale_mode.upper(), (200, 200, 200))
        gs_label = f"GRAY: {grayscale_mode.upper()}"
        cv2.putText(
            out,
            gs_label,
            (w - 180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gs_color,
            2,
            cv2.LINE_AA,
        )

    return out


# ================================================================
# Utility Helpers
# ================================================================


def _find_model_path(detector: Any, cfg: Any) -> Optional[str]:
    """Locate the trained model checkpoint."""
    candidates: List[str] = []

    if isinstance(cfg, dict):
        cfg_path = cfg.get("model_path")
        if cfg_path:
            candidates.append(str(cfg_path))

    if detector is not None:
        eng = getattr(detector, "ml_engine", None)
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
            return str(Path(c).resolve())
    return None


def _frame_result_to_dict(fr: Any) -> Dict[str, Any]:
    """Convert an OptimizedVideoProcessor FrameResult to dict."""
    d: Dict[str, Any] = {
        "metadata": {
            "frame_number": fr.frame_number,
            "processing_time_ms": fr.processing_ms,
        },
        "overall_quality": (fr.quality.value if fr.quality else "NO_DETECTION"),
        "overall_confidence": fr.confidence,
    }

    if fr.pupil_center and fr.pupil_axes:
        semi_a = max(fr.pupil_axes) / 2.0
        semi_b = min(fr.pupil_axes) / 2.0
        d["pupil"] = {
            "detected": True,
            "confidence": fr.confidence,
            "fit_type": getattr(fr, "pupil_fit_type", None),
            "ellipse": {
                "center_x": fr.pupil_center[0],
                "center_y": fr.pupil_center[1],
                "radius": semi_a,
                "semi_major": semi_a,
                "semi_minor": semi_b,
            },
        }
    else:
        d["pupil"] = {"detected": False, "ellipse": {}}

    if fr.limbus_center and fr.limbus_axes:
        semi_a = max(fr.limbus_axes) / 2.0
        semi_b = min(fr.limbus_axes) / 2.0
        d["limbus"] = {
            "detected": True,
            "confidence": min(0.95, fr.confidence + 0.05),
            "fit_type": getattr(fr, "limbus_fit_type", None),
            "ellipse": {
                "center_x": fr.limbus_center[0],
                "center_y": fr.limbus_center[1],
                "radius": semi_a,
                "semi_major": semi_a,
                "semi_minor": semi_b,
            },
        }
    else:
        d["limbus"] = {"detected": False, "ellipse": {}}

    if fr.limbus_axes is not None:
        limbus_full_dia = max(fr.limbus_axes)
        if limbus_full_dia > 0:
            px_per_mm = limbus_full_dia / _CORNEAL_DIAMETER_MM
            d["calibration"] = {
                "calibrated": True,
                "mm_per_px": 1.0 / px_per_mm,
                "px_per_mm": px_per_mm,
            }
        else:
            d["calibration"] = {"calibrated": False}
    else:
        d["calibration"] = {"calibrated": False}

    if fr.pupil_center and fr.limbus_center:
        dx = fr.pupil_center[0] - fr.limbus_center[0]
        dy = fr.pupil_center[1] - fr.limbus_center[1]
        mag_px = math.hypot(dx, dy)
        cal = d["calibration"]
        d["corneal_center"] = {
            "offset_magnitude_px": mag_px,
            "offset_magnitude_mm": (
                mag_px * cal["mm_per_px"] if cal.get("calibrated") else None
            ),
            "offset_angle_deg": math.degrees(math.atan2(dy, dx)),
        }
    else:
        d["corneal_center"] = {}

    d["ring_status"] = getattr(fr, "ring_status", "unknown")
    d["ring_confidence"] = getattr(fr, "ring_confidence", 0.0)
    d["image_category"] = getattr(fr, "image_category", "unknown")
    d["corneal_reference_source"] = getattr(
        fr,
        "corneal_reference_source",
        "limbus",
    )
    if hasattr(fr, "ring_center") and fr.ring_center is not None:
        d["ring_center_x"], d["ring_center_y"] = fr.ring_center
    if hasattr(fr, "ring_radius") and fr.ring_radius is not None:
        d["ring_radius"] = fr.ring_radius
    d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)

    return d


def _export_results_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Export results list to CSV."""
    if not results:
        return

    rows: List[Dict[str, Any]] = []
    prev_pupil_px = None
    prev_limbus_px = None
    for r in results:
        pupil = r.get("pupil", {})
        limbus = r.get("limbus", {})
        pe = pupil.get("ellipse", {})
        le = limbus.get("ellipse", {})
        cc = r.get("corneal_center", {})
        meta = r.get("metadata", {})
        cal_info = r.get("calibration", {})
        mm_px = cal_info.get("mm_per_px", 0)

        pupil_dia_px = pe.get("radius", 0) * 2 if pe.get("radius") else ""
        limbus_dia_px = le.get("radius", 0) * 2 if le.get("radius") else ""
        pupil_dia_mm = pupil_dia_px * mm_px if pupil_dia_px and mm_px else ""
        limbus_dia_mm = limbus_dia_px * mm_px if limbus_dia_px and mm_px else ""
        # compute deltas vs previous frame (px)
        pupil_delta = ""
        limbus_delta = ""
        try:
            if prev_pupil_px is not None and isinstance(pupil_dia_px, (int, float)):
                pupil_delta = pupil_dia_px - prev_pupil_px
        except Exception:
            pupil_delta = ""
        try:
            if prev_limbus_px is not None and isinstance(limbus_dia_px, (int, float)):
                limbus_delta = limbus_dia_px - prev_limbus_px
        except Exception:
            limbus_delta = ""

        rows.append(
            {
                "frame": meta.get("frame_number", ""),
                "ring_status": r.get("ring_status", ""),
                "pupil_detected": pupil.get("detected", False),
                "pupil_cx_px": pe.get("center_x", ""),
                "pupil_cy_px": pe.get("center_y", ""),
                "pupil_diameter_px": pupil_dia_px,
                "pupil_diameter_mm": pupil_dia_mm,
                "pupil_diameter_delta_px": pupil_delta,
                "pupil_fit_type": pupil.get("fit_type", ""),
                "pupil_confidence": pupil.get("confidence", ""),
                "limbus_detected": limbus.get("detected", False),
                "limbus_cx_px": le.get("center_x", ""),
                "limbus_cy_px": le.get("center_y", ""),
                "limbus_diameter_px": limbus_dia_px,
                "limbus_diameter_mm": limbus_dia_mm,
                "limbus_diameter_delta_px": limbus_delta,
                "limbus_fit_type": limbus.get("fit_type", ""),
                "limbus_confidence": limbus.get("confidence", ""),
                "offset_px": cc.get("offset_magnitude_px", ""),
                "offset_mm": cc.get("offset_magnitude_mm", ""),
                "offset_angle_deg": cc.get("offset_angle_deg", ""),
                "px_per_mm": cal_info.get("px_per_mm", ""),
                "quality": r.get("overall_quality", ""),
                "grayscale_mode": r.get("grayscale_mode", ""),
                "grayscale_applied": r.get("grayscale_applied", ""),
            }
        )

        # update previous frame values
        try:
            prev_pupil_px = float(pupil_dia_px) if isinstance(pupil_dia_px, (int, float)) else None
        except Exception:
            prev_pupil_px = None
        try:
            prev_limbus_px = float(limbus_dia_px) if isinstance(limbus_dia_px, (int, float)) else None
        except Exception:
            prev_limbus_px = None

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        # Append summary statistics: average pupil/limbus diameter pre/post dock
        try:
            # compute averages grouped by ring_status
            pre_pupil = []
            post_pupil = []
            pre_limbus = []
            post_limbus = []
            for r in rows:
                try:
                    rs = r.get("ring_status", "")
                    pd = r.get("pupil_diameter_mm")
                    ld = r.get("limbus_diameter_mm")
                    if isinstance(pd, (int, float)) and pd > 0:
                        if rs == "ring_present" or rs == "PRESENT":
                            post_pupil.append(pd)
                        else:
                            pre_pupil.append(pd)
                    if isinstance(ld, (int, float)) and ld > 0:
                        if rs == "ring_present" or rs == "PRESENT":
                            post_limbus.append(ld)
                        else:
                            pre_limbus.append(ld)
                except Exception:
                    continue

            def mean_or_empty(arr):
                return float(sum(arr) / len(arr)) if arr else ""

            summary = {
                "frame": "SUMMARY",
                "ring_status": "",
                "pupil_diameter_mm": mean_or_empty(pre_pupil),
                "limbus_diameter_mm": mean_or_empty(pre_limbus),
                "pupil_diameter_delta_px": "(pre_avg)",
            }
            writer.writerow({k: summary.get(k, "") for k in rows[0].keys()})
            summary2 = {
                "frame": "SUMMARY",
                "ring_status": "docked_avg",
                "pupil_diameter_mm": mean_or_empty(post_pupil),
                "limbus_diameter_mm": mean_or_empty(post_limbus),
                "pupil_diameter_delta_px": "(docked_avg)",
            }
            writer.writerow({k: summary2.get(k, "") for k in rows[0].keys()})
            # write diff row
            try:
                diff_pupil = ""
                if isinstance(summary2.get("pupil_diameter_mm"), float) and isinstance(summary.get("pupil_diameter_mm"), float):
                    diff_pupil = summary2.get("pupil_diameter_mm") - summary.get("pupil_diameter_mm")
                diff_row = {k: "" for k in rows[0].keys()}
                diff_row["frame"] = "SUMMARY"
                diff_row["ring_status"] = "docked_minus_pre"
                diff_row["pupil_diameter_mm"] = diff_pupil
                writer.writerow(diff_row)
            except Exception:
                # fail silently on diff row
                pass
        except Exception:
            # never fail CSV export for summary
            pass


# ================================================================
# Argument Parser
# ================================================================



# ==================================================================
# PENTACAM MANUAL ANNOTATION WORKSPACE
#
# The full workspace and data/geometry helpers live in this file.
# There is no separate annotation application module. GUI-only
# dependencies are loaded only when annotate mode is requested.
# ==================================================================

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})


class AnnotationValidationError(ValueError):
    """Raised when an annotation cannot safely be stored or exported."""


def discover_images(dataset_dir: Path | str) -> list[Path]:
    """Return supported raster images below *dataset_dir*, in stable order."""
    root = Path(dataset_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    excluded_parts = {".git", ".pentacam_annotations", "__pycache__", "venv", ".venv"}
    images: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images, key=lambda item: item.relative_to(root).as_posix().casefold())


def stable_image_id(relative_path: str | Path) -> str:
    """Create a stable, non-reversible identifier from a safe relative path."""
    from pathlib import PurePosixPath

    normalized = str(relative_path).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError("relative_path must be a non-empty path without traversal")
    normalized = relative.as_posix()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _image_dimensions(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive integers")
    return int(width), int(height)


def transform_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int,
    *,
    to_display: bool = True,
) -> tuple[float, float]:
    """Transform a point between original and display coordinates.

    The supported display transforms are 0 and 180 degrees. A 180-degree
    rotation is its own inverse: (x, y) -> (width - 1 - x, height - 1 - y).
    """
    width, height = _image_dimensions(width, height)
    if len(point) != 2:
        raise ValueError("point must contain exactly x and y")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("point coordinates must be finite")
    if rotation_degrees not in (0, 180):
        raise ValueError("only 0- and 180-degree orientation are supported")
    if rotation_degrees == 0:
        return x, y
    return float(width - 1) - x, float(height - 1) - y


def display_to_original_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int = 0,
    roi_box: Sequence[int] | None = None,
) -> tuple[float, float]:
    """Map a point in the current full-image/ROI display view to source pixels."""
    width, height = _image_dimensions(width, height)
    if roi_box is None:
        return transform_point(point, width, height, rotation_degrees, to_display=False)
    if len(roi_box) != 4:
        raise AnnotationValidationError("ROI box must contain four pixel coordinates")
    x0, y0, x1, y1 = (int(value) for value in roi_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")
    local_x, local_y = transform_point(
        point, x1 - x0, y1 - y0, rotation_degrees, to_display=False
    )
    return x0 + local_x, y0 + local_y


def original_to_display_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int = 0,
    roi_box: Sequence[int] | None = None,
) -> tuple[float, float] | None:
    """Map a source-image point to the current view; return None outside an ROI."""
    width, height = _image_dimensions(width, height)
    if roi_box is None:
        return transform_point(point, width, height, rotation_degrees)
    if len(roi_box) != 4:
        raise AnnotationValidationError("ROI box must contain four pixel coordinates")
    x0, y0, x1, y1 = (int(value) for value in roi_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")
    if len(point) != 2:
        raise ValueError("point must contain exactly x and y")
    x, y = float(point[0]), float(point[1])
    if not (x0 <= x < x1 and y0 <= y < y1):
        return None
    return transform_point((x - x0, y - y0), x1 - x0, y1 - y0, rotation_degrees)


def validate_points(
    points: Iterable[Sequence[float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Validate and normalize boundary points in original-image coordinates."""
    width, height = _image_dimensions(width, height)
    normalized: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise AnnotationValidationError(f"point {index} must contain x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError(f"point {index} is not numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise AnnotationValidationError(f"point {index} contains a non-finite coordinate")
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise AnnotationValidationError(
                f"point {index} ({x:.3f}, {y:.3f}) is outside {width}x{height} image bounds"
            )
        normalized.append((x, y))
    return normalized


def fit_ellipse_annotation(
    points: Iterable[Sequence[float]],
    width: int,
    height: int,
) -> dict[str, float]:
    """Fit and normalize an ellipse from at least five original-space points."""
    normalized = validate_points(points, width, height)
    if len(normalized) < 5:
        raise AnnotationValidationError("at least five boundary points are required")

    point_array = np.asarray(normalized, dtype=np.float32).reshape(-1, 1, 2)
    try:
        (center_x, center_y), (diameter_a, diameter_b), angle = cv2.fitEllipse(point_array)
    except cv2.error as exc:
        raise AnnotationValidationError(f"ellipse fitting failed: {exc}") from exc

    if diameter_a >= diameter_b:
        semi_major, semi_minor = diameter_a / 2.0, diameter_b / 2.0
        angle_deg = angle
    else:
        semi_major, semi_minor = diameter_b / 2.0, diameter_a / 2.0
        angle_deg = angle + 90.0

    values = (center_x, center_y, semi_major, semi_minor, angle_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise AnnotationValidationError("ellipse fit produced non-finite values")
    if semi_major <= 0 or semi_minor <= 0:
        raise AnnotationValidationError("ellipse axes must be positive")

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "semi_major": float(semi_major),
        "semi_minor": float(semi_minor),
        "angle_deg": float(angle_deg % 180.0),
    }


def normalize_roi_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> list[float]:
    """Normalize a half-open xyxy ROI to [0, 1] in original-image space."""
    width, height = _image_dimensions(width, height)
    if len(box) != 4:
        raise AnnotationValidationError("ROI box must be [x_min, y_min, x_max, y_max]")
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError("ROI coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise AnnotationValidationError("ROI coordinates must be finite")
    x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    if x1 <= x0 or y1 <= y0:
        raise AnnotationValidationError("ROI box must have positive width and height")
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def denormalize_roi_box(
    box_norm: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized half-open xyxy ROI coordinates to integer pixels."""
    width, height = _image_dimensions(width, height)
    if len(box_norm) != 4:
        raise AnnotationValidationError("normalized ROI must contain four values")
    try:
        values = [float(value) for value in box_norm]
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError("normalized ROI coordinates must be numeric") from exc
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AnnotationValidationError("normalized ROI coordinates must be finite and within [0, 1]")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise AnnotationValidationError("ROI box must have positive width and height")
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    return left, top, right, bottom


def validate_annotation_document(document: Mapping[str, Any]) -> None:
    """Validate the versioned workspace document before persistence/export."""
    if not isinstance(document, Mapping):
        raise AnnotationValidationError("annotation document must be a JSON object")
    version = str(document.get("schema_version", ""))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise AnnotationValidationError(f"unsupported annotation schema version: {version!r}")

    try:
        width, height = _image_dimensions(
            int(document["image_width"]), int(document["image_height"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationValidationError("positive image_width and image_height are required") from exc

    roi_box_norm = document.get("roi_box_norm")
    if roi_box_norm is not None:
        denormalize_roi_box(roi_box_norm, width, height)

    annotations = document.get("annotations", {})
    if not isinstance(annotations, Mapping):
        raise AnnotationValidationError("annotations must be an object")

    for mode, record in annotations.items():
        if mode not in {"pupil", "limbus", "ring"}:
            raise AnnotationValidationError(f"unsupported annotation mode: {mode!r}")
        if not isinstance(record, Mapping):
            raise AnnotationValidationError(f"{mode} annotation must be an object")
        points = record.get("boundary_points", [])
        if not isinstance(points, list):
            raise AnnotationValidationError(f"{mode} boundary_points must be a list")
        validate_points(points, width, height)
        ellipse = record.get("ellipse")
        if ellipse is not None:
            if not isinstance(ellipse, Mapping):
                raise AnnotationValidationError(f"{mode} ellipse must be an object")
            required = ("center_x", "center_y", "semi_major", "semi_minor", "angle_deg")
            try:
                values = {key: float(ellipse[key]) for key in required}
            except (KeyError, TypeError, ValueError) as exc:
                raise AnnotationValidationError(f"{mode} ellipse is missing numeric parameters") from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise AnnotationValidationError(f"{mode} ellipse contains non-finite values")
            if values["semi_major"] <= 0 or values["semi_minor"] <= 0:
                raise AnnotationValidationError(f"{mode} ellipse axes must be positive")
            if not (0 <= values["center_x"] < width and 0 <= values["center_y"] < height):
                raise AnnotationValidationError(f"{mode} ellipse center is outside image bounds")


def save_annotation_json(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Atomically write a validated annotation JSON file."""
    validate_annotation_document(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temporary_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


def load_annotation_json(path: Path | str) -> dict[str, Any]:
    """Load and validate a workspace annotation JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    validate_annotation_document(document)
    return dict(document)


def build_segmentation_mask(
    document: Mapping[str, Any],
    roi_box: Sequence[int] | None = None,
) -> np.ndarray:
    """Rasterize manually annotated ellipses into a uint8 class mask.

    *roi_box* is a half-open [x0, y0, x1, y1] rectangle in original-image
    pixels. When supplied, the returned mask has the exact crop dimensions.
    Class IDs: 0 background, 1 pupil, 2 iris region inside limbus, 3 ring outline.
    """
    validate_annotation_document(document)
    width, height = int(document["image_width"]), int(document["image_height"])
    if roi_box is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        if len(roi_box) != 4:
            raise AnnotationValidationError("ROI box must be [x_min, y_min, x_max, y_max]")
        try:
            x0, y0, x1, y1 = (int(value) for value in roi_box)
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError("ROI pixel coordinates must be integers") from exc
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    annotations = document.get("annotations", {})

    def draw_ellipse(mode: str, label: int, thickness: int) -> None:
        record = annotations.get(mode, {})
        ellipse = record.get("ellipse") if isinstance(record, Mapping) else None
        if not ellipse:
            return
        center = (
            int(round(float(ellipse["center_x"]) - x0)),
            int(round(float(ellipse["center_y"]) - y0)),
        )
        axes = (
            max(1, int(round(float(ellipse["semi_major"])))),
            max(1, int(round(float(ellipse["semi_minor"])))),
        )
        cv2.ellipse(
            mask, center, axes, float(ellipse["angle_deg"]), 0, 360,
            label, thickness=thickness,
        )

    draw_ellipse("limbus", 2, -1)
    ring_record = annotations.get("ring", {})
    ring = ring_record.get("ellipse") if isinstance(ring_record, Mapping) else None
    if ring:
        ring_axes = (
            max(1, int(round(float(ring["semi_major"])))),
            max(1, int(round(float(ring["semi_minor"])))),
        )
        draw_ellipse("ring", 3, max(2, int(round(min(ring_axes) * 0.04))))
    draw_ellipse("pupil", 1, -1)
    return mask

APP_TITLE = "Pentacam Manual Annotation Workspace"
SCHEMA_VERSION = "1.1"
ANNOTATION_MODES = ("pupil", "limbus", "ring")
MODE_COLORS = {
    "pupil": (0, 220, 80),
    "limbus": (255, 150, 0),
    "ring": (0, 80, 255),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PentacamAnnotationWorkspace:
    """Tkinter dataset browser and manual ellipse annotation workspace."""

    def __init__(
        self,
        root: tk.Tk,
        dataset_dir: Optional[Path] = None,
        annotation_dir: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x920")
        self.root.minsize(1000, 650)

        self.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else None
        self.annotation_dir = (
            Path(annotation_dir).expanduser().resolve()
            if annotation_dir
            else (Path.cwd() / ".pentacam_annotations").resolve()
        )
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        self._image_files: list[Path] = []
        self._image_index = -1
        self.current_image_path: Optional[Path] = None
        self.image_id: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._orientation_degrees = 0
        self._roi_box: Optional[tuple[int, int, int, int]] = None
        self._roi_history: list[Optional[tuple[int, int, int, int]]] = []
        self._roi_preview = False
        self._roi_selection_mode = False
        self._roi_drag_start: Optional[tuple[int, int]] = None
        self._scale = 1.0
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_anchor: Optional[tuple[int, int]] = None

        self._mode = "pupil"
        self._points: dict[str, list[tuple[float, float]]] = {
            mode: [] for mode in ANNOTATION_MODES
        }
        self._ellipses: dict[str, Optional[dict[str, float]]] = {
            mode: None for mode in ANNOTATION_MODES
        }
        self._undo_stack: list[tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]] = []
        self._redo_stack: list[tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]] = []
        self._loaded_document: Optional[dict[str, Any]] = None
        self._annotation_path: Optional[Path] = None
        self._dirty = False
        self._autosave_job: Optional[str] = None
        self._loading_metadata = False

        self._eye_var = tk.StringVar(value="OD")
        self._image_type_var = tk.StringVar(value=PentacamImageType.UNKNOWN.value)
        self._quality_var = tk.StringVar(value=PentacamQuality.NO_DETECTION.value)
        self._mode_var = tk.StringVar(value="pupil")
        self._orientation_var = tk.StringVar(value="0")
        self._annotator_var = tk.StringVar(value="")
        self._failure_reason_var = tk.StringVar(value="")
        self._roi_preview_var = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="Choose a dataset directory to begin")
        self._counter_var = tk.StringVar(value="0 / 0 images")

        self._build_ui()
        for variable in (
            self._eye_var,
            self._image_type_var,
            self._quality_var,
            self._annotator_var,
            self._failure_reason_var,
        ):
            variable.trace_add("write", self._on_metadata_changed)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.dataset_dir is not None:
            self._load_dataset_dir(self.dataset_dir)

    # ------------------------------ UI ---------------------------------
    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        body = ttk.Frame(self.root, padding=6)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        canvas_panel = ttk.LabelFrame(body, text="Original image (annotations use original pixel coordinates)")
        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_panel.grid_columnconfigure(0, weight=1)
        canvas_panel.grid_rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(canvas_panel, background="#17191d", cursor="crosshair", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<ButtonPress-1>", self._on_left_press)
        self._canvas.bind("<B1-Motion>", self._on_left_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_motion)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        controls_outer = ttk.Frame(body, width=320)
        controls_outer.grid(row=0, column=1, sticky="ns")
        controls_outer.grid_propagate(False)
        controls_canvas = tk.Canvas(
            controls_outer, width=300, highlightthickness=0, borderwidth=0
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_outer, orient="vertical", command=controls_canvas.yview
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scrollbar.pack(side="right", fill="y")
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        def _refresh_controls_scrollregion(_event: Any = None) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _fit_controls_width(event: tk.Event) -> None:
            controls_canvas.itemconfigure(controls_window, width=event.width)

        def _scroll_controls(event: tk.Event) -> None:
            if event.delta:
                controls_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _enable_controls_wheel(_event: tk.Event) -> None:
            self.root.bind_all("<MouseWheel>", _scroll_controls)

        def _disable_controls_wheel(_event: tk.Event) -> None:
            self.root.unbind_all("<MouseWheel>")

        controls.bind("<Configure>", _refresh_controls_scrollregion)
        controls_canvas.bind("<Configure>", _fit_controls_width)
        controls_outer.bind("<Enter>", _enable_controls_wheel)
        controls_outer.bind("<Leave>", _disable_controls_wheel)

        ttk.Label(controls, text="Dataset").pack(anchor="w", pady=(0, 3))
        ttk.Button(controls, text="Open dataset folder…", command=self._open_dataset_directory).pack(fill="x")
        ttk.Button(controls, text="Open one image…", command=self._open_single_image).pack(fill="x", pady=(3, 8))

        ttk.Label(controls, textvariable=self._counter_var).pack(anchor="w")
        ttk.Label(controls, text="Current image:", foreground="gray").pack(anchor="w")
        self._image_label = ttk.Label(controls, text="No image loaded", wraplength=280)
        self._image_label.pack(anchor="w", pady=(0, 4))
        navigation = ttk.Frame(controls)
        navigation.pack(fill="x")
        ttk.Button(navigation, text="◀ Previous", command=self._prev_image).pack(side="left", expand=True, fill="x")
        ttk.Button(navigation, text="Next ▶", command=self._next_image).pack(side="left", expand=True, fill="x", padx=(4, 0))

        roi_frame = ttk.LabelFrame(controls, text="Region of interest (ROI)")
        roi_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(roi_frame, text="Select / resize ROI (drag)", command=self._start_roi_selection).pack(
            fill="x", padx=5, pady=2
        )
        roi_actions = ttk.Frame(roi_frame)
        roi_actions.pack(fill="x", padx=5)
        ttk.Button(roi_actions, text="Undo ROI", command=self._undo_roi).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(roi_actions, text="Clear ROI", command=self._clear_roi).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        ttk.Checkbutton(
            roi_frame, text="Preview cropped ROI", variable=self._roi_preview_var,
            command=self._on_roi_preview_toggle,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Eye side").pack(anchor="w")
        ttk.Combobox(controls, textvariable=self._eye_var, values=("OD", "OS"), state="readonly").pack(fill="x")

        ttk.Label(controls, text="Image type").pack(anchor="w", pady=(7, 0))
        ttk.Combobox(
            controls, textvariable=self._image_type_var,
            values=tuple(item.value for item in PentacamImageType), state="readonly",
        ).pack(fill="x")

        ttk.Label(controls, text="Manual image quality").pack(anchor="w", pady=(7, 0))
        ttk.Combobox(
            controls, textvariable=self._quality_var,
            values=tuple(item.value for item in PentacamQuality), state="readonly",
        ).pack(fill="x")

        ttk.Label(controls, text="Annotator ID (optional)").pack(anchor="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self._annotator_var).pack(fill="x")
        ttk.Label(controls, text="Annotation note / failure reason").pack(anchor="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self._failure_reason_var).pack(fill="x")

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Annotation structure").pack(anchor="w")
        for mode, label in (("pupil", "Pupil"), ("limbus", "Limbus"), ("ring", "Suction ring")):
            ttk.Radiobutton(
                controls, text=label, variable=self._mode_var, value=mode,
                command=self._on_mode_changed,
            ).pack(anchor="w")

        actions = ttk.LabelFrame(controls, text="Annotation")
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Fit ellipse (≥5 points)", command=self._fit_ellipse).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Undo", command=self._undo_point).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Redo", command=self._redo_point).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Clear current structure", command=self._clear_current).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Save annotation", command=self._save_annotation).pack(fill="x", padx=5, pady=2)

        orientation = ttk.LabelFrame(controls, text="Display orientation")
        orientation.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(
            orientation, text="Original (0°)", variable=self._orientation_var,
            value="0", command=self._on_orientation_changed,
        ).pack(anchor="w", padx=5)
        ttk.Radiobutton(
            orientation, text="Rotate display 180°", variable=self._orientation_var,
            value="180", command=self._on_orientation_changed,
        ).pack(anchor="w", padx=5)

        zoom = ttk.Frame(controls)
        zoom.pack(fill="x", pady=(0, 8))
        ttk.Button(zoom, text="− Zoom", command=lambda: self._set_zoom(self._zoom / 1.2)).pack(side="left", expand=True, fill="x")
        ttk.Button(zoom, text="Fit", command=lambda: self._set_zoom(1.0, reset_pan=True)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(zoom, text="+ Zoom", command=lambda: self._set_zoom(self._zoom * 1.2)).pack(side="left", expand=True, fill="x")

        exports = ttk.LabelFrame(controls, text="Export (local files)")
        exports.pack(fill="x", pady=(0, 8))
        ttk.Button(exports, text="Export JSON…", command=self._export_json).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export ROI crop…", command=self._export_crop).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export overlay PNG…", command=self._export_overlay).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export segmentation mask…", command=self._export_mask).pack(fill="x", padx=5, pady=2)

        ttk.Label(
            controls,
            text="Left click: add point  |  Right click: undo\nMiddle-drag: pan  |  Mouse wheel: zoom\nOriginal images are never modified.",
            wraplength=285, foreground="gray",
        ).pack(anchor="w", pady=(3, 8))

        ttk.Label(self.root, textvariable=self._status_var, anchor="w", relief="sunken").grid(
            row=1, column=0, sticky="ew"
        )

    # --------------------------- Dataset --------------------------------
    def _open_dataset_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="Select Pentacam dataset directory",
            initialdir=str(self.dataset_dir) if self.dataset_dir else str(Path.cwd()),
        )
        if selected:
            self._load_dataset_dir(Path(selected))

    def _open_single_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open Pentacam raster image",
            filetypes=[("Supported images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if selected:
            path = Path(selected).resolve()
            self.dataset_dir = path.parent
            self._image_files = [path]
            self._image_index = -1
            self._load_image(0)

    def _load_dataset_dir(self, dataset_dir: Path) -> None:
        resolved_dir = dataset_dir.expanduser().resolve()
        try:
            images = discover_images(resolved_dir)
        except (OSError, NotADirectoryError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return
        self.dataset_dir = resolved_dir
        self._image_files = images
        self._image_index = -1
        if not images:
            self._clear_current_image()
            self._status_var.set("No supported raster images found (JPG, PNG, BMP, TIFF). DICOM/PDF import is not implemented.")
            return
        self._load_image(0)

    def _clear_current_image(self) -> None:
        self.current_image_path = None
        self.image_id = None
        self.original_image = None
        self.display_image = None
        self._points = {mode: [] for mode in ANNOTATION_MODES}
        self._ellipses = {mode: None for mode in ANNOTATION_MODES}
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._canvas.delete("all")
        self._image_label.config(text="No image loaded")
        self._counter_var.set(f"0 / {len(self._image_files)} images")

    def _load_image(self, index: int) -> None:
        if not self._image_files or not (0 <= index < len(self._image_files)):
            return
        if self.current_image_path is not None and self._dirty:
            self._flush_autosave()

        image_path = self._image_files[index]
        try:
            encoded = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV could not decode this file")
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Image load error", f"Could not load {image_path.name}: {exc}")
            return

        self.current_image_path = image_path
        self._image_index = index
        self.original_image = image
        self.display_image = image.copy()
        self._orientation_degrees = 0
        self._orientation_var.set("0")
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._roi_selection_mode = False
        self._roi_drag_start = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._points = {mode: [] for mode in ANNOTATION_MODES}
        self._ellipses = {mode: None for mode in ANNOTATION_MODES}

        if self.dataset_dir:
            try:
                relative_path = image_path.relative_to(self.dataset_dir).as_posix()
            except ValueError:
                relative_path = image_path.name
        else:
            relative_path = image_path.name
        self.image_id = stable_image_id(relative_path)
        self._annotation_path = self.annotation_dir / f"{self.image_id}.json"
        self._loaded_document = None
        self._dirty = False
        self._restore_saved_annotation()

        height, width = image.shape[:2]
        self._image_label.config(text=f"{image_path.name}\n{width} × {height}")
        self._counter_var.set(f"{index + 1} / {len(self._image_files)} images")
        self._status_var.set(f"Loaded {image_path.name}; coordinates are stored in original image space")
        self._refresh_canvas()

    def _restore_saved_annotation(self) -> None:
        self._loading_metadata = True
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        if self._annotation_path is None or not self._annotation_path.exists():
            self._loaded_document = None
            self._eye_var.set("OD")
            self._image_type_var.set(PentacamImageType.UNKNOWN.value)
            self._quality_var.set(PentacamQuality.NO_DETECTION.value)
            self._annotator_var.set("")
            self._failure_reason_var.set("")
            self._loading_metadata = False
            return
        try:
            document = load_annotation_json(self._annotation_path)
            if document.get("image_id") != self.image_id:
                raise AnnotationValidationError("saved annotation image_id does not match the selected image")
            self._loaded_document = document
            self._eye_var.set(document.get("eye_side", "OD"))
            self._image_type_var.set(document.get("image_type", PentacamImageType.UNKNOWN.value))
            self._quality_var.set(document.get("quality_grade", PentacamQuality.NO_DETECTION.value))
            self._annotator_var.set(document.get("annotator_id", ""))
            self._failure_reason_var.set(document.get("failure_reason", ""))
            orientation = document.get("orientation_transform", {})
            degrees = int(orientation.get("display_rotation_degrees", 0)) if isinstance(orientation, dict) else 0
            self._orientation_degrees = 180 if degrees == 180 else 0
            self._orientation_var.set(str(self._orientation_degrees))
            height, width = self.original_image.shape[:2]
            roi_norm = document.get("roi_box_norm")
            if roi_norm is not None:
                self._roi_box = denormalize_roi_box(roi_norm, width, height)
                self._roi_preview = True
                self._roi_preview_var.set(True)
            for mode in ANNOTATION_MODES:
                record = document.get("annotations", {}).get(mode, {})
                self._points[mode] = [tuple(point) for point in record.get("boundary_points", [])]
                self._ellipses[mode] = copy.deepcopy(record.get("ellipse"))
            self._dirty = False
            self._status_var.set("Restored saved annotation and ROI")
        except (OSError, ValueError, json.JSONDecodeError, AnnotationValidationError) as exc:
            self._loaded_document = None
            self._roi_box = None
            self._roi_preview = False
            self._roi_preview_var.set(False)
            self._status_var.set(f"Saved annotation could not be loaded: {exc}")
        finally:
            self._loading_metadata = False
        self._refresh_canvas()

    # ------------------------- Coordinate/UI ---------------------------
    def _view_dimensions(self) -> tuple[int, int]:
        if self.original_image is None:
            return 0, 0
        height, width = self.original_image.shape[:2]
        if self._roi_preview and self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            return x1 - x0, y1 - y0
        return width, height

    def _view_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        roi = self._roi_box if self._roi_preview else None
        return display_to_original_point(
            point, width, height, self._orientation_degrees, roi_box=roi
        )

    def _original_to_view(self, point: tuple[float, float]) -> Optional[tuple[float, float]]:
        if self.original_image is None:
            return None
        height, width = self.original_image.shape[:2]
        roi = self._roi_box if self._roi_preview else None
        return original_to_display_point(
            point, width, height, self._orientation_degrees, roi_box=roi
        )

    def _canvas_to_view(self, event: tk.Event) -> tuple[float, float]:
        return (
            (event.x - self._offset_x) / self._scale,
            (event.y - self._offset_y) / self._scale,
        )

    def _on_left_press(self, event: tk.Event) -> None:
        if self._roi_selection_mode:
            self._roi_drag_start = (event.x, event.y)
            self._canvas.delete("roi-drag")

    def _on_left_motion(self, event: tk.Event) -> None:
        if not self._roi_selection_mode or self._roi_drag_start is None:
            return
        x0, y0 = self._roi_drag_start
        self._canvas.delete("roi-drag")
        self._canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#00e5ff", width=2,
            dash=(6, 3), tags="roi-drag",
        )

    def _on_left_release(self, event: tk.Event) -> None:
        if self._roi_selection_mode:
            self._finish_roi_selection(event)
            return
        self._on_click(event)

    def _on_click(self, event: tk.Event) -> None:
        if self.original_image is None or self._roi_selection_mode:
            return
        view_x, view_y = self._canvas_to_view(event)
        view_width, view_height = self._view_dimensions()
        if not (0 <= view_x < view_width and 0 <= view_y < view_height):
            return
        original_point = self._view_to_original((view_x, view_y))
        height, width = self.original_image.shape[:2]
        try:
            validate_points([original_point], width, height)
        except AnnotationValidationError:
            return
        self._push_undo()
        self._points[self._mode].append(original_point)
        self._ellipses[self._mode] = None
        self._mark_dirty()
        self._status_var.set(f"{self._mode}: {len(self._points[self._mode])} boundary points")
        self._refresh_canvas()

    def _start_roi_selection(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image before selecting an ROI.")
            return
        self._roi_selection_mode = True
        self._roi_drag_start = None
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._canvas.configure(cursor="crosshair")
        self._refresh_canvas()
        self._status_var.set("Drag a rectangle around the eye image. Release to set or resize the ROI.")

    def _finish_roi_selection(self, event: tk.Event) -> None:
        self._canvas.delete("roi-drag")
        start = self._roi_drag_start
        self._roi_drag_start = None
        self._roi_selection_mode = False
        self._canvas.configure(cursor="crosshair")
        if self.original_image is None or start is None:
            return

        start_view = (
            (start[0] - self._offset_x) / self._scale,
            (start[1] - self._offset_y) / self._scale,
        )
        end_view = self._canvas_to_view(event)
        try:
            start_original = self._view_to_original(start_view)
            end_original = self._view_to_original(end_view)
        except (AnnotationValidationError, ValueError):
            self._status_var.set("ROI selection failed; try again inside the image.")
            return

        height, width = self.original_image.shape[:2]
        x0 = max(0, min(width - 1, int(np.floor(min(start_original[0], end_original[0])))))
        y0 = max(0, min(height - 1, int(np.floor(min(start_original[1], end_original[1])))))
        x1 = max(x0 + 1, min(width, int(np.ceil(max(start_original[0], end_original[0]))) + 1))
        y1 = max(y0 + 1, min(height, int(np.ceil(max(start_original[1], end_original[1]))) + 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            self._status_var.set("ROI is too small. Drag a larger rectangle.")
            self._refresh_canvas()
            return

        try:
            normalize_roi_box((x0, y0, x1, y1), width, height)
        except AnnotationValidationError as exc:
            self._status_var.set(f"Invalid ROI: {exc}")
            return

        self._roi_history.append(self._roi_box)
        if len(self._roi_history) > 50:
            self._roi_history.pop(0)
        self._roi_box = (x0, y0, x1, y1)
        self._roi_preview = True
        self._roi_preview_var.set(True)
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set(f"ROI selected: {x1 - x0} × {y1 - y0} px in original image coordinates.")

    def _undo_roi(self) -> None:
        if not self._roi_history:
            self._status_var.set("No previous ROI to restore")
            return
        self._roi_box = self._roi_history.pop()
        self._roi_preview = self._roi_box is not None
        self._roi_preview_var.set(self._roi_preview)
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set("Restored previous ROI")

    def _clear_roi(self) -> None:
        if self._roi_box is None:
            return
        self._roi_history.append(self._roi_box)
        self._roi_box = None
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set("ROI cleared; showing full image")

    def _on_roi_preview_toggle(self) -> None:
        if self._roi_preview_var.get() and self._roi_box is None:
            self._roi_preview_var.set(False)
            messagebox.showinfo("No ROI", "Select a rectangular ROI first.")
            return
        self._roi_preview = bool(self._roi_preview_var.get())
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._refresh_canvas()
        self._status_var.set("Showing ROI crop preview" if self._roi_preview else "Showing full image")

    def _on_right_click(self, _event: tk.Event) -> None:
        self._undo_point()

    def _on_mode_changed(self) -> None:
        self._mode = self._mode_var.get()
        self._status_var.set(f"Annotation mode: {self._mode}")

    def _on_metadata_changed(self, *_args: Any) -> None:
        if not self._loading_metadata and self.original_image is not None:
            self._mark_dirty()

    def _on_orientation_changed(self) -> None:
        self._apply_orientation_to_display(int(self._orientation_var.get()))

    def _apply_orientation_to_display(self, degrees: int) -> None:
        if degrees not in (0, 180):
            raise ValueError("only 0- and 180-degree orientation is supported")
        self._orientation_degrees = degrees
        self._mark_dirty()
        self._refresh_canvas()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        if self.original_image is None:
            return
        height, width = self.original_image.shape[:2]
        if self._roi_preview and self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            view_image = self.original_image[y0:y1, x0:x1].copy()
            view_width, view_height = x1 - x0, y1 - y0
        else:
            view_image = self.original_image.copy()
            view_width, view_height = width, height

        display = cv2.flip(view_image, -1) if self._orientation_degrees == 180 else view_image
        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        base_scale = min(canvas_width / view_width, canvas_height / view_height)
        self._scale = max(0.01, min(base_scale * self._zoom, 8.0))
        scaled_width = max(1, int(round(view_width * self._scale)))
        scaled_height = max(1, int(round(view_height * self._scale)))
        self._offset_x = (canvas_width - scaled_width) / 2 + self._pan_x
        self._offset_y = (canvas_height - scaled_height) / 2 + self._pan_y

        for mode in ANNOTATION_MODES:
            color = MODE_COLORS[mode]
            for point in self._points[mode]:
                view_point = self._original_to_view(point)
                if view_point is not None:
                    cv2.circle(
                        display, (int(round(view_point[0])), int(round(view_point[1]))),
                        3, color, -1,
                    )
            ellipse = self._ellipses[mode]
            if ellipse:
                center = self._original_to_view((ellipse["center_x"], ellipse["center_y"]))
                if center is not None:
                    axes = (
                        max(1, int(round(ellipse["semi_major"]))),
                        max(1, int(round(ellipse["semi_minor"]))),
                    )
                    cv2.ellipse(
                        display, (int(round(center[0])), int(round(center[1]))),
                        axes, float(ellipse["angle_deg"]), 0, 360, color, 2,
                    )

        if self._roi_box is not None and not self._roi_preview:
            x0, y0, x1, y1 = self._roi_box
            p0 = transform_point((x0, y0), width, height, self._orientation_degrees)
            p1 = transform_point((x1 - 1, y1 - 1), width, height, self._orientation_degrees)
            cv2.rectangle(
                display, (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))), (255, 220, 0), 2,
            )
        if self._roi_preview:
            cv2.putText(display, "ROI CROP PREVIEW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        self.display_image = display
        resized = cv2.resize(
            display, (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if self._scale < 1 else cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(
            self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo_image
        )

    def _set_zoom(self, value: float, reset_pan: bool = False) -> None:
        self._zoom = max(0.25, min(float(value), 6.0))
        if reset_pan:
            self._pan_x = self._pan_y = 0.0
        self._refresh_canvas()

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self._set_zoom(self._zoom * (1.15 if delta > 0 else 1 / 1.15))

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_anchor = (event.x, event.y)

    def _on_pan_motion(self, event: tk.Event) -> None:
        if self._pan_anchor is None:
            return
        previous_x, previous_y = self._pan_anchor
        self._pan_x += event.x - previous_x
        self._pan_y += event.y - previous_y
        self._pan_anchor = (event.x, event.y)
        self._refresh_canvas()

    # -------------------------- Annotation -----------------------------
    def _push_undo(self) -> None:
        self._undo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
        self._redo_stack.clear()
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _undo_point(self) -> None:
        if not self._undo_stack:
            # Allow one-point-at-a-time undo for annotations loaded from disk.
            if not self._points[self._mode]:
                self._status_var.set("Nothing to undo")
                return
            self._redo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
            self._points[self._mode].pop()
            self._ellipses[self._mode] = None
        else:
            self._redo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
            self._points, self._ellipses = self._undo_stack.pop()
        self._mark_dirty()
        self._refresh_canvas()

    def _redo_point(self) -> None:
        if not self._redo_stack:
            self._status_var.set("Nothing to redo")
            return
        self._undo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
        self._points, self._ellipses = self._redo_stack.pop()
        self._mark_dirty()
        self._refresh_canvas()

    def _clear_current(self) -> None:
        if self.original_image is None:
            return
        self._push_undo()
        self._points[self._mode] = []
        self._ellipses[self._mode] = None
        self._mark_dirty()
        self._refresh_canvas()

    def _fit_ellipse(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        height, width = self.original_image.shape[:2]
        try:
            ellipse = fit_ellipse_annotation(self._points[self._mode], width, height)
        except AnnotationValidationError as exc:
            messagebox.showwarning("Cannot fit ellipse", str(exc))
            return
        self._push_undo()
        self._ellipses[self._mode] = ellipse
        self._mark_dirty()
        self._status_var.set(
            f"{self._mode}: center=({ellipse['center_x']:.1f}, {ellipse['center_y']:.1f}), "
            f"axes=({ellipse['semi_major']:.1f}, {ellipse['semi_minor']:.1f}) px"
        )
        self._refresh_canvas()

    def _build_annotation_document(self) -> dict[str, Any]:
        if self.original_image is None or self.current_image_path is None or self.image_id is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        annotations: dict[str, Any] = {}
        for mode in ANNOTATION_MODES:
            points = validate_points(self._points[mode], width, height)
            if points or self._ellipses[mode]:
                annotations[mode] = {
                    "boundary_points": [[x, y] for x, y in points],
                    "ellipse": copy.deepcopy(self._ellipses[mode]),
                }

        now = _utc_now()
        created_at = (
            self._loaded_document.get("created_at", now)
            if self._loaded_document else now
        )
        status = (
            "OK" if self._ellipses["pupil"] and self._ellipses["limbus"]
            else "NO_PUPIL" if not self._ellipses["pupil"]
            else "NO_LIMBUS"
        )
        previous_revision = int(self._loaded_document.get("revision", 0)) if self._loaded_document else 0
        document = {
            "schema_version": SCHEMA_VERSION,
            "revision": previous_revision + 1,
            "image_id": self.image_id,
            "image_width": width,
            "image_height": height,
            "coordinate_system": "original_image_pixels",
            "eye_side": self._eye_var.get(),
            "image_type": self._image_type_var.get(),
            "quality_grade": self._quality_var.get(),
            "detection_status": status,
            "annotations": annotations,
            "roi_box_norm": (
                normalize_roi_box(self._roi_box, width, height)
                if self._roi_box is not None else None
            ),
            "roi_coordinate_system": "original_image_normalized_half_open_xyxy",
            "failure_reason": self._failure_reason_var.get().strip(),
            "annotator_id": self._annotator_var.get().strip(),
            "created_at": created_at,
            "annotated_at": now,
            "updated_at": now,
            "orientation_transform": {
                "display_rotation_degrees": self._orientation_degrees,
                "coordinates_stored_in": "original_image_pixels",
            },
        }
        validate_annotation_document(document)
        return document

    def _validate_current(self) -> None:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        for mode in ANNOTATION_MODES:
            validate_points(self._points[mode], width, height)
            ellipse = self._ellipses[mode]
            if ellipse:
                validate_annotation_document({
                    "schema_version": SCHEMA_VERSION,
                    "image_width": width,
                    "image_height": height,
                    "annotations": {mode: {"boundary_points": self._points[mode], "ellipse": ellipse}},
                })

    # ---------------------- Persistence/autosave ------------------------
    def _annotation_document_path(self) -> Path:
        if self._annotation_path is None:
            raise AnnotationValidationError("annotation path is not initialized")
        return self._annotation_path

    def _write_current_annotation(self) -> Path:
        document = self._build_annotation_document()
        path = save_annotation_json(self._annotation_document_path(), document)
        self._loaded_document = document
        self._dirty = False
        return path

    def _mark_dirty(self) -> None:
        if self.original_image is None:
            return
        self._dirty = True
        if self._autosave_job is not None:
            try:
                self.root.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
        self._autosave_job = self.root.after(900, self._autosave)

    def _autosave(self) -> None:
        self._autosave_job = None
        if not self._dirty or self.original_image is None:
            return
        try:
            self._write_current_annotation()
            self._status_var.set("Draft autosaved locally")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            self._status_var.set(f"Autosave failed: {exc}")

    def _flush_autosave(self) -> None:
        if self._autosave_job is not None:
            try:
                self.root.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
            self._autosave_job = None
        if self._dirty and self.original_image is not None:
            self._autosave()

    def _save_annotation(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            path = self._write_current_annotation()
            self._status_var.set(f"Saved private annotation: {path.name}")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Save error", str(exc))

    def _on_close(self) -> None:
        self._flush_autosave()
        self.root.destroy()

    # ------------------------------ Export ------------------------------
    def _export_json(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            document = self._build_annotation_document()
            path = filedialog.asksaveasfilename(
                title="Export annotation JSON",
                defaultextension=".json",
                initialfile=f"{self.image_id}.json",
                filetypes=[("JSON", "*.json")],
            )
            if path:
                save_annotation_json(path, document)
                self._status_var.set("Annotation JSON exported")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Export error", str(exc))

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        suffix = path.suffix.lower() or ".png"
        success, encoded = cv2.imencode(suffix, image)
        if not success:
            raise OSError(f"OpenCV could not encode {suffix}")
        encoded.tofile(str(path))

    def _render_original_overlay(self) -> np.ndarray:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        overlay = self.original_image.copy()
        if self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2)
        for mode in ANNOTATION_MODES:
            color = MODE_COLORS[mode]
            for x, y in self._points[mode]:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 3, color, -1)
            ellipse = self._ellipses[mode]
            if ellipse:
                center = (int(round(ellipse["center_x"])), int(round(ellipse["center_y"])))
                axes = (max(1, int(round(ellipse["semi_major"]))), max(1, int(round(ellipse["semi_minor"]))))
                cv2.ellipse(overlay, center, axes, ellipse["angle_deg"], 0, 360, color, 2)
        return overlay

    def _export_crop(self) -> None:
        """Export the raw ROI crop in original image orientation."""
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if self._roi_box is None:
            messagebox.showinfo("No ROI", "Select an ROI before exporting a crop.")
            return
        x0, y0, x1, y1 = self._roi_box
        crop = self.original_image[y0:y1, x0:x1].copy()
        path = filedialog.asksaveasfilename(
            title="Export ROI crop (original orientation)",
            defaultextension=".png",
            initialfile=f"{self.image_id}_crop.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            self._write_image(Path(path), crop)
            self._status_var.set(f"ROI crop exported: {x1 - x0} × {y1 - y0} px")
        except OSError as exc:
            messagebox.showerror("Crop export error", str(exc))

    def _export_overlay(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export original-coordinate overlay",
            defaultextension=".png",
            initialfile=f"{self.image_id}_overlay.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            self._write_image(Path(path), self._render_original_overlay())
            self._status_var.set("Overlay exported in original image orientation")
        except OSError as exc:
            messagebox.showerror("Overlay export error", str(exc))

    def _export_mask(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            document = self._build_annotation_document()
            mask = build_segmentation_mask(document, roi_box=self._roi_box)
            suffix = "_roi_mask.png" if self._roi_box is not None else "_mask.png"
            path = filedialog.asksaveasfilename(
                title="Export segmentation mask (class IDs 0–3; ROI crop if selected)",
                defaultextension=".png",
                initialfile=f"{self.image_id}{suffix}",
                filetypes=[("PNG", "*.png")],
            )
            if path:
                self._write_image(Path(path), mask)
                self._status_var.set(
                    f"Segmentation mask exported: {mask.shape[1]} × {mask.shape[0]} px "
                    "(0=background, 1=pupil, 2=iris, 3=ring outline)"
                )
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Mask export error", str(exc))

    # --------------------------- Navigation -----------------------------
    def _prev_image(self) -> None:
        if self._image_index > 0:
            self._flush_autosave()
            self._load_image(self._image_index - 1)

    def _next_image(self) -> None:
        if self._image_index + 1 < len(self._image_files):
            self._flush_autosave()
            self._load_image(self._image_index + 1)

def launch_pentacam_workspace(
    dataset_dir: Optional[Path] = None,
    annotation_dir: Optional[Path] = None,
) -> None:
    """Launch the integrated workspace; all annotation code lives in this file."""
    global tk, filedialog, messagebox, ttk, Image, ImageTk
    global PentacamImageType, PentacamQuality

    # Keep GUI-only imports lazy so normal detector/video/camera launch is unchanged.
    import tkinter as tk_module
    from tkinter import filedialog as filedialog_module
    from tkinter import messagebox as messagebox_module
    from tkinter import ttk as ttk_module
    from PIL import Image as image_module, ImageTk as image_tk_module
    from pupil_tracking.pentacam.types import (
        PentacamImageType as image_type_enum,
        PentacamQuality as quality_enum,
    )

    tk = tk_module
    filedialog = filedialog_module
    messagebox = messagebox_module
    ttk = ttk_module
    Image = image_module
    ImageTk = image_tk_module
    PentacamImageType = image_type_enum
    PentacamQuality = quality_enum

    root = tk.Tk()
    PentacamAnnotationWorkspace(
        root,
        dataset_dir=dataset_dir,
        annotation_dir=annotation_dir,
    )
    root.mainloop()



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch_gui.py",
        description=(
            "\n"
            "==============================================================\n"
            "Medevplus IXcentai surgical grade\n"
            "Deep learning pupil/iris detection & tracking\n"
            "==============================================================\n"
        ),
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=42, width=100
        ),
        epilog=(
            "\n"
            "----------------------- EXAMPLES -----------------------\n"
            "\n"
            "  Launch GUI (default):\n"
            "    python launch_gui.py\n"
            "    python launch_gui.py gui\n"
            "\n"
            "  Process single image:\n"
            "    python launch_gui.py image -i eye_photo.jpg\n"
            "    python launch_gui.py image -i eye.jpg --grayscale force\n"
            "\n"
            "  Process video file:\n"
            "    python launch_gui.py video -i recording.mp4\n"
            "    python launch_gui.py video -i rec.mp4 -o output.mp4\n"
            "    python launch_gui.py video -i rec.mp4 --grayscale force\n"
            "\n"
            "  Live camera:\n"
            "    python launch_gui.py camera\n"
            "    python launch_gui.py camera --grayscale force\n"
            "    python launch_gui.py camera  # press [G] to toggle\n"
            "\n"
            "  Pentacam manual annotation:\n"
            "    python launch_gui.py annotate\n"
            "    python launch_gui.py annotate --dataset-dir ./pentacam_images\n"
            "    python launch_gui.py annotate --dataset-dir ./pentacam_images --annotation-dir ./.pentacam_annotations\n"
            "\n"
            "---------------------------------------------------------\n"
        ),
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="gui",
        choices=["gui", "image", "video", "camera", "annotate"],
        help="Processing mode: gui / image / video / camera / annotate",
    )

    annotation_group = parser.add_argument_group("PENTACAM ANNOTATION")
    annotation_group.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Pentacam image directory (optional; can also be selected in the workspace)",
    )
    annotation_group.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Private annotation storage directory (default: ./.pentacam_annotations)",
    )

    io_group = parser.add_argument_group("INPUT / OUTPUT")
    io_group.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help="Input file path (image or video)",
    )
    io_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Output file path",
    )
    io_group.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to model weights (.pth)",
    )
    io_group.add_argument(
        "--camera-id",
        type=int,
        default=0,
        metavar="ID",
        help="Camera device index (default: 0)",
    )
    io_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )

    ring_group = parser.add_argument_group("RING DETECTION")
    ring_group.add_argument(
        "--ring-mode",
        type=str,
        default="auto",
        choices=["auto", "docked", "pre_docked"],
        help="Ring detection mode (default: auto)",
    )
    ring_group.add_argument(
        "--ring-classifier",
        type=str,
        default="models/ring_classifier.pth",
        metavar="PATH",
        help="Path to ring classifier model",
    )
    ring_group.add_argument(
        "--show-ring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw ring outline in output",
    )

    gray_group = parser.add_argument_group("GRAYSCALE MODE")
    gray_group.add_argument(
        "--grayscale",
        type=str,
        default="off",
        choices=["off", "auto", "force"],
        metavar="MODE",
        help=(
            "Grayscale processing mode.\n"
            "  off   = RGB passthrough (default)\n"
            "  auto  = detect grayscale -> enhance -> process\n"
            "  force = always convert to grayscale (IR look)\n"
            "In camera mode, press [G] to cycle modes."
        ),
    )

    pipe_group = parser.add_argument_group("PIPELINE")
    pipe_group.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use optimised pipeline",
    )
    pipe_group.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=_RUNTIME_PROFILE.recommended_fp16,
        help="FP16 half-precision",
    )
    pipe_group.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=_RUNTIME_PROFILE.recommended_compile,
        help="torch.compile JIT",
    )

    vid_group = parser.add_argument_group("VIDEO")
    vid_group.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1)",
    )
    vid_group.add_argument(
        "--resolution",
        type=int,
        default=_RUNTIME_PROFILE.recommended_resolution,
        metavar="PX",
        help="Inference resolution (default: machine profile)",
    )
    vid_group.add_argument(
        "--target-fps",
        type=float,
        default=_RUNTIME_PROFILE.recommended_target_fps,
        metavar="FPS",
        help="Target processing FPS (default: machine profile)",
    )

    roi_group = parser.add_argument_group("ROI & TRACKING")
    roi_group.add_argument(
        "--roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ROI tracking",
    )
    roi_group.add_argument(
        "--roi-cache",
        type=int,
        default=5,
        metavar="N",
        help="ROI cache lifetime in frames (default: 5)",
    )
    roi_group.add_argument(
        "--kalman-process-noise",
        type=float,
        default=0.03,
        metavar="F",
        help="Kalman process noise (default: 0.03)",
    )
    roi_group.add_argument(
        "--kalman-measure-noise",
        type=float,
        default=0.1,
        metavar="F",
        help="Kalman measurement noise (default: 0.1)",
    )

    # Calibration group
    cal_group = parser.add_argument_group("Calibration Options")
    cal_group.add_argument(
        "--calibration-mode",
        type=str,
        default="FIXED_PIXEL_SCALE",
        choices=["ANATOMICAL_ANCHOR", "FIXED_PIXEL_SCALE", "RING_REFLECTION"],
        help="Calibration mode: ANATOMICAL_ANCHOR, FIXED_PIXEL_SCALE, RING_REFLECTION (default: FIXED_PIXEL_SCALE)",
    )
    cal_group.add_argument(
        "--px-per-mm",
        type=float,
        default=58.2,
        metavar="F",
        help="Fixed manual scale in px/mm for FIXED_PIXEL_SCALE mode (default: 58.2)",
    )
    cal_group.add_argument(
        "--corneal-diameter-mm",
        type=float,
        default=12.0,
        metavar="F",
        help="Assumed horizontal corneal diameter in mm (default: 12.0)",
    )
    cal_group.add_argument(
        "--ring-diameter-mm",
        type=float,
        default=9.4,
        metavar="F",
        help="Known reference suction ring diameter in mm (default: 9.4)",
    )

    return parser



# ================================================================
# Main Entry Point
# ================================================================


# ================================================================
# Startup Self Check
# ================================================================


def _run_startup_self_check(args: argparse.Namespace, cfg: Any) -> int:
    warnings: List[str] = []
    errors: List[str] = []

    log_dir = Path(cfg.paths.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Cannot create log directory: {log_dir} ({exc})")

    if getattr(args, "input", None):
        input_path = Path(args.input)
        if not input_path.exists():
            errors.append(f"Input not found: {input_path}")
        elif args.mode in {"image", "video"} and not input_path.is_file():
            errors.append(f"Input is not a file: {input_path}")

    optimized_requested = bool(getattr(args, "optimized", True))
    if optimized_requested:
        if not _FAST_PIPELINE_AVAILABLE:
            warnings.append("Optimized pipeline unavailable - classic pipeline will be used.")
        else:
            model_candidates = []
            if getattr(args, "model", None):
                model_candidates.append(Path(args.model))
            model_candidates.extend(
                [
                    Path("models/best_model.pth"),
                    Path("models/segmentation_quantized.onnx"),
                ]
            )
            if not any(candidate.is_file() for candidate in model_candidates):
                warnings.append(
                    "No optimized model artifact found in expected locations - optimized mode may fall back."
                )

    for msg in warnings:
        print(f"  [WARN] {msg}")
    for msg in errors:
        print(f"  [ERROR] {msg}")
    return 1 if errors else 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Keep Pentacam annotation isolated from detector startup and settings.
    # The annotation workspace is launched lazily so existing GUI/image/video/
    # camera entry points retain their original initialization path.
    if args.mode == "annotate":
        dataset_dir = (
            Path(args.dataset_dir).expanduser()
            if args.dataset_dir else None
        )
        annotation_dir = (
            Path(args.annotation_dir).expanduser()
            if args.annotation_dir else None
        )
        launch_pentacam_workspace(
            dataset_dir=dataset_dir,
            annotation_dir=annotation_dir,
        )
        return

    print(_BANNER)
    print(
        f"Runtime profile: {_RUNTIME_PROFILE.name} "
        f"({int(round(_RUNTIME_PROFILE.ram_gb))} GB RAM, "
        f"{_RUNTIME_PROFILE.physical_cores} cores)"
    )

    cfg = get_config()
    logger = AuditLogger(log_dir=cfg.paths.log_dir)
    set_logger(logger)

    startup_status = _run_startup_self_check(args, cfg)
    if startup_status != 0:
        logger.close()
        sys.exit(startup_status)

    if args.mode == "image":
        if not args.input:
            print("  ✗ ERROR: --input / -i required for image mode")
            sys.exit(1)
        if not Path(args.input).is_file():
            print(f"  ✗ ERROR: File not found: {args.input}\n")
            sys.exit(1)
        process_image(args.input, model_path=args.model, device=args.device, args=args)

    elif args.mode == "video":
        if not args.input:
            print("  ✗ ERROR: --input / -i required for video mode")
            sys.exit(1)
        if not Path(args.input).is_file():
            print(f"  ✗ ERROR: File not found: {args.input}\n")
            sys.exit(1)
        process_video(
            args.input, output_path=args.output, model_path=args.model, args=args
        )

    elif args.mode == "camera":
        process_camera(camera_id=args.camera_id, model_path=args.model, args=args)

    elif args.mode == "gui":
        from pupil_tracking.interface.gui_app import PupilTrackingGUI, launch_gui

        def _launch_with_overrides() -> None:
            import tkinter as _tk
            from pupil_tracking.interface.theme import DarkTheme

            root = _tk.Tk()
            try:
                import os
                import sys
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
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
            app = PupilTrackingGUI(root, colors=colors)
            app._use_optimized_var.set(args.optimized)
            app._fp16_var.set(args.fp16)
            app._compile_var.set(getattr(args, "compile", True))
            app._resolution_var.set(args.resolution)
            app._stride_var.set(args.stride)
            app._target_fps_var.set(args.target_fps)
            app._roi_var.set(args.roi)
            app._roi_cache_var.set(args.roi_cache)
            app._kalman_process_var.set(args.kalman_process_noise)
            app._kalman_measure_var.set(args.kalman_measure_noise)
            if hasattr(app, "_res_display"):
                app._res_display.set(str(args.resolution))
            if hasattr(app, "_kp_display"):
                app._kp_display.set(f"{args.kalman_process_noise:.3f}")
            if hasattr(app, "_calibration_mode_var"):
                app._calibration_mode_var.set(args.calibration_mode)
            if hasattr(app, "_fixed_scale_var"):
                app._fixed_scale_var.set(args.px_per_mm)
            if hasattr(app, "_corneal_ref_mm_var"):
                app._corneal_ref_mm_var.set(args.corneal_diameter_mm)
            if hasattr(app, "_ring_ref_mm_var"):
                app._ring_ref_mm_var.set(args.ring_diameter_mm)
            if hasattr(app, "_grayscale_mode_var"):
                app._grayscale_mode_var.set(args.grayscale)
            app._invalidate_fast_engine()
            print("  ✓ CLI overrides applied to GUI Settings tab\n")
            root.mainloop()

        _defaults = _build_parser().parse_args([])
        _any_override = any(
            getattr(args, k) != getattr(_defaults, k)
            for k in [
                "optimized",
                "fp16",
                "compile",
                "resolution",
                "stride",
                "target_fps",
                "roi",
                "roi_cache",
                "kalman_process_noise",
                "kalman_measure_noise",
                "grayscale",
                "calibration_mode",
                "px_per_mm",
                "corneal_diameter_mm",
                "ring_diameter_mm",
            ]
        )

        if _any_override:
            print("  Launching GUI with CLI overrides…\n")
            _launch_with_overrides()
        else:
            print("  Launching GUI…\n")
            launch_gui()

    logger.close()


if __name__ == "__main__":
    main()
