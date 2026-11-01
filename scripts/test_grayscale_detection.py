#!/usr/bin/env python3
"""
Validate grayscale detection accuracy against RGB baseline.

This script processes a set of eye images in BOTH RGB and grayscale
modes, compares the detection results, and produces a detailed report
showing whether grayscale mode maintains detection accuracy.

PURPOSE
───────
After fine-tuning with ``scripts/finetune_grayscale.py``, run this
script to verify that:

    1. Grayscale detections match RGB detections (center offset < N px)
    2. Confidence scores are comparable
    3. No new detection failures are introduced
    4. Overall quality grades are maintained

The script also works on the ORIGINAL model (before fine-tuning) to
measure the baseline gap between RGB and grayscale — useful for
deciding whether fine-tuning is needed.

OUTPUT
──────
    - Console: per-image comparison table + aggregate statistics
    - CSV:     detailed per-image metrics (if ``--csv`` specified)
    - Images:  side-by-side visualisations (if ``--visualize`` specified)

USAGE
─────
  # Basic — test all images in a directory
  python scripts/test_grayscale_detection.py \\
      --image-dir clinical_data/clean \\
      --model models/best_model.pth

  # With output files
  python scripts/test_grayscale_detection.py \\
      --image-dir clinical_data/clean \\
      --model models/best_model.pth \\
      --csv diagnostic_output/grayscale_report.csv \\
      --visualize \\
      --output-dir diagnostic_output/grayscale_test

  # Test specific images
  python scripts/test_grayscale_detection.py \\
      --image-dir clinical_data/clean \\
      --model models/best_model.pth \\
      --max-images 20

  # Strict tolerance (tighter matching)
  python scripts/test_grayscale_detection.py \\
      --image-dir clinical_data/clean \\
      --model models/best_model.pth \\
      --center-tolerance 5.0 \\
      --radius-tolerance 3.0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Ensure project root is on sys.path ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import AuditLogger, set_logger, get_logger
from pupil_tracking.preprocessing.grayscale_handler import GrayscaleHandler

# ── Banner ──────────────────────────────────────────────────────
_BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        GRAYSCALE DETECTION VALIDATION — RGB vs GRAY         ║
╚══════════════════════════════════════════════════════════════╝
"""

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ================================================================
# Data classes for comparison results
# ================================================================

@dataclass
class SingleDetection:
    """Detection result for one structure (pupil or limbus)."""
    detected: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 0.0
    confidence: float = 0.0
    quality: str = ""


@dataclass
class ImageComparison:
    """Comparison of RGB vs grayscale detection for one image."""
    filename: str = ""
    image_width: int = 0
    image_height: int = 0

    # RGB results
    rgb_pupil: SingleDetection = field(default_factory=SingleDetection)
    rgb_limbus: SingleDetection = field(default_factory=SingleDetection)
    rgb_confidence: float = 0.0
    rgb_quality: str = ""
    rgb_time_ms: float = 0.0

    # Grayscale results
    gray_pupil: SingleDetection = field(default_factory=SingleDetection)
    gray_limbus: SingleDetection = field(default_factory=SingleDetection)
    gray_confidence: float = 0.0
    gray_quality: str = ""
    gray_time_ms: float = 0.0

    # Grayscale handler info
    was_input_grayscale: bool = False
    contrast_before: float = 0.0
    contrast_after: float = 0.0

    # Comparison metrics
    pupil_center_offset: float = float("nan")
    pupil_radius_diff: float = float("nan")
    limbus_center_offset: float = float("nan")
    limbus_radius_diff: float = float("nan")
    confidence_diff: float = 0.0

    # Verdict
    pupil_match: bool = False
    limbus_match: bool = False
    both_detected_rgb: bool = False
    both_detected_gray: bool = False
    new_failure: bool = False  # detected in RGB but not in gray


# ================================================================
# Extract detection info
# ================================================================

def _extract_detection(result: Any, structure: str) -> SingleDetection:
    """Extract pupil or limbus detection from an EyeDetectionResult.

    Parameters
    ----------
    result : EyeDetectionResult
        Detection result object.
    structure : str
        ``"pupil"`` or ``"limbus"``.

    Returns
    -------
    SingleDetection
    """
    det = SingleDetection()

    obj = getattr(result, structure, None)
    if obj is None:
        return det

    det.detected = getattr(obj, "detected", False)
    if not det.detected:
        return det

    ellipse = getattr(obj, "ellipse", None)
    if ellipse is None:
        det.detected = False
        return det

    det.center_x = float(getattr(ellipse, "center_x", 0.0))
    det.center_y = float(getattr(ellipse, "center_y", 0.0))
    det.radius = float(getattr(ellipse, "radius", 0.0))
    det.confidence = float(getattr(obj, "confidence", 0.0))

    quality = getattr(obj, "quality", None)
    if quality is not None:
        det.quality = quality.value if hasattr(quality, "value") else str(quality)

    return det


# ================================================================
# Compare two detections
# ================================================================

def _compare_detections(
    rgb: SingleDetection,
    gray: SingleDetection,
    center_tolerance: float,
    radius_tolerance: float,
) -> Tuple[float, float, bool]:
    """Compare RGB and grayscale detections for one structure.

    Parameters
    ----------
    rgb, gray : SingleDetection
        Detection results to compare.
    center_tolerance : float
        Maximum allowable center offset in pixels.
    radius_tolerance : float
        Maximum allowable radius difference in pixels.

    Returns
    -------
    (center_offset, radius_diff, is_match)
    """
    if not rgb.detected or not gray.detected:
        return float("nan"), float("nan"), False

    center_offset = math.sqrt(
        (rgb.center_x - gray.center_x) ** 2
        + (rgb.center_y - gray.center_y) ** 2
    )

    radius_diff = abs(rgb.radius - gray.radius)

    is_match = (
        center_offset <= center_tolerance
        and radius_diff <= radius_tolerance
    )

    return center_offset, radius_diff, is_match


# ================================================================
# Process one image
# ================================================================

def process_single_image(
    image_path: Path,
    detector: Any,
    grayscale_handler: GrayscaleHandler,
    center_tolerance: float,
    radius_tolerance: float,
) -> ImageComparison:
    """Process one image in RGB and grayscale modes, compare results.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    detector : UnifiedDetector
        The detector instance. Its grayscale mode will be toggled.
    grayscale_handler : GrayscaleHandler
        For quality metrics computation.
    center_tolerance : float
        Max center offset for a match (pixels).
    radius_tolerance : float
        Max radius difference for a match (pixels).

    Returns
    -------
    ImageComparison
    """
    comp = ImageComparison()
    comp.filename = image_path.name

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return comp

    comp.image_height, comp.image_width = image.shape[:2]

    # Check if input is inherently grayscale
    comp.was_input_grayscale = grayscale_handler.is_grayscale(image)

    # Quality metrics
    metrics = grayscale_handler.get_quality_metrics(image)
    comp.contrast_before = metrics.get("contrast", 0.0)

    # ── RGB detection ───────────────────────────────────────────
    detector.set_grayscale_mode("off")

    t0 = time.time()
    rgb_result = detector.detect(image, source=f"rgb:{image_path.name}")
    comp.rgb_time_ms = (time.time() - t0) * 1000.0

    comp.rgb_pupil = _extract_detection(rgb_result, "pupil")
    comp.rgb_limbus = _extract_detection(rgb_result, "limbus")
    comp.rgb_confidence = float(getattr(rgb_result, "overall_confidence", 0.0))

    rgb_quality = getattr(rgb_result, "overall_quality", None)
    if rgb_quality is not None:
        comp.rgb_quality = (
            rgb_quality.value if hasattr(rgb_quality, "value")
            else str(rgb_quality)
        )

    # ── Grayscale detection ─────────────────────────────────────
    detector.set_grayscale_mode("force")

    t0 = time.time()
    gray_result = detector.detect(image, source=f"gray:{image_path.name}")
    comp.gray_time_ms = (time.time() - t0) * 1000.0

    comp.gray_pupil = _extract_detection(gray_result, "pupil")
    comp.gray_limbus = _extract_detection(gray_result, "limbus")
    comp.gray_confidence = float(getattr(gray_result, "overall_confidence", 0.0))

    gray_quality = getattr(gray_result, "overall_quality", None)
    if gray_quality is not None:
        comp.gray_quality = (
            gray_quality.value if hasattr(gray_quality, "value")
            else str(gray_quality)
        )

    # Retrieve contrast after enhancement from detector info
    gs_info = detector.last_grayscale_info
    if gs_info is not None:
        comp.contrast_after = gs_info.contrast_after

    # ── Reset detector mode ─────────────────────────────────────
    detector.set_grayscale_mode("off")

    # ── Compare ─────────────────────────────────────────────────
    comp.both_detected_rgb = comp.rgb_pupil.detected and comp.rgb_limbus.detected
    comp.both_detected_gray = comp.gray_pupil.detected and comp.gray_limbus.detected

    # Pupil comparison
    (
        comp.pupil_center_offset,
        comp.pupil_radius_diff,
        comp.pupil_match,
    ) = _compare_detections(
        comp.rgb_pupil, comp.gray_pupil,
        center_tolerance, radius_tolerance,
    )

    # Limbus comparison
    (
        comp.limbus_center_offset,
        comp.limbus_radius_diff,
        comp.limbus_match,
    ) = _compare_detections(
        comp.rgb_limbus, comp.gray_limbus,
        center_tolerance, radius_tolerance,
    )

    # Confidence difference
    comp.confidence_diff = comp.gray_confidence - comp.rgb_confidence

    # New failure: detected in RGB but not in grayscale
    comp.new_failure = (
        (comp.rgb_pupil.detected and not comp.gray_pupil.detected)
        or (comp.rgb_limbus.detected and not comp.gray_limbus.detected)
    )

    return comp


# ================================================================
# Aggregate statistics
# ================================================================

@dataclass
class AggregateStats:
    """Summary statistics across all images."""
    total_images: int = 0
    images_with_rgb_pupil: int = 0
    images_with_gray_pupil: int = 0
    images_with_rgb_limbus: int = 0
    images_with_gray_limbus: int = 0
    images_both_rgb: int = 0
    images_both_gray: int = 0

    pupil_matches: int = 0
    pupil_comparable: int = 0
    limbus_matches: int = 0
    limbus_comparable: int = 0

    new_failures: int = 0

    pupil_offsets: List[float] = field(default_factory=list)
    pupil_radius_diffs: List[float] = field(default_factory=list)
    limbus_offsets: List[float] = field(default_factory=list)
    limbus_radius_diffs: List[float] = field(default_factory=list)

    rgb_confidences: List[float] = field(default_factory=list)
    gray_confidences: List[float] = field(default_factory=list)
    confidence_diffs: List[float] = field(default_factory=list)

    rgb_times: List[float] = field(default_factory=list)
    gray_times: List[float] = field(default_factory=list)

    inherently_grayscale: int = 0


def compute_aggregate(comparisons: List[ImageComparison]) -> AggregateStats:
    """Compute aggregate statistics from per-image comparisons."""
    stats = AggregateStats()
    stats.total_images = len(comparisons)

    for c in comparisons:
        if c.rgb_pupil.detected:
            stats.images_with_rgb_pupil += 1
        if c.gray_pupil.detected:
            stats.images_with_gray_pupil += 1
        if c.rgb_limbus.detected:
            stats.images_with_rgb_limbus += 1
        if c.gray_limbus.detected:
            stats.images_with_gray_limbus += 1
        if c.both_detected_rgb:
            stats.images_both_rgb += 1
        if c.both_detected_gray:
            stats.images_both_gray += 1

        if c.new_failure:
            stats.new_failures += 1

        if c.was_input_grayscale:
            stats.inherently_grayscale += 1

        # Pupil offsets (only when both detected)
        if c.rgb_pupil.detected and c.gray_pupil.detected:
            stats.pupil_comparable += 1
            if not math.isnan(c.pupil_center_offset):
                stats.pupil_offsets.append(c.pupil_center_offset)
            if not math.isnan(c.pupil_radius_diff):
                stats.pupil_radius_diffs.append(c.pupil_radius_diff)
            if c.pupil_match:
                stats.pupil_matches += 1

        # Limbus offsets
        if c.rgb_limbus.detected and c.gray_limbus.detected:
            stats.limbus_comparable += 1
            if not math.isnan(c.limbus_center_offset):
                stats.limbus_offsets.append(c.limbus_center_offset)
            if not math.isnan(c.limbus_radius_diff):
                stats.limbus_radius_diffs.append(c.limbus_radius_diff)
            if c.limbus_match:
                stats.limbus_matches += 1

        stats.rgb_confidences.append(c.rgb_confidence)
        stats.gray_confidences.append(c.gray_confidence)
        stats.confidence_diffs.append(c.confidence_diff)

        stats.rgb_times.append(c.rgb_time_ms)
        stats.gray_times.append(c.gray_time_ms)

    return stats


# ================================================================
# Reporting
# ================================================================

def _fmt(val: float, fmt: str = ".1f") -> str:
    """Format a float, handling NaN."""
    if math.isnan(val):
        return "  —  "
    return f"{val:{fmt}}"


def _percentile_str(values: List[float], label: str) -> str:
    """Format percentile summary for a list of values."""
    if not values:
        return f"    {label}: (no data)"
    arr = np.array(values)
    return (
        f"    {label}: "
        f"mean={np.mean(arr):.2f}, "
        f"median={np.median(arr):.2f}, "
        f"p95={np.percentile(arr, 95):.2f}, "
        f"max={np.max(arr):.2f}"
    )


def print_report(
    comparisons: List[ImageComparison],
    stats: AggregateStats,
    center_tolerance: float,
    radius_tolerance: float,
) -> None:
    """Print a comprehensive comparison report to console."""

    print(f"\n{'═' * 72}")
    print("  GRAYSCALE DETECTION VALIDATION REPORT")
    print(f"{'═' * 72}")

    # ── Per-image table ─────────────────────────────────────────
    print(f"\n  {'IMAGE':<25} {'RGB_P':>5} {'GRY_P':>5} "
          f"{'P_OFF':>6} {'RGB_L':>5} {'GRY_L':>5} "
          f"{'L_OFF':>6} {'CONF_Δ':>7} {'STATUS':>8}")
    print(f"  {'─' * 25} {'─' * 5} {'─' * 5} "
          f"{'─' * 6} {'─' * 5} {'─' * 5} "
          f"{'─' * 6} {'─' * 7} {'─' * 8}")

    for c in comparisons:
        name = c.filename[:24]

        rgb_p = "✓" if c.rgb_pupil.detected else "✗"
        gry_p = "✓" if c.gray_pupil.detected else "✗"
        p_off = _fmt(c.pupil_center_offset)

        rgb_l = "✓" if c.rgb_limbus.detected else "✗"
        gry_l = "✓" if c.gray_limbus.detected else "✗"
        l_off = _fmt(c.limbus_center_offset)

        conf_d = f"{c.confidence_diff:+.3f}"

        if c.new_failure:
            status = "  FAIL ✗"
        elif c.pupil_match and c.limbus_match:
            status = "  OK   ✓"
        elif c.pupil_match or c.limbus_match:
            status = " PARTIAL"
        elif not c.rgb_pupil.detected and not c.rgb_limbus.detected:
            status = " NO_DET"
        else:
            status = "  DRIFT"

        print(
            f"  {name:<25} {rgb_p:>5} {gry_p:>5} "
            f"{p_off:>6} {rgb_l:>5} {gry_l:>5} "
            f"{l_off:>6} {conf_d:>7} {status:>8}"
        )

    # ── Aggregate summary ───────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  AGGREGATE STATISTICS")
    print(f"{'─' * 72}")

    print(f"  Total images tested:       {stats.total_images}")
    print(f"  Inherently grayscale:      {stats.inherently_grayscale}")

    print(f"\n  DETECTION RATES:")
    print(f"    Pupil  — RGB: {stats.images_with_rgb_pupil}/{stats.total_images}  "
          f"Gray: {stats.images_with_gray_pupil}/{stats.total_images}")
    print(f"    Limbus — RGB: {stats.images_with_rgb_limbus}/{stats.total_images}  "
          f"Gray: {stats.images_with_gray_limbus}/{stats.total_images}")
    print(f"    Both   — RGB: {stats.images_both_rgb}/{stats.total_images}  "
          f"Gray: {stats.images_both_gray}/{stats.total_images}")

    print(f"\n  NEW FAILURES (detected in RGB, lost in gray): "
          f"{stats.new_failures}")

    # Match rates
    if stats.pupil_comparable > 0:
        pct = stats.pupil_matches / stats.pupil_comparable * 100
        print(f"\n  PUPIL MATCH RATE:  {stats.pupil_matches}/"
              f"{stats.pupil_comparable} = {pct:.1f}%  "
              f"(tolerance: {center_tolerance:.1f}px center, "
              f"{radius_tolerance:.1f}px radius)")
    if stats.limbus_comparable > 0:
        pct = stats.limbus_matches / stats.limbus_comparable * 100
        print(f"  LIMBUS MATCH RATE: {stats.limbus_matches}/"
              f"{stats.limbus_comparable} = {pct:.1f}%")

    # Offset distributions
    print(f"\n  OFFSET DISTRIBUTIONS:")
    print(_percentile_str(stats.pupil_offsets, "Pupil center offset (px) "))
    print(_percentile_str(stats.pupil_radius_diffs, "Pupil radius diff  (px) "))
    print(_percentile_str(stats.limbus_offsets, "Limbus center offset (px)"))
    print(_percentile_str(stats.limbus_radius_diffs, "Limbus radius diff  (px)"))

    # Confidence comparison
    print(f"\n  CONFIDENCE:")
    if stats.rgb_confidences:
        print(f"    RGB mean:       {np.mean(stats.rgb_confidences):.4f}")
    if stats.gray_confidences:
        print(f"    Gray mean:      {np.mean(stats.gray_confidences):.4f}")
    if stats.confidence_diffs:
        print(f"    Mean difference: {np.mean(stats.confidence_diffs):+.4f} "
              f"(negative = gray lower)")

    # Timing
    print(f"\n  TIMING:")
    if stats.rgb_times:
        print(f"    RGB  mean: {np.mean(stats.rgb_times):.1f} ms")
    if stats.gray_times:
        print(f"    Gray mean: {np.mean(stats.gray_times):.1f} ms  "
              f"(overhead: {np.mean(stats.gray_times) - np.mean(stats.rgb_times):+.1f} ms)")

    # ── Overall verdict ─────────────────────────────────────────
    print(f"\n{'═' * 72}")

    all_good = (
        stats.new_failures == 0
        and (
            stats.pupil_comparable == 0
            or stats.pupil_matches / stats.pupil_comparable >= 0.80
        )
        and (
            stats.limbus_comparable == 0
            or stats.limbus_matches / stats.limbus_comparable >= 0.80
        )
    )

    if all_good:
        print("  ✓ VERDICT: PASS — Grayscale detection is consistent with RGB")
    elif stats.new_failures == 0:
        print("  ~ VERDICT: PARTIAL — No new failures, but some drift detected")
        print("             Consider fine-tuning: python scripts/finetune_grayscale.py")
    else:
        print(f"  ✗ VERDICT: FAIL — {stats.new_failures} new detection failures")
        print("             Fine-tuning required: python scripts/finetune_grayscale.py")

    print(f"{'═' * 72}\n")


# ================================================================
# CSV export
# ================================================================

def export_csv(
    comparisons: List[ImageComparison],
    csv_path: str,
) -> None:
    """Export per-image comparison data to CSV."""

    rows = []
    for c in comparisons:
        rows.append({
            "filename": c.filename,
            "image_width": c.image_width,
            "image_height": c.image_height,
            "was_input_grayscale": c.was_input_grayscale,
            "contrast_before": f"{c.contrast_before:.2f}",
            "contrast_after": f"{c.contrast_after:.2f}",
            # RGB
            "rgb_pupil_detected": c.rgb_pupil.detected,
            "rgb_pupil_cx": f"{c.rgb_pupil.center_x:.2f}" if c.rgb_pupil.detected else "",
            "rgb_pupil_cy": f"{c.rgb_pupil.center_y:.2f}" if c.rgb_pupil.detected else "",
            "rgb_pupil_radius": f"{c.rgb_pupil.radius:.2f}" if c.rgb_pupil.detected else "",
            "rgb_pupil_confidence": f"{c.rgb_pupil.confidence:.4f}" if c.rgb_pupil.detected else "",
            "rgb_limbus_detected": c.rgb_limbus.detected,
            "rgb_limbus_cx": f"{c.rgb_limbus.center_x:.2f}" if c.rgb_limbus.detected else "",
            "rgb_limbus_cy": f"{c.rgb_limbus.center_y:.2f}" if c.rgb_limbus.detected else "",
            "rgb_limbus_radius": f"{c.rgb_limbus.radius:.2f}" if c.rgb_limbus.detected else "",
            "rgb_limbus_confidence": f"{c.rgb_limbus.confidence:.4f}" if c.rgb_limbus.detected else "",
            "rgb_overall_confidence": f"{c.rgb_confidence:.4f}",
            "rgb_quality": c.rgb_quality,
            "rgb_time_ms": f"{c.rgb_time_ms:.1f}",
            # Grayscale
            "gray_pupil_detected": c.gray_pupil.detected,
            "gray_pupil_cx": f"{c.gray_pupil.center_x:.2f}" if c.gray_pupil.detected else "",
            "gray_pupil_cy": f"{c.gray_pupil.center_y:.2f}" if c.gray_pupil.detected else "",
            "gray_pupil_radius": f"{c.gray_pupil.radius:.2f}" if c.gray_pupil.detected else "",
            "gray_pupil_confidence": f"{c.gray_pupil.confidence:.4f}" if c.gray_pupil.detected else "",
            "gray_limbus_detected": c.gray_limbus.detected,
            "gray_limbus_cx": f"{c.gray_limbus.center_x:.2f}" if c.gray_limbus.detected else "",
            "gray_limbus_cy": f"{c.gray_limbus.center_y:.2f}" if c.gray_limbus.detected else "",
            "gray_limbus_radius": f"{c.gray_limbus.radius:.2f}" if c.gray_limbus.detected else "",
            "gray_limbus_confidence": f"{c.gray_limbus.confidence:.4f}" if c.gray_limbus.detected else "",
            "gray_overall_confidence": f"{c.gray_confidence:.4f}",
            "gray_quality": c.gray_quality,
            "gray_time_ms": f"{c.gray_time_ms:.1f}",
            # Comparison
            "pupil_center_offset_px": _fmt(c.pupil_center_offset, ".2f").strip(),
            "pupil_radius_diff_px": _fmt(c.pupil_radius_diff, ".2f").strip(),
            "limbus_center_offset_px": _fmt(c.limbus_center_offset, ".2f").strip(),
            "limbus_radius_diff_px": _fmt(c.limbus_radius_diff, ".2f").strip(),
            "confidence_diff": f"{c.confidence_diff:+.4f}",
            "pupil_match": c.pupil_match,
            "limbus_match": c.limbus_match,
            "new_failure": c.new_failure,
        })

    if not rows:
        return

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV saved: {csv_path}")


# ================================================================
# Side-by-side visualisation
# ================================================================

def save_visualisation(
    image_path: Path,
    comp: ImageComparison,
    output_dir: Path,
) -> None:
    """Save a side-by-side RGB vs grayscale visualisation."""

    image = cv2.imread(str(image_path))
    if image is None:
        return

    handler = GrayscaleHandler()

    # Create grayscale version for display
    gray = handler.to_grayscale(image)
    enhanced = handler.enhance_grayscale(gray)
    gray_3ch = np.stack([enhanced, enhanced, enhanced], axis=2)

    # Draw detections on RGB
    rgb_vis = image.copy()
    if comp.rgb_pupil.detected:
        ct = (int(comp.rgb_pupil.center_x), int(comp.rgb_pupil.center_y))
        r = int(comp.rgb_pupil.radius)
        cv2.circle(rgb_vis, ct, r, (0, 255, 0), 2)
        cv2.circle(rgb_vis, ct, 3, (0, 255, 0), -1)
    if comp.rgb_limbus.detected:
        ct = (int(comp.rgb_limbus.center_x), int(comp.rgb_limbus.center_y))
        r = int(comp.rgb_limbus.radius)
        cv2.circle(rgb_vis, ct, r, (255, 100, 0), 2)

    cv2.putText(
        rgb_vis, f"RGB (conf={comp.rgb_confidence:.3f})",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    # Draw detections on grayscale
    gray_vis = gray_3ch.copy()
    if comp.gray_pupil.detected:
        ct = (int(comp.gray_pupil.center_x), int(comp.gray_pupil.center_y))
        r = int(comp.gray_pupil.radius)
        cv2.circle(gray_vis, ct, r, (0, 255, 0), 2)
        cv2.circle(gray_vis, ct, 3, (0, 255, 0), -1)
    if comp.gray_limbus.detected:
        ct = (int(comp.gray_limbus.center_x), int(comp.gray_limbus.center_y))
        r = int(comp.gray_limbus.radius)
        cv2.circle(gray_vis, ct, r, (255, 100, 0), 2)

    cv2.putText(
        gray_vis, f"GRAY (conf={comp.gray_confidence:.3f})",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )

    # Status bar
    status = "MATCH" if (comp.pupil_match and comp.limbus_match) else "DRIFT"
    if comp.new_failure:
        status = "FAILURE"
    status_color = (0, 200, 0) if status == "MATCH" else (0, 0, 255)
    cv2.putText(
        gray_vis, status,
        (10, gray_vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, status_color, 2,
    )

    # Combine side by side
    # Ensure same height
    h1, w1 = rgb_vis.shape[:2]
    h2, w2 = gray_vis.shape[:2]
    max_h = max(h1, h2)
    if h1 < max_h:
        rgb_vis = cv2.copyMakeBorder(
            rgb_vis, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=0,
        )
    if h2 < max_h:
        gray_vis = cv2.copyMakeBorder(
            gray_vis, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=0,
        )

    # Separator
    sep = np.full((max_h, 4, 3), 128, dtype=np.uint8)
    combined = np.hstack([rgb_vis, sep, gray_vis])

    # Save
    out_path = output_dir / f"{image_path.stem}_comparison.jpg"
    cv2.imwrite(str(out_path), combined)


# ================================================================
# Main pipeline
# ================================================================

def run_validation(args: argparse.Namespace) -> None:
    """Execute the full validation pipeline."""

    print(_BANNER)

    # ── Collect image files ─────────────────────────────────────
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"\n  ✗ ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    image_files = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"\n  ✗ ERROR: No images found in {image_dir}")
        sys.exit(1)

    if args.max_images and args.max_images < len(image_files):
        image_files = image_files[:args.max_images]

    print(f"  Image directory:    {image_dir}")
    print(f"  Images to test:     {len(image_files)}")
    print(f"  Model:              {args.model}")
    print(f"  Center tolerance:   {args.center_tolerance} px")
    print(f"  Radius tolerance:   {args.radius_tolerance} px")
    if args.csv:
        print(f"  CSV output:         {args.csv}")
    if args.visualize:
        print(f"  Visualisations:     {args.output_dir}")
    print(f"{'═' * 60}\n")

    # ── Initialise detector ─────────────────────────────────────
    from pupil_tracking.core.detector import UnifiedDetector

    cfg = get_config()
    detector = UnifiedDetector(
        model_path=args.model,
        config=cfg,
        grayscale_mode="off",  # we toggle per-image
    )

    grayscale_handler = GrayscaleHandler()

    # ── Output directory for visualisations ─────────────────────
    output_dir = None
    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process all images ──────────────────────────────────────
    comparisons: List[ImageComparison] = []

    for idx, img_path in enumerate(image_files):
        sys.stdout.write(
            f"\r  Processing {idx + 1}/{len(image_files)}: "
            f"{img_path.name[:40]:<40}"
        )
        sys.stdout.flush()

        comp = process_single_image(
            img_path, detector, grayscale_handler,
            args.center_tolerance, args.radius_tolerance,
        )
        comparisons.append(comp)

        # Save visualisation
        if output_dir is not None:
            save_visualisation(img_path, comp, output_dir)

    sys.stdout.write("\n\n")

    # ── Compute aggregate stats ─────────────────────────────────
    stats = compute_aggregate(comparisons)

    # ── Print report ────────────────────────────────────────────
    print_report(
        comparisons, stats,
        args.center_tolerance, args.radius_tolerance,
    )

    # ── Export CSV ──────────────────────────────────────────────
    if args.csv:
        export_csv(comparisons, args.csv)

    # ── Exit code ───────────────────────────────────────────────
    if stats.new_failures > 0:
        sys.exit(1)
    sys.exit(0)


# ================================================================
# Argument parser
# ================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_grayscale_detection.py",
        description=(
            "Compare pupil/limbus detection in RGB vs grayscale mode. "
            "Reports accuracy, offset, and confidence differences."
        ),
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=40, width=95
        ),
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="clinical_data/clean",
        metavar="PATH",
        help="Directory containing test eye images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        metavar="PATH",
        help="Path to model weights",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of images to test (default: all)",
    )
    parser.add_argument(
        "--center-tolerance",
        type=float,
        default=10.0,
        metavar="PX",
        help=(
            "Maximum center offset for a 'match' (default: 10 px). "
            "Tighter values catch smaller drift."
        ),
    )
    parser.add_argument(
        "--radius-tolerance",
        type=float,
        default=8.0,
        metavar="PX",
        help=(
            "Maximum radius difference for a 'match' (default: 8 px)."
        ),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Export per-image comparison to CSV",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Save side-by-side RGB vs grayscale visualisations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diagnostic_output/grayscale_test",
        metavar="PATH",
        help="Directory for visualisation images (with --visualize)",
    )

    return parser


# ================================================================
# Entry point
# ================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = get_config()
    audit_logger = AuditLogger(log_dir=cfg.paths.log_dir)
    set_logger(audit_logger)

    try:
        run_validation(args)
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted by user\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  ✗ ERROR: {exc}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        audit_logger.close()


if __name__ == "__main__":
    main()