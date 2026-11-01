#!/usr/bin/env python3
"""
Diagnostic tool — shows exactly what the model sees.

For each image, saves:
  1. Raw segmentation mask (colour-coded)
  2. Post-processed masks (pupil, iris)
  3. Detected contours + ellipse fits
  4. Final detection overlay

Use this to debug detection failures.

Usage:
    python scripts/diagnose_detection.py
    python scripts/diagnose_detection.py --input clinical_data/clean/eye_06.jpeg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import math
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from pupil_tracking.ml.inference import SegmentationInference
from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.utils.config import get_config


_CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # pupil — green
    2: (255, 100, 0),    # iris — blue
}


def diagnose_image(
    image_path: str,
    engine: SegmentationInference,
    detector: UnifiedDetector,
    output_dir: Path,
) -> None:
    """Run full diagnostics on one image."""
    stem = Path(image_path).stem
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Cannot read: {image_path}")
        return

    h, w = image.shape[:2]
    input_size = engine.input_size
    print(f"\n  Image: {stem} ({w}x{h})")
    print(f"  Model input: {input_size}x{input_size}")
    print(f"  Scale: x={w/input_size:.2f} y={h/input_size:.2f}")

    # -- raw mask at model resolution ----------------------------------
    pred_mask, probs = engine.get_raw_mask(image)

    # count pixels per class
    for c in range(3):
        count = int(np.sum(pred_mask == c))
        pct = 100 * count / (input_size * input_size)
        prob_mean = float(np.mean(probs[c][pred_mask == c])) if count > 0 else 0
        print(f"  Class {c} ({'bg' if c==0 else 'pupil' if c==1 else 'iris'}): "
              f"{count:6d}px ({pct:5.1f}%)  mean_prob={prob_mean:.3f}")

    # -- save colour mask ----------------------------------------------
    color_mask = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    for c, col in _CLASS_COLORS.items():
        color_mask[pred_mask == c] = col

    mask_path = output_dir / f"{stem}_01_mask.png"
    cv2.imwrite(str(mask_path), color_mask)

    # -- save probability heatmaps ------------------------------------
    for c, name in enumerate(["bg", "pupil", "iris"]):
        heatmap = (probs[c] * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heat_path = output_dir / f"{stem}_02_prob_{name}.png"
        cv2.imwrite(str(heat_path), heatmap_color)

    # -- save resized input (what model sees) --------------------------
    resized = cv2.resize(image, (input_size, input_size))
    blend = cv2.addWeighted(resized, 0.6, color_mask, 0.4, 0)
    blend_path = output_dir / f"{stem}_03_blend.png"
    cv2.imwrite(str(blend_path), blend)

    # -- run full detection --------------------------------------------
    result = detector.detect(image, source=str(image_path))

    # -- save detection overlay ----------------------------------------
    overlay = image.copy()

    if result.pupil.detected and result.pupil.ellipse is not None:
        e = result.pupil.ellipse
        center = (int(round(e.center_x)), int(round(e.center_y)))
        axes = (int(round(e.semi_major)), int(round(e.semi_minor)))
        angle = int(round(e.angle_deg))
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.circle(overlay, center, 5, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            f"PUPIL ({e.center_x:.0f},{e.center_y:.0f}) r={e.radius:.0f} "
            f"conf={result.pupil.confidence:.2f}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    if result.limbus.detected and result.limbus.ellipse is not None:
        e = result.limbus.ellipse
        center = (int(round(e.center_x)), int(round(e.center_y)))
        axes = (int(round(e.semi_major)), int(round(e.semi_minor)))
        angle = int(round(e.angle_deg))
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (255, 100, 0), 2)
        cv2.circle(overlay, center, 5, (255, 100, 0), -1)
        cv2.putText(
            overlay,
            f"LIMBUS ({e.center_x:.0f},{e.center_y:.0f}) r={e.radius:.0f} "
            f"conf={result.limbus.confidence:.2f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2,
        )

    # offset line
    if result.has_both:
        p = result.pupil.ellipse
        l = result.limbus.ellipse
        p_pt = (int(round(p.center_x)), int(round(p.center_y)))
        l_pt = (int(round(l.center_x)), int(round(l.center_y)))
        cv2.line(overlay, p_pt, l_pt, (0, 255, 255), 2)
        offset = math.sqrt(
            (p.center_x - l.center_x)**2 + (p.center_y - l.center_y)**2
        )
        cv2.putText(
            overlay,
            f"Offset: {offset:.1f}px",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

    quality = result.overall_quality.value
    cv2.putText(
        overlay,
        f"Quality: {quality} ({result.overall_confidence:.3f})",
        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    overlay_path = output_dir / f"{stem}_04_detection.png"
    cv2.imwrite(str(overlay_path), overlay)

    # -- print results -------------------------------------------------
    if result.pupil.detected:
        e = result.pupil.ellipse
        print(f"  PUPIL:  ({e.center_x:.1f}, {e.center_y:.1f}) "
              f"r={e.radius:.1f} conf={result.pupil.confidence:.3f} "
              f"[{result.pupil.quality.value}]")
    else:
        print(f"  PUPIL:  NOT DETECTED")

    if result.limbus.detected:
        e = result.limbus.ellipse
        print(f"  LIMBUS: ({e.center_x:.1f}, {e.center_y:.1f}) "
              f"r={e.radius:.1f} conf={result.limbus.confidence:.3f} "
              f"[{result.limbus.quality.value}]")
    else:
        print(f"  LIMBUS: NOT DETECTED")

    if result.alerts:
        for alert in result.alerts:
            print(f"  ALERT: {alert}")

    print(f"  Saved diagnostics to {output_dir}/{stem}_*.png")


def main():
    import math

    parser = argparse.ArgumentParser(
        description="Diagnose detection on images"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Single image path (if not given, processes all in clean/)",
    )
    parser.add_argument(
        "--model", type=str, default="models/best_model.pth",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="clinical_data/diagnostic_output",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DETECTION DIAGNOSTICS")
    print(f"{'='*60}")

    engine = SegmentationInference(model_path=args.model)
    detector = UnifiedDetector(model_path=args.model)

    if args.input:
        images = [Path(args.input)]
    else:
        img_dir = Path("clinical_data/clean")
        images = sorted(
            list(img_dir.glob("*.jpeg"))
            + list(img_dir.glob("*.jpg"))
            + list(img_dir.glob("*.png"))
        )

    for img_path in images:
        diagnose_image(str(img_path), engine, detector, output_dir)

    print(f"\n{'='*60}")
    print(f"  Diagnostics saved to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()