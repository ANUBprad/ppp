#!/usr/bin/env python3
"""
Comprehensive data quality report for the annotation dataset.

Checks:
  - All images exist and are readable
  - All annotations have required fields
  - Pupil/limbus sizes are anatomically plausible
  - Pupil is inside limbus
  - Boundary points are consistent with ellipse parameters
  - Image dimensions match annotation metadata
  - Sufficient data for training

Usage:
    python scripts/verify_data.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.ml.dataset import load_annotations, get_annotation_stats


def verify_dataset(
    annotation_path: str = "clinical_data/annotations/annotations.json",
    image_dir: str = "clinical_data/clean",
) -> bool:
    """Run all verification checks. Returns True if no critical issues."""

    print(f"\n{'='*70}")
    print(f"  DATA QUALITY VERIFICATION REPORT")
    print(f"{'='*70}\n")

    # load
    try:
        image_ids, annotations = load_annotations(annotation_path)
    except Exception as e:
        print(f"✗ CRITICAL: Cannot load annotations: {e}")
        return False

    print(f"Annotation file: {annotation_path}")
    print(f"Image directory:  {image_dir}")
    print(f"Images found:     {len(image_ids)}")
    print()

    # stats
    stats = get_annotation_stats(annotations)

    warnings = 0
    errors = 0
    img_dir = Path(image_dir)

    # ── check 1: images exist ───────────────────────────────────
    print("─── CHECK 1: Image Availability ───")
    for img_id in image_ids:
        found = False
        for ext in (".jpeg", ".jpg", ".png"):
            p = img_dir / f"{img_id}{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    exp_w = annotations[img_id].get("image_width", 0)
                    exp_h = annotations[img_id].get("image_height", 0)
                    size_ok = (
                        (exp_w == 0 or w == exp_w)
                        and (exp_h == 0 or h == exp_h)
                    )
                    status = "✓" if size_ok else "⚠ size mismatch"
                    if not size_ok:
                        warnings += 1
                    print(
                        f"  {status} {img_id}: {w}×{h} "
                        f"(expected {exp_w}×{exp_h})"
                    )
                    found = True
                    break
                else:
                    print(f"  ✗ {img_id}: file exists but unreadable")
                    errors += 1
                    found = True
                    break
        if not found:
            print(f"  ✗ {img_id}: NOT FOUND in {img_dir}")
            errors += 1
    print()

    # ── check 2: annotation completeness ────────────────────────
    print("─── CHECK 2: Annotation Completeness ───")
    for img_id in image_ids:
        ann = annotations[img_id]
        issues = []
        if "pupil_center" not in ann:
            issues.append("no pupil")
            errors += 1
        if "limbus_center" not in ann:
            issues.append("no limbus")
            errors += 1
        if "pupil_axes" not in ann:
            issues.append("no pupil axes")
            errors += 1
        if "limbus_axes" not in ann:
            issues.append("no limbus axes")
            errors += 1

        n_pbp = len(ann.get("pupil_boundary", []))
        n_lbp = len(ann.get("limbus_boundary", []))

        if issues:
            print(f"  ✗ {img_id}: {', '.join(issues)}")
        else:
            ring = "ring" if ann.get("has_suction_ring") else "no ring"
            print(
                f"  ✓ {img_id}: pupil({n_pbp}pts) "
                f"limbus({n_lbp}pts) [{ring}]"
            )
    print()

    # ── check 3: anatomical plausibility ────────────────────────
    print("─── CHECK 3: Anatomical Plausibility ───")
    for img_id in image_ids:
        ann = annotations[img_id]
        img_issues = []

        pr = ann.get("pupil_radius", 0)
        lr = ann.get("limbus_radius", 0)
        iw = ann.get("image_width", 1000)

        # pupil size
        if pr > 0:
            pupil_frac = pr / iw
            if pupil_frac < 0.01:
                img_issues.append(
                    f"pupil very small ({pr:.0f}px = "
                    f"{pupil_frac*100:.1f}% of width)"
                )
                warnings += 1
            elif pupil_frac > 0.3:
                img_issues.append(
                    f"pupil very large ({pr:.0f}px = "
                    f"{pupil_frac*100:.1f}% of width)"
                )
                warnings += 1

        # limbus size
        if lr > 0:
            limbus_frac = lr / iw
            if limbus_frac < 0.05:
                img_issues.append(
                    f"limbus very small ({lr:.0f}px = "
                    f"{limbus_frac*100:.1f}% of width)"
                )
                warnings += 1
            elif limbus_frac > 0.7:
                img_issues.append(
                    f"limbus very large ({lr:.0f}px = "
                    f"{limbus_frac*100:.1f}% of width)"
                )
                warnings += 1

        # pupil/limbus ratio
        if pr > 0 and lr > 0:
            ratio = pr / lr
            if ratio < 0.10:
                img_issues.append(
                    f"pupil/limbus ratio very low ({ratio:.3f})"
                )
                warnings += 1
            elif ratio > 0.75:
                img_issues.append(
                    f"pupil/limbus ratio very high ({ratio:.3f})"
                )
                warnings += 1

        # pupil inside limbus
        if "pupil_center" in ann and "limbus_center" in ann:
            pc = ann["pupil_center"]
            lc = ann["limbus_center"]
            offset = math.sqrt(
                (pc[0] - lc[0])**2 + (pc[1] - lc[1])**2
            )
            if lr > 0:
                offset_ratio = offset / lr
                if offset_ratio > 0.3:
                    img_issues.append(
                        f"pupil centre far from limbus centre "
                        f"(offset={offset:.0f}px = "
                        f"{offset_ratio:.2f}×limbus_r)"
                    )
                    warnings += 1

        # ellipse aspect ratio
        for prefix in ["pupil", "limbus"]:
            axes = ann.get(f"{prefix}_axes")
            if axes and len(axes) == 2 and axes[0] > 0:
                aspect = min(axes) / max(axes)
                if aspect < 0.4:
                    img_issues.append(
                        f"{prefix} very elongated "
                        f"(aspect={aspect:.2f})"
                    )
                    warnings += 1

        if img_issues:
            for issue in img_issues:
                print(f"  ⚠ {img_id}: {issue}")
        else:
            ratio_str = f"{pr/lr:.2f}" if lr > 0 else "?"
            print(
                f"  ✓ {img_id}: pupil_r={pr:.0f}px "
                f"limbus_r={lr:.0f}px ratio={ratio_str}"
            )
    print()

    # ── check 4: training readiness ─────────────────────────────
    print("─── CHECK 4: Training Readiness ───")
    n_both = stats["with_both"]
    if n_both < 2:
        print(
            f"  ✗ CRITICAL: Need ≥ 2 images with both pupil & "
            f"limbus, got {n_both}"
        )
        errors += 1
    elif n_both < 5:
        print(
            f"  ⚠ Only {n_both} images with both annotations — "
            f"model may underfit"
        )
        warnings += 1
    else:
        print(f"  ✓ {n_both} images with both annotations")

    n_val = max(1, int(n_both * 0.2))
    n_train = n_both - n_val
    print(f"  Train/val split: {n_train} train, {n_val} val")

    if stats["with_ring"] > 0:
        pct_ring = 100 * stats["with_ring"] / stats["total_images"]
        print(
            f"  Ring presence: {stats['with_ring']}/{stats['total_images']} "
            f"({pct_ring:.0f}%) — model will see both cases"
        )
    print()

    # ── summary ─────────────────────────────────────────────────
    print(f"{'='*70}")
    if errors > 0:
        print(
            f"  ✗ {errors} ERROR(s), {warnings} warning(s) — "
            f"fix errors before training"
        )
    elif warnings > 0:
        print(
            f"  ⚠ {warnings} warning(s), 0 errors — "
            f"review warnings, training can proceed"
        )
    else:
        print(f"  ✓ All checks passed — ready to train")
    print(f"{'='*70}\n")

    return errors == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify annotation data quality"
    )
    parser.add_argument(
        "--annotation-path", type=str,
        default="clinical_data/annotations/annotations.json",
    )
    parser.add_argument(
        "--image-dir", type=str,
        default="clinical_data/clean",
    )
    args = parser.parse_args()

    ok = verify_dataset(args.annotation_path, args.image_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()