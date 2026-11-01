#!/usr/bin/env python3
"""
Generate and verify segmentation masks from annotations.

This script:
  1. Loads annotations.json
  2. Generates multi-class masks for each image
  3. Saves masks as PNG files
  4. Creates visualisation overlays for verification
  5. Reports statistics and data quality issues

Usage:
    python scripts/generate_masks.py
    python scripts/generate_masks.py --annotation-path path/to/annotations.json
    python scripts/generate_masks.py --use-boundary-points
    python scripts/generate_masks.py --verify-only

Output:
    clinical_data/annotations/masks/       - mask PNGs
    clinical_data/annotations/masks/verify/ - overlay images for review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.ml.dataset import (
    load_annotations,
    generate_mask_from_annotation,
    get_annotation_stats,
)


# colour map for visualisation
_CLASS_COLORS = {
    0: (0, 0, 0),          # background — black
    1: (0, 255, 0),        # pupil — green
    2: (255, 100, 0),      # iris — blue-ish
    3: (0, 0, 255),        # ring — red (if used)
}


def generate_all_masks(
    annotation_path: str,
    image_dir: str,
    output_dir: str,
    use_boundary_points: bool = False,
    num_classes: int = 3,
) -> None:
    """Generate masks for all annotated images."""
    image_ids, annotations = load_annotations(annotation_path)

    mask_dir = Path(output_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    verify_dir = mask_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MASK GENERATION")
    print(f"{'='*60}")
    print(f"Annotation file: {annotation_path}")
    print(f"Image directory:  {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Use boundary pts: {use_boundary_points}")
    print(f"Num classes:      {num_classes}")
    print(f"Images found:     {len(image_ids)}")
    print(f"{'='*60}\n")

    # stats
    stats = get_annotation_stats(annotations)
    print(f"With pupil:  {stats['with_pupil']}/{stats['total_images']}")
    print(f"With limbus: {stats['with_limbus']}/{stats['total_images']}")
    print(f"With ring:   {stats['with_ring']}/{stats['total_images']}")
    print(f"With both:   {stats['with_both']}/{stats['total_images']}")
    if stats["issues"]:
        print(f"\n⚠ Data quality issues:")
        for issue in stats["issues"]:
            print(f"  - {issue}")
    print()

    img_dir = Path(image_dir)
    success = 0
    failed = 0

    for img_id in image_ids:
        ann = annotations[img_id]

        # find image
        img_path = _find_image(img_dir, img_id, ann)
        if img_path is None:
            print(f"  ✗ {img_id}: image not found in {img_dir}")
            failed += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  ✗ {img_id}: failed to read {img_path}")
            failed += 1
            continue

        h, w = image.shape[:2]

        # generate mask
        mask = generate_mask_from_annotation(
            (h, w), ann, num_classes,
            use_boundary_points=use_boundary_points,
        )

        # save mask
        mask_path = mask_dir / f"{img_id}.png"
        cv2.imwrite(str(mask_path), mask)

        # compute stats
        pupil_pixels = int(np.sum(mask == 1))
        iris_pixels = int(np.sum(mask == 2))
        total_pixels = h * w
        pupil_pct = 100.0 * pupil_pixels / total_pixels
        iris_pct = 100.0 * iris_pixels / total_pixels

        # create verification overlay
        overlay = _create_overlay(image, mask, ann)
        verify_path = verify_dir / f"{img_id}_verify.png"
        cv2.imwrite(str(verify_path), overlay)

        print(
            f"  ✓ {img_id}: {w}×{h}  "
            f"pupil={pupil_pct:.1f}%  iris={iris_pct:.1f}%  "
            f"→ {mask_path.name}"
        )
        success += 1

    print(f"\n{'='*60}")
    print(f"Generated {success} masks, {failed} failed")
    print(f"Masks saved to:         {mask_dir}")
    print(f"Verification images to: {verify_dir}")
    print(f"{'='*60}")
    print(f"\n★ REVIEW the images in {verify_dir}/ to verify masks")
    print(f"  are correct before training.\n")


def _find_image(
    img_dir: Path, img_id: str, ann: dict
) -> Path | None:
    """Find image file by ID, trying multiple locations and extensions."""
    extensions = (".jpeg", ".jpg", ".png", ".bmp", ".tiff")

    # try image_dir + stem + extension
    for ext in extensions:
        p = img_dir / f"{img_id}{ext}"
        if p.exists():
            return p

    # try image_filename from annotation
    fname = ann.get("image_filename", "")
    if fname:
        p = img_dir / fname
        if p.exists():
            return p

    # try image_path from annotation (relative or absolute)
    img_path_str = ann.get("image_path", "")
    if img_path_str:
        p = Path(img_path_str)
        if p.exists():
            return p
        # try just the filename
        p = img_dir / p.name
        if p.exists():
            return p

    return None


def _create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    ann: dict,
) -> np.ndarray:
    """Create a verification overlay with:
    - Original image
    - Colour mask overlay (semi-transparent)
    - Ellipse outlines from annotation parameters
    - Centre markers
    - Text labels
    """
    h, w = image.shape[:2]

    # create colour mask
    color_mask = np.zeros_like(image)
    for class_id, color in _CLASS_COLORS.items():
        if class_id == 0:
            continue
        color_mask[mask == class_id] = color

    # blend
    alpha = 0.35
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # draw ellipse outlines from annotation parameters
    # pupil — green outline
    if "pupil_center" in ann and "pupil_axes" in ann:
        pc = ann["pupil_center"]
        pa = ann["pupil_axes"]
        pang = ann.get("pupil_angle", 0)
        center = (int(round(pc[0])), int(round(pc[1])))
        axes = (int(round(pa[0])), int(round(pa[1])))
        cv2.ellipse(
            overlay, center, axes, int(round(pang)),
            0, 360, (0, 255, 0), 2,
        )
        cv2.circle(overlay, center, 4, (0, 255, 0), -1)
        cv2.putText(
            overlay,
            f"P ({pc[0]:.0f},{pc[1]:.0f})",
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    # limbus — blue outline
    if "limbus_center" in ann and "limbus_axes" in ann:
        lc = ann["limbus_center"]
        la = ann["limbus_axes"]
        lang = ann.get("limbus_angle", 0)
        center = (int(round(lc[0])), int(round(lc[1])))
        axes = (int(round(la[0])), int(round(la[1])))
        cv2.ellipse(
            overlay, center, axes, int(round(lang)),
            0, 360, (255, 100, 0), 2,
        )
        cv2.circle(overlay, center, 4, (255, 100, 0), -1)
        cv2.putText(
            overlay,
            f"L ({lc[0]:.0f},{lc[1]:.0f})",
            (center[0] + 8, center[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1,
        )

    # ring — red outline (if present)
    if ann.get("has_suction_ring") and "ring_axes" in ann:
        rc = ann["ring_center"]
        ra = ann["ring_axes"]
        rang = ann.get("ring_angle", 0)
        center = (int(round(rc[0])), int(round(rc[1])))
        axes = (int(round(ra[0])), int(round(ra[1])))
        cv2.ellipse(
            overlay, center, axes, int(round(rang)),
            0, 360, (0, 0, 255), 1,
        )

    # offset line (pupil centre → limbus centre)
    if "pupil_center" in ann and "limbus_center" in ann:
        pc = ann["pupil_center"]
        lc = ann["limbus_center"]
        p1 = (int(round(pc[0])), int(round(pc[1])))
        p2 = (int(round(lc[0])), int(round(lc[1])))
        cv2.line(overlay, p1, p2, (0, 255, 255), 2)

        # offset text
        dx = pc[0] - lc[0]
        dy = pc[1] - lc[1]
        offset = (dx ** 2 + dy ** 2) ** 0.5
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(
            overlay,
            f"offset={offset:.1f}px",
            (mid[0] + 5, mid[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
        )

    # legend
    y = 25
    for label, color in [
        ("Pupil (class 1)", (0, 255, 0)),
        ("Iris (class 2)", (255, 100, 0)),
        ("Ring", (0, 0, 255)),
        ("Offset", (0, 255, 255)),
    ]:
        cv2.rectangle(
            overlay, (10, y - 12), (25, y + 2), color, -1
        )
        cv2.putText(
            overlay, label, (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        y += 20

    # image id
    img_id = ann.get("image_id", "unknown")
    cv2.putText(
        overlay,
        f"{img_id} ({w}x{h})",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    return overlay


def verify_masks(
    annotation_path: str,
    image_dir: str,
    mask_dir: str,
) -> None:
    """Verify that existing masks match annotations."""
    image_ids, annotations = load_annotations(annotation_path)
    mask_path = Path(mask_dir)

    print(f"\nVerifying masks in {mask_dir}...")
    issues = []

    for img_id in image_ids:
        mp = mask_path / f"{img_id}.png"
        if not mp.exists():
            issues.append(f"{img_id}: mask file not found")
            continue

        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            issues.append(f"{img_id}: failed to read mask")
            continue

        ann = annotations[img_id]
        expected_h = ann.get("image_height", 0)
        expected_w = ann.get("image_width", 0)

        if expected_h > 0 and expected_w > 0:
            if mask.shape != (expected_h, expected_w):
                issues.append(
                    f"{img_id}: mask size {mask.shape} ≠ "
                    f"expected ({expected_h}, {expected_w})"
                )

        classes_present = set(np.unique(mask))
        has_pupil = 1 in classes_present
        has_iris = 2 in classes_present
        has_pupil_ann = "pupil_center" in ann
        has_limbus_ann = "limbus_center" in ann

        if has_pupil_ann and not has_pupil:
            issues.append(f"{img_id}: pupil annotated but not in mask")
        if has_limbus_ann and not has_iris:
            issues.append(f"{img_id}: limbus annotated but iris not in mask")

        # check centroid matches annotation centre
        if has_pupil:
            ys, xs = np.where(mask == 1)
            pred_cx, pred_cy = float(np.mean(xs)), float(np.mean(ys))
            ann_cx, ann_cy = ann["pupil_center"]
            dist = ((pred_cx - ann_cx)**2 + (pred_cy - ann_cy)**2)**0.5
            if dist > 20:
                issues.append(
                    f"{img_id}: pupil centroid off by {dist:.1f}px"
                )

        print(f"  ✓ {img_id}: classes={sorted(classes_present)}")

    if issues:
        print(f"\n⚠ {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ All {len(image_ids)} masks verified OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and verify segmentation masks"
    )
    parser.add_argument(
        "--annotation-path", type=str,
        default="clinical_data/annotations/annotations.json",
        help="Path to annotations.json",
    )
    parser.add_argument(
        "--image-dir", type=str,
        default="clinical_data/clean",
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="clinical_data/annotations/masks",
        help="Output directory for masks",
    )
    parser.add_argument(
        "--use-boundary-points", action="store_true",
        help="Use boundary-point polygons instead of ellipses "
             "(only when ≥20 points available)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing masks, don't generate",
    )
    parser.add_argument(
        "--num-classes", type=int, default=3,
        help="Number of segmentation classes (default: 3)",
    )

    args = parser.parse_args()

    if args.verify_only:
        verify_masks(
            args.annotation_path,
            args.image_dir,
            args.output_dir,
        )
    else:
        generate_all_masks(
            annotation_path=args.annotation_path,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            use_boundary_points=args.use_boundary_points,
            num_classes=args.num_classes,
        )


if __name__ == "__main__":
    main()