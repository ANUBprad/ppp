"""
Training Data Diagnostic — Run BEFORE retraining
=================================================
Checks mask quality, class distribution, image-mask pairing,
and identifies problems that would corrupt training.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys


def check_training_data(
    image_dir: str,
    mask_dir: str,
    num_classes: int = 4,
):
    """Comprehensive training data check."""
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)

    print("=" * 60)
    print("TRAINING DATA DIAGNOSTIC")
    print("=" * 60)

    # ── Check directories exist ──────────────────────
    if not image_path.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return False
    if not mask_path.exists():
        print(f"❌ Mask directory not found: {mask_dir}")
        return False

    # ── Find all images ──────────────────────────────
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp',
                  '.tiff', '.tif'}
    images = sorted([
        f for f in image_path.iterdir()
        if f.suffix.lower() in image_exts
    ])
    print(f"\n📁 Images found: {len(images)}")

    # ── Find all masks ───────────────────────────────
    masks = sorted(mask_path.glob("*_mask.png"))
    print(f"📁 Masks found:  {len(masks)}")

    # ── Match pairs ──────────────────────────────────
    pairs = []
    unmatched_masks = []
    unmatched_images = []

    mask_stems = {
        m.stem.replace("_mask", ""): m for m in masks
    }

    for img in images:
        stem = img.stem
        if stem in mask_stems:
            pairs.append((img, mask_stems[stem]))
        else:
            unmatched_images.append(img)

    for stem, mask in mask_stems.items():
        found = any(
            img.stem == stem for img in images)
        if not found:
            unmatched_masks.append(mask)

    print(f"✅ Matched pairs: {len(pairs)}")

    if unmatched_images:
        print(f"⚠️  Images without masks: "
              f"{len(unmatched_images)}")
        for img in unmatched_images[:5]:
            print(f"    {img.name}")

    if unmatched_masks:
        print(f"⚠️  Masks without images: "
              f"{len(unmatched_masks)}")
        for m in unmatched_masks[:5]:
            print(f"    {m.name}")

    if len(pairs) == 0:
        print("❌ No matched pairs found!")
        print("   Mask files must be named: "
              "<image_stem>_mask.png")
        return False

    # ── Check each pair ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("PAIR ANALYSIS")
    print(f"{'─' * 60}")

    class_totals = np.zeros(num_classes, dtype=np.int64)
    issues = []
    image_sizes = []

    for img_path, mask_path_item in pairs:
        img = cv2.imread(str(img_path))
        mask = cv2.imread(
            str(mask_path_item), cv2.IMREAD_GRAYSCALE)

        if img is None:
            issues.append(
                f"Cannot read image: {img_path.name}")
            continue
        if mask is None:
            issues.append(
                f"Cannot read mask: {mask_path_item.name}")
            continue

        ih, iw = img.shape[:2]
        mh, mw = mask.shape[:2]
        image_sizes.append((iw, ih))

        # Size mismatch
        if (ih, iw) != (mh, mw):
            issues.append(
                f"{img_path.name}: image {iw}×{ih} vs "
                f"mask {mw}×{mh} — SIZE MISMATCH")

        # Class values
        unique_vals = np.unique(mask)
        invalid = [
            v for v in unique_vals if v >= num_classes]
        if invalid:
            issues.append(
                f"{mask_path_item.name}: invalid values "
                f"{invalid} (max should be "
                f"{num_classes - 1})")

        # Class distribution for this mask
        for c in range(num_classes):
            class_totals[c] += np.sum(mask == c)

        # Check mask is not all background
        fg_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        fg_ratio = fg_pixels / total_pixels

        if fg_ratio < 0.01:
            issues.append(
                f"{mask_path_item.name}: only "
                f"{fg_ratio:.1%} foreground — "
                f"nearly empty mask")

        # Check for pupil (class 1)
        pupil_pixels = np.sum(mask == 1)
        if pupil_pixels == 0:
            issues.append(
                f"{mask_path_item.name}: no pupil "
                f"(class 1) pixels")

        print(
            f"  {img_path.name:30s} {iw:4d}×{ih:4d}  "
            f"fg={fg_ratio:5.1%}  "
            f"pup={np.sum(mask == 1):6d}  "
            f"iris={np.sum(mask == 2):6d}  "
            f"ring={np.sum(mask == 3):6d}")

    # ── Class distribution ───────────────────────────
    print(f"\n{'─' * 60}")
    print("CLASS DISTRIBUTION")
    print(f"{'─' * 60}")

    total_px = class_totals.sum()
    class_names = [
        "background", "pupil", "iris", "ring"]

    for c in range(num_classes):
        pct = (class_totals[c] / total_px * 100
               if total_px > 0 else 0)
        bar = "█" * int(pct / 2)
        status = "✅" if class_totals[c] > 0 else "❌"
        print(
            f"  {status} {class_names[c]:12s}: "
            f"{class_totals[c]:10d} px  "
            f"({pct:5.1f}%)  {bar}")

    # ── Image size statistics ────────────────────────
    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        print(f"\n{'─' * 60}")
        print("IMAGE SIZES")
        print(f"{'─' * 60}")
        print(f"  Width:  min={min(widths)} "
              f"max={max(widths)} "
              f"mean={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)} "
              f"max={max(heights)} "
              f"mean={np.mean(heights):.0f}")

    # ── Issues ───────────────────────────────────────
    if issues:
        print(f"\n{'─' * 60}")
        print(f"⚠️  ISSUES FOUND: {len(issues)}")
        print(f"{'─' * 60}")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"\n✅ No issues found!")

    # ── Recommendations ──────────────────────────────
    print(f"\n{'─' * 60}")
    print("RECOMMENDATIONS")
    print(f"{'─' * 60}")

    if len(pairs) < 5:
        print("  🔴 CRITICAL: Need at least 5 "
              "annotated images")
        print("     You have: "
              f"{len(pairs)} pairs")
    elif len(pairs) < 10:
        print(f"  🟡 {len(pairs)} pairs — minimum "
              f"viable, 15+ recommended")
    elif len(pairs) < 20:
        print(f"  🟢 {len(pairs)} pairs — good "
              f"for initial training")
    else:
        print(f"  🟢 {len(pairs)} pairs — excellent")

    if class_totals[1] == 0:
        print("  🔴 CRITICAL: No pupil annotations!")
    if class_totals[2] == 0:
        print("  🔴 CRITICAL: No iris annotations!")
        print("     Limbus detection requires iris masks")

    has_issues = len(issues) > 0
    has_enough = len(pairs) >= 5
    has_pupil = class_totals[1] > 0
    has_iris = class_totals[2] > 0

    ready = (has_enough and has_pupil and has_iris
             and not any("MISMATCH" in i for i in issues))

    print(f"\n{'═' * 60}")
    if ready:
        print("✅ READY TO TRAIN")
    else:
        print("❌ NOT READY — fix issues above first")
    print(f"{'═' * 60}")

    return ready


if __name__ == "__main__":
    # Default paths — adjust to your structure
    img_dir = "./clinical_data/annotations"
    mask_dir = "./clinical_data/annotations/masks"

    # Or from command line
    if len(sys.argv) >= 3:
        img_dir = sys.argv[1]
        mask_dir = sys.argv[2]

    check_training_data(img_dir, mask_dir)