"""
Train production EyeSegmentationModel on data from annotate_live_video.py.

Bridges the annotation format gap between the annotation tool
and the production training pipeline.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

from pupil_tracking.utils.logger import get_logger


def convert_annotations(
    src_path: str,
    dst_path: str,
    image_dir: str,
) -> int:
    """Convert annotate_live_video.py format to project training format.

    Source format (annotate_live_video.py)::

        {
          "frame_000044.jpg": {
            "pupil": {
              "cx": 1537, "cy": 564,
              "semi_major": 55, "semi_minor": 49,
              "angle_deg": 12.5
            },
            "limbus": {
              "cx": 1506, "cy": 570,
              "semi_major": 293, "semi_minor": 289,
              "angle_deg": 45.0
            },
            "timestamp_sec": 0.733,
            "frame_index": 44,
            "annotated_at": "2026-03-06T..."
          }
        }

    Target format (production pipeline)::

        {
          "frame_000044.jpg": {
            "image_path": "images/frame_000044.jpg",
            "image_width": 1920,
            "image_height": 1080,
            "annotations": {
              "PUPIL": {
                "class_id": 1,
                "center_x": 1537, "center_y": 564,
                "semi_major": 55, "semi_minor": 49,
                "angle_deg": 12.5,
                "boundary_points": []
              },
              "LIMBUS": {
                "class_id": 2,
                "center_x": 1506, "center_y": 570,
                "semi_major": 293, "semi_minor": 289,
                "angle_deg": 45.0,
                "boundary_points": []
              }
            }
          }
        }
    """
    logger = get_logger()

    with open(src_path, "r") as f:
        raw = json.load(f)

    import cv2

    converted: Dict[str, Any] = {}
    count = 0

    for filename, entry in raw.items():
        # Read image to get dimensions
        img_path = Path(image_dir) / filename
        if not img_path.exists():
            logger.warning(
                "Image not found: %s — skipping", img_path
            )
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(
                "Cannot read image: %s — skipping", img_path
            )
            continue

        h, w = img.shape[:2]

        out_entry: Dict[str, Any] = {
            "image_path": str(img_path),
            "image_width": w,
            "image_height": h,
            "annotations": {},
        }

        # Convert pupil
        pupil = entry.get("pupil")
        if pupil and "cx" in pupil:
            out_entry["annotations"]["PUPIL"] = {
                "class_id": 1,
                "center_x": float(pupil["cx"]),
                "center_y": float(pupil["cy"]),
                "semi_major": float(pupil["semi_major"]),
                "semi_minor": float(pupil["semi_minor"]),
                "angle_deg": float(pupil.get("angle_deg", 0)),
                "boundary_points": [],
            }

        # Convert limbus
        limbus = entry.get("limbus")
        if limbus and "cx" in limbus:
            out_entry["annotations"]["LIMBUS"] = {
                "class_id": 2,
                "center_x": float(limbus["cx"]),
                "center_y": float(limbus["cy"]),
                "semi_major": float(limbus["semi_major"]),
                "semi_minor": float(limbus["semi_minor"]),
                "angle_deg": float(limbus.get("angle_deg", 0)),
                "boundary_points": [],
            }

        if out_entry["annotations"]:
            converted[filename] = out_entry
            count += 1
            logger.info(
                "  Converted: %s (%dx%d) — %s",
                filename, w, h,
                ", ".join(out_entry["annotations"].keys()),
            )
        else:
            logger.warning(
                "  Skipped %s — no pupil or limbus", filename
            )

    # Write converted annotations
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        json.dump(converted, f, indent=2)

    logger.info(
        "Converted %d/%d annotations: %s -> %s",
        count, len(raw), src_path, dst_path,
    )
    return count


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Train production model on annotation tool data"
    )
    p.add_argument(
        "--annotations",
        default="clinical_data/annotations/annotations.json",
        help="Path to annotate_live_video.py annotations",
    )
    p.add_argument(
        "--image-dir",
        default="clinical_data/training_data/images",
        help="Directory containing annotated frame images",
    )
    p.add_argument(
        "--mask-dir",
        default="clinical_data/training_data/masks",
        help="Directory containing generated masks",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs (default: 200)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2, use small for few images)",
    )
    p.add_argument(
        "--copies-per-image",
        type=int,
        default=50,
        help="Augmented copies per image per epoch (default: 50)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    p.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained model",
    )
    p.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip annotation conversion (if already done)",
    )
    args = p.parse_args()

    logger = get_logger()

    # ── Step 1: Convert annotations ──
    converted_path = str(
        Path(args.annotations).parent / "annotations_production.json"
    )

    if not args.skip_convert:
        print("=" * 60)
        print("  Step 1: Converting annotation format")
        print("=" * 60)

        count = convert_annotations(
            src_path=args.annotations,
            dst_path=converted_path,
            image_dir=args.image_dir,
        )

        if count < 2:
            print(
                f"\nERROR: Need at least 2 converted annotations, "
                f"got {count}."
            )
            print("Add more annotations with annotate_live_video.py")
            sys.exit(1)

        print(f"\n  Converted {count} annotations")
        print(f"  Saved to: {converted_path}")
    else:
        if not Path(converted_path).exists():
            print(
                f"ERROR: --skip-convert but {converted_path} "
                f"does not exist"
            )
            sys.exit(1)
        print(f"  Using existing: {converted_path}")

    # ── Step 2: Configure and train ──
    print()
    print("=" * 60)
    print("  Step 2: Training EyeSegmentationModel (ResNet34 + UNet)")
    print("=" * 60)

    from pupil_tracking.utils.config import get_config

    cfg = get_config()

    # Override config for our data
    cfg.training.epochs = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.learning_rate
    cfg.training.augmentations_per_image = args.copies_per_image
    cfg.training.early_stopping_patience = max(30, args.epochs // 4)
    cfg.training.val_ratio = 0.15  # Keep more for training with small dataset
    cfg.paths.model_dir = args.output_dir

    print(f"  Annotations: {converted_path}")
    print(f"  Images:      {args.image_dir}")
    print(f"  Masks:       {args.mask_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Copies/img:  {args.copies_per_image}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  Output:      {args.output_dir}")
    print()

    from pupil_tracking.ml.trainer import Trainer

    try:
        trainer = Trainer(
            config=cfg,
            annotation_path=converted_path,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
        )
    except Exception as e:
        print(f"\nERROR creating trainer: {e}")
        print("\nDebug info:")
        print(f"  Annotation file exists: {Path(converted_path).exists()}")
        print(f"  Image dir exists: {Path(args.image_dir).exists()}")
        if Path(args.image_dir).exists():
            imgs = list(Path(args.image_dir).glob("*"))
            print(f"  Images found: {len(imgs)}")
            for img in imgs[:5]:
                print(f"    {img.name}")
        print(f"  Mask dir exists: {Path(args.mask_dir).exists()}")
        raise

    print(f"  Training set:   {len(trainer.train_ds)} samples")
    print(f"  Validation set: {len(trainer.val_ds)} samples")
    print(f"  Device:         {trainer.device}")
    print()

    result = trainer.train()

    # ── Step 3: Summary ──
    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best validation IoU: {result['best_val_iou']:.4f}")
    print(f"  Epochs trained:      {result['epochs_trained']}")
    print(f"  Model saved to:      {result['model_path']}")
    print()
    print("  Now restart the GUI:")
    print("    python launch_gui.py")
    print()

    # Verify the saved model loads correctly
    print("  Verifying model loads correctly...")
    try:
        from pupil_tracking.ml.architecture import EyeSegmentationModel
        model = EyeSegmentationModel.load(
            result["model_path"], device="cpu"
        )
        print(f"  ✓ Model loaded successfully")
        print(f"    Classes: {model.num_classes}")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Model verification failed: {e}")


if __name__ == "__main__":
    main()