#!/usr/bin/env python3
"""
Training entry point for eye segmentation models.

Supports both 3-class (legacy: background/pupil/iris) and 4-class
(ring-aware: background/pupil/iris/suction_ring) training.

Usage:
    # Standard 3-class training
    python scripts/train_model.py

    # 4-class ring-aware training
    python scripts/train_model.py --num-classes 4

    # Full customisation
    python scripts/train_model.py \\
        --epochs 300 \\
        --batch-size 8 \\
        --lr 0.0005 \\
        --num-classes 4 \\
        --annotation-path clinical_data/annotations/annotations.json \\
        --image-dir clinical_data/training_data/images \\
        --mask-dir clinical_data/training_data/masks \\
        --ring-labels clinical_data/ring_labels.json \\
        --device cuda

    # With focal loss for better ring-class handling
    python scripts/train_model.py \\
        --num-classes 4 \\
        --loss-type composite \\
        --use-focal

Notes:
    - When ``--num-classes 4`` is specified, the model learns to
      segment suction rings as a separate class.  Images without
      rings simply have no class-3 pixels in their masks, which
      the weighted loss handles correctly.

    - The ``--ring-labels`` flag is optional.  When provided, it
      supplements ring-presence flags from the main annotation file
      (useful when annotations were created before ring labelling
      was added).

    - Run ``python check_training_data.py`` before training to
      validate your dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.utils.config import get_config, set_config
from pupil_tracking.utils.logger import AuditLogger, set_logger
from pupil_tracking.ml.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train eye segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic 3-class training\n"
            "  python scripts/train_model.py\n\n"
            "  # 4-class ring-aware training\n"
            "  python scripts/train_model.py --num-classes 4\n\n"
            "  # Full customisation\n"
            "  python scripts/train_model.py \\\n"
            "      --epochs 300 --batch-size 8 --lr 0.0005 \\\n"
            "      --num-classes 4 --device cuda\n"
        ),
    )

    # ── Training hyperparameters ──────────────────────────────
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config, typically 200)",
    )
    training_group.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: from config; reduce for low-memory GPUs)",
    )
    training_group.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: from config, typically 1e-4)",
    )
    training_group.add_argument(
        "--input-size", type=int, default=None,
        help="Model input resolution in pixels (default: 512)",
    )
    training_group.add_argument(
        "--device", type=str, default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device (default: auto-detect)",
    )
    training_group.add_argument(
        "--copies-per-image", type=int, default=None,
        help="Augmented copies per source image per epoch (default: from config)",
    )

    # ── Model architecture ────────────────────────────────────
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--num-classes", type=int, default=None,
        choices=[3, 4],
        help=(
            "Number of segmentation classes: "
            "3 (background/pupil/iris) or "
            "4 (background/pupil/iris/suction_ring)"
        ),
    )
    model_group.add_argument(
        "--encoder", type=str, default=None,
        help="Backbone encoder name (default: resnet34)",
    )

    # ── Loss function ─────────────────────────────────────────
    loss_group = parser.add_argument_group("Loss Function")
    loss_group.add_argument(
        "--loss-type", type=str, default=None,
        choices=["composite", "ce_dice", "focal_dice", "ce"],
        help=(
            "Loss function type (default: composite). "
            "'composite' = CE + Dice + Boundary; "
            "'ce_dice' = CE + Dice; "
            "'focal_dice' = Focal + Dice; "
            "'ce' = CrossEntropy only"
        ),
    )
    loss_group.add_argument(
        "--use-focal", action="store_true",
        help=(
            "Replace CE with focal loss in composite mode "
            "(better for rare ring class)"
        ),
    )
    loss_group.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Focal loss gamma parameter (default: 2.0)",
    )
    loss_group.add_argument(
        "--class-weights", type=float, nargs="+", default=None,
        help=(
            "Per-class loss weights. "
            "3-class default: 0.3 1.0 1.0; "
            "4-class default: 0.3 1.0 1.0 1.2"
        ),
    )

    # ── Data paths ────────────────────────────────────────────
    data_group = parser.add_argument_group("Data Paths")
    data_group.add_argument(
        "--annotation-path", type=str, default=None,
        help="Path to annotations.json",
    )
    data_group.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory containing training images",
    )
    data_group.add_argument(
        "--mask-dir", type=str, default=None,
        help="Directory containing mask PNGs (optional — generated from annotations if absent)",
    )
    data_group.add_argument(
        "--ring-labels", type=str, default=None,
        help=(
            "Path to ring_labels.json (created by annotate_ring_data.py). "
            "Supplements ring-presence flags in the main annotation file."
        ),
    )

    # ── Output ────────────────────────────────────────────────
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save model checkpoints (default: models/)",
    )
    output_group.add_argument(
        "--model-name", type=str, default=None,
        help="Checkpoint filename (default: best_model.pth)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Configure ─────────────────────────────────────────────
    cfg = get_config()

    # Training hyperparameters
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.copies_per_image is not None:
        cfg.training.augmentations_per_image = args.copies_per_image

    # Model architecture
    if args.input_size is not None:
        cfg.model.input_size = args.input_size
    if args.device is not None:
        cfg.model.device = args.device
    if args.num_classes is not None:
        cfg.model.num_classes = args.num_classes
    if args.encoder is not None:
        cfg.model.encoder = args.encoder

    # Output paths
    if args.save_dir is not None:
        cfg.paths.model_dir = args.save_dir
    if args.model_name is not None:
        cfg.model.model_path = str(
            Path(cfg.paths.model_dir) / args.model_name
        )

    set_config(cfg)

    # ── Logger ────────────────────────────────────────────────
    logger = AuditLogger(log_dir=cfg.paths.log_dir, session_id="training")
    set_logger(logger)

    # ── Print configuration ───────────────────────────────────
    num_classes = cfg.model.num_classes
    class_names = {
        3: "background, pupil, iris",
        4: "background, pupil, iris, suction_ring",
    }

    logger.info("=" * 64)
    logger.info("  EYE SEGMENTATION MODEL TRAINING")
    logger.info("=" * 64)
    logger.info("  Classes:        %d (%s)", num_classes, class_names.get(num_classes, "custom"))
    logger.info("  Encoder:        %s", cfg.model.encoder)
    logger.info("  Input size:     %d", cfg.model.input_size)
    logger.info("  Epochs:         %d", cfg.training.epochs)
    logger.info("  Batch size:     %d", cfg.training.batch_size)
    logger.info("  Learning rate:  %s", cfg.training.learning_rate)
    logger.info("  Device:         %s", cfg.model.device)
    logger.info("  Augmentations:  %d per image", cfg.training.augmentations_per_image)

    if args.ring_labels:
        logger.info("  Ring labels:    %s", args.ring_labels)
    if args.loss_type:
        logger.info("  Loss type:      %s", args.loss_type)
    if args.use_focal:
        logger.info("  Focal loss:     enabled (gamma=%.1f)", args.focal_gamma)
    if args.class_weights:
        logger.info("  Class weights:  %s", args.class_weights)

    logger.info("=" * 64)

    # ── Validate ring-aware configuration ─────────────────────
    if num_classes == 4:
        logger.info("")
        logger.info("  Ring-aware (4-class) training enabled.")
        logger.info("  Class 3 = suction_ring")
        logger.info("")

        if args.ring_labels is not None:
            ring_labels_path = Path(args.ring_labels)
            if not ring_labels_path.exists():
                logger.warning(
                    "  ⚠ Ring labels file not found: %s",
                    ring_labels_path,
                )
                logger.warning(
                    "    Ring presence will be inferred from annotations only."
                )
        else:
            logger.info(
                "  No --ring-labels provided; ring presence will be "
                "inferred from RING entries in annotations.json."
            )

    # ── Build loss function kwargs ────────────────────────────
    loss_kwargs = {}
    if args.loss_type is not None:
        loss_kwargs["loss_type"] = args.loss_type
    if args.use_focal:
        loss_kwargs["use_focal"] = True
        loss_kwargs["focal_gamma"] = args.focal_gamma
    if args.class_weights is not None:
        if len(args.class_weights) != num_classes:
            logger.error(
                "  ✗ --class-weights has %d values but --num-classes is %d",
                len(args.class_weights), num_classes,
            )
            sys.exit(1)
        loss_kwargs["class_weights"] = args.class_weights

    # ── Train ─────────────────────────────────────────────────
    trainer = Trainer(
        config=cfg,
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        ring_labels_path=args.ring_labels,
        loss_kwargs=loss_kwargs if loss_kwargs else None,
    )

    results = trainer.train()

    # ── Report ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 64)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 64)
    logger.info("  Best IoU:       %.4f", results.get("best_val_iou", 0.0))
    logger.info("  Best Dice:      %.4f", results.get("best_val_dice", 0.0))
    logger.info("  Epochs trained: %d", results.get("epochs_trained", 0))
    logger.info("  Model saved:    %s", results.get("model_path", "N/A"))
    logger.info("  Classes:        %d", num_classes)

    # Per-class metrics if available
    per_class = results.get("per_class_iou", {})
    if per_class:
        logger.info("  Per-class IoU:")
        class_label_map = {
            0: "background", 1: "pupil", 2: "iris", 3: "suction_ring",
        }
        for cls_id, iou_val in per_class.items():
            name = class_label_map.get(int(cls_id), f"class_{cls_id}")
            logger.info("    %s: %.4f", name, iou_val)

    logger.info("=" * 64)

    # Quality assessment
    best_iou = results.get("best_val_iou", 0.0)
    if best_iou >= 0.90:
        logger.info("  ✅ Excellent model — ready for production use.")
    elif best_iou >= 0.80:
        logger.info("  ✅ Good model — should work well for most images.")
    elif best_iou >= 0.70:
        logger.info("  ⚠️  Fair model — consider more data or tuning.")
    else:
        logger.info("  ❌ Model needs improvement.")
        logger.info("     Suggestions:")
        logger.info("     • Add more annotated images")
        logger.info("     • Try lower learning rate: --lr 0.00005")
        logger.info("     • Train longer: --epochs 500")
        if num_classes == 4:
            logger.info(
                "     • Ensure ring annotations exist in training data"
            )
            logger.info(
                "     • Try --use-focal for better ring class handling"
            )

    logger.info("")

    # Next steps
    if num_classes == 4:
        logger.info("Next steps:")
        logger.info(
            "  1. Evaluate: python scripts/verify_data.py "
            "--model-path %s", results.get("model_path", "models/best_model.pth"),
        )
        logger.info(
            "  2. Test:     python launch_gui.py image -i <image> "
            "--ring-mode auto"
        )
        logger.info(
            "  3. Export:   python scripts/export_onnx.py "
            "--model %s", results.get("model_path", "models/best_model.pth"),
        )
    else:
        logger.info("Next steps:")
        logger.info(
            "  1. Evaluate: python scripts/verify_data.py "
            "--model-path %s", results.get("model_path", "models/best_model.pth"),
        )
        logger.info("  2. Test:     python launch_gui.py image -i <image>")
        logger.info(
            "  3. Upgrade:  Re-train with --num-classes 4 for ring support"
        )

    logger.info("")
    logger.close()


if __name__ == "__main__":
    main()