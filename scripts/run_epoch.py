#!/usr/bin/env python3
"""
Run a single training + validation epoch without albumentations.

Usage:
    python scripts/run_epoch.py \
        --annotation-path clinical_data/annotations/annotations.json \
        --image-dir clinical_data/clean \
        --mask-dir clinical_data/annotations/masks

This script monkeypatches the augmentation factory functions to
return `None` so the dataset uses the simple fallback transforms.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.utils.config import get_config, set_config
from pupil_tracking.utils.logger import AuditLogger, set_logger
from pupil_tracking.ml.trainer import Trainer
import pupil_tracking.ml.dataset as dataset_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--copies-per-image", type=int, default=5,
                        help="Augmented copies per image for this quick run")
    args = parser.parse_args()

    # disable albumentations-based pipelines (use dataset fallback)
    dataset_module._get_train_augmentation = lambda input_size: None
    dataset_module._get_val_augmentation = lambda input_size: None

    cfg = get_config()
    cfg.training.epochs = 1
    cfg.training.augmentations_per_image = args.copies_per_image
    cfg.training.batch_size = cfg.training.batch_size or 4
    set_config(cfg)

    logger = AuditLogger(log_dir=cfg.paths.log_dir, session_id="run_epoch")
    set_logger(logger)

    trainer = Trainer(
        config=cfg,
        annotation_path=args.annotation_path,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )

    # run one training epoch and validation
    train_metrics = trainer._train_epoch(1)
    val_metrics = trainer._val_epoch(1)

    logger.info("Single epoch complete")
    logger.info("  train: %s", train_metrics)
    logger.info("  val:   %s", val_metrics)


if __name__ == "__main__":
    main()
