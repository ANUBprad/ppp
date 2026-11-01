#!/usr/bin/env python3
"""
train_ring_classifier.py — Train the binary ring presence classifier.

Trains a lightweight MobileNetV2-based CNN to classify eye images as
either "ring present" (docked, class 1) or "ring absent" (pre-docked,
class 0).  The trained model is used as the first routing decision in
the adaptive detection pipeline.

Prerequisites
-------------
1. A directory of eye images (mixed docked and pre-docked).
2. A ``ring_labels.json`` file created by ``scripts/annotate_ring_data.py``.

Label file format::

    {
      "image_001.jpg": {"ring_present": true},
      "image_002.jpg": {"ring_present": false},
      ...
    }

Usage
-----
::

    # Minimal — uses defaults
    python scripts/train_ring_classifier.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json

    # Full customisation
    python scripts/train_ring_classifier.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json \\
        --epochs 80 \\
        --batch-size 16 \\
        --lr 0.0003 \\
        --val-split 0.2 \\
        --device cuda \\
        --save-path models/ring_classifier.pth \\
        --balance

Output
------
* ``models/ring_classifier.pth`` — best model weights (by validation loss).
* ``models/ring_classifier.meta.json`` — training metadata (epochs,
  best val loss / accuracy).
* Console log with per-epoch metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

# ── Ensure project root is importable ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pupil_tracking.ml.ring_classifier import (
    RingClassifierNet,
    RingClassificationDataset,
    RingClassifierTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train binary ring-present / ring-absent classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing eye images",
    )
    p.add_argument(
        "--labels", type=str, required=True,
        help="Path to ring_labels.json (created by annotate_ring_data.py)",
    )

    # Training hyper-parameters
    p.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum training epochs (default: 50)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32; reduce for low-memory GPUs)",
    )
    p.add_argument(
        "--lr", type=float, default=0.0005,
        help="Initial learning rate (default: 0.0005)",
    )
    p.add_argument(
        "--val-split", type=float, default=0.20,
        help="Fraction of data for validation (default: 0.20)",
    )
    p.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience in epochs (default: 10)",
    )

    # Balancing
    p.add_argument(
        "--balance", action="store_true",
        help="Use weighted random sampling to balance ring/no-ring classes",
    )

    # Device
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )

    # Output
    p.add_argument(
        "--save-path", type=str, default="models/ring_classifier.pth",
        help="Where to save the best model checkpoint",
    )

    # Workers
    p.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker processes (default: 2)",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Device resolution
# ═══════════════════════════════════════════════════════════════════════

def resolve_device(name: str) -> torch.device:
    """Resolve a device name string to a ``torch.device``."""
    if name == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(name)

    logger.info("Using device: %s", dev)
    if dev.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "  Memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_mem / 1e9,
        )
    return dev


# ═══════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════

def validate_inputs(args: argparse.Namespace, labels: Dict[str, dict]) -> None:
    """Check inputs are sane before starting training."""
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        logger.error("Image directory does not exist: %s", image_dir)
        sys.exit(1)

    if not labels:
        logger.error("Label file is empty — run annotate_ring_data.py first")
        sys.exit(1)

    ring_count = sum(1 for v in labels.values() if v.get("ring_present", False))
    no_ring_count = len(labels) - ring_count

    logger.info("Label statistics:")
    logger.info("  Total:         %d", len(labels))
    logger.info("  Ring present:  %d  (%.1f%%)", ring_count, ring_count / len(labels) * 100)
    logger.info("  Ring absent:   %d  (%.1f%%)", no_ring_count, no_ring_count / len(labels) * 100)

    if ring_count == 0:
        logger.error("No 'ring_present' labels found — need both classes!")
        sys.exit(1)
    if no_ring_count == 0:
        logger.error("No 'ring_absent' labels found — need both classes!")
        sys.exit(1)

    if ring_count < 10 or no_ring_count < 10:
        logger.warning(
            "Very few samples in one class (ring=%d, no-ring=%d). "
            "Consider collecting more data for better results.",
            ring_count, no_ring_count,
        )

    if abs(ring_count - no_ring_count) / max(ring_count, no_ring_count) > 0.5:
        logger.warning(
            "Classes are imbalanced (ring=%d, no-ring=%d). "
            "Consider using --balance flag.",
            ring_count, no_ring_count,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Load labels ───────────────────────────────────────────────
    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error("Label file not found: %s", labels_path)
        logger.error("Run: python scripts/annotate_ring_data.py --image-dir %s --output %s",
                      args.image_dir, labels_path)
        sys.exit(1)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # ── Validate ──────────────────────────────────────────────────
    validate_inputs(args, labels)

    # ── Device ────────────────────────────────────────────────────
    device = resolve_device(args.device)

    # ── Create full dataset (with augmentation) ───────────────────
    full_dataset = RingClassificationDataset(
        image_dir=args.image_dir,
        labels=labels,
        augment=True,
        size=224,
    )

    if len(full_dataset) == 0:
        logger.error(
            "Dataset is empty — no images matched the labels. "
            "Check that --image-dir contains the files listed in --labels."
        )
        sys.exit(1)

    # ── Train / validation split ──────────────────────────────────
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )

    logger.info("Split: train=%d  val=%d", train_size, val_size)

    # ── Validation set without augmentation ───────────────────────
    val_dataset_clean = RingClassificationDataset(
        image_dir=args.image_dir,
        labels=labels,
        augment=False,
        size=224,
    )
    val_subset_clean = torch.utils.data.Subset(
        val_dataset_clean, val_subset.indices,
    )

    # ── DataLoaders ───────────────────────────────────────────────
    train_sampler = None
    train_shuffle = True

    if args.balance:
        # Weighted random sampler to balance classes
        all_weights = full_dataset.get_sampler_weights()
        train_weights = torch.tensor(
            [all_weights[i] for i in train_subset.indices],
            dtype=torch.float64,
        )
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        train_shuffle = False  # sampler handles ordering
        logger.info("Using weighted random sampling for class balance")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_subset_clean,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Create model ──────────────────────────────────────────────
    model = RingClassifierNet(pretrained=True)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: RingClassifierNet — %.2fM params (%.2fM trainable)",
        param_count / 1e6, trainable_count / 1e6,
    )

    # ── Train ─────────────────────────────────────────────────────
    save_dir = Path(args.save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = RingClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        save_path=args.save_path,
        patience=args.patience,
    )

    print()
    print("=" * 64)
    print("  RING CLASSIFIER TRAINING")
    print("=" * 64)
    print(f"  Images:      {len(full_dataset)}")
    print(f"  Train:       {train_size}")
    print(f"  Validation:  {val_size}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Device:      {device}")
    print(f"  Save path:   {args.save_path}")
    print(f"  Balanced:    {args.balance}")
    print("=" * 64)
    print()

    history = trainer.train()

    # ── Final report ──────────────────────────────────────────────
    best_val_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
    best_val_loss = min(history["val_loss"]) if history["val_loss"] else float("inf")
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    epochs_trained = len(history["train_loss"])

    print()
    print("=" * 64)
    print("  TRAINING COMPLETE")
    print("=" * 64)
    print(f"  Epochs trained:    {epochs_trained}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    print(f"  Best val accuracy: {best_val_acc:.3f}  ({best_val_acc*100:.1f}%)")
    print(f"  Final val accuracy:{final_val_acc:.3f}  ({final_val_acc*100:.1f}%)")
    print(f"  Model saved to:    {args.save_path}")
    print("=" * 64)
    print()

    # Quality assessment
    if best_val_acc >= 0.95:
        print("  ✅ Excellent classifier — ready for production use.")
    elif best_val_acc >= 0.90:
        print("  ✅ Good classifier — should work well for most images.")
    elif best_val_acc >= 0.80:
        print("  ⚠️  Fair classifier — consider adding more training data.")
    else:
        print("  ❌ Poor classifier — needs more data or tuning.")
        print("     Suggestions:")
        print("     • Add more labelled images (especially the minority class)")
        print("     • Use --balance flag for class balancing")
        print("     • Try lower learning rate: --lr 0.0001")
        print("     • Train for more epochs: --epochs 100")

    print()
    print("Next steps:")
    print(f"  1. Evaluate:  python scripts/evaluate_ring_detection.py "
          f"--image-dir {args.image_dir} --labels {args.labels} "
          f"--classifier {args.save_path}")
    print(f"  2. Use:       python launch_gui.py image -i <image> "
          f"--ring-classifier {args.save_path}")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()