#!/usr/bin/env python3
"""
Fine-tune an existing trained model for grayscale robustness.

This script takes a pre-trained ``best_model.pth`` and fine-tunes it
with grayscale augmentation so that the model produces identical
segmentation quality on both colour (RGB) and grayscale inputs.

CRITICAL SAFETY GUARANTEE
──────────────────────────
The new checkpoint is **only saved** when BOTH of these conditions
are met:

    1. RGB validation Dice  ≥  baseline RGB Dice  (no regression)
    2. Grayscale validation Dice  ≥  ``--min-gray-dice``  (new capability)

If either condition fails, the original model is preserved unchanged.
This makes it impossible for fine-tuning to degrade existing accuracy.

HOW IT WORKS
────────────
1. Load the existing model weights (``--base-model``).
2. Run one validation pass on RGB data → record baseline Dice.
3. Create a training dataset with ``RandomGrayscaleConversion``
   augmentation (probability ``--grayscale-prob``).
4. Create TWO validation datasets:
   a. Standard RGB (no grayscale conversion)
   b. Forced grayscale (every image converted)
5. Fine-tune with a reduced learning rate for ``--epochs`` epochs.
6. After each epoch, compute Dice on BOTH validation sets.
7. Save checkpoint ONLY when both thresholds are satisfied.

USAGE
─────
  # Basic (uses defaults — recommended first run)
  python scripts/finetune_grayscale.py

  # Custom paths and parameters
  python scripts/finetune_grayscale.py \\
      --base-model models/best_model.pth \\
      --annotation-path clinical_data/annotations/annotations.json \\
      --image-dir clinical_data/clean \\
      --epochs 50 \\
      --lr 0.0001 \\
      --grayscale-prob 0.3 \\
      --min-gray-dice 0.88 \\
      --save-dir models/ \\
      --device auto

  # Aggressive grayscale training (more grayscale exposure)
  python scripts/finetune_grayscale.py \\
      --grayscale-prob 0.5 \\
      --epochs 80 \\
      --lr 0.00005

  # Dry run — validate only, no training
  python scripts/finetune_grayscale.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── Ensure project root is on sys.path ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import AuditLogger, set_logger, get_logger

# ── Banner ──────────────────────────────────────────────────────
_BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║      GRAYSCALE FINE-TUNING — Accuracy-Safe Augmentation     ║
╚══════════════════════════════════════════════════════════════╝
"""


# ================================================================
# Dice coefficient computation
# ================================================================

def compute_dice_per_class(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-6,
) -> Dict[int, float]:
    """Compute per-class Dice coefficient.

    Parameters
    ----------
    predictions : torch.Tensor
        Model output logits of shape ``(B, C, H, W)``.
    targets : torch.Tensor
        Ground-truth labels of shape ``(B, H, W)`` with integer
        class indices.
    num_classes : int
        Number of segmentation classes.
    smooth : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    dict[int, float]
        Dice coefficient per class.
    """
    preds = predictions.argmax(dim=1)  # (B, H, W)

    dice_scores: Dict[int, float] = {}
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores[c] = float(dice.item())

    return dice_scores


def mean_dice(
    dice_per_class: Dict[int, float],
    exclude_background: bool = True,
) -> float:
    """Compute mean Dice, optionally excluding background (class 0).

    Parameters
    ----------
    dice_per_class : dict
        Per-class Dice from :func:`compute_dice_per_class`.
    exclude_background : bool
        If True, exclude class 0 from the mean.

    Returns
    -------
    float
        Mean Dice coefficient.
    """
    classes = [c for c in dice_per_class if (c > 0 or not exclude_background)]
    if not classes:
        return 0.0
    return float(np.mean([dice_per_class[c] for c in classes]))


# ================================================================
# Validation pass
# ================================================================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    label: str = "val",
) -> Tuple[float, Dict[int, float]]:
    """Run one validation pass and return mean Dice.

    Parameters
    ----------
    model : nn.Module
        The segmentation model in eval mode.
    dataloader : DataLoader
        Validation data loader.
    device : torch.device
        Compute device.
    num_classes : int
        Number of classes.
    label : str
        Label for logging (e.g. "rgb_val", "gray_val").

    Returns
    -------
    (mean_dice_score, per_class_dice)
    """
    model.eval()

    all_dice: Dict[int, list] = {c: [] for c in range(num_classes)}
    total_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        dice = compute_dice_per_class(outputs, masks, num_classes)

        for c in range(num_classes):
            all_dice[c].append(dice[c])

        total_batches += 1

    # Average across batches
    avg_dice: Dict[int, float] = {}
    for c in range(num_classes):
        if all_dice[c]:
            avg_dice[c] = float(np.mean(all_dice[c]))
        else:
            avg_dice[c] = 0.0

    md = mean_dice(avg_dice, exclude_background=True)

    class_names = {0: "bg", 1: "pupil", 2: "iris", 3: "ring"}
    parts = []
    for c in range(num_classes):
        name = class_names.get(c, f"c{c}")
        parts.append(f"{name}={avg_dice[c]:.4f}")
    detail = ", ".join(parts)

    get_logger().info(
        "  [%s] mean_dice=%.4f (%s) — %d batches",
        label, md, detail, total_batches,
    )

    return md, avg_dice


# ================================================================
# Training pass
# ================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch and return average loss.

    Parameters
    ----------
    model : nn.Module
        Model in train mode.
    dataloader : DataLoader
        Training data loader.
    optimizer : Optimizer
        Parameter optimizer.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch (1-based).
    total_epochs : int
        Total number of epochs.

    Returns
    -------
    float
        Average loss for this epoch.
    """
    model.train()

    running_loss = 0.0
    batch_count = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_count += 1

        # Progress indicator every 20 batches
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            avg = running_loss / batch_count
            pct = (batch_idx + 1) / len(dataloader) * 100
            sys.stdout.write(
                f"\r  Epoch {epoch:3d}/{total_epochs} "
                f"[{'█' * int(pct // 3)}{'░' * (33 - int(pct // 3))}] "
                f"{pct:5.1f}%  loss={avg:.4f}"
            )
            sys.stdout.flush()

    avg_loss = running_loss / max(batch_count, 1)
    sys.stdout.write("\n")
    return avg_loss


# ================================================================
# Device selection
# ================================================================

def select_device(device_str: str) -> torch.device:
    """Select the best available compute device.

    Parameters
    ----------
    device_str : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# ================================================================
# Main fine-tuning logic
# ================================================================

def finetune_grayscale(args: argparse.Namespace) -> None:
    """Execute the grayscale fine-tuning pipeline."""

    print(_BANNER)

    logger_obj = get_logger()

    # ── Resolve paths ───────────────────────────────────────────
    base_model_path = Path(args.base_model)
    if not base_model_path.is_file():
        print(f"\n  ✗ ERROR: Base model not found: {base_model_path}")
        sys.exit(1)

    annotation_path = Path(args.annotation_path)
    if not annotation_path.is_file():
        print(f"\n  ✗ ERROR: Annotation file not found: {annotation_path}")
        sys.exit(1)

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"\n  ✗ ERROR: Image directory not found: {image_dir}")
        sys.exit(1)

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)

    print(f"  Base model:      {base_model_path}")
    print(f"  Annotations:     {annotation_path}")
    print(f"  Image dir:       {image_dir}")
    print(f"  Mask dir:        {mask_dir or '(generate from annotations)'}")
    print(f"  Save dir:        {save_dir}")
    print(f"  Device:          {device}")
    print(f"{'─' * 60}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Input size:      {args.input_size}")
    print(f"  Grayscale prob:  {args.grayscale_prob}")
    print(f"  Min gray Dice:   {args.min_gray_dice}")
    print(f"  Num classes:     {args.num_classes}")
    print(f"  Dry run:         {args.dry_run}")
    print(f"{'═' * 60}\n")

    # ── Load model ──────────────────────────────────────────────
    print("  Loading model architecture...")

    try:
        from pupil_tracking.ml.architecture import EyeSegmentationModel
    except ImportError:
        print("  ✗ ERROR: Cannot import EyeSegmentationModel")
        sys.exit(1)

    model = EyeSegmentationModel(num_classes=args.num_classes)

    print(f"  Loading weights from {base_model_path}...")
    checkpoint = torch.load(str(base_model_path), map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:      {param_count:,} total, {trainable:,} trainable")

    # ── Build datasets ──────────────────────────────────────────
    print("\n  Building datasets...")

    from pupil_tracking.ml.dataset import (
        load_annotations,
        split_by_images,
        EyeSegmentationDataset,
        _get_train_augmentation,
        _get_val_augmentation,
        _get_grayscale_val_augmentation,
    )

    image_ids, annotations = load_annotations(str(annotation_path))

    if len(image_ids) < 2:
        print(f"\n  ✗ ERROR: Need ≥2 annotated images, got {len(image_ids)}")
        sys.exit(1)

    train_ids, val_ids = split_by_images(
        image_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Training set — WITH grayscale augmentation
    train_transform = _get_train_augmentation(
        input_size=args.input_size,
        grayscale_prob=args.grayscale_prob,
    )

    # Validation set — standard RGB (no grayscale)
    rgb_val_transform = _get_val_augmentation(
        input_size=args.input_size,
    )

    # Validation set — forced grayscale (every image)
    gray_val_transform = _get_grayscale_val_augmentation(
        input_size=args.input_size,
    )

    mask_dir_str = str(mask_dir) if mask_dir else None

    train_ds = EyeSegmentationDataset(
        image_ids=train_ids,
        annotations=annotations,
        image_dir=str(image_dir),
        mask_dir=mask_dir_str,
        transform=train_transform,
        copies_per_image=args.copies_per_image,
        input_size=args.input_size,
        num_classes=args.num_classes,
    )

    rgb_val_ds = EyeSegmentationDataset(
        image_ids=val_ids,
        annotations=annotations,
        image_dir=str(image_dir),
        mask_dir=mask_dir_str,
        transform=rgb_val_transform,
        copies_per_image=max(5, args.copies_per_image // 5),
        input_size=args.input_size,
        num_classes=args.num_classes,
    )

    gray_val_ds = EyeSegmentationDataset(
        image_ids=val_ids,
        annotations=annotations,
        image_dir=str(image_dir),
        mask_dir=mask_dir_str,
        transform=gray_val_transform,
        copies_per_image=max(5, args.copies_per_image // 5),
        input_size=args.input_size,
        num_classes=args.num_classes,
    )

    num_workers = min(4, len(train_ds) // max(args.batch_size, 1))
    num_workers = max(0, num_workers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    rgb_val_loader = DataLoader(
        rgb_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    gray_val_loader = DataLoader(
        gray_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"  Train:  {len(train_ds)} samples ({len(train_ids)} images "
          f"× {args.copies_per_image} copies)")
    print(f"  Val:    {len(rgb_val_ds)} RGB + {len(gray_val_ds)} gray "
          f"({len(val_ids)} images)")

    # ── Baseline validation (RGB) ───────────────────────────────
    print(f"\n{'─' * 60}")
    print("  BASELINE VALIDATION (before fine-tuning)")
    print(f"{'─' * 60}")

    baseline_rgb_dice, baseline_rgb_per_class = validate(
        model, rgb_val_loader, device, args.num_classes, label="baseline_rgb",
    )

    baseline_gray_dice, baseline_gray_per_class = validate(
        model, gray_val_loader, device, args.num_classes, label="baseline_gray",
    )

    print(f"\n  Baseline RGB Dice:       {baseline_rgb_dice:.4f}")
    print(f"  Baseline Grayscale Dice: {baseline_gray_dice:.4f}")
    print(f"  Target RGB Dice:         ≥ {baseline_rgb_dice:.4f} (no regression)")
    print(f"  Target Gray Dice:        ≥ {args.min_gray_dice:.4f}")

    if args.dry_run:
        print(f"\n{'═' * 60}")
        print("  DRY RUN — no training performed")
        print(f"{'═' * 60}\n")
        return

    # ── Setup optimizer and loss ────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  FINE-TUNING")
    print(f"{'─' * 60}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing scheduler — gentle LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    criterion = nn.CrossEntropyLoss()

    # ── Training loop ───────────────────────────────────────────
    best_combined_score = 0.0
    best_epoch = 0
    best_rgb_dice = baseline_rgb_dice
    best_gray_dice = baseline_gray_dice
    saved_checkpoint = False
    patience_counter = 0

    history = {
        "baseline_rgb_dice": baseline_rgb_dice,
        "baseline_gray_dice": baseline_gray_dice,
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args.epochs,
        )

        # Validate RGB
        rgb_dice, rgb_per_class = validate(
            model, rgb_val_loader, device, args.num_classes,
            label="rgb_val",
        )

        # Validate Grayscale
        gray_dice, gray_per_class = validate(
            model, gray_val_loader, device, args.num_classes,
            label="gray_val",
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Check save conditions ───────────────────────────────
        rgb_ok = rgb_dice >= baseline_rgb_dice
        gray_ok = gray_dice >= args.min_gray_dice
        combined_score = (rgb_dice + gray_dice) / 2.0

        save_this = (
            rgb_ok
            and gray_ok
            and combined_score > best_combined_score
        )

        status = ""
        if save_this:
            best_combined_score = combined_score
            best_epoch = epoch
            best_rgb_dice = rgb_dice
            best_gray_dice = gray_dice
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), str(checkpoint_path))
            saved_checkpoint = True
            status = " ★ SAVED"

            # Save metadata
            meta = {
                "epoch": epoch,
                "rgb_dice": rgb_dice,
                "gray_dice": gray_dice,
                "combined_score": combined_score,
                "baseline_rgb_dice": baseline_rgb_dice,
                "baseline_gray_dice": baseline_gray_dice,
                "grayscale_prob": args.grayscale_prob,
                "learning_rate": args.lr,
                "input_size": args.input_size,
                "num_classes": args.num_classes,
                "fine_tuned_from": str(base_model_path),
                "grayscale_finetuned": True,
            }
            meta_path = save_dir / "checkpoint_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        else:
            patience_counter += 1

        # ── Epoch summary ───────────────────────────────────────
        rgb_indicator = "✓" if rgb_ok else "✗"
        gray_indicator = "✓" if gray_ok else "✗"

        print(
            f"  Epoch {epoch:3d}/{args.epochs} │ "
            f"loss={train_loss:.4f} │ "
            f"RGB={rgb_dice:.4f} [{rgb_indicator}] │ "
            f"Gray={gray_dice:.4f} [{gray_indicator}] │ "
            f"lr={current_lr:.6f} │ "
            f"{elapsed:.1f}s{status}"
        )

        # Record history
        history["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "rgb_dice": rgb_dice,
            "gray_dice": gray_dice,
            "combined_score": combined_score,
            "rgb_ok": rgb_ok,
            "gray_ok": gray_ok,
            "saved": save_this,
            "lr": current_lr,
        })

        # Early stopping
        if patience_counter >= args.patience:
            print(
                f"\n  ⚠ Early stopping — no improvement for "
                f"{args.patience} epochs"
            )
            break

    # ── Save training history ───────────────────────────────────
    history_path = save_dir / "finetune_grayscale_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Final report ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  FINE-TUNING COMPLETE")
    print(f"{'─' * 60}")
    print(f"  Epochs trained:           {epoch}")
    print(f"  Baseline RGB Dice:        {baseline_rgb_dice:.4f}")
    print(f"  Baseline Grayscale Dice:  {baseline_gray_dice:.4f}")

    if saved_checkpoint:
        print(f"{'─' * 60}")
        print(f"  ★ BEST CHECKPOINT (epoch {best_epoch}):")
        print(f"    RGB Dice:               {best_rgb_dice:.4f} "
              f"(baseline: {baseline_rgb_dice:.4f}, "
              f"delta: {best_rgb_dice - baseline_rgb_dice:+.4f})")
        print(f"    Grayscale Dice:         {best_gray_dice:.4f} "
              f"(baseline: {baseline_gray_dice:.4f}, "
              f"delta: {best_gray_dice - baseline_gray_dice:+.4f})")
        print(f"    Combined Score:         {best_combined_score:.4f}")
        print(f"    Saved to:               {save_dir / 'best_model.pth'}")
        print(f"    Metadata:               {save_dir / 'checkpoint_meta.json'}")

        improvement = best_gray_dice - baseline_gray_dice
        if improvement > 0.05:
            print(f"\n  ✓ EXCELLENT — Grayscale Dice improved by "
                  f"{improvement:+.4f}")
        elif improvement > 0.01:
            print(f"\n  ✓ GOOD — Grayscale Dice improved by "
                  f"{improvement:+.4f}")
        else:
            print(f"\n  ✓ Modest improvement ({improvement:+.4f}). "
                  f"Consider increasing --grayscale-prob or --epochs.")
    else:
        print(f"{'─' * 60}")
        print(f"  ⚠ NO CHECKPOINT SAVED")
        print(f"    The fine-tuned model did not meet both thresholds:")
        print(f"    - RGB Dice ≥ {baseline_rgb_dice:.4f} (baseline)")
        print(f"    - Gray Dice ≥ {args.min_gray_dice:.4f}")
        print(f"    Original model is PRESERVED UNCHANGED.")
        print(f"\n  SUGGESTIONS:")
        print(f"    1. Try lower learning rate: --lr 0.00005")
        print(f"    2. Try more epochs: --epochs 100")
        print(f"    3. Try lower grayscale threshold: --min-gray-dice 0.85")
        print(f"    4. Try higher grayscale exposure: --grayscale-prob 0.4")

    print(f"\n  Training history:         {history_path}")
    print(f"{'═' * 60}\n")


# ================================================================
# Argument parser
# ================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune_grayscale.py",
        description=(
            "Fine-tune a trained segmentation model for grayscale "
            "robustness without losing RGB accuracy."
        ),
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=40, width=95
        ),
    )

    # ── Model & data paths ──────────────────────────────────────
    parser.add_argument(
        "--base-model",
        type=str,
        default="models/best_model.pth",
        metavar="PATH",
        help="Path to existing trained model weights (.pth)",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="clinical_data/annotations/annotations.json",
        metavar="PATH",
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="clinical_data/clean",
        metavar="PATH",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory with pre-generated masks (optional)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/",
        metavar="PATH",
        help="Directory to save fine-tuned checkpoint",
    )

    # ── Training parameters ─────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Number of fine-tuning epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Batch size (default: 8, reduce for low VRAM)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="F",
        help=(
            "Learning rate (default: 0.0001). Should be 5-10x lower "
            "than initial training LR to avoid catastrophic forgetting."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="F",
        help="Weight decay / L2 regularisation (default: 1e-4)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        metavar="PX",
        help="Model input resolution (default: 512)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        metavar="N",
        help="Number of segmentation classes (default: 3)",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=50,
        metavar="N",
        help="Augmented copies per source image per epoch (default: 50)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        metavar="F",
        help="Fraction of images for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        metavar="N",
        help="Early stopping patience in epochs (default: 20)",
    )

    # ── Grayscale-specific parameters ───────────────────────────
    parser.add_argument(
        "--grayscale-prob",
        type=float,
        default=0.3,
        metavar="F",
        help=(
            "Probability of converting each training image to "
            "grayscale (default: 0.3 = 30%%). Higher values give "
            "more grayscale exposure but may need more epochs."
        ),
    )
    parser.add_argument(
        "--min-gray-dice",
        type=float,
        default=0.88,
        metavar="F",
        help=(
            "Minimum grayscale Dice required to save checkpoint "
            "(default: 0.88). Lower this if your dataset is "
            "inherently harder."
        ),
    )

    # ── Device & misc ───────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only run baseline validation, no training",
    )

    return parser


# ================================================================
# Entry point
# ================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Initialise logger
    cfg = get_config()
    audit_logger = AuditLogger(log_dir=cfg.paths.log_dir)
    set_logger(audit_logger)

    try:
        finetune_grayscale(args)
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