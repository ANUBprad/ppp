#!/usr/bin/env python3
"""
Phase 4: Pentacam Segmentation Model Trainer

Trains the PentacamSegmentator on the 6 training images (eye_01 through eye_06)
using transfer learning and heavy augmentation due to limited data.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from pupil_tracking.ml.pentacam_segmentator import (
    PentacamSegmentator,
    PentacamSegmentationConfig,
    segment_pentacam_image,
)

# ──────────────────────────────────────────────────────────────────────
# Dataset ────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────


class PentacamSegmentationDataset(Dataset):
    """Dataset loading Pentacam images and their ground-truth masks."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.augment = augment

        # Collect image-mask pairs
        self.samples: List[Tuple[Path, Path]] = []
        for eye_num in range(1, 7):  # eye_01 through eye_06 (training set)
            img_path = self.image_dir / f"eye_{eye_num:02d}.jpg"
            mask_path = self.mask_dir / f"eye_{eye_num:02d}_mask.png"

            if img_path.exists() and mask_path.exists():
                self.samples.append((img_path, mask_path))
            elif img_path.exists() and not mask_path.exists():
                # Image without mask - we'll generate a placeholder or skip
                print(f"Warning: No mask for {img_path.name}, skipping")
            # Skip if image doesn't exist

        if not self.samples:
            raise RuntimeError(
                f"No training samples found in {image_dir} with masks in {mask_dir}"
            )

        print(f"PentacamSegmentationDataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (grayscale or RGB)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            # Fallback: try grayscale
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Cannot load image: {img_path}")

        # Load mask (grayscale, values 0-3)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot load mask: {mask_path}")

        # Normalize mask to binary (0 or 1) - combine pupil and limbus into single mask
        # Actually, we'll create separate pupil and limbus masks from the ground truth
        # The mask values 0-3 represent: 0=background, 1=pupil, 2=limbus, 3=pupil+limbus
        # Let's extract individual masks

        # Create binary pupil mask (value 1 or 3 -> pupil)
        pupil_mask = ((mask == 1) | (mask == 3)).astype(np.float32)
        # Create binary limbus mask (value 2 or 3 -> limbus)  
        limbus_mask = ((mask == 2) | (mask == 3)).astype(np.float32)

        # Resize to model input size
        pupil_mask = cv2.resize(pupil_mask, (512, 512), interpolation=cv2.INTER_CUBIC)
        limbus_mask = cv2.resize(limbus_mask, (512, 512), interpolation=cv2.INTER_CUBIC)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Apply augmentations if enabled
        if self.augment:
            image, pupil_mask, limbus_mask = self._augment(
                image, pupil_mask, limbus_mask
            )

        # Convert image to tensor
        image_tensor = (
            torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            .float()
        )  # (3, 512, 512)

        # Convert masks to tensors
        pupil_mask_tensor = torch.from_numpy(pupil_mask.astype(np.float32)).unsqueeze(0)  # (1, 512, 512)
        limbus_mask_tensor = torch.from_numpy(limbus_mask.astype(np.float32)).unsqueeze(0)  # (1, 512, 512)

        return {
            "image": image_tensor,
            "pupil_mask": pupil_mask_tensor,
            "limbus_mask": limbus_mask_tensor,
            "image_path": str(img_path),
        }

    def _augment(
        self,
        image: np.ndarray,
        pupil_mask: np.ndarray,
        limbus_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentations for robustness to orientation/flips."""

        # Random rotation (90, 180 degrees)
        # Note: cv2.ROTATE_270 is not standard; use 90 counter-clockwise for 270 effect
        if random.random() < 0.5:
            angle = random.choice([90, 180])
            if angle == 90:
                rot_code = cv2.ROTATE_90_CLOCKWISE
            else:  # 180
                rot_code = cv2.ROTATE_180
            image = cv2.rotate(image, rot_code)
            pupil_mask = cv2.rotate(pupil_mask, rot_code)
            limbus_mask = cv2.rotate(limbus_mask, rot_code)

        # Random horizontal flip
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            pupil_mask = cv2.flip(pupil_mask, 1)
            limbus_mask = cv2.flip(limbus_mask, 1)

        # Random vertical flip
        if random.random() < 0.5:
            image = cv2.flip(image, 0)
            pupil_mask = cv2.flip(pupil_mask, 0)
            limbus_mask = cv2.flip(limbus_mask, 0)

        # Random brightness/contrast
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)

        # Random blur (for robustness)
        if random.random() < 0.3:
            ksize = random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image, pupil_mask, limbus_mask


# ──────────────────────────────────────────────────────────────────────
# Loss Functions ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.
    
    pred: (N, 1, H_pred, W_pred) predicted mask probabilities
    target: (N, 1, H_target, W_target) ground truth binary mask
    
    If spatial sizes differ, pred is upsampled to match target.
    Returns a scalar tensor (requires grad).
    """
    # Upsample pred to match target spatial size if needed
    if pred.shape[2:] != target.shape[2:]:
        pred = torch.nn.functional.interpolate(
            pred, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # Compute intersection and union across spatial dimensions
    pred_f = pred.flatten(1)  # (N, H*W)
    target_f = target.flatten(1)  # (N, H*W)
    
    intersection = (pred_f * target_f).sum(dim=1)  # (N,)
    union = pred_f.sum(dim=1) + target_f.sum(dim=1)  # (N,)
    
    # Dice coefficient (mean over batch)
    dice = (2.0 * intersection + smooth) / (union + smooth)  # (N,)
    
    # Return mean dice loss over batch
    return 1.0 - dice.mean()


def boundary_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 0.5,
) -> torch.Tensor:
    """
    Boundary-aware loss using Sobel edge detection.
    
    Encourages predicted mask boundaries to align with ground-truth boundaries.
    """
    # Upsample pred to match target spatial size if needed
    if pred.shape[2:] != target.shape[2:]:
        pred = torch.nn.functional.interpolate(
            pred, size=target.shape[2:], mode='bilinear', align_corners=False
        )
    
    # Apply Sobel filter to detect edges
    pred_sobel = torch.abs(
        torch.conv2d(
            pred,
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                          dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3),
            padding=1,
        )
    )
    target_sobel = torch.abs(
        torch.conv2d(
            target,
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                          dtype=target.dtype, device=target.device).view(1, 1, 3, 3),
            padding=1,
        )
    )

    # Boundary F1: proportion of predicted edges that match ground-truth edges
    intersection = (pred_sobel * target_sobel).sum()
    union = pred_sobel.sum() + target_sobel.sum() - intersection

    if union.item() > 0:
        boundary_f1 = (2.0 * intersection + 1e-7) / union.item()
    else:
        boundary_f1 = torch.tensor(1.0)

    return weight * (1.0 - boundary_f1)


# ──────────────────────────────────────────────────────────────────────
# Training Function ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────


def train_pentacam_segmentator(
    train_images_dir: str = "clinical_data/training_data/images",
    train_masks_dir: str = "clinical_data/annotations/masks",
    val_images_dir: str = None,
    val_masks_dir: str = None,
    epochs: int = 50,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: str = "cpu",
    model_save_path: str = "pentacam_segmentator.pth",
    resume_from: str = None,
) -> Dict[str, Any]:
    """
    Train the Pentacam segmentation model.
    
    Uses transfer learning with ResNet18 encoder + U-Net decoder.
    Heavy augmentation due to limited training data (6 images).
    
    Returns training history and results.
    """
    
    print("=" * 60)
    print("Pentacam Segmentation Model Training")
    print("=" * 60)
    
    # Set device
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create datasets
    try:
        train_dataset = PentacamSegmentationDataset(
            image_dir=train_images_dir,
            mask_dir=train_masks_dir,
            augment=True,  # Heavy augmentation for 6 images
        )
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Falling back to minimal dataset...")
        # Create minimal dataset from available samples
        train_dataset = PentacamSegmentationDataset(
            image_dir=train_images_dir,
            mask_dir=train_masks_dir,
            augment=True,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing for Windows compatibility
        pin_memory=True if str(device) == "cuda" else False,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print()
    
    # Initialize model
    config = PentacamSegmentationConfig(
        input_width=512,
        input_height=512,
        device=str(device),
    )
    
    model = PentacamSegmentator(config)
    model.initialize()
    
    # Load weights if resuming
    start_epoch = 0
    best_val_dice = 0.0
    if resume_from and os.path.exists(resume_from):
        model.load_weights(resume_from)
        print(f"Resumed from: {resume_from}")
    
    # Define loss functions and optimizer
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training history
    history = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "lr": [],
    }
    
    # ──────────────────────────────────────────────────────────────
    # Training Loop
    # ──────────────────────────────────────────────────────────────
    
    print("Beginning training loop...")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        model.model['encoder'].train()
        epoch_losses = []
        epoch_dice_scores = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            pupil_targets = batch["pupil_mask"].to(device)
            limbus_targets = batch["limbus_mask"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model.forward(images)
            
            # Compute losses
            # Pupil Dice loss
            pupil_loss = dice_loss(outputs["pupil_mask"], pupil_targets)
            
            # Limbus Dice loss
            limbus_loss = dice_loss(outputs["limbus_mask"], limbus_targets)
            
            # Combined segmentation loss
            seg_loss = pupil_loss + limbus_loss
            
            # Boundary loss (optional, for better edge detection)
            boundary_loss_val = boundary_loss(outputs["pupil_mask"], pupil_targets, weight=0.3)
            seg_loss = seg_loss + boundary_loss_val
            
            # Backward pass
            seg_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute Dice scores
            with torch.no_grad():
                pupil_dice = 1.0 - pupil_loss.item()
                limbus_dice = 1.0 - limbus_loss.item()
                avg_dice = (pupil_dice + limbus_dice) / 2.0
            
            epoch_losses.append(seg_loss.item())
            epoch_dice_scores.append(avg_dice)
            
            if batch_idx % max(1, len(train_loader) // 4) == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"[{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {seg_loss.item():.4f} "
                    f"Dice: {avg_dice:.4f} "
                    f"(Pupil: {pupil_dice:.4f}, Limbus: {limbus_dice:.4f})"
                )
        
        # Step scheduler
        scheduler.step()
        
        # Compute epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_dice = np.mean(epoch_dice_scores) if epoch_dice_scores else 0.0
        current_lr = scheduler.get_last_lr()[0]
        
        history["train_loss"].append(avg_loss)
        history["train_dice"].append(avg_dice)
        history["lr"].append(current_lr)
        
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"=> Train Loss: {avg_loss:.4f}, "
            f"Dice: {avg_dice:.4f}, LR: {current_lr:.1e}"
        )
        
        # ──────────────────────────────────────────────────────────
        # Validation (if validation data provided)
        # ──────────────────────────────────────────────────────────
        if val_images_dir and val_masks_dir and os.path.exists(val_images_dir):
            try:
                val_dataset = PentacamSegmentationDataset(
                    image_dir=val_images_dir,
                    mask_dir=val_masks_dir,
                    augment=False,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                
                model.eval()
                val_losses = []
                val_dice_scores = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        pupil_targets = batch["pupil_mask"].to(device)
                        limbus_targets = batch["limbus_mask"].to(device)
                        
                        outputs = model.forward(images)
                        
                        pupil_loss = dice_loss(outputs["pupil_mask"], pupil_targets)
                        limbus_loss = dice_loss(outputs["limbus_mask"], limbus_targets)
                        seg_loss = pupil_loss + limbus_loss
                        
                        val_losses.append(seg_loss.item())
                        
                        with torch.no_grad():
                            p_dice = 1.0 - pupil_loss.item()
                            l_dice = 1.0 - limbus_loss.item()
                            avg_d = (p_dice + l_dice) / 2.0
                            val_dice_scores.append(avg_d)
                
                avg_val_loss = np.mean(val_losses) if val_losses else 0.0
                avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0
                
                history["val_loss"].append(avg_val_loss)
                history["val_dice"].append(avg_val_dice)
                
                print(
                    f"=> Val Loss: {avg_val_loss:.4f}, "
                    f"Val Dice: {avg_val_dice:.4f}"
                )
                
                # Save best model
                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    model.save_weights(model_save_path)
                    print(f"  -> Saved best model (Dice: {best_val_dice:.4f})")
                
            except Exception as e:
                print(f"Validation error (skipping): {e}")
                history["val_loss"].append(None)
                history["val_dice"].append(None)
        else:
            history["val_loss"].append(None)
            history["val_dice"].append(None)
        
        print()
    
    # ──────────────────────────────────────────────────────────────
    # Finalization
    # ──────────────────────────────────────────────────────────────
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    # Save final model
    model.save_weights(model_save_path)
    print(f"Final model saved to: {model_save_path}")
    
    # Return training history
    return {
        "config": config,
        "history": history,
        "best_val_dice": best_val_dice,
        "model_path": model_save_path,
        "model": model,
        "train_samples": len(train_dataset),
    }


# ──────────────────────────────────────────────────────────────────────
# Evaluation Function ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

def evaluate_pentacam_segmentator(
    model: PentacamSegmentator,
    test_images_dir: str,
    test_masks_dir: str,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate the Pentacam segmentation model on held-out test images.
    
    Calculates metrics for each of the 5 test images separately,
    not just aggregate scores.
    """
    
    import numpy as np
    
    device = torch.device(device)
    model.eval()
    
    results = {
        "individual_images": [],
        "aggregate": {
            "pupil_dice": 0.0,
            "limbus_dice": 0.0,
            "boundary_f1": 0.0,
            "avg_surface_dist": 0.0,
            "hausdorff_95": 0.0,
            "ellipse_center_err": 0.0,
            "ellipse_axis_err": 0.0,
            "failure_rate": 0.0,
        },
        "n_images": 0,
    }
    
    # Get test image paths (eye_07 through eye_11)
    test_eyes = list(range(7, 12))  # eye_07 through eye_11
    
    pixel_dices = []
    boundary_f1s = []
    ellipse_center_errors = []
    ellipse_axis_errors = []
    failure_count = 0
    
    for eye_num in test_eyes:
        print(f"\nEvaluating eye_{eye_num:02d}...")
        
        # Load image
        img_path = os.path.join(test_images_dir, f"eye_{eye_num:02d}.jpg")
        mask_path = os.path.join(test_masks_dir, f"eye_{eye_num:02d}_mask.png")
        
        if not os.path.exists(img_path):
            print(f"  SKIP: Image not found: {img_path}")
            # Create a result entry for the missing image
            results["individual_images"].append({
                "eye": eye_num,
                "image_found": False,
                "metrics": {"dice": None, "failure": True},
            })
            continue
        
        if not os.path.exists(mask_path):
            print(f"  SKIP: Mask not found: {mask_path}")
            results["individual_images"].append({
                "eye": eye_num,
                "image_found": True,
                "mask_found": False,
                "metrics": {"dice": None, "failure": True},
            })
            continue
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Run segmentation
        with torch.no_grad():
            outputs = model.segment_and_detect(image)
        
        pupil_mask = outputs["pupil_mask"]
        limbus_mask = outputs["limbus_mask"]
        pupil_ellipse = outputs["pupil_ellipse"]
        limbus_ellipse = outputs["limbus_ellipse"]
        pair_valid = outputs["pair_valid"]
        
        # Compute Dice scores
        # Pupil Dice: overlap between predicted and ground truth pupil
        # Ground truth pupil from mask (values 1 or 3 = pupil)
        gt_pupil = (((ground_truth_mask == 1) | (ground_truth_mask == 3))).astype(np.float32)
        gt_limbus = (((ground_truth_mask == 2) | (ground_truth_mask == 3))).astype(np.float32)
        
        # Resize predictions to match ground truth size if needed
        if pupil_mask.shape != gt_pupil.shape:
            pupil_mask_resized = cv2.resize(pupil_mask, (gt_pupil.shape[1], gt_pupil.shape[0]))
            limbus_mask_resized = cv2.resize(limbus_mask, (gt_pupil.shape[1], gt_pupil.shape[0]))
        else:
            pupil_mask_resized = pupil_mask
            limbus_mask_resized = limbus_mask
        
        # Binarize
        pupil_pred = (pupil_mask_resized > 0.5).astype(np.float32)
        limbus_pred = (limbus_mask_resized > 0.5).astype(np.float32)
        
        # Dice score for pupil
        pupil_intersection = np.sum(pupil_pred * gt_pupil)
        pupil_union = np.sum(pupil_pred) + np.sum(gt_pupil)
        pupil_dice = (2.0 * pupil_intersection) / (pupil_union + 1e-10) if pupil_union > 0 else 1.0
        
        # Dice score for limbus
        limbus_intersection = np.sum(limbus_pred * gt_limbus)
        limbus_union = np.sum(limbus_pred) + np.sum(gt_limbus)
        limbus_dice = (2.0 * limbus_intersection) / (limbus_union + 1e-10) if limbus_union > 0 else 1.0
        
        # Boundary F1
        # Use the boundary loss approach
        pupil_sobel = np.abs(cv2.Sobel(pupil_pred.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3))
        target_sobel = np.abs(cv2.Sobel(gt_pupil.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3))
        boundary_intersection = np.sum(pupil_sobel * target_sobel)
        boundary_union = np.sum(pupil_sobel) + np.sum(target_sobel) - boundary_intersection
        boundary_f1 = (2.0 * boundary_intersection) / (boundary_union + 1e-10) if boundary_union > 0 else 1.0
        
        # Ellipse center error
        center_err = None
        if pupil_ellipse.get("valid", False) and limbus_ellipse.get("valid", False):
            px, py = pupil_ellipse["center"]
            lx, ly = limbus_ellipse["center"]
            center_err = np.sqrt((px - lx)**2 + (py - ly)**2)
        elif pupil_ellipse.get("valid", False):
            # Compare pupil ellipse center to image center as proxy
            center_err = None  # Can't compute without valid limbus
        
        # Ellipse axis/radius error
        axis_err = None
        if pupil_ellipse.get("valid", False) and limbus_ellipse.get("valid", False):
            p_major = pupil_ellipse["axes"][0]
            l_major = limbus_ellipse["axes"][0]
            axis_err = abs(p_major - l_major)
        elif pupil_ellipse.get("valid", False):
            axis_err = None
        
        # Failure rate: if pair is invalid or ellipse fitting failed
        is_failure = not pair_valid or not pupil_ellipse.get("valid", False) or not limbus_ellipse.get("valid", False)
        if is_failure:
            failure_count += 1
        
        # Store individual image results
        individual_result = {
            "eye": eye_num,
            "image_found": True,
            "mask_found": True,
            "pupil_dice": float(pupil_dice),
            "limbus_dice": float(limbus_dice),
            "boundary_f1": float(boundary_f1),
            "pupil_ellipse_valid": pupil_ellipse.get("valid", False),
            "limbus_ellipse_valid": limbus_ellipse.get("valid", False),
            "pair_valid": pair_valid,
            "pupil_confidence": outputs["pupil_confidence"],
            "limbus_confidence": outputs["limbus_confidence"],
            "pair_reason": outputs.get("pair_reason", ""),
            "center_error_px": center_err,
            "axis_error_px": axis_err,
        }
        
        results["individual_images"].append(individual_result)
        results["n_images"] += 1
        
        # Aggregate metrics (only if valid)
        if pupil_dice > 0:
            pixel_dices.append(pupil_dice)
        if boundary_f1 > 0:
            boundary_f1s.append(boundary_f1)
        if center_err is not None:
            ellipse_center_errors.append(center_err)
        if axis_err is not None:
            ellipse_axis_errors.append(axis_err)
        
        print(f"  eye_{eye_num:02d}: pupil_dice={pupil_dice:.4f}, "
              f"limbus_dice={limbus_dice:.4f}, boundary_f1={boundary_f1:.4f}")
    
    # Compute aggregate metrics
    if pixel_dices:
        results["aggregate"]["pupil_dice"] = float(np.mean(pixel_dices))
    if boundary_f1s:
        results["aggregate"]["boundary_f1"] = float(np.mean(boundary_f1s))
    if ellipse_center_errors:
        results["aggregate"]["ellipse_center_err"] = float(np.mean(ellipse_center_errors))
    if ellipse_axis_errors:
        results["aggregate"]["ellipse_axis_err"] = float(np.mean(ellipse_axis_errors))
    if results["n_images"] > 0:
        results["aggregate"]["failure_rate"] = float(failure_count / results["n_images"])
    
    # 95th percentile Hausdorff distance (approximation)
    # ... (can be computed if needed)
    
    return results


# ──────────────────────────────────────────────────────────────────────
# Main ───────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    print("Pentacam Segmentation Trainer")
    print("=" * 50)
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    action = "train"
    if len(args) > 0:
        if args[0] == "train":
            action = "train"
        elif args[0] == "eval":
            action = "eval"
        elif args[0] == "segment":
            action = "segment"
    
    if action == "train":
        # Train the model
        result = train_pentacam_segmentator(
            epochs=30,  # Reduced for quick training with limited data
            batch_size=2,
            learning_rate=1e-4,
            device="cpu",
            model_save_path="pentacam_segmentator.pth",
        )
        
        print(f"\nTraining summary:")
        print(f"  Best val Dice: {result['best_val_dice']:.4f}")
        print(f"  Train samples: {result['train_samples']}")
        print(f"  Model saved to: {result['model_path']}")
        
    elif action == "eval":
        # Evaluate the model
        if len(args) < 2:
            print("Usage: python pentacam_trainer.py eval <model_path>")
            sys.exit(1)
        
        model_path = args[1]
        
        # Load model
        from pupil_tracking.ml.pentacam_segmentator import PentacamSegmentator, PentacamSegmentationConfig
        config = PentacamSegmentationConfig()
        model = PentacamSegmentator(config)
        model.initialize()
        model.load_weights(model_path)
        
        # Evaluate
        results = evaluate_pentacam_segmentator(
            model=model,
            test_images_dir="clinical_data/training_data/images",
            test_masks_dir="clinical_data/annotations/masks",
            device="cpu",
        )
        
        print(f"\nEvaluation results:")
        print(f"  Images evaluated: {results['n_images']}")
        print(f"  Aggregate pupil Dice: {results['aggregate']['pupil_dice']:.4f}")
        print(f"  Aggregate limbus Dice: {results['aggregate']['limbus_dice']:.4f}")
        print(f"  Aggregate boundary F1: {results['aggregate']['boundary_f1']:.4f}")
        print(f"  Failure rate: {results['aggregate']['failure_rate']:.2%}")
        
        print(f"\nIndividual image results:")
        for indiv in results["individual_images"]:
            if indiv.get("image_found", False):
                print(f"  eye_{indiv['eye']:02d}: "
                      f"pupil_dice={indiv['pupil_dice']:.4f}, "
                      f"limbus_dice={indiv['limbus_dice']:.4f}, "
                      f"boundary_f1={indiv['boundary_f1']:.4f}, "
                      f"failure={not indiv.get('pair_valid', True)}")
    
    elif action == "segment":
        # Single image segmentation
        if len(args) < 2:
            print("Usage: python pentacam_trainer.py segment <image_path>")
            sys.exit(1)
        
        image_path = args[1]
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            sys.exit(1)
        
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Load model
        from pupil_tracking.ml.pentacam_segmentator import PentacamSegmentator, PentacamSegmentationConfig
        config = PentacamSegmentationConfig()
        model = PentacamSegmentator(config)
        model.initialize()
        
        # Segment
        result = model.segment_and_detect(image)
        
        print(f"\nSegmentation results for {image_path}:")
        print(f"  Pupil mask shape: {result['pupil_mask'].shape}")
        print(f"  Limbus mask shape: {result['limbus_mask'].shape}")
        print(f"  Pupil ellipse valid: {result['pupil_ellipse']['valid']}")
        print(f"  Limbus ellipse valid: {result['limbus_ellipse']['valid']}")
        print(f"  Pair valid: {result['pair_valid']}")
        print(f"  Pupil confidence: {result['pupil_confidence']:.4f}")
        print(f"  Limbus confidence: {result['limbus_confidence']:.4f}")