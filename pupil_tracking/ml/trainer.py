"""
Training loop with:
  - image-level train/val split (no data leakage)
  - mixed-precision training
  - cosine-annealing LR schedule
  - early stopping
  - per-class IoU tracking
  - automatic model checkpointing
  - full audit logging
  - Windows-compatible multiprocessing
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pupil_tracking.ml.architecture import EyeSegmentationModel, get_device
from pupil_tracking.ml.dataset import build_datasets
from pupil_tracking.ml.losses import CompositeLoss
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════


def compute_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    ignore_bg: bool = True,
) -> Dict[str, float]:
    """Per-class IoU + mean IoU.

    Parameters
    ----------
    preds : Tensor [N, H, W]  predicted class indices
    targets : Tensor [N, H, W]  ground-truth class indices
    """
    ious = {}
    class_names = ["background", "pupil", "iris"]

    start_c = 1 if ignore_bg else 0
    valid_ious = []

    for c in range(start_c, num_classes):
        pred_c = (preds == c)
        tgt_c = (targets == c)
        intersection = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()

        if union < 1:
            iou = float("nan")
        else:
            iou = float(intersection / union)
            valid_ious.append(iou)

        name = class_names[c] if c < len(class_names) else f"class_{c}"
        ious[f"iou_{name}"] = iou

    ious["iou_mean"] = float(np.nanmean(valid_ious)) if valid_ious else 0.0
    return ious


def compute_center_error(
    pred_mask: np.ndarray,
    gt_center: Tuple[float, float],
    class_id: int,
    mask_scale: float = 1.0,
) -> float:
    """Pixel error between predicted mask centroid and GT center.

    Returns
    -------
    float  Euclidean distance in pixels (at the original image scale).
    """
    ys, xs = np.where(pred_mask == class_id)
    if len(xs) == 0:
        return float("inf")
    pred_cx = float(np.mean(xs)) * mask_scale
    pred_cy = float(np.mean(ys)) * mask_scale
    dx = pred_cx - gt_center[0]
    dy = pred_cy - gt_center[1]
    return float(np.sqrt(dx * dx + dy * dy))


# ══════════════════════════════════════════════════════════════════════
# Safe num_workers for Windows
# ══════════════════════════════════════════════════════════════════════


def _safe_num_workers(requested: int) -> int:
    """On Windows, multiprocessing spawn can crash DataLoader workers.
    Use 0 (main-process loading) to avoid this entirely.
    On Linux/Mac, use the requested value.
    """
    if os.name == "nt":  # Windows
        return 0
    return requested


# ══════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════


class Trainer:
    """End-to-end training manager.

    Usage
    -----
    >>> trainer = Trainer()
    >>> trainer.train()
    """

    def __init__(
        self,
        config=None,
        annotation_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        ring_labels_path: Optional[str] = None,
        loss_kwargs: Optional[dict] = None,
    ) -> None:
        self.cfg = config or get_config()
        self.logger = get_logger()

        tc = self.cfg.training
        mc = self.cfg.model
        pc = self.cfg.paths

        self.annotation_path = annotation_path or str(
            Path(pc.annotation_dir) / "annotations.json"
        )
        self.image_dir = image_dir or str(
            Path(pc.annotation_dir) / "images"
        )
        self.mask_dir = mask_dir or str(
            Path(pc.annotation_dir) / "masks"
        )

        self.device = get_device(mc.device)
        self.logger.info("Training device: %s", self.device)

        # ── datasets ────────────────────────────────────────────
        self.train_ds, self.val_ds = build_datasets(
            annotation_path=self.annotation_path,
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            val_ratio=tc.val_ratio,
            copies_per_image=tc.augmentations_per_image,
            input_size=mc.input_size,
            num_classes=mc.num_classes,
            ring_labels_path=ring_labels_path,
        )

        safe_workers = _safe_num_workers(tc.num_workers)
        if safe_workers != tc.num_workers:
            self.logger.info(
                "Windows detected: using num_workers=0 "
                "(requested %d)", tc.num_workers
            )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=safe_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=tc.batch_size,
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        # ── model ───────────────────────────────────────────────
        self.model = EyeSegmentationModel(
            encoder=mc.encoder,
            num_classes=mc.num_classes,
            pretrained=mc.pretrained,
        ).to(self.device)

        # ── loss / optim / scheduler ────────────────────────────
        kw = {"num_classes": mc.num_classes}
        if loss_kwargs:
            kw.update(loss_kwargs)
        self.criterion = CompositeLoss(**kw).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc.epochs, eta_min=1e-7,
        )

        self.scaler = torch.amp.GradScaler(
            device=str(self.device),
            enabled=tc.use_amp and self.device.type == "cuda",
        )

        # ── tracking ────────────────────────────────────────────
        self.history: List[Dict[str, Any]] = []
        self.best_val_iou: float = 0.0
        self.patience_counter: int = 0

    # ════════════════════════════════════════════════════════════
    # Main training loop
    # ════════════════════════════════════════════════════════════

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns
        -------
        dict  with keys ``best_val_iou``, ``epochs_trained``,
              ``history``, ``model_path``
        """
        tc = self.cfg.training
        self.logger.info(
            "Starting training: %d epochs, batch=%d, lr=%.1e, "
            "train=%d samples, val=%d samples",
            tc.epochs,
            tc.batch_size,
            tc.learning_rate,
            len(self.train_ds),
            len(self.val_ds),
        )

        for epoch in range(1, tc.epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "lr": lr,
                "train": train_metrics,
                "val": val_metrics,
                "time_s": time.time() - t0,
            }
            self.history.append(record)

            # ── logging ─────────────────────────────────────────
            val_iou = val_metrics.get("iou_mean", 0.0)
            self.logger.info(
                "Epoch %3d/%d  |  train_loss=%.4f  val_loss=%.4f  "
                "val_iou=%.4f  lr=%.2e  [%.1fs]",
                epoch,
                tc.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_iou,
                lr,
                record["time_s"],
            )

            # per-class IoU
            for key in sorted(val_metrics):
                if key.startswith("iou_") and key != "iou_mean":
                    self.logger.info(
                        "  %s = %.4f", key, val_metrics[key]
                    )

            # ── checkpointing ───────────────────────────────────
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_iou)
                self.logger.info(
                    "  [BEST] New best model (IoU=%.4f)", val_iou
                )
            else:
                self.patience_counter += 1

            # ── early stopping ──────────────────────────────────
            if self.patience_counter >= tc.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch,
                    tc.early_stopping_patience,
                )
                break

        # ── temperature calibration ─────────────────────────────
        self.logger.info("Calibrating temperature on validation set...")
        try:
            temp = self.model.calibrate_temperature(
                self.val_loader, self.device
            )
            self.logger.info("Calibrated temperature: %.4f", temp)
        except Exception as e:
            self.logger.warning(
                "Temperature calibration failed: %s (using 1.0)", e
            )

        # re-save with calibrated temperature
        final_path = str(
            Path(self.cfg.paths.model_dir) / "best_model.pth"
        )
        self.model.save(final_path)

        # save history
        history_path = str(
            Path(self.cfg.paths.model_dir) / "training_history.json"
        )
        with open(history_path, "w") as fh:
            json.dump(self.history, fh, indent=2, default=str)

        self.logger.info(
            "Training complete. Best IoU=%.4f Model saved to %s",
            self.best_val_iou,
            final_path,
        )

        return {
            "best_val_iou": self.best_val_iou,
            "epochs_trained": len(self.history),
            "history": self.history,
            "model_path": final_path,
        }

    # ════════════════════════════════════════════════════════════
    # Single-epoch methods
    # ════════════════════════════════════════════════════════════

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=str(self.device.type),
                enabled=self.scaler.is_enabled(),
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        all_preds = []
        all_targets = []

        for images, masks in self.val_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        ious = compute_iou(
            all_preds, all_targets, self.cfg.model.num_classes
        )

        metrics = {"loss": total_loss / max(n_batches, 1)}
        metrics.update(ious)
        return metrics

    # ════════════════════════════════════════════════════════════
    # Checkpointing
    # ════════════════════════════════════════════════════════════

    def _save_checkpoint(
        self, epoch: int, val_iou: float
    ) -> None:
        model_dir = Path(self.cfg.paths.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = str(model_dir / "best_model.pth")
        self.model.save(path)

        meta = {
            "epoch": epoch,
            "val_iou": val_iou,
            "best_val_iou": self.best_val_iou,
        }
        meta_path = str(model_dir / "checkpoint_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)