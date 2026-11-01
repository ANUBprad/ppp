"""
Composite loss for surgical-grade segmentation.

Combines:
    1. Cross-entropy   — pixel-level classification
    2. Dice loss       — region overlap (handles class imbalance)
    3. Boundary loss   — penalises boundary inaccuracy specifically
    4. Focal loss      — handles hard negatives and rare classes (NEW)

The boundary component is critical for limbus detection, where the
exact edge of the iris-sclera junction determines surgical planning.

Supports both 3-class and 4-class configurations:
    3-class (legacy):      0=background  1=pupil  2=iris
    4-class (ring-aware):  0=background  1=pupil  2=iris  3=suction_ring

The ring class (3) is often sparse — it only appears in docked images.
The ``FocalLoss`` and class-weight balancing in ``CompositeLoss``
ensure that the model still learns ring boundaries despite the
class imbalance.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Default class weights for each configuration.
# Background is down-weighted; minority classes are up-weighted.
_DEFAULT_CLASS_WEIGHTS: Dict[int, List[float]] = {
    3: [0.3, 1.0, 1.0],
    4: [0.3, 1.0, 1.0, 1.2],
}


def get_default_class_weights(num_classes: int) -> List[float]:
    """Return default class weights for the given class count.

    Parameters
    ----------
    num_classes : int
        3 or 4.

    Returns
    -------
    list of float
        Per-class weights.
    """
    return list(_DEFAULT_CLASS_WEIGHTS.get(num_classes, [1.0] * num_classes))


# ═══════════════════════════════════════════════════════════════════════
#  Dice Loss
# ═══════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft Dice loss per class, averaged.

    Handles class imbalance inherently because Dice normalises by
    each class's area.

    When ``skip_background=True`` (default), class 0 is excluded
    from the average so that the vast background area does not
    dominate the gradient.

    When ``per_class_weights`` is provided, the per-class Dice
    losses are weighted before averaging.  This allows boosting
    the ring class (3) when it is rare.

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to avoid division by zero.
    num_classes : int
        Number of segmentation classes (3 or 4).
    skip_background : bool
        If True, exclude class 0 from the loss average.
    per_class_weights : list of float or None
        Optional per-class weights applied to individual Dice terms.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        num_classes: int = 3,
        skip_background: bool = True,
        per_class_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.skip_background = skip_background

        if per_class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(per_class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor [B, C, H, W]  raw logits
        targets : Tensor [B, H, W]  class indices (long)

        Returns
        -------
        Tensor  scalar loss value
        """
        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]

        # Clamp targets to valid range
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)

        targets_oh = F.one_hot(
            targets_clamped, self.num_classes
        ).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)  # reduce over batch, height, width
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (
            union + self.smooth
        )  # shape [C]

        # Select classes to include
        start_class = 1 if self.skip_background else 0
        dice_foreground = dice_per_class[start_class:]

        if self.class_weights is not None:
            weights = self.class_weights[start_class:]
            # Normalise weights
            weights = weights / (weights.sum() + 1e-8)
            weighted_dice = (dice_foreground * weights).sum()
            return 1.0 - weighted_dice
        else:
            return 1.0 - dice_foreground.mean()


# ═══════════════════════════════════════════════════════════════════════
#  Boundary Loss
# ═══════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    """Boundary-aware loss that penalises predicted boundaries that
    deviate from ground-truth boundaries.

    Computes the distance transform of the ground-truth boundary,
    then weights the prediction error by proximity to the boundary.
    Regions far from any boundary contribute almost nothing; boundary
    regions contribute heavily.

    This is essential for surgical limbus detection and, when using
    4-class mode, also helps with precise ring boundary delineation.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (3 or 4).
    theta0 : float
        Distance (in pixels) at which the boundary weight drops to 0.5.
    boundary_classes : list of int or None
        Which classes to compute boundary weights for.  Defaults to
        all foreground classes ``[1, 2]`` or ``[1, 2, 3]``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        theta0: float = 3.0,
        boundary_classes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.theta0 = theta0

        if boundary_classes is not None:
            self.boundary_classes = boundary_classes
        else:
            self.boundary_classes = list(range(1, num_classes))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor [B, C, H, W]
        targets : Tensor [B, H, W]  class indices

        Returns
        -------
        Tensor  scalar loss value
        """
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)

        targets_oh = F.one_hot(
            targets_clamped, self.num_classes
        ).permute(0, 3, 1, 2).float()

        # compute boundary distance weight maps (on CPU, then move)
        device = logits.device
        weight_maps = self._boundary_weights(
            targets_oh.detach().cpu().numpy()
        )
        weight_maps = torch.from_numpy(weight_maps).to(device)

        # weighted cross-entropy per pixel
        log_probs = torch.log_softmax(logits, dim=1)
        pixel_loss = -(targets_oh * log_probs).sum(dim=1)  # [B, H, W]

        weighted = (pixel_loss * weight_maps).mean()
        return weighted

    def _boundary_weights(self, targets_oh: np.ndarray) -> np.ndarray:
        """Compute per-pixel boundary proximity weights.

        Parameters
        ----------
        targets_oh : np.ndarray [B, C, H, W]

        Returns
        -------
        np.ndarray [B, H, W]
        """
        B, C, H, W = targets_oh.shape
        weights = np.ones((B, H, W), dtype=np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        for b in range(B):
            boundary = np.zeros((H, W), dtype=np.uint8)
            for c in self.boundary_classes:
                if c >= C:
                    continue
                mask_c = (targets_oh[b, c] > 0.5).astype(np.uint8)
                # morphological gradient = boundary
                grad = cv2.morphologyEx(
                    mask_c, cv2.MORPH_GRADIENT, kernel
                )
                boundary = np.maximum(boundary, grad)

            if boundary.sum() > 0:
                dist = cv2.distanceTransform(
                    1 - boundary, cv2.DIST_L2, 5
                ).astype(np.float32)
                # sigmoid-like weight: 1.0 at boundary, decays with distance
                w = 1.0 / (1.0 + (dist / self.theta0) ** 2)
                # scale so that boundary pixels get 5× weight,
                # far pixels get 1×
                weights[b] = 1.0 + 4.0 * w
            else:
                weights[b] = 1.0

        return weights


# ═══════════════════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance and hard negatives.

    Particularly useful when the suction ring class (3) is rare —
    focal loss down-weights easy (well-classified) pixels and focuses
    the gradient on hard pixels near class boundaries.

    ``FL(p) = -α · (1 - p)^γ · log(p)``

    Parameters
    ----------
    alpha : Tensor or list of float or None
        Per-class weighting factors.  When ``None``, uniform weights.
    gamma : float
        Focusing parameter.  ``γ = 0`` reduces to standard CE.
        ``γ = 2`` (default) strongly focuses on hard examples.
    num_classes : int
        Number of segmentation classes (3 or 4).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.register_buffer(
                "alpha",
                torch.ones(num_classes, dtype=torch.float32),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor [B, C, H, W]  raw logits
        targets : Tensor [B, H, W]  class indices (long)

        Returns
        -------
        Tensor  scalar loss value
        """
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)

        ce = F.cross_entropy(
            logits, targets_clamped,
            weight=self.alpha,
            reduction="none",
        )  # [B, H, W]

        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ═══════════════════════════════════════════════════════════════════════
#  Weighted Cross-Entropy + Dice (Simple Combined Loss)
# ═══════════════════════════════════════════════════════════════════════

class WeightedCrossEntropyDiceLoss(nn.Module):
    """Combined weighted cross-entropy + Dice loss.

    A simpler alternative to ``CompositeLoss`` that omits the
    boundary component.  Useful for faster training or when
    boundary precision is less critical.

    Supports 3 or 4 classes.  Class weights handle the imbalance
    caused by the suction ring class appearing only in docked images.

    Parameters
    ----------
    class_weights : Tensor or None
        Per-class CE weights.
    dice_weight : float
        Relative weight of the Dice component.
    ce_weight : float
        Relative weight of the CE component.
    smooth : float
        Dice smoothing constant.
    num_classes : int
        Number of segmentation classes (3 or 4).
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.num_classes = num_classes

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            default_w = get_default_class_weights(num_classes)
            self.register_buffer(
                "class_weights",
                torch.tensor(default_w, dtype=torch.float32),
            )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor [B, C, H, W]  logits
        target : Tensor [B, H, W]  class indices

        Returns
        -------
        Tensor  scalar loss value
        """
        target_clamped = target.long().clamp(0, self.num_classes - 1)

        # Cross-entropy
        ce_loss = F.cross_entropy(
            pred, target_clamped, weight=self.class_weights,
        )

        # Dice loss (per-class, weighted average)
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(
            target_clamped, self.num_classes
        ).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dice_loss = torch.tensor(0.0, device=pred.device)
        for c in range(self.num_classes):
            p = pred_soft[:, c]
            t = target_one_hot[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (
                union + self.smooth
            )
            dice_loss = dice_loss + (1.0 - dice) * self.class_weights[c]

        dice_loss = dice_loss / (self.class_weights.sum() + 1e-8)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ═══════════════════════════════════════════════════════════════════════
#  Composite Loss (Main Loss Function)
# ═══════════════════════════════════════════════════════════════════════

class CompositeLoss(nn.Module):
    """Combined loss for surgical eye segmentation.

    ``loss = α·CE + β·Dice + γ·Boundary``

    Default weights::

        CE       = 0.3   (pixel-level accuracy)
        Dice     = 0.4   (region overlap, class-balanced)
        Boundary = 0.3   (boundary precision for limbus and ring)

    Supports both 3-class and 4-class configurations.  When
    ``num_classes=4``, the boundary loss automatically includes the
    ring class boundary, and the Dice loss is weighted to account for
    the ring class being sparse.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (3 or 4).
    ce_weight : float
        Weight for cross-entropy component.
    dice_weight : float
        Weight for Dice component.
    boundary_weight : float
        Weight for boundary component.
    class_weights : list of float or None
        Per-class weights for cross-entropy.  When ``None``, uses
        sensible defaults that down-weight background and up-weight
        the ring class.
    dice_class_weights : list of float or None
        Optional per-class weights for the Dice loss.  When ``None``,
        the Dice loss uses uniform weighting across foreground classes.
    use_focal : bool
        If True, replace the CE component with focal loss for better
        handling of class imbalance (useful when ring class is rare).
    focal_gamma : float
        Focal loss gamma parameter (only used when ``use_focal=True``).
    """

    def __init__(
        self,
        num_classes: int = 3,
        ce_weight: float = 0.3,
        dice_weight: float = 0.4,
        boundary_weight: float = 0.3,
        class_weights: Optional[List[float]] = None,
        dice_class_weights: Optional[List[float]] = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.use_focal = use_focal

        # ── Class weights for CE / Focal ──────────────────────────
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32)
        else:
            cw = torch.tensor(
                get_default_class_weights(num_classes),
                dtype=torch.float32,
            )

        # ── CE or Focal loss ──────────────────────────────────────
        if use_focal:
            self.ce = FocalLoss(
                alpha=cw,
                gamma=focal_gamma,
                num_classes=num_classes,
            )
        else:
            self.ce = nn.CrossEntropyLoss(weight=cw)

        # ── Dice loss ─────────────────────────────────────────────
        self.dice = DiceLoss(
            num_classes=num_classes,
            per_class_weights=dice_class_weights,
        )

        # ── Boundary loss ─────────────────────────────────────────
        # Include all foreground classes (1, 2) or (1, 2, 3) for
        # boundary detection
        boundary_classes = list(range(1, num_classes))
        self.boundary = BoundaryLoss(
            num_classes=num_classes,
            boundary_classes=boundary_classes,
        )

        logger.info(
            "CompositeLoss: %d classes, weights CE=%.2f Dice=%.2f "
            "Boundary=%.2f, class_weights=%s, focal=%s",
            num_classes, ce_weight, dice_weight, boundary_weight,
            [round(w, 2) for w in cw.tolist()],
            use_focal,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the composite loss.

        Parameters
        ----------
        logits : Tensor [B, C, H, W]  raw logits
        targets : Tensor [B, H, W]  class indices (long)

        Returns
        -------
        Tensor  scalar loss value
        """
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)

        if self.use_focal:
            loss_ce = self.ce(logits, targets_clamped)
        else:
            loss_ce = self.ce(logits, targets_clamped)

        loss_dice = self.dice(logits, targets_clamped)
        loss_bnd = self.boundary(logits, targets_clamped)

        total = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.boundary_weight * loss_bnd
        )
        return total

    def forward_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """Return individual loss components for logging.

        Parameters
        ----------
        logits : Tensor [B, C, H, W]
        targets : Tensor [B, H, W]

        Returns
        -------
        dict
            Keys: ``"total"``, ``"ce"``, ``"dice"``, ``"boundary"``
            — each a detached scalar tensor.
        """
        targets_clamped = targets.long().clamp(0, self.num_classes - 1)

        if self.use_focal:
            loss_ce = self.ce(logits, targets_clamped)
        else:
            loss_ce = self.ce(logits, targets_clamped)

        loss_dice = self.dice(logits, targets_clamped)
        loss_bnd = self.boundary(logits, targets_clamped)

        total = (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.boundary_weight * loss_bnd
        )
        return {
            "total": total,
            "ce": loss_ce.detach(),
            "dice": loss_dice.detach(),
            "boundary": loss_bnd.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Factory Function
# ═══════════════════════════════════════════════════════════════════════

def create_loss(
    num_classes: int = 3,
    class_weights: Optional[List[float]] = None,
    loss_type: str = "composite",
    use_focal: bool = False,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory function to create the appropriate loss module.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (3 or 4).
    class_weights : list of float or None
        Per-class weights.  ``None`` uses defaults.
    loss_type : str
        ``"composite"`` (CE + Dice + Boundary),
        ``"ce_dice"`` (CE + Dice only),
        ``"focal_dice"`` (Focal + Dice),
        ``"ce"`` (CE only).
    use_focal : bool
        Replace CE with focal loss in composite mode.
    focal_gamma : float
        Focal loss gamma.

    Returns
    -------
    nn.Module
        Loss module.
    """
    if class_weights is None:
        class_weights = get_default_class_weights(num_classes)

    if loss_type == "composite":
        return CompositeLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
        )
    elif loss_type == "ce_dice":
        cw = torch.tensor(class_weights, dtype=torch.float32)
        return WeightedCrossEntropyDiceLoss(
            class_weights=cw,
            num_classes=num_classes,
        )
    elif loss_type == "focal_dice":
        cw = torch.tensor(class_weights, dtype=torch.float32)
        return WeightedCrossEntropyDiceLoss(
            class_weights=cw,
            num_classes=num_classes,
        )
    elif loss_type == "ce":
        cw = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=cw)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")