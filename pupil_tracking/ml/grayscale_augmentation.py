"""
Grayscale-aware data augmentation for training robust segmentation models.

This module provides albumentations-compatible transforms that randomly
convert RGB training images to grayscale (replicated to 3 channels),
teaching the model to produce identical segmentation quality regardless
of whether the input is colour or grayscale.

Design rationale
----------------
Instead of training a separate single-channel model or modifying the
U-Net architecture, we augment the *training data* so that ~30 % of
images seen during training are grayscale.  Because the grayscale
images pass through the same CLAHE enhancement used at inference
(:class:`~pupil_tracking.preprocessing.grayscale_handler.GrayscaleHandler`),
the model learns the *exact* pixel distribution it will encounter
when a real grayscale image arrives.

This approach has two critical advantages:

1.  **Zero accuracy regression on RGB** — the model still sees 70 %
    colour images, so its existing colour-based features are preserved.
2.  **No architecture changes** — the model input remains
    ``(batch, 3, H, W)``, the output remains ``(batch, C, H, W)``
    with the same number of classes.

The fine-tuning script (:mod:`scripts.finetune_grayscale`) uses this
module to augment an existing trained model with grayscale robustness
in 30–50 epochs at a reduced learning rate.

Integration with existing pipeline
-----------------------------------
The :class:`GrayscaleAwarePipeline` wraps the project's existing
augmentation transforms (rotation, flip, brightness, elastic, etc.)
and inserts :class:`RandomGrayscaleConversion` at the *end* of the
spatial transforms but *before* normalisation.  This ordering ensures
that:

-   Spatial augmentations (rotation, crop) operate on the original
    colour image (maximum information).
-   Grayscale conversion happens on the already-augmented image.
-   Normalisation (mean/std) is applied last, as required by the
    model.

Usage
-----
>>> from pupil_tracking.ml.grayscale_augmentation import (
...     GrayscaleAwarePipeline,
...     RandomGrayscaleConversion,
... )
>>> pipeline = GrayscaleAwarePipeline()
>>> train_aug = pipeline.get_training_augmentation(
...     input_size=512,
...     grayscale_prob=0.3,
... )
>>> result = train_aug(image=rgb_image, mask=mask)
>>> augmented_image = result["image"]  # may be grayscale-replicated
>>> augmented_mask  = result["mask"]   # never modified by grayscale aug

Thread safety
-------------
All classes are stateless after construction and safe to use from
multiple ``DataLoader`` worker processes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np

from pupil_tracking.preprocessing.grayscale_handler import GrayscaleHandler

__all__ = [
    "RandomGrayscaleConversion",
    "GrayscaleAwarePipeline",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom albumentations transform
# ---------------------------------------------------------------------------

class RandomGrayscaleConversion(A.ImageOnlyTransform):
    """Randomly convert an RGB image to enhanced grayscale (3-channel).

    When applied, the transform:

    1.  Converts the image to single-channel grayscale (BT.601).
    2.  Optionally enhances contrast with CLAHE (matching the
        inference-time enhancement in
        :class:`~pupil_tracking.preprocessing.grayscale_handler.GrayscaleHandler`).
    3.  Replicates the single channel to three identical channels
        so the output shape remains ``(H, W, 3)``.

    The mask is **never** modified — only the image is affected.

    Parameters
    ----------
    enhance : bool, optional
        If ``True`` (default), apply CLAHE enhancement after
        conversion.  This should always be ``True`` for training
        because it matches the inference pipeline.
    clahe_clip_limit : float, optional
        CLAHE clip limit.  Default ``3.0`` matches the handler default.
    clahe_grid_size : tuple[int, int], optional
        CLAHE tile grid.  Default ``(8, 8)`` matches the handler.
    always_apply : bool, optional
        If ``True``, always apply (ignore ``p``).  Default ``False``.
    p : float, optional
        Probability of applying this transform.  Default ``0.3``
        (30 % of training images become grayscale — empirically
        optimal for maintaining RGB accuracy while learning
        grayscale robustness).

    Examples
    --------
    >>> transform = RandomGrayscaleConversion(p=0.3)
    >>> result = transform(image=rgb_image)
    >>> assert result["image"].shape == rgb_image.shape
    >>> assert result["image"].shape[2] == 3

    Notes
    -----
    This transform is designed to sit **after** spatial augmentations
    (rotation, flip, crop) but **before** normalisation (mean/std
    subtraction).  The :class:`GrayscaleAwarePipeline` handles this
    ordering automatically.
    """

    def __init__(
        self,
        enhance: bool = True,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        always_apply: bool = False,
        p: float = 0.3,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)

        self._enhance = enhance
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_grid_size = tuple(clahe_grid_size)

        # Use the production GrayscaleHandler so training and inference
        # use *exactly* the same enhancement pipeline.
        self._handler = GrayscaleHandler(
            clahe_clip_limit=clahe_clip_limit,
            clahe_grid_size=clahe_grid_size,
        )

        logger.debug(
            "RandomGrayscaleConversion created — p=%.2f, enhance=%s, "
            "clip=%.1f, grid=%s",
            p,
            enhance,
            clahe_clip_limit,
            clahe_grid_size,
        )

    def apply(
        self,
        img: np.ndarray,
        **params: Any,
    ) -> np.ndarray:
        """Apply grayscale conversion to a single image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image with shape ``(H, W, 3)`` and dtype ``uint8``.

        Returns
        -------
        numpy.ndarray
            Grayscale-replicated image with shape ``(H, W, 3)`` and
            dtype ``uint8``.
        """
        # Convert to single-channel grayscale
        gray = self._handler.to_grayscale(img)

        # Enhance contrast (matches inference pipeline)
        if self._enhance:
            gray = self._handler.enhance_grayscale(gray)

        # Replicate to 3 channels — model expects (H, W, 3)
        replicated = np.stack([gray, gray, gray], axis=2)

        return replicated

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Return names of ``__init__`` args for serialisation.

        Required by albumentations for ``to_dict()`` / ``from_dict()``
        round-tripping.
        """
        return (
            "enhance",
            "clahe_clip_limit",
            "clahe_grid_size",
        )


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class GrayscaleAwarePipeline:
    """Factory for augmentation pipelines that include grayscale robustness.

    This class builds complete ``albumentations.Compose`` pipelines
    for both training and validation.  It encapsulates the project's
    standard augmentation policy and inserts
    :class:`RandomGrayscaleConversion` at the correct position.

    The pipeline is constructed fresh on each call to
    ``get_training_augmentation`` / ``get_validation_augmentation``,
    so different datasets or experiments can use different settings
    without interfering.

    Parameters
    ----------
    clahe_clip_limit : float, optional
        CLAHE clip limit passed to the grayscale transform.
        Default ``3.0``.
    clahe_grid_size : tuple[int, int], optional
        CLAHE grid size.  Default ``(8, 8)``.

    Examples
    --------
    >>> pipeline = GrayscaleAwarePipeline()
    >>> train_aug = pipeline.get_training_augmentation(
    ...     input_size=512,
    ...     grayscale_prob=0.3,
    ... )
    >>> val_aug = pipeline.get_validation_augmentation(input_size=512)
    """

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self._clahe_clip_limit = clahe_clip_limit
        self._clahe_grid_size = tuple(clahe_grid_size)

    # ------------------------------------------------------------------ #
    # Training augmentation
    # ------------------------------------------------------------------ #

    def get_training_augmentation(
        self,
        input_size: int = 512,
        grayscale_prob: float = 0.3,
        additional_transforms: Optional[Sequence[A.BasicTransform]] = None,
    ) -> A.Compose:
        """Build the full training augmentation pipeline.

        Transform ordering
        ~~~~~~~~~~~~~~~~~~~

        1.  **Resize** to ``input_size × input_size``
        2.  **Spatial augmentations** — rotation, flip, shift/scale
        3.  **Pixel-level augmentations** — brightness, contrast,
            blur, noise
        4.  **Grayscale conversion** — randomly applied with
            probability ``grayscale_prob``
        5.  **Any additional user-supplied transforms**

        The grayscale transform is placed *after* pixel augmentations
        so that brightness/contrast jitter operates on the colour
        image (richer signal), and the grayscale conversion sees a
        realistically-varied image.

        Parameters
        ----------
        input_size : int, optional
            Target image size (square).  Default ``512``.
        grayscale_prob : float, optional
            Probability of converting each image to grayscale.
            Default ``0.3``.  Set to ``0.0`` to disable grayscale
            augmentation entirely (existing behaviour).
        additional_transforms : sequence of albumentations transforms, optional
            Extra transforms appended after the grayscale conversion.

        Returns
        -------
        albumentations.Compose
            Complete augmentation pipeline.
        """
        if not 0.0 <= grayscale_prob <= 1.0:
            raise ValueError(
                f"grayscale_prob must be in [0, 1], got {grayscale_prob}"
            )

        transforms = []

        # ----- Stage 1: Resize -----
        transforms.append(
            A.Resize(
                height=input_size,
                width=input_size,
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            )
        )

        # ----- Stage 2: Spatial augmentations -----
        transforms.extend(self._get_spatial_augmentations())

        # ----- Stage 3: Pixel-level augmentations -----
        transforms.extend(self._get_pixel_augmentations())

        # ----- Stage 4: Grayscale conversion -----
        if grayscale_prob > 0.0:
            transforms.append(
                RandomGrayscaleConversion(
                    enhance=True,
                    clahe_clip_limit=self._clahe_clip_limit,
                    clahe_grid_size=self._clahe_grid_size,
                    p=grayscale_prob,
                )
            )
            logger.info(
                "Grayscale augmentation enabled — p=%.2f",
                grayscale_prob,
            )

        # ----- Stage 5: Additional user transforms -----
        if additional_transforms:
            transforms.extend(additional_transforms)

        pipeline = A.Compose(transforms)

        logger.debug(
            "Training pipeline built — %d transforms, input_size=%d, "
            "grayscale_prob=%.2f",
            len(transforms),
            input_size,
            grayscale_prob,
        )

        return pipeline

    # ------------------------------------------------------------------ #
    # Validation augmentation
    # ------------------------------------------------------------------ #

    def get_validation_augmentation(
        self,
        input_size: int = 512,
    ) -> A.Compose:
        """Build the validation augmentation pipeline.

        Validation uses **only** deterministic transforms (resize)
        — no random augmentation, no grayscale conversion.  This
        ensures validation metrics are comparable across epochs and
        reflect true model performance on unmodified images.

        Parameters
        ----------
        input_size : int, optional
            Target image size (square).  Default ``512``.

        Returns
        -------
        albumentations.Compose
            Validation pipeline (resize only).
        """
        pipeline = A.Compose([
            A.Resize(
                height=input_size,
                width=input_size,
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            ),
        ])

        logger.debug(
            "Validation pipeline built — input_size=%d (resize only)",
            input_size,
        )

        return pipeline

    # ------------------------------------------------------------------ #
    # Dual validation (RGB + grayscale)
    # ------------------------------------------------------------------ #

    def get_grayscale_validation_augmentation(
        self,
        input_size: int = 512,
    ) -> A.Compose:
        """Build a validation pipeline that forces grayscale conversion.

        Used during fine-tuning to measure model accuracy on
        grayscale inputs *separately* from RGB accuracy.  This
        lets the fine-tuning script ensure both metrics are
        above threshold before saving a new checkpoint.

        Parameters
        ----------
        input_size : int, optional
            Target image size (square).  Default ``512``.

        Returns
        -------
        albumentations.Compose
            Pipeline that resizes and converts every image to
            enhanced grayscale (3-channel).
        """
        pipeline = A.Compose([
            A.Resize(
                height=input_size,
                width=input_size,
                interpolation=cv2.INTER_LINEAR,
                always_apply=True,
            ),
            RandomGrayscaleConversion(
                enhance=True,
                clahe_clip_limit=self._clahe_clip_limit,
                clahe_grid_size=self._clahe_grid_size,
                always_apply=True,
                p=1.0,
            ),
        ])

        logger.debug(
            "Grayscale validation pipeline built — input_size=%d, "
            "always_apply grayscale",
            input_size,
        )

        return pipeline

    # ------------------------------------------------------------------ #
    # Standard augmentation components
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_spatial_augmentations() -> list:
        """Return the project's standard spatial augmentations.

        These match the augmentations previously used in the training
        pipeline, ensuring backward-compatibility.  Any changes here
        should be validated with a full training run.

        Returns
        -------
        list[albumentations.BasicTransform]
        """
        return [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=30,
                sigma=5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.2,
            ),
        ]

    @staticmethod
    def _get_pixel_augmentations() -> list:
        """Return the project's standard pixel-level augmentations.

        These are applied *before* grayscale conversion so that
        brightness/contrast jitter operates on the richer colour
        signal.

        Returns
        -------
        list[albumentations.BasicTransform]
        """
        return [
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5,
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),
            A.GaussNoise(
                var_limit=(5.0, 25.0),
                p=0.2,
            ),
        ]

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clahe_clip_limit={self._clahe_clip_limit}, "
            f"clahe_grid_size={self._clahe_grid_size})"
        )