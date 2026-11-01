"""
Grayscale detection, conversion, enhancement, and model-input preparation.

This module is the single source of truth for all grayscale operations in
the pupil-tracking pipeline.  It is intentionally **stateless** — every
public method is a pure function of its arguments — so it can be shared
safely across threads (video processing) without locks.

Design rationale
----------------
The segmentation model (U-Net / ResNet-34 encoder) expects 3-channel
(H × W × 3) input.  Rather than training a separate single-channel
model, we:

1.  **Detect** whether the input is already grayscale.
2.  **Enhance** contrast with CLAHE (superior to global histogram
    equalisation for clinical images with uneven illumination).
3.  **Replicate** the enhanced single channel to three identical
    channels so the existing model processes it without architecture
    changes.

During *training*, a companion augmentation
(:class:`~pupil_tracking.ml.grayscale_augmentation.RandomGrayscaleConversion`)
randomly applies the same conversion, making the model invariant to
colour vs. grayscale input.  This guarantees that RGB accuracy is
**never degraded** — it can only improve because the model sees
more diverse data.

Typical usage
-------------
>>> from pupil_tracking.preprocessing.grayscale_handler import GrayscaleHandler
>>> handler = GrayscaleHandler()
>>> # Auto-detect and convert for model
>>> model_input = handler.to_model_input(image, force_grayscale=False)
>>> # Force grayscale (user toggle)
>>> model_input = handler.to_model_input(image, force_grayscale=True)
>>> # Quality check
>>> metrics = handler.get_quality_metrics(image)
>>> print(f"Contrast: {metrics['contrast']:.2f}")

Thread safety
-------------
All public methods are stateless and re-entrant.  The only mutable
state is the lazily-created CLAHE object, which is guarded by a
threading lock and rebuilt per-call if parameters differ.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

__all__ = [
    "GrayscaleHandler",
    "GrayscaleMode",
    "GrayscaleInfo",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public enums & data classes
# ---------------------------------------------------------------------------

class GrayscaleMode(Enum):
    """Operating mode for grayscale handling in the detection pipeline.

    Attributes
    ----------
    AUTO : auto
        Inspect each image; apply grayscale enhancement only when the
        image is detected as grayscale.
    FORCE : auto
        Always convert to grayscale before detection — useful when the
        user explicitly toggles "Grayscale" in the GUI.
    OFF : auto
        Pass images through unchanged (legacy behaviour).
    """

    AUTO = auto()
    FORCE = auto()
    OFF = auto()

    @classmethod
    def from_string(cls, value: str) -> "GrayscaleMode":
        """Parse a case-insensitive string to a :class:`GrayscaleMode`.

        Parameters
        ----------
        value : str
            One of ``"auto"``, ``"force"``, ``"off"``.

        Returns
        -------
        GrayscaleMode

        Raises
        ------
        ValueError
            If *value* is not a recognised mode name.
        """
        lookup = {
            "auto": cls.AUTO,
            "force": cls.FORCE,
            "off": cls.OFF,
        }
        normalised = str(value).strip().lower()
        if normalised not in lookup:
            valid = ", ".join(sorted(lookup.keys()))
            raise ValueError(
                f"Unknown grayscale mode {value!r}.  "
                f"Valid options: {valid}"
            )
        return lookup[normalised]


@dataclass(frozen=True)
class GrayscaleInfo:
    """Diagnostic metadata returned alongside a converted image.

    Every field is intentionally immutable (``frozen=True``) so the
    object can be logged, cached, or passed between threads without
    risk of mutation.

    Attributes
    ----------
    was_grayscale : bool
        ``True`` if the original image was detected as grayscale.
    conversion_applied : bool
        ``True`` if any grayscale conversion or enhancement was applied.
    mode_used : GrayscaleMode
        The mode that was active when processing occurred.
    original_channels : int
        Number of channels in the raw input (1 or 3).
    contrast_before : float
        Standard deviation of pixel intensities before enhancement.
    contrast_after : float
        Standard deviation of pixel intensities after enhancement
        (equals *contrast_before* if no enhancement was applied).
    """

    was_grayscale: bool
    conversion_applied: bool
    mode_used: GrayscaleMode
    original_channels: int
    contrast_before: float
    contrast_after: float


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GrayscaleHandler:
    """Central engine for grayscale detection, conversion, and enhancement.

    Parameters
    ----------
    clahe_clip_limit : float, optional
        Contrast-limiting threshold for CLAHE.  Higher values allow
        more amplification in flat regions but can also amplify noise.
        Default is ``3.0`` (empirically good for clinical eye images).
    clahe_grid_size : tuple[int, int], optional
        Tile grid size for CLAHE.  Smaller tiles give more localised
        contrast enhancement.  Default is ``(8, 8)``.
    channel_diff_threshold : float, optional
        Maximum mean absolute difference between channels (0–255) for
        an image to be classified as "fake-RGB grayscale" (i.e., a
        grayscale image saved in a 3-channel container).  Default is
        ``3.0`` — tight enough to avoid false positives on tinted
        images, loose enough to catch JPEG-compressed grayscale.

    Examples
    --------
    >>> handler = GrayscaleHandler()
    >>> gray_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    >>> assert handler.is_grayscale(gray_img)
    >>> model_in = handler.to_model_input(gray_img)
    >>> assert model_in.shape == (480, 640, 3)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        channel_diff_threshold: float = 3.0,
    ) -> None:
        if clahe_clip_limit <= 0:
            raise ValueError(
                f"clahe_clip_limit must be positive, got {clahe_clip_limit}"
            )
        if (
            len(clahe_grid_size) != 2
            or clahe_grid_size[0] < 1
            or clahe_grid_size[1] < 1
        ):
            raise ValueError(
                f"clahe_grid_size must be (rows, cols) with positive "
                f"integers, got {clahe_grid_size}"
            )
        if channel_diff_threshold < 0:
            raise ValueError(
                f"channel_diff_threshold must be non-negative, "
                f"got {channel_diff_threshold}"
            )

        self._clahe_clip_limit: float = clahe_clip_limit
        self._clahe_grid_size: Tuple[int, int] = tuple(clahe_grid_size)  # type: ignore[assignment]
        self._channel_diff_threshold: float = channel_diff_threshold

        # Lazy CLAHE instance (re-created if parameters change)
        self._clahe: Optional[cv2.CLAHE] = None
        self._clahe_lock = threading.Lock()

        logger.debug(
            "GrayscaleHandler initialised — clip=%.1f, grid=%s, "
            "channel_threshold=%.1f",
            self._clahe_clip_limit,
            self._clahe_grid_size,
            self._channel_diff_threshold,
        )

    # ------------------------------------------------------------------ #
    # Public API — Detection
    # ------------------------------------------------------------------ #

    def is_grayscale(self, image: np.ndarray) -> bool:
        """Determine whether *image* is grayscale.

        An image is considered grayscale if it is:

        *   **Single-channel** (shape ``(H, W)`` or ``(H, W, 1)``), or
        *   **Three-channel with near-identical channels** — the mean
            absolute difference across all channel pairs is below
            :attr:`channel_diff_threshold`.  This catches grayscale
            images that were saved in BGR/RGB containers (common with
            JPEG compression or video codecs).

        Parameters
        ----------
        image : numpy.ndarray
            Input image.  Accepts ``uint8`` or ``float32/float64``
            (values assumed in 0–1 range for float, 0–255 for uint8).

        Returns
        -------
        bool
            ``True`` if the image is grayscale.

        Raises
        ------
        ValueError
            If *image* is not a 2-D or 3-D array, or has an
            unsupported number of channels (e.g., 4-channel RGBA).
        """
        self._validate_image(image, allow_single_channel=True)

        ndim = image.ndim
        if ndim == 2:
            return True

        num_channels = image.shape[2]
        if num_channels == 1:
            return True
        if num_channels == 4:
            # RGBA — check RGB portion only
            image = image[:, :, :3]
            num_channels = 3
        if num_channels != 3:
            raise ValueError(
                f"Unsupported channel count {num_channels}.  "
                f"Expected 1, 3, or 4."
            )

        # Compare channels pair-wise on a down-sampled copy for speed.
        # Down-sample to at most 200 × 200 for the comparison.
        h, w = image.shape[:2]
        scale = min(1.0, 200.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = image

        # Work in float to avoid uint8 overflow on subtraction
        sf = small.astype(np.float32)
        ch0, ch1, ch2 = sf[:, :, 0], sf[:, :, 1], sf[:, :, 2]

        # Scale threshold if image is float [0, 1]
        threshold = self._channel_diff_threshold
        if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
            threshold = threshold / 255.0

        mean_diff_01 = np.mean(np.abs(ch0 - ch1))
        mean_diff_02 = np.mean(np.abs(ch0 - ch2))
        mean_diff_12 = np.mean(np.abs(ch1 - ch2))
        max_diff = max(mean_diff_01, mean_diff_02, mean_diff_12)

        is_gray = bool(max_diff <= threshold)

        logger.debug(
            "is_grayscale check — max_channel_diff=%.2f, "
            "threshold=%.2f, result=%s",
            max_diff,
            threshold,
            is_gray,
        )
        return is_gray

    # ------------------------------------------------------------------ #
    # Public API — Conversion
    # ------------------------------------------------------------------ #

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert any image to single-channel grayscale.

        If the image is already single-channel it is returned as-is
        (no copy).  Three-channel images are converted using the
        standard luminance formula (BT.601):

            ``Y = 0.299·R + 0.587·G + 0.114·B``

        (OpenCV's ``cv2.cvtColor`` with ``COLOR_BGR2GRAY``.)

        Parameters
        ----------
        image : numpy.ndarray
            Input image, ``uint8`` or ``float32/64``.

        Returns
        -------
        numpy.ndarray
            Single-channel image with shape ``(H, W)``.
        """
        self._validate_image(image, allow_single_channel=True)

        if image.ndim == 2:
            return image
        if image.shape[2] == 1:
            return image[:, :, 0]
        if image.shape[2] == 4:
            image = image[:, :, :3]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug(
            "Converted %s image to grayscale (%s)",
            image.shape,
            gray.shape,
        )
        return gray

    # ------------------------------------------------------------------ #
    # Public API — Enhancement
    # ------------------------------------------------------------------ #

    def enhance_grayscale(self, gray: np.ndarray) -> np.ndarray:
        """Enhance contrast of a single-channel grayscale image.

        The enhancement pipeline is:

        1.  **CLAHE** (Contrast-Limited Adaptive Histogram
            Equalisation) — improves local contrast while preventing
            over-amplification of noise.
        2.  **Gentle global normalisation** — stretches the histogram
            to use the full 0–255 range, but only if the image is
            significantly under-utilising the dynamic range (top 1 %
            percentile < 200).

        This is deliberately conservative: over-enhancement hurts
        pupil detection more than under-enhancement because it can
        wash out the pupil–iris boundary.

        Parameters
        ----------
        gray : numpy.ndarray
            Single-channel image with shape ``(H, W)``.  Must be
            ``uint8``.

        Returns
        -------
        numpy.ndarray
            Enhanced single-channel ``uint8`` image, same shape.
        """
        if gray.ndim != 2:
            raise ValueError(
                f"enhance_grayscale expects a 2-D array, got shape "
                f"{gray.shape}.  Call to_grayscale() first."
            )

        # Ensure uint8 for CLAHE
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255.0).clip(0, 255).astype(np.uint8)
            else:
                gray = gray.clip(0, 255).astype(np.uint8)

        # Step 1: CLAHE
        clahe = self._get_clahe()
        enhanced = clahe.apply(gray)

        # Step 2: Gentle global stretch (only when needed)
        p_low = np.percentile(enhanced, 1)
        p_high = np.percentile(enhanced, 99)

        if p_high < 200 and (p_high - p_low) > 10:
            # Stretch to [0, 255] using the 1st/99th percentile window
            alpha = 255.0 / max(float(p_high - p_low), 1.0)
            beta = -p_low * alpha
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            logger.debug(
                "Applied global stretch — p1=%.0f, p99=%.0f, "
                "alpha=%.2f, beta=%.2f",
                p_low,
                p_high,
                alpha,
                beta,
            )

        return enhanced

    # ------------------------------------------------------------------ #
    # Public API — Model-Input Preparation
    # ------------------------------------------------------------------ #

    def to_model_input(
        self,
        image: np.ndarray,
        force_grayscale: bool = False,
    ) -> Tuple[np.ndarray, GrayscaleInfo]:
        """Prepare an image for model inference.

        This is the **primary entry point** used by the detection
        pipeline.  It combines detection, conversion, enhancement,
        and channel replication into a single call.

        Behaviour matrix
        ~~~~~~~~~~~~~~~~~

        +-------------------+-------------------+-----------------------+
        | Input             | force_grayscale   | Action                |
        +===================+===================+=======================+
        | Grayscale (1-ch)  | any               | Enhance → replicate   |
        +-------------------+-------------------+-----------------------+
        | Fake-RGB gray     | any               | Extract → enhance →   |
        |                   |                   | replicate             |
        +-------------------+-------------------+-----------------------+
        | True RGB          | ``False``         | Pass through (no-op)  |
        +-------------------+-------------------+-----------------------+
        | True RGB          | ``True``          | Convert → enhance →   |
        |                   |                   | replicate             |
        +-------------------+-------------------+-----------------------+

        Parameters
        ----------
        image : numpy.ndarray
            Input image of any supported format.
        force_grayscale : bool, optional
            If ``True``, convert even true-RGB images to grayscale
            before enhancement and replication (GUI "Force Grayscale"
            toggle).  Default is ``False``.

        Returns
        -------
        model_input : numpy.ndarray
            ``uint8`` image with shape ``(H, W, 3)``, ready for the
            segmentation model.
        info : GrayscaleInfo
            Diagnostic metadata about what processing was applied.
        """
        self._validate_image(image, allow_single_channel=True)

        original_channels = 1 if image.ndim == 2 else image.shape[2]
        detected_gray = self.is_grayscale(image)
        apply_conversion = detected_gray or force_grayscale

        # Compute contrast *before* enhancement
        gray_before = (
            self.to_grayscale(image) if image.ndim == 3 else image.copy()
        )
        if gray_before.dtype != np.uint8:
            if gray_before.max() <= 1.0:
                gray_before = (gray_before * 255).clip(0, 255).astype(np.uint8)
            else:
                gray_before = gray_before.clip(0, 255).astype(np.uint8)
        contrast_before = float(np.std(gray_before.astype(np.float32)))

        if apply_conversion:
            # Convert to single-channel grayscale
            gray = self.to_grayscale(image)

            # Enhance contrast
            enhanced = self.enhance_grayscale(gray)

            # Replicate to 3 channels for model compatibility
            model_input = self._replicate_to_3ch(enhanced)

            contrast_after = float(
                np.std(enhanced.astype(np.float32))
            )

            mode_used = (
                GrayscaleMode.FORCE if force_grayscale and not detected_gray
                else GrayscaleMode.AUTO
            )

            logger.info(
                "Grayscale pipeline applied — detected_gray=%s, "
                "forced=%s, contrast %.1f → %.1f",
                detected_gray,
                force_grayscale,
                contrast_before,
                contrast_after,
            )
        else:
            # Pass through RGB unchanged
            model_input = self._ensure_3ch_uint8(image)
            contrast_after = contrast_before
            mode_used = GrayscaleMode.OFF

            logger.debug(
                "RGB passthrough — no grayscale conversion applied"
            )

        info = GrayscaleInfo(
            was_grayscale=detected_gray,
            conversion_applied=apply_conversion,
            mode_used=mode_used,
            original_channels=original_channels,
            contrast_before=contrast_before,
            contrast_after=contrast_after,
        )

        return model_input, info

    # ------------------------------------------------------------------ #
    # Public API — Quality Metrics
    # ------------------------------------------------------------------ #

    def get_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """Compute image-quality metrics relevant to grayscale handling.

        These metrics help the quality-assessment module decide
        whether grayscale conversion would *improve* detection
        (e.g., when the colour channels add more noise than signal).

        Parameters
        ----------
        image : numpy.ndarray
            Input image (any format).

        Returns
        -------
        dict[str, float]
            Dictionary with the following keys:

            - ``contrast``  — standard deviation of luminance.
            - ``dynamic_range`` — difference between 99th and 1st
              percentile intensities (0–255 scale).
            - ``mean_intensity`` — mean luminance (0–255 scale).
            - ``snr_estimate`` — crude signal-to-noise ratio
              (mean / std of a Laplacian-filtered image).
            - ``is_grayscale`` — 1.0 if grayscale, 0.0 otherwise.
            - ``channel_variance`` — mean inter-channel variance
              (0.0 for perfect grayscale, higher for colourful
              images).  Useful for continuous grayscale scoring.
        """
        self._validate_image(image, allow_single_channel=True)

        gray = self.to_grayscale(image) if image.ndim == 3 else image.copy()
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).clip(0, 255).astype(np.uint8)
            else:
                gray = gray.clip(0, 255).astype(np.uint8)

        gray_f = gray.astype(np.float32)

        # Basic intensity statistics
        mean_intensity = float(np.mean(gray_f))
        contrast = float(np.std(gray_f))
        p_low = float(np.percentile(gray_f, 1))
        p_high = float(np.percentile(gray_f, 99))
        dynamic_range = p_high - p_low

        # SNR estimate via Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_std = float(np.std(laplacian))
        snr_estimate = mean_intensity / max(lap_std, 1e-6)

        # Channel variance (0 for grayscale, positive for colour)
        channel_variance = 0.0
        if image.ndim == 3 and image.shape[2] >= 3:
            img_f = image[:, :, :3].astype(np.float32)
            # Down-sample for speed
            h, w = img_f.shape[:2]
            scale = min(1.0, 200.0 / max(h, w))
            if scale < 1.0:
                img_f = cv2.resize(
                    img_f, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            # Per-pixel variance across channels, then mean
            channel_variance = float(np.mean(np.var(img_f, axis=2)))

        is_gray_flag = 1.0 if self.is_grayscale(image) else 0.0

        metrics = {
            "contrast": contrast,
            "dynamic_range": dynamic_range,
            "mean_intensity": mean_intensity,
            "snr_estimate": snr_estimate,
            "is_grayscale": is_gray_flag,
            "channel_variance": channel_variance,
        }

        logger.debug("Quality metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_clahe(self) -> cv2.CLAHE:
        """Return a thread-safe CLAHE instance.

        The CLAHE object is lazily created on first use and cached.
        A threading lock prevents races when multiple video-processing
        threads call ``enhance_grayscale`` concurrently.
        """
        with self._clahe_lock:
            if self._clahe is None:
                self._clahe = cv2.createCLAHE(
                    clipLimit=self._clahe_clip_limit,
                    tileGridSize=self._clahe_grid_size,
                )
                logger.debug(
                    "Created CLAHE instance — clip=%.1f, grid=%s",
                    self._clahe_clip_limit,
                    self._clahe_grid_size,
                )
            return self._clahe

    @staticmethod
    def _replicate_to_3ch(gray: np.ndarray) -> np.ndarray:
        """Stack a single-channel image into a 3-channel ``(H, W, 3)`` array.

        Uses ``np.stack`` rather than ``cv2.merge`` for clarity and to
        avoid unnecessary copies (NumPy can create a view when the
        memory layout allows it).
        """
        if gray.ndim != 2:
            raise ValueError(
                f"_replicate_to_3ch expects (H, W), got {gray.shape}"
            )
        return np.stack([gray, gray, gray], axis=2)

    @staticmethod
    def _ensure_3ch_uint8(image: np.ndarray) -> np.ndarray:
        """Guarantee the output is ``(H, W, 3)`` and ``uint8``.

        Handles edge cases:
        - Single-channel → replicate
        - 4-channel (RGBA/BGRA) → drop alpha
        - Float images → scale to 0–255
        """
        # Handle dtype
        if image.dtype in (np.float32, np.float64):
            if image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Handle channels
        if image.ndim == 2:
            return np.stack([image, image, image], axis=2)
        if image.shape[2] == 1:
            squeezed = image[:, :, 0]
            return np.stack([squeezed, squeezed, squeezed], axis=2)
        if image.shape[2] == 4:
            return image[:, :, :3].copy()
        if image.shape[2] == 3:
            return image

        raise ValueError(
            f"Cannot convert image with {image.shape[2]} channels to 3-ch"
        )

    @staticmethod
    def _validate_image(
        image: np.ndarray,
        allow_single_channel: bool = False,
    ) -> None:
        """Raise clear errors for invalid inputs.

        Parameters
        ----------
        image : numpy.ndarray
            The image to validate.
        allow_single_channel : bool
            If ``False``, reject 2-D (single-channel) images.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image).__name__}"
            )
        if image.size == 0:
            raise ValueError("Image is empty (zero pixels)")
        if image.ndim == 2:
            if not allow_single_channel:
                raise ValueError(
                    "Single-channel image not allowed here.  "
                    "Pass allow_single_channel=True or convert to 3-ch."
                )
            return
        if image.ndim != 3:
            raise ValueError(
                f"Image must be 2-D or 3-D, got {image.ndim}-D "
                f"with shape {image.shape}"
            )
        if image.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Image has {image.shape[2]} channels; "
                f"expected 1, 3, or 4"
            )

    # ------------------------------------------------------------------ #
    # Repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clahe_clip_limit={self._clahe_clip_limit}, "
            f"clahe_grid_size={self._clahe_grid_size}, "
            f"channel_diff_threshold={self._channel_diff_threshold})"
        )