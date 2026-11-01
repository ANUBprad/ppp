"""
U-Net with a pretrained ResNet-34 encoder for multi-class eye
segmentation.

Supports two class configurations:

* **3-class (legacy)**: background (0), pupil (1), iris (2).
* **4-class (ring-aware)**: background (0), pupil (1), iris (2),
  suction_ring (3).

The model auto-detects the class count when loading a checkpoint,
so existing 3-class models continue to work without modification.

Uses ``segmentation_models_pytorch`` for the backbone and adds
temperature scaling for probability calibration.

Usage
-----
>>> from pupil_tracking.ml.architecture import (
...     EyeSegmentationModel, create_model, load_model, get_class_names,
... )
>>>
>>> # Create a new 4-class model
>>> model = create_model(num_classes=4, pretrained=True, device="cuda")
>>>
>>> # Load an existing checkpoint (auto-detects 3 or 4 classes)
>>> model = load_model("models/best_model.pth", device="cuda")
>>> print(model.num_classes, model.class_names)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Dict, List

# PyTorch is required for this module (training and PyTorch inference)
# In production, ONNX Runtime is used instead via onnx_inference.py
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

try:
    import segmentation_models_pytorch as smp
    _HAS_SMP = True
except ImportError:
    smp = None
    _HAS_SMP = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Class Definitions
# ═══════════════════════════════════════════════════════════════════════

CLASSES_3: Dict[int, str] = {
    0: "background",
    1: "pupil",
    2: "iris",
}

CLASSES_4: Dict[int, str] = {
    0: "background",
    1: "pupil",
    2: "iris",
    3: "suction_ring",
}

_CLASS_MAPS: Dict[int, Dict[int, str]] = {
    3: CLASSES_3,
    4: CLASSES_4,
}


def get_class_names(num_classes: int) -> Dict[int, str]:
    """Return class-index → name mapping for *num_classes*.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (3 or 4).

    Returns
    -------
    dict
        ``{0: "background", 1: "pupil", …}``.
    """
    return dict(_CLASS_MAPS.get(num_classes, CLASSES_3))


def get_class_colours(num_classes: int) -> Dict[int, tuple]:
    """Return class-index → BGR colour mapping for visualisation.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (3 or 4).

    Returns
    -------
    dict
        ``{0: (0,0,0), 1: (0,0,255), …}`` in BGR.
    """
    colours = {
        0: (0, 0, 0),        # background — black
        1: (0, 0, 255),      # pupil — red
        2: (255, 0, 0),      # iris — blue
    }
    if num_classes >= 4:
        colours[3] = (0, 200, 0)  # suction_ring — green
    return colours


# ═══════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════

# Dynamically choose the base class so the module can be imported
# even when PyTorch is not installed (e.g. ONNX-only production).
_BaseClass = nn.Module if (_HAS_TORCH and nn is not None) else object


class EyeSegmentationModel(_BaseClass):
    """Multi-class eye segmentation model.

    Supported class configurations::

        3-class (legacy):      0=background  1=pupil  2=iris
        4-class (ring-aware):  0=background  1=pupil  2=iris  3=suction_ring

    Parameters
    ----------
    encoder : str
        Backbone encoder name (default ``"resnet34"``).
    num_classes : int
        Number of output segmentation classes (3 or 4).
    pretrained : bool
        Use ImageNet-pretrained encoder weights.
    in_channels : int
        Number of input image channels (default 3 = RGB).

    Raises
    ------
    ImportError
        If PyTorch or segmentation_models_pytorch is not installed.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        num_classes: int = 3,
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for EyeSegmentationModel. "
                "This should not be called in production — use ONNX "
                "Runtime instead.  Install with: pip install torch"
            )
        if not _HAS_SMP:
            raise ImportError(
                "segmentation_models_pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )

        super().__init__()

        self.num_classes = num_classes
        self.encoder_name = encoder
        self.class_names = get_class_names(num_classes)
        self.class_colours = get_class_colours(num_classes)

        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )

        # Temperature scaling (train-time fixed at 1.0, calibrated
        # post-hoc on a held-out set for surgical use)
        self.temperature = nn.Parameter(
            torch.ones(1), requires_grad=False,
        )

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            "EyeSegmentationModel: encoder=%s, classes=%d (%s), "
            "pretrained=%s, params=%.1fM",
            encoder,
            num_classes,
            list(self.class_names.values()),
            pretrained,
            param_count / 1e6,
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input batch ``[B, 3, H, W]``, float32, normalised.

        Returns
        -------
        Tensor
            Raw logits ``[B, C, H, W]`` (NOT softmaxed).
        """
        return self.model(x) / self.temperature

    def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return softmax probabilities.

        Parameters
        ----------
        x : Tensor
            Input batch ``[B, 3, H, W]``.

        Returns
        -------
        Tensor
            Probabilities ``[B, C, H, W]``, values in ``[0, 1]``.
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_classes(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return per-pixel class predictions.

        Parameters
        ----------
        x : Tensor
            Input batch ``[B, 3, H, W]``.

        Returns
        -------
        Tensor
            Class indices ``[B, H, W]``, dtype ``int64``.
        """
        with torch.no_grad():
            logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def has_ring_class(self) -> bool:
        """Return ``True`` if the model includes the suction ring class."""
        return self.num_classes >= 4

    # ── calibration ─────────────────────────────────────────────────

    def calibrate_temperature(
        self,
        val_loader,
        device: "torch.device",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Post-hoc temperature scaling on a validation set.

        Optimises NLL on the validation logits with temperature as the
        only parameter.  Returns the calibrated temperature value.

        Parameters
        ----------
        val_loader : DataLoader
            Validation set loader yielding ``(images, masks)`` tuples.
        device : torch.device
            Compute device.
        lr : float
            L-BFGS learning rate.
        max_iter : int
            Maximum L-BFGS iterations.

        Returns
        -------
        float
            Calibrated temperature value.
        """
        self.eval()
        self.temperature.requires_grad_(True)

        nll = nn.CrossEntropyLoss()
        opt = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                raw = self.model(images).cpu()
                logits_list.append(raw)
                labels_list.append(masks)

        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list).long()

        def _closure():
            opt.zero_grad()
            loss = nll(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        opt.step(_closure)
        self.temperature.requires_grad_(False)

        temp = float(self.temperature.item())
        logger.info("Temperature calibrated to %.4f", temp)
        return temp

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model checkpoint with metadata.

        Saves ``state_dict``, ``num_classes``, and ``encoder`` name
        so that ``load()`` can reconstruct the correct architecture.

        Parameters
        ----------
        path : str
            Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "encoder": self.encoder_name,
                "class_names": self.class_names,
            },
            path,
        )
        logger.info(
            "Model saved to %s (%d classes)", path, self.num_classes,
        )

    @classmethod
    def load(
        cls,
        path: str,
        device: str = "cpu",
        num_classes: Optional[int] = None,
    ) -> "EyeSegmentationModel":
        """Load a model checkpoint.

        Auto-detects the number of classes from the checkpoint
        metadata when *num_classes* is ``None``.  Falls back to
        probing the segmentation head weight shape, and finally
        defaults to 3 for backward compatibility.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        device : str
            Target device.
        num_classes : int or None
            Override class count (``None`` = auto-detect).

        Returns
        -------
        EyeSegmentationModel
            Loaded model in eval mode.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required to load EyeSegmentationModel. "
                "Install with: pip install torch"
            )

        ckpt = torch.load(path, map_location=device, weights_only=False)

        # ── Determine num_classes ─────────────────────────────────
        if num_classes is not None:
            n_cls = num_classes
        elif isinstance(ckpt, dict) and "num_classes" in ckpt:
            n_cls = ckpt["num_classes"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            n_cls = _detect_classes_from_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            # Raw state_dict (no wrapper)
            n_cls = _detect_classes_from_state_dict(ckpt)
        else:
            n_cls = 3

        # ── Determine encoder ─────────────────────────────────────
        encoder = "resnet34"
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            encoder = ckpt["encoder"]

        # ── Build model and load weights ──────────────────────────
        model = cls(encoder=encoder, num_classes=n_cls, pretrained=False)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise ValueError(f"Unexpected checkpoint format in {path}")

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        logger.info(
            "Model loaded from %s — %d classes (%s) on %s",
            path, n_cls, list(get_class_names(n_cls).values()), device,
        )
        return model


# ═══════════════════════════════════════════════════════════════════════
#  Factory Functions
# ═══════════════════════════════════════════════════════════════════════

def create_model(
    num_classes: int = 3,
    encoder: str = "resnet34",
    pretrained: bool = True,
    device: str = "cpu",
) -> EyeSegmentationModel:
    """Create a new segmentation model.

    Parameters
    ----------
    num_classes : int
        3 for legacy, 4 for ring-aware.
    encoder : str
        Backbone encoder name.
    pretrained : bool
        Use ImageNet pretrained weights.
    device : str
        Target device.

    Returns
    -------
    EyeSegmentationModel
        Initialised model on the specified device.

    Raises
    ------
    ImportError
        If PyTorch or segmentation_models_pytorch is not installed.
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required to create a model. "
            "Install with: pip install torch"
        )
    model = EyeSegmentationModel(
        encoder=encoder,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return model.to(device)


def load_model(
    path: str,
    device: str = "cpu",
    num_classes: Optional[int] = None,
) -> EyeSegmentationModel:
    """Load a trained model from a checkpoint file.

    This is a convenience wrapper around ``EyeSegmentationModel.load``
    that also handles raw state-dict files (without the metadata
    wrapper) by auto-detecting the number of classes from the
    segmentation head weight tensor shape.

    Tries the following in order:

    1. Full checkpoint with metadata (``num_classes`` key).
    2. Raw state-dict — probe segmentation head for class count.
    3. Try loading as 4-class, then 3-class (brute force).
    4. Default to 3 classes.

    Parameters
    ----------
    path : str
        Path to checkpoint file.
    device : str
        Target device.
    num_classes : int or None
        Override class count.

    Returns
    -------
    EyeSegmentationModel
        Model in eval mode.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    RuntimeError
        If the model cannot be loaded with any class count.
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required to load a model. "
            "Install with: pip install torch"
        )

    # First try the standard load path (handles metadata checkpoints)
    try:
        return EyeSegmentationModel.load(path, device=device, num_classes=num_classes)
    except Exception as exc:
        logger.debug("Standard load failed: %s — trying fallback", exc)

    # Fallback: brute-force load with raw state_dict
    state_dict = torch.load(path, map_location=device, weights_only=True)

    for n_cls in ([num_classes] if num_classes else [4, 3]):
        try:
            model = EyeSegmentationModel(
                num_classes=n_cls, pretrained=False,
            )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            logger.info(
                "Model loaded (fallback) from %s — %d classes on %s",
                path, n_cls, device,
            )
            return model
        except RuntimeError:
            continue

    raise RuntimeError(
        f"Could not load model from {path} with any class count"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _detect_classes_from_state_dict(state_dict: dict) -> int:
    """Probe the segmentation head weight shape to detect num_classes.

    Looks for keys matching the SMP U-Net segmentation head pattern.

    Parameters
    ----------
    state_dict : dict
        Model state dictionary.

    Returns
    -------
    int
        Detected number of classes, or 3 as default.
    """
    for key in state_dict:
        if "segmentation_head" in key and "weight" in key:
            shape = state_dict[key].shape
            if len(shape) >= 1:
                n = shape[0]
                logger.debug(
                    "Detected %d classes from %s shape %s",
                    n, key, shape,
                )
                return n
    logger.debug("Could not detect class count; defaulting to 3")
    return 3


def get_device(preference: str = "auto") -> Optional["torch.device"]:
    """Resolve a device preference string to a ``torch.device``.

    Parameters
    ----------
    preference : str
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    torch.device or None
        The resolved device, or ``None`` if PyTorch is not installed.
    """
    if not _HAS_TORCH:
        return None

    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)