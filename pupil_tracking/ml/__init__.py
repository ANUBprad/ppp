"""
Machine learning module for pupil/limbus segmentation.

In production (ONNX Runtime), only onnx_inference and inference_backend
are used. PyTorch-dependent modules (architecture, fast_inference, trainer,
dataset, losses) are optional and only available in development.
"""

# Always available (no torch dependency)
from pupil_tracking.ml.onnx_inference import ONNXInference
from pupil_tracking.ml.inference_backend import InferenceBackend

# PyTorch-dependent modules — optional
try:
    from pupil_tracking.ml.inference import SegmentationInference
except ImportError:
    SegmentationInference = None

try:
    from pupil_tracking.ml.fast_inference import FastInference
except ImportError:
    FastInference = None

try:
    from pupil_tracking.ml.architecture import EyeSegmentationModel
except ImportError:
    EyeSegmentationModel = None

__all__ = [
    "ONNXInference",
    "InferenceBackend",
    "SegmentationInference",
    "FastInference",
    "EyeSegmentationModel",
]