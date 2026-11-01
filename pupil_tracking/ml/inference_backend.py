# pupil_tracking/ml/inference_backend.py
"""
Abstraction layer — auto-selects best available inference backend.
Production: ONNX Runtime (lightweight, fast, cross-platform)
Development: PyTorch (for training + inference)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class InferenceBackend:
    """Factory that creates the best available inference engine."""

    @staticmethod
    def create(
        model_dir: Optional[str] = None,
        prefer_onnx: bool = True,
        input_size: int = 512,
        num_classes: int = 3,
        enable_gpu: bool = True,
        use_quantized: bool = True,
    ):
        """
        Create the best available inference engine.
        Returns an object with .infer(image) -> dict method.
        """

        # ── Try ONNX Runtime first (production) ──
        if prefer_onnx:
            engine = InferenceBackend._try_onnx(
                model_dir, input_size, num_classes,
                enable_gpu, use_quantized,
            )
            if engine is not None:
                return engine

        # ── Fallback to PyTorch (development) ──
        engine = InferenceBackend._try_pytorch(
            model_dir, input_size, num_classes,
        )
        if engine is not None:
            return engine

        raise RuntimeError(
            "No inference backend available.\n"
            "Install onnxruntime: pip install onnxruntime\n"
            "Or ensure models/onnx/ contains your .onnx model files."
        )

    @staticmethod
    def _try_onnx(model_dir, input_size, num_classes, enable_gpu, use_quantized):
        """Attempt to create ONNX Runtime inference engine."""
        try:
            from pupil_tracking.ml.onnx_inference import ONNXInference

            model_path = _find_onnx_model(model_dir, use_quantized)

            engine = ONNXInference(
                model_path=model_path,
                input_size=input_size,
                num_classes=num_classes,
                use_quantized=use_quantized,
                enable_gpu=enable_gpu,
            )

            if engine.is_loaded:
                info = engine.get_device_info()
                logger.info(
                    f"Using ONNX Runtime backend "
                    f"({info.get('model', 'unknown')}, "
                    f"{info.get('provider', 'CPU')})"
                )
                return engine
            else:
                logger.warning("ONNX model not found, trying PyTorch fallback")
                return None

        except ImportError:
            logger.info("onnxruntime not installed, trying PyTorch fallback")
            return None
        except Exception as e:
            logger.warning(f"ONNX backend failed: {e}, trying PyTorch fallback")
            return None

    @staticmethod
    def _try_pytorch(model_dir, input_size, num_classes):
        """Attempt to create PyTorch inference engine."""
        try:
            from pupil_tracking.ml.inference import SegmentationInference

            pth_path = _find_pytorch_model(model_dir)
            if pth_path is None:
                logger.warning("PyTorch model file not found")
                return None

            engine = SegmentationInference(
                input_size=input_size,
                num_classes=num_classes,
            )
            engine.load_model(pth_path)

            logger.info(f"Using PyTorch backend ({pth_path})")
            return engine

        except ImportError:
            logger.info("PyTorch not installed")
            return None
        except Exception as e:
            logger.warning(f"PyTorch backend failed: {e}")
            return None

    @staticmethod
    def create_ring_classifier(
        model_dir: Optional[str] = None,
        prefer_onnx: bool = True,
    ):
        """Create ring classifier with best available backend."""
        if prefer_onnx:
            try:
                from pupil_tracking.ml.onnx_inference import ONNXRingClassifier
                classifier = ONNXRingClassifier()
                if classifier.session is not None:
                    logger.info("Ring classifier: ONNX backend")
                    return classifier
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"ONNX ring classifier failed: {e}")

        # PyTorch fallback
        try:
            from pupil_tracking.ml.ring_classifier import RingClassifier
            classifier = RingClassifier()
            logger.info("Ring classifier: PyTorch backend")
            return classifier
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch ring classifier failed: {e}")

        logger.info("No ring classifier available — using heuristic detection")
        return None


def _find_onnx_model(
    model_dir: Optional[str],
    use_quantized: bool,
) -> Optional[str]:
    """Locate ONNX model file."""
    search_dirs = []

    if model_dir:
        search_dirs.append(Path(model_dir) / "onnx")
        search_dirs.append(Path(model_dir))

    # Frozen app (PyInstaller)
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            search_dirs.append(Path(sys._MEIPASS) / "models" / "onnx")
        search_dirs.append(Path(sys.executable).parent / "models" / "onnx")

    # Development
    search_dirs.extend([
        Path(__file__).parent.parent.parent / "models" / "onnx",
        Path.cwd() / "models" / "onnx",
    ])

    filename = (
        "segmentation_quantized.onnx" if use_quantized
        else "segmentation.onnx"
    )

    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return str(candidate)

    # Try non-quantized as fallback
    if use_quantized:
        return _find_onnx_model(model_dir, use_quantized=False)

    return None


def _find_pytorch_model(model_dir: Optional[str]) -> Optional[str]:
    """Locate PyTorch model file."""
    search_dirs = []
    if model_dir:
        search_dirs.append(Path(model_dir))

    search_dirs.extend([
        Path(__file__).parent.parent.parent / "models",
        Path.cwd() / "models",
    ])

    for d in search_dirs:
        candidate = d / "best_model.pth"
        if candidate.exists():
            return str(candidate)
    return None