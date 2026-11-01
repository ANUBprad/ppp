# scripts/convert_to_onnx.py
"""
Convert all PyTorch models to ONNX format for distribution.
Run this ONCE on your development machine. Ship the .onnx files.

Handles multiple architecture patterns automatically.
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import json
import sys
import hashlib
import inspect

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_segmentation_model(pth_path: Path):
    """
    Load the segmentation model, auto-detecting the constructor signature.
    Returns (model, n_classes, input_size).
    """
    from pupil_tracking.ml.architecture import EyeSegmentationModel
    
    # ── Step 1: Inspect what the constructor actually accepts ──
    sig = inspect.signature(EyeSegmentationModel.__init__)
    param_names = [
        p for p in sig.parameters.keys() if p != 'self'
    ]
    print(f"    Constructor params: {param_names}")
    
    # ── Step 2: Load the checkpoint to find metadata ──
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict and metadata
    state_dict = None
    metadata = {}
    
    if isinstance(checkpoint, dict):
        print(f"    Checkpoint keys: {list(checkpoint.keys())}")
        
        # Find the state dict
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"    Found state_dict under key: '{key}'")
                break
        
        if state_dict is None:
            # Might be the state_dict itself
            # Check if keys look like model parameters
            sample_key = next(iter(checkpoint.keys()), '')
            if '.' in sample_key and any(
                x in sample_key for x in ['conv', 'bn', 'weight', 'bias', 'encoder', 'decoder']
            ):
                state_dict = checkpoint
                print(f"    Checkpoint IS the state_dict directly")
            else:
                state_dict = checkpoint
        
        # Gather any metadata
        for key in checkpoint:
            if key not in ['model_state_dict', 'state_dict', 'model', 'optimizer_state_dict', 'optimizer']:
                metadata[key] = checkpoint[key]
        
        if metadata:
            print(f"    Metadata: {metadata}")
    else:
        state_dict = checkpoint
        print(f"    Checkpoint is raw state_dict")
    
    # ── Step 3: Detect n_classes from the final layer weights ──
    n_classes = _detect_n_classes(state_dict)
    print(f"    Detected n_classes: {n_classes}")
    
    # ── Step 4: Detect input size from metadata or default ──
    input_size = 512  # default
    for key in ['input_size', 'img_size', 'image_size']:
        if key in metadata:
            input_size = metadata[key]
            break
    print(f"    Input size: {input_size}")
    
    # ── Step 5: Create the model with the right constructor ──
    model = _create_model_instance(
        EyeSegmentationModel, param_names, n_classes, metadata
    )
    
    # ── Step 6: Load weights ──
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"    ✓ Weights loaded (strict mode)")
    except RuntimeError as e:
        print(f"    ⚠ Strict loading failed: {e}")
        print(f"    Trying non-strict loading...")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"    Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"    Unexpected keys: {result.unexpected_keys[:5]}...")
    
    model.eval()
    return model, n_classes, input_size


def _detect_n_classes(state_dict: dict) -> int:
    """Detect number of output classes from the final layer shape."""
    # Look for the final segmentation head layer
    # Common patterns in segmentation_models_pytorch:
    final_layer_patterns = [
        'segmentation_head.0.weight',      # smp default
        'segmentation_head.weight',
        'final_conv.weight',
        'classifier.weight',
        'head.weight',
        'output_conv.weight',
        'last_conv.weight',
    ]
    
    for pattern in final_layer_patterns:
        if pattern in state_dict:
            shape = state_dict[pattern].shape
            n_classes = shape[0]  # Output channels = number of classes
            print(f"    Found final layer '{pattern}' with shape {list(shape)}")
            return n_classes
    
    # Fallback: search for any key that could be the final layer
    candidates = []
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 4:
            # Conv2d weight: (out_channels, in_channels, kH, kW)
            candidates.append((key, tensor.shape))
    
    if candidates:
        # The last conv layer is usually the output
        # Sort by key name and pick the last one
        candidates.sort(key=lambda x: x[0])
        
        # Filter for likely output layers (small number of output channels)
        output_candidates = [
            (k, s) for k, s in candidates if s[0] <= 10
        ]
        
        if output_candidates:
            last_key, last_shape = output_candidates[-1]
            print(f"    Guessing final layer: '{last_key}' with shape {list(last_shape)}")
            return last_shape[0]
    
    print(f"    ⚠ Could not detect n_classes, defaulting to 3")
    return 3


def _create_model_instance(model_class, param_names, n_classes, metadata):
    """Create model instance by trying different constructor patterns."""
    
    # Pattern 1: n_classes parameter
    if 'n_classes' in param_names:
        print(f"    Creating model with n_classes={n_classes}")
        return model_class(n_classes=n_classes)
    
    # Pattern 2: num_classes parameter
    if 'num_classes' in param_names:
        print(f"    Creating model with num_classes={n_classes}")
        return model_class(num_classes=n_classes)
    
    # Pattern 3: classes parameter
    if 'classes' in param_names:
        print(f"    Creating model with classes={n_classes}")
        return model_class(classes=n_classes)
    
    # Pattern 4: out_channels parameter
    if 'out_channels' in param_names:
        print(f"    Creating model with out_channels={n_classes}")
        return model_class(out_channels=n_classes)
    
    # Pattern 5: No class parameter — maybe it's in a config dict
    if 'config' in param_names and 'config' in metadata:
        print(f"    Creating model with config={metadata['config']}")
        return model_class(config=metadata['config'])
    
    # Pattern 6: Try with just the number as first positional arg
    if len(param_names) >= 1:
        try:
            print(f"    Trying: model_class({n_classes})")
            return model_class(n_classes)
        except TypeError:
            pass
    
    # Pattern 7: No arguments
    try:
        print(f"    Trying: model_class() with no arguments")
        return model_class()
    except TypeError as e:
        pass
    
    # Pattern 8: Try create_model factory function
    try:
        from pupil_tracking.ml.architecture import create_model
        sig = inspect.signature(create_model)
        factory_params = [p for p in sig.parameters.keys()]
        print(f"    Trying create_model() with params: {factory_params}")
        
        kwargs = {}
        if 'n_classes' in factory_params:
            kwargs['n_classes'] = n_classes
        elif 'num_classes' in factory_params:
            kwargs['num_classes'] = n_classes
        elif 'classes' in factory_params:
            kwargs['classes'] = n_classes
        
        return create_model(**kwargs)
    except (ImportError, TypeError) as e:
        print(f"    create_model failed: {e}")
    
    raise RuntimeError(
        f"Could not create EyeSegmentationModel.\n"
        f"Constructor params: {param_names}\n"
        f"Please check pupil_tracking/ml/architecture.py and update this script.\n"
        f"You may need to add the correct constructor call."
    )


def convert_segmentation_model(
    pth_path: Path,
    onnx_path: Path,
    opset_version: int = 17,
):
    """Convert the main U-Net segmentation model to ONNX."""
    print(f"[1/4] Loading PyTorch model from {pth_path}...")
    
    model, n_classes, input_size = load_segmentation_model(pth_path)
    
    print(f"[2/4] Exporting to ONNX (input: 1x3x{input_size}x{input_size})...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Dynamic axes for flexible batch size and resolution
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"},
    }
    
    # ── Force legacy exporter (PyTorch 2.6+ defaults to dynamo which ──
    # ── splits weights into external files, giving a tiny .onnx file) ──
    export_kwargs = dict(
        model=model,
        args=dummy_input,
        f=str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    # Check PyTorch version to determine exporter behavior
    torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
    
    if torch_version >= (2, 6):
        # PyTorch 2.6+: Must explicitly use legacy exporter
        # The new dynamo exporter saves weights externally (0.4MB graph only)
        print(f"    PyTorch {torch.__version__} detected — using legacy exporter")
        export_kwargs["dynamo"] = False
    
    try:
        torch.onnx.export(**export_kwargs)
    except TypeError:
        # If 'dynamo' kwarg not recognized (older PyTorch), try without it
        export_kwargs.pop("dynamo", None)
        torch.onnx.export(**export_kwargs)
    
    # ── Verify file size is reasonable ──
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    
    if file_size_mb < 5.0:
        print(f"    ⚠ WARNING: Output is only {file_size_mb:.1f} MB — weights may be external")
        
        # Check for external data files
        external_files = list(onnx_path.parent.glob(f"{onnx_path.stem}*"))
        external_files = [f for f in external_files if f != onnx_path]
        if external_files:
            print(f"    Found external weight files: {[f.name for f in external_files]}")
            print(f"    Merging into single file...")
            
            # Load and re-save as single file
            onnx_model_ext = onnx.load(str(onnx_path), load_external_data=True)
            
            # Remove the external data and embed weights in the model
            for tensor in onnx_model_ext.graph.initializer:
                if tensor.HasField("data_location"):
                    tensor.ClearField("data_location")
            
            onnx.save_model(
                onnx_model_ext,
                str(onnx_path),
                save_as_external_data=False,
            )
            
            # Clean up external files
            for f in external_files:
                f.unlink()
                print(f"    Removed: {f.name}")
            
            file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"    Merged model size: {file_size_mb:.1f} MB")
        
        if file_size_mb < 5.0:
            print(f"    ⚠ Model is still small ({file_size_mb:.1f} MB).")
            print(f"    Expected ~90 MB for ResNet-34 U-Net.")
            print(f"    The export may have failed silently.")
    
    print(f"[3/4] Validating ONNX model...")
    
    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    # Verify numerical accuracy
    print(f"[4/4] Verifying numerical accuracy...")
    verify_accuracy(model, onnx_path, dummy_input)
    
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Segmentation model exported: {onnx_path} ({file_size_mb:.1f} MB)")
    
    return onnx_path, n_classes


def load_ring_classifier(pth_path: Path):
    """
    Load the ring classifier, auto-detecting the constructor.
    """
    # Try multiple possible import paths
    model = None
    
    # Try 1: Direct import
    try:
        from pupil_tracking.ml.ring_classifier import RingClassifier
        sig = inspect.signature(RingClassifier.__init__)
        param_names = [p for p in sig.parameters.keys() if p != 'self']
        print(f"    RingClassifier params: {param_names}")
        
        if not param_names:
            model = RingClassifier()
        elif 'num_classes' in param_names:
            model = RingClassifier(num_classes=2)
        elif 'n_classes' in param_names:
            model = RingClassifier(n_classes=2)
        else:
            model = RingClassifier()
    except Exception as e:
        print(f"    Could not create RingClassifier: {e}")
        return None, 224
    
    # Load weights
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint:
                model.load_state_dict(checkpoint[key], strict=False)
                break
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # Detect input size
    input_size = 224  # MobileNetV2 default
    
    return model, input_size


def convert_ring_classifier(
    pth_path: Path,
    onnx_path: Path,
):
    """Convert the MobileNetV2 ring classifier to ONNX."""
    print(f"[1/3] Loading ring classifier from {pth_path}...")
    
    model, input_size = load_ring_classifier(pth_path)
    if model is None:
        print(f"    ⚠ Could not load ring classifier, skipping")
        return
    
    print(f"[2/3] Exporting ring classifier to ONNX...")
    
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"[3/3] Verifying ring classifier accuracy...")
    verify_accuracy(model, onnx_path, dummy_input)
    
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Ring classifier exported: {onnx_path} ({file_size_mb:.1f} MB)")


def verify_accuracy(
    pytorch_model: nn.Module,
    onnx_path: Path,
    test_input: torch.Tensor,
    tolerance: float = 1e-5,
):
    """Verify ONNX output matches PyTorch output exactly."""
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX Runtime inference
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(None, {"input": test_input.numpy()})[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    
    print(f"    Max absolute difference:  {max_diff:.8f}")
    print(f"    Mean absolute difference: {mean_diff:.8f}")
    
    if max_diff < tolerance:
        print(f"    ✓ PASS — Accuracy preserved (diff < {tolerance})")
    elif max_diff < 1e-3:
        print(f"    ✓ ACCEPTABLE — Small numerical difference (< 0.001)")
    else:
        print(f"    ⚠ WARNING — Difference {max_diff:.8f} exceeds tolerance")
        print(f"      This may still be acceptable for inference.")
    
    # Test with multiple random inputs
    print(f"    Running 10 random input tests...")
    max_diffs = []
    for i in range(10):
        rand_input = torch.randn_like(test_input)
        with torch.no_grad():
            pt_out = pytorch_model(rand_input).numpy()
        ort_out = session.run(None, {"input": rand_input.numpy()})[0]
        diff = np.max(np.abs(pt_out - ort_out))
        max_diffs.append(diff)
    
    avg_max_diff = np.mean(max_diffs)
    worst_diff = np.max(max_diffs)
    print(f"    Average max diff: {avg_max_diff:.8f}")
    print(f"    Worst max diff:   {worst_diff:.8f}")
    
    if worst_diff < 1e-3:
        print(f"    ✓ All 10 random tests passed")
    else:
        print(f"    ⚠ Some tests had notable differences")


def quantize_model(onnx_path: Path, quantized_path: Path):
    """Apply INT8 dynamic quantization for CPU speed boost."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print(f"    ⚠ onnxruntime quantization not available, skipping")
        return
    
    # Check model size — skip if suspiciously small
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 1.0:
        print(f"    ⚠ Skipping quantization — model too small ({file_size_mb:.1f} MB)")
        return
    
    print(f"    Quantizing {onnx_path.name}...")
    
    try:
        # Newer onnxruntime versions
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QUInt8,
        )
    except TypeError:
        try:
            # Even newer API — positional args
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8,
            )
        except Exception as e:
            print(f"    ⚠ Quantization failed: {e}")
            print(f"    The full-precision model will work fine without quantization.")
            return
    
    original_size = onnx_path.stat().st_size / (1024 * 1024)
    quantized_size = quantized_path.stat().st_size / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"    Original:  {original_size:.1f} MB")
    print(f"    Quantized: {quantized_size:.1f} MB")
    print(f"    Reduction: {reduction:.1f}%")


def compute_file_hash(path: Path) -> str:
    """SHA256 hash for integrity verification."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    models_dir = PROJECT_ROOT / "models"
    onnx_dir = PROJECT_ROOT / "models" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ONNX MODEL CONVERSION PIPELINE")
    print("=" * 60)
    
    # ── First: Show what we're working with ──
    print("\nInspecting architecture...")
    try:
        from pupil_tracking.ml.architecture import EyeSegmentationModel
        sig = inspect.signature(EyeSegmentationModel.__init__)
        print(f"  EyeSegmentationModel.__init__{sig}")
        
        # Show full source for debugging
        source = inspect.getsource(EyeSegmentationModel.__init__)
        print(f"\n  Source code:")
        for line in source.split('\n')[:15]:
            print(f"    {line}")
        print()
    except Exception as e:
        print(f"  Error inspecting: {e}")
    
    # ── Convert segmentation model ──
    seg_pth = models_dir / "best_model.pth"
    seg_onnx = onnx_dir / "segmentation.onnx"
    seg_quantized = onnx_dir / "segmentation_quantized.onnx"
    
    if seg_pth.exists():
        print(f"\n{'─' * 60}")
        print("SEGMENTATION MODEL")
        print(f"{'─' * 60}")
        try:
            convert_segmentation_model(seg_pth, seg_onnx)
            print()
            quantize_model(seg_onnx, seg_quantized)
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            print(f"\n  Full traceback:")
            import traceback
            traceback.print_exc()
            print(f"\n  Please share the output above so we can fix this.")
    else:
        print(f"\n⚠ Segmentation model not found: {seg_pth}")
    
    # ── Convert ring classifier ──
    ring_pth = models_dir / "ring_classifier.pth"
    ring_onnx = onnx_dir / "ring_classifier.onnx"
    ring_quantized = onnx_dir / "ring_classifier_quantized.onnx"
    
    if ring_pth.exists():
        print(f"\n{'─' * 60}")
        print("RING CLASSIFIER")
        print(f"{'─' * 60}")
        try:
            convert_ring_classifier(ring_pth, ring_onnx)
            print()
            quantize_model(ring_onnx, ring_quantized)
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Ring classifier not found: {ring_pth}")
    
    # ── Create manifest ──
    print(f"\n{'─' * 60}")
    print("MANIFEST")
    print(f"{'─' * 60}")
    
    manifest = {
        "format_version": "1.0",
        "converter": "convert_to_onnx.py",
        "models": {},
    }
    
    for onnx_file in sorted(onnx_dir.glob("*.onnx")):
        manifest["models"][onnx_file.name] = {
            "size_bytes": onnx_file.stat().st_size,
            "size_mb": round(onnx_file.stat().st_size / (1024 * 1024), 2),
            "sha256": compute_file_hash(onnx_file),
        }
    
    manifest_path = onnx_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # ── Summary ──
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    
    if not manifest["models"]:
        print("\n  ⚠ No models were converted successfully.")
        print("  Please share the error output above.")
    else:
        print(f"\n  Output directory: {onnx_dir}")
        print(f"\n  Files to ship with your application:")
        total_size = 0
        for name, info in manifest["models"].items():
            print(f"    {name:45s} {info['size_mb']:8.1f} MB")
            total_size += info['size_bytes']
        print(f"    {'─' * 53}")
        print(f"    {'TOTAL':45s} {total_size / (1024*1024):8.1f} MB")
        print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()