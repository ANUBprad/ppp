#!/usr/bin/env python3
"""
Export trained model to ONNX for maximum inference speed.

Usage:
    python scripts/export_onnx.py --model models/best_model.pth --resolution 320
    python scripts/export_onnx.py --model models/best_model.pth --resolution 320 --verify
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export")


def export(model_path: str, resolution: int = 320, output_path: str = None, verify: bool = False):
    from pupil_tracking.ml.architecture import EyeSegmentationModel

    if output_path is None:
        base = os.path.splitext(model_path)[0]
        output_path = f"{base}_r{resolution}.onnx"

    # Load
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = EyeSegmentationModel(
        encoder=checkpoint.get("encoder", "resnet34"),
        num_classes=checkpoint.get("num_classes", 3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, resolution, resolution)

    # Export
    logger.info(f"Exporting to {output_path} (resolution={resolution}) ...")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=18,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / 1e6
    logger.info(f"✅ Exported: {output_path} ({file_size:.1f} MB)")

    # Verify
    if verify:
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            # Compare outputs
            with torch.no_grad():
                pt_out = model(dummy)
                if isinstance(pt_out, dict):
                    pt_out = list(pt_out.values())[0]
                pt_np = pt_out.numpy()

            onnx_out = session.run(None, {input_name: dummy.numpy()})[0]

            diff = np.abs(pt_np - onnx_out).max()
            logger.info(f"✅ Verification passed — max diff: {diff:.6f}")

            if diff > 0.01:
                logger.warning(f"⚠ Max difference {diff:.6f} is larger than expected")

        except ImportError:
            logger.warning("onnxruntime not installed — skipping verification")

    # Also try to optimize with onnxruntime
    try:
        import onnxruntime as ort

        opt_path = output_path.replace(".onnx", "_optimized.onnx")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.optimized_model_filepath = opt_path

        _ = ort.InferenceSession(output_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

        if os.path.exists(opt_path):
            opt_size = os.path.getsize(opt_path) / 1e6
            logger.info(f"✅ Optimized ONNX: {opt_path} ({opt_size:.1f} MB)")

    except Exception as e:
        logger.info(f"ONNX optimization skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    export(args.model, args.resolution, args.output, args.verify)


if __name__ == "__main__":
    main()