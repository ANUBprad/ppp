#!/usr/bin/env python3
"""
Benchmark script to measure FPS at different resolutions, 
with/without FP16, with/without ROI, etc.

Usage:
    python scripts/benchmark_fps.py --model models/best_model.pth
    python scripts/benchmark_fps.py --model models/best_model.pth --source 0       # camera
    python scripts/benchmark_fps.py --model models/best_model.pth --source video.mp4
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import cv2
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("benchmark")


def make_synthetic_frame(h=720, w=1280):
    """Generate a synthetic eye-like frame for benchmarking."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Background
    frame[:] = (60, 50, 45)
    # Iris
    cv2.circle(frame, (w // 2, h // 2), 180, (80, 120, 60), -1)
    # Pupil
    cv2.circle(frame, (w // 2, h // 2), 70, (10, 10, 10), -1)
    # Specular reflection
    cv2.circle(frame, (w // 2 - 30, h // 2 - 25), 15, (220, 220, 220), -1)
    # Noise
    noise = np.random.randint(0, 15, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def benchmark_engine_only(model_path: str, device: str = "auto"):
    """Benchmark raw inference speed at different resolutions."""
    from ml.fast_inference import FastInferenceEngine

    print("\n" + "=" * 70)
    print("  RAW INFERENCE BENCHMARK (model forward pass only)")
    print("=" * 70)

    resolutions = [256, 288, 320, 352, 384, 416, 448, 512]

    for fp16 in [True, False]:
        # Reset cache for fair comparison
        FastInferenceEngine.reset_cache()
        engine = FastInferenceEngine(
            model_path=model_path,
            device=device,
            use_fp16=fp16,
            use_compile=False,  # Benchmark without compile first
        )

        print(f"\n{'FP16' if fp16 else 'FP32'} on {engine.device}:")
        print(f"  {'Resolution':>10}  {'Avg (ms)':>10}  {'FPS':>8}  {'Min (ms)':>10}  {'Max (ms)':>10}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

        for res in resolutions:
            engine.warmup(res)

            # Prepare input
            dtype = torch.float16 if fp16 else torch.float32
            x = torch.randn(1, 3, res, res, dtype=dtype, device=engine.device)

            # Benchmark
            times = []
            N = 50
            for _ in range(N):
                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    _ = engine.model(x)

                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            avg = np.mean(times) * 1000
            fps = 1000.0 / avg
            mn = min(times) * 1000
            mx = max(times) * 1000
            print(f"  {res:>10}  {avg:>10.1f}  {fps:>8.1f}  {mn:>10.1f}  {mx:>10.1f}")

    # Now benchmark with torch.compile
    if hasattr(torch, "compile"):
        print(f"\n\nFP16 + torch.compile on CUDA:")
        FastInferenceEngine.reset_cache()
        engine = FastInferenceEngine(
            model_path=model_path, device=device,
            use_fp16=True, use_compile=True,
        )
        print(f"  {'Resolution':>10}  {'Avg (ms)':>10}  {'FPS':>8}")
        print(f"  {'-'*10}  {'-'*10}  {'-'*8}")

        for res in [320, 384, 512]:
            engine.warmup(res)
            dtype = torch.float16
            x = torch.randn(1, 3, res, res, dtype=dtype, device=engine.device)

            times = []
            for _ in range(80):  # More iterations for compile warmup
                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = engine.model(x)
                if engine.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            # Drop first 30 (compile warmup)
            times = times[30:]
            avg = np.mean(times) * 1000
            fps = 1000.0 / avg
            print(f"  {res:>10}  {avg:>10.1f}  {fps:>8.1f}")


def benchmark_full_pipeline(model_path: str, source=None, device: str = "auto"):
    """Benchmark complete pipeline: preprocess → infer → postprocess → Kalman."""
    from ml.fast_inference import FastInferenceEngine
    from video.optimized_processor import OptimizedVideoProcessor

    print("\n" + "=" * 70)
    print("  FULL PIPELINE BENCHMARK (preprocess + infer + ellipse + Kalman)")
    print("=" * 70)

    FastInferenceEngine.reset_cache()
    engine = FastInferenceEngine(
        model_path=model_path,
        device=device,
        video_resolution=320,
        use_fp16=True,
        use_compile=True,
    )
    engine.warmup(320)

    # We need a minimal detector stub for the processor
    processor = OptimizedVideoProcessor(
        detector=None,  # Not used directly in our implementation
        fast_engine=engine,
        target_fps=30.0,
    )

    # Generate or read frames
    if source is not None:
        cap = cv2.VideoCapture(source)
        frames = []
        for _ in range(200):
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            print("  ERROR: Could not read frames from source")
            return
    else:
        print("  Using synthetic frames (1280×720)")
        frames = [make_synthetic_frame() for _ in range(100)]

    # Warm up
    for i in range(min(5, len(frames))):
        processor.process_frame(frames[i], i)

    # Benchmark
    times = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        result = processor.process_frame(frame, i)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    stats = processor.get_stats()
    avg = np.mean(times) * 1000
    fps = 1000.0 / avg if avg > 0 else 0
    p95 = np.percentile(times, 95) * 1000
    p99 = np.percentile(times, 99) * 1000

    print(f"\n  Frames:       {len(frames)}")
    print(f"  Resolution:   {stats['resolution']}")
    print(f"  Frame skip:   {stats['frame_skip']}")
    print(f"  ROI active:   {stats['roi_active']}")
    print(f"  Processed:    {stats['frames_processed']}")
    print(f"  Interpolated: {stats['frames_interpolated']}")
    print(f"\n  Average:      {avg:.1f} ms  ({fps:.1f} FPS)")
    print(f"  P95:          {p95:.1f} ms")
    print(f"  P99:          {p99:.1f} ms")
    print(f"  Min:          {min(times)*1000:.1f} ms")
    print(f"  Max:          {max(times)*1000:.1f} ms")


def benchmark_optimizations_comparison(model_path: str, device: str = "auto"):
    """Compare different optimization combinations."""
    from ml.fast_inference import FastInferenceEngine

    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPARISON")
    print("=" * 70)

    frame = make_synthetic_frame()
    N = 40

    configs = [
        {"label": "Baseline (FP32, 512, no compile)", "fp16": False, "compile": False, "res": 512},
        {"label": "FP16 only (512)",                   "fp16": True,  "compile": False, "res": 512},
        {"label": "Reduced res (320, FP32)",           "fp16": False, "compile": False, "res": 320},
        {"label": "FP16 + reduced res (320)",          "fp16": True,  "compile": False, "res": 320},
        {"label": "FP16 + res 320 + compile",          "fp16": True,  "compile": True,  "res": 320},
        {"label": "FP16 + res 256 + compile",          "fp16": True,  "compile": True,  "res": 256},
    ]

    print(f"\n  {'Configuration':<40}  {'Avg (ms)':>10}  {'FPS':>8}  {'Speedup':>8}")
    print(f"  {'-'*40}  {'-'*10}  {'-'*8}  {'-'*8}")

    baseline_ms = None

    for cfg in configs:
        FastInferenceEngine.reset_cache()
        try:
            engine = FastInferenceEngine(
                model_path=model_path,
                device=device,
                video_resolution=cfg["res"],
                static_resolution=cfg["res"],
                use_fp16=cfg["fp16"],
                use_compile=cfg["compile"],
            )
            engine.warmup(cfg["res"])
        except Exception as e:
            print(f"  {cfg['label']:<40}  FAILED: {e}")
            continue

        # Full pipeline: preprocess + infer + postprocess
        times = []
        for _ in range(N + 10):
            t0 = time.perf_counter()
            tensor, meta = engine.preprocess(frame, cfg["res"])
            mask = engine.infer_mask(tensor)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        # Drop warmup
        times = times[10:]
        avg = np.mean(times) * 1000
        fps = 1000.0 / avg

        if baseline_ms is None:
            baseline_ms = avg
        speedup = baseline_ms / avg

        print(f"  {cfg['label']:<40}  {avg:>10.1f}  {fps:>8.1f}  {speedup:>7.1f}×")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS performance")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--source", default=None, help="Video file or camera ID")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu/mps)")
    parser.add_argument("--mode", default="all", choices=["engine", "pipeline", "compare", "all"])
    args = parser.parse_args()

    source = None
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source

    print(f"\nModel:  {args.model}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name()}")
        print(f"VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    if args.mode in ("engine", "all"):
        benchmark_engine_only(args.model, args.device)

    if args.mode in ("compare", "all"):
        benchmark_optimizations_comparison(args.model, args.device)

    if args.mode in ("pipeline", "all"):
        benchmark_full_pipeline(args.model, source, args.device)

    print("\n✅ Benchmark complete\n")


if __name__ == "__main__":
    main()