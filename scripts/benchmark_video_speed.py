#!/usr/bin/env python
"""
Video processing speed benchmark.

Measures decode speed, per-frame inference, batch inference, and
overall pipeline FPS before/after optimisations.

Usage:
    python scripts/benchmark_video_speed.py --input <video_path> [--frames 200] [--batch 4]
    python scripts/benchmark_video_speed.py --synthetic [--frames 200]

Outputs:
    • Total wall-clock time
    • Average FPS
    • Per-frame latency breakdown (decode / preprocess / infer / postprocess)
    • GPU utilisation estimate (if nvidia-smi is available)
    • Comparison between single-frame and batch inference
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_gpu_util() -> Optional[float]:
    """Return current GPU utilisation % via nvidia-smi, or None."""
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def _try_gpu_memory() -> Optional[Tuple[float, float]]:
    """Return (used_mb, total_mb) via nvidia-smi, or None."""
    try:
        import subprocess
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
        )
        parts = out.decode().strip().split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None


def _create_synthetic_video(
    path: str,
    n_frames: int = 200,
    size: int = 480,
    fps: float = 30.0,
) -> str:
    """Generate a synthetic test video with a fake 'eye'.

    Creates a video with:
        - Gray background (sclera)
        - Dark circle (iris) with texture
        - Smaller dark circle (pupil) with slight movement
        - Bright specular reflection spot
        - Optional red dots simulating suction ring markers
        - Random noise for realism
    """
    print(f"Generating synthetic test video: {path}")
    print(f"  Frames: {n_frames}, Size: {size}x{size}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (size, size))

    if not writer.isOpened():
        # Fallback codec
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        path = path.replace(".avi", ".avi")
        writer = cv2.VideoWriter(path, fourcc, fps, (size, size))

    rng = np.random.default_rng(42)
    cx_base, cy_base = size // 2, size // 2
    iris_r = size // 4
    pupil_r = size // 8

    for i in range(n_frames):
        # Background (sclera-like)
        frame = np.full((size, size, 3), 200, dtype=np.uint8)

        # Add slight gradient for realism
        gradient = np.linspace(180, 220, size).astype(np.uint8)
        for c in range(3):
            frame[:, :, c] = gradient[np.newaxis, :]

        # Iris (dark circle with slight colour)
        cv2.circle(
            frame, (cx_base, cy_base), iris_r,
            (60, 80, 50), -1,
        )

        # Iris texture (radial lines)
        for angle in np.linspace(0, 2 * np.pi, 36):
            x1 = int(cx_base + pupil_r * 1.2 * np.cos(angle))
            y1 = int(cy_base + pupil_r * 1.2 * np.sin(angle))
            x2 = int(cx_base + iris_r * 0.95 * np.cos(angle))
            y2 = int(cy_base + iris_r * 0.95 * np.sin(angle))
            colour = (
                50 + rng.integers(0, 20),
                70 + rng.integers(0, 20),
                40 + rng.integers(0, 20),
            )
            cv2.line(frame, (x1, y1), (x2, y2), colour, 1)

        # Pupil (slightly moving to simulate saccades)
        dx = int(5 * np.sin(i * 0.1))
        dy = int(3 * np.cos(i * 0.13))
        pcx, pcy = cx_base + dx, cy_base + dy
        cv2.circle(frame, (pcx, pcy), pupil_r, (8, 8, 8), -1)

        # Specular reflection (bright spot on pupil)
        ref_x = pcx - pupil_r // 3
        ref_y = pcy - pupil_r // 3
        cv2.circle(frame, (ref_x, ref_y), 6, (255, 255, 255), -1)
        cv2.circle(frame, (ref_x + 12, ref_y + 8), 3, (240, 240, 255), -1)

        # Suction ring markers (red dots in a circle, every other frame
        # to test detection stability)
        if i % 3 != 0:  # Present in 2/3 of frames
            n_dots = 16
            ring_r = int(iris_r * 1.05)
            for di in range(n_dots):
                angle = 2 * np.pi * di / n_dots
                dot_x = int(cx_base + ring_r * np.cos(angle))
                dot_y = int(cy_base + ring_r * np.sin(angle))
                cv2.circle(frame, (dot_x, dot_y), 3, (0, 0, 220), -1)

        # Gaussian noise
        noise = rng.integers(
            0, 8, frame.shape, dtype=np.uint8
        )
        frame = cv2.add(frame, noise)

        writer.write(frame)

    writer.release()
    print(f"  Synthetic video saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def _benchmark_decode(
    video_path: str, n_frames: int
) -> Tuple[float, int]:
    """Measure raw video decode speed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return 0.0, 0

    count = 0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    return elapsed, count


def _benchmark_preprocess(
    video_path: str, n_frames: int
) -> Tuple[float, int]:
    """Measure preprocessing speed (reflection removal + normalisation)."""
    try:
        from pupil_tracking.preprocessing.reflection_removal import (
            ReflectionRemover,
        )
        from pupil_tracking.preprocessing.suction_ring_masker import (
            SuctionRingMasker,
        )
        from pupil_tracking.preprocessing.normalizer import ImageNormalizer

        refl = ReflectionRemover(
            brightness_threshold=225,
            min_reflection_area=10,
            inpaint_radius=3,
        )
        ring = SuctionRingMasker()
        norm = ImageNormalizer(
            enable_clahe=True,
            enable_brightness=True,
            enable_white_balance=False,
            enable_gamma=False,
        )
    except ImportError as e:
        print(f"  Preprocessing modules not available: {e}")
        return 0.0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0

    count = 0
    t0 = time.perf_counter()
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Full preprocessing pipeline
        frame, _ = ring.remove(frame)
        frame, _ = refl.remove(frame)
        frame = norm.normalize(frame)
        count += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    return elapsed, count


def _benchmark_single_inference(
    video_path: str,
    n_frames: int,
    model_path: str,
    input_size: int = 320,
) -> Tuple[float, int, List[float]]:
    """Measure single-frame inference speed."""
    try:
        from pupil_tracking.ml.fast_inference import FastInference

        engine = FastInference(
            model_path=model_path,
            input_size=input_size,
        )
    except Exception as e:
        print(f"  FastInference not available: {e}")
        return 0.0, 0, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0, []

    latencies: List[float] = []
    count = 0
    t0 = time.perf_counter()

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t_frame = time.perf_counter()
        _ = engine.detect(frame)
        latencies.append((time.perf_counter() - t_frame) * 1000)
        count += 1

    elapsed = time.perf_counter() - t0
    cap.release()
    return elapsed, count, latencies


def _benchmark_batch_inference(
    video_path: str,
    n_frames: int,
    model_path: str,
    batch_size: int = 4,
    input_size: int = 320,
) -> Tuple[float, int, List[float]]:
    """Measure batch inference speed."""
    try:
        from pupil_tracking.ml.fast_inference import FastInference

        engine = FastInference(
            model_path=model_path,
            input_size=input_size,
        )
    except Exception as e:
        print(f"  FastInference not available: {e}")
        return 0.0, 0, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, 0, []

    latencies: List[float] = []
    count = 0
    t0 = time.perf_counter()

    batch_frames: List[np.ndarray] = []

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)

        if len(batch_frames) >= batch_size:
            t_batch = time.perf_counter()
            _ = engine.detect_batch(batch_frames)
            batch_ms = (time.perf_counter() - t_batch) * 1000
            per_frame_ms = batch_ms / len(batch_frames)
            for _ in batch_frames:
                latencies.append(per_frame_ms)
            count += len(batch_frames)
            batch_frames = []

    # Process remaining frames
    if batch_frames:
        t_batch = time.perf_counter()
        _ = engine.detect_batch(batch_frames)
        batch_ms = (time.perf_counter() - t_batch) * 1000
        per_frame_ms = batch_ms / len(batch_frames)
        for _ in batch_frames:
            latencies.append(per_frame_ms)
        count += len(batch_frames)

    elapsed = time.perf_counter() - t0
    cap.release()
    return elapsed, count, latencies


def _benchmark_full_pipeline(
    video_path: str,
    n_frames: int,
    model_path: str,
    batch_size: int = 4,
    input_size: int = 320,
) -> Tuple[float, int, List[float]]:
    """Measure full OptimizedVideoProcessor speed."""
    try:
        from pupil_tracking.processing.optimized_processor import (
            OptimizedVideoProcessor,
        )

        processor = OptimizedVideoProcessor(
            model_path=model_path,
            input_size=input_size,
            batch_size=batch_size,
            fast_mode=True,
            skip_quality_check=True,
        )
    except Exception as e:
        print(f"  OptimizedVideoProcessor not available: {e}")
        return 0.0, 0, []

    t0 = time.perf_counter()
    results = processor.process_video(
        video_path, max_frames=n_frames
    )
    elapsed = time.perf_counter() - t0

    latencies = [
        r.get("latency_ms", 0) for r in results
        if "latency_ms" in r
    ]
    return elapsed, len(results), latencies


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(
    video_path: str,
    max_frames: int = 200,
    batch_size: int = 4,
    input_size: int = 320,
    model_path: Optional[str] = None,
) -> None:
    """Run all benchmarks and print results."""

    # Find model
    if model_path is None:
        model_path = str(_PROJECT_ROOT / "models" / "best_model.pth")
        if not os.path.exists(model_path):
            models_dir = _PROJECT_ROOT / "models"
            if models_dir.is_dir():
                pth_files = list(models_dir.glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])

    has_model = os.path.exists(model_path) if model_path else False

    # Video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    n = min(max_frames, total) if total > 0 else max_frames

    # GPU info
    gpu_util_start = _try_gpu_util()
    gpu_mem = _try_gpu_memory()

    print("\n" + "=" * 70)
    print("  VIDEO PROCESSING SPEED BENCHMARK")
    print("=" * 70)
    print(f"  Video:       {video_path}")
    print(f"  Resolution:  {fw}x{fh} @ {fps_native:.1f} FPS")
    print(f"  Total:       {total} frames")
    print(f"  Benchmark:   {n} frames")
    print(f"  Model:       {model_path if has_model else 'NOT FOUND'}")
    print(f"  Input size:  {input_size}x{input_size}")
    print(f"  Batch size:  {batch_size}")
    if gpu_util_start is not None:
        print(f"  GPU util:    {gpu_util_start:.0f}%")
    if gpu_mem is not None:
        print(f"  GPU memory:  {gpu_mem[0]:.0f}/{gpu_mem[1]:.0f} MB")
    print("-" * 70)

    # ── 1. Decode speed ────────────────────────────────────────
    print("\n[1/5] Decode-only speed ...")
    decode_time, decode_count = _benchmark_decode(video_path, n)
    decode_fps = decode_count / decode_time if decode_time > 0 else 0
    print(
        f"  {decode_count} frames in {decode_time:.2f}s "
        f"= {decode_fps:.1f} FPS"
    )

    # ── 2. Preprocessing speed ─────────────────────────────────
    print("\n[2/5] Preprocessing speed (refl + ring + norm) ...")
    preproc_time, preproc_count = _benchmark_preprocess(video_path, n)
    if preproc_count > 0:
        preproc_fps = preproc_count / preproc_time if preproc_time > 0 else 0
        preproc_ms = preproc_time / preproc_count * 1000
        print(
            f"  {preproc_count} frames in {preproc_time:.2f}s "
            f"= {preproc_fps:.1f} FPS ({preproc_ms:.1f} ms/frame)"
        )
    else:
        print("  (skipped)")

    if not has_model:
        print("\n  Model not found — skipping inference benchmarks")
        print("=" * 70)
        return

    # ── 3. Single-frame inference ──────────────────────────────
    print(f"\n[3/5] Single-frame inference ({input_size}x{input_size}) ...")
    single_time, single_count, single_lat = _benchmark_single_inference(
        video_path, n, model_path, input_size
    )
    if single_count > 0:
        single_fps = single_count / single_time if single_time > 0 else 0
        avg_single = np.mean(single_lat) if single_lat else 0
        p95_single = np.percentile(single_lat, 95) if single_lat else 0
        print(
            f"  {single_count} frames in {single_time:.2f}s "
            f"= {single_fps:.1f} FPS"
        )
        print(
            f"  Latency: avg={avg_single:.1f}ms  "
            f"p95={p95_single:.1f}ms"
        )
    else:
        print("  (skipped)")
        single_fps = 0

    # ── 4. Batch inference ─────────────────────────────────────
    print(
        f"\n[4/5] Batch inference (batch={batch_size}, "
        f"{input_size}x{input_size}) ..."
    )
    batch_time, batch_count, batch_lat = _benchmark_batch_inference(
        video_path, n, model_path, batch_size, input_size
    )
    if batch_count > 0:
        batch_fps = batch_count / batch_time if batch_time > 0 else 0
        avg_batch = np.mean(batch_lat) if batch_lat else 0
        p95_batch = np.percentile(batch_lat, 95) if batch_lat else 0
        print(
            f"  {batch_count} frames in {batch_time:.2f}s "
            f"= {batch_fps:.1f} FPS"
        )
        print(
            f"  Latency: avg={avg_batch:.1f}ms  "
            f"p95={p95_batch:.1f}ms"
        )
    else:
        print("  (skipped)")
        batch_fps = 0

    # ── 5. Full pipeline ──────────────────────────────────────
    print(
        f"\n[5/5] Full pipeline (OptimizedVideoProcessor, "
        f"batch={batch_size}) ..."
    )
    pipe_time, pipe_count, pipe_lat = _benchmark_full_pipeline(
        video_path, n, model_path, batch_size, input_size
    )
    if pipe_count > 0:
        pipe_fps = pipe_count / pipe_time if pipe_time > 0 else 0
        avg_pipe = np.mean(pipe_lat) if pipe_lat else 0
        p95_pipe = np.percentile(pipe_lat, 95) if pipe_lat else 0
        print(
            f"  {pipe_count} frames in {pipe_time:.2f}s "
            f"= {pipe_fps:.1f} FPS"
        )
        print(
            f"  Latency: avg={avg_pipe:.1f}ms  "
            f"p95={p95_pipe:.1f}ms"
        )
    else:
        print("  (skipped)")
        pipe_fps = 0

    # ── GPU stats after ─────────────────────────────────────────
    gpu_util_end = _try_gpu_util()
    gpu_mem_end = _try_gpu_memory()

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Stage':<35} {'FPS':>8} {'ms/frame':>10}")
    print(f"  {'-' * 55}")
    print(
        f"  {'Decode only':<35} {decode_fps:>8.1f} "
        f"{1000/decode_fps if decode_fps > 0 else 0:>10.1f}"
    )
    if preproc_count > 0:
        print(
            f"  {'Preprocess (refl+ring+norm)':<35} "
            f"{preproc_fps:>8.1f} {preproc_ms:>10.1f}"
        )
    if single_count > 0:
        print(
            f"  {'Single-frame inference':<35} {single_fps:>8.1f} "
            f"{avg_single:>10.1f}"
        )
    if batch_count > 0:
        print(
            f"  {'Batch inference (batch={bs})':<35} {batch_fps:>8.1f} "
            f"{avg_batch:>10.1f}".format(bs=batch_size)
        )
    if pipe_count > 0:
        print(
            f"  {'Full pipeline':<35} {pipe_fps:>8.1f} "
            f"{avg_pipe:>10.1f}"
        )
    print(f"  {'-' * 55}")

    # Speedup calculation
    if single_fps > 0 and batch_fps > 0:
        speedup = batch_fps / single_fps
        print(
            f"  Batch vs single speedup: {speedup:.1f}x"
        )
    if single_fps > 0 and pipe_fps > 0:
        speedup = pipe_fps / single_fps
        print(
            f"  Pipeline vs single speedup: {speedup:.1f}x"
        )

    if gpu_util_end is not None:
        print(f"  GPU util (end): {gpu_util_end:.0f}%")
    if gpu_mem_end is not None:
        print(
            f"  GPU memory (end): "
            f"{gpu_mem_end[0]:.0f}/{gpu_mem_end[1]:.0f} MB"
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Video processing speed benchmark"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to video file",
    )
    parser.add_argument(
        "--frames", type=int, default=200,
        help="Max frames to process (default: 200)",
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Batch size for batch inference (default: 4)",
    )
    parser.add_argument(
        "--input-size", type=int, default=320,
        help="Model input resolution (default: 320)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate and use a synthetic test video",
    )
    args = parser.parse_args()

    if args.input is None or args.synthetic:
        synth_path = str(_PROJECT_ROOT / "test_synthetic_eye.avi")
        if not os.path.exists(synth_path) or args.synthetic:
            _create_synthetic_video(
                synth_path,
                n_frames=max(args.frames, 200),
            )
        video_path = synth_path
    else:
        video_path = args.input

    if not os.path.exists(video_path):
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    benchmark(
        video_path,
        max_frames=args.frames,
        batch_size=args.batch,
        input_size=args.input_size,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()