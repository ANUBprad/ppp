#!/usr/bin/env python3
"""
Quick-start script for optimised video processing.

Usage:
    python scripts/process_video.py --input video.mp4
    python scripts/process_video.py --input video.mp4 --output result.mp4 --csv results.csv
    python scripts/process_video.py --camera 0
    python scripts/process_video.py --input video.mp4 --stride 2 --device cuda --preview

Run with --help for all options.
"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pupil_tracking.video.optimized_processor import OptimizedVideoProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Optimised Pupil & Limbus Video Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input eye_video.mp4
  %(prog)s --input eye_video.mp4 --output annotated.mp4 --csv results.csv
  %(prog)s --input eye_video.mp4 --stride 2 --device cuda --preview
  %(prog)s --camera 0
  %(prog)s --benchmark --input eye_video.mp4
        """,
    )

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str,
                       help="Path to input video file")
    group.add_argument("--camera", "-c", type=int, default=None,
                       help="Camera device ID (0 for default webcam)")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path for annotated output video")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path for CSV results file")
    parser.add_argument("--json", type=str, default=None,
                        help="Path for JSON results file")

    # Model
    parser.add_argument("--model", "-m", type=str,
                        default="models/best_model.pth",
                        help="Path to model checkpoint (default: models/best_model.pth)")
    parser.add_argument("--device", "-d", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Compute device (default: auto)")

    # Performance
    parser.add_argument("--input-size", type=int, default=256,
                        help="Model input resolution (default: 256, use 512 for accuracy)")
    parser.add_argument("--stride", "-s", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N processed frames")
    parser.add_argument("--no-half", action="store_true",
                        help="Disable FP16 half-precision (use if NaN issues)")

    # ROI
    parser.add_argument("--roi-cache", type=int, default=12,
                        help="Reuse cached eye ROI for N frames (default: 12)")
    parser.add_argument("--roi-padding", type=float, default=0.6,
                        help="Extra padding around detected eye (default: 0.6)")

    # Smoothing
    parser.add_argument("--process-noise", type=float, default=2.0,
                        help="Kalman process noise (lower = smoother)")
    parser.add_argument("--measurement-noise", type=float, default=4.0,
                        help="Kalman measurement noise (lower = more responsive)")

    # Display
    parser.add_argument("--preview", action="store_true",
                        help="Show live preview window")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    # Camera-specific
    parser.add_argument("--resolution", type=str, default="1280x720",
                        help="Camera resolution WxH (default: 1280x720)")
    parser.add_argument("--flip", action="store_true",
                        help="Mirror camera horizontally")

    # Benchmark
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmark on first 300 frames")

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build processor
    processor = OptimizedVideoProcessor(
        model_path=args.model,
        device=args.device,
        input_size=args.input_size,
        half_precision=not args.no_half,
        roi_cache_ttl=args.roi_cache,
        roi_padding=args.roi_padding,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )

    if args.camera is not None:
        # --- Camera mode ---
        w, h = map(int, args.resolution.split("x"))
        processor.process_camera(
            camera_id=args.camera,
            resolution=(w, h),
            flip_horizontal=args.flip,
            save_frames_dir="camera_captures",
        )
    else:
        # --- Video file mode ---
        if not os.path.isfile(args.input):
            print(f"ERROR: File not found: {args.input}")
            sys.exit(1)

        # Auto-generate output paths if not specified
        stem = os.path.splitext(os.path.basename(args.input))[0]
        output_path = args.output
        csv_path = args.csv
        json_path = args.json

        if output_path is None and not args.benchmark:
            output_path = f"{stem}_tracked.mp4"

        if csv_path is None and not args.benchmark:
            csv_path = f"{stem}_results.csv"

        max_frames = args.max_frames
        if args.benchmark:
            max_frames = 300
            output_path = None
            csv_path = None

        results = processor.process_video(
            input_path=args.input,
            output_path=output_path,
            csv_path=csv_path,
            stride=args.stride,
            max_frames=max_frames,
            show_preview=args.preview,
        )

        # Save JSON
        if json_path or (args.benchmark and not json_path):
            json_out = json_path or f"{stem}_results.json"
            OptimizedVideoProcessor.save_results_json(results, json_out)

        # Print summary
        if results:
            import numpy as np
            latencies = [r.get("latency_ms", 0) for r in results]
            detected = sum(1 for r in results if r.get("pupil_detected"))
            print("\n" + "=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)
            print(f"  Frames processed : {len(results)}")
            print(f"  Pupil detected   : {detected}/{len(results)} "
                  f"({detected/len(results)*100:.0f}%)")
            print(f"  Avg latency      : {np.mean(latencies):.1f} ms")
            print(f"  Median latency   : {np.median(latencies):.1f} ms")
            print(f"  95th percentile  : {np.percentile(latencies, 95):.1f} ms")
            print(f"  Max latency      : {np.max(latencies):.1f} ms")
            print(f"  Effective FPS    : {1000/np.mean(latencies):.1f}")
            if output_path:
                print(f"  Output video     : {output_path}")
            if csv_path:
                print(f"  CSV results      : {csv_path}")
            print("=" * 60)


if __name__ == "__main__":
    main()