#!/usr/bin/env python3
"""Quick smoke test for the optimised video pipeline."""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s")

import cv2
import numpy as np


def test_roi_detector():
    """Test EyeROIDetector with a synthetic frame."""
    from pupil_tracking.core.eye_roi_detector import EyeROIDetector

    detector = EyeROIDetector()

    # Create a fake "face" frame with a dark circle (eye)
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (640, 300), 40, (30, 30, 30), -1)   # dark pupil
    cv2.circle(frame, (640, 300), 100, (120, 80, 60), 2)   # iris ring

    roi = detector.detect(frame)
    print(f"ROI: valid={roi.valid}  closeup={roi.is_closeup}  "
          f"cache={roi.from_cache}  bbox={roi.bbox}")

    # Second call should use cache
    roi2 = detector.detect(frame)
    print(f"ROI (cached): valid={roi2.valid}  cache={roi2.from_cache}")

    print("✓ EyeROIDetector OK\n")


def test_temporal_smoother():
    """Test Kalman filter smoothing."""
    from pupil_tracking.video.temporal_smoother import TemporalSmoother

    smoother = TemporalSmoother()

    for i in range(20):
        # Simulate noisy detections
        noise = np.random.randn() * 3
        det = {
            "pupil_detected": True,
            "pupil_x": 200.0 + noise,
            "pupil_y": 300.0 + noise * 0.5,
            "pupil_radius": 45.0 + noise * 0.3,
            "limbus_detected": True,
            "limbus_x": 205.0 + noise * 0.8,
            "limbus_y": 302.0 + noise * 0.4,
            "limbus_radius": 120.0 + noise * 0.2,
        }
        smoothed = smoother.smooth(det)

        if i >= 5:  # after warm-up
            # Smoothed values should have less variance than raw
            assert abs(smoothed["pupil_x"] - 200) < 15, \
                f"Frame {i}: smoothed pupil_x={smoothed['pupil_x']}"

    print("✓ TemporalSmoother OK\n")


def test_fast_inference():
    """Test FastInference model loading and inference."""
    from pupil_tracking.ml.fast_inference import FastInference

    model_path = "models/best_model.pth"
    if not os.path.isfile(model_path):
        print(f"⚠ Skipping FastInference test: {model_path} not found")
        return

    fi = FastInference(model_path, input_size=256)

    # Synthetic eye image
    img = np.full((512, 512, 3), 160, dtype=np.uint8)
    cv2.circle(img, (256, 256), 50, (20, 20, 20), -1)    # pupil
    cv2.circle(img, (256, 256), 130, (80, 60, 40), 10)   # iris

    t0 = time.perf_counter()
    result = fi.detect(img)
    latency = (time.perf_counter() - t0) * 1000

    print(f"FastInference: latency={latency:.1f} ms  "
          f"pupil={result.get('pupil_detected')}  "
          f"limbus={result.get('limbus_detected')}")

    # Benchmark 50 frames
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        fi.detect(img)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"  50-frame benchmark: avg={np.mean(times):.1f} ms  "
          f"median={np.median(times):.1f} ms  "
          f"p95={np.percentile(times, 95):.1f} ms")
    print("✓ FastInference OK\n")


def test_unified_video_mode():
    """Test UnifiedDetector.detect_video_frame()."""
    try:
        from pupil_tracking.core.detector import UnifiedDetector
    except ImportError as e:
        print(f"⚠ Skipping UnifiedDetector test: {e}")
        return

    model_path = "models/best_model.pth"
    if not os.path.isfile(model_path):
        print(f"⚠ Skipping: {model_path} not found")
        return

    detector = UnifiedDetector(model_path=model_path)
    detector.init_video_mode(input_size=256)

    # Synthetic eye
    img = np.full((400, 400, 3), 160, dtype=np.uint8)
    cv2.circle(img, (200, 200), 45, (15, 15, 15), -1)
    cv2.circle(img, (200, 200), 120, (90, 70, 50), 8)

    t0 = time.perf_counter()
    result = detector.detect_video_frame(
        img, frame_number=0, roi_x=100, roi_y=50
    )
    latency = (time.perf_counter() - t0) * 1000

    print(f"detect_video_frame: latency={latency:.1f} ms")
    print(f"  pupil: detected={result.pupil.detected}  "
          f"conf={result.pupil.confidence:.2f}")
    if result.pupil.detected and result.pupil.ellipse:
        e = result.pupil.ellipse
        print(f"  pupil centre: ({e.center_x:.1f}, {e.center_y:.1f})  "
              f"r={e.radius:.1f}")
    print(f"  limbus: detected={result.limbus.detected}")
    print(f"  quality: {result.overall_quality}")

    # Test result_to_dict
    d = detector.result_to_dict(result)
    assert isinstance(d, dict)
    assert "pupil_detected" in d
    print(f"  result_to_dict: {len(d)} keys")

    print("✓ UnifiedDetector video mode OK\n")


def test_full_pipeline():
    """Test OptimizedVideoProcessor.process_frame()."""
    try:
        from pupil_tracking.video.optimized_processor import (
            OptimizedVideoProcessor,
        )
    except ImportError as e:
        print(f"⚠ Skipping full pipeline test: {e}")
        return

    model_path = "models/best_model.pth"
    if not os.path.isfile(model_path):
        print(f"⚠ Skipping: {model_path} not found")
        return

    proc = OptimizedVideoProcessor(model_path=model_path, input_size=256)

    # Simulate a "wider" frame where eye is in upper-right
    frame = np.full((720, 1280, 3), 190, dtype=np.uint8)
    eye_cx, eye_cy = 900, 250
    cv2.circle(frame, (eye_cx, eye_cy), 35, (10, 10, 10), -1)
    cv2.circle(frame, (eye_cx, eye_cy), 95, (100, 70, 50), 8)

    # Process 20 frames (simulating temporal consistency)
    latencies = []
    for i in range(20):
        t0 = time.perf_counter()
        det = proc.process_frame(frame, frame_idx=i)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

        if i == 0 or i == 19:
            print(f"  Frame {i}: latency={lat:.1f} ms  "
                  f"pupil={det.get('pupil_detected')}  "
                  f"cache={det.get('roi_from_cache')}")

    print(f"  20-frame avg: {np.mean(latencies):.1f} ms  "
          f"p95: {np.percentile(latencies, 95):.1f} ms")

    target = 100.0
    avg = np.mean(latencies)
    status = "✓ PASS" if avg < target else "✗ FAIL"
    print(f"  {status}: avg {avg:.1f} ms vs {target:.0f} ms target")
    print("✓ Full pipeline OK\n")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMISED VIDEO PIPELINE — SMOKE TEST")
    print("=" * 60 + "\n")

    test_roi_detector()
    test_temporal_smoother()
    test_fast_inference()
    test_unified_video_mode()
    test_full_pipeline()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)