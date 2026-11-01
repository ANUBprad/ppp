"""evaluate_pentacam_detection.py — Evaluates detection on clean clinical images.

Measures:
- Pupil detection success rate
- Limbus detection success rate
- Pupil center error (pixels): median, mean, p95, max
- Limbus center error (pixels): median, mean, p95, max
- Pupil radius error (pixels): median, mean, p95, max
- Limbus radius error (pixels): median, mean, p95, max
- Overall confidence and processing latency (ms)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

# Ensure project root in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.interface.gui_app import PupilTrackingGUI


def evaluate_dataset(ann_path: Path, img_dir: Path) -> Dict[str, Any]:
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    detector = UnifiedDetector()

    results: List[Dict[str, Any]] = []
    pupil_center_errs: List[float] = []
    pupil_radius_errs: List[float] = []
    limbus_center_errs: List[float] = []
    limbus_radius_errs: List[float] = []
    latencies_ms: List[float] = []

    print("=" * 82)
    print("  CLINICAL DATASET DETECTION EVALUATION")
    print("=" * 82)
    print(f"  {'Image':14s} | {'Pupil':7s} | {'P-Err':7s} | {'Limbus':7s} | {'L-Err':7s} | {'Conf':5s} | {'Latency':7s}")
    print("  " + "-" * 78)

    for img_name, entry in gt_data.items():
        img_file = img_dir / img_name
        if not img_file.is_file():
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        t0 = time.perf_counter()
        res = detector.detect(img, frame_number=-1, source=f"pentacam:{img_name}")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        ann = entry.get("annotations", {})
        gt_p = ann.get("PUPIL")
        gt_l = ann.get("LIMBUS")

        pupil_obj = getattr(res, "pupil", None)
        p_ok = bool(getattr(pupil_obj, "detected", False)) and getattr(pupil_obj, "ellipse", None) is not None
        limbus_obj = getattr(res, "limbus", None)
        l_ok = bool(getattr(limbus_obj, "detected", False)) and getattr(limbus_obj, "ellipse", None) is not None

        p_err = None
        p_r_err = None
        if p_ok and gt_p:
            pe = pupil_obj.ellipse
            p_err = math.hypot(pe.center_x - gt_p["center_x"], pe.center_y - gt_p["center_y"])
            pupil_center_errs.append(p_err)
            gt_pr = 0.5 * (gt_p["semi_major"] + gt_p["semi_minor"])
            det_pr = 0.5 * (pe.semi_major + pe.semi_minor)
            p_r_err = abs(det_pr - gt_pr)
            pupil_radius_errs.append(p_r_err)

        l_err = None
        l_r_err = None
        if l_ok and gt_l:
            le = limbus_obj.ellipse
            l_err = math.hypot(le.center_x - gt_l["center_x"], le.center_y - gt_l["center_y"])
            limbus_center_errs.append(l_err)
            gt_lr = 0.5 * (gt_l["semi_major"] + gt_l["semi_minor"])
            det_lr = 0.5 * (le.semi_major + le.semi_minor)
            l_r_err = abs(det_lr - gt_lr)
            limbus_radius_errs.append(l_r_err)

        p_str = "OK" if p_ok else "FAIL"
        l_str = "OK" if l_ok else "FAIL"
        p_err_str = f"{p_err:5.1f}px" if p_err is not None else "  -  "
        l_err_str = f"{l_err:5.1f}px" if l_err is not None else "  -  "
        conf_str = f"{res.overall_confidence:.2f}" if res is not None else "0.00"

        print(f"  {img_name:14s} | {p_str:7s} | {p_err_str:7s} | {l_str:7s} | {l_err_str:7s} | {conf_str:5s} | {dt_ms:6.1f}ms")

        results.append({
            "image": img_name,
            "pupil_detected": p_ok,
            "limbus_detected": l_ok,
            "pupil_center_err": p_err,
            "limbus_center_err": l_err,
            "pupil_radius_err": p_r_err,
            "limbus_radius_err": l_r_err,
            "confidence": getattr(res, "overall_confidence", 0.0),
            "latency_ms": dt_ms,
        })

    print("  " + "-" * 78)
    n = len(results)
    p_success = sum(1 for r in results if r["pupil_detected"])
    l_success = sum(1 for r in results if r["limbus_detected"])

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    summary = {
        "num_images": n,
        "pupil_success_count": p_success,
        "pupil_success_rate": p_success / n if n > 0 else 0.0,
        "limbus_success_count": l_success,
        "limbus_success_rate": l_success / n if n > 0 else 0.0,
        "pupil_center_err_stats": _stats(pupil_center_errs),
        "pupil_radius_err_stats": _stats(pupil_radius_errs),
        "limbus_center_err_stats": _stats(limbus_center_errs),
        "limbus_radius_err_stats": _stats(limbus_radius_errs),
        "latency_stats": {
            "mean_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
            "median_ms": float(np.median(latencies_ms)) if latencies_ms else 0.0,
            "min_ms": float(np.min(latencies_ms)) if latencies_ms else 0.0,
            "max_ms": float(np.max(latencies_ms)) if latencies_ms else 0.0,
        },
    }

    print(f"  Summary ({n} images):")
    print(f"  Pupil Detection:  {p_success}/{n} ({summary['pupil_success_rate']:.1%})")
    print(f"  Limbus Detection: {l_success}/{n} ({summary['limbus_success_rate']:.1%})")
    p_c = summary["pupil_center_err_stats"]
    print(f"  Pupil Center Err: median={p_c['median']:.1f}px, mean={p_c['mean']:.1f}px, p95={p_c['p95']:.1f}px, max={p_c['max']:.1f}px")
    l_c = summary["limbus_center_err_stats"]
    print(f"  Limbus Center Err: median={l_c['median']:.1f}px, mean={l_c['mean']:.1f}px, p95={l_c['p95']:.1f}px, max={l_c['max']:.1f}px")
    lat = summary["latency_stats"]
    print(f"  Latency: mean={lat['mean_ms']:.1f}ms, min={lat['min_ms']:.1f}ms, max={lat['max_ms']:.1f}ms")
    print("=" * 82)

    return summary


if __name__ == "__main__":
    ann = _ROOT / "clinical_data" / "clean" / "annotations" / "annotations.json"
    imgs = _ROOT / "clinical_data" / "clean"
    evaluate_dataset(ann, imgs)
