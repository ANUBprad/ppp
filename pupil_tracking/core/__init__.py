"""
Core detection and geometry modules.

Provides:
    - UnifiedDetector        : Main detection orchestrator (plan-aligned)
    - SmartContourFitter     : Adaptive circle/ellipse fitting with RANSAC
    - CornealCenterCalculator: Corneal center and offset calculation
    - EllipseFitter          : Legacy ellipse fitting (kept for backward compat)
    - EyeROIDetector         : Eye region-of-interest detection for video

The UnifiedDetector is the primary entry point for all detection tasks.
It orchestrates ML segmentation, smart fitting, cross-validation,
calibration, and corneal center calculation.

Usage
-----
>>> from pupil_tracking.core import UnifiedDetector
>>> detector = UnifiedDetector()
>>> result = detector.detect(image_bgr)
>>> if result.has_both:
...     print(f"Pupil: {result.pupil.ellipse.center_x:.1f}, "
...           f"{result.pupil.ellipse.center_y:.1f}")
...     print(f"Corneal offset: {result.corneal_center.offset_magnitude_mm:.2f} mm")
"""

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.smart_fitter import (
    SmartContourFitter,
    FitResult,
    FitType,
    smart_fit,
)

try:
    from pupil_tracking.core.corneal_center import CornealCenterCalculator
except ImportError:
    CornealCenterCalculator = None

try:
    from pupil_tracking.core.ellipse_fitter import EllipseFitter
except ImportError:
    EllipseFitter = None

try:
    from pupil_tracking.core.eye_roi_detector import EyeROIDetector
except ImportError:
    EyeROIDetector = None

__all__ = [
    "UnifiedDetector",
    "SmartContourFitter",
    "FitResult",
    "FitType",
    "smart_fit",
    "CornealCenterCalculator",
    "EllipseFitter",
    "EyeROIDetector",
]