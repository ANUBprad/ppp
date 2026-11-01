"""static_mode.py — Static Mode (image-based) detection analysis.

ADDITIVE + display-only
-----------------------
"Static Mode" is the *still-image* branch of the workflow, as opposed to the
live video pipeline::

    pre-dock   : the doctor centres and focuses the eye, then captures a still
    post-dock  : a second still is captured for comparison

This module is the pure-logic core behind that window.  Given two already
loaded stills — a **Pentacam** reference and an **ELITA** surgical image — it:

1. runs the *same* ``UnifiedDetector`` pupil/limbus detection the rest of the
   app uses (never modified, only called);
2. builds the iris annulus ROI and unwraps it into a straight polar strip
   (Daugman rubber-sheet — iris texture is *preserved*, nothing is cropped
   outside the annulus);
3. extracts iris features twice: from the source image (``iris.detect``) and
   from the unwrapped strip (``cyclotorsion.iris_features``);
4. estimates the cyclotorsion between the two strips;
5. validates the whole result and serialises it so pre-dock and post-dock
   details can be saved.

Nothing here feeds surgical control, and no existing centration / pupil /
limbus / WTW / calibration behaviour is altered.  No Tk code lives here — the
window lives in :mod:`pupil_tracking.interface.static_mode_window`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ``(height, width)`` of every strip in Static Mode.  Both sources use the
#: *same* size so the engine can correlate them directly.
DEFAULT_OUTPUT_SIZE: Tuple[int, int] = (64, 512)

#: ``angle_offset_deg`` for every Static Mode strip: column 0 is image angle
#: 0° (+x axis) and columns then advance monotonically to 360°, so the ruler
#: reads 0° → 90° → 180° → 270° → 360° and the ``angle_deg`` reported by
#: :func:`cyclotorsion.iris_features.extract_strip_features` is already a true
#: image angle needing no correction.  (The main window's *disc* unwrap panel
#: uses 180° so its centre column is 0°; the offset is a pure cyclic roll, so
#: it cannot bias the correlation either way — only the labels differ.)
DEFAULT_ANGLE_OFFSET_DEG = 0.0

#: Accepted-iris-feature floor for a source to pass validation.
DEFAULT_MIN_FEATURES = 5

#: Correlation-score gate (mirrors ``cyclotorsion_engine.DEFAULT_MIN_SCORE``).
DEFAULT_MIN_SCORE = 0.25

#: |angle| above which an estimate is flagged as implausible.  Post-dock
#: cyclotorsion is clinically a matter of a few degrees; 30° is a deliberately
#: loose honesty gate, **not** a clinical claim.
MAX_ABS_ANGLE_DEG = 30.0

PHASE_PRE_DOCK = "pre_dock"
PHASE_POST_DOCK = "post_dock"
PHASES: Tuple[str, ...] = (PHASE_PRE_DOCK, PHASE_POST_DOCK)

PHASE_LABELS: Dict[str, str] = {
    PHASE_PRE_DOCK: "Pre-dock",
    PHASE_POST_DOCK: "Post-dock",
}

VALID_EYES: Tuple[str, ...] = ("OD", "OS")

#: Extensions accepted by the Browse buttons (JPG / JPEG / PNG / PDF are the
#: contract; the loader also understands DICOM and the usual TIFF/BMP).
IMAGE_FILETYPES: Tuple[Tuple[str, str], ...] = (
    (
        "Eye image files",
        "*.bmp *.BMP *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.tif *.tiff *.TIF *.TIFF *.pdf *.PDF *.dcm *.DCM",
    ),
    ("BMP files (*.bmp; *.BMP)", "*.bmp *.BMP"),
    ("JPEG files (*.jpg; *.jpeg)", "*.jpg *.jpeg *.JPG *.JPEG"),
    ("PNG files (*.png)", "*.png *.PNG"),
    ("TIFF files (*.tif; *.tiff)", "*.tif *.tiff *.TIF *.TIFF"),
    ("DICOM files (*.dcm)", "*.dcm *.DCM"),
    ("PDF files (*.pdf)", "*.pdf *.PDF"),
    ("All files", "*.*"),
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".pdf", ".dcm"}

SCHEMA = "static_mode_cyclotorsion/1"
HISTORY_SCHEMA = "static_mode_cyclotorsion_history/1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_confidence(features: Optional[Iterable[Any]]) -> float:
    """Mean ``confidence`` over *features*; 0.0 when there are none."""
    vals: List[float] = []
    for feat in features or []:
        conf = getattr(feat, "confidence", None)
        if conf is None:
            continue
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if math.isfinite(conf):
            vals.append(conf)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _as_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce gray / 1-channel / BGRA input to a 3-channel BGR image."""
    img = np.asarray(image)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SourceAnalysis:
    """Detection, unwrap and feature summary for ONE still image.

    Holds no heavyweight state: ``strip`` is the only array (and is excluded
    from :meth:`to_dict` so records stay small and JSON-safe).
    """

    label: str = ""            #: "Pentacam" | "Elita"
    path: str = ""
    flipped: bool = False      #: whether the Pentacam 180° auto-flip was applied

    # ── load ────────────────────────────────────────────────────────────
    loaded: bool = False
    height: int = 0
    width: int = 0

    # ── shared pupil/limbus detection (read-only call) ─────────────────
    pupil_detected: bool = False
    pupil_confidence: float = 0.0
    limbus_detected: bool = False
    limbus_confidence: float = 0.0
    overall_confidence: float = 0.0
    image_quality: str = ""

    # ── iris annulus ───────────────────────────────────────────────────
    roi_valid: bool = False
    roi_reason: str = ""
    roi_source: str = ""       #: "detected" | "recorded box" | ""
    pupil_radius_px: float = 0.0
    limbus_radius_px: float = 0.0

    # ── unwrapped strip ────────────────────────────────────────────────
    strip: Optional[np.ndarray] = None
    strip_height: int = 0
    strip_width: int = 0

    # ── detector-level iris features (``iris.detect``) ─────────────────
    iris_features: int = 0
    iris_feature_confidence: float = 0.0
    iris_feature_coverage: float = 0.0
    iris_usable_fraction: float = 0.0
    iris_candidates: int = 0
    iris_status: str = ""

    # ── strip-level iris features (``cyclotorsion.iris_features``) ─────
    strip_features: int = 0
    strip_feature_confidence: float = 0.0
    strip_feature_candidates: int = 0
    strip_feature_status: str = ""
    #: The features themselves, kept for drawing markers on the strip.
    #: Never serialised — :meth:`to_dict` only reports the counts.
    strip_feature_list: List[Any] = field(
        default_factory=list, repr=False, compare=False
    )

    # ── bookkeeping ────────────────────────────────────────────────────
    reason: str = ""
    error: str = ""
    processing_time_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True when the source produced a usable unwrapped iris strip."""
        return bool(self.loaded and self.roi_valid and self.strip is not None)

    @property
    def detected(self) -> bool:
        return bool(self.pupil_detected and self.limbus_detected)

    @property
    def strip_size(self) -> Tuple[int, int]:
        return (self.strip_height, self.strip_width)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary (the strip itself is never serialised)."""
        return {
            "label": self.label,
            "path": self.path,
            "flipped": bool(self.flipped),
            "loaded": bool(self.loaded),
            "image_size": [int(self.height), int(self.width)],
            "pupil": {
                "detected": bool(self.pupil_detected),
                "confidence": round(float(self.pupil_confidence), 4),
            },
            "limbus": {
                "detected": bool(self.limbus_detected),
                "confidence": round(float(self.limbus_confidence), 4),
            },
            "overall_confidence": round(float(self.overall_confidence), 4),
            "image_quality": self.image_quality,
            "iris_roi": {
                "valid": bool(self.roi_valid),
                "reason": self.roi_reason,
                "source": self.roi_source,
                "pupil_radius_px": round(float(self.pupil_radius_px), 2),
                "limbus_radius_px": round(float(self.limbus_radius_px), 2),
            },
            "strip": {
                "size": [int(self.strip_height), int(self.strip_width)],
                "available": self.strip is not None,
            },
            "iris_features": {
                "count": int(self.iris_features),
                "candidates": int(self.iris_candidates),
                "confidence": round(float(self.iris_feature_confidence), 4),
                "coverage": round(float(self.iris_feature_coverage), 4),
                "usable_fraction": round(float(self.iris_usable_fraction), 4),
                "status": self.iris_status,
            },
            "strip_features": {
                "count": int(self.strip_features),
                "candidates": int(self.strip_feature_candidates),
                "confidence": round(float(self.strip_feature_confidence), 4),
                "status": self.strip_feature_status,
            },
            "reason": self.reason,
            "error": self.error,
            "processing_time_ms": round(float(self.processing_time_ms), 1),
        }


@dataclass
class ValidationCheck:
    """One PASS/FAIL item produced by :func:`validate_analysis`."""

    name: str
    passed: bool
    detail: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def line(self) -> str:
        return f"[{self.status}] {self.name}" + (f" — {self.detail}" if self.detail else "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class StaticAnalysis:
    """A full Static Mode result: two sources + cyclotorsion + validation.

    The **reference** is the Pentacam still (sitting, auto-flipped) and the
    **target** is the ELITA still for the selected phase.  ``angle_deg`` is the
    signed rotation that must be applied to the reference strip to align it
    with the target strip (see :mod:`cyclotorsion_engine` for the convention).
    """

    eye: str = "OD"
    phase: str = PHASE_PRE_DOCK

    reference: SourceAnalysis = field(
        default_factory=lambda: SourceAnalysis(label="Pentacam")
    )
    target: SourceAnalysis = field(
        default_factory=lambda: SourceAnalysis(label="Elita")
    )

    # ── cyclotorsion (reference -> target) ─────────────────────────────
    angle_deg: float = 0.0
    score: float = 0.0
    cyclotorsion_valid: bool = False
    cyclotorsion_reason: str = ""
    n_valid: int = 0
    n_total: int = 0
    angle_range_deg: float = 180.0
    min_score: float = DEFAULT_MIN_SCORE

    # ── validation thresholds used for this run ────────────────────────
    min_features: int = DEFAULT_MIN_FEATURES
    max_abs_angle_deg: float = MAX_ABS_ANGLE_DEG

    checks: List[ValidationCheck] = field(default_factory=list)
    timestamp: str = ""
    elapsed_ms: float = 0.0

    @property
    def valid(self) -> bool:
        """True when **every** validation check passed."""
        if not self.checks:
            return False
        return all(c.passed for c in self.checks)

    def passed_count(self) -> Tuple[int, int]:
        passed = sum(1 for c in self.checks if c.passed)
        return passed, len(self.checks)

    def summary(self) -> str:
        passed, total = self.passed_count()
        return f"{passed}/{total} checks passed"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe record (never contains arrays)."""
        passed, total = self.passed_count()
        return {
            "schema": SCHEMA,
            "eye": self.eye,
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, self.phase),
            "timestamp": self.timestamp,
            "elapsed_ms": round(float(self.elapsed_ms), 1),
            "reference": self.reference.to_dict(),
            "target": self.target.to_dict(),
            "cyclotorsion": {
                "reference": self.reference.label,
                "target": self.target.label,
                "angle_deg": round(float(self.angle_deg), 4),
                "score": round(float(self.score), 4),
                "valid": bool(self.cyclotorsion_valid),
                "reason": self.cyclotorsion_reason,
                "n_valid": int(self.n_valid),
                "n_total": int(self.n_total),
                "angle_range_deg": round(float(self.angle_range_deg), 3),
                "min_score": round(float(self.min_score), 4),
            },
            "validation": {
                "valid": self.valid,
                "passed": passed,
                "total": total,
                "checks": [c.to_dict() for c in self.checks],
            },
            "thresholds": {
                "min_features": int(self.min_features),
                "max_abs_angle_deg": float(self.max_abs_angle_deg),
            },
        }


# ---------------------------------------------------------------------------
# Per-source analysis
# ---------------------------------------------------------------------------

def analyse_source(
    image: Optional[np.ndarray],
    *,
    label: str,
    path: str = "",
    detector: Any = None,
    iris_detector: Any = None,
    flip: bool = False,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    angle_offset_deg: float = DEFAULT_ANGLE_OFFSET_DEG,
    source_tag: str = "",
) -> SourceAnalysis:
    """Detect, unwrap and extract features for ONE still.

    Parameters
    ----------
    image:
        BGR/gray still, or ``None`` when the file could not be read.
    label, path:
        Shown in the window and written into saved records.
    detector:
        The app's ``UnifiedDetector``.  ``None`` skips detection (only a
        recorded ``roi_norm`` box can then produce a strip).
    iris_detector:
        Optional reused ``IrisFeatureDetector``; one is created on demand.
        Reusing keeps the (potentially torch-backed) RL agent warm and is the
        recommended path for a window that analyses several stills.
    flip:
        Apply the Pentacam 180° auto-flip (``cv2.flip(image, -1)``).  ELITA
        stills come from the live surgical feed and must NOT be flipped.
    roi_norm:
        Optional recorded normalised iris-ROI box ``(x0, y0, x1, y1)``, used
        only as a fallback when detection cannot produce a usable annulus.
    output_size, angle_offset_deg:
        Strip geometry — identical for every source in a Static Mode run.

    Returns
    -------
    SourceAnalysis
        Never ``None``; failures are reported in ``reason`` / ``error``.
    """
    out = SourceAnalysis(label=label, path=path)
    t0 = time.perf_counter()

    if image is None:
        out.reason = "no image"
        out.error = "image could not be read"
        out.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        return out

    try:
        img = _as_bgr(image)
        if flip:
            img = cv2.flip(img, -1)
            out.flipped = True
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            raise ValueError("empty image")
    except Exception as exc:  # malformed input — report, never raise
        out.reason = "invalid image"
        out.error = str(exc)
        out.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        return out

    out.loaded = True
    out.height, out.width = h, w

    # ── 1. shared pupil/limbus detection (read-only) ──────────────────
    result = None
    if detector is not None:
        try:
            result = detector.detect(
                img, frame_number=-1, source=source_tag or label
            )
        except Exception as exc:
            out.error = f"detection failed: {exc}"
            result = None

    if result is not None:
        out.pupil_detected = bool(getattr(result.pupil, "detected", False))
        out.pupil_confidence = float(getattr(result.pupil, "confidence", 0.0) or 0.0)
        out.limbus_detected = bool(getattr(result.limbus, "detected", False))
        out.limbus_confidence = float(getattr(result.limbus, "confidence", 0.0) or 0.0)
        out.overall_confidence = float(getattr(result, "overall_confidence", 0.0) or 0.0)
        quality = getattr(result, "overall_quality", None)
        out.image_quality = str(getattr(quality, "value", quality) or "")

    # ── 2. iris annulus ROI (detected geometry, else recorded box) ────
    roi = None
    if result is not None:
        from pupil_tracking.iris.roi import IrisROIExtractor

        roi = IrisROIExtractor().build_from_detection(result)
        if getattr(roi, "valid", False):
            out.roi_source = "detected"

    if not getattr(roi, "valid", False) and roi_norm is not None:
        from pupil_tracking.cyclotorsion.pentacam_loader import roi_norm_to_iris_roi

        roi = roi_norm_to_iris_roi(roi_norm, w, h)
        if getattr(roi, "valid", False):
            out.roi_source = "recorded box"

    out.roi_valid = bool(getattr(roi, "valid", False))
    out.roi_reason = str(getattr(roi, "reason", "") or "")
    if out.roi_valid:
        out.pupil_radius_px = float(getattr(roi, "pupil_radius_px", 0.0))
        out.limbus_radius_px = float(getattr(roi, "limbus_radius_px", 0.0))

    # ── 3. unwrap the annulus (iris texture preserved) ────────────────
    if out.roi_valid:
        from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_iris

        try:
            strip = unwrap_iris(img, roi, output_size, angle_offset_deg)
        except Exception as exc:
            strip = None
            out.error = f"unwrap failed: {exc}"
        if strip is None:
            out.reason = "iris unwrap failed"
        else:
            out.strip = strip
            out.strip_height = int(strip.shape[0])
            out.strip_width = int(strip.shape[1])

    # ── 4. detector-level iris features ───────────────────────────────
    if out.pupil_detected and out.limbus_detected:
        try:
            if iris_detector is None:
                from pupil_tracking.iris.detect import IrisFeatureDetector

                iris_detector = IrisFeatureDetector()
            ir = iris_detector.detect(
                img,
                getattr(result.pupil, "ellipse", None),
                getattr(result.limbus, "ellipse", None),
            )
            feature_set = ir.feature_set
            status = getattr(ir, "status", None)
            out.iris_status = str(getattr(status, "value", status) or "")
            out.iris_features = len(getattr(feature_set, "features", []) or [])
            out.iris_candidates = int(getattr(feature_set, "num_candidates", 0) or 0)
            out.iris_feature_coverage = float(
                getattr(feature_set, "region_coverage", 0.0) or 0.0
            )
            out.iris_usable_fraction = float(
                getattr(feature_set, "usable_fraction", 0.0) or 0.0
            )
            out.iris_feature_confidence = _mean_confidence(
                getattr(feature_set, "features", []) or []
            )
        except Exception as exc:
            out.iris_status = "ERROR"
            out.error = out.error or f"iris feature detection failed: {exc}"

    # ── 5. strip-level iris features ──────────────────────────────────
    if out.strip is not None:
        try:
            from pupil_tracking.cyclotorsion.iris_features import extract_strip_features

            sfs = extract_strip_features(out.strip)
            out.strip_feature_status = str(getattr(sfs, "status", "") or "")
            out.strip_features = len(getattr(sfs, "features", []) or [])
            out.strip_feature_candidates = int(
                getattr(sfs, "num_candidates", 0) or 0
            )
            out.strip_feature_list = list(getattr(sfs, "features", []) or [])
            out.strip_feature_confidence = _mean_confidence(
                out.strip_feature_list
            )
        except Exception as exc:
            out.strip_feature_status = "ERROR"
            out.error = out.error or f"strip feature extraction failed: {exc}"

    if not out.reason:
        out.reason = "ok" if out.ok else "no usable iris strip"

    out.processing_time_ms = (time.perf_counter() - t0) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------------

def analyse_static_pair(
    pentacam_image: Optional[np.ndarray],
    elita_image: Optional[np.ndarray],
    *,
    eye: str = "OD",
    phase: str = PHASE_PRE_DOCK,
    detector: Any = None,
    iris_detector: Any = None,
    pentacam_path: str = "",
    elita_path: str = "",
    flip_pentacam: bool = True,
    flip_elita: bool = False,
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    angle_offset_deg: float = DEFAULT_ANGLE_OFFSET_DEG,
    angle_range_deg: float = 180.0,
    min_score: float = DEFAULT_MIN_SCORE,
    min_features: int = DEFAULT_MIN_FEATURES,
    max_abs_angle_deg: float = MAX_ABS_ANGLE_DEG,
) -> StaticAnalysis:
    """Analyse a Pentacam/ELITA pair for one eye and one (pre/post-dock) phase.

    ``pentacam_image`` is the reference (auto-flipped 180° so it matches the
    live camera orientation) and ``elita_image`` is the target.  The reported
    ``angle_deg`` is the signed rotation aligning the reference strip with the
    target strip.

    An ``eye`` outside :data:`VALID_EYES` falls back to ``"OD"`` and an unknown
    ``phase`` falls back to :data:`PHASE_PRE_DOCK` — the window only ever
    offers those choices, so a stray value is a caller bug, not a clinical one.
    The fallback is logged as a warning so a bad value is still visible.
    """
    if eye not in VALID_EYES:
        logger.warning("analyse_static_pair: unknown eye %r, using 'OD'", eye)
        eye = "OD"
    if phase not in PHASES:
        logger.warning(
            "analyse_static_pair: unknown phase %r, using %r", phase, PHASE_PRE_DOCK
        )
        phase = PHASE_PRE_DOCK
    t0 = time.perf_counter()

    out = StaticAnalysis(eye=eye, phase=phase)
    out.angle_range_deg = float(abs(angle_range_deg))
    out.min_score = float(min_score)
    out.min_features = int(min_features)
    out.max_abs_angle_deg = float(max_abs_angle_deg)
    out.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    tag = f"static:{eye}:{phase}"
    out.reference = analyse_source(
        pentacam_image,
        label="Pentacam",
        path=pentacam_path,
        detector=detector,
        iris_detector=iris_detector,
        flip=flip_pentacam,
        roi_norm=roi_norm,
        output_size=output_size,
        angle_offset_deg=angle_offset_deg,
        source_tag=f"{tag}:pentacam",
    )
    out.target = analyse_source(
        elita_image,
        label="Elita",
        path=elita_path,
        detector=detector,
        iris_detector=iris_detector,
        flip=flip_elita,
        roi_norm=None,  # a different eye/image — its ROI must be its own
        output_size=output_size,
        angle_offset_deg=angle_offset_deg,
        source_tag=f"{tag}:elita",
    )

    ref_strip = out.reference.strip
    tgt_strip = out.target.strip
    if ref_strip is not None and tgt_strip is not None:
        from pupil_tracking.cyclotorsion.cyclotorsion_engine import (
            estimate_cyclotorsion,
        )

        try:
            cres = estimate_cyclotorsion(
                ref_strip,
                tgt_strip,
                angle_range_deg=angle_range_deg,
                min_score=min_score,
            )
            out.angle_deg = float(cres.angle_deg)
            out.score = float(cres.score)
            out.cyclotorsion_valid = bool(cres.valid)
            out.cyclotorsion_reason = str(cres.reason or "")
            out.n_valid = int(cres.n_valid)
            out.n_total = int(cres.n_total)
        except Exception as exc:
            out.cyclotorsion_reason = f"engine error: {exc}"
    else:
        out.cyclotorsion_reason = (
            "missing Pentacam strip" if ref_strip is None else "missing Elita strip"
        )
        if ref_strip is None and tgt_strip is None:
            out.cyclotorsion_reason = "missing both iris strips"

    out.checks = validate_analysis(out)
    out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_analysis(analysis: StaticAnalysis) -> List[ValidationCheck]:
    """Return the PASS/FAIL checklist for *analysis*.

    Checks are deliberately explicit (source loaded → geometry detected →
    annulus valid → strip unwrapped → features present → correlation score →
    angle plausible) so the window can show a doctor *why* a run is or is not
    trustworthy rather than a bare boolean.
    """
    checks: List[ValidationCheck] = []
    min_features = int(analysis.min_features)
    max_angle = float(analysis.max_abs_angle_deg)
    min_score = float(analysis.min_score)

    def add(name: str, passed: Any, detail: str = "") -> None:
        checks.append(
            ValidationCheck(name=name, passed=bool(passed), detail=detail)
        )

    for src in (analysis.reference, analysis.target):
        add(
            f"{src.label} image loaded",
            src.loaded,
            src.path or "no file selected",
        )
        add(
            f"{src.label} pupil + limbus detected",
            src.detected,
            f"pupil {src.pupil_confidence:.2f}, limbus {src.limbus_confidence:.2f}",
        )
        add(
            f"{src.label} iris ROI valid",
            src.roi_valid,
            f"{src.roi_reason or 'n/a'}"
            + (f" ({src.roi_source})" if src.roi_source else ""),
        )
        has_strip = src.strip is not None
        add(
            f"{src.label} iris annulus unwrapped",
            has_strip,
            f"{src.strip_height}x{src.strip_width} strip" if has_strip else "",
        )
        add(
            f"{src.label} iris features ≥ {min_features}",
            src.iris_features >= min_features,
            f"{src.iris_features} features, conf {src.iris_feature_confidence:.2f}, "
            f"coverage {src.iris_feature_coverage:.2f}",
        )
        add(
            f"{src.label} strip features ≥ {min_features}",
            src.strip_features >= min_features,
            f"{src.strip_features} features, conf {src.strip_feature_confidence:.2f}",
        )

    add(
        "Cyclotorsion estimated",
        analysis.n_total > 0,
        analysis.cyclotorsion_reason or "ok",
    )
    add(
        f"Correlation score ≥ {min_score:.2f}",
        analysis.score >= min_score,
        f"score {analysis.score:.3f} over {analysis.n_valid}/{analysis.n_total} px",
    )
    add(
        "Engine accepted the estimate",
        analysis.cyclotorsion_valid,
        analysis.cyclotorsion_reason or "ok",
    )
    add(
        f"|angle| ≤ {max_angle:g}°",
        abs(analysis.angle_deg) <= max_angle,
        f"{analysis.angle_deg:+.2f}°",
    )
    return checks


# ---------------------------------------------------------------------------
# Saving (pre- and post-dock cyclotorsion details)
# ---------------------------------------------------------------------------

def default_history_path(patient_dir: Any, eye: str) -> Path:
    """``<patient>/<eye>/Cyclotorsion/static_cyclotorsion.json``."""
    return Path(patient_dir) / eye / "Cyclotorsion" / "static_cyclotorsion.json"


def save_analysis(
    analysis: StaticAnalysis,
    patient_dir: Any,
    *,
    path: Optional[Any] = None,
) -> Path:
    """Append *analysis* to the patient's OD/OS cyclotorsion history.

    The JSON payload keeps a full ``records`` audit trail plus a ``latest``
    dict keyed by phase, so both the pre-dock and post-dock details are always
    directly readable.  Pass ``path`` for an explicit "Save As…" destination;
    the file is then written as a one-record history.

    Raises ``OSError`` on an unwritable destination so the caller can report it.
    """
    target = Path(path) if path is not None else default_history_path(
        patient_dir, analysis.eye
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    record = analysis.to_dict()

    history: List[Dict[str, Any]] = []
    latest: Dict[str, Any] = {}
    if target.is_file():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, dict):
            if isinstance(existing.get("records"), list):
                history = [r for r in existing["records"] if isinstance(r, dict)]
            if isinstance(existing.get("latest"), dict):
                latest = dict(existing["latest"])
        elif isinstance(existing, list):
            history = [r for r in existing if isinstance(r, dict)]

    history.append(record)
    latest[analysis.phase] = record

    payload = {
        "schema": HISTORY_SCHEMA,
        "patient_dir": str(patient_dir),
        "eye": analysis.eye,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest": latest,
        "records": history,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def load_history(patient_dir: Any, eye: str) -> Dict[str, Any]:
    """Read the saved history payload for *eye* (``{}`` when absent/bad)."""
    target = default_history_path(patient_dir, eye)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def latest_records(patient_dir: Any, eye: str) -> Dict[str, Dict[str, Any]]:
    """``{phase: record}`` of the most recent pre/post-dock save for *eye*."""
    data = load_history(patient_dir, eye)
    latest = data.get("latest")
    return dict(latest) if isinstance(latest, dict) else {}


# ---------------------------------------------------------------------------
# OD/OS routing helpers
# ---------------------------------------------------------------------------

def pentacam_reference_dir(patient_dir: Any, eye: str) -> Path:
    """``<patient>/<eye>/Pentacam`` — where that eye's stills live."""
    return Path(patient_dir) / eye / "Pentacam"


def find_reference_image(patient_dir: Any, eye: str) -> Optional[Path]:
    """Most recent Pentacam still recorded for *eye* (OD/OS aware).

    Returns ``None`` when the folder is absent or holds no readable still.
    """
    folder = pentacam_reference_dir(patient_dir, eye)
    if not folder.is_dir():
        return None
    try:
        candidates = [
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in _IMAGE_EXTS
            and not p.name.lower().endswith("_unwrapped.png")
        ]
    except OSError:
        return None
    if not candidates:
        return None
    try:
        eye_lower = eye.lower()
        for pref in (f"pentacam_{eye_lower}_flipped", "pentacam_original"):
            for c in candidates:
                if c.stem.lower() == pref or c.name.lower().startswith(pref):
                    return c
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def read_roi_norm(
    patient_dir: Any, eye: str
) -> Optional[Tuple[float, float, float, float]]:
    """The iris-ROI box recorded for *eye* in ``pentacam_values.txt``.

    Used as the fallback annulus when detection cannot find a usable ring on
    the Pentacam reference.  Returns ``None`` when absent or malformed.
    """
    from pupil_tracking.cyclotorsion.pentacam_loader import parse_roi_norm

    values = pentacam_reference_dir(patient_dir, eye) / "pentacam_values.txt"
    try:
        text = values.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("iris_roi_norm="):
            return parse_roi_norm(line.split("=", 1)[1])
    return None


__all__ = [
    "DEFAULT_OUTPUT_SIZE",
    "DEFAULT_ANGLE_OFFSET_DEG",
    "DEFAULT_MIN_FEATURES",
    "DEFAULT_MIN_SCORE",
    "MAX_ABS_ANGLE_DEG",
    "PHASES",
    "PHASE_LABELS",
    "PHASE_PRE_DOCK",
    "PHASE_POST_DOCK",
    "VALID_EYES",
    "IMAGE_FILETYPES",
    "SourceAnalysis",
    "ValidationCheck",
    "StaticAnalysis",
    "analyse_source",
    "analyse_static_pair",
    "validate_analysis",
    "default_history_path",
    "save_analysis",
    "load_history",
    "latest_records",
    "pentacam_reference_dir",
    "find_reference_image",
    "read_roi_norm",
]
