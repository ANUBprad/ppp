"""dynamic_mode.py — Dynamic (Live) Mode cyclotorsion monitoring.

ADDITIVE + display-only
-----------------------
"Dynamic Mode" is the **live** branch of the cyclotorsion feature, as opposed
to :mod:`pupil_tracking.cyclotorsion.static_mode` (the still-image branch)::

    live camera  ->  one strip per frame  ->  compared with the Pentacam
                     (still) baseline every frame, continuously

This module is the pure-logic core behind that window.  Unlike Static Mode it
does **not** run the ML pupil/limbus detector on every frame — the main window's
existing live detection already produced the pupil and limbus ellipses for the
current frame.  Given those ellipses, the live frame's iris annulus is:

1. turned into the same ``IrisROI`` annulus band the still-image path uses;
2. unwrapped into a straight polar strip (Daugman normalisation — iris texture
   is *preserved*, nothing outside the annulus is used, nothing is cropped);
3. correlated against a cached Pentacam baseline strip of the same geometry to
   recover the frame's torsional offset.

The unwrap is geometry-driven and read-only.  **Nothing here feeds surgical
control, and no existing centration / pupil / limbus / WTW / calibration
behaviour is altered.**  No Tk code lives here — the window lives in
:mod:`pupil_tracking.interface.dynamic_mode_window`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.cyclotorsion.static_mode import (
    DEFAULT_ANGLE_OFFSET_DEG,
    DEFAULT_MIN_SCORE,
    DEFAULT_OUTPUT_SIZE,
    MAX_ABS_ANGLE_DEG,
    VALID_EYES,
    _as_bgr,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Full-circle search by default, matching the still-image engine.
DEFAULT_ANGLE_RANGE_DEG = 180.0

#: Accepted |angle| for a live sample to count as plausible (mirrors the
#: still-image plausibility gate; a deliberately loose honesty gate only).
DEFAULT_MAX_ABS_ANGLE_DEG = MAX_ABS_ANGLE_DEG

SCHEMA = "dynamic_mode_cyclotorsion/1"
HISTORY_SCHEMA = "dynamic_mode_cyclotorsion_history/1"


# ---------------------------------------------------------------------------
# Pentacam baseline (built once per eye)
# ---------------------------------------------------------------------------

@dataclass
class ReferenceStrip:
    """Detection + unwrap summary for the Pentacam baseline still.

    Mirrors :class:`~pupil_tracking.cyclotorsion.static_mode.SourceAnalysis`
    but deliberately keeps only the strip geometry (no iris-feature pass), so
    building a baseline is cheap and never loads a torch model.
    """

    label: str = "Pentacam"
    path: str = ""
    flipped: bool = False

    loaded: bool = False
    height: int = 0
    width: int = 0

    pupil_detected: bool = False
    pupil_confidence: float = 0.0
    limbus_detected: bool = False
    limbus_confidence: float = 0.0
    overall_confidence: float = 0.0
    image_quality: str = ""

    roi_valid: bool = False
    roi_reason: str = ""
    roi_source: str = ""
    pupil_radius_px: float = 0.0
    limbus_radius_px: float = 0.0

    #: The unwrapped strip (excluded from ``to_dict``).
    strip: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    strip_height: int = 0
    strip_width: int = 0

    reason: str = ""
    error: str = ""
    processing_time_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.loaded and self.roi_valid and self.strip is not None)

    @property
    def detected(self) -> bool:
        return bool(self.pupil_detected and self.limbus_detected)

    def to_dict(self) -> Dict[str, Any]:
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
            "reason": self.reason,
            "error": self.error,
            "processing_time_ms": round(float(self.processing_time_ms), 1),
        }


def build_reference_strip(
    image: Optional[np.ndarray],
    *,
    detector: Any = None,
    flip: bool = True,
    path: str = "",
    label: str = "Pentacam",
    roi_norm: Optional[Tuple[float, float, float, float]] = None,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    angle_offset_deg: float = DEFAULT_ANGLE_OFFSET_DEG,
    source_tag: str = "",
) -> ReferenceStrip:
    """Detect + unwrap the Pentacam baseline once for a Dynamic Mode session.

    ``flip`` applies the Pentacam 180° auto-flip (``cv2.flip(image, -1)``) so
    the baseline sits in the live camera's frame; the flip is read-only.  The
    ``roi_norm`` box is used only when detection cannot find a usable annulus
    (OD/OS-aware recorded box).  Never raises — failures are reported in
    ``reason`` / ``error``.
    """
    out = ReferenceStrip(label=label, path=path)
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
        out.image_quality = str(getattr(getattr(result, "overall_quality", None), "value", "") or "")

    roi = None
    if result is not None:
        try:
            from pupil_tracking.iris.roi import IrisROIExtractor

            roi = IrisROIExtractor().build_from_detection(result)
        except Exception:
            roi = None
        if roi is not None and getattr(roi, "valid", False):
            out.roi_source = "detected"

    if (roi is None or not getattr(roi, "valid", False)) and roi_norm is not None:
        try:
            from pupil_tracking.cyclotorsion.pentacam_loader import roi_norm_to_iris_roi

            roi = roi_norm_to_iris_roi(roi_norm, h, w)
        except Exception:
            roi = None
        if roi is not None and getattr(roi, "valid", False):
            out.roi_source = "recorded box"

    out.roi_valid = bool(roi is not None and getattr(roi, "valid", False))
    out.roi_reason = str(getattr(roi, "reason", "") or "")
    if out.roi_valid:
        out.pupil_radius_px = float(getattr(roi, "pupil_radius_px", 0.0))
        out.limbus_radius_px = float(getattr(roi, "limbus_radius_px", 0.0))
        try:
            from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_iris

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

    if not out.reason:
        out.reason = "ok" if out.ok else "no usable iris strip"

    out.processing_time_ms = (time.perf_counter() - t0) * 1000.0
    return out


# ---------------------------------------------------------------------------
# One live frame
# ---------------------------------------------------------------------------

@dataclass
class LiveFrameAnalysis:
    """Unwrap + cyclotorsion for ONE live frame.

    ``angle_deg`` is the signed rotation that must be applied to the Pentacam
    (*reference*) strip to line it up with this frame's strip, i.e. exactly the
    static-mode convention.  ``within_limit`` is the loose plausibility gate.
    The strip array is never serialised.
    """

    eye: str = "OD"
    frame_number: int = 0
    timestamp: str = ""
    reference_label: str = "Pentacam"

    loaded: bool = False
    height: int = 0
    width: int = 0

    pupil_detected: bool = False
    pupil_confidence: float = 0.0
    limbus_detected: bool = False
    limbus_confidence: float = 0.0
    overall_confidence: float = 0.0
    image_quality: str = ""

    roi_valid: bool = False
    roi_reason: str = ""
    roi_source: str = ""
    pupil_radius_px: float = 0.0
    limbus_radius_px: float = 0.0

    strip: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    strip_height: int = 0
    strip_width: int = 0

    angle_deg: float = 0.0
    score: float = 0.0
    valid: bool = False
    within_limit: bool = False
    reason: str = ""
    n_valid: int = 0
    n_total: int = 0

    elapsed_ms: float = 0.0

    @property
    def has_strip(self) -> bool:
        return self.strip is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "eye": self.eye,
            "frame_number": int(self.frame_number),
            "timestamp": self.timestamp,
            "reference": self.reference_label,
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
            "cyclotorsion": {
                "angle_deg": round(float(self.angle_deg), 4),
                "score": round(float(self.score), 4),
                "valid": bool(self.valid),
                "within_limit": bool(self.within_limit),
                "reason": self.reason,
                "n_valid": int(self.n_valid),
                "n_total": int(self.n_total),
            },
            "elapsed_ms": round(float(self.elapsed_ms), 1),
        }


def analyse_live_frame(
    image: Optional[np.ndarray],
    detection: Any,
    reference: Optional[ReferenceStrip],
    *,
    eye: str = "OD",
    frame_number: int = 0,
    reference_label: str = "Pentacam",
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    angle_offset_deg: float = DEFAULT_ANGLE_OFFSET_DEG,
    angle_range_deg: float = DEFAULT_ANGLE_RANGE_DEG,
    min_score: float = DEFAULT_MIN_SCORE,
    max_abs_angle_deg: float = DEFAULT_MAX_ABS_ANGLE_DEG,
) -> LiveFrameAnalysis:
    """Unwrap ONE live frame and estimate its torsion against *reference*.

    Parameters
    ----------
    image:
        The live BGR frame (never flipped — the camera already defines the
        orientation; the *reference* carries the flip).
    detection:
        The frame's already-computed ``EyeDetectionResult`` (``pupil`` /
        ``limbus``).  ``None`` or an undetected limbus yields no strip.  This
        module never runs its own detector on the live path.
    reference:
        The Pentacam baseline produced by :func:`build_reference_strip`.  When
        it has no strip the cyclotorsion is reported as invalid.
    eye:
        Coerced to OD/OS (the window only offers those).  An unknown value is
        logged as a warning.
    """
    if eye not in VALID_EYES:
        logger.warning("analyse_live_frame: unknown eye %r, using 'OD'", eye)
        eye = "OD"
    out = LiveFrameAnalysis(
        eye=eye,
        frame_number=int(frame_number),
        reference_label=reference_label,
    )
    t0 = time.perf_counter()
    out.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if image is None:
        out.reason = "no live frame"
        out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out

    try:
        img = _as_bgr(image)
        h, w = int(img.shape[0]), int(img.shape[1])
        if h <= 0 or w <= 0:
            raise ValueError("empty image")
    except Exception as exc:
        out.reason = "invalid frame"
        out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        out.roi_reason = str(exc)
        return out

    out.loaded = True
    out.height, out.width = h, w

    if detection is not None:
        out.pupil_detected = bool(getattr(getattr(detection, "pupil", None), "detected", False))
        out.pupil_confidence = float(getattr(getattr(detection, "pupil", None), "confidence", 0.0) or 0.0)
        out.limbus_detected = bool(getattr(getattr(detection, "limbus", None), "detected", False))
        out.limbus_confidence = float(getattr(getattr(detection, "limbus", None), "confidence", 0.0) or 0.0)
        out.overall_confidence = float(getattr(detection, "overall_confidence", 0.0) or 0.0)
        out.image_quality = str(getattr(getattr(detection, "overall_quality", None), "value", "") or "")

    # ── 1. iris annulus ROI from the frame's own detection ──────────────
    roi = None
    if detection is not None:
        try:
            from pupil_tracking.iris.roi import IrisROIExtractor

            roi = IrisROIExtractor().build_from_detection(detection)
        except Exception:
            roi = None
    out.roi_valid = bool(getattr(roi, "valid", False))
    out.roi_reason = str(getattr(roi, "reason", "") or "")
    if out.roi_valid:
        out.roi_source = "detected"
        out.pupil_radius_px = float(getattr(roi, "pupil_radius_px", 0.0))
        out.limbus_radius_px = float(getattr(roi, "limbus_radius_px", 0.0))

    # ── 2. unwrap the annulus (iris texture preserved) ──────────────────
    if out.roi_valid:
        from pupil_tracking.cyclotorsion.polar_unwrap import unwrap_iris

        strip = unwrap_iris(img, roi, output_size, angle_offset_deg)
        if strip is None:
            out.reason = "iris unwrap failed"
        else:
            out.strip = strip
            out.strip_height = int(strip.shape[0])
            out.strip_width = int(strip.shape[1])

    if out.strip is None:
        if not out.reason:
            out.reason = "no usable iris strip"
        out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out

    ref_strip = reference.strip if reference is not None else None
    if ref_strip is None:
        out.reason = "no Pentacam baseline strip"
        out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return out

    # ── 3. cyclotorsion (reference -> this frame) ───────────────────────
    from pupil_tracking.cyclotorsion.cyclotorsion_engine import (
        estimate_cyclotorsion,
    )

    cres = estimate_cyclotorsion(
        ref_strip,
        out.strip,
        angle_range_deg=angle_range_deg,
        min_score=min_score,
    )
    out.angle_deg = float(cres.angle_deg)
    out.score = float(cres.score)
    out.valid = bool(cres.valid)
    out.reason = str(cres.reason or "")
    out.n_valid = int(cres.n_valid)
    out.n_total = int(cres.n_total)
    out.within_limit = abs(out.angle_deg) <= float(max_abs_angle_deg)

    out.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return out


# ---------------------------------------------------------------------------
# Saving live samples
# ---------------------------------------------------------------------------

def default_history_path(patient_dir: Any, eye: str) -> Path:
    """``<patient>/<eye>/Cyclotorsion/dynamic_cyclotorsion.json``."""
    return Path(patient_dir) / eye / "Cyclotorsion" / "dynamic_cyclotorsion.json"


def save_sample(
    analysis: LiveFrameAnalysis,
    patient_dir: Any,
    *,
    path: Optional[Any] = None,
) -> Path:
    """Append *analysis* to the patient's OD/OS live cyclotorsion history.

    The payload keeps a ``records`` audit trail plus a ``latest`` record, so a
    live sample is always directly readable.  Pass ``path`` for an explicit
    "Save As…" destination.  Raises ``OSError`` on an unwritable destination.
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
    latest = record

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
    """Read the saved live history payload for *eye* (``{}`` when absent)."""
    target = default_history_path(patient_dir, eye)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


__all__ = [
    "DEFAULT_ANGLE_RANGE_DEG",
    "DEFAULT_MAX_ABS_ANGLE_DEG",
    "SCHEMA",
    "HISTORY_SCHEMA",
    "ReferenceStrip",
    "LiveFrameAnalysis",
    "build_reference_strip",
    "analyse_live_frame",
    "default_history_path",
    "save_sample",
    "load_history",
]
