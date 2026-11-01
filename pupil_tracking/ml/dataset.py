"""
Training dataset with IMAGE-LEVEL splits and video-grade augmentation.

Fixes:
    - Data-leakage: augmented copies of the same source image NEVER
      appear in both train and validation sets.
    - Annotation format: handles the exact project annotation format
      with PUPIL/LIMBUS/RING entries, ellipse parameters, and
      optional boundary points.
    - Video augmentation: motion blur, JPEG compression, resolution
      jitter, partial occlusion.
    - Mask generation: uses ellipse parameters for smooth masks, with
      optional boundary-point polygon fill for dense annotations.
    - Grayscale augmentation: optionally converts a fraction of
      training images to enhanced grayscale (3-channel replicated),
      making the model robust to both colour and grayscale input
      without any architecture changes.

Class configurations:
    - 3-class (legacy):      0=background  1=pupil  2=iris
    - 4-class (ring-aware):  0=background  1=pupil  2=iris  3=suction_ring

When ``num_classes=4``, the RING annotation from the project format is
drawn onto the mask as class 3.  Images without a ring annotation simply
have no class-3 pixels, which is correct — the loss function class
weights handle the resulting imbalance.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALB = True
except ImportError:
    _HAS_ALB = False

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger

# ═══════════════════════════════════════════════════════════════════════
# CHANGE 1 of 4 — Import grayscale augmentation transform
# ═══════════════════════════════════════════════════════════════════════
try:
    from pupil_tracking.ml.grayscale_augmentation import (
        RandomGrayscaleConversion,
    )
    _HAS_GRAYSCALE_AUG = True
except ImportError:
    _HAS_GRAYSCALE_AUG = False
# ═══════════════════════════════════════════════════════════════════════
# END CHANGE 1
# ═══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Annotation loading — handles the exact project format
# ══════════════════════════════════════════════════════════════════════


def load_annotations(
    annotation_path: str,
    ring_labels_path: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Load annotations.json in the project-specific format.

    Expected format::

        {
          "eye_01.jpeg": {
            "image_path": "clinical_data/clean/eye_01.jpeg",
            "image_width": 698,
            "image_height": 655,
            "annotations": {
              "PUPIL": {
                "class_id": 1,
                "center_x": ..., "center_y": ...,
                "semi_major": ..., "semi_minor": ...,
                "angle_deg": ...,
                "boundary_points": [[x, y], ...]
              },
              "LIMBUS": { ... },
              "RING": { ... }          // optional
            }
          },
          ...
        }

    Parameters
    ----------
    annotation_path : str
        Path to the main annotations JSON file.
    ring_labels_path : str or None
        Optional path to ``ring_labels.json`` containing per-image
        ring-presence labels.  When provided, this supplements the
        ``has_suction_ring`` flag in annotations that lack a RING
        entry.  Format::

            {
              "image_001.jpg": {"ring_present": true},
              "image_002.jpg": {"ring_present": false},
              ...
            }

    Returns
    -------
    (image_ids, annotations_dict)
        image_ids : sorted list of stem names (e.g. ``["eye_01", "eye_02", ...]``)
        annotations_dict : ``{stem: normalised_annotation}``
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with open(path) as fh:
        raw = json.load(fh)

    logger = get_logger()
    annotations: Dict[str, Dict[str, Any]] = {}

    # ── detect format ───────────────────────────────────────────
    # Project format: top-level keys are filenames
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None)
        if isinstance(first_val, dict) and "annotations" in first_val:
            # ── PROJECT FORMAT ──────────────────────────────────
            annotations = _parse_project_format(raw, logger)
        elif isinstance(first_val, dict) and (
            "pupil_center" in first_val or "pupil" in first_val
        ):
            # ── FLAT FORMAT (legacy) ────────────────────────────
            annotations = _parse_flat_format(raw, logger)
        elif "images" in raw or "annotations" in raw:
            # ── LIST FORMAT ─────────────────────────────────────
            entries = raw.get("images") or raw.get("annotations", [])
            annotations = _parse_list_format(entries, logger)
        else:
            # ── ASSUME {stem: annotation} ───────────────────────
            annotations = _parse_flat_format(raw, logger)
    elif isinstance(raw, list):
        annotations = _parse_list_format(raw, logger)
    else:
        raise ValueError(
            f"Unrecognised annotation format: {type(raw)}"
        )

    # ── Merge ring labels if provided ───────────────────────────
    if ring_labels_path is not None:
        annotations = _merge_ring_labels(annotations, ring_labels_path, logger)

    image_ids = sorted(annotations.keys())
    logger.info(
        "Loaded %d annotations from %s", len(image_ids), annotation_path
    )
    return image_ids, annotations


def _merge_ring_labels(
    annotations: Dict[str, Dict[str, Any]],
    ring_labels_path: str,
    logger,
) -> Dict[str, Dict[str, Any]]:
    """Merge ring_labels.json into the annotation dict.

    For each image in ``ring_labels.json``, sets the
    ``has_suction_ring`` flag in the annotation if not already set
    from a RING annotation entry.  Format::

        {
          "image_001.jpg": {"ring_present": true},
          "image_002.jpg": {"ring_present": false},
          ...
        }

    Parameters
    ----------
    annotations : dict
        Existing parsed annotations.
    ring_labels_path : str
        Path to ``ring_labels.json``.
    logger
        Logger instance.

    Returns
    -------
    dict
        Updated annotations dict.
    """
    rl_path = Path(ring_labels_path)
    if not rl_path.exists():
        logger.debug("Ring labels file not found: %s — skipping", rl_path)
        return annotations

    try:
        with open(rl_path) as f:
            ring_labels = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Could not parse ring labels %s: %s", rl_path, exc)
        return annotations

    merged_count = 0
    for fname, info in ring_labels.items():
        stem = Path(fname).stem
        if stem in annotations:
            ring_present = bool(info.get("ring_present", False))
            # Only update if annotation doesn't already have ring info
            if not annotations[stem].get("has_suction_ring", False):
                annotations[stem]["has_suction_ring"] = ring_present
                if ring_present:
                    annotations[stem]["ring_visibility"] = info.get(
                        "ring_visibility", "full"
                    )
                    merged_count += 1

    if merged_count > 0:
        logger.info(
            "Merged ring labels: %d images marked as ring-present from %s",
            merged_count, rl_path,
        )

    return annotations


def _parse_project_format(
    raw: Dict[str, Any], logger
) -> Dict[str, Dict[str, Any]]:
    """Parse the project's native annotation format."""
    annotations = {}

    for filename, entry in raw.items():
        img_id = Path(filename).stem  # "eye_01"
        structs = entry.get("annotations", {})

        ann: Dict[str, Any] = {
            "image_id": img_id,
            "image_filename": filename,
            "image_path": str(
                Path(entry.get("image_path", "")).as_posix()
            ),
            "image_width": int(entry.get("image_width", 0)),
            "image_height": int(entry.get("image_height", 0)),
        }

        # ── PUPIL ─────────────────────────────────────────────
        pupil = structs.get("PUPIL", {})
        if pupil and "center_x" in pupil:
            ann["pupil_center"] = [
                float(pupil["center_x"]),
                float(pupil["center_y"]),
            ]
            ann["pupil_axes"] = [
                float(pupil["semi_major"]),
                float(pupil["semi_minor"]),
            ]
            ann["pupil_radius"] = (
                float(pupil["semi_major"]) + float(pupil["semi_minor"])
            ) / 2.0
            ann["pupil_angle"] = float(pupil.get("angle_deg", 0))
            ann["pupil_boundary"] = [
                [float(p[0]), float(p[1])]
                for p in pupil.get("boundary_points", [])
            ]
        else:
            logger.warning(
                "Image '%s' has no PUPIL annotation", img_id
            )

        # ── LIMBUS ────────────────────────────────────────────
        limbus = structs.get("LIMBUS", {})
        if limbus and "center_x" in limbus:
            ann["limbus_center"] = [
                float(limbus["center_x"]),
                float(limbus["center_y"]),
            ]
            ann["limbus_axes"] = [
                float(limbus["semi_major"]),
                float(limbus["semi_minor"]),
            ]
            ann["limbus_radius"] = (
                float(limbus["semi_major"])
                + float(limbus["semi_minor"])
            ) / 2.0
            ann["limbus_angle"] = float(limbus.get("angle_deg", 0))
            ann["limbus_boundary"] = [
                [float(p[0]), float(p[1])]
                for p in limbus.get("boundary_points", [])
            ]
        else:
            logger.warning(
                "Image '%s' has no LIMBUS annotation", img_id
            )

        # ── RING ──────────────────────────────────────────────
        ring = structs.get("RING", {})
        if ring and "center_x" in ring:
            ann["has_suction_ring"] = True
            ann["ring_center"] = [
                float(ring["center_x"]),
                float(ring["center_y"]),
            ]
            ann["ring_axes"] = [
                float(ring["semi_major"]),
                float(ring["semi_minor"]),
            ]
            ann["ring_angle"] = float(ring.get("angle_deg", 0))
            ann["ring_boundary"] = [
                [float(p[0]), float(p[1])]
                for p in ring.get("boundary_points", [])
            ]
            # Compute inner radius if available
            ring_inner = ring.get("inner_semi_major")
            ring_inner_minor = ring.get("inner_semi_minor")
            if ring_inner is not None and ring_inner_minor is not None:
                ann["ring_inner_axes"] = [
                    float(ring_inner),
                    float(ring_inner_minor),
                ]
            else:
                # Estimate inner opening as 80% of outer
                ann["ring_inner_axes"] = [
                    float(ring["semi_major"]) * 0.80,
                    float(ring["semi_minor"]) * 0.80,
                ]
        else:
            ann["has_suction_ring"] = False

        annotations[img_id] = ann

    return annotations


def _parse_ellipse_dict(
    d: Dict[str, Any],
) -> Tuple[Optional[List[float]], Optional[List[float]], float]:
    """Extract (center, axes, angle) from an ellipse sub-dict.

    Supports two annotation styles:

    *Style A* — ``cx``/``cy``/``semi_major``/``semi_minor``/``angle_deg``
    (written by ``annotate_live_video.py`` in this project)::

        {"cx": 1525.31, "cy": 568.91,
         "semi_major": 59.29, "semi_minor": 46.77, "angle_deg": 76.13}

    *Style B* — ``center``/``radius``|``axes``/``angle``
    (legacy / external tools)::

        {"center": [x, y], "axes": [a, b], "angle": 0.0}

    Returns
    -------
    center : [x, y] or None
    axes   : [semi_major, semi_minor] or None
    angle  : float (degrees)
    """
    if not d:
        return None, None, 0.0

    # ── Style A: cx / cy / semi_major / semi_minor / angle_deg ──
    if "cx" in d and "cy" in d:
        center = [float(d["cx"]), float(d["cy"])]
        sm = d.get("semi_major")
        sn = d.get("semi_minor")
        axes = [float(sm), float(sn)] if sm is not None and sn is not None else None
        angle = float(d.get("angle_deg", d.get("angle", 0)))
        return center, axes, angle

    # ── Style B: center list / radius or axes ───────────────────
    center_raw = d.get("center")
    center = [float(center_raw[0]), float(center_raw[1])] if center_raw is not None else None

    axes_raw = d.get("axes") or d.get("radius")
    if axes_raw is None:
        axes = None
    elif isinstance(axes_raw, (int, float)):
        axes = [float(axes_raw), float(axes_raw)]
    else:
        axes = [float(axes_raw[0]), float(axes_raw[1])]

    angle = float(d.get("angle", d.get("angle_deg", 0)))
    return center, axes, angle


def _parse_flat_format(
    raw: Dict[str, Any], logger
) -> Dict[str, Dict[str, Any]]:
    """Parse a flat {image_id: {pupil_center, ...}} format.

    Handles two ellipse sub-formats:
    - cx/cy/semi_major/semi_minor/angle_deg  (annotate_live_video.py output)
    - center/axes/angle  (legacy)
    """
    annotations = {}
    for key, entry in raw.items():
        img_id = Path(key).stem
        ann: Dict[str, Any] = {"image_id": img_id}

        # ── PUPIL ─────────────────────────────────────────────────
        pupil_dict = entry.get("pupil", {})
        pc_raw = entry.get("pupil_center")
        pr_raw = entry.get("pupil_radius") or entry.get("pupil_axes")

        if pc_raw is not None:
            # Legacy top-level keys
            ann["pupil_center"] = [float(pc_raw[0]), float(pc_raw[1])]
            if pr_raw is not None:
                if isinstance(pr_raw, (int, float)):
                    ann["pupil_radius"] = float(pr_raw)
                    ann["pupil_axes"] = [float(pr_raw), float(pr_raw)]
                else:
                    ann["pupil_axes"] = [float(pr_raw[0]), float(pr_raw[1])]
                    ann["pupil_radius"] = (float(pr_raw[0]) + float(pr_raw[1])) / 2.0
            ann["pupil_angle"] = float(entry.get("pupil_angle", 0))
        elif pupil_dict:
            # Sub-dict style (both Style A and B handled by helper)
            p_center, p_axes, p_angle = _parse_ellipse_dict(pupil_dict)
            if p_center is not None:
                ann["pupil_center"] = p_center
            if p_axes is not None:
                ann["pupil_axes"] = p_axes
                ann["pupil_radius"] = (p_axes[0] + p_axes[1]) / 2.0
            ann["pupil_angle"] = p_angle
        else:
            ann["pupil_angle"] = 0.0

        # ── LIMBUS ────────────────────────────────────────────────
        limbus_dict = entry.get("limbus", {})
        lc_raw = entry.get("limbus_center")
        lr_raw = entry.get("limbus_radius") or entry.get("limbus_axes")

        if lc_raw is not None:
            # Legacy top-level keys
            ann["limbus_center"] = [float(lc_raw[0]), float(lc_raw[1])]
            if lr_raw is not None:
                if isinstance(lr_raw, (int, float)):
                    ann["limbus_radius"] = float(lr_raw)
                    ann["limbus_axes"] = [float(lr_raw), float(lr_raw)]
                else:
                    ann["limbus_axes"] = [float(lr_raw[0]), float(lr_raw[1])]
                    ann["limbus_radius"] = (float(lr_raw[0]) + float(lr_raw[1])) / 2.0
            ann["limbus_angle"] = float(entry.get("limbus_angle", 0))
        elif limbus_dict:
            l_center, l_axes, l_angle = _parse_ellipse_dict(limbus_dict)
            if l_center is not None:
                ann["limbus_center"] = l_center
            if l_axes is not None:
                ann["limbus_axes"] = l_axes
                ann["limbus_radius"] = (l_axes[0] + l_axes[1]) / 2.0
            ann["limbus_angle"] = l_angle
        else:
            ann["limbus_angle"] = 0.0

        ann["has_suction_ring"] = bool(entry.get("has_suction_ring", False))

        # ── RING (flat format) ────────────────────────────────────
        ring_data = entry.get("ring", {})
        if ring_data:
            r_center, r_axes, r_angle = _parse_ellipse_dict(ring_data)
            if r_center is not None:
                ann["has_suction_ring"] = True
                ann["ring_center"] = r_center
                if r_axes is not None:
                    ann["ring_axes"] = r_axes
                ann["ring_angle"] = r_angle

        annotations[img_id] = ann

    return annotations


def _parse_list_format(
    entries: list, logger
) -> Dict[str, Dict[str, Any]]:
    """Parse a list-of-dicts format."""
    annotations = {}
    for entry in entries:
        img_id = (
            entry.get("image_id")
            or entry.get("image")
            or entry.get("filename")
            or entry.get("file")
        )
        if img_id is None:
            continue
        img_id = Path(img_id).stem
        entry["image_id"] = img_id
        # Ensure ring flag exists
        if "has_suction_ring" not in entry:
            entry["has_suction_ring"] = bool(
                entry.get("ring_center") or entry.get("ring", {}).get("center")
            )
        annotations[img_id] = entry
    return annotations


# ══════════════════════════════════════════════════════════════════════
# Mask generation from annotations
# ══════════════════════════════════════════════════════════════════════


def _order_boundary_points(points: np.ndarray) -> np.ndarray:
    """Order boundary points by angle around their centroid.

    Prevents self-intersecting polygons when some annotation points
    backtrack or are out of order.
    """
    if len(points) < 3:
        return points
    cx = float(np.mean(points[:, 0]))
    cy = float(np.mean(points[:, 1]))
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    order = np.argsort(angles)
    return points[order]


def generate_mask_from_annotation(
    image_shape: Tuple[int, int],
    annotation: Dict[str, Any],
    num_classes: int = 3,
    use_boundary_points: bool = False,
    min_boundary_points: int = 20,
) -> np.ndarray:
    """Create a multi-class segmentation mask from an annotation dict.

    Drawing order (later overwrites earlier)::

        3-class mode:
            1. Background    (class 0) — full image
            2. Iris / Limbus (class 2) — iris region bounded by limbus
            3. Pupil         (class 1) — overwrites iris within pupil

        4-class mode (ring-aware):
            1. Background    (class 0) — full image
            2. Suction ring  (class 3) — ring annulus region
            3. Iris / Limbus (class 2) — iris region (inside ring if present)
            4. Pupil         (class 1) — overwrites iris within pupil

    The ring is drawn as an annulus: the area between the outer ring
    boundary and the inner ring opening.  The iris and pupil are drawn
    inside the ring opening, so they correctly overwrite the inner
    portion.

    Parameters
    ----------
    image_shape : (height, width)
    annotation : normalised annotation dict
    num_classes : int
        3 = background/pupil/iris, 4 = adds suction_ring.
    use_boundary_points : bool
        If True, prefer boundary-point polygons over ellipses when
        enough points are available.
    min_boundary_points : int
        Minimum boundary points required to use polygon fill.

    Returns
    -------
    np.ndarray  shape (H, W)  dtype uint8  values in {0, 1, 2} or {0, 1, 2, 3}
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # ── SUCTION RING (class 3) — drawn first (4-class only) ────
    if num_classes >= 4 and annotation.get("has_suction_ring", False):
        _draw_ring_annulus(
            mask,
            annotation,
            class_id=3,
            use_boundary=use_boundary_points,
            min_pts=min_boundary_points,
        )

    # ── LIMBUS / IRIS (class 2) — drawn second ────────────────
    _draw_structure(
        mask,
        annotation,
        prefix="limbus",
        class_id=2,
        use_boundary=use_boundary_points,
        min_pts=min_boundary_points,
    )

    # ── PUPIL (class 1) — drawn on top ─────────────────────────
    _draw_structure(
        mask,
        annotation,
        prefix="pupil",
        class_id=1,
        use_boundary=use_boundary_points,
        min_pts=min_boundary_points,
    )

    return mask


def _draw_structure(
    mask: np.ndarray,
    annotation: Dict[str, Any],
    prefix: str,
    class_id: int,
    use_boundary: bool = False,
    min_pts: int = 20,
) -> None:
    """Draw one anatomical structure (pupil or limbus) onto a mask.

    Modifies *mask* in-place.
    """
    center = annotation.get(f"{prefix}_center")
    axes = annotation.get(f"{prefix}_axes")
    radius = annotation.get(f"{prefix}_radius")
    angle = annotation.get(f"{prefix}_angle", 0.0)
    boundary = annotation.get(f"{prefix}_boundary", [])

    if center is None:
        return  # no annotation for this structure

    # decide: boundary points or ellipse?
    if (
        use_boundary
        and len(boundary) >= min_pts
    ):
        # polygon fill from boundary points
        pts = np.array(boundary, dtype=np.float64).reshape(-1, 2)
        pts = _order_boundary_points(pts)  # prevent self-intersection
        pts_int = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts_int], int(class_id))
    else:
        # ellipse fill
        cx = int(round(float(center[0])))
        cy = int(round(float(center[1])))

        if axes is not None:
            ax_a = int(round(float(axes[0])))
            ax_b = int(round(float(axes[1])))
        elif radius is not None:
            ax_a = int(round(float(radius)))
            ax_b = ax_a
        else:
            return  # no size information

        ang = int(round(float(angle)))

        if ax_a <= 0 or ax_b <= 0:
            return

        cv2.ellipse(
            mask, (cx, cy), (ax_a, ax_b), ang, 0, 360,
            int(class_id), -1,
        )


def _draw_ring_annulus(
    mask: np.ndarray,
    annotation: Dict[str, Any],
    class_id: int = 3,
    use_boundary: bool = False,
    min_pts: int = 20,
) -> None:
    """Draw the suction ring as an annulus (class 3) onto the mask.

    The ring is the region between the outer ring boundary and the
    inner ring opening.  The inner opening is where the iris and
    pupil are visible.

    Strategy:
        1. Fill the entire outer ring ellipse with class_id.
        2. Cut out the inner opening by filling it with 0 (background).
           The iris and pupil will be drawn on top of this later.

    If boundary points are available, uses polygon fill for the outer
    boundary.  The inner opening is always an ellipse (estimated or
    annotated).

    Modifies *mask* in-place.
    """
    ring_center = annotation.get("ring_center")
    ring_axes = annotation.get("ring_axes")
    ring_angle = annotation.get("ring_angle", 0.0)
    ring_boundary = annotation.get("ring_boundary", [])
    ring_inner_axes = annotation.get("ring_inner_axes")

    if ring_center is None:
        return

    cx = int(round(float(ring_center[0])))
    cy = int(round(float(ring_center[1])))

    # ── Draw outer ring boundary ──────────────────────────────
    if use_boundary and len(ring_boundary) >= min_pts:
        pts = np.array(ring_boundary, dtype=np.float64).reshape(-1, 2)
        pts = _order_boundary_points(pts)
        pts_int = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts_int], int(class_id))
    elif ring_axes is not None:
        ax_a = int(round(float(ring_axes[0])))
        ax_b = int(round(float(ring_axes[1])))
        ang = int(round(float(ring_angle)))

        if ax_a <= 0 or ax_b <= 0:
            return

        cv2.ellipse(
            mask, (cx, cy), (ax_a, ax_b), ang, 0, 360,
            int(class_id), -1,
        )
    else:
        return  # no geometry

    # ── Cut out inner opening (set to 0 = background) ─────────
    if ring_inner_axes is not None:
        inner_a = int(round(float(ring_inner_axes[0])))
        inner_b = int(round(float(ring_inner_axes[1])))
    elif ring_axes is not None:
        # Default: inner opening is ~80% of outer
        inner_a = int(round(float(ring_axes[0]) * 0.80))
        inner_b = int(round(float(ring_axes[1]) * 0.80))
    else:
        return

    ang = int(round(float(ring_angle)))
    if inner_a > 0 and inner_b > 0:
        cv2.ellipse(
            mask, (cx, cy), (inner_a, inner_b), ang, 0, 360,
            0, -1,  # fill with background
        )


def get_annotation_stats(
    annotations: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute statistics about the annotation dataset.

    Returns a summary dict with counts, quality flags, etc.
    """
    stats = {
        "total_images": len(annotations),
        "with_pupil": 0,
        "with_limbus": 0,
        "with_ring": 0,
        "with_both": 0,
        "with_all_three": 0,
        "pupil_boundary_counts": [],
        "limbus_boundary_counts": [],
        "ring_boundary_counts": [],
        "pupil_radii": [],
        "limbus_radii": [],
        "ring_radii": [],
        "issues": [],
    }

    for img_id, ann in annotations.items():
        has_pupil = "pupil_center" in ann
        has_limbus = "limbus_center" in ann
        has_ring = ann.get("has_suction_ring", False)

        if has_pupil:
            stats["with_pupil"] += 1
            n_bp = len(ann.get("pupil_boundary", []))
            stats["pupil_boundary_counts"].append(n_bp)
            pr = ann.get("pupil_radius", 0)
            if pr > 0:
                stats["pupil_radii"].append(pr)
            if n_bp < 5:
                stats["issues"].append(
                    f"{img_id}: PUPIL has only {n_bp} boundary points"
                )

        if has_limbus:
            stats["with_limbus"] += 1
            n_bp = len(ann.get("limbus_boundary", []))
            stats["limbus_boundary_counts"].append(n_bp)
            lr = ann.get("limbus_radius", 0)
            if lr > 0:
                stats["limbus_radii"].append(lr)
            if n_bp < 5:
                stats["issues"].append(
                    f"{img_id}: LIMBUS has only {n_bp} boundary points"
                )

        if has_ring:
            stats["with_ring"] += 1
            n_bp = len(ann.get("ring_boundary", []))
            stats["ring_boundary_counts"].append(n_bp)
            ring_axes = ann.get("ring_axes")
            if ring_axes:
                ring_r = (float(ring_axes[0]) + float(ring_axes[1])) / 2.0
                stats["ring_radii"].append(ring_r)

        if has_pupil and has_limbus:
            stats["with_both"] += 1
        if has_pupil and has_limbus and has_ring:
            stats["with_all_three"] += 1

        # cross-validation
        if has_pupil and has_limbus:
            pr = ann.get("pupil_radius", 0)
            lr = ann.get("limbus_radius", 0)
            if lr > 0 and pr > 0:
                ratio = pr / lr
                if ratio < 0.10 or ratio > 0.80:
                    stats["issues"].append(
                        f"{img_id}: unusual pupil/limbus ratio "
                        f"{ratio:.3f}"
                    )
                pc = ann["pupil_center"]
                lc = ann["limbus_center"]
                offset = math.sqrt(
                    (pc[0] - lc[0]) ** 2 + (pc[1] - lc[1]) ** 2
                )
                offset_ratio = offset / lr
                if offset_ratio > 0.3:
                    stats["issues"].append(
                        f"{img_id}: large pupil-limbus offset "
                        f"({offset:.1f}px = {offset_ratio:.2f} of "
                        f"limbus radius)"
                    )

        # Ring-specific validation
        if has_ring and has_limbus:
            ring_axes = ann.get("ring_axes")
            if ring_axes:
                ring_r = (float(ring_axes[0]) + float(ring_axes[1])) / 2.0
                lr = ann.get("limbus_radius", 0)
                if lr > 0 and ring_r > 0:
                    if ring_r < lr:
                        stats["issues"].append(
                            f"{img_id}: ring radius ({ring_r:.1f}) < "
                            f"limbus radius ({lr:.1f}) — unusual"
                        )

    return stats


# ══════════════════════════════════════════════════════════════════════
# Class weight computation
# ══════════════════════════════════════════════════════════════════════


def compute_class_weights(
    annotations: Dict[str, Dict[str, Any]],
    image_ids: List[str],
    image_dir: str,
    num_classes: int = 3,
    use_boundary_points: bool = False,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from annotations.

    Generates masks for a sample of images and counts pixels per class
    to derive balanced loss weights.  Handles the case where the ring
    class (3) appears only in docked images.

    Parameters
    ----------
    annotations : dict
        Annotation dictionary.
    image_ids : list of str
        Image IDs to sample from.
    image_dir : str
        Image directory (for reading image dimensions).
    num_classes : int
        Number of segmentation classes (3 or 4).
    use_boundary_points : bool
        Whether to use boundary points for mask generation.

    Returns
    -------
    torch.Tensor
        Class weights of shape ``(num_classes,)``.
    """
    logger = get_logger()
    class_counts = np.zeros(num_classes, dtype=np.float64)
    image_dir_path = Path(image_dir)

    sample_ids = image_ids[:50]  # Sample up to 50 images for speed

    for img_id in sample_ids:
        ann = annotations.get(img_id, {})

        # Determine image shape
        img_w = ann.get("image_width", 0)
        img_h = ann.get("image_height", 0)

        if img_w == 0 or img_h == 0:
            # Try reading the image
            for ext in (".jpeg", ".jpg", ".png", ".bmp"):
                img_path = image_dir_path / f"{img_id}{ext}"
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_h, img_w = img.shape[:2]
                        break

        if img_w == 0 or img_h == 0:
            continue

        mask = generate_mask_from_annotation(
            (img_h, img_w), ann, num_classes,
            use_boundary_points=use_boundary_points,
        )

        for c in range(num_classes):
            class_counts[c] += np.count_nonzero(mask == c)

    # Compute inverse-frequency weights
    total = class_counts.sum()
    if total == 0:
        logger.warning("No pixels counted — returning uniform weights")
        return torch.ones(num_classes, dtype=torch.float32)

    weights = total / (num_classes * class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    weights = np.clip(weights, 0.1, 10.0)

    class_names = {
        0: "background", 1: "pupil", 2: "iris", 3: "suction_ring",
    }
    for c in range(num_classes):
        name = class_names.get(c, f"class_{c}")
        pct = class_counts[c] / total * 100 if total > 0 else 0
        logger.info(
            "  Class %d (%s): %d pixels (%.1f%%), weight=%.3f",
            c, name, int(class_counts[c]), pct, weights[c],
        )

    return torch.tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════
# Augmentation pipelines — compatible with albumentations >= 2.0
# ══════════════════════════════════════════════════════════════════════


def _get_train_augmentation(
    input_size: int = 512,
    # ═══════════════════════════════════════════════════════════════════
    # CHANGE 2 of 4 — New parameter for grayscale augmentation
    # ═══════════════════════════════════════════════════════════════════
    grayscale_prob: float = 0.0,
    # ═══════════════════════════════════════════════════════════════════
    # END CHANGE 2 (signature)
    # ═══════════════════════════════════════════════════════════════════
) -> Any:
    """Heavy augmentation including video-specific transforms.

    Compatible with albumentations 2.x API.

    This pipeline ensures the model sees:
    - Motion blur           (video frames)
    - JPEG compression      (video codec artefacts)
    - Resolution jitter     (different camera zoom levels)
    - Gaussian noise        (sensor noise)
    - Brightness/contrast   (varying illumination)
    - Partial occlusion     (eyelid, reflections)
    - Geometric variation   (slight rotation, scale, perspective)
    - Grayscale conversion  (when grayscale_prob > 0)

    Parameters
    ----------
    input_size : int
        Target spatial resolution (square).
    grayscale_prob : float
        Probability of converting each training image to enhanced
        grayscale (3-channel replicated).  Default ``0.0`` preserves
        the original behaviour.  Set to ``0.3`` for grayscale
        robustness training.
    """
    if not _HAS_ALB:
        raise ImportError(
            "albumentations is required for training.  "
            "pip install albumentations"
        )

    transforms_list = [
        # ── spatial ──────────────────────────────────────────
        A.Resize(input_size, input_size),
        A.HorizontalFlip(p=0.5),
    ]

    # Affine (replaces deprecated ShiftScaleRotate)
    try:
        transforms_list.append(
            A.Affine(
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                scale=(0.85, 1.15),
                rotate=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.6,
            )
        )
    except TypeError:
        # fallback for older albumentations
        try:
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.15,
                    rotate_limit=25,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.6,
                )
            )
        except Exception:
            pass

    transforms_list.append(
        A.Perspective(scale=(0.02, 0.06), p=0.3),
    )

    try:
        transforms_list.append(
            A.ElasticTransform(
                alpha=40,
                sigma=40 * 0.05,
                p=0.2,
            )
        )
    except TypeError:
        transforms_list.append(
            A.ElasticTransform(p=0.2)
        )

    # ── colour / exposure ────────────────────────────────────
    transforms_list.append(
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,
                    contrast_limit=0.25,
                    p=1.0,
                ),
                A.CLAHE(clip_limit=3.0, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            ],
            p=0.7,
        ),
    )
    transforms_list.append(
        A.HueSaturationValue(
            hue_shift_limit=12,
            sat_shift_limit=20,
            val_shift_limit=18,
            p=0.4,
        ),
    )

    # ── noise (sensor) ───────────────────────────────────────
    noise_transforms = []

    # GaussNoise — API changed in v2.0
    try:
        # albumentations >= 2.0: uses std_range
        noise_transforms.append(
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0)
        )
    except TypeError:
        try:
            # albumentations < 2.0: uses var_limit
            noise_transforms.append(
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0)
            )
        except TypeError:
            noise_transforms.append(A.GaussNoise(p=1.0))

    try:
        noise_transforms.append(
            A.ISONoise(
                color_shift=(0.01, 0.04),
                intensity=(0.05, 0.2),
                p=1.0,
            )
        )
    except TypeError:
        noise_transforms.append(A.ISONoise(p=1.0))

    try:
        noise_transforms.append(
            A.MultiplicativeNoise(
                multiplier=(0.90, 1.10), p=1.0
            )
        )
    except (TypeError, AttributeError):
        pass  # skip if not available

    if noise_transforms:
        transforms_list.append(A.OneOf(noise_transforms, p=0.5))

    # ── blur (focus + motion) ────────────────────────────────
    transforms_list.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=0.4,
        ),
    )

    # ── video codec artefacts ────────────────────────────────
    try:
        # albumentations >= 2.0
        transforms_list.append(
            A.ImageCompression(
                quality_range=(55, 95),
                p=0.35,
            )
        )
    except TypeError:
        try:
            # albumentations < 2.0
            transforms_list.append(
                A.ImageCompression(
                    quality_lower=55,
                    quality_upper=95,
                    p=0.35,
                )
            )
        except TypeError:
            pass  # skip if neither API works

    # ── partial occlusion (eyelid / reflections) ────────────
    try:
        # albumentations v2.x API
        transforms_list.append(
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(input_size * 0.04), int(input_size * 0.08)),
                hole_width_range=(int(input_size * 0.04), int(input_size * 0.08)),
                fill=0,
                p=0.2,
            )
        )
    except TypeError:
        try:
            # albumentations v1.x API
            transforms_list.append(
                A.CoarseDropout(
                    max_holes=3,
                    max_height=int(input_size * 0.08),
                    max_width=int(input_size * 0.08),
                    min_holes=1,
                    fill_value=0,
                    p=0.2,
                )
            )
        except TypeError:
            pass

    # ── downscale then upscale (resolution jitter) ──────────
    try:
        # albumentations v2.x API
        transforms_list.append(
            A.Downscale(
                scale_range=(0.5, 0.85),
                p=0.25,
            )
        )
    except TypeError:
        try:
            # albumentations v1.x API
            transforms_list.append(
                A.Downscale(
                    scale_min=0.5,
                    scale_max=0.85,
                    p=0.25,
                )
            )
        except TypeError:
            pass  # skip if not available

    # ═══════════════════════════════════════════════════════════════════
    # CHANGE 2 of 4 (continued) — Insert grayscale augmentation
    #
    # Placed AFTER all pixel-level augmentations (brightness, blur,
    # noise, compression, occlusion, downscale) but BEFORE Normalize
    # and ToTensorV2.  This ordering ensures:
    #   - Pixel augmentations operate on the richer colour image
    #   - Grayscale conversion produces the exact same enhanced
    #     3-channel output that the inference pipeline produces
    #   - Normalize sees the final pixel values
    #
    # When grayscale_prob == 0.0 (default), this block is skipped
    # entirely and the pipeline is identical to the original.
    # ═══════════════════════════════════════════════════════════════════
    if grayscale_prob > 0.0 and _HAS_GRAYSCALE_AUG:
        transforms_list.append(
            RandomGrayscaleConversion(
                enhance=True,
                clahe_clip_limit=3.0,
                clahe_grid_size=(8, 8),
                p=grayscale_prob,
            )
        )
        get_logger().info(
            "Grayscale augmentation enabled in training pipeline "
            "(p=%.2f)",
            grayscale_prob,
        )
    elif grayscale_prob > 0.0 and not _HAS_GRAYSCALE_AUG:
        get_logger().warning(
            "Grayscale augmentation requested (p=%.2f) but "
            "grayscale_augmentation module not available — skipping. "
            "Ensure pupil_tracking.ml.grayscale_augmentation is "
            "importable.",
            grayscale_prob,
        )
    # ═══════════════════════════════════════════════════════════════════
    # END CHANGE 2
    # ═══════════════════════════════════════════════════════════════════

    # ── normalise ────────────────────────────────────────────
    transforms_list.append(
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    )
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def _get_val_augmentation(input_size: int = 512) -> Any:
    """Deterministic resize + normalise — no randomness."""
    if not _HAS_ALB:
        raise ImportError("albumentations is required.")

    return A.Compose(
        [
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def _get_inference_transform(input_size: int = 512) -> Any:
    """Inference-time preprocessing (same as val)."""
    return _get_val_augmentation(input_size)


# ═══════════════════════════════════════════════════════════════════════
# CHANGE 3 of 4 — Grayscale-forced validation pipeline
#
# Used by the fine-tuning script (scripts/finetune_grayscale.py) to
# measure model accuracy on grayscale inputs SEPARATELY from RGB
# accuracy.  This ensures the fine-tuned model passes BOTH thresholds
# before a checkpoint is saved.
#
# When not called, this function has zero impact on the existing code.
# ═══════════════════════════════════════════════════════════════════════


def _get_grayscale_val_augmentation(input_size: int = 512) -> Any:
    """Validation pipeline that forces every image to enhanced grayscale.

    Every image is converted to grayscale, enhanced with CLAHE, and
    replicated to 3 channels — matching the inference-time grayscale
    pipeline exactly.  This lets the fine-tuning script compute Dice
    on grayscale inputs independently from RGB Dice.

    Parameters
    ----------
    input_size : int
        Target spatial resolution (square).

    Returns
    -------
    albumentations.Compose
        Pipeline: resize → grayscale(always) → normalize → tensor.

    Raises
    ------
    ImportError
        If albumentations or the grayscale augmentation module is
        not available.
    """
    if not _HAS_ALB:
        raise ImportError("albumentations is required.")
    if not _HAS_GRAYSCALE_AUG:
        raise ImportError(
            "pupil_tracking.ml.grayscale_augmentation is required "
            "for grayscale validation pipeline."
        )

    return A.Compose(
        [
            A.Resize(input_size, input_size),
            RandomGrayscaleConversion(
                enhance=True,
                clahe_clip_limit=3.0,
                clahe_grid_size=(8, 8),
                always_apply=True,
                p=1.0,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


# ═══════════════════════════════════════════════════════════════════════
# END CHANGE 3
# ═══════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════
# Image-level split
# ══════════════════════════════════════════════════════════════════════


def split_by_images(
    image_ids: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Split image IDs into train / val at the IMAGE level.

    Every augmented copy of one image ends up in the SAME split.
    This prevents the data-leakage bug in the original code.
    """
    ids = sorted(set(image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_ratio))
    n_val = min(n_val, len(ids) - 1)  # keep ≥ 1 for training

    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    logger = get_logger()
    logger.info(
        "Image-level split: %d train, %d val (ratio=%.2f, seed=%d)",
        len(train_ids), len(val_ids), val_ratio, seed,
    )
    for vid in val_ids:
        logger.info("  VAL image: %s", vid)
    for tid in train_ids:
        logger.info("  TRAIN image: %s", tid)

    return train_ids, val_ids


# ══════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════


class EyeSegmentationDataset(Dataset):
    """Multi-class eye segmentation dataset with image-level integrity.

    Each ``__getitem__`` call returns one augmented (image, mask) pair.
    The *same* source image is augmented ``copies_per_image`` times,
    but all copies share the split assignment of their source image.

    Supports both 3-class (legacy) and 4-class (ring-aware) masks.
    When ``num_classes=4``, images with ring annotations will have
    class-3 pixels in their masks; images without rings will have
    only classes 0–2 (no class 3 pixels), which is handled correctly
    by the weighted loss function.

    Parameters
    ----------
    image_ids : list of str
        Image identifiers (stem names, e.g. ``"eye_01"``).
    annotations : dict
        ``{image_id: annotation_dict}`` as returned by ``load_annotations``.
    image_dir : str | Path
        Directory containing source images.
    mask_dir : str | Path | None
        Directory with pre-generated mask PNGs.  If a mask for an
        image is not found here, one is generated from the annotation.
    transform
        albumentations pipeline.
    copies_per_image : int
        How many augmented versions per source image per epoch.
    input_size : int
        Target spatial resolution.
    num_classes : int
        Number of segmentation classes (3 = legacy, 4 = ring-aware).
    use_boundary_points : bool
        If True, use boundary-point polygons for mask generation
        when enough points are available.
    """

    _EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".tif")

    def __init__(
        self,
        image_ids: List[str],
        annotations: Dict[str, Dict[str, Any]],
        image_dir: str | Path,
        mask_dir: Optional[str | Path] = None,
        transform=None,
        copies_per_image: int = 50,
        input_size: int = 512,
        num_classes: int = 3,
        use_boundary_points: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.copies_per_image = copies_per_image
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_boundary_points = use_boundary_points

        # keep annotations available for _find_image
        self.annotations = annotations

        # resolve actual image files
        self.samples: List[Tuple[str, Path]] = []
        for img_id in image_ids:
            img_path = self._find_image(img_id)
            if img_path is None:
                get_logger().warning(
                    "Image not found for id '%s' in %s — skipping",
                    img_id, self.image_dir,
                )
                continue
            self.samples.append((img_id, img_path))

        # pre-load and cache images + masks for speed
        # (13 images easily fit in memory)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for img_id, img_path in self.samples:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                get_logger().error(
                    "Failed to read image: %s", img_path
                )
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._load_or_generate_mask(img_id, image.shape)
            self._cache[img_id] = (image, mask)

        # Count ring vs no-ring images
        ring_count = sum(
            1 for img_id, _ in self.samples
            if annotations.get(img_id, {}).get("has_suction_ring", False)
        )

        get_logger().info(
            "Dataset: %d images x %d copies = %d samples "
            "(cached %d, %d with ring, %d classes)",
            len(self.samples), copies_per_image, len(self),
            len(self._cache), ring_count, num_classes,
        )

    def __len__(self) -> int:
        return len(self.samples) * self.copies_per_image

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = idx // self.copies_per_image
        img_id, _ = self.samples[img_idx]

        if img_id in self._cache:
            image, mask = self._cache[img_id]
        else:
            img_path = self.samples[img_idx][1]
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"Failed to read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = self._load_or_generate_mask(img_id, image.shape)

        # make copies to avoid mutating cache
        image = image.copy()
        mask = mask.copy()

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image_t = transformed["image"]       # Tensor [3, H, W]
            mask_t = transformed["mask"].long()   # Tensor [H, W]
        else:
            # fallback: resize + to tensor
            image = cv2.resize(
                image, (self.input_size, self.input_size)
            )
            mask = cv2.resize(
                mask, (self.input_size, self.input_size),
                interpolation=cv2.INTER_NEAREST,
            )
            image_t = torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            mask_t = torch.from_numpy(mask).long()

        return image_t, mask_t

    # ── mask loading / generation ───────────────────────────────

    def _load_or_generate_mask(
        self, img_id: str, image_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Try to load a pre-generated mask; fall back to generation."""
        if self.mask_dir is not None:
            # composite multi-class mask
            for ext in (".png", ".npy"):
                mask_path = self.mask_dir / f"{img_id}{ext}"
                if mask_path.exists():
                    if ext == ".npy":
                        mask = np.load(str(mask_path))
                    else:
                        mask = cv2.imread(
                            str(mask_path), cv2.IMREAD_GRAYSCALE
                        )
                    if mask is not None:
                        mask = mask.astype(np.uint8)
                        mask[mask >= self.num_classes] = 0
                        return mask

            # separate per-class masks
            pupil_path = self.mask_dir / f"{img_id}_pupil.png"
            iris_path = self.mask_dir / f"{img_id}_iris.png"
            ring_path = self.mask_dir / f"{img_id}_ring.png"

            if pupil_path.exists():
                merged = np.zeros(
                    image_shape[:2], dtype=np.uint8
                )

                # Ring (class 3) — drawn first
                if self.num_classes >= 4 and ring_path.exists():
                    ring_m = cv2.imread(
                        str(ring_path), cv2.IMREAD_GRAYSCALE
                    )
                    if ring_m is not None:
                        merged[ring_m > 127] = 3

                # Iris (class 2) — drawn second
                if iris_path.exists():
                    iris_m = cv2.imread(
                        str(iris_path), cv2.IMREAD_GRAYSCALE
                    )
                    if iris_m is not None:
                        merged[iris_m > 127] = 2

                # Pupil (class 1) — drawn last (on top)
                pupil_m = cv2.imread(
                    str(pupil_path), cv2.IMREAD_GRAYSCALE
                )
                if pupil_m is not None:
                    merged[pupil_m > 127] = 1

                return merged

        # generate from annotation
        ann = self.annotations.get(img_id, {})
        return generate_mask_from_annotation(
            image_shape[:2], ann, self.num_classes,
            use_boundary_points=self.use_boundary_points,
        )

    # ── file-finding ────────────────────────────────────────────

    def _find_image(self, img_id: str) -> Optional[Path]:
        """Find the image file for a given ID.

        Searches: exact match, stem+extension, image_path from
        annotation.
        """
        # try annotation image_path first
        ann = self.annotations if hasattr(self, 'annotations') else {}
        if img_id in ann:
            img_path_str = ann[img_id].get("image_path", "")
            if img_path_str:
                # try relative to image_dir
                p = self.image_dir / Path(img_path_str).name
                if p.exists():
                    return p
                # try as-is (absolute or relative to cwd)
                p = Path(img_path_str)
                if p.exists():
                    return p

        # try stem + extensions
        for ext in self._EXTENSIONS:
            p = self.image_dir / f"{img_id}{ext}"
            if p.exists():
                return p

        # try with image_filename
        if img_id in ann:
            fname = ann[img_id].get("image_filename", "")
            if fname:
                p = self.image_dir / fname
                if p.exists():
                    return p

        return None

    # ── dataset statistics ──────────────────────────────────────

    def get_class_distribution(self) -> Dict[int, int]:
        """Count pixels per class across all cached masks.

        Returns
        -------
        dict
            ``{class_id: pixel_count}``.
        """
        counts = {c: 0 for c in range(self.num_classes)}
        for img_id, (_, mask) in self._cache.items():
            for c in range(self.num_classes):
                counts[c] += int(np.count_nonzero(mask == c))
        return counts

    def get_ring_image_count(self) -> Tuple[int, int]:
        """Count images with and without ring annotations.

        Returns
        -------
        (ring_count, no_ring_count)
        """
        ring = 0
        no_ring = 0
        for img_id, _ in self.samples:
            ann = self.annotations.get(img_id, {})
            if ann.get("has_suction_ring", False):
                ring += 1
            else:
                no_ring += 1
        return ring, no_ring


# ══════════════════════════════════════════════════════════════════════
# Convenience: build train + val datasets in one call
# ══════════════════════════════════════════════════════════════════════


def build_datasets(
    annotation_path: str,
    image_dir: str,
    mask_dir: Optional[str] = None,
    val_ratio: float = 0.2,
    copies_per_image: int = 50,
    input_size: int = 512,
    num_classes: int = 3,
    seed: int = 42,
    use_boundary_points: bool = False,
    ring_labels_path: Optional[str] = None,
    # ═══════════════════════════════════════════════════════════════════
    # CHANGE 4 of 4 — New parameters for grayscale augmentation
    # ═══════════════════════════════════════════════════════════════════
    enable_grayscale_aug: bool = False,
    grayscale_prob: float = 0.3,
    # ═══════════════════════════════════════════════════════════════════
    # END CHANGE 4 (signature)
    # ═══════════════════════════════════════════════════════════════════
) -> Tuple[EyeSegmentationDataset, EyeSegmentationDataset]:
    """Build train + val datasets with proper image-level split.

    Parameters
    ----------
    annotation_path : str
        Path to the main annotations JSON file.
    image_dir : str
        Directory containing source images.
    mask_dir : str or None
        Directory with pre-generated mask PNGs.
    val_ratio : float
        Fraction of images for validation.
    copies_per_image : int
        Augmented copies per source image per epoch.
    input_size : int
        Target spatial resolution.
    num_classes : int
        3 (legacy) or 4 (ring-aware).
    seed : int
        Random seed for reproducible splits.
    use_boundary_points : bool
        Use boundary-point polygons for mask generation.
    ring_labels_path : str or None
        Optional path to ring_labels.json for supplementing
        ring-presence flags.
    enable_grayscale_aug : bool
        If ``True``, add random grayscale conversion to the training
        pipeline.  Default ``False`` preserves the original behaviour
        exactly.  Set to ``True`` when fine-tuning for grayscale
        robustness.
    grayscale_prob : float
        Probability of converting each training image to grayscale
        when ``enable_grayscale_aug`` is ``True``.  Default ``0.3``
        (30 %).  Ignored when ``enable_grayscale_aug`` is ``False``.

    Returns
    -------
    (train_dataset, val_dataset)
    """
    image_ids, annotations = load_annotations(
        annotation_path,
        ring_labels_path=ring_labels_path,
    )

    # print stats
    stats = get_annotation_stats(annotations)
    logger = get_logger()
    logger.info("Annotation stats:")
    logger.info(
        "  Total: %d, with pupil: %d, with limbus: %d, "
        "with both: %d, with ring: %d",
        stats["total_images"], stats["with_pupil"],
        stats["with_limbus"], stats["with_both"],
        stats["with_ring"],
    )

    if num_classes >= 4:
        logger.info(
            "  4-class mode: %d images have ring annotations",
            stats["with_ring"],
        )
        if stats["with_ring"] == 0:
            logger.warning(
                "  ⚠ No ring annotations found! 4-class training will "
                "have no class-3 supervision.  Consider using "
                "--num-classes 3 or adding ring annotations."
            )

    if stats["issues"]:
        for issue in stats["issues"]:
            logger.warning("  WARNING: %s", issue)

    if len(image_ids) < 2:
        raise ValueError(
            f"Need >= 2 annotated images for train/val split, got "
            f"{len(image_ids)}"
        )

    train_ids, val_ids = split_by_images(
        image_ids, val_ratio=val_ratio, seed=seed
    )

    # ═══════════════════════════════════════════════════════════════════
    # CHANGE 4 of 4 (continued) — Pass grayscale_prob to train pipeline
    #
    # When enable_grayscale_aug is False (default), grayscale_prob is
    # forced to 0.0 so the pipeline is identical to the original.
    # ═══════════════════════════════════════════════════════════════════
    effective_grayscale_prob = grayscale_prob if enable_grayscale_aug else 0.0
    train_transform = _get_train_augmentation(
        input_size,
        grayscale_prob=effective_grayscale_prob,
    )
    # ═══════════════════════════════════════════════════════════════════
    # END CHANGE 4
    # ═══════════════════════════════════════════════════════════════════

    val_transform = _get_val_augmentation(input_size)

    train_ds = EyeSegmentationDataset(
        image_ids=train_ids,
        annotations=annotations,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        copies_per_image=copies_per_image,
        input_size=input_size,
        num_classes=num_classes,
        use_boundary_points=use_boundary_points,
    )

    val_ds = EyeSegmentationDataset(
        image_ids=val_ids,
        annotations=annotations,
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=val_transform,
        copies_per_image=max(5, copies_per_image // 5),
        input_size=input_size,
        num_classes=num_classes,
        use_boundary_points=use_boundary_points,
    )

    return train_ds, val_ds