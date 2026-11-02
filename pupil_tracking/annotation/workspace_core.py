"""Testable data and geometry primitives for the Pentacam annotation workspace.

All geometry uses original-image pixel coordinates. Display orientation is a
view-only transform and never changes the saved annotation coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1"})


class AnnotationValidationError(ValueError):
    """Raised when an annotation cannot safely be stored or exported."""


def discover_images(dataset_dir: Path | str) -> list[Path]:
    """Return supported raster images below *dataset_dir*, in stable order."""
    root = Path(dataset_dir).expanduser()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    excluded_parts = {".git", ".pentacam_annotations", "__pycache__", "venv", ".venv"}
    images: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images, key=lambda item: item.relative_to(root).as_posix().casefold())


def stable_image_id(relative_path: str | Path) -> str:
    """Create a stable, non-reversible identifier from a safe relative path."""
    from pathlib import PurePosixPath

    normalized = str(relative_path).replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or not relative.parts or any(part == ".." for part in relative.parts):
        raise ValueError("relative_path must be a non-empty path without traversal")
    normalized = relative.as_posix()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _image_dimensions(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive integers")
    return int(width), int(height)


def transform_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int,
    *,
    to_display: bool = True,
) -> tuple[float, float]:
    """Transform a point between original and display coordinates.

    The supported display transforms are 0 and 180 degrees. A 180-degree
    rotation is its own inverse: (x, y) -> (width - 1 - x, height - 1 - y).
    """
    width, height = _image_dimensions(width, height)
    if len(point) != 2:
        raise ValueError("point must contain exactly x and y")
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("point coordinates must be finite")
    if rotation_degrees not in (0, 180):
        raise ValueError("only 0- and 180-degree orientation are supported")
    if rotation_degrees == 0:
        return x, y
    return float(width - 1) - x, float(height - 1) - y


def display_to_original_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int = 0,
    roi_box: Sequence[int] | None = None,
) -> tuple[float, float]:
    """Map a point in the current full-image/ROI display view to source pixels."""
    width, height = _image_dimensions(width, height)
    if roi_box is None:
        return transform_point(point, width, height, rotation_degrees, to_display=False)
    if len(roi_box) != 4:
        raise AnnotationValidationError("ROI box must contain four pixel coordinates")
    x0, y0, x1, y1 = (int(value) for value in roi_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")
    local_x, local_y = transform_point(
        point, x1 - x0, y1 - y0, rotation_degrees, to_display=False
    )
    return x0 + local_x, y0 + local_y


def original_to_display_point(
    point: Sequence[float],
    width: int,
    height: int,
    rotation_degrees: int = 0,
    roi_box: Sequence[int] | None = None,
) -> tuple[float, float] | None:
    """Map a source-image point to the current view; return None outside an ROI."""
    width, height = _image_dimensions(width, height)
    if roi_box is None:
        return transform_point(point, width, height, rotation_degrees)
    if len(roi_box) != 4:
        raise AnnotationValidationError("ROI box must contain four pixel coordinates")
    x0, y0, x1, y1 = (int(value) for value in roi_box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")
    if len(point) != 2:
        raise ValueError("point must contain exactly x and y")
    x, y = float(point[0]), float(point[1])
    if not (x0 <= x < x1 and y0 <= y < y1):
        return None
    return transform_point((x - x0, y - y0), x1 - x0, y1 - y0, rotation_degrees)


def validate_points(
    points: Iterable[Sequence[float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    """Validate and normalize boundary points in original-image coordinates."""
    width, height = _image_dimensions(width, height)
    normalized: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if len(point) != 2:
            raise AnnotationValidationError(f"point {index} must contain x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError(f"point {index} is not numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise AnnotationValidationError(f"point {index} contains a non-finite coordinate")
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise AnnotationValidationError(
                f"point {index} ({x:.3f}, {y:.3f}) is outside {width}x{height} image bounds"
            )
        normalized.append((x, y))
    return normalized


def fit_ellipse_annotation(
    points: Iterable[Sequence[float]],
    width: int,
    height: int,
) -> dict[str, float]:
    """Fit and normalize an ellipse from at least five original-space points."""
    normalized = validate_points(points, width, height)
    if len(normalized) < 5:
        raise AnnotationValidationError("at least five boundary points are required")

    point_array = np.asarray(normalized, dtype=np.float32).reshape(-1, 1, 2)
    try:
        (center_x, center_y), (diameter_a, diameter_b), angle = cv2.fitEllipse(point_array)
    except cv2.error as exc:
        raise AnnotationValidationError(f"ellipse fitting failed: {exc}") from exc

    if diameter_a >= diameter_b:
        semi_major, semi_minor = diameter_a / 2.0, diameter_b / 2.0
        angle_deg = angle
    else:
        semi_major, semi_minor = diameter_b / 2.0, diameter_a / 2.0
        angle_deg = angle + 90.0

    values = (center_x, center_y, semi_major, semi_minor, angle_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise AnnotationValidationError("ellipse fit produced non-finite values")
    if semi_major <= 0 or semi_minor <= 0:
        raise AnnotationValidationError("ellipse axes must be positive")

    return {
        "center_x": float(center_x),
        "center_y": float(center_y),
        "semi_major": float(semi_major),
        "semi_minor": float(semi_minor),
        "angle_deg": float(angle_deg % 180.0),
    }


def normalize_roi_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> list[float]:
    """Normalize a half-open xyxy ROI to [0, 1] in original-image space."""
    width, height = _image_dimensions(width, height)
    if len(box) != 4:
        raise AnnotationValidationError("ROI box must be [x_min, y_min, x_max, y_max]")
    try:
        x0, y0, x1, y1 = (float(value) for value in box)
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError("ROI coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise AnnotationValidationError("ROI coordinates must be finite")
    x0, x1 = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    y0, y1 = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    if x1 <= x0 or y1 <= y0:
        raise AnnotationValidationError("ROI box must have positive width and height")
    return [x0 / width, y0 / height, x1 / width, y1 / height]


def denormalize_roi_box(
    box_norm: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized half-open xyxy ROI coordinates to integer pixels."""
    width, height = _image_dimensions(width, height)
    if len(box_norm) != 4:
        raise AnnotationValidationError("normalized ROI must contain four values")
    try:
        values = [float(value) for value in box_norm]
    except (TypeError, ValueError) as exc:
        raise AnnotationValidationError("normalized ROI coordinates must be numeric") from exc
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise AnnotationValidationError("normalized ROI coordinates must be finite and within [0, 1]")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise AnnotationValidationError("ROI box must have positive width and height")
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    return left, top, right, bottom


def validate_annotation_document(document: Mapping[str, Any]) -> None:
    """Validate the versioned workspace document before persistence/export."""
    if not isinstance(document, Mapping):
        raise AnnotationValidationError("annotation document must be a JSON object")
    version = str(document.get("schema_version", ""))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise AnnotationValidationError(f"unsupported annotation schema version: {version!r}")

    try:
        width, height = _image_dimensions(
            int(document["image_width"]), int(document["image_height"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationValidationError("positive image_width and image_height are required") from exc

    roi_box_norm = document.get("roi_box_norm")
    if roi_box_norm is not None:
        denormalize_roi_box(roi_box_norm, width, height)

    annotations = document.get("annotations", {})
    if not isinstance(annotations, Mapping):
        raise AnnotationValidationError("annotations must be an object")

    for mode, record in annotations.items():
        if mode not in {"pupil", "limbus", "ring"}:
            raise AnnotationValidationError(f"unsupported annotation mode: {mode!r}")
        if not isinstance(record, Mapping):
            raise AnnotationValidationError(f"{mode} annotation must be an object")
        points = record.get("boundary_points", [])
        if not isinstance(points, list):
            raise AnnotationValidationError(f"{mode} boundary_points must be a list")
        validate_points(points, width, height)
        ellipse = record.get("ellipse")
        if ellipse is not None:
            if not isinstance(ellipse, Mapping):
                raise AnnotationValidationError(f"{mode} ellipse must be an object")
            required = ("center_x", "center_y", "semi_major", "semi_minor", "angle_deg")
            try:
                values = {key: float(ellipse[key]) for key in required}
            except (KeyError, TypeError, ValueError) as exc:
                raise AnnotationValidationError(f"{mode} ellipse is missing numeric parameters") from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise AnnotationValidationError(f"{mode} ellipse contains non-finite values")
            if values["semi_major"] <= 0 or values["semi_minor"] <= 0:
                raise AnnotationValidationError(f"{mode} ellipse axes must be positive")
            if not (0 <= values["center_x"] < width and 0 <= values["center_y"] < height):
                raise AnnotationValidationError(f"{mode} ellipse center is outside image bounds")


def save_annotation_json(path: Path | str, document: Mapping[str, Any]) -> Path:
    """Atomically write a validated annotation JSON file."""
    validate_annotation_document(document)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temporary_path = Path(temp_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


def load_annotation_json(path: Path | str) -> dict[str, Any]:
    """Load and validate a workspace annotation JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    validate_annotation_document(document)
    return dict(document)


def build_segmentation_mask(
    document: Mapping[str, Any],
    roi_box: Sequence[int] | None = None,
) -> np.ndarray:
    """Rasterize manually annotated ellipses into a uint8 class mask.

    *roi_box* is a half-open [x0, y0, x1, y1] rectangle in original-image
    pixels. When supplied, the returned mask has the exact crop dimensions.
    Class IDs: 0 background, 1 pupil, 2 iris region inside limbus, 3 ring outline.
    """
    validate_annotation_document(document)
    width, height = int(document["image_width"]), int(document["image_height"])
    if roi_box is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        if len(roi_box) != 4:
            raise AnnotationValidationError("ROI box must be [x_min, y_min, x_max, y_max]")
        try:
            x0, y0, x1, y1 = (int(value) for value in roi_box)
        except (TypeError, ValueError) as exc:
            raise AnnotationValidationError("ROI pixel coordinates must be integers") from exc
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise AnnotationValidationError("ROI pixel box is outside the image or degenerate")

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    annotations = document.get("annotations", {})

    def draw_ellipse(mode: str, label: int, thickness: int) -> None:
        record = annotations.get(mode, {})
        ellipse = record.get("ellipse") if isinstance(record, Mapping) else None
        if not ellipse:
            return
        center = (
            int(round(float(ellipse["center_x"]) - x0)),
            int(round(float(ellipse["center_y"]) - y0)),
        )
        axes = (
            max(1, int(round(float(ellipse["semi_major"])))),
            max(1, int(round(float(ellipse["semi_minor"])))),
        )
        cv2.ellipse(
            mask, center, axes, float(ellipse["angle_deg"]), 0, 360,
            label, thickness=thickness,
        )

    draw_ellipse("limbus", 2, -1)
    ring_record = annotations.get("ring", {})
    ring = ring_record.get("ellipse") if isinstance(ring_record, Mapping) else None
    if ring:
        ring_axes = (
            max(1, int(round(float(ring["semi_major"])))),
            max(1, int(round(float(ring["semi_minor"])))),
        )
        draw_ellipse("ring", 3, max(2, int(round(min(ring_axes) * 0.04))))
    draw_ellipse("pupil", 1, -1)
    return mask
