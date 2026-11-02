from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pupil_tracking.annotation.workspace_core import (
    AnnotationValidationError,
    build_segmentation_mask,
    denormalize_roi_box,
    display_to_original_point,
    original_to_display_point,
    discover_images,
    fit_ellipse_annotation,
    load_annotation_json,
    normalize_roi_box,
    save_annotation_json,
    stable_image_id,
    transform_point,
    validate_annotation_document,
)


def _document():
    return {
        "schema_version": "1.1",
        "image_id": "abc123",
        "image_width": 100,
        "image_height": 80,
        "annotations": {
            "limbus": {
                "boundary_points": [],
                "ellipse": {
                    "center_x": 50.0, "center_y": 40.0,
                    "semi_major": 25.0, "semi_minor": 20.0, "angle_deg": 0.0,
                },
            },
            "pupil": {
                "boundary_points": [],
                "ellipse": {
                    "center_x": 50.0, "center_y": 40.0,
                    "semi_major": 10.0, "semi_minor": 8.0, "angle_deg": 0.0,
                },
            },
        },
    }


@pytest.mark.parametrize(
    "point,width,height",
    [((0, 0), 1200, 838), ((1199, 837), 1200, 838), ((320.25, 112.5), 640, 480)],
)
def test_180_degree_transform_is_its_own_inverse(point, width, height):
    displayed = transform_point(point, width, height, 180)
    restored = transform_point(displayed, width, height, 180, to_display=False)
    assert restored == pytest.approx(point)


def test_zero_degree_transform_is_identity():
    point = (12.5, 20.25)
    assert transform_point(point, 100, 80, 0) == point


def test_unsupported_rotation_is_rejected():
    with pytest.raises(ValueError, match="only 0- and 180"):
        transform_point((1, 2), 100, 80, 90)


def test_stable_image_ids_are_deterministic_and_path_specific():
    assert stable_image_id("case_a/image.bmp") == stable_image_id("case_a/image.bmp")
    assert stable_image_id("case_a/image.bmp") != stable_image_id("case_b/image.bmp")
    assert len(stable_image_id("case_a/image.bmp")) == 24


@pytest.mark.parametrize("unsafe_path", ["../patient/image.bmp", "/absolute/image.bmp", ""])
def test_stable_image_id_rejects_unsafe_paths(unsafe_path):
    with pytest.raises(ValueError):
        stable_image_id(unsafe_path)


def test_discover_images_filters_extensions_and_private_artifacts(tmp_path):
    (tmp_path / "a.JPG").write_bytes(b"not decoded here")
    (tmp_path / "b.tiff").write_bytes(b"not decoded here")
    (tmp_path / "notes.pdf").write_bytes(b"not an image")
    private = tmp_path / ".pentacam_annotations"
    private.mkdir()
    (private / "overlay.png").write_bytes(b"private")
    found = discover_images(tmp_path)
    assert [item.name for item in found] == ["a.JPG", "b.tiff"]


def test_discover_images_rejects_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        discover_images(tmp_path / "missing")


def test_fit_ellipse_annotation_returns_positive_axes():
    points = [
        (50 + 20 * np.cos(angle), 40 + 10 * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ]
    ellipse = fit_ellipse_annotation(points, 100, 80)
    assert ellipse["center_x"] == pytest.approx(50, abs=0.5)
    assert ellipse["center_y"] == pytest.approx(40, abs=0.5)
    assert ellipse["semi_major"] >= ellipse["semi_minor"] > 0


def test_fit_ellipse_requires_five_points():
    with pytest.raises(AnnotationValidationError, match="five"):
        fit_ellipse_annotation([(1, 1), (2, 2), (3, 3), (4, 4)], 20, 20)


@pytest.mark.parametrize("bad_point", [(-1, 2), (10, 2), (float("nan"), 1)])
def test_fit_ellipse_rejects_invalid_points(bad_point):
    with pytest.raises(AnnotationValidationError):
        fit_ellipse_annotation(
            [(1, 1), (2, 2), (3, 3), (4, 4), bad_point], 10, 10
        )


def test_roi_normalization_clamps_and_normalizes():
    assert normalize_roi_box((-10, 10, 50, 90), 100, 80) == pytest.approx(
        [0.0, 0.125, 0.5, 1.0]
    )


def test_roi_normalization_rejects_degenerate_box():
    with pytest.raises(AnnotationValidationError, match="positive"):
        normalize_roi_box((10, 10, 10, 20), 100, 80)


def test_roi_normalize_denormalize_round_trip():
    pixel_box = (120, 80, 900, 700)
    normalized = normalize_roi_box(pixel_box, 1200, 838)
    assert denormalize_roi_box(normalized, 1200, 838) == pixel_box


def test_denormalize_roi_rejects_out_of_range_coordinates():
    with pytest.raises(AnnotationValidationError, match="within"):
        denormalize_roi_box((-0.1, 0.1, 0.8, 0.9), 100, 80)


def test_roi_coordinate_mapping_preserves_source_points_at_zero_degrees():
    roi = (100, 50, 500, 350)
    source = display_to_original_point((25.5, 40.25), 800, 600, 0, roi)
    assert source == pytest.approx((125.5, 90.25))
    assert original_to_display_point(source, 800, 600, 0, roi) == pytest.approx((25.5, 40.25))


def test_roi_coordinate_mapping_round_trips_after_180_degree_rotation():
    roi = (100, 50, 500, 350)
    display_point = (12.25, 18.5)
    source = display_to_original_point(display_point, 800, 600, 180, roi)
    assert source == pytest.approx((486.75, 330.5))
    assert original_to_display_point(source, 800, 600, 180, roi) == pytest.approx(display_point)


def test_original_point_outside_roi_is_not_drawn_in_crop_preview():
    roi = (100, 50, 500, 350)
    assert original_to_display_point((99.5, 100), 800, 600, 0, roi) is None


def test_document_validation_accepts_and_rejects_normalized_roi():
    document = _document()
    document["roi_box_norm"] = [0.1, 0.1, 0.9, 0.9]
    validate_annotation_document(document)
    document["roi_box_norm"] = [0.8, 0.2, 0.2, 0.9]
    with pytest.raises(AnnotationValidationError, match="positive"):
        validate_annotation_document(document)


def test_roi_mask_is_cropped_and_uses_original_coordinate_offset():
    document = _document()
    cropped = build_segmentation_mask(document, roi_box=(40, 30, 80, 70))
    assert cropped.shape == (40, 40)
    assert cropped.dtype == np.uint8
    assert cropped[10, 10] == 1
    assert cropped[10, 30] == 2


def test_roi_mask_rejects_invalid_pixel_bounds():
    with pytest.raises(AnnotationValidationError, match="outside"):
        build_segmentation_mask(_document(), roi_box=(90, 70, 110, 90))


def test_document_validation_accepts_valid_schema():
    validate_annotation_document(_document())


def test_document_validation_rejects_out_of_bounds_points():
    document = _document()
    document["annotations"]["pupil"]["boundary_points"] = [[100, 2]]
    with pytest.raises(AnnotationValidationError, match="outside"):
        validate_annotation_document(document)


def test_document_validation_rejects_nonpositive_axes():
    document = _document()
    document["annotations"]["pupil"]["ellipse"]["semi_major"] = 0
    with pytest.raises(AnnotationValidationError, match="positive"):
        validate_annotation_document(document)


def test_annotation_json_round_trip_is_atomic_and_validated(tmp_path):
    destination = tmp_path / "nested" / "annotation.json"
    source = _document()
    save_annotation_json(destination, source)
    restored = load_annotation_json(destination)
    assert restored == source
    assert not list(destination.parent.glob("*.tmp"))


def test_annotation_json_rejects_invalid_document(tmp_path):
    with pytest.raises(AnnotationValidationError):
        save_annotation_json(tmp_path / "bad.json", {"schema_version": "99"})


def test_segmentation_mask_uses_expected_class_ids_and_geometry():
    mask = build_segmentation_mask(_document())
    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8
    assert mask[40, 50] == 1       # pupil
    assert mask[40, 70] == 2       # iris inside limbus
    assert mask[5, 5] == 0         # background


def test_segmentation_mask_supports_missing_optional_structures():
    document = _document()
    document["annotations"] = {}
    mask = build_segmentation_mask(document)
    assert mask.shape == (80, 100)
    assert np.count_nonzero(mask) == 0
