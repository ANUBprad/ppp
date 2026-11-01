from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pupil_tracking.annotation.workspace_core import (
    AnnotationValidationError,
    build_segmentation_mask,
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
