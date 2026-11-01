"""Tests for Module 3 — iris features on the unwrapped strip.

Deterministic and synthetic: no clinical imagery is used.  The properties that
matter for cyclotorsion are asserted explicitly:

* the angular profile is exactly cyclic-shift equivariant, and
* rolling the strip shifts every feature by the same number of columns while
  leaving its descriptor unchanged (rotation-invariant matching).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.iris_features import (
    DEFAULT_DESCRIPTOR_BINS,
    STATUS_EMPTY,
    STATUS_NO_FEATURES,
    STATUS_OK,
    StripFeatureType,
    angular_profile,
    angular_signature,
    extract_features_from_image,
    extract_strip_features,
)
from pupil_tracking.iris.types import IrisROI


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

W = 256
H = 64
CX = CY = 100.0
PUPIL_R = 30.0
LIMBUS_R = 80.0
IMG_SIZE = 200


def _blob_strip(col: int = 62, row: int = 32, h: int = H, w: int = W,
                sigma: float = 3.0, amp: float = 255.0) -> np.ndarray:
    """A flat strip with a single Gaussian bright blob."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (amp * np.exp(-(((xx - col) ** 2 + (yy - row) ** 2) / (2 * sigma ** 2)))
            ).astype(np.uint8)


def _circular_roi(pupil_r: float = PUPIL_R, limbus_r: float = LIMBUS_R) -> IrisROI:
    return IrisROI(
        valid=True, reason="ok",
        center_x=CX, center_y=CY,
        pupil_semi_major=pupil_r, pupil_semi_minor=pupil_r, pupil_angle_deg=0.0,
        limbus_semi_major=limbus_r, limbus_semi_minor=limbus_r, limbus_angle_deg=0.0,
        inner_inset_frac=0.0, outer_inset_frac=0.0,
        pupil_radius_px=pupil_r, limbus_radius_px=limbus_r,
    )


def _spoke_image(angle_deg: float = 0.0, width: int = 5) -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    a = math.radians(angle_deg)
    p0 = (int(round(CX + (PUPIL_R - 2) * math.cos(a))),
          int(round(CY + (PUPIL_R - 2) * math.sin(a))))
    p1 = (int(round(CX + (LIMBUS_R - 5) * math.cos(a))),
          int(round(CY + (LIMBUS_R - 5) * math.sin(a))))
    cv2.line(img, p0, p1, 255, width)
    return img


def _single_feature(strip: np.ndarray):
    """Extract exactly one feature (suppression wide enough to keep just one)."""
    return extract_strip_features(strip, min_angular_sep_deg=180.0, min_radial_sep=1.0)


# ---------------------------------------------------------------------------
# Angular profile
# ---------------------------------------------------------------------------

def test_angular_profile_length_and_dtype():
    p = angular_profile(_blob_strip())
    assert p is not None
    assert p.shape == (W,)
    assert p.dtype == np.float32
    # The blob's column should still dominate after radial normalisation.
    assert abs(int(np.argmax(p)) - 62) <= 4


def test_angular_profile_rotation_equivariance():
    strip = _blob_strip()
    k = 40
    p = angular_profile(strip)
    pr = angular_profile(np.roll(strip, k, axis=1))
    assert np.allclose(pr, np.roll(p, k), atol=1e-4)


def test_angular_profile_median_reducer():
    p = angular_profile(_blob_strip(), reducer="median")
    assert p is not None and p.shape == (W,)


def test_angular_profile_masked():
    strip = _blob_strip(col=62)
    mask = np.zeros((H, W), dtype=bool)
    mask[:, 150:250] = True
    p = angular_profile(strip, mask, radial_normalize=False)
    assert p is not None
    assert p.shape == (W,)
    # Columns outside the mask contribute nothing.
    assert np.allclose(p[0:150], 0.0, atol=1e-6)
    assert np.allclose(p[250:], 0.0, atol=1e-6)


def test_angular_profile_invalid_inputs():
    assert angular_profile(None) is None
    assert angular_profile(np.zeros(10)) is None
    assert angular_profile(np.zeros((0, 0), dtype=np.uint8)) is None
    # Supplied-but-unusable mask -> invalid.
    assert angular_profile(_blob_strip(), np.ones((3, 3), dtype=bool)) is None


# ---------------------------------------------------------------------------
# Angular signature
# ---------------------------------------------------------------------------

def test_angular_signature_zero_mean_unit_norm():
    sig = angular_signature(_blob_strip())
    assert sig is not None
    assert abs(float(sig.mean())) < 1e-5
    assert abs(float(np.linalg.norm(sig)) - 1.0) < 1e-5


def test_angular_signature_flat_strip_is_zero():
    sig = angular_signature(np.full((H, W), 100, dtype=np.uint8))
    assert sig is not None
    assert np.allclose(sig, 0.0)


def test_angular_signature_roll_is_cyclic():
    strip = _blob_strip()
    k = 33
    a = angular_signature(strip)
    b = angular_signature(np.roll(strip, k, axis=1))
    assert np.allclose(b, np.roll(a, k), atol=1e-4)


def test_angular_signature_accepts_bgr():
    sig = angular_signature(cv2.cvtColor(_blob_strip(), cv2.COLOR_GRAY2BGR))
    assert sig is not None and sig.shape == (W,)


def test_angular_signature_invalid_returns_none():
    assert angular_signature(None) is None
    assert angular_signature(np.zeros(5)) is None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_features_finds_blob():
    res = extract_strip_features(_blob_strip(col=62, row=32))
    assert res.status == STATUS_OK
    assert res.num_accepted >= 1
    assert res.angular_profile is not None
    best = min(res.features, key=lambda f: abs(f.column - 62) + abs(f.row - 32))
    assert abs(best.column - 62) <= 6
    assert abs(best.row - 32) <= 6


def test_feature_angle_matches_column_and_ranges():
    res = extract_strip_features(_blob_strip())
    assert res.features
    for f in res.features:
        assert abs(f.angle_deg - f.column / res.width * 360.0) < 1e-6
        assert 0.0 <= f.radial_norm <= 1.0
        assert f.response >= 0.0
        assert 0.0 <= f.local_contrast <= 1.0
        assert 0.0 <= f.confidence <= 1.0
        assert isinstance(f.feature_type, StripFeatureType)


def test_feature_descriptors_are_unit_norm():
    res = extract_strip_features(_blob_strip())
    assert res.features
    for f in res.features:
        assert f.descriptor is not None
        assert f.descriptor.shape == (DEFAULT_DESCRIPTOR_BINS,)
        assert abs(float(np.linalg.norm(f.descriptor)) - 1.0) < 1e-5


def test_roll_shifts_feature_and_preserves_descriptor():
    strip = _blob_strip(col=62, row=32)
    k = 40
    s1 = _single_feature(strip)
    s2 = _single_feature(np.roll(strip, k, axis=1))

    assert len(s1.features) == 1
    assert len(s2.features) == 1
    f1, f2 = s1.features[0], s2.features[0]
    w = s1.width

    assert (f2.column - f1.column) % w == k % w
    assert f2.row == f1.row
    # Descriptor is unchanged by the roll -> rotation-invariant matching.
    assert np.allclose(f1.descriptor, f2.descriptor, atol=1e-5)
    assert abs(f1.confidence - f2.confidence) < 1e-6
    assert abs(f1.response - f2.response) < 1e-6


def test_mask_excludes_features():
    strip = _blob_strip(col=62)
    mask = np.zeros((H, W), dtype=bool)
    mask[:, 150:250] = True
    res = extract_strip_features(strip, mask=mask)
    # The blob is outside the mask (and the rest is flat) -> honest refusal.
    assert res.num_accepted == 0
    assert res.status == STATUS_NO_FEATURES


def test_flat_strip_has_no_features():
    res = extract_strip_features(np.full((H, W), 128, dtype=np.uint8))
    assert res.status == STATUS_NO_FEATURES
    assert res.num_accepted == 0


def test_small_but_valid_strip():
    # A tiny strip must still be handled without raising.
    res = extract_strip_features(np.zeros((2, 4), dtype=np.uint8))
    assert res.status in (STATUS_EMPTY, STATUS_NO_FEATURES)


def test_invalid_strip_inputs_are_empty():
    assert extract_strip_features(None).status == STATUS_EMPTY
    assert extract_strip_features(np.zeros(10)).status == STATUS_EMPTY
    assert extract_strip_features(np.zeros((1, 1), dtype=np.uint8)).status == STATUS_EMPTY
    # Supplied-but-unusable mask.
    assert extract_strip_features(_blob_strip(), np.ones((3, 3), bool)).status == STATUS_EMPTY


def test_confidence_is_bounded_and_status_consistent():
    res = extract_strip_features(_blob_strip())
    assert res.status == STATUS_OK
    assert res.num_accepted == len(res.features)
    assert res.num_candidates >= res.num_accepted
    for f in res.features:
        assert 0.0 <= f.confidence <= 1.0


# ---------------------------------------------------------------------------
# Convenience: image + ROI
# ---------------------------------------------------------------------------

def test_features_from_image_spoke_near_zero_angle():
    res = extract_features_from_image(_spoke_image(0.0), _circular_roi(), output_size=(64, 512))
    assert res.status == STATUS_OK
    w = res.width
    near = [f for f in res.features if min(f.column, w - f.column) <= 8]
    assert near, "expected a feature near the 0-degree spoke"


def test_features_from_image_applies_angle_offset():
    res = extract_features_from_image(
        _spoke_image(0.0), _circular_roi(),
        output_size=(64, 512), angle_offset_deg=90.0,
    )
    assert res.status == STATUS_OK
    # With angle_offset_deg=90 the 0-degree spoke lands at strip angle 270,
    # and reporting the image-frame angle adds the offset back (270+90=360=0).
    near = [f for f in res.features if min(f.angle_deg % 360.0, 360.0 - f.angle_deg % 360.0) <= 8.0]
    assert near, "angle_offset_deg must be added back to reported angles"


def test_features_from_image_invalid_roi_is_empty():
    res = extract_features_from_image(_spoke_image(), IrisROI(valid=False))
    assert res.status == STATUS_EMPTY
    assert res.num_accepted == 0


# ---------------------------------------------------------------------------
# Serialisation contract
# ---------------------------------------------------------------------------

def test_to_dict_contract():
    res = extract_strip_features(_blob_strip())
    d = res.to_dict()
    for key in ("height", "width", "num_candidates", "num_accepted",
                "status", "features", "has_profile"):
        assert key in d
    assert d["has_profile"] is True
    assert d["width"] == W
    if res.features:
        fd = res.features[0].to_dict()
        for key in ("id", "column", "row", "angle_deg", "radial_norm",
                    "response", "local_contrast", "confidence",
                    "feature_type", "descriptor_len"):
            assert key in fd
        assert fd["descriptor_len"] == DEFAULT_DESCRIPTOR_BINS
