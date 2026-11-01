"""Tests for Module 4 — cyclotorsion (torsional angle) estimation.

Deterministic and synthetic.  The core contract exercised here is:

    roll_strip(reference, result.angle_deg / 360 * width) ~= target

and its image-space counterpart (an image rotated by ``+theta`` with
``cv2.getRotationMatrix2D`` yields ``angle_deg == -theta``).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from pupil_tracking.cyclotorsion.cyclotorsion_engine import (
    CyclotorsionResult,
    align_strip,
    estimate_cyclotorsion,
    estimate_from_images,
)
from pupil_tracking.cyclotorsion.polar_unwrap import roll_strip, unwrap_iris
from pupil_tracking.iris.types import IrisROI


H, W = 32, 128


def _textured_strip(seed: int = 0, height: int = H, width: int = W) -> np.ndarray:
    """A deterministic, cyclically-smoothed random strip (ideal for shifts)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    # Cyclic (wrap-around) smoothing along the angular axis so the strip stays
    # consistent under np.roll — GaussianBlur cannot use BORDER_WRAP here.
    smooth = noise.copy()
    for s in (1, 2):
        smooth = smooth + np.roll(noise, s, axis=1) + np.roll(noise, -s, axis=1)
    smooth /= 5.0
    lo = float(smooth.min())
    hi = float(smooth.max())
    smooth = (smooth - lo) / max(hi - lo, 1e-6)
    return (smooth * 200.0 + 20.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Identity / basic behaviour
# ---------------------------------------------------------------------------

def test_identical_strips_give_zero_angle():
    strip = _textured_strip()
    res = estimate_cyclotorsion(strip, strip.copy())
    assert isinstance(res, CyclotorsionResult)
    assert res.valid
    assert res.reason == "ok"
    assert abs(res.angle_deg) < 0.5
    assert res.score > 0.99


def test_result_to_dict_round_trip():
    res = estimate_cyclotorsion(_textured_strip(), _textured_strip())
    d = res.to_dict()
    for key in ("angle_deg", "shift_cols", "score", "valid", "reason",
                "n_valid", "n_total", "search_range_deg"):
        assert key in d
    assert d["valid"] is True


# ---------------------------------------------------------------------------
# Known integer shifts
# ---------------------------------------------------------------------------

def test_known_positive_roll():
    ref = _textured_strip()
    k = 17
    target = roll_strip(ref, k)
    res = estimate_cyclotorsion(ref, target)
    expected = k / W * 360.0
    assert res.valid
    assert abs(res.angle_deg - expected) <= 0.6
    assert abs(res.shift_cols - k) <= 0.1


def test_known_negative_roll():
    ref = _textured_strip()
    k = -23
    target = roll_strip(ref, k)
    res = estimate_cyclotorsion(ref, target)
    expected = k / W * 360.0
    assert res.valid
    assert abs(res.angle_deg - expected) <= 0.6


def test_shift_cols_is_consistent_with_angle():
    ref = _textured_strip()
    target = roll_strip(ref, 11)
    res = estimate_cyclotorsion(ref, target, angle_range_deg=60.0)
    assert abs(res.shift_cols - res.angle_deg / 360.0 * W) < 1e-6


def test_align_strip_reproduces_target():
    ref = _textured_strip()
    target = roll_strip(ref, 14)
    res = estimate_cyclotorsion(ref, target)
    aligned = align_strip(ref, res.angle_deg)
    # Near-integer shift: aligned and target should be almost identical.
    assert np.mean(np.abs(aligned.astype(np.float32) - target.astype(np.float32))) < 3.0


# ---------------------------------------------------------------------------
# Sub-pixel accuracy
# ---------------------------------------------------------------------------

def test_subpixel_angular_offset():
    # Unwrapping with an angle offset rotates the *sampling grid*, so the strip
    # is rolled by a fractional number of columns; the engine must recover the
    # sub-pixel angle (here -10.5 deg).
    base = _spoke_image()
    roi = _circular_roi()
    offset_deg = 10.5
    ref = unwrap_iris(base, roi, (64, 512), angle_offset_deg=0.0)
    target = unwrap_iris(base, roi, (64, 512), angle_offset_deg=offset_deg)
    res = estimate_cyclotorsion(ref, target, angle_range_deg=90.0)
    assert res.valid
    assert abs(res.angle_deg - (-offset_deg)) <= 0.5


# ---------------------------------------------------------------------------
# Image-space sign convention
# ---------------------------------------------------------------------------

CX = CY = 100.0
PUPIL_R = 30.0
LIMBUS_R = 80.0
IMG_SIZE = 200


def _circular_roi() -> IrisROI:
    return IrisROI(
        valid=True, reason="ok", center_x=CX, center_y=CY,
        pupil_semi_major=PUPIL_R, pupil_semi_minor=PUPIL_R, pupil_angle_deg=0.0,
        limbus_semi_major=LIMBUS_R, limbus_semi_minor=LIMBUS_R, limbus_angle_deg=0.0,
        inner_inset_frac=0.0, outer_inset_frac=0.0,
        pupil_radius_px=PUPIL_R, limbus_radius_px=LIMBUS_R,
    )


def _spoke_image(angle_deg: float = 0.0) -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    a = math.radians(angle_deg)
    p0 = (int(round(CX + (PUPIL_R - 2) * math.cos(a))),
          int(round(CY + (PUPIL_R - 2) * math.sin(a))))
    p1 = (int(round(CX + (LIMBUS_R - 5) * math.cos(a))),
          int(round(CY + (LIMBUS_R - 5) * math.sin(a))))
    cv2.line(img, p0, p1, 255, 5)
    return img


def _multi_spoke_image() -> np.ndarray:
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for ang in (0.0, 40.0, 110.0, 200.0, 290.0):
        a = math.radians(ang)
        p0 = (int(round(CX + (PUPIL_R - 2) * math.cos(a))),
              int(round(CY + (PUPIL_R - 2) * math.sin(a))))
        p1 = (int(round(CX + (LIMBUS_R - 5) * math.cos(a))),
              int(round(CY + (LIMBUS_R - 5) * math.sin(a))))
        cv2.line(img, p0, p1, 255, 4)
    return img


def test_image_rotation_sign_convention():
    base = _multi_spoke_image()
    roi = _circular_roi()
    for theta in (10.0, 25.0, 45.0):
        m = cv2.getRotationMatrix2D((CX, CY), theta, 1.0)
        rotated = cv2.warpAffine(base, m, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)
        res = estimate_from_images(base, rotated, roi, output_size=(64, 512),
                                   angle_range_deg=90.0)
        assert res.valid
        # getRotationMatrix2D(+theta) => engine reports -theta.
        assert abs(res.angle_deg - (-theta)) <= 1.5


# ---------------------------------------------------------------------------
# Robustness: masks, occlusion, gates
# ---------------------------------------------------------------------------

def test_mask_excludes_fixed_occlusion():
    ref = _textured_strip(seed=3)
    k = 19
    target = roll_strip(ref, k)

    # A bright occluder at the SAME strip location in both images (e.g. an
    # eyelid) would bias the NCC toward zero shift.  Masking it out recovers
    # the true shift.
    occ = np.zeros_like(ref, dtype=bool)
    occ[6:18, 40:70] = True
    ref_occ = ref.copy()
    target_occ = target.copy()
    ref_occ[occ] = 255
    target_occ[occ] = 255
    mask = ~occ

    biased = estimate_cyclotorsion(ref_occ, target_occ, radial_normalize=False)
    corrected = estimate_cyclotorsion(
        ref_occ, target_occ,
        reference_mask=mask, target_mask=mask,
        radial_normalize=False,
    )
    expected = k / W * 360.0
    assert corrected.valid
    assert abs(corrected.angle_deg - expected) <= 1.0
    assert corrected.score > biased.score


def test_flat_strips_are_invalid():
    flat = np.full((H, W), 128, dtype=np.uint8)
    res = estimate_cyclotorsion(flat, flat)
    assert not res.valid
    assert "correlation" in res.reason


def test_uncorrelated_strips_are_invalid():
    a = _textured_strip(seed=1)
    b = _textured_strip(seed=99)
    res = estimate_cyclotorsion(a, b, min_score=0.5)
    assert not res.valid
    assert "correlation" in res.reason


def test_shape_mismatch_is_invalid():
    a = _textured_strip()
    b = _textured_strip(height=16, width=64)
    res = estimate_cyclotorsion(a, b)
    assert not res.valid
    assert "mismatch" in res.reason


def test_missing_strip_is_invalid():
    res = estimate_cyclotorsion(None, _textured_strip())
    assert not res.valid
    assert "missing" in res.reason


def test_angle_range_limits_search():
    ref = _textured_strip()
    target = roll_strip(ref, 40)  # 112.5 deg at width 128
    limited = estimate_cyclotorsion(ref, target, angle_range_deg=20.0)
    assert abs(limited.angle_deg) <= 20.0
    assert limited.search_range_deg == 20.0


def test_bgr_and_gray_agree():
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(H, W), dtype=np.uint8)
    gray = np.ascontiguousarray(noise)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    res_gray = estimate_cyclotorsion(gray, roll_strip(gray, 9))
    res_bgr = estimate_cyclotorsion(bgr, roll_strip(bgr, 9))
    assert abs(res_gray.angle_deg - res_bgr.angle_deg) < 1e-3


def test_estimate_from_images_invalid_roi():
    img = _spoke_image()
    res = estimate_from_images(img, img, IrisROI(valid=False))
    assert not res.valid
    assert "unwrap" in res.reason
