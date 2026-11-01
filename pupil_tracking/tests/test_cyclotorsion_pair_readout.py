"""Tests for the display-only OD/OS cyclotorsion pair readout.

These are deterministic and synthetic: a textured synthetic iris is rendered,
the "current" image is produced by rotating it about its centre by a known
angle, and the readout must recover that angle's magnitude.  Nothing here uses
clinical imagery, and nothing touches the centration / pupil / limbus pipeline.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.pair_readout import (
    PairReadout,
    estimate_pair_readout,
)
from pupil_tracking.cyclotorsion.pentacam_loader import (
    DEFAULT_PUPIL_LIMBUS_RATIO,
    roi_norm_to_iris_roi,
)

SIZE = 400
ROI_BOX = (0.08, 0.08, 0.92, 0.92)


def _synthetic_iris(size: int = SIZE) -> np.ndarray:
    """A deterministic, richly-textured synthetic iris (unique, non-periodic)."""
    img = np.full((size, size), 20, dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), int(0.45 * size), 90, -1)   # iris/limbus disc
    rng = np.random.default_rng(7)
    for _ in range(18):
        angle = math.radians(rng.uniform(0.0, 360.0))
        radius = rng.uniform(0.18, 0.38) * size
        value = int(rng.integers(120, 256))
        pos = (int(c + radius * math.cos(angle)), int(c + radius * math.sin(angle)))
        cv2.circle(img, pos, 7, value, -1)
    return img


def _rotate(image: np.ndarray, deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _write(path, image):
    cv2.imwrite(str(path), image)
    return str(path)


# ---------------------------------------------------------------------------
# roi_norm_to_iris_roi
# ---------------------------------------------------------------------------

def test_roi_box_to_iris_roi_geometry():
    roi = roi_norm_to_iris_roi((0.2, 0.1, 0.8, 0.9), 1000, 800)
    assert roi.valid
    assert roi.center_x == pytest.approx(500.0)
    assert roi.center_y == pytest.approx(400.0)
    assert roi.limbus_semi_major == pytest.approx(300.0)
    assert roi.limbus_semi_minor == pytest.approx(320.0)
    assert roi.pupil_semi_major == pytest.approx(300.0 * DEFAULT_PUPIL_LIMBUS_RATIO)
    assert roi.pupil_semi_minor == pytest.approx(320.0 * DEFAULT_PUPIL_LIMBUS_RATIO)


def test_roi_box_to_iris_roi_respects_pupil_ratio():
    roi = roi_norm_to_iris_roi((0.0, 0.0, 1.0, 1.0), 100, 100, pupil_ratio=0.5)
    assert roi.valid
    assert roi.limbus_semi_major == pytest.approx(50.0)
    assert roi.pupil_semi_major == pytest.approx(25.0)


def test_roi_box_to_iris_roi_rejects_bad_input():
    assert not roi_norm_to_iris_roi((0.5, 0.5, 0.5, 0.9)).valid      # degenerate
    assert not roi_norm_to_iris_roi((0.9, 0.1, 0.2, 0.9)).valid      # inverted
    assert not roi_norm_to_iris_roi("nonsense").valid                # malformed
    assert not roi_norm_to_iris_roi((0.1, 0.1, 0.9, 0.9), 0, 0).valid  # bad size


# ---------------------------------------------------------------------------
# estimate_pair_readout
# ---------------------------------------------------------------------------

def test_pair_readout_recovers_known_rotation(tmp_path):
    ref = _synthetic_iris()
    cur = _rotate(ref, 30.0)
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", cur)

    out = estimate_pair_readout(cp, rp, ROI_BOX, flip=False)
    assert out.valid, out.reason
    assert abs(abs(out.angle_deg) - 30.0) <= 6.0, out.angle_deg
    assert out.score > 0.0


def test_pair_readout_flip_preserves_rotation_magnitude(tmp_path):
    ref = _synthetic_iris()
    cur = _rotate(ref, -25.0)
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", cur)

    out = estimate_pair_readout(cp, rp, ROI_BOX, flip=True)
    assert out.valid, out.reason
    assert abs(abs(out.angle_deg) - 25.0) <= 6.0, out.angle_deg


def test_pair_readout_identical_images_give_zero(tmp_path):
    ref = _synthetic_iris()
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", ref.copy())
    out = estimate_pair_readout(cp, rp, ROI_BOX, flip=False)
    assert out.valid
    assert abs(out.angle_deg) <= 1.0


def test_pair_readout_missing_paths():
    assert estimate_pair_readout("", "x.png", ROI_BOX).reason == "missing image path"
    assert estimate_pair_readout("x.png", "", ROI_BOX).reason == "missing image path"
    r = estimate_pair_readout("", "x.png", ROI_BOX)
    assert not r.valid


def test_pair_readout_no_roi(tmp_path):
    ref = _synthetic_iris()
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", ref.copy())
    out = estimate_pair_readout(cp, rp, None)
    assert not out.valid
    assert out.reason == "no iris ROI"


def test_pair_readout_unreadable_images(tmp_path):
    out = estimate_pair_readout(
        str(tmp_path / "nope.png"), str(tmp_path / "nope2.png"), ROI_BOX
    )
    assert not out.valid
    assert out.reason == "unreadable image"


def test_pair_readout_degenerate_roi(tmp_path):
    ref = _synthetic_iris()
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", ref.copy())
    out = estimate_pair_readout(cp, rp, (0.5, 0.5, 0.5, 0.9))
    assert not out.valid
    assert out.reason == "invalid ROI box"


def test_pair_readout_to_dict_contract(tmp_path):
    ref = _synthetic_iris()
    rp = _write(tmp_path / "ref.png", ref)
    cp = _write(tmp_path / "cur.png", _rotate(ref, 12.0))
    out = estimate_pair_readout(cp, rp, ROI_BOX, flip=False)
    d = out.to_dict()
    for key in ("valid", "angle_deg", "score", "reason", "n_valid", "n_total"):
        assert key in d
    assert isinstance(out, PairReadout)
