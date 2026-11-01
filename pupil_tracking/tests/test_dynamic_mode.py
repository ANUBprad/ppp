"""Tests for the Dynamic (Live) Mode analysis core.

``pupil_tracking.cyclotorsion.dynamic_mode`` is the pure-logic half of the
live cyclotorsion monitor: for every *live* frame it unwraps the iris annulus
using the pupil/limbus geometry the main pipeline already produced, and
correlates that strip with a cached Pentacam baseline strip.

The tests are deterministic and synthetic.  Two stubs keep the ML out:

* :class:`_StubDetector` stands in for ``UnifiedDetector`` when the Pentacam
  baseline is built (the live path never runs its own detector — it consumes
  the main window's ``_current_result``).
* the synthetic iris itself is richly textured but periodic-free, so the
  sign/shape of the rotation estimate is meaningful.

Nothing here touches the live centration / pupil / limbus pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion import dynamic_mode as dm
from pupil_tracking.utils.types import (
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)

SIZE = 400
CENTRE = SIZE // 2
PUPIL_R = 70.0
LIMBUS_R = 175.0


# ---------------------------------------------------------------------------
# Synthetic eye
# ---------------------------------------------------------------------------

def _synthetic_iris(size: int = SIZE) -> np.ndarray:
    """A deterministic, richly-textured synthetic iris (unique, non-periodic)."""
    img = np.full((size, size), 20, dtype=np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), 180, 90, -1)  # iris/limbus disc
    rng = np.random.default_rng(7)
    for _ in range(24):
        angle = math.radians(rng.uniform(0.0, 360.0))
        radius = rng.uniform(80.0, 165.0)
        value = int(rng.integers(120, 256))
        pos = (int(c + radius * math.cos(angle)), int(c + radius * math.sin(angle)))
        cv2.circle(img, pos, 8, value, -1)
    return img


def _rotate(image: np.ndarray, deg: float, centre=(CENTRE, CENTRE)) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((centre[0], centre[1]), deg, 1.0)
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubDetector:
    """Deterministic stand-in for ``UnifiedDetector`` (matches the synthetic)."""

    def __init__(self, *, pupil_conf: float = 0.93, limbus_conf: float = 0.87) -> None:
        self.calls = 0
        self._pupil_conf = pupil_conf
        self._limbus_conf = limbus_conf

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        self.calls += 1
        return _detection(self._pupil_conf, self._limbus_conf)


def _detection(pupil_conf: float = 0.93, limbus_conf: float = 0.87) -> EyeDetectionResult:
    return EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=float(CENTRE), center_y=float(CENTRE),
                semi_major=float(PUPIL_R), semi_minor=float(PUPIL_R) * 0.96,
                angle_deg=0.0,
            ),
            confidence=pupil_conf,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=float(CENTRE), center_y=float(CENTRE),
                semi_major=float(LIMBUS_R), semi_minor=float(LIMBUS_R) * 0.97,
                angle_deg=0.0,
            ),
            confidence=limbus_conf,
        ),
        overall_quality=DetectionQuality.RESEARCH,
        overall_confidence=(pupil_conf + limbus_conf) / 2.0,
    )


def _no_detection() -> EyeDetectionResult:
    return EyeDetectionResult(
        pupil=PupilDetection(detected=False, confidence=0.1),
        limbus=LimbusDetection(detected=False, confidence=0.1),
        overall_quality=DetectionQuality.INSUFFICIENT,
        overall_confidence=0.1,
    )


def _reference(image: np.ndarray, **kw) -> dm.ReferenceStrip:
    kw.setdefault("detector", _StubDetector())
    kw.setdefault("flip", False)
    return dm.build_reference_strip(image, **kw)


# ---------------------------------------------------------------------------
# build_reference_strip
# ---------------------------------------------------------------------------

def test_reference_builds_a_strip_with_detection():
    ref = _reference(_synthetic_iris())
    assert ref.loaded
    assert ref.pupil_detected and ref.limbus_detected
    assert ref.pupil_confidence == pytest.approx(0.93)
    assert ref.limbus_confidence == pytest.approx(0.87)
    assert dm.DEFAULT_OUTPUT_SIZE == (64, 512)
    assert ref.strip is not None
    assert (ref.strip_height, ref.strip_width) == (dm.DEFAULT_OUTPUT_SIZE)
    # the annulus is brighter than the dark background outside the limbus, so
    # a cropped/collapsed strip would lose this contrast entirely
    assert float(ref.strip.mean()) > 40.0


def test_reference_reports_none_without_an_image():
    out = dm.build_reference_strip(None)
    assert not out.loaded
    assert out.strip is None
    assert not out.ok
    assert out.reason == "no image"


def test_reference_without_detector_and_no_box_has_no_strip():
    out = dm.build_reference_strip(_synthetic_iris(), detector=None)
    assert out.loaded
    assert not out.roi_valid
    assert out.strip is None
    assert not out.ok


def test_reference_uses_the_recorded_roi_norm_fallback():
    ref = dm.build_reference_strip(
        _synthetic_iris(), detector=None, roi_norm=(0.2, 0.2, 0.8, 0.8)
    )
    assert ref.roi_valid
    assert ref.roi_source == "recorded box"
    assert ref.strip is not None


def test_reference_handles_a_none_image():
    ref = dm.build_reference_strip(None, detector=_StubDetector())
    assert not ref.loaded
    assert ref.reason == "no image"
    assert ref.error


def test_reference_to_dict_is_json_safe_and_omits_the_strip():
    ref = _reference(_synthetic_iris())
    payload = json.loads(json.dumps(ref.to_dict()))
    assert "strip" in payload
    assert payload["strip"]["available"] is True
    assert "array" not in json.dumps(payload)
    assert payload["pupil"]["confidence"] == pytest.approx(0.93)


# ---------------------------------------------------------------------------
# analyse_live_frame
# ---------------------------------------------------------------------------

def test_live_identical_strip_reports_zero():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _detection(), _reference(base))
    assert res.loaded
    assert res.strip is not None
    assert res.strip.shape == (dm.DEFAULT_OUTPUT_SIZE[0], dm.DEFAULT_OUTPUT_SIZE[1], 3)
    assert res.roi_valid
    assert res.angle_deg == pytest.approx(0.0, abs=0.5)
    assert res.score == pytest.approx(1.0, abs=1e-3)
    assert res.valid
    assert res.within_limit
    assert res.n_total > 0
    assert res.n_valid == res.n_total


def test_live_rotation_is_antisymmetric():
    base = _synthetic_iris()
    ref = _reference(base)
    plus = dm.analyse_live_frame(_rotate(base, 9.0), _detection(), ref)
    minus = dm.analyse_live_frame(_rotate(base, -9.0), _detection(), ref)
    assert plus.angle_deg < 0
    assert minus.angle_deg > 0
    assert plus.valid and minus.valid


def test_live_recovers_a_known_rotation():
    base = _synthetic_iris()
    ref = _reference(base)
    res = dm.analyse_live_frame(_rotate(base, -12.0), _detection(), ref)
    assert res.valid
    assert res.angle_deg == pytest.approx(12.0, abs=3.0)
    assert res.score >= dm.DEFAULT_MIN_SCORE


def test_live_without_a_frame_reports_the_reason():
    res = dm.analyse_live_frame(None, _detection(), _reference(_synthetic_iris()))
    assert not res.loaded
    assert res.reason == "no live frame"
    assert not res.has_strip


def test_live_without_detection_has_no_roi():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, None, _reference(base))
    assert res.loaded
    assert not res.roi_valid
    assert res.strip is None
    assert not res.valid


def test_live_without_a_baseline_strip_is_reported():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _detection(), None)
    assert res.strip is not None
    assert res.reason == "no Pentacam baseline strip"
    assert not res.valid


def test_live_reports_no_strip_when_detection_is_blind():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _no_detection(), _reference(base))
    assert res.strip is None
    assert not res.roi_valid
    assert not res.valid


def test_live_handles_a_none_image():
    res = dm.analyse_live_frame(None, _detection(), _reference(_synthetic_iris()))
    assert res.reason == "no live frame"
    assert res.angle_deg == 0.0


def test_live_coerces_an_unknown_eye(caplog):
    base = _synthetic_iris()
    with caplog.at_level("WARNING"):
        res = dm.analyse_live_frame(base, _detection(), _reference(base), eye="XX")
    assert res.eye == "OD"
    assert any("unknown eye" in r.message for r in caplog.records)


def test_live_respects_the_max_angle_gate():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(
        _rotate(base, -25.0), _detection(), _reference(base),
        max_abs_angle_deg=10.0,
    )
    assert res.valid
    assert not res.within_limit


def test_live_to_dict_is_json_safe_and_omits_the_strip():
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _detection(), _reference(base))
    payload = json.loads(json.dumps(res.to_dict()))
    assert payload["schema"] == dm.SCHEMA
    assert payload["strip"]["available"] is True
    assert payload["cyclotorsion"]["valid"] is True
    assert "array" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# saving
# ---------------------------------------------------------------------------

def test_default_history_path_is_eye_scoped(tmp_path: Path):
    p = dm.default_history_path(tmp_path, "OS")
    assert p == tmp_path / "OS" / "Cyclotorsion" / "dynamic_cyclotorsion.json"


def test_save_then_load_history(tmp_path: Path):
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _detection(), _reference(base), frame_number=3)
    path = dm.save_sample(res, tmp_path)
    assert path.is_file()
    payload = dm.load_history(tmp_path, "OD")
    assert payload["schema"] == dm.HISTORY_SCHEMA
    assert payload["latest"]["cyclotorsion"]["angle_deg"] == pytest.approx(
        res.angle_deg, abs=1e-3
    )
    assert len(payload["records"]) == 1
    # a second eye is kept separate
    assert dm.load_history(tmp_path, "OS") == {}


def test_save_appends_records(tmp_path: Path):
    base = _synthetic_iris()
    ref = _reference(base)
    for i in range(3):
        res = dm.analyse_live_frame(base, _detection(), ref, frame_number=i)
        dm.save_sample(res, tmp_path)
    payload = dm.load_history(tmp_path, "OD")
    assert len(payload["records"]) == 3


def test_save_as_writes_the_requested_file(tmp_path: Path):
    base = _synthetic_iris()
    res = dm.analyse_live_frame(base, _detection(), _reference(base))
    target = tmp_path / "explicit.json"
    dm.save_sample(res, tmp_path, path=target)
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8"))["schema"] == dm.HISTORY_SCHEMA


def test_load_history_survives_a_corrupt_file(tmp_path: Path):
    res = dm.analyse_live_frame(_synthetic_iris(), _detection(), _reference(_synthetic_iris()))
    path = dm.save_sample(res, tmp_path)
    path.write_text("{not json", encoding="utf-8")
    assert dm.load_history(tmp_path, "OD") == {}


def test_load_history_is_empty_when_absent(tmp_path: Path):
    assert dm.load_history(tmp_path, "OD") == {}


# ---------------------------------------------------------------------------
# module surface
# ---------------------------------------------------------------------------

def test_constants_are_exported():
    assert dm.DEFAULT_ANGLE_RANGE_DEG == 180.0
    assert dm.DEFAULT_MAX_ABS_ANGLE_DEG == 30.0
    assert dm.SCHEMA and dm.HISTORY_SCHEMA


def test_all_is_consistent():
    for name in dm.__all__:
        assert hasattr(dm, name)
