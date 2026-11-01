"""Tests for the Static Mode (image-based detection) analysis core.

``pupil_tracking.cyclotorsion.static_mode`` is the pure-logic half of the
Static Mode window: it runs the same read-only detection the rest of the app
uses, unwraps the iris annulus of a Pentacam reference and an ELITA target,
extracts iris features, estimates the cyclotorsion between them, validates the
result and saves the pre-/post-dock details into the patient folder.

The tests are deterministic and synthetic.  Two stubs keep the ML out:

* :class:`_StubDetector` / :class:`_BlindDetector` stand in for
  ``UnifiedDetector`` — the real model does not detect pupils in hand-drawn
  synthetic eyes, so real-image verification lives in the clinical smoke
  scripts instead.
* :class:`_StubIrisDetector` stands in for ``iris.detect.IrisFeatureDetector``
  so no torch import or model load happens here.

Nothing in this module touches the live centration / pupil / limbus pipeline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion import static_mode as sm
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
# Stub detectors
# ---------------------------------------------------------------------------

class _StubDetector:
    """Deterministic stand-in for ``UnifiedDetector``.

    Always reports the same pupil/limbus ellipses, which match the geometry of
    :func:`_synthetic_iris` (pupil radius 70, limbus radius 175).
    """

    def __init__(
        self,
        *,
        pupil=(CENTRE, CENTRE, PUPIL_R),
        limbus=(CENTRE, CENTRE, LIMBUS_R),
        pupil_conf: float = 0.93,
        limbus_conf: float = 0.87,
    ) -> None:
        self.calls = 0
        self._pupil = pupil
        self._limbus = limbus
        self._pupil_conf = pupil_conf
        self._limbus_conf = limbus_conf

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        self.calls += 1
        px, py, pr = self._pupil
        lx, ly, lr = self._limbus
        return EyeDetectionResult(
            pupil=PupilDetection(
                detected=True,
                ellipse=EllipseParams(
                    center_x=float(px), center_y=float(py),
                    semi_major=float(pr), semi_minor=float(pr) * 0.96,
                    angle_deg=0.0,
                ),
                confidence=self._pupil_conf,
            ),
            limbus=LimbusDetection(
                detected=True,
                ellipse=EllipseParams(
                    center_x=float(lx), center_y=float(ly),
                    semi_major=float(lr), semi_minor=float(lr) * 0.97,
                    angle_deg=0.0,
                ),
                confidence=self._limbus_conf,
            ),
            overall_quality=DetectionQuality.RESEARCH,
            overall_confidence=(self._pupil_conf + self._limbus_conf) / 2.0,
        )


class _BlindDetector:
    """Reports a clean no-detection (every field has a safe default)."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        self.calls += 1
        return EyeDetectionResult(
            pupil=PupilDetection(detected=False, confidence=0.1),
            limbus=LimbusDetection(detected=False, confidence=0.1),
            overall_quality=DetectionQuality.INSUFFICIENT,
            overall_confidence=0.1,
        )


class _DetectsReferenceOnly:
    """Detects the reference (first call) and goes blind on the target."""

    def __init__(self) -> None:
        self._good = _StubDetector()
        self._blind = _BlindDetector()
        self.calls = 0

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        self.calls += 1
        if self.calls == 1:
            return self._good.detect(image)
        return self._blind.detect(image)


class _DetectsTargetOnly:
    """The mirror of :class:`_DetectsReferenceOnly`."""

    def __init__(self) -> None:
        self._good = _StubDetector()
        self._blind = _BlindDetector()
        self.calls = 0

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        self.calls += 1
        if self.calls == 1:
            return self._blind.detect(image)
        return self._good.detect(image)


class _BoomDetector:
    """Raises on detection, so the soft-failure path is exercised."""

    def detect(self, image, frame_number=-1, source="", force_mode=None):
        raise RuntimeError("model exploded")


class _StubIrisDetector:
    """Stand-in for ``IrisFeatureDetector`` — no torch, no model load."""

    def __init__(self, n: int = 8, confidence: float = 0.61) -> None:
        self.n = n
        self.confidence = confidence
        self.calls = 0

    def detect(self, image, pupil, limbus):
        self.calls += 1
        feats = [SimpleNamespace(confidence=self.confidence) for _ in range(self.n)]
        return SimpleNamespace(
            feature_set=SimpleNamespace(
                features=feats,
                num_candidates=self.n * 2,
                region_coverage=0.31,
                usable_fraction=0.84,
            ),
            status=SimpleNamespace(value="OK"),
        )


def _analyse_pair(reference=None, target=None, **kw):
    """Run a full pair analysis with both stubs wired in by default.

    ``flip_pentacam`` defaults to **False** here: these synthetic tests compare
    an image against itself or against a rotation of itself, and the Pentacam
    auto-flip would only add a constant half-turn.  The flip itself is covered
    by :func:`test_the_pentacam_flip_makes_a_shared_image_read_as_half_a_turn`.
    """
    reference = _synthetic_iris() if reference is None else reference
    target = reference if target is None else target
    kw.setdefault("detector", _StubDetector())
    kw.setdefault("iris_detector", _StubIrisDetector())
    kw.setdefault("flip_pentacam", False)
    return sm.analyse_static_pair(reference, target, **kw)


# ---------------------------------------------------------------------------
# Constants / conventions
# ---------------------------------------------------------------------------

def test_constants_match_the_documented_convention():
    assert sm.DEFAULT_OUTPUT_SIZE == (64, 512)
    # column 0 = image angle 0°, columns advance monotonically to 360°
    assert sm.DEFAULT_ANGLE_OFFSET_DEG == 0.0
    assert sm.DEFAULT_MIN_FEATURES == 5
    assert sm.DEFAULT_MIN_SCORE == 0.25
    assert sm.MAX_ABS_ANGLE_DEG == 30.0


def test_phases_and_eyes_are_constrained():
    assert sm.PHASES == ("pre_dock", "post_dock")
    assert sm.PHASE_LABELS["pre_dock"] == "Pre-dock"
    assert sm.PHASE_LABELS["post_dock"] == "Post-dock"
    assert sm.VALID_EYES == ("OD", "OS")


def test_image_filetypes_cover_the_required_formats():
    listed = " ".join(pattern for _label, pattern in sm.IMAGE_FILETYPES).lower()
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.pdf"):
        assert ext in listed


# ---------------------------------------------------------------------------
# analyse_source — detection, ROI, unwrap
# ---------------------------------------------------------------------------

def test_analyse_source_detects_roi_and_unwraps():
    src = sm.analyse_source(
        _synthetic_iris(), label="Pentacam", detector=_StubDetector(),
        iris_detector=_StubIrisDetector(),
    )
    assert src.loaded
    assert src.detected
    assert src.pupil_confidence == pytest.approx(0.93)
    assert src.limbus_confidence == pytest.approx(0.87)
    assert src.overall_confidence == pytest.approx(0.90)
    assert src.roi_valid
    assert src.roi_source == "detected"
    assert src.pupil_radius_px == pytest.approx(PUPIL_R, rel=0.05)
    assert src.limbus_radius_px == pytest.approx(LIMBUS_R, rel=0.05)
    assert src.strip is not None
    assert src.strip_size == sm.DEFAULT_OUTPUT_SIZE
    assert src.reason == "ok"
    assert src.ok


def test_analyse_source_records_detector_and_strip_features():
    src = sm.analyse_source(
        _synthetic_iris(), label="Elita", detector=_StubDetector(),
        iris_detector=_StubIrisDetector(n=8, confidence=0.61),
    )
    assert src.iris_features == 8
    assert src.iris_feature_confidence == pytest.approx(0.61)
    assert src.iris_candidates == 16
    assert src.iris_feature_coverage == pytest.approx(0.31)
    assert src.iris_usable_fraction == pytest.approx(0.84)
    assert src.iris_status == "OK"
    # the strip features come from the real extractor on real texture
    assert src.strip_features >= sm.DEFAULT_MIN_FEATURES
    assert 0.0 <= src.strip_feature_confidence <= 1.0
    assert len(src.strip_feature_list) == src.strip_features


def test_analyse_source_applies_the_pentacam_flip():
    img = _synthetic_iris()
    flipped = sm.analyse_source(
        img, label="Pentacam", detector=_StubDetector(), flip=True
    )
    plain = sm.analyse_source(
        img, label="Pentacam", detector=_StubDetector(), flip=False
    )
    assert flipped.flipped is True
    assert plain.flipped is False
    assert flipped.strip is not None and plain.strip is not None
    # a 180° rotation about the iris centre is exactly a half-width cyclic
    # roll of the strip, because the annulus samples are angle-indexed
    rolled = np.roll(plain.strip, plain.strip.shape[1] // 2, axis=1)
    assert float(np.mean(cv2.absdiff(flipped.strip, rolled))) < 6.0


def test_analyse_source_accepts_the_recorded_roi_box_as_fallback():
    src = sm.analyse_source(
        _synthetic_iris(), label="Pentacam", detector=_BlindDetector(),
        roi_norm=(0.1, 0.1, 0.9, 0.9),
    )
    assert not src.detected
    assert src.roi_valid
    assert src.roi_source == "recorded box"
    assert src.strip is not None
    assert src.ok


def test_analyse_source_without_any_roi_leaves_no_strip():
    src = sm.analyse_source(
        _synthetic_iris(), label="Elita", detector=_BlindDetector()
    )
    assert not src.detected
    assert not src.roi_valid
    assert src.roi_reason  # explains *why* the annulus is unusable
    assert src.strip is None
    assert not src.ok
    assert src.reason == "no usable iris strip"


def test_analyse_source_tolerates_a_detector_that_raises():
    src = sm.analyse_source(
        _synthetic_iris(), label="Elita", detector=_BoomDetector()
    )
    assert src.error.startswith("detection failed")
    assert not src.detected
    assert src.strip is None
    assert not src.ok


def test_analyse_source_handles_a_none_image():
    src = sm.analyse_source(None, label="Pentacam", detector=_StubDetector())
    assert not src.loaded
    assert src.strip is None
    assert not src.ok


def test_analyse_source_keeps_the_iris_annulus_uncropped():
    """The whole pupil→limbus annulus must survive the unwrap."""
    src = sm.analyse_source(
        _synthetic_iris(), label="Elita", detector=_StubDetector()
    )
    assert src.strip is not None
    assert src.strip_height == sm.DEFAULT_OUTPUT_SIZE[0]
    assert src.strip_width == sm.DEFAULT_OUTPUT_SIZE[1]
    # the annulus is brighter than the dark background outside the limbus,
    # so a cropped/collapsed strip would lose this contrast entirely
    assert float(src.strip.mean()) > 40.0


# ---------------------------------------------------------------------------
# analyse_static_pair — cyclotorsion
# ---------------------------------------------------------------------------

def test_pair_of_identical_images_reports_zero_cyclotorsion():
    res = _analyse_pair()
    assert res.angle_deg == pytest.approx(0.0, abs=0.5)
    assert res.score == pytest.approx(1.0, abs=1e-3)
    assert res.cyclotorsion_valid
    assert res.n_total > 0
    assert res.n_valid == res.n_total


def test_the_pentacam_flip_makes_a_shared_image_read_as_half_a_turn():
    """The Pentacam reference is auto-flipped, so handing the pipeline the very
    same still twice genuinely lands the two strips half a turn apart — and the
    plausibility gate must then refuse the estimate rather than report 180° of
    cyclotorsion.
    """
    img = _synthetic_iris()
    # flip_pentacam is explicit here: the helper defaults it to False.
    res = _analyse_pair(img, img, flip_pentacam=True)
    assert abs(abs(res.angle_deg) - 180.0) < 2.0
    # the *engine* is right — the two strips really are half a turn apart — but
    # the plausibility gate refuses to call that clinical cyclotorsion
    assert res.cyclotorsion_valid
    assert not res.valid
    failed = [c.name for c in res.checks if not c.passed]
    assert failed == [f"|angle| ≤ {sm.MAX_ABS_ANGLE_DEG:g}°"]


def test_pair_recovers_a_known_rotation():
    base = _synthetic_iris()
    # The reference is auto-flipped by the pipeline, so the target is built
    # from the *flipped* reference to keep the comparison purely rotational.
    target = _rotate(cv2.flip(base, -1), -12.0)
    res = _analyse_pair(
        base, target,
        eye="OD", phase=sm.PHASE_POST_DOCK,
        flip_pentacam=True, flip_elita=False,
    )
    assert res.cyclotorsion_valid
    assert res.score >= sm.DEFAULT_MIN_SCORE
    # rotating the target one way reads as the opposite sign on the reference
    assert res.angle_deg == pytest.approx(12.0, abs=3.0)


def test_pair_sign_is_antisymmetric():
    base = cv2.flip(_synthetic_iris(), -1)
    plus = _analyse_pair(
        _synthetic_iris(), _rotate(base, 9.0), flip_pentacam=True
    )
    minus = _analyse_pair(
        _synthetic_iris(), _rotate(base, -9.0), flip_pentacam=True
    )
    assert plus.angle_deg < 0
    assert minus.angle_deg > 0


def test_pair_reports_the_missing_elita_strip():
    res = _analyse_pair(detector=_DetectsReferenceOnly())
    assert res.reference.strip is not None
    assert res.target.strip is None
    assert res.cyclotorsion_reason == "missing Elita strip"
    assert not res.cyclotorsion_valid
    assert res.n_total == 0


def test_pair_reports_the_missing_pentacam_strip():
    res = _analyse_pair(detector=_DetectsTargetOnly())
    assert res.reference.strip is None
    assert res.target.strip is not None
    assert res.cyclotorsion_reason == "missing Pentacam strip"
    assert not res.cyclotorsion_valid


def test_pair_reports_when_both_strips_are_missing():
    res = _analyse_pair(detector=_BlindDetector())
    assert res.reference.strip is None
    assert res.target.strip is None
    assert res.cyclotorsion_reason == "missing both iris strips"
    assert not res.cyclotorsion_valid
    assert res.n_total == 0


def test_pair_gives_the_elita_its_own_roi():
    """The recorded ROI box is a Pentacam fallback only — never the target's."""
    res = _analyse_pair(
        _synthetic_iris(), _synthetic_iris(),
        detector=_BlindDetector(), roi_norm=(0.1, 0.1, 0.9, 0.9),
    )
    assert res.reference.roi_source == "recorded box"
    assert res.target.roi_source == ""
    assert not res.target.roi_valid
    assert res.target.strip is None


def test_pair_carries_eye_phase_and_timing():
    res = _analyse_pair(eye="OS", phase=sm.PHASE_POST_DOCK)
    assert res.eye == "OS"
    assert res.phase == sm.PHASE_POST_DOCK
    assert res.elapsed_ms >= 0.0
    assert res.timestamp


def test_pair_coerces_an_unknown_eye_to_od(caplog):
    """The window only ever offers OD/OS, so anything else falls back."""
    with caplog.at_level("WARNING"):
        res = _analyse_pair(eye="XX")
    assert res.eye == "OD"
    assert any("unknown eye" in r.message for r in caplog.records)


def test_pair_coerces_an_unknown_phase_to_pre_dock(caplog):
    with caplog.at_level("WARNING"):
        res = _analyse_pair(phase="mid_dock")
    assert res.phase == sm.PHASE_PRE_DOCK
    assert any("unknown phase" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# validate_analysis — the checklist
# ---------------------------------------------------------------------------

def test_validation_produces_the_documented_checklist():
    res = _analyse_pair()
    names = [c.name for c in res.checks]
    assert len(res.checks) == 16
    for label in ("Pentacam", "Elita"):
        assert f"{label} image loaded" in names
        assert f"{label} pupil + limbus detected" in names
        assert f"{label} iris ROI valid" in names
        assert f"{label} iris annulus unwrapped" in names
        assert f"{label} iris features ≥ {sm.DEFAULT_MIN_FEATURES}" in names
        assert f"{label} strip features ≥ {sm.DEFAULT_MIN_FEATURES}" in names
    assert "Cyclotorsion estimated" in names
    assert f"Correlation score ≥ {sm.DEFAULT_MIN_SCORE:.2f}" in names
    assert "Engine accepted the estimate" in names
    assert f"|angle| ≤ {sm.MAX_ABS_ANGLE_DEG:g}°" in names


def test_validation_passes_on_a_clean_pair():
    res = _analyse_pair()
    assert res.passed_count() == (16, 16)
    assert res.valid
    assert res.summary() == "16/16 checks passed"


def test_validation_flags_an_implausible_angle():
    res = _analyse_pair()
    res.angle_deg = 41.0
    res.checks = sm.validate_analysis(res)
    failing = [c.name for c in res.checks if not c.passed]
    assert failing == [f"|angle| ≤ {sm.MAX_ABS_ANGLE_DEG:g}°"]
    assert not res.valid
    assert res.passed_count() == (15, 16)


def test_validation_flags_a_low_correlation_score():
    res = _analyse_pair()
    res.score = 0.11
    res.checks = sm.validate_analysis(res)
    failing = [c.name for c in res.checks if not c.passed]
    assert f"Correlation score ≥ {sm.DEFAULT_MIN_SCORE:.2f}" in failing


def test_validation_flags_a_rejected_engine_estimate():
    res = _analyse_pair()
    res.cyclotorsion_valid = False
    res.cyclotorsion_reason = "score below threshold"
    res.checks = sm.validate_analysis(res)
    rejected = [
        c for c in res.checks if c.name == "Engine accepted the estimate"
    ]
    assert len(rejected) == 1
    assert not rejected[0].passed
    assert rejected[0].detail == "score below threshold"


def test_validation_flags_undetected_sources():
    """A still that loads fine but yields no geometry must fail downstream."""
    res = _analyse_pair(detector=_BlindDetector())
    checks = {c.name: c for c in sm.validate_analysis(res)}
    # the still itself read successfully…
    assert checks["Pentacam image loaded"].passed
    # …but everything downstream of detection fails
    assert not checks["Pentacam pupil + limbus detected"].passed
    assert not checks["Pentacam iris ROI valid"].passed
    assert not checks["Pentacam iris annulus unwrapped"].passed
    assert not checks["Pentacam iris features ≥ 5"].passed
    assert not checks["Pentacam strip features ≥ 5"].passed
    assert not checks["Cyclotorsion estimated"].passed
    assert not checks["Engine accepted the estimate"].passed
    # the detection detail reports the confidences it did see
    assert "pupil 0.10, limbus 0.10" in checks["Pentacam pupil + limbus detected"].detail


def test_validation_flags_an_unreadable_image():
    res = sm.analyse_static_pair(
        None, _synthetic_iris(),
        detector=_StubDetector(), iris_detector=_StubIrisDetector(),
        flip_pentacam=False,
    )
    checks = {c.name: c for c in sm.validate_analysis(res)}
    assert not checks["Pentacam image loaded"].passed
    assert checks["Pentacam image loaded"].detail == "no file selected"
    assert checks["Elita image loaded"].passed
    # the target is fine, so its own checks still pass
    assert checks["Elita iris annulus unwrapped"].passed


def test_validation_handles_a_numpy_strip_without_ambiguous_truthiness():
    """Regression: a non-empty strip array must never be tested for truthiness.

    ``if src.strip else ...`` raises ``ValueError: The truth value of an array
    with more than one element is ambiguous`` as soon as a source actually
    produced a strip — i.e. on every real run.
    """
    res = _analyse_pair()
    assert res.reference.strip is not None  # a real, non-empty ndarray
    checks = sm.validate_analysis(res)
    strip_check = [
        c for c in checks if c.name == "Pentacam iris annulus unwrapped"
    ][0]
    assert strip_check.passed
    assert strip_check.detail == f"{res.reference.strip_height}x{res.reference.strip_width} strip"


def test_check_line_and_dict_are_plain_text():
    check = sm.ValidationCheck(
        name="Pentacam image loaded", passed=True, detail="eye_01.jpeg"
    )
    assert check.status == "PASS"
    assert check.line() == "[PASS] Pentacam image loaded — eye_01.jpeg"
    assert check.to_dict() == {
        "name": "Pentacam image loaded",
        "passed": True,
        "status": "PASS",
        "detail": "eye_01.jpeg",
    }
    assert sm.ValidationCheck("x", False).status == "FAIL"


def test_analysis_valid_is_false_before_any_checks_run():
    assert sm.StaticAnalysis().checks == []
    assert not sm.StaticAnalysis().valid
    assert sm.StaticAnalysis().passed_count() == (0, 0)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_is_json_safe_and_omits_the_strip():
    res = _analyse_pair(eye="OS", phase=sm.PHASE_POST_DOCK)
    payload = res.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["eye"] == "OS"
    assert payload["phase"] == "post_dock"
    assert payload["phase_label"] == "Post-dock"
    assert payload["cyclotorsion"]["angle_deg"] == pytest.approx(res.angle_deg, abs=1e-3)
    assert payload["cyclotorsion"]["target"] == "Elita"
    assert payload["validation"]["valid"] is True
    assert payload["validation"]["total"] == 16
    assert payload["thresholds"]["max_abs_angle_deg"] == sm.MAX_ABS_ANGLE_DEG
    # the strip arrays are never serialised — only a size/availability summary
    assert payload["reference"]["strip"] == {
        "size": list(sm.DEFAULT_OUTPUT_SIZE),
        "available": True,
    }


def test_to_dict_carries_both_feature_levels():
    payload = _analyse_pair().reference.to_dict()
    assert payload["iris_features"]["count"] == 8
    assert payload["iris_features"]["confidence"] == pytest.approx(0.61)
    assert payload["strip_features"]["count"] >= sm.DEFAULT_MIN_FEATURES
    # the marker list is in-memory only
    assert "strip_feature_list" not in payload


def test_to_dict_round_trips_through_json():
    payload = _analyse_pair().to_dict()
    again = json.loads(json.dumps(payload))
    assert again == payload


# ---------------------------------------------------------------------------
# Saving pre- and post-dock details
# ---------------------------------------------------------------------------

@pytest.fixture()
def patient_dir(tmp_path: Path) -> Path:
    d = tmp_path / "PX_123_Smoke"
    (d / "OD").mkdir(parents=True)
    (d / "OS").mkdir(parents=True)
    return d


def test_default_history_path_is_eye_scoped(patient_dir):
    assert sm.default_history_path(patient_dir, "OD") == (
        patient_dir / "OD" / "Cyclotorsion" / "static_cyclotorsion.json"
    )
    assert sm.default_history_path(patient_dir, "OS") != (
        sm.default_history_path(patient_dir, "OD")
    )


def test_save_then_load_history(patient_dir):
    res = _analyse_pair(eye="OD", phase=sm.PHASE_PRE_DOCK)
    path = sm.save_analysis(res, patient_dir)
    assert path == sm.default_history_path(patient_dir, "OD")
    assert path.is_file()

    payload = sm.load_history(patient_dir, "OD")
    assert payload["eye"] == "OD"
    assert len(payload["records"]) == 1
    assert sm.PHASE_PRE_DOCK in payload["latest"]
    assert payload["latest"][sm.PHASE_PRE_DOCK]["cyclotorsion"]["angle_deg"] == pytest.approx(
        res.angle_deg, abs=1e-3
    )


def test_saving_both_phases_keeps_history_and_latest(patient_dir):
    pre = _analyse_pair(eye="OD", phase=sm.PHASE_PRE_DOCK)
    post = _analyse_pair(eye="OD", phase=sm.PHASE_POST_DOCK)
    sm.save_analysis(pre, patient_dir)
    sm.save_analysis(post, patient_dir)

    payload = sm.load_history(patient_dir, "OD")
    assert len(payload["records"]) == 2
    assert sorted(payload["latest"]) == [sm.PHASE_POST_DOCK, sm.PHASE_PRE_DOCK]
    for phase in sm.PHASES:
        assert payload["latest"][phase]["phase"] == phase

    latest = sm.latest_records(patient_dir, "OD")
    assert set(latest) == set(sm.PHASES)


def test_history_is_per_eye(patient_dir):
    sm.save_analysis(_analyse_pair(eye="OD"), patient_dir)
    assert sm.load_history(patient_dir, "OS") == {}
    assert sm.latest_records(patient_dir, "OS") == {}


def test_load_history_is_empty_for_a_missing_file(patient_dir):
    assert sm.load_history(patient_dir, "OD") == {}
    assert sm.latest_records(patient_dir, "OD") == {}


def test_load_history_survives_a_corrupt_file(patient_dir):
    path = sm.default_history_path(patient_dir, "OD")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert sm.load_history(patient_dir, "OD") == {}


def test_save_as_writes_the_requested_file(patient_dir):
    res = _analyse_pair(eye="OS", phase=sm.PHASE_PRE_DOCK)
    explicit = patient_dir / "OS" / "Cyclotorsion" / "review_copy.json"
    written = sm.save_analysis(res, patient_dir, path=explicit)
    assert written == explicit
    assert explicit.is_file()
    payload = json.loads(explicit.read_text(encoding="utf-8"))
    assert payload["eye"] == "OS"
    assert len(payload["records"]) == 1


def test_save_creates_missing_folders(patient_dir):
    nested = patient_dir / "OS" / "Cyclotorsion" / "sub"
    sm.save_analysis(_analyse_pair(eye="OS"), patient_dir, path=nested / "x.json")
    assert (nested / "x.json").is_file()


def test_save_as_works_without_an_active_patient(tmp_path):
    """Save As… must work when no patient folder is open (records "")."""
    res = _analyse_pair(eye="OD", phase=sm.PHASE_PRE_DOCK)
    explicit = tmp_path / "Cyclotorsion" / "review_copy.json"
    written = sm.save_analysis(res, "", path=explicit)
    assert written == explicit
    assert explicit.is_file()
    payload = json.loads(explicit.read_text(encoding="utf-8"))
    assert payload["patient_dir"] == ""
    assert payload["eye"] == "OD"
    assert len(payload["records"]) == 1


# ---------------------------------------------------------------------------
# OD/OS routing helpers
# ---------------------------------------------------------------------------

def test_find_reference_image_and_read_roi_norm(patient_dir):
    folder = patient_dir / "OD" / "Pentacam"
    folder.mkdir(parents=True, exist_ok=True)
    still = folder / "eye_01.jpeg"
    still.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (folder / "pentacam_values.txt").write_text(
        "iris_roi_norm=0.115,0.102,0.893,0.921\n", encoding="utf-8"
    )

    assert sm.find_reference_image(patient_dir, "OD") == still
    assert sm.read_roi_norm(patient_dir, "OD") == pytest.approx(
        (0.115, 0.102, 0.893, 0.921)
    )
    # OD/OS routing: the other eye has nothing recorded
    assert sm.find_reference_image(patient_dir, "OS") is None
    assert sm.read_roi_norm(patient_dir, "OS") is None


def test_find_reference_image_ignores_non_images(patient_dir):
    folder = patient_dir / "OD" / "Pentacam"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    assert sm.find_reference_image(patient_dir, "OD") is None


def test_read_roi_norm_is_none_when_malformed(patient_dir):
    folder = patient_dir / "OD" / "Pentacam"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pentacam_values.txt").write_text(
        "iris_roi_norm=not,a,box\n", encoding="utf-8"
    )
    assert sm.read_roi_norm(patient_dir, "OD") is None


def test_read_roi_norm_is_none_when_absent(patient_dir):
    assert sm.read_roi_norm(patient_dir, "OD") is None


# ---------------------------------------------------------------------------
# The window module (wiring only — Tk is skipped when unavailable)
# ---------------------------------------------------------------------------

def _tk_usable() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tk_usable(), reason="Tk unavailable in this environment")
def test_static_mode_window_builds_and_renders(tmp_path):
    import tkinter as tk

    from pupil_tracking.interface import static_mode_window as win_mod
    from pupil_tracking.interface.theme import Colors

    class _FakeApp:
        def __init__(self):
            self._colors = Colors  # DarkTheme.apply() hands back the palette
            self.root = tk.Tk()
            self.root.withdraw()
            self._detector = _StubDetector()
            self._selected_eye_var = tk.StringVar(value="OD")
            self._active_patient_dir = None
            self._static_mode_open = False

        def _update_patient_controls(self):
            pass

    app = _FakeApp()
    window = win_mod.StaticModeWindow(app)
    try:
        window.open()
        assert window._win is not None
        assert window._eye_var.get() == "OD"
        assert window._phase_var.get() == sm.PHASE_PRE_DOCK
        assert set(window._strip_canvases) == {"reference", "target"}

        # apply a real analysis and confirm every panel is populated
        res = _analyse_pair(eye="OD", phase=sm.PHASE_PRE_DOCK)
        window._apply(res)
        assert window._analysis is res
        assert "Cyclotorsion" in window._cyc_var.get()
        assert "VALID" in window._cyc_var.get()
        assert window._readout_cells["strip:reference"].cget("text") == "64 × 512"
        assert window._readout_cells["det:reference"].cget("text") == "0.93 / 0.87"
        body = window._val_text.get("1.0", tk.END)
        assert "[PASS]  Pentacam image loaded" in body
        assert "16/16 checks passed" in window._val_header.get()

        # the strip canvases re-render when the window is resized
        strip = np.zeros((64, 512, 3), np.uint8)
        window._draw_strip("reference", strip, [])
        window._on_strip_resize("reference", 640, 110)
        assert window._strip_drawn["reference"] == (640, 110)
        assert "reference" in window._photos
        window._on_strip_resize("reference", 640, 110)  # unchanged -> no redraw
        assert window._strip_drawn["reference"] == (640, 110)

        # the ELITA target is remembered per (eye, phase)
        window._eye_var.set("OD")
        window._phase_var.set(sm.PHASE_PRE_DOCK)
        window._elita_var.set(r"C:\tmp\od_pre.jpeg")
        window._set_phase(sm.PHASE_POST_DOCK)          # stores (OD, pre_dock)
        assert window._elita_var.get() == ""
        window._elita_var.set(r"C:\tmp\od_post.jpeg")
        window._set_phase(sm.PHASE_PRE_DOCK)
        assert window._elita_var.get() == r"C:\tmp\od_pre.jpeg"
        window._set_phase(sm.PHASE_POST_DOCK)
        assert window._elita_var.get() == r"C:\tmp\od_post.jpeg"

        # OD/OS routing: each eye keeps its own reference and target, so a still
        # is never shown — or saved — under the wrong eye label
        window._pentacam_paths["OD"] = r"C:\tmp\od_pentacam.jpeg"
        window._set_eye("OS")
        assert window._eye_var.get() == "OS"
        assert window._pentacam_var.get() == ""        # OS has no still yet
        assert window._elita_var.get() == ""           # nor an (OS, post) target
        window._pentacam_paths["OS"] = r"C:\tmp\os_pentacam.jpeg"
        window._elita_var.set(r"C:\tmp\os_post.jpeg")
        window._set_eye("OD")
        assert window._pentacam_var.get() == r"C:\tmp\od_pentacam.jpeg"
        assert window._elita_var.get() == r"C:\tmp\od_post.jpeg"
        window._set_eye("OS")
        assert window._pentacam_var.get() == r"C:\tmp\os_pentacam.jpeg"
        assert window._elita_var.get() == r"C:\tmp\os_post.jpeg"

        # cancelling an in-flight run must release Analyse and bump the token,
        # otherwise a superseded worker (which now returns early) would leave
        # the button disabled forever
        window._busy = True
        window._analyse_btn.config(state=tk.DISABLED)
        token = window._token
        window._show_cached_or_clear()
        assert window._token == token + 1
        assert window._busy is False
        assert str(window._analyse_btn.cget("state")) == "normal"

        # the empty-state path must be safe too
        window._analysis = None
        window._show_cached_or_clear()
        assert window._analysis is None
        assert "awaiting analysis" in window._cyc_var.get()
        assert window._last_strip["reference"][0] is None

        # re-opening raises the existing window instead of duplicating it
        first = window._win
        window.open()
        assert window._win is first
    finally:
        window.close()
        try:
            app.root.destroy()
        except tk.TclError:
            pass
    assert app._static_mode_open is False
