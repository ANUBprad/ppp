"""Unit tests for the cyclotorsion Pentacam loader + iris ROI helpers.

Covers Module 1 of the cyclotorsion feature:
- image load / auto-flip / preview resize
- normalised iris ROI serialisation, parsing and pixel conversion

Synthetic data only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pupil_tracking.cyclotorsion.pentacam_loader import (
    PENTACAM_FILETYPES,
    PENTACAM_HEIGHT,
    PENTACAM_WIDTH,
    copy_to_patient_folder,
    flip_pentacam,
    format_roi_norm,
    load_pentacam_image,
    parse_roi_norm,
    resize_for_preview,
    roi_norm_to_px,
)


# ── Dimensions ──────────────────────────────────────────────────────

def test_pentacam_dimensions_are_1200x838():
    assert PENTACAM_WIDTH == 1200
    assert PENTACAM_HEIGHT == 838


# ── Load / flip / resize ────────────────────────────────────────────

def test_load_and_flip_synthetic_image(tmp_path):
    img = np.zeros((PENTACAM_HEIGHT, PENTACAM_WIDTH, 3), dtype=np.uint8)
    img[0:5, 100:1100] = (0, 0, 255)  # red strip on the top edge
    img[100:800, 0:5] = (0, 255, 0)   # green strip on the left edge
    path = tmp_path / "eye.png"
    assert cv2.imwrite(str(path), img)

    loaded = load_pentacam_image(str(path))
    assert loaded is not None
    assert loaded.shape == (PENTACAM_HEIGHT, PENTACAM_WIDTH, 3)
    # Before flipping: red on top, green on the left.
    assert loaded[2, PENTACAM_WIDTH // 2].tolist() == [0, 0, 255]
    assert loaded[PENTACAM_HEIGHT // 2, 2].tolist() == [0, 255, 0]

    flipped = flip_pentacam(loaded)
    # A 180° flip moves the top strip to the bottom and the left strip to
    # the right (vertical upside-down kept + right-to-left mirror).
    assert flipped[PENTACAM_HEIGHT - 3, PENTACAM_WIDTH // 2].tolist() == [0, 0, 255]
    assert flipped[PENTACAM_HEIGHT // 2, PENTACAM_WIDTH - 3].tolist() == [0, 255, 0]


def test_load_missing_file_returns_none():
    assert load_pentacam_image("definitely_missing_file.png") is None


def test_resize_for_preview_keeps_aspect_ratio():
    img = np.zeros((PENTACAM_HEIGHT, PENTACAM_WIDTH, 3), dtype=np.uint8)
    thumb = resize_for_preview(img, max_width=560, max_height=196)
    assert thumb.shape[1] <= 560
    assert thumb.shape[0] <= 196
    # Aspect ratio preserved to within rounding.
    assert abs((thumb.shape[1] / thumb.shape[0]) - (PENTACAM_WIDTH / PENTACAM_HEIGHT)) < 0.02


def test_resize_for_preview_does_not_upscale():
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    out = resize_for_preview(small, max_width=560, max_height=320)
    assert out.shape[:2] == (100, 100)


# ── ROI serialisation ───────────────────────────────────────────────

def test_roi_round_trip():
    roi = (0.1, 0.2, 0.6, 0.75)
    text = format_roi_norm(roi)
    assert text == "0.100000,0.200000,0.600000,0.750000"
    assert parse_roi_norm(text) == pytest.approx(roi)


def test_parse_roi_rejects_malformed():
    assert parse_roi_norm("") is None
    assert parse_roi_norm("nope") is None
    assert parse_roi_norm("0.1,0.2,0.3") is None
    assert parse_roi_norm("0.1,0.2,0.3,0.4,0.5") is None


def test_parse_roi_rejects_degenerate():
    assert parse_roi_norm("0.6,0.2,0.1,0.75") is None  # x1 < x0
    assert parse_roi_norm("0.1,0.75,0.6,0.2") is None  # y1 < y0


def test_parse_roi_clamps_out_of_range():
    parsed = parse_roi_norm("1.5,-0.2,2.0,0.9")
    assert parsed == pytest.approx((1.0, 0.0, 1.0, 0.9))


def test_roi_norm_to_px_matches_pentacam_size():
    px = roi_norm_to_px((0.1, 0.2, 0.6, 0.75))
    assert px == (120, 168, 720, 628)


# ── Regression tests: BMP, uppercase, spaces, unicode, invalid files ──

def test_load_bmp_image(tmp_path):
    """BMP image loading works correctly and preserves BGR pixel values."""
    img = np.zeros((PENTACAM_HEIGHT, PENTACAM_WIDTH, 3), dtype=np.uint8)
    img[10:20, 10:20] = (255, 128, 64)  # BGR
    path = tmp_path / "sample_eye.bmp"
    _, buf = cv2.imencode(".bmp", img)
    path.write_bytes(buf.tobytes())

    loaded = load_pentacam_image(str(path))
    assert loaded is not None
    assert loaded.shape == (PENTACAM_HEIGHT, PENTACAM_WIDTH, 3)
    assert loaded[15, 15].tolist() == [255, 128, 64]


def test_load_uppercase_bmp_extension(tmp_path):
    """Files with uppercase .BMP extensions load correctly."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (50, 100, 150)
    path = tmp_path / "CLINICAL_EYE.BMP"
    _, buf = cv2.imencode(".bmp", img)
    path.write_bytes(buf.tobytes())

    loaded = load_pentacam_image(str(path))
    assert loaded is not None
    assert loaded.shape == (100, 100, 3)
    assert loaded[50, 50].tolist() == [50, 100, 150]


def test_load_uppercase_extensions(tmp_path):
    """Files with uppercase extensions like .JPG and .PNG load properly."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:] = (20, 40, 60)

    jpg_path = tmp_path / "TEST_IMAGE.JPG"
    _, buf_jpg = cv2.imencode(".jpg", img)
    jpg_path.write_bytes(buf_jpg.tobytes())
    loaded_jpg = load_pentacam_image(str(jpg_path))
    assert loaded_jpg is not None
    assert loaded_jpg.shape == (60, 80, 3)

    png_path = tmp_path / "TEST_IMAGE.PNG"
    _, buf_png = cv2.imencode(".png", img)
    png_path.write_bytes(buf_png.tobytes())
    loaded_png = load_pentacam_image(str(png_path))
    assert loaded_png is not None
    assert loaded_png.shape == (60, 80, 3)


def test_load_path_with_spaces_and_quotes(tmp_path):
    """Paths containing spaces and optional surrounding quotes load reliably."""
    folder_with_spaces = tmp_path / "Patient Pentacam Data (OD)"
    folder_with_spaces.mkdir(parents=True, exist_ok=True)
    img_path = folder_with_spaces / "Sitting Eye Scan 01.bmp"

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    _, buf = cv2.imencode(".bmp", img)
    img_path.write_bytes(buf.tobytes())

    # Raw path with spaces
    loaded = load_pentacam_image(str(img_path))
    assert loaded is not None
    assert loaded.shape == (50, 50, 3)

    # Path with leading/trailing spaces and surrounding double quotes
    quoted_path = f'  "{img_path}"  '
    loaded_quoted = load_pentacam_image(quoted_path)
    assert loaded_quoted is not None
    assert loaded_quoted.shape == (50, 50, 3)

    # Path with surrounding single quotes
    single_quoted = f"'{img_path}'"
    loaded_sq = load_pentacam_image(single_quoted)
    assert loaded_sq is not None
    assert loaded_sq.shape == (50, 50, 3)


def test_load_path_with_unicode_characters(tmp_path):
    """Windows non-ASCII / Unicode paths (accents, international names) load properly."""
    unicode_dir = tmp_path / "patient_œil_пациент_№1"
    unicode_dir.mkdir(parents=True, exist_ok=True)
    unicode_path = unicode_dir / "œil_gauche_中文.bmp"

    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:] = (77, 88, 99)
    _, buf = cv2.imencode(".bmp", img)
    unicode_path.write_bytes(buf.tobytes())

    loaded = load_pentacam_image(str(unicode_path))
    assert loaded is not None
    assert loaded.shape == (80, 80, 3)
    assert loaded[0, 0].tolist() == [77, 88, 99]


def test_load_invalid_and_corrupt_files(tmp_path):
    """Invalid, empty, corrupted, and missing files return None safely without error."""
    # 1. Non-existent file
    assert load_pentacam_image(str(tmp_path / "non_existent.bmp")) is None

    # 2. None or empty string
    assert load_pentacam_image(None) is None
    assert load_pentacam_image("") is None
    assert load_pentacam_image("   ") is None
    assert load_pentacam_image('""') is None

    # 3. Empty (0-byte) file with .bmp extension
    empty_file = tmp_path / "empty.bmp"
    empty_file.write_bytes(b"")
    assert load_pentacam_image(str(empty_file)) is None

    # 4. Corrupted / non-image text file with .bmp extension
    corrupt_file = tmp_path / "corrupt.bmp"
    corrupt_file.write_text("This is plain text, not a valid BMP header.")
    assert load_pentacam_image(str(corrupt_file)) is None

    # 5. Truncated header
    truncated_file = tmp_path / "truncated.bmp"
    truncated_file.write_bytes(b"BM\x00\x00")
    assert load_pentacam_image(str(truncated_file)) is None


def test_pentacam_filetypes_cover_all_supported_formats():
    """File dialog filters include BMP, JPG, PNG, and uppercase variants."""
    all_patterns = " ".join(pat for _lbl, pat in PENTACAM_FILETYPES)
    for ext in ("*.bmp", "*.BMP", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        assert ext in all_patterns


def test_auto_flip_with_bmp_image(tmp_path):
    """Auto-flip 180° rotation works correctly on loaded BMP images."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[0, 50] = (0, 0, 255)  # Red pixel at top center
    path = tmp_path / "flip_test.bmp"
    _, buf = cv2.imencode(".bmp", img)
    path.write_bytes(buf.tobytes())

    loaded = load_pentacam_image(str(path))
    assert loaded is not None
    assert loaded[0, 50].tolist() == [0, 0, 255]

    flipped = flip_pentacam(loaded)
    # Flipped 180°: top becomes bottom, center-x stays center-x (99 - 50 = 49 or 50)
    assert flipped[99, 49].tolist() == [0, 0, 255]


def test_copy_to_patient_folder_with_bmp_and_spaces(tmp_path):
    """copy_to_patient_folder copies BMP images from paths with spaces successfully."""
    src_folder = tmp_path / "External Pentacam Export"
    src_folder.mkdir(parents=True, exist_ok=True)
    src_file = src_folder / "Patient Sitting OD.bmp"

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".bmp", img)
    src_file.write_bytes(buf.tobytes())

    patient_dir = tmp_path / "PX_001_Test"
    dest = copy_to_patient_folder(f'"{src_file}"', patient_dir, "OD")
    assert dest is not None
    assert dest.is_file()
    assert dest.name == "Patient Sitting OD.bmp"
    assert dest.parent == patient_dir / "OD" / "Pentacam"

