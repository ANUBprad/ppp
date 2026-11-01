"""
pentacam_loader.py — Pentacam image import, auto-flip, and OD/OS routing.

Responsibilities:
- Load Pentacam images from local disk (JPG, PNG, PDF) or DICOM files.
- Auto-flip the image 180° (upside down + right-to-left mirror correction).
- Route the image into the correct OD or OS patient subfolder.
- Provide a preview-ready numpy array for the Add PX dialog.

Expected Pentacam image resolution: 1200 x 838 px.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PENTACAM_WIDTH  = 1200
PENTACAM_HEIGHT = 838

# Supported local file extensions
_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".dcm", ".pdf"}

#: Extensions accepted by Pentacam file dialogs (covers BMP, uppercase, and all loader formats)
PENTACAM_FILETYPES: Tuple[Tuple[str, str], ...] = (
    (
        "Pentacam / Image files",
        "*.bmp *.BMP *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.tif *.tiff *.TIF *.TIFF *.pdf *.PDF *.dcm *.DCM",
    ),
    ("BMP files (*.bmp; *.BMP)", "*.bmp *.BMP"),
    ("JPEG files (*.jpg; *.jpeg)", "*.jpg *.jpeg *.JPG *.JPEG"),
    ("PNG files (*.png)", "*.png *.PNG"),
    ("TIFF files (*.tif; *.tiff)", "*.tif *.tiff *.TIF *.TIFF"),
    ("DICOM files (*.dcm)", "*.dcm *.DCM"),
    ("PDF files (*.pdf)", "*.pdf *.PDF"),
    ("All files", "*.*"),
)


# ---------------------------------------------------------------------------
# Robust Image Loading Helper
# ---------------------------------------------------------------------------

def _read_image_robust(path_obj: Path) -> Optional[np.ndarray]:
    """Read an image file robustly across platforms (especially Windows).

    cv2.imread() fails on Windows for paths containing non-ASCII characters.
    This helper reads the raw bytes via Python's UTF-8-aware file I/O,
    decodes it using cv2.imdecode(), and falls back to cv2.imread and PIL
    if needed.  Returns a BGR numpy array or None if decoding fails.
    """
    # 1. Primary method: read bytes via Python I/O, decode in memory (safe for non-ASCII & spaces)
    try:
        raw_bytes = path_obj.read_bytes()
        if raw_bytes:
            buf = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    # 2. cv2.imread fallback
    try:
        img = cv2.imread(str(path_obj), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 3. Pillow fallback: handles unusual BMP variants (e.g. RLE, 16-bit, OS/2) or TIFFs
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path_obj) as pil_img:
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pentacam_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load a Pentacam image from *path* and return a BGR numpy array.

    Handles:
    - Standard image files (BMP, JPG, PNG, TIFF, etc.).
    - Case-insensitive extensions (e.g. .bmp, .BMP, .jpg, .JPG).
    - Unicode/non-ASCII characters and spaces in file paths on Windows.
    - Paths with surrounding quotes or whitespace.
    - DICOM files (.dcm) when pydicom is available.
    - PDF files: extracts the first page as an image when pdf2image is available.

    Returns None if the file cannot be read or decoded.
    """
    if path is None:
        return None
    clean_path = str(path).strip().strip('"').strip("'")
    if not clean_path:
        return None

    p = Path(clean_path)
    if not p.is_file():
        return None

    ext = p.suffix.lower()

    if ext == ".dcm":
        return _load_dicom(str(p))

    if ext == ".pdf":
        return _load_pdf(str(p))

    return _read_image_robust(p)


def flip_pentacam(image: np.ndarray) -> np.ndarray:
    """Flip a Pentacam image 180° (upside down **and** right-to-left).

    Pentacam captures in a sitting position; the image must be flipped
    before pupil/limbus detection so anatomical orientation matches the
    live camera feed.  This combines the vertical (upside-down) flip with
    a horizontal right-to-left mirror — equivalent to a 180° rotation —
    implemented as ``cv2.flip(image, -1)``.
    """
    return cv2.flip(image, -1)


def prepare_for_detection(
    path: str,
    flip: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load, optionally flip, and return (original, display_ready) arrays.

    Parameters
    ----------
    path:
        Absolute path to the Pentacam image file.
    flip:
        When True (default) the image is flipped 180° (upside down and
        right-to-left) before detection.  Pass False to skip the flip
        (e.g. for debugging).

    Returns
    -------
    (original, processed)
        *original* — raw loaded image (BGR).
        *processed* — flipped image ready for detection (BGR).
        Both are None if the file cannot be read.
    """
    original = load_pentacam_image(path)
    if original is None:
        return None, None
    processed = flip_pentacam(original) if flip else original.copy()
    return original, processed


def copy_to_patient_folder(
    src_path: str,
    patient_dir: Path,
    eye: str,
) -> Optional[Path]:
    """Copy a Pentacam image into the patient's OD or OS Pentacam subfolder.

    Parameters
    ----------
    src_path:
        Source file path.
    patient_dir:
        Root patient directory (e.g. ``PX_001_Patient_001/``).
    eye:
        ``"OD"`` or ``"OS"``.

    Returns
    -------
    Destination Path on success, None on failure.
    """
    import shutil

    if src_path is None:
        return None
    clean_src = str(src_path).strip().strip('"').strip("'")
    if not clean_src:
        return None

    src = Path(clean_src)
    if not src.is_file():
        return None

    dest_dir = patient_dir / eye / "Pentacam"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    try:
        shutil.copy2(src, dest)
        return dest
    except OSError:
        return None


def resize_for_preview(
    image: np.ndarray,
    max_width: int = 560,
    max_height: int = 320,
) -> np.ndarray:
    """Resize *image* to fit within *max_width* × *max_height*, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image.copy()


# ---------------------------------------------------------------------------
# Iris ROI helpers
#
# An iris ROI is stored as four *normalised* coordinates (0..1) in
# flipped-image space: (x0, y0, x1, y1) with x0 < x1 and y0 < y1.  Storing
# normalised coordinates keeps the ROI valid regardless of the preview
# scale used in the Add PX dialog.  This is purely metadata for the
# (additive) cyclotorsion pipeline — it never alters the centration /
# pupil / limbus detection path.
# ---------------------------------------------------------------------------

def format_roi_norm(roi_norm: Tuple[float, float, float, float]) -> str:
    """Serialise a normalised ROI tuple to ``x0,y0,x1,y1`` (6 dp)."""
    x0, y0, x1, y1 = roi_norm
    return f"{x0:.6f},{y0:.6f},{x1:.6f},{y1:.6f}"


def parse_roi_norm(value: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a ``x0,y0,x1,y1`` normalised ROI string.

    Returns None when the value is malformed or degenerate (x1 < x0 or
    y1 < y0).  The coordinates are clamped to the 0..1 range.
    """
    try:
        parts = [float(part) for part in value.strip().split(",")]
    except (AttributeError, TypeError, ValueError):
        return None
    if len(parts) != 4:
        return None

    def _clamp(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    x0, y0, x1, y1 = (_clamp(p) for p in parts)
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def roi_norm_to_px(
    roi_norm: Tuple[float, float, float, float],
    width: int = PENTACAM_WIDTH,
    height: int = PENTACAM_HEIGHT,
) -> Tuple[int, int, int, int]:
    """Convert a normalised ROI to integer pixel (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = roi_norm
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


#: Assumed pupil / limbus size ratio when only a bounding *box* ROI is known
#: (the Add PX dialog draws a single box, not separate pupil + limbus
#: ellipses).  ~0.40 is a typical photopic pupil/limbus diameter ratio.
DEFAULT_PUPIL_LIMBUS_RATIO = 0.40


def roi_norm_to_iris_roi(
    roi_norm: Tuple[float, float, float, float],
    width: int = PENTACAM_WIDTH,
    height: int = PENTACAM_HEIGHT,
    *,
    pupil_ratio: float = DEFAULT_PUPIL_LIMBUS_RATIO,
    inner_inset_frac: float = 0.05,
    outer_inset_frac: float = 0.05,
):
    """Approximate an :class:`~pupil_tracking.iris.types.IrisROI` from a box.

    The Add PX dialog records the iris ROI as a single normalised *box*
    ``(x0, y0, x1, y1)`` in flipped-image space.  The cyclotorsion unwrap
    needs a full pupil/limbus annulus instead, so here the box is treated as
    the **limbus** bounding box and a concentric **pupil** ellipse is assumed
    with radius ``pupil_ratio`` of the limbus.

    This is a deliberate, clearly-labelled *approximation* used only by the
    additive, display-only cyclotorsion readout.  It never feeds the
    centration / pupil / limbus detection path and makes no clinical claim.

    ``IrisROI`` is imported lazily so that importing this module stays
    lightweight (no torch / heavy iris package load).

    Returns an ``IrisROI`` with ``valid=False`` (and a ``reason``) for
    malformed input; it never raises.
    """
    from pupil_tracking.iris.types import IrisROI

    try:
        x0, y0, x1, y1 = (float(v) for v in roi_norm)
    except (TypeError, ValueError):
        return IrisROI(valid=False, reason="malformed roi")

    def _clamp(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    x0, y0, x1, y1 = _clamp(x0), _clamp(y0), _clamp(x1), _clamp(y1)
    if x1 <= x0 or y1 <= y0:
        return IrisROI(valid=False, reason="degenerate roi box")
    if width <= 0 or height <= 0:
        return IrisROI(valid=False, reason="invalid image size")

    cx = (x0 + x1) * 0.5 * width
    cy = (y0 + y1) * 0.5 * height
    limbus_semi_major = (x1 - x0) * 0.5 * width
    limbus_semi_minor = (y1 - y0) * 0.5 * height
    if limbus_semi_major <= 0.0 or limbus_semi_minor <= 0.0:
        return IrisROI(valid=False, reason="degenerate limbus")

    ratio = min(max(float(pupil_ratio), 0.05), 0.95)
    pupil_semi_major = limbus_semi_major * ratio
    pupil_semi_minor = limbus_semi_minor * ratio

    return IrisROI(
        valid=True,
        reason="ok (approximate box ROI)",
        center_x=float(cx),
        center_y=float(cy),
        pupil_semi_major=float(pupil_semi_major),
        pupil_semi_minor=float(pupil_semi_minor),
        pupil_angle_deg=0.0,
        limbus_semi_major=float(limbus_semi_major),
        limbus_semi_minor=float(limbus_semi_minor),
        limbus_angle_deg=0.0,
        inner_inset_frac=float(inner_inset_frac),
        outer_inset_frac=float(outer_inset_frac),
        pupil_radius_px=float((pupil_semi_major + pupil_semi_minor) * 0.5),
        limbus_radius_px=float((limbus_semi_major + limbus_semi_minor) * 0.5),
    )


# ---------------------------------------------------------------------------
# DICOM support (optional — requires pydicom)
# ---------------------------------------------------------------------------

def _load_dicom(path: str) -> Optional[np.ndarray]:
    """Extract pixel data from a DICOM file and return a BGR numpy array."""
    try:
        import pydicom  # type: ignore
    except ImportError:
        return None

    try:
        ds = pydicom.dcmread(path)
        pixel_array = ds.pixel_array  # shape: (H, W) or (H, W, 3)

        if pixel_array.ndim == 2:
            # Grayscale — convert to BGR
            norm = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
            gray = norm.astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if pixel_array.ndim == 3:
            # RGB from DICOM → BGR for OpenCV
            rgb = pixel_array.astype(np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# PDF support (optional — requires pdf2image + poppler)
# ---------------------------------------------------------------------------

def _load_pdf(path: str) -> Optional[np.ndarray]:
    """Rasterise the first page of a PDF and return a BGR numpy array."""
    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        return None

    try:
        pages = convert_from_path(path, dpi=150, first_page=1, last_page=1)
        if not pages:
            return None
        pil_img = pages[0]
        rgb = np.array(pil_img, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None
