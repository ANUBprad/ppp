#!/usr/bin/env python3
"""
annotate_ring_data.py — Label images as docked (ring) vs pre-docked (no ring).

Launches a simple OpenCV GUI that shows each image in sequence.
The user presses a single key to classify it and the labels are
accumulated in a JSON file that the ring classifier trainer expects.

Controls
--------
    R   → Label as "ring present" (docked)
    N   → Label as "no ring" (pre-docked / natural eye)
    P   → Label as "partial ring" (ring partially visible / occluded)
    U   → Undo last label
    S   → Save progress to disk immediately
    Q   → Save and quit
    ESC → Save and quit

Label file format (``ring_labels.json``)::

    {
      "image_001.jpg": {"ring_present": true,  "ring_visibility": "full"},
      "image_002.jpg": {"ring_present": false, "ring_visibility": "none"},
      "image_003.jpg": {"ring_present": true,  "ring_visibility": "partial"},
      ...
    }

Usage
-----
::

    # Start fresh
    python scripts/annotate_ring_data.py \\
        --image-dir clinical_data/training_data/images \\
        --output clinical_data/ring_labels.json

    # Resume an earlier session
    python scripts/annotate_ring_data.py \\
        --image-dir clinical_data/training_data/images \\
        --output clinical_data/ring_labels.json \\
        --resume

    # Only show images matching a glob pattern
    python scripts/annotate_ring_data.py \\
        --image-dir clinical_data/training_data/images \\
        --output clinical_data/ring_labels.json \\
        --pattern "docked_*"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ═══════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate eye images as ring-present / ring-absent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Key bindings inside the GUI:\n"
            "  R  = Ring PRESENT (docked)\n"
            "  N  = No ring (pre-docked)\n"
            "  P  = Partially visible ring\n"
            "  U  = Undo last label\n"
            "  S  = Save progress\n"
            "  Q / ESC = Save & quit"
        ),
    )
    p.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing eye images to label",
    )
    p.add_argument(
        "--output", type=str,
        default="clinical_data/ring_labels.json",
        help="Output JSON label file (default: clinical_data/ring_labels.json)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing label file (skip already-labelled images)",
    )
    p.add_argument(
        "--pattern", type=str, default=None,
        help="Optional glob pattern to filter filenames (e.g. 'docked_*')",
    )
    p.add_argument(
        "--max-display", type=int, default=900,
        help="Maximum display window dimension in pixels (default: 900)",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def discover_images(
    image_dir: Path,
    pattern: Optional[str] = None,
) -> List[Path]:
    """Return sorted list of image paths, optionally filtered by glob."""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if pattern is not None:
        images = [p for p in images if fnmatch.fnmatch(p.name, pattern)]
    return images


def load_existing_labels(path: Path) -> Dict[str, dict]:
    """Load labels from JSON file if it exists, else return empty dict."""
    if path.exists():
        try:
            with open(path, "r") as f:
                data = json.load(f)
            logger.info("Loaded %d existing labels from %s", len(data), path)
            return data
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not parse %s: %s — starting fresh", path, exc)
    return {}


def save_labels(labels: Dict[str, dict], path: Path) -> None:
    """Atomically write labels to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    tmp.replace(path)  # atomic on most OSes


def make_display_image(
    image: np.ndarray,
    filename: str,
    idx: int,
    total: int,
    max_dim: int,
) -> np.ndarray:
    """
    Resize image for display and add informational text overlay.

    Returns a copy — the original is not modified.
    """
    display = image.copy()
    h, w = display.shape[:2]

    # Scale down if necessary
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        display = cv2.resize(
            display, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    dh, dw = display.shape[:2]

    # Semi-transparent header bar
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (dw, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # File info
    info_text = f"[{idx + 1}/{total}]  {filename}"
    cv2.putText(
        display, info_text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
    )

    # Controls reminder
    controls = "R=Ring  N=No-ring  P=Partial  U=Undo  S=Save  Q=Quit"
    cv2.putText(
        display, controls, (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
    )

    # Bottom bar with image dimensions
    dim_text = f"{w}x{h}"
    cv2.putText(
        display, dim_text, (dw - 100, dh - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
    )

    return display


def show_confirmation(
    display: np.ndarray,
    label_text: str,
    colour: tuple,
) -> np.ndarray:
    """Flash a confirmation label on the display image."""
    confirm = display.copy()
    dh, dw = confirm.shape[:2]

    # Large centred text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(label_text, font, scale, thickness)
    tx = (dw - tw) // 2
    ty = (dh + th) // 2

    # Background rectangle
    pad = 20
    cv2.rectangle(
        confirm,
        (tx - pad, ty - th - pad),
        (tx + tw + pad, ty + pad),
        (0, 0, 0), -1,
    )
    cv2.putText(
        confirm, label_text, (tx, ty),
        font, scale, colour, thickness,
    )
    return confirm


# ═══════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    if not image_dir.is_dir():
        logger.error("Image directory does not exist: %s", image_dir)
        sys.exit(1)

    # Discover images
    all_images = discover_images(image_dir, args.pattern)
    logger.info("Found %d images in %s", len(all_images), image_dir)

    if not all_images:
        logger.error("No images found — check --image-dir and --pattern")
        sys.exit(1)

    # Load existing labels
    labels: Dict[str, dict] = {}
    if args.resume:
        labels = load_existing_labels(output_path)

    # Filter out already-labelled images
    unlabelled = [img for img in all_images if img.name not in labels]
    logger.info(
        "Already labelled: %d  |  Remaining: %d",
        len(all_images) - len(unlabelled), len(unlabelled),
    )

    if not unlabelled:
        logger.info("All images already labelled — nothing to do!")
        _print_summary(labels)
        return

    # Print instructions
    print()
    print("=" * 64)
    print("  RING ANNOTATION TOOL")
    print("=" * 64)
    print("  R  = Ring PRESENT  (docked image)")
    print("  N  = No ring       (pre-docked / natural eye)")
    print("  P  = Partial ring  (ring partially visible)")
    print("  U  = Undo last label")
    print("  S  = Save progress to disk")
    print("  Q  = Save & quit")
    print("=" * 64)
    print()

    window_name = "Ring Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    history: List[str] = []   # stack of labelled filenames for undo
    idx = 0
    save_pending = False

    while idx < len(unlabelled):
        img_path = unlabelled[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image (skipping): %s", img_path)
            idx += 1
            continue

        # Build display
        display = make_display_image(
            image, img_path.name, idx, len(unlabelled), args.max_display,
        )
        cv2.imshow(window_name, display)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF

        # ── R: Ring present ───────────────────────────────────────
        if key == ord("r"):
            labels[img_path.name] = {
                "ring_present": True,
                "ring_visibility": "full",
            }
            history.append(img_path.name)
            save_pending = True
            logger.info("  ✓ %s → RING PRESENT", img_path.name)

            confirm = show_confirmation(display, "RING PRESENT", (0, 255, 0))
            cv2.imshow(window_name, confirm)
            cv2.waitKey(250)
            idx += 1

        # ── N: No ring ────────────────────────────────────────────
        elif key == ord("n"):
            labels[img_path.name] = {
                "ring_present": False,
                "ring_visibility": "none",
            }
            history.append(img_path.name)
            save_pending = True
            logger.info("  ✓ %s → NO RING", img_path.name)

            confirm = show_confirmation(display, "NO RING", (0, 200, 255))
            cv2.imshow(window_name, confirm)
            cv2.waitKey(250)
            idx += 1

        # ── P: Partial ring ───────────────────────────────────────
        elif key == ord("p"):
            labels[img_path.name] = {
                "ring_present": True,
                "ring_visibility": "partial",
            }
            history.append(img_path.name)
            save_pending = True
            logger.info("  ✓ %s → PARTIAL RING", img_path.name)

            confirm = show_confirmation(display, "PARTIAL RING", (0, 255, 200))
            cv2.imshow(window_name, confirm)
            cv2.waitKey(250)
            idx += 1

        # ── U: Undo ───────────────────────────────────────────────
        elif key == ord("u"):
            if history:
                last = history.pop()
                if last in labels:
                    del labels[last]
                idx = max(0, idx - 1)
                save_pending = True
                logger.info("  ↩ Undid: %s", last)
            else:
                logger.info("  Nothing to undo")

        # ── S: Save ───────────────────────────────────────────────
        elif key == ord("s"):
            save_labels(labels, output_path)
            save_pending = False
            logger.info("  💾 Saved %d labels to %s", len(labels), output_path)

        # ── Q / ESC: Quit ─────────────────────────────────────────
        elif key == ord("q") or key == 27:
            save_labels(labels, output_path)
            save_pending = False
            logger.info("Saved %d labels and quit", len(labels))
            break

        # ── Auto-save every 25 labels ─────────────────────────────
        if save_pending and len(history) % 25 == 0 and len(history) > 0:
            save_labels(labels, output_path)
            save_pending = False
            logger.info("  (auto-saved %d labels)", len(labels))

    # Final save
    if save_pending:
        save_labels(labels, output_path)

    cv2.destroyAllWindows()
    _print_summary(labels)


# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════

def _print_summary(labels: Dict[str, dict]) -> None:
    """Print a summary of the labelling session."""
    total = len(labels)
    ring_full = sum(
        1 for v in labels.values()
        if v.get("ring_present") and v.get("ring_visibility") == "full"
    )
    ring_partial = sum(
        1 for v in labels.values()
        if v.get("ring_present") and v.get("ring_visibility") == "partial"
    )
    no_ring = sum(
        1 for v in labels.values()
        if not v.get("ring_present")
    )

    print()
    print("=" * 64)
    print("  ANNOTATION SUMMARY")
    print("=" * 64)
    print(f"  Total labelled:      {total}")
    print(f"  Ring present (full): {ring_full}")
    print(f"  Ring partial:        {ring_partial}")
    print(f"  No ring:             {no_ring}")
    if total > 0:
        balance = (ring_full + ring_partial) / total * 100
        print(f"  Ring ratio:          {balance:.1f}%")
    print("=" * 64)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()