#!/usr/bin/env python3
"""Check that all required project files exist."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    # utils
    "pupil_tracking/__init__.py",
    "pupil_tracking/utils/__init__.py",
    "pupil_tracking/utils/types.py",
    "pupil_tracking/utils/config.py",
    "pupil_tracking/utils/logger.py",
    # core
    "pupil_tracking/core/__init__.py",
    "pupil_tracking/core/ellipse_fitter.py",
    "pupil_tracking/core/corneal_center.py",
    "pupil_tracking/core/detector.py",
    # ml
    "pupil_tracking/ml/__init__.py",
    "pupil_tracking/ml/architecture.py",
    "pupil_tracking/ml/dataset.py",
    "pupil_tracking/ml/losses.py",
    "pupil_tracking/ml/trainer.py",
    "pupil_tracking/ml/inference.py",
    "pupil_tracking/ml/postprocess.py",
    # video
    "pupil_tracking/video/__init__.py",
    "pupil_tracking/video/kalman_tracker.py",
    "pupil_tracking/video/video_processor.py",
    # scripts
    "scripts/train_model.py",
    "scripts/generate_masks.py",
    "scripts/verify_data.py",
    # root
    "launch_gui.py",
]

REQUIRED_DIRS = [
    "pupil_tracking",
    "pupil_tracking/utils",
    "pupil_tracking/core",
    "pupil_tracking/ml",
    "pupil_tracking/video",
    "pupil_tracking/preprocessing",
    "pupil_tracking/calibration",
    "pupil_tracking/annotation",
    "pupil_tracking/interface",
    "scripts",
    "tests",
    "models",
    "logs",
    "clinical_data/annotations/masks",
]


def main():
    print(f"\n{'='*60}")
    print(f"  PROJECT FILE CHECK")
    print(f"  Root: {PROJECT_ROOT}")
    print(f"{'='*60}\n")

    # Check directories
    print("── Directories ──")
    missing_dirs = 0
    for d in REQUIRED_DIRS:
        p = PROJECT_ROOT / d
        if p.is_dir():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/  ← MISSING — run: mkdir -p {d}")
            missing_dirs += 1
    print()

    # Check files
    print("── Files ──")
    missing_files = 0
    for f in REQUIRED_FILES:
        p = PROJECT_ROOT / f
        if p.is_file():
            size = p.stat().st_size
            print(f"  ✓ {f}  ({size:,} bytes)")
        else:
            print(f"  ✗ {f}  ← MISSING")
            missing_files += 1
    print()

    # Check data
    print("── Data Files ──")
    ann_path = PROJECT_ROOT / "clinical_data" / "annotations" / "annotations.json"
    if ann_path.exists():
        print(f"  ✓ annotations.json ({ann_path.stat().st_size:,} bytes)")
    else:
        print(f"  ✗ annotations.json ← MISSING")
        missing_files += 1

    clean_dir = PROJECT_ROOT / "clinical_data" / "clean"
    if clean_dir.is_dir():
        images = list(clean_dir.glob("*.jpeg")) + list(clean_dir.glob("*.jpg")) + list(clean_dir.glob("*.png"))
        print(f"  ✓ clinical_data/clean/ ({len(images)} images)")
        for img in sorted(images)[:5]:
            print(f"      {img.name}")
        if len(images) > 5:
            print(f"      ... and {len(images)-5} more")
    else:
        print(f"  ✗ clinical_data/clean/ ← MISSING")
        missing_dirs += 1
    print()

    # Summary
    print(f"{'='*60}")
    if missing_files == 0 and missing_dirs == 0:
        print(f"  ✓ All {len(REQUIRED_FILES)} files and "
              f"{len(REQUIRED_DIRS)} directories present")
    else:
        print(f"  ✗ {missing_files} missing file(s), "
              f"{missing_dirs} missing directory(ies)")
        print(f"\n  Create missing directories:")
        for d in REQUIRED_DIRS:
            p = PROJECT_ROOT / d
            if not p.is_dir():
                print(f"    mkdir -p \"{d}\"")
        if missing_files > 0:
            print(f"\n  Create missing files from the code provided")
            print(f"  in the previous conversation messages.")
    print(f"{'='*60}\n")

    return 0 if (missing_files == 0 and missing_dirs == 0) else 1


if __name__ == "__main__":
    sys.exit(main())