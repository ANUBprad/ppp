#!/usr/bin/env python3
"""Launch the annotation tool GUI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.annotation.annotation_tool import launch_annotation_tool

if __name__ == "__main__":
    launch_annotation_tool()