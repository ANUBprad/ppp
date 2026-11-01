from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.utils.logger import get_logger
from pupil_tracking.pentacam.types import (
    PentacamImageType,
    PentacamDetectionStatus,
    PentacamQuality,
    PentacamGeometry,
    PentacamFeature,
    PentacamFeatureSet,
    PentacamDetectionResult,
)

logger = get_logger()


# ── Pentacam-specific enums and constants ────────────────────────────────

class WorkspaceAnnotationSchemaVersion(str, Enum):
    """Version of the annotation schema used by this workspace."""
    V1_0 = "1.0"
    V1_1 = "1.1"  # added orientation_transform, annotator_id


# ── Main Workspace Window ────────────────────────────────────────────────

class PentacamAnnotationWorkspace:
    """Pentacam manual annotation workspace.

    Extends AnnotationTool with:
    - Image-type classification (PentacamImageType)
    - Quality grading (PentacamQuality)
    - OD/OS eye-side selection
    - 180° orientation handling with coordinate transform preservation
    - Versioned annotation persistence
    - Batch navigation
    - Export capabilities
    """

    def __init__(
        self,
        root: tk.Tk,
        dataset_dir: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.root.title("Pentacam Manual Annotation Workspace")
        self.root.geometry("1400x900")

        self.dataset_dir: Optional[Path] = dataset_dir
        self.patient_eye: Optional[Tuple[str, str]] = None  # (OD/OS, patient_id)
        self.current_image_path: Optional[Path] = None
        self.original_image: Optional[np.ndarray] = None  # canonical, unmodified
        self.display_image: Optional[np.ndarray] = None  # may be flipped/rotated
        self.annotation_result: Optional[PentacamDetectionResult] = None

        # Orientation state
        self._orientation_degrees: float = 0.0  # 0 or 180, user-applied
        self._orientation_applied: bool = False  # whether flip has been done

        # UI state
        self._image_index: int = -1
        self._image_files: List[Path] = []
        self._annotations_by_image: Dict[str, Dict[str, Any]] = {}

        # Mode state
        self._mode: str = "pupil"  # "pupil" | "limbus" | "ring"
        self._pupil_points: List[Tuple[float, float]] = []
        self._limbus_points: List[Tuple[float, float]] = []
        self._ring_points: List[Tuple[float, float]] = []

        self._build_ui()

        # If a dataset directory was provided, pre-load file list
        if dataset_dir is not None:
            self._load_dataset_dir(dataset_dir)

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Main layout: left panel = image canvas, right panel = controls
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Left: canvas frame
        self.canvas_frame = ttk.LabelFrame(self.root, text="Image Canvas")
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            self.canvas_frame,
            bg="#1a1a1a",
            cursor="crosshair",
            highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<Configure>", self._on_configure)

        # Right: control panel
        self.control_frame = ttk.LabelFrame(self.root, text="Controls")
        self.control_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)
        self.control_frame.grid_rowconfigure(0, weight=1)

        # ── Image navigation ──────────────────────────────────────────────
        nav_label = ttk.Label(self.control_frame, text="Image Navigation")
        nav_label.pack(anchor="w", padx=5, pady=2)

        self._image_index_var = tk.StringVar(value="---")
        ttk.Label(self.control_frame, text="Image:").pack(anchor="w", padx=5)
        self._image_label = ttk.Label(
            self.control_frame, text="No dataset loaded", foreground="gray"
        )
        self._image_label.pack(anchor="w", padx=5)

        ttk.Button(self.control_frame, text="<< Prev", command=self._prev_image).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(self.control_frame, text="Next >>", command=self._next_image).pack(
            fill=tk.X, padx=5, pady=2
        )

        # ── Eye side selection ──────────────────────────────────────────
        eye_label = ttk.Label(self.control_frame, text="Eye Side")
        eye_label.pack(anchor="w", padx=5, pady=(10, 0))
        self._eye_var = tk.StringVar(value="OD")
        for eye in ["OD", "OS"]:
            ttk.Radiobutton(
                self.control_frame, text=eye, variable=self._eye_var, value=eye
            ).pack(anchor="w", padx=5, pady=1)

        # ── Image-type classification ───────────────────────────────────
        it_label = ttk.Label(self.control_frame, text="Image Type")
        it_label.pack(anchor="w", padx=5, pady=(10, 0))
        self._image_type_var = tk.StringVar(value="UNKNOWN")
        for pit in PentacamImageType:
            ttk.Radiobutton(
                self.control_frame,
                text=pit.value,
                variable=self._image_type_var,
                value=pit.value,
            ).pack(anchor="w", padx=5, pady=1)

        # ── Quality grading ─────────────────────────────────────────────
        ql_label = ttk.Label(self.control_frame, text="Quality Grade")
        ql_label.pack(anchor="w", padx=5, pady=(10, 0))
        self._quality_var = tk.StringVar(value="NO_DETECTION")
        for q in PentacamQuality:
            ttk.Radiobutton(
                self.control_frame, text=q.value, variable=self._quality_var, value=q.value
            ).pack(anchor="w", padx=5, pady=1)

        # ── Orientation ──────────────────────────────────────────────────
        orient_label = ttk.Label(self.control_frame, text="Orientation")
        orient_label.pack(anchor="w", padx=5, pady=(10, 0))
        self._orientation_var = tk.StringVar(value="0°")
        ttk.Radiobutton(
            self.control_frame, text="0° (normal)", variable=self._orientation_var, value="0°"
        ).pack(anchor="w", padx=5, pady=1)
        ttk.Radiobutton(
            self.control_frame, text="180° (flipped)", variable=self._orientation_var, value="180°"
        ).pack(anchor="w", padx=5, pady=1)

        # ── Annotation mode ─────────────────────────────────────────────
        mode_label = ttk.Label(self.control_frame, text="Annotation Mode")
        mode_label.pack(anchor="w", padx=5, pady=(10, 0))
        self._mode_var = tk.StringVar(value="pupil")
        for mode, color in [("pupil", "green"), ("limbus", "blue"), ("ring", "red")]:
            ttk.Radiobutton(
                self.control_frame, text=mode.capitalize(),
                variable=self._mode_var, value=mode,
            ).pack(anchor="w", padx=5, pady=1)

        # ── Action buttons ──────────────────────────────────────────────
        btn_frame = ttk.LabelFrame(self.control_frame, text="Actions")
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="Add Points", command=self._add_points).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(btn_frame, text="Fit Ellipse", command=self._fit_ellipse).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(btn_frame, text="Undo Point", command=self._undo_point).pack(
            fill=tk.X, padx=5, pady=2
        )
        ttk.Button(btn_frame, text="Reset Annotation", command=self._reset_annotation).pack(
            fill=tk.X, padx=5, pady=2
        )

        # ── Status ──────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Load a Pentacam image to begin")
        ttk.Label(self.control_frame, textvariable=self._status_var, font=("Consolas", 9))\
            .pack(anchor="w", padx=5, pady=5)

        # ── Menu bar ────────────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Dataset Directory...",
            command=self._open_dataset_directory
        )
        file_menu.add_command(
            label="Open Single Image...",
            command=self._open_single_image
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Annotation", command=self._save_annotation)
        file_menu.add_command(
            label="Save Annotation As...",
            command=self._save_annotation_as
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    # ── Dataset Browser (Phase 2) ────────────────────────────────────────

    def _load_dataset_dir(self, dataset_dir: Path) -> None:
        """Load a directory of Pentacam images."""
        # Supported raster formats initially
        raster_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        self._image_files = []
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in raster_exts:
                self._image_files.append(p)

        self._image_index = 0
        self._annotations_by_image = {}
        self._load_existing_annotations_for_dataset()
        self._refresh_image_display()

    def _open_dataset_directory(self) -> None:
        """Phase 2: Browse and select a dataset directory."""
        path = filedialog.askdirectory(
            title="Select Pentacam Dataset Directory",
            initialdir=self.dataset_dir,
        )
        if path:
            self.dataset_dir = Path(path)
            self._load_dataset_dir(self.dataset_dir)
            self._status_var.set(
                f"Loaded dataset: {len(self._image_files)} images"
            )

    def _open_single_image(self) -> None:
        """Open a single image file (bypass dataset browser)."""
        path = filedialog.askopenfilename(
            title="Open Pentacam Image",
            filetypes=[
                ("Pentacam images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.dcm *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if path:
            p = Path(path)
            self.dataset_dir = p.parent
            self._image_files = [p]
            self._image_index = 0
            self._annotations_by_image = {}
            self._load_existing_annotations_for_dataset()
            self._refresh_image_display()
            self._status_var.set(f"Opened: {p.name}")

    def _load_existing_annotations_for_dataset(self) -> None:
        """Load any existing annotations for the current dataset images."""
        if self._image_files is None:
            return

        # Load annotations from the workspace's private data directory
        # (not from the images themselves, to keep images git-ignored)
        data_dir = Path(".pentacam_annotations")
        data_dir.mkdir(exist_ok=True)

        for img_path in self._image_files:
            ann_path = data_dir / f"{img_path.stem}.json"
            if ann_path.exists():
                try:
                    with open(ann_path) as f:
                        self._annotations_by_image[img_path.name] = json.load(f)
                except Exception:
                    self._annotations_by_image[img_path.name] = {}

    # ── Annotation Canvas (Phase 3) ─────────────────────────────────────

    def _add_points(self) -> None:
        """Add a point in the current annotation mode."""
        if self._image is None:
            return

        mode = self._mode_var.get()
        self._status_var.set(
            f"{mode}: Click on the image to add a point (current: {len(self._get_current_points())})"
        )

    def _on_click(self, event) -> None:
        """Handle canvas clicks for point placement."""
        if self._image is None:
            return

        # Convert canvas coords to image coords
        ix = (event.x - self._offset_x) / self._scale
        iy = (event.y - self._offset_y) / self._scale

        h, w = self._image.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return

        mode = self._mode_var.get()
        if mode == "pupil":
            self._pupil_points.append((ix, iy))
        elif mode == "limbus":
            self._limbus_points.append((ix, iy))
        elif mode == "ring":
            self._ring_points.append((ix, iy))

        n = len(self._get_current_points())
        self._status_var.set(
            f"{mode}: {n} points (need >= 5 for ellipse fit)"
        )
        self._refresh()

    def _on_right_click(self, event) -> None:
        """Handle right-click: undo point."""
        self._undo_point()

    def _undo_point(self) -> None:
        """Remove the last point from the current mode."""
        mode = self._mode_var.get()
        pts = self._get_current_points()
        if pts:
            pts.pop()
        self._refresh()
        self._status_var.set(
            f"{mode}: {len(self._get_current_points())} points"
        )

    def _clear_current(self) -> None:
        """Clear all points from the current mode."""
        mode = self._mode_var.get()
        pts = self._get_current_points()
        pts.clear()
        self._refresh()
        self._status_var.set(f"{mode}: 0 points")

    def _fit_ellipse(self) -> None:
        """Fit ellipse to the current mode's points."""
        pts = self._get_current_points()
        mode = self._mode_var.get()

        if len(pts) < 5:
            messagebox.showinfo(
                "Not enough points",
                f"Need >= 5 points to fit ellipse, got {len(pts)}"
            )
            return

        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            ellipse = cv2.fitEllipse(pts_np)
            (cx, cy), (d1, d2), angle = ellipse
            sa = max(d1, d2) / 2
            sb = min(d1, d2) / 2
            if d2 > d1:
                angle = (angle + 90) % 180

            # Store the fitted ellipse in the annotation result
            self._status_var.set(
                f"{mode}: center=({cx:.1f},{cy:.1f}) axes=({sa:.1f},{sb:.1f}) angle={angle:.1f}"
            )

            # Refresh display with overlay
            self._draw_ellipse_overlay(mode, (cx, cy), (sa, sb), angle)

        except cv2.error as e:
            messagebox.showerror("Fit Error", str(e))

    def _draw_ellipse_overlay(self, mode: str, center: Tuple[float, float], 
                              axes: Tuple[float, float], angle: float) -> None:
        """Draw the fitted ellipse on the canvas display."""
        if self._image is None:
            return

        # Draw on the display image
        colors = {
            "pupil": (0, 255, 0),
            "limbus": (255, 100, 0),
            "ring": (0, 0, 255),
        }
        color = colors.get(mode, (255, 255, 255))

        # Draw the ellipse
        cv2.ellipse(self.display_image, 
                    (int(center[0]), int(center[1])), 
                    (int(axes[0]), int(axes[1])), 
                    angle, 0, 360, color, 2)

        # Draw center point
        cx, cy = center
        cv2.circle(self.display_image, (int(cx), int(cy)), 4, color, -1)

        # Refresh canvas
        self._refresh_from_display()

    def _refresh_from_display(self) -> None:
        """Refresh the canvas from the current display image."""
        if self._image is None or self.display_image is None:
            return

        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        # Scale display image to canvas
        self._scale = min(canvas_w / self.display_image.shape[1], 
                         canvas_h / self.display_image.shape[0], 1.0)
        new_w = int(self.display_image.shape[1] * self._scale)
        new_h = int(self.display_image.shape[0] * self._scale)
        self._offset_x = (canvas_w - new_w) // 2
        self._offset_y = (canvas_h - new_h) // 2

        # Resize and convert
        resized = cv2.resize(self.display_image, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image_tk = ImageTk.PhotoImage(pil_img)

        self._canvas.delete("all")
        self._canvas.create_image(
            self._offset_x, self._offset_y,
            anchor=tk.NW, image=self._display_image_tk,
        )

    def _reset_annotation(self) -> None:
        """Reset all points for the current mode."""
        self._clear_current()

    # ── Image Orientation (Phase 4) ──────────────────────────────────────

    def _apply_orientation_to_display(self, degrees: float) -> None:
        """Apply orientation transform to the display image.

        The original image is never modified. This updates display_image.
        """
        if degrees == 0 or self.original_image is None:
            self.display_image = self.original_image.copy() if self.original_image is not None else None
            return

        if degrees == 180 and self.original_image is not None:
            self.display_image = cv2.flip(self.original_image, -1)
            self._orientation_degrees = 180.0
            self._orientation_applied = True
        else:
            self.display_image = self.original_image.copy()
            self._orientation_degrees = 0.0
            self._orientation_applied = False

        self._refresh_from_display()

    def _original_image_is_valid(self) -> bool:
        return self._original_image is not None

    # ── Annotation Metadata (Phase 5) ───────────────────────────────────

    def _build_annotation_result(self) -> PentacamDetectionResult:
        """Build the PentacamDetectionResult from current UI state."""
        if self.annotation_result is None:
            self.annotation_result = PentacamDetectionResult()

        # Map UI state to result
        self.annotation_result.valid = len(self._get_current_points()) >= 5
        self.annotation_result.status = PentacamDetectionStatus.OK

        # Image type
        try:
            self.annotation_result.image_type = PentacamImageType(
                self._image_type_var.get()
            )
        except ValueError:
            self.annotation_result.image_type = PentacamImageType.UNKNOWN

        # Quality grade
        try:
            self.annotation_result.quality = PentacamQuality(
                self._quality_var.get()
            )
        except ValueError:
            self.annotation_result.quality = PentacamQuality.NO_DETECTION

        # Detection status
        self.annotation_result.status = PentacamDetectionStatus.OK

        # Pupil ellipse - from clicked points
        pupil_pts = self._pupil_points
        if len(pupil_pts) >= 5:
            pts_np = np.array(pupil_pts, dtype=np.float32).reshape(-1, 1, 2)
            try:
                ellipse = cv2.fitEllipse(pts_np)
                (cx, cy), (d1, d2), angle = ellipse
                sa = max(d1, d2) / 2
                sb = min(d1, d2) / 2
                if d2 > d1:
                    angle = (angle + 90) % 180
                
                self.annotation_result.geometry.pupil_detected = True
                self.annotation_result.geometry.pupil_radius_px = sa
            except cv2.error:
                pass

        # Limbus ellipse - from clicked points
        limbus_pts = self._limbus_points
        if len(limbus_pts) >= 5:
            pts_np = np.array(limbus_pts, dtype=np.float32).reshape(-1, 1, 2)
            try:
                ellipse = cv2.fitEllipse(pts_np)
                (cx, cy), (d1, d2), angle = ellipse
                sa = max(d1, d2) / 2
                sb = min(d1, d2) / 2
                if d2 > d1:
                    angle = (angle + 90) % 180
                
                self.annotation_result.geometry.limbus_detected = True
                self.annotation_result.geometry.limbus_radius_px = sa
            except cv2.error:
                pass

        # Boundary points (raw clicked points stored as feature set)
        features = []
        for i, (x, y) in enumerate(self._pupil_points[:36]):
            features.append({
                "id": i,
                "x": float(x),
                "y": float(y),
                "confidence": 0.9,
            })
        self.annotation_result.feature_set.features = features

        # Annotator ID and timestamp
        import time
        self.annotation_result.confidence = 0.9
        self.annotation_result.processing_time_ms = 0.0

        # Schema version
        self.annotation_result.annotation_schema_version = WorkspaceAnnotationSchemaVersion.V1_1.value

        # Orientation transform
        orient_str = f"0°->{int(self._orientation_degrees)}°" if self._orientation_applied else "0°"
        self.annotation_result.orientation_transform = orient_str

        return self.annotation_result

    # ── Annotation Persistence (Phase 6) ────────────────────────────────

    def _save_annotation(self) -> None:
        """Save the current annotation."""
        if self._image is None:
            messagebox.showinfo("No Image", "Load an image first")
            return

        # Build the annotation result
        result = self._build_annotation_result()

        # Determine save path
        if self._annotation_path is None:
            # Save to private data directory
            data_dir = Path(".pentacam_annotations")
            data_dir.mkdir(exist_ok=True)
            ann_path = data_dir / f"{self.current_image_path.stem}.json"
        else:
            ann_path = Path(self._annotation_path)

        # Check if file already exists — don't overwrite without confirmation
        if ann_path.exists():
            result_msg = messagebox.askyesno(
                "Existing Annotation",
                f"A annotation already exists at {ann_path.name}.\n"
                "Overwrite?"
            )
            if not result_msg:
                return

        # Serialize the result
        try:
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
                if "schema_version" not in result_dict:
                    result_dict["schema_version"] = WorkspaceAnnotationSchemaVersion.V1_1.value
            else:
                result_dict = {
                    "valid": result.valid,
                    "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                    "image_type": result.image_type.value if hasattr(result.image_type, 'value') else str(result.image_type),
                    "quality": result.quality.value if hasattr(result.quality, 'value') else str(result.quality),
                    "geometry": result.geometry.to_dict() if hasattr(result.geometry, 'to_dict') else {},
                    "feature_set": result.feature_set.to_dict() if hasattr(result.feature_set, 'to_dict') else {},
                    "image_width": getattr(result, 'image_width', 0),
                    "image_height": getattr(result, 'image_height', 0),
                    "confidence": round(result.confidence, 4) if result.confidence else 0.0,
                    "processing_time_ms": round(result.processing_time_ms, 2) if result.processing_time_ms else 0.0,
                    "schema_version": result.annotation_schema_version,
                    "orientation_transform": result.orientation_transform,
                    "failure_reason": result.failure_reason,
                    "annotator_id": getattr(result, 'annotator_id', ''),
                    "annotated_at": getattr(result, 'annotated_at', 0.0),
                }

            with open(ann_path, "w") as f:
                json.dump(result_dict, f, indent=2)

            self._status_var.set(f"Saved annotation to {ann_path.name}")

            # Track in memory
            self._annotations_by_image[self.current_image_path.name] = result_dict

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save annotation: {e}")
            logger.error(f"Annotation save error: {e}")

    def _save_annotation_as(self) -> None:
        """Save annotation to a chosen path."""
        path = filedialog.asksaveasfilename(
            title="Save Annotation As",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            self._annotation_path = path
            self._save_annotation()

    def _reopen_annotation(self, image_path: Path) -> None:
        """Re-open a previously saved annotation."""
        data_dir = Path(".pentacam_annotations")
        ann_path = data_dir / f"{image_path.stem}.json"

        if ann_path.exists():
            try:
                with open(ann_path) as f:
                    data = json.load(f)
                self._annotations_by_image[image_path.name] = data
                if "image_type" in data:
                    self._image_type_var.set(data["image_type"])
                if "quality" in data:
                    self._quality_var.set(data["quality"])
                if "orientation_transform" in data:
                    orient = data["orientation_transform"]
                    if "180°" in orient:
                        self._orientation_var.set("180°")
                        self._orientation_applied = True
                        self._apply_orientation_to_display(180)
                self._status_var.set(
                    f"Re-opened annotation from {ann_path.name}"
                )
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load annotation: {e}")
                logger.error(f"Annotation load error: {e}")
        else:
            self._annotations_by_image[image_path.name] = {}
            self._status_var.set("No saved annotation found — starting fresh")

    # ── Quality Control (Phase 7) ───────────────────────────────────────

    def _validate_annotation(self) -> Tuple[bool, str]:
        """Validate the current annotation.

        Returns (is_valid, error_message).
        """
        if self._image is None:
            return False, "No image loaded"

        h, w = self._image.shape[:2]
        for pts in [self._pupil_points, self._limbus_points, self._ring_points]:
            for (x, y) in pts:
                if x < 0 or x >= w or y < 0 or y >= h:
                    return False, f"Point ({x:.1f},{y:.1f}) is outside image bounds ({w}x{h})"

        return True, ""

    # ── Export (Phase 8) ────────────────────────────────────────────────

    def _export_annotations(self) -> None:
        """Export annotations in versioned JSON format."""
        if self._image is None:
            messagebox.showinfo("No Image", "Load an image first")
            return

        result = self._build_annotation_result()

        path = filedialog.asksaveasfilename(
            title="Export Annotations",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        try:
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
                if "schema_version" not in result_dict:
                    result_dict["schema_version"] = WorkspaceAnnotationSchemaVersion.V1_1.value
            else:
                result_dict = {
                    "schema_version": WorkspaceAnnotationSchemaVersion.V1_1.value,
                    "valid": result.valid,
                    "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                    "image_type": result.image_type.value if hasattr(result.image_type, 'value') else str(result.image_type),
                    "quality": result.quality.value if hasattr(result.quality, 'value') else str(result.quality),
                    "orientation_transform": result.orientation_transform,
                    "failure_reason": result.failure_reason,
                    "annotator_id": getattr(result, 'annotator_id', ''),
                    "annotated_at": getattr(result, 'annotated_at', 0.0),
                }

            with open(path, "w") as f:
                json.dump(result_dict, f, indent=2)

            self._export_annotated_overlay(path)
            self._export_mask(path)

            self._status_var.set(f"Exported to {Path(path).name}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
            logger.error(f"Export error: {e}")

    def _export_annotated_overlay(self, json_path: Path) -> None:
        """Generate an annotated image overlay next to the JSON."""
        if self._image is None or self.display_image is None:
            return

        data_dir = Path(".pentacam_annotations")
        base_name = Path(json_path).stem
        overlay_path = data_dir / f"{base_name}_overlay.png"

        disp = self.display_image.copy()
        colors = {
            "pupil": (0, 255, 0),
            "limbus": (255, 100, 0),
            "ring": (0, 0, 255),
        }

        for mode, pts, color in [
            ("pupil", self._pupil_points, colors["pupil"]),
            ("limbus", self._limbus_points, colors["limbus"]),
        ]:
            for (x, y) in pts:
                cv2.circle(disp, (int(x), int(y)), 3, color, -1)

        for mode, pts, color in [
            ("pupil", self._pupil_points, colors["pupil"]),
            ("limbus", self._limbus_points, colors["limbus"]),
        ]:
            if len(pts) >= 5:
                try:
                    pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                    ellipse = cv2.fitEllipse(pts_np)
                    cv2.ellipse(disp, ellipse, color, 2)
                except cv2.error:
                    pass

        if self._orientation_applied:
            cv2.putText(disp, "180° flipped", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        overlay_path = overlay_path.parent / f"{overlay_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), disp)
        logger.info(f"Exported annotated overlay: {overlay_path}")

    def _export_mask(self, json_path: Path) -> None:
        """Generate segmentation mask where applicable."""
        data_dir = Path(".pentacam_annotations")
        base_name = Path(json_path).stem
        mask_path = data_dir / f"{base_name}_mask.png"
        logger.info(f"Mask export noted: {mask_path} (implementation: pupil/limbus binary masks pending)")

    # ── Navigation ───────────────────────────────────────────────────────

    def _prev_image(self) -> None:
        if not self._image_files or self._image_index <= 0:
            return
        self._image_index -= 1
        self._load_image(self._image_files[self._image_index])

    def _next_image(self) -> None:
        if not self._image_files or self._image_index >= len(self._image_files) - 1:
            return
        self._image_index += 1
        self._load_image(self._image_files[self._image_index])

    def _load_image(self, image_path: Path) -> None:
        """Load a specific image from the dataset."""
        self.current_image_path = image_path

        try:
            original = cv2.imread(str(image_path))
            if original is None:
                messagebox.showerror("Error", f"Cannot read: {image_path}")
                return

            self.original_image = original
            self._orientation_degrees = 0.0
            self._orientation_applied = False
            self.display_image = self.original_image.copy()

            self._reopen_annotation(image_path)

            self._image_label.config(
                text=f"{image_path.name} ({self.original_image.shape[1]}x{self.original_image.shape[0]})"
            )
            self._refresh_from_display()
            self._status_var.set(
                f"Loaded: {image_path.name} — Use annotation tools above"
            )

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load image: {e}")
            logger.error(f"Image load error: {e}")

    # ── Utility methods ─────────────────────────────────────────────────

    def _show_about(self) -> None:
        msg = (
            "Pentacam Manual Annotation Workspace\n"
            "Version 1.0\n\n"
            "Extension of the existing ellipse-based annotation tool\n"
            "with Pentacam-specific features:\n"
            "- Image-type classification (PentacamImageType enum)\n"
            "- Quality grading (PentacamQuality enum)\n"
            "- OD/OS eye-side selection\n"
            "- 180° orientation handling with coordinate transform\n"
            "- Versioned annotation schema (V1.0/V1.1)\n"
            "- Annotation persistence in private .pentacam_annotations/ dir\n"
            "- Export with annotated overlays\n\n"
            "Dataset remains private; no patient images uploaded."
        )
        messagebox.showinfo("About", msg)

    def run(self) -> None:
        self.root.mainloop()


def launch_pentacam_workspace(dataset_dir: Optional[Path] = None) -> None:
    """Launch the Pentacam annotation workspace."""
    root = tk.Tk()
    app = PentacamAnnotationWorkspace(root, dataset_dir=dataset_dir)
    app.run()


if __name__ == "__main__":
    launch_pentacam_workspace()