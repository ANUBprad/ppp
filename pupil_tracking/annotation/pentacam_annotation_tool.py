"""Manual Pentacam image annotation workspace.

The original image is immutable. All point coordinates and fitted ellipses are
stored in original-image pixel coordinates; 0/180-degree rotation is a
reversible display-only transform. Annotation JSON and drafts are written to
a local private directory and must not be committed with clinical data.
"""
from __future__ import annotations

import copy
import json
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.pentacam.types import PentacamImageType, PentacamQuality
from .workspace_core import (
    AnnotationValidationError,
    build_segmentation_mask,
    discover_images,
    fit_ellipse_annotation,
    load_annotation_json,
    normalize_roi_box,
    denormalize_roi_box,
    display_to_original_point,
    original_to_display_point,
    save_annotation_json,
    stable_image_id,
    transform_point,
    validate_annotation_document,
    validate_points,
)

APP_TITLE = "Pentacam Manual Annotation Workspace"
SCHEMA_VERSION = "1.1"
ANNOTATION_MODES = ("pupil", "limbus", "ring")
MODE_COLORS = {
    "pupil": (0, 220, 80),
    "limbus": (255, 150, 0),
    "ring": (0, 80, 255),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PentacamAnnotationWorkspace:
    """Tkinter dataset browser and manual ellipse annotation workspace."""

    def __init__(
        self,
        root: tk.Tk,
        dataset_dir: Optional[Path] = None,
        annotation_dir: Optional[Path] = None,
    ) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1500x920")
        self.root.minsize(1000, 650)

        self.dataset_dir = Path(dataset_dir).expanduser().resolve() if dataset_dir else None
        self.annotation_dir = (
            Path(annotation_dir).expanduser().resolve()
            if annotation_dir
            else (Path.cwd() / ".pentacam_annotations").resolve()
        )
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        self._image_files: list[Path] = []
        self._image_index = -1
        self.current_image_path: Optional[Path] = None
        self.image_id: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.display_image: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None

        self._orientation_degrees = 0
        self._roi_box: Optional[tuple[int, int, int, int]] = None
        self._roi_history: list[Optional[tuple[int, int, int, int]]] = []
        self._roi_preview = False
        self._roi_selection_mode = False
        self._roi_drag_start: Optional[tuple[int, int]] = None
        self._scale = 1.0
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._pan_anchor: Optional[tuple[int, int]] = None

        self._mode = "pupil"
        self._points: dict[str, list[tuple[float, float]]] = {
            mode: [] for mode in ANNOTATION_MODES
        }
        self._ellipses: dict[str, Optional[dict[str, float]]] = {
            mode: None for mode in ANNOTATION_MODES
        }
        self._undo_stack: list[tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]] = []
        self._redo_stack: list[tuple[dict[str, list[tuple[float, float]]], dict[str, Any]]] = []
        self._loaded_document: Optional[dict[str, Any]] = None
        self._annotation_path: Optional[Path] = None
        self._dirty = False
        self._autosave_job: Optional[str] = None
        self._loading_metadata = False

        self._eye_var = tk.StringVar(value="OD")
        self._image_type_var = tk.StringVar(value=PentacamImageType.UNKNOWN.value)
        self._quality_var = tk.StringVar(value=PentacamQuality.NO_DETECTION.value)
        self._mode_var = tk.StringVar(value="pupil")
        self._orientation_var = tk.StringVar(value="0")
        self._annotator_var = tk.StringVar(value="")
        self._failure_reason_var = tk.StringVar(value="")
        self._roi_preview_var = tk.BooleanVar(value=False)
        self._status_var = tk.StringVar(value="Choose a dataset directory to begin")
        self._counter_var = tk.StringVar(value="0 / 0 images")

        self._build_ui()
        for variable in (
            self._eye_var,
            self._image_type_var,
            self._quality_var,
            self._annotator_var,
            self._failure_reason_var,
        ):
            variable.trace_add("write", self._on_metadata_changed)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.dataset_dir is not None:
            self._load_dataset_dir(self.dataset_dir)

    # ------------------------------ UI ---------------------------------
    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        body = ttk.Frame(self.root, padding=6)
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        canvas_panel = ttk.LabelFrame(body, text="Original image (annotations use original pixel coordinates)")
        canvas_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        canvas_panel.grid_columnconfigure(0, weight=1)
        canvas_panel.grid_rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(canvas_panel, background="#17191d", cursor="crosshair", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._canvas.bind("<ButtonPress-1>", self._on_left_press)
        self._canvas.bind("<B1-Motion>", self._on_left_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_motion)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        controls_outer = ttk.Frame(body, width=320)
        controls_outer.grid(row=0, column=1, sticky="ns")
        controls_outer.grid_propagate(False)
        controls_canvas = tk.Canvas(
            controls_outer, width=300, highlightthickness=0, borderwidth=0
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_outer, orient="vertical", command=controls_canvas.yview
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_canvas.pack(side="left", fill="both", expand=True)
        controls_scrollbar.pack(side="right", fill="y")
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )

        def _refresh_controls_scrollregion(_event: Any = None) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _fit_controls_width(event: tk.Event) -> None:
            controls_canvas.itemconfigure(controls_window, width=event.width)

        def _scroll_controls(event: tk.Event) -> None:
            if event.delta:
                controls_canvas.yview_scroll(int(-event.delta / 120), "units")

        def _enable_controls_wheel(_event: tk.Event) -> None:
            self.root.bind_all("<MouseWheel>", _scroll_controls)

        def _disable_controls_wheel(_event: tk.Event) -> None:
            self.root.unbind_all("<MouseWheel>")

        controls.bind("<Configure>", _refresh_controls_scrollregion)
        controls_canvas.bind("<Configure>", _fit_controls_width)
        controls_outer.bind("<Enter>", _enable_controls_wheel)
        controls_outer.bind("<Leave>", _disable_controls_wheel)

        ttk.Label(controls, text="Dataset").pack(anchor="w", pady=(0, 3))
        ttk.Button(controls, text="Open dataset folder…", command=self._open_dataset_directory).pack(fill="x")
        ttk.Button(controls, text="Open one image…", command=self._open_single_image).pack(fill="x", pady=(3, 8))

        ttk.Label(controls, textvariable=self._counter_var).pack(anchor="w")
        ttk.Label(controls, text="Current image:", foreground="gray").pack(anchor="w")
        self._image_label = ttk.Label(controls, text="No image loaded", wraplength=280)
        self._image_label.pack(anchor="w", pady=(0, 4))
        navigation = ttk.Frame(controls)
        navigation.pack(fill="x")
        ttk.Button(navigation, text="◀ Previous", command=self._prev_image).pack(side="left", expand=True, fill="x")
        ttk.Button(navigation, text="Next ▶", command=self._next_image).pack(side="left", expand=True, fill="x", padx=(4, 0))

        roi_frame = ttk.LabelFrame(controls, text="Region of interest (ROI)")
        roi_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(roi_frame, text="Select / resize ROI (drag)", command=self._start_roi_selection).pack(
            fill="x", padx=5, pady=2
        )
        roi_actions = ttk.Frame(roi_frame)
        roi_actions.pack(fill="x", padx=5)
        ttk.Button(roi_actions, text="Undo ROI", command=self._undo_roi).pack(
            side="left", expand=True, fill="x"
        )
        ttk.Button(roi_actions, text="Clear ROI", command=self._clear_roi).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )
        ttk.Checkbutton(
            roi_frame, text="Preview cropped ROI", variable=self._roi_preview_var,
            command=self._on_roi_preview_toggle,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Eye side").pack(anchor="w")
        ttk.Combobox(controls, textvariable=self._eye_var, values=("OD", "OS"), state="readonly").pack(fill="x")

        ttk.Label(controls, text="Image type").pack(anchor="w", pady=(7, 0))
        ttk.Combobox(
            controls, textvariable=self._image_type_var,
            values=tuple(item.value for item in PentacamImageType), state="readonly",
        ).pack(fill="x")

        ttk.Label(controls, text="Manual image quality").pack(anchor="w", pady=(7, 0))
        ttk.Combobox(
            controls, textvariable=self._quality_var,
            values=tuple(item.value for item in PentacamQuality), state="readonly",
        ).pack(fill="x")

        ttk.Label(controls, text="Annotator ID (optional)").pack(anchor="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self._annotator_var).pack(fill="x")
        ttk.Label(controls, text="Annotation note / failure reason").pack(anchor="w", pady=(7, 0))
        ttk.Entry(controls, textvariable=self._failure_reason_var).pack(fill="x")

        ttk.Separator(controls).pack(fill="x", pady=8)
        ttk.Label(controls, text="Annotation structure").pack(anchor="w")
        for mode, label in (("pupil", "Pupil"), ("limbus", "Limbus"), ("ring", "Suction ring")):
            ttk.Radiobutton(
                controls, text=label, variable=self._mode_var, value=mode,
                command=self._on_mode_changed,
            ).pack(anchor="w")

        actions = ttk.LabelFrame(controls, text="Annotation")
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Fit ellipse (≥5 points)", command=self._fit_ellipse).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Undo", command=self._undo_point).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Redo", command=self._redo_point).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Clear current structure", command=self._clear_current).pack(fill="x", padx=5, pady=2)
        ttk.Button(actions, text="Save annotation", command=self._save_annotation).pack(fill="x", padx=5, pady=2)

        orientation = ttk.LabelFrame(controls, text="Display orientation")
        orientation.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(
            orientation, text="Original (0°)", variable=self._orientation_var,
            value="0", command=self._on_orientation_changed,
        ).pack(anchor="w", padx=5)
        ttk.Radiobutton(
            orientation, text="Rotate display 180°", variable=self._orientation_var,
            value="180", command=self._on_orientation_changed,
        ).pack(anchor="w", padx=5)

        zoom = ttk.Frame(controls)
        zoom.pack(fill="x", pady=(0, 8))
        ttk.Button(zoom, text="− Zoom", command=lambda: self._set_zoom(self._zoom / 1.2)).pack(side="left", expand=True, fill="x")
        ttk.Button(zoom, text="Fit", command=lambda: self._set_zoom(1.0, reset_pan=True)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(zoom, text="+ Zoom", command=lambda: self._set_zoom(self._zoom * 1.2)).pack(side="left", expand=True, fill="x")

        exports = ttk.LabelFrame(controls, text="Export (local files)")
        exports.pack(fill="x", pady=(0, 8))
        ttk.Button(exports, text="Export JSON…", command=self._export_json).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export ROI crop…", command=self._export_crop).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export overlay PNG…", command=self._export_overlay).pack(fill="x", padx=5, pady=2)
        ttk.Button(exports, text="Export segmentation mask…", command=self._export_mask).pack(fill="x", padx=5, pady=2)

        ttk.Label(
            controls,
            text="Left click: add point  |  Right click: undo\nMiddle-drag: pan  |  Mouse wheel: zoom\nOriginal images are never modified.",
            wraplength=285, foreground="gray",
        ).pack(anchor="w", pady=(3, 8))

        ttk.Label(self.root, textvariable=self._status_var, anchor="w", relief="sunken").grid(
            row=1, column=0, sticky="ew"
        )

    # --------------------------- Dataset --------------------------------
    def _open_dataset_directory(self) -> None:
        selected = filedialog.askdirectory(
            title="Select Pentacam dataset directory",
            initialdir=str(self.dataset_dir) if self.dataset_dir else str(Path.cwd()),
        )
        if selected:
            self._load_dataset_dir(Path(selected))

    def _open_single_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Open Pentacam raster image",
            filetypes=[("Supported images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if selected:
            path = Path(selected).resolve()
            self.dataset_dir = path.parent
            self._image_files = [path]
            self._image_index = -1
            self._load_image(0)

    def _load_dataset_dir(self, dataset_dir: Path) -> None:
        resolved_dir = dataset_dir.expanduser().resolve()
        try:
            images = discover_images(resolved_dir)
        except (OSError, NotADirectoryError) as exc:
            messagebox.showerror("Dataset error", str(exc))
            return
        self.dataset_dir = resolved_dir
        self._image_files = images
        self._image_index = -1
        if not images:
            self._clear_current_image()
            self._status_var.set("No supported raster images found (JPG, PNG, BMP, TIFF). DICOM/PDF import is not implemented.")
            return
        self._load_image(0)

    def _clear_current_image(self) -> None:
        self.current_image_path = None
        self.image_id = None
        self.original_image = None
        self.display_image = None
        self._points = {mode: [] for mode in ANNOTATION_MODES}
        self._ellipses = {mode: None for mode in ANNOTATION_MODES}
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._canvas.delete("all")
        self._image_label.config(text="No image loaded")
        self._counter_var.set(f"0 / {len(self._image_files)} images")

    def _load_image(self, index: int) -> None:
        if not self._image_files or not (0 <= index < len(self._image_files)):
            return
        if self.current_image_path is not None and self._dirty:
            self._flush_autosave()

        image_path = self._image_files[index]
        try:
            encoded = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV could not decode this file")
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("Image load error", f"Could not load {image_path.name}: {exc}")
            return

        self.current_image_path = image_path
        self._image_index = index
        self.original_image = image
        self.display_image = image.copy()
        self._orientation_degrees = 0
        self._orientation_var.set("0")
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._roi_selection_mode = False
        self._roi_drag_start = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._points = {mode: [] for mode in ANNOTATION_MODES}
        self._ellipses = {mode: None for mode in ANNOTATION_MODES}

        if self.dataset_dir:
            try:
                relative_path = image_path.relative_to(self.dataset_dir).as_posix()
            except ValueError:
                relative_path = image_path.name
        else:
            relative_path = image_path.name
        self.image_id = stable_image_id(relative_path)
        self._annotation_path = self.annotation_dir / f"{self.image_id}.json"
        self._loaded_document = None
        self._dirty = False
        self._restore_saved_annotation()

        height, width = image.shape[:2]
        self._image_label.config(text=f"{image_path.name}\n{width} × {height}")
        self._counter_var.set(f"{index + 1} / {len(self._image_files)} images")
        self._status_var.set(f"Loaded {image_path.name}; coordinates are stored in original image space")
        self._refresh_canvas()

    def _restore_saved_annotation(self) -> None:
        self._loading_metadata = True
        self._roi_box = None
        self._roi_history.clear()
        self._roi_preview = False
        self._roi_preview_var.set(False)
        if self._annotation_path is None or not self._annotation_path.exists():
            self._loaded_document = None
            self._eye_var.set("OD")
            self._image_type_var.set(PentacamImageType.UNKNOWN.value)
            self._quality_var.set(PentacamQuality.NO_DETECTION.value)
            self._annotator_var.set("")
            self._failure_reason_var.set("")
            self._loading_metadata = False
            return
        try:
            document = load_annotation_json(self._annotation_path)
            if document.get("image_id") != self.image_id:
                raise AnnotationValidationError("saved annotation image_id does not match the selected image")
            self._loaded_document = document
            self._eye_var.set(document.get("eye_side", "OD"))
            self._image_type_var.set(document.get("image_type", PentacamImageType.UNKNOWN.value))
            self._quality_var.set(document.get("quality_grade", PentacamQuality.NO_DETECTION.value))
            self._annotator_var.set(document.get("annotator_id", ""))
            self._failure_reason_var.set(document.get("failure_reason", ""))
            orientation = document.get("orientation_transform", {})
            degrees = int(orientation.get("display_rotation_degrees", 0)) if isinstance(orientation, dict) else 0
            self._orientation_degrees = 180 if degrees == 180 else 0
            self._orientation_var.set(str(self._orientation_degrees))
            height, width = self.original_image.shape[:2]
            roi_norm = document.get("roi_box_norm")
            if roi_norm is not None:
                self._roi_box = denormalize_roi_box(roi_norm, width, height)
                self._roi_preview = True
                self._roi_preview_var.set(True)
            for mode in ANNOTATION_MODES:
                record = document.get("annotations", {}).get(mode, {})
                self._points[mode] = [tuple(point) for point in record.get("boundary_points", [])]
                self._ellipses[mode] = copy.deepcopy(record.get("ellipse"))
            self._dirty = False
            self._status_var.set("Restored saved annotation and ROI")
        except (OSError, ValueError, json.JSONDecodeError, AnnotationValidationError) as exc:
            self._loaded_document = None
            self._roi_box = None
            self._roi_preview = False
            self._roi_preview_var.set(False)
            self._status_var.set(f"Saved annotation could not be loaded: {exc}")
        finally:
            self._loading_metadata = False
        self._refresh_canvas()

    # ------------------------- Coordinate/UI ---------------------------
    def _view_dimensions(self) -> tuple[int, int]:
        if self.original_image is None:
            return 0, 0
        height, width = self.original_image.shape[:2]
        if self._roi_preview and self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            return x1 - x0, y1 - y0
        return width, height

    def _view_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        roi = self._roi_box if self._roi_preview else None
        return display_to_original_point(
            point, width, height, self._orientation_degrees, roi_box=roi
        )

    def _original_to_view(self, point: tuple[float, float]) -> Optional[tuple[float, float]]:
        if self.original_image is None:
            return None
        height, width = self.original_image.shape[:2]
        roi = self._roi_box if self._roi_preview else None
        return original_to_display_point(
            point, width, height, self._orientation_degrees, roi_box=roi
        )

    def _canvas_to_view(self, event: tk.Event) -> tuple[float, float]:
        return (
            (event.x - self._offset_x) / self._scale,
            (event.y - self._offset_y) / self._scale,
        )

    def _on_left_press(self, event: tk.Event) -> None:
        if self._roi_selection_mode:
            self._roi_drag_start = (event.x, event.y)
            self._canvas.delete("roi-drag")

    def _on_left_motion(self, event: tk.Event) -> None:
        if not self._roi_selection_mode or self._roi_drag_start is None:
            return
        x0, y0 = self._roi_drag_start
        self._canvas.delete("roi-drag")
        self._canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#00e5ff", width=2,
            dash=(6, 3), tags="roi-drag",
        )

    def _on_left_release(self, event: tk.Event) -> None:
        if self._roi_selection_mode:
            self._finish_roi_selection(event)
            return
        self._on_click(event)

    def _on_click(self, event: tk.Event) -> None:
        if self.original_image is None or self._roi_selection_mode:
            return
        view_x, view_y = self._canvas_to_view(event)
        view_width, view_height = self._view_dimensions()
        if not (0 <= view_x < view_width and 0 <= view_y < view_height):
            return
        original_point = self._view_to_original((view_x, view_y))
        height, width = self.original_image.shape[:2]
        try:
            validate_points([original_point], width, height)
        except AnnotationValidationError:
            return
        self._push_undo()
        self._points[self._mode].append(original_point)
        self._ellipses[self._mode] = None
        self._mark_dirty()
        self._status_var.set(f"{self._mode}: {len(self._points[self._mode])} boundary points")
        self._refresh_canvas()

    def _start_roi_selection(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image before selecting an ROI.")
            return
        self._roi_selection_mode = True
        self._roi_drag_start = None
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._canvas.configure(cursor="crosshair")
        self._refresh_canvas()
        self._status_var.set("Drag a rectangle around the eye image. Release to set or resize the ROI.")

    def _finish_roi_selection(self, event: tk.Event) -> None:
        self._canvas.delete("roi-drag")
        start = self._roi_drag_start
        self._roi_drag_start = None
        self._roi_selection_mode = False
        self._canvas.configure(cursor="crosshair")
        if self.original_image is None or start is None:
            return

        start_view = (
            (start[0] - self._offset_x) / self._scale,
            (start[1] - self._offset_y) / self._scale,
        )
        end_view = self._canvas_to_view(event)
        try:
            start_original = self._view_to_original(start_view)
            end_original = self._view_to_original(end_view)
        except (AnnotationValidationError, ValueError):
            self._status_var.set("ROI selection failed; try again inside the image.")
            return

        height, width = self.original_image.shape[:2]
        x0 = max(0, min(width - 1, int(np.floor(min(start_original[0], end_original[0])))))
        y0 = max(0, min(height - 1, int(np.floor(min(start_original[1], end_original[1])))))
        x1 = max(x0 + 1, min(width, int(np.ceil(max(start_original[0], end_original[0]))) + 1))
        y1 = max(y0 + 1, min(height, int(np.ceil(max(start_original[1], end_original[1]))) + 1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            self._status_var.set("ROI is too small. Drag a larger rectangle.")
            self._refresh_canvas()
            return

        try:
            normalize_roi_box((x0, y0, x1, y1), width, height)
        except AnnotationValidationError as exc:
            self._status_var.set(f"Invalid ROI: {exc}")
            return

        self._roi_history.append(self._roi_box)
        if len(self._roi_history) > 50:
            self._roi_history.pop(0)
        self._roi_box = (x0, y0, x1, y1)
        self._roi_preview = True
        self._roi_preview_var.set(True)
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set(f"ROI selected: {x1 - x0} × {y1 - y0} px in original image coordinates.")

    def _undo_roi(self) -> None:
        if not self._roi_history:
            self._status_var.set("No previous ROI to restore")
            return
        self._roi_box = self._roi_history.pop()
        self._roi_preview = self._roi_box is not None
        self._roi_preview_var.set(self._roi_preview)
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set("Restored previous ROI")

    def _clear_roi(self) -> None:
        if self._roi_box is None:
            return
        self._roi_history.append(self._roi_box)
        self._roi_box = None
        self._roi_preview = False
        self._roi_preview_var.set(False)
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._mark_dirty()
        self._refresh_canvas()
        self._status_var.set("ROI cleared; showing full image")

    def _on_roi_preview_toggle(self) -> None:
        if self._roi_preview_var.get() and self._roi_box is None:
            self._roi_preview_var.set(False)
            messagebox.showinfo("No ROI", "Select a rectangular ROI first.")
            return
        self._roi_preview = bool(self._roi_preview_var.get())
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._refresh_canvas()
        self._status_var.set("Showing ROI crop preview" if self._roi_preview else "Showing full image")

    def _on_right_click(self, _event: tk.Event) -> None:
        self._undo_point()

    def _on_mode_changed(self) -> None:
        self._mode = self._mode_var.get()
        self._status_var.set(f"Annotation mode: {self._mode}")

    def _on_metadata_changed(self, *_args: Any) -> None:
        if not self._loading_metadata and self.original_image is not None:
            self._mark_dirty()

    def _on_orientation_changed(self) -> None:
        self._apply_orientation_to_display(int(self._orientation_var.get()))

    def _apply_orientation_to_display(self, degrees: int) -> None:
        if degrees not in (0, 180):
            raise ValueError("only 0- and 180-degree orientation is supported")
        self._orientation_degrees = degrees
        self._mark_dirty()
        self._refresh_canvas()

    def _on_canvas_configure(self, _event: tk.Event) -> None:
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        if self.original_image is None:
            return
        height, width = self.original_image.shape[:2]
        if self._roi_preview and self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            view_image = self.original_image[y0:y1, x0:x1].copy()
            view_width, view_height = x1 - x0, y1 - y0
        else:
            view_image = self.original_image.copy()
            view_width, view_height = width, height

        display = cv2.flip(view_image, -1) if self._orientation_degrees == 180 else view_image
        canvas_width = max(1, self._canvas.winfo_width())
        canvas_height = max(1, self._canvas.winfo_height())
        base_scale = min(canvas_width / view_width, canvas_height / view_height)
        self._scale = max(0.01, min(base_scale * self._zoom, 8.0))
        scaled_width = max(1, int(round(view_width * self._scale)))
        scaled_height = max(1, int(round(view_height * self._scale)))
        self._offset_x = (canvas_width - scaled_width) / 2 + self._pan_x
        self._offset_y = (canvas_height - scaled_height) / 2 + self._pan_y

        for mode in ANNOTATION_MODES:
            color = MODE_COLORS[mode]
            for point in self._points[mode]:
                view_point = self._original_to_view(point)
                if view_point is not None:
                    cv2.circle(
                        display, (int(round(view_point[0])), int(round(view_point[1]))),
                        3, color, -1,
                    )
            ellipse = self._ellipses[mode]
            if ellipse:
                center = self._original_to_view((ellipse["center_x"], ellipse["center_y"]))
                if center is not None:
                    axes = (
                        max(1, int(round(ellipse["semi_major"]))),
                        max(1, int(round(ellipse["semi_minor"]))),
                    )
                    cv2.ellipse(
                        display, (int(round(center[0])), int(round(center[1]))),
                        axes, float(ellipse["angle_deg"]), 0, 360, color, 2,
                    )

        if self._roi_box is not None and not self._roi_preview:
            x0, y0, x1, y1 = self._roi_box
            p0 = transform_point((x0, y0), width, height, self._orientation_degrees)
            p1 = transform_point((x1 - 1, y1 - 1), width, height, self._orientation_degrees)
            cv2.rectangle(
                display, (int(round(p0[0])), int(round(p0[1]))),
                (int(round(p1[0])), int(round(p1[1]))), (255, 220, 0), 2,
            )
        if self._roi_preview:
            cv2.putText(display, "ROI CROP PREVIEW", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        self.display_image = display
        resized = cv2.resize(
            display, (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if self._scale < 1 else cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._photo_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(
            self._offset_x, self._offset_y, anchor=tk.NW, image=self._photo_image
        )

    def _set_zoom(self, value: float, reset_pan: bool = False) -> None:
        self._zoom = max(0.25, min(float(value), 6.0))
        if reset_pan:
            self._pan_x = self._pan_y = 0.0
        self._refresh_canvas()

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            self._set_zoom(self._zoom * (1.15 if delta > 0 else 1 / 1.15))

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_anchor = (event.x, event.y)

    def _on_pan_motion(self, event: tk.Event) -> None:
        if self._pan_anchor is None:
            return
        previous_x, previous_y = self._pan_anchor
        self._pan_x += event.x - previous_x
        self._pan_y += event.y - previous_y
        self._pan_anchor = (event.x, event.y)
        self._refresh_canvas()

    # -------------------------- Annotation -----------------------------
    def _push_undo(self) -> None:
        self._undo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
        self._redo_stack.clear()
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _undo_point(self) -> None:
        if not self._undo_stack:
            # Allow one-point-at-a-time undo for annotations loaded from disk.
            if not self._points[self._mode]:
                self._status_var.set("Nothing to undo")
                return
            self._redo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
            self._points[self._mode].pop()
            self._ellipses[self._mode] = None
        else:
            self._redo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
            self._points, self._ellipses = self._undo_stack.pop()
        self._mark_dirty()
        self._refresh_canvas()

    def _redo_point(self) -> None:
        if not self._redo_stack:
            self._status_var.set("Nothing to redo")
            return
        self._undo_stack.append((copy.deepcopy(self._points), copy.deepcopy(self._ellipses)))
        self._points, self._ellipses = self._redo_stack.pop()
        self._mark_dirty()
        self._refresh_canvas()

    def _clear_current(self) -> None:
        if self.original_image is None:
            return
        self._push_undo()
        self._points[self._mode] = []
        self._ellipses[self._mode] = None
        self._mark_dirty()
        self._refresh_canvas()

    def _fit_ellipse(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        height, width = self.original_image.shape[:2]
        try:
            ellipse = fit_ellipse_annotation(self._points[self._mode], width, height)
        except AnnotationValidationError as exc:
            messagebox.showwarning("Cannot fit ellipse", str(exc))
            return
        self._push_undo()
        self._ellipses[self._mode] = ellipse
        self._mark_dirty()
        self._status_var.set(
            f"{self._mode}: center=({ellipse['center_x']:.1f}, {ellipse['center_y']:.1f}), "
            f"axes=({ellipse['semi_major']:.1f}, {ellipse['semi_minor']:.1f}) px"
        )
        self._refresh_canvas()

    def _build_annotation_document(self) -> dict[str, Any]:
        if self.original_image is None or self.current_image_path is None or self.image_id is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        annotations: dict[str, Any] = {}
        for mode in ANNOTATION_MODES:
            points = validate_points(self._points[mode], width, height)
            if points or self._ellipses[mode]:
                annotations[mode] = {
                    "boundary_points": [[x, y] for x, y in points],
                    "ellipse": copy.deepcopy(self._ellipses[mode]),
                }

        now = _utc_now()
        created_at = (
            self._loaded_document.get("created_at", now)
            if self._loaded_document else now
        )
        status = (
            "OK" if self._ellipses["pupil"] and self._ellipses["limbus"]
            else "NO_PUPIL" if not self._ellipses["pupil"]
            else "NO_LIMBUS"
        )
        previous_revision = int(self._loaded_document.get("revision", 0)) if self._loaded_document else 0
        document = {
            "schema_version": SCHEMA_VERSION,
            "revision": previous_revision + 1,
            "image_id": self.image_id,
            "image_width": width,
            "image_height": height,
            "coordinate_system": "original_image_pixels",
            "eye_side": self._eye_var.get(),
            "image_type": self._image_type_var.get(),
            "quality_grade": self._quality_var.get(),
            "detection_status": status,
            "annotations": annotations,
            "roi_box_norm": (
                normalize_roi_box(self._roi_box, width, height)
                if self._roi_box is not None else None
            ),
            "roi_coordinate_system": "original_image_normalized_half_open_xyxy",
            "failure_reason": self._failure_reason_var.get().strip(),
            "annotator_id": self._annotator_var.get().strip(),
            "created_at": created_at,
            "annotated_at": now,
            "updated_at": now,
            "orientation_transform": {
                "display_rotation_degrees": self._orientation_degrees,
                "coordinates_stored_in": "original_image_pixels",
            },
        }
        validate_annotation_document(document)
        return document

    def _validate_current(self) -> None:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        height, width = self.original_image.shape[:2]
        for mode in ANNOTATION_MODES:
            validate_points(self._points[mode], width, height)
            ellipse = self._ellipses[mode]
            if ellipse:
                validate_annotation_document({
                    "schema_version": SCHEMA_VERSION,
                    "image_width": width,
                    "image_height": height,
                    "annotations": {mode: {"boundary_points": self._points[mode], "ellipse": ellipse}},
                })

    # ---------------------- Persistence/autosave ------------------------
    def _annotation_document_path(self) -> Path:
        if self._annotation_path is None:
            raise AnnotationValidationError("annotation path is not initialized")
        return self._annotation_path

    def _write_current_annotation(self) -> Path:
        document = self._build_annotation_document()
        path = save_annotation_json(self._annotation_document_path(), document)
        self._loaded_document = document
        self._dirty = False
        return path

    def _mark_dirty(self) -> None:
        if self.original_image is None:
            return
        self._dirty = True
        if self._autosave_job is not None:
            try:
                self.root.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
        self._autosave_job = self.root.after(900, self._autosave)

    def _autosave(self) -> None:
        self._autosave_job = None
        if not self._dirty or self.original_image is None:
            return
        try:
            self._write_current_annotation()
            self._status_var.set("Draft autosaved locally")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            self._status_var.set(f"Autosave failed: {exc}")

    def _flush_autosave(self) -> None:
        if self._autosave_job is not None:
            try:
                self.root.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
            self._autosave_job = None
        if self._dirty and self.original_image is not None:
            self._autosave()

    def _save_annotation(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            path = self._write_current_annotation()
            self._status_var.set(f"Saved private annotation: {path.name}")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Save error", str(exc))

    def _on_close(self) -> None:
        self._flush_autosave()
        self.root.destroy()

    # ------------------------------ Export ------------------------------
    def _export_json(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            document = self._build_annotation_document()
            path = filedialog.asksaveasfilename(
                title="Export annotation JSON",
                defaultextension=".json",
                initialfile=f"{self.image_id}.json",
                filetypes=[("JSON", "*.json")],
            )
            if path:
                save_annotation_json(path, document)
                self._status_var.set("Annotation JSON exported")
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Export error", str(exc))

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        suffix = path.suffix.lower() or ".png"
        success, encoded = cv2.imencode(suffix, image)
        if not success:
            raise OSError(f"OpenCV could not encode {suffix}")
        encoded.tofile(str(path))

    def _render_original_overlay(self) -> np.ndarray:
        if self.original_image is None:
            raise AnnotationValidationError("no image is loaded")
        overlay = self.original_image.copy()
        if self._roi_box is not None:
            x0, y0, x1, y1 = self._roi_box
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2)
        for mode in ANNOTATION_MODES:
            color = MODE_COLORS[mode]
            for x, y in self._points[mode]:
                cv2.circle(overlay, (int(round(x)), int(round(y))), 3, color, -1)
            ellipse = self._ellipses[mode]
            if ellipse:
                center = (int(round(ellipse["center_x"])), int(round(ellipse["center_y"])))
                axes = (max(1, int(round(ellipse["semi_major"]))), max(1, int(round(ellipse["semi_minor"]))))
                cv2.ellipse(overlay, center, axes, ellipse["angle_deg"], 0, 360, color, 2)
        return overlay

    def _export_crop(self) -> None:
        """Export the raw ROI crop in original image orientation."""
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        if self._roi_box is None:
            messagebox.showinfo("No ROI", "Select an ROI before exporting a crop.")
            return
        x0, y0, x1, y1 = self._roi_box
        crop = self.original_image[y0:y1, x0:x1].copy()
        path = filedialog.asksaveasfilename(
            title="Export ROI crop (original orientation)",
            defaultextension=".png",
            initialfile=f"{self.image_id}_crop.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            self._write_image(Path(path), crop)
            self._status_var.set(f"ROI crop exported: {x1 - x0} × {y1 - y0} px")
        except OSError as exc:
            messagebox.showerror("Crop export error", str(exc))

    def _export_overlay(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export original-coordinate overlay",
            defaultextension=".png",
            initialfile=f"{self.image_id}_overlay.png",
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            self._write_image(Path(path), self._render_original_overlay())
            self._status_var.set("Overlay exported in original image orientation")
        except OSError as exc:
            messagebox.showerror("Overlay export error", str(exc))

    def _export_mask(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return
        try:
            document = self._build_annotation_document()
            mask = build_segmentation_mask(document, roi_box=self._roi_box)
            suffix = "_roi_mask.png" if self._roi_box is not None else "_mask.png"
            path = filedialog.asksaveasfilename(
                title="Export segmentation mask (class IDs 0–3; ROI crop if selected)",
                defaultextension=".png",
                initialfile=f"{self.image_id}{suffix}",
                filetypes=[("PNG", "*.png")],
            )
            if path:
                self._write_image(Path(path), mask)
                self._status_var.set(
                    f"Segmentation mask exported: {mask.shape[1]} × {mask.shape[0]} px "
                    "(0=background, 1=pupil, 2=iris, 3=ring outline)"
                )
        except (OSError, AnnotationValidationError, TypeError, ValueError) as exc:
            messagebox.showerror("Mask export error", str(exc))

    # --------------------------- Navigation -----------------------------
    def _prev_image(self) -> None:
        if self._image_index > 0:
            self._flush_autosave()
            self._load_image(self._image_index - 1)

    def _next_image(self) -> None:
        if self._image_index + 1 < len(self._image_files):
            self._flush_autosave()
            self._load_image(self._image_index + 1)


def launch_pentacam_workspace(
    dataset_dir: Optional[Path] = None,
    annotation_dir: Optional[Path] = None,
) -> None:
    """Launch the workspace; private annotations default to ./.pentacam_annotations."""
    root = tk.Tk()
    PentacamAnnotationWorkspace(root, dataset_dir=dataset_dir, annotation_dir=annotation_dir)
    root.mainloop()


if __name__ == "__main__":
    launch_pentacam_workspace()
