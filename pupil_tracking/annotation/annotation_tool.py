"""
GUI annotation tool for labeling pupil and limbus boundaries.

Allows the user to:
    1. Load clinical eye images
    2. Click to define ellipse boundaries for pupil and limbus
    3. Adjust ellipse parameters with sliders
    4. Save annotations in the project's JSON format
    5. Generate training masks

Usage:
    python -m pupil_tracking.annotation.annotation_tool
    python scripts/annotate_data.py
"""

from __future__ import annotations

import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from pupil_tracking.utils.logger import get_logger


class AnnotationTool:
    """Simple ellipse annotation tool for eye images.

    The user clicks points on the pupil boundary, then limbus
    boundary. An ellipse is fitted to each set of points.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Eye Annotation Tool")
        self.root.geometry("1200x800")

        self.logger = get_logger()

        # state
        self._image: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._display_image: Optional[ImageTk.PhotoImage] = None
        self._scale: float = 1.0
        self._offset_x: int = 0
        self._offset_y: int = 0

        self._mode: str = "pupil"  # "pupil" | "limbus" | "ring"
        self._pupil_points: List[Tuple[float, float]] = []
        self._limbus_points: List[Tuple[float, float]] = []
        self._ring_points: List[Tuple[float, float]] = []

        self._annotations: Dict[str, Dict[str, Any]] = {}
        self._annotation_path: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        # toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(
            toolbar, text="Open Image", command=self._open_image
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar, text="Load Annotations", command=self._load_annotations
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar, text="Save", command=self._save_annotations
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )

        self._mode_var = tk.StringVar(value="pupil")
        for mode, color in [
            ("pupil", "green"), ("limbus", "blue"), ("ring", "red")
        ]:
            ttk.Radiobutton(
                toolbar, text=mode.capitalize(),
                variable=self._mode_var, value=mode,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=5
        )

        ttk.Button(
            toolbar, text="Fit Ellipse", command=self._fit_current
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar, text="Clear Points", command=self._clear_current
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            toolbar, text="Undo Point", command=self._undo_point
        ).pack(side=tk.LEFT, padx=2)

        # status
        self._status_var = tk.StringVar(value="Load an image to begin")
        ttk.Label(
            toolbar, textvariable=self._status_var, font=("Consolas", 9)
        ).pack(side=tk.RIGHT, padx=5)

        # canvas
        self._canvas = tk.Canvas(self.root, bg="#1a1a1a", cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<Button-3>", self._on_right_click)
        self._canvas.bind("<Configure>", self._on_resize)

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open Eye Image",
            filetypes=[
                ("Image files", "*.jpeg *.jpg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", f"Cannot read: {path}")
            return

        self._image = image
        self._image_path = path
        self._pupil_points.clear()
        self._limbus_points.clear()
        self._ring_points.clear()

        # check if annotation already exists
        stem = Path(path).stem
        if stem in self._annotations:
            self._load_existing_annotation(stem)

        self._refresh()
        self._status_var.set(
            f"{Path(path).name} ({image.shape[1]}x{image.shape[0]}) "
            f"- Click to add {self._mode_var.get()} points"
        )

    def _on_click(self, event) -> None:
        if self._image is None:
            return

        # convert canvas coords to image coords
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
        self._undo_point()

    def _get_current_points(self) -> list:
        mode = self._mode_var.get()
        if mode == "pupil":
            return self._pupil_points
        elif mode == "limbus":
            return self._limbus_points
        return self._ring_points

    def _undo_point(self) -> None:
        pts = self._get_current_points()
        if pts:
            pts.pop()
            self._refresh()

    def _clear_current(self) -> None:
        pts = self._get_current_points()
        pts.clear()
        self._refresh()

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        n = len(self._get_current_points())
        self._status_var.set(f"{mode}: {n} points")
        self._refresh()

    def _fit_current(self) -> None:
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

            self._status_var.set(
                f"{mode}: center=({cx:.0f},{cy:.0f}) "
                f"axes=({sa:.0f},{sb:.0f}) angle={angle:.0f}"
            )
            self._refresh()

        except cv2.error as e:
            messagebox.showerror("Fit Error", str(e))

    def _save_annotations(self) -> None:
        if self._image_path is None:
            messagebox.showinfo("No Image", "Load an image first")
            return

        if not self._annotation_path:
            self._annotation_path = filedialog.asksaveasfilename(
                title="Save Annotations",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialfile="annotations.json",
            )
        if not self._annotation_path:
            return

        stem = Path(self._image_path).stem
        filename = Path(self._image_path).name
        h, w = self._image.shape[:2]

        entry: Dict[str, Any] = {
            "image_path": str(self._image_path),
            "image_width": w,
            "image_height": h,
            "annotations": {},
        }

        # fit and save each structure
        for mode, points, class_id in [
            ("PUPIL", self._pupil_points, 1),
            ("LIMBUS", self._limbus_points, 2),
            ("RING", self._ring_points, 3),
        ]:
            if len(points) >= 5:
                pts_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
                try:
                    (cx, cy), (d1, d2), angle = cv2.fitEllipse(pts_np)
                    sa = max(d1, d2) / 2
                    sb = min(d1, d2) / 2
                    if d2 > d1:
                        angle = (angle + 90) % 180

                    entry["annotations"][mode] = {
                        "class_id": class_id,
                        "class_name": mode,
                        "center_x": float(cx),
                        "center_y": float(cy),
                        "semi_major": float(sa),
                        "semi_minor": float(sb),
                        "angle_deg": float(angle),
                        "boundary_points": [
                            [float(p[0]), float(p[1])] for p in points
                        ],
                    }
                except cv2.error:
                    pass

        self._annotations[filename] = entry

        # load existing, merge, save
        ann_path = Path(self._annotation_path)
        existing = {}
        if ann_path.exists():
            with open(ann_path) as fh:
                existing = json.load(fh)

        existing[filename] = entry

        with open(ann_path, "w") as fh:
            json.dump(existing, fh, indent=2)

        self._status_var.set(f"Saved annotation for {filename}")

    def _load_annotations(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        with open(path) as fh:
            self._annotations = json.load(fh)

        self._annotation_path = path
        self._status_var.set(
            f"Loaded {len(self._annotations)} annotations"
        )

    def _load_existing_annotation(self, stem: str) -> None:
        for key, entry in self._annotations.items():
            if Path(key).stem == stem:
                structs = entry.get("annotations", {})
                for mode, points_list in [
                    ("PUPIL", self._pupil_points),
                    ("LIMBUS", self._limbus_points),
                    ("RING", self._ring_points),
                ]:
                    s = structs.get(mode, {})
                    bp = s.get("boundary_points", [])
                    points_list.clear()
                    points_list.extend(
                        [(p[0], p[1]) for p in bp]
                    )
                break

    def _refresh(self) -> None:
        if self._image is None:
            return

        canvas_w = self._canvas.winfo_width()
        canvas_h = self._canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        image = self._image.copy()
        h, w = image.shape[:2]

        # draw points and fitted ellipses
        colors = {
            "pupil": (0, 255, 0),
            "limbus": (255, 100, 0),
            "ring": (0, 0, 255),
        }
        all_points = {
            "pupil": self._pupil_points,
            "limbus": self._limbus_points,
            "ring": self._ring_points,
        }

        for mode, pts in all_points.items():
            color = colors[mode]
            for px, py in pts:
                cv2.circle(
                    image, (int(px), int(py)), 3, color, -1
                )

            if len(pts) >= 5:
                pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
                try:
                    ellipse = cv2.fitEllipse(pts_np)
                    cv2.ellipse(image, ellipse, color, 2)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    cv2.circle(image, center, 4, color, -1)
                except cv2.error:
                    pass

        # highlight current mode
        mode = self._mode_var.get()
        color = colors.get(mode, (255, 255, 255))
        cv2.putText(
            image, f"Mode: {mode.upper()}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )

        # scale to canvas
        self._scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w = int(w * self._scale)
        new_h = int(h * self._scale)
        self._offset_x = (canvas_w - new_w) // 2
        self._offset_y = (canvas_h - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._display_image = ImageTk.PhotoImage(pil_img)

        self._canvas.delete("all")
        self._canvas.create_image(
            self._offset_x, self._offset_y,
            anchor=tk.NW, image=self._display_image,
        )

    def _on_resize(self, event) -> None:
        self._refresh()


def launch_annotation_tool() -> None:
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()


if __name__ == "__main__":
    launch_annotation_tool()