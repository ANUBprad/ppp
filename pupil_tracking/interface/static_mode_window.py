"""static_mode_window.py — the Static Mode (image-based detection) window.

Separate top-level window for the still-image workflow::

    pre-dock   : doctor centres + focuses the eye, captures an image
    post-dock  : a second image is captured for comparison

It shows the **Pentacam** and **ELITA** iris annuli unwrapped into straight
strips (top-bottom), reports detected iris features, detection confidence and
the cyclotorsion angle, runs a validation checklist, and saves the pre- and
post-dock cyclotorsion details into the patient folder (plus Save As…).

All measurement logic lives in :mod:`pupil_tracking.cyclotorsion.static_mode`;
this module is display + wiring only.  It never touches the live centration /
pupil / limbus pipeline and never feeds surgical control.

The heavy work (detection ~1 s/image, iris features) runs on a worker thread
with a generation token so a stale run can never overwrite a newer one; Tk
widgets are only ever touched on the main thread.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.cyclotorsion.static_mode import (
    IMAGE_FILETYPES,
    PHASES,
    PHASE_LABELS,
    PHASE_POST_DOCK,
    PHASE_PRE_DOCK,
    VALID_EYES,
    StaticAnalysis,
    analyse_static_pair,
    find_reference_image,
    read_roi_norm,
    save_analysis,
    validate_analysis,
)

#: Strip display size (canvas box) — the strip itself is 64 x 512.
_STRIP_CANVAS_W = 780
_STRIP_CANVAS_H = 120
_RULER_H = 18

#: Feature-marker colour thresholds on the strip features' confidence.
_CONF_HIGH = 0.60
_CONF_MID = 0.30


class StaticModeWindow:
    """Static Mode window (one instance, parented to the main app)."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._colors = app._colors

        self._token = 0
        self._busy = False
        self._analysis: Optional[StaticAnalysis] = None
        self._iris_detector: Any = None  # built once, reused across runs

        # (eye, phase) -> ELITA target path and eye -> Pentacam reference path,
        # so a still is never silently carried across to another eye or phase.
        self._elita_paths: Dict[Tuple[str, str], str] = {}
        self._pentacam_paths: Dict[str, str] = {}
        # (eye, phase, pentacam, elita) -> StaticAnalysis
        self._cache: Dict[Tuple[str, str, str, str], StaticAnalysis] = {}

        self._win: Optional[tk.Toplevel] = None
        # key -> PhotoImage (kept alive so Tk does not garbage-collect it)
        self._photos: Dict[str, Any] = {}
        # key -> (strip, features) so a resize can re-render without the caller
        self._last_strip: Dict[str, Any] = {}
        self._strip_drawn: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Show the window (or raise it if it is already open)."""
        if self._win is not None:
            try:
                if self._win.winfo_exists():
                    self._win.deiconify()
                    self._win.lift()
                    self._win.focus_set()
                    return
            except tk.TclError:
                pass
        self._build()
        self._sync_from_patient()
        self._maybe_auto_analyse()

    def _build(self) -> None:
        c = self._colors
        win = tk.Toplevel(self.app.root)
        self._win = win
        win.title("Static Mode — Image-Based Detection")
        win.transient(self.app.root)
        win.configure(bg=c.BG_PRIMARY)
        # a fresh build always starts idle, even if a previous run was
        # orphaned by close() while still in flight
        self._busy = False

        try:
            from PIL import Image as _Img, ImageTk as _ITk
            import pathlib as _pl

            icon_path = _pl.Path(__file__).parent / "add-patient.png.png"
            icon = _ITk.PhotoImage(_Img.open(icon_path).resize((32, 32), _Img.LANCZOS))
            win.iconphoto(False, icon)
            win._icon_ref = icon
        except Exception:
            pass

        # Size to the screen.  The layout wants ~960x900, but clinical laptops
        # are often only 768 px tall, where an undocked 900 px window would push
        # the validation checklist and the status line off the display.
        self.app.root.update_idletasks()
        screen_w = self.app.root.winfo_screenwidth()
        screen_h = self.app.root.winfo_screenheight()
        w = max(760, min(960, screen_w - 60))
        h = max(520, min(900, screen_h - 80))
        rx = max(0, min(
            self.app.root.winfo_rootx() + (self.app.root.winfo_width() - w) // 2,
            max(0, screen_w - w - 20),
        ))
        ry = max(0, min(
            self.app.root.winfo_rooty() + (self.app.root.winfo_height() - h) // 2,
            max(0, screen_h - h - 40),
        ))
        win.geometry(f"{w}x{h}+{rx}+{ry}")
        win.minsize(720, 480)

        F = {"bg": c.BG_PRIMARY}
        H1 = ("Segoe UI", 15, "bold")
        H2 = ("Segoe UI", 10, "bold")
        BODY = ("Segoe UI", 10)
        SMALL = ("Segoe UI", 9)

        # ── header (fixed — never scrolls away) ───────────────────────
        hdr = tk.Frame(win, **F)
        hdr.pack(fill=tk.X, padx=20, pady=(12, 2))
        tk.Label(
            hdr, text="Static Mode — Image-Based Detection",
            font=H1, bg=c.BG_PRIMARY, fg=c.FG_PRIMARY,
        ).pack(anchor=tk.W)
        tk.Label(
            hdr,
            text="Pre-dock: centre & focus, capture · Post-dock: capture the "
                 "second image for comparison",
            font=SMALL, bg=c.BG_PRIMARY, fg=c.FG_SECONDARY,
        ).pack(anchor=tk.W, pady=(2, 0))

        # ── control bar (fixed): actions right, eye/phase left ────────
        ctl = tk.Frame(win, **F)
        ctl.pack(fill=tk.X, padx=20, pady=(8, 0))

        def _btn(text, cmd, primary=False):
            return tk.Button(
                ctl, text=text, font=H2,
                bg=c.BTN_ACTION if primary else c.BTN_BG,
                fg="#FFFFFF" if primary else c.FG_PRIMARY,
                activebackground=c.ACCENT_HOVER if primary else c.BTN_HOVER,
                activeforeground="#FFFFFF", relief=tk.FLAT, bd=0,
                padx=14, pady=5, cursor="hand2", command=cmd,
            )

        # packed right-to-left so the whole action row stays on screen
        _btn("Close", self.close).pack(side=tk.RIGHT)
        _btn("Save As…", self._save_as).pack(side=tk.RIGHT, padx=(0, 6))
        _btn("Save", self._save).pack(side=tk.RIGHT, padx=(0, 6))
        _btn("Validate", self._validate_now).pack(side=tk.RIGHT, padx=(0, 6))
        self._analyse_btn = _btn("Analyse", self._analyse, primary=True)
        self._analyse_btn.pack(side=tk.RIGHT, padx=(0, 6))

        self._eye_var = tk.StringVar(value=self._default_eye())
        tk.Label(ctl, text="Eye", font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY).pack(side=tk.LEFT, padx=(0, 8))
        eye_btns: Dict[str, tk.Button] = {}
        for eye in VALID_EYES:
            b = tk.Button(
                ctl, text=eye, font=BODY, width=4, relief=tk.FLAT, bd=0,
                cursor="hand2", command=lambda e=eye: self._set_eye(e),
            )
            b.pack(side=tk.LEFT, padx=(0, 4))
            eye_btns[eye] = b
        self._eye_btns = eye_btns

        tk.Label(ctl, text="Phase", font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY).pack(side=tk.LEFT, padx=(20, 8))
        self._phase_var = tk.StringVar(value=PHASE_PRE_DOCK)
        phase_btns: Dict[str, tk.Button] = {}
        for phase in PHASES:
            b = tk.Button(
                ctl, text=PHASE_LABELS[phase], font=BODY, relief=tk.FLAT, bd=0,
                cursor="hand2", command=lambda p=phase: self._set_phase(p),
            )
            b.pack(side=tk.LEFT, padx=(0, 4))
            phase_btns[phase] = b
        self._phase_btns = phase_btns

        self._eye_phase_var = tk.StringVar(value="")
        tk.Label(
            win, textvariable=self._eye_phase_var, font=SMALL,
            bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W,
        ).pack(fill=tk.X, padx=20, pady=(4, 0))
        tk.Frame(win, bg=c.BORDER, height=1).pack(fill=tk.X, padx=20, pady=(8, 0))

        # ── scrollable body: sources · strips · readouts · validation ─
        shell = tk.Frame(win, **F)
        shell.pack(fill=tk.BOTH, expand=True)
        vbar = tk.Scrollbar(
            shell, orient=tk.VERTICAL, bg=c.BG_SECONDARY,
            troughcolor=c.BG_PRIMARY, activebackground=c.BTN_HOVER,
            relief=tk.FLAT, bd=0, highlightthickness=0,
        )
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroller = tk.Canvas(shell, bg=c.BG_PRIMARY, highlightthickness=0,
                             yscrollcommand=vbar.set)
        scroller.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=scroller.yview)

        body = tk.Frame(scroller, **F)
        body_id = scroller.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda _e: scroller.configure(scrollregion=scroller.bbox("all")),
        )
        scroller.bind(
            "<Configure>",
            lambda e: scroller.itemconfigure(body_id, width=e.width),
        )
        self._scroller = scroller

        # ── sources ───────────────────────────────────────────────────
        src = tk.Frame(body, **F)
        src.pack(fill=tk.X, padx=20, pady=(10, 0))
        src.columnconfigure(1, weight=1)

        self._pentacam_var = tk.StringVar()
        self._elita_var = tk.StringVar()

        def _source_row(row: int, label: str, var: tk.StringVar, browse) -> None:
            tk.Label(src, text=label, font=H2, bg=c.BG_PRIMARY, fg=c.FG_PRIMARY,
                     anchor=tk.W, width=20).grid(
                row=row, column=0, sticky=tk.W, pady=3, padx=(0, 10)
            )
            entry = tk.Entry(
                src, textvariable=var, font=BODY, readonlybackground=c.BG_INPUT,
                fg=c.FG_PRIMARY, insertbackground=c.FG_PRIMARY, relief=tk.FLAT,
                bd=0, highlightthickness=1, highlightbackground=c.BORDER,
                highlightcolor=c.ACCENT,
            )
            entry.grid(row=row, column=1, sticky=tk.EW, pady=3, ipady=4)
            tk.Button(
                src, text="Browse…", font=BODY, bg=c.BTN_BG, fg=c.FG_PRIMARY,
                activebackground=c.BTN_HOVER, activeforeground=c.FG_PRIMARY,
                relief=tk.FLAT, bd=0, padx=12, pady=4, cursor="hand2",
                command=browse,
            ).grid(row=row, column=2, sticky=tk.E, pady=3, padx=(10, 0))

        self._src_status = tk.StringVar(value="No images selected")
        _source_row(0, "Pentacam (reference)", self._pentacam_var,
                    lambda: self._browse("pentacam"))
        _source_row(1, "ELITA (target)", self._elita_var,
                    lambda: self._browse("elita"))
        tk.Label(src, textvariable=self._src_status, font=SMALL, bg=c.BG_PRIMARY,
                 fg=c.FG_SECONDARY, anchor=tk.W).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(3, 0)
        )

        # ── strips (top-bottom: Pentacam above ELITA) ─────────────────
        strips = tk.Frame(body, **F)
        strips.pack(fill=tk.X, padx=20, pady=(12, 0))
        strips.columnconfigure(0, weight=1)

        self._strip_canvases: Dict[str, tk.Canvas] = {}
        for row, (key, title) in enumerate(
            (("reference", "Pentacam — unwrapped iris annulus"),
             ("target", "ELITA — unwrapped iris annulus"))
        ):
            tk.Label(strips, text=title, font=H2, bg=c.BG_PRIMARY,
                     fg=c.FG_PRIMARY, anchor=tk.W).grid(
                row=row * 2, column=0, sticky=tk.W, pady=(8 if row else 0, 2)
            )
            canvas = tk.Canvas(
                strips, width=_STRIP_CANVAS_W, height=_STRIP_CANVAS_H,
                bg="#0a0a0a", highlightthickness=1, highlightbackground=c.BORDER,
            )
            canvas.grid(row=row * 2 + 1, column=0, sticky=tk.EW)
            canvas._photo = None
            canvas.bind(
                "<Configure>",
                lambda e, k=key: self._on_strip_resize(k, e.width, e.height),
            )
            self._strip_canvases[key] = canvas

        tk.Label(
            strips,
            text="Daugman normalisation of the full iris annulus "
                 "(pupil boundary → limbus boundary) — no iris detail is "
                 "cropped or discarded.  Markers: strip features "
                 "(green = conf ≥ 0.60, amber ≥ 0.30, grey below).",
            font=SMALL, bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky=tk.W, pady=(5, 0))

        # ── cyclotorsion + readouts ───────────────────────────────────
        ro = tk.Frame(body, bg=c.BG_SECONDARY, highlightthickness=1,
                      highlightbackground=c.BORDER)
        ro.pack(fill=tk.X, padx=20, pady=(12, 0))

        self._cyc_var = tk.StringVar(value="Cyclotorsion: awaiting analysis")
        self._cyc_label = tk.Label(
            ro, textvariable=self._cyc_var, font=("Segoe UI", 11, "bold"),
            bg=c.BG_SECONDARY, fg=c.FG_PRIMARY, anchor=tk.W, wraplength=w - 80,
        )
        self._cyc_label.pack(fill=tk.X, padx=12, pady=(8, 2))
        self._cyc_detail_var = tk.StringVar(value="")
        tk.Label(ro, textvariable=self._cyc_detail_var, font=SMALL,
                 bg=c.BG_SECONDARY, fg=c.FG_SECONDARY, anchor=tk.W,
                 justify=tk.LEFT, wraplength=w - 80).pack(
            fill=tk.X, padx=12, pady=(0, 4)
        )

        grid = tk.Frame(ro, bg=c.BG_SECONDARY)
        grid.pack(fill=tk.X, padx=12, pady=(2, 8))
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        self._readout_cells: Dict[str, tk.Label] = {}

        def _row(r: int, label: str) -> None:
            tk.Label(grid, text=label, font=SMALL, bg=c.BG_SECONDARY,
                     fg=c.FG_SECONDARY, anchor=tk.W).grid(
                row=r, column=0, sticky=tk.W, pady=1, padx=(0, 10)
            )

        self._readout_cols = ("reference", "target")
        # column headers
        for col, title in enumerate(("Pentacam", "ELITA"), start=1):
            tk.Label(grid, text=title, font=H2, bg=c.BG_SECONDARY,
                     fg=c.FG_PRIMARY, anchor=tk.W).grid(
                row=0, column=col, sticky=tk.W, pady=1, padx=(0, 8)
            )

        readout_rows = (
            ("det", "Pupil / limbus conf."),
            ("roi", "Iris ROI"),
            ("strip", "Unwrapped strip"),
            ("iris", "Detector iris features"),
            ("sfeat", "Strip features"),
        )
        for r, (key, label) in enumerate(readout_rows, start=1):
            _row(r, label)
            for col, cname in enumerate(self._readout_cols, start=1):
                lbl = tk.Label(
                    grid, text="—", font=SMALL, bg=c.BG_SECONDARY,
                    fg=c.FG_PRIMARY, anchor=tk.W,
                )
                lbl.grid(row=r, column=col, sticky=tk.W, pady=1, padx=(0, 8))
                self._readout_cells[f"{key}:{cname}"] = lbl

        # ── validation ────────────────────────────────────────────────
        val = tk.Frame(body, **F)
        val.pack(fill=tk.X, padx=20, pady=(12, 12))
        self._val_header = tk.StringVar(value="Validation — not run yet")
        tk.Label(val, textvariable=self._val_header, font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY, anchor=tk.W).pack(anchor=tk.W)
        self._val_text = tk.Text(
            val, height=7, font=("Consolas", 9), bg=c.BG_INPUT,
            fg=c.FG_PRIMARY, relief=tk.FLAT, bd=0, highlightthickness=1,
            highlightbackground=c.BORDER, highlightcolor=c.ACCENT,
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self._val_text.pack(fill=tk.X, pady=(5, 0))
        self._val_text.tag_configure("pass", foreground=c.SURGICAL)
        self._val_text.tag_configure("fail", foreground=c.INSUFFICIENT)
        self._val_text.tag_configure("dim", foreground=c.FG_SECONDARY)

        # ── status (fixed) ────────────────────────────────────────────
        tk.Frame(win, bg=c.BORDER, height=1).pack(
            fill=tk.X, padx=20, pady=(8, 4)
        )
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(win, textvariable=self._status_var, font=SMALL,
                 bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W).pack(
            fill=tk.X, padx=20, pady=(0, 10)
        )

        win.protocol("WM_DELETE_WINDOW", self.close)
        self._paint_toggles()
        self._bind_wheel(win, scroller)

    def _bind_wheel(self, widget: tk.Misc, scroller: tk.Canvas) -> None:
        """Route the mouse wheel to the body scroller from anywhere inside."""
        widget.bind(
            "<MouseWheel>",
            lambda e: self._scroll_by(scroller, -int(e.delta / 120)),
            add="+",
        )
        for child in widget.winfo_children():
            self._bind_wheel(child, scroller)

    @staticmethod
    def _scroll_by(scroller: tk.Canvas, units: int) -> None:
        try:
            scroller.yview_scroll(units, "units")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # small UI helpers
    # ------------------------------------------------------------------
    def _default_eye(self) -> str:
        try:
            eye = str(self.app._selected_eye_var.get())
            return eye if eye in VALID_EYES else "OD"
        except Exception:
            return "OD"

    def _paint_toggles(self) -> None:
        c = self._colors
        active, idle = c.BTN_ACTION, c.BTN_BG
        for eye, btn in self._eye_btns.items():
            on = self._eye_var.get() == eye
            btn.config(bg=active if on else idle,
                       fg="#FFFFFF" if on else c.FG_PRIMARY)
        for phase, btn in self._phase_btns.items():
            on = self._phase_var.get() == phase
            btn.config(bg=active if on else idle,
                       fg="#FFFFFF" if on else c.FG_PRIMARY)

    def _set_status(self, text: str) -> None:
        try:
            self._status_var.set(text)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # eye / phase / patient routing
    # ------------------------------------------------------------------
    def _elita_key(self) -> Tuple[str, str]:
        """The ELITA target is per (eye, phase) — pre/post-dock, OD/OS."""
        return (self._eye_var.get(), self._phase_var.get())

    def _set_eye(self, eye: str) -> None:
        if eye not in VALID_EYES or eye == self._eye_var.get():
            return
        # Remember the ELITA still of the eye we are leaving, then re-point
        # both fields at the new eye.
        self._elita_paths[self._elita_key()] = self._elita_var.get().strip()
        self._eye_var.set(eye)
        self._paint_toggles()
        self._sync_from_patient()
        self._show_cached_or_clear()

    def _set_phase(self, phase: str) -> None:
        if phase not in PHASES or phase == self._phase_var.get():
            return
        # Remember the current ELITA path for (eye, phase) we are leaving.
        self._elita_paths[self._elita_key()] = self._elita_var.get().strip()
        self._phase_var.set(phase)
        self._paint_toggles()
        # Restore the target image captured for the phase we are entering.
        self._elita_var.set(self._elita_paths.get(self._elita_key(), "").strip())
        self._load_images_silently()
        self._show_cached_or_clear()

    def _sync_from_patient(self) -> None:
        """OD/OS-aware: pull that eye's Pentacam reference + recorded ROI box.

        The reference always belongs to the selected eye — a still is never
        silently carried across from the other eye, because the eye label is
        written into the saved cyclotorsion record.
        """
        eye = self._eye_var.get()
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        recorded = find_reference_image(patient_dir, eye) if patient_dir else None
        if recorded is not None:
            self._pentacam_paths[eye] = str(recorded)
            note = f"reference: {recorded.name}"
            if read_roi_norm(patient_dir, eye) is not None:
                note += " · recorded iris ROI available"
        elif patient_dir is None:
            note = (
                "reference: image browsed for this eye"
                if self._pentacam_paths.get(eye)
                else "no active patient — browse both images manually"
            )
        else:
            note = f"no {eye} Pentacam image recorded yet"
        # the reference is always this eye's still, never the other eye's
        self._pentacam_var.set(self._pentacam_paths.get(eye, ""))
        self._eye_phase_var.set(note)
        self._elita_var.set(self._elita_paths.get(self._elita_key(), ""))
        self._load_images_silently()

    def _show_cached_or_clear(self) -> None:
        """Restore the cached result for (eye, phase) or clear the panels."""
        cached = self._cache.get(self._cache_key())
        if cached is not None:
            self._cancel_inflight()
            self._apply(cached, from_cache=True)
            return
        self._analysis = None
        self._cancel_inflight()
        for key in ("reference", "target"):
            self._draw_strip(key, None, [])
        self._cyc_var.set("Cyclotorsion: awaiting analysis")
        self._cyc_detail_var.set("")
        for cell in self._readout_cells.values():
            cell.config(text="—")
        self._set_validation(None)
        self._set_status("Ready — press Analyse.")

    def _cancel_inflight(self) -> None:
        """Invalidate any in-flight worker and release the Analyse button.

        Bumping the token turns a late result into a no-op.  The busy flag and
        the button must be reset *here* too, because a superseded worker now
        returns early and no longer owns that UI state — otherwise cancelling
        during a run would leave Analyse disabled forever.
        """
        self._token += 1
        self._busy = False
        try:
            self._analyse_btn.config(state=tk.NORMAL)
        except tk.TclError:
            pass

    def _cache_key(self) -> Tuple[str, str, str, str]:
        return (
            self._eye_var.get(),
            self._phase_var.get(),
            self._pentacam_var.get().strip(),
            self._elita_var.get().strip(),
        )

    # ------------------------------------------------------------------
    # image loading
    # ------------------------------------------------------------------
    def _browse(self, which: str) -> None:
        initial = None
        var = self._pentacam_var if which == "pentacam" else self._elita_var
        try:
            start = var.get().strip() or str(
                getattr(self.app, "_active_patient_dir", "") or ""
            )
            if start:
                initial = start
        except Exception:
            initial = None

        path = filedialog.askopenfilename(
            title="Select Pentacam image" if which == "pentacam"
            else "Select ELITA image",
            filetypes=list(IMAGE_FILETYPES),
            initialdir=initial,
        )
        if not path:
            return
        from pupil_tracking.cyclotorsion.pentacam_loader import load_pentacam_image
        clean_path = path.strip().strip('"').strip("'")
        loaded_img = load_pentacam_image(clean_path)
        if loaded_img is None:
            messagebox.showerror(
                "Image Load Error",
                f"Failed to load image from:\n{clean_path}\n\n"
                "The file could not be decoded or is corrupted.\n"
                "Supported formats: .bmp, .jpg, .jpeg, .png, .tif, .dcm, .pdf",
            )
            return
        var.set(clean_path)
        if which == "elita":
            self._elita_paths[self._elita_key()] = clean_path
        else:
            self._pentacam_paths[self._eye_var.get()] = clean_path
        self._load_images_silently()
        self._maybe_auto_analyse()

    def _load_images_silently(self) -> None:
        """(Re)load the two stills without running detection."""
        from pupil_tracking.cyclotorsion.pentacam_loader import load_pentacam_image

        pentacam_path = self._pentacam_var.get().strip().strip('"').strip("'")
        elita_path = self._elita_var.get().strip().strip('"').strip("'")
        self._images: Dict[str, Optional[np.ndarray]] = {
            "reference": load_pentacam_image(pentacam_path) if pentacam_path else None,
            "target": load_pentacam_image(elita_path) if elita_path else None,
        }
        parts = []
        for key, label in (("reference", "Pentacam"), ("target", "ELITA")):
            img = self._images.get(key)
            parts.append(f"{label}: {img.shape[1]}x{img.shape[0]}" if img is not None
                         else f"{label}: —")
        self._src_status.set("  |  ".join(parts))

    def _maybe_auto_analyse(self) -> None:
        """Analyse as soon as both stills are available (doctor-friendly)."""
        if (
            self._pentacam_var.get().strip()
            and self._elita_var.get().strip()
            and self._cache.get(self._cache_key()) is None
        ):
            self._analyse()

    # ------------------------------------------------------------------
    # analysis (worker thread)
    # ------------------------------------------------------------------
    def _get_iris_detector(self) -> Any:
        """Return the cached IrisFeatureDetector (built once, then reused).

        Reusing one instance keeps the torch-backed RL agent warm across the
        Pentacam and ELITA analyses instead of re-loading the model twice.
        Returns None if the iris module cannot be built — analysis still runs,
        it just reports no detector-level iris features.
        """
        if self._iris_detector is not None:
            return self._iris_detector
        try:
            from pupil_tracking.iris.detect import IrisFeatureDetector

            self._iris_detector = IrisFeatureDetector()
        except Exception as exc:
            self._set_status(f"Iris feature detector unavailable: {exc}")
            self._iris_detector = None
        return self._iris_detector

    def _analyse(self) -> None:
        if self._busy:
            return
        pentacam_path = self._pentacam_var.get().strip()
        elita_path = self._elita_var.get().strip()
        if not pentacam_path or not elita_path:
            messagebox.showinfo(
                "Static Mode",
                "Select both a Pentacam and an ELITA image first.",
                parent=self._win,
            )
            return

        self._load_images_silently()
        pentacam = self._images.get("reference")
        elita = self._images.get("target")
        if pentacam is None or elita is None:
            missing = "Pentacam" if pentacam is None else "ELITA"
            messagebox.showerror(
                "Static Mode",
                f"The {missing} image could not be read.\n"
                "Supported: JPG, JPEG, PNG, PDF.",
                parent=self._win,
            )
            return

        eye = self._eye_var.get()
        phase = self._phase_var.get()
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        roi_norm = None
        if patient_dir is not None:
            roi_norm = read_roi_norm(patient_dir, eye)

        detector = getattr(self.app, "_detector", None)
        if detector is None:
            self._set_status("Detector unavailable — check the ML backend.")
            return
        iris_detector = self._get_iris_detector()

        self._token += 1
        token = self._token
        self._busy = True
        self._analyse_btn.config(state=tk.DISABLED)
        self._set_status(
            f"Analysing {PHASE_LABELS[phase]} · {eye} "
            f"(detect + unwrap + features on both images)…"
        )

        def work() -> None:
            try:
                result = analyse_static_pair(
                    pentacam, elita,
                    eye=eye,
                    phase=phase,
                    detector=detector,
                    iris_detector=iris_detector,
                    pentacam_path=pentacam_path,
                    elita_path=elita_path,
                    flip_pentacam=True,
                    flip_elita=False,
                    roi_norm=roi_norm,
                )
            except Exception as exc:  # never let a worker die silently
                result = None
                err = repr(exc)
            else:
                err = ""
            self.app.root.after(0, lambda: self._on_done(token, result, err))

        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, token: int, result: Optional[StaticAnalysis],
                 err: str) -> None:
        if token != self._token:
            return  # superseded — the newer run owns the busy/button state
        self._busy = False
        try:
            self._analyse_btn.config(state=tk.NORMAL)
        except tk.TclError:
            return
        if result is None:
            self._set_status(f"Analysis failed: {err}")
            messagebox.showerror("Static Mode", f"Analysis failed:\n{err}",
                                 parent=self._win)
            return
        self._cache[self._cache_key()] = result
        self._apply(result)

    # ------------------------------------------------------------------
    # applying / rendering
    # ------------------------------------------------------------------
    def _apply(self, result: StaticAnalysis, from_cache: bool = False) -> None:
        self._analysis = result
        ref, tgt = result.reference, result.target

        self._photos = {}  # drop PhotoImage refs from the previous render
        for key, src in (("reference", ref), ("target", tgt)):
            self._draw_strip(key, src.strip, list(src.strip_feature_list))

        self._update_readout(result)
        self._set_validation(result)

        cyc = result.cyclotorsion_valid
        sign = "+" if result.angle_deg >= 0 else "\u2212"
        self._cyc_var.set(
            f"Cyclotorsion  {sign}{abs(result.angle_deg):.2f}\u00b0   \u00b7   "
            f"score {result.score:.3f}   \u00b7   "
            f"{'VALID' if cyc else 'NOT ACCEPTED'}"
        )
        try:
            self._cyc_label.config(
                fg=self._colors.SURGICAL if cyc else self._colors.INSUFFICIENT
            )
        except tk.TclError:
            pass
        self._cyc_detail_var.set(
            f"Pentacam (reference, 180° auto-flipped) → ELITA (target)   ·   "
            f"eye {result.eye}   ·   {PHASE_LABELS[result.phase]}   ·   "
            f"engine: {result.cyclotorsion_reason or 'ok'}   ·   "
            f"{result.n_valid}/{result.n_total} px compared   ·   "
            f"search ±{result.angle_range_deg:g}°   ·   "
            f"took {result.elapsed_ms:.0f} ms"
            + ("   ·   (cached)" if from_cache else "")
        )
        self._set_status(
            f"{PHASE_LABELS[result.phase]} · {result.eye}: "
            f"{result.summary()}  ·  angle {result.angle_deg:+.2f}°  ·  "
            f"score {result.score:.3f}"
        )

    def _draw_strip(self, key: str, strip: Optional[np.ndarray],
                    features: List[Any]) -> None:
        """Remember *strip* for *key* and render it at the canvas' real size."""
        self._last_strip[key] = (strip, list(features))
        canvas = self._strip_canvases.get(key)
        if canvas is None:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        # before the window is mapped winfo_* is 1 — fall back to the request
        if width <= 1:
            width = int(canvas.cget("width"))
        if height <= 1:
            height = int(canvas.cget("height"))
        self._render_strip(key, strip, list(features), width, height)

    def _on_strip_resize(self, key: str, width: int, height: int) -> None:
        """Re-render a strip when the window resizes (the canvas is sticky EW)."""
        if (width, height) == self._strip_drawn.get(key):
            return
        last = self._last_strip.get(key)
        if last is None:
            return
        strip, features = last
        self._render_strip(key, strip, features, width, height)

    def _render_strip(self, key: str, strip: Optional[np.ndarray],
                      features: List[Any], width: int, height: int) -> None:
        canvas = self._strip_canvases.get(key)
        if canvas is None or width <= 1 or height <= 1:
            return
        self._strip_drawn[key] = (width, height)
        canvas.delete("all")

        if strip is None:
            canvas.create_text(
                width // 2, height // 2, text="no strip — run Analyse",
                fill=self._colors.FG_TERTIARY, font=("Segoe UI", 10),
            )
            canvas._photo = None
            self._photos.pop(key, None)
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            canvas.create_text(width // 2, height // 2, text="PIL unavailable",
                               fill=self._colors.INSUFFICIENT)
            return

        arr = np.asarray(strip)
        if arr.ndim == 2:
            rgb = arr
        else:
            import cv2

            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        sh, sw = rgb.shape[0], rgb.shape[1]
        avail_w = width - 10
        avail_h = height - _RULER_H - 6
        if avail_w < 8 or avail_h < 8:
            return
        scale = min(avail_w / float(sw), avail_h / float(sh))
        disp_w = max(1, int(round(sw * scale)))
        disp_h = max(1, int(round(sh * scale)))
        x0 = (width - disp_w) // 2
        y0 = _RULER_H + max(0, (avail_h - disp_h) // 2)

        photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(
            (disp_w, disp_h), Image.NEAREST if scale > 1 else Image.BILINEAR
        ))
        canvas.create_image(x0, y0, image=photo, anchor="nw")
        canvas._photo = photo
        self._photos[key] = photo

        # ── angle ruler (0° → 360° across the strip) ──────────────────
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + int(round(frac * (disp_w - 1)))
            deg = int(round(frac * 360.0))
            anchor = tk.W if frac == 0.0 else (tk.E if frac == 1.0 else tk.N)
            canvas.create_text(
                max(2, min(width - 2, x)), 9, text=f"{deg}°",
                fill=self._colors.FG_SECONDARY, font=("Segoe UI", 8),
                anchor=anchor,
            )
            canvas.create_line(x, _RULER_H - 3, x, height - 3,
                               fill=self._colors.BORDER, dash=(2, 4))

        # ── strip-feature markers ─────────────────────────────────────
        sx = disp_w / float(sw)
        sy = disp_h / float(sh)
        for feat in features:
            try:
                fx = x0 + float(feat.column) * sx + sx / 2.0
                fy = y0 + float(feat.row) * sy + sy / 2.0
                conf = float(getattr(feat, "confidence", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            colour = (self._colors.SURGICAL if conf >= _CONF_HIGH
                      else self._colors.RESEARCH if conf >= _CONF_MID
                      else self._colors.FG_TERTIARY)
            r = 4
            canvas.create_oval(fx - r, fy - r, fx + r, fy + r,
                               outline=colour, width=1)

    # ------------------------------------------------------------------
    # readout + validation
    # ------------------------------------------------------------------
    def _update_readout(self, result: StaticAnalysis) -> None:
        c = self._colors
        for cname, src in (("reference", result.reference),
                           ("target", result.target)):
            self._readout_cells[f"det:{cname}"].config(
                text=(f"{src.pupil_confidence:.2f} / {src.limbus_confidence:.2f}"
                      if src.detected else "not detected")
            )
            if src.roi_valid:
                self._readout_cells[f"roi:{cname}"].config(
                    text=f"{src.roi_source or 'ok'}, r {src.pupil_radius_px:.0f} / "
                         f"{src.limbus_radius_px:.0f} px"
                )
            else:
                self._readout_cells[f"roi:{cname}"].config(
                    text=f"invalid — {src.roi_reason or 'n/a'}"
                )
            self._readout_cells[f"strip:{cname}"].config(
                text=(f"{src.strip_height} × {src.strip_width}"
                      if src.strip is not None else "none")
            )
            self._readout_cells[f"iris:{cname}"].config(
                text=f"{src.iris_features} (conf {src.iris_feature_confidence:.2f}, "
                     f"coverage {src.iris_feature_coverage:.2f}, "
                     f"usable {src.iris_usable_fraction:.2f})"
            )
            self._readout_cells[f"sfeat:{cname}"].config(
                text=f"{src.strip_features} (conf "
                     f"{src.strip_feature_confidence:.2f} of "
                     f"{src.strip_feature_candidates} candidates)"
            )
        # colour the detection cells by success
        for cname in ("reference", "target"):
            det_ok = (result.reference if cname == "reference"
                      else result.target).detected
            self._readout_cells[f"det:{cname}"].config(
                fg=c.SURGICAL if det_ok else c.INSUFFICIENT
            )

    def _set_validation(self, result: Optional[StaticAnalysis]) -> None:
        text = self._val_text
        text.config(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        if result is None:
            text.insert(tk.END, "No analysis yet — press Analyse.\n", "dim")
            self._val_header.set("Validation — not run yet")
        else:
            for check in result.checks:
                tag = "pass" if check.passed else "fail"
                line = f"[{check.status}]  {check.name}"
                if check.detail:
                    line += f"   {check.detail}"
                text.insert(tk.END, line + "\n", tag)
            passed, total = result.passed_count()
            self._val_header.set(
                f"Validation — {passed}/{total} checks passed  ·  "
                f"{'PASS' if result.valid else 'FAIL'}   "
                f"(≥{result.min_features} features, score ≥ {result.min_score:.2f}, "
                f"|angle| ≤ {result.max_abs_angle_deg:g}°)"
            )
        text.config(state=tk.DISABLED)

    def _validate_now(self) -> None:
        """Re-run the validation function on the current analysis."""
        if self._analysis is None:
            self._set_status("Nothing to validate yet — run Analyse first.")
            return
        self._analysis.checks = validate_analysis(self._analysis)
        self._set_validation(self._analysis)
        passed, total = self._analysis.passed_count()
        self._set_status(
            f"Validation re-run: {passed}/{total} checks passed "
            f"→ {'PASS' if self._analysis.valid else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # saving
    # ------------------------------------------------------------------
    def _require_analysis(self) -> Optional[StaticAnalysis]:
        if self._analysis is None:
            messagebox.showinfo(
                "Static Mode",
                "Run Analyse before saving cyclotorsion details.",
                parent=self._win,
            )
            return None
        return self._analysis

    def _save(self) -> None:
        analysis = self._require_analysis()
        if analysis is None:
            return
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        if patient_dir is None:
            messagebox.showwarning(
                "Static Mode",
                "No active patient folder.\n"
                "Add a patient first, or use Save As…",
                parent=self._win,
            )
            return
        try:
            path = save_analysis(analysis, patient_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Static Mode", f"Could not save:\n{exc}",
                                 parent=self._win)
            return
        self._set_status(f"Saved {analysis.phase} cyclotorsion → {path}")

    def _save_as(self) -> None:
        analysis = self._require_analysis()
        if analysis is None:
            return
        default = f"static_cyclotorsion_{analysis.eye}_{analysis.phase}.json"
        path = filedialog.asksaveasfilename(
            title="Save Cyclotorsion Details As",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile=default,
        )
        if not path:
            return
        try:
            # patient_dir is only recorded as a label when saving explicitly,
            # so an empty string honestly says "no active patient".
            save_analysis(
                analysis,
                getattr(self.app, "_active_patient_dir", None) or "",
                path=path,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Static Mode", f"Could not save:\n{exc}",
                                 parent=self._win)
            return
        self._set_status(f"Saved a copy of the {analysis.phase} details → {path}")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._token += 1  # orphan any in-flight worker
        win = self._win
        self._win = None
        self._analysis = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
        try:
            self.app._static_mode_open = False
            if hasattr(self.app, "_update_patient_controls"):
                self.app._update_patient_controls()
        except Exception:
            pass


def open_static_mode(app: Any) -> None:
    """Open (or raise) the Static Mode window for *app*."""
    win = getattr(app, "_static_mode_window", None)
    if win is None or getattr(win, "_win", None) is None:
        win = StaticModeWindow(app)
        app._static_mode_window = win
        app._static_mode_open = True
    win.open()


__all__ = ["StaticModeWindow", "open_static_mode"]
