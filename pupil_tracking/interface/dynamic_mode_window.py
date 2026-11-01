"""dynamic_mode_window.py — the Dynamic (Live) Mode window.

A separate top-level window that continuously unwraps the live camera's iris
annulus (the same reprojection Static Mode uses for stills) and compares it
with the patient's Pentacam baseline still, reporting the ongoing cyclotorsion
angle and detection confidence frame after frame.

    live camera  ──▶  strip per frame  ──▶  correlation vs Pentacam strip
                                              │
                                              ▼
                                  angle ± score ± confidence

All measurement logic lives in
:mod:`pupil_tracking.cyclotorsion.dynamic_mode`; this module is display +
wiring only.  It never touches the live centration / pupil / limbus pipeline
and never feeds surgical control — the estimate is for validation and
monitoring.

The heavy work (unwrap + correlation, and building the baseline) runs on a
worker thread; Tk widgets are only ever touched on the main thread.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.cyclotorsion.dynamic_mode import (
    LiveFrameAnalysis,
    ReferenceStrip,
    analyse_live_frame,
    build_reference_strip,
    save_sample,
)
from pupil_tracking.cyclotorsion.static_mode import (
    IMAGE_FILETYPES,
    VALID_EYES,
    find_reference_image,
    read_roi_norm,
)

#: Strip display size (canvas box) — the strip itself is 64 x 512.
_STRIP_CANVAS_W = 780
_STRIP_CANVAS_H = 120
_RULER_H = 18

#: How long the monitor waits between frame grabs (seconds).
_POLL_INTERVAL_S = 0.02

#: Rolling live log length (in samples).
_LOG_LEN = 200
_LOG_SHOWN = 12


class DynamicModeWindow:
    """Live cyclotorsion monitor for one app instance."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self._colors = app._colors

        self._win: Optional[tk.Toplevel] = None
        self._photos: Dict[str, Any] = {}

        self._eye = "OD"
        self._pentacam_path = ""
        self._load_images: Dict[str, Optional[np.ndarray]] = {}

        self._reference: Optional[ReferenceStrip] = None
        self._ref_building = False
        self._baseline_image: Optional[np.ndarray] = None

        self._running = False
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_frame_token: Any = None
        self._frames = 0
        # Worker -> main-thread hand-off: the worker only ever *puts* a
        # result here; the main thread drains it on its own timer, so no Tk
        # call is ever made from the worker thread.
        self._result_queue: "queue.Queue[LiveFrameAnalysis]" = queue.Queue()
        self._pump_after_id: Optional[str] = None
        # Same hand-off for the one-shot Pentacam baseline build.
        self._ref_queue: "queue.Queue[ReferenceStrip]" = queue.Queue()

        self._samples: List[LiveFrameAnalysis] = []
        self._last_strip: Dict[str, Any] = {}
        self._strip_drawn: Dict[str, Tuple[int, int]] = {}
        # Display-only compensation marker: where the Pentacam reference's 0°
        # meridian currently maps to on the live strip.  Never fed anywhere.
        self._target_marker_deg: Optional[float] = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def open(self) -> None:
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
        self._rebuild_reference()

    def _build(self) -> None:
        c = self._colors
        win = tk.Toplevel(self.app.root)
        self._win = win
        win.title("Dynamic (Live) Mode — Cyclotorsion Monitoring")
        win.transient(self.app.root)
        win.configure(bg=c.BG_PRIMARY)

        try:
            from PIL import Image as _Img, ImageTk as _ITk
            import pathlib as _pl

            icon_path = _pl.Path(__file__).parent / "add-patient.png.png"
            icon = _ITk.PhotoImage(_Img.open(icon_path).resize((32, 32), _Img.LANCZOS))
            win.iconphoto(False, icon)
            win._icon_ref = icon
        except Exception:
            pass

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

        # ── header ────────────────────────────────────────────────────
        hdr = tk.Frame(win, **F)
        hdr.pack(fill=tk.X, padx=20, pady=(12, 2))
        tk.Label(hdr, text="Dynamic (Live) Mode — Cyclotorsion Monitoring",
                 font=H1, bg=c.BG_PRIMARY, fg=c.FG_PRIMARY).pack(anchor=tk.W)
        tk.Label(
            hdr,
            text="Every live frame is unwrapped from its own detected iris and "
                 "compared with the Pentacam baseline.  Display-only: nothing "
                 "here changes the centration / pupil / limbus pipeline.",
            font=SMALL, bg=c.BG_PRIMARY, fg=c.FG_SECONDARY,
        ).pack(anchor=tk.W, pady=(2, 0))

        # ── control bar ───────────────────────────────────────────────
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

        _btn("Close", self.close).pack(side=tk.RIGHT)
        _btn("Save As…", self._save_as).pack(side=tk.RIGHT, padx=(0, 6))
        _btn("Save", self._save).pack(side=tk.RIGHT, padx=(0, 6))
        _btn("Reset log", self._reset_log).pack(side=tk.RIGHT, padx=(0, 6))
        self._monitor_btn = _btn("▶ Start monitoring", self._toggle_monitor, primary=True)
        self._monitor_btn.pack(side=tk.RIGHT, padx=(0, 6))

        tk.Label(ctl, text="Eye", font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY).pack(side=tk.LEFT, padx=(0, 8))
        eye_btns: Dict[str, tk.Button] = {}
        for eye in VALID_EYES:
            b = tk.Button(ctl, text=eye, font=BODY, width=4, relief=tk.FLAT,
                          bd=0, cursor="hand2", command=lambda e=eye: self._set_eye(e))
            b.pack(side=tk.LEFT, padx=(0, 4))
            eye_btns[eye] = b
        self._eye_btns = eye_btns

        self._eye_phase_var = tk.StringVar(value="")
        tk.Label(win, textvariable=self._eye_phase_var, font=SMALL,
                 bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W).pack(
            fill=tk.X, padx=20, pady=(6, 0))
        tk.Frame(win, bg=c.BORDER, height=1).pack(fill=tk.X, padx=20, pady=(8, 0))

        # ── scrollable body ───────────────────────────────────────────
        shell = tk.Frame(win, **F)
        shell.pack(fill=tk.BOTH, expand=True)
        vbar = tk.Scrollbar(shell, orient=tk.VERTICAL, bg=c.BG_SECONDARY,
                            troughcolor=c.BG_PRIMARY, activebackground=c.BTN_HOVER,
                            relief=tk.FLAT, bd=0, highlightthickness=0)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroller = tk.Canvas(shell, bg=c.BG_PRIMARY, highlightthickness=0,
                             yscrollcommand=vbar.set)
        scroller.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vbar.config(command=scroller.yview)

        body = tk.Frame(scroller, **F)
        body_id = scroller.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda _e: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>",
                      lambda e: scroller.itemconfigure(body_id, width=e.width))
        self._scroller = scroller

        # ── reference source ──────────────────────────────────────────
        src = tk.Frame(body, **F)
        src.pack(fill=tk.X, padx=20, pady=(10, 0))
        src.columnconfigure(1, weight=1)

        tk.Label(src, text="Pentacam baseline", font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY, anchor=tk.W, width=20).grid(
            row=0, column=0, sticky=tk.W, pady=3, padx=(0, 10))
        self._pentacam_var = tk.StringVar()
        tk.Entry(src, textvariable=self._pentacam_var, font=BODY,
                 readonlybackground=c.BG_INPUT, fg=c.FG_PRIMARY,
                 insertbackground=c.FG_PRIMARY, relief=tk.FLAT, bd=0,
                 highlightthickness=1, highlightbackground=c.BORDER,
                 highlightcolor=c.ACCENT).grid(
            row=0, column=1, sticky=tk.EW, pady=3, ipady=4)
        tk.Button(src, text="Browse…", font=BODY, bg=c.BTN_BG, fg=c.FG_PRIMARY,
                  activebackground=c.BTN_HOVER, activeforeground=c.FG_PRIMARY,
                  relief=tk.FLAT, bd=0, padx=12, pady=4, cursor="hand2",
                  command=self._browse_reference).grid(
            row=0, column=2, sticky=tk.E, pady=3, padx=(10, 0))

        self._src_status = tk.StringVar(value="No baseline selected")
        tk.Label(src, textvariable=self._src_status, font=SMALL, bg=c.BG_PRIMARY,
                 fg=c.FG_SECONDARY, anchor=tk.W).grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(3, 0))

        # ── strips (top-bottom: Pentacam above live ELITA) ────────────
        strips = tk.Frame(body, **F)
        strips.pack(fill=tk.X, padx=20, pady=(12, 0))
        strips.columnconfigure(0, weight=1)

        self._strip_canvases: Dict[str, tk.Canvas] = {}
        for row, (key, title) in enumerate(
            (("reference", "Pentacam — baseline iris annulus"),
             ("target", "ELITA live — current frame iris annulus"))
        ):
            tk.Label(strips, text=title, font=H2, bg=c.BG_PRIMARY,
                     fg=c.FG_PRIMARY, anchor=tk.W).grid(
                row=row * 2, column=0, sticky=tk.W, pady=(8 if row else 0, 2))
            canvas = tk.Canvas(strips, width=_STRIP_CANVAS_W, height=_STRIP_CANVAS_H,
                               bg="#0a0a0a", highlightthickness=1,
                               highlightbackground=c.BORDER)
            canvas.grid(row=row * 2 + 1, column=0, sticky=tk.EW)
            canvas._photo = None
            canvas.bind("<Configure>",
                        lambda e, k=key: self._on_strip_resize(k, e.width, e.height))
            self._strip_canvases[key] = canvas

        tk.Label(
            strips,
            text="Daugman normalisation of the full iris annulus "
                 "(pupil boundary → limbus boundary) — no iris detail is "
                 "cropped or discarded.  Both strips share column 0 = 0°.",
            font=SMALL, bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W,
            justify=tk.LEFT,
        ).grid(row=4, column=0, sticky=tk.W, pady=(5, 0))

        # ── live readout ──────────────────────────────────────────────
        ro = tk.Frame(body, bg=c.BG_SECONDARY, highlightthickness=1,
                      highlightbackground=c.BORDER)
        ro.pack(fill=tk.X, padx=20, pady=(12, 0))

        self._cyc_var = tk.StringVar(value="Cyclotorsion: awaiting the live feed")
        self._cyc_label = tk.Label(
            ro, textvariable=self._cyc_var, font=("Segoe UI", 11, "bold"),
            bg=c.BG_SECONDARY, fg=c.FG_PRIMARY, anchor=tk.W, wraplength=w - 80,
        )
        self._cyc_label.pack(fill=tk.X, padx=12, pady=(8, 2))
        self._cyc_detail_var = tk.StringVar(value="")
        tk.Label(ro, textvariable=self._cyc_detail_var, font=SMALL,
                 bg=c.BG_SECONDARY, fg=c.FG_SECONDARY, anchor=tk.W,
                 justify=tk.LEFT, wraplength=w - 80).pack(
            fill=tk.X, padx=12, pady=(0, 4))

        grid = tk.Frame(ro, bg=c.BG_SECONDARY)
        grid.pack(fill=tk.X, padx=12, pady=(2, 8))
        grid.columnconfigure(1, weight=1)

        self._cells: Dict[str, tk.Label] = {}
        for r, (key, label) in enumerate((
            ("pupil", "Pupil confidence"),
            ("limbus", "Limbus confidence"),
            ("quality", "Frame quality"),
            ("roi", "Iris ROI"),
            ("strip", "Strip size"),
            ("angle", "Cyclotorsion angle"),
            ("score", "Correlation score"),
        )):
            tk.Label(grid, text=label, font=SMALL, bg=c.BG_SECONDARY,
                     fg=c.FG_SECONDARY, anchor=tk.W).grid(
                row=r, column=0, sticky=tk.W, pady=1, padx=(0, 12))
            lbl = tk.Label(grid, text="—", font=SMALL, bg=c.BG_SECONDARY,
                           fg=c.FG_PRIMARY, anchor=tk.W)
            lbl.grid(row=r, column=1, sticky=tk.W, pady=1)
            self._cells[key] = lbl

        # ── live log ──────────────────────────────────────────────────
        log = tk.Frame(body, **F)
        log.pack(fill=tk.X, padx=20, pady=(12, 12))
        tk.Label(log, text="Recent live frames", font=H2, bg=c.BG_PRIMARY,
                 fg=c.FG_PRIMARY, anchor=tk.W).pack(anchor=tk.W)
        self._log_text = tk.Text(
            log, height=7, font=("Consolas", 9), bg=c.BG_INPUT, fg=c.FG_PRIMARY,
            relief=tk.FLAT, bd=0, highlightthickness=1,
            highlightbackground=c.BORDER, highlightcolor=c.ACCENT,
            state=tk.DISABLED, wrap=tk.NONE,
        )
        self._log_text.pack(fill=tk.X, pady=(5, 0))
        self._log_text.tag_configure("ok", foreground=c.SURGICAL)
        self._log_text.tag_configure("warn", foreground=c.RESEARCH)
        self._log_text.tag_configure("bad", foreground=c.INSUFFICIENT)
        self._log_text.tag_configure("dim", foreground=c.FG_SECONDARY)

        # ── status ────────────────────────────────────────────────────
        tk.Frame(win, bg=c.BORDER, height=1).pack(fill=tk.X, padx=20, pady=(8, 4))
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(win, textvariable=self._status_var, font=SMALL,
                 bg=c.BG_PRIMARY, fg=c.FG_SECONDARY, anchor=tk.W).pack(
            fill=tk.X, padx=20, pady=(0, 10))

        win.protocol("WM_DELETE_WINDOW", self.close)
        self._paint_toggles()
        self._bind_wheel(win, scroller)

    def _bind_wheel(self, widget: tk.Misc, scroller: tk.Canvas) -> None:
        widget.bind("<MouseWheel>",
                    lambda e: self._scroll_by(scroller, -int(e.delta / 120)), add="+")
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
            on = self._eye == eye
            btn.config(bg=active if on else idle,
                       fg="#FFFFFF" if on else c.FG_PRIMARY)

    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    def _set_eye(self, eye: str) -> None:
        if eye not in VALID_EYES or eye == self._eye:
            return
        self._eye = eye
        self._paint_toggles()
        self._sync_from_patient()
        self._rebuild_reference()

    # ------------------------------------------------------------------
    # reference baseline
    # ------------------------------------------------------------------
    def _sync_from_patient(self) -> None:
        eye = self._eye
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        recorded = find_reference_image(patient_dir, eye) if patient_dir else None
        if recorded is not None:
            self._pentacam_path = str(recorded)
            note = f"{eye} reference: {recorded.name}"
        elif patient_dir is None:
            self._pentacam_path = ""
            note = "no active patient — browse a Pentacam image"
        else:
            self._pentacam_path = ""
            note = f"no {eye} Pentacam image recorded yet"
        self._pentacam_var.set(self._pentacam_path)
        self._eye_phase_var.set(note)

    def _browse_reference(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Pentacam baseline image",
            filetypes=list(IMAGE_FILETYPES),
            initialdir=self._pentacam_path or str(
                getattr(self.app, "_active_patient_dir", "") or ""),
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
        self._pentacam_path = clean_path
        self._pentacam_var.set(clean_path)
        self._rebuild_reference()

    def _rebuild_reference(self) -> None:
        """Load + build the Pentacam baseline strip (off-thread detection)."""
        from pupil_tracking.cyclotorsion.pentacam_loader import load_pentacam_image

        path = self._pentacam_var.get().strip().strip('"').strip("'")
        self._baseline_image = load_pentacam_image(path) if path else None
        self._reference = None
        self._draw_strip("reference", None)
        if self._baseline_image is None:
            if path:
                self._src_status.set("Failed to load baseline image")
                self._set_status(f"Error: Failed to load image {path}")
            else:
                self._src_status.set("No baseline selected")
                self._set_status("No Pentacam baseline selected.")
            return

        detector = getattr(self.app, "_detector", None)
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        roi_norm = read_roi_norm(patient_dir, self._eye) if patient_dir else None
        image = self._baseline_image
        self._src_status.set("Building baseline…")
        self._set_status("Building the Pentacam baseline strip…")
        self._ref_building = True

        # Drop any stale result from a previous build.
        while True:
            try:
                self._ref_queue.get_nowait()
            except queue.Empty:
                break

        def work() -> None:
            strip = build_reference_strip(
                image, detector=detector, flip=True, path=path,
                roi_norm=roi_norm, label=f"Pentacam {self._eye}",
            )
            # Never touch Tk here — the main-thread poll reads this.
            self._ref_queue.put(strip)

        threading.Thread(target=work, daemon=True).start()
        self._schedule_reference_poll()

    def _schedule_reference_poll(self) -> None:
        """Main-thread: keep checking for the finished baseline build."""
        if self._win is None:
            return
        try:
            self._win.after(50, self._poll_reference)
        except tk.TclError:
            pass

    def _poll_reference(self) -> None:
        if self._win is None:
            return
        try:
            strip = self._ref_queue.get_nowait()
        except queue.Empty:
            if self._ref_building:
                self._schedule_reference_poll()
            return
        self._on_reference(strip)

    def _on_reference(self, strip: ReferenceStrip) -> None:
        self._ref_building = False
        self._reference = strip
        self._draw_strip("reference", strip.strip)
        if strip.strip is not None:
            self._src_status.set(
                f"Baseline ready — {strip.strip_height}×{strip.strip_width} px "
                f"({strip.roi_source or 'ok'})"
            )
        else:
            self._src_status.set(f"Baseline unavailable — {strip.reason}")
        self._set_status(
            f"Baseline ready ({self._eye}): {strip.reason}"
            + ("" if strip.ok else " — live comparison may be unavailable")
        )

    # ------------------------------------------------------------------
    # live monitoring
    # ------------------------------------------------------------------
    def _toggle_monitor(self) -> None:
        if self._running:
            self._stop_monitor()
        else:
            self._start_monitor()

    def _start_monitor(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_evt.clear()
        self._last_frame_token = None
        self._drain_queue()
        try:
            self._monitor_btn.config(text="■ Stop monitoring")
        except tk.TclError:
            pass
        self._set_status("Monitoring the live feed…")
        self._worker = threading.Thread(target=self._live_loop, daemon=True)
        self._worker.start()
        self._schedule_pump()

    def _stop_monitor(self) -> None:
        self._running = False
        self._stop_evt.set()
        after_id = self._pump_after_id
        self._pump_after_id = None
        if after_id is not None and self._win is not None:
            try:
                self._win.after_cancel(after_id)
            except tk.TclError:
                pass
        try:
            self._monitor_btn.config(text="▶ Start monitoring")
        except tk.TclError:
            pass
        self._set_status("Monitoring paused.")

    # ── worker -> main thread hand-off ────────────────────────────────
    def _schedule_pump(self) -> None:
        """(Re)arm the main-thread queue pump while monitoring."""
        if self._win is None or not self._running:
            return
        try:
            self._pump_after_id = self._win.after(25, self._pump)
        except tk.TclError:
            self._pump_after_id = None

    def _pump(self) -> None:
        """Main-thread: apply any analyses the worker produced, then rearm."""
        self._pump_after_id = None
        if self._win is None:
            return
        self._drain_queue()
        if self._running:
            self._schedule_pump()

    def _drain_queue(self) -> None:
        while True:
            try:
                analysis = self._result_queue.get_nowait()
            except queue.Empty:
                return
            if analysis is not None:
                self._apply_live(analysis)

    def _live_loop(self) -> None:
        app = self.app
        while not self._stop_evt.is_set():
            ref = self._reference
            image = getattr(app, "_current_image", None)
            result = getattr(app, "_current_result", None)
            if image is None or result is None or ref is None or ref.strip is None:
                self._stop_evt.wait(_POLL_INTERVAL_S * 4)
                continue
            token = (id(image), id(result))
            if token == self._last_frame_token:
                self._stop_evt.wait(_POLL_INTERVAL_S)
                continue
            self._last_frame_token = token
            try:
                analysis = analyse_live_frame(
                    image, result, ref, eye=self._eye,
                    frame_number=self._frames,
                    reference_label=ref.label or "Pentacam",
                )
            except Exception:
                analysis = None
            if analysis is not None:
                # Never touch Tk here — the main thread's pump reads this.
                self._result_queue.put(analysis)
            self._stop_evt.wait(_POLL_INTERVAL_S)

    def _apply_live(self, analysis: LiveFrameAnalysis) -> None:
        if self._win is None:
            return
        self._frames += 1
        self._samples.append(analysis)
        if len(self._samples) > _LOG_LEN:
            self._samples = self._samples[-_LOG_LEN:]

        self._target_marker_deg = (
            float(analysis.angle_deg) if analysis.valid else None
        )
        self._draw_strip("target", analysis.strip)
        self._update_cells(analysis)
        self._append_log(analysis)
        sign = "+" if analysis.angle_deg >= 0 else "\u2212"
        accepted = analysis.valid
        self._cyc_var.set(
            f"Live cyclotorsion  {sign}{abs(analysis.angle_deg):.2f}\u00b0   \u00b7   "
            f"score {analysis.score:.3f}   \u00b7   "
            f"{'VALID' if accepted else 'NOT ACCEPTED'}"
        )
        try:
            self._cyc_label.config(
                fg=self._colors.SURGICAL if accepted else self._colors.INSUFFICIENT
            )
        except tk.TclError:
            pass
        self._cyc_detail_var.set(
            f"Pentacam ({analysis.reference_label}) \u2192 live ELITA   \u00b7   "
            f"eye {analysis.eye}   \u00b7   engine: {analysis.reason or 'ok'}   \u00b7   "
            f"{analysis.n_valid}/{analysis.n_total} px compared   \u00b7   "
            f"took {analysis.elapsed_ms:.0f} ms"
        )
        self._set_status(
            f"Live · {analysis.eye}: angle {sign}{abs(analysis.angle_deg):.2f}\u00b0 · "
            f"score {analysis.score:.3f} · {self._frames} frames"
        )

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _update_cells(self, a: LiveFrameAnalysis) -> None:
        self._cells["pupil"].config(text=_fmt(a.pupil_confidence if a.pupil_detected else None))
        self._cells["limbus"].config(text=_fmt(a.limbus_confidence if a.limbus_detected else None))
        self._cells["quality"].config(text=a.image_quality or "—")
        if a.roi_valid:
            self._cells["roi"].config(
                text=f"{a.roi_source or 'ok'}, r {a.pupil_radius_px:.0f} / "
                     f"{a.limbus_radius_px:.0f} px")
        else:
            self._cells["roi"].config(text=f"invalid — {a.roi_reason or 'n/a'}")
        self._cells["strip"].config(
            text=(f"{a.strip_height} × {a.strip_width}" if a.has_strip else "none"))
        self._cells["angle"].config(
            text=(f"{a.angle_deg:+.2f}°" if a.has_strip else "—"),
            fg=self._colors.SURGICAL if a.valid else self._colors.INSUFFICIENT,
        )
        self._cells["score"].config(text=f"{a.score:.3f}")

    def _append_log(self, a: LiveFrameAnalysis) -> None:
        text = self._log_text
        try:
            text.config(state=tk.NORMAL)
            if len(self._samples) <= _LOG_SHOWN:
                text.delete("1.0", tk.END)
            else:
                text.delete("1.0", "2.0")  # drop the oldest line
            if a.valid and a.within_limit:
                tag = "ok" if a.score >= 0.55 else "warn"
            elif a.valid:
                tag = "warn"
            else:
                tag = "bad"
            stamp = time.strftime("%H:%M:%S")
            line = (f"{stamp}  {a.angle_deg:+7.2f}°  score {a.score:5.3f}  "
                    f"pupil {a.pupil_confidence:.2f} limbus {a.limbus_confidence:.2f}  "
                    f"{a.reason or 'ok'}\n")
            text.insert(tk.END, line, tag)
            text.see(tk.END)
        except tk.TclError:
            pass
        finally:
            try:
                text.config(state=tk.DISABLED)
            except tk.TclError:
                pass

    def _draw_strip(self, key: str, strip: Optional[np.ndarray]) -> None:
        self._last_strip[key] = strip
        canvas = self._strip_canvases.get(key)
        if canvas is None:
            return
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1:
            width = int(canvas.cget("width"))
        if height <= 1:
            height = int(canvas.cget("height"))
        self._render_strip(key, strip, width, height)

    def _on_strip_resize(self, key: str, width: int, height: int) -> None:
        if (width, height) == self._strip_drawn.get(key):
            return
        if key not in self._last_strip:
            return
        self._render_strip(key, self._last_strip[key], width, height)

    def _render_strip(self, key: str, strip: Optional[np.ndarray],
                      width: int, height: int) -> None:
        canvas = self._strip_canvases.get(key)
        if canvas is None or width <= 1 or height <= 1:
            return
        self._strip_drawn[key] = (width, height)
        canvas.delete("all")

        if strip is None:
            canvas.create_text(width // 2, height // 2,
                               text="no strip yet", fill=self._colors.FG_TERTIARY,
                               font=("Segoe UI", 10))
            canvas._photo = None
            self._photos.pop(key, None)
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
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
            (disp_w, disp_h), Image.NEAREST if scale > 1 else Image.BILINEAR))
        canvas.create_image(x0, y0, image=photo, anchor="nw")
        canvas._photo = photo
        self._photos[key] = photo

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + int(round(frac * (disp_w - 1)))
            deg = int(round(frac * 360.0))
            anchor = tk.W if frac == 0.0 else (tk.E if frac == 1.0 else tk.N)
            canvas.create_text(max(2, min(width - 2, x)), 9, text=f"{deg}°",
                               fill=self._colors.FG_SECONDARY,
                               font=("Segoe UI", 8), anchor=anchor)
            canvas.create_line(x, _RULER_H - 3, x, height - 3,
                               fill=self._colors.BORDER, dash=(2, 4))

        # Display-only compensation marker on the LIVE strip: the column the
        # Pentacam reference's 0° meridian currently maps to after the measured
        # cyclotorsion.  It is a pure visual guide — never fed to any control.
        if key == "target" and self._target_marker_deg is not None:
            frac = ((-float(self._target_marker_deg)) % 360.0) / 360.0
            mx = x0 + int(round(frac * disp_w))
            mx = max(x0, min(x0 + disp_w, mx))
            canvas.create_line(mx, _RULER_H, mx, y0 + disp_h,
                               fill=self._colors.ACCENT, width=2, dash=(3, 3))
            canvas.create_text(
                max(2, min(width - 2, mx + 5)), y0 + disp_h - 2,
                anchor="sw", text=f"correction {(-self._target_marker_deg):+.1f}°",
                fill=self._colors.ACCENT, font=("Segoe UI", 8),
            )

    # ------------------------------------------------------------------
    # saving / log
    # ------------------------------------------------------------------
    def _latest(self) -> Optional[LiveFrameAnalysis]:
        if not self._samples:
            messagebox.showinfo("Dynamic Mode",
                                "No live sample to save yet.",
                                parent=self._win)
            return None
        return self._samples[-1]

    def _save(self) -> None:
        analysis = self._latest()
        if analysis is None:
            return
        patient_dir = getattr(self.app, "_active_patient_dir", None)
        if patient_dir is None:
            messagebox.showwarning(
                "Dynamic Mode",
                "No active patient folder.\nAdd a patient first, or use Save As…",
                parent=self._win)
            return
        try:
            path = save_sample(analysis, patient_dir)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dynamic Mode", f"Could not save:\n{exc}",
                                 parent=self._win)
            return
        self._set_status(f"Saved live sample → {path}")

    def _save_as(self) -> None:
        analysis = self._latest()
        if analysis is None:
            return
        default = f"dynamic_cyclotorsion_{analysis.eye}.json"
        path = filedialog.asksaveasfilename(
            title="Save Live Cyclotorsion Details As",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialfile=default,
        )
        if not path:
            return
        try:
            save_sample(analysis,
                        getattr(self.app, "_active_patient_dir", None) or "",
                        path=path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Dynamic Mode", f"Could not save:\n{exc}",
                                 parent=self._win)
            return
        self._set_status(f"Saved a copy of the live sample → {path}")

    def _reset_log(self) -> None:
        self._samples = []
        self._frames = 0
        self._target_marker_deg = None
        try:
            self._log_text.config(state=tk.NORMAL)
            self._log_text.delete("1.0", tk.END)
            self._log_text.config(state=tk.DISABLED)
        except tk.TclError:
            pass
        self._set_status("Live log cleared.")
        # Redraw the live strip so the stale compensation marker disappears.
        self._draw_strip("target", self._last_strip.get("target"))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._running = False
        self._stop_evt.set()
        win = self._win
        self._win = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass
        try:
            self.app._dynamic_mode_open = False
        except Exception:
            pass


def _fmt(value: Optional[float]) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def open_dynamic_mode(app: Any) -> None:
    """Open (or raise) the Dynamic Mode window for *app*."""
    win = getattr(app, "_dynamic_mode_window", None)
    if win is None or getattr(win, "_win", None) is None:
        win = DynamicModeWindow(app)
        app._dynamic_mode_window = win
        app._dynamic_mode_open = True
    win.open()


__all__ = ["DynamicModeWindow", "open_dynamic_mode"]
