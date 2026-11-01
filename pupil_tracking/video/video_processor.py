"""
Video & real-time stream processor.

Reads frames from:
  - Video files (mp4, avi, mov, etc.)
  - Camera streams (webcam, clinical camera)
  - Image sequences

For each frame:
  1. Run UnifiedDetector.detect()
  2. Apply Kalman smoothing
  3. Compute corneal centre
  4. Render overlay (optional)
  5. Log to audit trail

Supports:
  - Frame skipping for performance
  - Real-time display with overlay
  - Batch processing to CSV/JSON
  - Callback hooks for GUI integration
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    CalibrationInfo,
    FrameMetadata,
    assign_quality_grade,
)
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger


class VideoProcessor:
    """Production video processing pipeline.

    Usage
    -----
    >>> processor = VideoProcessor()
    >>> results = processor.process_file("surgery_recording.mp4")

    Real-time with callback:
    >>> processor = VideoProcessor()
    >>> processor.process_stream(
    ...     camera_id=0,
    ...     on_frame=lambda frame, result: display(frame, result),
    ... )
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config=None,
    ) -> None:
        self.cfg = config or get_config()
        self.logger = get_logger()

        # enable video-mode relaxations
        self.cfg.apply_video_mode()

        self.detector = UnifiedDetector(
            model_path=model_path, config=self.cfg
        )
        self.tracker = EyeKalmanTracker(config=self.cfg)
        self.corneal_calc = CornealCenterCalculator(config=self.cfg)
        self.eye: str = getattr(self.cfg, "eye", "OD")

        self._results: List[Dict[str, Any]] = []
        self._running = False

    # ════════════════════════════════════════════════════════════
    # File processing
    # ════════════════════════════════════════════════════════════

    def process_file(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        on_frame: Optional[Callable] = None,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process a video file end-to-end.

        Parameters
        ----------
        video_path : str  path to video file
        output_path : str | None  if given, save annotated video
        on_frame : callable  ``fn(frame, result, frame_num)``
        max_frames : int | None  stop after N frames
        show_preview : bool  display cv2 window

        Returns
        -------
        list of dicts  — per-frame results (serialisable)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Cannot open video: %s", video_path)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.info(
            "Processing video: %s (%dx%d, %.1f fps, %d frames)",
            video_path, w, h, fps, total_frames,
        )

        # output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        self.tracker.reset()
        self._results = []
        self._running = True
        frame_num = 0
        skip = self.cfg.video.skip_frames

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_num >= max_frames:
                    break

                # frame skipping
                if skip > 0 and frame_num % (skip + 1) != 0:
                    frame_num += 1
                    continue

                # detect
                raw_result = self.detector.detect(
                    frame,
                    frame_number=frame_num,
                    source=video_path,
                )

                # kalman smooth
                smoothed = self.tracker.update(raw_result)

                # corneal centre (on smoothed)
                smoothed.corneal_center = self.corneal_calc.calculate(
                    smoothed.pupil,
                    smoothed.limbus,
                    smoothed.calibration,
                )

                # store
                result_dict = smoothed.to_dict()
                self._results.append(result_dict)

                # callback
                if on_frame:
                    on_frame(frame, smoothed, frame_num)

                # overlay + write
                if writer or show_preview:
                    annotated = self._draw_overlay(frame, smoothed)
                    if writer:
                        writer.write(annotated)
                    if show_preview:
                        cv2.imshow("Pupil Tracking", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                # audit log every 30th frame
                if frame_num % 30 == 0:
                    self.logger.log_detection(result_dict, frame_num)

                frame_num += 1

        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
            self._running = False

        self.logger.info(
            "Processed %d frames, %d results",
            frame_num, len(self._results),
        )

        return self._results

    # ════════════════════════════════════════════════════════════
    # Live stream
    # ════════════════════════════════════════════════════════════

    def process_stream(
        self,
        camera_id: int = 0,
        on_frame: Optional[Callable] = None,
        show_preview: bool = True,
    ) -> None:
        """Process a live camera stream until stopped.

        Parameters
        ----------
        camera_id : int  camera device index
        on_frame : callable  ``fn(frame, result, frame_num)``
        show_preview : bool
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self.logger.error("Cannot open camera: %d", camera_id)
            return

        self.tracker.reset()
        self._running = True
        frame_num = 0
        consecutive_failures = 0

        self.logger.info("Live stream started (camera=%d)", camera_id)

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self.logger.warning(
                            "Camera read failed %d times — stopping",
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                raw_result = self.detector.detect(
                    frame, frame_number=frame_num, source=f"camera_{camera_id}",
                )
                smoothed = self.tracker.update(raw_result)

                smoothed.corneal_center = self.corneal_calc.calculate(
                    smoothed.pupil,
                    smoothed.limbus,
                    smoothed.calibration,
                )

                if on_frame:
                    on_frame(frame, smoothed, frame_num)

                if show_preview:
                    annotated = self._draw_overlay(frame, smoothed)
                    cv2.imshow("Pupil Tracking — Live", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break

                frame_num += 1

        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            self._running = False
            self.logger.info("Live stream stopped after %d frames", frame_num)

    def stop(self) -> None:
        """Signal the processor to stop."""
        self._running = False

    # ════════════════════════════════════════════════════════════
    # Export
    # ════════════════════════════════════════════════════════════

    def export_csv(self, path: str) -> None:
        """Export results to CSV."""
        if not self._results:
            self.logger.warning("No results to export")
            return

        flat_rows = []
        for r in self._results:
            row = {
                "frame": r.get("metadata", {}).get("frame_number", -1),
                "timestamp": r.get("metadata", {}).get("timestamp", 0),
                "pupil_detected": r.get("pupil", {}).get("detected", False),
                "pupil_cx": r.get("pupil", {}).get("ellipse", {}).get("center_x", ""),
                "pupil_cy": r.get("pupil", {}).get("ellipse", {}).get("center_y", ""),
                "pupil_radius": r.get("pupil", {}).get("ellipse", {}).get("radius", ""),
                "pupil_confidence": r.get("pupil", {}).get("confidence", 0),
                "limbus_detected": r.get("limbus", {}).get("detected", False),
                "limbus_cx": r.get("limbus", {}).get("ellipse", {}).get("center_x", ""),
                "limbus_cy": r.get("limbus", {}).get("ellipse", {}).get("center_y", ""),
                "limbus_radius": r.get("limbus", {}).get("ellipse", {}).get("radius", ""),
                "limbus_confidence": r.get("limbus", {}).get("confidence", 0),
                "offset_px": r.get("corneal_center", {}).get("offset_magnitude_px", ""),
                "offset_mm": r.get("corneal_center", {}).get("offset_magnitude_mm", ""),
                "quality": r.get("overall_quality", ""),
            }
            flat_rows.append(row)

        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)

        self.logger.info("Exported %d rows to %s", len(flat_rows), path)

    def export_json(self, path: str) -> None:
        """Export results to JSON."""
        with open(path, "w") as fh:
            json.dump(self._results, fh, indent=2, default=str)
        self.logger.info("Exported %d results to %s", len(self._results), path)

    # ════════════════════════════════════════════════════════════
    # Overlay rendering
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        result: EyeDetectionResult,
    ) -> np.ndarray:
        """Draw detection overlay on a frame copy."""
        out = frame.copy()
        h, w = out.shape[:2]

        # ── pupil ───────────────────────────────────────────────
        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            center = (int(round(e.center_x)), int(round(e.center_y)))
            axes = (
                int(round(e.semi_major)),
                int(round(e.semi_minor)),
            )
            angle = int(round(e.angle_deg))

            # green for pupil
            cv2.ellipse(out, center, axes, angle, 0, 360, (0, 255, 0), 2)
            cv2.circle(out, center, 3, (0, 255, 0), -1)

            # label
            cv2.putText(
                out,
                f"P ({e.center_x:.1f}, {e.center_y:.1f}) r={e.radius:.1f}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
            )

        # ── limbus ──────────────────────────────────────────────
        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            center = (int(round(e.center_x)), int(round(e.center_y)))
            axes = (
                int(round(e.semi_major)),
                int(round(e.semi_minor)),
            )
            angle = int(round(e.angle_deg))

            # blue for limbus
            cv2.ellipse(out, center, axes, angle, 0, 360, (255, 100, 0), 2)
            cv2.circle(out, center, 3, (255, 100, 0), -1)

            cv2.putText(
                out,
                f"L ({e.center_x:.1f}, {e.center_y:.1f}) r={e.radius:.1f}",
                (center[0] + 10, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1,
            )

        # ── corneal centre + offset ────────────────────────────
        if result.corneal_center.valid:
            cc = result.corneal_center
            cc_pt = (
                int(round(cc.center_px[0])),
                int(round(cc.center_px[1])),
            )
            cv2.drawMarker(
                out, cc_pt, (10, 10, 10),
                cv2.MARKER_CROSS, 45, 4, cv2.LINE_AA
            )
            cv2.drawMarker(
                out, cc_pt, (255, 0, 255),
                cv2.MARKER_CROSS, 45, 3, cv2.LINE_AA
            )

            # draw offset line from corneal centre to pupil centre
            if result.pupil.detected and result.pupil.ellipse is not None:
                p_pt = (
                    int(round(result.pupil.ellipse.center_x)),
                    int(round(result.pupil.ellipse.center_y)),
                )
                cv2.line(out, cc_pt, p_pt, (0, 255, 255), 1)

            # offset text
            offset_text = f"Offset: {cc.offset_magnitude_px:.1f}px"
            if cc.offset_magnitude_mm is not None:
                offset_text += f" ({cc.offset_magnitude_mm:.2f}mm)"
            cv2.putText(
                out, offset_text, (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

        # ── cross section ─────────────────────────────────────────
        if (
            result.pupil.detected
            and result.pupil.ellipse is not None
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            import math
            p_e = result.pupil.ellipse
            l_e = result.limbus.ellipse

            p_cx = p_e.center_x
            p_cy = p_e.center_y

            # Define local helper for ellipse intersection at scale = 1.0
            def get_intersection(px, py, dx, dy):
                cx = l_e.center_x
                cy = l_e.center_y
                a = max(1.0, l_e.semi_major)
                b = max(1.0, l_e.semi_minor)
                angle_rad = math.radians(l_e.angle_deg)

                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                # Local coordinates of start point relative to ellipse center
                x_loc = (px - cx) * cos_a + (py - cy) * sin_a
                y_loc = -(px - cx) * sin_a + (py - cy) * cos_a

                # Local direction vector
                dx_loc = dx * cos_a + dy * sin_a
                dy_loc = -dx * sin_a + dy * cos_a

                # Quadratic equation coefficients: A * t^2 + 2 * B * t + C = 0
                A_coef = (dx_loc / a) ** 2 + (dy_loc / b) ** 2
                B_coef = (x_loc * dx_loc) / (a ** 2) + (y_loc * dy_loc) / (b ** 2)
                C_coef = (x_loc / a) ** 2 + (y_loc / b) ** 2 - 1.0

                disc = B_coef ** 2 - A_coef * C_coef
                if disc < 0 or A_coef == 0:
                    return px + dx * a, py + dy * b

                t = (-B_coef + math.sqrt(disc)) / A_coef
                return px + t * dx, py + t * dy

            up_pt = get_intersection(p_cx, p_cy, 0.0, -1.0)
            down_pt = get_intersection(p_cx, p_cy, 0.0, 1.0)
            left_pt = get_intersection(p_cx, p_cy, -1.0, 0.0)
            right_pt = get_intersection(p_cx, p_cy, 1.0, 0.0)

            p_center = (int(round(p_cx)), int(round(p_cy)))
            green_color = (0, 255, 0)
            blue_color = (255, 100, 0)

            cv2.line(out, p_center, (int(round(up_pt[0])), int(round(up_pt[1]))), green_color, 1, cv2.LINE_AA)
            cv2.line(out, p_center, (int(round(down_pt[0])), int(round(down_pt[1]))), blue_color, 1, cv2.LINE_AA)
            cv2.line(out, p_center, (int(round(left_pt[0])), int(round(left_pt[1]))), green_color, 1, cv2.LINE_AA)
            cv2.line(out, p_center, (int(round(right_pt[0])), int(round(right_pt[1]))), blue_color, 1, cv2.LINE_AA)

            # Degree labels (clinical anatomical convention: Nose is central reference)
            from pupil_tracking.interface.gui_helpers import get_centration_meridian_labels
            eye_clean = (
                getattr(result, "eye", None)
                or getattr(self, "eye", None)
                or "OD"
            )
            left_lbl, right_lbl, up_lbl, down_lbl = get_centration_meridian_labels(eye_clean)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_sz = 0.4
            lbl_color = (220, 220, 220)

            cv2.putText(out, up_lbl, (int(round(up_pt[0])) - 10, int(round(up_pt[1])) - 5), font, font_sz, lbl_color, 1, cv2.LINE_AA)
            cv2.putText(out, down_lbl, (int(round(down_pt[0])) - 7, int(round(down_pt[1])) + 12), font, font_sz, lbl_color, 1, cv2.LINE_AA)
            cv2.putText(out, left_lbl, (int(round(left_pt[0])) - 15, int(round(left_pt[1])) + 4), font, font_sz, lbl_color, 1, cv2.LINE_AA)
            cv2.putText(out, right_lbl, (int(round(right_pt[0])) + 5, int(round(right_pt[1])) + 4), font, font_sz, lbl_color, 1, cv2.LINE_AA)


        # ── quality badge ───────────────────────────────────────
        quality_colors = {
            "SURGICAL": (0, 255, 0),
            "CLINICAL": (0, 200, 255),
            "RESEARCH": (0, 165, 255),
            "INSUFFICIENT": (0, 0, 255),
            "NO_DETECTION": (128, 128, 128),
        }
        quality_str = result.overall_quality.value
        color = quality_colors.get(quality_str, (128, 128, 128))

        cv2.putText(
            out,
            f"Quality: {quality_str} ({result.overall_confidence:.2f})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

        # ── processing time ─────────────────────────────────────
        cv2.putText(
            out,
            f"{result.metadata.processing_time_ms:.0f}ms",
            (w - 80, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # ── alerts ──────────────────────────────────────────────
        for i, alert in enumerate(result.alerts[:3]):
            cv2.putText(
                out, alert[:80], (10, h - 10 - i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )

        return out