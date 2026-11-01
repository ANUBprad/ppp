"""
pentacam_detector.py — ELITA-based Pentacam pupil and limbus detection adapter.

Phase 4: Replaces classical UnifiedDetector in Pentacam workflows with the proven
ELITA neural network segmentation pipeline (FastInference / U-Net ResNet-34).

Key characteristics:
1. Reuses proven ELITA segmentation model weights (segmentation_quantized.onnx / best_model.pth).
2. Pentacam image adapter: normalises color channels (grayscale/BGR/BGRA), aspect ratio,
   and handles specular reflections via ReflectionRemover.
3. Independent extraction of pupil (class 1) and limbus (outer iris+pupil combined contour).
4. Genuine ellipse fitting via cv2.fitEllipse: preserves true corneal ellipticity,
   semi-major/minor axes, and orientation angle (never clamps to circles).
5. Returns standard EyeDetectionResult compatible with all downstream cyclotorsion,
   polar unwrapping, centration display, and GUI components.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from pupil_tracking.ml.fast_inference import FastInference
from pupil_tracking.utils.types import (
    CalibrationInfo,
    CornealCenterResult,
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    FrameMetadata,
    LimbusDetection,
    PupilDetection,
)

logger = logging.getLogger(__name__)


class PentacamDetector:
    """ELITA-based Pentacam detector.

    Wraps the ELITA FastInference neural segmentation engine with a Pentacam-specific
    image adapter and coordinate mapping.
    """

    def __init__(
        self,
        engine: Optional[FastInference] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        input_size: int = 320,
    ) -> None:
        self._engine = engine
        self._model_path = str(model_path) if model_path else None
        self._device = device
        self._input_size = input_size
        self._initialized = False

    def _ensure_engine(self) -> Optional[FastInference]:
        if self._engine is not None:
            return self._engine

        # Try to locate model weights
        candidate_paths = []
        if self._model_path:
            candidate_paths.append(Path(self._model_path))

        # Standard project locations
        base_dir = Path(__file__).resolve().parent.parent.parent
        candidate_paths.extend([
            base_dir / "models" / "onnx" / "segmentation_quantized.onnx",
            base_dir / "models" / "best_model.pth",
            base_dir / "models" / "onnx" / "segmentation.onnx",
            Path("models/onnx/segmentation_quantized.onnx"),
            Path("models/best_model.pth"),
        ])

        engine_size = 512
        for path in candidate_paths:
            if path.is_file():
                try:
                    self._engine = FastInference(
                        model_path=str(path),
                        device=self._device,
                        input_size=engine_size,
                        reflection_removal=True,
                        suction_ring_removal=False,
                    )
                    logger.info("PentacamDetector initialized FastInference with %s (size %d)", path, engine_size)
                    self._initialized = True
                    return self._engine
                except Exception as exc:
                    logger.warning("Failed to load FastInference from %s: %s", path, exc)

        logger.error("PentacamDetector could not locate ELITA model weights.")
        return None

    def _extract_corneal_structures(
        self,
        pupil_mask: np.ndarray,
        iris_mask: np.ndarray,
        probs_np: np.ndarray,
        img_w: int,
        img_h: int,
        scale_x: float,
        scale_y: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        is_full_pentacam: bool = False,
    ) -> Dict[str, Any]:
        """Extract genuine pupil and limbus ellipses from segmentation masks."""
        res: Dict[str, Any] = {
            "pupil_detected": False,
            "pupil_x": 0.0,
            "pupil_y": 0.0,
            "pupil_radius": 0.0,
            "pupil_major": 0.0,
            "pupil_minor": 0.0,
            "pupil_angle": 0.0,
            "pupil_confidence": 0.0,
            "pupil_fit_type": "ellipse",
            "limbus_detected": False,
            "limbus_x": 0.0,
            "limbus_y": 0.0,
            "limbus_radius": 0.0,
            "limbus_major": 0.0,
            "limbus_minor": 0.0,
            "limbus_angle": 0.0,
            "limbus_confidence": 0.0,
            "limbus_fit_type": "ellipse",
        }

        # 1. Pupil extraction
        cnts_p, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_p_cnts = [c for c in cnts_p if cv2.contourArea(c) >= 20]
        if valid_p_cnts:
            p_cnt = max(valid_p_cnts, key=cv2.contourArea)
            if len(p_cnt) >= 5:
                # Map contour points directly to image space to prevent model aspect ratio distortion
                p_cnt_real = p_cnt.astype(np.float32)
                p_cnt_real[:, 0, 0] = p_cnt_real[:, 0, 0] * scale_x + offset_x
                p_cnt_real[:, 0, 1] = p_cnt_real[:, 0, 1] * scale_y + offset_y
                (pcx, pcy), (pw, ph), pang = cv2.fitEllipse(p_cnt_real)
                p_major = max(pw, ph)
                p_minor = min(pw, ph)
                p_r = (p_major + p_minor) / 4.0
                if p_r >= 2.0:
                    res["pupil_detected"] = True
                    res["pupil_x"] = pcx
                    res["pupil_y"] = pcy
                    res["pupil_radius"] = p_r
                    res["pupil_major"] = p_major
                    res["pupil_minor"] = p_minor
                    res["pupil_angle"] = pang
                    res["pupil_fit_type"] = "circle" if (p_minor / p_major) >= 0.96 else "ellipse"
                    p_region = pupil_mask > 127
                    if np.any(p_region) and probs_np is not None:
                        res["pupil_confidence"] = float(np.mean(probs_np[1][p_region]))
                    else:
                        res["pupil_confidence"] = 0.85

        # 2. Limbus extraction
        # Exclude Pentacam report sidebar (columns 0..344 in flipped 1200-wide image)
        i_clean = iris_mask.copy()
        if is_full_pentacam and img_w >= 1000:
            # Zero out the sidebar area (first ~29% of width)
            sidebar_col = int(pupil_mask.shape[1] * 0.29)
            i_clean[:, :sidebar_col] = 0

        # Combine iris and pupil
        comb = np.maximum(i_clean, pupil_mask)

        # Bridge neural network gap between pupil and iris using dilated pupil ellipse
        if res["pupil_detected"]:
            # Model space coordinates of pupil
            m_pcx = (res["pupil_x"] - offset_x) / scale_x
            m_pcy = (res["pupil_y"] - offset_y) / scale_y
            m_pw = res["pupil_major"] / scale_x
            m_ph = res["pupil_minor"] / scale_y
            cv2.ellipse(comb, ((m_pcx, m_pcy), (m_pw * 1.35, m_ph * 1.35), res["pupil_angle"]), 255, -1)

        # Morphological bridge to ensure solid cornea
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        comb = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k_close)

        # Fill internal holes (e.g. reflections or central unsegmented areas)
        cnts_fill, hierarchy = cv2.findContours(comb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for idx in range(len(cnts_fill)):
                if hierarchy[0][idx][3] != -1:
                    cv2.drawContours(comb, cnts_fill, idx, 255, -1)

        # Find the cornea component
        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(comb)
        cornea_comp = None
        if res["pupil_detected"]:
            m_pcx = int(np.clip((res["pupil_x"] - offset_x) / scale_x, 0, comb.shape[1] - 1))
            m_pcy = int(np.clip((res["pupil_y"] - offset_y) / scale_y, 0, comb.shape[0] - 1))
            p_lbl = labels[m_pcy, m_pcx]
            if p_lbl > 0 and stats[p_lbl, cv2.CC_STAT_AREA] > 300:
                cornea_comp = (labels == p_lbl).astype(np.uint8) * 255

        if cornea_comp is None and num_l > 1:
            # Pick largest component in area
            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_l)]
            max_idx = int(np.argmax(areas)) + 1
            if stats[max_idx, cv2.CC_STAT_AREA] > 300:
                cornea_comp = (labels == max_idx).astype(np.uint8) * 255

        if cornea_comp is not None:
            # Sample radial boundary rays from pupil center to eliminate eyelid occlusion
            m_h, m_w = cornea_comp.shape[:2]
            c_center_x = (res["pupil_x"] - offset_x) / scale_x if res["pupil_detected"] else m_w / 2.0
            c_center_y = (res["pupil_y"] - offset_y) / scale_y if res["pupil_detected"] else m_h / 2.0
            base_r = (res["pupil_radius"] / ((scale_x + scale_y) / 2.0)) if res["pupil_detected"] else 30.0

            rays = []
            max_search_r = min(int(min(m_w, m_h) * 0.48), 250)
            for deg in range(0, 360, 2):
                rad = math.radians(deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                rs = np.arange(int(base_r * 1.15), max_search_r)
                if len(rs) == 0:
                    continue
                xs = np.clip(np.round(c_center_x + rs * cos_a).astype(int), 0, m_w - 1)
                ys = np.clip(np.round(c_center_y + rs * sin_a).astype(int), 0, m_h - 1)
                vals = cornea_comp[ys, xs]
                hits = np.where(vals > 0)[0]
                if len(hits) > 0:
                    last_hit = hits[-1]
                    if last_hit < len(rs) - 2 and xs[last_hit] > (m_w * 0.29 if is_full_pentacam else 2):
                        x_real = xs[last_hit] * scale_x + offset_x
                        y_real = ys[last_hit] * scale_y + offset_y
                        r_real = math.hypot(x_real - res["pupil_x"], y_real - res["pupil_y"])
                        rays.append((deg, r_real, x_real, y_real))

            # Lateral rays (around horizontal axis: nasal and temporal sectors) are free from eyelid occlusion
            lateral_rs = [r for d, r, x, y in rays if (d <= 35 or d >= 325 or (145 <= d <= 215))]
            median_lateral_r = float(np.median(lateral_rs)) if lateral_rs else (res["pupil_radius"] * 2.8)

            # Filter out eyelid occlusion points where boundary radius is significantly foreshortened
            valid_boundary_pts = []
            for d, r, x, y in rays:
                if 0.80 * median_lateral_r <= r <= 1.25 * median_lateral_r:
                    valid_boundary_pts.append([x, y])

            # If sufficient unoccluded boundary points exist, fit genuine ellipse
            if len(valid_boundary_pts) >= 15:
                pts_arr = np.array(valid_boundary_pts, dtype=np.float32).reshape(-1, 1, 2)
                (lcx, lcy), (lw, lh), lang = cv2.fitEllipse(pts_arr)
                l_major = max(lw, lh)
                l_minor = min(lw, lh)
                l_r = (l_major + l_minor) / 4.0
                if l_r > 10.0:
                    res["limbus_detected"] = True
                    res["limbus_x"] = lcx
                    res["limbus_y"] = lcy
                    res["limbus_radius"] = l_r
                    res["limbus_major"] = l_major
                    res["limbus_minor"] = l_minor
                    res["limbus_angle"] = lang
                    res["limbus_fit_type"] = "circle" if (l_minor / l_major) >= 0.96 else "ellipse"
                    i_region = i_clean > 127
                    if np.any(i_region) and probs_np is not None:
                        res["limbus_confidence"] = float(np.mean(probs_np[2][i_region]))
                    else:
                        res["limbus_confidence"] = 0.85
            else:
                # Fallback to direct contour fit if radial rays are sparse
                cnts_c, _ = cv2.findContours(cornea_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_c:
                    c_max = max(cnts_c, key=cv2.contourArea)
                    if len(c_max) >= 5 and cv2.contourArea(c_max) >= 100:
                        hull = cv2.convexHull(c_max).astype(np.float32)
                        hull[:, 0, 0] = hull[:, 0, 0] * scale_x + offset_x
                        hull[:, 0, 1] = hull[:, 0, 1] * scale_y + offset_y
                        (lcx, lcy), (lw, lh), lang = cv2.fitEllipse(hull)
                        l_major = max(lw, lh)
                        l_minor = min(lw, lh)
                        l_r = (l_major + l_minor) / 4.0
                        if l_r > 10.0:
                            res["limbus_detected"] = True
                            res["limbus_x"] = lcx
                            res["limbus_y"] = lcy
                            res["limbus_radius"] = l_r
                            res["limbus_major"] = l_major
                            res["limbus_minor"] = l_minor
                            res["limbus_angle"] = lang
                            res["limbus_fit_type"] = "circle" if (l_minor / l_major) >= 0.96 else "ellipse"
                            res["limbus_confidence"] = 0.80

        return res

    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[float, float, float, float]] = None,
        eye: str = "OD",
        source: Optional[str] = None,
    ) -> EyeDetectionResult:
        """Detect pupil and limbus on a Pentacam image."""
        if image is None or getattr(image, "size", 0) == 0:
            return EyeDetectionResult(overall_quality=DetectionQuality.NO_DETECTION)

        # 1. Image normalisation: Ensure 3-channel BGR
        img_bgr = image
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.ndim == 3:
            if img_bgr.shape[2] == 1:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            elif img_bgr.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = img_bgr.shape[:2]
        engine = self._ensure_engine()
        if engine is None:
            logger.error("FastInference engine unavailable for Pentacam detection.")
            return EyeDetectionResult(overall_quality=DetectionQuality.NO_DETECTION)

        engine_size = getattr(engine, "input_size", 512)
        is_pentacam_report = (orig_w >= 1000 and orig_h >= 700)

        # 2. Run detection on full frame
        scale_x = orig_w / float(engine_size)
        scale_y = orig_h / float(engine_size)
        p_mask, i_mask, probs = engine.segment(img_bgr)
        raw_det = self._extract_corneal_structures(
            p_mask, i_mask, probs, orig_w, orig_h, scale_x, scale_y, 0.0, 0.0, is_full_pentacam=is_pentacam_report
        )

        # 3. If an ROI was provided, run detection on ROI crop
        crop_det = None
        if roi is not None:
            if isinstance(roi, dict):
                rx0 = float(roi.get("x", 0.0))
                ry0 = float(roi.get("y", 0.0))
                rx1 = rx0 + float(roi.get("width", 1.0))
                ry1 = ry0 + float(roi.get("height", 1.0))
            else:
                rx0, ry0, rx1, ry1 = roi
            ix0 = max(0, min(orig_w - 1, int(round(rx0 * orig_w))))
            iy0 = max(0, min(orig_h - 1, int(round(ry0 * orig_h))))
            ix1 = max(ix0 + 10, min(orig_w, int(round(rx1 * orig_w))))
            iy1 = max(iy0 + 10, min(orig_h, int(round(ry1 * orig_h))))

            # Do not pad into the sidebar if Pentacam report
            pad_w = int((ix1 - ix0) * 0.10)
            pad_h = int((iy1 - iy0) * 0.10)
            min_x0 = 344 if is_pentacam_report else 0
            cx0 = max(min_x0, ix0 - pad_w)
            cy0 = max(0, iy0 - pad_h)
            cx1 = min(orig_w, ix1 + pad_w)
            cy1 = min(orig_h, iy1 + pad_h)
            crop = img_bgr[cy0:cy1, cx0:cx1]
            ch, cw = crop.shape[:2]
            if ch >= 30 and cw >= 30:
                c_scale_x = cw / float(engine_size)
                c_scale_y = ch / float(engine_size)
                try:
                    cp_mask, ci_mask, cprobs = engine.segment(crop)
                    crop_det = self._extract_corneal_structures(
                        cp_mask, ci_mask, cprobs, cw, ch, c_scale_x, c_scale_y, float(cx0), float(cy0), is_full_pentacam=False
                    )
                except Exception as exc:
                    logger.warning("Pentacam ROI crop detection failed: %s", exc)

        # Choose the best detection
        det = raw_det
        if crop_det is not None and crop_det.get("pupil_detected") and crop_det.get("limbus_detected"):
            crop_lr = float(crop_det.get("limbus_radius", 0.0))
            crop_pr = float(crop_det.get("pupil_radius", 0.0))
            raw_lr = float(raw_det.get("limbus_radius", 0.0))
            raw_valid = raw_det.get("limbus_detected") and (raw_lr >= 150.0 if is_pentacam_report else raw_lr >= 20.0)

            crop_plausible = (crop_lr >= 150.0 if is_pentacam_report else crop_lr >= 20.0) and (crop_lr > crop_pr * 1.2)
            if crop_plausible:
                if not raw_valid or float(crop_det.get("limbus_confidence", 0)) > float(det.get("limbus_confidence", 0)):
                    det = crop_det

        # 4. Construct PupilDetection
        pupil_det = PupilDetection()
        p_detected = bool(det.get("pupil_detected", False))
        p_cx = float(det.get("pupil_x", 0.0))
        p_cy = float(det.get("pupil_y", 0.0))
        p_r = float(det.get("pupil_radius", 0.0))
        p_maj = float(det.get("pupil_major", p_r * 2.0))
        p_min = float(det.get("pupil_minor", p_r * 2.0))
        p_angle = float(det.get("pupil_angle", 0.0))
        p_conf = float(det.get("pupil_confidence", 0.0))

        if p_detected and p_r > 2.0 and p_maj > 0 and p_min > 0:
            pupil_det = PupilDetection(
                detected=True,
                confidence=p_conf,
                ellipse=EllipseParams(
                    center_x=p_cx,
                    center_y=p_cy,
                    semi_major=p_maj / 2.0,
                    semi_minor=p_min / 2.0,
                    angle_deg=p_angle,
                ),
            )
            pupil_det.fit_type = str(det.get("pupil_fit_type", "ellipse"))

        # 5. Construct LimbusDetection
        limbus_det = LimbusDetection()
        l_detected = bool(det.get("limbus_detected", False))
        l_cx = float(det.get("limbus_x", 0.0))
        l_cy = float(det.get("limbus_y", 0.0))
        l_r = float(det.get("limbus_radius", 0.0))
        l_maj = float(det.get("limbus_major", l_r * 2.0))
        l_min = float(det.get("limbus_minor", l_r * 2.0))
        l_angle = float(det.get("limbus_angle", 0.0))
        l_conf = float(det.get("limbus_confidence", 0.0))

        if l_detected and l_r > 10.0 and l_maj > 0 and l_min > 0:
            limbus_det = LimbusDetection(
                detected=True,
                confidence=l_conf,
                ellipse=EllipseParams(
                    center_x=l_cx,
                    center_y=l_cy,
                    semi_major=l_maj / 2.0,
                    semi_minor=l_min / 2.0,
                    angle_deg=l_angle,
                ),
            )
            limbus_det.fit_type = str(det.get("limbus_fit_type", "ellipse"))

        # 6. Anatomical Consistency Validation
        consistent = True
        if pupil_det.detected and limbus_det.detected:
            # Pupil center must be inside limbus
            c_dist = math.hypot(p_cx - l_cx, p_cy - l_cy)
            if c_dist > 0.6 * (l_maj / 2.0):
                consistent = False
            # Limbus must be substantially larger than pupil
            if (l_maj / 2.0) < 1.15 * (p_maj / 2.0):
                consistent = False
            # Pupil shouldn't be microscopic compared to limbus (> 7.0x ratio)
            if (l_maj / 2.0) > 7.0 * (p_maj / 2.0):
                consistent = False

        if not consistent and not (p_detected and l_detected and p_conf > 0.8 and l_conf > 0.8):
            # If anatomically impossible and confidence is low, flag limbus as unconfirmed
            if (l_maj / 2.0) < (p_maj / 2.0):
                limbus_det.detected = False

        # 7. Corneal Center & Decentration Offset
        corneal_center = CornealCenterResult()
        if limbus_det.detected and limbus_det.ellipse is not None:
            corneal_center.center_px = (limbus_det.ellipse.center_x, limbus_det.ellipse.center_y)
            corneal_center.valid = True
            if pupil_det.detected and pupil_det.ellipse is not None:
                dx = pupil_det.ellipse.center_x - limbus_det.ellipse.center_x
                dy = pupil_det.ellipse.center_y - limbus_det.ellipse.center_y
                corneal_center.offset_x_px = dx
                corneal_center.offset_y_px = dy
                corneal_center.offset_magnitude_px = math.hypot(dx, dy)
                corneal_center.offset_angle_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        # 8. Detection Quality & Overall Result
        has_both = pupil_det.detected and limbus_det.detected
        overall_conf = (p_conf + l_conf) / 2.0 if has_both else (p_conf if pupil_det.detected else l_conf)
        if has_both and overall_conf >= 0.75:
            quality = DetectionQuality.SURGICAL
        elif has_both or (pupil_det.detected and p_conf >= 0.7):
            quality = DetectionQuality.CLINICAL
        elif pupil_det.detected or limbus_det.detected:
            quality = DetectionQuality.RESEARCH
        else:
            quality = DetectionQuality.NO_DETECTION

        res = EyeDetectionResult(
            pupil=pupil_det,
            limbus=limbus_det,
            corneal_center=corneal_center,
            calibration=CalibrationInfo(calibrated=False),
            metadata=FrameMetadata(
                image_width=orig_w,
                image_height=orig_h,
            ),
            overall_quality=quality,
            overall_confidence=overall_conf,
        )
        # Store eye attribute directly on result
        res.eye = eye
        return res
