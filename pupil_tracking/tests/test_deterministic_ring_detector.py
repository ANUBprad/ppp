import math

import cv2
import numpy as np

from pupil_tracking.core.deterministic_ring_detector import (
    HeuristicRingDetector,
    RingStatus,
)


def _make_base_eye(size: int = 768) -> np.ndarray:
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    center = (size // 2, size // 2)

    cv2.circle(img, center, 315, (225, 228, 215), -1, cv2.LINE_AA)
    cv2.circle(img, center, 170, (72, 84, 108), -1, cv2.LINE_AA)
    cv2.circle(img, center, 78, (28, 32, 38), -1, cv2.LINE_AA)
    cv2.circle(img, center, 220, (180, 182, 176), 3, cv2.LINE_AA)
    return img


def _make_preop_image() -> np.ndarray:
    return _make_base_eye()


def _make_postop_image() -> np.ndarray:
    img = _make_base_eye()
    center = (img.shape[1] // 2 + 8, img.shape[0] // 2 - 6)
    radius = 298

    cv2.circle(img, center, radius, (0, 0, 235), 9, cv2.LINE_AA)
    for angle_deg in range(0, 360, 24):
        angle = math.radians(angle_deg)
        x = int(round(center[0] + radius * math.cos(angle)))
        y = int(round(center[1] + radius * math.sin(angle)))
        cv2.circle(img, (x, y), 10, (25, 55, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (120, 180, 255), -1, cv2.LINE_AA)
    return img


def test_preop_image_is_not_classified_as_docked():
    detector = HeuristicRingDetector()
    result = detector.detect(_make_preop_image())

    assert result.status == RingStatus.ABSENT
    assert result.confidence >= 0.5


def test_postop_image_returns_ring_geometry():
    detector = HeuristicRingDetector()
    result = detector.detect(_make_postop_image())

    assert result.status == RingStatus.PRESENT
    assert result.ring_center is not None
    assert result.ring_radius is not None
    assert result.ring_contour is not None
    assert result.corneal_reference_source == "suction_ring"
    assert abs(result.ring_center[0] - 392) < 25
    assert abs(result.ring_center[1] - 378) < 25
    assert abs(result.ring_radius - 298) < 30
