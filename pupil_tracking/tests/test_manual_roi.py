import numpy as np

from pupil_tracking.core.eye_roi_detector import ROIResult
from pupil_tracking.video.temporal_smoother import TemporalSmoother
from pupil_tracking.video.optimized_processor import (
    ManualCircularROI,
    OptimizedVideoProcessor,
    _annotate_quality,
)


class _DummyDetector:
    def __init__(self, roi: ROIResult):
        self.roi = roi
        self.called = False

    def detect(self, frame):
        self.called = True
        return self.roi


def _make_processor():
    proc = object.__new__(OptimizedVideoProcessor)
    proc._manual_roi = None
    proc._enable_auto_roi = True
    proc.roi_detector = None
    proc._input_size = 320
    proc._use_unified = False
    proc._batch_size = 4
    from collections import deque
    proc._latency_history_ms = deque(maxlen=120)
    proc._processing_history_ms = deque(maxlen=120)
    proc._roi_history_ms = deque(maxlen=120)
    proc._quality_fail_count = 0
    proc._processed_frames = 0
    proc._stale_frames = 0
    proc._dropped_frames = 0
    proc._last_source_frame_idx = None
    proc._adaptive_quality = True
    proc._adaptive_stable_frames = 4
    proc._adaptive_quality_skip_stride = 1
    proc._stable_tracking_streak = 0
    proc._quality_check_skip_count = 0
    proc._quality_check_skip_total = 0
    proc._last_quality_usable = True
    return proc


def test_manual_roi_resolves_to_bounded_crop():
    proc = _make_processor()
    proc._manual_roi = ManualCircularROI(
        center_x=50.0,
        center_y=40.0,
        radius=12.0,
        frame_width=120,
        frame_height=80,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    roi = proc._resolve_roi(frame)

    assert roi.valid
    assert (roi.x, roi.y, roi.width, roi.height) == (38, 28, 24, 24)
    assert roi.cropped.shape == (24, 24, 3)
    assert roi.from_cache is True


def test_manual_roi_falls_back_when_frame_size_changes():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    fallback = ROIResult(
        x=5,
        y=6,
        width=40,
        height=30,
        cropped=frame[6:36, 5:45],
        is_closeup=False,
        from_cache=False,
        confidence=0.8,
    )
    detector = _DummyDetector(fallback)
    proc = _make_processor()
    proc._manual_roi = ManualCircularROI(
        center_x=50.0,
        center_y=40.0,
        radius=10.0,
        frame_width=640,
        frame_height=480,
    )
    proc.roi_detector = detector

    roi = proc._resolve_roi(frame)

    assert detector.called is True
    assert roi is fallback


def test_annotate_quality_marks_detected_frame():
    det = _annotate_quality(
        {
            "pupil_detected": True,
            "limbus_detected": True,
            "pupil_confidence": 0.82,
            "limbus_confidence": 0.78,
        }
    )

    assert det["overall_confidence"] == 0.8
    assert det["overall_quality"] == "SURGICAL"


def test_stats_capture_latency_and_drop_counters():
    proc = _make_processor()

    proc.note_source_frame(1)
    proc.note_source_frame(4)
    proc.note_stale_frame()
    proc._record_metrics(
        {"processing_time_ms": 12.0, "frame_quality": "ok"},
        latency_ms=20.0,
        roi_ms=1.5,
    )

    stats = proc.get_stats()

    assert stats["dropped_frames"] == 2
    assert stats["stale_frames"] == 1
    assert stats["processed_frames"] == 1
    assert stats["latency_avg_ms"] == 20.0
    assert stats["processing_avg_ms"] == 12.0
    assert stats["roi_avg_ms"] == 1.5


def test_adaptive_quality_skip_only_when_tracking_stable():
    class _CountingQualityChecker:
        def __init__(self):
            self.skip_check = False
            self.calls = 0

        def is_usable(self, image):
            self.calls += 1
            return True, "ok"

    proc = _make_processor()
    proc.quality_checker = _CountingQualityChecker()
    proc._manual_roi = ManualCircularROI(
        center_x=40.0,
        center_y=40.0,
        radius=20.0,
        frame_width=80,
        frame_height=80,
    )
    proc._stable_tracking_streak = 4
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    usable_1, reason_1, skipped_1 = proc._evaluate_frame_quality(frame)
    usable_2, reason_2, skipped_2 = proc._evaluate_frame_quality(frame)

    assert (usable_1, reason_1, skipped_1) == (True, "ok", True)
    assert (usable_2, reason_2, skipped_2) == (True, "ok", False)
    assert proc.quality_checker.calls == 1


def test_stats_report_adaptive_quality_state():
    proc = _make_processor()
    proc._manual_roi = ManualCircularROI(
        center_x=50.0,
        center_y=40.0,
        radius=12.0,
        frame_width=120,
        frame_height=80,
    )
    proc._adaptive_stable_frames = 1
    proc._record_metrics(
        {
            "pupil_detected": True,
            "limbus_detected": True,
            "overall_confidence": 0.81,
            "frame_quality": "ok",
            "processing_time_ms": 8.0,
        },
        latency_ms=10.0,
        roi_ms=0.5,
    )

    stats = proc.get_stats()

    assert stats["stable_tracking_streak"] == 1
    assert stats["adaptive_quality_active"] is True


def test_update_runtime_settings_applies_live_roi_and_smoother_changes():
    proc = _make_processor()
    proc.smoother = TemporalSmoother(process_noise=0.01, measurement_noise=1.0)

    proc.update_runtime_settings(
        enable_auto_roi=True,
        roi_cache_ttl=9,
        process_noise=0.2,
        measurement_noise=0.4,
    )

    assert proc._enable_auto_roi is True
    assert proc.roi_detector is not None
    assert proc.roi_detector.cache_ttl == 9
    assert proc.smoother.process_noise == 0.2
    assert proc.smoother.measurement_noise == 0.4
