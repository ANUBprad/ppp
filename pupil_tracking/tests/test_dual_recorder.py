"""Unit tests for Dual Video Recording in the Centration Application."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.interface.frame_recorder import DualFrameRecorder, FrameRecorder


def test_default_recorder_avoids_qsv_h264_by_default():
    recorder = DualFrameRecorder()
    assert recorder._prefer_qsv is False


def _count_video_frames(video_path: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    return count, w, h


def test_dual_recorder_saves_synchronized_raw_and_overlay_in_centration_folder(tmp_path: Path):
    centration_dir = tmp_path / "Centration" / "PX_001_Test" / "OD"
    centration_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = str(centration_dir / "detection_20260930_120000.mp4")

    raw_w, raw_h = 640, 480
    panel_w = 360
    comp_w, comp_h = raw_w + panel_w, raw_h

    def dummy_composer(annotated: np.ndarray, result_obj) -> np.ndarray:
        panel = np.full((raw_h, panel_w, 3), 30, dtype=np.uint8)
        cv2.putText(
            panel,
            f"VAL:{result_obj}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return np.concatenate([annotated, panel], axis=1)

    recorder = DualFrameRecorder()
    assert recorder.start(
        overlay_path,
        comp_w,
        comp_h,
        30.0,
        raw_width=raw_w,
        raw_height=raw_h,
        overlay_composer=dummy_composer,
    )
    assert recorder.is_recording
    assert recorder.is_dual_recording

    num_frames = 15
    write_latencies_ms = []
    for idx in range(num_frames):
        # Simulated raw J&J surgical screen frame (no overlays)
        raw_frame = np.full((raw_h, raw_w, 3), idx * 10, dtype=np.uint8)
        # Simulated annotated frame (with detection overlays)
        annotated = raw_frame.copy()
        cv2.circle(annotated, (320, 240), 50, (0, 255, 0), 2)

        t0 = time.perf_counter()
        ok = recorder.write(
            annotated,
            raw_frame=raw_frame,
            frame_index=100 + idx,
            result=idx,
            compose_in_background=True,
        )
        write_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        assert ok

    # Latency impact on detection thread must remain well below <150 ms
    assert max(write_latencies_ms) < 50.0

    saved_overlay = recorder.stop()
    assert saved_overlay == overlay_path
    assert not recorder.is_recording

    saved_raw = recorder.last_raw_path
    saved_sync = recorder.last_sync_path
    assert saved_raw is not None and Path(saved_raw).exists()
    assert Path(saved_overlay).exists()
    assert saved_sync is not None and Path(saved_sync).exists()

    # Verify both videos have identical frame counts and expected dimensions
    raw_count, rw, rh = _count_video_frames(saved_raw)
    ov_count, ow, oh = _count_video_frames(saved_overlay)
    assert raw_count == num_frames
    assert ov_count == num_frames
    assert (rw, rh) == (raw_w, raw_h)
    assert (ow, oh) == (comp_w, comp_h)

    # Verify frame count & elapsed time remain accessible after stop()
    assert recorder.frame_count == num_frames
    assert recorder.raw_frame_count == num_frames
    assert recorder.elapsed_time > 0.0

    # Verify sync manifest CSV
    with open(saved_sync, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == num_frames
    for i, row in enumerate(rows):
        assert int(row["recorded_frame_idx"]) == i
        assert int(row["source_frame_idx"]) == 100 + i
        assert row["raw_video_file"] == Path(saved_raw).name
        assert row["overlay_video_file"] == Path(saved_overlay).name


def test_single_stream_backward_compatibility(tmp_path: Path):
    out_path = str(tmp_path / "single_stream.mp4")
    recorder = FrameRecorder()
    assert recorder.start(out_path, 320, 240, 25.0)
    for _ in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert recorder.write(frame)
    saved = recorder.stop()
    assert saved == out_path
    assert Path(out_path).exists()
    count, w, h = _count_video_frames(out_path)
    assert count == 5
    assert (w, h) == (320, 240)
