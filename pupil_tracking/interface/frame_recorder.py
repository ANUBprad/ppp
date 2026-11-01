"""
Professional Dual & Single FrameRecorder for the Centration Application.

Features:
    - Synchronised dual-stream recording:
        1. Raw J&J surgical screen feed (unmodified video for iris registration training)
        2. Centration application overlay feed (processed detection overlays & values)
    - Intel Quick Sync Video (QSV) hardware acceleration (CAP_INTEL_MFX /
      VIDEO_ACCELERATION_MFX / VIDEO_ACCELERATION_D3D11 / FFmpeg h264_qsv & hevc_qsv)
    - Lightweight H.264 (AVC) and H.265 (HEVC) compression with automatic fallback
    - Lockstep frame-index & timestamp synchronisation with companion sync CSV manifest
    - Dedicated background writer threads per stream to isolate encoding and
      keep detection latency well below <150 ms
"""

from __future__ import annotations

import csv
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


class _FFmpegQSVPipeWriter:
    """Lightweight FFmpeg pipe writer supporting Intel QSV (h264_qsv / hevc_qsv)."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        encoder: str = "h264_qsv",
    ) -> None:
        self._path = path
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._encoder = encoder
        self._proc: Optional[subprocess.Popen] = None
        self._opened = False
        self._open()

    def _open(self) -> None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            f"{self._fps:.3f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            self._encoder,
        ]
        if "qsv" in self._encoder:
            cmd.extend(["-preset", "veryfast", "-global_quality", "23"])
        elif self._encoder in ("libx264", "libx265"):
            cmd.extend(["-preset", "ultrafast", "-crf", "23"])
        cmd.extend(["-pix_fmt", "yuv420p", self._path])

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            # Verify process did not exit immediately
            if self._proc.poll() is None:
                self._opened = True
            else:
                self._proc = None
        except Exception:
            self._proc = None
            self._opened = False

    def isOpened(self) -> bool:
        return bool(
            self._opened
            and self._proc is not None
            and self._proc.poll() is None
            and self._proc.stdin is not None
        )

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            return
        try:
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height))
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            self._proc.stdin.write(frame.tobytes())
        except Exception:
            self._opened = False

    def release(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            finally:
                self._proc = None
                self._opened = False


class FrameRecorder:
    """High-performance single/dual frame recorder with Intel QSV & H.264/H.265 support.

    Supports both legacy single-stream recording and synchronised dual-stream
    recording (raw J&J surgical screen feed + centration application overlay feed)
    using dedicated background threads so detection latency remains unaffected (<150 ms).
    """

    # Prioritise lightweight H.264 / H.265 compression (with Intel QSV hardware
    # acceleration probed first), followed by standard container fallbacks.
    SUPPORTED_CODECS = [
        ("H264", ".mp4", "H.264/AVC (QSV/HW)"),
        ("avc1", ".mp4", "H.264/AVC (MP4)"),
        ("HEVC", ".mp4", "H.265/HEVC (QSV/HW)"),
        ("hvc1", ".mp4", "H.265/HEVC (MP4)"),
        ("X264", ".mp4", "H.264 (MP4)"),
        ("mp4v", ".mp4", "MPEG-4 (MP4)"),
        ("XVID", ".avi", "XviD (AVI)"),
        ("MJPG", ".avi", "Motion JPEG (AVI)"),
        ("DIVX", ".avi", "DivX (AVI)"),
        ("WMV2", ".wmv", "WMV2 (WMV)"),
    ]

    # Cached best encoder configuration discovered on this system:
    # (api_preference, hw_accel_prop, codec_fourcc_str, hw_label)
    _CACHED_ENCODER_SPEC: Optional[Tuple[int, Optional[int], str, str]] = None
    _PROBE_LOCK = threading.Lock()

    def __init__(self, prefer_qsv: bool = False, max_queue_size: int = 90) -> None:
        # Default to software encoding so Windows machines without a compatible
        # OpenH264/FFmpeg stack do not trigger the "Incorrect library version" crash
        # during recording startup. Explicit opt-in remains available via
        # prefer_qsv=True for users who know their environment is valid.
        self._prefer_qsv: bool = bool(prefer_qsv)
        self._max_queue_size: int = max(30, int(max_queue_size))

        self._writer_thread: Optional[threading.Thread] = None
        self._raw_writer_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self._max_queue_size)
        self._raw_frame_queue: queue.Queue = queue.Queue(maxsize=self._max_queue_size)

        self._writer: Optional[Any] = None
        self._raw_writer: Optional[Any] = None

        self._path: Optional[str] = None
        self._raw_path: Optional[str] = None
        self._sync_path: Optional[str] = None

        self.last_overlay_path: Optional[str] = None
        self.last_raw_path: Optional[str] = None
        self.last_sync_path: Optional[str] = None
        self.last_frame_count: int = 0
        self.last_raw_frame_count: int = 0
        self.last_elapsed_time: float = 0.0
        self.last_codec: str = "mp4v"
        self.last_hw_backend: str = "CPU"

        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._raw_width: int = 0
        self._raw_height: int = 0
        self._dual_mode: bool = False

        self._running: bool = False
        self._codec: str = "mp4v"
        self._hw_backend: str = "CPU"
        self._frame_count: int = 0
        self._raw_frame_count: int = 0
        self._queued_pairs: int = 0
        self._dropped_frames: int = 0
        self._start_time: float = 0.0
        self._start_wall_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._lock = threading.Lock()

        self._sync_records: List[Tuple[int, int, float, float, float]] = []
        self._overlay_composer: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None
        self._status_callback: Optional[Callable[[dict], None]] = None

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        with self._lock:
            return self._running and self._writer is not None

    @property
    def is_dual_recording(self) -> bool:
        """Check if synchronised dual-stream recording is active."""
        with self._lock:
            return (
                self._running
                and self._dual_mode
                and self._writer is not None
                and self._raw_writer is not None
            )

    @property
    def raw_path(self) -> Optional[str]:
        """Path to the raw surgical feed video (current or most recent)."""
        with self._lock:
            return self._raw_path or self.last_raw_path

    @property
    def sync_path(self) -> Optional[str]:
        """Path to the frame-synchronisation CSV manifest (current or most recent)."""
        with self._lock:
            return self._sync_path or self.last_sync_path

    @property
    def frame_count(self) -> int:
        """Get number of frames written (preserves last session count after stop)."""
        with self._lock:
            return self._frame_count if self._running else self.last_frame_count

    @property
    def raw_frame_count(self) -> int:
        """Get number of raw frames written."""
        with self._lock:
            return self._raw_frame_count if self._running else self.last_raw_frame_count

    @property
    def dropped_frames(self) -> int:
        """Get number of dropped frames."""
        with self._lock:
            return self._dropped_frames

    @property
    def elapsed_time(self) -> float:
        """Get elapsed recording time (preserves last session duration after stop)."""
        with self._lock:
            if self._running and self._start_time > 0:
                return time.monotonic() - self._start_time
            return self.last_elapsed_time

    def set_status_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for status updates."""
        self._status_callback = callback

    def set_overlay_composer(
        self, composer: Optional[Callable[[np.ndarray, Any], np.ndarray]]
    ) -> None:
        """Set an optional function to compose the full overlay panel on the writer thread."""
        self._overlay_composer = composer

    @staticmethod
    def derive_companion_paths(
        overlay_path: str,
        raw_path: Optional[str] = None,
        sync_path: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Derive synchronised raw video path and sync CSV path in the same folder."""
        p = Path(overlay_path)
        stem = p.stem
        suffix = p.suffix or ".mp4"
        parent = p.parent

        if raw_path is None:
            if stem.startswith("detection_"):
                ts_part = stem[len("detection_") :]
                raw_name = f"raw_{ts_part}{suffix}"
            else:
                raw_name = f"{stem}_raw{suffix}"
            raw_path = str(parent / raw_name)

        if sync_path is None:
            if stem.startswith("detection_"):
                ts_part = stem[len("detection_") :]
                sync_name = f"sync_{ts_part}.csv"
            else:
                sync_name = f"{stem}_sync.csv"
            sync_path = str(parent / sync_name)

        return str(p), raw_path, sync_path

    def _get_best_codec(self, path: str) -> Tuple[int, str]:
        """Select the best available codec fourcc for the output format."""
        ext = Path(path).suffix.lower() or ".mp4"
        if FrameRecorder._CACHED_ENCODER_SPEC is not None:
            _, _, cached_codec, _ = FrameRecorder._CACHED_ENCODER_SPEC
            return cv2.VideoWriter_fourcc(*cached_codec), cached_codec

        for codec, codec_ext, _ in self.SUPPORTED_CODECS:
            if ext == codec_ext:
                return cv2.VideoWriter_fourcc(*codec), codec

        return cv2.VideoWriter_fourcc(*"mp4v"), "mp4v"

    def _open_accelerated_writer(
        self, path: str, width: int, height: int, fps: float
    ) -> Tuple[Optional[Any], str, str]:
        """Open a VideoWriter prioritising Intel QSV (Quick Sync Video) & H.264/H.265.

        Probes hardware-accelerated Intel QSV (`CAP_INTEL_MFX` / `VIDEO_ACCELERATION_MFX` /
        `VIDEO_ACCELERATION_D3D11`) with H.264/H.265 first, caches the working backend
        for instantaneous subsequent opens, and falls back to MPEG-4 (`mp4v`) so recording
        always succeeds on any machine.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # 1. Fast path: use cached encoder specification if already probed
        if FrameRecorder._CACHED_ENCODER_SPEC is not None:
            api_pref, hw_accel, codec_name, hw_label = FrameRecorder._CACHED_ENCODER_SPEC
            writer = self._try_create_cv2_writer(
                path, width, height, fps, api_pref, hw_accel, codec_name
            )
            if writer is not None:
                return writer, codec_name, hw_label

        # 2. One-time silent probe for Intel QSV -> D3D11 HW -> H.264/H.265 -> mp4v
        with FrameRecorder._PROBE_LOCK:
            if FrameRecorder._CACHED_ENCODER_SPEC is not None:
                api_pref, hw_accel, codec_name, hw_label = FrameRecorder._CACHED_ENCODER_SPEC
                writer = self._try_create_cv2_writer(
                    path, width, height, fps, api_pref, hw_accel, codec_name
                )
                if writer is not None:
                    return writer, codec_name, hw_label

            prev_log_level = None
            if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
                try:
                    prev_log_level = cv2.utils.logging.getLogLevel()
                    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
                except Exception:
                    prev_log_level = None

            try:
                candidates: List[Tuple[int, Optional[int], str, str]] = []
                cap_mfx = getattr(cv2, "CAP_INTEL_MFX", None)
                cap_msmf = getattr(cv2, "CAP_MSMF", None)
                cap_ffmpeg = getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY)
                hw_mfx = getattr(cv2, "VIDEO_ACCELERATION_MFX", None)
                hw_d3d11 = getattr(cv2, "VIDEO_ACCELERATION_D3D11", None)
                hw_any = getattr(cv2, "VIDEO_ACCELERATION_ANY", None)

                if self._prefer_qsv:
                    # Priority 1: Intel Quick Sync Video (QSV) hardware acceleration
                    if cap_mfx is not None:
                        for c in ("H264", "HEVC", "avc1"):
                            candidates.append((cap_mfx, hw_mfx, c, f"Intel QSV ({c})"))
                    if hw_mfx is not None:
                        for api in (cap_msmf, cap_ffmpeg):
                            if api is not None:
                                for c in ("H264", "avc1", "HEVC", "hvc1"):
                                    candidates.append((api, hw_mfx, c, f"Intel QSV ({c})"))
                    # Priority 2: D3D11 / Windows Media Foundation Hardware QSV
                    if hw_d3d11 is not None and cap_msmf is not None:
                        for c in ("H264", "HEVC"):
                            candidates.append((cap_msmf, hw_d3d11, c, f"QSV/D3D11 ({c})"))

                # Priority 3: Prefer safe software encoders first so the app does not
                # trip over incompatible OpenH264/FFmpeg stacks on Windows. H.264/HEVC
                # remain available as an explicit opt-in when prefer_qsv=True.
                for c in ("mp4v", "XVID", "MJPG", "avc1", "H264", "HEVC"):
                    if hw_any is not None:
                        candidates.append((cap_ffmpeg, hw_any, c, f"HW/CPU ({c})"))
                    candidates.append((cv2.CAP_ANY, None, c, f"CPU ({c})"))

                for api_pref, hw_accel, codec_name, hw_label in candidates:
                    writer = self._try_create_cv2_writer(
                        path, width, height, fps, api_pref, hw_accel, codec_name
                    )
                    if writer is not None:
                        FrameRecorder._CACHED_ENCODER_SPEC = (
                            api_pref,
                            hw_accel,
                            codec_name,
                            hw_label,
                        )
                        return writer, codec_name, hw_label
            finally:
                if prev_log_level is not None:
                    try:
                        cv2.utils.logging.setLogLevel(prev_log_level)
                    except Exception:
                        pass

        # 3. Optional FFmpeg QSV subprocess fallback if ffmpeg is on PATH
        if shutil.which("ffmpeg"):
            for enc, label in (
                ("h264_qsv", "FFmpeg Intel QSV (H.264)"),
                ("hevc_qsv", "FFmpeg Intel QSV (H.265)"),
                ("libx264", "FFmpeg H.264"),
            ):
                ff_writer = _FFmpegQSVPipeWriter(path, width, height, fps, encoder=enc)
                if ff_writer.isOpened():
                    return ff_writer, enc, label

        return None, "none", "none"

    @staticmethod
    def _try_create_cv2_writer(
        path: str,
        width: int,
        height: int,
        fps: float,
        api_pref: int,
        hw_accel: Optional[int],
        codec_name: str,
    ) -> Optional[cv2.VideoWriter]:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            prop_hw = getattr(cv2, "VIDEOWRITER_PROP_HW_ACCELERATION", None)
            if hw_accel is not None and prop_hw is not None:
                writer = cv2.VideoWriter(
                    path,
                    api_pref,
                    fourcc,
                    fps,
                    (int(width), int(height)),
                    [prop_hw, int(hw_accel)],
                )
            else:
                writer = cv2.VideoWriter(
                    path,
                    api_pref,
                    fourcc,
                    fps,
                    (int(width), int(height)),
                )
            if writer is not None and writer.isOpened():
                return writer
            if writer is not None:
                writer.release()
        except Exception:
            pass
        return None

    def start(
        self,
        path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        *,
        raw_path: Optional[str] = None,
        raw_width: Optional[int] = None,
        raw_height: Optional[int] = None,
        sync_path: Optional[str] = None,
        overlay_composer: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
    ) -> bool:
        """Start recording overlay stream and (optionally) synchronised raw surgical stream.

        Args:
            path: Output video file path for the overlay stream
            width: Overlay frame width
            height: Overlay frame height
            fps: Target frames per second
            raw_path: Optional output file path for the raw J&J surgical stream
            raw_width: Raw surgical frame width (enables dual recording when > 0)
            raw_height: Raw surgical frame height (enables dual recording when > 0)
            sync_path: Optional CSV path for frame-level timestamp/index sync manifest
            overlay_composer: Optional callable(annotated_frame, result) -> composite_frame
                executed inside the background overlay writer thread to avoid UI/detection latency.

        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            self.stop()

        if width <= 0 or height <= 0:
            return False

        if overlay_composer is not None:
            self._overlay_composer = overlay_composer

        enable_dual = bool(
            raw_path is not None
            or (raw_width is not None and raw_height is not None and raw_width > 0 and raw_height > 0)
        )

        if enable_dual:
            path, resolved_raw_path, resolved_sync_path = self.derive_companion_paths(
                path, raw_path=raw_path, sync_path=sync_path
            )
            rw = int(raw_width) if (raw_width and raw_width > 0) else int(width)
            rh = int(raw_height) if (raw_height and raw_height > 0) else int(height)
        else:
            resolved_raw_path = None
            resolved_sync_path = sync_path
            rw, rh = 0, 0

        # Open overlay writer
        overlay_writer, codec_name, hw_label = self._open_accelerated_writer(
            path, width, height, fps
        )
        if overlay_writer is None:
            return False

        # Open raw surgical stream writer if dual mode is enabled
        raw_writer = None
        if enable_dual and resolved_raw_path is not None:
            raw_writer, _, _ = self._open_accelerated_writer(
                resolved_raw_path, rw, rh, fps
            )
            if raw_writer is None:
                overlay_writer.release()
                return False

        with self._lock:
            self._path = path
            self._raw_path = resolved_raw_path
            self._sync_path = resolved_sync_path
            self._fps = float(fps)
            self._width = int(width)
            self._height = int(height)
            self._raw_width = rw
            self._raw_height = rh
            self._dual_mode = bool(enable_dual and raw_writer is not None)

            self._writer = overlay_writer
            self._raw_writer = raw_writer
            self._codec = codec_name
            self._hw_backend = hw_label

            self._frame_count = 0
            self._raw_frame_count = 0
            self._queued_pairs = 0
            self._dropped_frames = 0
            self._last_frame_time = 0.0
            self._sync_records = []

            self.last_overlay_path = path
            self.last_raw_path = resolved_raw_path
            self.last_sync_path = resolved_sync_path
            self.last_frame_count = 0
            self.last_raw_frame_count = 0
            self.last_elapsed_time = 0.0
            self.last_codec = codec_name
            self.last_hw_backend = hw_label

            self._running = True
            self._start_time = time.monotonic()
            self._start_wall_time = time.time()
            self._frame_queue = queue.Queue(maxsize=self._max_queue_size)
            self._raw_frame_queue = queue.Queue(maxsize=self._max_queue_size)

        # Start dedicated parallel writer threads so raw & overlay encode simultaneously
        if self._dual_mode:
            self._raw_writer_thread = threading.Thread(
                target=self._raw_writer_loop,
                daemon=True,
                name="FrameRecorderRawWriter",
            )
            self._raw_writer_thread.start()

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="FrameRecorderOverlayWriter",
        )
        self._writer_thread.start()

        return True

    def _raw_writer_loop(self) -> None:
        """Dedicated background thread that encodes the raw J&J surgical stream."""
        raw_written = 0
        while self._running or not self._raw_frame_queue.empty():
            try:
                try:
                    item = self._raw_frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    break

                raw_frame, _frame_idx, _ts_mono = item
                if self._raw_writer is not None and self._raw_writer.isOpened():
                    if (
                        raw_frame.shape[1] != self._raw_width
                        or raw_frame.shape[0] != self._raw_height
                    ):
                        raw_frame = cv2.resize(
                            raw_frame, (self._raw_width, self._raw_height)
                        )
                    self._raw_writer.write(raw_frame)
                    raw_written += 1
                    with self._lock:
                        self._raw_frame_count = raw_written
                        self.last_raw_frame_count = raw_written
            except Exception:
                pass

        if self._raw_writer is not None:
            try:
                self._raw_writer.release()
            except Exception:
                pass

    def _writer_loop(self) -> None:
        """Dedicated background thread that composes & encodes the overlay stream."""
        frames_written = 0
        last_status_update = 0.0

        while self._running or not self._frame_queue.empty():
            try:
                try:
                    frame_data = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame_data is None:
                    break

                frame, timestamp, result_obj, needs_compose = frame_data

                # Perform overlay panel composition in the background thread if requested
                if needs_compose and self._overlay_composer is not None:
                    try:
                        frame = self._overlay_composer(frame, result_obj)
                    except Exception:
                        pass

                if self._writer is not None and self._writer.isOpened():
                    if (
                        frame.shape[1] != self._width
                        or frame.shape[0] != self._height
                    ):
                        frame = cv2.resize(frame, (self._width, self._height))
                    self._writer.write(frame)
                    frames_written += 1

                    with self._lock:
                        self._frame_count = frames_written
                        self.last_frame_count = frames_written
                        if self._start_time > 0:
                            self.last_elapsed_time = time.monotonic() - self._start_time

                now = time.monotonic()
                if now - last_status_update >= 0.1 and self._status_callback:
                    elapsed = now - self._start_time
                    actual_fps = frames_written / elapsed if elapsed > 0 else 0

                    self._status_callback(
                        {
                            "is_recording": True,
                            "is_dual": self._dual_mode,
                            "frame_count": frames_written,
                            "raw_frame_count": self._raw_frame_count,
                            "elapsed_time": elapsed,
                            "fps": actual_fps,
                            "dropped_frames": self._dropped_frames,
                            "queue_size": self._frame_queue.qsize(),
                            "codec": self._codec,
                            "hw_backend": self._hw_backend,
                        }
                    )
                    last_status_update = now

            except Exception:
                pass

        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass

    def write_dual(
        self,
        raw_frame: np.ndarray,
        overlay_frame: np.ndarray,
        *,
        frame_index: Optional[int] = None,
        result: Any = None,
        compose_in_background: bool = False,
    ) -> bool:
        """Write a synchronised (raw_frame, overlay_frame) pair without blocking detection.

        Guarantees 1-to-1 frame lockstep between the raw surgical video and the
        application overlay video: either BOTH frames are enqueued with the same
        frame index and timestamp, or BOTH are dropped together if the queue is full.
        """
        if not self.is_recording:
            return False

        if raw_frame is None or overlay_frame is None:
            return False

        ts_mono = time.monotonic()
        ts_wall = time.time()

        with self._lock:
            if not self._running or self._writer is None:
                return False

            # Lockstep check: ensure both queues have capacity so raw & overlay
            # never drift out of synchronisation by even a single frame.
            if self._dual_mode:
                if self._frame_queue.full() or self._raw_frame_queue.full():
                    self._dropped_frames += 1
                    return False
            else:
                if self._frame_queue.full():
                    self._dropped_frames += 1
                    return False

            rec_idx = self._queued_pairs
            src_idx = int(frame_index) if frame_index is not None else rec_idx
            elapsed = ts_mono - self._start_time if self._start_time > 0 else 0.0

            try:
                if self._dual_mode:
                    self._raw_frame_queue.put_nowait(
                        (raw_frame.copy(), src_idx, ts_mono)
                    )
                self._frame_queue.put_nowait(
                    (
                        overlay_frame.copy(),
                        ts_mono,
                        result,
                        bool(compose_in_background),
                    )
                )
                self._queued_pairs += 1
                self._sync_records.append(
                    (rec_idx, src_idx, ts_mono, elapsed, ts_wall)
                )
                return True
            except queue.Full:
                self._dropped_frames += 1
                return False

    def write(
        self,
        frame: np.ndarray,
        *,
        raw_frame: Optional[np.ndarray] = None,
        frame_index: Optional[int] = None,
        result: Any = None,
        compose_in_background: bool = False,
    ) -> bool:
        """Write a frame (or synchronised raw + overlay pair) to the recording.

        Thread-safe and non-blocking. Preserves full backward compatibility with
        single-argument `recorder.write(frame)` calls while supporting dual-stream
        synchronised recording.
        """
        if not self.is_recording:
            return False

        if self._dual_mode:
            raw_src = raw_frame if raw_frame is not None else frame
            return self.write_dual(
                raw_src,
                frame,
                frame_index=frame_index,
                result=result,
                compose_in_background=compose_in_background,
            )

        # Single-stream validation (only reject if dimensions are invalid)
        if frame is None or frame.shape[1] <= 0 or frame.shape[0] <= 0:
            return False
        if (
            not compose_in_background
            and (frame.shape[1] != self._width or frame.shape[0] != self._height)
        ):
            return False

        return self.write_dual(
            frame,
            frame,
            frame_index=frame_index,
            result=result,
            compose_in_background=compose_in_background,
        )

    def write_with_retry(self, frame: np.ndarray, max_retries: int = 3) -> bool:
        """Write a frame with retry logic for better frame preservation."""
        for _attempt in range(max_retries):
            if self.write(frame):
                return True
            time.sleep(0.001)
        return False

    def _write_sync_manifest(self) -> Optional[str]:
        """Write the frame-index and timestamp synchronisation CSV manifest."""
        sync_file = self._sync_path
        if not sync_file or not self._sync_records:
            return None
        try:
            Path(sync_file).parent.mkdir(parents=True, exist_ok=True)
            raw_name = Path(self._raw_path).name if self._raw_path else ""
            overlay_name = Path(self._path).name if self._path else ""
            with open(sync_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "recorded_frame_idx",
                        "source_frame_idx",
                        "elapsed_s",
                        "timestamp_monotonic_s",
                        "wall_timestamp_s",
                        "raw_video_file",
                        "overlay_video_file",
                    ]
                )
                for rec_idx, src_idx, ts_mono, elapsed, ts_wall in self._sync_records:
                    writer.writerow(
                        [
                            rec_idx,
                            src_idx,
                            f"{elapsed:.6f}",
                            f"{ts_mono:.6f}",
                            f"{ts_wall:.6f}",
                            raw_name,
                            overlay_name,
                        ]
                    )
            return sync_file
        except Exception:
            return None

    def stop(self) -> Optional[str]:
        """Stop recording, flush both raw and overlay streams, and write sync manifest.

        Returns:
            Path to the saved overlay video file, or None if not recording.
            Companion paths are available on `self.last_raw_path` and `self.last_sync_path`.
        """
        if not self.is_recording:
            return None

        path = self._path
        elapsed_now = (
            time.monotonic() - self._start_time if self._start_time > 0 else 0.0
        )

        # Signal writer threads to finish flushing remaining queued frames
        with self._lock:
            self._running = False
            self.last_elapsed_time = elapsed_now

        try:
            self._frame_queue.put_nowait(None)
        except queue.Full:
            pass

        if self._dual_mode:
            try:
                self._raw_frame_queue.put_nowait(None)
            except queue.Full:
                pass

        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        if self._raw_writer_thread is not None:
            self._raw_writer_thread.join(timeout=5.0)

        # Write synchronisation manifest CSV
        self._write_sync_manifest()

        with self._lock:
            final_frames = self._frame_count
            final_raw_frames = self._raw_frame_count
            final_elapsed = (
                self.last_elapsed_time if self.last_elapsed_time > 0 else elapsed_now
            )
            self.last_frame_count = final_frames
            self.last_raw_frame_count = final_raw_frames
            self.last_elapsed_time = final_elapsed

        if self._status_callback:
            self._status_callback(
                {
                    "is_recording": False,
                    "is_dual": self._dual_mode,
                    "frame_count": final_frames,
                    "raw_frame_count": final_raw_frames,
                    "elapsed_time": final_elapsed,
                    "fps": final_frames / final_elapsed if final_elapsed > 0 else 0,
                    "dropped_frames": self._dropped_frames,
                    "queue_size": 0,
                    "codec": self._codec,
                    "hw_backend": self._hw_backend,
                }
            )

        with self._lock:
            self._writer = None
            self._raw_writer = None
            self._path = None
            self._raw_path = None
            self._start_time = 0.0
            self._running = False

        return path

    def pause(self) -> None:
        """Pause recording (frames will be dropped)."""
        with self._lock:
            self._running = False

    def resume(self) -> None:
        """Resume recording."""
        with self._lock:
            if self._writer is not None:
                self._running = True

    def get_status(self) -> dict:
        """Get current recording status."""
        with self._lock:
            elapsed = (
                time.monotonic() - self._start_time
                if (self._running and self._start_time > 0)
                else self.last_elapsed_time
            )
            frames = self._frame_count if self._running else self.last_frame_count
            raw_frames = (
                self._raw_frame_count if self._running else self.last_raw_frame_count
            )
            return {
                "is_recording": self._running and self._writer is not None,
                "is_dual": self._dual_mode,
                "frame_count": frames,
                "raw_frame_count": raw_frames,
                "elapsed_time": elapsed,
                "fps": frames / elapsed if elapsed > 0 else 0,
                "dropped_frames": self._dropped_frames,
                "queue_size": self._frame_queue.qsize()
                if hasattr(self, "_frame_queue")
                else 0,
                "raw_queue_size": self._raw_frame_queue.qsize()
                if hasattr(self, "_raw_frame_queue")
                else 0,
                "path": self._path or self.last_overlay_path,
                "raw_path": self._raw_path or self.last_raw_path,
                "sync_path": self._sync_path or self.last_sync_path,
                "codec": self._codec,
                "hw_backend": self._hw_backend,
            }


class DualFrameRecorder(FrameRecorder):
    """Dedicated Dual Video Recorder for the Centration Application.

    Simultaneously records:
      1. The original J&J surgical screen feed (raw video for iris registration model training)
      2. The centration application overlay feed (processed detection & values for validation)
    with Intel Quick Sync Video (QSV) hardware encoding, H.264/H.265 compression,
    and frame-level timestamp/index synchronisation in separate background threads.
    """

    def __init__(self, prefer_qsv: bool = False, max_queue_size: int = 90) -> None:
        super().__init__(prefer_qsv=prefer_qsv, max_queue_size=max_queue_size)

