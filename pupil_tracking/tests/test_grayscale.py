"""
Unit tests for the grayscale detection pipeline.

Tests cover:
    - GrayscaleHandler: detection, conversion, enhancement, model input
    - GrayscaleMode: enum parsing and cycling
    - GrayscaleInfo: immutability and field correctness
    - Thread safety: concurrent calls to handler methods
    - Edge cases: empty images, float images, RGBA, single-channel

Run with:
    python -m pytest pupil_tracking/tests/test_grayscale.py -v
    python -m pytest pupil_tracking/tests/test_grayscale.py -v -x  # stop on first failure
"""

from __future__ import annotations

import threading
from typing import List, Tuple

import cv2
import numpy as np
import pytest

from pupil_tracking.preprocessing.grayscale_handler import (
    GrayscaleHandler,
    GrayscaleInfo,
    GrayscaleMode,
)


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def handler() -> GrayscaleHandler:
    """Default GrayscaleHandler instance."""
    return GrayscaleHandler(
        clahe_clip_limit=3.0,
        clahe_grid_size=(8, 8),
        channel_diff_threshold=3.0,
    )


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Synthetic 200×200 RGB image with distinct channel values.

    Creates an image where R, G, B channels are clearly different
    so it should NOT be detected as grayscale.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Blue
    img[:, :, 1] = 120  # Green
    img[:, :, 2] = 200  # Red
    # Add some gradient to make it realistic
    for i in range(200):
        img[i, :, 0] = min(255, 50 + i // 3)
        img[i, :, 1] = min(255, 120 - i // 5)
    return img


@pytest.fixture
def gray_single_channel() -> np.ndarray:
    """Single-channel 200×200 grayscale image."""
    img = np.zeros((200, 200), dtype=np.uint8)
    # Circular gradient simulating an eye
    cy, cx = 100, 100
    for y in range(200):
        for x in range(200):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            img[y, x] = max(0, min(255, int(180 - dist * 1.5)))
    return img


@pytest.fixture
def gray_replicated(gray_single_channel: np.ndarray) -> np.ndarray:
    """3-channel image with identical channels (fake-RGB grayscale).

    This simulates a grayscale image saved as JPEG/PNG in BGR format.
    """
    return np.stack([
        gray_single_channel,
        gray_single_channel,
        gray_single_channel,
    ], axis=2)


@pytest.fixture
def low_contrast_gray() -> np.ndarray:
    """Very low contrast grayscale image.

    Pixel values concentrated in [80, 120] range.
    Enhancement should significantly improve contrast.
    """
    rng = np.random.RandomState(42)
    img = rng.randint(80, 121, (200, 200), dtype=np.uint8)
    return img


@pytest.fixture
def rgba_image(rgb_image: np.ndarray) -> np.ndarray:
    """4-channel RGBA image."""
    alpha = np.full(
        (rgb_image.shape[0], rgb_image.shape[1], 1),
        255,
        dtype=np.uint8,
    )
    return np.concatenate([rgb_image, alpha], axis=2)


@pytest.fixture
def float_image(gray_single_channel: np.ndarray) -> np.ndarray:
    """Float32 image with values in [0, 1]."""
    return gray_single_channel.astype(np.float32) / 255.0


# ================================================================
# GrayscaleMode Tests
# ================================================================

class TestGrayscaleMode:
    """Tests for the GrayscaleMode enum."""

    def test_from_string_auto(self):
        assert GrayscaleMode.from_string("auto") == GrayscaleMode.AUTO

    def test_from_string_force(self):
        assert GrayscaleMode.from_string("force") == GrayscaleMode.FORCE

    def test_from_string_off(self):
        assert GrayscaleMode.from_string("off") == GrayscaleMode.OFF

    def test_from_string_case_insensitive(self):
        assert GrayscaleMode.from_string("AUTO") == GrayscaleMode.AUTO
        assert GrayscaleMode.from_string("Force") == GrayscaleMode.FORCE
        assert GrayscaleMode.from_string("OFF") == GrayscaleMode.OFF

    def test_from_string_whitespace(self):
        assert GrayscaleMode.from_string("  auto  ") == GrayscaleMode.AUTO

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown grayscale mode"):
            GrayscaleMode.from_string("invalid")

    def test_from_string_empty(self):
        with pytest.raises(ValueError):
            GrayscaleMode.from_string("")

    def test_all_modes_exist(self):
        modes = list(GrayscaleMode)
        assert len(modes) == 3
        assert GrayscaleMode.AUTO in modes
        assert GrayscaleMode.FORCE in modes
        assert GrayscaleMode.OFF in modes


# ================================================================
# GrayscaleHandler — is_grayscale() Tests
# ================================================================

class TestIsGrayscale:
    """Tests for GrayscaleHandler.is_grayscale()."""

    def test_single_channel_detected(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """A 2-D array should always be detected as grayscale."""
        assert handler.is_grayscale(gray_single_channel) is True

    def test_single_channel_3d(self, handler: GrayscaleHandler):
        """Shape (H, W, 1) should be detected as grayscale."""
        img = np.zeros((100, 100, 1), dtype=np.uint8)
        assert handler.is_grayscale(img) is True

    def test_replicated_channels_detected(
        self, handler: GrayscaleHandler, gray_replicated: np.ndarray
    ):
        """3-channel image with identical channels should be grayscale."""
        assert handler.is_grayscale(gray_replicated) is True

    def test_rgb_not_detected(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """A true RGB image should NOT be detected as grayscale."""
        assert handler.is_grayscale(rgb_image) is False

    def test_near_grayscale_with_noise(self, handler: GrayscaleHandler):
        """Channels with tiny differences (< threshold) → grayscale."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        # Add noise < 3.0 mean diff
        rng = np.random.RandomState(42)
        noise = rng.randint(-1, 2, (100, 100), dtype=np.int16)
        ch0 = np.clip(gray.astype(np.int16), 0, 255).astype(np.uint8)
        ch1 = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        ch2 = np.clip(gray.astype(np.int16) - noise, 0, 255).astype(np.uint8)
        img = np.stack([ch0, ch1, ch2], axis=2)
        assert handler.is_grayscale(img) is True

    def test_rgba_rgb_portion_checked(
        self, handler: GrayscaleHandler, rgba_image: np.ndarray
    ):
        """RGBA image — only RGB portion matters."""
        # rgba_image has distinct R/G/B → not grayscale
        assert handler.is_grayscale(rgba_image) is False

    def test_float_image(self, handler: GrayscaleHandler, float_image: np.ndarray):
        """Float [0,1] single-channel image should be grayscale."""
        assert handler.is_grayscale(float_image) is True

    def test_large_image_performance(self, handler: GrayscaleHandler):
        """is_grayscale should handle large images quickly via downsampling."""
        img = np.full((4000, 4000, 3), 128, dtype=np.uint8)
        # Should complete in < 100ms due to downsampling
        import time
        t0 = time.time()
        result = handler.is_grayscale(img)
        elapsed = time.time() - t0
        assert result is True
        assert elapsed < 0.5, f"is_grayscale took {elapsed:.3f}s on 4K image"

    def test_empty_image_raises(self, handler: GrayscaleHandler):
        """Empty image should raise ValueError."""
        img = np.array([], dtype=np.uint8).reshape(0, 0)
        with pytest.raises(ValueError, match="empty"):
            handler.is_grayscale(img)

    def test_invalid_type_raises(self, handler: GrayscaleHandler):
        """Non-ndarray should raise TypeError."""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            handler.is_grayscale([[1, 2], [3, 4]])  # type: ignore

    def test_5d_array_raises(self, handler: GrayscaleHandler):
        """5-D array should raise ValueError."""
        img = np.zeros((1, 1, 1, 1, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            handler.is_grayscale(img)


# ================================================================
# GrayscaleHandler — to_grayscale() Tests
# ================================================================

class TestToGrayscale:
    """Tests for GrayscaleHandler.to_grayscale()."""

    def test_single_channel_passthrough(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Single-channel image should be returned as-is."""
        result = handler.to_grayscale(gray_single_channel)
        assert result.ndim == 2
        assert np.array_equal(result, gray_single_channel)

    def test_3ch_to_single(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """3-channel image should become 2-D."""
        result = handler.to_grayscale(rgb_image)
        assert result.ndim == 2
        assert result.shape == rgb_image.shape[:2]
        assert result.dtype == np.uint8

    def test_rgba_drops_alpha(
        self, handler: GrayscaleHandler, rgba_image: np.ndarray
    ):
        """4-channel image — alpha dropped, then converted."""
        result = handler.to_grayscale(rgba_image)
        assert result.ndim == 2
        assert result.shape == rgba_image.shape[:2]

    def test_shape_1ch_squeezed(self, handler: GrayscaleHandler):
        """(H, W, 1) should be squeezed to (H, W)."""
        img = np.full((50, 50, 1), 100, dtype=np.uint8)
        result = handler.to_grayscale(img)
        assert result.shape == (50, 50)
        assert np.all(result == 100)


# ================================================================
# GrayscaleHandler — enhance_grayscale() Tests
# ================================================================

class TestEnhanceGrayscale:
    """Tests for GrayscaleHandler.enhance_grayscale()."""

    def test_output_is_uint8(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Output should always be uint8."""
        result = handler.enhance_grayscale(gray_single_channel)
        assert result.dtype == np.uint8

    def test_output_same_shape(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Output should have same shape as input."""
        result = handler.enhance_grayscale(gray_single_channel)
        assert result.shape == gray_single_channel.shape

    def test_improves_low_contrast(
        self, handler: GrayscaleHandler, low_contrast_gray: np.ndarray
    ):
        """Enhancement should increase standard deviation of low-contrast image."""
        std_before = float(np.std(low_contrast_gray.astype(np.float32)))
        enhanced = handler.enhance_grayscale(low_contrast_gray)
        std_after = float(np.std(enhanced.astype(np.float32)))
        assert std_after > std_before, (
            f"Enhancement did not improve contrast: "
            f"before={std_before:.2f}, after={std_after:.2f}"
        )

    def test_does_not_destroy_high_contrast(self, handler: GrayscaleHandler):
        """A well-contrasted image should not be degraded."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 20   # dark pupil region
        img[50:, :] = 220  # bright sclera region
        std_before = float(np.std(img.astype(np.float32)))
        enhanced = handler.enhance_grayscale(img)
        std_after = float(np.std(enhanced.astype(np.float32)))
        # Should not drop by more than 30%
        assert std_after >= std_before * 0.7, (
            f"Enhancement degraded good contrast: "
            f"before={std_before:.2f}, after={std_after:.2f}"
        )

    def test_rejects_3d_input(self, handler: GrayscaleHandler):
        """3-D input should raise ValueError."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="2-D array"):
            handler.enhance_grayscale(img)

    def test_float_input_converted(self, handler: GrayscaleHandler):
        """Float [0,1] input should be converted to uint8 internally."""
        img = np.random.rand(100, 100).astype(np.float32) * 0.5
        result = handler.enhance_grayscale(img)
        assert result.dtype == np.uint8
        assert result.shape == (100, 100)

    def test_pixel_values_in_range(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Output values should be in [0, 255]."""
        result = handler.enhance_grayscale(gray_single_channel)
        assert result.min() >= 0
        assert result.max() <= 255


# ================================================================
# GrayscaleHandler — to_model_input() Tests
# ================================================================

class TestToModelInput:
    """Tests for GrayscaleHandler.to_model_input()."""

    def test_rgb_passthrough_when_not_forced(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """RGB image with force=False should pass through unchanged."""
        result, info = handler.to_model_input(rgb_image, force_grayscale=False)
        assert result.shape == rgb_image.shape
        assert result.dtype == np.uint8
        assert info.conversion_applied is False
        assert info.was_grayscale is False
        assert info.mode_used == GrayscaleMode.OFF

    def test_rgb_forced_to_grayscale(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """RGB image with force=True should be converted."""
        result, info = handler.to_model_input(rgb_image, force_grayscale=True)
        assert result.shape[2] == 3  # still 3 channels
        assert result.dtype == np.uint8
        assert info.conversion_applied is True
        assert info.was_grayscale is False
        assert info.mode_used == GrayscaleMode.FORCE
        # All 3 channels should be identical (replicated)
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])

    def test_gray_auto_detected_and_enhanced(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Single-channel grayscale should be auto-detected and enhanced."""
        result, info = handler.to_model_input(
            gray_single_channel, force_grayscale=False
        )
        assert result.shape == (200, 200, 3)
        assert result.dtype == np.uint8
        assert info.conversion_applied is True
        assert info.was_grayscale is True
        assert info.mode_used == GrayscaleMode.AUTO

    def test_fake_rgb_gray_detected(
        self, handler: GrayscaleHandler, gray_replicated: np.ndarray
    ):
        """3-channel fake-RGB grayscale should be detected and enhanced."""
        result, info = handler.to_model_input(
            gray_replicated, force_grayscale=False
        )
        assert result.shape == gray_replicated.shape
        assert info.conversion_applied is True
        assert info.was_grayscale is True

    def test_output_always_3ch(self, handler: GrayscaleHandler):
        """Output should always be (H, W, 3) regardless of input."""
        # 1-channel
        img1 = np.zeros((50, 50), dtype=np.uint8)
        r1, _ = handler.to_model_input(img1)
        assert r1.shape == (50, 50, 3)

        # 3-channel
        img3 = np.zeros((50, 50, 3), dtype=np.uint8)
        r3, _ = handler.to_model_input(img3)
        assert r3.shape == (50, 50, 3)

        # 4-channel
        img4 = np.zeros((50, 50, 4), dtype=np.uint8)
        r4, _ = handler.to_model_input(img4)
        assert r4.shape == (50, 50, 3)

    def test_output_always_uint8(
        self, handler: GrayscaleHandler, float_image: np.ndarray
    ):
        """Output should always be uint8."""
        result, _ = handler.to_model_input(float_image, force_grayscale=True)
        assert result.dtype == np.uint8

    def test_contrast_reported_in_info(
        self, handler: GrayscaleHandler, low_contrast_gray: np.ndarray
    ):
        """GrayscaleInfo should report contrast before and after."""
        _, info = handler.to_model_input(low_contrast_gray, force_grayscale=True)
        assert info.contrast_before > 0
        assert info.contrast_after > 0
        assert info.contrast_after >= info.contrast_before  # enhancement improves

    def test_info_immutable(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """GrayscaleInfo should be frozen (immutable)."""
        _, info = handler.to_model_input(gray_single_channel)
        with pytest.raises(AttributeError):
            info.was_grayscale = False  # type: ignore

    def test_info_original_channels_correct(
        self, handler: GrayscaleHandler
    ):
        """original_channels should match actual input channel count."""
        img1 = np.zeros((50, 50), dtype=np.uint8)
        _, info1 = handler.to_model_input(img1)
        assert info1.original_channels == 1

        img3 = np.zeros((50, 50, 3), dtype=np.uint8)
        _, info3 = handler.to_model_input(img3, force_grayscale=True)
        assert info3.original_channels == 3


# ================================================================
# GrayscaleHandler — get_quality_metrics() Tests
# ================================================================

class TestQualityMetrics:
    """Tests for GrayscaleHandler.get_quality_metrics()."""

    def test_returns_all_keys(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """Should return all expected metric keys."""
        metrics = handler.get_quality_metrics(rgb_image)
        expected_keys = {
            "contrast",
            "dynamic_range",
            "mean_intensity",
            "snr_estimate",
            "is_grayscale",
            "channel_variance",
        }
        assert set(metrics.keys()) == expected_keys

    def test_contrast_positive(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """Contrast should be positive for non-uniform images."""
        metrics = handler.get_quality_metrics(rgb_image)
        assert metrics["contrast"] > 0

    def test_is_grayscale_flag(
        self,
        handler: GrayscaleHandler,
        rgb_image: np.ndarray,
        gray_single_channel: np.ndarray,
    ):
        """is_grayscale metric should match is_grayscale() method."""
        rgb_metrics = handler.get_quality_metrics(rgb_image)
        assert rgb_metrics["is_grayscale"] == 0.0

        gray_metrics = handler.get_quality_metrics(gray_single_channel)
        assert gray_metrics["is_grayscale"] == 1.0

    def test_channel_variance_zero_for_gray(
        self, handler: GrayscaleHandler, gray_replicated: np.ndarray
    ):
        """Channel variance should be ~0 for replicated grayscale."""
        metrics = handler.get_quality_metrics(gray_replicated)
        assert metrics["channel_variance"] < 1.0

    def test_channel_variance_positive_for_rgb(
        self, handler: GrayscaleHandler, rgb_image: np.ndarray
    ):
        """Channel variance should be clearly positive for RGB."""
        metrics = handler.get_quality_metrics(rgb_image)
        assert metrics["channel_variance"] > 10.0

    def test_dynamic_range_reasonable(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """Dynamic range should be in [0, 255]."""
        metrics = handler.get_quality_metrics(gray_single_channel)
        assert 0 <= metrics["dynamic_range"] <= 255

    def test_snr_positive(
        self, handler: GrayscaleHandler, gray_single_channel: np.ndarray
    ):
        """SNR estimate should be positive."""
        metrics = handler.get_quality_metrics(gray_single_channel)
        assert metrics["snr_estimate"] > 0


# ================================================================
# GrayscaleHandler — Construction & Validation Tests
# ================================================================

class TestConstruction:
    """Tests for GrayscaleHandler construction and parameter validation."""

    def test_default_construction(self):
        """Default construction should work without errors."""
        handler = GrayscaleHandler()
        assert handler is not None

    def test_custom_parameters(self):
        """Custom parameters should be accepted."""
        handler = GrayscaleHandler(
            clahe_clip_limit=5.0,
            clahe_grid_size=(16, 16),
            channel_diff_threshold=5.0,
        )
        assert handler is not None

    def test_invalid_clip_limit(self):
        with pytest.raises(ValueError, match="positive"):
            GrayscaleHandler(clahe_clip_limit=-1.0)

    def test_invalid_grid_size(self):
        with pytest.raises(ValueError, match="positive integers"):
            GrayscaleHandler(clahe_grid_size=(0, 8))

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="non-negative"):
            GrayscaleHandler(channel_diff_threshold=-0.5)

    def test_repr(self):
        handler = GrayscaleHandler(clahe_clip_limit=2.0)
        r = repr(handler)
        assert "GrayscaleHandler" in r
        assert "2.0" in r


# ================================================================
# Thread Safety Tests
# ================================================================

class TestThreadSafety:
    """Tests for concurrent access to GrayscaleHandler."""

    def test_concurrent_is_grayscale(self, handler: GrayscaleHandler):
        """Multiple threads calling is_grayscale should not crash."""
        results: List[bool] = []
        errors: List[Exception] = []

        img = np.full((100, 100, 3), 128, dtype=np.uint8)

        def worker():
            try:
                for _ in range(50):
                    r = handler.is_grayscale(img)
                    results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 8 * 50
        assert all(r is True for r in results)

    def test_concurrent_enhance(self, handler: GrayscaleHandler):
        """Multiple threads calling enhance_grayscale should not crash."""
        errors: List[Exception] = []
        results: List[Tuple[int, int]] = []

        img = np.random.randint(50, 150, (100, 100), dtype=np.uint8)

        def worker():
            try:
                for _ in range(30):
                    enhanced = handler.enhance_grayscale(img.copy())
                    results.append(enhanced.shape)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(s == (100, 100) for s in results)

    def test_concurrent_to_model_input(self, handler: GrayscaleHandler):
        """Multiple threads calling to_model_input should not crash."""
        errors: List[Exception] = []
        shapes: List[Tuple[int, ...]] = []

        img = np.random.randint(0, 255, (80, 80), dtype=np.uint8)

        def worker():
            try:
                for _ in range(20):
                    result, info = handler.to_model_input(
                        img.copy(), force_grayscale=True
                    )
                    shapes.append(result.shape)
                    assert info.conversion_applied is True
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert all(s == (80, 80, 3) for s in shapes)


# ================================================================
# Edge Case Tests
# ================================================================

class TestEdgeCases:
    """Tests for unusual inputs and boundary conditions."""

    def test_uniform_black_image(self, handler: GrayscaleHandler):
        """All-black image should work without errors."""
        img = np.zeros((50, 50), dtype=np.uint8)
        result, info = handler.to_model_input(img)
        assert result.shape == (50, 50, 3)
        assert info.was_grayscale is True

    def test_uniform_white_image(self, handler: GrayscaleHandler):
        """All-white image should work without errors."""
        img = np.full((50, 50), 255, dtype=np.uint8)
        result, info = handler.to_model_input(img)
        assert result.shape == (50, 50, 3)

    def test_single_pixel_image(self, handler: GrayscaleHandler):
        """1×1 image should work."""
        img = np.array([[128]], dtype=np.uint8)
        assert handler.is_grayscale(img) is True
        result, info = handler.to_model_input(img)
        assert result.shape == (1, 1, 3)

    def test_very_small_image(self, handler: GrayscaleHandler):
        """2×2 image should work."""
        img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        result, info = handler.to_model_input(img, force_grayscale=True)
        assert result.shape == (2, 2, 3)

    def test_non_square_image(self, handler: GrayscaleHandler):
        """Non-square image should work correctly."""
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        result, info = handler.to_model_input(img, force_grayscale=True)
        assert result.shape == (100, 300, 3)

    def test_float64_image(self, handler: GrayscaleHandler):
        """Float64 image should be handled."""
        img = np.random.rand(50, 50).astype(np.float64) * 0.8
        result, info = handler.to_model_input(img)
        assert result.dtype == np.uint8
        assert result.shape == (50, 50, 3)

    def test_float_image_over_1(self, handler: GrayscaleHandler):
        """Float image with values > 1 should be clipped to [0, 255]."""
        img = np.random.rand(50, 50).astype(np.float32) * 300
        result, info = handler.to_model_input(img)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_uint16_image(self, handler: GrayscaleHandler):
        """uint16 image should be converted to uint8."""
        img = np.random.randint(0, 65535, (50, 50, 3), dtype=np.uint16)
        result, info = handler.to_model_input(img, force_grayscale=True)
        assert result.dtype == np.uint8

    def test_replicated_channels_after_force(self, handler: GrayscaleHandler):
        """After force grayscale, all 3 output channels must be identical."""
        img = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        # Make it clearly RGB
        img[:, :, 0] = 50
        img[:, :, 1] = 150
        img[:, :, 2] = 200

        result, info = handler.to_model_input(img, force_grayscale=True)
        assert info.conversion_applied is True
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])


# ================================================================
# Integration-style test (end-to-end)
# ================================================================

class TestEndToEnd:
    """Integration tests simulating real pipeline usage."""

    def test_full_pipeline_gray_input(self, handler: GrayscaleHandler):
        """Simulate: grayscale camera → handler → model-ready output."""
        # Simulate grayscale camera image (single channel)
        camera_frame = np.random.randint(30, 180, (480, 640), dtype=np.uint8)

        # Step 1: Detect
        assert handler.is_grayscale(camera_frame) is True

        # Step 2: Prepare for model
        model_input, info = handler.to_model_input(
            camera_frame, force_grayscale=False
        )

        # Verify
        assert model_input.shape == (480, 640, 3)
        assert model_input.dtype == np.uint8
        assert info.conversion_applied is True
        assert info.was_grayscale is True
        assert info.original_channels == 1
        assert info.contrast_after >= info.contrast_before

    def test_full_pipeline_rgb_auto_mode(self, handler: GrayscaleHandler):
        """Simulate: RGB camera → auto mode → passthrough."""
        camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Make channels distinct
        camera_frame[:, :, 0] = np.random.randint(0, 100, (480, 640), dtype=np.uint8)
        camera_frame[:, :, 1] = np.random.randint(100, 200, (480, 640), dtype=np.uint8)
        camera_frame[:, :, 2] = np.random.randint(150, 255, (480, 640), dtype=np.uint8)

        model_input, info = handler.to_model_input(
            camera_frame, force_grayscale=False
        )

        assert model_input.shape == camera_frame.shape
        assert info.conversion_applied is False
        assert info.was_grayscale is False

    def test_full_pipeline_rgb_force_mode(self, handler: GrayscaleHandler):
        """Simulate: RGB camera → force grayscale → enhanced output."""
        camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        camera_frame[:, :, 0] = 50
        camera_frame[:, :, 1] = 150
        camera_frame[:, :, 2] = 200

        model_input, info = handler.to_model_input(
            camera_frame, force_grayscale=True
        )

        assert model_input.shape == (480, 640, 3)
        assert info.conversion_applied is True
        assert info.was_grayscale is False
        assert info.mode_used == GrayscaleMode.FORCE
        # Channels should be replicated
        assert np.array_equal(model_input[:, :, 0], model_input[:, :, 1])

    def test_quality_metrics_consistency(self, handler: GrayscaleHandler):
        """Quality metrics should be consistent with is_grayscale."""
        gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        metrics = handler.get_quality_metrics(gray)
        assert metrics["is_grayscale"] == 1.0
        assert metrics["channel_variance"] < 1.0

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = 50
        rgb[:, :, 1] = 150
        rgb[:, :, 2] = 200
        metrics = handler.get_quality_metrics(rgb)
        assert metrics["is_grayscale"] == 0.0
        assert metrics["channel_variance"] > 10.0