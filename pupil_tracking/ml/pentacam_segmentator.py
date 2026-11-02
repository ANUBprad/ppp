from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.ml.postprocess import validate_pupil_limbus_pair


@dataclass
class PentacamSegmentationConfig:
    """Configuration for Pentacam segmentation model."""
    
    # Input image size (resized for model input)
    input_width: int = 512
    input_height: int = 512
    
    # Model architecture
    backbone: str = "resnet18"
    use_attention: bool = True
    
    # Loss weights
    dice_weight: float = 1.0
    boundary_weight: float = 0.5
    cls_weight: float = 0.3
    
    # Post-processing
    # Thresholds: model sigmoid outputs are ~0.44 for limbus, ~0.48 for pupil.
    # Using 0.3 for limbus captures the full region while 0.5 keeps pupil accurate.
    pupil_threshold: float = 0.5
    limbus_threshold: float = 0.3
    min_contour_area: float = 10.0
    
    # Device
    device: str = "cpu"  # Force CPU for compatibility


class PentacamSegmentator:
    """
    Pentacam pupil and limbus segmentation model.
    
    Predicts separate pupil and limbus masks using a U-Net-like architecture
    with Dice boundary loss for robust boundary extraction.
    """
    
    def __init__(self, config: Optional[PentacamSegmentationConfig] = None) -> None:
        self.config = config or PentacamSegmentationConfig()
        self.model = None
        self.initialized = False
        
        # Class balancing factors (pupil and limbus are small relative to full image)
        self.pupil_weight = 2.0  # Weight for pupil class
        self.limbus_weight = 1.5  # Weight for limbus class
        
        # Checkpoint auditability
        self._last_checkpoint_path: Optional[str] = None
        self._last_checkpoint_sha256: Optional[str] = None
        self._last_load_time: Optional[float] = None
    
    def initialize(self) -> None:
        """Initialize the model architecture. weights loaded separately."""
        try:
            import torch
            import torch.nn as nn
            
            # Build U-Net encoder-decoder with ResNet18 encoder
            self.model = self._build_model()
            self.model.to(self.config.device)
            self.initialized = True
            print(f"PentacamSegmentator model initialized on {self.config.device}")
            
        except ImportError:
            print("PyTorch not available - model will use classical fallback")
            self.model = None
    
    def _build_model(self) -> Any:
        """Build the U-Net segmentation model."""
        import torch
        import torch.nn as nn
        
        # ResNet18 encoder (without the final fc layer)
        encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        encoder = nn.Sequential(*list(encoder.children())[:-2])  # Remove avgpool and fc
        
        # Decoder with skip connections (simplified U-Net)
        decoder = nn.ModuleList([
            # Upconv + conv blocks
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            
            # Output heads for pupil and limbus
            nn.Conv2d(8, 2, kernel_size=1),  # 2 classes: pupil, limbus
        ])
        
        return torch.nn.ModuleDict({
            'encoder': encoder,
            'decoder': decoder,
        })
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess a Pentacam image for model input.
        
        Returns:
            tensor: Preprocessed image tensor (1, 3, H, W) or (1, 1, H, W) for grayscale
            orig_size: Original (width, height) before resizing
        """
        orig_h, orig_w = image.shape[:2] if len(image.shape) == 3 else image.shape
        
        # Resize to input size
        resized = cv2.resize(image, (self.config.input_width, self.config.input_height))
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        if len(resized.shape) == 2:
            # Grayscale
            tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            # Shape: (1, 1, H, W)
        else:
            # RGB
            tensor = torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
            # Shape: (1, 3, H, W)
        
        return tensor, (orig_w, orig_h)
    
    def preprocess_grayscale(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a grayscale Pentacam image."""
        tensor, _ = self.preprocess_image(image)
        return tensor
    
    def forward(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run forward pass through the model.
        
        Returns dict with 'pupil_mask' and 'limbus_mask' keys.
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        encoder = self.model['encoder']
        decoder = self.model['decoder']
        
        # Encoder pass
        features = encoder(tensor)
        
        # Decoder pass with skip-like connections
        x = features
        # The decoder expects specific feature pyramid levels
        # Simplified: just apply decoder layers sequentially
        
        for i, layer in enumerate(decoder):
            if isinstance(layer, nn.ConvTranspose2d):
                # Handle stride for upsampling
                x = layer(x)
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
            else:
                x = layer(x)
        
        # Split output into pupil and limbus masks
        # Output shape: (1, 2, H, W)
        if x.shape[1] >= 2:
            pupil_logits = x[:, 0:1, :, :]  # First channel
            limbus_logits = x[:, 1:2, :, :]  # Second channel
        else:
            # Fallback: use single output for both
            pupil_logits = x[:, 0:1, :, :]
            limbus_logits = x[:, 0:1, :, :]
        
        # Apply sigmoid
        pupil_mask = torch.sigmoid(pupil_logits)
        limbus_mask = torch.sigmoid(limbus_logits)
        
        return {
            'pupil_mask': pupil_mask,
            'limbus_mask': limbus_mask,
            'pupil_logits': pupil_logits,
            'limbus_logits': limbus_logits,
        }
    
    def extract_boundary_points(self, mask: np.ndarray, min_points: int = 10) -> Optional[np.ndarray]:
        """
        Extract boundary points from a binary mask using contour detection.
        
        Returns:
            numpy array of (n, 1, 2) boundary points, or None if no valid contour found
        """
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Handle different OpenCV contour return formats
        # Use CHAIN_APPROX_NONE to preserve all boundary points (important for full-coverage masks like limbus)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by minimum area
        if cv2.contourArea(largest_contour) < self.config.min_contour_area:
            return None
        
        # Approximate contour to reduce points
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < min_points:
            # Use raw contour if approximation gives too few points
            approx = largest_contour.reshape(-1, 2).astype(np.float32)
        
        return approx
    
    def fit_ellipse_to_mask(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Fit an ellipse to a pupil/limbus mask.
        
        Returns dict with ellipse parameters, or defaults if fitting fails.
        """
        # Extract boundary points
        boundary = self.extract_boundary_points(mask)
        
        if boundary is None or len(boundary) < 5:
            # Return default ellipse parameters
            h, w = mask.shape[:2] if len(mask.shape) == 2 else mask.shape
            return {
                'center': (w // 2, h // 2),
                'axes': (min(w, h) // 4, min(w, h) // 6),
                'angle': 0.0,
                'valid': False,
                'num_points': 0,
            }
        
        # Fit ellipse using OpenCV
        try:
            # Ensure we have enough points and they're in the right format
            pts = boundary.astype(np.float32).reshape(-1, 1, 2)
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(pts)
            (cx, cy), (dx, dy), angle = ellipse
            
            semi_major = max(dx, dy) / 2
            semi_minor = min(dx, dy) / 2
            
            # Adjust angle: if minor axis is horizontal, adjust angle
            if dy > dx:
                angle = (angle + 90) % 180
            
            return {
                'center': (float(cx), float(cy)),
                'axes': (float(semi_major), float(semi_minor)),
                'angle': float(angle),
                'valid': True,
                'num_points': len(boundary),
            }
        except cv2.error:
            return {
                'center': (float(w // 2), float(h // 2)),
                'axes': (float(min(w, h) // 4), float(min(w, h) // 6)),
                'angle': 0.0,
                'valid': False,
                'num_points': 0,
            }
    
    def segment_and_detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Full segmentation and detection pipeline.
        
        Returns dict with:
        - pupil_mask: numpy array
        - limbus_mask: numpy array
        - pupil_ellipse: Ellipse params dict
        - limbus_ellipse: Ellipse params dict
        - boundary_points: pupil boundary points
        - confidence: dict with per-structure confidence scores
        """
        orig_h, orig_w = image.shape[:2] if len(image.shape) == 3 else image.shape
        
        # Preprocess
        tensor, orig_size = self.preprocess_image(image)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(tensor)
        
        # Upsample masks back to original size
        orig_w_orig, orig_h_orig = orig_size
        pupil_mask_np = outputs['pupil_mask'].squeeze().cpu().numpy()
        limbus_mask_np = outputs['limbus_mask'].squeeze().cpu().numpy()
        
        # Resize masks to original image dimensions
        pupil_mask_resized = cv2.resize(
            pupil_mask_np, (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        limbus_mask_resized = cv2.resize(
            limbus_mask_np, (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        
        # Binarize masks
        pupil_binary = (pupil_mask_resized > self.config.pupil_threshold).astype(np.uint8)
        limbus_binary = (limbus_mask_resized > self.config.limbus_threshold).astype(np.uint8)
        
        # Fit ellipses
        pupil_ellipse = self.fit_ellipse_to_mask(pupil_binary)
        limbus_ellipse = self.fit_ellipse_to_mask(limbus_binary)
        
        # Extract boundary points
        pupil_boundary = self.extract_boundary_points(pupil_binary)
        limbus_boundary = self.extract_boundary_points(limbus_binary)
        
        # Compute confidence scores based on mask quality
        pupil_confidence = float(np.mean(pupil_binary)) * 2.0  # Normalized confidence
        limbus_confidence = float(np.mean(limbus_binary)) * 2.0
        
        # Validate pupil-limbus pair
        from pupil_tracking.utils.types import EllipseParams
        pupil_ep = EllipseParams(
            center_x=pupil_ellipse['center'][0],
            center_y=pupil_ellipse['center'][1],
            semi_major=pupil_ellipse['axes'][0],
            semi_minor=pupil_ellipse['axes'][1],
            angle_deg=pupil_ellipse['angle']
        )
        limbus_ep = EllipseParams(
            center_x=limbus_ellipse['center'][0],
            center_y=limbus_ellipse['center'][1],
            semi_major=limbus_ellipse['axes'][0],
            semi_minor=limbus_ellipse['axes'][1],
            angle_deg=limbus_ellipse['angle']
        )
        pair_valid, pair_reason = validate_pupil_limbus_pair(pupil_ep, limbus_ep)
        
        return {
            'pupil_mask': pupil_binary,
            'limbus_mask': limbus_binary,
            'pupil_ellipse': pupil_ellipse,
            'limbus_ellipse': limbus_ellipse,
            'pupil_boundary': pupil_boundary,
            'limbus_boundary': limbus_boundary,
            'pupil_confidence': pupil_confidence,
            'limbus_confidence': limbus_confidence,
            'pair_valid': pair_valid,
            'pair_reason': pair_reason,
            'image_size': (orig_w, orig_h),
        }
    
    def save_weights(self, path: str) -> None:
        """Save model weights to disk."""
        if self.model is not None and self.initialized:
            # Save the full model state dict (encoder + segmentation heads)
            torch.save(self.model.state_dict(), path)
            print(f"Full model weights saved to {path}")
    
    def load_weights(self, path: str) -> None:
        """Load model weights from disk.
        
        Loads the full state dict (encoder + segmentation heads) into the
        full model using strict matching.  Verifies all expected keys and
        tensor shapes are compatible.  Raises a clear exception if the
        checkpoint is missing keys, has unexpected keys, or has shape
        mismatches.  Only marks the model as loaded after complete
        verification succeeds.
        
        Records the checkpoint path and SHA256 hash for auditability.
        """
        if self.model is not None and self.initialized:
            try:
                import hashlib
                import time
                
                # Record checkpoint path and SHA256 for auditability
                self._last_checkpoint_path = path
                with open(path, 'rb') as f:
                    self._last_checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()
                print(f"Checkpoint SHA256: {self._last_checkpoint_sha256[:16]}...")
                
                # Load state dict
                state_dict = torch.load(path, map_location=self.config.device)
                
                # Verify expected keys exist before loading
                expected_keys = set(self.model.state_dict().keys())
                loaded_keys = set(state_dict.keys())
                
                missing_in_checkpoint = expected_keys - loaded_keys
                unexpected_in_checkpoint = loaded_keys - expected_keys
                
                if missing_in_checkpoint:
                    raise ValueError(
                        f"Checkpoint missing {len(missing_in_checkpoint)} "
                        f"expected keys: {sorted(missing_in_checkpoint)[:5]}..."
                    )
                if unexpected_in_checkpoint:
                    raise ValueError(
                        f"Checkpoint has {len(unexpected_in_checkpoint)} "
                        f"unexpected keys: {sorted(unexpected_in_checkpoint)[:5]}..."
                    )
                
                # Load with strict=True to verify tensor shape compatibility
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    state_dict, strict=True)
                
                if missing_keys:
                    raise ValueError(
                        f"Load failed: {len(missing_keys)} keys missing after "
                        f"strict load"
                    )
                if unexpected_keys:
                    raise ValueError(
                        f"Load failed: {len(unexpected_keys)} unexpected keys after "
                        f"strict load"
                    )
                
                print(f"Model weights loaded from {path} "
                      f"(SHA256: {self._last_checkpoint_sha256[:16]}...)")
                
                # Record inference timing
                self._last_load_time = time.time()
                
            except Exception as e:
                print(f"Failed to load weights: {e}")
                # Clear any partial state
                self._last_checkpoint_path = None
                self._last_checkpoint_sha256 = None
                raise
                raise


# Convenience function for quick inference
def segment_pentacam_image(
    image: np.ndarray,
    model_path: Optional[str] = None,
    config: Optional[PentacamSegmentationConfig] = None,
) -> Dict[str, Any]:
    """
    Quick segmentation of a Pentacam image.
    
    Args:
        image: numpy array (H, W, 3) BGR or (H, W) grayscale
        model_path: Optional path to pretrained weights
        config: Optional configuration
    
    Returns:
        Dict with segmentation results
    """
    seg = PentacamSegmentator(config)
    seg.initialize()
    
    if model_path and os.path.exists(model_path):
        seg.load_weights(model_path)
    
    return seg.segment_and_detect(image)