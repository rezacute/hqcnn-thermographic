"""
Preprocessing Utilities
========================
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image


class ThermalPreprocessor:
    """Preprocessing utilities for thermal images"""
    
    @staticmethod
    def normalize_thermal(
        image: np.ndarray,
        method: str = 'minmax'
    ) -> np.ndarray:
        """Normalize thermal image values
        
        Args:
            image: Input thermal image
            method: 'minmax', 'zscore', or 'histogram'
            
        Returns:
            Normalized image
        """
        if method == 'minmax':
            min_val = image.min()
            max_val = image.max()
            if max_val - min_val > 0:
                return (image - min_val) / (max_val - min_val)
            return image
        
        elif method == 'zscore':
            mean = image.mean()
            std = image.std()
            if std > 0:
                return (image - mean) / std
            return image - mean
        
        elif method == 'histogram':
            # Histogram equalization for better contrast
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return cv2.equalizeHist(image)
        
        return image
    
    @staticmethod
    def apply_colormap(
        image: np.ndarray,
        colormap: str = 'thermal'
    ) -> np.ndarray:
        """Apply colormap to thermal image for visualization
        
        Args:
            image: Input thermal image (grayscale)
            colormap: Colormap name ('thermal', 'iron', 'rainbow', 'grayscale')
            
        Returns:
            Colorized image
        """
        # Normalize to 0-255
        if image.max() > image.min():
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
        
        colormaps = {
            'thermal': cv2.COLORMAP_JET,
            'iron': cv2.COLORMAP_IRIS,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'grayscale': cv2.COLORMAP_BONE
        }
        
        cmap = colormaps.get(colormap, cv2.COLORMAP_JET)
        return cv2.applyColorMap(image, cmap)
    
    @staticmethod
    def detect_hotspots(
        image: np.ndarray,
        threshold: float = 0.8
    ) -> Tuple[np.ndarray, list]:
        """Detect hotspots in thermal image
        
        Args:
            image: Normalized thermal image (0-1)
            threshold: Threshold for hotspot detection
            
        Returns:
            Binary mask and hotspot coordinates
        """
        # Find hotspots
        hotspot_mask = (image > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            hotspot_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Get hotspot centers
        hotspots = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                hotspots.append((cx, cy))
        
        return hotspot_mask, hotspots
    
    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """Apply CLAHE for contrast enhancement"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe.apply(image)
    
    @staticmethod
    def resize_keep_aspect(
        image: np.ndarray,
        target_size: Tuple[int, int],
        padding: bool = True
    ) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if padding:
            # Create padded image
            padded = np.zeros((target_h, target_w), dtype=image.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return padded
        
        return resized
