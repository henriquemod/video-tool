"""
Basic upscaler implementation for non-AI methods.

This module provides implementations for basic upscaling methods like bicubic
and lanczos interpolation using OpenCV.
"""

from typing import Optional, Callable
import cv2
import numpy as np

from .base_upscaler import BaseUpscaler


class BasicUpscaler(BaseUpscaler):
    """Basic upscaler implementation for bicubic and lanczos methods."""

    def __init__(self, model_id: str):
        """
        Initialize the basic upscaler.

        Args:
            model_id (str): Identifier for the upscaling method (e.g., 'bicubic_2x')
        """
        super().__init__()
        self.model_id = model_id
        self.scale = int(model_id.split('_')[1][0])  # Extract scale from model_id (e.g., '2' from '2x')
        self.method = cv2.INTER_CUBIC if 'bicubic' in model_id else cv2.INTER_LANCZOS4

    def load_model(self) -> None:
        """
        No model loading needed for basic upscaling methods.
        """
        pass

    def upscale(self, img: np.ndarray, progress_callback: Optional[Callable[[int], None]] = None) -> np.ndarray:
        """
        Upscale the input image using basic interpolation methods.

        Args:
            img (np.ndarray): Input image as a numpy array (BGR format)
            progress_callback (Optional[Callable[[int], None]]): Callback for progress updates

        Returns:
            np.ndarray: Upscaled image
        """
        if progress_callback:
            progress_callback(0)

        h, w = img.shape[:2]
        upscaled = cv2.resize(
            img,
            (w * self.scale, h * self.scale),
            interpolation=self.method
        )

        if progress_callback:
            progress_callback(100)

        return upscaled 