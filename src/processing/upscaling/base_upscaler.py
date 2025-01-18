"""
Base upscaler module that defines the interface for all AI upscaling implementations.

This module provides the abstract base class that all upscaler implementations must
inherit from, ensuring a consistent interface across different upscaling methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import torch


class BaseUpscaler(ABC):
    """Abstract base class for all upscaler implementations."""

    def __init__(self):
        """Initialize the base upscaler with common attributes."""
        self.device = self._get_device()
        self._is_cancelled = False
        self.model = None
        self.model_path = None

    def _get_device(self) -> str:
        """
        Determine the best available device for processing.

        Returns:
            str: Device identifier ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the AI model into memory.

        This method must be implemented by each upscaler class to handle
        model-specific loading requirements.

        Raises:
            UpscaleError: If model loading fails
        """
        pass

    @abstractmethod
    def upscale(self,
                img: np.ndarray,
                progress_callback: Optional[Callable[[int], None]] = None
                ) -> np.ndarray:
        """
        Upscale the input image.

        Args:
            img (np.ndarray): Input image as a numpy array (BGR format)
            progress_callback (Optional[Callable[[int], None]]): Callback for progress updates

        Returns:
            np.ndarray: Upscaled image

        Raises:
            UpscaleError: If upscaling fails
        """
        pass

    def cancel(self) -> None:
        """Cancel the current upscaling operation."""
        self._is_cancelled = True

    def reset(self) -> None:
        """Reset the cancellation flag."""
        self._is_cancelled = False

    def cleanup(self) -> None:
        """
        Clean up resources used by the upscaler.

        This method should be called when the upscaler is no longer needed.
        """
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Ensure resources are cleaned up when the upscaler is deleted."""
        self.cleanup()
