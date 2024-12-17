"""
Real-ESRGAN upscaler implementation.

This module provides the implementation of Real-ESRGAN and its variants for
high-quality image upscaling. It supports multiple models including Real-ESRGAN,
Real-ESRNet, and standard ESRGAN.
"""

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

from ...exceptions.upscale_error import UpscaleError
from ...utils.config import MODEL_URLS
from .base_upscaler import BaseUpscaler


class RealESRGANUpscaler(BaseUpscaler):
    """Real-ESRGAN upscaler implementation."""

    def __init__(self, model_name: str, scale: int = 4):
        """
        Initialize the Real-ESRGAN upscaler.

        Args:
            model_name (str): Name of the model to use
            scale (int): Upscaling factor (default: 4)
        """
        super().__init__()
        self.model_name = model_name
        self.scale = scale
        self.num_feat = 64
        self.num_block = 23
        self.upsampler = None

        # Configure model architecture based on model name
        if "anime" in model_name.lower():
            self.num_feat = 64
            self.num_block = 6
        elif "general" in model_name.lower():
            self.num_feat = 48
            self.num_block = 16

    def load_model(self) -> None:
        """
        Load the Real-ESRGAN model.

        Raises:
            UpscaleError: If model loading fails
        """
        try:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=self.num_feat,
                num_block=self.num_block,
                num_grow_ch=32,
                scale=self.scale
            )

            # Get model URL and download if necessary
            model_url = MODEL_URLS.get(self.model_name)
            if not model_url:
                raise UpscaleError(
                    f"Model URL not found for {self.model_name}",
                    model_id=self.model_name
                )

            self.model_path = load_file_from_url(
                model_url,
                model_dir='models'
            )

            # Initialize RealESRGANer
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=self.model_path,
                model=model,
                tile=512,
                tile_pad=32,
                pre_pad=0,
                half=self.device == 'cuda',
                device=self.device
            )

        except Exception as e:
            raise UpscaleError(
                f"Failed to load Real-ESRGAN model: {str(e)}",
                model_id=self.model_name,
                error_type="ModelLoading"
            ) from e

    def upscale(self, img: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Upscale an image using Real-ESRGAN.

        Args:
            img (np.ndarray): Input image
            progress_callback (Optional[Callable[[int], None]]): Progress callback

        Returns:
            np.ndarray: Upscaled image

        Raises:
            UpscaleError: If upscaling fails
        """
        try:
            if self.upsampler is None:
                self.load_model()

            if progress_callback:
                progress_callback(10)

            if self._is_cancelled:
                raise UpscaleError(
                    "Upscaling cancelled by user",
                    model_id=self.model_name,
                    error_type="Cancelled"
                )

            # Process the image
            output, _ = self.upsampler.enhance(
                img,
                outscale=self.scale
            )

            if progress_callback:
                progress_callback(100)

            if output is None:
                raise UpscaleError(
                    "Upscaling failed: No output received",
                    model_id=self.model_name,
                    error_type="Processing"
                )

            return output

        except Exception as e:
            if not isinstance(e, UpscaleError):
                e = UpscaleError(
                    f"Real-ESRGAN processing failed: {str(e)}",
                    model_id=self.model_name,
                    error_type="Processing"
                )
            raise e
