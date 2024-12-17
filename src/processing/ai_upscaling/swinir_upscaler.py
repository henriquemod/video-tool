"""
SwinIR upscaler implementation.

This module provides the implementation of the SwinIR model for image upscaling,
offering high-quality results with efficient transformer-based architecture.
"""

import numpy as np
import torch
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY

from ...exceptions.upscale_error import UpscaleError
from ...utils.config import MODEL_URLS
from .base_upscaler import BaseUpscaler


class SwinIRUpscaler(BaseUpscaler):
    """SwinIR upscaler implementation."""

    def __init__(self, scale: int = 4):
        """
        Initialize the SwinIR upscaler.

        Args:
            scale (int): Upscaling factor (default: 4)
        """
        super().__init__()
        self.scale = scale
        self.model_name = f'SwinIR-{scale}x'

    def load_model(self) -> None:
        """
        Load the SwinIR model.

        Raises:
            UpscaleError: If model loading fails
        """
        try:
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

            # Initialize SwinIR model
            self.model = ARCH_REGISTRY.get('SwinIR')(
                upscale=self.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.,
                depths=[6, 6, 6, 6],
                embed_dim=60,
                num_heads=[6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffledirect',
                resi_connection='1conv'
            )

            # Load pretrained weights
            pretrained_model = torch.load(
                self.model_path,
                map_location=self.device
            )

            if 'params_ema' in pretrained_model:
                pretrained_model = pretrained_model['params_ema']
            elif 'params' in pretrained_model:
                pretrained_model = pretrained_model['params']

            self.model.load_state_dict(pretrained_model, strict=True)
            self.model.eval()
            self.model.to(self.device)

        except Exception as e:
            raise UpscaleError(
                f"Failed to load SwinIR model: {str(e)}",
                model_id=self.model_name,
                error_type="ModelLoading"
            ) from e

    def upscale(self, img: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Upscale an image using SwinIR.

        Args:
            img (np.ndarray): Input image
            progress_callback (Optional[Callable[[int], None]]): Progress callback

        Returns:
            np.ndarray: Upscaled image

        Raises:
            UpscaleError: If upscaling fails
        """
        try:
            if self.model is None:
                self.load_model()

            if progress_callback:
                progress_callback(10)

            if self._is_cancelled:
                raise UpscaleError(
                    "Upscaling cancelled by user",
                    model_id=self.model_name,
                    error_type="Cancelled"
                )

            # Convert to RGB if necessary
            if img.shape[2] == 3:
                img = img[..., ::-1]  # BGR to RGB

            # Pad image if necessary
            h, w = img.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode='reflect'
                )

            if progress_callback:
                progress_callback(30)

            # Prepare input tensor
            img_tensor = torch.from_numpy(img).float().permute(
                2, 0, 1).unsqueeze(0) / 255.
            img_tensor = img_tensor.to(self.device)

            # Process image
            with torch.no_grad():
                output = self.model(img_tensor)

            if progress_callback:
                progress_callback(70)

            # Post-process output
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            output = output.transpose(1, 2, 0)

            # Remove padding if necessary
            if pad_h > 0 or pad_w > 0:
                output = output[:h*self.scale, :w*self.scale, :]

            # Convert back to BGR
            output = output[..., ::-1]  # RGB to BGR

            if progress_callback:
                progress_callback(100)

            return output

        except Exception as e:
            if not isinstance(e, UpscaleError):
                e = UpscaleError(
                    f"SwinIR processing failed: {str(e)}",
                    model_id=self.model_name,
                    error_type="Processing"
                )
            raise e
