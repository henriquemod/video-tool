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
from ...utils.config import MODEL_URLS, MODELS_DIR
from .base_upscaler import BaseUpscaler


class SwinIRUpscaler(BaseUpscaler):
    """SwinIR upscaler implementation."""

    def __init__(self, model_id: str, scale: int = 4):
        """
        Initialize the SwinIR upscaler.

        Args:
            model_id (str): Identifier for the model
            scale (int): Upscaling factor (default: 4)
        """
        super().__init__()
        self.scale = scale
        self.model_id = model_id
        self.model = None
        self.tile_size = 512  # Default tile size
        self.tile_overlap = 32  # Overlap between tiles to avoid boundary artifacts

    def load_model(self) -> None:
        """
        Load the SwinIR model.

        Raises:
            UpscaleError: If model loading fails
        """
        try:
            # Clear CUDA cache before loading model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get model URL and download if necessary
            model_url = MODEL_URLS.get(self.model_id)
            if not model_url:
                raise UpscaleError(
                    f"Model URL not found for {self.model_id}",
                    model_id=self.model_id
                )

            self.model_path = load_file_from_url(
                model_url,
                model_dir=str(MODELS_DIR)
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

            # Clear memory after loading
            del pretrained_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            raise UpscaleError(
                f"Failed to load SwinIR model: {str(e)}",
                model_id=self.model_id,
                error_type="ModelLoading"
            ) from e

    def _process_tile(self, tile: np.ndarray) -> np.ndarray:
        """Process a single tile of the image."""
        # Convert to tensor
        tile_tensor = torch.from_numpy(tile).float().permute(2, 0, 1).unsqueeze(0) / 255.
        tile_tensor = tile_tensor.to(self.device)

        # Process
        with torch.no_grad():
            output = self.model(tile_tensor)

        # Convert back to numpy
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output = output.transpose(1, 2, 0)

        # Clear GPU memory
        del tile_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output

    def upscale(self, img: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Upscale an image using SwinIR with tiling for memory efficiency.

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
                    model_id=self.model_id,
                    error_type="Cancelled"
                )

            # Convert to RGB if necessary
            if img.shape[2] == 3:
                img = img[..., ::-1]  # BGR to RGB
                
            # Ensure array is contiguous in memory
            img = np.ascontiguousarray(img)

            # Get dimensions
            h, w = img.shape[:2]

            # Initialize output array
            output = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)

            # Calculate number of tiles
            num_tiles_x = int(np.ceil(w / (self.tile_size - self.tile_overlap * 2)))
            num_tiles_y = int(np.ceil(h / (self.tile_size - self.tile_overlap * 2)))
            total_tiles = num_tiles_x * num_tiles_y

            # Process tiles
            tile_count = 0
            for y in range(num_tiles_y):
                for x in range(num_tiles_x):
                    # Calculate tile coordinates
                    x_start = max(0, x * (self.tile_size - self.tile_overlap * 2))
                    y_start = max(0, y * (self.tile_size - self.tile_overlap * 2))
                    x_end = min(w, x_start + self.tile_size)
                    y_end = min(h, y_start + self.tile_size)

                    # Extract tile
                    tile = img[y_start:y_end, x_start:x_end]

                    # Pad tile if necessary
                    pad_h = (8 - tile.shape[0] % 8) % 8
                    pad_w = (8 - tile.shape[1] % 8) % 8
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(
                            tile,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='reflect'
                        )

                    # Process tile
                    processed_tile = self._process_tile(tile)

                    # Remove padding if necessary
                    if pad_h > 0 or pad_w > 0:
                        processed_tile = processed_tile[:(y_end - y_start) * self.scale,
                                                    :(x_end - x_start) * self.scale]

                    # Place tile in output array
                    output[y_start * self.scale:y_end * self.scale,
                          x_start * self.scale:x_end * self.scale] = processed_tile

                    # Update progress
                    tile_count += 1
                    if progress_callback:
                        progress = int(30 + (tile_count / total_tiles) * 60)
                        progress_callback(progress)

                    if self._is_cancelled:
                        raise UpscaleError(
                            "Upscaling cancelled by user",
                            model_id=self.model_id,
                            error_type="Cancelled"
                        )

            # Convert back to BGR
            output = output[..., ::-1]  # RGB to BGR

            if progress_callback:
                progress_callback(100)

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output

        except Exception as e:
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not isinstance(e, UpscaleError):
                e = UpscaleError(
                    f"SwinIR processing failed: {str(e)}",
                    model_id=self.model_id,
                    error_type="Processing"
                )
            raise e
