import torch
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from dataclasses import dataclass
from typing import List

# Model URLs for SwinIR and Real-ESRGAN
MODEL_URLS = {
    'SwinIR-2x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    'SwinIR-4x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    # X4 model for general images
    'Real-ESRGAN-2x-plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'Real-ESRGAN-4x-plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    # X4 model with MSE loss (over-smooth effects)
    'Real-ESRNet-4x-plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
    # official ESRGAN model
    'ESRGAN-4x': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth',
    # X4 (can also be used for X1, X2, X3) | A tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity
    'ESRGAN-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
    # The following models are discriminators, which are usually used for fine-tuning.
    # RealESRGAN discriminator models
    'Real-ESRGAN-4x-plus-netD': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth',
    'Real-ESRGAN-2x-plus-netD': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x2plus_netD.pth',
    # For Anime Images / Illustrations
    # For Anime Images / Illustrations
    'Real-ESRGAN-4x-anime': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth',
    # Discriminator model for anime upscaling
    'Real-ESRGAN-4x-anime-netD': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B_netD.pth',
}

# At the top of the file, add:


@dataclass
class UpscaleModel:
    """Represents an upscaling model configuration"""
    id: str
    name: str
    scale: int


def get_available_models() -> List[UpscaleModel]:
    """Returns a list of available upscaling model configurations."""
    return [
        UpscaleModel("no_upscale", "No Upscaling", 1),
        UpscaleModel("bicubic_2x", "Bicubic (2x)", 2),
        UpscaleModel("bicubic_3x", "Bicubic (3x)", 3),
        UpscaleModel("bicubic_4x", "Bicubic (4x)", 4),
        UpscaleModel("lanczos_2x", "Lanczos (2x)", 2),
        UpscaleModel("lanczos_3x", "Lanczos (3x)", 3),
        UpscaleModel("lanczos_4x", "Lanczos (4x)", 4),
        UpscaleModel("swinir_2x", "SwinIR (2x)", 2),
        UpscaleModel("swinir_4x", "SwinIR (4x)", 4),
        # X4 model for general images
        UpscaleModel("realesrgan_2x_plus", "Real-ESRGAN (2x+)", 2),
        UpscaleModel("realesrgan_4x_plus", "Real-ESRGAN (4x+)", 4),
        # X4 model with MSE loss (over-smooth effects)
        UpscaleModel("realesrnet_4x_plus", "Real-ESRNet (4x+)", 4),
        # official ESRGAN model
        UpscaleModel("esrgan_4x", "ESRGAN (4x)", 4),
        # X4 (can also be used for X1, X2, X3) | A tiny small model (consume much fewer GPU memory and time); not too strong deblur and denoise capacity
        UpscaleModel("esrgan_general_4x", "ESRGAN General (4x)", 4),
        # The following models are discriminators, which are usually used for fine-tuning.
        # RealESRGAN discriminator models
        UpscaleModel("realesrgan_4x_plus_netd", "Real-ESRGAN (4x+ netD)", 4),
        UpscaleModel("realesrgan_2x_plus_netd", "Real-ESRGAN (2x+ netD)", 2),
        # For Anime Images / Illustrations
        UpscaleModel("realesrgan_anime_4x", "Real-ESRGAN Anime (4x)", 4),
        # Discriminator model for anime upscaling
        UpscaleModel("realesrgan_anime_4x_netd",
                     "Real-ESRGAN Anime (4x netD)", 4)
    ]


def get_model_by_id(model_id: str) -> UpscaleModel:
    """Returns a model configuration by its ID."""
    models = {model.id: model for model in get_available_models()}
    return models.get(model_id)


def get_model_names() -> List[str]:
    """Returns a list of model display names."""
    return [model.name for model in get_available_models()]


"""
AI Upscaling Module - Advanced Image Enhancement System

This module implements a sophisticated image upscaling system that combines multiple
state-of-the-art AI models with classical upscaling methods. It provides a flexible
and efficient interface for high-quality image enhancement.

Key Features:
- Multiple AI model support (SwinIR, Real-ESRGAN)
- Classical upscaling methods (Bicubic, Lanczos)
- Automatic device selection (CUDA, MPS, CPU)
- Tile-based processing for large images
- Memory-efficient implementation
- Progress tracking support

Models:
1. SwinIR:
   - Lightweight SR model
   - Efficient transformer-based architecture
   - Support for 2x and 4x upscaling
   - Memory-efficient processing
   - Tile-based handling for large images

2. Real-ESRGAN:
   - Enhanced SRGAN architecture
   - Robust to real-world degradation
   - 2x and 4x upscaling support
   - Improved detail preservation
   - Better artifact handling

Technical Implementation:
1. Device Management:
   - Automatic GPU detection
   - CUDA support for NVIDIA GPUs
   - MPS support for Apple Silicon
   - Graceful fallback to CPU
   - Dynamic memory optimization

2. Processing Pipeline:
   - Image preprocessing
   - Padding handling
   - Tiled processing for large images
   - Color space conversion
   - Result post-processing

3. Memory Management:
   - Efficient tensor operations
   - Automatic garbage collection
   - Batch size optimization
   - Resource cleanup
   - Memory usage monitoring

Functions:
    get_device():
        Determines the best available processing device
        Returns:
            str: 'cuda', 'mps', or 'cpu'

    upscale_image(img, method, scale, device=None):
        Main interface for image upscaling
        Args:
            img: numpy array (BGR format)
            method: upscaling method name
            scale: upscaling factor (2 or 4)
            device: processing device (optional)
        Returns:
            numpy array: upscaled image

    upscale_realesrgan(img, scale, device):
        Real-ESRGAN specific implementation
        Args:
            img: input image
            scale: upscaling factor
            device: processing device
        Returns:
            numpy array: upscaled image

    upscale_swinir(img, scale, device):
        SwinIR specific implementation
        Args:
            img: input image
            scale: upscaling factor
            device: processing device
        Returns:
            numpy array: upscaled image

Model Configuration:
- SwinIR:
    * Architecture: Swin Transformer
    * Parameters: 60 embedding dim, 6 depth
    * Window size: 8
    * Upsampler: pixelshuffledirect

- Real-ESRGAN:
    * Architecture: RRDB
    * Features: 64 channels
    * Blocks: 23
    * Growth channels: 32

Dependencies:
- torch: Deep learning framework
- basicsr: Base SR toolkit
- realesrgan: Real-ESRGAN implementation
- numpy: Numerical operations
- opencv-python: Image processing

Performance Considerations:
- GPU memory management
- Tile size optimization
- Batch processing efficiency
- Color space conversions
- Memory cleanup

Error Handling:
- Device availability checks
- Model loading failures
- Memory exhaustion
- Processing errors
- Invalid inputs

@see @Project Structure#ai_upscaling.py
@see @Project#AI Enhancement
@see @Project#Image Processing
"""


class AIUpscaler:
    """Centralized class for handling all AI upscaling operations"""

    def __init__(self):
        self.device = self.get_device()
        self._is_cancelled = False

    def get_device(self):
        """Get the best available device for AI processing."""
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def upscale(self, img, model_id: str, progress_callback=None):
        """
        Unified method for upscaling images using various methods.

        Args:
            img: numpy array of the image (BGR format)
            model_id: str, ID of the upscaling model to use
            progress_callback: Optional callback function for progress updates

        Returns:
            numpy array of the upscaled image
        """
        try:
            if progress_callback:
                progress_callback(10)

            if self._is_cancelled:
                return None

            model = get_model_by_id(model_id)
            if not model:
                raise ValueError(f"Invalid model ID: {model_id}")

            # Store current method for use in specific upscalers
            self.current_method = model.name
            scale = model.scale

            if "realesrgan" in model_id or "esrgan" in model_id or "esrnet" in model_id:
                return self.upscale_realesrgan(img, scale, progress_callback)
            elif "swinir" in model_id:
                return self.upscale_swinir(img, scale, progress_callback)
            elif "bicubic" in model_id:
                return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            elif "lanczos" in model_id:
                return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            elif model_id == "no_upscale":
                return img
            else:
                raise ValueError(f"Unsupported upscaling method: {model_id}")

        except Exception as e:
            raise Exception(f"Upscaling failed: {str(e)}")

    def cancel(self):
        """Cancel ongoing upscaling operation"""
        self._is_cancelled = True

    def reset(self):
        """Reset cancellation flag"""
        self._is_cancelled = False

    def upscale_realesrgan(self, img, scale, progress_callback):
        """Upscale image using Real-ESRGAN and its variants."""
        try:
            # Initialize model based on selected scale and type
            model_key = None
            num_feat = 64
            num_block = 23

            # Determine which model to use based on scale and current method
            if "anime" in self.current_method.lower():
                model_key = 'Real-ESRGAN-4x-anime'
                num_feat = 64
                num_block = 6  # Anime model uses 6 blocks instead of 23
            elif "esrnet" in self.current_method.lower():
                model_key = 'Real-ESRNet-4x-plus'
            elif "general" in self.current_method.lower():
                model_key = 'ESRGAN-general-x4v3'
                num_feat = 48  # Adjusted for general model
                num_block = 16
            else:
                model_key = f'Real-ESRGAN-{scale}x-plus'

            # Initialize model
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=num_feat,
                num_block=num_block,
                num_grow_ch=32,
                scale=scale
            )

            # Get model path
            model_url = MODEL_URLS[model_key]
            model_path = load_file_from_url(model_url, model_dir='models')

            # Initialize upsampler with appropriate settings
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=512,
                tile_pad=32,
                pre_pad=0,
                half=True if self.device == 'cuda' else False,
                device=self.device
            )

            # Process image
            if progress_callback:
                progress_callback(50)  # Update progress

            output, _ = upsampler.enhance(img, outscale=scale)

            if progress_callback:
                progress_callback(90)  # Update progress

            return output

        except Exception as e:
            raise Exception(f"Real-ESRGAN processing failed: {str(e)}")

    def upscale_swinir(self, img, scale, progress_callback):
        """Upscale image using SwinIR."""
        try:
            from basicsr.utils.registry import ARCH_REGISTRY

            # Prepare the image
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Handle padding
            h, w = img.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            # Load model
            model_url = MODEL_URLS[f'SwinIR-{scale}x']
            model_path = load_file_from_url(model_url, model_dir='models')

            # Initialize model
            model = ARCH_REGISTRY.get('SwinIR')(
                upscale=scale,
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

            # Load weights
            pretrained_model = torch.load(model_path, map_location=self.device)
            if 'params_ema' in pretrained_model:
                pretrained_model = pretrained_model['params_ema']
            elif 'params' in pretrained_model:
                pretrained_model = pretrained_model['params']

            model.load_state_dict(pretrained_model, strict=True)
            model.eval()
            model.to(self.device)

            # Process image
            img_tensor = torch.from_numpy(img).float().permute(
                2, 0, 1).unsqueeze(0) / 255.
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                output = model(img_tensor)

            # Final processing
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            output = output.transpose(1, 2, 0)

            if pad_h > 0 or pad_w > 0:
                output = output[:h*scale, :w*scale, :]

            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return output

        except Exception as e:
            raise Exception(f"SwinIR processing failed: {str(e)}")
