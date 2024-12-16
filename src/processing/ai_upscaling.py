"""
AI Upscaling Module - Provides advanced image upscaling capabilities using various AI models
and traditional methods. Supports multiple architectures including Real-ESRGAN and SwinIR.
"""
from dataclasses import dataclass
from typing import List

import cv2  # pylint: disable=no-member
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from realesrgan import RealESRGANer

class UpscalingError(Exception):
    """Base exception for upscaling errors"""


class ModelError(UpscalingError):
    """Exception raised for errors related to model loading or processing"""


class ProcessingError(UpscalingError):
    """Exception raised for errors during image processing"""

# Model URLs for SwinIR and Real-ESRGAN
MODEL_URLS = {
    'SwinIR-2x': (
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
        '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth'
    ),
    'SwinIR-4x': (
        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
        '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'
    ),
    'Real-ESRGAN-2x-plus': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/'
        'RealESRGAN_x2plus.pth'
    ),
    'Real-ESRGAN-4x-plus': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/'
        'RealESRGAN_x4plus.pth'
    ),
    'Real-ESRNet-4x-plus': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/'
        'RealESRNet_x4plus.pth'
    ),
    'ESRGAN-4x': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/'
        'ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
    ),
    'ESRGAN-general-x4v3': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'
        'realesr-general-x4v3.pth'
    ),
    'Real-ESRGAN-4x-plus-netD': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/'
        'RealESRGAN_x4plus_netD.pth'
    ),
    'Real-ESRGAN-2x-plus-netD': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/'
        'RealESRGAN_x2plus_netD.pth'
    ),
    'Real-ESRGAN-4x-anime': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/'
        'RealESRGAN_x4plus_anime_6B.pth'
    ),
    'Real-ESRGAN-4x-anime-netD': (
        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/'
        'RealESRGAN_x4plus_anime_6B_netD.pth'
    ),
}


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
        UpscaleModel("realesrgan_2x_plus", "Real-ESRGAN (2x+)", 2),
        UpscaleModel("realesrgan_4x_plus", "Real-ESRGAN (4x+)", 4),
        UpscaleModel("realesrnet_4x_plus", "Real-ESRNet (4x+)", 4),
        UpscaleModel("esrgan_4x", "ESRGAN (4x)", 4),
        UpscaleModel("esrgan_general_4x", "ESRGAN General (4x)", 4),
        UpscaleModel("realesrgan_4x_plus_netd", "Real-ESRGAN (4x+ netD)", 4),
        UpscaleModel("realesrgan_2x_plus_netd", "Real-ESRGAN (2x+ netD)", 2),
        UpscaleModel("realesrgan_anime_4x", "Real-ESRGAN Anime (4x)", 4),
        UpscaleModel("realesrgan_anime_4x_netd", "Real-ESRGAN Anime (4x netD)", 4)
    ]


def get_model_by_id(model_id: str) -> UpscaleModel:
    """Returns a model configuration by its ID."""
    models = {model.id: model for model in get_available_models()}
    return models.get(model_id)


def get_model_names() -> List[str]:
    """Returns a list of model display names."""
    return [model.name for model in get_available_models()]


class AIUpscaler:
    """Centralized class for handling all AI upscaling operations"""

    def __init__(self):
        """Initialize the AIUpscaler with device detection and state variables."""
        self.device = self.get_device()
        self._is_cancelled = False
        self.current_method = None

    def get_device(self):
        """Get the best available device for AI processing."""
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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

            self.current_method = model.name
            scale = model.scale

            if "realesrgan" in model_id or "esrgan" in model_id or "esrnet" in model_id:
                return self.upscale_realesrgan(img, scale, progress_callback)
            if "swinir" in model_id:
                return self.upscale_swinir(img, scale, progress_callback)
            if "bicubic" in model_id:
                return cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC)
            if "lanczos" in model_id:
                return cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LANCZOS4)
            if model_id == "no_upscale":
                return img

            raise ValueError(f"Unsupported upscaling method: {model_id}")

        except ValueError as e:
            raise ModelError(f"Invalid model configuration: {str(e)}") from e
        except Exception as e:
            raise ProcessingError(f"Upscaling failed: {str(e)}") from e

    def cancel(self):
        """Cancel ongoing upscaling operation"""
        self._is_cancelled = True

    def reset(self):
        """Reset cancellation flag"""
        self._is_cancelled = False

    def upscale_realesrgan(self, img, scale, progress_callback):
        """Upscale image using Real-ESRGAN and its variants."""
        try:
            model_key = None
            num_feat = 64
            num_block = 23

            if "anime" in self.current_method.lower():
                model_key = 'Real-ESRGAN-4x-anime'
                num_feat = 64
                num_block = 6
            elif "esrnet" in self.current_method.lower():
                model_key = 'Real-ESRNet-4x-plus'
            elif "general" in self.current_method.lower():
                model_key = 'ESRGAN-general-x4v3'
                num_feat = 48
                num_block = 16
            else:
                model_key = f'Real-ESRGAN-{scale}x-plus'

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=num_feat,
                num_block=num_block,
                num_grow_ch=32,
                scale=scale
            )

            model_url = MODEL_URLS[model_key]
            model_path = load_file_from_url(model_url, model_dir='models')

            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=512,
                tile_pad=32,
                pre_pad=0,
                half=self.device == 'cuda',
                device=self.device
            )

            if progress_callback:
                progress_callback(50)

            output, _ = upsampler.enhance(img, outscale=scale)

            if progress_callback:
                progress_callback(90)

            return output

        except Exception as e:
            raise ProcessingError(f"Real-ESRGAN processing failed: {str(e)}") from e

    def upscale_swinir(self, img, scale, progress_callback):  # pylint: disable=unused-argument
        """Upscale image using SwinIR."""
        try:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            if pad_h > 0 or pad_w > 0:
                img = np.pad(
                    img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            model_url = MODEL_URLS[f'SwinIR-{scale}x']
            model_path = load_file_from_url(model_url, model_dir='models')

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

            pretrained_model = torch.load(model_path, map_location=self.device)
            if 'params_ema' in pretrained_model:
                pretrained_model = pretrained_model['params_ema']
            elif 'params' in pretrained_model:
                pretrained_model = pretrained_model['params']

            model.load_state_dict(pretrained_model, strict=True)
            model.eval()
            model.to(self.device)

            img_tensor = torch.from_numpy(img).float().permute(
                2, 0, 1).unsqueeze(0) / 255.
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                output = model(img_tensor)

            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 255.0).round().astype(np.uint8)
            output = output.transpose(1, 2, 0)

            if pad_h > 0 or pad_w > 0:
                output = output[:h*scale, :w*scale, :]

            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            return output

        except Exception as e:
            raise ProcessingError(f"SwinIR processing failed: {str(e)}") from e
