"""
AI Upscaling package for the Multimedia Assistant application.

This package provides various AI-powered upscaling implementations and model management.
"""

from dataclasses import dataclass
from typing import List

from .base_upscaler import BaseUpscaler
from .basic_upscaler import BasicUpscaler
from .realesrgan_upscaler import RealESRGANUpscaler
from .swinir_upscaler import SwinIRUpscaler


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
        UpscaleModel("SwinIR-2x", "SwinIR (2x)", 2),
        UpscaleModel("SwinIR-4x", "SwinIR (4x)", 4),
        UpscaleModel("ESRGAN-4x", "ESRGAN (4x)", 4),
        UpscaleModel("Real-ESRGAN-2x-plus", "Real-ESRGAN (2x+)", 2),
        UpscaleModel("Real-ESRGAN-4x-plus", "Real-ESRGAN (4x+)", 4),
        UpscaleModel("Real-ESRNet-4x-plus", "Real-ESRNet (4x+)", 4),
        UpscaleModel("Real-ESRGAN-4x-anime", "Real-ESRGAN Anime (4x)", 4),
    ]


def get_model_names() -> List[str]:
    """Returns a list of model display names."""
    return [model.name for model in get_available_models()]


def create_upscaler(model_id: str) -> BaseUpscaler:
    """
    Factory function to create the appropriate upscaler instance.

    Args:
        model_id (str): ID of the model to use

    Returns:
        BaseUpscaler: An instance of the appropriate upscaler
    """
    if model_id == "no_upscale":
        return None
    elif model_id.startswith('bicubic') or model_id.startswith('lanczos'):
        return BasicUpscaler(model_id)
    elif model_id.startswith('SwinIR'):
        scale = 4 if '4x' in model_id else 2
        return SwinIRUpscaler(model_id, scale)
    elif any(model_id.startswith(prefix) for prefix in ['Real-ESRGAN', 'ESRGAN', 'Real-ESRNet']):
        scale = 4 if '4x' in model_id else 2
        return RealESRGANUpscaler(model_id, scale)
    else:
        return None
