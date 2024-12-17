"""
AI Upscaling package for the Multimedia Assistant application.

This package provides various AI-powered upscaling implementations and model management.
"""

from dataclasses import dataclass
from typing import List

from .base_upscaler import BaseUpscaler
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
        UpscaleModel("swinir_2x", "SwinIR (2x)", 2),
        UpscaleModel("swinir_4x", "SwinIR (4x)", 4),
        UpscaleModel("realesrgan_2x_plus", "Real-ESRGAN (2x+)", 2),
        UpscaleModel("realesrgan_4x_plus", "Real-ESRGAN (4x+)", 4),
        UpscaleModel("realesrnet_4x_plus", "Real-ESRNet (4x+)", 4),
        UpscaleModel("esrgan_4x", "ESRGAN (4x)", 4),
        UpscaleModel("esrgan_general_4x", "ESRGAN General (4x)", 4),
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
    if model_id.startswith('realesrgan') or model_id.startswith('esrgan'):
        return RealESRGANUpscaler(model_id)
    elif model_id.startswith('swinir'):
        scale = 4 if '4x' in model_id else 2
        return SwinIRUpscaler(scale)
    else:
        # For basic methods like bicubic and lanczos, we can use a simple upscaler
        # or return None to indicate no AI upscaling needed
        return None
