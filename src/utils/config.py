"""
Configuration module for the Multimedia Assistant application.

This module centralizes all configuration settings including paths, model URLs,
and other application-wide constants. It provides a single source of truth for
configuration parameters used throughout the application.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR.parent / "data"
TEMP_DIR = BASE_DIR.parent / "temp"
OUTPUT_DIR = BASE_DIR.parent / "output"
MODELS_DIR = BASE_DIR.parent / "models"

# Ensure required directories exist
for directory in [DATA_DIR, TEMP_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model URLs for AI upscaling
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

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'}

# Image file extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

# Default application settings
DEFAULT_SETTINGS = {
    'output_directory': str(OUTPUT_DIR),
    'temp_directory': str(TEMP_DIR),
    'models_directory': str(MODELS_DIR),
    'default_video_quality': 'high',
    'default_upscale_model': 'realesrgan_4x_plus',
    'enable_gpu': True,
    'max_threads': os.cpu_count() or 4,
}

# GUI Settings
GUI_SETTINGS = {
    'window_title': 'Multimedia Assistant',
    'default_width': 1200,
    'default_height': 800,
    'min_width': 800,
    'min_height': 600,
}
