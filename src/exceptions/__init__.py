"""
Exceptions package for the Multimedia Assistant application.

This package provides custom exceptions for various error scenarios in the application.
"""

from .base_exception import MultimediaAssistantError
from .crop_error import CropError
from .download_error import DownloadError
from .resize_error import ResizeError
from .upscale_error import UpscaleError
from .video_processing_error import VideoProcessingError

__all__ = [
    'MultimediaAssistantError',
    'CropError',
    'DownloadError',
    'ResizeError',
    'UpscaleError',
    'VideoProcessingError'
]
