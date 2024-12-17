"""
Dialog modules for the Multimedia Assistant application.

This package contains all dialog-based GUI components used in the application,
including dialogs for cropping, downloading, resizing, and upscaling operations.
"""

from .crop_dialog import CropDialog
from .download_dialog import DownloadDialog
from .resize_dialog import ResizeDialog
from .upscale_dialog import UpscaleDialog

__all__ = [
    'CropDialog',
    'DownloadDialog',
    'ResizeDialog',
    'UpscaleDialog'
]
