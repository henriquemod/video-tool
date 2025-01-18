"""
Module: crop_error.py

Provides custom exception class for image cropping related errors.
"""

from .base_exception import MultimediaAssistantError


class CropError(MultimediaAssistantError):
    """Custom exception for image cropping related errors"""

    def __init__(self, message: str, details: str = None):
        """
        Initialize CropError.

        Args:
            message (str): Main error message
            details (str, optional): Additional error details/context
        """
        super().__init__(message, details)
