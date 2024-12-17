"""
Module: resize_error.py

Provides custom exception class for image resizing related errors.
"""

from .base_exception import MultimediaAssistantError


class ResizeError(MultimediaAssistantError):
    """Custom exception for image resizing related errors"""

    def __init__(self, message: str, details: str = None):
        """
        Initialize ResizeError.

        Args:
            message (str): Main error message
            details (str, optional): Additional error details/context
        """
        super().__init__(message, details)
