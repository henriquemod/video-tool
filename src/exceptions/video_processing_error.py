"""
video_processing_error.py

This module defines custom exception classes for handling errors related to video processing operations.
"""

from .base_exception import MultimediaAssistantError


class VideoProcessingError(MultimediaAssistantError):
    """Custom exception for video processing related errors"""

    def __init__(self, message: str, details: str = None):
        """
        Initialize VideoProcessingError.

        Args:
            message (str): Main error message
            details (str, optional): Additional error details/context
        """
        super().__init__(message, details)
