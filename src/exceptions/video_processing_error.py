"""
video_processing_error.py

This module defines custom exception classes for handling errors related to video processing operations.
"""


class VideoProcessingError(Exception):
    """Custom exception for video processing related errors"""

    def __init__(self, message: str, details: str = None):
        """
        Initialize VideoProcessingError.

        Args:
            message (str): Main error message
            details (str, optional): Additional error details/context
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message
