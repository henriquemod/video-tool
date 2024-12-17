"""
Module: upscale_error.py

Provides custom exception classes for AI upscaling related errors.
"""

from .base_exception import MultimediaAssistantError


class UpscaleError(MultimediaAssistantError):
    """Custom exception for AI upscaling related errors"""

    def __init__(self, message: str, model_id: str = None, error_type: str = None):
        """
        Initialize UpscaleError.

        Args:
            message (str): Main error message
            model_id (str, optional): ID of the model that failed
            error_type (str, optional): Type of upscaling error (e.g. "CUDA", "Memory", "Model")
        """
        self.model_id = model_id
        self.error_type = error_type
        details = self._format_details()
        super().__init__(message, details)

    def _format_details(self) -> str:
        """
        Format the error details string.

        Returns:
            str: Formatted details string or None if no details available
        """
        details = []
        if self.model_id:
            details.append(f"Model: {self.model_id}")
        if self.error_type:
            details.append(f"Error Type: {self.error_type}")

        return "\n".join(details) if details else None
