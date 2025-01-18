"""
Module: download_error.py

Provides custom exception class for video download related errors.
"""

from .base_exception import MultimediaAssistantError


class DownloadError(MultimediaAssistantError):
    """Custom exception for video download related errors"""

    def __init__(self, message: str, url: str = None, details: str = None):
        """
        Initialize DownloadError.

        Args:
            message (str): Main error message
            url (str, optional): URL that failed to download
            details (str, optional): Additional error details/context
        """
        self.url = url
        formatted_details = self._format_details(details)
        super().__init__(message, formatted_details)

    def _format_details(self, details: str) -> str:
        """Format the error details string."""
        parts = []
        if self.url:
            parts.append(f"URL: {self.url}")
        if details:
            parts.append(details)
        return "\n".join(parts) if parts else None
