"""
Base exception module for the Multimedia Assistant application.

This module defines the root exception class that all other application-specific
exceptions should inherit from, ensuring consistent error handling throughout
the application.
"""


class MultimediaAssistantError(Exception):
    """Base exception class for all application-specific errors."""

    def __init__(self, message: str, details: str = None):
        """
        Initialize the base exception.

        Args:
            message (str): Main error message
            details (str, optional): Additional error details/context
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        """
        Format the error message with optional details.

        Returns:
            str: Formatted error message
        """
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message
