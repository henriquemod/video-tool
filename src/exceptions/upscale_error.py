"""
Module: upscale_error.py

Provides custom exception classes for AI upscaling related errors.

Classes:
    UpscaleError: Custom exception for AI upscaling related issues, including details about the model and error type.
"""


class UpscaleError(Exception):
    """Custom exception for AI upscaling related errors"""

    def __init__(self, message: str, model_id: str = None, error_type: str = None):
        """
        Initialize UpscaleError.

        Args:
            message (str): Main error message
            model_id (str, optional): ID of the model that failed
            error_type (str, optional): Type of upscaling error (e.g. "CUDA", "Memory", "Model")
        """
        self.message = message
        self.model_id = model_id
        self.error_type = error_type
        super().__init__(self.message)

    def __str__(self):
        error_msg = self.message
        if self.model_id:
            error_msg += f"\nModel: {self.model_id}"
        if self.error_type:
            error_msg += f"\nError Type: {self.error_type}"
        return error_msg
