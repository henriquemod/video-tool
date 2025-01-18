"""
Temporary file management system for the multimedia assistant application.

This module provides a centralized system for creating, tracking, and cleaning up
temporary files used during image and video processing operations. It ensures that
temporary files are properly managed and cleaned up when the application exits.
"""

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Set, Optional

from .config import TEMP_DIR


class TempFileManager:
    """
    Manages temporary files and cleanup for the application.

    This class implements the Singleton pattern to ensure only one instance
    manages temporary files throughout the application lifecycle.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TempFileManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the temporary file manager if not already initialized."""
        if not getattr(self, '_initialized', False):
            self.temp_dir = TEMP_DIR
            self.temp_files: Set[Path] = set()
            self.setup_temp_directory()
            atexit.register(self.cleanup)
            self._initialized = True

    def setup_temp_directory(self):
        """Create temporary directory if it doesn't exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, prefix: str = "", suffix: str = "") -> Path:
        """
        Generate a temporary file path.

        Args:
            prefix (str): Prefix for the temporary file name
            suffix (str): Suffix (extension) for the temporary file

        Returns:
            Path: Path object pointing to the temporary file location
        """
        temp_file = Path(
            tempfile.mktemp(
                prefix=prefix,
                suffix=suffix,
                dir=str(self.temp_dir)
            )
        )
        self.temp_files.add(temp_file)
        return temp_file

    def remove_temp_file(self, file_path: Optional[Path | str]) -> None:
        """
        Remove a specific temporary file.

        Args:
            file_path (Path | str): Path to the temporary file to remove
        """
        if file_path is None:
            return

        path = Path(file_path)
        try:
            path.unlink(missing_ok=True)
            self.temp_files.discard(path)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"Error removing temporary file {path}: {e}")

    def cleanup(self) -> None:
        """Remove all temporary files and directory on application exit."""
        # Remove individual temp files
        for file_path in self.temp_files:
            try:
                file_path.unlink(missing_ok=True)
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"Error cleaning up temporary file {file_path}: {e}")

        # Clear the set of temp files
        self.temp_files.clear()

        # Remove the temp directory and its contents
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError as e:
            print(f"Error removing temporary directory: {e}")


# Global instance
temp_manager = TempFileManager()
