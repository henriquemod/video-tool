"""
Temporary file management system for the multimedia assistant application.

This module provides a centralized system for creating, tracking, and cleaning up
temporary files used during image and video processing operations. It ensures that
temporary files are properly managed and cleaned up when the application exits.

The module includes:
    - TempFileManager: Main class for temporary file operations
    - temp_manager: Global instance of TempFileManager for application-wide use

Features:
    - Automatic cleanup of temporary files on application exit
    - Dedicated temporary directory for the application
    - Safe file removal with error handling
    - Path generation for temporary files with custom prefixes and suffixes

Usage:
    from utils.temp_file_manager import temp_manager
    
    # Generate a temporary file path
    temp_path = temp_manager.get_temp_path(prefix="video_", suffix=".mp4")
    
    # Remove a specific temporary file
    temp_manager.remove_temp_file(temp_path)
"""
import tempfile
import atexit
import shutil
from pathlib import Path


class TempFileManager:
    """Manages temporary files and cleanup for the application."""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "multimedia_assistant"
        self.temp_files = set()
        self.setup_temp_directory()
        atexit.register(self.cleanup)

    def setup_temp_directory(self):
        """Create temporary directory if it doesn't exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, prefix="", suffix=""):
        """Generate a temporary file path."""
        temp_file = Path(
            tempfile.mktemp(prefix=prefix, suffix=suffix,
                            dir=str(self.temp_dir))
        )
        self.temp_files.add(temp_file)
        return temp_file

    def remove_temp_file(self, file_path):
        """Remove a specific temporary file."""
        try:
            Path(file_path).unlink(missing_ok=True)
            self.temp_files.discard(Path(file_path))
        except FileNotFoundError:
            print(f"Temporary file {file_path} not found.")
        except PermissionError:
            print(f"Permission denied when removing {file_path}.")
        except OSError as e:
            print(f"Error removing temporary file {file_path}: {e}")

    def cleanup(self):
        """Remove all temporary files and directory."""
        for file_path in self.temp_files:
            try:
                file_path.unlink(missing_ok=True)
            except FileNotFoundError:
                print(f"Temporary file {file_path} not found during cleanup.")
            except PermissionError:
                print(
                    f"Permission denied when removing {file_path} during cleanup.")
            except OSError as e:
                print(f"Error cleaning up temporary file {file_path}: {e}")

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except OSError as e:
            print(f"Error removing temporary directory: {e}")


# Global instance
temp_manager = TempFileManager()
