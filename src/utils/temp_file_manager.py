import os
import tempfile
import atexit
import shutil
from pathlib import Path


class TempFileManager:
    """Manages temporary files and cleanup for the application"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "multimedia_assistant"
        self.temp_files = set()
        self.setup_temp_directory()
        atexit.register(self.cleanup)

    def setup_temp_directory(self):
        """Create temporary directory if it doesn't exist"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_temp_path(self, prefix="", suffix=""):
        """Generate a temporary file path"""
        temp_file = Path(tempfile.mktemp(
            prefix=prefix, suffix=suffix, dir=str(self.temp_dir)))
        self.temp_files.add(temp_file)
        return temp_file

    def remove_temp_file(self, file_path):
        """Remove a specific temporary file"""
        try:
            Path(file_path).unlink(missing_ok=True)
            self.temp_files.discard(Path(file_path))
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")

    def cleanup(self):
        """Remove all temporary files and directory"""
        for file_path in self.temp_files:
            try:
                file_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"Error cleaning up temporary file {file_path}: {e}")

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Error removing temporary directory: {e}")


# Global instance
temp_manager = TempFileManager()
