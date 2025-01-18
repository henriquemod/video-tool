"""
Multimedia Assistant - Desktop Application for Media Processing

This is the main entry point for the Multimedia Assistant application, a desktop tool
for processing and enhancing images and videos. The application provides features like:

- Video downloading from multiple platforms
- Image resizing and batch processing
- Temporary file management
- Cross-platform GUI interface

The application uses PyQt5 for the GUI and various multimedia processing libraries
like OpenCV, yt-dlp, and Pillow for media handling.

Usage:
    Run this script directly to launch the application:
    $ python main.py
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Apply torchvision compatibility patch before importing any other modules
from src.utils.torchvision_patch import apply_patch
apply_patch()

from src.app import run_app

if __name__ == "__main__":
    run_app()
