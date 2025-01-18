"""
Video player module for the multimedia assistant application.
Provides functionality for video playback, frame extraction, and AI upscaling.
"""

import sys
import os
import shutil
import cv2
import torch
from PyQt5.QtCore import QStandardPaths
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QCheckBox, QComboBox,
    QProgressDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir, QThread, pyqtSignal

from ..dialogs.crop_dialog import CropDialog
from ...processing.upscaling import (
    get_available_models,
    get_model_names,
    create_upscaler
)
from ...utils.temp_file_manager import temp_manager
from ...utils.icon_utils import generateIcon
from ...exceptions.upscale_error import UpscaleError
from ...exceptions.video_processing_error import VideoProcessingError


class UpscaleThread(QThread):
    """Thread for handling AI upscaling operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img, model_id: str):
        super().__init__()
        self.img = img
        self.model_id = model_id
        self.upscaler = create_upscaler(model_id)
        self._is_cancelled = False

    def run(self):
        """
        Execute the upscaling operation in a separate thread.

        This method runs the AI upscaling process using the configured model,
        emitting progress updates and handling errors appropriately.

        Signals:
            progress (int): Emits progress percentage during upscaling
            finished (object): Emits the upscaled image result on success
            error (str): Emits error message if upscaling fails
        """
        try:
            if self._is_cancelled:
                raise UpscaleError("Upscaling cancelled by user")

            if self.upscaler is None:
                # Handle basic upscaling methods
                if self.model_id.startswith('bicubic'):
                    scale = int(self.model_id.split('_')[1][0])
                    result = cv2.resize(self.img, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_CUBIC)
                elif self.model_id.startswith('lanczos'):
                    scale = int(self.model_id.split('_')[1][0])
                    result = cv2.resize(self.img, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_LANCZOS4)
                else:
                    raise UpscaleError(
                        f"Unsupported upscaling method: {self.model_id}")
            else:
                # Use AI upscaler
                result = self.upscaler.upscale(
                    self.img,
                    progress_callback=self.progress.emit
                )

            if result is not None:
                self.finished.emit(result)
            else:
                raise UpscaleError("Upscaling failed: No output received")

        except (RuntimeError, ValueError, IOError) as e:
            # Handle specific exceptions that might occur during upscaling
            self.error.emit(f"Upscaling error: {str(e)}")
        except UpscaleError as e:
            # Handle custom upscaling errors
            self.error.emit(str(e))

    def cancel(self):
        """Cancel the upscaling operation."""
        self._is_cancelled = True
        if self.upscaler is not None:
            self.upscaler.cancel()


class VideoPlayer(QWidget):
    """
    VideoPlayer Widget for Multimedia Assistant Application

    This module implements a professional-grade video player widget with integrated AI upscaling
    capabilities and advanced playback controls.

    Key Features:
    - Professional video playback with frame-accurate navigation
    - AI-powered screenshot enhancement using SwinIR and Real-ESRGAN
    - Advanced screenshot capabilities with cropping support
    - Frame-by-frame navigation with keyboard shortcuts
    - Multi-threaded processing for responsive UI
    - Cross-platform compatibility (Windows, macOS, Linux)
    - GPU acceleration support for AI operations

    Technical Implementation:
    - Hybrid architecture using QMediaPlayer for playback and OpenCV for frame processing
    - Integrated AI models (SwinIR, Real-ESRGAN) for high-quality upscaling
    - Efficient memory management with tile-based processing for large images
    - Multi-threaded design for background processing
    - Responsive GUI with platform-specific optimizations

    Components:
    1. Core Playback:
    - QMediaPlayer for smooth video/audio playback
    - Frame-accurate seeking and navigation
    - Custom slider implementation for precise control

    2. AI Enhancement:
    - Real-ESRGAN and SwinIR integration
    - GPU acceleration when available
    - Progress tracking and cancellation support
    - Tile-based processing for large images

    3. Screenshot System:
    - High-quality frame capture
    - Multiple upscaling options (2x, 4x)
    - Optional cropping with aspect ratio control
    - Configurable output directory

    4. Navigation Controls:
    - Frame-by-frame navigation (←/→)
    - Second jumps (Shift + ←/→)
    - Minute jumps (Ctrl + ←/→)
    - Play/Pause toggle
    - Volume control

    Dependencies:
    - PyQt5: GUI framework and media playback
    - OpenCV (cv2): Frame capture and image processing
    - torch: AI model backend
    - basicsr: AI model architectures
    - realesrgan: Real-ESRGAN implementation

    Example usage:
        player = VideoPlayer()
        player.load_video("path/to/video.mp4")
        player.show()

    Performance Considerations:
    - GPU acceleration automatically detected and utilized
    - Tile-based processing for large images
    - Background processing for AI operations
    - Memory-efficient handling of high-resolution content

    @see @Project Structure#video_player.py
    @see @Project#AI Enhancement
    @see @Project#Screenshot System
    """

    def __init__(self):
        super().__init__()
        
        # Add comprehensive media system check for Windows
        if sys.platform == 'win32':
            # Check codecs first
            codec_status = self.check_video_codecs()
            if codec_status:
                print("\nCodec Status:")
                for codec, status in codec_status.items():
                    print(f"{codec}: {'Installed' if status else 'Missing'}")
            
            # Check media service
            from PyQt5.QtMultimedia import QMediaPlayer
            test_player = QMediaPlayer()
            
            # Check media service availability
            availability = test_player.availability()
            if availability == QMediaPlayer.ServiceMissingError:
                print("WARNING: Media service is not available!")
                print("Please check DirectShow installation and codecs on Windows")
                QMessageBox.warning(self, "Media Service Warning",
                    "Media playback service is not available.\n\n"
                    "Please ensure you have the necessary codecs installed:\n"
                    "1. Install K-Lite Codec Pack (Basic)\n"
                    "2. Install LAV Filters\n"
                    "3. Install FFmpeg using winget or chocolatey"
                )
        
        # Initialize instance attributes
        self.outputFolder = QDir.currentPath()
        self.ai_capable = self.check_ai_capabilities()
        
        # Initialize GUI components with audio and video output
        self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer.setNotifyInterval(250)
        self.videoWidget = QVideoWidget()

        # Set default volume
        self.mediaPlayer.setVolume(50)  # 50% volume

        # Connect error handling
        self.mediaPlayer.error.connect(self._handle_error)

        # Initialize OpenCV VideoCapture
        self.cap = None

        # Initialize UI elements
        self.currentTimeLabel = QLabel("00:00:00:00")
        self.totalTimeLabel = QLabel("00:00:00:00")
        self.back30SecondsButton = None
        self.back15SecondsButton = None
        self.backFrameButton = None
        self.playButton = None
        self.stopButton = None
        self.forwardFrameButton = None
        self.forward15SecondsButton = None
        self.forward30SecondsButton = None
        self.screenshotButton = None
        self.upscale_thread = None

        self.allowCrop = False

        # Set up the video player
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # Set up the main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)

        # Create the position slider
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.set_position)

        # Apply stylesheet to reduce slider handle size on macOS
        if sys.platform == "darwin":
            self.positionSlider.setStyleSheet("""
                QSlider::handle:horizontal {
                    background: #5A5A5A;
                    border: 1px solid #5A5A5A;
                    width: 10px;
                    height: 10px;
                    margin: -5px 0;
                    border-radius: 5px;
                }
                QSlider::groove:horizontal {
                    height: 4px;
                    background: #C0C0C0;
                    border-radius: 2px;
                }
            """)

        # Set fixed heights for time labels
        self.currentTimeLabel.setFixedHeight(20)
        self.totalTimeLabel.setFixedHeight(20)

        # Set up playback controls and volume
        self.setup_playback_controls(mainLayout)

        # Add additional options (Upscale, Crop, and Volume) in the same row
        if self.ai_capable:
            self.add_upscale_and_crop_controls(mainLayout)

        self.setLayout(mainLayout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)

    def get_data_directory(self):
        """
        Returns the directory where video data is stored.
        Modify this method to return the appropriate path as needed.

        Returns:
            str: Path to the data directory
        """
        # Example: Return user's Videos directory
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    def check_ai_capabilities(self):
        """Check if system can run AI upscaling."""
        try:
            # First check if torch is properly installed with CUDA
            if not hasattr(torch, 'cuda'):
                print("PyTorch CUDA support not available - reinstall PyTorch with CUDA support")
                print("Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                return False

            # Check for GPU support
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

            if has_cuda:
                try:
                    # Test CUDA functionality
                    test_tensor = torch.tensor([1.0], device='cuda')
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                    print(f"CUDA GPU detected with {gpu_memory:.2f}GB memory")
                    print(f"CUDA Version: {torch.version.cuda}")
                    return True
                except RuntimeError as e:
                    print(f"CUDA initialization error: {str(e)}")
                    print("Please ensure NVIDIA drivers are up to date")
                    print("Download latest drivers from: https://www.nvidia.com/download/index.aspx")
                    return False
            elif has_mps:
                print("Apple Metal GPU detected")
                return True
            else:
                print("No compatible GPU detected, AI upscaling will use CPU (slow)")
                print("For NVIDIA GPUs, please install PyTorch with CUDA support:")
                print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                return False

        except Exception as e:
            print(f"Error checking AI capabilities: {str(e)}")
            print("\nTo enable GPU acceleration on Windows with NVIDIA GPUs:")
            print("1. Install NVIDIA drivers from https://www.nvidia.com/download/index.aspx")
            print("2. Install PyTorch with CUDA support:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False

    def setup_playback_controls(self, parent_layout):
        """Set up the playback control buttons in sequence."""
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(5)

        slider_layout.addWidget(self.currentTimeLabel)
        slider_layout.addWidget(self.positionSlider)
        slider_layout.addWidget(self.totalTimeLabel)
        parent_layout.addLayout(slider_layout)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.create_navigation_buttons()
        self.create_screenshot_button()
        self.connect_button_signals()
        self.add_buttons_to_layout(controls_layout)

        parent_layout.addLayout(controls_layout)

    def create_navigation_buttons(self):
        """Create all navigation buttons with their properties."""
        # Back buttons
        self.back30SecondsButton = self._create_button(
            " x2", "Jump back 30 seconds", "media-seek-backward")
        self.back15SecondsButton = self._create_button(
            "", "Jump back 15 seconds", "media-seek-backward")
        self.backFrameButton = self._create_button(
            "-", "Previous frame", style="font-weight: bold")

        # Playback buttons
        self.playButton = self._create_button(
            "", "Play/Pause", "media-playback-start")
        self.stopButton = self._create_button(
            "", "Stop", "media-playback-stop")

        # Forward buttons
        self.forwardFrameButton = self._create_button(
            "+", "Next frame", style="font-weight: bold")
        self.forward15SecondsButton = self._create_button(
            "", "Jump forward 15 seconds", "media-seek-forward")
        self.forward30SecondsButton = self._create_button(
            " x2", "Jump forward 30 seconds", "media-seek-forward")

    def _create_button(self, text="", tooltip="", icon_name=None, style=None):
        """Create a button with standard properties."""
        button = QPushButton(text)
        button.setFixedSize(100, 36)
        button.setToolTip(tooltip)
        if icon_name:
            button.setIcon(generateIcon(icon_name))
        if style:
            button.setStyleSheet(f"QPushButton {{ {style} }}")
        return button

    def create_screenshot_button(self):
        """Create and configure the screenshot button."""
        self.screenshotButton = QPushButton(" Screenshot")
        self.screenshotButton.setFixedSize(120, 36)
        self.screenshotButton.setToolTip("Take a screenshot (Ctrl+S)")
        self.screenshotButton.setShortcut("Ctrl+S")
        self.screenshotButton.setIcon(generateIcon("camera-photo", True))
        self.screenshotButton.setStyleSheet("""
            QPushButton { 
                background-color: #678dc6; 
                color: white;
                padding-left: 5px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7699d1;
            }
            QPushButton:pressed {
                background-color: #5a7eb2;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
                color: #e0e0e0;
            }
            QPushButton QToolTip { 
                background-color: none;
            }
        """)
        self.screenshotButton.clicked.connect(self.take_screenshot)
        # Initialize button state
        self.update_screenshot_button_state()

    def connect_button_signals(self):
        """Connect all button signals to their respective slots."""
        self.back30SecondsButton.clicked.connect(
            lambda: self.seek_relative(-30000))
        self.back15SecondsButton.clicked.connect(
            lambda: self.seek_relative(-15000))
        self.backFrameButton.clicked.connect(lambda: self.seek_frames(-1))
        self.playButton.clicked.connect(self.toggle_playback)
        self.stopButton.clicked.connect(self.stop_video)
        self.forwardFrameButton.clicked.connect(lambda: self.seek_frames(1))
        self.forward15SecondsButton.clicked.connect(
            lambda: self.seek_relative(15000))
        self.forward30SecondsButton.clicked.connect(
            lambda: self.seek_relative(30000))

    def add_buttons_to_layout(self, layout):
        """Add all buttons to the layout in sequence."""
        buttons = [
            self.back30SecondsButton,
            self.back15SecondsButton,
            self.backFrameButton,
            self.playButton,
            self.stopButton,
            self.forwardFrameButton,
            self.forward15SecondsButton,
            self.forward30SecondsButton,
            self.screenshotButton
        ]

        for button in buttons:
            layout.addWidget(button)

    def add_upscale_and_crop_controls(self, parent_layout):
        """Add upscale, crop controls, and volume in the same line."""
        controls_layout = QHBoxLayout()

        # Add upscale label and combo
        upscale_label = QLabel("Upscale:")
        controls_layout.addWidget(upscale_label)

        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(get_model_names())
        controls_layout.addWidget(self.upscale_combo)

        # Add performance warning label
        self.performance_label = QLabel()
        self.performance_label.setStyleSheet("""
            QLabel {
                color: #FF8C00;
                font-weight: bold;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(255, 140, 0, 0.1);
            }
        """)
        controls_layout.addWidget(self.performance_label)
        self.performance_label.hide()

        # Add some spacing
        controls_layout.addSpacing(20)

        # Add crop checkbox
        self.cropCheckbox = QCheckBox("Crop Image")
        self.cropCheckbox.setChecked(self.allowCrop)
        self.cropCheckbox.stateChanged.connect(self.toggle_crop)
        controls_layout.addWidget(self.cropCheckbox)

        # Add spacing before volume control
        controls_layout.addSpacing(20)

        # Add volume control
        volumeLabel = QLabel("Volume")
        controls_layout.addWidget(volumeLabel)

        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(50)  # Default volume
        self.volumeSlider.setFixedWidth(100)
        self.volumeSlider.valueChanged.connect(self.set_volume)
        controls_layout.addWidget(self.volumeSlider)

        # Add stretch to push everything to the left
        controls_layout.addStretch()

        parent_layout.addLayout(controls_layout)

    def set_volume(self, value):
        """Set the audio volume."""
        self.mediaPlayer.setVolume(value)

    def load_video(self, file_path):
        """
        Initializes video playback for the specified file path.

        Args:
            file_path (str): Path to the video file
        """
        if not os.path.exists(file_path):
            QMessageBox.critical(
                self, "Error", f"File not found: {file_path}")
            return

        # Log video file details
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"\nAttempting to load video:")
            print(f"Path: {file_path}")
            print(f"Size: {file_size:.2f} MB")
            
            # Check video file with OpenCV first
            test_cap = cv2.VideoCapture(file_path)
            if test_cap.isOpened():
                fps = test_cap.get(cv2.CAP_PROP_FPS)
                width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"OpenCV Info - FPS: {fps}, Resolution: {width}x{height}")
                test_cap.release()
            else:
                print("Warning: OpenCV cannot open the video file")
        except Exception as e:
            print(f"Error checking video file: {str(e)}")

        # Create QMediaContent with local file URL
        url = QUrl.fromLocalFile(os.path.abspath(file_path))
        media = QMediaContent(url)
        
        print("\nMedia Content Info:")
        print(f"URL: {url.toString()}")
        print(f"Is Local: {url.isLocalFile()}")
        
        # Set the media and verify it was loaded
        self.mediaPlayer.setMedia(media)
        
        # Wait briefly for media to be processed
        import time
        time.sleep(0.5)  # Give media player time to process
        
        # Check media status
        status = self.mediaPlayer.mediaStatus()
        print(f"\nMedia Status After Loading: {self._get_status_string(status)}")
        
        # Check for media player errors
        error = self.mediaPlayer.error()
        if error != QMediaPlayer.NoError:
            error_msg = f"Media Player Error: {self._get_error_string(error)}"
            print(f"\nError Details: {error_msg}")
            print(f"Error String: {self.mediaPlayer.errorString()}")
            
            # Show detailed error to user
            QMessageBox.critical(self, "Media Error", 
                f"Failed to load video:\n\n{error_msg}\n\n"
                f"System Info:\n"
                f"- OS: {sys.platform}\n"
                f"- Qt Version: {QT_VERSION_STR}\n\n"
                f"Please ensure you have the necessary video codecs installed."
            )
            return
        
        # Initialize OpenCV capture
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.critical(
                self, "Error", "Failed to open video with OpenCV.")
            return

        # Update screenshot button state
        self.update_screenshot_button_state()

        # Start playback automatically
        self.mediaPlayer.play()
        
        # Update play button icon
        self.playButton.setIcon(generateIcon("media-playback-pause"))
        
        # Log final status
        self._log_media_status()

    def _get_status_string(self, status):
        """Convert QMediaPlayer status to string for debugging."""
        status_map = {
            QMediaPlayer.UnknownMediaStatus: "Unknown",
            QMediaPlayer.NoMedia: "No Media",
            QMediaPlayer.LoadingMedia: "Loading",
            QMediaPlayer.LoadedMedia: "Loaded",
            QMediaPlayer.StalledMedia: "Stalled",
            QMediaPlayer.BufferingMedia: "Buffering",
            QMediaPlayer.BufferedMedia: "Buffered",
            QMediaPlayer.EndOfMedia: "End of Media",
            QMediaPlayer.InvalidMedia: "Invalid"
        }
        return status_map.get(status, f"Unknown Status: {status}")

    def _get_error_string(self, error):
        """Convert QMediaPlayer error to descriptive string."""
        error_map = {
            QMediaPlayer.NoError: "No Error",
            QMediaPlayer.ResourceError: "Resource Error (missing codecs or file access issue)",
            QMediaPlayer.FormatError: "Format Error (unsupported format)",
            QMediaPlayer.NetworkError: "Network Error",
            QMediaPlayer.AccessDeniedError: "Access Denied",
            QMediaPlayer.ServiceMissingError: "Media Service Missing (check DirectShow installation)",
        }
        return error_map.get(error, f"Unknown Error: {error}")

    def _log_media_status(self):
        """Log detailed media player status for debugging."""
        status = self.mediaPlayer.mediaStatus()
        state = self.mediaPlayer.state()
        error = self.mediaPlayer.error()
        
        print("\nDetailed Media Player Status:")
        print(f"Media Status: {self._get_status_string(status)}")
        print(f"Player State: {state}")
        print(f"Current Volume: {self.mediaPlayer.volume()}")
        print(f"Media Player Available: {self.mediaPlayer.availability()}")
        print(f"Media Duration: {self.mediaPlayer.duration()} ms")
        print(f"Is Seekable: {self.mediaPlayer.isSeekable()}")
        print(f"Is Audio Available: {self.mediaPlayer.isAudioAvailable()}")
        print(f"Is Video Available: {self.mediaPlayer.isVideoAvailable()}")
        
        if error != QMediaPlayer.NoError:
            print(f"Error: {self._get_error_string(error)}")
            print(f"Error String: {self.mediaPlayer.errorString()}")

    def toggle_playback(self):
        """Toggle between play and pause."""
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setIcon(generateIcon("media-playback-start"))
        else:
            self.mediaPlayer.play()
            self.playButton.setIcon(generateIcon("media-playback-pause"))

    def stop_video(self):
        """Stop video playback."""
        self.mediaPlayer.stop()
        self.playButton.setIcon(generateIcon("media-playback-start"))

    def set_position(self, position):
        """Set the media player position."""
        self.mediaPlayer.setPosition(position)

    def position_changed(self, position):
        """Update the slider position and current time label."""
        self.positionSlider.setValue(position)
        self.currentTimeLabel.setText(self.ms_to_time(position))

    def duration_changed(self, duration):
        """Update the slider range and total time label."""
        self.positionSlider.setRange(0, duration)
        self.totalTimeLabel.setText(self.ms_to_time(duration))

    def media_state_changed(self, state):
        """Handle media state changes."""
        if state == QMediaPlayer.StoppedState:
            self.playButton.setIcon(generateIcon("media-playback-start"))
        elif state == QMediaPlayer.PlayingState:
            self._log_media_status()  # Log status when playing starts

    def ms_to_time(self, ms, for_filename=False):
        """
        Convert milliseconds to time format.
        
        Args:
            ms (int): Milliseconds to convert
            for_filename (bool): If True, returns filename-safe format
        
        Returns:
            str: Formatted time string
        """
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = (ms % 1000) // 10
        
        if for_filename:
            # Use underscores and hyphens instead of colons for filenames
            return f"{int(hours):02d}-{int(minutes):02d}-{int(seconds):02d}-{int(milliseconds):02d}"
        else:
            # Regular format for display
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{int(milliseconds):02}"

    def set_playback_speed(self, value):
        """Set the playback speed."""
        speed = value / 100.0  # Convert slider value to speed factor
        self.mediaPlayer.setPlaybackRate(speed)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreenButton.setText("Fullscreen")
        else:
            self.showFullScreen()
            self.fullscreenButton.setText("Exit Fullscreen")

    def take_screenshot(self):
        """Captures the current frame as a high-quality screenshot."""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(
                self,
                "No Video Loaded",
                "Please load a video before attempting to take a screenshot."
            )
            return

        try:
            # Capture and save the screenshot
            frame, current_position = self.capture_frame()
            temp_screenshot = self.save_temp_screenshot(frame)

            # Process the screenshot (crop and upscale)
            processing_path = self.process_screenshot(temp_screenshot)

            # If processing_path is None, upscaling is happening asynchronously
            if processing_path is None:
                # Cleanup the temporary screenshot - upscaling thread will handle the rest
                self.cleanup_temporary_files(temp_screenshot, None)
                return

            # Save the final screenshot
            final_path = self.save_final_screenshot(
                processing_path, current_position)

            # Cleanup temporary files
            self.cleanup_temporary_files(temp_screenshot, processing_path)

            # Show success message if file was saved
            if final_path and os.path.exists(final_path):
                QMessageBox.information(
                    self, "Success", f"Image saved: {final_path}"
                )

        except VideoProcessingError as e:
            QMessageBox.critical(self, "Error", str(e))
        except UpscaleError as e:
            QMessageBox.critical(self, "Upscaling Error", str(e))
        except IOError as e:
            QMessageBox.critical(self, "File Error",
                                 f"Failed to process image: {str(e)}")

    def capture_frame(self):
        """
        Capture the current frame from the video.

        Returns:
            tuple: (frame image, current position in milliseconds)
        """
        # Get the current position of the video
        current_position = self.mediaPlayer.position()

        # Set the video capture to the current position
        self.cap.set(cv2.CAP_PROP_POS_MSEC, current_position)

        # Read the current frame
        ret, frame = self.cap.read()
        if not ret:
            raise VideoProcessingError("Failed to capture frame from video.")

        return frame, current_position

    def save_temp_screenshot(self, frame):
        """
        Save the captured frame to a temporary file.

        Args:
            frame: The captured frame image.

        Returns:
            Path to the temporary screenshot.
        """
        # Generate temp path for original screenshot
        temp_screenshot = temp_manager.get_temp_path(
            prefix="screenshot_", suffix=".png")

        # Save the original screenshot to temp location
        cv2.imwrite(str(temp_screenshot), frame)

        return temp_screenshot

    def process_screenshot(self, processing_path):
        """
        Process the screenshot by handling cropping and upscaling.

        Args:
            processing_path (Path): Path to the temporary screenshot.

        Returns:
            Path to the processed screenshot.
        """
        # Handle cropping if enabled
        if self.allowCrop:
            crop_dialog = CropDialog(str(processing_path), self)
            if crop_dialog.exec_() == QDialog.Accepted:
                new_path = crop_dialog.get_result_path()
                if not new_path:
                    raise VideoProcessingError("Cropping failed.")
                processing_path = new_path

        # Handle upscaling based on selected option
        selected_model = get_available_models(
        )[self.upscale_combo.currentIndex()]
        if selected_model.id != "no_upscale":
            img = cv2.imread(str(processing_path))
            if img is None:
                raise VideoProcessingError(
                    "Failed to load image for upscaling.")
            self.start_upscaling_thread(img, selected_model)
            # Return None to indicate async processing is happening
            return None

        return processing_path

    def start_upscaling_thread(self, img, selected_model):
        """
        Start the upscaling thread.

        Args:
            img: Image to upscale.
            selected_model: Model selected for upscaling.
        """
        # Clean up any existing thread
        if self.upscale_thread:
            self.upscale_thread.quit()
            self.upscale_thread.wait()
            self.upscale_thread = None

        # Create filename-safe timestamp
        timestamp = self.ms_to_time(self.mediaPlayer.position(), for_filename=True)
        default_filename = f"enhanced_screenshot_{timestamp}.png"
        
        # Get the Pictures directory
        pictures_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        
        # Create the full path
        save_path = os.path.join(pictures_dir, default_filename)

        final_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Screenshot",
            save_path,
            "PNG Images (*.png);;All Files (*)"
        )

        if not final_path:  # User cancelled
            return

        # Create progress dialog
        progress = QProgressDialog(
            "Upscaling image...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)

        # Create and configure upscale thread
        self.upscale_thread = UpscaleThread(img, selected_model.id)

        # Connect signals
        self.upscale_thread.progress.connect(progress.setValue)
        self.upscale_thread.finished.connect(
            lambda result: self._handle_upscale_finished(
                result, final_path, progress)
        )
        self.upscale_thread.error.connect(
            lambda err: self.handle_upscale_error(err, progress)
        )

        # Connect cancel button
        progress.canceled.connect(self.upscale_thread.cancel)

        # Start thread
        self.upscale_thread.start()

        # Show progress dialog
        progress.exec_()

    def save_final_screenshot(self, processing_path, current_position):
        """
        Save the final screenshot based on processing.

        Args:
            processing_path (Path): Path to the processed screenshot.
            current_position (int): Current video position.

        Returns:
            Path to the final saved screenshot.
        """
        # Use filename-safe format for the timestamp
        if processing_path.suffix == ".png":
            default_filename = f"screenshot_{self.ms_to_time(current_position, for_filename=True)}.png"
        else:
            default_filename = f"screenshot_{self.ms_to_time(current_position, for_filename=True)}{processing_path.suffix}"

        # Get the Pictures directory
        pictures_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        
        # Ensure the path is valid for Windows
        save_path = os.path.join(pictures_dir, default_filename)
        
        final_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            save_path,
            "PNG Images (*.png);;All Files (*)"
        )

        if final_path:
            try:
                shutil.copy2(str(processing_path), final_path)
            except IOError as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save screenshot: {str(e)}\nPath: {final_path}"
                )
                return None

        return final_path

    def cleanup_temporary_files(self, temp_screenshot, processing_path):
        """
        Cleanup temporary screenshot files.

        Args:
            temp_screenshot (Path): Path to the temporary original screenshot.
            processing_path (Path): Path to the processed screenshot.
        """
        if temp_screenshot != processing_path:
            temp_manager.remove_temp_file(temp_screenshot)
        if processing_path and processing_path != temp_screenshot:
            temp_manager.remove_temp_file(processing_path)

    def closeEvent(self, event):
        """
        Handles cleanup when the video player widget is closed.

        Ensures proper release of system resources:
        - Closes OpenCV video capture
        - Releases media player resources

        Args:
            event (QCloseEvent): The close event to handle

        @anchor #resource-cleanup
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.update_screenshot_button_state()
        event.accept()

    def toggle_crop(self, state):
        """Toggle crop functionality."""
        self.allowCrop = bool(state)

    def seek_relative(self, ms):
        """
        Seek relative to current position by specified milliseconds.

        Args:
            ms (int): Milliseconds to seek (positive or negative)
        """
        current = self.mediaPlayer.position()
        new_pos = max(0, min(current + ms, self.mediaPlayer.duration()))
        self.mediaPlayer.setPosition(new_pos)

    def seek_frames(self, frames):
        """
        Seek by specified number of frames forward or backward.

        Args:
            frames (int): Number of frames to seek (positive or negative)
        """
        if not self.cap:
            return

        # Get current frame rate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return

        # Calculate milliseconds per frame
        ms_per_frame = int(1000 / fps)

        # Seek by calculated milliseconds
        self.seek_relative(frames * ms_per_frame)

    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts for frame navigation.

        Left/Right arrows for frame-by-frame
        Shift + Left/Right for second jumps
        Ctrl + Left/Right for minute jumps
        """
        if event.key() == Qt.Key_Left:
            if event.modifiers() & Qt.ControlModifier:
                self.seek_relative(-60000)  # Back 1 minute
            elif event.modifiers() & Qt.ShiftModifier:
                self.seek_relative(-1000)   # Back 1 second
            else:
                self.seek_frames(-1)        # Back 1 frame
        elif event.key() == Qt.Key_Right:
            if event.modifiers() & Qt.ControlModifier:
                self.seek_relative(60000)   # Forward 1 minute
            elif event.modifiers() & Qt.ShiftModifier:
                self.seek_relative(1000)    # Forward 1 second
            else:
                self.seek_frames(1)         # Forward 1 frame
        else:
            super().keyPressEvent(event)

    def _handle_upscale_finished(self, result, final_path, progress_dialog):
        """
        Handle the completion of the upscaling process.

        Args:
            result: The upscaled image
            final_path (str): Path where the image should be saved
            progress_dialog (QProgressDialog): Progress dialog to close
        """
        try:
            # Close progress dialog first
            if progress_dialog and progress_dialog.isVisible():
                progress_dialog.close()

            if result is None:
                raise UpscaleError("Upscaling failed: No output received")

            # Ensure the file extension is supported
            file_ext = os.path.splitext(final_path)[1].lower()
            if not file_ext:
                final_path += '.png'  # Add default extension if none provided
                file_ext = '.png'
            
            # Convert BGR to RGB for certain formats
            if file_ext in ['.jpg', '.jpeg']:
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # Save the upscaled image
            success = cv2.imwrite(final_path, result)
            if not success:
                # Try alternative approach for problematic formats
                try:
                    from PIL import Image
                    import numpy as np
                    # Convert from BGR to RGB for PIL
                    rgb_img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb_img)
                    img_pil.save(final_path)
                    success = True
                except Exception as e:
                    raise VideoProcessingError(
                        f"Failed to save image using alternative method: {str(e)}")

            if not success:
                raise VideoProcessingError(
                    f"Failed to save image to {final_path}")

            # Clean up the upscale thread
            if self.upscale_thread:
                self.upscale_thread.quit()
                self.upscale_thread.wait()
                self.upscale_thread = None

            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Upscaled image saved:\n{final_path}"
            )

        except (UpscaleError, VideoProcessingError) as e:
            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

    def handle_upscale_error(self, error_message, progress_dialog):
        """
        Handle errors that occur during upscaling.

        Args:
            error_message (str): The error message
            progress_dialog (QProgressDialog): Progress dialog to close
        """
        # Close progress dialog
        if progress_dialog and progress_dialog.isVisible():
            progress_dialog.close()

        # Clean up the upscale thread
        if self.upscale_thread:
            self.upscale_thread.quit()
            self.upscale_thread.wait()
            self.upscale_thread = None

        # Show error message
        QMessageBox.critical(
            self,
            "Upscaling Error",
            f"An error occurred during upscaling:\n{error_message}"
        )

    def _handle_error(self, error):
        """Handle media player errors."""
        if error != QMediaPlayer.NoError:
            error_msg = f"Media Player Error: {error} - {self.mediaPlayer.errorString()}"
            print(error_msg)  # Print to console for debugging
            QMessageBox.critical(self, "Error", error_msg)

    def update_screenshot_button_state(self):
        """Enable or disable screenshot button based on video state."""
        has_video = self.cap is not None and self.cap.isOpened()
        self.screenshotButton.setEnabled(has_video)
        if not has_video:
            self.screenshotButton.setToolTip("Load a video first to take screenshots")
        else:
            self.screenshotButton.setToolTip("Take a screenshot (Ctrl+S)")

    def check_video_codecs(self):
        """Check for required video codecs and DirectShow filters."""
        import subprocess
        import winreg
        
        codec_status = {
            'directshow': False,
            # 'ffmpeg': False,
            # 'lav_filters': False,
            'k_lite': False
        }
        
        try:
            # Check for FFmpeg
            # try:
            #     result = subprocess.run(['ffmpeg', '-version'], 
            #                          capture_output=True, 
            #                          text=True)
            #     codec_status['ffmpeg'] = result.returncode == 0
            # except FileNotFoundError:
            #     print("FFmpeg not found in system PATH")
            
            # Check for LAV Filters
            # try:
            #     winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
            #                   r'CLSID\{B98D13E7-55DB-4385-A33D-09FD1BA26338}')
            #     codec_status['lav_filters'] = True
            # except WindowsError:
            #     print("LAV Filters not found")
            
            # Check for K-Lite Codec Pack
            try:
                winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r'SOFTWARE\WOW6432Node\KLCodecPack')
                codec_status['k_lite'] = True
            except WindowsError:
                print("K-Lite Codec Pack not found")
            
            # Check DirectShow filters
            try:
                winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,
                              r'DirectShow\MediaObjects')
                codec_status['directshow'] = True
            except WindowsError:
                print("DirectShow filters not properly registered")
            
            missing_components = [k for k, v in codec_status.items() if not v]
            if missing_components:
                print("\nMissing or unregistered components:")
                print('\n'.join(f"- {component}" for component in missing_components))
                
                QMessageBox.warning(self, "Missing Video Components",
                    "This application requires the following components to be installed:\n\n"
                    "1. K-Lite Codec Pack (Basic):\n"
                    "   https://www.codecguide.com/download_k-lite_codec_pack_basic.htm\n\n"
                    "The application may not work properly without these components."
                )
                
            return codec_status
            
        except Exception as e:
            print(f"Error checking codecs: {str(e)}")
            return None
