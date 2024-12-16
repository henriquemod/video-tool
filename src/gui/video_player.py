"""
Video player module for the multimedia assistant application.
Provides functionality for video playback, frame extraction, and AI upscaling.
"""

import sys
import os
import shutil
import cv2
from PyQt5.QtCore import QStandardPaths
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QCheckBox, QComboBox,
    QProgressDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir, QThread, pyqtSignal

from .crop_dialog import CropDialog
from ..processing.ai_upscaling import AIUpscaler, get_available_models, get_model_names
from ..utils.temp_file_manager import temp_manager
from ..utils.icon_utils import generateIcon


class UpscaleThread(QThread):
    """Thread for handling AI upscaling operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img, model_id: str, progress_callback=None):
        super().__init__()
        self.img = img
        self.model_id = model_id
        self.upscaler = AIUpscaler()

    def run(self):
        try:
            # Use the centralized upscaler
            result = self.upscaler.upscale(
                self.img,
                self.model_id,
                progress_callback=self.progress.emit
            )

            if result is not None:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
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

        # Initialize outputFolder with a default path
        self.outputFolder = QDir.currentPath()

        # Initialize ai_capable attribute
        self.ai_capable = self.check_ai_capabilities()

        # Set up the video player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setNotifyInterval(250)
        self.videoWidget = QVideoWidget()

        # Initialize OpenCV VideoCapture
        self.cap = None

        # Set up the main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)

        # Set up the media player
        self.mediaPlayer.setVideoOutput(self.videoWidget)

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

        # Add playback controls and volume
        self.setup_playback_controls(mainLayout)

        # Add additional options (Upscale, Crop, and Volume) in the same row
        if self.ai_capable:
            self.add_upscale_and_crop_controls(mainLayout)

        self.setLayout(mainLayout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)

        # Initialize crop checkbox as False by default
        self.allowCrop = False

        # Initialize upscaler
        self.upscaler = AIUpscaler()

    def get_data_directory(self):
        """
        Returns the directory where video data is stored.
        Modify this method to return the appropriate path as needed.
        """
        # Example: Return user's Videos directory
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

    def check_ai_capabilities(self):
        """Check if system can run AI upscaling."""
        try:
            # First try to import required modules
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Check for GPU support
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(
                torch.backends, 'mps') and torch.backends.mps.is_available()

            if has_cuda:
                gpu_memory = torch.cuda.get_device_properties(
                    0).total_memory / 1024**3  # Convert to GB
                print(f"CUDA GPU detected with {gpu_memory:.2f}GB memory")
                return True
            elif has_mps:
                print("Apple Metal GPU detected")
                return True
            else:
                print("No compatible GPU detected, AI upscaling will use CPU (slow)")
                return False

        except ImportError as e:
            print(f"AI capabilities not available: {str(e)}")
            print("To enable AI upscaling, install required packages with:")
            print(
                "pip install numpy==1.24.3 torch==2.0.1 torchvision==0.15.2 basicsr realesrgan")
            return False
        except Exception as e:
            print(f"Error checking AI capabilities: {str(e)}")
            return False

    def setup_playback_controls(self, parent_layout):
        """Set up the playback control buttons in sequence."""
        # First row - slider only
        sliderLayout = QHBoxLayout()
        sliderLayout.setContentsMargins(0, 0, 0, 0)
        sliderLayout.setSpacing(5)

        self.currentTimeLabel = QLabel("00:00:00:00")
        self.totalTimeLabel = QLabel("00:00:00:00")
        self.currentTimeLabel.setFixedHeight(20)
        self.totalTimeLabel.setFixedHeight(20)
        self.positionSlider.setFixedHeight(20)

        sliderLayout.addWidget(self.currentTimeLabel)
        sliderLayout.addWidget(self.positionSlider)
        sliderLayout.addWidget(self.totalTimeLabel)
        parent_layout.addLayout(sliderLayout)

        # Second row - playback controls and screenshot
        controlsLayout = QHBoxLayout()
        controlsLayout.setContentsMargins(0, 0, 0, 0)

        # Navigation buttons with standard icons and tooltips
        self.back30SecondsButton = QPushButton(" x2")
        self.back30SecondsButton.setFixedSize(100, 36)
        self.back30SecondsButton.setToolTip("Jump back 30 seconds")
        self.back30SecondsButton.setIcon(
            generateIcon("media-seek-backward"))

        self.back15SecondsButton = QPushButton()
        self.back15SecondsButton.setFixedSize(100, 36)
        self.back15SecondsButton.setToolTip("Jump back 15 seconds")
        self.back15SecondsButton.setIcon(
            generateIcon("media-seek-backward"))

        self.backFrameButton = QPushButton("-")
        self.backFrameButton.setFixedSize(100, 36)
        self.backFrameButton.setToolTip("Previous frame")
        self.backFrameButton.setStyleSheet(
            "QPushButton { font-weight: bold; } ")

        self.playButton = QPushButton()
        self.playButton.setFixedSize(100, 36)
        self.playButton.setToolTip("Play/Pause")
        self.playButton.setIcon(generateIcon("media-playback-start"))

        self.stopButton = QPushButton()
        self.stopButton.setFixedSize(100, 36)
        self.stopButton.setToolTip("Stop")
        self.stopButton.setIcon(generateIcon("media-playback-stop"))

        self.forwardFrameButton = QPushButton("+")
        self.forwardFrameButton.setFixedSize(100, 36)
        self.forwardFrameButton.setToolTip("Next frame")
        self.forwardFrameButton.setStyleSheet(
            "QPushButton { font-weight: bold; } ")

        self.forward15SecondsButton = QPushButton()
        self.forward15SecondsButton.setFixedSize(100, 36)
        self.forward15SecondsButton.setToolTip("Jump forward 15 seconds")
        self.forward15SecondsButton.setIcon(
            generateIcon("media-seek-forward"))

        self.forward30SecondsButton = QPushButton(" x2")
        self.forward30SecondsButton.setFixedSize(100, 36)
        self.forward30SecondsButton.setToolTip("Jump forward 30 seconds")
        self.forward30SecondsButton.setIcon(
            generateIcon("media-seek-forward"))

        # Screenshot button
        self.screenshotButton = QPushButton(
            " Screenshot")  # Added space before text
        self.screenshotButton.setFixedSize(120, 36)
        self.screenshotButton.setToolTip("Take a screenshot (Ctrl+S)")
        self.screenshotButton.setShortcut("Ctrl+S")
        # https://specifications.freedesktop.org/icon-naming-spec/latest/
        self.screenshotButton.setIcon(generateIcon("camera-photo", True))
        self.screenshotButton.setStyleSheet(
            "QPushButton { background-color: #678dc6; padding-left: 5px; } QPushButton QToolTip { background-color: none; }")
        self.screenshotButton.clicked.connect(self.take_screenshot)

        # Connect button signals
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

        # Add buttons to layout in the desired sequence
        controlsLayout.addWidget(self.back30SecondsButton)
        controlsLayout.addWidget(self.back15SecondsButton)
        controlsLayout.addWidget(self.backFrameButton)
        controlsLayout.addWidget(self.playButton)
        controlsLayout.addWidget(self.stopButton)
        controlsLayout.addWidget(self.forwardFrameButton)
        controlsLayout.addWidget(self.forward15SecondsButton)
        controlsLayout.addWidget(self.forward30SecondsButton)
        controlsLayout.addWidget(self.screenshotButton)

        parent_layout.addLayout(controlsLayout)

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
        self.allowCrop = False
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
        # Set fixed width for the volume slider
        self.volumeSlider.setFixedWidth(100)
        self.volumeSlider.valueChanged.connect(self.mediaPlayer.setVolume)
        controls_layout.addWidget(self.volumeSlider)

        # Add stretch to push everything to the left
        controls_layout.addStretch()

        parent_layout.addLayout(controls_layout)

    def load_video(self, file_path):
        """
        Initializes video playback for the specified file path.

        Args:
            file_path (str): Path to the video file
        """
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.critical(
                self, "Error", "Failed to open video with OpenCV.")
        else:
            # Start playback automatically
            self.mediaPlayer.play()
            self.playButton.setIcon(generateIcon("media-playback-pause"))

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

    def ms_to_time(self, ms):
        """Convert milliseconds to hh:mm:ss:ms format."""
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = (ms % 1000) // 10
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
        # Get the current position of the video
        current_position = self.mediaPlayer.position()

        # Set the video capture to the current position
        self.cap.set(cv2.CAP_PROP_POS_MSEC, current_position)

        # Read the current frame
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.critical(
                self, "Error", "Failed to capture frame from video.")
            return

        # Generate temp path for original screenshot
        temp_screenshot = temp_manager.get_temp_path(
            prefix="screenshot_", suffix=".png")

        # Save the original screenshot to temp location
        cv2.imwrite(str(temp_screenshot), frame)

        processing_path = temp_screenshot
        final_path = None

        try:
            # Handle cropping if enabled
            if hasattr(self, 'allowCrop') and self.allowCrop:
                crop_dialog = CropDialog(str(processing_path), self)
                if crop_dialog.exec_() == QDialog.Accepted:
                    processing_path = crop_dialog.get_result_path()
                    if not processing_path:
                        raise Exception("Cropping failed")

            # Handle upscaling based on selected option
            selected_model = get_available_models(
            )[self.upscale_combo.currentIndex()]

            if selected_model.id != "no_upscale":
                # Read the image to upscale
                img = cv2.imread(str(processing_path))
                if img is None:
                    raise Exception("Failed to load image for upscaling")

                # Now we'll ask for the final save location
                default_filename = f"enhanced_screenshot_{current_position}.png"
                final_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Enhanced Screenshot",
                    os.path.join(QStandardPaths.writableLocation(
                        QStandardPaths.PicturesLocation), default_filename),
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
                    lambda result: self.handle_upscale_finished(
                        result, final_path, selected_model.id, selected_model.scale, progress
                    )
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
                return

            # If no upscaling, but cropping was done, save the cropped image
            elif processing_path != temp_screenshot:
                # Ask for save location for cropped image
                default_filename = f"cropped_screenshot_{current_position}.png"
                final_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Cropped Screenshot",
                    os.path.join(QStandardPaths.writableLocation(
                        QStandardPaths.PicturesLocation), default_filename),
                    "PNG Images (*.png);;All Files (*)"
                )

                if final_path:
                    shutil.copy2(processing_path, final_path)

            # If no processing was done, save original screenshot
            else:
                default_filename = f"screenshot_{current_position}.png"
                final_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Screenshot",
                    os.path.join(QStandardPaths.writableLocation(
                        QStandardPaths.PicturesLocation), default_filename),
                    "PNG Images (*.png);;All Files (*)"
                )

                if final_path:
                    shutil.copy2(temp_screenshot, final_path)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to process image: {str(e)}")
        finally:
            # Cleanup temporary files
            if temp_screenshot != processing_path:
                temp_manager.remove_temp_file(temp_screenshot)
            if processing_path and processing_path != temp_screenshot:
                temp_manager.remove_temp_file(processing_path)

        # Show success message if file was saved
        if final_path and os.path.exists(final_path):
            QMessageBox.information(
                self, "Success", f"Image saved: {final_path}")

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
        event.accept()

    def toggle_crop(self, state):
        """Toggle crop functionality"""
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

    def handle_upscale_finished(self, result, processing_path, method, scale, progress_dialog):
        """
        Handle the completion of the upscaling process.

        Args:
            result: The upscaled image
            processing_path (str): Path of the input image
            method (str): Upscaling method used
            scale (int): Upscaling factor
            progress_dialog (QProgressDialog): Progress dialog to close
        """
        try:
            # Close progress dialog
            progress_dialog.close()

            if result is None:
                raise Exception("Upscaling failed: No output received")

            # Generate output filename
            base_path = os.path.splitext(processing_path)[0]
            output_path = f"{base_path}_{method}_{scale}x.png"

            # Save the upscaled image
            cv2.imwrite(output_path, result)

            # Show success message
            QMessageBox.information(
                self,
                "Success",
                f"Upscaled image saved:\n{output_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save upscaled image: {str(e)}"
            )

    def handle_upscale_error(self, error_message, progress_dialog):
        """
        Handle errors that occur during upscaling.

        Args:
            error_message (str): The error message
            progress_dialog (QProgressDialog): Progress dialog to close
        """
        # Close progress dialog
        progress_dialog.close()

        # Show error message
        QMessageBox.critical(
            self,
            "Upscaling Error",
            f"An error occurred during upscaling:\n{error_message}"
        )
