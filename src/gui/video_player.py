import cv2
import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QCheckBox, QComboBox,
    QProgressDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir, QThread, pyqtSignal
from .crop_dialog import CropDialog

import torch
import numpy as np


# Model URLs for SwinIR and Real-ESRGAN
MODEL_URLS = {
    'SwinIR-2x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    'SwinIR-4x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    'Real-ESRGAN-2x': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    'Real-ESRGAN-4x': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}


class UpscaleThread(QThread):
    """Thread for handling AI upscaling operations"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, img, method, scale, device):
        super().__init__()
        self.img = img
        self.method = method
        self.scale = scale
        self.device = device
        self._is_cancelled = False

    def run(self):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer

            # Update progress
            self.progress.emit(10)

            output = None  # Initialize output variable

            if "SwinIR" in self.method:
                from basicsr.utils.registry import ARCH_REGISTRY

                # Prepare the image
                if self.img.shape[2] == 3:
                    img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                # Update progress
                self.progress.emit(20)

                # Handle padding
                h, w = img.shape[:2]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8

                if pad_h > 0 or pad_w > 0:
                    img = np.pad(
                        img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

                # Load model
                if self.scale == 2:
                    model_path = load_file_from_url(
                        MODEL_URLS['SwinIR-2x'], model_dir='models')
                else:
                    model_path = load_file_from_url(
                        MODEL_URLS['SwinIR-4x'], model_dir='models')

                # Update progress
                self.progress.emit(40)

                if self._is_cancelled:
                    return

                # Initialize model
                model = ARCH_REGISTRY.get('SwinIR')(
                    upscale=self.scale,
                    in_chans=3,
                    img_size=64,
                    window_size=8,
                    img_range=1.,
                    depths=[6, 6, 6, 6],
                    embed_dim=60,
                    num_heads=[6, 6, 6, 6],
                    mlp_ratio=2,
                    upsampler='pixelshuffledirect',
                    resi_connection='1conv'
                )

                # Load weights
                pretrained_model = torch.load(
                    model_path, map_location=self.device)
                if 'params_ema' in pretrained_model:
                    pretrained_model = pretrained_model['params_ema']
                elif 'params' in pretrained_model:
                    pretrained_model = pretrained_model['params']

                model.load_state_dict(pretrained_model, strict=True)
                model.eval()
                model.to(self.device)

                # Update progress
                self.progress.emit(60)

                if self._is_cancelled:
                    return

                # Process image
                img_tensor = torch.from_numpy(img).float().permute(
                    2, 0, 1).unsqueeze(0) / 255.
                img_tensor = img_tensor.to(self.device)

                # Process in tiles if needed
                tile_size = 640
                tile_overlap = 32

                b, c, h, w = img_tensor.shape
                output = torch.zeros(
                    (b, c, h * self.scale, w * self.scale), device=self.device)

                if h > tile_size or w > tile_size:
                    total_tiles = ((h - 1) // (tile_size - tile_overlap) + 1) * \
                        ((w - 1) // (tile_size - tile_overlap) + 1)
                    current_tile = 0

                    for top in range(0, h, tile_size - tile_overlap):
                        for left in range(0, w, tile_size - tile_overlap):
                            if self._is_cancelled:
                                return

                            bottom = min(top + tile_size, h)
                            right = min(left + tile_size, w)

                            tile = img_tensor[:, :, top:bottom, left:right]

                            with torch.no_grad():
                                tile_output = model(tile)

                            out_top = top * self.scale
                            out_left = left * self.scale
                            out_bottom = bottom * self.scale
                            out_right = right * self.scale

                            output[:, :, out_top:out_bottom,
                                   out_left:out_right] = tile_output

                            current_tile += 1
                            progress = 60 + (current_tile / total_tiles) * 35
                            self.progress.emit(int(progress))
                else:
                    with torch.no_grad():
                        output = model(img_tensor)

                # Final processing
                output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = (output * 255.0).round().astype(np.uint8)
                output = output.transpose(1, 2, 0)

                if pad_h > 0 or pad_w > 0:
                    output = output[:h*self.scale, :w*self.scale, :]

                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            elif "Real-ESRGAN" in self.method:
                try:
                    # Initialize model with correct architecture for Real-ESRGAN
                    model = RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=self.scale
                    )

                    # Select appropriate model based on scale
                    if self.scale == 2:
                        model_path = load_file_from_url(
                            MODEL_URLS['Real-ESRGAN-2x'], model_dir='models')
                    else:
                        model_path = load_file_from_url(
                            MODEL_URLS['Real-ESRGAN-4x'], model_dir='models')

                    # Update progress
                    self.progress.emit(30)

                    if self._is_cancelled:
                        return

                    # Load pre-trained model state
                    pretrained_model = torch.load(
                        model_path, map_location=self.device)
                    if 'params_ema' in pretrained_model:
                        pretrained_model = pretrained_model['params_ema']
                    elif 'params' in pretrained_model:
                        pretrained_model = pretrained_model['params']
                    else:
                        raise KeyError(
                            "No 'params_ema' or 'params' keys found in the model file.")

                    model.load_state_dict(pretrained_model, strict=True)
                    model.eval()
                    model.to(self.device)

                    # Initialize upsampler
                    upsampler = RealESRGANer(
                        scale=self.scale,
                        model_path=model_path,
                        model=model,
                        tile=512,
                        tile_pad=32,
                        pre_pad=0,
                        half=True if self.device == 'cuda' else False,
                        device=self.device
                    )

                    # Update progress
                    self.progress.emit(50)

                    if self._is_cancelled:
                        return

                    # Process image
                    output, _ = upsampler.enhance(
                        self.img, outscale=self.scale)

                except Exception as e:
                    raise Exception(f"Real-ESRGAN processing failed: {str(e)}")

            else:
                raise Exception(f"Unsupported upscaling method: {self.method}")

            # Ensure we have a valid output
            if output is None:
                raise Exception("Failed to generate output image")

            self.progress.emit(100)
            self.finished.emit(output)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._is_cancelled = True


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
        self.videoWidget = QVideoWidget()

        # Initialize OpenCV VideoCapture
        self.cap = None

        # Set up the main layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)

        # Set up the media player
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # Add a slider for video seeking with custom style for macOS
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

        # Add time labels
        self.currentTimeLabel = QLabel("00:00")
        self.totalTimeLabel = QLabel("00:00")

        # Adjust the layout for slider and labels
        sliderLayout = QHBoxLayout()
        sliderLayout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        sliderLayout.setSpacing(5)  # Reduce spacing between widgets

        # Set fixed height for labels and slider to minimize height usage
        self.currentTimeLabel.setFixedHeight(20)
        self.totalTimeLabel.setFixedHeight(20)
        self.positionSlider.setFixedHeight(20)

        sliderLayout.addWidget(self.currentTimeLabel)
        sliderLayout.addWidget(self.positionSlider)
        sliderLayout.addWidget(self.totalTimeLabel)

        # Add screenshot button
        self.screenshotButton = QPushButton("Screenshot")
        self.screenshotButton.clicked.connect(self.take_screenshot)
        sliderLayout.addWidget(self.screenshotButton)

        mainLayout.addLayout(sliderLayout)

        # Add playback controls
        self.setup_playback_controls(mainLayout)

        # Add additional options (Upscale and Crop)
        if self.ai_capable:
            self.add_upscale_controls(mainLayout)
        self.add_crop_controls(mainLayout)

        self.setLayout(mainLayout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)

        # Initialize crop checkbox as False by default
        self.allowCrop = False

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
        controlsLayout = QHBoxLayout()

        # Navigation buttons with icons or text
        self.backMinButton = QPushButton("◀◀ 1m")
        self.backSecButton = QPushButton("◀ 1s")
        self.backFrameButton = QPushButton("◀ 1f")
        self.playButton = QPushButton("Play")
        self.stopButton = QPushButton("Stop")
        self.forwardFrameButton = QPushButton("1f ▶")
        self.forwardSecButton = QPushButton("1s ▶")
        self.forwardMinButton = QPushButton("1m ▶▶")

        # Connect button signals
        self.backMinButton.clicked.connect(
            lambda: self.seek_relative(-60000))  # -1 min
        self.backSecButton.clicked.connect(
            lambda: self.seek_relative(-1000))   # -1 sec
        self.backFrameButton.clicked.connect(lambda: self.seek_frames(-1))
        self.playButton.clicked.connect(self.toggle_playback)
        self.stopButton.clicked.connect(self.stop_video)
        self.forwardFrameButton.clicked.connect(lambda: self.seek_frames(1))
        self.forwardSecButton.clicked.connect(
            lambda: self.seek_relative(1000))    # +1 sec
        self.forwardMinButton.clicked.connect(
            lambda: self.seek_relative(60000))   # +1 min

        # Add buttons to layout in the desired sequence
        controlsLayout.addWidget(self.backMinButton)
        controlsLayout.addWidget(self.backSecButton)
        controlsLayout.addWidget(self.backFrameButton)
        controlsLayout.addWidget(self.playButton)
        controlsLayout.addWidget(self.stopButton)
        controlsLayout.addWidget(self.forwardFrameButton)
        controlsLayout.addWidget(self.forwardSecButton)
        controlsLayout.addWidget(self.forwardMinButton)

        # Volume Control
        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(50)  # Default volume
        self.volumeSlider.setFixedWidth(100)
        self.volumeSlider.valueChanged.connect(self.mediaPlayer.setVolume)
        controlsLayout.addWidget(QLabel("Volume"))
        controlsLayout.addWidget(self.volumeSlider)

        parent_layout.addLayout(controlsLayout)

    def add_upscale_controls(self, parent_layout):
        """Add upscale options to the layout."""
        upscale_layout = QHBoxLayout()

        # Add upscale label
        upscale_label = QLabel("Upscale:")
        upscale_layout.addWidget(upscale_label)

        # Create combo box
        self.upscale_combo = QComboBox()

        # Add all options
        options = [
            "No Upscaling",
            "Bicubic (2x)",
            "Bicubic (3x)",
            "Bicubic (4x)",
            "Lanczos (2x)",
            "Lanczos (3x)",
            "Lanczos (4x)",
            "Real-ESRGAN (2x)",
            "Real-ESRGAN (4x)",
            "SwinIR (2x)",
            "SwinIR (4x)",
        ]
        self.upscale_combo.addItems(options)
        upscale_layout.addWidget(self.upscale_combo)

        # Add performance warning label
        self.performance_label = QLabel()
        self.performance_label.setStyleSheet("""
            QLabel {
                color: #FF8C00;  /* Dark Orange */
                font-weight: bold;
                padding: 2px 5px;
                border-radius: 3px;
                background-color: rgba(255, 140, 0, 0.1);
            }
        """)
        upscale_layout.addWidget(self.performance_label)
        self.performance_label.hide()  # Initially hidden

        # Check AI capabilities and update UI
        has_gpu = self.check_ai_capabilities()

        # Connect combo box change event
        self.upscale_combo.currentTextChanged.connect(
            lambda text: self.on_upscale_option_changed(text, has_gpu)
        )

        parent_layout.addLayout(upscale_layout)

    def add_crop_controls(self, parent_layout):
        """Add crop checkbox to the layout."""
        crop_layout = QHBoxLayout()

        # Create and setup crop checkbox
        self.cropCheckbox = QCheckBox("Crop Image")
        self.allowCrop = False  # Initialize the crop flag
        self.cropCheckbox.setChecked(self.allowCrop)
        self.cropCheckbox.stateChanged.connect(self.toggle_crop)

        # Add checkbox to layout
        crop_layout.addWidget(self.cropCheckbox)

        parent_layout.addLayout(crop_layout)

    def on_upscale_option_changed(self, text, has_gpu):
        """Handle upscale option changes and update performance warning."""
        if "Real-ESRGAN" in text:
            if not has_gpu:
                self.performance_label.setText("⚠️ CPU Mode (Slow)")
                self.performance_label.show()
            else:
                self.performance_label.hide()
        else:
            self.performance_label.hide()

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
            self.playButton.setText("Pause")

    def toggle_playback(self):
        """Toggle between play and pause."""
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.playButton.setText("Play")
        else:
            self.mediaPlayer.play()
            self.playButton.setText("Pause")

    def stop_video(self):
        """Stop video playback."""
        self.mediaPlayer.stop()
        self.playButton.setText("Play")

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
            self.playButton.setText("Play")

    def ms_to_time(self, ms):
        """Convert milliseconds to mm:ss format."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes):02}:{int(seconds):02}"

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
        """
        Captures the current frame as a high-quality screenshot.

        Features:
        - Captures frame at original video resolution
        - Saves with timestamp-based filename
        - Optional cropping through CropDialog
        - Configurable output directory

        @anchor #screenshot-functionality
        @see @Project#Screenshot Capability
        """

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

        # Generate filename with timestamp
        filename = f"screenshot_{current_position}.png"

        # Use outputFolder if defined, otherwise use root project path
        save_path = os.path.join(getattr(
            self, 'outputFolder', os.path.dirname(os.path.abspath(__file__))), filename)

        # Save the original screenshot
        cv2.imwrite(save_path, frame)
        print(f"Original screenshot saved: {save_path}")

        processing_path = save_path
        final_path = save_path

        # Handle cropping if enabled
        if hasattr(self, 'allowCrop') and self.allowCrop:
            try:
                crop_dialog = CropDialog(save_path, self)
                if crop_dialog.exec_() == QDialog.Accepted:
                    processing_path = crop_dialog.get_cropped_path()
                    if processing_path:
                        print(f"Cropped image saved: {processing_path}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to crop image: {str(e)}")
                return

        # Handle upscaling based on selected option
        selected_upscale = self.upscale_combo.currentText()
        if selected_upscale != "No Upscaling":
            try:
                # Parse method and scale
                method, scale = selected_upscale.split(" ")
                scale = int(scale.strip("()x"))

                # Read the image to upscale
                img = cv2.imread(processing_path)
                if img is None:
                    raise Exception("Failed to load image for upscaling")

                # Create progress dialog
                progress = QProgressDialog(
                    "Upscaling image...", "Cancel", 0, 100, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setAutoClose(True)
                progress.setAutoReset(True)

                # Create and configure upscale thread
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                self.upscale_thread = UpscaleThread(img, method, scale, device)

                # Connect signals
                self.upscale_thread.progress.connect(progress.setValue)
                self.upscale_thread.finished.connect(lambda result: self.handle_upscale_finished(
                    result, processing_path, method, scale, progress))
                self.upscale_thread.error.connect(
                    lambda err: self.handle_upscale_error(err, progress))

                # Connect cancel button
                progress.canceled.connect(self.upscale_thread.cancel)

                # Start thread
                self.upscale_thread.start()

                # Show progress dialog
                progress.exec_()
                return

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to upscale image: {str(e)}")
                return

        # Show success message with all saved paths
        saved_paths = [p for p in [save_path, processing_path, final_path]
                       if p and os.path.exists(p) and p != save_path]

        if saved_paths:
            message = "Files saved:\n" + "\n".join(saved_paths)
        else:
            message = f"Screenshot saved: {save_path}"

        QMessageBox.information(self, "Success", message)

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
