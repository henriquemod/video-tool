import cv2
import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog, QCheckBox, QComboBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir
from PyQt5.QtWidgets import QApplication
from .crop_dialog import CropDialog
from .upscale_dialog import UpscaleDialog

# Add these new imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

"""
VideoPlayer Widget for Multimedia Assistant Application

This module implements a comprehensive video player widget using PyQt5 and OpenCV, providing
professional-grade video playback capabilities with synchronized audio support.

Key Features:
- Video playback with support for common formats (MP4, AVI, MKV, MOV)
- Advanced playback controls (play, pause, stop, seek)
- Volume and playback speed adjustment
- High-quality screenshot capture with optional cropping
- Fullscreen support
- Cross-platform compatibility (Windows, macOS, Linux)

Technical Implementation:
- Uses QMediaPlayer for smooth video/audio playback
- Integrates OpenCV for frame capture and processing
- Implements responsive GUI with custom styling for different platforms
- Employs efficient memory management for large video files

Dependencies:
- PyQt5: GUI framework and media playback
- OpenCV (cv2): Frame capture and image processing
- Standard libraries: os, sys

Example usage:
    player = VideoPlayer()
    player.load_video("path/to/video.mp4")
    player.show()

@anchor #video-player-implementation
@see @Project Structure#video_player.py
"""

# Model URLs for SwinIR
MODEL_URLS = {
    'SwinIR-2x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
    'SwinIR-4x': 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
    'ESRGAN-2x': 'https://github.com/xinntao/ESRGAN/releases/download/v0.4.0/ESRGAN_x2.pth',
    'ESRGAN-4x': 'https://github.com/xinntao/ESRGAN/releases/download/v0.4.0/ESRGAN_x4.pth'
}

class VideoPlayer(QWidget):
    """
    A feature-rich video player widget implementing professional multimedia playback capabilities.
    
    This widget provides a complete video playback interface with:
    - Video preview screen
    - Playback controls (play/pause, stop, seek)
    - Volume and playback speed controls
    - Screenshot functionality with optional cropping
    - Fullscreen support
    
    Attributes:
        outputFolder (str): Directory for saving screenshots
        mediaPlayer (QMediaPlayer): Core media playback engine
        videoWidget (QVideoWidget): Widget for video display
        cap (cv2.VideoCapture): OpenCV video capture object
        positionSlider (QSlider): Seek control for video navigation
        volumeSlider (QSlider): Volume control slider
        speedSlider (QSlider): Playback speed control
    
    @anchor #video-player-class
    """

    def __init__(self):
        super().__init__()

        # Initialize outputFolder with a default path
        self.outputFolder = QDir.currentPath()

        # Set up the video player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

        # Initialize OpenCV VideoCapture
        self.cap = None

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)

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

        layout.addLayout(sliderLayout)

        # Add playback controls
        controlsLayout = QHBoxLayout()

        # Play/Pause Button
        self.playButton = QPushButton("Play")
        self.playButton.clicked.connect(self.toggle_playback)
        controlsLayout.addWidget(self.playButton)

        # Stop Button
        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stop_video)
        controlsLayout.addWidget(self.stopButton)

        # Volume Control
        self.volumeSlider = QSlider(Qt.Horizontal)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setValue(50)  # Default volume
        self.volumeSlider.setFixedWidth(100)
        self.volumeSlider.valueChanged.connect(self.mediaPlayer.setVolume)
        controlsLayout.addWidget(QLabel("Volume"))
        controlsLayout.addWidget(self.volumeSlider)

        # Playback Speed Control
        self.speedSlider = QSlider(Qt.Horizontal)
        self.speedSlider.setRange(50, 200)  # 0.5x to 2.0x
        self.speedSlider.setValue(100)  # Default speed (1.0x)
        self.speedSlider.setFixedWidth(100)
        self.speedSlider.valueChanged.connect(self.set_playback_speed)
        controlsLayout.addWidget(QLabel("Speed"))
        controlsLayout.addWidget(self.speedSlider)

        # FullScreen Button
        self.fullscreenButton = QPushButton("Fullscreen")
        self.fullscreenButton.clicked.connect(self.toggle_fullscreen)
        controlsLayout.addWidget(self.fullscreenButton)

        layout.addLayout(controlsLayout)

        self.setLayout(layout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)
        self.mediaPlayer.stateChanged.connect(self.media_state_changed)

        # Initialize crop checkbox as False by default
        self.allowCrop = False

        # Check AI capabilities before adding upscale options
        self.ai_capable = self.check_ai_capabilities()
        self.add_upscale_checkbox()

    def open_file(self):
        """
        Opens a file dialog for video selection and loads the selected video.
        
        Supports common video formats (MP4, AVI, MKV, MOV) and handles file loading errors.
        
        @anchor #video-file-loading
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", QDir.homePath(),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        """
        Initializes video playback for the specified file path.
        
        Args:
            file_path (str): Path to the video file
            
        Handles both QMediaPlayer and OpenCV initialization for synchronized
        playback and frame capture capabilities.
        
        @anchor #video-loading-implementation
        """
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video with OpenCV.")
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
            QMessageBox.critical(self, "Error", "Failed to capture frame from video.")
            return
        
        # Generate filename with timestamp
        filename = f"screenshot_{current_position}.png"
        
        # Use outputFolder if defined, otherwise use root project path
        save_path = os.path.join(getattr(self, 'outputFolder', os.path.dirname(os.path.abspath(__file__))), filename)
        
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
                QMessageBox.critical(self, "Error", f"Failed to crop image: {str(e)}")
                return
        
        # Handle upscaling based on selected option
        selected_upscale = self.upscale_combo.currentText()
        if selected_upscale != "No Upscaling":
            try:
                # Parse method and scale from selection
                method, scale = selected_upscale.split(" ")
                scale = int(scale.strip("()x"))
                
                # Read the image to upscale (either original or cropped)
                img = cv2.imread(processing_path)
                if img is None:
                    raise Exception("Failed to load image for upscaling")
                
                # Get dimensions for resizing
                height, width = img.shape[:2]
                new_height = int(height * scale)
                new_width = int(width * scale)
                
                # Handle different upscaling methods
                if any(ai_method in method for ai_method in ["Real-ESRGAN", "ESRGAN", "SwinIR"]):
                    upscaled = self.upscale_with_ai(img, method, scale)
                elif method == "Bicubic":
                    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                elif method == "Lanczos":
                    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
                elif method == "Nearest":
                    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                elif method == "Area":
                    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                else:
                    # Default to bilinear interpolation
                    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Generate upscaled filename
                base_path = os.path.splitext(processing_path)[0]
                final_path = f"{base_path}_upscaled_{method}_{scale}x.png"
                
                # Save upscaled image
                cv2.imwrite(final_path, upscaled)
                print(f"Upscaled image saved: {final_path}")
            
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to upscale image: {str(e)}")
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

    def add_upscale_checkbox(self):
        """Add upscale options to controls."""
        # Create layout for upscale controls
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
            "ESRGAN (2x)",  # New advanced AI method
            "ESRGAN (4x)",  # New advanced AI method
            "SwinIR (2x)",  # New advanced AI method
            "SwinIR (4x)",  # New advanced AI method
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
        has_gpu = self.check_gpu_capabilities()
        
        # Connect combo box change event
        self.upscale_combo.currentTextChanged.connect(
            lambda text: self.on_upscale_option_changed(text, has_gpu)
        )
        
        # Get the controls layout
        layout = self.layout()
        controlsLayout = layout.itemAt(2).layout()  # Get the controls layout
        
        # Add the upscale layout to controls
        controlsLayout.insertLayout(2, upscale_layout)

    def check_gpu_capabilities(self):
        """Check for GPU acceleration capabilities."""
        try:
            import torch
            
            # Check for CUDA (NVIDIA) or MPS (Apple Silicon) support
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if has_cuda:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                print(f"CUDA GPU detected with {gpu_memory:.2f}GB memory")
                return True
            elif has_mps:
                print("Apple Metal GPU detected")
                return True
            else:
                print("No compatible GPU detected, AI upscaling will use CPU")
                return False
            
        except ImportError:
            print("PyTorch not installed, AI upscaling will use CPU")
            return False
        except Exception as e:
            print(f"Error checking GPU capabilities: {str(e)}")
            return False

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

    def upscale_with_ai(self, img, method, scale):
        """Upscale image using AI models"""
        processing_dialog = None
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer

            # Option 1: Enable MPS fallback
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            # Option 2: Or use this device selection instead if you prefer CPU-only on Mac
            device = torch.device('cpu')
            if torch.cuda.is_available():  # Will be True for NVIDIA GPUs
                device = torch.device('cuda')
                if not hasattr(self, '_showed_gpu_info'):
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    print(f"Using GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
                    self._showed_gpu_info = True
            else:
                if not hasattr(self, '_showed_cpu_warning'):
                    QMessageBox.warning(self, "Performance Warning", 
                                      "Using CPU for AI upscaling. This will be slower than GPU acceleration.")
                    self._showed_cpu_warning = True

            # Show processing dialog
            processing_dialog = QMessageBox(self)
            processing_dialog.setIcon(QMessageBox.Information)
            processing_dialog.setText("Processing image with AI upscaling...")
            processing_dialog.setStandardButtons(QMessageBox.NoButton)
            processing_dialog.show()
            QApplication.processEvents()

            if "SwinIR" in method:
                # Import SwinIR components
                from basicsr.utils.registry import ARCH_REGISTRY
                
                # Load the appropriate model based on scale
                if scale == 2:
                    model_path = load_file_from_url(
                        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth',
                        model_dir='models'
                    )
                else:  # scale == 4
                    model_path = load_file_from_url(
                        'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth',
                        model_dir='models'
                    )

                # Initialize model with the exact lightweight configuration
                model = ARCH_REGISTRY.get('SwinIR')(
                    upscale=scale,
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

                # Load pretrained weights
                pretrained_model = torch.load(model_path, map_location=device)
                if 'params_ema' in pretrained_model:
                    pretrained_model = pretrained_model['params_ema']
                elif 'params' in pretrained_model:
                    pretrained_model = pretrained_model['params']
                    
                model.load_state_dict(pretrained_model, strict=True)
                model.eval()
                model.to(device)

                # Process the image
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.
                img_tensor = img_tensor.to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                
                output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = (output * 255.0).round().astype(np.uint8)
                output = output.transpose(1, 2, 0)
                
                if processing_dialog:
                    processing_dialog.close()
                
                return output

            elif "ESRGAN" in method and "Real-ESRGAN" not in method:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
                model_key = f'ESRGAN-{scale}x'
                model_path = load_file_from_url(MODEL_URLS[model_key], model_dir='models')
                loadnet = torch.load(model_path)
                model.load_state_dict(loadnet)
                model.eval()
                model.to(device)

            else:  # Real-ESRGAN
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
                
                if scale == 2:
                    model_path = load_file_from_url(
                        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                        model_dir='models'
                    )
                else:
                    model_path = load_file_from_url(
                        'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                        model_dir='models'
                    )

            # Initialize upscaler for non-SwinIR models
            if "SwinIR" not in method:
                tile_size = 512 if torch.cuda.is_available() else 256  # Smaller tiles for CPU
                upscaler = RealESRGANer(
                    scale=scale,
                    model_path=model_path,
                    model=model,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=torch.cuda.is_available(),
                    device=device
                )

                # Upscale image
                output, _ = upscaler.enhance(img, outscale=scale)
                
                if processing_dialog:
                    processing_dialog.close()
                
                return output

        except ImportError as e:
            error_message = str(e)
            if "numpy" in error_message.lower():
                error_message += "\n\nPlease run:\npip install numpy==1.24.3"
            elif "torch" in error_message.lower():
                error_message += "\n\nPlease run:\npip install torch==2.0.1 torchvision==0.15.2"
            elif "basicsr" in error_message.lower() or "realesrgan" in error_message.lower():
                error_message += "\n\nPlease run:\npip install basicsr realesrgan"
            
            QMessageBox.critical(self, "Error", f"Missing required package:\n{error_message}")
            raise
        except Exception as e:
            QMessageBox.critical(self, "Error", f"AI upscaling failed: {str(e)}")
            raise

    def check_ai_capabilities(self):
        """Check if system can run AI upscaling."""
        try:
            # First try to import numpy with specific version
            import numpy as np
            if np.__version__.startswith('2'):
                print("Warning: NumPy 2.x detected, downgrading may be required")
            
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Check if CUDA is available
            has_cuda = torch.cuda.is_available()
            
            # Check if MPS (Metal Performance Shaders for Mac) is available
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            # Check available GPU memory
            if has_cuda:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                print(f"CUDA GPU detected with {gpu_memory:.2f}GB memory")
            elif has_mps:
                print("Apple Metal GPU detected")
            else:
                print("No compatible GPU detected, AI upscaling will use CPU (slow)")
            
            return True
            
        except ImportError as e:
            print(f"AI capabilities not available: {str(e)}")
            print("To enable AI upscaling, install required packages with:")
            print("pip install numpy==1.24.3 torch==2.0.1 torchvision==0.15.2 basicsr realesrgan")
            return False
        except Exception as e:
            print(f"Error checking AI capabilities: {str(e)}")
            return False

    def add_crop_checkbox(self):
        """Add crop checkbox to controls layout"""
        # Create layout for crop controls
        crop_layout = QHBoxLayout()
        
        # Create and setup crop checkbox
        self.cropCheckbox = QCheckBox("Crop Image")
        self.allowCrop = False  # Initialize the crop flag
        self.cropCheckbox.setChecked(self.allowCrop)
        self.cropCheckbox.stateChanged.connect(self.toggle_crop)
        
        # Add checkbox to layout
        crop_layout.addWidget(self.cropCheckbox)
        
        # Get the controls layout
        layout = self.layout()
        controlsLayout = layout.itemAt(2).layout()  # Get the controls layout
        
        # Add the crop layout to controls
        controlsLayout.insertLayout(2, crop_layout)

    def toggle_crop(self, state):
        """Toggle crop functionality"""
        self.allowCrop = bool(state)
