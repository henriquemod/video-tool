import cv2
import sys
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox, QDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from .crop_dialog import CropDialog

class VideoPlayer(QWidget):
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

        # Load Video Button
        self.loadButton = QPushButton("Load Video")
        self.loadButton.clicked.connect(self.open_file)
        controlsLayout.addWidget(self.loadButton)

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

    def open_file(self):
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", QDir.homePath(),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path):
        """Load a video file and initialize OpenCV capture."""
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video with OpenCV.")
        else:
            QMessageBox.information(self, "Success", "Video loaded successfully with OpenCV.")
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
        """Capture the current frame as a screenshot using OpenCV and optionally crop it."""
        if self.mediaPlayer.state() != QMediaPlayer.PlayingState:
            return
        
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
        
        # Save the screenshot using OpenCV
        cv2.imwrite(save_path, frame)
        print(f"Screenshot saved: {save_path}")
        
        # If crop is allowed, open the crop dialog with the saved image
        if hasattr(self, 'allowCrop') and self.allowCrop:
            try:
                # Create and show crop dialog with the saved image
                crop_dialog = CropDialog(save_path, self)
                if crop_dialog.exec_() == QDialog.Accepted:
                    # Get the cropped image path (CropDialog will handle saving)
                    cropped_path = crop_dialog.get_cropped_path()
                    if cropped_path:
                        print(f"Cropped image saved: {cropped_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to crop image: {str(e)}")

    def closeEvent(self, event):
        """Handle the widget closing."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()
