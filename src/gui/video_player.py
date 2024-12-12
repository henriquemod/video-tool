import cv2
import sys
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QSlider,
    QHBoxLayout, QFileDialog, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir, pyqtSlot
from PyQt5.QtGui import QImage

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
        self.screenshotButton.clicked.connect(self.capture_screenshot)
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

        # Fullscreen Button
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

    def capture_screenshot(self):
        """Capture the current frame and save it as an image using OpenCV."""
        if self.cap and self.cap.isOpened():
            position_ms = self.mediaPlayer.position()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                QMessageBox.warning(self, "Error", "Cannot retrieve FPS from video.")
                return
            frame_number = int((position_ms / 1000.0) * fps)
            # Ensure frame_number is within the video length
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number >= total_frames:
                frame_number = total_frames - 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR (OpenCV format) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to QImage
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                
                # Open file dialog to choose save location and format
                options = QFileDialog.Options()
                save_path, selected_filter = QFileDialog.getSaveFileName(
                    self, "Save Screenshot",
                    f"screenshot_{self.ms_to_time(position_ms).replace(':', '-')}",
                    "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;BMP Files (*.bmp);;All Files (*)",
                    options=options
                )
                if save_path:
                    if q_img.save(save_path):
                        QMessageBox.information(self, "Success", f"Screenshot saved to:\n{save_path}")
                    else:
                        QMessageBox.warning(self, "Save Failed", "Failed to save the screenshot.")
            else:
                QMessageBox.warning(self, "Error", "Failed to read the frame from OpenCV.")
        else:
            QMessageBox.warning(self, "Error", "Video is not loaded with OpenCV. Cannot capture screenshot.")

    def closeEvent(self, event):
        """Handle the widget closing."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()
