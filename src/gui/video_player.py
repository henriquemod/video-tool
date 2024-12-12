import cv2
import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QHBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QVideoProbe
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QDir
from PyQt5.QtGui import QImage

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize outputFolder with a default path
        self.outputFolder = QDir.currentPath()

        # Set up the video player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        self.videoProbe = QVideoProbe()

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
        self.playButton = QPushButton("Play")
        self.playButton.clicked.connect(self.toggle_playback)
        layout.addWidget(self.playButton)

        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stop_video)
        layout.addWidget(self.stopButton)

        self.setLayout(layout)

        # Connect media player signals
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def process_video_frame(self, frame):
        """Process the video frame received from QVideoProbe."""
        image = frame.image()
        if not image.isNull():
            # Optionally, store or display the frame as needed
            self.latestFrame = image


    def load_video(self, file_path):
        """Load a video file."""
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("Failed to open video with OpenCV.")
        else:
            print("Video loaded successfully with OpenCV.")
        

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

    def ms_to_time(self, ms):
        """Convert milliseconds to mm:ss format."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes):02}:{int(seconds):02}"

    def capture_screenshot(self):
        """Capture the current frame and save it as an image using OpenCV."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            position_ms = self.mediaPlayer.position()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_number = int((position_ms / 1000.0) * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR (OpenCV format) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Save using QImage
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                timestamp = self.ms_to_time(position_ms).replace(":", "-")
                output_path = f"{self.outputFolder}/screenshot_{timestamp}.png"
                if q_img.save(output_path):
                    print(f"Screenshot captured and saved to: {output_path}")
                else:
                    print("Failed to save the screenshot.")
            else:
                print("Failed to read the frame from OpenCV.")
        else:
            print("Video is not loaded with OpenCV. Cannot capture screenshot.")