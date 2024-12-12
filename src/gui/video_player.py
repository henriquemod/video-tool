import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QSlider, QHBoxLayout
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the video player
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)

        # Set up the media player
        self.mediaPlayer.setVideoOutput(videoWidget)

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

    def load_video(self, file_path):
        """Load a video file."""
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))

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
 