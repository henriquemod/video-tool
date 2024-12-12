from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

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

        # Add playback controls
        self.playButton = QPushButton("Play")
        self.playButton.clicked.connect(self.toggle_playback)
        layout.addWidget(self.playButton)

        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stop_video)
        layout.addWidget(self.stopButton)

        self.setLayout(layout)

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