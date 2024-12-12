from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from .video_player import VideoPlayer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Assistant")
        self.setGeometry(100, 100, 800, 600)

        # Set up the central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout()

        # Video Player
        self.videoPlayer = VideoPlayer()
        layout.addWidget(self.videoPlayer)

        # Load Video Button
        loadButton = QPushButton("Load Video")
        loadButton.clicked.connect(self.load_video)
        layout.addWidget(loadButton)

        central_widget.setLayout(layout)

    def load_video(self):
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.videoPlayer.load_video(file_path) 