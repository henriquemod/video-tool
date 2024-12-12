from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QAction,
)
from .video_player import VideoPlayer
from .settings import SettingsDialog
from PyQt5.QtCore import QDir

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

        central_widget.setLayout(layout)

        # Set up the menu bar
        self.setup_menu_bar()

    def setup_menu_bar(self):
        """Create and configure the menu bar with 'App' and 'File' menus."""
        menu_bar = self.menuBar()

        # 'App' Menu
        app_menu = menu_bar.addMenu("App")

        # Exit Action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        app_menu.addAction(exit_action)

        # 'File' Menu
        file_menu = menu_bar.addMenu("File")

        # Open Video Action
        open_video_action = QAction("Open Video", self)
        open_video_action.triggered.connect(self.load_video)
        file_menu.addAction(open_video_action)

        # Add settings action
        settingsAction = QAction("Settings", self)
        settingsAction.triggered.connect(self.open_settings)
        file_menu.addAction(settingsAction)

    def load_video(self):
        """Open a file dialog to select a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_path:
            self.videoPlayer.load_video(file_path) 

    def open_settings(self):
        """Open the settings dialog."""
        settingsDialog = SettingsDialog(self)
        if settingsDialog.exec_():
            # Retrieve settings here
            self.videoPlayer.allowCrop = settingsDialog.cropCheckbox.isChecked()
            output_folder = settingsDialog.outputFolderLabel.text().replace("Output Folder: ", "")
            self.videoPlayer.outputFolder = output_folder if output_folder else QDir.currentPath()
            print(f"Settings saved: Crop Images - {self.videoPlayer.allowCrop}, Output Folder - {self.videoPlayer.outputFolder}")