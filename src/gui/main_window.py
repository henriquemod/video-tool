from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QAction,
    QHBoxLayout,
    QTreeView,
    QFileSystemModel,
)
from .video_player import VideoPlayer
from .settings import SettingsDialog
from PyQt5.QtCore import QDir, Qt
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Assistant")
        self.resize(1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create and setup file browser
        self.setup_file_browser()

        # Create video player widget
        self.video_player = VideoPlayer()

        # Add widgets to main layout
        main_layout.addWidget(self.file_browser, 1)  # Smaller width for file browser
        main_layout.addWidget(self.video_player, 4)  # Larger width for video player

        # Set up the menu bar
        self.setup_menu_bar()

    def setup_file_browser(self):
        # Create file system model
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.get_data_directory())

        # Set filters to show only video files
        self.file_model.setNameFilters(["*.mp4", "*.avi", "*.mkv", "*.mov"])
        self.file_model.setNameFilterDisables(False)

        # Create tree view
        self.file_browser = QTreeView()
        self.file_browser.setModel(self.file_model)
        self.file_browser.setRootIndex(self.file_model.index(self.get_data_directory()))

        # Hide unnecessary columns
        self.file_browser.setColumnHidden(1, True)  # Size
        self.file_browser.setColumnHidden(2, True)  # Type
        self.file_browser.setColumnHidden(3, True)  # Date Modified

        # Connect double-click signal
        self.file_browser.doubleClicked.connect(self.on_file_double_clicked)

    def get_data_directory(self):
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_dir = os.path.join(current_dir, 'data')

        # Create the data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        return data_dir

    def on_file_double_clicked(self, index):
        # Get the full path of the selected file
        file_path = self.file_model.filePath(index)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            # Load the video in the video player
            self.video_player.load_video(file_path)

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
            self.video_player.load_video(file_path) 

    def open_settings(self):
        """Open the settings dialog."""
        settingsDialog = SettingsDialog(self)
        if settingsDialog.exec_():
            # Retrieve settings here
            self.video_player.allowCrop = settingsDialog.cropCheckbox.isChecked()
            output_folder = settingsDialog.outputFolderLabel.text().replace("Output Folder: ", "")
            self.video_player.outputFolder = output_folder if output_folder else QDir.currentPath()
            print(f"Settings saved: Crop Images - {self.video_player.allowCrop}, Output Folder - {self.video_player.outputFolder}")