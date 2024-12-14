from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QAction,
    QHBoxLayout,
    QTreeView,
    QFileSystemModel,
    QPushButton,
    QVBoxLayout,
)
from .video_player import VideoPlayer
from .download_dialog import DownloadDialog
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

        # Create a container widget for file browser and download button
        file_browser_container = QWidget()
        file_browser_layout = QVBoxLayout(file_browser_container)
        
        # Create and setup file browser
        self.setup_file_browser()

        # Add file browser
        file_browser_layout.addWidget(self.file_browser)
        
        # Create and add download button
        download_button = QPushButton("Download video")
        download_button.clicked.connect(self.show_download_dialog)
        file_browser_layout.addWidget(download_button)
        
        # Create video player widget
        self.video_player = VideoPlayer()

        # Add crop checkbox to the video player's control layout
        self.video_player.add_crop_checkbox()
        
        # Set the output folder directly
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.video_player.outputFolder = output_dir

        # Add widgets to main layout
        main_layout.addWidget(file_browser_container, 1)  # Smaller width for file browser
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

    def load_video(self):
        """Open file dialog and load selected video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            self.get_data_directory(),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.video_player.load_video(file_path)

    def show_download_dialog(self):
        """Show the download dialog when download button is clicked"""
        dialog = DownloadDialog(self)
        dialog.exec_()