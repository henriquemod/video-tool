"""MainWindow - Primary Application Window for Multimedia Assistant

This module implements the main application window that serves as the central hub for all multimedia
operations and user interactions. It integrates various components including video playback,
file management, and media processing features.
"""
import os
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
from PyQt5.QtCore import QStandardPaths

from .widgets.video_player import VideoPlayer
from .dialogs.download_dialog import DownloadDialog
from .dialogs.upscale_dialog import UpscaleDialog
from .dialogs.resize_dialog import ResizeDialog
from ..utils.icon_utils import generateIcon


class MainWindow(QMainWindow):
    """
    MainWindow - Primary Application Window for Multimedia Assistant

    This class sets up the main window, including the video player, file browser, and various
    dialogs for downloading, upscaling, and resizing media files.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Assistant")
        self.resize(1200, 800)

        # Initialize current_browser_path
        self.current_browser_path = os.path.join(
            QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        )

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create video player widget
        self.video_player = VideoPlayer()

        # Set the output folder directly
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(__file__))), 'output'
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.video_player.outputFolder = output_dir

        # Create a container widget for file browser and buttons
        file_browser_container = QWidget()
        file_browser_layout = QVBoxLayout(file_browser_container)

        # Create and setup file browser
        self.setup_file_browser()

        # Add file browser
        file_browser_layout.addWidget(self.file_browser)

        # Add Load from folder button with 'folder-open' icon
        load_folder_button = QPushButton(" Load from folder")
        load_folder_button.setIcon(generateIcon("folder-open", True))
        load_folder_button.setStyleSheet(
            "text-align: left; padding-left: 15px; width: 100%; height: 36px;"
        )
        load_folder_button.clicked.connect(self.change_browser_folder)
        file_browser_layout.addWidget(load_folder_button)

        # Add download button with 'document-save' icon
        download_button = QPushButton(" Download video")
        download_button.setIcon(generateIcon("document-save", True))
        download_button.setStyleSheet(
            "text-align: left; padding-left: 15px; width: 100%; height: 36px;"
        )
        download_button.clicked.connect(self.show_download_dialog)
        file_browser_layout.addWidget(download_button)

        # Add the upscale batch button with 'zoom-fit-best' icon
        upscale_button = QPushButton(" Upscale batch")
        upscale_button.setIcon(generateIcon("zoom-fit-best", True))
        upscale_button.setStyleSheet(
            "text-align: left; padding-left: 15px; width: 100%; height: 36px;"
        )
        upscale_button.clicked.connect(self.show_upscale_dialog)
        file_browser_layout.addWidget(upscale_button)

        # Add the resize batch button with 'view-fullscreen' icon
        resize_button = QPushButton(" Resize batch")
        resize_button.setIcon(generateIcon("view-fullscreen", True))
        resize_button.setStyleSheet(
            "text-align: left; padding-left: 15px; width: 100%; height: 36px;"
        )
        resize_button.clicked.connect(self.show_resize_dialog)
        file_browser_layout.addWidget(resize_button)

        # Add widgets to main layout
        # Smaller width for file browser
        main_layout.addWidget(file_browser_container, 1)
        # Larger width for video player
        main_layout.addWidget(self.video_player, 4)

        # Set up the menu bar
        self.setup_menu_bar()

    def setup_file_browser(self):
        """Set up the file browser with a filtered view for video files."""
        # Create file system model
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.current_browser_path)

        # Set filters to show only video files
        self.file_model.setNameFilters(["*.mp4", "*.avi", "*.mkv", "*.mov"])
        self.file_model.setNameFilterDisables(False)

        # Create tree view
        self.file_browser = QTreeView()
        self.file_browser.setModel(self.file_model)
        self.file_browser.setRootIndex(
            self.file_model.index(self.current_browser_path))

        # Hide unnecessary columns
        self.file_browser.setColumnHidden(1, True)  # Size
        self.file_browser.setColumnHidden(2, True)  # Type
        self.file_browser.setColumnHidden(3, True)  # Date Modified

        # Connect double-click signal
        self.file_browser.doubleClicked.connect(self.on_file_double_clicked)

    def on_file_double_clicked(self, index):
        """Handle double-click events on the file browser to load videos."""
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
        """Open file dialog and load selected video."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            self.video_player.get_data_directory(),
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)",
        )
        if file_path:
            self.video_player.load_video(file_path)

    def show_download_dialog(self):
        """Show the download dialog when download button is clicked."""
        dialog = DownloadDialog(self)
        dialog.exec_()

    def show_upscale_dialog(self):
        """Show the upscale dialog when upscale batch button is clicked."""
        dialog = UpscaleDialog(self)
        dialog.exec_()

    def show_resize_dialog(self):
        """Show the resize dialog when resize batch button is clicked."""
        dialog = ResizeDialog(self)
        dialog.exec_()

    def change_browser_folder(self):
        """Allow user to select a new folder to browse."""
        new_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder",
            self.current_browser_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )

        if new_folder:
            self.current_browser_path = new_folder
            self.file_browser.setRootIndex(self.file_model.index(new_folder))
