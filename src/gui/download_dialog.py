"""
Download Dialog Module for Multimedia Assistant

This module provides a sophisticated video download interface that supports multiple video sources
and offers progress tracking. It utilizes yt-dlp as the backend for reliable video downloading
with format selection and quality control.
"""

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, QStandardPaths


import yt_dlp


class VideoItem:
    """
    Container class for video information.

    Attributes:
        provider (str): Source platform (e.g., YouTube).
        url (str): Video URL.
    """

    def __init__(self, provider, url):
        """
        Initialize a VideoItem instance.

        Args:
            provider (str): The video provider.
            url (str): The URL of the video.
        """
        self.provider = provider
        self.url = url

    def __str__(self):
        """
        Return a string representation of the VideoItem.

        Returns:
            str: Formatted string with provider and URL.
        """
        return f"{self.provider}: {self.url}"


class DownloadThread(QThread):
    """
    Background worker for download operations.

    Signals:
        progress(int): Download progress percentage.
        video_progress(int, int): Current video index and total videos.
        finished(bool, str): Success status and message.
    """

    progress = pyqtSignal(int)
    video_progress = pyqtSignal(int, int)  # current video, total videos
    finished = pyqtSignal(bool, str)

    def __init__(self, videos, output_dir):
        """
        Initialize the DownloadThread.

        Args:
            videos (list): List of VideoItem instances to download.
            output_dir (str): Directory where videos will be saved.
        """
        super().__init__()
        self.videos = videos
        self.output_dir = output_dir
        self.current_video = 0

    def progress_hook(self, d):
        """
        Hook to report download progress.

        Args:
            d (dict): Dictionary containing download status and bytes information.
        """
        if d.get('status') == 'downloading':
            downloaded = d.get('downloaded_bytes', 0)
            total = d.get('total_bytes', 1)
            p = downloaded / total * 100
            self.progress.emit(int(p))

    def run(self):
        """
        Execute the download process for each video.
        """
        try:
            total_videos = len(self.videos)
            for i, video in enumerate(self.videos, 1):
                self.current_video = i
                self.video_progress.emit(i, total_videos)

                ydl_opts = {
                    'format': (
                        'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/'
                        'best[ext=mp4]/best'
                    ),
                    'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                    'progress_hooks': [self.progress_hook],
                    'merge_output_format': 'mp4',
                    'postprocessor_args': [
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-movflags', '+faststart'
                    ],
                    'verbose': True,
                    'prefer_ffmpeg': True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video.url])

            self.finished.emit(
                True, f"Successfully downloaded {total_videos} videos!"
            )
        except yt_dlp.utils.DownloadError as e:
            self.finished.emit(False, str(e))
        except (OSError, IOError) as e:
            # Handle file system related errors
            self.finished.emit(False, f"File system error: {str(e)}")
        except ValueError as e:
            # Handle value-related errors
            self.finished.emit(False, f"Value error: {str(e)}")
        except RuntimeError as e:
            # Handle runtime-specific errors
            self.finished.emit(False, f"Runtime error: {str(e)}")


class DownloadDialog(QDialog):
    """
    Video Download Manager for Multimedia Assistant.

    This dialog interface supports adding multiple video URLs, selecting providers,
    managing a download queue, and tracking download progress.
    """

    def __init__(self, parent=None):
        """
        Initialize the DownloadDialog.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Download Video")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.videos = []
        self.download_thread = None
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the user interface components.
        """
        layout = QVBoxLayout()

        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("YouTube")
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)

        # URL input
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)

        # Add button
        self.add_button = QPushButton("Add")
        url_layout.addWidget(self.add_button)

        layout.addLayout(provider_layout)
        layout.addLayout(url_layout)

        # Video list
        self.video_list = QListWidget()
        layout.addWidget(self.video_list)

        # Remove button
        self.remove_button = QPushButton("Remove Selected")
        layout.addWidget(self.remove_button)

        # Progress information
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_label = QLabel("0/0")
        self.progress_bar.hide()
        self.progress_label.hide()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.download_button = QPushButton("Download")
        self.download_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.download_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
        self.add_button.clicked.connect(self.add_video)
        self.remove_button.clicked.connect(self.remove_video)
        self.cancel_button.clicked.connect(self.handle_cancel)
        self.download_button.clicked.connect(self.download_videos)

    def add_video(self):
        """
        Add a video to the download queue.
        """
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a valid URL")
            return

        provider = self.provider_combo.currentText()
        video = VideoItem(provider, url)
        self.videos.append(video)

        self.video_list.addItem(QListWidgetItem(str(video)))
        self.url_input.clear()

    def remove_video(self):
        """
        Remove the selected video from the download queue.
        """
        current_row = self.video_list.currentRow()
        if current_row >= 0:
            self.video_list.takeItem(current_row)
            self.videos.pop(current_row)

    def handle_cancel(self):
        """
        Handle the cancellation of the download process.
        """
        if self.download_thread and self.download_thread.isRunning():
            self.download_thread.terminate()
            self.download_thread.wait()
            self.progress_bar.hide()
            self.progress_label.hide()
            self.download_button.setEnabled(True)
        self.reject()

    def download_videos(self):
        """
        Initiate the download process for all queued videos.
        """
        if not self.videos:
            QMessageBox.warning(
                self, "Error", "Please add at least one video to download"
            )
            return

        # Ask user for save location
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Download Location",
            QStandardPaths.writableLocation(QStandardPaths.MoviesLocation),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not output_dir:  # User cancelled
            return

        # Show progress bar and disable buttons
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.download_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)

        # Create and start download thread with user-selected directory
        self.download_thread = DownloadThread(self.videos, output_dir)
        self.download_thread.progress.connect(self.progress_bar.setValue)
        self.download_thread.video_progress.connect(self.update_progress_label)
        self.download_thread.finished.connect(self.download_finished)

        # Start download
        self.download_thread.start()

    def update_progress_label(self, current, total):
        """
        Update the progress label with the current download status.

        Args:
            current (int): The index of the current video being downloaded.
            total (int): The total number of videos to download.
        """
        self.progress_label.setText(f"{current}/{total}")

    def download_finished(self, success, message):
        """
        Handle the completion of the download process.

        Args:
            success (bool): Indicates if the download was successful.
            message (str): Success or error message.
        """
        self.progress_bar.hide()
        self.progress_label.hide()
        self.download_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.remove_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(
                self, "Error", f"Failed to download videos: {message}"
            )
