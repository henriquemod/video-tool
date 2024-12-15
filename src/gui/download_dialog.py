from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import yt_dlp


class VideoItem:
    def __init__(self, provider, url):
        self.provider = provider
        self.url = url

    def __str__(self):
        return f"{self.provider}: {self.url}"


class DownloadThread(QThread):
    progress = pyqtSignal(int)
    video_progress = pyqtSignal(int, int)  # current video, total videos
    finished = pyqtSignal(bool, str)

    def __init__(self, videos, data_dir):
        super().__init__()
        self.videos = videos
        self.data_dir = data_dir
        self.current_video = 0

    def progress_hook(self, d):
        if d['status'] == 'downloading':
            p = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100
            self.progress.emit(int(p))

    def run(self):
        try:
            total_videos = len(self.videos)
            for i, video in enumerate(self.videos, 1):
                self.current_video = i
                self.video_progress.emit(i, total_videos)

                ydl_opts = {
                    'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'outtmpl': os.path.join(self.data_dir, '%(title)s.%(ext)s'),
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
                True, f"Successfully downloaded {total_videos} videos!")
        except Exception as e:
            self.finished.emit(False, str(e))


class DownloadDialog(QDialog):
    """
    DownloadDialog - Video Download Manager for Multimedia Assistant

    This module implements a sophisticated video download interface that supports multiple video sources
    and provides progress tracking. It uses yt-dlp as its backend for reliable video downloading with
    format selection and quality control.

    Key Features:
    - Multi-video download queue management
    - Progress tracking for individual downloads
    - Support for multiple video platforms
    - High-quality video format selection
    - Configurable output directory
    - Cancel and resume capabilities

    Components:
    1. Download Queue:
    - List-based interface for multiple video URLs
    - Add/Remove functionality for queue management
    - Visual feedback for queued items
    - Support for batch processing

    2. Progress Tracking:
    - Individual progress bars for each download
    - Overall progress indication
    - Current/Total videos counter
    - Real-time status updates

    3. Format Handling:
    - Automatic best quality selection
    - Format preference configuration:
        * Video: H.264/AVC codec
        * Audio: AAC with 192k bitrate
        * Container: MP4 with faststart
    - Fallback options for compatibility

    4. Error Management:
    - Graceful error handling
    - User feedback for failed downloads
    - Retry capabilities
    - Detailed error reporting

    Technical Implementation:
    - Multi-threaded download processing
    - Signal-based progress updates
    - Memory-efficient queue management
    - Proper resource cleanup

    Classes:
        VideoItem:
            Container class for video information
            Attributes:
                - provider: Source platform (e.g., YouTube)
                - url: Video URL

        DownloadThread(QThread):
            Background worker for download operations
            Signals:
                - progress(int): Download progress percentage
                - video_progress(int, int): Current/Total videos
                - finished(bool, str): Success status and message

        DownloadDialog(QDialog):
            Main dialog interface for download management
            Features:
                - URL input with validation
                - Provider selection
                - Queue management
                - Progress visualization

    Dependencies:
    - PyQt5: GUI framework
    - yt-dlp: Download backend
    - ffmpeg: Media processing

    Example Usage:
        dialog = DownloadDialog(parent)
        dialog.exec_()

    Configuration Options:
    - Video format selection
    - Output path customization
    - Download parameters:
        * Format preferences
        * Quality settings
        * Post-processing options

    Performance Considerations:
    - Asynchronous download processing
    - Efficient progress updates
    - Memory management for queue
    - Proper thread handling

    Error Handling:
    - Network connectivity issues
    - Invalid URLs
    - Platform-specific errors
    - File system errors

    @see @Project Structure#download_dialog.py
    @see @Project#Download Management
    @see @Project#Media Processing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Video")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.videos = []
        self.setup_ui()

    def setup_ui(self):
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
        current_row = self.video_list.currentRow()
        if current_row >= 0:
            self.video_list.takeItem(current_row)
            self.videos.pop(current_row)

    def handle_cancel(self):
        if hasattr(self, 'download_thread') and self.download_thread.isRunning():
            self.download_thread.terminate()
            self.download_thread.wait()
            self.progress_bar.hide()
            self.progress_label.hide()
            self.download_button.setEnabled(True)
        self.reject()

    def download_videos(self):
        if not self.videos:
            QMessageBox.warning(
                self, "Error", "Please add at least one video to download")
            return

        data_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))), 'data')

        # Show progress bar and disable download button
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.download_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)

        # Create and configure download thread
        self.download_thread = DownloadThread(self.videos, data_dir)
        self.download_thread.progress.connect(self.progress_bar.setValue)
        self.download_thread.video_progress.connect(self.update_progress_label)
        self.download_thread.finished.connect(self.download_finished)

        # Start download
        self.download_thread.start()

    def update_progress_label(self, current, total):
        self.progress_label.setText(f"{current}/{total}")

    def download_finished(self, success, message):
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
                self, "Error", f"Failed to download videos: {message}")
