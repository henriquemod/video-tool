from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QMessageBox,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import yt_dlp

class DownloadThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, url, data_dir):
        super().__init__()
        self.url = url
        self.data_dir = data_dir
        
    def progress_hook(self, d):
        if d['status'] == 'downloading':
            p = d.get('downloaded_bytes', 0) / d.get('total_bytes', 1) * 100
            self.progress.emit(int(p))
            
    def run(self):
        try:
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(self.data_dir, '%(title)s.%(ext)s'),
                'progress_hooks': [self.progress_hook],
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            self.finished.emit(True, "Video downloaded successfully!")
        except Exception as e:
            self.finished.emit(False, str(e))

class DownloadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Video")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel("Provider:")
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("YouTube")
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self.provider_combo)
        layout.addLayout(provider_layout)
        
        # URL input
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self.url_input = QLineEdit()
        url_layout.addWidget(url_label)
        url_layout.addWidget(self.url_input)
        layout.addLayout(url_layout)
        
        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
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
        self.cancel_button.clicked.connect(self.handle_cancel)
        self.download_button.clicked.connect(self.download_video)
        
    def handle_cancel(self):
        if hasattr(self, 'download_thread') and self.download_thread.isRunning():
            self.download_thread.terminate()
            self.download_thread.wait()
            self.progress_bar.hide()
            self.download_button.setEnabled(True)
        self.reject()
        
    def download_video(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a valid URL")
            return
            
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        
        # Show progress bar and disable download button
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.download_button.setEnabled(False)
        
        # Create and configure download thread
        self.download_thread = DownloadThread(url, data_dir)
        self.download_thread.progress.connect(self.progress_bar.setValue)
        self.download_thread.finished.connect(self.download_finished)
        
        # Start download
        self.download_thread.start()
        
    def download_finished(self, success, message):
        self.progress_bar.hide()
        self.download_button.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to download video: {message}") 