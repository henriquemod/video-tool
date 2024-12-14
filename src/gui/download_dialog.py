from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QMessageBox
)
from PyQt5.QtCore import Qt
import os
import yt_dlp

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
        self.cancel_button.clicked.connect(self.reject)
        self.download_button.clicked.connect(self.download_video)
        
    def download_video(self):
        url = self.url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Error", "Please enter a valid URL")
            return
            
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            
            ydl_opts = {
                'format': 'best',
                'outtmpl': os.path.join(data_dir, '%(title)s.%(ext)s'),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            QMessageBox.information(self, "Success", "Video downloaded successfully!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to download video: {str(e)}") 