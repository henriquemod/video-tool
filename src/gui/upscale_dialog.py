from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QComboBox, 
                           QLabel, QPushButton, QMessageBox)
from PyQt5.QtCore import Qt
import os
import cv2
import numpy as np

class UpscaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_scale = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Select Upscale Options")
        layout = QVBoxLayout()
        
        # Add upscale method selector
        method_label = QLabel("Upscale Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "No Upscaling",
            "Bicubic",  # Basic CV2 upscaling
            "Lanczos",  # Better quality for general purposes
            # Future AI models can be added here
            # "Real-ESRGAN",
            # "SRCNN",
        ])
        layout.addWidget(method_label)
        layout.addWidget(self.method_combo)
        
        # Add scale factor selector
        scale_label = QLabel("Scale Factor:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "2x",
            "3x",
            "4x"
        ])
        self.scale_combo.setEnabled(False)  # Initially disabled
        layout.addWidget(scale_label)
        layout.addWidget(self.scale_combo)
        
        # Connect method changes to enable/disable scale selector
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        
        # Add OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)
        
        self.setLayout(layout)
    
    def on_method_changed(self, text):
        """Enable/disable scale selector based on method."""
        self.scale_combo.setEnabled(text != "No Upscaling")
    
    def get_settings(self):
        """Return the selected upscale method and scale factor."""
        method = self.method_combo.currentText()
        if method == "No Upscaling":
            return None, None
        
        scale = int(self.scale_combo.currentText().replace('x', ''))
        return method, scale 