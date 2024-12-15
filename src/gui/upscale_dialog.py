from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
from pathlib import Path
from ..processing.ai_upscaling import upscale_image, get_device

class ImageItem:
    def __init__(self, path):
        self.path = path
    
    def __str__(self):
        return str(Path(self.path).name)

class UpscaleThread(QThread):
    progress = pyqtSignal(int)
    image_progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, images, output_dir, method, scale):
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.method = method
        self.scale = scale
        self.device = get_device()
        
    def run(self):
        try:
            total_images = len(self.images)
            for i, image in enumerate(self.images, 1):
                self.image_progress.emit(i, total_images)
                
                # Calculate progress
                progress = (i - 1) / total_images * 100
                self.progress.emit(int(progress))
                
                # Read image
                img = cv2.imread(image.path)
                if img is None:
                    raise Exception(f"Failed to load image: {image.path}")
                
                # Perform upscaling
                upscaled = upscale_image(img, self.method, self.scale, self.device)
                
                # Save output
                output_path = os.path.join(
                    self.output_dir,
                    f"upscaled_{self.method}_{self.scale}x_{Path(image.path).name}"
                )
                cv2.imwrite(output_path, upscaled)
                
                # Update progress
                progress = i / total_images * 100
                self.progress.emit(int(progress))
                
            self.finished.emit(True, f"Successfully upscaled {total_images} images!")
        except Exception as e:
            self.finished.emit(False, str(e))

class UpscaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Image Upscaler")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.images = []
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Path input
        path_layout = QHBoxLayout()
        path_label = QLabel("Image Path:")
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_button)
        
        # Add button
        self.add_button = QPushButton("Add")
        path_layout.addWidget(self.add_button)
        
        layout.addLayout(path_layout)
        
        # Image list
        self.image_list = QListWidget()
        layout.addWidget(self.image_list)
        
        # Remove button
        self.remove_button = QPushButton("Remove Selected")
        layout.addWidget(self.remove_button)
        
        # Add upscale options
        options_layout = QHBoxLayout()
        
        # Method selection
        method_label = QLabel("Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Bicubic",
            "Lanczos",
            "Real-ESRGAN",
            "SwinIR"
        ])
        options_layout.addWidget(method_label)
        options_layout.addWidget(self.method_combo)
        
        # Scale selection
        scale_label = QLabel("Scale:")
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "4x"])
        options_layout.addWidget(scale_label)
        options_layout.addWidget(self.scale_combo)
        
        layout.addLayout(options_layout)
        
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
        self.upscale_button = QPushButton("Upscale")
        self.upscale_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.upscale_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.browse_button.clicked.connect(self.browse_images)
        self.add_button.clicked.connect(self.add_images)
        self.remove_button.clicked.connect(self.remove_image)
        self.cancel_button.clicked.connect(self.handle_cancel)
        self.upscale_button.clicked.connect(self.upscale_images)
        
    def browse_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_paths:
            self.path_input.setText(";".join(file_paths))
            
    def add_images(self):
        paths = self.path_input.text().strip().split(";")
        for path in paths:
            if not path:
                continue
                
            if not os.path.isfile(path):
                QMessageBox.warning(self, "Error", f"File not found: {path}")
                continue
                
            image = ImageItem(path)
            self.images.append(image)
            self.image_list.addItem(QListWidgetItem(str(image)))
            
        self.path_input.clear()
        
    def remove_image(self):
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            self.image_list.takeItem(current_row)
            self.images.pop(current_row)
        
    def handle_cancel(self):
        if hasattr(self, 'upscale_thread') and self.upscale_thread.isRunning():
            self.upscale_thread.terminate()
            self.upscale_thread.wait()
            self.progress_bar.hide()
            self.progress_label.hide()
            self.upscale_button.setEnabled(True)
        self.reject()
        
    def upscale_images(self):
        if not self.images:
            QMessageBox.warning(self, "Error", "Please add at least one image to upscale")
            return
            
        # Get selected method and scale
        method = self.method_combo.currentText()
        scale = int(self.scale_combo.currentText().replace('x', ''))
        
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Show progress bar and disable buttons
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.upscale_button.setEnabled(False)
        self.add_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        
        # Create and configure upscale thread
        self.upscale_thread = UpscaleThread(self.images, output_dir, method, scale)
        self.upscale_thread.progress.connect(self.progress_bar.setValue)
        self.upscale_thread.image_progress.connect(self.update_progress_label)
        self.upscale_thread.finished.connect(self.upscale_finished)
        
        # Start upscaling
        self.upscale_thread.start()
        
    def update_progress_label(self, current, total):
        self.progress_label.setText(f"{current}/{total}")
        
    def upscale_finished(self, success, message):
        self.progress_bar.hide()
        self.progress_label.hide()
        self.upscale_button.setEnabled(True)
        self.add_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to upscale images: {message}")