from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, 
                           QLabel, QRubberBand, QMessageBox)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage
import os

class CropDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.cropped_path = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Crop Image")
        layout = QVBoxLayout()
        
        # Load and display the image
        self.image_label = QLabel()
        self.pixmap = QPixmap(self.image_path)
        
        # Scale pixmap if too large while maintaining aspect ratio
        scaled_pixmap = self.pixmap.scaled(800, 600, Qt.KeepAspectRatio, 
                                         Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        
        # Enable mouse tracking for rubber band selection
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label)
        
        # Add crop button
        self.crop_button = QPushButton("Crop")
        self.crop_button.clicked.connect(self.crop_image)
        layout.addWidget(self.crop_button)
        
        self.setLayout(layout)
        
        # Initialize rubber band
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.image_label)
        self.origin = None
        
    def mousePressEvent(self, event):
        """Handle mouse press to start rubber band selection."""
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, self.origin))
            self.rubber_band.show()
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement to update rubber band size."""
        if self.origin:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finalize selection."""
        if event.button() == Qt.LeftButton:
            self.rubber_band.hide()
            self.selected_rect = QRect(self.origin, event.pos()).normalized()
            self.rubber_band.setGeometry(self.selected_rect)
            self.rubber_band.show()
            
    def crop_image(self):
        """Crop the image using the selected area and save it."""
        if not hasattr(self, 'selected_rect'):
            QMessageBox.warning(self, "Warning", "Please select an area to crop")
            return
            
        try:
            # Calculate the scale factor between original and displayed image
            scale_x = self.pixmap.width() / self.image_label.pixmap().width()
            scale_y = self.pixmap.height() / self.image_label.pixmap().height()
            
            # Scale the selection rectangle to match original image dimensions
            scaled_rect = QRect(
                int(self.selected_rect.x() * scale_x),
                int(self.selected_rect.y() * scale_y),
                int(self.selected_rect.width() * scale_x),
                int(self.selected_rect.height() * scale_y)
            )
            
            # Crop the original image
            cropped_pixmap = self.pixmap.copy(scaled_rect)
            
            # Generate cropped image filename
            base_path, ext = os.path.splitext(self.image_path)
            self.cropped_path = f"{base_path}_cropped{ext}"
            
            # Save the cropped image
            if cropped_pixmap.save(self.cropped_path):
                self.accept()
            else:
                raise Exception("Failed to save cropped image")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to crop image: {str(e)}")
            
    def get_cropped_path(self):
        """Return the path of the cropped image."""
        return self.cropped_path 