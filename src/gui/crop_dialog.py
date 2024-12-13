from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton, 
                           QLabel, QRubberBand, QMessageBox)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
import os

class CustomRubberBand(QRubberBand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_enabled = True

    def paintEvent(self, event):
        """Custom paint event to draw dashed border and grid."""
        painter = QPainter(self)
        pen = QPen(QColor(255, 255, 255))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        
        rect = self.rect()
        painter.drawRect(rect)

        if self.grid_enabled:
            # Draw grid lines (3x3 grid)
            cell_width = rect.width() / 3
            cell_height = rect.height() / 3
            
            # Draw horizontal lines
            for i in range(1, 3):
                y = int(rect.top() + (i * cell_height))
                painter.drawLine(rect.left(), y, rect.right(), y)
            
            # Draw vertical lines
            for i in range(1, 3):
                x = int(rect.left() + (i * cell_width))
                painter.drawLine(x, rect.top(), x, rect.bottom())

class CropDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.cropped_path = None
        self.is_dragging = False
        self.drag_start_pos = None
        self.initial_rect = None
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Crop Image")
        layout = QVBoxLayout()
        
        # Load and display the image
        self.image_label = QLabel()
        self.pixmap = QPixmap(self.image_path)
        
        # Check if the pixmap is valid
        if self.pixmap.isNull():
            QMessageBox.critical(self, "Error", "Failed to load image for cropping.")
            self.reject()
            return
        
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
        self.rubber_band = CustomRubberBand(QRubberBand.Rectangle, self.image_label)
        self.origin = None
        
    def mousePressEvent(self, event):
        """Handle mouse press for both creating and moving selection."""
        if event.button() == Qt.LeftButton:
            # Check if clicking inside existing selection
            if (hasattr(self, 'selected_rect') and 
                self.selected_rect.contains(event.pos())):
                self.is_dragging = True
                self.drag_start_pos = event.pos()
                self.initial_rect = QRect(self.selected_rect)
            else:
                # Start new selection
                self.is_dragging = False
                self.origin = event.pos()
                self.rubber_band.setGeometry(QRect(self.origin, self.origin))
                self.rubber_band.show()
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement for both resizing and moving selection."""
        if self.is_dragging and self.drag_start_pos:
            # Move existing selection
            delta = event.pos() - self.drag_start_pos
            new_rect = self.initial_rect.translated(delta)
            
            # Keep selection within image bounds
            label_rect = self.image_label.rect()
            if label_rect.contains(new_rect):
                self.selected_rect = new_rect
                self.rubber_band.setGeometry(self.selected_rect)
        elif self.origin:
            # Resize new selection
            self.rubber_band.setGeometry(
                QRect(self.origin, event.pos()).normalized())
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release for both creation and moving."""
        if event.button() == Qt.LeftButton:
            if self.is_dragging:
                self.is_dragging = False
                self.drag_start_pos = None
                self.initial_rect = None
            else:
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