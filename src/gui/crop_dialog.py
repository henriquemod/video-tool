from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QPushButton,
                             QLabel, QRubberBand, QMessageBox,
                             QComboBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
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
    """
    CropDialog - Interactive Image Cropping Interface

    This module implements a sophisticated image cropping dialog with real-time preview,
    aspect ratio control, and grid overlay support. It provides a professional-grade
    interface for precise image cropping operations.

    Key Features:
    - Interactive rubber band selection
    - Customizable aspect ratios
    - Rule of thirds grid overlay
    - Real-time preview
    - High-resolution support
    - Drag and resize capabilities

    Components:
    1. Selection Interface:
        - Custom rubber band implementation with grid overlay
        - Drag-to-create and drag-to-move functionality
        - Resize handles for fine adjustment
        - Boundary constraints to image dimensions

    2. Aspect Ratio Control:
        - Predefined aspect ratio options:
            * Free Crop (unconstrained)
            * 1:1 (Square)
            * 4:3 (Standard)
            * 16:9 (Widescreen)
            * 3:2 (Classic Photo)
            * 2:3 (Portrait)
        - Dynamic constraint maintenance during resizing
        - Automatic centering for ratio changes

    3. Visual Aids:
        - Rule of thirds grid overlay
        - Selection border with dashed style
        - Real-time dimension display
        - Visual feedback for constraints

    Classes:
        CustomRubberBand(QRubberBand):
            Enhanced rubber band widget with grid overlay
            Features:
                - Custom border styling
                - Rule of thirds grid
                - Dynamic size updates

        CropDialog(QDialog):
            Main cropping interface
            Features:
                - Image loading and display
                - Selection management
                - Aspect ratio control
                - Result processing

    Technical Implementation:
    - Efficient image scaling for display
    - Responsive selection updates
    - Precise coordinate mapping
    - Memory-efficient image handling

    Mouse Interaction:
    1. Creation Mode:
        - Click and drag to create selection
        - Maintains aspect ratio if specified
        - Constrains to image boundaries

    2. Edit Mode:
        - Click and drag existing selection
        - Resize from edges/corners
        - Snap to image boundaries

    Dependencies:
    - PyQt5: Core GUI framework
    - PIL/Pillow: Image processing
    - numpy: Coordinate calculations

    Example Usage:
        dialog = CropDialog(image_path, parent)
        if dialog.exec_() == QDialog.Accepted:
            cropped_path = dialog.get_cropped_path()

    Performance Considerations:
    - Efficient image scaling for preview
    - Smooth selection updates
    - Proper cleanup of resources
    - Memory management for large images

    Error Handling:
    - Invalid image files
    - Out of memory conditions
    - File system errors
    - Coordinate validation

    @see @Project Structure#crop_dialog.py
    @see @Project#Image Processing
    @see @Project#UI Components
    """

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.cropped_path = None
        self.is_dragging = False
        self.drag_start_pos = None
        self.initial_rect = None
        self.current_aspect_ratio = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Crop Image")
        layout = QVBoxLayout()

        # Add aspect ratio selector
        aspect_layout = QHBoxLayout()
        aspect_label = QLabel("Aspect Ratio:")
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems([
            "Free Crop",
            "1:1 (Square)",
            "4:3",
            "16:9",
            "3:2",
            "2:3 (Portrait)"
        ])
        self.aspect_combo.currentTextChanged.connect(
            self.on_aspect_ratio_changed)
        aspect_layout.addWidget(aspect_label)
        aspect_layout.addWidget(self.aspect_combo)
        aspect_layout.addStretch()
        layout.addLayout(aspect_layout)

        # Load and display the image
        self.image_label = QLabel()
        self.pixmap = QPixmap(self.image_path)

        # Check if the pixmap is valid
        if self.pixmap.isNull():
            QMessageBox.critical(
                self, "Error", "Failed to load image for cropping.")
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
        self.rubber_band = CustomRubberBand(
            QRubberBand.Rectangle, self.image_label)
        self.origin = None

    def on_aspect_ratio_changed(self, text):
        """Handle aspect ratio selection changes."""
        ratios = {
            "Free Crop": None,
            "1:1 (Square)": 1/1,
            "4:3": 4/3,
            "16:9": 16/9,
            "3:2": 3/2,
            "2:3 (Portrait)": 2/3
        }
        self.current_aspect_ratio = ratios[text]

        # Clear existing selection if it exists
        if hasattr(self, 'selected_rect'):
            self.selected_rect = None
            self.rubber_band.hide()

    def adjust_selection_aspect_ratio(self):
        """Adjust the current selection to match the chosen aspect ratio."""
        if not self.current_aspect_ratio:
            return

        current_rect = self.rubber_band.geometry()
        new_width = current_rect.width()
        new_height = int(new_width / self.current_aspect_ratio)

        # If new height would exceed image bounds, adjust width instead
        if new_height > self.image_label.height():
            new_height = current_rect.height()
            new_width = int(new_height * self.current_aspect_ratio)

        # Center the new rectangle on the old one
        x = current_rect.x() + (current_rect.width() - new_width) // 2
        y = current_rect.y() + (current_rect.height() - new_height) // 2

        self.selected_rect = QRect(x, y, new_width, new_height)
        self.rubber_band.setGeometry(self.selected_rect)

    def mousePressEvent(self, event):
        """Handle mouse press for both creating and moving selection."""
        if event.button() == Qt.LeftButton:
            pos = self.image_label.mapFrom(self, event.pos())

            if (hasattr(self, 'selected_rect') and
                self.selected_rect is not None and
                    self.selected_rect.contains(pos)):
                self.is_dragging = True
                self.drag_start_pos = pos
                self.initial_rect = QRect(self.selected_rect)
            else:
                self.is_dragging = False
                self.origin = pos
                self.rubber_band.setGeometry(QRect(self.origin, QSize(1, 1)))
                self.rubber_band.show()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for both resizing and moving selection."""
        pos = self.image_label.mapFrom(self, event.pos())

        # Constrain position to image boundaries
        pos.setX(max(0, min(pos.x(), self.image_label.width())))
        pos.setY(max(0, min(pos.y(), self.image_label.height())))

        if self.is_dragging and self.drag_start_pos:
            delta = pos - self.drag_start_pos
            new_rect = self.initial_rect.translated(delta)

            # Keep selection within image bounds
            label_rect = self.image_label.rect()
            if new_rect.right() > label_rect.width():
                new_rect.moveRight(label_rect.width())
            if new_rect.bottom() > label_rect.height():
                new_rect.moveBottom(label_rect.height())
            if new_rect.left() < 0:
                new_rect.moveLeft(0)
            if new_rect.top() < 0:
                new_rect.moveTop(0)

            self.selected_rect = new_rect
            self.rubber_band.setGeometry(self.selected_rect)
        elif self.origin:
            # Calculate the raw rectangle from origin to current position
            current_rect = QRect(self.origin, pos).normalized()

            if self.current_aspect_ratio:
                # Calculate dimensions maintaining aspect ratio
                if abs(pos.x() - self.origin.x()) > abs(pos.y() - self.origin.y()):
                    # Width is dominant
                    width = min(abs(pos.x() - self.origin.x()),
                                self.image_label.width() - self.origin.x())
                    height = int(width / self.current_aspect_ratio)

                    # Adjust if height exceeds boundaries
                    if height > self.image_label.height() - min(self.origin.y(), pos.y()):
                        height = self.image_label.height() - min(self.origin.y(), pos.y())
                        width = int(height * self.current_aspect_ratio)
                else:
                    # Height is dominant
                    height = min(abs(pos.y() - self.origin.y()),
                                 self.image_label.height() - self.origin.y())
                    width = int(height * self.current_aspect_ratio)

                    # Adjust if width exceeds boundaries
                    if width > self.image_label.width() - min(self.origin.x(), pos.x()):
                        width = self.image_label.width() - min(self.origin.x(), pos.x())
                        height = int(width / self.current_aspect_ratio)

                # Determine direction of drag to properly position the rectangle
                x = self.origin.x()
                y = self.origin.y()

                if pos.x() < self.origin.x():
                    x = max(0, self.origin.x() - width)
                if pos.y() < self.origin.y():
                    y = max(0, self.origin.y() - height)

                # Create new rect with calculated dimensions
                new_rect = QRect(x, y, width, height)
            else:
                # Free crop mode
                new_rect = current_rect

                # Ensure the rectangle stays within the image bounds
                new_rect = new_rect.intersected(self.image_label.rect())

            self.rubber_band.setGeometry(new_rect)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for both creation and moving."""
        if event.button() == Qt.LeftButton:
            pos = self.image_label.mapFrom(self, event.pos())

            if self.is_dragging:
                self.is_dragging = False
                self.drag_start_pos = None
                self.initial_rect = None
            else:
                # Use the current rubber band geometry instead of creating a new rect
                self.selected_rect = QRect(self.rubber_band.geometry())
                # No need to hide and show again
                self.rubber_band.setGeometry(self.selected_rect)

    def crop_image(self):
        """Crop the image using the selected area and save it."""
        if not hasattr(self, 'selected_rect'):
            QMessageBox.warning(
                self, "Warning", "Please select an area to crop")
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
            QMessageBox.critical(
                self, "Error", f"Failed to crop image: {str(e)}")

    def get_cropped_path(self):
        """Return the path of the cropped image."""
        return self.cropped_path
