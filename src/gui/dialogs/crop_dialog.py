"""
Crop Dialog Module - Provides an interactive image cropping interface with aspect ratio control.

This module implements a sophisticated image cropping dialog with real-time preview,
aspect ratio control, and grid overlay support for precise image cropping operations.
"""

from PyQt5 import QtWidgets, QtCore, QtGui

from ...utils.temp_file_manager import temp_manager
from ...exceptions import CropError


class CustomRubberBand(QtWidgets.QRubberBand):
    """
    Enhanced rubber band widget with grid overlay for image cropping.

    Provides visual guides including rule of thirds grid and custom border styling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_enabled = True

    def paint_event(self, _):
        """
        Custom paint event to draw dashed border and grid.

        Args:
            _: Unused paint event parameter
        """
        painter = QtGui.QPainter(self)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
        pen.setStyle(QtCore.Qt.DashLine)
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


class CropDialog(QtWidgets.QDialog):
    """
    Main dialog for image cropping with aspect ratio control and interactive selection.

    Attributes:
        image_path (str): Path to the image being cropped
        cropped_path (str): Path where the cropped image will be saved
        is_dragging (bool): Flag indicating if selection is being dragged
        drag_start_pos (QtCore.QPoint): Starting position of drag operation
        initial_rect (QtCore.QRect): Initial rectangle before drag
        current_aspect_ratio (float): Current aspect ratio constraint
    """

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        # Initialize paths
        self.image_path = image_path
        self.cropped_path = None
        self.result_path = None

        # Initialize selection state
        self.is_dragging = False
        self.drag_start_pos = None
        self.initial_rect = None
        self.current_aspect_ratio = None
        self.selected_rect = None
        self.origin = None

        # Initialize UI components
        self.image_label = None
        self.pixmap = None
        self.aspect_combo = None
        self.crop_button = None
        self.rubber_band = None

        self.setup_ui()

        # Add event filter to image_label
        self.image_label.installEventFilter(self)

    def setup_ui(self):
        """Set up the dialog UI components and layouts."""
        self.setWindowTitle("Crop Image")
        layout = QtWidgets.QVBoxLayout()

        # Add aspect ratio selector
        aspect_layout = QtWidgets.QHBoxLayout()
        aspect_label = QtWidgets.QLabel("Aspect Ratio:")
        self.aspect_combo = QtWidgets.QComboBox()
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
        self.image_label = QtWidgets.QLabel()
        self.pixmap = QtGui.QPixmap(self.image_path)

        # Check if the pixmap is valid
        if self.pixmap.isNull():
            QtWidgets.QMessageBox.critical(
                self, "Error", "Failed to load image for cropping.")
            self.reject()
            return

        # Scale pixmap if too large while maintaining aspect ratio
        scaled_pixmap = self.pixmap.scaled(
            800, 600,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Enable mouse tracking for rubber band selection
        self.image_label.setMouseTracking(True)
        layout.addWidget(self.image_label)

        # Add crop button
        self.crop_button = QtWidgets.QPushButton("Crop")
        self.crop_button.clicked.connect(self.crop_image)
        layout.addWidget(self.crop_button)

        self.setLayout(layout)

        # Initialize rubber band
        self.rubber_band = CustomRubberBand(
            QtWidgets.QRubberBand.Rectangle, self.image_label)

    def eventFilter(self, obj, event):
        """
        Handle mouse events from the image label.

        Args:
            obj: Object that triggered the event
            event: Event to be processed
        """
        if obj == self.image_label:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.handle_mouse_press(event)
                return True
            elif event.type() == QtCore.QEvent.MouseMove:
                self.handle_mouse_move(event)
                return True
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        """Handle dialog close event."""
        self.result_path = None
        event.accept()

    def handle_mouse_press(self, event):
        """
        Handle mouse press events for selection creation and movement.

        Args:
            event: Mouse event containing position information
        """
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()  # Event pos is already relative to image_label
            # Check if the click is within the image label bounds
            if not self.image_label.rect().contains(pos):
                return

            if (hasattr(self, 'selected_rect') and
                self.selected_rect is not None and
                    self.selected_rect.contains(pos)):
                self.is_dragging = True
                self.drag_start_pos = pos
                self.initial_rect = QtCore.QRect(self.selected_rect)
            else:
                self.is_dragging = False
                self.origin = pos
                self.rubber_band.setGeometry(QtCore.QRect(pos, QtCore.QSize()))
                self.rubber_band.show()

    def handle_mouse_move(self, event):
        """
        Handle mouse movement for selection resizing and moving.

        Args:
            event: Mouse event containing position information
        """
        pos = event.pos()  # Event pos is already relative to image_label
        # Constrain position to image boundaries
        pos.setX(max(0, min(pos.x(), self.image_label.width())))
        pos.setY(max(0, min(pos.y(), self.image_label.height())))

        if self.is_dragging and self.drag_start_pos:
            # Move the existing selection
            delta = pos - self.drag_start_pos
            new_rect = self.initial_rect.translated(delta)
            # Constrain to image boundaries
            if new_rect.left() < 0:
                new_rect.moveLeft(0)
            if new_rect.right() > self.image_label.width():
                new_rect.moveRight(self.image_label.width())
            if new_rect.top() < 0:
                new_rect.moveTop(0)
            if new_rect.bottom() > self.image_label.height():
                new_rect.moveBottom(self.image_label.height())
            self.selected_rect = new_rect
            self.rubber_band.setGeometry(self.selected_rect)
        elif hasattr(self, 'origin') and self.origin:
            # Create new selection
            rect = QtCore.QRect(self.origin, pos).normalized()
            if self.current_aspect_ratio:
                # Maintain aspect ratio while dragging
                width = rect.width()
                height = int(width / self.current_aspect_ratio)
                if height > self.image_label.height():
                    height = self.image_label.height()
                    width = int(height * self.current_aspect_ratio)
                rect.setHeight(height)
                rect.setWidth(width)
            self.rubber_band.setGeometry(rect)

    def handle_mouse_release(self, event):
        """
        Handle mouse release events for selection completion.

        Args:
            event: Mouse event containing position information
        """
        if event.button() == QtCore.Qt.LeftButton:
            if self.is_dragging:
                self.is_dragging = False
                self.drag_start_pos = None
                self.initial_rect = None
            elif hasattr(self, 'origin') and self.origin:
                pos = event.pos()  # Event pos is already relative to image_label
                rect = QtCore.QRect(self.origin, pos).normalized()
                if self.current_aspect_ratio:
                    # Maintain aspect ratio for final selection
                    width = rect.width()
                    height = int(width / self.current_aspect_ratio)
                    if height > self.image_label.height():
                        height = self.image_label.height()
                        width = int(height * self.current_aspect_ratio)
                    rect.setHeight(height)
                    rect.setWidth(width)
                self.selected_rect = rect
                self.rubber_band.setGeometry(rect)
                self.origin = None

    def crop_image(self):
        """Crop the image using the selected area and save it."""
        if not hasattr(self, 'selected_rect'):
            QtWidgets.QMessageBox.warning(
                self, "Warning", "Please select an area to crop")
            return

        try:
            # Calculate the scale factor between original and displayed image
            scale_x = self.pixmap.width() / self.image_label.pixmap().width()
            scale_y = self.pixmap.height() / self.image_label.pixmap().height()

            # Scale the selection rectangle to match original image dimensions
            scaled_rect = QtCore.QRect(
                int(self.selected_rect.x() * scale_x),
                int(self.selected_rect.y() * scale_y),
                int(self.selected_rect.width() * scale_x),
                int(self.selected_rect.height() * scale_y)
            )

            # Crop the original image
            cropped_pixmap = self.pixmap.copy(scaled_rect)

            # Save to temporary location
            temp_path = temp_manager.get_temp_path(
                prefix="cropped_", suffix=".png")
            if not cropped_pixmap.save(str(temp_path)):
                raise CropError("Failed to save cropped image")

            self.result_path = temp_path
            self.accept()

        except Exception as e:
            if not isinstance(e, CropError):
                raise CropError(f"Failed to crop image: {str(e)}")
            raise

    def get_result_path(self):
        """Return the path of the cropped image."""
        return getattr(self, 'result_path', None)

    def on_aspect_ratio_changed(self, text):
        """
        Handle aspect ratio changes from the combo box.

        Args:
            text (str): Selected aspect ratio text from combo box
        """
        # Parse the aspect ratio from the selected text
        if text == "Free Crop":
            self.current_aspect_ratio = None
        else:
            try:
                # Extract ratio values from text (e.g., "16:9" -> 16/9)
                if ":" in text:
                    width, height = map(float, text.split(
                        "(")[0].strip().split(":"))
                    self.current_aspect_ratio = width / height
                elif text == "1:1 (Square)":
                    self.current_aspect_ratio = 1.0
            except (ValueError, IndexError):
                self.current_aspect_ratio = None

        # Update the selection if it exists
        if hasattr(self, 'selected_rect') and self.selected_rect:
            if self.current_aspect_ratio:
                # Maintain the current center while adjusting to new aspect ratio
                center = self.selected_rect.center()
                width = self.selected_rect.width()
                new_height = int(width / self.current_aspect_ratio)
                new_rect = QtCore.QRect(
                    self.selected_rect.x(),
                    center.y() - new_height // 2,
                    width,
                    new_height
                )
                # Ensure the new rectangle fits within the image bounds
                image_rect = self.image_label.rect()
                if new_rect.top() < 0:
                    new_rect.moveTop(0)
                if new_rect.bottom() > image_rect.bottom():
                    new_rect.moveBottom(image_rect.bottom())
                self.selected_rect = new_rect
                self.rubber_band.setGeometry(self.selected_rect)
