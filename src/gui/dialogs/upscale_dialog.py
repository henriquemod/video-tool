"""
Module for the Upscale Dialog GUI component.

This module provides the user interface for batch image upscaling using AI-powered
and classical methods. It allows users to select images, choose upscaling methods,
track progress, and handle the upscaling process in a separate thread to maintain
a responsive interface.
"""

from pathlib import Path
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QGroupBox, QRadioButton, QButtonGroup, QCheckBox
)
from PyQt5.QtCore import QStandardPaths, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPainter
import cv2

from ...processing.upscaling import (
    get_available_models,
    get_model_names,
    create_upscaler
)


class ImageItem:
    """
    Container for image processing information.

    Attributes:
        path (str): Source image path.
    """

    def __init__(self, path: str):
        """
        Initialize an ImageItem instance.

        Args:
            path (str): The file path of the image.
        """
        self.path = path

    def __str__(self) -> str:
        """
        Return the name of the image file.

        Returns:
            str: Name of the image file.
        """
        return str(Path(self.path).name)


class UpscaleThread(QThread):
    """
    Background worker for upscaling operations.

    Signals:
        progress(int): Processing progress percentage.
        image_progress(int, int): Current and total number of images processed.
        finished(bool, str): Success status and message.
    """

    progress = pyqtSignal(int)
    image_progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, images: list, output_dir: str, model_ids: list, output_prefix: str = ""):
        """
        Initialize the UpscaleThread.

        Args:
            images (list): List of ImageItem instances.
            output_dir (str): Directory to save upscaled images.
            model_ids (list): List of model IDs to use for upscaling.
            output_prefix (str, optional): Prefix for output filenames. Defaults to "".
        """
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.model_ids = model_ids
        self.output_prefix = output_prefix
        
        # Create upscalers
        self.upscalers = {}
        for model_id in model_ids:
            try:
                self.upscalers[model_id] = create_upscaler(model_id)
            except Exception as e:
                self.finished.emit(False, f"Failed to create upscaler for model {model_id}: {str(e)}")
                return

    def run(self):
        """
        Execute the upscaling process in a separate thread.
        """
        try:
            if not self.upscalers:
                raise RuntimeError("No valid upscalers available")

            total_operations = len(self.images) * len(self.model_ids)
            current_operation = 0

            for i, image in enumerate(self.images, 1):
                # Read image
                img = cv2.imread(image.path)
                if img is None:
                    raise FileNotFoundError(f"Failed to load image: {image.path}")

                for j, model_id in enumerate(self.model_ids, 1):
                    current_operation += 1
                    self.image_progress.emit(current_operation, total_operations)

                    # Calculate progress
                    progress = (current_operation - 1) / total_operations * 100
                    self.progress.emit(int(progress))

                    # Get upscaler for this model
                    upscaler = self.upscalers.get(model_id)
                    if upscaler is None:
                        raise RuntimeError(f"Upscaler not found for model {model_id}")

                    # Use AIUpscaler instance for upscaling
                    upscaled = upscaler.upscale(
                        img,
                        progress_callback=lambda p: self.progress.emit(
                            int((current_operation - 1) / total_operations * 100 + p / total_operations)
                        )
                    )

                    # Generate output filename
                    base_name = Path(image.path).stem
                    if self.output_prefix:
                        output_name = f"{self.output_prefix}_{current_operation}.png"
                    else:
                        model_name = model_id.split('/')[-1]  # Get last part of model ID
                        output_name = f"{base_name}_{model_name}.png"

                    # Save output
                    output_path = os.path.join(self.output_dir, output_name)
                    success = cv2.imwrite(output_path, upscaled)
                    if not success:
                        raise IOError(f"Failed to save upscaled image: {output_path}")

                    # Update progress
                    progress = current_operation / total_operations * 100
                    self.progress.emit(int(progress))

            self.finished.emit(True, f"Successfully upscaled {total_operations} images!")
        except (FileNotFoundError, IOError) as e:
            self.finished.emit(False, str(e))
        except RuntimeError as e:
            self.finished.emit(False, f"Runtime error: {str(e)}")
        except Exception as e:
            self.finished.emit(False, f"An unexpected error occurred: {str(e)}")


class DragDropListWidget(QListWidget):
    """
    Custom QListWidget that supports drag and drop of image files.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._placeholder_text = ""
        self.setMinimumHeight(150)

    def get_dialog(self):
        """
        Helper method to get the UpscaleDialog parent.
        """
        parent = self.parent()
        while parent and not isinstance(parent, UpscaleDialog):
            parent = parent.parent()
        return parent

    def setPlaceholderText(self, text: str):
        """
        Set the placeholder text to be shown when the list is empty.

        Args:
            text (str): The placeholder text to display.
        """
        self._placeholder_text = text
        self.update()

    def paintEvent(self, event):
        """
        Override paint event to draw the placeholder text when the list is empty.
        """
        super().paintEvent(event)
        if self.count() == 0 and self._placeholder_text:
            painter = QPainter(self.viewport())
            painter.save()
            col = self.palette().placeholderText().color()
            painter.setPen(col)
            fm = self.fontMetrics()
            elided_text = fm.elidedText(
                self._placeholder_text, Qt.ElideRight, self.viewport().width() - 10
            )
            painter.drawText(
                self.viewport().rect(),
                Qt.AlignCenter,
                elided_text
            )
            painter.restore()

    def dragEnterEvent(self, event: QDragEnterEvent):
        # Accept more MIME types to handle different drag sources
        if event.mimeData().hasUrls() or event.mimeData().hasText() or event.mimeData().hasImage():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        # Accept more MIME types in dragMove as well
        if event.mimeData().hasUrls() or event.mimeData().hasText() or event.mimeData().hasImage():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        dialog = self.get_dialog()
        if not dialog:
            event.ignore()
            return

        mime_data = event.mimeData()
        
        # Try to handle URLs first
        if mime_data.hasUrls():
            files = [url.toLocalFile() for url in mime_data.urls()]
            for file_path in files:
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    dialog.add_image_path(file_path)
        # Try to handle text (might be a file path)
        elif mime_data.hasText():
            file_path = mime_data.text()
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                dialog.add_image_path(file_path)
        
        event.accept()


class UpscaleDialog(QDialog):
    """
    Batch Image Upscaling Interface.

    This dialog provides a user-friendly interface for batch upscaling images using various
    AI-powered and classical methods. It supports features like progress tracking, GPU
    acceleration, and configurable output options.
    """

    def __init__(self, parent=None):
        """
        Initialize the UpscaleDialog.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Image Upscaler")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.images = []
        self.upscale_thread = None
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the user interface components.
        """
        layout = QVBoxLayout()

        # Batch Mode Selection
        mode_group = QGroupBox("Batch Mode")
        mode_layout = QVBoxLayout()
        self.mode_group = QButtonGroup()
        
        self.one_to_multiple = QRadioButton("One-to-Multiple (One image → Multiple methods)")
        self.multiple_to_one = QRadioButton("Multiple-to-One (Multiple images → One method)")
        self.multiple_to_multiple = QRadioButton("Multiple-to-Multiple (Multiple images → Multiple methods)")
        
        self.one_to_multiple.setToolTip("Process a single image through multiple upscaling methods")
        self.multiple_to_one.setToolTip("Process multiple images using a single upscaling method")
        self.multiple_to_multiple.setToolTip("Process multiple images through multiple upscaling methods")
        
        self.mode_group.addButton(self.one_to_multiple)
        self.mode_group.addButton(self.multiple_to_one)
        self.mode_group.addButton(self.multiple_to_multiple)
        
        mode_layout.addWidget(self.one_to_multiple)
        mode_layout.addWidget(self.multiple_to_one)
        mode_layout.addWidget(self.multiple_to_multiple)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Red Box: Input Area
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        # Path input
        path_layout = QHBoxLayout()
        path_label = QLabel("Images:")
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse Files")
        self.add_button = QPushButton("Add")
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.add_button)
        path_layout.addWidget(self.browse_button)
        input_layout.addLayout(path_layout)

        # Image list with drag & drop support
        self.image_list = DragDropListWidget(self)
        self.image_list.setPlaceholderText("Drop images here or click Browse Files")
        input_layout.addWidget(self.image_list)

        # Remove button
        self.remove_button = QPushButton("Remove Selected")
        input_layout.addWidget(self.remove_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Blue Box: Upscale Method Selection
        method_group = QGroupBox("Upscale Methods")
        method_layout = QVBoxLayout()
        
        # Method selection
        self.method_list = QListWidget()
        self.method_list.setSelectionMode(QListWidget.MultiSelection)
        for method in get_model_names():
            item = QListWidgetItem(method)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.method_list.addItem(item)
        
        method_layout.addWidget(self.method_list)
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Green Box: Output Options & Actions
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout()

        # Output prefix
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("Output Name Prefix:")
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Optional - Leave empty for auto-naming")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        output_layout.addLayout(prefix_layout)

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
        output_layout.addLayout(progress_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.upscale_button = QPushButton("Upscale")
        self.upscale_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.upscale_button)
        output_layout.addLayout(button_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        self.setLayout(layout)

        # Set default mode
        self.multiple_to_one.setChecked(True)
        self.mode_group.buttonClicked.connect(self.update_ui_for_mode)

        # Connect signals
        self.browse_button.clicked.connect(self.browse_images)
        self.add_button.clicked.connect(self.add_image)
        self.remove_button.clicked.connect(self.remove_image)
        self.cancel_button.clicked.connect(self.handle_cancel)
        self.upscale_button.clicked.connect(self.upscale_images)

        # Initial UI update
        self.update_ui_for_mode()

    def update_ui_for_mode(self):
        """
        Update UI elements based on selected batch mode.
        """
        one_to_multiple = self.one_to_multiple.isChecked()
        multiple_to_one = self.multiple_to_one.isChecked()
        
        # Update image list selection mode
        self.image_list.setSelectionMode(
            QListWidget.SingleSelection if one_to_multiple else QListWidget.MultiSelection
        )
        
        # Update method list selection mode and behavior
        if multiple_to_one:
            self.method_list.setSelectionMode(QListWidget.SingleSelection)
            # Uncheck all items since we're using selection instead
            for i in range(self.method_list.count()):
                item = self.method_list.item(i)
                item.setCheckState(Qt.Unchecked)
        else:
            self.method_list.setSelectionMode(QListWidget.NoSelection)
            # Clear selection since we're using checkboxes
            self.method_list.clearSelection()
            
        # Update the UI to reflect the changes
        self.method_list.update()

    def add_image_path(self, path: str):
        """
        Add an image to the list from a file path.

        Args:
            path (str): Path to the image file.
        """
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return

        # Check if we're in one-to-multiple mode and already have an image
        if self.one_to_multiple.isChecked() and self.images:
            QMessageBox.warning(
                self, "Error", 
                "One-to-Multiple mode only supports a single input image. "
                "Please remove the existing image first."
            )
            return

        image = ImageItem(path)
        self.images.append(image)
        self.image_list.addItem(QListWidgetItem(str(image)))

    def browse_images(self):
        """
        Open a file dialog to select images and add them to the list.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        for path in file_paths:
            self.add_image_path(path)

    def add_image(self):
        """
        Add a single image to the list based on the input path.
        """
        path = self.path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Please enter an image path")
            return

        self.add_image_path(path)
        self.path_input.clear()

    def remove_image(self):
        """
        Remove the selected image(s) from the list.
        """
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            row = self.image_list.row(item)
            self.image_list.takeItem(row)
            self.images.pop(row)

    def get_selected_model_ids(self) -> list:
        """
        Get the IDs of selected upscaling models.

        Returns:
            list: List of selected model IDs.
        """
        selected_models = []
        models = get_available_models()
        
        if self.multiple_to_one.isChecked():
            # In Multiple-to-One mode, use the current selection
            current_row = self.method_list.currentRow()
            if current_row >= 0:  # Make sure there is a selection
                selected_models.append(models[current_row].id)
        else:
            # In One-to-Multiple or Multiple-to-Multiple mode, use checked items
            for i in range(self.method_list.count()):
                item = self.method_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected_models.append(models[i].id)
        
        return selected_models

    def validate_selections(self) -> bool:
        """
        Validate user selections before starting the upscale process.

        Returns:
            bool: True if selections are valid, False otherwise.
        """
        if not self.images:
            QMessageBox.warning(
                self, "Error", "Please add at least one image to upscale"
            )
            return False

        selected_models = self.get_selected_model_ids()
        if not selected_models:
            QMessageBox.warning(
                self, "Error", "Please select at least one upscaling method"
            )
            return False

        if self.one_to_multiple.isChecked() and len(self.images) > 1:
            QMessageBox.warning(
                self, "Error", 
                "One-to-Multiple mode only supports a single input image"
            )
            return False

        return True

    def handle_cancel(self):
        """
        Handle the cancellation of the upscaling process and close the dialog.
        """
        if self.upscale_thread and self.upscale_thread.isRunning():
            self.upscale_thread.terminate()
            self.upscale_thread.wait()
            self.progress_bar.hide()
            self.progress_label.hide()
            self.upscale_button.setEnabled(True)
        self.reject()

    def upscale_images(self):
        """
        Initiate the upscaling process for the selected images.
        """
        if not self.validate_selections():
            return

        # Ask user for save location
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Location for Upscaled Images",
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not output_dir:  # User cancelled
            return

        # Show progress bar and disable buttons
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.upscale_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.browse_button.setEnabled(False)

        # Create and start upscale thread
        selected_models = self.get_selected_model_ids()
        self.upscale_thread = UpscaleThread(
            self.images,
            output_dir,
            selected_models,
            self.prefix_input.text().strip()
        )
        self.upscale_thread.progress.connect(self.progress_bar.setValue)
        self.upscale_thread.image_progress.connect(self.update_progress_label)
        self.upscale_thread.finished.connect(self.upscale_finished)

        # Start upscaling
        self.upscale_thread.start()

    def update_progress_label(self, current: int, total: int):
        """
        Update the progress label with current and total counts.

        Args:
            current (int): Current operation number.
            total (int): Total number of operations.
        """
        self.progress_label.setText(f"{current}/{total}")

    def upscale_finished(self, success: bool, message: str):
        """
        Handle the completion of the upscaling process.

        Args:
            success (bool): Indicates if the process was successful.
            message (str): Success or error message.
        """
        self.progress_bar.hide()
        self.progress_label.hide()
        self.upscale_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.browse_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(
                self, "Error", f"Failed to upscale images: {message}"
            )
