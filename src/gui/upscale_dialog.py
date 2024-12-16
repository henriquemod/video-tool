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
    QFileDialog
)
from PyQt5.QtCore import QStandardPaths, QThread, pyqtSignal
import cv2

from ..processing.ai_upscaling import AIUpscaler, get_available_models, get_model_names


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

    def __init__(self, images: list, output_dir: str, model_id: str):
        """
        Initialize the UpscaleThread.

        Args:
            images (list): List of ImageItem instances.
            output_dir (str): Directory to save upscaled images.
            model_id (str): Identifier for the AI model to use.
        """
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.model_id = model_id
        self.upscaler = AIUpscaler()

    def run(self):
        """
        Execute the upscaling process in a separate thread.
        """
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
                    raise FileNotFoundError(
                        f"Failed to load image: {image.path}")

                # Use AIUpscaler instance for upscaling
                upscaled = self.upscaler.upscale(
                    img,
                    self.model_id,
                    progress_callback=lambda p, current=i: self.progress.emit(
                        int((current - 1) / total_images * 100 + p / total_images)
                    )
                )

                # Save output
                output_path = os.path.join(
                    self.output_dir,
                    f"upscaled_{Path(image.path).name}"
                )
                success = cv2.imwrite(output_path, upscaled)
                if not success:
                    raise IOError(
                        f"Failed to save upscaled image: {output_path}")

                # Update progress
                progress = i / total_images * 100
                self.progress.emit(int(progress))

            self.finished.emit(
                True, f"Successfully upscaled {total_images} images!"
            )
        except (FileNotFoundError, IOError) as e:
            self.finished.emit(False, str(e))
        except RuntimeError as e:
            # Handle runtime-specific errors from AIUpscaler
            self.finished.emit(False, f"Runtime error: {str(e)}")
        except Exception as e:
            # Handle any other unexpected exceptions
            self.finished.emit(
                False, f"An unexpected error occurred: {str(e)}")


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
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.images = []
        self.upscale_thread = None
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the user interface components.
        """
        layout = QVBoxLayout()

        # Path input
        path_layout = QHBoxLayout()
        path_label = QLabel("Images:")
        self.path_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.add_button = QPushButton("Add")
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.add_button)
        path_layout.addWidget(self.browse_button)
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
        self.method_combo.addItems(get_model_names())
        options_layout.addWidget(method_label)
        options_layout.addWidget(self.method_combo)

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
        self.add_button.clicked.connect(self.add_image)
        self.remove_button.clicked.connect(self.remove_image)
        self.cancel_button.clicked.connect(self.handle_cancel)
        self.upscale_button.clicked.connect(self.upscale_images)

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
        if file_paths:
            for path in file_paths:
                if not os.path.isfile(path):
                    QMessageBox.warning(
                        self, "Error", f"File not found: {path}"
                    )
                    continue

                image = ImageItem(path)
                self.images.append(image)
                self.image_list.addItem(QListWidgetItem(str(image)))

    def add_image(self):
        """
        Add a single image to the list based on the input path.
        """
        path = self.path_input.text().strip()
        if not path:
            QMessageBox.warning(self, "Error", "Please enter an image path")
            return

        if not os.path.isfile(path):
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return

        image = ImageItem(path)
        self.images.append(image)
        self.image_list.addItem(QListWidgetItem(str(image)))
        self.path_input.clear()

    def remove_image(self):
        """
        Remove the selected image from the list.
        """
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            self.image_list.takeItem(current_row)
            self.images.pop(current_row)

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
        if not self.images:
            QMessageBox.warning(
                self, "Error", "Please add at least one image to upscale"
            )
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

        # Create and start upscale thread with user-selected directory
        selected_model = get_available_models(
        )[self.method_combo.currentIndex()]
        self.upscale_thread = UpscaleThread(
            self.images,
            output_dir,
            selected_model.id
        )
        self.upscale_thread.progress.connect(self.progress_bar.setValue)
        self.upscale_thread.image_progress.connect(self.update_progress_label)
        self.upscale_thread.finished.connect(self.upscale_finished)

        # Start upscaling
        self.upscale_thread.start()

    def update_progress_label(self, current: int, total: int):
        """
        Update the progress label with current and total image counts.

        Args:
            current (int): Current image number.
            total (int): Total number of images.
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
