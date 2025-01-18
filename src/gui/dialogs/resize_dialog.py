"""
Module for batch image resizing dialog in the multimedia assistant application.
"""

import os
from pathlib import Path

import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QSpinBox
)
from PyQt5.QtCore import QStandardPaths, QThread, pyqtSignal
from ...exceptions import ResizeError


class ImageItem:
    """
    Represents an image file with its path.
    """

    def __init__(self, path: str):
        self.path = path

    def __str__(self):
        return str(Path(self.path).name)


class ResizeConfig:
    """
    Configuration for resizing images.
    """

    def __init__(self,
                 output_dir: str,
                 target_width: int,
                 target_height: int,
                 maintain_ratio: bool):
        self.output_dir = output_dir
        self.target_width = target_width
        self.target_height = target_height
        self.maintain_ratio = maintain_ratio


class ResizeThread(QThread):
    """
    Thread to handle image resizing operations.
    """
    progress = pyqtSignal(int)
    image_progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, images, config: ResizeConfig):
        """
        Initialize the resize thread.

        Args:
            images: List of images to process
            config: ResizeConfig object with resize settings
        """
        super().__init__()
        self.images = images
        self.config = config

    def run(self):
        """
        Execute the image resizing process.
        """
        try:
            self._process_images()
            total_images = len(self.images)
            self.finished.emit(
                True, f"Successfully resized {total_images} images!")
        except Exception as e:
            if not isinstance(e, ResizeError):
                raise ResizeError(str(e))
            raise

    def _process_images(self):
        """
        Process all images in the queue.
        """
        total_images = len(self.images)
        for i, image in enumerate(self.images, 1):
            self._process_single_image(image, i, total_images)

    def _process_single_image(self, image, current_index: int, total_images: int):
        """
        Process a single image.

        Args:
            image: The ImageItem to process
            current_index: The current image index
            total_images: Total number of images to process
        """
        self.image_progress.emit(current_index, total_images)
        progress = (current_index - 1) / total_images * 100
        self.progress.emit(int(progress))

        try:
            # Read image with cv2
            img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ResizeError(f"Failed to load image: {image.path}")

            # Calculate dimensions
            h, w = img.shape[:2]
            if self.config.maintain_ratio:
                # Calculate aspect ratio
                aspect = w / h
                target_aspect = self.config.target_width / self.config.target_height
                if target_aspect > aspect:
                    new_w = int(self.config.target_height * aspect)
                    new_h = self.config.target_height
                else:
                    new_w = self.config.target_width
                    new_h = int(self.config.target_width / aspect)
            else:
                new_w = self.config.target_width
                new_h = self.config.target_height

            # Resize image using Lanczos algorithm for best quality
            resized = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            # Save with maximum quality
            output_path = os.path.join(
                self.config.output_dir,
                f"resized_{Path(image.path).name}"
            )

            # Save with maximum quality settings
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                success = cv2.imwrite(
                    output_path,
                    resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 100]
                )
            elif output_path.lower().endswith('.png'):
                success = cv2.imwrite(
                    output_path,
                    resized,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0]
                )
            else:
                success = cv2.imwrite(output_path, resized)

            if not success:
                raise ResizeError(
                    f"Failed to save resized image: {output_path}")

            progress = current_index / total_images * 100
            self.progress.emit(int(progress))

        except Exception as e:
            if not isinstance(e, ResizeError):
                raise ResizeError(f"Failed to process image: {str(e)}")
            raise


class AspectRatio:
    """
    Represents an aspect ratio option.
    """

    def __init__(self, name: str, ratio: float = None):
        self.name = name
        self.ratio = ratio  # width:height ratio as float, None means free form

    def __str__(self):
        return self.name

    def calculate_height(self, width: int) -> int:
        """
        Calculate the height based on the given width and aspect ratio.

        Args:
            width: The width to calculate height for

        Returns:
            Calculated height as integer
        """
        if self.ratio is None:
            return None
        return int(width / self.ratio)

    def calculate_width(self, height: int) -> int:
        """
        Calculate the width based on the given height and aspect ratio.

        Args:
            height: The height to calculate width for

        Returns:
            Calculated width as integer
        """
        if self.ratio is None:
            return None
        return int(height * self.ratio)


class ResizeDialog(QDialog):
    """
    Dialog for batch resizing images.
    """

    def __init__(self, parent=None):
        """
        Initialize the resize dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Batch Image Resizer")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._updating = False  # Flag to prevent recursive updates
        self.images = []
        self.resize_thread = None  # Initialize here

        # Initialize all attributes
        self.main_layout: QVBoxLayout = QVBoxLayout()
        self.path_input: QLineEdit = QLineEdit()
        self.browse_button: QPushButton = QPushButton("Browse")
        self.add_button: QPushButton = QPushButton("Add")
        self.image_list: QListWidget = QListWidget()
        self.remove_button: QPushButton = QPushButton("Remove Selected")
        self.aspect_combo: QComboBox = QComboBox()
        self.width_spin: QSpinBox = QSpinBox()
        self.height_spin: QSpinBox = QSpinBox()
        self.progress_bar: QProgressBar = QProgressBar()
        self.progress_label: QLabel = QLabel("0/0")
        self.cancel_button: QPushButton = QPushButton("Cancel")
        self.resize_button: QPushButton = QPushButton("Resize")
        self.aspect_ratios = [
            AspectRatio("No aspect ratio"),
            AspectRatio("1:1 (Square)", 1.0),
            AspectRatio("4:3 (Standard)", 4/3),
            AspectRatio("16:9 (Widescreen)", 16/9),
            AspectRatio("16:10 (Widescreen)", 16/10),
            AspectRatio("21:9 (Ultrawide)", 21/9),
            AspectRatio("3:2 (Photo)", 3/2),
            AspectRatio("2:3 (Portrait)", 2/3),
            AspectRatio("5:4 (Monitor)", 5/4),
        ]

        self.setup_ui()

    def setup_ui(self):
        """
        Sets up the user interface components.
        """
        self._setup_layout()
        self._setup_path_section()
        self._setup_image_section()
        self._setup_aspect_ratio_section()
        self._setup_dimensions_section()
        self._setup_progress_section()
        self._setup_action_buttons()
        self._connect_signals()

    def _setup_layout(self):
        """
        Sets up the main layout.
        """
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

    def _setup_path_section(self):
        """
        Sets up the path input section.
        """
        path_layout = QHBoxLayout()
        path_label = QLabel("Images:")
        path_layout.addWidget(path_label)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.add_button)
        path_layout.addWidget(self.browse_button)
        self.main_layout.addLayout(path_layout)

    def _setup_image_section(self):
        """
        Sets up the image list section.
        """
        self.main_layout.addWidget(self.image_list)
        self.main_layout.addWidget(self.remove_button)

    def _setup_aspect_ratio_section(self):
        """
        Sets up the aspect ratio selection section.
        """
        aspect_layout = QHBoxLayout()
        aspect_label = QLabel("Aspect Ratio:")
        self.aspect_combo.addItems([str(ratio)
                                   for ratio in self.aspect_ratios])
        aspect_layout.addWidget(aspect_label)
        aspect_layout.addWidget(self.aspect_combo)
        self.main_layout.addLayout(aspect_layout)

    def _setup_dimensions_section(self):
        """
        Sets up the width and height input section.
        """
        dimensions_layout = QHBoxLayout()

        # Width input
        width_label = QLabel("Width:")
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(1920)
        dimensions_layout.addWidget(width_label)
        dimensions_layout.addWidget(self.width_spin)

        # Height input
        height_label = QLabel("Height:")
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(1080)
        dimensions_layout.addWidget(height_label)
        dimensions_layout.addWidget(self.height_spin)

        self.main_layout.addLayout(dimensions_layout)

    def _setup_progress_section(self):
        """
        Sets up the progress bar and label section.
        """
        progress_layout = QHBoxLayout()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.hide()
        self.progress_label.hide()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        self.main_layout.addLayout(progress_layout)

    def _setup_action_buttons(self):
        """
        Sets up the action buttons section.
        """
        button_layout = QHBoxLayout()
        self.cancel_button.setText("Cancel")
        self.resize_button.setText("Resize")
        self.resize_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.resize_button)
        self.main_layout.addLayout(button_layout)

    def _connect_signals(self):
        """
        Connects all signal handlers.
        """
        self.browse_button.clicked.connect(self.browse_images)
        self.add_button.clicked.connect(self.add_image)
        self.remove_button.clicked.connect(self.remove_image)
        self.cancel_button.clicked.connect(self.reject)
        self.resize_button.clicked.connect(self.resize_images)
        self.width_spin.valueChanged.connect(self.on_width_changed)
        self.height_spin.valueChanged.connect(self.on_height_changed)
        self.aspect_combo.currentIndexChanged.connect(
            self.on_aspect_ratio_changed)

    def browse_images(self):
        """
        Opens a file dialog to select images and adds them to the list.
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
        Adds a single image from the path input to the list.
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
        Removes the selected image from the list.
        """
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            self.image_list.takeItem(current_row)
            self.images.pop(current_row)

    def resize_images(self):
        """
        Initiates the image resizing process in a separate thread.
        """
        if not self.images:
            QMessageBox.warning(
                self, "Error", "Please add at least one image to resize"
            )
            return

        # Ask user for save location
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Location for Resized Images",
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not output_dir:  # User cancelled
            return

        # Show progress bar and disable buttons
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.resize_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.browse_button.setEnabled(False)

        # Create resize configuration
        config = ResizeConfig(
            output_dir=output_dir,
            target_width=self.width_spin.value(),
            target_height=self.height_spin.value(),
            maintain_ratio=True  # Always maintain ratio during resize if aspect ratio is selected
        )

        # Create and start resize thread with user-selected directory
        self.resize_thread = ResizeThread(
            self.images,
            config
        )
        self.resize_thread.progress.connect(self.progress_bar.setValue)
        self.resize_thread.image_progress.connect(self.update_progress_label)
        self.resize_thread.finished.connect(self.resize_finished)
        self.resize_thread.start()

    def update_progress_label(self, current: int, total: int):
        """
        Updates the progress label with the current progress.

        Args:
            current: Current image index
            total: Total number of images
        """
        self.progress_label.setText(f"{current}/{total}")

    def resize_finished(self, success: bool, message: str):
        """
        Handles the completion of the resizing process.

        Args:
            success: Boolean indicating if resizing was successful
            message: Success or error message
        """
        self.progress_bar.hide()
        self.progress_label.hide()
        self.resize_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.browse_button.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            QMessageBox.critical(
                self, "Error", f"Failed to resize images: {message}"
            )

    def get_current_aspect_ratio(self) -> AspectRatio:
        """
        Retrieves the currently selected aspect ratio.

        Returns:
            The selected AspectRatio object
        """
        return self.aspect_ratios[self.aspect_combo.currentIndex()]

    def on_width_changed(self, new_width: int):
        """
        Updates the height spin box based on the new width and selected aspect ratio.

        Args:
            new_width: The new width value
        """
        if self._updating:
            return

        aspect_ratio = self.get_current_aspect_ratio()
        if aspect_ratio.ratio is not None:
            self._updating = True
            new_height = aspect_ratio.calculate_height(new_width)
            if new_height is not None:
                self.height_spin.setValue(new_height)
            self._updating = False

    def on_height_changed(self, new_height: int):
        """
        Updates the width spin box based on the new height and selected aspect ratio.

        Args:
            new_height: The new height value
        """
        if self._updating:
            return

        aspect_ratio = self.get_current_aspect_ratio()
        if aspect_ratio.ratio is not None:
            self._updating = True
            new_width = aspect_ratio.calculate_width(new_height)
            if new_width is not None:
                self.width_spin.setValue(new_width)
            self._updating = False

    def on_aspect_ratio_changed(self, index: int):
        """
        Updates the height spin box based on the selected aspect ratio.

        Args:
            index: The index of the selected aspect ratio
        """
        if index == 0:  # "No aspect ratio" selected
            return

        # Always update height based on current width and new aspect ratio
        self._updating = True
        new_height = self.aspect_ratios[index].calculate_height(
            self.width_spin.value()
        )
        if new_height is not None:
            self.height_spin.setValue(new_height)
        self._updating = False
