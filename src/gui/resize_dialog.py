from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QMessageBox, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QProgressDialog, QCheckBox, QSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
from pathlib import Path


class ImageItem:
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return str(Path(self.path).name)


class ResizeThread(QThread):
    progress = pyqtSignal(int)
    image_progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, images, output_dir, target_width, target_height, maintain_ratio):
        super().__init__()
        self.images = images
        self.output_dir = output_dir
        self.target_width = target_width
        self.target_height = target_height
        self.maintain_ratio = maintain_ratio

    def run(self):
        try:
            total_images = len(self.images)
            for i, image in enumerate(self.images, 1):
                self.image_progress.emit(i, total_images)
                progress = (i - 1) / total_images * 100
                self.progress.emit(int(progress))

                # Read image with cv2
                img = cv2.imread(image.path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise Exception(f"Failed to load image: {image.path}")

                # Calculate dimensions
                h, w = img.shape[:2]
                if self.maintain_ratio:
                    # Calculate aspect ratio
                    aspect = w / h
                    if self.target_width / self.target_height > aspect:
                        new_w = int(self.target_height * aspect)
                        new_h = self.target_height
                    else:
                        new_w = self.target_width
                        new_h = int(self.target_width / aspect)
                else:
                    new_w = self.target_width
                    new_h = self.target_height

                # Resize image using Lanczos algorithm for best quality
                resized = cv2.resize(img, (new_w, new_h),
                                     interpolation=cv2.INTER_LANCZOS4)

                # Save with maximum quality
                output_path = os.path.join(
                    self.output_dir,
                    f"resized_{Path(image.path).name}"
                )

                # Save with maximum quality settings
                if output_path.lower().endswith(('.jpg', '.jpeg')):
                    cv2.imwrite(output_path, resized, [
                                cv2.IMWRITE_JPEG_QUALITY, 100])
                elif output_path.lower().endswith('.png'):
                    cv2.imwrite(output_path, resized, [
                                cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(output_path, resized)

                progress = i / total_images * 100
                self.progress.emit(int(progress))

            self.finished.emit(
                True, f"Successfully resized {total_images} images!")
        except Exception as e:
            self.finished.emit(False, str(e))


class AspectRatio:
    def __init__(self, name, ratio=None):
        self.name = name
        self.ratio = ratio  # width:height ratio as float, None means free form

    def __str__(self):
        return self.name

    def calculate_height(self, width):
        if self.ratio is None:
            return None
        return int(width / self.ratio)

    def calculate_width(self, height):
        if self.ratio is None:
            return None
        return int(height * self.ratio)


class ResizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Image Resizer")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._updating = False  # Flag to prevent recursive updates
        self.images = []
        self.setup_ui()

    def setup_ui(self):
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

        # Define common aspect ratios
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

        # Resolution options
        resolution_layout = QVBoxLayout()  # Changed to vertical layout

        # Aspect ratio selection
        aspect_layout = QHBoxLayout()
        aspect_label = QLabel("Aspect Ratio:")
        self.aspect_combo = QComboBox()
        for ratio in self.aspect_ratios:
            self.aspect_combo.addItem(str(ratio))
        aspect_layout.addWidget(aspect_label)
        aspect_layout.addWidget(self.aspect_combo)
        resolution_layout.addLayout(aspect_layout)

        # Dimensions layout
        dimensions_layout = QHBoxLayout()

        # Width input
        width_label = QLabel("Width:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(1920)
        dimensions_layout.addWidget(width_label)
        dimensions_layout.addWidget(self.width_spin)

        # Height input
        height_label = QLabel("Height:")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(1080)
        dimensions_layout.addWidget(height_label)
        dimensions_layout.addWidget(self.height_spin)

        resolution_layout.addLayout(dimensions_layout)
        layout.addLayout(resolution_layout)

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
        self.resize_button = QPushButton("Resize")
        self.resize_button.setDefault(True)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.resize_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect signals
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
                        self, "Error", f"File not found: {path}")
                    continue

                image = ImageItem(path)
                self.images.append(image)
                self.image_list.addItem(QListWidgetItem(str(image)))

    def add_image(self):
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
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            self.image_list.takeItem(current_row)
            self.images.pop(current_row)

    def resize_images(self):
        if not self.images:
            QMessageBox.warning(
                self, "Error", "Please add at least one image to resize")
            return

        output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))), 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Show progress bar and disable buttons
        self.progress_bar.show()
        self.progress_label.show()
        self.progress_bar.setValue(0)
        self.resize_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.browse_button.setEnabled(False)

        # Create and start resize thread
        self.resize_thread = ResizeThread(
            self.images,
            output_dir,
            self.width_spin.value(),
            self.height_spin.value(),
            True  # Always maintain ratio during resize if aspect ratio is selected
        )
        self.resize_thread.progress.connect(self.progress_bar.setValue)
        self.resize_thread.image_progress.connect(self.update_progress_label)
        self.resize_thread.finished.connect(self.resize_finished)
        self.resize_thread.start()

    def update_progress_label(self, current, total):
        self.progress_label.setText(f"{current}/{total}")

    def resize_finished(self, success, message):
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
                self, "Error", f"Failed to resize images: {message}")

    def get_current_aspect_ratio(self):
        return self.aspect_ratios[self.aspect_combo.currentIndex()]

    def on_width_changed(self, new_width):
        if self._updating:
            return

        aspect_ratio = self.get_current_aspect_ratio()
        if aspect_ratio.ratio is not None:
            self._updating = True
            new_height = aspect_ratio.calculate_height(new_width)
            self.height_spin.setValue(new_height)
            self._updating = False

    def on_height_changed(self, new_height):
        if self._updating:
            return

        aspect_ratio = self.get_current_aspect_ratio()
        if aspect_ratio.ratio is not None:
            self._updating = True
            new_width = aspect_ratio.calculate_width(new_height)
            self.width_spin.setValue(new_width)
            self._updating = False

    def on_aspect_ratio_changed(self, index):
        if index == 0:  # "No aspect ratio" selected
            return

        # Always update height based on current width and new aspect ratio
        self._updating = True
        new_height = self.aspect_ratios[index].calculate_height(
            self.width_spin.value())
        self.height_spin.setValue(new_height)
        self._updating = False
