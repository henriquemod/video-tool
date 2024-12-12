from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QComboBox, QLabel, QWidget
)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

class CropDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setModal(True)
        self.original_pixmap = pixmap
        self.current_pixmap = pixmap.copy()
        self.crop_rect = None
        self.dragging = False
        self.start_pos = None
        
        # Setup UI
        layout = QVBoxLayout(self)
        
        # Add aspect ratio selector
        ratio_layout = QHBoxLayout()
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["Free Form", "16:9", "4:3", "1:1"])
        self.ratio_combo.currentTextChanged.connect(self.on_ratio_changed)
        ratio_layout.addWidget(QLabel("Aspect Ratio:"))
        ratio_layout.addWidget(self.ratio_combo)
        ratio_layout.addStretch()
        layout.addLayout(ratio_layout)
        
        # Add preview area
        self.preview = QLabel()
        self.preview.setMinimumSize(640, 480)
        self.preview.setPixmap(self.current_pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        layout.addWidget(self.preview)
        
        # Add buttons
        button_layout = QHBoxLayout()
        crop_button = QPushButton("Crop")
        cancel_button = QPushButton("Cancel")
        crop_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(crop_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.preview.setMouseTracking(True)
        self.current_ratio = None
    
    def get_cropped_pixmap(self):
        if self.crop_rect:
            # Convert preview coordinates to original image coordinates
            preview_rect = self.preview.rect()
            scale_x = self.original_pixmap.width() / preview_rect.width()
            scale_y = self.original_pixmap.height() / preview_rect.height()
            
            original_rect = QRect(
                int(self.crop_rect.x() * scale_x),
                int(self.crop_rect.y() * scale_y),
                int(self.crop_rect.width() * scale_x),
                int(self.crop_rect.height() * scale_y)
            )
            return self.original_pixmap.copy(original_rect)
        return None

    def on_ratio_changed(self, ratio_text):
        if ratio_text == "Free Form":
            self.current_ratio = None
        elif ratio_text == "16:9":
            self.current_ratio = 16/9
        elif ratio_text == "4:3":
            self.current_ratio = 4/3
        elif ratio_text == "1:1":
            self.current_ratio = 1
        self.update_crop_rect()

    def update_crop_rect(self):
        if self.crop_rect and self.current_ratio:
            new_width = self.crop_rect.height() * self.current_ratio
            self.crop_rect.setWidth(int(new_width))
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.start_pos = event.pos() - self.preview.pos()
            self.crop_rect = QRect(self.start_pos, QPoint(0, 0))
            self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            current_pos = event.pos() - self.preview.pos()
            self.crop_rect = QRect(self.start_pos, current_pos).normalized()
            if self.current_ratio:
                new_width = self.crop_rect.height() * self.current_ratio
                self.crop_rect.setWidth(int(new_width))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.crop_rect:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(self.crop_rect.translated(self.preview.pos())) 