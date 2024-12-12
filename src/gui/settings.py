from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QPushButton, QFileDialog, QLabel

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")

        layout = QVBoxLayout()

        # Checkbox for cropping option
        self.cropCheckbox = QCheckBox("Allow image cropping")
        layout.addWidget(self.cropCheckbox)

        # Button to select output folder
        self.outputFolderLabel = QLabel("Output Folder: Not selected")
        layout.addWidget(self.outputFolderLabel)
        self.selectFolderButton = QPushButton("Select Output Folder")
        self.selectFolderButton.clicked.connect(self.select_output_folder)
        layout.addWidget(self.selectFolderButton)

        # Save button
        self.saveButton = QPushButton("Save Settings")
        self.saveButton.clicked.connect(self.accept)
        layout.addWidget(self.saveButton)

        self.setLayout(layout)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.outputFolderLabel.setText(f"Output Folder: {folder}") 