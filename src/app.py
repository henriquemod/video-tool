"""
Main application entry point for the multimedia processing application.
This module initializes the PyQt application and launches the main window.
"""

import sys
from PyQt5 import QtWidgets, QtGui
from .gui.main_window import MainWindow


def setup_application_style(app):
    """
    Set up consistent application-wide styling and fonts.
    
    Args:
        app: QApplication instance to apply the styling to
    """
    # Create and set default application font
    default_font = QtGui.QFont("Segoe UI", 9)  # Segoe UI is available on Windows, fallbacks work on other OS
    app.setFont(default_font)

    # Apply stylesheet for consistent look
    stylesheet = """
        QWidget {
            font-size: 9pt;
        }
        QLabel {
            font-size: 9pt;
        }
        QPushButton {
            font-size: 9pt;
            padding: 5px;
        }
        QMenuBar {
            font-size: 9pt;
        }
        QMenu {
            font-size: 9pt;
        }
    """
    app.setStyleSheet(stylesheet)


def run_app():
    """
    Initialize and run the main application.

    Creates the QApplication instance, instantiates the main window,
    displays it, and starts the event loop.

    Returns:
        None. The function will exit the application when the main window is closed.
    """
    app = QtWidgets.QApplication(sys.argv)
    setup_application_style(app)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
