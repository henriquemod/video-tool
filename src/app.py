"""
Main application entry point for the multimedia processing application.
This module initializes the PyQt application and launches the main window.
"""

import sys
from PyQt5 import QtWidgets
from .gui.main_window import MainWindow


def run_app():
    """
    Initialize and run the main application.

    Creates the QApplication instance, instantiates the main window,
    displays it, and starts the event loop.

    Returns:
        None. The function will exit the application when the main window is closed.
    """
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
