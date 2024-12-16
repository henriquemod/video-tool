"""
Utility functions for handling icons in the application.

This module provides helper functions for generating and manipulating QIcons,
including functionality to convert theme icons to specific colors. It's primarily
used for consistent icon handling across the application's user interface.

Functions:
    generateIcon: Creates a QIcon from a theme icon name with optional color conversion.
"""
from PyQt5.QtGui import QIcon, QColor, QPainter
from PyQt5.QtCore import QSize


def generateIcon(icon_name, fromTheme=False):
    """
    Generate a QIcon from a theme icon name, optionally converting it to black.

    Args:
        icon_name (str): Name of the icon from the system theme
        fromTheme (bool): If True, return the theme icon directly; if False, convert to black

    Returns:
        QIcon: The generated icon
    """
    icon = QIcon.fromTheme(icon_name)
    if fromTheme:
        return icon
    else:
        pixmap = icon.pixmap(QSize(32, 32))  # Get pixmap of appropriate size
        # Create painter to modify the pixmap
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(0, 0, 0))  # Fill with black
        painter.end()
        return QIcon(pixmap)
