"""
Utility functions for handling icons in the application.

This module provides helper functions for generating and manipulating QIcons,
including functionality to convert theme icons to specific colors. It's primarily
used for consistent icon handling across the application's user interface.
"""

from PyQt5.QtGui import QIcon, QColor, QPainter, QPixmap
from PyQt5.QtCore import QSize
import qtawesome as qta

# Map theme icon names to Font Awesome icons
ICON_MAP = {
    'media-playback-start': 'fa5s.play',
    'media-playback-pause': 'fa5s.pause',
    'media-playback-stop': 'fa5s.stop',
    'media-skip-forward': 'fa5s.step-forward',
    'media-skip-backward': 'fa5s.step-backward',
    'media-seek-forward': 'fa5s.forward',
    'media-seek-backward': 'fa5s.backward',
    'folder-open': 'fa5s.folder-open',
    'document-save': 'fa5s.save',
    'camera-photo': 'fa5s.camera',
}

def generateIcon(icon_name: str, fromTheme: bool = False) -> QIcon:
    """
    Generate a QIcon, falling back to Font Awesome if theme icon is not available.

    Args:
        icon_name (str): Name of the icon from the system theme
        fromTheme (bool): If True, return the theme icon directly; if False, convert to black

    Returns:
        QIcon: The generated icon
    """
    icon = QIcon.fromTheme(icon_name)
    
    # If theme icon is not available, use Font Awesome
    if icon.isNull() and icon_name in ICON_MAP:
        return qta.icon(ICON_MAP[icon_name], color='black')
        
    if fromTheme:
        return icon

    # Get pixmap of appropriate size
    pixmap = icon.pixmap(QSize(32, 32))
    if pixmap.isNull():
        return icon

    # Create painter to modify the pixmap
    new_pixmap = QPixmap(pixmap.size())
    new_pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background

    painter = QPainter(new_pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(new_pixmap.rect(), QColor(0, 0, 0))  # Fill with black
    painter.end()

    return QIcon(new_pixmap)


def createColoredIcon(icon_name: str, color: QColor) -> QIcon:
    """
    Create an icon with a specific color from a theme icon.

    Args:
        icon_name (str): Name of the icon from the system theme
        color (QColor): Color to apply to the icon

    Returns:
        QIcon: The colored icon
    """
    icon = QIcon.fromTheme(icon_name)
    pixmap = icon.pixmap(QSize(32, 32))
    if pixmap.isNull():
        return icon

    new_pixmap = QPixmap(pixmap.size())
    new_pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(new_pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(new_pixmap.rect(), color)
    painter.end()

    return QIcon(new_pixmap)
