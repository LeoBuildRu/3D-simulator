#!/usr/bin/env python3
"""
Inline PyQt5 to PyQt6 migration for gui.py and main.py
"""

import re

def migrate(content):
    """Apply PyQt5 to PyQt6 migrations to content."""

    # Replace imports
    content = content.replace(
        'from PyQt5.QtWidgets import *',
        '''from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QAbstractItemView, QScrollArea, QTabWidget, QFrame, QFileDialog, QMessageBox,
    QDialog, QSizePolicy, QApplication, QMainWindow, QStatusBar, QLineEdit,
    QProgressBar, QGroupBox, QFormLayout, QGridLayout, QListView, QThreadPool
)'''
    )

    content = content.replace(
        'from PyQt5.QtCore import *',
        '''from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QPoint, QRect,
    QEvent, QRunnable, QThreadPool
)'''
    )

    content = content.replace(
        'from PyQt5.QtGui import *',
        '''from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen, QBrush
)'''
    )

    # Module path replacements
    content = content.replace('from PyQt5', 'from PyQt6')
    content = content.replace('PyQt5.QtWidgets', 'PyQt6.QtWidgets')
    content = content.replace('PyQt5.QtCore', 'PyQt6.QtCore')
    content = content.replace('PyQt5.QtGui', 'PyQt6.QtGui')

    # Enum replacements - IMPORTANT: order matters, do longest patterns first
    replacements = [
        # Window types
        ('Qt.WindowStaysOnTopHint', 'Qt.WindowType.WindowStaysOnTopHint'),
        ('Qt.FramelessWindowHint', 'Qt.WindowType.FramelessWindowHint'),
        ('Qt.WindowType.Tool', 'Qt.WindowType.Tool'),  already updated
        ('Qt.Tool', 'Qt.WindowType.Tool'),
        ('Qt.ToolTip', 'Qt.WindowType.ToolTip'),

        # Widget attributes
        ('Qt.WA_TranslucentBackground', 'Qt.WidgetAttribute.WA_TranslucentBackground'),
        ('Qt.WA_TransparentForMouseEvents', 'Qt.WidgetAttribute.WA_TransparentForMouseEvents'),

        # Orientation
        ('Qt.Horizontal', 'Qt.Orientation.Horizontal'),
        ('Qt.Vertical', 'Qt.Orientation.Vertical'),

        # Alignment
        ('Qt.AlignHCenter', 'Qt.AlignmentFlag.AlignHCenter'),
        ('Qt.AlignVCenter', 'Qt.AlignmentFlag.AlignVCenter'),
        ('Qt.AlignCenter', 'Qt.AlignmentFlag.AlignCenter'),
        ('Qt.AlignLeft', 'Qt.AlignmentFlag.AlignLeft'),
        ('Qt.AlignRight', 'Qt.AlignmentFlag.AlignRight'),
        ('Qt.AlignTop', 'Qt.AlignmentFlag.AlignTop'),
        ('Qt.AlignBottom', 'Qt.AlignmentFlag.AlignBottom'),

        # Aspect ratio
        ('Qt.KeepAspectRatio', 'Qt.AspectRatioMode.KeepAspectRatio'),

        # Transformation
        ('Qt.SmoothTransformation', 'Qt.TransformationMode.SmoothTransformation'),

        # Item role
        ('Qt.UserRole', 'Qt.ItemDataRole.UserRole'),

        # Focus policy
        ('Qt.StrongFocus', 'Qt.FocusPolicy.StrongFocus'),
        ('Qt.NoFocus', 'Qt.FocusPolicy.NoFocus'),

        # Scroll bar policy
        ('Qt.ScrollBarAlwaysOff', 'Qt.ScrollBarPolicy.ScrollBarAlwaysOff'),
        ('Qt.ScrollBarAsNeeded', 'Qt.ScrollBarPolicy.ScrollBarAsNeeded'),

        # Mouse buttons
        ('Qt.LeftButton', 'Qt.MouseButton.LeftButton'),
        ('Qt.RightButton', 'Qt.MouseButton.RightButton'),

        # Key
        ('Qt.Key_Escape', 'Qt.Key.Key_Escape'),

        # QFrame
        ('QFrame.HLine', 'QFrame.Shape.HLine'),
        ('QFrame.VLine', 'QFrame.Shape.VLine'),
        ('QFrame.Sunken', 'QFrame.Shadow.Sunken'),
        ('QFrame.NoFrame', 'QFrame.Shape.NoFrame'),

        # QSizePolicy
        ('QSizePolicy.Fixed', 'QSizePolicy.Policy.Fixed'),
        ('QSizePolicy.Expanding', 'QSizePolicy.Policy.Expanding'),
        ('QSizePolicy.Preferred', 'QSizePolicy.Policy.Preferred'),

        # QAbstractItemView
        ('QAbstractItemView.SingleSelection', 'QAbstractItemView.SelectionMode.SingleSelection'),

        # QDialog
        ('QDialog.Accepted', 'QDialog.DialogCode.Accepted'),

        # QMessageBox
        ('QMessageBox.Yes', 'QMessageBox.StandardButton.Yes'),
        ('QMessageBox.No', 'QMessageBox.StandardButton.No'),

        # QSlider
        ('QSlider.TicksBelow', 'QSlider.TickPosition.TicksBelow'),
        ('QSlider.TicksAbove', 'QSlider.TickPosition.TicksAbove'),

        # QTabWidget
        ('QTabWidget.North', 'QTabWidget.TabPosition.North'),
        ('QTabWidget.South', 'QTabWidget.TabPosition.South'),
        ('QTabWidget.East', 'QTabWidget.TabPosition.East'),
        ('QTabWidget.West', 'QTabWidget.TabPosition.West'),

        # Application methods
        ('app.exec_()', 'app.exec()'),
        ('.exec_()', '.exec()'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    return content

# Read gui.py
with open(r'C:\Users\larle\toner_project\gui.py', 'r', encoding='utf-8') as f:
    gui_content = f.read()

# Read main.py
with open(r'C:\Users\larle\toner_project\main.py', 'r', encoding='utf-8') as f:
    main_content = f.read()

# Migrate both
gui_migrated = migrate(gui_content)
main_migrated = migrate(main_content)

# Write migrated versions
with open(r'C:\Users\larle\toner_project\gui.py', 'w', encoding='utf-8') as f:
    f.write(gui_migrated)

with open(r'C:\Users\larle\toner_project\main.py', 'w', encoding='utf-8') as f:
    f.write(main_migrated)

print("Migration complete!")
print("- Updated gui.py")
print("- Updated main.py")
