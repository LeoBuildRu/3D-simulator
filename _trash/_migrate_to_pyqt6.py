#!/usr/bin/env python3
"""
PyQt5 to PyQt6 migration script for gui.py and main.py
This script performs systematic enum and import replacements.
"""

import re
import os

def migrate_file(input_path, output_path):
    """Read a file and apply PyQt5->PyQt6 migrations."""

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

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

    # Replace PyQt5 module paths with PyQt6 where they appear
    content = content.replace('PyQt5.QtWidgets', 'PyQt6.QtWidgets')
    content = content.replace('PyQt5.QtCore', 'PyQt6.QtCore')
    content = content.replace('PyQt5.QtGui', 'PyQt6.QtGui')
    content = content.replace('PyQt5', 'PyQt6')

    # Replace Qt enum paths
    enum_replacements = {
        # Orientation
        r'\bQt\.Horizontal\b': 'Qt.Orientation.Horizontal',
        r'\bQt\.Vertical\b': 'Qt.Orientation.Vertical',

        # Alignment
        r'\bQt\.AlignCenter\b': 'Qt.AlignmentFlag.AlignCenter',
        r'\bQt\.AlignLeft\b': 'Qt.AlignmentFlag.AlignLeft',
        r'\bQt\.AlignRight\b': 'Qt.AlignmentFlag.AlignRight',
        r'\bQt\.AlignTop\b': 'Qt.AlignmentFlag.AlignTop',
        r'\bQt\.AlignBottom\b': 'Qt.AlignmentFlag.AlignBottom',
        r'\bQt\.AlignVCenter\b': 'Qt.AlignmentFlag.AlignVCenter',
        r'\bQt\.AlignHCenter\b': 'Qt.AlignmentFlag.AlignHCenter',

        # Aspect ratio
        r'\bQt\.KeepAspectRatio\b': 'Qt.AspectRatioMode.KeepAspectRatio',

        # Transformation
        r'\bQt\.SmoothTransformation\b': 'Qt.TransformationMode.SmoothTransformation',

        # Window types
        r'\bQt\.FramelessWindowHint\b': 'Qt.WindowType.FramelessWindowHint',
        r'\bQt\.Tool\b': 'Qt.WindowType.Tool',
        r'\bQt\.WindowStaysOnTopHint\b': 'Qt.WindowType.WindowStaysOnTopHint',
        r'\bQt\.ToolTip\b': 'Qt.WindowType.ToolTip',

        # Widget attributes
        r'\bQt\.WA_TranslucentBackground\b': 'Qt.WidgetAttribute.WA_TranslucentBackground',
        r'\bQt\.WA_TransparentForMouseEvents\b': 'Qt.WidgetAttribute.WA_TransparentForMouseEvents',

        # Item role
        r'\bQt\.UserRole\b': 'Qt.ItemDataRole.UserRole',

        # Focus policy
        r'\bQt\.NoFocus\b': 'Qt.FocusPolicy.NoFocus',
        r'\bQt\.StrongFocus\b': 'Qt.FocusPolicy.StrongFocus',

        # Scroll bar policy
        r'\bQt\.ScrollBarAlwaysOff\b': 'Qt.ScrollBarPolicy.ScrollBarAlwaysOff',
        r'\bQt\.ScrollBarAsNeeded\b': 'Qt.ScrollBarPolicy.ScrollBarAsNeeded',

        # Mouse buttons
        r'\bQt\.LeftButton\b': 'Qt.MouseButton.LeftButton',
        r'\bQt\.RightButton\b': 'Qt.MouseButton.RightButton',

        # Key
        r'\bQt\.Key_Escape\b': 'Qt.Key.Key_Escape',
    }

    for pattern, replacement in enum_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QFrame enums
    frame_replacements = {
        r'\bQFrame\.HLine\b': 'QFrame.Shape.HLine',
        r'\bQFrame\.VLine\b': 'QFrame.Shape.VLine',
        r'\bQFrame\.Sunken\b': 'QFrame.Shadow.Sunken',
        r'\bQFrame\.NoFrame\b': 'QFrame.Shape.NoFrame',
    }

    for pattern, replacement in frame_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QSizePolicy enums
    sizepolicy_replacements = {
        r'\bQSizePolicy\.Fixed\b': 'QSizePolicy.Policy.Fixed',
        r'\bQSizePolicy\.Expanding\b': 'QSizePolicy.Policy.Expanding',
        r'\bQSizePolicy\.Preferred\b': 'QSizePolicy.Policy.Preferred',
    }

    for pattern, replacement in sizepolicy_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QAbstractItemView enums
    itemview_replacements = {
        r'\bQAbstractItemView\.SingleSelection\b': 'QAbstractItemView.SelectionMode.SingleSelection',
    }

    for pattern, replacement in itemview_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QListView enums
    listview_replacements = {
        r'\bQListView\.Movement\.Static\b': 'QListView.Movement.Static',
    }

    for pattern, replacement in listview_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QDialog enums
    dialog_replacements = {
        r'\bQDialog\.Accepted\b': 'QDialog.DialogCode.Accepted',
    }

    for pattern, replacement in dialog_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QMessageBox enums
    msgbox_replacements = {
        r'\bQMessageBox\.Yes\b': 'QMessageBox.StandardButton.Yes',
        r'\bQMessageBox\.No\b': 'QMessageBox.StandardButton.No',
    }

    for pattern, replacement in msgbox_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QSlider enums
    slider_replacements = {
        r'\bQSlider\.TicksBelow\b': 'QSlider.TickPosition.TicksBelow',
        r'\bQSlider\.TicksAbove\b': 'QSlider.TickPosition.TicksAbove',
    }

    for pattern, replacement in slider_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace QTabWidget enums
    tabwidget_replacements = {
        r'\bQTabWidget\.North\b': 'QTabWidget.TabPosition.North',
        r'\bQTabWidget\.South\b': 'QTabWidget.TabPosition.South',
        r'\bQTabWidget\.East\b': 'QTabWidget.TabPosition.East',
        r'\bQTabWidget\.West\b': 'QTabWidget.TabPosition.West',
    }

    for pattern, replacement in tabwidget_replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace app.exec_() with app.exec()
    content = content.replace('app.exec_()', 'app.exec()')
    content = re.sub(r'\.exec_\(\)', '.exec()', content)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Migrated: {input_path} -> {output_path}")

if __name__ == '__main__':
    project_root = r'C:\Users\larle\toner_project'

    # Migrate gui.py
    migrate_file(
        os.path.join(project_root, 'gui.py'),
        os.path.join(project_root, 'gui_pyqt6.py')
    )

    # Migrate main.py
    migrate_file(
        os.path.join(project_root, 'main.py'),
        os.path.join(project_root, 'main_pyqt6.py')
    )

    print("Migration complete!")
