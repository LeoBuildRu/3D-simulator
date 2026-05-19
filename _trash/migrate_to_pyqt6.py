#!/usr/bin/env python3
"""PyQt5 to PyQt6 migration script for gui.py"""

import re
import sys

# Read the file
gui_path = r"C:\Users\larle\toner_project\gui.py"
with open(gui_path, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content
replacement_count = 0

def replace_with_count(pattern, replacement, text, flags=0):
    """Replace and count replacements"""
    global replacement_count
    new_text, count = re.subn(pattern, replacement, text, flags=flags)
    replacement_count += count
    return new_text

# Order matters: do specific replacements before more general ones

# 1. Qt enum replacements (use word boundaries to avoid double-replacement)
replacements = [
    # Orientation
    (r'\bQt\.Horizontal\b', r'Qt.Orientation.Horizontal'),
    (r'\bQt\.Vertical\b', r'Qt.Orientation.Vertical'),

    # Alignment
    (r'\bQt\.AlignCenter\b', r'Qt.AlignmentFlag.AlignCenter'),
    (r'\bQt\.AlignLeft\b', r'Qt.AlignmentFlag.AlignLeft'),
    (r'\bQt\.AlignRight\b', r'Qt.AlignmentFlag.AlignRight'),
    (r'\bQt\.AlignTop\b', r'Qt.AlignmentFlag.AlignTop'),
    (r'\bQt\.AlignBottom\b', r'Qt.AlignmentFlag.AlignBottom'),
    (r'\bQt\.AlignVCenter\b', r'Qt.AlignmentFlag.AlignVCenter'),
    (r'\bQt\.AlignHCenter\b', r'Qt.AlignmentFlag.AlignHCenter'),

    # AspectRatioMode
    (r'\bQt\.KeepAspectRatio\b', r'Qt.AspectRatioMode.KeepAspectRatio'),
    (r'\bQt\.IgnoreAspectRatio\b', r'Qt.AspectRatioMode.IgnoreAspectRatio'),
    (r'\bQt\.KeepAspectRatioByExpanding\b', r'Qt.AspectRatioMode.KeepAspectRatioByExpanding'),

    # TransformationMode
    (r'\bQt\.SmoothTransformation\b', r'Qt.TransformationMode.SmoothTransformation'),
    (r'\bQt\.FastTransformation\b', r'Qt.TransformationMode.FastTransformation'),

    # WindowType
    (r'\bQt\.FramelessWindowHint\b', r'Qt.WindowType.FramelessWindowHint'),
    (r'\bQt\.Tool\b', r'Qt.WindowType.Tool'),
    (r'\bQt\.WindowStaysOnTopHint\b', r'Qt.WindowType.WindowStaysOnTopHint'),
    (r'\bQt\.ToolTip\b', r'Qt.WindowType.ToolTip'),
    (r'\bQt\.Dialog\b', r'Qt.WindowType.Dialog'),
    (r'\bQt\.Popup\b', r'Qt.WindowType.Popup'),

    # WidgetAttribute
    (r'\bQt\.WA_TranslucentBackground\b', r'Qt.WidgetAttribute.WA_TranslucentBackground'),
    (r'\bQt\.WA_TransparentForMouseEvents\b', r'Qt.WidgetAttribute.WA_TransparentForMouseEvents'),
    (r'\bQt\.WA_DeleteOnClose\b', r'Qt.WidgetAttribute.WA_DeleteOnClose'),
    (r'\bQt\.WA_StyledBackground\b', r'Qt.WidgetAttribute.WA_StyledBackground'),

    # ItemDataRole
    (r'\bQt\.UserRole\b', r'Qt.ItemDataRole.UserRole'),
    (r'\bQt\.DisplayRole\b', r'Qt.ItemDataRole.DisplayRole'),

    # FocusPolicy
    (r'\bQt\.NoFocus\b', r'Qt.FocusPolicy.NoFocus'),
    (r'\bQt\.StrongFocus\b', r'Qt.FocusPolicy.StrongFocus'),
    (r'\bQt\.ClickFocus\b', r'Qt.FocusPolicy.ClickFocus'),

    # ScrollBarPolicy
    (r'\bQt\.ScrollBarAlwaysOff\b', r'Qt.ScrollBarPolicy.ScrollBarAlwaysOff'),
    (r'\bQt\.ScrollBarAsNeeded\b', r'Qt.ScrollBarPolicy.ScrollBarAsNeeded'),
    (r'\bQt\.ScrollBarAlwaysOn\b', r'Qt.ScrollBarPolicy.ScrollBarAlwaysOn'),

    # MouseButton
    (r'\bQt\.LeftButton\b', r'Qt.MouseButton.LeftButton'),
    (r'\bQt\.RightButton\b', r'Qt.MouseButton.RightButton'),
    (r'\bQt\.MiddleButton\b', r'Qt.MouseButton.MiddleButton'),

    # Key
    (r'\bQt\.Key_Escape\b', r'Qt.Key.Key_Escape'),
    (r'\bQt\.Key_Return\b', r'Qt.Key.Key_Return'),
    (r'\bQt\.Key_Enter\b', r'Qt.Key.Key_Enter'),
    (r'\bQt\.Key_Delete\b', r'Qt.Key.Key_Delete'),

    # ContextMenuPolicy
    (r'\bQt\.CustomContextMenu\b', r'Qt.ContextMenuPolicy.CustomContextMenu'),

    # CursorShape
    (r'\bQt\.PointingHandCursor\b', r'Qt.CursorShape.PointingHandCursor'),
    (r'\bQt\.ArrowCursor\b', r'Qt.CursorShape.ArrowCursor'),

    # GlobalColor (more specific patterns first to avoid conflicts)
    (r'\bQt\.white\b', r'Qt.GlobalColor.white'),
    (r'\bQt\.black\b', r'Qt.GlobalColor.black'),
    (r'\bQt\.red\b', r'Qt.GlobalColor.red'),
    (r'\bQt\.green\b', r'Qt.GlobalColor.green'),
    (r'\bQt\.blue\b', r'Qt.GlobalColor.blue'),
    (r'\bQt\.transparent\b', r'Qt.GlobalColor.transparent'),

    # PenStyle
    (r'\bQt\.NoPen\b', r'Qt.PenStyle.NoPen'),
    (r'\bQt\.SolidLine\b', r'Qt.PenStyle.SolidLine'),

    # BrushStyle
    (r'\bQt\.SolidPattern\b', r'Qt.BrushStyle.SolidPattern'),

    # QFrame
    (r'\bQFrame\.HLine\b', r'QFrame.Shape.HLine'),
    (r'\bQFrame\.VLine\b', r'QFrame.Shape.VLine'),
    (r'\bQFrame\.NoFrame\b', r'QFrame.Shape.NoFrame'),
    (r'\bQFrame\.Box\b', r'QFrame.Shape.Box'),
    (r'\bQFrame\.StyledPanel\b', r'QFrame.Shape.StyledPanel'),
    (r'\bQFrame\.Sunken\b', r'QFrame.Shadow.Sunken'),
    (r'\bQFrame\.Raised\b', r'QFrame.Shadow.Raised'),
    (r'\bQFrame\.Plain\b', r'QFrame.Shadow.Plain'),

    # QSizePolicy
    (r'\bQSizePolicy\.Fixed\b', r'QSizePolicy.Policy.Fixed'),
    (r'\bQSizePolicy\.Expanding\b', r'QSizePolicy.Policy.Expanding'),
    (r'\bQSizePolicy\.Preferred\b', r'QSizePolicy.Policy.Preferred'),
    (r'\bQSizePolicy\.MinimumExpanding\b', r'QSizePolicy.Policy.MinimumExpanding'),
    (r'\bQSizePolicy\.Minimum\b', r'QSizePolicy.Policy.Minimum'),
    (r'\bQSizePolicy\.Maximum\b', r'QSizePolicy.Policy.Maximum'),

    # QAbstractItemView
    (r'\bQAbstractItemView\.SingleSelection\b', r'QAbstractItemView.SelectionMode.SingleSelection'),
    (r'\bQAbstractItemView\.ExtendedSelection\b', r'QAbstractItemView.SelectionMode.ExtendedSelection'),
    (r'\bQAbstractItemView\.NoSelection\b', r'QAbstractItemView.SelectionMode.NoSelection'),
    (r'\bQAbstractItemView\.SelectRows\b', r'QAbstractItemView.SelectionBehavior.SelectRows'),

    # QDialog
    (r'\bQDialog\.Accepted\b', r'QDialog.DialogCode.Accepted'),
    (r'\bQDialog\.Rejected\b', r'QDialog.DialogCode.Rejected'),

    # QMessageBox
    (r'\bQMessageBox\.Yes\b', r'QMessageBox.StandardButton.Yes'),
    (r'\bQMessageBox\.No\b', r'QMessageBox.StandardButton.No'),
    (r'\bQMessageBox\.Ok\b', r'QMessageBox.StandardButton.Ok'),
    (r'\bQMessageBox\.Cancel\b', r'QMessageBox.StandardButton.Cancel'),
    (r'\bQMessageBox\.Warning\b', r'QMessageBox.Icon.Warning'),
    (r'\bQMessageBox\.Critical\b', r'QMessageBox.Icon.Critical'),
    (r'\bQMessageBox\.Information\b', r'QMessageBox.Icon.Information'),
    (r'\bQMessageBox\.Question\b', r'QMessageBox.Icon.Question'),

    # QSlider
    (r'\bQSlider\.TicksBelow\b', r'QSlider.TickPosition.TicksBelow'),
    (r'\bQSlider\.TicksAbove\b', r'QSlider.TickPosition.TicksAbove'),
    (r'\bQSlider\.NoTicks\b', r'QSlider.TickPosition.NoTicks'),

    # QTabWidget
    (r'\bQTabWidget\.North\b', r'QTabWidget.TabPosition.North'),
    (r'\bQTabWidget\.South\b', r'QTabWidget.TabPosition.South'),

    # QListView
    (r'\bQListView\.Static\b', r'QListView.Movement.Static'),
]

for pattern, replacement in replacements:
    content = replace_with_count(pattern, replacement, content)

# 2. Replace .exec_() with .exec()
content = replace_with_count(r'\.exec_\(\)', r'.exec()', content)

# 3. Replace imports
old_imports = """from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *"""

new_imports = """from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem, QAbstractItemView,
    QScrollArea, QTabWidget, QFrame, QFileDialog, QMessageBox, QDialog, QSizePolicy,
    QApplication, QMainWindow, QStatusBar, QLineEdit, QProgressBar, QGroupBox,
    QFormLayout, QGridLayout, QListView, QStyle, QStyledItemDelegate, QStyleOptionViewItem
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QPoint, QRect, QEvent,
    QRunnable, QThreadPool, QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen, QBrush, QAction,
    QCursor, QFontMetrics, QPalette
)"""

content = content.replace(old_imports, new_imports)
replacement_count += 1  # Count as one replacement

# 4. Check for any remaining PyQt5 references and replace them with PyQt6
content = replace_with_count(r'PyQt5', 'PyQt6', content)

# Write the file
with open(gui_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Total replacements made: {replacement_count}")
print(f"Migration complete!")

# Verify: check for any remaining PyQt5 references
if 'PyQt5' in content:
    print("WARNING: PyQt5 references still found in file!")
    sys.exit(1)
else:
    print("Verified: No PyQt5 references remain")

# Check for unqualified enum usages (basic check)
unqualified_patterns = [
    r'\bQt\.Align[A-Z]',
    r'\bQt\.Key_',
    r'\bQFrame\.[A-Z][a-zA-Z]+(?!\.)',
]
found_unqualified = False
for pattern in unqualified_patterns:
    matches = re.findall(pattern, content)
    if matches:
        print(f"Potential unqualified enums found for pattern {pattern}: {matches[:5]}")
        found_unqualified = True

if not found_unqualified:
    print("Verified: No bare Qt enum usages without proper qualification")

sys.exit(0)
