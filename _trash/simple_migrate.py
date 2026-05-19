import sys

# Read gui.py
with open('gui.py', 'r', encoding='utf-8') as f:
    gui = f.read()

#Simple replacements
gui = gui.replace('from PyQt5.QtWidgets import *', '''from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QAbstractItemView, QScrollArea, QTabWidget, QFrame, QFileDialog, QMessageBox,
    QDialog, QSizePolicy, QApplication, QMainWindow, QStatusBar, QLineEdit,
    QProgressBar, QGroupBox, QFormLayout, QGridLayout, QListView, QThreadPool
)''')
gui = gui.replace('from PyQt5.QtCore import *', '''from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QPoint, QRect,
    QEvent, QRunnable, QThreadPool
)''')
gui = gui.replace('from PyQt5.QtGui import *', '''from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen, QBrush
)''')

# Enum replacements
enums = {
    'Qt.WindowStaysOnTopHint': 'Qt.WindowType.WindowStaysOnTopHint',
    'Qt.FramelessWindowHint': 'Qt.WindowType.FramelessWindowHint',
    'Qt.Tool': 'Qt.WindowType.Tool',
    'Qt.ToolTip': 'Qt.WindowType.ToolTip',
    'Qt.WA_TranslucentBackground': 'Qt.WidgetAttribute.WA_TranslucentBackground',
    'Qt.WA_TransparentForMouseEvents': 'Qt.WidgetAttribute.WA_TransparentForMouseEvents',
    'Qt.Horizontal': 'Qt.Orientation.Horizontal',
    'Qt.Vertical': 'Qt.Orientation.Vertical',
    'Qt.AlignHCenter': 'Qt.AlignmentFlag.AlignHCenter',
    'Qt.AlignVCenter': 'Qt.AlignmentFlag.AlignVCenter',
    'Qt.AlignCenter': 'Qt.AlignmentFlag.AlignCenter',
    'Qt.AlignLeft': 'Qt.AlignmentFlag.AlignLeft',
    'Qt.AlignRight': 'Qt.AlignmentFlag.AlignRight',
    'Qt.AlignTop': 'Qt.AlignmentFlag.AlignTop',
    'Qt.AlignBottom': 'Qt.AlignmentFlag.AlignBottom',
    'Qt.KeepAspectRatio': 'Qt.AspectRatioMode.KeepAspectRatio',
    'Qt.SmoothTransformation': 'Qt.TransformationMode.SmoothTransformation',
    'Qt.UserRole': 'Qt.ItemDataRole.UserRole',
    'Qt.StrongFocus': 'Qt.FocusPolicy.StrongFocus',
    'Qt.NoFocus': 'Qt.FocusPolicy.NoFocus',
    'Qt.ScrollBarAlwaysOff': 'Qt.ScrollBarPolicy.ScrollBarAlwaysOff',
    'Qt.ScrollBarAsNeeded': 'Qt.ScrollBarPolicy.ScrollBarAsNeeded',
    'Qt.LeftButton': 'Qt.MouseButton.LeftButton',
    'Qt.RightButton': 'Qt.MouseButton.RightButton',
    'Qt.Key_Escape': 'Qt.Key.Key_Escape',
    'QFrame.HLine': 'QFrame.Shape.HLine',
    'QFrame.VLine': 'QFrame.Shape.VLine',
    'QFrame.Sunken': 'QFrame.Shadow.Sunken',
    'QFrame.NoFrame': 'QFrame.Shape.NoFrame',
    'QSizePolicy.Fixed': 'QSizePolicy.Policy.Fixed',
    'QSizePolicy.Expanding': 'QSizePolicy.Policy.Expanding',
    'QSizePolicy.Preferred': 'QSizePolicy.Policy.Preferred',
    'QAbstractItemView.SingleSelection': 'QAbstractItemView.SelectionMode.SingleSelection',
    'QDialog.Accepted': 'QDialog.DialogCode.Accepted',
    'QMessageBox.Yes': 'QMessageBox.StandardButton.Yes',
    'QMessageBox.No': 'QMessageBox.StandardButton.No',
    'QSlider.TicksBelow': 'QSlider.TickPosition.TicksBelow',
    'QSlider.TicksAbove': 'QSlider.TickPosition.TicksAbove',
    'QTabWidget.North': 'QTabWidget.TabPosition.North',
    'QTabWidget.South': 'QTabWidget.TabPosition.South',
    'QTabWidget.East': 'QTabWidget.TabPosition.East',
    'QTabWidget.West': 'QTabWidget.TabPosition.West',
    'app.exec_()': 'app.exec()',
}

for old, new in enums.items():
    gui = gui.replace(old, new)

with open('gui.py', 'w', encoding='utf-8') as f:
    f.write(gui)

# Do same for main.py
with open('main.py', 'r', encoding='utf-8') as f:
    main = f.read()

main = main.replace('from PyQt5.QtWidgets import *', '''from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QAbstractItemView, QScrollArea, QTabWidget, QFrame, QFileDialog, QMessageBox,
    QDialog, QSizePolicy, QApplication, QMainWindow, QStatusBar, QLineEdit,
    QProgressBar, QGroupBox, QFormLayout, QGridLayout, QListView, QThreadPool
)''')
main = main.replace('from PyQt5.QtCore import *', '''from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QPoint, QRect,
    QEvent, QRunnable, QThreadPool
)''')
main = main.replace('from PyQt5.QtGui import *', '''from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen, QBrush
)''')

for old, new in enums.items():
    main = main.replace(old, new)

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(main)

print("Done")
