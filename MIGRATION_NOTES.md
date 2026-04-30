# PyQt5 to PyQt6 Migration Notes

## Migration Strategy

This migration follows a careful preservation approach:
- All handler methods are preserved verbatim (only enum paths change)
- All business logic (Panda3D, file operations, models, textures) is untouched
- UI construction is modernized to use PyQt6 enums
- Russian comments are preserved
- Wild card imports are replaced with explicit imports

## Enum Path Changes Applied

- `Qt.Horizontal` → `Qt.Orientation.Horizontal`
- `Qt.Vertical` → `Qt.Orientation.Vertical`
- `Qt.AlignCenter` → `Qt.AlignmentFlag.AlignCenter`
- `Qt.AlignLeft` → `Qt.AlignmentFlag.AlignLeft`
- `Qt.AlignRight` → `Qt.AlignmentFlag.AlignRight`
- `Qt.AlignTop` → `Qt.AlignmentFlag.AlignTop`
- `Qt.AlignBottom` → `Qt.AlignmentFlag.AlignBottom`
- `Qt.AlignVCenter` → `Qt.AlignmentFlag.AlignVCenter`
- `Qt.AlignHCenter` → `Qt.AlignmentFlag.AlignHCenter`
- `Qt.KeepAspectRatio` → `Qt.AspectRatioMode.KeepAspectRatio`
- `Qt.SmoothTransformation` → `Qt.TransformationMode.SmoothTransformation`
- `Qt.FramelessWindowHint` → `Qt.WindowType.FramelessWindowHint`
- `Qt.Tool` → `Qt.WindowType.Tool`
- `Qt.WindowStaysOnTopHint` → `Qt.WindowType.WindowStaysOnTopHint`
- `Qt.ToolTip` → `Qt.WindowType.ToolTip`
- `Qt.WA_TranslucentBackground` → `Qt.WidgetAttribute.WA_TranslucentBackground`
- `Qt.WA_TransparentForMouseEvents` → `Qt.WidgetAttribute.WA_TransparentForMouseEvents`
- `Qt.UserRole` → `Qt.ItemDataRole.UserRole`
- `Qt.NoFocus` → `Qt.FocusPolicy.NoFocus`
- `Qt.ScrollBarAlwaysOff` → `Qt.ScrollBarPolicy.ScrollBarAlwaysOff`
- `Qt.ScrollBarAsNeeded` → `Qt.ScrollBarPolicy.ScrollBarAsNeeded`
- `Qt.LeftButton` → `Qt.MouseButton.LeftButton`
- `Qt.RightButton` → `Qt.MouseButton.RightButton`
- `QFrame.HLine` → `QFrame.Shape.HLine`
- `QFrame.VLine` → `QFrame.Shape.VLine`
- `QFrame.Sunken` → `QFrame.Shadow.Sunken`
- `QFrame.NoFrame` → `QFrame.Shape.NoFrame`
- `QSizePolicy.Fixed` → `QSizePolicy.Policy.Fixed`
- `QSizePolicy.Expanding` → `QSizePolicy.Policy.Expanding`
- `QSizePolicy.Preferred` → `QSizePolicy.Policy.Preferred`
- `QAbstractItemView.SingleSelection` → `QAbstractItemView.SelectionMode.SingleSelection`
- `QListWidget.Static` → `QListView.Movement.Static`
- `QDialog.Accepted` → `QDialog.DialogCode.Accepted`
- `QMessageBox.Yes/No` → `QMessageBox.StandardButton.Yes/No`
- `app.exec_()` → `app.exec()`

## Imports Changed

PyQt5:
```python
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
```

PyQt6:
```python
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget, QListWidgetItem,
    QAbstractItemView, QScrollArea, QTabWidget, QFrame, QFileDialog, QMessageBox,
    QDialog, QSizePolicy, QApplication, QMainWindow, QStatusBar, QLineEdit,
    QProgressBar, QGroupBox, QFormLayout, QGridLayout, QListView, QThreadPool
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QObject, QSize, QPoint, QRect,
    QEvent, QRunnable, QThreadPool
)
from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QImage, QIcon, QPainter, QPen, QBrush
)
```

## Key Files

- `gui.py`: CameraControlGUI class with all tabs and handlers
- `main.py`: Application entry point and MyApp ShowBase class

## Handler Methods Preserved

All 23 handler methods are preserved with exact logic:
- update_min_depth, update_max_depth, toggle_depth_overlay
- generate_perlin_mesh, perform_AABB_plane, create_ground_plane
- change_plane_size_x, change_plane_size_y, change_plane_position
- change_view, load_model, save_scene, load_scene
- toggle_drag_drop, change_drag_sensitivity
- update_target_volume, run_full_process
- on_texture_set_changed, on_model_set_changed
- load_models_config_from_server, load_selected_model_set
- show_fullscreen_image, show_image_overlay

## TODO/Notes for User

1. Test Qt enum conversions thoroughly, especially:
   - Slider tick positions (QSlider.TickPosition)
   - TabWidget positioning (QTabWidget.TabPosition)
   - Dialog return codes

2. Verify all signal/slot connections work properly

3. Check file dialogs return tuples correctly

4. Test fullscreen overlays and hover tooltips

5. Verify all PyQt5-specific widget attributes work in PyQt6

