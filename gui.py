# gui.py
import random
import math
import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml
import trimesh

import json
import tempfile
from datetime import datetime

from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    GeomTriangles, NodePath, Vec3, TextureStage, Texture,
    Material, TransparencyAttrib, Shader, GeomVertexReader
)
include_files = [
    ("models", "models"),
    ("textures", "textures"),
    ("PLY_examples", "PLY_examples"),

    ("models_config.yaml", "models_config.yaml"),
    ("textures_config.yaml", "textures_config.yaml"),

    # RenderPipeline целиком (на всякий случай)
    ("render_pipeline", "render_pipeline"),

    # RenderPipeline для MountManager (ОБЯЗАТЕЛЬНЫЕ ПАПКИ)
    ("render_pipeline/config", "lib/config"),
    ("render_pipeline/effects", "lib/effects"),
    ("render_pipeline/data", "lib/data"),
    ("render_pipeline/rpplugins", "lib/rpplugins"),
]
from panda_widget import Panda3DWidget

if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"


class HoverInfoWidget(QWidget):
    """Единый виджет для отображения всей информации при наведении"""
    def __init__(self, pixmap, entry, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 40, 240);
                border: 1px solid #5a5a7a;
                border-radius: 6px;
                color: #e0e0e0;
            }
        """)

        # Вертикальный layout: картинка сверху, текст снизу
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Изображение ---
        img_label = QLabel()
        scaled = pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(scaled)
        img_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(img_label)

        # --- Текстовая информация ---
        text_lines = []
        car_number = entry.get('car_number', 'N/A')
        text_lines.append(f"<b style='font-size:16px;'>🚗 {car_number}</b>")

        time_str = entry.get('time', '')
        if time_str:
            text_lines.append(f"🕒 {time_str}")

        model_name = entry.get('model', '')
        if model_name:
            text_lines.append(f"📦 Модель: {model_name}")

        data_type = entry.get('data_type', '')
        if data_type:
            text_lines.append(f"📊 Тип: {data_type}")

        filler = entry.get('filler', '')
        if filler:
            text_lines.append(f"🧪 Наполнитель: {filler}")

        target_volume = entry.get('target_volume')
        if target_volume:
            text_lines.append(f"📐 Объём: {target_volume}")

        full_text = "<br>".join(text_lines)
        text_label = QLabel(full_text)
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        text_label.setStyleSheet("font-size:13px;")
        main_layout.addWidget(text_label)


class ReconListItemWidget(QWidget):
    # Сигналы для оповещения о наведении/уходе мыши
    entered = pyqtSignal(object, QPoint)  # передаём сам виджет и глобальную позицию его верхнего левого угла
    left = pyqtSignal()
    show_image_requested = pyqtSignal(object)  # запрос на показ полноэкранного изображения

    def __init__(self, car_number, time_str, parent=None):
        super().__init__(parent)
        self.car_number = car_number
        self.time_str = time_str
        self.current_pixmap = None
        self.item = None  # ссылка на QListWidgetItem

        # Включаем отслеживание мыши
        self.setMouseTracking(True)

        # Основной горизонтальный layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # QLabel для иконки фиксированного размера
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(32, 26)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("background-color: #2a2a35; border-radius: 3px;")
        # Устанавливаем временную заглушку
        self.icon_label.setText("⏳")
        layout.addWidget(self.icon_label)

        # Вертикальный layout для текста
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        # Номер автомобиля (жирный шрифт)
        self.car_label = QLabel(car_number)
        self.car_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        text_layout.addWidget(self.car_label)

        # Время (менее яркое)
        self.time_label = QLabel(time_str)
        self.time_label.setStyleSheet("color: #a0a0b0; font-size: 10px;")
        text_layout.addWidget(self.time_label)

        layout.addLayout(text_layout)
        layout.addStretch()

        # Кнопка для полноэкранного просмотра изображения
        self.image_btn = QPushButton("🔍")
        self.image_btn.setFixedSize(20, 20)
        self.image_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a4a;
                border: 1px solid #5a5a7a;
                border-radius: 4px;
                color: white;
                font-size: 11px;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #4a4a5a;
                border: 1px solid #6a6a8a;
            }
        """)
        self.image_btn.clicked.connect(self.on_image_button_clicked)
        layout.addWidget(self.image_btn)

    def enterEvent(self, event):
        """Вызывается, когда мышь входит в область виджета"""
        self.entered.emit(self, self.mapToGlobal(QPoint(0, 0)))
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Вызывается, когда мышь покидает область виджета"""
        self.left.emit()
        super().leaveEvent(event)

    def set_item(self, item):
        """Сохраняет ссылку на соответствующий QListWidgetItem"""
        self.item = item

    def set_icon_pixmap(self, pixmap):
        """Сохраняем pixmap и отображаем его в иконке"""
        self.current_pixmap = pixmap
        if not pixmap.isNull():
            scaled = pixmap.scaled(32, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.icon_label.setPixmap(scaled)
            self.icon_label.setText("")
        else:
            self.icon_label.setText("❌")

    def get_pixmap(self):
        """Возвращает сохранённый pixmap (может быть None)"""
        return self.current_pixmap

    def on_image_button_clicked(self):
        """Обработчик клика по кнопке просмотра изображения"""
        if self.item:
            entry = self.item.data(Qt.UserRole)
            if entry:
                self.show_image_requested.emit(entry)


class ImageDownloadTask(QRunnable):
    class Signals(QObject):
        downloaded = pyqtSignal(QPixmap)

    def __init__(self, client, img_filename, item_widget, temp_dir):
        super().__init__()
        self.client = client
        self.img_filename = img_filename
        self.item_widget = item_widget
        self.temp_dir = temp_dir
        self.signals = self.Signals()
        # Подключаем сигнал к слоту виджета
        self.signals.downloaded.connect(self.item_widget.set_icon_pixmap)

    def run(self):
        if not self.img_filename:
            return
        local_path = os.path.join(self.temp_dir, self.img_filename)
        try:
            self.client.download_file(self.img_filename, local_path)
            pixmap = QPixmap(local_path)
        except Exception as e:
            print(f"Ошибка загрузки изображения {self.img_filename}: {e}")
            pixmap = QPixmap()
        self.signals.downloaded.emit(pixmap)


class ImageOverlay(QWidget):
    """Полноэкранный оверлей с затемнением и изображением по центру"""
    def __init__(self, pixmap, parent=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Затемнение как в show_overlay
        self.setStyleSheet("background-color: rgba(0, 0, 0, 200);")
        self.setFocusPolicy(Qt.StrongFocus)

        # Основной layout центрирует содержимое
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 50, 50, 50)
        main_layout.setAlignment(Qt.AlignCenter)

        # Контейнер для изображения и кнопки (позволяет наложить кнопку поверх)
        self.image_container = QWidget()
        self.image_container.setStyleSheet("background: transparent;")
        container_layout = QGridLayout(self.image_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Изображение
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #5a5a7a; background-color: #1a1a21;")
        container_layout.addWidget(self.image_label, 0, 0)

        # Кнопка закрытия в стиле Chrome (поверх изображения, справа сверху)
        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(36, 36)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(60, 60, 70, 200);
                color: white;
                border: none;
                border-radius: 2px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        container_layout.addWidget(self.close_btn, 0, 0, Qt.AlignRight | Qt.AlignTop)

        main_layout.addWidget(self.image_container)

        self.set_pixmap(pixmap)

    def set_pixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Доступный размер с учётом отступов main_layout (50 пикселей с каждой стороны)
            available = self.size() - QSize(100, 100)
            if available.width() > 0 and available.height() > 0:
                scaled = self.original_pixmap.scaled(available, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scaled_pixmap()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class CameraControlGUI(QWidget):
    def __init__(self, panda_app, main_window=None):
        super().__init__()
        self.panda_app = panda_app
        self.main_window = main_window
        self.panda_widget = Panda3DWidget()

        # Фиксированные значения поворота камеры
        self.fixed_camera_rotation = {
            'h': 0,
            'p': -90.0,
            'r': 0.0
        }

        self.models_config = self.load_models_config()
        self.textures_config = self.load_textures_config()

        self.setup_styles()
        self.init_ui()

        self.status_timer = QTimer()
        self.status_timer.setSingleShot(True)
        self.status_timer.timeout.connect(self.clear_status)

        default_combo = self.textures_config["default"]
        self.on_texture_set_changed(default_combo)
        self.textures_combo.setCurrentText(default_combo)

        self.hide_overlay_timer = QTimer()
        self.hide_overlay_timer.setSingleShot(True)
        self.hide_overlay_timer.timeout.connect(self.hide_overlay)

        self.overlay = None
        self.image_overlay = None  # для полноэкранного просмотра изображений

        # Для всплывающего тултипа
        self.hover_tooltip = None

    def show_overlay(self, message="подождите"):
        """Показать полупрозрачный overlay с логом поверх всего окна"""
        parent = self.main_window if self.main_window is not None else self.window()
        if parent is None:
            return

        if self.overlay is None:
            self.overlay = QWidget(parent)
            self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 200);")
            self.overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self.overlay.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

            layout = QVBoxLayout(self.overlay)
            layout.setAlignment(Qt.AlignCenter)

            # Заголовок
            title = QLabel("Выполняется операция...")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; margin-bottom: 20px;")
            layout.addWidget(title)

            # Метка для текущего сообщения (вместо QTextEdit)
            self.overlay_label = QLabel(message)
            self.overlay_label.setAlignment(Qt.AlignCenter)
            self.overlay_label.setWordWrap(True)
            self.overlay_label.setStyleSheet("""
                color: white;
                font-size: 14pt;
                font-weight: 500;
                padding: 20px;
            """)
            layout.addWidget(self.overlay_label)

            self.overlay.hide()

        # Очищаем и устанавливаем начальное сообщение
        self.overlay_label.setText(message)
        # Устанавливаем размер равным родительскому окну
        self.overlay.setGeometry(parent.rect())
        self.overlay.show()
        self.overlay.raise_()
        QApplication.processEvents()

    def log_message(self, message):
        """Обновить текст в центре оверлея"""
        if hasattr(self, 'overlay_label') and self.overlay_label is not None:
            self.overlay_label.setText(message)
            QApplication.processEvents()

    def hide_overlay(self):
        """Скрыть overlay"""
        if self.overlay is not None:
            self.overlay.hide()
            QApplication.processEvents()

    def load_models_config(self):
        config_path = os.path.join(PROJECT_ROOT, "models_config.yaml")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                    # Обрабатываем относительные пути для всех моделей
                    for model_set in config.values():
                        for key in ['cuzov', 'napolnitel', 'other']:
                            if key in model_set and model_set[key]:
                                # Если путь не абсолютный, делаем его абсолютным относительно корня проекта
                                if not os.path.isabs(model_set[key]):
                                    model_set[key] = os.path.join(PROJECT_ROOT, model_set[key])

                    return config
            else:
                return {}
        except Exception as e:
            print(f"Ошибка загрузки конфигурации моделей: {e}")
            return {}

    def load_textures_config(self):
        config_path = "textures_config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return config
            else:
                return {}
        except Exception as e:
            return {}

    def setup_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0f0f13;
                color: #e0e0e0;
                font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
                font-size: 11px;
                border: none;
            }

            QGroupBox {
                background-color: #1a1a21;
                border: 1px solid #2a2a35;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                padding-bottom: 15px;
            }

            QGroupBox::title {
                color: #a0a0b0;
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                font-weight: 600;
                font-size: 11px;
                letter-spacing: 0.5px;
            }

            QPushButton {
                background-color: #252532;
                color: #d0d0e0;
                border: 1px solid #3a3a4a;
                border-radius: 6px;
                padding: 8px 12px;
                margin: 2px;
                font-weight: 500;
                min-height: 20px;
            }

            QPushButton:hover {
                background-color: #2d2d3a;
                border: 1px solid #4a4a5a;
            }

            QPushButton:pressed {
                background-color: #1d1d2a;
            }

            QPushButton:checked {
                background-color: #3a3a5a;
                border: 1px solid #5a5a7a;
            }

            QPushButton[accent="true"] {
                background-color: #4a7fbe;
                color: #ffffff;
                border: 1px solid #5a8fce;
                font-weight: 600;
            }

            QPushButton[accent="true"]:hover {
                background-color: #5a8fce;
                border: 1px solid #6a9fde;
            }

            QPushButton[accent="true"]:pressed {
                background-color: #3a6fae;
            }

            QPushButton[danger="true"] {
                background-color: #be4a4a;
                color: #ffffff;
                border: 1px solid #ce5a5a;
            }

            QPushButton[danger="true"]:hover {
                background-color: #ce5a5a;
                border: 1px solid #de6a6a;
            }

            QPushButton[mini="true"] {
                padding: 4px 8px;
                font-size: 10px;
                min-height: 16px;
            }

            QDoubleSpinBox, QSpinBox {
                background-color: #1a1a21;
                border: 1px solid #3a3a4a;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e0e0e0;
                min-height: 20px;
            }

            QDoubleSpinBox:hover, QSpinBox:hover {
                border: 1px solid #4a4a5a;
            }

            QDoubleSpinBox::up-button, QSpinBox::up-button {
                background-color: #2a2a35;
                border-left: 1px solid #3a3a4a;
                border-radius: 0px 3px 3px 0px;
                width: 16px;
            }

            QDoubleSpinBox::down-button, QSpinBox::down-button {
                background-color: #2a2a35;
                border-left: 1px solid #3a3a4a;
                border-radius: 0px 3px 3px 0px;
                width: 16px;
            }

            QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
                width: 6px;
                height: 6px;
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-bottom: 6px solid #a0a0b0;
            }

            QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
                width: 6px;
                height: 6px;
                image: none;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 6px solid #a0a0b0;
            }

            QSlider::groove:horizontal {
                background-color: #2a2a35;
                height: 3px;
                border-radius: 1px;
            }

            QSlider::handle:horizontal {
                background-color: #4a7fbe;
                border: 1px solid #5a8fce;
                width: 12px;
                height: 12px;
                border-radius: 6px;
                margin: -5px 0;
            }

            QSlider::handle:horizontal:hover {
                background-color: #5a8fce;
                width: 14px;
                height: 14px;
                border-radius: 7px;
            }

            QSlider::sub-page:horizontal {
                background-color: #4a7fbe;
                border-radius: 1px;
            }

            QLabel {
                color: #b0b0c0;
                padding: 2px 0px;
            }

            QLabel[title="true"] {
                color: #d0d0e0;
                font-weight: 600;
                font-size: 12px;
            }

            QFrame[line="true"] {
                background-color: #2a2a35;
                border: none;
                height: 1px;
                margin: 8px 0px;
            }

            QScrollBar:horizontal {
                height: 12px;
                background-color: #1a1a21;
            }

            QScrollBar:vertical {
                width: 12px;
                background-color: #1a1a21;
            }

            QScrollBar::handle {
                background-color: #3a3a4a;
                border-radius: 6px;
            }

            QScrollBar::handle:hover {
                background-color: #4a4a5a;
            }
        """)

    def create_section_title(self, text):
        label = QLabel(text)
        label.setProperty("title", True)
        return label

    def create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setProperty("line", True)
        return line

    def create_accent_button(self, text, callback=None, mini=False):
        btn = QPushButton(text)
        btn.setProperty("accent", True)
        if mini:
            btn.setProperty("mini", True)
        if callback:
            btn.clicked.connect(callback)
        return btn

    def create_danger_button(self, text, callback=None):
        btn = QPushButton(text)
        btn.setProperty("danger", True)
        if callback:
            btn.clicked.connect(callback)
        return btn

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                background-color: #1a1a21;
                border: 1px solid #2a2a35;
            }
            QTabBar::tab {
                background-color: #252532;
                color: #b0b0c0;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #2a2a35;
                border-bottom: none;

                font-weight: 500;
            }
            QTabBar::tab:hover {
                background-color: #2d2d3a;
                color: #d0d0e0;
            }
        """)

        self.scene_content_tab = QWidget()
        self.setup_scene_content_tab()
        self.tab_widget.addTab(self.scene_content_tab, "🎮 СОДЕРЖАНИЕ СЦЕНЫ")

        self.scene_control_tab = QWidget()
        self.setup_scene_control_tab()
        self.tab_widget.addTab(self.scene_control_tab, "🎛️ УПРАВЛЕНИЕ СЦЕНОЙ")

        self.debug_tab = QWidget()
        self.setup_debug_tab()
        self.tab_widget.addTab(self.debug_tab, "🐛 DEBUG")

        main_layout.addWidget(self.tab_widget)

        self.status_bar = QLabel()
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("""
            background-color: #252532;
            color: #a0a0b0;
            padding: 6px 12px;
            font-size: 10px;
            border-top: 1px solid #2a2a35;
            min-height: 24px;
        """)
        self.status_bar.setText("Готов к работе")
        main_layout.addWidget(self.status_bar)

        self.setLayout(main_layout)

        self.setWindowTitle('🎮 3D Viewer')
        self.setFixedWidth(380)
        self.setMinimumHeight(700)

    def setup_scene_content_tab(self):
        layout = QVBoxLayout(self.scene_content_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        if self.models_config:
            model_section = QGroupBox("НАБОРЫ МОДЕЛЕЙ")
            model_layout = QVBoxLayout()
            model_layout.setSpacing(6)

            model_combo_group = QWidget()
            model_combo_layout = QHBoxLayout(model_combo_group)
            model_combo_layout.setContentsMargins(0, 0, 0, 0)

            model_combo_layout.addWidget(QLabel("Набор:"))
            self.model_set_combo = QComboBox()
            self.model_set_combo.setMinimumHeight(25)
            for model_set_name in self.models_config.keys():
                self.model_set_combo.addItem(model_set_name)
            self.model_set_combo.currentTextChanged.connect(self.on_model_set_changed)
            model_combo_layout.addWidget(self.model_set_combo)
            model_layout.addWidget(model_combo_group)

            self.model_set_info = QLabel("Выберите набор моделей")
            self.model_set_info.setStyleSheet("""
                background-color: #1a1a21;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #2a2a35;
                color: #808090;
                font-size: 10px;
            """)
            self.model_set_info.setWordWrap(True)
            model_layout.addWidget(self.model_set_info)

            self.load_model_set_btn = self.create_accent_button(
                "🚚 Загрузить набор моделей",
                self.load_selected_model_set
            )
            model_layout.addWidget(self.load_model_set_btn)

            model_section.setLayout(model_layout)
            layout.addWidget(model_section)

        if self.textures_config:
            texture_section = QGroupBox("НАБОРЫ ТЕКСТУР")
            texture_layout = QVBoxLayout()
            texture_layout.setSpacing(6)

            texture_combo_group = QWidget()
            texture_combo_layout = QHBoxLayout(texture_combo_group)
            texture_combo_layout.setContentsMargins(0, 0, 0, 0)

            texture_combo_layout.addWidget(QLabel("Текстуры:"))
            self.textures_combo = QComboBox()
            self.textures_combo.setMinimumHeight(25)
            for texture_set_name in self.textures_config.keys():
                if texture_set_name != "default":
                    self.textures_combo.addItem(texture_set_name)
            self.textures_combo.currentTextChanged.connect(self.on_texture_set_changed)
            texture_combo_layout.addWidget(self.textures_combo)
            texture_layout.addWidget(texture_combo_group)

            self.texture_set_info = QLabel("Выберите набор текстур")
            self.texture_set_info.setStyleSheet("""
                background-color: #1a1a21;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #2a2a35;
                color: #808090;
                font-size: 10px;
            """)
            self.texture_set_info.setWordWrap(True)
            texture_layout.addWidget(self.texture_set_info)

            texture_section.setLayout(texture_layout)
            layout.addWidget(texture_section)

        process_section = QGroupBox("НАПОЛНЕНИЕ")
        process_layout = QVBoxLayout()
        process_layout.setSpacing(8)

        volume_group = QWidget()
        volume_layout = QHBoxLayout(volume_group)
        volume_layout.setContentsMargins(0, 0, 0, 0)

        volume_layout.addWidget(QLabel("Target Volume:"))
        self.target_volume_spinbox = QDoubleSpinBox()
        self.target_volume_spinbox.setRange(0.1, 200.0)
        self.target_volume_spinbox.setValue(20.0)
        self.target_volume_spinbox.setSingleStep(0.5)
        self.target_volume_spinbox.valueChanged.connect(self.update_target_volume)
        volume_layout.addWidget(self.target_volume_spinbox)
        process_layout.addWidget(volume_group)

        self.run_full_process_btn_scene = self.create_accent_button(
            "🚀 Построить наполнение",
            self.run_full_process
        )
        process_layout.addWidget(self.run_full_process_btn_scene)

        process_section.setLayout(process_layout)
        layout.addWidget(process_section)

        # === НОВАЯ СЕКЦИЯ: ПАРАМЕТРЫ ЧАСТИЦ ===
        particle_section = QGroupBox("ПАРАМЕТРЫ ЧАСТИЦ")
        particle_layout = QVBoxLayout()
        particle_layout.setSpacing(8)

        self.particle_flag_checkbox = QCheckBox("Распределять частицы")
        self.particle_flag_checkbox.setChecked(self.panda_app.canDistributeMeshes)  # используем canDistributeMeshes
        self.particle_flag_checkbox.stateChanged.connect(self.on_particle_flag_changed)

        particle_layout.addWidget(self.particle_flag_checkbox)
        particle_section.setLayout(particle_layout)
        layout.addWidget(particle_section)

        # === Секция: 2D в 3D реконструкция ===
        recon_section = QGroupBox("2D В 3D РЕКОНСТРУКЦИЯ")
        recon_layout = QVBoxLayout()
        recon_layout.setSpacing(8)

        # --- Папки (всегда читаем обе) ---
        self.recon_base_folders = {
            "ply": os.path.join(PROJECT_ROOT, "PLY_examples"),
            "height": os.path.join(PROJECT_ROOT, "height_examples")
        }

        self.recon_all_files = []

        # ======================
        # СПИСОК
        # ======================
        self.recon_json_list = QListWidget()
        self.recon_json_list.setMinimumHeight(250)
        recon_layout.addWidget(self.recon_json_list)

        # ======================
        # ЗАГРУЗКА ФАЙЛОВ
        # ======================
        def load_recon_jsons():
            # Очищаем список
            self.recon_json_list.clear()

            # --- 1. Получаем данные с сервера (как и раньше) ---
            try:
                files_from_server = self.panda_app.tls_client.get_verified_models()
            except Exception as e:
                print(f"Ошибка получения списка с сервера: {e}")
                files_from_server = []

            # --- 2. Сканируем локальную папку height_examples ---
            local_files = []
            height_folder = self.recon_base_folders["height"]  # путь к height_examples
            if os.path.exists(height_folder):
                for filename in os.listdir(height_folder):
                    if filename.lower().endswith('.json'):
                        json_path = os.path.join(height_folder, filename)
                        try:
                            with open(json_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            # Парсим дату (ожидается поле "time")
                            time_str = data.get("time", "")
                            dt = None
                            try:
                                dt = datetime.strptime(time_str, "%d.%m.%Y %H:%M")
                            except:
                                try:
                                    dt = datetime.fromisoformat(time_str)
                                except:
                                    dt = datetime.now()

                            # Формируем запись
                            entry = {
                                "name": filename,
                                "path": json_path,                # локальный путь
                                "model": data.get("model", ""),
                                "datetime": dt,
                                "data_type": "height",            # помечаем как height
                                "source_folder": "height_examples",
                                "car_number": data.get("car_number", "Неизвестно"),
                                "time": time_str,
                                "img_file": data.get("img_file", ""),
                                "ply_file": data.get("ply_file", None),  # может отсутствовать
                                "is_local": True,                  # флаг локальности
                                "filler": data.get("filler", ""),  # добавляем наполнитель
                                "target_volume": data.get("target_volume", None)  # добавляем объём
                            }
                            local_files.append(entry)
                        except Exception as e:
                            print(f"Ошибка чтения локального JSON {filename}: {e}")

            # --- 3. Объединяем и сортируем ---
            all_entries = []
            # Серверные файлы
            for f in files_from_server:
                dt = None
                try:
                    dt = datetime.strptime(f.get("datetime", ""), "%d.%m.%Y %H:%M")
                except:
                    try:
                        dt = datetime.fromisoformat(f.get("datetime", ""))
                    except:
                        dt = datetime.now()
                entry = {
                    "name": f["name"],
                    "path": f["path"],                # серверный путь (не используется напрямую)
                    "model": f["model"],
                    "datetime": dt,
                    "data_type": f.get("data_type", "ply"),
                    "source_folder": "PLY_examples",
                    "car_number": f.get("car_number", "Неизвестно"),
                    "time": f.get("time", ""),
                    "img_file": f.get("img_file", ""),
                    "is_local": False,
                    "filler": f.get("filler", ""),      # может отсутствовать
                    "target_volume": f.get("target_volume", None)
                }
                all_entries.append(entry)

            # Добавляем локальные
            all_entries.extend(local_files)

            # Сортируем по дате (новые сверху)
            sorted_entries = sorted(all_entries, key=lambda x: x["datetime"], reverse=True)

            # Временная директория для миниатюр
            temp_dir = os.path.join(tempfile.gettempdir(), "vizutil_recon_thumbnails")
            os.makedirs(temp_dir, exist_ok=True)

            for entry in sorted_entries:
                item = QListWidgetItem()
                item.setData(Qt.UserRole, entry)

                display_time = entry["datetime"].strftime("%d.%m.%Y %H:%M") if entry["datetime"] else ""
                widget = ReconListItemWidget(
                    car_number=entry["car_number"],
                    time_str=display_time
                )
                widget.set_item(item)  # сохраняем ссылку на item

                # Подключаем сигналы для тултипа и кнопки изображения
                widget.entered.connect(self.on_widget_entered)
                widget.left.connect(self.hide_hover_tooltip)
                widget.show_image_requested.connect(self.show_fullscreen_image)

                item.setSizeHint(widget.sizeHint())
                self.recon_json_list.addItem(item)
                self.recon_json_list.setItemWidget(item, widget)

                # Загрузка миниатюры (если есть изображение)
                if entry["img_file"]:
                    if entry["is_local"]:
                        # Для локальных файлов пытаемся загрузить изображение напрямую
                        img_local_path = os.path.join(height_folder, entry["img_file"])
                        if os.path.exists(img_local_path):
                            pixmap = QPixmap(img_local_path)
                            widget.set_icon_pixmap(pixmap)
                        else:
                            # Можно попробовать скачать с сервера, если картинка там же
                            task = ImageDownloadTask(
                                client=self.panda_app.tls_client,
                                img_filename=entry["img_file"],
                                item_widget=widget,
                                temp_dir=temp_dir
                            )
                            QThreadPool.globalInstance().start(task)
                    else:
                        # Для серверных – асинхронная загрузка
                        task = ImageDownloadTask(
                            client=self.panda_app.tls_client,
                            img_filename=entry["img_file"],
                            item_widget=widget,
                            temp_dir=temp_dir
                        )
                        QThreadPool.globalInstance().start(task)

        # ======================
        # КЛИК
        # ======================
        def on_recon_file_clicked(item):
            file_data = item.data(Qt.UserRole)
            if not file_data:
                return

            # Показываем оверлей
            self.show_overlay("🚀 Загрузка данных...")

            # Определяем путь к JSON (локальный или будем скачивать)
            if file_data.get("is_local"):
                local_json_path = file_data["path"]
                self.log_message(f"📁 Используется локальный JSON: {local_json_path}")
            else:
                # Скачиваем с сервера
                temp_dir = os.path.join(tempfile.gettempdir(), "vizutil_recon")
                os.makedirs(temp_dir, exist_ok=True)
                server_json_name = file_data["name"]
                local_json_path = os.path.join(temp_dir, server_json_name)
                try:
                    self.panda_app.tls_client.download_file(server_json_name, local_json_path)
                    self.log_message(f"✅ JSON загружен: {server_json_name}")
                except Exception as e:
                    self.log_message(f"❌ Ошибка загрузки JSON: {e}")
                    self.hide_overlay_timer.start(2000)
                    return

            # Читаем JSON (локальный или скачанный)
            try:
                with open(local_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    car_number = json_data.get("car_number")
                    filler = json_data.get("filler")
                    time = json_data.get("time")
                    target_volume = json_data.get("target_volume")
                    self.panda_app.update_overlay_info(
                        texture=filler,
                        car_number=car_number,
                        initial_volume=target_volume,
                        time=time
                    )

                    # Установка набора текстур по filler (как и раньше)
                    if filler:
                        selected_texture_key = None
                        selected_config = None
                        for key, config in self.textures_config.items():
                            if key == "default":
                                continue
                            if config.get("name") == filler:
                                selected_texture_key = key
                                selected_config = config
                                break
                        if selected_config:
                            self.panda_app.set_texture_set(selected_config)
                            self.textures_combo.setCurrentText(selected_texture_key)
                            self.log_message(f"🎨 Установлен набор текстур: {selected_texture_key}")
                        else:
                            self.log_message(f"⚠️ Набор текстур для '{filler}' не найден, используется текущий")
            except Exception as e:
                self.log_message(f"❌ Ошибка чтения JSON: {e}")
                self.hide_overlay_timer.start(2000)
                return

            # Загрузка PLY-файла (если указан)
            ply_filename = json_data.get("ply_file")
            local_ply_path = None
            if ply_filename:
                # Для локальных файлов ищем PLY в той же папке, что и JSON
                if file_data.get("is_local"):
                    ply_dir = os.path.dirname(local_json_path)
                    local_ply_path = os.path.join(ply_dir, ply_filename)
                    if not os.path.exists(local_ply_path):
                        self.log_message(f"⚠️ Локальный PLY не найден: {local_ply_path}")
                        local_ply_path = None
                else:
                    temp_dir = os.path.dirname(local_json_path)
                    local_ply_path = os.path.join(temp_dir, ply_filename)
                    if not os.path.exists(local_ply_path):
                        self.log_message("📁 Загрузка PLY файла...")
                        try:
                            self.panda_app.tls_client.download_file(ply_filename, local_ply_path)
                            self.log_message(f"✅ PLY загружен: {ply_filename}")
                        except Exception as e:
                            self.log_message(f"❌ Ошибка загрузки PLY: {e}")
                            self.hide_overlay_timer.start(2000)
                            return
                    else:
                        self.log_message(f"📁 PLY уже загружен: {ply_filename}")

            # Загружаем модель по имени из JSON
            model_name = file_data["model"]
            self.log_message(f"📁 Загружается модель: {model_name}")
            if model_name:
                model_kayname = ""
                for key, config in self.models_config.items():
                    if config.get("model") == model_name:
                        model_kayname = key
                if model_kayname:
                    self.load_model_set(model_kayname)
                    # Синхронизируем выпадающий список
                    self.model_set_combo.setCurrentText(model_kayname)
                else:
                    self.set_status(f"Модель '{model_name}' не найдена в конфигурации", True)
            self.log_message("✅ Модель загружена")

            recon_module = getattr(self.panda_app, "mesh_reconstruction", None)
            if not recon_module:
                self.log_message("❌ Модуль реконструкции не найден")
                self.hide_overlay_timer.start(2000)
                return

            if file_data["data_type"] == "height":
                # Получаем имя файла карты высот из JSON
                heightmap_filename = json_data.get("heightmap_path", "")
                if not heightmap_filename:
                    self.log_message("❌ В JSON не указан путь к карте высот (heightmap_path)")
                    self.hide_overlay_timer.start(2000)
                    return

                # Определяем локальный путь, где должна лежать карта высот
                # Она должна находиться в той же папке, что и JSON (так ожидает MeshReconstruction.load_height_map)
                target_dir = os.path.dirname(local_json_path)
                local_heightmap_path = os.path.join(target_dir, heightmap_filename)

                # Если файл ещё не существует – пытаемся его получить
                if not os.path.exists(local_heightmap_path):
                    if file_data.get("is_local"):
                        # Локальный файл: ищем в исходной папке JSON
                        src_dir = os.path.dirname(file_data["path"])
                        src_heightmap = os.path.join(src_dir, heightmap_filename)
                        if os.path.exists(src_heightmap):
                            # Копируем или просто запоминаем путь (можно использовать исходный, но load_height_map ожидает рядом с JSON)
                            # Проще скопировать, чтобы не менять логику загрузки
                            import shutil
                            shutil.copy2(src_heightmap, local_heightmap_path)
                            self.log_message(f"📁 Карта высот скопирована из локальной папки: {heightmap_filename}")
                        else:
                            self.log_message(f"❌ Локальная карта высот не найдена: {src_heightmap}")
                            self.hide_overlay_timer.start(2000)
                            return
                    else:
                        # Серверный файл: скачиваем
                        self.log_message(f"📁 Загрузка карты высот: {heightmap_filename}")
                        try:
                            self.panda_app.tls_client.download_file(heightmap_filename, local_heightmap_path)
                            self.log_message(f"✅ Карта высот загружена: {heightmap_filename}")
                        except Exception as e:
                            self.log_message(f"❌ Ошибка загрузки карты высот: {e}")
                            self.hide_overlay_timer.start(2000)
                            return

                # Всё готово, запускаем реконструкцию (ply_path не передаём)
                self.log_message("📊 Запуск реконструкции по карте высот...")
                try:
                    recon_module.run_2d_to_3d_reconstruction_from(json_path=local_json_path)
                    self.log_message("✅ Реконструкция по карте высот завершена")
                except Exception as e:
                    self.log_message(f"❌ Ошибка реконструкции: {e}")

                self.hide_overlay_timer.start(2000)
            else:
                # Для PLY
                self.log_message("📊 Запуск реконструкции по PLY...")
                try:
                    recon_module.run_2d_to_3d_reconstruction_from(
                        json_path=local_json_path,
                        ply_path=local_ply_path
                    )
                    self.log_message("✅ Реконструкция по PLY завершена")
                except Exception as e:
                    self.log_message(f"❌ Ошибка реконструкции: {e}")
                self.hide_overlay_timer.start(2000)

        # ======================
        # СИГНАЛЫ
        # ======================
        self.recon_json_list.itemClicked.connect(on_recon_file_clicked)

        load_recon_jsons()

        recon_section.setLayout(recon_layout)
        layout.addWidget(recon_section)

        layout.addStretch()

    # ---------- Методы для тултипа ----------
    def on_widget_entered(self, widget, global_pos):
        """Обработчик входа мыши на виджет элемента"""
        self.hide_hover_tooltip()  # скрываем предыдущий, если был

        pixmap = widget.get_pixmap()
        if pixmap is None or pixmap.isNull():
            return

        # Получаем entry через сохранённый item
        if widget.item is None:
            return
        entry = widget.item.data(Qt.UserRole)
        if not entry:
            return

        self.show_tooltip_at(pixmap, entry, global_pos, widget.width(), widget.height())

    def show_tooltip_at(self, pixmap, entry, widget_top_left, widget_width, widget_height):
        """Создаёт и показывает единый тултип с изображением и информацией"""
        tooltip = HoverInfoWidget(pixmap, entry)
        tooltip_size = tooltip.sizeHint()
        screen = QApplication.primaryScreen().availableGeometry()

        # Пытаемся разместить слева
        x = widget_top_left.x() - tooltip_size.width() - 10
        y = widget_top_left.y() + (widget_height - tooltip_size.height()) // 2

        # Если слева не влезает, пробуем справа
        if x < screen.left():
            x = widget_top_left.x() + widget_width + 10

        # Корректировка по вертикали, чтобы не выходить за экран
        if y < screen.top():
            y = screen.top()
        if y + tooltip_size.height() > screen.bottom():
            y = screen.bottom() - tooltip_size.height()

        tooltip.move(x, y)
        tooltip.show()
        self.hover_tooltip = tooltip

    def hide_hover_tooltip(self):
        """Скрывает активный тултип"""
        if self.hover_tooltip:
            self.hover_tooltip.close()
            self.hover_tooltip = None
    # ----------------------------------------

    # ---------- Методы для полноэкранного изображения ----------
    def show_fullscreen_image(self, entry):
        """Показать полноэкранное изображение из entry"""
        img_file = entry.get('img_file')
        if not img_file:
            return

        # Определяем путь к изображению
        if entry.get('is_local'):
            # локальный файл — лежит в папке height_examples
            img_path = os.path.join(self.recon_base_folders["height"], img_file)
            if not os.path.exists(img_path):
                self.set_status("Файл изображения не найден локально", True)
                return
            pixmap = QPixmap(img_path)
        else:
            # серверный — скачиваем во временную папку
            temp_dir = os.path.join(tempfile.gettempdir(), "vizutil_fullsize_images")
            os.makedirs(temp_dir, exist_ok=True)
            local_img_path = os.path.join(temp_dir, img_file)
            if not os.path.exists(local_img_path):
                try:
                    self.panda_app.tls_client.download_file(img_file, local_img_path)
                except Exception as e:
                    self.set_status(f"Ошибка загрузки изображения: {e}", True)
                    return
            pixmap = QPixmap(local_img_path)

        if pixmap.isNull():
            self.set_status("Не удалось загрузить изображение", True)
            return

        # Показываем оверлей с изображением
        self.show_image_overlay(pixmap)

    def show_image_overlay(self, pixmap):
        """Показать оверлей с изображением на весь экран"""
        if self.image_overlay:
            self.image_overlay.close()

        parent = self.main_window if self.main_window else self.window()
        self.image_overlay = ImageOverlay(pixmap, parent)
        self.image_overlay.setGeometry(parent.rect())
        self.image_overlay.show()
        self.image_overlay.raise_()

    def hide_image_overlay(self):
        if self.image_overlay:
            self.image_overlay.close()
            self.image_overlay = None
    # ---------------------------------------------------------

    def on_particle_flag_changed(self, state):
        self.panda_app.canDistributeMeshes = (state == Qt.Checked)
        self.set_status(f"Распределение частиц: {'включено' if self.panda_app.canDistributeMeshes else 'выключено'}")

    def setup_scene_control_tab(self):
        layout = QVBoxLayout(self.scene_control_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        camera_section = QGroupBox("ВИДЫ КАМЕРЫ")
        camera_layout = QGridLayout()
        camera_layout.setSpacing(6)

        views = [
            ('Перспектива', 'perspective'), ('Сверху', 'top'), ('Снизу', 'bottom'),
            ('Спереди', 'front'), ('Сзади', 'back'), ('Слева', 'left'), ('Справа', 'right')
        ]

        for i, (name, view) in enumerate(views):
            btn = self.create_accent_button(name, self.change_view, mini=True)
            btn.setProperty("view", view)
            camera_layout.addWidget(btn, i // 4, i % 4)

        camera_section.setLayout(camera_layout)
        layout.addWidget(camera_section)

        # === НОВАЯ СЕКЦИЯ: ВРЕМЯ СУТОК ===
        time_section = QGroupBox("ВРЕМЯ СУТОК")
        time_layout = QVBoxLayout()
        time_layout.setSpacing(8)

        # Слайдер для выбора времени
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 1439)  # От 00:00 до 23:59 в минутах
        self.time_slider.setValue(400)  # Начальное значение 6:40 (6*60 + 40 = 400)
        self.time_slider.setSingleStep(10)  # Шаг 10 минут
        self.time_slider.setTickInterval(60)  # Метки каждый час
        self.time_slider.setTickPosition(QSlider.TicksBelow)
        self.time_slider.valueChanged.connect(self.change_time_of_day)

        # Метка для отображения текущего времени
        self.time_label = QLabel("Время: 06:40")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("""
            font-size: 11px;
            font-weight: 500;
            color: #a0a0b0;
        """)

        # Примеры времени для быстрого выбора
        time_presets_layout = QHBoxLayout()
        time_presets_layout.setSpacing(4)

        time_presets = [
            ("🌅 06:40", 400),
            ("☀️ 12:00", 720),
            ("🌆 16:50", 1010),
            ("🌙 20:30", 1230),
            ("🌌 00:00", 0)
        ]

        for preset_name, minutes in time_presets:
            btn = QPushButton(preset_name)
            btn.setProperty("mini", True)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 4px 8px;
                    font-size: 10px;
                    background-color: #252532;
                    border: 1px solid #3a3a4a;
                    border-radius: 4px;
                    color: #b0b0c0;
                }
                QPushButton:hover {
                    background-color: #2d2d3a;
                    border: 1px solid #4a4a5a;
                }
            """)
            btn.clicked.connect(lambda checked, m=minutes: self.set_time_preset(m))
            time_presets_layout.addWidget(btn)

        time_presets_widget = QWidget()
        time_presets_widget.setLayout(time_presets_layout)

        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.time_slider)
        time_layout.addWidget(time_presets_widget)

        time_section.setLayout(time_layout)
        layout.addWidget(time_section)

        render_section = QGroupBox("РЕНДЕРИНГ")
        render_layout = QVBoxLayout()
        render_layout.setSpacing(8)

        self.save_single_render_button = self.create_accent_button(
            "🖼️ Одиночный рендер",
            self.panda_app.renderer_utils.save_single_render
        )
        render_layout.addWidget(self.save_single_render_button)

        self.save_dataset_button = self.create_accent_button(
            "📊 Рендер датасета",
            self.panda_app.renderer_utils.save_dataset_render
        )
        render_layout.addWidget(self.save_dataset_button)

        self.log_camera_button = self.create_accent_button(
            "📷 Параметры камеры",
            self.panda_app.log_camera_parameters
        )
        render_layout.addWidget(self.log_camera_button)

        render_section.setLayout(render_layout)
        layout.addWidget(render_section)

        depth_section = QGroupBox("КАРТА ГЛУБИНЫ")
        depth_layout = QVBoxLayout()
        depth_layout.setSpacing(8)

        self.toggle_depth_btn = QPushButton("🌊 Включить карту глубины")
        self.toggle_depth_btn.setCheckable(True)
        self.toggle_depth_btn.clicked.connect(self.toggle_depth_overlay)
        depth_layout.addWidget(self.toggle_depth_btn)

        depth_settings_group = QWidget()
        depth_settings_layout = QVBoxLayout(depth_settings_group)
        depth_settings_layout.setSpacing(6)

        near_far_group = QWidget()
        near_far_layout = QGridLayout(near_far_group)
        near_far_layout.setContentsMargins(0, 0, 0, 0)

        near_far_layout.addWidget(QLabel("Ближняя:"), 0, 0)
        self.min_depth_spinbox = QDoubleSpinBox()
        self.min_depth_spinbox.setRange(0.01, 1000.0)
        self.min_depth_spinbox.setValue(0.1)
        self.min_depth_spinbox.setSingleStep(0.1)
        self.min_depth_spinbox.valueChanged.connect(self.update_min_depth)
        near_far_layout.addWidget(self.min_depth_spinbox, 0, 1)

        near_far_layout.addWidget(QLabel("Дальняя:"), 1, 0)
        self.max_depth_spinbox = QDoubleSpinBox()
        self.max_depth_spinbox.setRange(0.1, 10000.0)
        self.max_depth_spinbox.setValue(100.0)
        self.max_depth_spinbox.setSingleStep(1.0)
        self.max_depth_spinbox.valueChanged.connect(self.update_max_depth)
        near_far_layout.addWidget(self.max_depth_spinbox, 1, 1)

        depth_settings_layout.addWidget(near_far_group)

        gradient_group = QWidget()
        gradient_layout = QGridLayout(gradient_group)
        gradient_layout.setContentsMargins(0, 0, 0, 0)

        gradient_layout.addWidget(QLabel("Начало:"), 0, 0)
        self.gradient_start_spinbox = QDoubleSpinBox()
        self.gradient_start_spinbox.setRange(0.0, 1.0)
        self.gradient_start_spinbox.setValue(0.2)
        self.gradient_start_spinbox.setSingleStep(0.05)
        self.gradient_start_spinbox.valueChanged.connect(self.update_gradient_start)
        gradient_layout.addWidget(self.gradient_start_spinbox, 0, 1)

        gradient_layout.addWidget(QLabel("Конец:"), 1, 0)
        self.gradient_end_spinbox = QDoubleSpinBox()
        self.gradient_end_spinbox.setRange(0.0, 1.0)
        self.gradient_end_spinbox.setValue(0.4)
        self.gradient_end_spinbox.setSingleStep(0.05)
        self.gradient_end_spinbox.valueChanged.connect(self.update_gradient_end)
        gradient_layout.addWidget(self.gradient_end_spinbox, 1, 1)

        depth_settings_layout.addWidget(gradient_group)
        depth_layout.addWidget(depth_settings_group)

        depth_section.setLayout(depth_layout)
        layout.addWidget(depth_section)

        drag_section = QGroupBox("DRAG & DROP")
        drag_layout = QVBoxLayout()
        drag_layout.setSpacing(8)

        self.drag_drop_btn = QPushButton("👆 Включить Drag & Drop")
        self.drag_drop_btn.setCheckable(True)
        self.drag_drop_btn.clicked.connect(self.toggle_drag_drop)
        drag_layout.addWidget(self.drag_drop_btn)

        sens_group = QWidget()
        sens_layout = QHBoxLayout(sens_group)
        sens_layout.setContentsMargins(0, 0, 0, 0)

        sens_layout.addWidget(QLabel("Чувствительность:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(20)
        self.sensitivity_slider.valueChanged.connect(self.change_drag_sensitivity)
        self.sensitivity_label = QLabel("2.0")

        sens_layout.addWidget(self.sensitivity_slider)
        sens_layout.addWidget(self.sensitivity_label)
        drag_layout.addWidget(sens_group)

        drag_section.setLayout(drag_layout)
        layout.addWidget(drag_section)

        tips_label = QLabel(
            "🖱️ Управление: WASD - движение • Space/Shift - высота • ЛКМ - вращение\n"
            "🔄 Для Drag & Drop включите режим выше и используйте ЛКМ"
        )
        tips_label.setAlignment(Qt.AlignCenter)
        tips_label.setStyleSheet("""
            background-color: #252532;
            color: #808090;
            font-size: 9px;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #2a2a35;
        """)
        tips_label.setWordWrap(True)
        layout.addWidget(tips_label)

        layout.addStretch()

    def setup_debug_tab(self):
        """Вкладка Debug"""
        layout = QVBoxLayout(self.debug_tab)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        load_section = QGroupBox("ЗАГРУЗКА МОДЕЛЕЙ")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(8)

        self.load_btn = self.create_accent_button(
            "📁 Загрузить GLTF модель",
            self.load_model
        )
        load_layout.addWidget(self.load_btn)

        load_section.setLayout(load_layout)
        layout.addWidget(load_section)

        mesh_section = QGroupBox("ГЕНЕРАЦИЯ МЕШЕЙ")
        mesh_layout = QVBoxLayout()
        mesh_layout.setSpacing(8)

        self.perlin_btn = self.create_accent_button(
            "🌄 Сгенерировать Perlin Mesh",
            self.generate_perlin_mesh
        )
        mesh_layout.addWidget(self.perlin_btn)

        plane_buttons_group = QWidget()
        plane_buttons_layout = QHBoxLayout(plane_buttons_group)
        plane_buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.create_plane_btn = self.create_accent_button(
            "📐 Создать плоскость",
            self.create_ground_plane,
            mini=True
        )
        plane_buttons_layout.addWidget(self.create_plane_btn)

        self.perform_plane_AABB = self.create_accent_button(
            "📏 AABB Plane",
            self.perform_AABB_plane,
            mini=True
        )
        plane_buttons_layout.addWidget(self.perform_plane_AABB)

        mesh_layout.addWidget(plane_buttons_group)
        mesh_section.setLayout(mesh_layout)
        layout.addWidget(mesh_section)

        plane_settings_section = QGroupBox("НАСТРОЙКИ ПЛОСКОСТИ")
        plane_settings_layout = QVBoxLayout()
        plane_settings_layout.setSpacing(8)

        size_group = QWidget()
        size_layout = QGridLayout(size_group)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setHorizontalSpacing(10)

        size_layout.addWidget(QLabel("Размер X:"), 0, 0)
        self.plane_size_x_spinbox = QDoubleSpinBox()
        self.plane_size_x_spinbox.setRange(0.1, 1000.0)
        self.plane_size_x_spinbox.setValue(100.0)
        self.plane_size_x_spinbox.valueChanged.connect(self.change_plane_size_x)
        size_layout.addWidget(self.plane_size_x_spinbox, 0, 1)

        size_layout.addWidget(QLabel("Размер Y:"), 1, 0)
        self.plane_size_y_spinbox = QDoubleSpinBox()
        self.plane_size_y_spinbox.setRange(0.1, 1000.0)
        self.plane_size_y_spinbox.setValue(100.0)
        self.plane_size_y_spinbox.valueChanged.connect(self.change_plane_size_y)
        size_layout.addWidget(self.plane_size_y_spinbox, 1, 1)

        plane_settings_layout.addWidget(size_group)

        pos_group = QWidget()
        pos_layout = QGridLayout(pos_group)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setHorizontalSpacing(10)

        pos_layout.addWidget(QLabel("Позиция X:"), 0, 0)
        self.plane_pos_x_spinbox = QDoubleSpinBox()
        self.plane_pos_x_spinbox.setRange(-10000, 10000)
        self.plane_pos_x_spinbox.setValue(0)
        self.plane_pos_x_spinbox.valueChanged.connect(lambda: self.change_plane_position('x'))
        pos_layout.addWidget(self.plane_pos_x_spinbox, 0, 1)

        pos_layout.addWidget(QLabel("Позиция Y:"), 1, 0)
        self.plane_pos_y_spinbox = QDoubleSpinBox()
        self.plane_pos_y_spinbox.setRange(-10000, 10000)
        self.plane_pos_y_spinbox.setValue(0)
        self.plane_pos_y_spinbox.valueChanged.connect(lambda: self.change_plane_position('y'))
        pos_layout.addWidget(self.plane_pos_y_spinbox, 1, 1)

        pos_layout.addWidget(QLabel("Позиция Z:"), 2, 0)
        self.plane_pos_z_spinbox = QDoubleSpinBox()
        self.plane_pos_z_spinbox.setRange(-10000, 10000)
        self.plane_pos_z_spinbox.setValue(0)
        self.plane_pos_z_spinbox.valueChanged.connect(lambda: self.change_plane_position('z'))
        pos_layout.addWidget(self.plane_pos_z_spinbox, 2, 1)

        plane_settings_layout.addWidget(pos_group)

        apply_pos_btn = self.create_accent_button(
            "📍 Применить позицию",
            lambda: self.change_plane_position('all'),
            mini=True
        )
        plane_settings_layout.addWidget(apply_pos_btn)

        plane_settings_section.setLayout(plane_settings_layout)
        layout.addWidget(plane_settings_section)

        layout.addStretch()

    def change_time_of_day(self, minutes):
        """Изменение времени суток по значению слайдера (в минутах)"""
        hours = minutes // 60
        mins = minutes % 60

        # Форматируем время как строку
        time_str = f"{hours:02d}:{mins:02d}"

        # Обновляем метку
        time_names = {
            (0, 5): "🌌 Ночь",
            (6, 11): "🌅 Утро",
            (12, 17): "☀️ День",
            (18, 23): "🌆 Вечер"
        }

        time_name = "🌌 Ночь"
        for (start, end), name in time_names.items():
            if start <= hours <= end:
                time_name = name
                break

        self.time_label.setText(f"{time_name}: {time_str}")

        # Применяем время к сцене
        if hasattr(self.panda_app, 'render_pipeline'):
            try:
                self.panda_app.render_pipeline.daytime_mgr.time = time_str
                self.set_status(f"Время суток: {time_str}")
            except Exception as e:
                self.set_status(f"Ошибка установки времени: {str(e)}", True)

    def set_time_preset(self, minutes):
        """Установка предустановленного времени"""
        self.time_slider.setValue(minutes)

    def create_accent_button(self, text, callback=None, mini=False):
        btn = QPushButton(text)
        btn.setProperty("accent", True)
        if mini:
            btn.setProperty("mini", True)
        if callback:
            btn.clicked.connect(callback)
        return btn

    def set_status(self, message, is_error=False):
        """Установить статусное сообщение"""
        color = "#be4a4a" if is_error else "#4a7fbe"
        self.status_bar.setStyleSheet(f"""
            background-color: #252532;
            color: {color};
            padding: 6px 12px;
            font-size: 10px;
            border-top: 1px solid #2a2a35;
            min-height: 24px;
        """)
        self.status_bar.setText(message)

        self.status_timer.start(5000)

    def clear_status(self):
        self.status_bar.setStyleSheet("""
            background-color: #252532;
            color: #a0a0b0;
            padding: 6px 12px;
            font-size: 10px;
            border-top: 1px solid #2a2a35;
            min-height: 24px;
        """)
        self.status_bar.setText("Готов к работе")

    def _setup_transparent_material(self, model):
        """Настраивает прозрачный материал для отображения"""
        material = Material()
        material.setDiffuse((0.3, 0.7, 0.9, 1))
        material.setAmbient((0.15, 0.35, 0.45, 1))
        material.setSpecular((0.8, 0.8, 0.8, 1))
        material.setShininess(50)
        model.setMaterial(material)
        model.setShaderAuto()
        model.setTransparency(TransparencyAttrib.MAlpha)
        model.setAlphaScale(0.7)
        model.setTwoSided(True)
        model.setScale(1, 1, 1)
        model.setPos(0, 0, 0)

    def _prepare_target_model_for_boolean(self, target_model):
        """Подготавливает целевую модель для boolean операций"""
        original_min_bound, original_max_bound = target_model.getTightBounds()

        original_size_x = original_max_bound.x - original_min_bound.x
        original_size_y = original_max_bound.y - original_min_bound.y
        original_size_z = original_max_bound.z - original_min_bound.z

        original_center_x = (original_min_bound.x + original_max_bound.x) / 2
        original_center_y = (original_min_bound.y + original_max_bound.y) / 2
        original_center_z = (original_min_bound.z + original_max_bound.z) / 2

        target_model_trimesh = self.panda_app.panda_to_trimesh(target_model)

        self.processed_model = self.panda_app.trimesh_to_panda(target_model_trimesh)

        target_model_trimesh = None

        advanced_min_bound, advanced_max_bound = self.processed_model.getTightBounds()

        advanced_size_x = advanced_max_bound.x - advanced_min_bound.x
        advanced_size_y = advanced_max_bound.y - advanced_min_bound.y
        advanced_size_z = advanced_max_bound.z - advanced_min_bound.z

        advanced_center_x = (advanced_min_bound.x + advanced_max_bound.x) / 2
        advanced_center_y = (advanced_min_bound.y + advanced_max_bound.y) / 2
        advanced_center_z = (advanced_min_bound.z + advanced_max_bound.z) / 2

        scale_x = original_size_x / advanced_size_x
        scale_y = original_size_y / advanced_size_y
        scale_z = original_size_z / advanced_size_z

        self.processed_model.setScale(scale_x, scale_y, scale_z)

        new_pos_x = original_center_x - (advanced_center_x * scale_x)
        new_pos_y = original_center_y - (advanced_center_y * scale_y)
        new_pos_z = original_center_z - (advanced_center_z * scale_z)

        self.processed_model.setPos(new_pos_x, new_pos_y, new_pos_z)

        target_model_copy = target_model.copyTo(target_model.getParent())

        target_model_copy.setScale(scale_x, scale_y, scale_z)
        target_model_copy.setPos(new_pos_x, new_pos_y, new_pos_z)

        self.processed_model.hide()

        target_model_trimesh = self.panda_app.panda_to_trimesh(target_model_copy)

        target_model_copy.removeNode()

        return target_model_trimesh

    def on_texture_set_changed(self, texture_set_name):
        if texture_set_name in self.textures_config:
            config = self.textures_config[texture_set_name].copy()
            info_text = f"<b>{texture_set_name}</b><br>"

            for key in ['diffuse', 'albedo']:
                if key in config:
                    info_text += f"Основная: {os.path.basename(config[key])}<br>"
                    break

            self.texture_set_info.setText(info_text)
            self.panda_app.set_texture_set(config)

            self.set_status(f"Выбран набор текстур: {texture_set_name}")

    def on_model_set_changed(self, model_set_name):
        if model_set_name in self.models_config:
            config = self.models_config[model_set_name]
            max_volume = config.get('max_volume', 'N/A')
            info_text = f"<b>{model_set_name}</b><br>"
            info_text += f"Макс. объем: {max_volume}<br>"

            # Используем базовое имя файла для отображения
            for key in ['cuzov', 'napolnitel']:
                if key in config:
                    info_text += f"{key.capitalize()}: {os.path.basename(config[key])}<br>"

            self.model_set_info.setText(info_text)
            self.target_volume_spinbox.setValue(max_volume)

            self.set_status(f"Выбран набор моделей: {model_set_name}")
        else:
            self.model_set_info.setText("Неизвестный набор моделей")

    def load_selected_model_set(self):
        model_set_name = self.model_set_combo.currentText()
        self.load_model_set(model_set_name)

    def load_model_set(self, model_set_name):
        if not model_set_name or model_set_name not in self.models_config:
            self.set_status("⚠️ Не выбран набор моделей!", True)
            return

        config = self.models_config[model_set_name]
        success = self.panda_app.load_model_set(config, model_set_name)

        if success:
            self.set_status(f"✅ Набор моделей '{model_set_name}' успешно загружен")
        else:
            self.set_status("❌ Не удалось загрузить набор моделей", True)

    def update_gradient_start(self, value):
        if hasattr(self.panda_app, 'depth_renderer') and self.panda_app.depth_renderer:
            self.panda_app.depth_renderer.set_gradient_start(value)

    def update_gradient_end(self, value):
        if hasattr(self.panda_app, 'depth_renderer') and self.panda_app.depth_renderer:
            self.panda_app.depth_renderer.set_gradient_end(value)

    def setup_animations(self):
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(300)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()

    def update_min_depth(self, value):
        if hasattr(self.panda_app, 'depth_renderer') and self.panda_app.depth_renderer:
            self.panda_app.depth_renderer.min_depth = value
            if self.panda_app.depth_renderer.depth_camera_np:
                lens = self.panda_app.depth_renderer.depth_camera_np.node().get_lens()
                if lens:
                    lens.set_near_far(value, self.panda_app.depth_renderer.max_depth)
            if self.panda_app.depth_renderer.overlay_node:
                self.panda_app.depth_renderer.overlay_node.setShaderInput("near", value)

    def update_max_depth(self, value):
        if hasattr(self.panda_app, 'depth_renderer') and self.panda_app.depth_renderer:
            self.panda_app.depth_renderer.max_depth = value
            if self.panda_app.depth_renderer.depth_camera_np:
                lens = self.panda_app.depth_renderer.depth_camera_np.node().get_lens()
                if lens:
                    lens.set_near_far(self.panda_app.depth_renderer.min_depth, value)
            if self.panda_app.depth_renderer.overlay_node:
                self.panda_app.depth_renderer.overlay_node.setShaderInput("far", value)

    def toggle_depth_overlay(self):
        is_enabled = self.panda_app.toggle_depth_overlay()

        if is_enabled:
            self.toggle_depth_btn.setProperty("accent", True)
            self.toggle_depth_btn.setText("🌊 Выключить карту глубины")
            self.set_status("Карта глубины включена")
        else:
            self.toggle_depth_btn.setProperty("accent", False)
            self.toggle_depth_btn.setText("🌊 Включить карту глубины")
            self.set_status("Карта глубины выключена")

        self.toggle_depth_btn.style().unpolish(self.toggle_depth_btn)
        self.toggle_depth_btn.style().polish(self.toggle_depth_btn)

        return is_enabled

    def generate_perlin_mesh(self):
        success = self.panda_app.perlin_generator.generate_perlin_mesh_from_csg()

    def perform_AABB_plane(self):
        success = self.panda_app.perform_AABB_plane()

    def create_ground_plane(self):
        self.panda_app.create_ground_plane()

    def change_plane_size_x(self, value):
        self.panda_app.set_plane_size_x(value)

    def change_plane_size_y(self, value):
        self.panda_app.set_plane_size_y(value)

    def change_plane_position(self, axis):
        x = self.plane_pos_x_spinbox.value()
        y = self.plane_pos_y_spinbox.value()
        z = self.plane_pos_z_spinbox.value()
        self.panda_app.set_plane_position(x, y, z)

    def change_view(self):
        sender = self.sender()
        view_name = sender.property("view")

        view_methods = {
            "perspective": self.panda_app.set_perspective_view,
            "top": self.panda_app.set_top_view,
            "bottom": self.panda_app.set_bottom_view,
            "front": self.panda_app.set_front_view,
            "back": self.panda_app.set_back_view,
            "left": self.panda_app.set_left_view,
            "right": self.panda_app.set_right_view
        }

        if view_name in view_methods:
            view_methods[view_name]()
            self.set_status(f"Вид камеры: {sender.text()}")

    def load_model(self):
        file_path = self.panda_widget.load_model_dialog()
        if file_path:
            self.panda_app.load_gltf_model(file_path)

    def save_scene(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.panda_app.save_scene_to_json(file_path)

    def load_scene(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Scene", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.panda_app.load_scene_from_json(file_path)

    def toggle_drag_drop(self):
        is_enabled = self.drag_drop_btn.isChecked()
        self.panda_app.toggle_drag_drop_mode(is_enabled)

        if is_enabled:
            self.drag_drop_btn.setText("👆 Выключить Drag & Drop")
            self.set_status("Режим Drag & Drop включен")
        else:
            self.drag_drop_btn.setText("👆 Включить Drag & Drop")
            self.set_status("Режим Drag & Drop выключен")

    def change_drag_sensitivity(self, value):
        sensitivity = value / 10.0
        self.panda_app.set_drag_sensitivity(sensitivity)
        self.sensitivity_label.setText(f"{sensitivity:.1f}")

    def change_quarry_scale(self, value):
        self.panda_app.set_quarry_scale(value)

    def change_quarry_position(self, axis):
        x = self.pos_x_spinbox.value()
        y = self.pos_y_spinbox.value()
        z = self.pos_z_spinbox.value()
        self.panda_app.set_quarry_position(x, y, z)

    def update_target_volume(self, value):
        self.panda_app.Target_Volume = value

    def run_full_process(self):
        self.show_overlay()
        if self.hide_overlay_timer.isActive():
            self.hide_overlay_timer.stop()
        try:
            self.log_message("🔄 Запуск полного процесса построения наполнения...")
            target_volume = self.target_volume_spinbox.value()
            self.panda_app.Target_Volume = target_volume
            self.log_message(f"✅ Целевой объём установлен: {target_volume}")

            current_model_set = self.model_set_combo.currentText()
            current_texture_set = self.textures_combo.currentText() if hasattr(self, 'textures_combo') else None

            if current_model_set and current_model_set in self.models_config:
                config = self.models_config[current_model_set]
                ground_plane_z = config.get('ground_plane', 0)

                if current_texture_set and current_texture_set in self.textures_config:
                    try:
                        self.textures_combo.currentTextChanged.disconnect(self.on_texture_set_changed)
                        texture_config = self.textures_config[current_texture_set]
                        self.panda_app.set_texture_set(texture_config)
                        self.textures_combo.setCurrentText(current_texture_set)
                        self.log_message(f"✅ Набор текстур '{current_texture_set}' загружен")
                    finally:
                        self.textures_combo.currentTextChanged.connect(self.on_texture_set_changed)
                    QApplication.processEvents()

                self.log_message("🛠️ Создание плоскости земли...")
                self.panda_app.create_ground_plane()
                self.panda_app.ground_plane.setPos(0, 0, ground_plane_z)
                self.plane_pos_z_spinbox.setValue(ground_plane_z)
                QApplication.processEvents()

                self.log_message("📐 Выполнение AABB plane...")
                success_aabb = self.panda_app.perform_AABB_plane()
                QApplication.processEvents()

                if success_aabb:
                    self.log_message("🌄 Генерация Perlin mesh...")
                    success_perlin = self.panda_app.perlin_generator.generate_perlin_mesh_from_csg()
                    QApplication.processEvents()

                    if success_perlin:
                        self.log_message("✅ Все операции завершены успешно!")
                        self.set_status(
                            f"✅ Полный процесс выполнен успешно!\n"
                            f"Target Volume: {target_volume}\n"
                            f"Позиция ground_plane: Z={ground_plane_z}"
                        )
                    else:
                        self.log_message("❌ Ошибка генерации Perlin mesh")
                        self.set_status("⚠️ Не удалось сгенерировать перлин-меш", True)
                else:
                    self.log_message("❌ Ошибка выполнения AABB plane")
                    self.set_status("⚠️ Не удалось выполнить AABB plane", True)
            else:
                self.log_message("❌ Не выбран набор моделей")
                self.set_status("❌ Не выбран набор моделей или набор не найден", True)
        finally:
            self.hide_overlay_timer.start(10)