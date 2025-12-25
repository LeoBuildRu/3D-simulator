# gui.py
from height_map_mesh_generator import HeightMapMeshGenerator
import random
import math
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import yaml

from panda_widget import Panda3DWidget

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class CameraControlGUI(QWidget):
    def __init__(self, panda_app):
        super().__init__()
        self.panda_app = panda_app
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

    def load_models_config(self):
        config_path = "models_config.yaml"
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

        # === Секция: 2D в 3D реконструкция ===
        recon_section = QGroupBox("2D В 3D РЕКОНСТРУКЦИЯ")
        recon_layout = QVBoxLayout()
        recon_layout.setSpacing(8)

        # Выбор JSON-файла (СУЩЕСТВУЮЩИЙ ФУНКЦИОНАЛ)
        file_group = QWidget()
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)

        file_layout.addWidget(QLabel("Данные (.json):"))
        self.recon_json_path = QLineEdit()
        self.recon_json_path.setPlaceholderText("Выберите файл с 2D-данными")
        self.recon_json_path.setReadOnly(True)
        file_layout.addWidget(self.recon_json_path)

        def pick_recon_config():
            newConf = self.panda_app.mesh_reconstruction.browse_recon_json()
            if newConf:
                self.recon_json_path.setText(newConf)

        browse_btn = QPushButton("📂")
        browse_btn.setFixedWidth(45)
        browse_btn.clicked.connect(pick_recon_config)
        file_layout.addWidget(browse_btn)

        recon_layout.addWidget(file_group)

        # Кнопка запуска реконструкции из JSON (СУЩЕСТВУЮЩИЙ ФУНКЦИОНАЛ)
        self.run_reconstruction_btn = self.create_accent_button(
            "🔄 Запустить реконструкцию (из JSON)",
            self.panda_app.mesh_reconstruction.run_2d_to_3d_reconstruction
        )
        recon_layout.addWidget(self.run_reconstruction_btn)

        # НОВАЯ кнопка для HeightMapMeshGenerator
        self.run_height_map_btn = self.create_accent_button(
            "🗻 Сгенерировать HeightMap меш",
            self.run_height_map_reconstruction
        )
        recon_layout.addWidget(self.run_height_map_btn)

        recon_section.setLayout(recon_layout)
        layout.addWidget(recon_section)
        
        layout.addStretch()

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

    def run_height_map_reconstruction(self):
        """Запуск реконструкции через HeightMapMeshGenerator с учетом разницы поворотов"""
        try:
            # Если генератор еще не создан, создаем его
            if not hasattr(self.panda_app, 'height_map_generator'):
                self.panda_app.height_map_generator = HeightMapMeshGenerator(self.panda_app)
            
            generator = self.panda_app.height_map_generator
            
            # Получаем текущие параметры камеры
            camera_data = self.panda_app.log_camera_parameters()
            
            # Извлекаем текущий поворот камеры
            try:
                # Если camera_data уже является словарем
                if isinstance(camera_data, dict):
                    current_rotation = camera_data.get('rotation', {})
                else:
                    # Если это JSON строка
                    import json
                    camera_dict = json.loads(camera_data)
                    current_rotation = camera_dict.get('rotation', {})
            except:
                # Fallback: запросим напрямую из камеры
                cam_orientation = self.panda_app.get_camera_orientation()
                current_rotation = {
                    'h': cam_orientation['hpr'][0],
                    'p': cam_orientation['hpr'][1],
                    'r': cam_orientation['hpr'][2]
                }
            
            # Вычисляем разницу между текущим и фиксированным поворотом
            rotation_diff = {
                'h': current_rotation.get('h', 0) - self.fixed_camera_rotation['h'],
                'p': current_rotation.get('p', 0) - self.fixed_camera_rotation['p'],
                'r': current_rotation.get('r', 0) - self.fixed_camera_rotation['r']
            }
            
            # Отладочный вывод
            print(f"Текущий поворот: {current_rotation}")
            print(f"Фиксированный поворот: {self.fixed_camera_rotation}")
            print(f"Разница поворотов: {rotation_diff}")
            
            # Настройка параметров шума
            generator.set_noise_scale(3.0)
            generator.set_noise_strength(0.42)
            generator.set_noise_octaves(4)
            generator.set_noise_persistence(0.01)
            generator.set_noise_lacunarity(1.0)
            generator.set_noise_seed(random.randint(0, 10000))
            
            # Настройка параметров унифицированного подхода
            generator.set_interpolation_method('rbf')
            generator.set_rbf_smooth(0.1)
            generator.set_use_smoothing(True)
            generator.set_smoothing_iterations(1)
            
            # Настройка параметров адаптивного подъема
            generator.set_adaptive_lift_enabled(True)
            generator.set_lift_parameters(
                base_distance=0.5,
                min_distance=0.1,
                max_distance=3.0,
                intensity=1.0
            )
            
            # Настройки для устранения волнообразности
            generator.set_lift_smoothing_enabled(True)
            generator.set_lift_smoothing_sigma(1.5)
            generator.set_lift_blur_enabled(True)
            generator.set_lift_blur_radius(2)
            
            # Настройки для сглаживания исходного меша
            generator.set_source_mesh_smoothing_enabled(True)
            generator.set_source_mesh_smoothing_iterations(1)
            generator.set_source_mesh_smoothing_sigma(0.1)
            generator.set_source_mesh_edge_preserving(True)
            
            # Настройка области для меша
            generator.set_extended_area(2.5, 5.0)
            generator.set_grid_resolution(60)
            generator.set_base_height(0.0)
            
            # Создание унифицированного перлин-меша с адаптивным подъемом
            # Применяем разницу поворотов к мешу
            height_map_mesh = generator.add_extended_mesh_to_scene(
                position=(0, 0, 0.0),
                source_scale_x=2.5,
                source_scale_y=3.2,
                source_scale_z=1.75,
                source_offset_x=0.0,
                source_offset_y=0.0,
                source_offset_z=0.0,
                source_rotation_x=0,  # pitch -> rotation_x
                source_rotation_y=0,  # roll -> rotation_y
                source_rotation_z=rotation_diff['h']   # heading -> rotation_z
            )
            
            # Сохраняем ссылку на меш
            self.panda_app.height_map_mesh = height_map_mesh
            
            # Выводим информацию в статус
            self.set_status(
                f"✅ HeightMap меш создан с учетом разницы поворотов:\n"
                f"ΔH={rotation_diff['h']:.1f}°, "
                f"ΔP={rotation_diff['p']:.1f}°, "
                f"ΔR={rotation_diff['r']:.1f}°"
            )
            
            # Выводим в консоль для отладки
            print(f"Параметры, переданные в add_extended_mesh_to_scene:")
            print(f"  source_rotation_x = {rotation_diff['p']}")
            print(f"  source_rotation_y = {rotation_diff['r']}")
            print(f"  source_rotation_z = {rotation_diff['h']}")
            
        except Exception as e:
            self.set_status(f"❌ Ошибка реконструкции: {str(e)}", True)
            print(f"Ошибка при создании HeightMap: {e}")
            import traceback
            traceback.print_exc()

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
        target_volume = self.target_volume_spinbox.value()
        self.panda_app.Target_Volume = target_volume
        
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
                finally:
                    self.textures_combo.currentTextChanged.connect(self.on_texture_set_changed)
            
            self.panda_app.create_ground_plane()
            self.panda_app.ground_plane.setPos(0, 0, ground_plane_z)
            self.plane_pos_z_spinbox.setValue(ground_plane_z)
            
            success_aabb = self.panda_app.perform_AABB_plane()
            
            if success_aabb:
                success_perlin = self.panda_app.perlin_generator.generate_perlin_mesh_from_csg()
                
                if success_perlin:
                    self.set_status(
                        f"✅ Полный процесс выполнен успешно!\n"
                        f"Target Volume: {target_volume}\n"
                        f"Позиция ground_plane: Z={ground_plane_z}"
                    )
                else:
                    self.set_status("⚠️ Не удалось сгенерировать перлин-меш", True)
            else:
                self.set_status("⚠️ Не удалось выполнить AABB plane", True)
        else:
            self.set_status("❌ Не выбран набор моделей или набор не найден", True)