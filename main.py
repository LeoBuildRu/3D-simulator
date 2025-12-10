import sys
import os
import math
import time
import json
import random
import datetime
import tempfile
import glob
import re
import yaml
import subprocess
from pathlib import Path

import win32gui
import win32con

import trimesh
import numpy as np
from scipy.spatial import cKDTree

from PIL import Image, ImageDraw, ImageFilter

import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from panda3d.core import *

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from direct.task import Task
from direct.showbase.DirectObject import DirectObject

from noise import pnoise2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RP_PATH = os.path.join(PROJECT_ROOT, "render_pipeline")
if RP_PATH not in sys.path:
    sys.path.insert(0, RP_PATH)

from rpcore import PointLight
from rpcore import RenderPipeline
from rpcore.util.movement_controller import MovementController

NOISE_AVAILABLE = True
USE_SCIPY = True

class MainWindowManager:
    """Управляет главным окном, содержащим сцену и control panel"""
    def __init__(self, panda_app):
        self.panda_app = panda_app
        self.qt_app = QApplication.instance() or QApplication(sys.argv)
        
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle('3D Scene Viewer')
        
        self.main_window.resize(2300, 1080)
        self.main_window.setMinimumSize(2300, 1080)
        
        try:
            self.main_window.setWindowIcon(QIcon('icon.png'))
        except:
            pass
        
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.panda_container = QWidget()
        self.panda_container.setMinimumSize(1920, 1080)  
        self.panda_container.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #2a2a35;
        """)
        
        self.control_panel = CameraControlGUI(panda_app)
        self.control_panel.setMinimumWidth(380)
        self.control_panel.setMaximumWidth(380)
        
        main_layout.addWidget(self.panda_container, 1)  
        main_layout.addWidget(self.control_panel, 0)    
        
        self.main_window.setCentralWidget(central_widget)
        
        self.panda_timer = QTimer()
        self.panda_timer.timeout.connect(self.update_panda)
        self.panda_timer.start(16)  
        
        self.init_timer = QTimer()
        self.init_timer.setSingleShot(True)
        self.init_timer.timeout.connect(self.initialize_window_integration)
        
        self.panda_app_reference = panda_app
        
        self.panda_window_handle = None
        
    def update_panda(self):
        self.panda_app.taskMgr.step()
        
    def initialize_window_integration(self):
        if hasattr(self.panda_app, 'win') and self.panda_app.win:
            self.panda_window_handle = self.panda_app.win.getWindowHandle()
            
            if self.panda_window_handle:
                container_hwnd = int(self.panda_container.winId())
                
                win32gui.SetParent(self.panda_window_handle, container_hwnd)
                
                style = win32gui.GetWindowLong(self.panda_window_handle, win32con.GWL_STYLE)
                style = style & ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | 
                                 win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | 
                                 win32con.WS_SYSMENU | win32con.WS_BORDER | 
                                 win32con.WS_DLGFRAME)
                style = style | win32con.WS_CHILD
                win32gui.SetWindowLong(self.panda_window_handle, win32con.GWL_STYLE, style)
                
                self.update_panda_window_position()
                
    
    def update_panda_window_position(self):
        if self.panda_window_handle:
            container_rect = self.panda_container.geometry()
            width = container_rect.width()
            height = container_rect.height()
            
            win32gui.MoveWindow(self.panda_window_handle, 0, 0, width, height, True)
            
            win32gui.ShowWindow(self.panda_window_handle, win32con.SW_SHOW)
            win32gui.UpdateWindow(self.panda_window_handle)
                
    
    def resizeEvent(self, event):
        self.update_panda_window_position()
        super().resizeEvent(event)
    
    def show(self):
        screen_geometry = self.qt_app.primaryScreen().geometry()
        window_size = self.main_window.size()
        x = (screen_geometry.width() - window_size.width()) // 2
        y = (screen_geometry.height() - window_size.height()) // 2
        self.main_window.move(x, y)
        
        self.main_window.show()
        
        self.init_timer.start(500)
        
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_panda_window_position)
        self.position_timer.start(100)  
        
        self.main_window.raise_()
        self.main_window.activateWindow()
        
    def run(self):
        self.show()
        return self.qt_app.exec_()

class DepthMapRenderer:
    def __init__(self, base):
        self.base = base
        self.depth_texture = None
        self.overlay_node = None
        self.min_depth = 0.1
        self.max_depth = 100.0
        self.gradient_start = 0.2
        self.gradient_end = 0.4
        self.depth_buffer = None
        self.depth_camera_np = None
        self.setup_depth_render()
        self.setup_depth_overlay()

    def set_gradient_start(self, value):
        self.gradient_start = value
        if self.overlay_node:
            self.overlay_node.setShaderInput("gradientStart", value)
    
    def set_gradient_end(self, value):
        self.gradient_end = value
        if self.overlay_node:
            self.overlay_node.setShaderInput("gradientEnd", value)
    
    def setup_depth_render(self):
        win_width = 1920
        win_height = 1080
        
        self.depth_texture = Texture()
        self.depth_texture.setup_2d_texture(win_width, win_height, Texture.T_float, Texture.F_depth_component32)
        
        fb_props = FrameBufferProperties()
        fb_props.set_depth_bits(32)
        fb_props.set_float_depth(True)
        
        self.depth_buffer = self.base.win.make_texture_buffer("depth_buffer", win_width, win_height, 
                                                              self.depth_texture, to_ram=True)
        
        if self.depth_buffer is None:
            self.depth_buffer = self.base.graphicsEngine.make_output(
                self.base.pipe, "depth_buffer", 0, 
                fb_props,
                WindowProperties.size(win_width, win_height),
                GraphicsPipe.BF_refuse_window,
                self.base.win.get_gsg(), self.base.win
            )
            if self.depth_buffer:
                self.depth_buffer.add_render_texture(
                    self.depth_texture, 
                    GraphicsOutput.RTM_copy_ram, 
                    GraphicsOutput.RTP_depth
                )
        
        self.depth_buffer.set_clear_color_active(False)
        self.depth_buffer.set_clear_depth_active(True)
        self.depth_buffer.set_clear_depth(1.0)
        
        lens = PerspectiveLens()
        lens.set_near_far(self.min_depth, self.max_depth)
        
        main_lens = self.base.cam.node().get_lens()
        if hasattr(main_lens, 'get_fov'):
            main_fov = main_lens.get_fov()
            lens.set_fov(main_fov)
        else:
            lens.set_fov(60)
        
        depth_camera = Camera('depth_camera', lens)
        self.depth_camera_np = self.base.render.attach_new_node(depth_camera)
        
        depth_region = self.depth_buffer.make_display_region(0, 1, 0, 1)
        depth_region.set_camera(self.depth_camera_np)
        depth_region.set_clear_depth_active(True)
        depth_region.set_clear_depth(1.0)
        
        depth_region.set_clear_color_active(False)
            
    def setup_depth_overlay(self):
        win_width = self.base.win.getXSize()
        win_height = self.base.win.getYSize()

        cm = CardMaker('depth_overlay')
        cm.setFrame(-1, 1, -1, 1)

        self.overlay_node = self.base.render2d.attachNewNode(cm.generate())
        self.overlay_node.setPos(0, 0, 0)

        vertex_shader = """
        #version 330
        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        out vec2 texcoord;
        void main() {
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            texcoord = p3d_MultiTexCoord0;
        }
        """

        fragment_shader = """
        #version 330
        uniform sampler2D depthMap;
        uniform float near;
        uniform float far;
        uniform float gradientStart;
        uniform float gradientEnd;
        in vec2 texcoord;
        out vec4 fragColor;

        float linearize_depth(float depth) {
            return (2.0 * near) / (far + near - depth * (far - near));
        }

        void main() {
            float depth = texture(depthMap, texcoord).r;
            float linear_depth = linearize_depth(depth);
            float normalized_depth = (linear_depth - gradientStart) / (gradientEnd - gradientStart);
            normalized_depth = clamp(normalized_depth, 0.0, 1.0);
            float t = 1.0 - normalized_depth;
            vec3 color;

            if (t >= 0.9) {
                float segment_t = (t - 0.9) / 0.1;
                color = mix(vec3(1.0, 0.0, 0.0), vec3(0.5, 0.0, 0.0), segment_t);
            } else if (t >= 0.7) {
                float segment_t = (t - 0.7) / 0.2;
                color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), segment_t);
            } else if (t >= 0.5) {
                float segment_t = (t - 0.5) / 0.2;
                color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), segment_t);
            } else if (t >= 0.3) {
                // Изумрудный -> жёлтый
                float segment_t = (t - 0.3) / 0.2;
                color = mix(vec3(0.1, 0.7, 0.4), vec3(1.0, 1.0, 0.0), segment_t);
            } else if (t >= 0.1) {
                // Голубой -> изумрудный
                float segment_t = (t - 0.1) / 0.2;
                color = mix(vec3(0.0, 0.0, 1.0), vec3(0.1, 0.7, 0.4), segment_t);
            } else {
                float segment_t = t / 0.1;
                // Тёмно-синий -> синий
                color = mix(vec3(0.0, 0.0, 0.3), vec3(0.0, 0.0, 1.0), segment_t);
            }

            fragColor = vec4(color, 1.0);
        }
        """

        shader = Shader.make(Shader.SL_GLSL, vertex_shader, fragment_shader)
        self.overlay_node.setShader(shader)
        depth_stage = TextureStage('depth_stage')
        depth_stage.setMode(TextureStage.M_modulate)
        self.overlay_node.setTexture(depth_stage, self.depth_texture)
        self.overlay_node.setShaderInput("depthMap", self.depth_texture)
        self.overlay_node.setShaderInput("near", self.min_depth)
        self.overlay_node.setShaderInput("far", self.max_depth)
        self.overlay_node.setShaderInput("gradientStart", self.gradient_start)
        self.overlay_node.setShaderInput("gradientEnd", self.gradient_end)
        self.overlay_node.setTransparency(TransparencyAttrib.MAlpha)
        self.overlay_node.setBin("fixed", 50)
        self.overlay_node.setDepthTest(False)
        self.overlay_node.setDepthWrite(False)
        self.overlay_node.hide()
    
    def update_depth_texture(self):
        if not hasattr(self, 'depth_buffer') or not self.depth_buffer:
            return False
        
        original_camera = self.base.camera
        
        try:
            main_cam_pos = original_camera.get_pos(self.base.render)
            main_cam_hpr = original_camera.get_hpr(self.base.render)
            
            self.depth_camera_np.set_pos(main_cam_pos)
            self.depth_camera_np.set_hpr(main_cam_hpr)
            
            if hasattr(original_camera.node(), 'get_lens'):
                main_lens = original_camera.node().get_lens()
                depth_lens = self.depth_camera_np.node().get_lens()
                
                if hasattr(main_lens, 'get_near') and hasattr(main_lens, 'get_far'):
                    self.min_depth = main_lens.get_near()
                    self.max_depth = main_lens.get_far()
                    depth_lens.set_near_far(self.min_depth, self.max_depth)
                
                if hasattr(main_lens, 'get_fov'):
                    fov = main_lens.get_fov()
                    depth_lens.set_fov(fov)
            
            self.base.camera = self.depth_camera_np
            
            self.depth_buffer.set_active(True)
            self.base.graphicsEngine.render_frame()
            
            if self.depth_texture:
                self.depth_texture.reload()
            
            if self.overlay_node and self.depth_texture:
                self.overlay_node.setShaderInput("depthMap", self.depth_texture)
                self.overlay_node.setShaderInput("near", self.min_depth)
                self.overlay_node.setShaderInput("far", self.max_depth)
                self.overlay_node.setShaderInput("gradientStart", self.gradient_start)
                self.overlay_node.setShaderInput("gradientEnd", self.gradient_end)
            
            return True
            
        finally:
            self.base.camera = original_camera
            self.depth_buffer.set_active(False)
    
    def toggle_overlay(self):
        if self.overlay_node:
            if self.overlay_node.isHidden():
                self.overlay_node.show()
            else:
                self.overlay_node.hide()
            return not self.overlay_node.isHidden()
        return False
    
    def set_overlay_visibility(self, visible):
        if self.overlay_node:
            if visible:
                self.overlay_node.show()
            else:
                self.overlay_node.hide()

class Panda3DWidget:
    def __init__(self):
        self.app = None
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()
        self.tk_root.attributes('-topmost', True)
        
    def load_model_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select .gltf model",
            filetypes=[("GLTF files", "*.gltf"), ("GLB files", "*.glb"), ("All files", "*.*")]
        )
        
        if file_path:
            return file_path
        return None

class CameraControlGUI(QWidget):
    def __init__(self, panda_app):
        super().__init__()
        self.panda_app = panda_app
        self.panda_widget = Panda3DWidget()
        
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
            self.panda_app.save_single_render
        )
        render_layout.addWidget(self.save_single_render_button)
        
        self.save_dataset_button = self.create_accent_button(
            "📊 Рендер датасета",
            self.panda_app.save_dataset_render
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

class PerlinMeshGenerator:
    """Класс для генерации перлин-мешей и связанных операций"""
    
    def __init__(self, panda_app):
        self.panda_app = panda_app
        self.last_target_model_trimesh = None
        self.last_best_z = None
        self.test_perlin_mesh = None
        self.last_grid_size = 48
        self.perlin_vertices_before_displace = None
        self.perlin_texcoords_before_displace = None
        self.processed_model = None
        self.current_display_model = None
        
    def generate_perlin_mesh(self, grid_size=48):
        """Генерация перлин-меша с указанным размером сетки"""
        base_vertex_count = grid_size * grid_size
        self.last_grid_size = grid_size
        
        csg_info = self.panda_app.csg_results[-1]
        csg_node = csg_info["result_node"]
        pos = csg_node.getPos()
        min_bound, max_bound = csg_node.getTightBounds()
        
        if min_bound is None or max_bound is None:
            size_x = 10.0
            size_y = 10.0
            size_z = 1.0
        else:
            size_x = max_bound.x - min_bound.x
            size_y = max_bound.y - min_bound.y
            size_z = max_bound.z - min_bound.z
            
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(project_root, "models")

            max_volumes = {
                os.path.join(models_dir, "Scania-Napolnitel.gltf"): 24,
                os.path.join(models_dir, "Kamaz-Napolnitel.gltf"): 10,
                os.path.join(models_dir, "Hooper1-Napolnitel.gltf"): 48,
                os.path.join(models_dir, "Hooper2-Napolnitel.gltf"): 88
            }
            
            max_coefficient = 48
            min_coefficient = 14
            
            target_model_path = None
            for model in self.panda_app.loaded_models:
                model_id = id(model)
                if model_id in self.panda_app.model_paths:
                    if self.panda_app.Target_Napolnitel in self.panda_app.model_paths[model_id]:
                        target_model_path = self.panda_app.model_paths[model_id]
                        break
            
            if target_model_path in max_volumes:
                max_volume = max_volumes[target_model_path]
                volume_ratio = self.panda_app.Target_Volume / max_volume
                coefficient = max_coefficient - (max_coefficient - min_coefficient) * volume_ratio
                coefficient = max(min_coefficient, min(max_coefficient, coefficient))
            else:
                coefficient = (max_coefficient + min_coefficient) / 2
            
            size_z = size_z * coefficient
        
        texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)
        
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("perlin_data", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        step_x = size_x / (grid_size - 1) if grid_size > 1 else 0
        step_y = size_y / (grid_size - 1) if grid_size > 1 else 0
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        base_z = pos.getZ()
        
        noise_scale = 4
        octaves = 12
        persistence = 0.01
        lacunarity = 1.0
        seed = random.randint(0, 10000)
        
        vertices = []
        normals_list = []
        texcoords_list = []
        
        for y in range(grid_size):
            for x in range(grid_size):
                world_x = x * step_x - half_size_x
                world_y = y * step_y - half_size_y
                
                nx = (x / grid_size) * noise_scale
                ny = (y / grid_size) * noise_scale
                value = 0.0
                amplitude = 1.0
                frequency = 1.0
                if NOISE_AVAILABLE:
                    for i in range(octaves):
                        value += pnoise2(
                            nx * frequency + seed, 
                            ny * frequency + seed, 
                            octaves=1
                        ) * amplitude
                        amplitude *= persistence
                        frequency *= lacunarity
                    value = (value + 1.0) / 2.0
                else:
                    value = (math.sin(x / 10.0) * 0.5 + math.cos(y / 10.0) * 0.5 + 1.0) / 2.0
                
                height_scale = size_z * 0.66
                world_z = base_z + value * height_scale
                
                vertex.addData3f(world_x, world_y, world_z)
                vertices.append((world_x, world_y, world_z))
                
                # UV-координаты с учетом textureRepeatX/Y
                normalized_u = x / (grid_size - 1) if grid_size > 1 else 0.0
                normalized_v = y / (grid_size - 1) if grid_size > 1 else 0.0
                u = normalized_u * texture_repeatX
                v = normalized_v * texture_repeatY
                texcoord.addData2f(u, v)
                texcoords_list.append((u, v))

        self.extract_perlin_mesh_data(vertices, texcoords_list, grid_size)

        self.perlin_vertices_before_displace = vertices.copy()
        self.perlin_texcoords_before_displace = texcoords_list.copy()
        
        strength = self.panda_app.current_texture_set.get('strength', 0.14)
        height_texture_path = self._get_height_texture_path()
        
        height_array, tex_width, tex_height = self._load_height_array(height_texture_path)
        
        vertices = self._apply_displacement(vertices, texcoords_list, height_array, tex_width, tex_height, strength)
        
        falloff_config = self._get_falloff_config()
        vertices = self._apply_falloff(vertices, size_x, size_y, base_z, falloff_config)
        
        normals_list = self._calculate_normals(vertices, grid_size)
        
        base_vdata = GeomVertexData("perlin_base_data_with_displacement", vdata.getFormat(), Geom.UHStatic)
        base_vertex = GeomVertexWriter(base_vdata, "vertex")
        base_normal = GeomVertexWriter(base_vdata, "normal")
        base_texcoord = GeomVertexWriter(base_vdata, "texcoord")
        
        for i in range(base_vertex_count):
            base_vertex.addData3f(vertices[i][0], vertices[i][1], vertices[i][2])
            base_normal.addData3f(normals_list[i][0], normals_list[i][1], normals_list[i][2])
            base_texcoord.addData2f(texcoords_list[i][0], texcoords_list[i][1])
        
        base_prim = GeomTriangles(Geom.UHStatic)
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i1 = y * grid_size + x
                i2 = y * grid_size + (x + 1)
                i3 = (y + 1) * grid_size + x
                i4 = (y + 1) * grid_size + (x + 1)
                base_prim.addVertices(i1, i3, i2)
                base_prim.addVertices(i2, i3, i4)
        base_prim.closePrimitive()
        
        base_geom = Geom(base_vdata)
        base_geom.addPrimitive(base_prim)
        
        base_node = GeomNode("perlin_base_mesh_with_displacement")
        base_node.addGeom(base_geom)
        
        perlin_base_np = NodePath(base_node)
        
        return perlin_base_np
    
    def extract_perlin_mesh_data(self, vertices, texcoords_list, grid_size):
        """Извлечение и логирование данных перлин-меша"""
        if vertices:
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]
            z_coords = [v[2] for v in vertices]
            
            size_x = max(x_coords) - min(x_coords)
            size_y = max(y_coords) - min(y_coords)
            size_z = max(z_coords) - min(z_coords)
        
        for i in range(min(10, len(vertices))):
            v = vertices[i]
            t = texcoords_list[i] if i < len(texcoords_list) else (0, 0)
        
        corners = [
            (0, 0),  
            (grid_size-1, 0),  
            (0, grid_size-1),  
            (grid_size-1, grid_size-1)  
        ]
        
        for corner_x, corner_y in corners:
            index = corner_y * grid_size + corner_x
            if index < len(vertices):
                v = vertices[index]
                t = texcoords_list[index] if index < len(texcoords_list) else (0, 0)
        
        if grid_size > 1 and vertices:
            step_x = abs(vertices[1][0] - vertices[0][0])  
            step_y = abs(vertices[grid_size][1] - vertices[0][1])  
    
    def create_mesh_from_perlin_data(self):
        """Создание меша из сохраненных данных перлина"""
        if hasattr(self.panda_app, 'final_model') and self.panda_app.final_model:
            self.panda_app.final_model.removeNode()
            self.panda_app.final_model = None

        if not hasattr(self, 'perlin_vertices_before_displace') or not self.perlin_vertices_before_displace:
            print("Нет сохраненных данных перлина")
            return False
        
        if not hasattr(self, 'last_target_model_trimesh') or self.last_target_model_trimesh is None:
            print("Нет сохраненной target_model_trimesh")
            return False
        
        texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)
        strength = self.panda_app.current_texture_set.get('strength', 0.14)
        
        vertices = self.perlin_vertices_before_displace.copy()
        grid_size = self.last_grid_size
        
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        
        size_x = max(x_coords) - min(x_coords)
        size_y = max(y_coords) - min(y_coords)
        
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        
        texcoords_list = []
        for i, (vx, vy, vz) in enumerate(vertices):
            normalized_u = (vx + half_size_x) / size_x
            normalized_v = (vy + half_size_y) / size_y
            u = normalized_u * texture_repeatX
            v = normalized_v * texture_repeatY
            texcoords_list.append((u, v))
        
        height_texture_path = self._get_height_texture_path()
        height_array, tex_width, tex_height = self._load_height_array(height_texture_path)
        
        vertices = self._apply_displacement(vertices, texcoords_list, height_array, tex_width, tex_height, strength)
        
        falloff_config = self._get_falloff_config()
        vertices = self._apply_falloff(vertices, size_x, size_y, vertices[0][2] if vertices else 0, falloff_config)
        
        normals = self._calculate_normals(vertices, grid_size)
        
        perlin_np = self._create_geom_from_vertices(vertices, normals, texcoords_list, grid_size, "recreated_perlin_mesh")
        
        perlin_np.setPos(0, 0, self.last_best_z if hasattr(self, 'last_best_z') else 0)
        
        perlin_model_trimesh = self.panda_app.panda_to_trimesh(perlin_np)
        
        try:
            final_result_trimesh = trimesh.boolean.difference(
                [self.last_target_model_trimesh, perlin_model_trimesh],
                engine='blender'
            )
            
            if final_result_trimesh.is_empty:
                print("Boolean разность вернула пустой меш")
                perlin_np.removeNode()
                return False
                
            self.panda_app.final_model = self.panda_app.trimesh_to_panda(final_result_trimesh)
            perlin_np.removeNode()
            
        except Exception as e:
            print(f"Ошибка при выполнении boolean разности: {e}")
            perlin_np.removeNode()
            return False
        
        # self.panda_app.fix_uv_coordinates_for_final_model(self.panda_app.final_model, texture_repeatX, texture_repeatY)
        
        geom_node = self.panda_app.final_model.node()
        if geom_node.getNumGeoms() > 0:
            geom = geom_node.getGeom(0)
            vdata = geom.getVertexData()
            
            new_vdata = GeomVertexData(vdata)
            new_geom = Geom(new_vdata)
            
            vertex_reader = GeomVertexReader(new_vdata, "vertex")
            normal_reader = GeomVertexReader(new_vdata, "normal")
            texcoord_writer = GeomVertexWriter(new_vdata, "texcoord")
            
            vertices_with_normals = []
            while not vertex_reader.isAtEnd():
                vertex = vertex_reader.getData3f()
                normal = normal_reader.getData3f() if not normal_reader.isAtEnd() else Vec3(0, 0, 1)
                vertices_with_normals.append((vertex, normal))
            
            texcoord_writer.setRow(0)
            
            up_threshold = 0.7
            
            csg_info = self.panda_app.csg_results[-1]
            csg_node = csg_info["result_node"]
            min_bound, max_bound = csg_node.getTightBounds()
            if min_bound is not None and max_bound is not None:
                size_x = max_bound.x - min_bound.x
                size_y = max_bound.y - min_bound.y
                size_z = max_bound.z - min_bound.z
            else:
                size_x = 10.0
                size_y = 10.0
                size_z = 1.0
            
            half_size_x = size_x / 2.0
            half_size_y = size_y / 2.0
            
            for vertex, normal in vertices_with_normals:
                u = (vertex.x + half_size_x) / size_x * texture_repeatX
                v = (vertex.y + half_size_y) / size_y * texture_repeatY
                texcoord_writer.setData2f(u, v)
            
            for i in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(i)
                new_geom.addPrimitive(prim)
            
            new_geom_node = GeomNode("textured_final_mesh")
            new_geom_node.addGeom(new_geom)
            
            self.panda_app.final_model.removeNode()
            self.panda_app.final_model = self.panda_app.render.attachNewNode(new_geom_node)
            self.panda_app.final_model.setPos(0, 0, 0)
        
        self._apply_textures_and_material(self.panda_app.final_model)
        
        return True
    
    def generate_perlin_mesh_from_csg(self):
        """Генерация перлин-меша на основе CSG операции"""
        if hasattr(self.panda_app, 'test_perlin_mesh') and self.panda_app.test_perlin_mesh is not None:
            if self.panda_app.test_perlin_mesh in self.panda_app.loaded_models:
                self.panda_app.loaded_models.remove(self.panda_app.test_perlin_mesh)
            self.panda_app.test_perlin_mesh.removeNode()
            self.panda_app.test_perlin_mesh = None

        if hasattr(self.panda_app, 'final_model') and self.panda_app.final_model:
            if self.panda_app.final_model in self.panda_app.loaded_models:
                self.panda_app.loaded_models.remove(self.panda_app.final_model)
            self.panda_app.final_model.removeNode()
            self.panda_app.final_model = None

        target_model = None
        target_model_path = None
        for model in self.panda_app.loaded_models:
            model_id = id(model)
            if model_id in self.panda_app.model_paths:
                if self.panda_app.Target_Napolnitel in self.panda_app.model_paths[model_id]:
                    target_model = model
                    target_model_path = self.panda_app.model_paths[model_id]
                    break
        
        if target_model is None:
            if self.panda_app.loaded_models:
                for model in self.panda_app.loaded_models:
                    model_id = id(model)
            return False
        
        target_model_trimesh = self._prepare_target_model_for_boolean(target_model)
        self.last_target_model_trimesh = target_model_trimesh
        
        if target_model_trimesh is None:
            target_model.setScale(1.0, 1.0, 1.0)
            return False
        
        target_model.setScale(1.0, 1.0, 1.0)
        target_model.setPos(0.0, 0.0, 0.0)

        if hasattr(self.panda_app, 'dynamic_perlin_model') and self.panda_app.dynamic_perlin_model:
            self.panda_app.dynamic_perlin_model.removeNode()
            self.panda_app.dynamic_perlin_model = None

        perlin_base_np = self.generate_perlin_mesh(grid_size=48)
        
        ground_pos = self.panda_app.ground_plane.getPos()
        perlin_base_np.setPos(ground_pos.x, ground_pos.y, ground_pos.z - 2.25)
        
        self.panda_app.loaded_models.append(perlin_base_np)

        target_volume = self.panda_app.Target_Volume
        tolerance = 0.2
        min_z = -2
        max_z = 2
        max_iterations = 50

        best_z = perlin_base_np.getZ()
        best_volume = None
        best_error = float('inf')

        if hasattr(self.panda_app, 'current_display_model') and self.panda_app.current_display_model:
            self.panda_app.current_display_model.removeNode()
        self.panda_app.current_display_model = None

        search_points = [min_z + (max_z - min_z) * i / 10 for i in range(11)]
        search_volumes = []

        for z in search_points:
            perlin_base_np.setPos(0, 0, z)
            perlin_model_trimesh_ = self.panda_app.panda_to_trimesh(perlin_base_np)
            
            result_csg = trimesh.boolean.difference(
                [target_model_trimesh, perlin_model_trimesh_],
                engine='blender'
            )
            
            model_csg_plane_1 = self.panda_app.trimesh_to_panda(result_csg)
            self._setup_transparent_material(model_csg_plane_1)
            
            if self.current_display_model is not None:
                self.current_display_model.removeNode()
            
            self.current_display_model = model_csg_plane_1
            self.current_display_model.reparentTo(self.panda_app.render)
            
            volume = self.panda_app.calculate_mesh_volume(self.current_display_model)
            error = abs(volume - target_volume)
            search_volumes.append((z, volume, error))
            
            if error < best_error:
                best_error = error
                best_z = z
                best_volume = volume
        
        if best_error <= tolerance:
            perlin_base_np.setPos(0, 0, best_z)
            if self.current_display_model is not None:
                self.current_display_model.removeNode()
            return best_z

        search_volumes.sort(key=lambda x: x[2])
        best_points = search_volumes[:3]

        if len(best_points) >= 2:
            z_values = [p[0] for p in best_points]
            min_search_z = min(z_values)
            max_search_z = max(z_values)
            
            range_expand = (max_search_z - min_search_z) * 0.2
            min_search_z = max(min_z, min_search_z - range_expand)
            max_search_z = min(max_z, max_search_z + range_expand)
            
            phi = 0.618
            a = min_search_z
            b = max_search_z
            
            x1 = b - phi * (b - a)
            x2 = a + phi * (b - a)
            
            iteration = 0
            while (b - a) > 0.01 and iteration < max_iterations:
                iteration += 1
                
                vol1, err1 = self._evaluate_z_position(perlin_base_np, target_model_trimesh, x1, target_volume)
                vol2, err2 = self._evaluate_z_position(perlin_base_np, target_model_trimesh, x2, target_volume)
                
                if err1 < best_error:
                    best_error = err1
                    best_z = x1
                    best_volume = vol1
                
                if err2 < best_error:
                    best_error = err2
                    best_z = x2
                    best_volume = vol2
                
                if best_error <= tolerance:
                    break
                
                if err1 < err2:
                    b = x2
                    x2 = x1
                    err2 = err1
                    x1 = b - phi * (b - a)
                    vol1, err1 = self._evaluate_z_position(perlin_base_np, target_model_trimesh, x1, target_volume)
                else:
                    a = x1
                    x1 = x2
                    err1 = err2
                    x2 = a + phi * (b - a)
                    vol2, err2 = self._evaluate_z_position(perlin_base_np, target_model_trimesh, x2, target_volume)
                
                if best_error <= tolerance:
                    break

        perlin_base_np.setPos(0, 0, best_z)
        self.last_best_z = best_z

        if self.current_display_model is not None:
            self.current_display_model.removeNode()
            self.current_display_model = None

        perlin_base_np.removeNode()
        
        perlin_detailed_np = self.generate_perlin_mesh(grid_size=1024)
        perlin_detailed_np.setPos(0, 0, best_z)

        perlin_model_trimesh = self.panda_app.panda_to_trimesh(perlin_detailed_np)

        final_result_trimesh = trimesh.boolean.difference(
            [target_model_trimesh, perlin_model_trimesh],
            engine='blender'
        )

        self.panda_app.final_model = self.panda_app.trimesh_to_panda(final_result_trimesh)

        texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)
        
        # self.panda_app.fix_uv_coordinates_for_final_model(self.panda_app.final_model, texture_repeatX, texture_repeatY)
        
        geom_node = self.panda_app.final_model.node()
        if geom_node.getNumGeoms() > 0:
            geom = geom_node.getGeom(0)
            vdata = geom.getVertexData()
            
            new_vdata = GeomVertexData(vdata)
            new_geom = Geom(new_vdata)
            
            vertex_reader = GeomVertexReader(new_vdata, "vertex")
            normal_reader = GeomVertexReader(new_vdata, "normal")
            texcoord_writer = GeomVertexWriter(new_vdata, "texcoord")
            
            vertices_with_normals = []
            while not vertex_reader.isAtEnd():
                vertex = vertex_reader.getData3f()
                normal = normal_reader.getData3f() if not normal_reader.isAtEnd() else Vec3(0, 0, 1)
                vertices_with_normals.append((vertex, normal))
            
            texcoord_writer.setRow(0)
            
            up_threshold = 0.7
            
            csg_info = self.panda_app.csg_results[-1]
            csg_node = csg_info["result_node"]
            min_bound, max_bound = csg_node.getTightBounds()
            if min_bound is not None and max_bound is not None:
                size_x = max_bound.x - min_bound.x
                size_y = max_bound.y - min_bound.y
                size_z = max_bound.z - min_bound.z
            else:
                size_x = 10.0
                size_y = 10.0
                size_z = 1.0
            
            half_size_x = size_x / 2.0
            half_size_y = size_y / 2.0
            
            for vertex, normal in vertices_with_normals:
                u = (vertex.x + half_size_x) / size_x * texture_repeatX
                v = (vertex.y + half_size_y) / size_y * texture_repeatY
                texcoord_writer.setData2f(u, v)
            
            for i in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(i)
                new_geom.addPrimitive(prim)
            
            new_geom_node = GeomNode("textured_final_mesh")
            new_geom_node.addGeom(new_geom)
            
            self.panda_app.final_model.removeNode()
            self.panda_app.final_model = self.panda_app.render.attachNewNode(new_geom_node)
            self.panda_app.final_model.setPos(0, 0, 0)
        
        self._apply_textures_and_material(self.panda_app.final_model)

        target_model.hide()
        perlin_detailed_np.hide()
        
        return True
    
    def _load_height_array(self, height_texture_path):
        """Загружает и обрабатывает текстуру высот"""
        height_image = Image.open(height_texture_path).convert('L')
        height_array = np.array(height_image, dtype=np.float32)
        tex_height, tex_width = height_array.shape
        
        height_min = np.min(height_array)
        height_max = np.max(height_array)
        height_array = (height_array - height_min) / (height_max - height_min)
        height_array = np.power(height_array, 0.7)
        
        return height_array, tex_width, tex_height
    
    def _get_falloff_config(self):
        """Возвращает конфигурацию falloff"""
        return {
            'left': {
                'width_ratio': 0.20,
                'target_offset': -0.15
            },
            'right': {
                'width_ratio': 0.20,
                'target_offset': -0.15
            },
            'front': {
                'width_ratio': 0.15,
                'target_offset': -0.2
            },
            'back': {
                'width_ratio': 0.25,
                'target_offset': -0.2
            }
        }
    
    def _apply_falloff(self, vertices, size_x, size_y, base_z, falloff_config):
        """Применяет falloff к вершинам"""
        falloff_widths = {
            'left': size_x * falloff_config['left']['width_ratio'],
            'right': size_x * falloff_config['right']['width_ratio'],
            'front': size_y * falloff_config['front']['width_ratio'],
            'back': size_y * falloff_config['back']['width_ratio']
        }
        
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        
        edge_samples = {'left': [], 'right': [], 'front': [], 'back': []}
        
        for i, (vx, vy, vz) in enumerate(vertices):
            if vx <= (-half_size_x + falloff_widths['left'] * 0.5):
                edge_samples['left'].append(vz)
            if vx >= (half_size_x - falloff_widths['right'] * 0.5):
                edge_samples['right'].append(vz)
            if vy <= (-half_size_y + falloff_widths['front'] * 0.5):
                edge_samples['front'].append(vz)
            if vy >= (half_size_y - falloff_widths['back'] * 0.5):
                edge_samples['back'].append(vz)
        
        target_heights = {}
        for side in ['left', 'right', 'front', 'back']:
            if edge_samples[side]:
                avg_height = sum(edge_samples[side]) / len(edge_samples[side])
                target_heights[side] = avg_height + falloff_config[side]['target_offset']
            else:
                target_heights[side] = base_z + falloff_config[side]['target_offset']

        falloff_vertices = []
        
        for i, (vx, vy, original_z) in enumerate(vertices):
            side_factors = {}
            
            dist_to_left = vx - (-half_size_x)
            if dist_to_left <= falloff_widths['left']:
                normalized_dist = dist_to_left / falloff_widths['left']
                side_factors['left'] = normalized_dist * normalized_dist * (3 - 2 * normalized_dist)
            else:
                side_factors['left'] = 1.0
                
            dist_to_right = half_size_x - vx
            if dist_to_right <= falloff_widths['right']:
                normalized_dist = dist_to_right / falloff_widths['right']
                side_factors['right'] = normalized_dist * normalized_dist * (3 - 2 * normalized_dist)
            else:
                side_factors['right'] = 1.0
                
            dist_to_front = vy - (-half_size_y)
            if dist_to_front <= falloff_widths['front']:
                normalized_dist = dist_to_front / falloff_widths['front']
                side_factors['front'] = normalized_dist * normalized_dist * (3 - 2 * normalized_dist)
            else:
                side_factors['front'] = 1.0
                
            dist_to_back = half_size_y - vy
            if dist_to_back <= falloff_widths['back']:
                normalized_dist = dist_to_back / falloff_widths['back']
                side_factors['back'] = normalized_dist * normalized_dist * (3 - 2 * normalized_dist)
            else:
                side_factors['back'] = 1.0
            
            overall_factor = min(side_factors.values())
            
            active_sides = []
            active_weights = []
            
            for side in ['left', 'right', 'front', 'back']:
                if side_factors[side] < 1.0:
                    active_sides.append(side)
                    active_weights.append(1.0 - side_factors[side])
            
            if active_sides:
                total_weight = sum(active_weights)
                normalized_weights = [w / total_weight for w in active_weights]
                
                weighted_target = sum(target_heights[side] * weight 
                                    for side, weight in zip(active_sides, normalized_weights))
            else:
                weighted_target = original_z
            
            new_z = original_z * overall_factor + weighted_target * (1.0 - overall_factor)
            
            falloff_vertices.append((vx, vy, new_z))
        
        return falloff_vertices
    
    def _calculate_normals(self, vertices, grid_size):
        """Рассчитывает нормали для сетки вершин"""
        calculated_normals = [Vec3(0, 0, 1) for _ in range(len(vertices))]
        
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                idx = y * grid_size + x
                
                p = vertices[idx]
                px1 = vertices[idx + 1]
                px0 = vertices[idx - 1]
                py1 = vertices[idx + grid_size]
                py0 = vertices[idx - grid_size]
                
                dx_vec = Vec3(px1[0] - px0[0], px1[1] - px0[1], px1[2] - px0[2])
                dy_vec = Vec3(py1[0] - py0[0], py1[1] - py0[1], py1[2] - py0[2])
                
                normal_vec = dx_vec.cross(dy_vec)
                if normal_vec.length() > 0:
                    normal_vec.normalize()
                    calculated_normals[idx] = normal_vec
        
        for i in range(len(vertices)):
            if calculated_normals[i].length() == 0:
                x = i % grid_size
                y = i // grid_size
                
                avg_normal = Vec3(0, 0, 0)
                count = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size:
                            neighbor_idx = ny * grid_size + nx
                            if calculated_normals[neighbor_idx].length() > 0:
                                avg_normal += calculated_normals[neighbor_idx]
                                count += 1
                
                if count > 0 and avg_normal.length() > 0:
                    avg_normal.normalize()
                    calculated_normals[i] = avg_normal
        
        normals_list = []
        for i in range(len(vertices)):
            n = calculated_normals[i]
            normals_list.append((n.x, n.y, n.z))
        
        return normals_list
    
    def _create_geom_from_vertices(self, vertices, normals, texcoords_list, grid_size, name="perlin_mesh"):
        """Создает геометрию из вершин, нормалей и UV-координат"""
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData(name, format, Geom.UHStatic)
        
        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")
        
        for i, (vx, vy, vz) in enumerate(vertices):
            vertex_writer.addData3f(vx, vy, vz)
            normal_writer.addData3f(normals[i][0], normals[i][1], normals[i][2])
            texcoord_writer.addData2f(texcoords_list[i][0], texcoords_list[i][1])
        
        prim = GeomTriangles(Geom.UHStatic)
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i1 = y * grid_size + x
                i2 = y * grid_size + (x + 1)
                i3 = (y + 1) * grid_size + x
                i4 = (y + 1) * grid_size + (x + 1)
                prim.addVertices(i1, i3, i2)
                prim.addVertices(i2, i3, i4)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode(name)
        geom_node.addGeom(geom)
        
        return NodePath(geom_node)
    
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
    
    def _evaluate_z_position(self, perlin_base_np, target_model_trimesh, z, target_volume):
        """Оценивает объем при определенной Z-позиции"""
        perlin_base_np.setPos(0, 0, z)
        perlin_model_trimesh_ = self.panda_app.panda_to_trimesh(perlin_base_np)
        
        result_csg = trimesh.boolean.difference(
            [target_model_trimesh, perlin_model_trimesh_],
            engine='blender'
        )
        
        model_csg_plane_1 = self.panda_app.trimesh_to_panda(result_csg)
        self._setup_transparent_material(model_csg_plane_1)
        
        if self.current_display_model is not None:
            self.current_display_model.removeNode()
        
        self.current_display_model = model_csg_plane_1
        self.current_display_model.reparentTo(self.panda_app.render)
        
        volume = self.panda_app.calculate_mesh_volume(self.current_display_model)
        error = abs(volume - target_volume)
        
        return volume, error
    
    # Вспомогательные методы, которые использовались в перенесенных методах
    def _get_height_texture_path(self):
        """Получает путь к текстуре высот"""
        if 'displacement' in self.panda_app.current_texture_set:
            height_texture_path = self.panda_app.current_texture_set['displacement']
        elif 'height' in self.panda_app.current_texture_set:
            height_texture_path = self.panda_app.current_texture_set['height']
        else:
            height_texture_path = "textures/stones_8k/rocks_ground_01_disp_8k.jpg"
        
        if not os.path.exists(height_texture_path):
            height_texture_path = "textures/stones_8k/rocks_ground_01_disp_8k.jpg"
        
        return height_texture_path
    
    def _apply_displacement(self, vertices, texcoords_list, height_array, tex_width, tex_height, strength):
        """Применяет displacement к вершинам"""
        displaced_vertices = []
        for i, (vx, vy, vz) in enumerate(vertices):
            u, v = texcoords_list[i]
            
            u_repeated = u % 1.0
            v_repeated = v % 1.0
            
            tex_x = u_repeated * (tex_width - 1)
            tex_y = v_repeated * (tex_height - 1)
            
            x1 = max(0, min(tex_width - 1, int(tex_x)))
            y1 = max(0, min(tex_height - 1, int(tex_y)))
            x2 = max(0, min(tex_width - 1, x1 + 1))
            y2 = max(0, min(tex_height - 1, y1 + 1))
            
            dx = tex_x - x1
            dy = tex_y - y1
            
            h11 = height_array[y1, x1]
            h12 = height_array[y2, x1]
            h21 = height_array[y1, x2]
            h22 = height_array[y2, x2]
            
            hx1 = h11 * (1 - dx) + h21 * dx
            hx2 = h12 * (1 - dx) + h22 * dx
            height_value = hx1 * (1 - dy) + hx2 * dy
            
            displacement = (height_value - 0.5) * strength
            new_z = vz + displacement
            displaced_vertices.append((vx, vy, new_z))
        
        return displaced_vertices
    
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
    
    def _apply_textures_and_material(self, model_np):
        """Применяет текстуры и материал к модели"""
        if 'diffuse' in self.panda_app.current_texture_set:
            diffuse_path = self.panda_app.current_texture_set['diffuse']
        elif 'albedo' in self.panda_app.current_texture_set:
            diffuse_path = self.panda_app.current_texture_set['albedo']
        else:
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        
        normal_path = self.panda_app.current_texture_set.get('normal', 
            "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg")
        
        roughness_path = self.panda_app.current_texture_set.get('roughness', None)
        
        if not os.path.exists(diffuse_path):
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        
        if not os.path.exists(normal_path):
            normal_path = "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"
        
        diffuse_tex = self.panda_app.loader.loadTexture(diffuse_path)
        if diffuse_tex:
            diffuse_tex.set_format(Texture.F_srgb)
            diffuse_tex.setMinfilter(Texture.FTLinearMipmapLinear)
            diffuse_tex.setMagfilter(Texture.FTLinear)
            diffuse_tex.setWrapU(Texture.WMRepeat)
            diffuse_tex.setWrapV(Texture.WMRepeat)
            model_np.setTexture(diffuse_tex, 1)
        
        normal_tex = self.panda_app.loader.loadTexture(normal_path)
        if normal_tex:
            normal_tex.setMinfilter(Texture.FTLinearMipmapLinear)
            normal_tex.setMagfilter(Texture.FTLinear)
            normal_tex.setWrapU(Texture.WMRepeat)
            normal_tex.setWrapV(Texture.WMRepeat)
            
            normal_stage = TextureStage('normal')
            normal_stage.setMode(TextureStage.MNormal)
            model_np.setTexture(normal_stage, normal_tex)
        
        if roughness_path and os.path.exists(roughness_path):
            roughness_tex = self.panda_app.loader.loadTexture(roughness_path)
            if roughness_tex:
                roughness_tex.setMinfilter(Texture.FTLinearMipmapLinear)
                roughness_tex.setMagfilter(Texture.FTLinear)
                roughness_tex.setWrapU(Texture.WMRepeat)
                roughness_tex.setWrapV(Texture.WMRepeat)
                
                roughness_stage = TextureStage('roughness')
                roughness_stage.setMode(TextureStage.MModulate)
                model_np.setTexture(roughness_stage, roughness_tex)
        
        base_material = Material("perlin_base_material_with_displacement")
        base_material.setDiffuse((0.4, 0.4, 0.4, 1.0))
        base_material.setAmbient((0.7, 0.7, 0.7, 1.0))
        base_material.setSpecular((0.1, 0.1, 0.1, 1.0))
        base_material.setShininess(5.0)
        base_material.setRoughness(0.85)
        base_material.setMetallic(0.0)
        base_material.setRefractiveIndex(1.5)
        model_np.setMaterial(base_material, 1)
        
        model_np.setShaderAuto()
        model_np.setTwoSided(True)
        model_np.setBin("fixed", 0)
        model_np.setDepthOffset(1)

class MyApp(ShowBase):
    def __init__(self):
        self.render_pipeline = RenderPipeline()
        self.render_pipeline.pre_showbase_init()
        
        loadPrcFileData("", "win-size 1920 1080")  
        loadPrcFileData("", "window-type embedded")
        loadPrcFileData("", "fullscreen false")
        loadPrcFileData("", "undecorated true")  

        ShowBase.__init__(self)

        self.render_pipeline.create(self)

        self.current_texture_set = {
            'diffuse': "textures/stones_8k/rocks_ground_01_diff_8k.jpg",
            'displacement': "textures/stones_8k/rocks_ground_01_disp_8k.jpg",
            'normal': "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg",
            'roughness': "textures/stones_8k/rocks_ground_01_rough_8k.jpg",
            'textureRepeatX': 1.35,
            'textureRepeatY': 3.2,
            'strength': 0.14,
            'textureRepeatU': 160.0,
            'textureRepeatV': 160.0
        }
        
        self.last_target_model_trimesh = None
        self.last_best_z = None
        self.test_perlin_mesh = None  
        self.last_grid_size = 48  

        self.loaded_models = []
        self.model_paths = {}

        self.setup_scene()

        self.next_model_x = 0

        self.current_model_set = None

        self.Target_Cuzov = "Scania-Cuzov.gltf"
        self.Target_Y_offset = 0
        self.Target_Volume = 20
        self.Target_Napolnitel = "Scania-Napolnitel.gltf"
        self.Target_height_val = 66

        self.current_view = "perspective"

        self.mouse_rotation_enabled = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.camera_rotation_speed = 0.5

        self.drag_drop_mode = False
        self.selected_model = None
        self.drag_start_pos = None
        self.drag_start_mouse_pos = None
        self.drag_sensitivity = 2.0

        self.disable_mouse()

        self.key_map = {
            "forward": False, 
            "backward": False,
            "left": False,    
            "right": False,   
            "up": False,      
            "down": False     
        }
        
        self.accept("w", self.set_key, ["forward", True])
        self.accept("w-up", self.set_key, ["forward", False])
        self.accept("s", self.set_key, ["backward", True])
        self.accept("s-up", self.set_key, ["backward", False])
        self.accept("a", self.set_key, ["left", True])
        self.accept("a-up", self.set_key, ["left", False])
        self.accept("d", self.set_key, ["right", True])
        self.accept("d-up", self.set_key, ["right", False])
        self.accept("space", self.set_key, ["up", True])
        self.accept("space-up", self.set_key, ["up", False])
        self.accept("shift", self.set_key, ["down", True])
        self.accept("shift-up", self.set_key, ["down", False])

        self.base_perlin_model = None
        self.dynamic_perlin_model = None

        self.height_values = []

        self.accept("mouse1", self.handle_mouse_left)
        self.accept("mouse1-up", self.handle_mouse_left_up)
        self.accept("mouse3", self.handle_mouse_right)

        self.accept("wheel_up", self.zoom_camera, [1.1])
        self.accept("wheel_down", self.zoom_camera, [0.9])

        self.zoom_sensitivity = 1.1

        self.current_z = 0

        self.final_model = None

        self.ground_plane = None
        self.plane_size_x = 100.0
        self.plane_size_y = 100.0
        
        self.taskMgr.add(self.move_camera_task, "move_camera_task")
        self.taskMgr.add(self.mouse_rotation_task, "mouse_rotation_task")
        self.taskMgr.add(self.drag_drop_task, "drag_drop_task")
        
        self.depth_renderer = None
        self.init_depth_renderer()

        self.perlin_generator = PerlinMeshGenerator(self)

    def setup_window_for_parenting(self, parent_hwnd):
        if hasattr(self, 'win') and self.win:
            try:
                hwnd = self.win.getWindowHandle()
                
                win32gui.SetParent(hwnd, parent_hwnd)
                
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                style = style & ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | 
                                 win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | 
                                 win32con.WS_SYSMENU | win32con.WS_BORDER | 
                                 win32con.WS_DLGFRAME)
                style = style | win32con.WS_CHILD
                win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
                
                return hwnd
            except Exception as e:
                return None
        return None

    def set_texture_set(self, texture_config):
        new_texture_set = texture_config.copy()
        
        defaults = {
            'textureRepeatX': 1.35,
            'textureRepeatY': 3.2,
            'strength': 0.14,
            'textureRepeatU': 160.0,
            'textureRepeatV': 160.0
        }
        
        for key, default_value in defaults.items():
            if key not in new_texture_set:
                new_texture_set[key] = default_value
        
        if 'albedo' in new_texture_set and 'diffuse' not in new_texture_set:
            new_texture_set['diffuse'] = new_texture_set['albedo']
        if 'height' in new_texture_set and 'displacement' not in new_texture_set:
            new_texture_set['displacement'] = new_texture_set['height']
        
        self.current_texture_set = new_texture_set
        
        if hasattr(self, 'final_model') and self.final_model is not None:
            self.perlin_generator.create_mesh_from_perlin_data()
        
        return new_texture_set

    def add_scene_points(self):
        top_points = [(-1.03, -2.25, 2.4), (-1.03, 2.4, 2.4), (1.075, 2.4, 2.4), (1.075, -2.25, 2.4)]
        
        mid_points = []
        for idx in range(len(top_points)):
            next_idx = (idx + 1) % len(top_points)
            p1 = top_points[idx]
            p2 = top_points[next_idx]
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            mid_z = (p1[2] + p2[2]) / 2.0
            mid_points.append((mid_x, mid_y, mid_z))
        
        bottom_points = [(-1.03, -1.85, 1.375), (-1.03, 1.90, 1.375), (1.075, 1.90, 1.375), (1.075, -1.85, 1.375)]
        
        points_node = self.render.attachNewNode("scene_points")
        
        def create_point(pos, color, name, point_size=5):
            format = GeomVertexFormat.getV3n3cp()
            vdata = GeomVertexData(name, format, Geom.UHStatic)
            
            vertex = GeomVertexWriter(vdata, 'vertex')
            normal = GeomVertexWriter(vdata, 'normal')
            color_writer = GeomVertexWriter(vdata, 'color')
            
            vertex.addData3f(0, 0, 0)
            normal.addData3f(0, 0, 1)
            color_writer.addData4f(color[0], color[1], color[2], color[3])
            
            points = GeomPoints(Geom.UHStatic)
            points.addVertex(0)
            points.closePrimitive()
            
            geom = Geom(vdata)
            geom.addPrimitive(points)
            
            node = GeomNode(name)
            node.addGeom(geom)
            
            np = points_node.attachNewNode(node)
            np.setPos(pos[0], pos[1], pos[2])
            
            np.setAttrib(RenderModeAttrib.make(RenderModeAttrib.M_point, point_size))
            
            return np
        
        for i, point in enumerate(top_points):
            create_point(point, (1, 0, 0, 1), f"top_point_{i}", 5) 
        
        for i, point in enumerate(mid_points):
            create_point(point, (0, 1, 0, 1), f"mid_point_{i}", 5) 
        
        for i, point in enumerate(bottom_points):
            create_point(point, (0, 0, 1, 1), f"bottom_point_{i}", 5) 

    def init_depth_renderer(self):
        self.taskMgr.do_method_later(0.5, self._delayed_depth_init, "delayed_depth_init")

    def _delayed_depth_init(self, task):
        self.depth_renderer = DepthMapRenderer(self)
        self.taskMgr.add(self.update_depth_overlay_task, "update_depth_overlay_task")
            
        return task.done

    def update_depth_overlay_task(self, task):
        if self.depth_renderer and hasattr(self.depth_renderer, 'overlay_node'):
            if not self.depth_renderer.overlay_node.isHidden():
                success = self.depth_renderer.update_depth_texture()
                if not success:
                    print("Failed to update depth texture")
        return task.cont

    def toggle_depth_overlay(self):
        if hasattr(self, 'depth_renderer') and self.depth_renderer:
            is_enabled = self.depth_renderer.toggle_overlay()
            return is_enabled
        else:
            return False
    
    def apply_barrel_distortion(self, img, k1, k2, crop_to_content=False):
        if k1 == 0.0 and k2 == 0.0:
            if crop_to_content:
                width = img.get_x_size()
                height = img.get_y_size()
                num_channels = img.get_num_channels()
                
                if num_channels == 1:
                    mode = 'L'
                    pil_img = Image.new(mode, (width, height))
                    pil_pixels = pil_img.load()
                    for y in range(height):
                        for x in range(width):
                            val = int(img.get_gray(x, y) * 255)
                            pil_pixels[x, y] = val
                elif num_channels >= 3:
                    mode = 'RGB' if num_channels == 3 else 'RGBA'
                    pil_img = Image.new(mode, (width, height))
                    pil_pixels = pil_img.load()
                    for y in range(height):
                        for x in range(width):
                            r = int(img.get_red(x, y) * 255)
                            g = int(img.get_green(x, y) * 255)
                            b = int(img.get_blue(x, y) * 255)
                            if num_channels == 3:
                                pil_pixels[x, y] = (r, g, b)
                            else: 
                                a = int(img.get_alpha(x, y) * 255) if img.has_alpha() else 255
                                pil_pixels[x, y] = (r, g, b, a)
                
                left, top, right, bottom = 270, 155, 1850, 925
                bbox = (left, top, right, bottom)
                cropped_pil_img = pil_img.crop(bbox)
                
                target_width = 1920 
                target_height = 1080 
                aspect_ratio = cropped_pil_img.width / cropped_pil_img.height
                target_ratio = target_width / target_height

                if aspect_ratio > target_ratio:
                    new_height = int(cropped_pil_img.width / target_ratio)
                    resized = cropped_pil_img.resize((target_width, new_height), Image.LANCZOS)
                    padding = (new_height - target_height) // 2
                    final_img = resized.crop((0, padding, target_width, padding + target_height))
                else:
                    new_width = int(cropped_pil_img.height * target_ratio)
                    resized = cropped_pil_img.resize((new_width, target_height), Image.LANCZOS)
                    padding = (new_width - target_width) // 2
                    final_img = resized.crop((padding, 0, padding + target_width, target_height))
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                    tmp_filename = tmpfile.name
                try:
                    resized_pil_img.save(tmp_filename, 'PNG')
                    temp_pnm = PNMImage()
                    if temp_pnm.read(Filename.from_os_specific(tmp_filename)):
                        os.unlink(tmp_filename)
                        return temp_pnm
                finally:
                    if os.path.exists(tmp_filename):
                        os.unlink(tmp_filename)       
            
            return img

        width = img.get_x_size()
        height = img.get_y_size()
        num_channels = img.get_num_channels()
        distorted_img = PNMImage(width, height, num_channels)
        cx = width / 2.0
        cy = height / 2.0
        max_r = math.sqrt((width/2)**2 + (height/2)**2)

        for y in range(height):
            for x in range(width):
                dx = x - cx
                dy = y - cy
                r = math.sqrt(dx*dx + dy*dy)
                if r > 0:
                    r_norm = r / max_r
                    r_distorted_norm = r_norm * (1 + k1 * r_norm**2 + k2 * r_norm**4)
                    r_distorted = r_distorted_norm * max_r
                    scale_factor = r_distorted / r
                else:
                    scale_factor = 1.0
                src_x = cx + dx * scale_factor
                src_y = cy + dy * scale_factor
                if 0 <= src_x < width-1 and 0 <= src_y < height-1:
                    x1 = int(src_x)
                    y1 = int(src_y)
                    x2 = x1 + 1
                    y2 = y1 + 1
                    dx_interp = src_x - x1
                    dy_interp = src_y - y1
                    c11 = [img.get_channel(x1, y1, c) for c in range(num_channels)]
                    c12 = [img.get_channel(x1, y2, c) for c in range(num_channels)]
                    c21 = [img.get_channel(x2, y1, c) for c in range(num_channels)]
                    c22 = [img.get_channel(x2, y2, c) for c in range(num_channels)]
                    c_top = [c11[c] * (1 - dx_interp) + c21[c] * dx_interp for c in range(num_channels)]
                    c_bottom = [c12[c] * (1 - dx_interp) + c22[c] * dx_interp for c in range(num_channels)]
                    c_final = [c_top[c] * (1 - dy_interp) + c_bottom[c] * dy_interp for c in range(num_channels)]
                    for c in range(num_channels):
                        distorted_img.set_channel(x, y, c, c_final[c])
                else:
                    src_x_clamped = max(0, min(width-1, src_x))
                    src_y_clamped = max(0, min(height-1, src_y))
                    x1 = int(src_x_clamped)
                    y1 = int(src_y_clamped)
                    x2 = min(width-1, x1 + 1)
                    y2 = min(height-1, y1 + 1)
                    dx_interp = src_x_clamped - x1
                    dy_interp = src_y_clamped - y1
                    c11 = [img.get_channel(x1, y1, c) for c in range(num_channels)]
                    c12 = [img.get_channel(x1, y2, c) for c in range(num_channels)]
                    c21 = [img.get_channel(x2, y1, c) for c in range(num_channels)]
                    c22 = [img.get_channel(x2, y2, c) for c in range(num_channels)]
                    c_top = [c11[c] * (1 - dx_interp) + c21[c] * dx_interp for c in range(num_channels)]
                    c_bottom = [c12[c] * (1 - dx_interp) + c22[c] * dx_interp for c in range(num_channels)]
                    c_final = [c_top[c] * (1 - dy_interp) + c_bottom[c] * dy_interp for c in range(num_channels)]
                    outside_dist = max(0, 
                                    abs(src_x - cx) - width/2 + 1, 
                                    abs(src_y - cy) - height/2 + 1)
                    fade_factor = max(0, min(1, 1 - (outside_dist / 15.0)**2))
                    for c in range(num_channels):
                        distorted_img.set_channel(x, y, c, c_final[c] * fade_factor)

        if crop_to_content:
            if num_channels == 1:
                mode = 'L'
                pil_img = Image.new(mode, (width, height))
                pil_pixels = pil_img.load()
                for y in range(height):
                    for x in range(width):
                        val = int(distorted_img.get_gray(x, y) * 255)
                        pil_pixels[x, y] = val
            elif num_channels >= 3:
                mode = 'RGB' if num_channels == 3 else 'RGBA'
                pil_img = Image.new(mode, (width, height))
                pil_pixels = pil_img.load()
                for y in range(height):
                    for x in range(width):
                        r = int(distorted_img.get_red(x, y) * 255)
                        g = int(distorted_img.get_green(x, y) * 255)
                        b = int(distorted_img.get_blue(x, y) * 255)
                        if num_channels == 3:
                            pil_pixels[x, y] = (r, g, b)
                        else: 
                            a = int(distorted_img.get_alpha(x, y) * 255) if distorted_img.has_alpha() else 255
                            pil_pixels[x, y] = (r, g, b, a)
            
            left, top, right, bottom = 270, 155, 1650, 925
            bbox = (left, top, right, bottom)
            cropped_pil_img = pil_img.crop(bbox)
            
            target_width = 1920 
            target_height = 1080
            resized_pil_img = cropped_pil_img.resize((target_width, target_height), Image.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                tmp_filename = tmpfile.name
            try:
                resized_pil_img.save(tmp_filename, 'PNG')
                temp_pnm = PNMImage()
                if temp_pnm.read(Filename.from_os_specific(tmp_filename)):
                    os.unlink(tmp_filename)
                    return temp_pnm
            finally:
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                        
        return distorted_img
    
    def create_video_from_frames(self, output_dir="renders/datasets_metric_/", video_name="camera_rotation.mp4", fps=20):
        search_pattern = os.path.join(output_dir, "render_*_frame_*.png")
        frame_files = glob.glob(search_pattern)
        
        frame_files.sort(key=lambda x: int(re.search(r'frame_(\d{3})', x).group(1)))
        
        if not frame_files:
            return False
        
        list_file = os.path.join(os.getcwd(), "frames_list.txt")
        with open(list_file, "w") as f:
            for frame in frame_files:
                abs_path = os.path.abspath(frame).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-r', str(fps),
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '22',
                '-pix_fmt', 'yuv420p',
                '-y',
                video_name
            ]
            
            return True
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def is_point_visible(self, point_3d):
        cam_pos = self.camera.getPos()
        point_pos = Point3(point_3d[0], point_3d[1], point_3d[2])
        
        direction = point_pos - cam_pos
        distance = direction.length()
        if distance < 0.001:
            return True
        direction.normalize()
        
        ray = CollisionRay()
        ray.setOrigin(cam_pos)
        ray.setDirection(direction)
        
        ray_node = CollisionNode('visibility_ray')
        
        ray_node.setFromCollideMask(BitMask32.bit(1))
        ray_node.setIntoCollideMask(BitMask32.allOff())
        ray_node.addSolid(ray)
        
        ray_np = self.render.attachNewNode(ray_node)
        
        queue = CollisionHandlerQueue()
        traverser = CollisionTraverser('visibility_traverser')
        traverser.addCollider(ray_np, queue)
        
        traverser.traverse(self.render)
        
        visible = True
        if queue.getNumEntries() > 0:
            queue.sortEntries()
            
            for i in range(queue.getNumEntries()):
                entry = queue.getEntry(i)
                hit_distance = (entry.getSurfacePoint(self.render) - cam_pos).length()
                
                if hit_distance < 0.1:
                    continue
                    
                if hit_distance < distance - 0.01:
                    visible = False
                    break
        
        ray_np.removeNode()
        return visible

    def _process_render_image(self, img, camera_fov_x=None, camera_fov_y=None, output_dir="renders", 
                              filename_prefix="render", metadata=None):
        k1 = 0.15
        k2 = 0.35
        
        left = 270
        top = 155
        right = 1650
        bottom = 925
        crop_width = right - left
        crop_height = bottom - top
        
        target_width = 1920
        target_height = 1080
        
        distorted_img = self.apply_barrel_distortion(
            img, 
            k1=k1, 
            k2=k2, 
            crop_to_content=False
        )
        
        cropped_img = PNMImage(crop_width, crop_height, 3)
        for y in range(crop_height):
            for x in range(crop_width):
                src_x = left + x
                src_y = top + y
                if (0 <= src_x < distorted_img.getXSize() and 
                    0 <= src_y < distorted_img.getYSize()):
                    r, g, b = distorted_img.getXel(src_x, src_y)
                    cropped_img.setXel(x, y, r, g, b)
        
        pil_img = Image.new("RGB", (crop_width, crop_height))
        pixels = pil_img.load()
        for y in range(crop_height):
            for x in range(crop_width):
                r, g, b = cropped_img.getXel(x, y)
                pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))
        
        resized_pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
        
        final_img = PNMImage(target_width, target_height, 3)
        for y in range(target_height):
            for x in range(target_width):
                r, g, b = resized_pil_img.getpixel((x, y))
                final_img.setXel(x, y, r/255.0, g/255.0, b/255.0)
        
        orig_width = img.getXSize()
        orig_height = img.getYSize()
        scale_x = target_width / crop_width
        scale_y = target_height / crop_height
        
        fx = fy = cx = cy = None
        if camera_fov_x is not None and camera_fov_y is not None:
            fx = (target_width / 2.0) / math.tan(math.radians(camera_fov_x / 2.0)) * scale_x
            fy = (target_height / 2.0) / math.tan(math.radians(camera_fov_y / 2.0)) * scale_y
            cx = target_width / 2.0
            cy = target_height / 2.0
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        final_img.write(Filename.from_os_specific(output_path))
        
        if metadata is None:
            metadata = {}
        
        render_metadata = {
            "image_path": output_path,
            "distortion_params": {
                "k1": k1,
                "k2": k2
            },
            "crop_params": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "crop_width": crop_width,
                "crop_height": crop_height
            },
            "scale_params": {
                "target_width": target_width,
                "target_height": target_height,
                "scale_x": scale_x,
                "scale_y": scale_y
            },
            **metadata
        }
        
        if camera_fov_x is not None and camera_fov_y is not None:
            render_metadata["camera_params"] = {
                "fov_x": camera_fov_x,
                "fov_y": camera_fov_y,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
        
        json_path = output_path.replace(".png", ".json")
        with open(json_path, 'w') as f:
            json.dump(render_metadata, f, indent=2)
        
        return output_path
    
    def save_single_render(self):
        lens = self.cam.node().getLens()
        if isinstance(lens, PerspectiveLens):
            fov = lens.getFov()
            camera_fov_x = fov[0]
            camera_fov_y = fov[1]
        else:
            camera_fov_x = camera_fov_y = None
        
        img = PNMImage()
        if not self.win.getScreenshot(img):
            return False
        
        output_path = self._process_render_image(
            img, 
            camera_fov_x=camera_fov_x,
            camera_fov_y=camera_fov_y,
            output_dir="renders/single",
            filename_prefix="single_render",
            metadata={
                "render_type": "single",
                "camera_position": {
                    "x": float(self.camera.getX()),
                    "y": float(self.camera.getY()),
                    "z": float(self.camera.getZ())
                },
                "camera_rotation": {
                    "h": float(self.camera.getH()),
                    "p": float(self.camera.getP()),
                    "r": float(self.camera.getR())
                },
                "model_set": self.current_model_set if hasattr(self, 'current_model_set') else None,
                "target_volume": self.Target_Volume
            }
        )
        
        return True
    
    def save_dataset_render(self):
        original_pos = self.camera.getPos()
        original_hpr = self.camera.getHpr()
        original_view = self.current_view
        original_target_volume = self.Target_Volume
        
        if not self.current_model_set:
            return False
        
        if not all([self.current_other_path, self.current_cuzov_path, self.current_napolnitel_path]):
            return False
        
        fixed_pos = (8.599995136260986, 6.0011109376791865e-05, 21.70002269744873)
        fixed_hpr = (89.99999237060547, -66.110355377197266, 0.0)
        fixed_fov_x = 48.0
        fixed_fov_y = 26.14091682434082
        
        volumes = [0.5 + i * 0.5 for i in range(99)] 
        passes_per_volume = 4
        
        total_renders = len(volumes) * passes_per_volume
        current_render = 0
        
        for i, volume in enumerate(volumes):
            for pass_num in range(passes_per_volume):
                current_render += 1
                
                self.Target_Volume = volume
                
                self.clear_scene()
                
                self.load_gltf_model(self.current_other_path)
                self.load_gltf_model(self.current_cuzov_path)
                self.load_gltf_model(self.current_napolnitel_path)
                
                self.create_ground_plane()
                self.ground_plane.setPos(0, 0, self.current_ground_plane_z)
                
                success_aabb = self.perform_AABB_plane()
                if not success_aabb:
                    continue
                
                self.Perlin_Seed = random.randint(0, 10000000) + pass_num * 1000000 + i * 100000000
                
                success_perlin = self.perlin_generator.generate_perlin_mesh_from_csg()
                if not success_perlin:
                    continue
                
                time.sleep(1.0)
                
                self.camera.setPos(*fixed_pos)
                self.camera.setHpr(*fixed_hpr)
                
                lens = self.cam.node().getLens()
                if isinstance(lens, PerspectiveLens):
                    lens.setFov(fixed_fov_x, fixed_fov_y)
                
                for _ in range(120):  
                    self.taskMgr.step()
                    time.sleep(0.01)
                
                current_pos = self.camera.getPos()
                current_hpr = self.camera.getHpr()
                
                img = PNMImage()
                if not self.win.getScreenshot(img):
                    continue
                
                output_path = self._process_render_image(
                    img,
                    camera_fov_x=fixed_fov_x,
                    camera_fov_y=fixed_fov_y,
                    output_dir="renders/datasets_metric_",
                    filename_prefix=f"render_volume_{volume:.1f}_pass_{pass_num:02d}",
                    metadata={
                        "render_type": "dataset",
                        "target_volume": volume,
                        "pass_number": pass_num,
                        "volume_index": i,
                        "perlin_seed": self.Perlin_Seed,
                        "model_set": self.current_model_set,
                        "camera_position": {
                            "x": float(current_pos.x),
                            "y": float(current_pos.y),
                            "z": float(current_pos.z)
                        },
                        "camera_rotation": {
                            "h": float(current_hpr.x),
                            "p": float(current_hpr.y),
                            "r": float(current_hpr.z)
                        }
                    }
                )
                
        self.Target_Volume = original_target_volume
        self.camera.setPos(original_pos)
        self.camera.setHpr(original_hpr)
        self.current_view = original_view
        
        return True
            

    def log_camera_parameters(self):
        lens = self.cam.node().get_lens()
        camera_zoom_data = {}
        if isinstance(lens, OrthographicLens):
            film_size = lens.get_film_size()
            camera_zoom_data = {
                "type": "orthographic",
                "film_size": {
                    "x": float(film_size.x),
                    "y": float(film_size.y)
                }
            }
        elif hasattr(lens, 'get_fov'):
            fov = lens.get_fov()
            camera_zoom_data = {
                "type": "perspective",
                "fov": {
                    "x": float(fov.x),
                    "y": float(fov.y)
                }
            }
        
        camera_data = {
            "position": {
                "x": float(self.camera.get_x()),
                "y": float(self.camera.get_y()),
                "z": float(self.camera.get_z())
            },
            "rotation": {
                "h": float(self.camera.get_h()),
                "p": float(self.camera.get_p()),
                "r": float(self.camera.get_r())
            },
            "view": self.current_view,
            "zoom": camera_zoom_data
        }
        
        camera_json = json.dumps(camera_data, indent=4)

    def calculate_mesh_volume(self, model):
        if not model:
            return 0.0
            
        node = model.node()
        if isinstance(node, GeomNode):
            geom_node = node
        else:
            geom_node_path = model.find("**/+GeomNode")
            if geom_node_path.isEmpty():
                return 0.0
            geom_node = geom_node_path.node()
        
        if geom_node.getNumGeoms() == 0:
            return 0.0
        
        geom = geom_node.getGeom(0)
        
        if geom.getNumPrimitives() == 0:
            return 0.0
        
        primitive = geom.getPrimitive(0)
        
        if not isinstance(primitive, GeomTriangles):
            return 0.0
        
        transform = model.getNetTransform().getMat()
        
        vertex_data = geom.getVertexData()
        vertex_reader = GeomVertexReader(vertex_data, "vertex")
        
        volume = 0.0
        num_tris = primitive.getNumPrimitives()
        
        for i in range(num_tris):
            vi0 = primitive.getVertex((i * 3) + 0)
            vi1 = primitive.getVertex((i * 3) + 1)
            vi2 = primitive.getVertex((i * 3) + 2)
            
            vertex_reader.setRow(vi0)
            v0 = vertex_reader.getData3f()
            
            vertex_reader.setRow(vi1)
            v1 = vertex_reader.getData3f()
            
            vertex_reader.setRow(vi2)
            v2 = vertex_reader.getData3f()
            
            v0 = transform.xformPoint(v0)
            v1 = transform.xformPoint(v1)
            v2 = transform.xformPoint(v2)
            
            volume += v0.dot(v1.cross(v2))
        
        volume = abs(volume) / 6.0
        return volume
    
    def perform_AABB_plane(self):
        target_model = None
        target_model_path = None
        for model in self.loaded_models:
            model_id = id(model)
            if model_id in self.model_paths:
                model_filename = self.model_paths[model_id].split('/')[-1]
                if model_filename == self.Target_Cuzov:
                    target_model = model
                    target_model_path = self.model_paths[model_id]
                    break

        if target_model is None:
            if self.loaded_models:
                for model in self.loaded_models:
                    model_id = id(model)
            return False

        min_point, max_point = target_model.getTightBounds()
        aabb_center = (min_point + max_point) / 2.0
        aabb_size = max_point - min_point
        
        ground_pos = self.ground_plane.getPos()
        
        plane_thickness = 0.05
        
        full_plane_mesh = trimesh.creation.box(
            extents=[self.plane_size_x, self.plane_size_y, plane_thickness]
        )
        
        aabb_mesh = trimesh.creation.box(
            extents=[aabb_size.x, aabb_size.y, aabb_size.z]
        )
        
        aabb_transform = trimesh.transformations.translation_matrix([
            aabb_center.x - ground_pos.x,
            aabb_center.y - ground_pos.y, 
            aabb_center.z - ground_pos.z
        ])
        aabb_mesh.apply_transform(aabb_transform)
        
        result_mesh = full_plane_mesh.intersection(aabb_mesh, engine='blender')
        
        if result_mesh.is_empty:
            return False
            
        csg_result_panda = self.trimesh_to_panda(result_mesh)

        material = Material()
        material.setDiffuse((0, 0.7, 0, 1))
        material.setAmbient((0, 0.3, 0, 1))
        material.setSpecular((0.5, 0.5, 0.5, 1))
        material.setShininess(50)
        csg_result_panda.setMaterial(material)
        csg_result_panda.setShaderAuto()

        old_pos = self.ground_plane.getPos()
        old_hpr = self.ground_plane.getHpr()
        old_scale = self.ground_plane.getScale()

        self.ground_plane.removeNode()

        csg_result_panda.reparentTo(self.render)
        csg_result_panda.setPos(old_pos)
        csg_result_panda.setHpr(old_hpr)
        csg_result_panda.setScale(old_scale)

        self.ground_plane = csg_result_panda

        if not hasattr(self, 'csg_results'):
            self.csg_results = []
        
        self.csg_results.append({
            "target_model_path": target_model_path,
            "result_node": csg_result_panda,
            "original_model": target_model
        })

        self.ground_plane.hide()

        return True

    def panda_to_trimesh(self, node_path):
        geom_node = node_path.node()
        if not isinstance(geom_node, GeomNode):
            geom_node_path = node_path.find("**/+GeomNode")
            geom_node = geom_node_path.node()
        
        transform = node_path.getNetTransform().getMat()
        
        vertices = []
        faces = []
        
        for i in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(i)
            vdata = geom.getVertexData()
            
            vertex_reader = GeomVertexReader(vdata, "vertex")
            while not vertex_reader.isAtEnd():
                pos = vertex_reader.getData3f()
                pos = transform.xformPoint(pos)
                vertices.append([pos.x, pos.y, pos.z])
            
            for j in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(j)
                if isinstance(prim, GeomTriangles):
                    for k in range(prim.getNumPrimitives()):
                        start = prim.getPrimitiveStart(k)
                        end = prim.getPrimitiveEnd(k)
                        face = []
                        for idx in range(start, end):
                            vi = prim.getVertex(idx)
                            face.append(vi)
                        if len(face) == 3:
                            faces.append(face)
        
        if not vertices or not faces:
            return None
            
        return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    def trimesh_to_panda(self, trimesh_mesh):
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        if not hasattr(trimesh_mesh, 'vertex_normals') or len(trimesh_mesh.vertex_normals) != len(vertices):
            trimesh_mesh.compute_vertex_normals()
        
        normals = trimesh_mesh.vertex_normals
        
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("trimesh_result", format, Geom.UHStatic)
        
        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")
        
        for i, vertex in enumerate(vertices):
            vertex_writer.addData3f(vertex[0], vertex[1], vertex[2])
            
            if i < len(normals):
                normal = normals[i]
                if np.any(np.isnan(normal)) or np.linalg.norm(normal) < 0.1:
                    normal = [0, 0, 1] 
                normal_writer.addData3f(normal[0], normal[1], normal[2])
            else:
                normal_writer.addData3f(0, 0, 1) 
            
            texcoord_writer.addData2f(0, 0) 
        
        prim = GeomTriangles(Geom.UHStatic)
        for face in faces:
            prim.addVertices(face[0], face[1], face[2])
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode("trimesh_result")
        node.addGeom(geom)
        
        result_np = self.render.attachNewNode(node)
        
        return result_np

    def fix_uv_coordinates_for_final_model(self, node, texture_repeatX=1.0, texture_repeatY=1.0):
        """Назначает правильные UV-координаты для всех поверхностей модели"""
        # Получаем размеры модели в мировых координатах
        min_bound, max_bound = node.getTightBounds()
        
        if min_bound is None or max_bound is None:
            print("Не удалось получить границы модели для UV-развертки")
            return
        
        size_x = max_bound.x - min_bound.x
        size_y = max_bound.y - min_bound.y
        size_z = max_bound.z - min_bound.z
        
        # Получаем геометрию модели
        geom_node = node.node()
        
        for geom_index in range(geom_node.getNumGeoms()):
            geom = geom_node.modifyGeom(geom_index)
            vdata = geom.modifyVertexData()
            
            # Получаем доступ к данным вершин
            vertex_reader = GeomVertexReader(vdata, 'vertex')
            normal_reader = GeomVertexReader(vdata, 'normal')
            texcoord_writer = GeomVertexWriter(vdata, 'texcoord')
            
            num_vertices = vdata.getNumRows()
            
            # Сбрасываем позиции ридеров
            vertex_reader.setRow(0)
            normal_reader.setRow(0)
            texcoord_writer.setRow(0)
            
            for i in range(num_vertices):
                if vertex_reader.isAtEnd() or normal_reader.isAtEnd():
                    break
                
                # Читаем позицию и нормаль вершины
                vertex = vertex_reader.getData3f()
                normal = normal_reader.getData3f()
                
                # Нормализуем нормаль (на всякий случай)
                normal_length = normal.length()
                if normal_length > 0:
                    normal = normal / normal_length
                
                # Используем проекционную UV-развертку в зависимости от направления нормали
                u = 0.0
                v = 0.0
                
                # Определяем доминирующую ось нормали
                abs_normal = Vec3(abs(normal.x), abs(normal.y), abs(normal.z))
                
                if abs_normal.x >= abs_normal.y and abs_normal.x >= abs_normal.z:
                    # Нормаль преимущественно по X - используем YZ плоскость
                    # Для консистентности, используем тот же repeat что и для XY
                    u = (vertex.y - min_bound.y) / size_y * texture_repeatY
                    v = (vertex.z - min_bound.z) / size_z * texture_repeatX
                elif abs_normal.y >= abs_normal.x and abs_normal.y >= abs_normal.z:
                    # Нормаль преимущественно по Y - используем XZ плоскость
                    u = (vertex.x - min_bound.x) / size_x * texture_repeatX
                    v = (vertex.z - min_bound.z) / size_z * texture_repeatY
                else:
                    # Нормаль преимущественно по Z - используем XY плоскость (верх/низ)
                    u = (vertex.x - min_bound.x) / size_x * texture_repeatX
                    v = (vertex.y - min_bound.y) / size_y * texture_repeatY
                
                texcoord_writer.setData2f(u, v)

    def create_ground_plane(self):
        if self.ground_plane:
            self.ground_plane.removeNode()
        
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("ground_plane", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        size_x = self.plane_size_x
        size_y = self.plane_size_y
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        
        vertices = [
            (-half_size_x, -half_size_y, 0),  
            (half_size_x, -half_size_y, 0),   
            (half_size_x, half_size_y, 0),    
            (-half_size_x, half_size_y, 0)    
        ]
        
        plane_normal = (0, 0, 1)
        
        for v in vertices:
            vertex.addData3f(v[0], v[1], v[2])
            normal.addData3f(plane_normal[0], plane_normal[1], plane_normal[2])
            u = (v[0] + half_size_x) / size_x
            v_coord = (v[1] + half_size_y) / size_y
            texcoord.addData2f(u, v_coord)
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode("ground_plane")
        node.addGeom(geom)
        
        self.ground_plane = self.render.attachNewNode(node)
        
        material = Material()
        material.setDiffuse((0, 0.7, 0, 1))
        material.setAmbient((0, 0.3, 0, 1))
        material.setSpecular((0.5, 0.5, 0.5, 1))
        material.setShininess(50)
        self.ground_plane.setMaterial(material)
        
        self.ground_plane.setPos(0, 0, 0)
        
        self.ground_plane.setShaderAuto()
        
        self.ground_plane.setTwoSided(True)

    def set_plane_size_x(self, size_x):
        self.plane_size_x = size_x

    def set_plane_size_y(self, size_y):
        self.plane_size_y = size_y

    def set_plane_position(self, x, y, z):
        self.ground_plane.setPos(x, y, z)

    def zoom_camera(self, factor):
        cam_pos = self.camera.get_pos()
        cam_hpr = self.camera.get_hpr()

        if self.current_view == "perspective":
            distance = cam_pos.length()
            if distance > 0.1:
                new_distance = distance * (1.0 / factor)
                scale_factor = new_distance / distance
                self.camera.set_pos(cam_pos * scale_factor)

        elif self.current_view in ["front", "back", "left", "right", "top", "bottom"]:
            lens = self.cam.node().get_lens()
            if hasattr(lens, 'get_fov') and hasattr(lens, 'set_fov'):
                if isinstance(lens, OrthographicLens):
                     old_film_size = lens.get_film_size()
                     new_film_size = old_film_size * (1.0 / factor)
                     lens.set_film_size(new_film_size)
                else:
                     old_fov = lens.get_fov()
                     new_fov_x = max(1.0, min(179.0, old_fov.x * (1.0 / factor)))
                     new_fov_y = max(1.0, min(179.0, old_fov.y * (1.0 / factor)))
                     lens.set_fov(LVecBase2f(new_fov_x, new_fov_y))

    def fix_shadow_camera_aspect(self):
        if hasattr(self.render_pipeline, 'light_manager'):
            light_mgr = self.render_pipeline.light_manager

    def create_perlin_noise_mesh(self):
        size_x = 2000.0
        size_y = 2000.0
        size_z = 2.0
        position = LPoint3f(0.0, 0.0, -1.0)
        
        texture_repeat_u = self.current_texture_set.get('textureRepeatU', 160.0)
        texture_repeat_v = self.current_texture_set.get('textureRepeatV', 160.0)
        
        format = GeomVertexFormat.getV3n3t2() 
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("simplified_perlin_data", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        grid_size = 64
        step_x = size_x / (grid_size - 1) if grid_size > 1 else 0
        step_y = size_y / (grid_size - 1) if grid_size > 1 else 0
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        pos_z = position.getZ() + (size_z / 2.0)
        
        for y in range(grid_size):
            for x in range(grid_size):
                world_x = x * step_x - half_size_x
                world_y = y * step_y - half_size_y
                world_z = pos_z
                vertex.addData3f(world_x, world_y, world_z)
                normal.addData3f(0, 0, 1)
                normalized_u = x / (grid_size - 1) if grid_size > 1 else 0.0
                normalized_v = y / (grid_size - 1) if grid_size > 1 else 0.0
                u = normalized_u * texture_repeat_u  
                v = normalized_v * texture_repeat_v  
                texcoord.addData2f(u, v)
        
        prim = GeomTriangles(Geom.UHStatic)
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i1 = y * grid_size + x
                i2 = y * grid_size + (x + 1)
                i3 = (y + 1) * grid_size + x
                i4 = (y + 1) * grid_size + (x + 1)
                prim.addVertices(i1, i2, i3)
                prim.addVertices(i2, i4, i3)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode("simplified_perlin_noise_mesh")
        node.addGeom(geom)
        
        self.perlin_model = self.render.attachNewNode(node)
        self.perlin_model.setPos(0, 0, 0)
        
        if not hasattr(self, 'loaded_models'):
            self.loaded_models = []
        self.loaded_models.append(self.perlin_model)
        
        if not hasattr(self, 'model_paths'):
            self.model_paths = {}
        self.model_paths[id(self.perlin_model)] = "perlin_noise_mesh"
        
        diffuse_texture_path = "textures/groundPerlin_8k/aerial_beach_03_diff_8k.jpg"
        texture = self.loader.loadTexture(diffuse_texture_path)
        if texture:
            texture.setMinfilter(Texture.FTLinearMipmapLinear)
            texture.setMagfilter(Texture.FTLinear)
            texture.setWrapU(Texture.WMRepeat) 
            texture.setWrapV(Texture.WMRepeat) 
            self.perlin_model.setTexture(texture, 1)
        
        material = Material("simplified_perlin_material")
        material.setDiffuse((1.0, 1.0, 1.0, 1.0))
        material.setAmbient((0.3, 0.3, 0.3, 1.0))
        material.setSpecular((0.5, 0.5, 0.5, 1.0))
        material.setShininess(10.0)
        self.perlin_model.setMaterial(material, 1)
        
        self.perlin_model.setShaderAuto()
        self.perlin_model.setTwoSided(True)
        self.perlin_model.setBin("fixed", 0)
        self.perlin_model.setDepthOffset(1)

    def _set_initial_time(self, task):
        """Установка начального времени суток"""
        if hasattr(self, 'render_pipeline') and hasattr(self.render_pipeline, 'daytime_mgr'):
            self.render_pipeline.daytime_mgr.time = "06:40"
        return task.done

    def setup_scene(self):
        self.quarry_model = None

        self.create_perlin_noise_mesh()

        self.taskMgr.do_method_later(0.5, self._set_initial_time, "set_initial_time")
        
        # # Очищаем существующие источники света
        # self._night_lights = []
        # 
        ## Создаем точечный свет (Point Light)
        #main_light = PointLight()
        #
        ## Устанавливаем позицию из Transform
        #main_light.pos = (4.0762, 1.0055, 5.9039)
        #
        ## Устанавливаем цвет из температуры (6500K)
        #main_light.set_color_from_temperature(6500)
        #
        ## Устанавливаем мощность (Power/Exposure 1000.000)
        #main_light.energy = 100.0
        #
        ## Устанавливаем радиус влияния (Custom Distance 40m)
        #main_light.radius = 20.0
        #
        ## Настройки теней
        #main_light.casts_shadows = True
        #main_light.shadow_map_resolution = 1024
        #
        ## В RenderPipeline влияние на диффузные/глянцевые материалы обычно 
        ## настраивается через материалы, а не через свет
        #
        ## Добавляем свет в сцену
        #self.render_pipeline.add_light(main_light)
        #self._night_lights.append(main_light)
        #
        #print(f"Main light added at position {main_light.pos}")
        #print(f"Light parameters: temperature=6500K, energy=1000.0, radius=40.0m")

        self.camera.set_pos(0, -20, 5)
        self.camera.look_at(0, 0, 0)
        self.disable_mouse()

    def animate_street_lights(self, task):
        frame_time = self.taskMgr.globalClock.get_frame_time()
        
        if hasattr(self, '_night_lights') and len(self._night_lights) >= 3:
            for i, light in enumerate(self._night_lights):
                if hasattr(light, 'energy'):
                    flicker = 1.0 + (random.random() - 0.5) * 0.1
                    light.energy = 100 * flicker
        
        return task.cont

    def set_drag_sensitivity(self, sensitivity):
        self.drag_sensitivity = sensitivity

    def set_quarry_scale(self, scale):
        self.quarry_model.set_scale(scale)

    def set_quarry_position(self, x, y, z):
        self.quarry_model.set_pos(x, y, z)

    def set_cube_size_x(self, size_x):
        self.cube_size_x = size_x
        self.cube_model.setScale(size_x, self.cube_size_y, self.cube_size_z)

    def set_cube_size_y(self, size_y):
        self.cube_size_y = size_y
        self.cube_model.setScale(self.cube_size_x, size_y, self.cube_size_z)

    def set_cube_size_z(self, size_z):
        self.cube_size_z = size_z
        self.cube_model.setScale(self.cube_size_x, self.cube_size_y, size_z)

    def set_cube_position(self, x, y, z):
        self.cube_model.setPos(x, y, z)

    def toggle_drag_drop_mode(self, enabled):
        self.drag_drop_mode = enabled
        if not enabled:
            self.selected_model = None
            self.drag_start_pos = None
            self.drag_start_mouse_pos = None

    def handle_mouse_left(self):
        if self.drag_drop_mode:
            if self.selected_model:
                self.start_drag_drop()
            else:
                self.select_model_under_mouse()
                if self.selected_model and self.selected_model != self.quarry_model:
                    self.start_drag_drop()
        else:
            if self.current_view == "perspective":
                self.mouse_rotation_enabled = True
                if self.mouseWatcherNode.hasMouse():
                    self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                    self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def handle_mouse_left_up(self):
        if self.drag_drop_mode:
            self.stop_drag_drop()
        else:
            self.mouse_rotation_enabled = False

    def handle_mouse_right(self):
        if self.drag_drop_mode:
            self.select_model_under_mouse()

    def start_drag_drop(self):
        if self.selected_model and self.selected_model != self.quarry_model:
            if self.mouseWatcherNode.hasMouse():
                self.drag_start_pos = self.selected_model.get_pos()
                self.drag_start_mouse_pos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())

    def stop_drag_drop(self):
        self.drag_start_pos = None
        self.drag_start_mouse_pos = None

    def select_model_under_mouse(self):
        if self.mouseWatcherNode.hasMouse():
            if self.ground_plane and not self.ground_plane.isHidden():
                self.selected_model = self.ground_plane
                return
                
            for model in reversed(self.loaded_models):
                if model != self.quarry_model and not model.isHidden():
                    self.selected_model = model
                    return
            
            self.selected_model = None

    def drag_drop_task(self, task):
        if self.drag_drop_mode and self.selected_model and self.drag_start_pos and self.drag_start_mouse_pos:
            if self.mouseWatcherNode.hasMouse():
                current_mouse_x, current_mouse_y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
                
                dx_mouse = current_mouse_x - self.drag_start_mouse_pos[0]
                dy_mouse = current_mouse_y - self.drag_start_mouse_pos[1]
                
                drag_speed = self.drag_sensitivity
                
                new_x, new_y, new_z = self.drag_start_pos.getX(), self.drag_start_pos.getY(), self.drag_start_pos.getZ()
                
                if self.current_view == "top":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                elif self.current_view == "bottom":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                elif self.current_view == "front":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dy_mouse * drag_speed
                elif self.current_view == "back":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dy_mouse * drag_speed
                elif self.current_view == "left":
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dx_mouse * drag_speed
                elif self.current_view == "right":
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dx_mouse * drag_speed
                else: 
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ()
                
                self.selected_model.set_pos(new_x, new_y, new_z)
                
        return task.cont

    def start_mouse_rotation(self):
        if self.drag_drop_mode:
            return
            
        if self.current_view == "perspective":
            self.mouse_rotation_enabled = True
            if self.mouseWatcherNode.hasMouse():
                self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def stop_mouse_rotation(self):
        self.mouse_rotation_enabled = False

    def mouse_rotation_task(self, task):
        if self.mouse_rotation_enabled and self.current_view == "perspective":
            if self.mouseWatcherNode.hasMouse():
                mouse_x = self.mouseWatcherNode.getMouseX()
                mouse_y = self.mouseWatcherNode.getMouseY()
                
                dx = mouse_x - self.last_mouse_x
                dy = mouse_y - self.last_mouse_y
                
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    h = self.camera.get_h() - dx * self.camera_rotation_speed * 100
                    p = self.camera.get_p() + dy * self.camera_rotation_speed * 100
                    
                    p = max(-89, min(89, p))
                    
                    self.camera.set_hpr(h, p, 0)
                
                self.last_mouse_x = mouse_x
                self.last_mouse_y = mouse_y
                
        return task.cont

    def set_top_view(self):
        self.current_view = "top"
        self.camera.set_pos(0, 0, 20)
        self.camera.set_hpr(0, -90, 0)

    def set_bottom_view(self):
        self.current_view = "bottom"
        self.camera.set_pos(0, 0, -20)
        self.camera.set_hpr(0, 90, 0)

    def set_front_view(self):
        self.current_view = "front"
        self.camera.set_pos(0, -20, 0)
        self.camera.look_at(0, 0, 0)

    def set_back_view(self):
        self.current_view = "back"
        self.camera.set_pos(0, 20, 0)
        self.camera.set_hpr(180, 0, 0)

    def set_left_view(self):
        self.current_view = "left"
        self.camera.set_pos(20, 0, 0)
        self.camera.set_hpr(90, 0, 0)

    def set_right_view(self):
        self.current_view = "right"
        self.camera.set_pos(-20, 0, 0)
        self.camera.set_hpr(-90, 0, 0)

    def set_perspective_view(self):
        self.current_view = "perspective"
        self.camera.set_pos(0, -20, 5)
        self.camera.look_at(0, 0, 0)

    def set_key(self, key, value):
        self.key_map[key] = value

    def move_camera_task(self, task):
        speed = 0.1
        
        if self.current_view == "top" or self.current_view == "bottom":
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
                
        elif self.current_view == "front" or self.current_view == "back":
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
                
        elif self.current_view == "left" or self.current_view == "right":
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
                
        else: 
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
        
        return task.cont

    def load_gltf_model(self, file_path):
        model_filename = Filename.from_os_specific(file_path)
        
        model_np = self.loader.load_model(model_filename, noCache=True) 
        
        model_np.reparent_to(self.render)
        self.render_pipeline.prepare_scene(model_np)
        model_np.set_pos(0, 0, 0)
        model_np.set_hpr(0, 0, 0) 
        model_np.set_scale(1)
        
        self.loaded_models.append(model_np)
        self.model_paths[id(model_np)] = file_path
        
        return model_np 

    def load_model_set(self, config, model_set_name):
        self.clear_scene()
        
        if not hasattr(self, 'perlin_model') or self.perlin_model is None:
            self.create_perlin_noise_mesh()
        
        models_loaded = []
        
        # Используем PROJECT_ROOT для построения абсолютных путей
        def get_absolute_path(relative_path):
            if os.path.isabs(relative_path):
                return relative_path
            return os.path.join(PROJECT_ROOT, relative_path)
        
        if 'other' in config and config['other']:
            other_path = get_absolute_path(config['other'])
            if os.path.exists(other_path):
                other_model = self.load_gltf_model(other_path)
                if other_model:
                    models_loaded.append('other')
                    self.current_other_path = other_path
        
        if 'cuzov' in config and config['cuzov']:
            cuzov_path = get_absolute_path(config['cuzov'])
            if os.path.exists(cuzov_path):
                cuzov_model = self.load_gltf_model(cuzov_path)
                if cuzov_model:
                    models_loaded.append('cuzov')
                    self.Target_Cuzov = os.path.basename(cuzov_path)
                    self.current_cuzov_path = cuzov_path
        
        if 'napolnitel' in config and config['napolnitel']:
            napolnitel_path = get_absolute_path(config['napolnitel'])
            if os.path.exists(napolnitel_path):
                napolnitel_model = self.load_gltf_model(napolnitel_path)
                if napolnitel_model:
                    napolnitel_model.hide()
                    models_loaded.append('napolnitel')
                    self.Target_Napolnitel = os.path.basename(napolnitel_path)
                    self.current_napolnitel_path = napolnitel_path
        
        if 'max_volume' in config:
            self.Target_Volume = config['max_volume']
        
        if 'ground_plane' in config:
            self.current_ground_plane_z = config['ground_plane']
        
        self.current_model_set = model_set_name
        
        if hasattr(self, 'perlin_model') and self.perlin_model:
            if self.perlin_model.isHidden():
                self.perlin_model.show()
        
        return True

    def clear_scene(self):
        if hasattr(self, 'test_perlin_mesh') and self.test_perlin_mesh:
            if self.test_perlin_mesh in self.loaded_models:
                self.loaded_models.remove(self.test_perlin_mesh)
            self.test_perlin_mesh.removeNode()
            self.test_perlin_mesh = None
        
        self.last_target_model_trimesh = None
        self.last_best_z = None

        if hasattr(self, 'dynamic_perlin_model') and self.dynamic_perlin_model:
            if self.dynamic_perlin_model in self.loaded_models:
                self.loaded_models.remove(self.dynamic_perlin_model)
            self.dynamic_perlin_model.removeNode()
            self.dynamic_perlin_model = None
        
        if hasattr(self, 'final_model') and self.final_model:
            if self.final_model in self.loaded_models:
                self.loaded_models.remove(self.final_model)
            self.final_model.removeNode()
            self.final_model = None
            
        if hasattr(self, 'csg_results'):
            for csg_info in self.csg_results:
                if "result_node" in csg_info and csg_info["result_node"]:
                    if csg_info["result_node"] in self.loaded_models:
                        self.loaded_models.remove(csg_info["result_node"])
                    csg_info["result_node"].removeNode()
            self.csg_results = []

        models_to_keep = []
        
        if hasattr(self, 'perlin_model') and self.perlin_model:
            models_to_keep.append(self.perlin_model)
        
        if hasattr(self, 'ground_plane') and self.ground_plane:
            models_to_keep.append(self.ground_plane)
        
        if hasattr(self, 'base_perlin_model') and self.base_perlin_model:
            models_to_keep.append(self.base_perlin_model)
        
        models_to_remove = []
        for model in self.loaded_models:
            if model not in models_to_keep:
                models_to_remove.append(model)
        
        for model in models_to_remove:
            model.removeNode()
            model_id = id(model)
            if model_id in self.model_paths:
                del self.model_paths[model_id]
        
        self.loaded_models = models_to_keep
        
def main():
    app = QApplication(sys.argv)
    
    main_window = QMainWindow()
    main_window.setWindowTitle('3D simulator')
    
    main_window.resize(2300, 1080)
    main_window.setMinimumSize(2300, 1080)
    
    central_widget = QWidget()
    main_layout = QHBoxLayout(central_widget)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)
    
    panda_container = QWidget()
    panda_container.setMinimumSize(1920, 1080)  
    panda_container.setMaximumSize(1920, 1080)  
    panda_container.setStyleSheet("background-color: #000000;")
    
    loadPrcFileData("", "win-size 1920 1080")
    loadPrcFileData("", "undecorated true")
    loadPrcFileData("", "fullscreen false")
    loadPrcFileData("", "window-type offscreen")
    
    panda_app = MyApp()
    
    control_panel = CameraControlGUI(panda_app)
    control_panel.setMinimumWidth(380)
    control_panel.setMaximumWidth(380)
    control_panel.setMinimumHeight(1080)
    
    main_layout.addWidget(panda_container, 1)
    main_layout.addWidget(control_panel, 0)
    
    main_window.setCentralWidget(central_widget)
    
    def update_panda():
        panda_app.taskMgr.step()
    
    timer = QTimer()
    timer.timeout.connect(update_panda)
    timer.start(16)
    
    def initialize_integration():
        if hasattr(panda_app, 'win') and panda_app.win:
            
            container_hwnd = panda_container.winId().__int__()
            
            def enum_windows_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    class_name = win32gui.GetClassName(hwnd)
                    window_text = win32gui.GetWindowText(hwnd)
                    if 'Panda' in window_text or 'panda' in class_name.lower():
                        results.append(hwnd)
                return True
            
            results = []
            win32gui.EnumWindows(enum_windows_callback, results)
            
            if results:
                hwnd = results[0]
                win32gui.SetParent(hwnd, container_hwnd)
                
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                style = style & ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | 
                                win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | 
                                win32con.WS_SYSMENU | win32con.WS_BORDER | 
                                win32con.WS_DLGFRAME)
                style = style | win32con.WS_CHILD
                win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
                
                win32gui.MoveWindow(hwnd, 0, 0, 1920, 1080, True)
                
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                win32gui.UpdateWindow(hwnd)
                
                panda_app.panda_hwnd = hwnd
                        
                    
    
    QTimer.singleShot(500, initialize_integration)
    
    position_timer = QTimer()
    position_timer.timeout.connect(lambda: 
        win32gui.MoveWindow(panda_app.panda_hwnd, 0, 0, 1920, 1080, True) 
        if hasattr(panda_app, 'panda_hwnd') and panda_app.panda_hwnd else None)
    position_timer.start(100)
    
    main_window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()