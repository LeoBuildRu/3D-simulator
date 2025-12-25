from gui import CameraControlGUI
from panda_widget import Panda3DWidget
from depth_map_renderer import DepthMapRenderer
from perlin_mesh_generator import PerlinMeshGenerator
from renderer_utils import RendererUtils
from mesh_reconstruction import MeshReconstruction

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
        self.renderer_utils = RendererUtils(self)

        self.mesh_reconstruction = MeshReconstruction(self)

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
        top_points = [(-1.03, -2.22, 2.4), (-1.03, 2.4, 2.4), (1.045, 2.4, 2.4), (1.045, -2.22, 2.4)]
        
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
    
    def log_camera_parameters(self):
        lens = self.cam.node().get_lens()
        camera_zoom_data = {}
        focal_length_pixels = None
        perspective_angle_x = None
        perspective_angle_y = None
        
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
            
            # Расчет фокусного расстояния в пикселях
            if self.win:
                # Получаем размер окна в пикселях
                width = self.win.getXSize()
                height = self.win.getYSize()
                
                # Конвертируем FOV из градусов в радианы
                fov_x_rad = math.radians(fov.x)
                fov_y_rad = math.radians(fov.y)
                
                # Вычисляем фокусное расстояние по горизонтали и вертикали
                focal_length_x = (width / 2.0) / math.tan(fov_x_rad / 2.0)
                focal_length_y = (height / 2.0) / math.tan(fov_y_rad / 2.0)
                
                # Берем среднее значение для общего фокусного расстояния
                focal_length_pixels = (focal_length_x + focal_length_y) / 2.0
            else:
                # Если окно недоступно, используем стандартные размеры
                width, height = 1920, 1080
                
                fov_x_rad = math.radians(fov.x)
                fov_y_rad = math.radians(fov.y)
                
                focal_length_x = (width / 2.0) / math.tan(fov_x_rad / 2.0)
                focal_length_y = (height / 2.0) / math.tan(fov_y_rad / 2.0)
                
                focal_length_pixels = (focal_length_x + focal_length_y) / 2.0
            
            # Углы наклона камеры по X и Y (в градусах)
            # В Panda3D:
            # - pitch (P) - наклон вверх/вниз (вращение вокруг оси X)
            # - heading (H) - поворот влево/вправо (вращение вокруг оси Z)
            
            # Получаем текущую ориентацию камеры
            camera_hpr = self.camera.getHpr()
            perspective_angle_x = float(camera_hpr.x)  # heading (вращение вокруг Z)
            perspective_angle_y = float(camera_hpr.y)  # pitch (вращение вокруг X)
        
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
            "zoom": camera_zoom_data,
            "perspective_focal_length": float(focal_length_pixels) if focal_length_pixels is not None else None,
            "perspective_angle_x": perspective_angle_x,
            "perspective_angle_y": perspective_angle_y
        }
        
        camera_json = json.dumps(camera_data, indent=4)
        
        # Вместо вывода в консоль, отображаем диалог с информацией
        # Для этого нужно иметь доступ к GUI
        if hasattr(self, 'control_panel') and self.control_panel:
            self.control_panel.show_camera_info_dialog(camera_json)
        else:
            # Fallback: выводим в консоль
            print(camera_json)
        
        return camera_json

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
        
        result_mesh = full_plane_mesh.intersection(aabb_mesh, engine='auto')
        
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
        self.add_scene_points()
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

    def get_camera_orientation(self):
        """Получить ориентацию камеры в удобном формате"""
        pos = self.camera.getPos()
        quat = self.camera.getQuat()
        hpr = quat.getHpr()
        
        return {
            'position': (pos.x, pos.y, pos.z),
            'quaternion': (quat.x, quat.y, quat.z, quat.w),
            'hpr': (hpr.x, hpr.y, hpr.z),
            'forward_vector': self.camera.getQuat().getForward()
        }

    def set_camera_look_at(self, target_point):
        """Направить камеру на целевую точку"""
        self.camera.lookAt(target_point)

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