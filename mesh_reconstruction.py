# mesh_reconstruction.py
import tkinter as tk
from tkinter import filedialog
from PyQt5 import *
from PyQt5.QtWidgets import *
from panda3d.core import *
import os
import json
import numpy as np
import math
from PIL import Image
import noise
import random
from scipy.spatial import KDTree
import trimesh

class MeshReconstruction:
    def __init__(self, panda_app, image_path = None):
        self.panda_app = panda_app
        self.recon_json_path = ""

        # for testing only
        self.image_path = image_path or "height_example/Example-1-3-final.png"
        self.alpha_threshold = 0.5

        # Калиброванные параметры глубины
        self.min_depth = 14.6
        self.max_depth = 19.5
        
        # Параметры экстраполяции
        self.extrapolation_enabled = False
        self.target_width = 15.0  # Ширина целевой области в метрах
        self.target_height = 10.0  # Высота целевой области в метрах
        self.grid_resolution = 80  # Разрешение сетки экстраполяции
        
        # Параметры шума Перлина
        self.noise_scale = 0.5
        self.noise_strength = 0.15
        self.noise_octaves = 4
        self.noise_persistence = 0.5
        self.noise_lacunarity = 2.0
        self.noise_seed = random.randint(0, 10000)
        
        # Для хранения мешей
        self.source_mesh_node = None
        self.mesh_node = None
        
        print(f"[DEBUG] Инициализирован MeshReconstruction")

    def calibrate_depth_from_json(self, json_data):
        self.min_depth = 14.6
        self.max_depth = 18.0
        
        print(f"[DEBUG] Откалиброванные глубины: min={self.min_depth}, max={self.max_depth}")

    def browse_recon_json(self):
        file_path = filedialog.askopenfilename(
            title="Select .json config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        self.recon_json_path = file_path

        if file_path:
            return file_path
        return None
        
    def normalize(self, v):
        v = np.array(v, dtype=float)
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def build_local_to_world_matrix_np(self, A, B, C):
        A, B, C = map(np.array, (A, B, C))
        x = self.normalize(B - A)
        tempY = C - A
        proj = np.dot(tempY, x) * x
        y = self.normalize(tempY - proj)
        z = self.normalize(np.cross(x, y))
        m = np.eye(4)
        m[:3, 0] = x
        m[:3, 1] = y
        m[:3, 2] = z
        m[:3, 3] = A
        return m

    def compute_transform_np(self, p1A, p1B, p1C, p2A, p2B, p2C):
        world_from_local1 = self.build_local_to_world_matrix_np(p1A, p1B, p1C)
        world_from_local2 = self.build_local_to_world_matrix_np(p2A, p2B, p2C)
        return world_from_local1 @ np.linalg.inv(world_from_local2)

    def panda_vec3_to_np(self, v):
        return np.array([v.x, v.y, v.z], dtype=float)

    def np_to_panda_point(self, v):
        return LPoint3f(float(v[0]), float(v[1]), float(v[2]))

    def np_to_panda_vec(self, v):
        return LVector3f(float(v[0]), float(v[1]), float(v[2]))

    def viewport_to_world_point_geometric(
        self,
        camera,
        u, v,
        distance,
        fov_y_deg,
        aspect_ratio
    ):
        half_fov_y = math.radians(fov_y_deg) * 0.5
        half_fov_x = math.atan(aspect_ratio * math.tan(half_fov_y))

        x_angle = (u - 0.5) * 2.0 * half_fov_x
        y_angle = (0.5 - v) * 2.0 * half_fov_y

        dir_x = math.tan(x_angle)
        dir_y = math.tan(y_angle)
        dir_z = 1.0

        length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
        dir_local = LVector3f(dir_x / length, dir_z / length, dir_y / length)

        dir_world = self.panda_app.render.get_relative_vector(camera, dir_local)

        cam_pos = camera.get_pos()
        world_point = cam_pos + dir_world * distance

        return world_point

    def reconstruct_camera_pos_hpr(self, data):
        key_points = data.get("points_2d_original", data["points_2d"])
        distances = data["distances_to_camera"]
        camera = self.panda_app.camera

        lens = self.panda_app.cam.node().getLens()
        lens.setFov(data["camera_params"]["fov_x"], data["camera_params"]["fov_y"])

        img_w = data["camera_params"]["cx"] * 2.0
        img_h = data["camera_params"]["cy"] * 2.0

        proj_3d = []
        cam_pos = self.panda_vec3_to_np(camera.get_pos(camera.get_parent()))

        self.fov_y = data["camera_params"]["fov_y"]
        self.aspect_ratio = img_w/img_h

        for i in range(3):
            u = key_points[i]["x"] / img_w
            v = key_points[i]["y"] / img_h
            distance = distances[i]

            point = self.viewport_to_world_point_geometric(camera, u, v, distance, self.fov_y, self.aspect_ratio)
            proj_3d.append(point)

        scene_3d = [np.array(p, dtype=float) for p in data["points_3d"][:3]]

        M = self.compute_transform_np(
            scene_3d[0], scene_3d[1], scene_3d[2],
            proj_3d[0], proj_3d[1], proj_3d[2]
        )

        old_pos = cam_pos
        new_pos = M @ np.append(old_pos, 1.0)
        camera.set_pos(self.np_to_panda_point(new_pos[:3]))

        current_mat = camera.get_net_transform().get_mat()
        world_right = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(1, 0, 0)))
        world_up = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 1, 0)))
        world_forward = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 0, 1)))

        R = M[:3, :3]
        new_right = R @ world_right
        new_up = R @ world_up
        new_forward = R @ world_forward

        new_right /= np.linalg.norm(new_right)
        new_up /= np.linalg.norm(new_up)
        new_forward /= np.linalg.norm(new_forward)

        mat3 = Mat3()
        mat3.set_row(0, self.np_to_panda_vec(new_right))
        mat3.set_row(1, self.np_to_panda_vec(new_up))
        mat3.set_row(2, self.np_to_panda_vec(new_forward))

        quat = LQuaternionf()
        quat.set_from_matrix(mat3)

        camera.set_quat(quat)

    def load_height_map(self):
        try:
            print(f"[DEBUG] Начало загрузки height map: {self.image_path}")
            img = Image.open(self.image_path)
            print(f"[DEBUG] Изображение загружено: {img.size}, mode: {img.mode}")
            
            # Уменьшаем изображение в 10 раз
            scale_factor = 1  # Восстановим уменьшение для производительности
            new_width = max(1, img.width // scale_factor)
            new_height = max(1, img.height // scale_factor)
            
            print(f"[DEBUG] Уменьшаем изображение с {img.width}x{img.height} до {new_width}x{new_height}")
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                print(f"[DEBUG] Изображение содержит альфа-канал")
                img_rgba = img.convert('RGBA')
                r, g, b, a = img_rgba.split()
                
                alpha = np.array(a, dtype=np.float32) / 255.0
                
                if img.mode != 'L':
                    img_gray = img.convert('L')
                    height_data = np.array(img_gray, dtype=np.float32) / 255.0
                else:
                    height_data = np.array(img, dtype=np.float32) / 255.0
                
                mask = alpha > self.alpha_threshold
                
                self.height_map = height_data * mask
                self.mask = mask
            else:
                print(f"[DEBUG] Изображение без альфа-канала")
                img_gray = img.convert('L')
                self.height_map = np.array(img_gray, dtype=np.float32) / 255.0
                mask = self.height_map > 0.05
                self.mask = mask
            
            img.close()
            print(f"[DEBUG] Загрузка height map завершена успешно")
            print(f"[DEBUG] Размер сетки: {self.height_map.shape[1]}x{self.height_map.shape[0]}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки height map: {e}")
            return False
    
    def _generate_perlin_noise(self, x, y):
        """Генерирует шум Перлина"""
        try:
            if self.noise_octaves > 1:
                value = 0.0
                max_value = 0.0
                frequency = 1.0
                amplitude = 1.0
                
                for i in range(self.noise_octaves):
                    nx = x * frequency * self.noise_scale + self.noise_seed
                    ny = y * frequency * self.noise_scale + self.noise_seed
                    
                    octave_value = noise.pnoise2(nx, ny, octaves=1)
                    value += octave_value * amplitude
                    max_value += amplitude
                    
                    frequency *= self.noise_lacunarity
                    amplitude *= self.noise_persistence
                
                if max_value > 0:
                    value /= max_value
            else:
                nx = x * self.noise_scale + self.noise_seed
                ny = y * self.noise_scale + self.noise_seed
                value = noise.pnoise2(nx, ny, octaves=1)
            
            return value
            
        except Exception as e:
            print(f"[WARNING] Ошибка генерации шума Перлина: {e}")
            return 0.0

    def create_extrapolated_mesh(self):
        """
        Создает меш с экстраполяцией, используя сферическую проекцию для исходного меша
        и плоскую экстраполяцию для расширения
        """
        if not self.load_height_map():
            return None
            
        h, w = self.height_map.shape
        
        print(f"[DEBUG] Создание экстраполированного меша: {w}x{h}")
        
        camera = self.panda_app.camera
        cam_pos = camera.get_pos()
        cam_mat = camera.get_mat()
        
        # Векторы камеры для плоской экстраполяции
        forward = cam_mat.xform_vec(LVector3f(0, 1, 0))
        right = cam_mat.xform_vec(LVector3f(1, 0, 0))
        up = cam_mat.xform_vec(LVector3f(0, 0, 1))
        
        forward.normalize()
        up.normalize()
        right.normalize()
        
        # Собираем исходные вершины (сферическая проекция) - ТОЛЬКО для расчета трансформации
        source_vertices = []
        source_points_2d = []
        source_heights = []
        source_positions_original = []  # Храним оригинальные позиции для трансформации
        
        for y in range(h):
            for x in range(w):
                if not self.mask[y, x]:
                    continue
                    
                u_norm = x / w
                v_norm = y / h
                
                # Сферическая проекция для исходных точек
                height_value = 1 - self.height_map[y, x]
                distance = height_value * (self.max_depth - self.min_depth) + self.min_depth
                
                point = self.viewport_to_world_point_geometric(
                    camera, u_norm, v_norm, distance, 
                    self.fov_y, self.aspect_ratio
                )
                
                source_vertices.append((x, y, point, u_norm, v_norm))
                source_points_2d.append([point.x, point.y])
                source_heights.append(point.z)
                source_positions_original.append((point.x, point.y, point.z))
        
        if len(source_points_2d) == 0:
            print(f"[ERROR] Нет активных точек")
            return None
        
        source_points_2d = np.array(source_points_2d)
        source_heights = np.array(source_heights)
        
        print(f"[DEBUG] Создано исходных вершин для расчета трансформации: {len(source_vertices)}")
        
        # --- ВЫЧИСЛЕНИЕ ТРАНСФОРМАЦИИ ДЛЯ RESIZE ---
        
        # Целевые точки (прямоугольник в мировых координатах)
        target_points = [
            [-1.03, -2.22, 2.4],
            [-1.03,  2.4,  2.4],
            [ 1.045, 2.4,  2.4],
            [ 1.045, -2.22, 2.4]
        ]
        
        target_points_np = np.array(target_points)
        
        # Целевой bounding box из заданных точек
        target_min_x = np.min(target_points_np[:, 0])
        target_max_x = np.max(target_points_np[:, 0])
        target_min_y = np.min(target_points_np[:, 1])
        target_max_y = np.max(target_points_np[:, 1])
        
        target_center_x = (target_min_x + target_max_x) / 2
        target_center_y = (target_min_y + target_max_y) / 2
        target_center_z = np.mean(target_points_np[:, 2])
        
        print(f"[DEBUG] Целевой центр: ({target_center_x:.3f}, {target_center_y:.3f}, {target_center_z:.3f})")
        
        # Вычисляем текущий bounding box исходных вершин
        if len(source_points_2d) > 0:
            current_min_x = np.min(source_points_2d[:, 0])
            current_max_x = np.max(source_points_2d[:, 0])
            current_min_y = np.min(source_points_2d[:, 1])
            current_max_y = np.max(source_points_2d[:, 1])
            
            current_center_x = (current_min_x + current_max_x) / 2
            current_center_y = (current_min_y + current_max_y) / 2
            current_mean_z = np.mean(source_heights)
            
            print(f"[DEBUG] Текущий центр исходных вершин: ({current_center_x:.3f}, {current_center_y:.3f}, {current_mean_z:.3f})")
            
            # Вычисляем scaling и translation
            if abs(current_max_x - current_min_x) > 1e-6:
                scale_x = (target_max_x - target_min_x) / (current_max_x - current_min_x)
            else:
                scale_x = 1.0
                
            if abs(current_max_y - current_min_y) > 1e-6:
                scale_y = (target_max_y - target_min_y) / (current_max_y - current_min_y)
            else:
                scale_y = 1.0
                
            # Вектор смещения для центрирования
            translate_x = target_center_x - current_center_x
            translate_y = target_center_y - current_center_y
            
            print(f"[DEBUG] Трансформация: scale=({scale_x:.3f}, {scale_y:.3f}), translate=({translate_x:.3f}, {translate_y:.3f})")
            
            # Вычисляем трансформацию для всех вершин (для расчета границ)
            transformed_source_vertices = []
            for i, (orig_x, orig_y, orig_z) in enumerate(source_positions_original):
                # Применяем трансформацию: сначала scaling относительно центра, затем translation
                new_x = (orig_x - current_center_x) * scale_x + target_center_x
                new_y = (orig_y - current_center_y) * scale_y + target_center_y
                # Z-координата сохраняется
                new_z = orig_z
                
                transformed_source_vertices.append((new_x, new_y, new_z))
                
                # Обновляем source_vertices для дальнейшей обработки
                x, y, point, u_norm, v_norm = source_vertices[i]
                # Создаем новую точку с трансформированными координатами
                new_point = LPoint3f(float(new_x), float(new_y), float(new_z))
                source_vertices[i] = (x, y, new_point, u_norm, v_norm)
                
            print(f"[DEBUG] Применен resize исходной области:")
            print(f"  Масштаб: X={scale_x:.3f}, Y={scale_y:.3f}")
            print(f"  Смещение: X={translate_x:.3f}, Y={translate_y:.3f}")
        else:
            scale_x = scale_y = 1.0
            translate_x = translate_y = 0.0
        
        # Создаем KD-дерево для интерполяции (на основе трансформированных вершин)
        print(f"[DEBUG] Создание KD-дерева...")
        # Обновляем source_points_2d с трансформированными координатами
        source_points_2d = np.array([[v[2].x, v[2].y] for v in source_vertices])
        kd_tree = KDTree(source_points_2d)
        
        # Определяем границы исходного меша (после трансформации)
        if len(source_points_2d) > 0:
            min_x, max_x = np.min(source_points_2d[:, 0]), np.max(source_points_2d[:, 0])
            min_y, max_y = np.min(source_points_2d[:, 1]), np.max(source_points_2d[:, 1])
            min_z, max_z = np.min(source_heights), np.max(source_heights)
        else:
            min_x = max_x = min_y = max_y = min_z = max_z = 0.0
        
        print(f"[DEBUG] Границы исходного меша после трансформации:")
        print(f"  X: [{min_x:.2f}, {max_x:.2f}]")
        print(f"  Y: [{min_y:.2f}, {max_y:.2f}]")
        print(f"  Z: [{min_z:.2f}, {max_z:.2f}]")
        
        # ВЫЧИСЛЯЕМ ГРАНИЦЫ ДЛЯ ЭКСТРАПОЛЯЦИИ ОТНОСИТЕЛЬНО ЦЕЛЕВОГО ЦЕНТРА
        # Ширина и высота целевого прямоугольника
        target_width = target_max_x - target_min_x
        target_height = target_max_y - target_min_y
        
        # Добавляем padding (50% от размера целевого прямоугольника) для экстраполяции
        padding_x = target_width * 0.5
        padding_y = target_height * 0.5
        
        # Границы экстраполяции центрируются относительно того же центра
        target_min_x_ext = target_center_x - target_width * 0.5 - padding_x
        target_max_x_ext = target_center_x + target_width * 0.5 + padding_x
        target_min_y_ext = target_center_y - target_height * 0.5 - padding_y
        target_max_y_ext = target_center_y + target_height * 0.5 + padding_y
        
        print(f"[DEBUG] Целевой прямоугольник: width={target_width:.2f}, height={target_height:.2f}")
        print(f"[DEBUG] Целевая область экстраполяции:")
        print(f"  X: [{target_min_x_ext:.2f}, {target_max_x_ext:.2f}]")
        print(f"  Y: [{target_min_y_ext:.2f}, {target_max_y_ext:.2f}]")
        
        # Создаем сетку экстраполяции
        grid_resolution = self.grid_resolution
        x_grid = np.linspace(target_min_x_ext, target_max_x_ext, grid_resolution)
        y_grid = np.linspace(target_min_y_ext, target_max_y_ext, grid_resolution)
        
        print(f"[DEBUG] Создана сетка экстраполяции: {grid_resolution}x{grid_resolution}")
        print(f"[DEBUG] Центр сетки: ({np.mean(x_grid):.2f}, {np.mean(y_grid):.2f})")
        
        # Создаем меш ТОЛЬКО с сеткой экстраполяции
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("extrapolated_mesh", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        # Создаем сетку экстраполяции (единственный меш)
        grid_vertex_indices = np.zeros((len(y_grid), len(x_grid)), dtype=int)
        vertex_count = 0
        
        print(f"[DEBUG] Создание вершин сетки экстраполяции...")
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x = x_grid[j]
                y = y_grid[i]
                
                # Определяем высоту
                if x >= target_min_x and x <= target_max_x and y >= target_min_y and y <= target_max_y:
                    # Внутри целевого прямоугольника - интерполируем из исходных вершин
                    if len(source_points_2d) > 0:
                        distances, indices = kd_tree.query([[x, y]], k=3)
                        
                        if len(indices) > 0 and len(indices[0]) > 0:
                            idx_list = indices[0]
                            if distances[0][0] < 0.1:
                                # Ближайшая точка
                                z = source_heights[idx_list[0]]
                            else:
                                # Взвешенная интерполяция
                                weights = 1.0 / (distances[0] + 1e-8)
                                weights = weights / np.sum(weights)
                                z = np.sum(source_heights[idx_list] * weights)
                        else:
                            z = np.mean(source_heights)
                    else:
                        z = 0.0
                else:
                    # Вне целевого прямоугольника - шум Перлина
                    # Нормализуем координаты для шума
                    norm_x = (x - target_min_x_ext) / (target_max_x_ext - target_min_x_ext)
                    norm_y = (y - target_min_y_ext) / (target_max_y_ext - target_min_y_ext)
                    
                    noise_value = self._generate_perlin_noise(norm_x * 5, norm_y * 5)
                    normalized_noise = (noise_value + 1.0) / 2.0
                    
                    # Определяем расстояние до границы целевого прямоугольника
                    dist_to_border = min(
                        abs(x - target_min_x), abs(x - target_max_x),
                        abs(y - target_min_y), abs(y - target_max_y)
                    )
                    
                    # Плавный переход на границе
                    border_factor = max(0, 1 - dist_to_border / max(padding_x, padding_y))
                    
                    if len(source_heights) > 0:
                        base_height = np.mean(source_heights)
                    else:
                        base_height = 0.0
                        
                    noise_height = base_height + normalized_noise * self.noise_strength
                    
                    z = base_height * border_factor + noise_height * (1 - border_factor)
                
                # Добавляем вершину сетки
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1)
                texcoord.addData2f(
                    (x - target_min_x_ext) / (target_max_x_ext - target_min_x_ext),
                    (y - target_min_y_ext) / (target_max_y_ext - target_min_y_ext)
                )
                
                grid_vertex_indices[i, j] = vertex_count
                vertex_count += 1
        
        print(f"[DEBUG] Всего вершин в сетке: {vertex_count}")
        
        # Создаем треугольники ТОЛЬКО для сетки экстраполяции
        triangles = GeomTriangles(Geom.UHStatic)
        grid_triangles = 0
        
        print(f"[DEBUG] Создание треугольников сетки экстраполяции...")
        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                v1 = grid_vertex_indices[i, j]
                v2 = grid_vertex_indices[i, j + 1]
                v3 = grid_vertex_indices[i + 1, j]
                v4 = grid_vertex_indices[i + 1, j + 1]
                
                triangles.addVertices(v1, v2, v3)
                triangles.addVertices(v2, v4, v3)
                grid_triangles += 2
        
        triangles.closePrimitive()
        print(f"[DEBUG] Треугольников в сетке: {grid_triangles}")
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("extrapolated_mesh")
        node.addGeom(geom)
        
        # Рассчитываем нормали
        self._calculate_normals(vdata, geom)
        
        print(f"[DEBUG] Экстраполированный меш создан успешно")
        print(f"[DEBUG] Центр меша: ({target_center_x:.3f}, {target_center_y:.3f})")
        
        return node

    def _calculate_normals(self, vdata, geom):
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        
        vertices = []
        while not vertex_reader.isAtEnd():
            pos = vertex_reader.getData3f()
            vertices.append((pos.x, pos.y, pos.z))
        
        normals = [(0.0, 0.0, 0.0) for _ in range(len(vertices))]
        
        for i in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(i)
            if prim.getNumPrimitives() > 0:
                for j in range(prim.getNumPrimitives()):
                    start = prim.getPrimitiveStart(j)
                    end = prim.getPrimitiveEnd(j)
                    
                    if end - start == 3:
                        vi0 = prim.getVertex(start)
                        vi1 = prim.getVertex(start + 1)
                        vi2 = prim.getVertex(start + 2)
                        
                        v0 = np.array(vertices[vi0])
                        v1 = np.array(vertices[vi1])
                        v2 = np.array(vertices[vi2])
                        
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        norm = np.linalg.norm(normal)
                        
                        if norm > 0:
                            normal = normal / norm
                            
                            for vi in [vi0, vi1, vi2]:
                                curr = normals[vi]
                                normals[vi] = (
                                    curr[0] + normal[0],
                                    curr[1] + normal[1],
                                    curr[2] + normal[2]
                                )
        
        normal_writer.setRow(0)
        for i in range(len(normals)):
            nx, ny, nz = normals[i]
            norm = (nx*nx + ny*ny + nz*nz) ** 0.5
            if norm > 0:
                nx /= norm
                ny /= norm
                nz /= norm

            normal_writer.setData3f(nx, ny, nz)

    def add_mesh_to_scene(self, node):
        if hasattr(self, "mesh_node") and self.mesh_node:
            self.mesh_node.removeNode()
        
        if node:
            self.mesh_node = self.panda_app.render.attachNewNode(node)
            
            # Отладочная информация
            print(f"[DEBUG] Меш добавлен на сцену:")
            print(f"  Позиция меша: {self.mesh_node.get_pos()}")
            print(f"  Масштаб меша: {self.mesh_node.get_scale()}")
            
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            self.mesh_node.setShaderAuto()
            self.mesh_node.setScale(1, 1, 1)
            
            # НЕ устанавливаем позицию - меш уже в правильных координатах
            # self.mesh_node.setPos(-1.0, 2.2, -1.7)  # ЗАКОММЕНТИРОВАТЬ!
            
            # Проверяем позицию после всех установок
            print(f"[DEBUG] Позиция меша после установки: {self.mesh_node.get_pos()}")
            
            if not hasattr(self.panda_app, 'loaded_models'):
                self.panda_app.loaded_models = []
            if not hasattr(self.panda_app, 'model_paths'):
                self.panda_app.model_paths = {}
            
            if self.mesh_node not in self.panda_app.loaded_models:
                self.panda_app.loaded_models.append(self.mesh_node)
                self.panda_app.model_paths[id(self.mesh_node)] = "height_map_mesh"
            
            return self.mesh_node
        
        return None

    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            print(f"[ERROR] JSON-файл не выбран или не существует")
            return
        
        print(f"[DEBUG] ====== НАЧАЛО РЕКОНСТРУКЦИИ 2D->3D ======")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Калибруем глубины по JSON
        self.calibrate_depth_from_json(data)
        
        print(f"[DEBUG] Восстановление позиции камеры...")
        self.reconstruct_camera_pos_hpr(data)
    
        
        print(f"[DEBUG] Загрузка height map...")
        if not self.load_height_map():
            print(f"[ERROR] Не удалось загрузить height map")
            return
        
        node = self.create_extrapolated_mesh()
        
        if node is None:
            print(f"[ERROR] Не удалось создать меш")
            return
        
        print(f"[DEBUG] Добавление меша на сцену...")
        mesh_node = self.add_mesh_to_scene(node)

        mesh_node.setZ(-0.5)
        
        print(f"[DEBUG] ====== РЕКОНСТРУКЦИЯ 2D->3D ЗАВЕРШЕНА ======")

    