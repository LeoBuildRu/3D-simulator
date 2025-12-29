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
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation, binary_erosion
import trimesh

class MeshReconstruction:
    def __init__(self, panda_app, image_path = None):
        self.panda_app = panda_app
        self.recon_json_path = ""

        # for testing only
        self.image_path = None
        self.alpha_threshold = 0.5

        # Калиброванные параметры глубины
        self.min_depth = 14.6
        self.max_depth = 19.5

        # Параметры displacement
        self.displacement_texture_path = "textures/stones_8k/rocks_ground_01_disp_4k.jpg"
        self.displacement_strength = 0.14
        self.texture_repeatX = 1.35
        self.texture_repeatY = 3.2
        self.use_displacement = True  # Включено по умолчанию
        
        print(f"[DEBUG] Инициализирован MeshReconstruction с displacement")
        
        # Параметры экстраполяции
        self.extrapolation_enabled = True
        self.target_width = 15.0  # Ширина целевой области в метрах
        self.target_height = 10.0  # Высота целевой области в метрах
        self.grid_resolution = 512  # Разрешение сетки экстраполяции
        
        # Параметры шума Перлина
        self.noise_scale = 10.5
        self.noise_strength = 5.75
        self.noise_octaves = 4
        self.noise_persistence = 0.5
        self.noise_lacunarity = 2.0
        self.noise_seed = random.randint(0, 10000)
        
        # Параметры адаптивного подъема вершин (заимствованы из height_map_mesh_generator.py)
        self.adaptive_lift_enabled = True
        self.base_distance = 1.0  # Базовое расстояние влияния
        self.min_distance = 0.3   # Минимальное расстояние
        self.max_distance = 0.5   # Максимальное расстояние
        self.lift_intensity = 0.8  # Интенсивность подъема
        
        # Параметры сглаживания
        self.lift_smoothing_enabled = True
        self.lift_smoothing_sigma = 2.0
        self.lift_blur_enabled = True
        self.lift_blur_radius = 3
        
        # Параметры сглаживания исходного меша
        self.source_mesh_smoothing_enabled = True
        self.source_mesh_smoothing_iterations = 1
        self.source_mesh_smoothing_sigma = 0.5
        self.source_mesh_edge_preserving = True
        
        # Для хранения мешей
        self.source_mesh_node = None
        self.mesh_node = None
        
        print(f"[DEBUG] Инициализирован MeshReconstruction")
        print(f"[DEBUG] Параметры адаптивного подъема: base={self.base_distance}, intensity={self.lift_intensity}")

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
        """Генерирует шум Перлина с улучшенной обработкой параметров"""
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
                    
                # Нормализуем значение к диапазону [-1, 1]
                value = max(-1.0, min(1.0, value))
            else:
                nx = x * self.noise_scale + self.noise_seed
                ny = y * self.noise_scale + self.noise_seed
                value = noise.pnoise2(nx, ny, octaves=1)
            
            return value
            
        except Exception as e:
            print(f"[WARNING] Ошибка генерации шума Перлина: {e}")
            return 0.0
        
    def _load_displacement_texture(self):
        """Загружает и обрабатывает текстуру высот для displacement"""
        if not hasattr(self, 'displacement_texture_path') or self.displacement_texture_path is None:
            print(f"[WARNING] Displacement texture path не указан")
            return None, 0, 0
        
        try:
            height_image = Image.open(self.displacement_texture_path).convert('L')
            height_array = np.array(height_image, dtype=np.float32)
            tex_height, tex_width = height_array.shape
            
            # Нормализуем к диапазону [0, 1]
            height_min = np.min(height_array)
            height_max = np.max(height_array)
            if height_max - height_min > 0:
                height_array = (height_array - height_min) / (height_max - height_min)
            else:
                height_array = np.zeros_like(height_array)
            
            # Немного сжимаем диапазон для менее агрессивного эффекта
            height_array = np.power(height_array, 0.7)
            
            print(f"[DEBUG] Displacement texture загружена: {tex_width}x{tex_height}")
            return height_array, tex_width, tex_height
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки displacement texture: {e}")
            return None, 0, 0
        
    def _apply_displacement_to_source(self, height_grid, source_mask, x_grid, y_grid):
        """
        Применяет displacement ТОЛЬКО к исходной области (source_mask)
        С УСИЛЕННЫМ эффектом для лучшей видимости
        """
        if not self.use_displacement:
            print(f"[DEBUG] Displacement выключен, пропускаем")
            return height_grid
        
        # Получаем параметры повторения текстур из current_texture_set, если есть
        if hasattr(self.panda_app, 'current_texture_set') and self.panda_app.current_texture_set:
            texture_set = self.panda_app.current_texture_set
            texture_repeatX = texture_set.get('textureRepeatX', self.texture_repeatX)
            texture_repeatY = texture_set.get('textureRepeatY', self.texture_repeatY)
        else:
            texture_repeatX = self.texture_repeatX
            texture_repeatY = self.texture_repeatY
        
        # Сохраняем копию для отладки
        before_displacement = height_grid.copy()
        
        height_array, tex_width, tex_height = self._load_displacement_texture()
        if height_array is None:
            print(f"[WARNING] Не удалось загрузить displacement texture, пропускаем displacement")
            return height_grid
        
        h, w = height_grid.shape
        displaced_grid = height_grid.copy()
        
        print(f"[DEBUG] Применение displacement ТОЛЬКО к исходной области")
        print(f"[DEBUG] Сетка: {w}x{h}, текстура: {tex_width}x{tex_height}")
        print(f"[DEBUG] Пикселей в исходной области: {np.sum(source_mask)}")
        print(f"[DEBUG] Displacement strength: {self.displacement_strength}")
        print(f"[DEBUG] Texture repeat: ({texture_repeatX}, {texture_repeatY})")
        
        # НОВОЕ: Анализируем диапазон высот
        height_min = np.min(height_grid[source_mask])
        height_max = np.max(height_grid[source_mask])
        height_range = height_max - height_min
        print(f"[DEBUG] Исходный диапазон высот: {height_range:.6f}")
        
        # Вычисляем целевой эффект displacement
        target_effect = height_range * 0.3  # 30% от диапазона высот
        print(f"[DEBUG] Целевой эффект displacement: {target_effect:.6f}")
        
        processed_points = 0
        source_points = np.sum(source_mask)
        
        # Для статистики
        displacement_values = []
        
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    # UV-координаты с учетом повторения текстуры
                    u = (j / (w - 1)) * texture_repeatX if w > 1 else 0
                    v = (i / (h - 1)) * texture_repeatY if h > 1 else 0
                    
                    # Билинейная интерполяция
                    tex_x = (u % 1.0) * (tex_width - 1)
                    tex_y = (v % 1.0) * (tex_height - 1)
                    
                    x1 = int(tex_x)
                    y1 = int(tex_y)
                    x2 = min(x1 + 1, tex_width - 1)
                    y2 = min(y1 + 1, tex_height - 1)
                    
                    dx = tex_x - x1
                    dy = tex_y - y1
                    
                    # Интерполяция
                    h11 = height_array[y1, x1]
                    h12 = height_array[y2, x1]
                    h21 = height_array[y1, x2]
                    h22 = height_array[y2, x2]
                    
                    hx1 = h11 * (1 - dx) + h21 * dx
                    hx2 = h12 * (1 - dx) + h22 * dx
                    texture_height = hx1 * (1 - dy) + hx2 * dy
                    
                    # УСИЛЕННЫЙ displacement
                    displacement = (texture_height - 0.5) * self.displacement_strength * height_range * 0.5
                    
                    displaced_grid[i, j] += displacement
                    displacement_values.append(displacement)
                    
                    processed_points += 1
        
        # Анализ результатов
        if displacement_values:
            displacement_array = np.array(displacement_values)
            print(f"[DEBUG] Статистика displacement:")
            print(f"  Applied to: {len(displacement_values)} points")
            print(f"  Min displacement: {np.min(displacement_array):.6f}")
            print(f"  Max displacement: {np.max(displacement_array):.6f}")
            print(f"  Mean displacement: {np.mean(displacement_array):.6f}")
            print(f"  Std displacement: {np.std(displacement_array):.6f}")
            
            # Проверяем, насколько изменилась общая геометрия
            new_min = np.min(displaced_grid[source_mask])
            new_max = np.max(displaced_grid[source_mask])
            new_range = new_max - new_min
            print(f"[DEBUG] Новый диапазон высот: {new_range:.6f}")
            print(f"[DEBUG] Изменение диапазона: {(new_range - height_range)/height_range*100:.1f}%")
        
        # Отладочная визуализация
        if hasattr(self, 'enable_displacement_debug') and self.enable_displacement_debug:
            self._debug_displacement_effect(displaced_grid, source_mask, before_displacement)
        
        print(f"[DEBUG] Displacement применен успешно")
        return displaced_grid
    
    def _debug_displacement_effect(self, height_grid, source_mask, before_displacement):
        """
        Отладочный метод для проверки эффекта displacement
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        
        self._debug_counter += 1
        
        # Выбираем срез по центру для визуализации
        center_y = height_grid.shape[0] // 2
        slice_width = min(50, height_grid.shape[1] // 2)
        start_x = height_grid.shape[1] // 2 - slice_width // 2
        
        before_slice = before_displacement[center_y, start_x:start_x+slice_width]
        after_slice = height_grid[center_y, start_x:start_x+slice_width]
        
        # Создаем простой график
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(before_slice, 'b-', label='Before Displacement', alpha=0.7)
        plt.plot(after_slice, 'r-', label='After Displacement', alpha=0.7)
        plt.fill_between(range(len(before_slice)), before_slice, after_slice, alpha=0.3)
        plt.title(f'Displacement Effect (Slice {self._debug_counter})')
        plt.xlabel('X position')
        plt.ylabel('Height')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Разница
        plt.subplot(1, 2, 2)
        diff = after_slice - before_slice
        plt.bar(range(len(diff)), diff, color='purple', alpha=0.7)
        plt.title('Height Difference (After - Before)')
        plt.xlabel('X position')
        plt.ylabel('Height Difference')
        plt.grid(True, alpha=0.3)
        
        # Статистика
        print(f"[DEBUG DISPLACEMENT] Статистика среза {self._debug_counter}:")
        print(f"  Max difference: {np.max(diff):.6f}")
        print(f"  Min difference: {np.min(diff):.6f}")
        print(f"  Mean difference: {np.mean(diff):.6f}")
        print(f"  Std difference: {np.std(diff):.6f}")
        
        plt.tight_layout()
        plt.savefig(f'debug_displacement_{self._debug_counter}.png')
        plt.close()
        
        # Также сохраняем в текстовый файл
        with open('displacement_debug.txt', 'a') as f:
            f.write(f"=== Displacement Debug {self._debug_counter} ===\n")
            f.write(f"Shape: {height_grid.shape}\n")
            f.write(f"Source points: {np.sum(source_mask)}\n")
            f.write(f"Max diff: {np.max(diff):.6f}\n")
            f.write(f"Min diff: {np.min(diff):.6f}\n")
            f.write(f"Mean diff: {np.mean(diff):.6f}\n")
            f.write(f"Std diff: {np.std(diff):.6f}\n")
            f.write(f"Total height range: {np.max(height_grid) - np.min(height_grid):.6f}\n")
            f.write("\n")

    def create_extrapolated_mesh(self):
        """
        Создает меш с экстраполяцией, используя сферическую проекцию для исходного меша
        и плоскую экстраполяцию для расширения с бесшовным переходом
        """
        if not self.load_height_map():
            return None
            
        h, w = self.height_map.shape
        
        print(f"[DEBUG] Создание экстраполированного меша: {w}x{h}")
        
        # Вместо этого - используем параметры, установленные в __init__ или оставляем как есть
        # Если хотим настроить отдельно для экстраполяции, делаем это явно:
        extrapolation_noise_scale = 0.7  # Восстанавливаем нормальный масштаб
        extrapolation_noise_strength = 1.05  # Восстанавливаем нормальную силу
        
        # ОБНОВЛЕНО: Параметры для более плавного перехода
        self.adaptive_lift_enabled = True
        self.base_distance = 2.0  # Увеличен радиус влияния
        self.min_distance = 0.5
        self.max_distance = 6.0
        self.lift_intensity = 0.6  # Уменьшена интенсивность подъема
        
        # ОБНОВЛЕНО: Усилено сглаживание
        self.lift_smoothing_enabled = True
        self.lift_smoothing_sigma = 3.0  # Увеличено сглаживание
        self.lift_blur_enabled = True
        self.lift_blur_radius = 4  # Увеличен радиус размытия
        
        self.source_mesh_smoothing_enabled = True
        self.source_mesh_smoothing_iterations = 2  # Увеличено количество итераций
        self.source_mesh_smoothing_sigma = 1.0  # Усилено сглаживание
        self.source_mesh_edge_preserving = False  # Отключено сохранение границ для лучшего слияния
        
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
        
        # Собираем исходные вершины (сферическая проекция)
        source_vertices = []
        source_points_2d = []
        source_heights = []
        source_positions_original = []
        
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
        
        print(f"[DEBUG] Создано исходных вершин: {len(source_vertices)}")
        
        # --- ВЫЧИСЛЕНИЕ ТРАНСФОРМАЦИИ ДЛЯ RESIZE ---
        
        target_points = [
            [-1.03, -2.22, 2.4],
            [-1.03,  2.4,  2.4],
            [ 1.045, 2.4,  2.4],
            [ 1.045, -2.22, 2.4]
        ]
        
        target_points_np = np.array(target_points)
        
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
            
            if abs(current_max_x - current_min_x) > 1e-6:
                scale_x = (target_max_x - target_min_x) / (current_max_x - current_min_x)
            else:
                scale_x = 1.0
                
            if abs(current_max_y - current_min_y) > 1e-6:
                scale_y = (target_max_y - target_min_y) / (current_max_y - current_min_y)
            else:
                scale_y = 1.0
                
            translate_x = target_center_x - current_center_x
            translate_y = target_center_y - current_center_y
            
            print(f"[DEBUG] Трансформация: scale=({scale_x:.3f}, {scale_y:.3f}), translate=({translate_x:.3f}, {translate_y:.3f})")
            
            for i, (orig_x, orig_y, orig_z) in enumerate(source_positions_original):
                new_x = (orig_x - current_center_x) * scale_x + target_center_x
                new_y = (orig_y - current_center_y) * scale_y + target_center_y
                new_z = orig_z
                
                x_idx, y_idx, point, u_norm, v_norm = source_vertices[i]
                new_point = LPoint3f(float(new_x), float(new_y), float(new_z))
                source_vertices[i] = (x_idx, y_idx, new_point, u_norm, v_norm)
                
            print(f"[DEBUG] Применен resize исходной области")
        else:
            scale_x = scale_y = 1.0
            translate_x = translate_y = 0.0
        
        # Создаем KD-дерево для интерполяции
        print(f"[DEBUG] Создание KD-дерева...")
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
        
        # ВЫЧИСЛЯЕМ ГРАНИЦЫ ДЛЯ ЭКСТРАПОЛЯЦИИ
        target_width = target_max_x - target_min_x
        target_height = target_max_y - target_min_y
        
        padding_x = target_width * 0.8
        padding_y = target_height * 0.8
        
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
        
        # ВЫЧИСЛЯЕМ ВЫСОТЫ ДЛЯ ВСЕЙ СЕТКИ
        print(f"[DEBUG] Вычисление высот для сетки с бесшовным переходом...")
        height_grid = np.zeros((len(y_grid), len(x_grid)))
        source_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
        
        if len(source_heights) > 0:
            base_height = np.mean(source_heights)
            height_range = np.max(source_heights) - np.min(source_heights)
        else:
            base_height = 0.0
            height_range = 1.0
        
        print(f"[DEBUG] Базовая высота: {base_height:.3f}, Диапазон высот: {height_range:.3f}")
        
        # Определяем граничные точки исходного меша
        boundary_points = []
        
        # ИНИЦИАЛИЗИРУЕМ СЧЕТЧИКИ ДО ИСПОЛЬЗОВАНИЯ
        noise_points_count = 0
        source_points_count = 0
        
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x = x_grid[j]
                y = y_grid[i]
                
                is_inside_target = (x >= target_min_x and x <= target_max_x and 
                                y >= target_min_y and y <= target_max_y)
                
                if is_inside_target and len(source_points_2d) > 0:
                    distances, indices = kd_tree.query([[x, y]], k=5)
                    
                    if len(indices) > 0 and len(indices[0]) > 0:
                        idx_list = indices[0]
                        
                        weights = np.exp(-distances[0] / 0.2)
                        weights = weights / np.sum(weights)
                        z = np.sum(source_heights[idx_list] * weights)
                        
                        source_mask[i, j] = True
                        height_grid[i, j] = z
                        source_points_count += 1
                        
                        # ПРОВЕРЯЕМ, ЧТО СЧЕТЧИК УЖЕ ИНИЦИАЛИЗИРОВАН
                        if noise_points_count < 5:
                            # Этот код будет выполнен только когда noise_points_count уже инициализирован
                            pass
                        
                        if (i > 0 and i < len(y_grid)-1 and j > 0 and j < len(x_grid)-1):
                            is_boundary = False
                            for di in [-2, -1, 0, 1, 2]:
                                for dj in [-2, -1, 0, 1, 2]:
                                    if di == 0 and dj == 0:
                                        continue
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < len(y_grid) and 0 <= nj < len(x_grid):
                                        nx, ny = x_grid[nj], y_grid[ni]
                                        if not (nx >= target_min_x and nx <= target_max_x and 
                                                ny >= target_min_y and ny <= target_max_y):
                                            is_boundary = True
                                            break
                            if is_boundary:
                                boundary_points.append([x, y, z])
                    else:
                        z = base_height
                        source_mask[i, j] = False
                        height_grid[i, j] = z
                else:
                    # ОБЛАСТЬ ЭКСТРАПОЛЯЦИИ - здесь должен применяться шум Перлина
                    norm_x = (x - target_min_x_ext) / (target_max_x_ext - target_min_x_ext)
                    norm_y = (y - target_min_y_ext) / (target_max_y_ext - target_min_y_ext)
                    
                    # Используем улучшенную генерацию шума с нормальными параметрами
                    # Вместо умножения на 0.05 используем нормальный масштаб
                    noise_value = self._generate_perlin_noise(
                        norm_x * extrapolation_noise_scale, 
                        norm_y * extrapolation_noise_scale
                    )
                    
                    # Отладочная информация для первых нескольких точек
                    # ПРОВЕРЯЕМ, ЧТО СЧЕТЧИК УЖЕ ИНИЦИАЛИЗИРОВАН
                    if noise_points_count < 5:
                        print(f"[DEBUG ШУМ] Точка экстраполяции [{i},{j}]:")
                        print(f"  Координаты: ({x:.3f}, {y:.3f})")
                        print(f"  Норм.координаты: ({norm_x:.3f}, {norm_y:.3f})")
                        print(f"  Значение шума: {noise_value:.6f}")
                        print(f"  Параметры шума: scale={extrapolation_noise_scale}, strength={extrapolation_noise_strength}")
                    
                    distance_to_center = np.sqrt((x - target_center_x)**2 + (y - target_center_y)**2)
                    max_distance = np.sqrt((target_max_x_ext - target_center_x)**2 + 
                                        (target_max_y_ext - target_center_y)**2)
                    normalized_distance = distance_to_center / max_distance
                    
                    # Используем параметры шума для экстраполяции
                    adaptive_strength = extrapolation_noise_strength * (0.3 + 0.7 * normalized_distance)
                    noise_height = base_height + noise_value * adaptive_strength * height_range
                    
                    source_mask[i, j] = False
                    height_grid[i, j] = noise_height
                    noise_points_count += 1
        
        print(f"[DEBUG] Статистика по точкам:")
        print(f"  Исходных точек (интерполяция): {source_points_count}")
        print(f"  Точек экстраполяции (шум): {noise_points_count}")
        print(f"  Всего точек: {len(y_grid) * len(x_grid)}")
        print(f"[DEBUG] Найдено граничных точек: {len(boundary_points)}")
        
        # ПРИМЕНЯЕМ АДАПТИВНЫЙ ПОДЪЕМ ВЕРШИН ЭКСТРАПОЛЯЦИИ
        if self.extrapolation_enabled and self.adaptive_lift_enabled and len(boundary_points) > 0:
            print(f"[DEBUG] Применение адаптивного подъема вершин...")
            height_grid, lifted_mask = self._apply_adaptive_vertex_lift(
                height_grid, source_mask, x_grid, y_grid, boundary_points
            )
            
            if self.lift_smoothing_enabled and np.any(lifted_mask):
                print(f"[DEBUG] Сглаживание поднятой области...")
                height_grid = self._smooth_lifted_area(
                    height_grid, lifted_mask, sigma=self.lift_smoothing_sigma
                )
            
            if self.lift_blur_enabled and self.lift_blur_radius > 0:
                print(f"[DEBUG] Размытие границ подъема...")
                height_grid = self._blur_lift_boundary(
                    height_grid, source_mask, boundary_points, blur_radius=self.lift_blur_radius
                )
        else:
            lifted_mask = np.zeros_like(source_mask, dtype=bool)
            print(f"[DEBUG] Адаптивный подъем не применен")
        
        # СГЛАЖИВАНИЕ ИСХОДНОГО МЕША
        if self.source_mesh_smoothing_enabled and np.any(source_mask):
            print(f"[DEBUG] Сглаживание исходного меша...")
            height_grid = self._smooth_source_mesh(
                height_grid, source_mask,
                sigma=self.source_mesh_smoothing_sigma,
                iterations=self.source_mesh_smoothing_iterations,
                preserve_edges=self.source_mesh_edge_preserving
            )
        
        # ОБНОВЛЕНО: Улучшенная постобработка граничной зоны
        print(f"[DEBUG] Улучшенная постобработка граничной зоны...")
        height_grid = self._improved_boundary_blending(height_grid, source_mask, boundary_width=8)
        
        if self.use_displacement:
            print(f"[DEBUG] Применение displacement к исходной области...")
            height_grid = self._apply_displacement_to_source(
                height_grid, source_mask, x_grid, y_grid
            )

        # ДОПОЛНИТЕЛЬНОЕ СГЛАЖИВАНИЕ ВСЕГО МЕША
        print(f"[DEBUG] Финальное сглаживание всего меша...")
        height_grid = gaussian_filter(height_grid, sigma=1.0)
        
        # СОЗДАЕМ МЕШ
        print(f"[DEBUG] Создание вершин меша...")
        
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("extrapolated_mesh", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        grid_vertex_indices = np.zeros((len(y_grid), len(x_grid)), dtype=int)
        vertex_count = 0
        
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x = x_grid[j]
                y = y_grid[i]
                z = height_grid[i, j]
                
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1)
                
                u = (x - target_min_x_ext) / (target_max_x_ext - target_min_x_ext)
                v = (y - target_min_y_ext) / (target_max_y_ext - target_min_y_ext)
                texcoord.addData2f(u, v)
                
                grid_vertex_indices[i, j] = vertex_count
                vertex_count += 1
        
        print(f"[DEBUG] Всего вершин в сетке: {vertex_count}")
        
        triangles = GeomTriangles(Geom.UHStatic)
        grid_triangles = 0
        
        print(f"[DEBUG] Создание треугольников сетки...")
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
        
        self._calculate_normals(vdata, geom)
        
        print(f"[DEBUG] Экстраполированный меш создан успешно с улучшенным бесшовным переходом")
        print(f"[DEBUG] Центр меша: ({target_center_x:.3f}, {target_center_y:.3f})")
        
        return node

    def _improved_boundary_blending(self, height_grid, source_mask, boundary_width=8):
        """
        Улучшенное смешивание границы между исходной областью и экстраполяцией
        Устраняет бардюр путем плавного перехода высот
        """
        print(f"[DEBUG] Улучшенное смешивание границы: width={boundary_width}")
        
        h, w = height_grid.shape
        blended_grid = height_grid.copy()
        
        # Создаем маску граничной зоны
        from scipy.ndimage import distance_transform_edt
        
        # Расстояние до границы исходной области
        distance_to_source = distance_transform_edt(~source_mask)
        distance_from_source = distance_transform_edt(source_mask)
        
        # Маска граничной зоны (внутренняя и внешняя части)
        inner_boundary_mask = (distance_from_source <= boundary_width) & source_mask
        outer_boundary_mask = (distance_to_source <= boundary_width) & ~source_mask
        boundary_zone = inner_boundary_mask | outer_boundary_mask
        
        print(f"[DEBUG] Размер граничной зоны: {np.sum(boundary_zone)} точек")
        
        # Для каждой точки в граничной зоне
        for i in range(h):
            for j in range(w):
                if boundary_zone[i, j]:
                    # Определяем расстояние до границы
                    if source_mask[i, j]:
                        distance = distance_from_source[i, j]
                    else:
                        distance = distance_to_source[i, j]
                    
                    # Нормализуем расстояние (0 на границе, 1 на краю зоны)
                    normalized_distance = distance / boundary_width
                    
                    # Собираем высоты в окрестности
                    neighborhood_size = 3
                    heights = []
                    weights = []
                    
                    for di in range(-neighborhood_size, neighborhood_size + 1):
                        for dj in range(-neighborhood_size, neighborhood_size + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                dist = np.sqrt(di*di + dj*dj)
                                weight = np.exp(-dist*dist / 2.0)
                                heights.append(height_grid[ni, nj] * weight)
                                weights.append(weight)
                    
                    if weights:
                        neighborhood_avg = np.sum(heights) / np.sum(weights)
                        
                        # Плавное смешивание с учетом расстояния до границы
                        # Чем ближе к границе, тем больше влияние соседних высот
                        blend_factor = 1.0 - normalized_distance
                        blended_grid[i, j] = (height_grid[i, j] * (1 - blend_factor) + 
                                            neighborhood_avg * blend_factor)
        
        print(f"[DEBUG] Улучшенное смешивание границы завершено")
        return blended_grid

    # МЕТОДЫ АДАПТИВНОГО ПОДЪЕМА И СГЛАЖИВАНИЯ (ЗАИМСТВОВАНЫ ИЗ height_map_mesh_generator.py)
    
    def _smooth_lifted_area(self, height_grid, lifted_mask, sigma=2.0):
        print(f"[DEBUG] Начало сглаживания поднятой области: sigma={sigma}")
        if not np.any(lifted_mask):
            print(f"[DEBUG] Поднятая область пуста, сглаживание не требуется")
            return height_grid
        
        smoothed_height = height_grid.copy()
        
        try:
            gaussian_smoothed = gaussian_filter(height_grid, sigma=sigma)
            
            smoothed_height[lifted_mask] = gaussian_smoothed[lifted_mask]
            
        except Exception as e:
            print(f"[DEBUG] Ошибка при гауссовом сглаживании поднятой области: {e}, используем упрощенный метод")
            for iteration in range(3):
                print(f"[DEBUG] Упрощенное сглаживание, итерация {iteration + 1}/3")
                new_heights = smoothed_height.copy()
                for i in range(1, smoothed_height.shape[0]-1):
                    for j in range(1, smoothed_height.shape[1]-1):
                        if lifted_mask[i, j]:
                            neighbors = []
                            for di in range(-2, 3):
                                for dj in range(-2, 3):
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < smoothed_height.shape[0] and 0 <= nj < smoothed_height.shape[1]:
                                        neighbors.append(smoothed_height[ni, nj])
                            new_heights[i, j] = np.mean(neighbors)
                smoothed_height = new_heights
        
        print(f"[DEBUG] Сглаживание поднятой области завершено")
        return smoothed_height
    
    def _blur_lift_boundary(self, height_grid, source_mask, boundary_points, blur_radius=3):
        print(f"[DEBUG] Начало размытия границ подъема: blur_radius={blur_radius}")
        if len(boundary_points) == 0 or blur_radius <= 0:
            print(f"[DEBUG] Нет граничных точек или радиус размытия = 0")
            return height_grid
        
        boundary_points_array = np.array(boundary_points)
        print(f"[DEBUG] Граничные точки: {len(boundary_points)} точек")
        
        blurred_height = height_grid.copy()
        
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
            print(f"[DEBUG] KDTree построен для {len(boundary_points_array)} точек")
        except Exception as e:
            print(f"[DEBUG] Ошибка построения KDTree: {e}")
            return height_grid
        
        h, w = height_grid.shape
        boundary_influence_mask = np.zeros((h, w), dtype=bool)
        
        processed_points = 0
        total_points = h * w - np.sum(source_mask)
        print(f"[DEBUG] Обработка влияния границ: всего {total_points} точек для проверки")
        
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    continue
                
                x = self._current_x_grid[j] if hasattr(self, '_current_x_grid') else j
                y = self._current_y_grid[i] if hasattr(self, '_current_y_grid') else i
                
                distances, _ = boundary_tree.query([x, y], k=1)
                
                if distances <= blur_radius:
                    boundary_influence_mask[i, j] = True
                
                processed_points += 1
                if processed_points % 10000 == 0:
                    print(f"[DEBUG] Обработано {processed_points}/{total_points} точек ({processed_points/total_points*100:.1f}%)")
        
        print(f"[DEBUG] Точки под влиянием границ: {np.sum(boundary_influence_mask)}")
        
        if np.any(boundary_influence_mask):
            points_to_blur = np.sum(boundary_influence_mask)
            print(f"[DEBUG] Начинаем размытие {points_to_blur} точек")
            
            blurred_points = 0
            for i in range(h):
                for j in range(w):
                    if boundary_influence_mask[i, j]:
                        neighbors = []
                        weights = []
                        
                        for di in range(-blur_radius, blur_radius + 1):
                            for dj in range(-blur_radius, blur_radius + 1):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    dist = np.sqrt(di*di + dj*dj)
                                    weight = np.exp(-dist*dist / (2 * (blur_radius/2)**2))
                                    neighbors.append(height_grid[ni, nj] * weight)
                                    weights.append(weight)
                        
                        if weights:
                            blurred_height[i, j] = np.sum(neighbors) / np.sum(weights)
                        
                        blurred_points += 1
                        if blurred_points % 1000 == 0:
                            print(f"[DEBUG] Размыто {blurred_points}/{points_to_blur} точек ({blurred_points/points_to_blur*100:.1f}%)")
            
        print(f"[DEBUG] Размытие границ подъема завершено")
        return blurred_height
    
    def _calculate_adaptive_distance_parameters(self, boundary_heights):
        if len(boundary_heights) == 0:
            return self.base_distance, self.min_distance, self.max_distance
        
        min_boundary_height = np.min(boundary_heights)
        max_boundary_height = np.max(boundary_heights)
        height_range = max_boundary_height - min_boundary_height
        
        base_distance = self.base_distance + min(height_range * 0.5, 0.7) 
        min_distance = max(self.min_distance, base_distance * 0.5)        
        max_distance = self.max_distance + min(height_range * 2.0, 4.0)   
        
        print(f"[DEBUG] Адаптивные параметры расстояния: base={base_distance:.2f}, min={min_distance:.2f}, max={max_distance:.2f}")
        return base_distance, min_distance, max_distance
    
    def _apply_adaptive_vertex_lift(self, height_grid, source_mask, x_grid, y_grid, boundary_points):
        """
        Улучшенный адаптивный подъем вершин с более точным определением границ
        """
        print(f"[DEBUG] Начало улучшенного адаптивного подъема вершин")
        if len(boundary_points) == 0:
            print(f"[DEBUG] Нет граничных точек, подъем не требуется")
            return height_grid, np.zeros_like(source_mask, dtype=bool)
        
        self._current_x_grid = x_grid
        self._current_y_grid = y_grid
        
        boundary_points_array = np.array(boundary_points)
        boundary_heights = boundary_points_array[:, 2]
        
        # ОБНОВЛЕНО: Уменьшаем параметры для более точного подъема
        base_distance = 0.3  # Уменьшаем радиус влияния
        max_distance = 1.0   # Уменьшаем максимальное расстояние
        
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
            use_kdtree = True
            print(f"[DEBUG] KDTree построен для адаптивного подъема")
        except Exception as e:
            use_kdtree = False
            print(f"[DEBUG] Ошибка построения KDTree: {e}, используем линейный поиск")
        
        corrected_height = height_grid.copy()
        lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        lifted_vertices_count = 0
        total_vertices = np.sum(~source_mask)
        print(f"[DEBUG] Всего вершин для проверки подъема: {total_vertices}")
        
        # Создаем маску для точек, которые находятся близко к границе
        print(f"[DEBUG] Создание маски близких к границе точек...")
        
        h, w = height_grid.shape
        distance_map = np.full((h, w), np.inf)
        
        # Вычисляем расстояния до границы для каждой точки
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    continue
                    
                x = x_grid[j]
                y = y_grid[i]
                
                # Ищем ближайшую граничную точку
                min_dist = np.inf
                nearest_height = 0
                
                if use_kdtree:
                    distances, indices = boundary_tree.query([[x, y]], k=1)
                    if len(distances) > 0:
                        min_dist = distances[0]
                        if len(indices) > 0:
                            nearest_height = boundary_heights[indices[0]]
                else:
                    distances = np.sqrt((boundary_points_array[:, 0] - x)**2 + 
                                    (boundary_points_array[:, 1] - y)**2)
                    min_dist = np.min(distances)
                    nearest_idx = np.argmin(distances)
                    nearest_height = boundary_points_array[nearest_idx, 2]
                
                distance_map[i, j] = min_dist
                
                # Применяем подъем только если точка близко к границе
                if min_dist <= max_distance:
                    current_z = height_grid[i, j]
                    height_diff = nearest_height - current_z
                    
                    if height_diff > 0:  # Только если граница выше текущей точки
                        # Плавная функция перехода
                        normalized_distance = min_dist / max_distance
                        
                        # Квадратичная функция плавности для более резкого перехода
                        t = normalized_distance
                        smooth_factor = 1.0 - t * t  # Более резкий переход
                        
                        lift_amount = height_diff * smooth_factor * self.lift_intensity * 0.5  # Уменьшаем интенсивность
                        
                        corrected_height[i, j] += lift_amount
                        lifted_mask[i, j] = True
                        lifted_vertices_count += 1
        
        # Отладочная информация о расстояниях
        valid_distances = distance_map[~source_mask & (distance_map < np.inf)]
        if len(valid_distances) > 0:
            print(f"[DEBUG] Статистика расстояний до границы:")
            print(f"  Минимальное расстояние: {np.min(valid_distances):.3f}")
            print(f"  Среднее расстояние: {np.mean(valid_distances):.3f}")
            print(f"  Максимальное расстояние: {np.max(valid_distances):.3f}")
            print(f"  Точек в пределах max_distance: {np.sum(valid_distances <= max_distance)}")
        
        print(f"[DEBUG] Адаптивный подъем завершен: поднято {lifted_vertices_count} вершин")
        return corrected_height, lifted_mask

    def _smooth_source_mesh(self, height_grid, source_mask, sigma=0.5, iterations=1, preserve_edges=True):
        print(f"[DEBUG] Начало сглаживания исходного меша: {height_grid.shape}, sigma={sigma}, iterations={iterations}")
        if not np.any(source_mask):
            print(f"[DEBUG] Исходный меш пуст, сглаживание не требуется")
            return height_grid
        
        smoothed_height = height_grid.copy()
        
        protected_mask = source_mask.copy()
        
        if preserve_edges:
            print(f"[DEBUG] Сохранение границ включено")
            edge_mask = self._create_edge_mask(source_mask)
            
            protection_radius = max(1, int(np.ceil(sigma)))
            
            protected_mask = binary_dilation(edge_mask, 
                                            structure=np.ones((2*protection_radius+1, 2*protection_radius+1)))
            
            protected_mask = protected_mask & source_mask
            
        masked_height = height_grid.copy()
        masked_height[~source_mask] = np.nan
        
        for iteration in range(iterations):
            print(f"[DEBUG] Итерация сглаживания исходного меша {iteration + 1}/{iterations}")
            gaussian_smoothed = self._gaussian_filter_with_nan(masked_height, sigma)
            
            replace_mask = source_mask & ~protected_mask
            smoothed_height[replace_mask] = gaussian_smoothed[replace_mask]
            
            masked_height[replace_mask] = smoothed_height[replace_mask]
            
        print(f"[DEBUG] Сглаживание исходного меша завершено")
        return smoothed_height
    
    def _gaussian_filter_with_nan(self, data, sigma):
        from scipy.ndimage import gaussian_filter
        
        valid_mask = ~np.isnan(data)
        
        if not np.any(valid_mask):
            return data
        
        data_filled = np.where(valid_mask, data, 0)
        
        smoothed_data = gaussian_filter(data_filled, sigma=sigma)
        smoothed_weights = gaussian_filter(valid_mask.astype(float), sigma=sigma)
        
        smoothed_weights = np.where(smoothed_weights > 0, smoothed_weights, 1)
        
        result = smoothed_data / smoothed_weights
        
        result[smoothed_weights < 0.01] = data[smoothed_weights < 0.01]
        
        return result

    def _create_edge_mask(self, source_mask):
        from scipy.ndimage import binary_dilation, binary_erosion
        
        dilated = binary_dilation(source_mask, structure=np.ones((3, 3)))
        eroded = binary_erosion(source_mask, structure=np.ones((3, 3)))
        
        edge_mask = dilated & ~eroded
        
        return edge_mask
    
    def _postprocess_boundary_zone(self, height_grid, source_mask, boundary_width=5):
        print(f"[DEBUG] Постобработка граничной зоны шириной {boundary_width}")
        
        corrected_height = height_grid.copy()
        
        h, w = height_grid.shape
        
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    min_boundary_height = height_grid[i, j]
                    
                    for di in range(-boundary_width, boundary_width + 1):
                        for dj in range(-boundary_width, boundary_width + 1):
                            ni, nj = i + di, j + dj
                            if (0 <= ni < h and 0 <= nj < w):
                                if not source_mask[ni, nj]:
                                    boundary_dist = np.sqrt(di*di + dj*dj)
                                    if boundary_dist < boundary_width:
                                        weight = 1.0 - (boundary_dist / boundary_width)
                                        corrected_height[i, j] = max(
                                            corrected_height[i, j],
                                            height_grid[ni, nj] * (1.0 - weight) + 
                                            corrected_height[i, j] * weight
                                        )
        
        return corrected_height

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

    def _setup_uv_coordinates_after_boolean(self, boolean_mesh_np, original_mesh_np):
        """
        Настраивает UV-координаты для меша после boolean операции
        с использованием мировых координат для правильного повторения текстуры
        """
        print(f"[DEBUG] Настройка UV-координат после boolean операции с использованием мировых координат")
        
        if not boolean_mesh_np or not boolean_mesh_np.node():
            print(f"[ERROR] Boolean mesh не существует")
            return None
        
        geom_node = boolean_mesh_np.node()
        if geom_node.getNumGeoms() == 0:
            print(f"[ERROR] В boolean mesh нет геометрии")
            return None
        
        # Получаем параметры повторения текстур
        if hasattr(self.panda_app, 'current_texture_set') and self.panda_app.current_texture_set:
            texture_set = self.panda_app.current_texture_set
        else:
            texture_set = {}
        
        texture_repeatX = texture_set.get('textureRepeatX', self.texture_repeatX)
        texture_repeatY = texture_set.get('textureRepeatY', self.texture_repeatY)
        
        print(f"[DEBUG] Параметры повторения для UV: repeatX={texture_repeatX}, repeatY={texture_repeatY}")
        
        # Получаем геометрию boolean меша
        geom = geom_node.getGeom(0)
        vdata = geom.getVertexData()
        
        # Создаем новую vertex data с UV-координатами
        format = GeomVertexFormat.getV3n3t2()
        new_vdata = GeomVertexData("boolean_mesh_with_uv", format, Geom.UHStatic)
        
        # Копируем вершины и нормали
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_reader = GeomVertexReader(vdata, "normal")
        
        vertex_writer = GeomVertexWriter(new_vdata, "vertex")
        normal_writer = GeomVertexWriter(new_vdata, "normal")
        texcoord_writer = GeomVertexWriter(new_vdata, "texcoord")
        
        vertices = []
        normals = []
        uv_coords = []
        
        # Получаем все вершины для анализа
        while not vertex_reader.isAtEnd():
            vertex = vertex_reader.getData3f()
            normal = normal_reader.getData3f() if not normal_reader.isAtEnd() else LVector3f(0, 0, 1)
            vertices.append(vertex)
            normals.append(normal)
        
        # Находим минимальные и максимальные координаты текущего меша
        min_x = min(v.x for v in vertices)
        max_x = max(v.x for v in vertices)
        min_y = min(v.y for v in vertices)
        max_y = max(v.y for v in vertices)
        
        print(f"[DEBUG] Текущий меш: X=[{min_x:.3f}, {max_x:.3f}], Y=[{min_y:.3f}, {max_y:.3f}]")
        
        # Сбрасываем ридеры для записи
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_reader = GeomVertexReader(vdata, "normal")
        
        # Вычисляем размеры текущего меша
        size_x = max_x - min_x
        size_y = max_y - min_y
        
        # Определяем базовый масштаб UV (1 текстура на 1 метр)
        base_scale = 1.0
        
        # Вычисляем повторение текстуры на основе размеров меша
        if size_x > 0 and size_y > 0:
            # Автоматически вычисляем repeat на основе размера меша и желаемой плотности текстур
            desired_texture_density = 0.3  # 1 текстура на каждые 0.3 метра
            auto_repeat_x = size_x / desired_texture_density
            auto_repeat_y = size_y / desired_texture_density
            
            # Используем либо заданные пользователем значения, либо автоматически вычисленные
            final_repeat_x = texture_repeatX if texture_repeatX > 0 else auto_repeat_x
            final_repeat_y = texture_repeatY if texture_repeatY > 0 else auto_repeat_y
            
            print(f"[DEBUG] Автоматические repeat: X={auto_repeat_x:.2f}, Y={auto_repeat_y:.2f}")
            print(f"[DEBUG] Финальные repeat: X={final_repeat_x:.2f}, Y={final_repeat_y:.2f}")
            
            # Создаем UV-координаты
            vertex_idx = 0
            while not vertex_reader.isAtEnd():
                vertex = vertex_reader.getData3f()
                normal = normal_reader.getData3f() if not normal_reader.isAtEnd() else LVector3f(0, 0, 1)
                
                vertex_writer.addData3f(vertex)
                normal_writer.addData3f(normal)
                
                # Вычисляем UV на основе положения вершины относительно минимальных координат
                u = (vertex.x - min_x) / size_x * final_repeat_x if size_x != 0 else 0
                v = (vertex.y - min_y) / size_y * final_repeat_y if size_y != 0 else 0
                
                texcoord_writer.addData2f(u, v)
                uv_coords.append((u, v))
                vertex_idx += 1
        else:
            # Fallback: используем простые UV
            print(f"[WARNING] Нулевые размеры меша, использую простые UV")
            vertex_idx = 0
            while not vertex_reader.isAtEnd():
                vertex = vertex_reader.getData3f()
                normal = normal_reader.getData3f() if not normal_reader.isAtEnd() else LVector3f(0, 0, 1)
                
                vertex_writer.addData3f(vertex)
                normal_writer.addData3f(normal)
                
                # Простые UV
                u = vertex_idx % 10 / 10.0
                v = (vertex_idx // 10) % 10 / 10.0
                texcoord_writer.addData2f(u, v)
                uv_coords.append((u, v))
                vertex_idx += 1
        
        print(f"[DEBUG] Обработано вершин для UV: {len(vertices)}")
        if uv_coords:
            min_u = min(uv[0] for uv in uv_coords)
            max_u = max(uv[0] for uv in uv_coords)
            min_v = min(uv[1] for uv in uv_coords)
            max_v = max(uv[1] for uv in uv_coords)
            print(f"[DEBUG UV] UV диапазон после применения repeat: U=[{min_u:.3f}, {max_u:.3f}], V=[{min_v:.3f}, {max_v:.3f}]")
            print(f"[DEBUG] Пример UV для первой вершины: ({vertices[0].x:.3f}, {vertices[0].y:.3f}) -> ({uv_coords[0][0]:.3f}, {uv_coords[0][1]:.3f})")
        
        # Копируем примитивы
        new_geom = Geom(new_vdata)
        for i in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(i)
            new_geom.addPrimitive(prim)
        
        # Создаем новый узел с UV
        new_geom_node = GeomNode("boolean_mesh_with_uv")
        new_geom_node.addGeom(new_geom)
        
        # Заменяем узел в модели
        boolean_mesh_np.removeNode()
        
        # СОЗДАЕМ НОВЫЙ NodePath и прикрепляем его к render
        new_mesh_np = NodePath(new_geom_node)
        
        # Прикрепляем к сцене
        new_mesh_np.reparentTo(self.panda_app.render)
        
        print(f"[DEBUG] Создан новый NodePath с UV на основе мировых координат")
        return new_mesh_np
    
    def _apply_textures_and_material(self, model_np):
        """Применяет текстуры и материал к модели"""
        print(f"[DEBUG] Начало применения текстур к модели...")
        
        # Проверяем, что model_np не пустой
        if model_np is None or model_np.is_empty():
            print(f"[ERROR] Передан пустой NodePath для применения текстур")
            return
        
        print(f"[DEBUG] model_np.is_empty() = {model_np.is_empty()}")
        
        # Получаем набор текстур из panda_app
        if hasattr(self.panda_app, 'current_texture_set') and self.panda_app.current_texture_set:
            texture_set = self.panda_app.current_texture_set
        else:
            texture_set = {}
            print(f"[DEBUG] current_texture_set не найден, используем значения по умолчанию")
        
        # Определяем пути к текстурам
        if 'diffuse' in texture_set:
            diffuse_path = texture_set['diffuse']
        elif 'albedo' in texture_set:
            diffuse_path = texture_set['albedo']
        else:
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        
        normal_path = texture_set.get('normal', 
            "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg")
        
        roughness_path = texture_set.get('roughness', None)
        
        # Проверяем существование файлов
        if not os.path.exists(diffuse_path):
            print(f"[WARNING] Диффузная текстура не найдена: {diffuse_path}")
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        
        if not os.path.exists(normal_path):
            print(f"[WARNING] Нормальная текстура не найдена: {normal_path}")
            normal_path = "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"
        
        print(f"[DEBUG] Загрузка текстур:")
        print(f"  Диффузная: {diffuse_path}")
        print(f"  Нормальная: {normal_path}")
        if roughness_path:
            print(f"  Шероховатость: {roughness_path}")
        
        # Создаем переменные для текстурных стадий
        diffuse_stage = None
        normal_stage = None
        roughness_stage = None
        
        # Загружаем и настраиваем диффузную текстуру
        try:
            print(f"[DEBUG] Попытка загрузки диффузной текстуры...")
            diffuse_tex = self.panda_app.loader.loadTexture(diffuse_path)
            if diffuse_tex:
                diffuse_tex.set_format(Texture.F_srgb)
                diffuse_tex.setMinfilter(Texture.FTLinearMipmapLinear)
                diffuse_tex.setMagfilter(Texture.FTLinear)
                diffuse_tex.setWrapU(Texture.WMRepeat)
                diffuse_tex.setWrapV(Texture.WMRepeat)
                
                # Создаем TextureStage для диффузной текстуры
                diffuse_stage = TextureStage('diffuse')
                diffuse_stage.setMode(TextureStage.MModulate)
                model_np.setTexture(diffuse_stage, diffuse_tex)
                print(f"[DEBUG] Диффузная текстура успешно загружена и настроена")
            else:
                print(f"[WARNING] Не удалось загрузить диффузную текстуру")
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки диффузной текстуры: {e}")
        
        # Загружаем и настраиваем нормальную текстуру
        try:
            print(f"[DEBUG] Попытка загрузки нормальной текстуры...")
            normal_tex = self.panda_app.loader.loadTexture(normal_path)
            if normal_tex:
                normal_tex.setMinfilter(Texture.FTLinearMipmapLinear)
                normal_tex.setMagfilter(Texture.FTLinear)
                normal_tex.setWrapU(Texture.WMRepeat)
                normal_tex.setWrapV(Texture.WMRepeat)
                
                normal_stage = TextureStage('normal')
                normal_stage.setMode(TextureStage.MNormal)
                model_np.setTexture(normal_stage, normal_tex)
                print(f"[DEBUG] Нормальная текстура успешно загружена")
            else:
                print(f"[WARNING] Не удалось загрузить нормальную текстуру")
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки нормальной текстуры: {e}")
        
        # Загружаем и настраиваем текстуру шероховатости (если есть)
        if roughness_path and os.path.exists(roughness_path):
            try:
                print(f"[DEBUG] Попытка загрузки текстуры шероховатости...")
                roughness_tex = self.panda_app.loader.loadTexture(roughness_path)
                if roughness_tex:
                    roughness_tex.setMinfilter(Texture.FTLinearMipmapLinear)
                    roughness_tex.setMagfilter(Texture.FTLinear)
                    roughness_tex.setWrapU(Texture.WMRepeat)
                    roughness_tex.setWrapV(Texture.WMRepeat)
                    
                    roughness_stage = TextureStage('roughness')
                    roughness_stage.setMode(TextureStage.MModulate)
                    model_np.setTexture(roughness_stage, roughness_tex)
                    print(f"[DEBUG] Текстура шероховатости успешно загружена")
                else:
                    print(f"[WARNING] Не удалось загрузить текстуру шероховатости")
            except Exception as e:
                print(f"[ERROR] Ошибка загрузки текстуры шероховатости: {e}")
        
        # Создаем и настраиваем материал
        base_material = Material("mesh_reconstruction_material")
        base_material.setDiffuse((0.8, 0.8, 0.8, 1.0))
        base_material.setAmbient((0.5, 0.5, 0.5, 1.0))
        base_material.setSpecular((0.3, 0.3, 0.3, 1.0))
        base_material.setShininess(20.0)
        base_material.setRoughness(0.7)
        base_material.setMetallic(0.1)
        base_material.setRefractiveIndex(1.5)
        model_np.setMaterial(base_material, 1)
        
        # Включаем шейдеры и настраиваем рендеринг
        model_np.setShaderAuto()
        model_np.setTwoSided(True)
        model_np.setBin("fixed", 0)
        model_np.setDepthOffset(1)
        
        # Включаем освещение для материала
        model_np.setLightOff()
        model_np.setRenderModeFilled()
        
        print(f"[DEBUG] Текстуры и материал успешно применены к модели")
    
    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            print(f"[ERROR] JSON-файл не выбран или не существует")
            return
        
        print(f"[DEBUG] ====== НАЧАЛО РЕКОНСТРУКЦИИ 2D->3D ======")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        source_height_path = data['source_height']
        self.image_path = source_height_path

        # Калибруем глубины по JSON
        self.calibrate_depth_from_json(data)
        
        # Настройка displacement параметров
        self.set_displacement_parameters(
            texture_path="textures/stones_8k/rocks_ground_01_disp_4k.jpg",
            strength=1.14,
            repeatX=1.35,
            repeatY=2.8,
            enabled=True
        )
        
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

        z_offset = data['z_offset']
        mesh_node.setZ(z_offset)

        target_model = None
        for model in self.panda_app.loaded_models:
            model_id = id(model)
            if model_id in self.panda_app.model_paths:
                if self.panda_app.Target_Napolnitel in self.panda_app.model_paths[model_id]:
                    target_model = model
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
        
        mesh_node_trimesh = self.panda_app.panda_to_trimesh(mesh_node)
        
        mesh_node_result_trimesh = trimesh.boolean.intersection(
            [target_model_trimesh, mesh_node_trimesh],
            engine='blender'
        )
        
        final_mesh_node = self.panda_app.trimesh_to_panda(mesh_node_result_trimesh)
        
        # ВАЖНО: После boolean операции нужно правильно установить UV-координаты
        print(f"[DEBUG] Настройка UV-координат после boolean операции...")
        final_mesh_node = self._setup_uv_coordinates_after_boolean(final_mesh_node, mesh_node)
        
        if final_mesh_node is None or final_mesh_node.is_empty():
            print(f"[ERROR] Не удалось создать меш с UV-координатами")
            return
        
        print(f"[DEBUG] final_mesh_node после настройки UV: is_empty={final_mesh_node.is_empty()}")
        
        # Применяем текстуры и материал
        print(f"[DEBUG] Применение текстур к финальному мешу...")
        self._apply_textures_and_material(final_mesh_node)
        # final_mesh_node.setPos(0, 0, 4)
        
        mesh_node.removeNode()
        
        print(f"[DEBUG] ====== РЕКОНСТРУКЦИЯ 2D->3D ЗАВЕРШЕНА ======")
    
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

    def set_displacement_parameters(self, texture_path=None, strength=None, repeatX=None, repeatY=None, enabled=None):
        """Установка параметров displacement"""
        if texture_path is not None:
            self.displacement_texture_path = texture_path
        if strength is not None:
            self.displacement_strength = strength
        if repeatX is not None:
            self.texture_repeatX = repeatX
        if repeatY is not None:
            self.texture_repeatY = repeatY
        if enabled is not None:
            self.use_displacement = enabled
        
        print(f"[DEBUG] Displacement параметры обновлены:")
        print(f"  texture: {self.displacement_texture_path}")
        print(f"  strength: {self.displacement_strength}")
        print(f"  repeat: ({self.texture_repeatX}, {self.texture_repeatY})")
        print(f"  enabled: {self.use_displacement}")

    def set_displacement_enabled(self, enabled):
        """Включение/выключение displacement"""
        self.use_displacement = enabled
        print(f"[DEBUG] Displacement enabled: {enabled}")

    