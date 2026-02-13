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
import point_cloud_utils as pcu

import cProfile
import pstats
import io

class MeshReconstruction:
    def __init__(self, panda_app):
        self.panda_app = panda_app
        self.recon_json_path = ""

        # for testing only
        self.alpha_threshold = 0.5

        self.adaptive_lift_enabled = True
        self.lift_intensity = 40.0
        self.lift_smoothing_sigma = 40.0
        self.lift_blur_radius = 40
        self.boundary_zone_width = 40
        self.smoothing_iterations = 4

        # Параметры экстраполяции
        self.extrapolation_enabled = False
        self.target_width = 15.0  # Ширина целевой области в метрах
        self.target_height = 10.0  # Высота целевой области в метрах
        self.grid_resolution = 812  # Разрешение сетки экстраполяции
        
        # Параметры шума Перлина
        self.noise_scale = 0.5
        self.noise_strength = 0.15
        self.noise_octaves = 4
        self.noise_persistence = 0.5
        self.noise_lacunarity = 2.0
        self.noise_seed = random.randint(0, 10000)
        
        # Параметры displacement map
        self.displacement_texture_path = "textures/sand_8k/dune_height.jpg"
        self.displacement_strength = 0.15
        self.texture_repeatX = 1.75
        self.texture_repeatY = 2.4
        self.use_displacement = True
        
        # Для хранения мешей
        self.source_mesh_node = None
        self.mesh_node = None
        
        print(f"[DEBUG] Инициализирован MeshReconstruction")

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
        # Pack points into arrays for easier indexing
        tri1 = np.array([p1A, p1B, p1C])  # shape (3, 3)
        tri2 = np.array([p2A, p2B, p2C])  # shape (3, 3)

        # Find which vertex in tri1 is the right-angle vertex
        right_angle_idx = self._find_right_angle_vertex(tri1)

        # Reorder both triangles so that the right-angle vertex is at index 0
        if right_angle_idx != 0:
            # Swap index 0 with right_angle_idx in both triangles
            tri1[[0, right_angle_idx]] = tri1[[right_angle_idx, 0]]
            tri2[[0, right_angle_idx]] = tri2[[right_angle_idx, 0]]

        # Proceed with transform computation
        world_from_local1 = self.build_local_to_world_matrix_np(tri1[0], tri1[1], tri1[2])
        world_from_local2 = self.build_local_to_world_matrix_np(tri2[0], tri2[1], tri2[2])
        return world_from_local1 @ np.linalg.inv(world_from_local2)


    def _find_right_angle_vertex(self, tri):
        """
        Given a triangle with 3 points (3x3 array), find the index of the vertex
        that is (approximately) the right angle by checking which side is longest
        (opposite the right angle).
        
        Returns:
            int: index (0, 1, or 2) of the right-angle vertex.
        """
        # Compute squared distances between all pairs
        d01 = np.sum((tri[0] - tri[1])**2)
        d02 = np.sum((tri[0] - tri[2])**2)
        d12 = np.sum((tri[1] - tri[2])**2)

        # The longest side is opposite the right angle
        if d12 >= d02 and d12 >= d01:
            return 0
        elif d02 >= d12 and d02 >= d01:
            return 1
        else:
            return 2

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
        distance
    ):
        lens = self.panda_app.cam.node().getLens()
        nearPoint = Point3()   
        farPoint = Point3()
        film_point = Point2((u - 0.5) * 2, (0.5 - v) * 2)
        lens.extrude(film_point, nearPoint, farPoint)     
        fromvec = Vec3(farPoint - nearPoint)   
        fromvec.normalize()

        # mult = 1 / fromvec[1]
        # for i in range(3):
        #     fromvec[i] = fromvec[i] * mult

        dir_world = self.panda_app.render.get_relative_vector(camera, fromvec)

        cam_pos = camera.get_pos()
        world_point = cam_pos + dir_world * distance

        return world_point
    

    def correct_depth(self, depth):
        return depth
        return math.sqrt(depth) 

    
    def map_value(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def setVertFOV(self, lens, vfov_deg):
        # Convert VFOV → HFOV
        hfov_rad = 2.0 * math.atan(math.tan(math.radians(vfov_deg) / 2.0) * self.aspect_ratio)
        hfov_deg = math.degrees(hfov_rad)

        self.fov_y = vfov_deg

        lens.setFov(hfov_deg, vfov_deg)

    def apply_transform(self, M, camera):
        cam_pos = self.panda_vec3_to_np(camera.get_pos(camera.get_parent()))
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

    def resolve_keypoints(self, scene_3d, points_2d, camera, min_depth, max_depth, debug):
        proj_3d = []

        for i in range(3):
            u = points_2d[i][0]
            v = points_2d[i][1]
            
            x = points_2d[i][2]
            y = points_2d[i][3]
            
            depth = self.correct_depth(self.height_map[y, x])
            z = depth * (max_depth - min_depth) + min_depth

            point = self.viewport_to_world_point(camera, u, v, z)
            proj_3d.append(self.panda_vec3_to_np(point))

        M = self.compute_transform_np(
            scene_3d[0], scene_3d[1], scene_3d[2],
            proj_3d[0], proj_3d[1], proj_3d[2]
        )

        self.apply_transform(M, camera)

        error = 0

        for i in range(self.matching_points):
            u = points_2d[i][0]
            v = points_2d[i][1]

            x = points_2d[i][2]
            y = points_2d[i][3]
            
            depth = self.correct_depth(self.height_map[y, x])
            z = depth * (max_depth - min_depth) + min_depth

            point = self.viewport_to_world_point(camera, u, v, z)
            diff = point - scene_3d[i]
            distance = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if(debug):
                self.add_cube_at(point)
            error += distance

        return error

    def lerp(self, a, b, t):
        return a + t * (b - a)
    
    def add_cube_at(self, pos, color=(1, 0, 0, 1), scale=0.10):
        box = self.panda_app.loader.load_model("models/smiley")
        box.set_pos(pos)
        box.set_scale(scale)
        box.reparent_to(self.panda_app.render)
        return box

    def cv_to_panda(self, p):
        """
        Convert point from OpenCV-style camera coordinates
        (X right, Y down, Z forward)
        to Panda3D camera coordinates
        (X right, Y forward, Z up)
        """
        x, y, z = p
        return LVector3f(
            x,   # right -> right
            z,   # forward -> forward
            -y   # up (inverted Y) -> up
        )


    def reconstruct_camera_pos_hpr_fov_depth(self, data):
        camera = self.panda_app.camera

        lens = self.panda_app.cam.node().getLens()

        self.aspect_ratio = 16/9
        
        lerpT = 1.0 / 2

        iterations = 24

        scene_3d = [np.array(p, dtype=float) for p in data["points_3d"]]
        self.scene_3d = scene_3d

        if "keypoints_3d" in data:
            print("Using 3d keypoints")

            min_scale = 0.0001
            max_scale = 1000

            best_scale_error = 10**10
            best_scale = 1

            find_scale = False

            #self.setVertFOV(lens, 50.6)

            keypoints_3d = [np.array(p, dtype=float) for p in data["keypoints_3d"]]

            for i in range(len(keypoints_3d)):
                keypoints_3d[i] = self.cv_to_panda(self.np_to_panda_point(keypoints_3d[i]))

            def ApplyScale(scale):
                #camera.set_pos(self.np_to_panda_point(cam_pos))
                #camera.set_quat(quat)

                transformed_keypoints = keypoints_3d.copy()
                for i in range(len(keypoints_3d)):
                    #transformed_keypoints[i] = self.panda_app.render.get_relative_point(camera, keypoints_3d[i] * scale)
                    transformed_keypoints[i] = keypoints_3d[i] * scale

                M = self.compute_transform_np(
                    scene_3d[0], scene_3d[1], scene_3d[2],
                    transformed_keypoints[0], transformed_keypoints[1], transformed_keypoints[2]
                )

                self.trs_matrix = M
                
                error = 0
                for i in range(len(transformed_keypoints)):
                    op = self.trs_matrix @ np.append(transformed_keypoints[i], 1.0)
                    sp = np.append(scene_3d[i], 1.0)
                    diff = self.np_to_panda_point(op - sp)
                    error += math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
                return error

                #self.apply_transform(M, camera)

                error = 0
                for i in range(len(transformed_keypoints)):
                    op = self.panda_app.render.get_relative_point(camera, keypoints_3d[i] * scale)
                    sp = scene_3d[i]
                    diff = self.np_to_panda_point(op - sp)
                    error += math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
                return error

            if(find_scale):
                scaleStart = min_scale
                scaleEnd = max_scale

                for i in range(iterations):
                    scaleStep = (scaleEnd - scaleStart) / 2
                    
                    if scaleStep == 0:
                        break

                    scale = scaleStart - scaleStep
                    while scale < scaleEnd:
                        scale += scaleStep
                        error = ApplyScale(scale)
                        if error < best_scale_error:
                            best_scale_error = error
                            best_scale = scale

                    scaleStart = self.lerp(best_scale, scaleStart, lerpT)
                    scaleEnd = self.lerp(best_scale, scaleEnd, lerpT)

            best_scale_error = ApplyScale(best_scale)
            print(f"Best error: {best_scale_error} at {best_scale}")

            trs_points = [self.cv_to_panda(self.np_to_panda_point(v)) for v in self.point_cloud]
            trs_points = [self.trs_matrix @ np.append(self.panda_vec3_to_np(v), 1.0) for v in trs_points]
            self.trs_points = trs_points

            return
        else:
            print("WARNING: No 3D keypoints! behaviour is undefined, results may be inaccurate. Please provide 3d keypoints for better reconstruction.")

        img_w = data["metadata"]["image_size"]["width"]
        img_h = data["metadata"]["image_size"]["height"]

        self.aspect_ratio = img_w/img_h

        key_points = data["keypoints"]
            
        self.fov_y = lens.getFov()[1]

        points_2d = []

        for i in range(len(key_points)):
            x = key_points[i]["x"]
            y = key_points[i]["y"]

            u = x / img_w
            v = y / img_h

            points_2d.append([u, v, x, y])

        # 99-55 для стационарного решения
        # 0-41 для блендера
        fov_known = True
        known_fov_x = 0
        known_fov_y = 41.1

        self.matching_points = 4

        min_fov = 15
        max_fov = 130
        fov_acc = 1.0

        start_min_depth = 1
        start_max_depth = 200

        bestError = 10**10

        bestMinDepth = 0
        bestMaxDepth = 0
        bestFOV = 0

        render_depth_known = True

        # из блендера
        render_min_depth = 3
        render_max_depth = 6
        render_camera_offset = 5.35774
        render_mesh_offset = 5.4816

        render_camera_offset -= render_mesh_offset

        render_min_depth += render_camera_offset
        render_max_depth += render_camera_offset

        ratio = render_min_depth / render_max_depth

        fov = min_fov - fov_acc
        while fov < max_fov:
            fov += fov_acc
            self.fov_y = fov

            if fov_known:
                if(known_fov_x != 0):
                    lens.setFov(known_fov_x, known_fov_y)
                    self.fov_y = known_fov_y
                else:
                    self.setVertFOV(lens, known_fov_y)
            else:
                self.setVertFOV(lens, fov)

            scaleStart = start_min_depth
            scaleEnd = start_max_depth
            localMaxDepthStart = start_min_depth
            localMaxDepthEnd = start_max_depth

            localBest = 10**10

            for i in range(iterations):
                scaleStep = (localMaxDepthEnd - localMaxDepthStart) / 2
                maxDepthStep = (localMaxDepthEnd - localMaxDepthStart) / 2

                if scaleStep == 0 or maxDepthStep == 0:
                    break
                
                scale = localMaxDepthStart - maxDepthStep
                while scale < localMaxDepthEnd:
                    scale += maxDepthStep
                    min_depth = scaleStart - scaleStep
                    while min_depth < scaleEnd:
                        min_depth += scaleStep

                        if(render_depth_known):
                            min_depth = scale * ratio

                        try:
                            error = self.resolve_keypoints(scene_3d, points_2d, camera, min_depth, scale, False)
                            if(error < localBest):
                                localBest = error
                            if(error < bestError):
                                bestError = error
                                bestFOV = fov
                                bestMinDepth = min_depth
                                bestMaxDepth = scale
                        except:
                            continue

                        if(render_depth_known):
                            break
                
                scaleStart = self.lerp(bestMinDepth, scaleStart, lerpT)
                scaleEnd = self.lerp(bestMinDepth, scaleEnd, lerpT)
                localMaxDepthStart = self.lerp(bestMaxDepth, localMaxDepthStart, lerpT)
                localMaxDepthEnd = self.lerp(bestMaxDepth, localMaxDepthEnd, lerpT)
            
            if fov_known:
                break

            print(f"local best for FOV {fov} is {localBest}")


        self.min_depth = bestMinDepth
        self.max_depth = bestMaxDepth + (bestMaxDepth - bestMinDepth) * 0.3 # чтобы насыпь сходилась с дном, некий магический коэффицент коррекции

        fov = bestFOV
        self.fov_y = fov

        if(known_fov_x != 0):
            lens.setFov(known_fov_x, known_fov_y)
            self.fov_y = known_fov_y
        else:
            self.setVertFOV(lens, known_fov_y)
    
        print(f"camera keypoints final error: {bestError} FOV: {self.fov_y} minDepth: {bestMinDepth} maxDepth: {bestMaxDepth}")
        
        self.resolve_keypoints(scene_3d, points_2d, camera, bestMinDepth, bestMaxDepth, False)

    def load_height_map(self):
        try:
            print(f"[DEBUG] Начало загрузки height map: {self.heightmap_path}")
            img = Image.open(self.heightmap_path)
            print(f"[DEBUG] Изображение загружено: {img.size}, mode: {img.mode}")
            
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
            
            # Закрываем изображение для освобождения памяти
            img.close()
            print(f"[DEBUG] Загрузка height map завершена успешно")
            return True
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки height map: {e}")
            return False
    
    def _load_displacement_texture(self, texture_path):
        """Загружает текстуру высот для displacement map (аналогично perlin_mesh_generator.py)"""
        try:
            print(f"[DEBUG] Загрузка displacement texture: {texture_path}")
            if not os.path.exists(texture_path):
                # Пробуем найти текстуру в стандартных путях
                if 'displacement' in self.panda_app.current_texture_set:
                    texture_path = self.panda_app.current_texture_set['displacement']
                elif 'height' in self.panda_app.current_texture_set:
                    texture_path = self.panda_app.current_texture_set['height']
                else:
                    texture_path = "textures/sand_8k/dune_height.jpg"
                    
                print(f"[DEBUG] Используем текстуру по умолчанию: {texture_path}")
            
            height_image = Image.open(texture_path).convert('L')
            height_array = np.array(height_image, dtype=np.float32)
            tex_height, tex_width = height_array.shape
            
            # Нормализация и предобработка как в perlin_mesh_generator.py
            height_min = np.min(height_array)
            height_max = np.max(height_array)
            
            if height_max > height_min:
                height_array = (height_array - height_min) / (height_max - height_min)
            else:
                height_array = np.zeros_like(height_array)
            
            # Применяем гамма-коррекцию как в оригинале
            height_array = np.power(height_array, 0.7)
            
            height_image.close()
            print(f"[DEBUG] Displacement texture загружена: {tex_width}x{tex_height}")
            
            return height_array, tex_width, tex_height
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки displacement texture: {e}")
            # Возвращаем пустой массив
            return np.zeros((1, 1), dtype=np.float32), 1, 1
    
    def _apply_displacement_to_grid(self, height_grid, grid_res, plane_center, target_width, target_height):
        """Применяет displacement map к сетке высот (аналогично perlin_mesh_generator.py)"""
        if not self.use_displacement:
            print("[DEBUG] Displacement отключен")
            return height_grid
        
        print("[DEBUG] Применение displacement map к сетке...")
        
        # Загружаем текстуру высот
        height_array, tex_width, tex_height = self._load_displacement_texture(self.displacement_texture_path)
        
        # Создаем сетку координат для вершин
        x_vals = np.linspace(-target_width/2, target_width/2, grid_res)
        y_vals = np.linspace(-target_height/2, target_height/2, grid_res)
        
        # Применяем displacement к каждой вершине сетки
        for i in range(grid_res):
            for j in range(grid_res):
                # Вычисляем UV-координаты с учетом повторения текстуры
                normalized_u = j / (grid_res - 1) if grid_res > 1 else 0.0
                normalized_v = i / (grid_res - 1) if grid_res > 1 else 0.0
                
                u = normalized_u * self.texture_repeatX
                v = normalized_v * self.texture_repeatY
                
                # Обрабатываем повторение текстуры
                u_repeated = u % 1.0
                v_repeated = v % 1.0
                
                # Преобразуем UV в координаты текстуры
                tex_x = u_repeated * (tex_width - 1)
                tex_y = v_repeated * (tex_height - 1)
                
                # Билинейная интерполяция (как в perlin_mesh_generator.py)
                x1 = max(0, min(tex_width - 1, int(tex_x)))
                y1 = max(0, min(tex_height - 1, int(tex_y)))
                x2 = max(0, min(tex_width - 1, x1 + 1))
                y2 = max(0, min(tex_height - 1, y1 + 1))
                
                dx = tex_x - x1
                dy = tex_y - y1
                
                # Значения в четырех соседних точках
                h11 = height_array[y1, x1]
                h12 = height_array[y2, x1]
                h21 = height_array[y1, x2]
                h22 = height_array[y2, x2]
                
                # Интерполяция
                hx1 = h11 * (1 - dx) + h21 * dx
                hx2 = h12 * (1 - dx) + h22 * dx
                height_value = hx1 * (1 - dy) + hx2 * dy
                
                # Применяем displacement
                displacement = (height_value - 0.5) * self.displacement_strength
                height_grid[i, j] += displacement
        
        print(f"[DEBUG] Displacement map применен к {grid_res}x{grid_res} вершинам")
        return height_grid

    def create_unified_perlin_mesh_with_lift(self):
        print(f"[DEBUG] Загрузка height map...")
        if not self.load_height_map():
            print(f"[ERROR] Не удалось загрузить height map")
            return None

        h, w = self.height_map.shape
        
        # ВОЗВРАЩАЕМ ГАУССОВО СГЛАЖИВАНИЕ
        print("[DEBUG] Применение гауссова сглаживания...")
        from scipy.ndimage import gaussian_filter
        import numpy as np

        masked_height = self.height_map.copy()
        masked_height[~self.mask] = 0

        blurred_masked = gaussian_filter(masked_height, sigma=20)
        blurred_mask = gaussian_filter(self.mask.astype(float), sigma=20)

        blurred_height_normalized = np.divide(
            blurred_masked, 
            blurred_mask + 1e-10,
            where=blurred_mask > 0.01
        )

        self.height_map = np.where(self.mask, blurred_height_normalized, self.height_map)
        
        camera = self.panda_app.camera
        
        # 1. СОЗДАЕМ ПЛОСКУЮ СЕТКУ 6x10 МЕТРОВ
        print("[DEBUG] Создание плоской сетки 6x10 метров...")
        
        # Размеры целевой области
        target_width = 6.0
        target_height = 10.0
        
        # Разрешение сетки
        grid_res = self.grid_resolution
        
        # Создаем сетку вершин
        x_vals = np.linspace(-target_width/2, target_width/2, grid_res)
        y_vals = np.linspace(-target_height/2, target_height/2, grid_res)
        
        # Определяем центр плоской сетки (центр исходного меша или камеры)
        cam_pos = self.panda_vec3_to_np(camera.get_pos())
        plane_center = np.array([cam_pos[0], cam_pos[1], 0])  # Z=0 для начала
        
        # 2. СОЗДАЕМ ВЕРШИНЫ ИСХОДНОГО МЕША И ИХ 2D ПРОЕКЦИИ
        print("[DEBUG] Создание вершин исходного меша для деформации плоскости...")
        
        # Собираем вершины исходного меша с прореживанием для производительности
        original_vertices_3d = []  # 3D координаты
        original_vertices_2d = []  # 2D проекции (X, Y)
        original_heights = []      # Z-координаты
        
        # Прореживание для ускорения
        step = max(1, int(np.sqrt(h * w / 50000)))  # ~50k точек
        print(f"[DEBUG] Шаг прореживания исходного меша: {step}")
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                if not self.mask[y, x]:
                    continue
                    
                # Вычисляем 3D координаты
                z = self.correct_depth(self.height_map[y, x])
                distance = z * (self.max_depth - self.min_depth) + self.min_depth
                u = x / w
                v = y / h
                point_3d = self.viewport_to_world_point_geometric(camera, u, v, distance)
                np_point = self.panda_vec3_to_np(point_3d)
                
                original_vertices_3d.append(np_point)
                original_vertices_2d.append(np_point[:2])  # Только X, Y
                original_heights.append(np_point[2])
        
        print(f"[DEBUG] Собрано {len(original_vertices_3d)} вершин исходного меша")
        
        if len(original_vertices_3d) == 0:
            print("[ERROR] Нет вершин исходного меша")
            return None
        
        # 3. СОЗДАЕМ KD-ДЕРЕВО ДЛЯ БЫСТРОГО ПОИСКА БЛИЖАЙШИХ ВЕРШИН
        print("[DEBUG] Создание KD-дерева для вершин исходного меша...")
        original_tree = KDTree(original_vertices_2d)
        
        # 4. СОЗДАЕМ СЕТКУ ВЫСОТ ДЛЯ ПЛОСКОСТИ
        print("[DEBUG] Создание сетки высот...")
        height_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        source_mask = np.zeros((grid_res, grid_res), dtype=bool)
        
        # Заполняем высоты исходного меша на сетке
        for idx, (x_2d, z) in enumerate(zip(original_vertices_2d, original_heights)):
            # Находим ближайшую точку на сетке
            x_idx = np.argmin(np.abs(x_vals - (x_2d[0] - plane_center[0])))
            y_idx = np.argmin(np.abs(y_vals - (x_2d[1] - plane_center[1])))
            
            if 0 <= x_idx < grid_res and 0 <= y_idx < grid_res:
                height_grid[y_idx, x_idx] = z
                source_mask[y_idx, x_idx] = True
        
        # 5. НАХОДИМ ГРАНИЧНЫЕ ТОЧКИ ИСХОДНОГО МЕША
        print("[DEBUG] Поиск граничных точек исходного меша...")
        boundary_points = []
        
        for i in range(grid_res):
            for j in range(grid_res):
                if source_mask[i, j]:
                    # Проверяем соседей
                    is_boundary = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_res and 0 <= nj < grid_res:
                                if not source_mask[ni, nj]:
                                    is_boundary = True
                                    break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        x = plane_center[0] + x_vals[j]
                        y = plane_center[1] + y_vals[i]
                        z = height_grid[i, j]
                        boundary_points.append((x, y, z))
        
        print(f"[DEBUG] Найдено {len(boundary_points)} граничных точек")
        
        # 6. ПРИМЕНЯЕМ АДАПТИВНЫЙ ПОДЪЕМ ВЕРШИН
        if len(boundary_points) > 0:
            print("[DEBUG] Применение адаптивного подъема вершин...")
            lifted_mask = np.zeros((grid_res, grid_res), dtype=bool)
            
            # Создаем KD-дерево для граничных точек
            boundary_array = np.array(boundary_points)
            boundary_tree = KDTree(boundary_array[:, :2])
            
            # Параметры адаптивного подъема
            base_distance = 0.5
            min_distance = 0.1
            max_distance = 3.0
            lift_intensity = 1.0
            
            # Вычисляем диапазон высот граничных точек
            boundary_heights = boundary_array[:, 2]
            min_boundary_height = np.min(boundary_heights)
            max_boundary_height = np.max(boundary_heights)
            height_range = max_boundary_height - min_boundary_height
            
            # Адаптивный подъем для каждой точки сетки
            for i in range(grid_res):
                for j in range(grid_res):
                    if source_mask[i, j]:
                        continue
                    
                    x = plane_center[0] + x_vals[j]
                    y = plane_center[1] + y_vals[i]
                    current_z = 0  # Базовая высота плоскости
                    
                    # Находим ближайшую граничную точку
                    distances, indices = boundary_tree.query([x, y], k=1)
                    nearest_idx = indices
                    nearest_dist = distances
                    
                    if nearest_dist < max_distance:
                        nearest_point = boundary_array[nearest_idx]
                        height_diff = nearest_point[2] - current_z
                        
                        if height_diff > 0:
                            # Адаптивное расстояние в зависимости от разницы высот
                            if height_range > 0:
                                normalized_height_diff = min(height_diff / height_range, 1.0)
                            else:
                                normalized_height_diff = 0.5
                            
                            # Динамическое расстояние
                            if normalized_height_diff < 0.1:
                                dynamic_distance = min_distance * 1.2
                            elif normalized_height_diff < 0.3:
                                dynamic_distance = base_distance * (1.0 + normalized_height_diff)
                            else:
                                dynamic_distance = base_distance * (1.0 + normalized_height_diff * 3.0)
                            
                            dynamic_distance = max(min_distance, min(dynamic_distance, max_distance))
                            
                            if nearest_dist <= dynamic_distance:
                                normalized_distance = nearest_dist / dynamic_distance
                                # Функция smoothstep для плавности
                                t = normalized_distance
                                smooth_factor = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
                                
                                lift_amount = height_diff * smooth_factor * lift_intensity
                                height_grid[i, j] = current_z + lift_amount
                                lifted_mask[i, j] = True
            
            print(f"[DEBUG] Поднято {np.sum(lifted_mask)} вершин")
            
            # 7. СГЛАЖИВАНИЕ ПОДНЯТОЙ ОБЛАСТИ
            if np.any(lifted_mask):
                print("[DEBUG] Сглаживание поднятой области...")
                height_grid = self._smooth_lifted_area(height_grid, lifted_mask)
            
            # 8. РАЗМЫТИЕ ГРАНИЦ ПЕРЕХОДА
            print("[DEBUG] Размытие границ перехода...")
            height_grid = self._blur_lift_boundary(height_grid, source_mask, boundary_points)
        
        # 9. ПОСТОБРАБОТКА ГРАНИЧНОЙ ЗОНЫ
        print("[DEBUG] Постобработка граничной зоны...")
        height_grid = self._postprocess_boundary_zone(height_grid, source_mask)
        
        # 10. ОБЩЕЕ СГЛАЖИВАНИЕ СЕТКИ
        print("[DEBUG] Общее сглаживание сетки...")
        height_grid = self._smooth_heightfield(height_grid)
        
        # 11. ПРИМЕНЕНИЕ DISPLACEMENT MAP (после всех сглаживаний)
        print("[DEBUG] Применение displacement map...")
        height_grid = self._apply_displacement_to_grid(height_grid, grid_res, plane_center, target_width, target_height)
        
        # 12. СОЗДАЕМ ВЕРШИНЫ ЕДИНОГО МЕША
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_extended_mesh", format, Geom.UHStatic)
        
        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")
        
        # Создаем массив для хранения индексов вершин в сетке
        vertex_indices = np.zeros((grid_res, grid_res), dtype=int)
        total_vertex_count = 0
        
        # Заполняем вершины с учетом обработанных высот
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                world_x = plane_center[0] + x
                world_y = plane_center[1] + y
                z = height_grid[i, j]
                
                vertex_writer.addData3f(world_x, world_y, z)
                normal_writer.addData3f(0, 0, 1)  # Временная нормаль
                texcoord_writer.addData2f(j/(grid_res-1), i/(grid_res-1))
                
                vertex_indices[i, j] = total_vertex_count
                total_vertex_count += 1
        
        print(f"[DEBUG] Создано {total_vertex_count} вершин деформированной плоскости")
        
        # 13. СОЗДАЕМ ТРЕУГОЛЬНИКИ ДЛЯ СЕТКИ
        print("[DEBUG] Создание треугольников для сетки...")
        triangles = GeomTriangles(Geom.UHStatic)
        
        triangle_count = 0
        for i in range(grid_res - 1):
            for j in range(grid_res - 1):
                v00 = vertex_indices[i, j]
                v10 = vertex_indices[i + 1, j]
                v01 = vertex_indices[i, j + 1]
                v11 = vertex_indices[i + 1, j + 1]
                
                # Создаем два треугольника для каждого квадрата сетки
                triangles.addVertices(v00, v10, v01)
                triangles.addVertices(v01, v10, v11)
                triangle_count += 2
        
        triangles.closePrimitive()
        print(f"[DEBUG] Создано {triangle_count} треугольников")
        
        # 14. СОЗДАЕМ ГЕОМЕТРИЮ И ПЕРЕСЧИТЫВАЕМ НОРМАЛИ
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        # Используем оригинальный метод расчета нормалей
        self._calculate_normals(vdata, geom)
        
        # 15. СОЗДАЕМ УЗЕЛ
        node = GeomNode("unified_extended_mesh")
        node.addGeom(geom)
        
        print(f"[DEBUG] Единый меш создан успешно")
        print(f"[DEBUG] Всего вершин: {total_vertex_count}")
        print(f"[DEBUG] Всего треугольников: {triangle_count}")
        
        return node
    
    # ---- optimized mesh builder (calls _calculate_normals) ----
    def create_mesh_from_point_cloud_smoothed(self, size = 128, n_samples = 2000):
        h = size
        w = size

        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", fmt, Geom.UHStatic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")

        # Preallocate writers' row growth (optional)
        # vdata.setNumRows(h * w)  # sometimes useful, but depends on Panda3D version
        
        source_points_2d = np.array([[v[0], v[1]] for v in self.trs_points])
        kd_tree = KDTree(source_points_2d)

        min_x = 10**10
        min_y = 10**10
        max_x = -min_x
        max_y = -min_y

        for p in self.scene_3d:
            min_x = min(min_x, p[0])
            max_x = max(max_x, p[0])
            min_y = min(min_y, p[1])
            max_y = max(max_y, p[1])

        x_range = max_x - min_x
        y_range = max_y - min_y

        w = int(w * x_range)
        h = int(h * y_range)

        print(x_range, y_range)
        print(w, h)

        inv_w = 1.0 / w
        inv_h = 1.0 / h

        source_heights = np.array([v[2] for v in self.trs_points])

        # VERTICES
        for y in range(h):
            v = y * inv_h
            for x in range(w):
                u = x * inv_w

                sx = min_x + u * x_range
                sy = min_y + v * y_range

                max_dist = 0.1

                distances, indices = kd_tree.query([[sx, sy]], k=n_samples)

                idx_list = indices[0]

                z_sum = 0
                w_sum = 0

                dst = distances[0]

                #max_dist = max(dst)

                for i in range(n_samples):
                    if(dst[i] > max_dist): continue

                    id = idx_list[i]

                    z_l = source_heights[id]

                    dist = max_dist - dst[i]

                    z_sum += z_l * dist
                    w_sum += dist

                if(w_sum > 0):
                    z = z_sum / w_sum
                else:
                    z = 0

                point = LPoint3f(sx, sy, z)

                vertex.addData3f(point)
                normal.addData3f(0.0, 0.0, 1.0)   # placeholder
                texcoord.addData2f(u, v)

        # TRIANGLES (index math: idx = y*w + x)
        tris = GeomTriangles(Geom.UHStatic)
        for y in range(h - 1):
            row = y * w
            next_row = (y + 1) * w
            for x in range(w - 1):
                # early mask check for shared edges
                #if not mask[y + 1, x] or not mask[y, x + 1]:
                #    continue

                v1 = row + x
                v2 = row + x + 1
                v3 = next_row + x
                v4 = next_row + x + 1

                #if mask[y, x]:
                tris.addVertices(v1, v3, v2)

                #if mask[y + 1, x + 1]:
                tris.addVertices(v2, v3, v4)

        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        # Compute normals using the Python routine (fast, in-place)
        self._calculate_normals(vdata, geom)

        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        return node
    
    def create_mesh_from_point_cloud(self, size=512, n_samples=3):
        """
        Создаёт единый меш с плоской базой и адаптивным подъёмом вершин,
        используя точки из self.trs_points в качестве источника высот.
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from scipy.spatial import KDTree
        from panda3d.core import GeomVertexFormat, GeomVertexData, GeomVertexWriter
        #from panda3d.core import GeomTriangles, Geom, GeomNode, GeomUHSprite, GeomEnums

        # ------------------------------------------------------------
        # 1. Исходные точки – self.trs_points
        # ------------------------------------------------------------
        if not hasattr(self, 'trs_points') or len(self.trs_points) == 0:
            print("[ERROR] Нет точек для построения меша (self.trs_points пуст)")
            return None

        source_points = np.array(self.trs_points, dtype=np.float32)  # (N, 3)
        print(f"[DEBUG] Загружено {len(source_points)} точек из self.trs_points")

        # ------------------------------------------------------------
        # 2. Границы облака и параметры плоской сетки
        # ------------------------------------------------------------
        min_x, min_y = np.min(source_points[:, :2], axis=0)
        max_x, max_y = np.max(source_points[:, :2], axis=0)

        # Размеры целевой области (можно переопределить через атрибуты)
        if hasattr(self, 'target_width') and self.target_width > 0:
            target_width = self.target_width
        else:
            target_width = max(max_x - min_x, 1.0) * 1.2   # запас 20%

        if hasattr(self, 'target_height') and self.target_height > 0:
            target_height = self.target_height
        else:
            target_height = max(max_y - min_y, 1.0) * 1.2

        # Разрешение сетки
        grid_res = getattr(self, 'grid_resolution', size)

        # Центр плоской сетки – центр ограничивающего прямоугольника, Z = 0
        plane_center = np.array([(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, 0.0])

        # Координаты узлов сетки в локальной системе (относительно центра)
        x_vals = np.linspace(-target_width / 2, target_width / 2, grid_res)
        y_vals = np.linspace(-target_height / 2, target_height / 2, grid_res)

        print(f"[DEBUG] Плоская сетка: {grid_res}x{grid_res}, размер {target_width:.2f} x {target_height:.2f} м")
        print(f"[DEBUG] Центр плоскости: {plane_center[0]:.2f}, {plane_center[1]:.2f}, 0")

        # ------------------------------------------------------------
        # 3. KD-дерево для исходных точек (2D проекции)
        # ------------------------------------------------------------
        source_xy = source_points[:, :2]
        source_z = source_points[:, 2]
        source_tree = KDTree(source_xy)

        # ------------------------------------------------------------
        # 4. Инициализация сетки высот и маски исходных значений
        # ------------------------------------------------------------
        height_grid = np.zeros((grid_res, grid_res), dtype=np.float32)
        source_mask = np.zeros((grid_res, grid_res), dtype=bool)

        # Заполняем ячейки, ближайшие к каждой исходной точке
        # (прореживание не требуется, т.к. точек обычно немного)
        for i in range(len(source_points)):
            x_2d, y_2d = source_xy[i]
            # локальные координаты относительно центра плоскости
            local_x = x_2d - plane_center[0]
            local_y = y_2d - plane_center[1]

            idx_x = np.argmin(np.abs(x_vals - local_x))
            idx_y = np.argmin(np.abs(y_vals - local_y))

            if 0 <= idx_x < grid_res and 0 <= idx_y < grid_res:
                # Если несколько точек попадают в один узел – берём максимум (можно и среднее)
                if not source_mask[idx_y, idx_x] or source_z[i] > height_grid[idx_y, idx_x]:
                    height_grid[idx_y, idx_x] = source_z[i]
                    source_mask[idx_y, idx_x] = True

        print(f"[DEBUG] Ячеек с прямыми значениями: {np.sum(source_mask)}")

        if np.sum(source_mask) == 0:
            print("[ERROR] Ни одна исходная точка не попала в плоскую сетку")
            return None

        # ------------------------------------------------------------
        # 5. Поиск граничных точек (на source_mask)
        # ------------------------------------------------------------
        boundary_points = []
        for i in range(grid_res):
            for j in range(grid_res):
                if source_mask[i, j]:
                    # проверяем соседние ячейки
                    is_boundary = False
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_res and 0 <= nj < grid_res:
                                if not source_mask[ni, nj]:
                                    is_boundary = True
                                    break
                        if is_boundary:
                            break
                    if is_boundary:
                        x = plane_center[0] + x_vals[j]
                        y = plane_center[1] + y_vals[i]
                        z = height_grid[i, j]
                        boundary_points.append((x, y, z))

        print(f"[DEBUG] Найдено {len(boundary_points)} граничных точек")

        # ------------------------------------------------------------
        # 6. Адаптивный подъём вершин вне source_mask
        # ------------------------------------------------------------
        lifted_mask = np.zeros((grid_res, grid_res), dtype=bool)
        if len(boundary_points) > 0:
            boundary_array = np.array(boundary_points)
            boundary_tree = KDTree(boundary_array[:, :2])

            # Параметры подъёма
            base_distance = 0.5
            min_distance = 0.1
            max_distance = 3.0
            lift_intensity = 1.0

            boundary_heights = boundary_array[:, 2]
            min_boundary_height = np.min(boundary_heights)
            max_boundary_height = np.max(boundary_heights)
            height_range = max_boundary_height - min_boundary_height

            for i in range(grid_res):
                for j in range(grid_res):
                    if source_mask[i, j]:
                        continue

                    x = plane_center[0] + x_vals[j]
                    y = plane_center[1] + y_vals[i]
                    current_z = 0.0  # базовая плоскость

                    distances, indices = boundary_tree.query([[x, y]], k=1)
                    nearest_dist = distances[0]
                    nearest_idx = indices[0]

                    if nearest_dist < max_distance:
                        nearest_point = boundary_array[nearest_idx]
                        height_diff = nearest_point[2] - current_z

                        if height_diff > 0:
                            # адаптивное расстояние в зависимости от разницы высот
                            if height_range > 0:
                                normalized_height_diff = min(height_diff / height_range, 1.0)
                            else:
                                normalized_height_diff = 0.5

                            if normalized_height_diff < 0.1:
                                dynamic_distance = min_distance * 1.2
                            elif normalized_height_diff < 0.3:
                                dynamic_distance = base_distance * (1.0 + normalized_height_diff)
                            else:
                                dynamic_distance = base_distance * (1.0 + normalized_height_diff * 3.0)

                            dynamic_distance = max(min_distance, min(dynamic_distance, max_distance))

                            if nearest_dist <= dynamic_distance:
                                normalized_distance = nearest_dist / dynamic_distance
                                # smoothstep
                                t = normalized_distance
                                smooth_factor = 1.0 - (3.0 * t * t - 2.0 * t * t * t)

                                lift_amount = height_diff * smooth_factor * lift_intensity
                                height_grid[i, j] = current_z + lift_amount
                                lifted_mask[i, j] = True

            print(f"[DEBUG] Поднято {np.sum(lifted_mask)} вершин")

        # ------------------------------------------------------------
        # 7. Сглаживание поднятой области
        # ------------------------------------------------------------
        if np.any(lifted_mask):
            print("[DEBUG] Сглаживание поднятой области...")
            height_grid = self._smooth_lifted_area(height_grid, lifted_mask)

        # ------------------------------------------------------------
        # 8. Размытие границ перехода
        # ------------------------------------------------------------
        if len(boundary_points) > 0:
            print("[DEBUG] Размытие границ перехода...")
            height_grid = self._blur_lift_boundary(height_grid, source_mask, boundary_points)

        # ------------------------------------------------------------
        # 9. Постобработка граничной зоны
        # ------------------------------------------------------------
        print("[DEBUG] Постобработка граничной зоны...")
        height_grid = self._postprocess_boundary_zone(height_grid, source_mask)

        # ------------------------------------------------------------
        # 10. Общее сглаживание сетки
        # ------------------------------------------------------------
        print("[DEBUG] Общее сглаживание сетки...")
        height_grid = self._smooth_heightfield(height_grid)

        # ------------------------------------------------------------
        # 11. ПРИМЕНЕНИЕ DISPLACEMENT MAP (после всех сглаживаний)
        # ------------------------------------------------------------
        print("[DEBUG] Применение displacement map...")
        height_grid = self._apply_displacement_to_grid(height_grid, grid_res, plane_center, target_width, target_height)

        # ------------------------------------------------------------
        # 12. Создание геометрии
        # ------------------------------------------------------------
        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_mesh_from_cloud", fmt, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")

        vertex_indices = np.zeros((grid_res, grid_res), dtype=int)
        total_vertex_count = 0

        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                world_x = plane_center[0] + x
                world_y = plane_center[1] + y
                z = height_grid[i, j]

                vertex_writer.addData3f(world_x, world_y, z)
                normal_writer.addData3f(0, 0, 1)   # временная нормаль
                texcoord_writer.addData2f(j / (grid_res - 1), i / (grid_res - 1))

                vertex_indices[i, j] = total_vertex_count
                total_vertex_count += 1

        print(f"[DEBUG] Создано {total_vertex_count} вершин")

        # ------------------------------------------------------------
        # 13. Треугольники
        # ------------------------------------------------------------
        tris = GeomTriangles(Geom.UHStatic)
        triangle_count = 0
        for i in range(grid_res - 1):
            for j in range(grid_res - 1):
                v00 = vertex_indices[i, j]
                v10 = vertex_indices[i + 1, j]
                v01 = vertex_indices[i, j + 1]
                v11 = vertex_indices[i + 1, j + 1]

                tris.addVertices(v00, v10, v01)
                tris.addVertices(v01, v10, v11)
                triangle_count += 2

        tris.closePrimitive()
        print(f"[DEBUG] Создано {triangle_count} треугольников")

        # ------------------------------------------------------------
        # 14. Пересчёт нормалей и сборка узла
        # ------------------------------------------------------------
        geom = Geom(vdata)
        geom.addPrimitive(tris)

        self._calculate_normals(vdata, geom)   # уже реализован в исходном классе

        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)

        print(f"[DEBUG] Единый меш успешно создан")
        return node

    def _smooth_lifted_area(self, height_grid, lifted_mask, sigma=2.0):
        sigma = self.lift_smoothing_sigma
        """Сглаживание поднятой области гауссовым фильтром"""
        if not np.any(lifted_mask):
            return height_grid
        
        from scipy.ndimage import gaussian_filter
        
        smoothed_height = height_grid.copy()
        
        try:
            # Применяем гауссово сглаживание ко всей сетке
            gaussian_smoothed = gaussian_filter(height_grid, sigma=sigma)
            
            # Обновляем только поднятые области
            smoothed_height[lifted_mask] = gaussian_smoothed[lifted_mask]
            
        except Exception as e:
            print(f"[DEBUG] Ошибка при гауссовом сглаживании: {e}")
            # Упрощенное сглаживание
            for iteration in range(3):
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
        
        return smoothed_height

    def _blur_lift_boundary(self, height_grid, source_mask, boundary_points):
        """Размытие границ перехода"""
        if len(boundary_points) == 0 or self.lift_blur_radius <= 0:
            return height_grid
        
        from scipy.spatial import KDTree
        
        boundary_array = np.array(boundary_points)
        boundary_tree = KDTree(boundary_array[:, :2])
        
        h, w = height_grid.shape
        blurred_height = height_grid.copy()
        
        # Применяем размытие к точкам вблизи границ
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    continue
                
                # Вычисляем расстояние до ближайшей граничной точки
                x = j  # Индексы как приближение координат
                y = i
                distances, _ = boundary_tree.query([x, y], k=1)
                
                if distances <= self.lift_blur_radius:
                    # Собираем соседей для размытия
                    neighbors = []
                    weights = []
                    
                    for di in range(-self.lift_blur_radius, self.lift_blur_radius + 1):
                        for dj in range(-self.lift_blur_radius, self.lift_blur_radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w:
                                dist = np.sqrt(di*di + dj*dj)
                                weight = np.exp(-dist*dist / (2 * (self.lift_blur_radius/2)**2))
                                neighbors.append(height_grid[ni, nj] * weight)
                                weights.append(weight)
                    
                    if weights:
                        blurred_height[i, j] = np.sum(neighbors) / np.sum(weights)
        
        return blurred_height

    def _postprocess_boundary_zone(self, height_grid, source_mask):
        boundary_width = self.boundary_zone_width
        """Постобработка граничной зоны с использованием distance transform"""
        from scipy.ndimage import distance_transform_edt
        
        # Вычисляем расстояние до исходного меша
        distance_in = distance_transform_edt(~source_mask)
        distance_out = distance_transform_edt(source_mask)
        
        corrected_height = height_grid.copy()
        
        # Корректируем высоты в граничной зоне
        for i in range(height_grid.shape[0]):
            for j in range(height_grid.shape[1]):
                if source_mask[i, j]:
                    dist = distance_out[i, j]
                    if dist < boundary_width:
                        # Плавное уменьшение влияния исходного меша у границ
                        weight = 1.0 - (dist / boundary_width)
                        # Находим среднюю высоту в окрестности
                        neighborhood = []
                        for di in range(-boundary_width, boundary_width + 1):
                            for dj in range(-boundary_width, boundary_width + 1):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < height_grid.shape[0] and 0 <= nj < height_grid.shape[1]:
                                    neighborhood.append(height_grid[ni, nj])
                        avg_neighbor = np.mean(neighborhood) if neighborhood else height_grid[i, j]
                        
                        # Смешиваем с соседними высотами
                        corrected_height[i, j] = height_grid[i, j] * weight + avg_neighbor * (1.0 - weight)
        
        return corrected_height

    def _smooth_heightfield(self, height_grid, smoothing_iterations=4):
        """Общее сглаживание heightfield"""
        print(f"[DEBUG] Начало сглаживания heightfield: {height_grid.shape}, итераций: {smoothing_iterations}")
        smoothed = height_grid.copy()
        h, w = smoothed.shape
        
        for iteration in range(smoothing_iterations):
            print(f"[DEBUG] Итерация сглаживания {iteration + 1}/{smoothing_iterations}")
            new_heights = smoothed.copy()
            for i in range(1, h-1):
                for j in range(1, w-1):
                    neighbors = [
                        smoothed[i-1, j-1], smoothed[i-1, j], smoothed[i-1, j+1],
                        smoothed[i, j-1], smoothed[i, j], smoothed[i, j+1],
                        smoothed[i+1, j-1], smoothed[i+1, j], smoothed[i+1, j+1]
                    ]
                    new_heights[i, j] = np.mean(neighbors)
            smoothed = new_heights
        
        print(f"[DEBUG] Сглаживание heightfield завершено")
        return smoothed
    
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

                            normal = -normal
                            
                            for vi in [vi0, vi1, vi2]:
                                curr = normals[vi]
                                normals[vi] = (
                                    curr[0] + normal[0],
                                    curr[1] + normal[1],
                                    curr[2] + normal[2]
                                )
        
        for i in range(len(normals)):
            nx, ny, nz = normals[i]
            norm = (nx*nx + ny*ny + nz*nz) ** 0.5
            if norm > 0:
                nx /= norm
                ny /= norm
                nz /= norm

            normal_writer.setData3f(nx, ny, nz)


    def add_extended_mesh_to_scene(self, node):
        if hasattr(self, "mesh_node") and self.mesh_node:
            self.mesh_node.removeNode()
        
        if node:
            self.mesh_node = self.panda_app.render.attachNewNode(node)
            
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            self.mesh_node.setShaderAuto()
            self.mesh_node.setPos(0, 0, 0)
            
            if not hasattr(self.panda_app, 'loaded_models'):
                self.panda_app.loaded_models = []
            if not hasattr(self.panda_app, 'model_paths'):
                self.panda_app.model_paths = {}
            
            if self.mesh_node not in self.panda_app.loaded_models:
                self.panda_app.loaded_models.append(self.mesh_node)
                self.panda_app.model_paths[id(self.mesh_node)] = "extended_height_map_mesh"
            
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
        """Apply PBR textures correctly for tobspr RenderPipeline"""

        import os
        from panda3d.core import Texture, TextureStage, Material

        texset = self.panda_app.current_texture_set

        # ------------------------------------------------------------------
        # Resolve paths
        # ------------------------------------------------------------------
        diffuse_path = (
            texset.get("diffuse")
            or texset.get("albedo")
            or "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        )

        normal_path = texset.get(
            "normal",
            "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"
        )

        roughness_path = texset.get("roughness")
        metallic_path  = texset.get("metallic")   # optional

        if not os.path.exists(diffuse_path):
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"

        if not os.path.exists(normal_path):
            normal_path = "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"

        # ------------------------------------------------------------------
        # Create RP-compatible material
        # ------------------------------------------------------------------
        mat = Material()

        # MUST be white for PBR
        mat.set_base_color((1, 1, 1, 1))

        # enable normal map strength (RP trick)
        mat.set_emission((0, 1, 0, 0))

        model_np.set_material(mat)

        # ------------------------------------------------------------------
        # Helper to configure textures
        # ------------------------------------------------------------------
        def setup_tex(tex, srgb=False):
            if srgb:
                tex.set_format(Texture.F_srgb)

            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)

        # ------------------------------------------------------------------
        # Texture stages (STRICT ORDER)
        # ------------------------------------------------------------------
        ts_color = TextureStage("0-color")
        ts_color.set_sort(0)
        ts_color.set_priority(0)

        ts_normal = TextureStage("1-normal")
        ts_normal.set_sort(1)
        ts_normal.set_priority(1)

        ts_metal = TextureStage("2-metallic")
        ts_metal.set_sort(2)
        ts_metal.set_priority(2)

        ts_rough = TextureStage("3-roughness")
        ts_rough.set_sort(3)
        ts_rough.set_priority(3)

        # ------------------------------------------------------------------
        # Load + assign textures
        # ------------------------------------------------------------------

        # Albedo
        diffuse_tex = self.panda_app.loader.loadTexture(diffuse_path)
        setup_tex(diffuse_tex, srgb=True)
        model_np.set_texture(ts_color, diffuse_tex)

        # Normal
        normal_tex = self.panda_app.loader.loadTexture(normal_path)
        setup_tex(normal_tex)
        model_np.set_texture(ts_normal, normal_tex)

        # Metallic (REQUIRED SLOT even if dummy)
        if metallic_path and os.path.exists(metallic_path):
            metal_tex = self.panda_app.loader.loadTexture(metallic_path)
        else:
            # dummy white metallic map
            metal_tex = Texture("dummy_metal")
            metal_tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
            metal_tex.setRamImage(b"\x00")

        setup_tex(metal_tex)
        model_np.set_texture(ts_metal, metal_tex)

        # Roughness
        if roughness_path and os.path.exists(roughness_path):
            rough_tex = self.panda_app.loader.loadTexture(roughness_path)
            setup_tex(rough_tex)
            model_np.set_texture(ts_rough, rough_tex)

        # ------------------------------------------------------------------
        # RP required flags
        # ------------------------------------------------------------------
        model_np.set_shader_auto()
        model_np.set_two_sided(True)

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
    

    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            #QMessageBox.warning(self.panda_app, "Ошибка", "Пожалуйста, выберите корректный JSON-файл.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.ply_path =  json_path.replace(".json", ".ply")
        if(os.path.exists(self.ply_path)):
            self.using_ply = True
            self.point_cloud = pcu.load_mesh_v(self.ply_path)
        else:
            self.heightmap_path = os.path.dirname(json_path) + "/" + data["metadata"]["image_path"].replace("corrected_", "height_map_").replace(".jpg", ".png")
            if not self.load_height_map(): 
                return
        
        self.cam_node = self.panda_app.cam.node()

        self.reconstruct_camera_pos_hpr_fov_depth(data)

        # self.heightmap_path = os.path.dirname(json_path) + "/" + data["metadata"]["mask_path"].replace("corrected_", "height_map_").replace(".jpg", ".png")

        if self.using_ply:
            node = self.create_mesh_from_point_cloud_smoothed()
        else:
            node = self.create_unified_perlin_mesh_with_lift()

        mesh_node = self.add_extended_mesh_to_scene(node)

        return

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
        
        mesh_node_result_trimesh = trimesh.boolean.difference(
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
        print(self.panda_app.calculate_mesh_volume(final_mesh_node))
        # final_mesh_node.setPos(0, 0, 4)
        
        mesh_node.removeNode()