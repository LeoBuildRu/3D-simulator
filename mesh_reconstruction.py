# panda_widget.py
import tkinter as tk
from tkinter import filedialog
from PyQt5 import *
from PyQt5.QtWidgets import *
from panda3d.core import *
import os
import json
import numpy as np
import math

class MeshReconstruction:
    def __init__(self, panda_app):
        self.panda_app = panda_app
        self.recon_json_path = ""
        
    def add_cube_at(self, pos, color=(1, 0, 0, 1), scale=1.0):
        box = self.panda_app.loader.load_model("models/box")  # Panda3D включает "box"
        box.set_pos(pos)
        box.set_scale(scale)
        box.set_color(color)
        box.reparent_to(self.panda_app.render)
        return box

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
        # 1. Переводим FOV в радианы
        half_fov_y = math.radians(fov_y_deg) * 0.5
        half_fov_x = math.atan(aspect_ratio * math.tan(half_fov_y))

        # 2. Преобразуем u,v → углы от центра
        x_angle = (u - 0.5) * 2.0 * half_fov_x      # [-half_fov_x, +half_fov_x]
        y_angle = (0.5 - v) * 2.0 * half_fov_y      # v=0 (низ) → +half_fov_y; v=1 (верх) → -half_fov_y

        # 3. Направление в локальной системе координат камеры (Z-вперёд)
        dir_x = math.tan(x_angle)
        dir_y = math.tan(y_angle)
        dir_z = 1.0  # камера смотрит в -Z

        # 4. Нормализуем направление
        length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
        dir_local = LVector3f(dir_x / length, dir_z / length, dir_y / length)

        # 5. Преобразуем направление в мировые координаты
        dir_world = self.panda_app.render.get_relative_vector(camera, dir_local)

        # 6. Вычисляем точку на заданном расстоянии
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

        # === Шаг 1: Получаем 3D точки по проекции + расстоянию ===
        proj_3d = []
        cam_pos = self.panda_vec3_to_np(camera.get_pos(camera.get_parent()))

        for i in range(3):  # используем первые 3 точки
            # Нормализуем координаты
            u = key_points[i]["x"] / img_w
            v = key_points[i]["y"] / img_h
            distance = distances[i]

            point = self.viewport_to_world_point_geometric(camera, u, v, distance, data["camera_params"]["fov_y"], img_w/img_h)

            #self.add_cube_at(point)
            proj_3d.append(point)

        # === Шаг 2: Целевые 3D точки из сцены ===
        scene_3d = [np.array(p, dtype=float) for p in data["points_3d"][:3]]

        # === Шаг 3: Вычисляем трансформацию ===
        M = self.compute_transform_np(
            scene_3d[0], scene_3d[1], scene_3d[2],
            proj_3d[0], proj_3d[1], proj_3d[2]
        )

        # === Шаг 4: Применяем к камере ===
        old_pos = cam_pos
        new_pos = M @ np.append(old_pos, 1.0)
        camera.set_pos(self.np_to_panda_point(new_pos[:3]))

        # Преобразуем ориентацию: берём forward и up векторы
        # 1. Получаем текущие мировые forward и up
        current_mat = camera.get_net_transform().get_mat()
        world_right = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(1, 0, 0)))
        world_up = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 1, 0)))
        world_forward = self.panda_vec3_to_np(current_mat.xform_vec(LVector3f(0, 0, 1)))

        # 2. Применяем поворотную часть M
        R = M[:3, :3]
        new_right = R @ world_right
        new_up = R @ world_up
        new_forward = R @ world_forward

        # Нормализуем
        new_right /= np.linalg.norm(new_right)
        new_up /= np.linalg.norm(new_up)
        new_forward /= np.linalg.norm(new_forward)

        # 4. Собираем Mat3
        mat3 = Mat3()
        mat3.set_row(0, self.np_to_panda_vec(new_right))   # X = right
        mat3.set_row(1, self.np_to_panda_vec(new_up)) # Y = up
        mat3.set_row(2, self.np_to_panda_vec(new_forward))      # Z = forward

        # 5. Конвертируем в кватернион
        quat = LQuaternionf()
        quat.set_from_matrix(mat3)

        # 6. Устанавливаем ориентацию (позиция не меняется!)
        camera.set_quat(quat)

    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            #QMessageBox.warning(self.panda_app, "Ошибка", "Пожалуйста, выберите корректный JSON-файл.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.reconstruct_camera_pos_hpr(data)

        