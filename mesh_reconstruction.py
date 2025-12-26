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
from PIL import Image

class MeshReconstruction:
    def __init__(self, panda_app, image_path = None):
        self.panda_app = panda_app
        self.recon_json_path = ""

        # for testing only
        self.image_path = image_path or "height_example/Example-1-3-final.png"
        self.alpha_threshold = 0.5
        
    def add_cube_at(self, pos, color=(1, 0, 0, 1), scale=1.0):
        box = self.panda_app.loader.load_model("models/box")
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

        self.fov_y = data["camera_params"]["fov_y"]
        self.aspect_ratio = img_w/img_h

        for i in range(3):  # используем первые 3 точки
            # Нормализуем координаты
            u = key_points[i]["x"] / img_w
            v = key_points[i]["y"] / img_h
            distance = distances[i]

            point = self.viewport_to_world_point_geometric(camera, u, v, distance, self.fov_y, self.aspect_ratio)

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

    def load_height_map(self):
        try:
            print(f"[DEBUG] Начало загрузки height map: {self.image_path}")
            img = Image.open(self.image_path)
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
        
    def create_unified_perlin_mesh_with_lift(self):
        h, w = self.height_map.shape

        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        vertices = 0
        vertex_indices = {}

        camera = self.panda_app.camera
        
        for y in range(h):
            for x in range(w):
                z = 1 - self.height_map[y, x]

                min_depth = 14.6
                max_depth = 19.5

                distance = z * (max_depth - min_depth) + min_depth

                u = (x / w)
                v = (y / h)

                point = self.viewport_to_world_point_geometric(camera, u, v, distance, self.fov_y, self.aspect_ratio)
                
                vertex.addData3f(point)
                normal.addData3f(0, 0, 1) 
                texcoord.addData2f(x / w, y / h)
                
                vertex_indices[(y, x)] = vertices
                vertices += 1
        
        triangles = GeomTriangles(Geom.UHStatic)
        
        for y in range(h - 1):
            for x in range(w - 1):
                if not self.mask[y + 1, x] or not self.mask[y, x + 1]: continue #used in both tris

                v1 = vertex_indices[(y, x)]
                v2 = vertex_indices[(y, x + 1)]
                v3 = vertex_indices[(y + 1, x)]
                v4 = vertex_indices[(y + 1, x + 1)]
                
                if self.mask[y, x]:
                    triangles.addVertices(v1, v3, v2)
                
                if self.mask[y+1, x+1]:
                    triangles.addVertices(v2, v3, v4)
        
        triangles.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        
        self._calculate_normals(vdata, geom)
        
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

    def run_2d_to_3d_reconstruction(self):
        json_path = self.recon_json_path
        if not json_path or not os.path.isfile(json_path):
            #QMessageBox.warning(self.panda_app, "Ошибка", "Пожалуйста, выберите корректный JSON-файл.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.reconstruct_camera_pos_hpr(data)

        if not self.load_height_map(): return

        node = self.create_unified_perlin_mesh_with_lift()

        self.add_extended_mesh_to_scene(node)


        