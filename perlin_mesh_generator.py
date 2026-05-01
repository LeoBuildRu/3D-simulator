import sys
import os

def get_resource_path(relative_path):
    """Ищет файл сначала в папке с exe, затем в _internal, затем рядом с модулем."""
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
        internal = os.path.join(base, '_internal')
        for loc in (base, internal):
            full = os.path.join(loc, relative_path)
            if os.path.exists(full):
                return full
    # Режим разработки или не найдено
    return os.path.join(os.path.dirname(__file__), relative_path)

import os
import random
import math
import numpy as np
from PIL import Image
import traceback

import trimesh
from noise import pnoise2

from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    GeomTriangles, NodePath, Vec3, TextureStage, Texture,
    Material, TransparencyAttrib, Shader, GeomVertexReader,
    Filename  
)

NOISE_AVAILABLE = True

class PerlinMeshGenerator:
    """Класс для генерации перлин-мешей и связанных операций"""
    
    def __init__(self, panda_app, tls_client=None, server_host='78.25.191.12', server_port=9998):
        self.panda_app = panda_app
        self.last_target_model_trimesh = None
        self.last_best_z = None
        self.test_perlin_mesh = None
        self.last_grid_size = 48
        self.perlin_vertices_before_displace = None
        self.perlin_texcoords_before_displace = None
        self.processed_model = None
        self.current_display_model = None

        self.gui = panda_app.gui

        self.last_size_x = None
        self.last_size_y = None
        self.last_half_size_x = None
        self.last_half_size_y = None

        # Создаём клиент для общения с сервером
        if tls_client is not None:
            self.tls_client = tls_client
        else:
            # Для обратной совместимости, если клиент не передан
            self.tls_client = TLS_client(host=server_host, port=server_port)
        
    def generate_perlin_mesh(self, grid_size=48):
        """
        Генерация ландшафта через Blender ANT Landscape (замена перлин-меша).
        Параметр grid_size оставлен для совместимости, но не используется.
        """
        if self.gui:
            self.gui.log_message("🔄 Отправка запроса на сервер для генерации ландшафта (Blender ANT Landscape)...")

        # -----------------------------------------------------------------
        # 1. Получение размеров из последней CSG‑операции (как и раньше)
        # -----------------------------------------------------------------
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

            # (опционально) масштабирование по объёму – оставляем как было
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

        self.last_size_x = size_x
        self.last_size_y = size_y
        self.last_half_size_x = size_x / 2.0
        self.last_half_size_y = size_y / 2.0
        self.last_grid_size = grid_size  # сохраняем для возможной обратной совместимости

        # -----------------------------------------------------------------
        # 2. Параметры текстуры и шума (из текущего набора)
        # -----------------------------------------------------------------
        texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)
        strength = self.panda_app.current_texture_set.get('strength', 0.14)
        seed = random.randint(0, 10000)

        # -----------------------------------------------------------------
        # 3. Параметры для Blender ANT Landscape
        #    (подбираются в соответствии с логикой старого генератора)
        # -----------------------------------------------------------------
        # Можно задать фиксированные значения или вычислить из контекста
        subdivisions = 48               # высокое разрешение сразу, чтобы не делать subdivision
        mesh_size_x_blender = size_x     # передаём реальные размеры области
        mesh_size_y_blender = size_y
        noise_scale_blender = 1.36       # аналог noise_scale в старом коде
        height_blender = 0.77            # можно связать с strength / size_z
        distortion_blender = 1.39
        depth_blender = 8
        edge_falloff = "3"               # тип спада по краям
        edge_level = -0.12
        falloff_x = 3.70
        falloff_y = 4.00

        try:
            landscape_data = self.tls_client.generate_landscape(
                name="Landscape",
                subdivisions=subdivisions,
                subdivisions_x=subdivisions,
                subdivisions_y=subdivisions,
                mesh_size_x=mesh_size_x_blender,
                mesh_size_y=mesh_size_y_blender,
                size_x=1.45,
                size_y=2.23,
                noise_scale=noise_scale_blender,
                noise_type="rocks_noise",
                noise_basis="BLENDER",
                offset_x=-1.05,
                offset_y=0.0,
                height=height_blender,
                height_offset=0.0,
                depth=depth_blender,
                distortion=distortion_blender,
                hard_noise="0",
                maximum=10000.0,
                minimum=-10000.0,
                edge_falloff=edge_falloff,
                edge_level=edge_level,
                falloff_x=falloff_x,
                falloff_y=falloff_y,
                strata_type="0",
                seed=seed,
                texture_repeat_x=texture_repeatX,
                texture_repeat_y=texture_repeatY,
                output_format="ply"
            )
        except Exception as e:
            if self.gui:
                self.gui.log_message(f"❌ Ошибка вызова generate_landscape: {e}")
            raise RuntimeError("Не удалось получить данные от сервера ландшафта")

        if landscape_data is None:
            raise RuntimeError("Сервер не вернул данные ландшафта")

                # Ожидаем, что landscape_data — это кортеж: (vertices, normals, uvs, triangles)
        if len(landscape_data) != 4:
            raise RuntimeError(f"Неверный формат данных от generate_landscape: ожидался кортеж из 4 элементов, получено {len(landscape_data)}")

                # Правильный порядок: vertices, triangles, normals, uvs
        vertices_list, triangles, normals_list, texcoords_list = landscape_data

        if self.gui:
            self.gui.log_message(f"✅ Получено {len(vertices_list)} вершин, {len(triangles)} треугольников")

        # -----------------------------------------------------------------
        # 5. Построение геометрии Panda3D
        # -----------------------------------------------------------------
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("landscape_mesh", format, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")

        # vertices_list, normals_list, texcoords_list — NumPy массивы (N,3), (N,3), (N,2)
        for i in range(len(vertices_list)):
            v = vertices_list[i]
            n = normals_list[i]
            t = texcoords_list[i]
            vertex_writer.addData3f(v[0], v[1], v[2])
            normal_writer.addData3f(n[0], n[1], n[2])
            texcoord_writer.addData2f(t[0], t[1])

        # triangles — NumPy массив (M,3) uint32
        prim = GeomTriangles(Geom.UHStatic)
        for tri in triangles:
            prim.addVertices(int(tri[0]), int(tri[1]), int(tri[2]))
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode("landscape_node")
        geom_node.addGeom(geom)
        perlin_np = NodePath(geom_node)

        self.perlin_vertices_before_displace = vertices_list.copy()
        self.perlin_texcoords_before_displace = texcoords_list.copy()

        if self.gui:
            self.gui.log_message("✅ Ландшафт успешно создан")

        return perlin_np
    
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
        if not(self.panda_app.canDistributeMeshes):
            for distrib in self.panda_app.mesh_distributions:
                distrib.stop_rendering()
            self.panda_app.mesh_distributions.clear()

        if hasattr(self.panda_app, 'final_model') and self.panda_app.final_model:
            self.panda_app.final_model.removeNode()
            self.panda_app.final_model = None

        if not hasattr(self, 'perlin_vertices_before_displace'):
            print("Нет сохраненных данных перлина")
            return False
        
        if not hasattr(self, 'last_target_model_trimesh') or self.last_target_model_trimesh is None:
            print("Нет сохраненной target_model_trimesh")
            return False
        
        if self.gui:
            self.gui.log_message("🔄 Создание меша из сохранённых данных перлина...")
        
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
        
        if self.gui:
            self.gui.log_message("✅ Displacement применён")

        falloff_config = self._get_falloff_config()
        vertices = self._apply_falloff(vertices, size_x, size_y, vertices[0][2] if vertices else 0, falloff_config)
        
        if self.gui:
            self.gui.log_message("✅ Falloff применён")

        normals = self._calculate_normals(vertices, grid_size)

        if self.gui:
            self.gui.log_message("✅ Нормали пересчитаны")
        
        perlin_np = self._create_geom_from_vertices(vertices, normals, texcoords_list, grid_size, "recreated_perlin_mesh")
        
        if self.gui:
            self.gui.log_message("✅ Геометрия создана, выполнение булевой разности...")
        
        perlin_np.setPos(0, 0, self.last_best_z if hasattr(self, 'last_best_z') else 0)
        
        perlin_model_trimesh = self.panda_app.panda_to_trimesh(perlin_np)
        
        try:
            result_vertices, result_triangles = self.tls_client.send_boolean_request(
                self.last_target_model_trimesh.vertices,
                self.last_target_model_trimesh.faces,
                perlin_model_trimesh.vertices,
                perlin_model_trimesh.faces,
                return_volume_only=False
            )
            final_result_trimesh = trimesh.Trimesh(vertices=result_vertices, faces=result_triangles)
            
            if final_result_trimesh.is_empty:
                print("Boolean разность вернула пустой меш")
                perlin_np.removeNode()
                return False
                
            self.panda_app.final_model = self.panda_app.trimesh_to_panda(final_result_trimesh)
            perlin_np.removeNode()

            if self.gui:
                self.gui.log_message("✅ Булева разность выполнена, обновление UV...")
            
        except Exception as e:
            print(f"Ошибка при выполнении boolean разности: {e}")
            perlin_np.removeNode()
            return False
        
        geom_node = self.panda_app.final_model.node()
        if geom_node.getNumGeoms() > 0:
            if(self.panda_app.canDistributeMeshes):
                self.panda_app.distribute_meshes(geom_node)
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

        if self.gui:
            self.gui.log_message("✅ UV-координаты обновлены")
        
        self._apply_textures_and_material(self.panda_app.final_model)

        if self.gui:
            self.gui.log_message("✅ Текстуры применены")

        if self.gui:
            self.gui.log_message("✅ Меш успешно создан")
        
        return True
    
    def generate_perlin_mesh_from_csg(self):
        if self.gui:
            self.gui.log_message("🌄 Начало генерации Perlin mesh...")

        if hasattr(self.panda_app, 'final_mesh_node') and self.panda_app.final_mesh_node:
            self.panda_app.final_mesh_node.removeNode()
            self.panda_app.final_mesh_node = None

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

        # Поиск целевой модели (как в оригинале)
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
            print("Целевая модель не найдена")
            return False

        if self.gui:
            self.gui.log_message("✅ Целевая модель найдена")
        
        # Подготовка target_model для boolean (без изменений)
        target_model_trimesh = self._prepare_target_model_for_boolean(target_model)
        
        if self.gui:
            self.gui.log_message("✅ Модель подготовлена для булевых операций")

        self.last_target_model_trimesh = target_model_trimesh
        if target_model_trimesh is None:
            target_model.setScale(1.0, 1.0, 1.0)
            return False
        target_model.setScale(1.0, 1.0, 1.0)
        target_model.setPos(0.0, 0.0, 0.0)

        # Генерация базового перлин-меша (grid_size=48)
        if self.gui:
            self.gui.log_message("🔄 Генерация базового перлин-меша (размер сетки 48)...")
        perlin_base_np = self.generate_perlin_mesh(grid_size=48)
        if self.gui:
            self.gui.log_message("✅ Базовый перлин-меш создан")
        ground_pos = self.panda_app.ground_plane.getPos()
        perlin_base_np.setPos(ground_pos.x, ground_pos.y, ground_pos.z - 2.25)
        self.panda_app.loaded_models.append(perlin_base_np)

        target_volume = self.panda_app.Target_Volume

        if self.gui:
            self.gui.log_message("🔍 Поиск оптимальной высоты...")

        # Поиск оптимальной Z (использует обновлённый _evaluate_z_position)
        best_z = self.find_best_z_position(
            perlin_base_np,
            target_model_trimesh,
            target_volume,
            initial_z=perlin_base_np.getZ()
        )
        self.last_best_z = best_z

        if self.gui:
            self.gui.log_message(f"✅ Оптимальная высота найдена: {best_z:.4f}")

        if self.current_display_model is not None:
            self.current_display_model.removeNode()
            self.current_display_model = None

        perlin_base_np.removeNode()

        # Получаем данные для subdivision
        height_texture_path = self._get_height_texture_path()
        height_array, tex_width, tex_height = self._load_height_array(height_texture_path)
        strength = self.panda_app.current_texture_set.get('strength', 0.14)
        texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)

        # Повышаем разрешение сетки (old_size=48 -> new_size=512)
        new_vertices, new_normals, new_texcoords = self.subdivide_mesh(
            self.perlin_vertices_before_displace,   # от грубой сетки
            self.perlin_texcoords_before_displace,
            old_size=48,
            new_size=512,
            height_array=height_array,
            tex_width=tex_width,
            tex_height=tex_height,
            strength=strength
        )

        # Создаём детальный меш
        perlin_detailed_np = self._create_geom_from_vertices(
            new_vertices, new_normals, new_texcoords, 512, "subdivided_perlin_mesh"
        )
        perlin_detailed_np.setPos(0, 0, best_z)

        if self.gui:
            self.gui.log_message("✅ Детальный перлин-меш создан")

        # Получаем trimesh детального меша
        perlin_trimesh = self.panda_app.panda_to_trimesh(perlin_detailed_np)

        if self.gui:
            self.gui.log_message("✂️ Выполнение булевой разности на сервере...")

        # Отправляем запрос на сервер для булевой разности (получаем геометрию)
        result_vertices, result_triangles = self.tls_client.send_boolean_request(
            target_model_trimesh.vertices,
            target_model_trimesh.faces,
            perlin_trimesh.vertices,
            perlin_trimesh.faces,
            return_volume_only=False
        )
        print(f"[DEBUG] Boolean request completed: vertices={len(result_vertices)}, triangles={len(result_triangles)}")

        if self.gui:
            self.gui.log_message(f"✅ Булева разность выполнена, получено {len(result_vertices)} вершин")

        result_trimesh = trimesh.Trimesh(vertices=result_vertices, faces=result_triangles)

        self.panda_app.particle_flag = True
        self.panda_app.final_model = self.panda_app.trimesh_to_panda(result_trimesh)
        self.panda_app.particle_flag = False

        if self.gui:
            self.gui.log_message("🔄 Пересчёт UV-координат...")

        # ---- НОВЫЙ КОД ВМЕСТО СТАРОГО ПЕРЕСЧЁТА UV ----
        # Используем сохранённые размеры перлин-меша
        if (self.last_size_x is not None and self.last_size_y is not None and
            self.last_half_size_x is not None and self.last_half_size_y is not None):
            size_x = self.last_size_x
            size_y = self.last_size_y
            half_size_x = self.last_half_size_x
            half_size_y = self.last_half_size_y
        else:
            # fallback – если по какой-то причине размеры не сохранены
            bounds = self.panda_app.final_model.getTightBounds()
            if bounds[0] is not None and bounds[1] is not None:
                min_pt, max_pt = bounds
                size_x = max_pt.x - min_pt.x
                size_y = max_pt.y - min_pt.y
                half_size_x = (min_pt.x + max_pt.x) / 2.0
                half_size_y = (min_pt.y + max_pt.y) / 2.0
            else:
                size_x = 10.0
                size_y = 10.0
                half_size_x = 0.0
                half_size_y = 0.0

        tex_repeat_x = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
        tex_repeat_y = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)

        # Извлекаем геометрию финальной модели
        try:
            geom_node = self.panda_app.final_model.node()
            if geom_node.getNumGeoms() > 0:
                geom = geom_node.getGeom(0)
                vdata = geom.getVertexData()
                vertex_reader = GeomVertexReader(vdata, "vertex")

                vertices = []
                while not vertex_reader.isAtEnd():
                    v = vertex_reader.getData3f()
                    vertices.append((v.x, v.y, v.z))

                # Создаём новые данные с UV
                format = GeomVertexFormat.getV3n3t2()
                new_vdata = GeomVertexData("final_mesh_with_uv", format, Geom.UHStatic)
                vertex_writer = GeomVertexWriter(new_vdata, "vertex")
                normal_writer = GeomVertexWriter(new_vdata, "normal")
                texcoord_writer = GeomVertexWriter(new_vdata, "texcoord")

                normal_reader = GeomVertexReader(vdata, "normal") if vdata.hasColumn("normal") else None

                for vx, vy, vz in vertices:
                    vertex_writer.addData3f(vx, vy, vz)
                    n = normal_reader.getData3f()
                    normal_writer.addData3f(-n.x, -n.y, -n.z)

                    # Генерация UV на основе исходных размеров перлин-меша
                    u = ((vx + half_size_x) / size_x) * tex_repeat_x
                    v = ((vy + half_size_y) / size_y) * tex_repeat_y
                    texcoord_writer.addData2f(u, v)

                # Копируем индексы треугольников
                prim = geom.getPrimitive(0)
                new_prim = GeomTriangles(Geom.UHStatic)
                for i in range(prim.getNumPrimitives()):
                    start = prim.getPrimitiveStart(i)
                    end = prim.getPrimitiveEnd(i)
                    for j in range(start, end):
                        idx = prim.getVertex(j)
                        new_prim.addVertex(idx)
                new_prim.closePrimitive()

                new_geom = Geom(new_vdata)
                new_geom.addPrimitive(new_prim)
                new_geom_node = GeomNode("final_mesh_with_uv")
                new_geom_node.addGeom(new_geom)

                # Заменяем старую модель
                self.panda_app.final_model.removeNode()
                self.panda_app.final_model = self.panda_app.render.attachNewNode(new_geom_node)

                if self.gui:
                    self.gui.log_message("✅ UV-координаты обновлены")
            else:
                print("[WARN] final_model не содержит геометрии")
        except Exception as e:
            print(f"[ERROR] Ошибка при пересчёте UV: {e}")
            traceback.print_exc()

        if self.gui:
            self.gui.log_message("🎨 Применение текстур и материала...")

        # Применение текстур и материала (как было)
        self._apply_textures_and_material(self.panda_app.final_model)
        print(self.panda_app.calculate_mesh_volume(self.panda_app.final_model))

        if self.gui:
            self.gui.log_message("✅ Текстуры применены")

        target_model.hide()
        perlin_detailed_np.hide()

        if self.gui:
            self.gui.log_message("✅ Perlin mesh успешно сгенерирован!")

        if hasattr(self.panda_app, 'final_model') and self.panda_app.final_model:
            volume = self.panda_app.calculate_mesh_volume(self.panda_app.final_model)
            self.panda_app.update_overlay_info(volume=volume)

    def subdivide_mesh(self, vertices, texcoords, old_size, new_size,
                    height_array, tex_width, tex_height, strength):
        """
        Повышает разрешение сетки с old_size x old_size до new_size x new_size
        путём билинейной интерполяции вершин и текстурных координат,
        затем применяет displacement по карте высот и пересчитывает нормали.
        """
        # Преобразуем списки в двумерные массивы для удобства интерполяции
        V = [[None] * old_size for _ in range(old_size)]
        T = [[None] * old_size for _ in range(old_size)]
        for j in range(old_size):
            for i in range(old_size):
                idx = j * old_size + i
                V[j][i] = vertices[idx]
                T[j][i] = texcoords[idx]

        new_vertices = []
        new_texcoords = []

        # Для каждой точки новой сетки
        for j_new in range(new_size):
            for i_new in range(new_size):
                # Параметрические координаты в [0, 1]
                s = i_new / (new_size - 1)
                t = j_new / (new_size - 1)

                # Координаты в исходной сетке (вещественные)
                u = s * (old_size - 1)
                v = t * (old_size - 1)

                i0 = int(u)
                j0 = int(v)
                i1 = min(i0 + 1, old_size - 1)
                j1 = min(j0 + 1, old_size - 1)

                dx = u - i0
                dy = v - j0

                # Билинейная интерполяция для трёхкомпонентной вершины
                def interp3(c00, c10, c01, c11, dx, dy):
                    x = (1-dx)*(1-dy)*c00[0] + dx*(1-dy)*c10[0] + (1-dx)*dy*c01[0] + dx*dy*c11[0]
                    y = (1-dx)*(1-dy)*c00[1] + dx*(1-dy)*c10[1] + (1-dx)*dy*c01[1] + dx*dy*c11[1]
                    z = (1-dx)*(1-dy)*c00[2] + dx*(1-dy)*c10[2] + (1-dx)*dy*c01[2] + dx*dy*c11[2]
                    return (x, y, z)

                def interp2(c00, c10, c01, c11, dx, dy):
                    uu = (1-dx)*(1-dy)*c00[0] + dx*(1-dy)*c10[0] + (1-dx)*dy*c01[0] + dx*dy*c11[0]
                    vv = (1-dx)*(1-dy)*c00[1] + dx*(1-dy)*c10[1] + (1-dx)*dy*c01[1] + dx*dy*c11[1]
                    return (uu, vv)

                new_v = interp3(V[j0][i0], V[j0][i1], V[j1][i0], V[j1][i1], dx, dy)
                new_t = interp2(T[j0][i0], T[j0][i1], T[j1][i0], T[j1][i1], dx, dy)

                new_vertices.append(new_v)
                new_texcoords.append(new_t)

        # Применяем displacement к новым вершинам
        displaced_vertices = self._apply_displacement(
            new_vertices, new_texcoords,
            height_array, tex_width, tex_height, strength
        )

        # Пересчитываем нормали
        new_normals = self._calculate_normals(displaced_vertices, new_size)

        return displaced_vertices, new_normals, new_texcoords
    
    def find_best_z_position(self, mesh_np, target_model_trimesh, target_volume, initial_z=0):
        """Поиск оптимальной Z-позиции меша для достижения целевого объема"""
        tolerance = 0.15
        min_z = -3
        max_z = 3
        max_iterations = 50

        best_z = initial_z
        best_volume = None
        best_error = float('inf')

        if hasattr(self, 'current_display_model') and self.current_display_model:
            self.current_display_model.removeNode()
        self.current_display_model = None

        search_points = [min_z + (max_z - min_z) * i / 10 for i in range(11)]
        search_volumes = []

        # DEBUG: Начальные параметры
        print("=== DEBUG: Начало поиска best_z ===")
        print(f"Целевой объем: {target_volume}")
        print(f"Допуск: {tolerance}")
        print(f"Диапазон поиска: [{min_z}, {max_z}]")
        print(f"Макс. итераций: {max_iterations}")
        print(f"Начальные точки поиска ({len(search_points)}): {search_points}")

        if self.gui:
            self.gui.log_message("🔍 Поиск оптимальной высоты (начальный перебор)...")

        for z in search_points:
            mesh_np.setPos(0, 0, z)
            perlin_model_trimesh_ = self.panda_app.panda_to_trimesh(mesh_np)
            
            if self.gui:
                self.gui.log_message(f"Проверка z={z:.4f}...")
            result_vertices, result_triangles = self.tls_client.send_boolean_request(
                target_model_trimesh.vertices,
                target_model_trimesh.faces,
                perlin_model_trimesh_.vertices,
                perlin_model_trimesh_.faces,
                return_volume_only=False
            )
            result_csg = trimesh.Trimesh(vertices=result_vertices, faces=result_triangles)
            model_csg_plane_1 = self.panda_app.trimesh_to_panda(result_csg)
            self._setup_transparent_material(model_csg_plane_1)
            
            if self.current_display_model is not None:
                self.current_display_model.removeNode()
            
            self.current_display_model = model_csg_plane_1
            self.current_display_model.reparentTo(self.panda_app.render)
            
            volume = self.panda_app.calculate_mesh_volume(self.current_display_model)
            error = abs(volume - target_volume)
            if self.gui:
                self.gui.log_message(f"  объём={volume:.4f}, ошибка={error:.4f}")
            search_volumes.append((z, volume, error))
            
            # DEBUG: Результаты для каждой точки поиска
            print(f"  z={z:.4f}: объем={volume:.6f}, ошибка={error:.6f}, best_error={best_error:.6f}, best_z={best_z:.4f}")
            
            if error < best_error:
                best_error = error
                best_z = z
                best_volume = volume
                print(f"    -> НОВЫЙ ЛУЧШИЙ! Обновлен best_z={best_z:.4f}, best_error={best_error:.6f}")
        
        # DEBUG: Результаты начального поиска
        print(f"\n=== DEBUG: Результаты начального поиска ===")
        print(f"Лучшая точка: z={best_z:.4f}, объем={best_volume:.6f}, ошибка={best_error:.6f}")
        
        if best_error <= tolerance:
            mesh_np.setPos(0, 0, best_z)
            if self.current_display_model is not None:
                self.current_display_model.removeNode()
            print(f"Достигнута требуемая точность! Возвращаем best_z={best_z:.4f}")
            return best_z

        search_volumes.sort(key=lambda x: x[2])
        best_points = search_volumes[:3]
        
        # DEBUG: Лучшие точки для уточнения
        print(f"\n=== DEBUG: Лучшие точки для уточнения ===")
        for i, (z, vol, err) in enumerate(best_points):
            print(f"  {i+1}: z={z:.4f}, объем={vol:.6f}, ошибка={err:.6f}")

        if len(best_points) >= 2:
            z_values = [p[0] for p in best_points]
            min_search_z = min(z_values)
            max_search_z = max(z_values)
            
            range_expand = (max_search_z - min_search_z) * 0.2
            min_search_z = max(min_z, min_search_z - range_expand)
            max_search_z = min(max_z, max_search_z + range_expand)
            
            # DEBUG: Параметры золотого сечения
            print(f"\n=== DEBUG: Настройка золотого сечения ===")
            print(f"Исходный диапазон: [{min(z_values):.4f}, {max(z_values):.4f}]")
            print(f"Расширенный диапазон: [{min_search_z:.4f}, {max_search_z:.4f}]")
            print(f"Коэффициент золотого сечения (phi): {0.618}")
            
            phi = 0.618
            a = min_search_z
            b = max_search_z
            
            x1 = b - phi * (b - a)
            x2 = a + phi * (b - a)
            
            iteration = 0
            while (b - a) > 0.01 and iteration < max_iterations:
                iteration += 1
                
                # DEBUG: Начало итерации
                print(f"\n--- Итерация {iteration} ---")
                print(f"Текущий интервал: a={a:.6f}, b={b:.6f}, ширина={b-a:.6f}")
                print(f"Точки проверки: x1={x1:.6f}, x2={x2:.6f}")
                
                vol1, err1 = self._evaluate_z_position(mesh_np, target_model_trimesh, x1, target_volume)
                vol2, err2 = self._evaluate_z_position(mesh_np, target_model_trimesh, x2, target_volume)
                
                # DEBUG: Результаты проверки точек
                print(f"  x1={x1:.6f}: объем={vol1:.6f}, ошибка={err1:.6f}")
                print(f"  x2={x2:.6f}: объем={vol2:.6f}, ошибка={err2:.6f}")
                print(f"  Текущий best_error={best_error:.6f}, best_z={best_z:.6f}")
                
                if err1 < best_error:
                    best_error = err1
                    best_z = x1
                    best_volume = vol1
                    print(f"    -> Обновление по x1! Новый best_z={best_z:.6f}, best_error={best_error:.6f}")
                
                if err2 < best_error:
                    best_error = err2
                    best_z = x2
                    best_volume = vol2
                    print(f"    -> Обновление по x2! Новый best_z={best_z:.6f}, best_error={best_error:.6f}")
                
                if best_error <= tolerance:
                    print(f"Достигнута требуемая точность! Ошибка={best_error:.6f} <= {tolerance}")
                    break
                
                if err1 < err2:
                    print(f"  err1({err1:.6f}) < err2({err2:.6f}) -> обновляем b")
                    b = x2
                    x2 = x1
                    err2 = err1
                    x1 = b - phi * (b - a)
                    vol1, err1 = self._evaluate_z_position(mesh_np, target_model_trimesh, x1, target_volume)
                else:
                    print(f"  err2({err2:.6f}) <= err1({err1:.6f}) -> обновляем a")
                    a = x1
                    x1 = x2
                    err1 = err2
                    x2 = a + phi * (b - a)
                    vol2, err2 = self._evaluate_z_position(mesh_np, target_model_trimesh, x2, target_volume)
                
                if best_error <= tolerance:
                    print(f"Достигнута требуемая точность после обновления интервала!")
                    break
        
        mesh_np.setPos(0, 0, best_z)

        self.panda_app.actual_volume_generated = best_volume
        
        if self.current_display_model is not None:
            self.current_display_model.removeNode()
            self.current_display_model = None

        if self.gui:
            self.gui.log_message(f"✅ Поиск завершён, выбрана высота {best_z:.4f}")
        
        return best_z
    
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
    
    def _evaluate_z_position(self, mesh_np, target_model_trimesh, z, target_volume):
        mesh_np.setPos(0, 0, z)
        perlin_trimesh = self.panda_app.panda_to_trimesh(mesh_np)

        # Всегда запрашиваем геометрию, чтобы избежать расхождений в расчёте объёма
        result_vertices, result_triangles = self.tls_client.send_boolean_request(
            target_model_trimesh.vertices,
            target_model_trimesh.faces,
            perlin_trimesh.vertices,
            perlin_trimesh.faces,
            return_volume_only=False
        )

        result_trimesh = trimesh.Trimesh(vertices=result_vertices, faces=result_triangles)
        temp_np = self.panda_app.trimesh_to_panda(result_trimesh)
        volume = self.panda_app.calculate_mesh_volume(temp_np)
        temp_np.removeNode()

        error = abs(volume - target_volume)
        return volume, error
    
    # Вспомогательные методы, которые использовались в перенесенных методах
    def _get_height_texture_path(self):
        """Получает путь к текстуре высот"""
        if 'displacement' in self.panda_app.current_texture_set:
            rel_path = self.panda_app.current_texture_set['displacement']
        elif 'height' in self.panda_app.current_texture_set:
            rel_path = self.panda_app.current_texture_set['height']
        else:
            print("В текущем наборе текстур нет displacement или height")
            return None
        
        # Преобразуем относительный путь в абсолютный с учётом сборки
        abs_path = get_resource_path(rel_path)
        if not os.path.exists(abs_path):
            print(f"Текстура не найдена по пути: {abs_path}")
            return None
        return abs_path

    
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
        import os
        from panda3d.core import Texture, TextureStage, Material

        texset = self.panda_app.current_texture_set

        # Пути к текстурам
        diffuse_path = texset.get("diffuse") or texset.get("albedo") or "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        normal_path = texset.get("normal", "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg")
        roughness_path = texset.get("roughness")
        metallic_path = texset.get("metallic")

        # Проверка существования файлов (с резервными)
        if not os.path.exists(diffuse_path):
            diffuse_path = "textures/stones_8k/rocks_ground_01_diff_8k.jpg"
        if not os.path.exists(normal_path):
            normal_path = "textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg"

        # Создаём PBR-материал
        mat = Material()
        mat.set_base_color((1, 1, 1, 1))
        # Для RP: зелёный канал emission = сила нормалей (если требуется)
        mat.set_emission((0, 1, 0, 0))
        model_np.set_material(mat)

        # Вспомогательная функция настройки текстур
        def setup_tex(tex, srgb=False):
            if srgb:
                tex.set_format(Texture.F_srgb)
            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)

        # Слоты с правильным порядком
        ts_color = TextureStage("0-color")
        ts_color.set_sort(0)
        ts_normal = TextureStage("1-normal")
        ts_normal.set_sort(1)
        ts_metal = TextureStage("2-metallic")
        ts_metal.set_sort(2)
        ts_rough = TextureStage("3-roughness")
        ts_rough.set_sort(3)

        # Albedo
        diffuse_tex = self.panda_app.loader.loadTexture(diffuse_path)
        setup_tex(diffuse_tex, srgb=True)
        model_np.set_texture(ts_color, diffuse_tex)

        # Normal map
        normal_tex = self.panda_app.loader.loadTexture(normal_path)
        setup_tex(normal_tex)
        model_np.set_texture(ts_normal, normal_tex)

        # Metallic (всегда заполняем)
        if metallic_path and os.path.exists(metallic_path):
            metal_tex = self.panda_app.loader.loadTexture(metallic_path)
        else:
            metal_tex = Texture("dummy_metal")
            metal_tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
            metal_tex.set_ram_image(b"\x00")  # чёрный = 0 металличности
        setup_tex(metal_tex)
        model_np.set_texture(ts_metal, metal_tex)

        # Roughness (всегда заполняем заглушкой)
        if roughness_path and os.path.exists(roughness_path):
            rough_tex = self.panda_app.loader.loadTexture(roughness_path)
        else:
            rough_tex = Texture("dummy_rough")
            rough_tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
            rough_tex.set_ram_image(b"\x80")  # 0x80 = 0.5 в линейном (средняя шероховатость)
        setup_tex(rough_tex)
        model_np.set_texture(ts_rough, rough_tex)

        # Флаги RP
        # model_np.set_shader_auto()  # Для RP обычно не требуется, но можно оставить
        model_np.set_two_sided(True)
