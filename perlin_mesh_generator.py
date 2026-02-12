import os
import random
import math
import numpy as np
from PIL import Image

import trimesh

from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    GeomTriangles, NodePath, Vec3, TextureStage, Texture,
    Material, TransparencyAttrib, Shader, GeomVertexReader
)

# Импорт клиента
from TLS_client import PerlinMeshClient

import time
from datetime import datetime


class PerlinMeshGenerator:
    """Класс для генерации перлин-мешей через удалённый сервер и выполнения булевых операций"""

    def __init__(self, panda_app, client=None, server_host='192.168.123.53', server_port=9999):
        """
        :param panda_app: ссылка на главное приложение Panda3D
        :param client: готовый экземпляр PerlinMeshClient (если None, будет создан новый)
        :param server_host: адрес сервера (используется только если client не передан)
        :param server_port: порт сервера (используется только если client не передан)
        """
        self.panda_app = panda_app
        self.last_target_model_trimesh = None
        self.last_best_z = None
        self.test_perlin_mesh = None
        self.last_grid_size = 48
        self.perlin_vertices_before_displace = None
        self.perlin_texcoords_before_displace = None
        self.processed_model = None
        self.current_display_model = None
        self.metrics_history = []

        # Создаём клиента, если он не передан
        if client is None:
            print(f"Создание нового клиента PerlinMeshClient ({server_host}:{server_port})")
            self.client = PerlinMeshClient(host=server_host, port=server_port, timeout=60.0)
        else:
            self.client = client
    
    def generate_perlin_mesh(self, grid_size=48):
        """Генерация перлин-меша через удалённый сервер"""
        print(f"Генерация перлин-меша (grid_size={grid_size})...")
        
        total_start = time.time()
        self.last_grid_size = grid_size
        
        try:
            # Получаем параметры из CSG операции
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
                
                # Адаптация размера Z (остается без изменений)
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
            
            # Параметры текстур
            texture_repeatX = self.panda_app.current_texture_set.get('textureRepeatX', 1.35)
            texture_repeatY = self.panda_app.current_texture_set.get('textureRepeatY', 3.2)
            
            base_z = pos.getZ()
            
            # Параметры шума
            noise_scale = 4
            octaves = 12
            persistence = 0.01
            lacunarity = 1.0
            seed = random.randint(0, 10000)
            
            # Загружаем height текстуру
            height_texture_path = self._get_height_texture_path()
            height_array, tex_width, tex_height = self._load_height_array(height_texture_path)
            
            strength = self.panda_app.current_texture_set.get('strength', 0.14)
            
            # Замеряем время создания базовых вершин
            base_vertices_start = time.time()
            
            # Подготавливаем базовые вершины
            base_vertices = []
            base_texcoords = []
            
            step_x = size_x / (grid_size - 1) if grid_size > 1 else 0
            step_y = size_y / (grid_size - 1) if grid_size > 1 else 0
            half_size_x = size_x / 2.0
            half_size_y = size_y / 2.0
            
            for y in range(grid_size):
                for x in range(grid_size):
                    world_x = x * step_x - half_size_x
                    world_y = y * step_y - half_size_y
                    
                    # Простой шум для базовой формы
                    value = (math.sin(x / 10.0) * 0.5 + math.cos(y / 10.0) * 0.5 + 1.0) / 2.0
                    
                    height_scale = size_z * 0.42
                    world_z = base_z + value * height_scale
                    
                    base_vertices.append((world_x, world_y, world_z))
                    
                    # UV координаты
                    normalized_u = x / (grid_size - 1) if grid_size > 1 else 0.0
                    normalized_v = y / (grid_size - 1) if grid_size > 1 else 0.0
                    u = normalized_u * texture_repeatX
                    v = normalized_v * texture_repeatY
                    base_texcoords.append((u, v))
            
            base_vertices_end = time.time()
            base_vertices_time = base_vertices_end - base_vertices_start
            
            # Сохраняем данные
            self.perlin_vertices_before_displace = base_vertices.copy()
            self.perlin_texcoords_before_displace = base_texcoords.copy()
            
            # Создаем клиент и отправляем запрос
            client = PerlinMeshClient(host='192.168.123.53', port=9999, timeout=60.0)
            
            print("Отправка запроса на сервер...")
            result = client.send_perlin_request(
                grid_size=grid_size,
                size_x=size_x,
                size_y=size_y,
                size_z=size_z,
                base_z=base_z,
                noise_scale=noise_scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                seed=seed,
                texture_repeatX=texture_repeatX,
                texture_repeatY=texture_repeatY,
                strength=strength,
                height_array=height_array,
                vertices_before=base_vertices,
                texcoords_before=base_texcoords
            )
            
            if result is None:
                raise Exception("Не удалось сгенерировать меш на удалённом сервере")
            
            vertices, normals, texcoords = result
            
            # Замеряем время применения falloff
            falloff_start = time.time()
            
            vertices_list = vertices.tolist()
            vertices_list = self._apply_falloff(vertices_list, size_x, size_y, base_z, self._get_falloff_config())
            vertices = np.array(vertices_list, dtype=np.float32)
            
            falloff_end = time.time()
            falloff_time = falloff_end - falloff_start
            
            # Замеряем время создания меша
            mesh_creation_start = time.time()
            
            mesh = self._create_mesh_from_data(vertices, normals, texcoords, grid_size, "perlin_remote_mesh")
            
            mesh_creation_end = time.time()
            mesh_creation_time = mesh_creation_end - mesh_creation_start
            
            # Получаем метрики от клиента
            client_metrics = client.get_metrics()
            
            # Выводим общие метрики
            total_time = time.time() - total_start
            
            print(f"\n{'='*60}")
            print("МЕТРИКИ ГЕНЕРАЦИИ ПЕРЛИН-МЕША:")
            print(f"{'='*60}")
            print(f"Grid Size:                 {grid_size}")
            print(f"Размер меша:               {size_x:.2f}x{size_y:.2f}x{size_z:.2f}")
            print(f"{'='*60}")
            print(f"• Базовые вершины:         {base_vertices_time*1000:.1f} мс")
            print(f"• Применение falloff:      {falloff_time*1000:.1f} мс")
            print(f"• Создание геометрии:      {mesh_creation_time*1000:.1f} мс")
            
            if client_metrics:
                print(f"• Запрос к серверу:        {client_metrics.get('total_time', 0)*1000:.1f} мс")
                print(f"  ├─ Подготовка:          {client_metrics.get('prep_time', 0)*1000:.1f} мс")
                print(f"  ├─ Соединение:          {client_metrics.get('connect_time', 0)*1000:.1f} мс")
                print(f"  ├─ Отправка:            {client_metrics.get('send_time', 0)*1000:.1f} мс")
                print(f"  ├─ Получение:           {client_metrics.get('recv_time', 0)*1000:.1f} мс")
                print(f"  ├─ Парсинг:             {client_metrics.get('parse_time', 0)*1000:.1f} мс")
                print(f"  └─ Конвертация:         {client_metrics.get('convert_time', 0)*1000:.1f} мс")
                print(f"• Серверная обработка:    ~{client_metrics.get('server_time', 0)*1000:.1f} мс")
            
            print(f"{'='*60}")
            print(f"ОБЩЕЕ ВРЕМЯ:               {total_time*1000:.1f} мс ({total_time:.3f} сек)")
            print(f"ВЕРШИН В МЕШЕ:             {len(vertices)}")
            print(f"{'='*60}\n")
            
            # Сохраняем метрики в историю
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'grid_size': grid_size,
                'total_time': total_time,
                'base_vertices_time': base_vertices_time,
                'falloff_time': falloff_time,
                'mesh_creation_time': mesh_creation_time,
                'client_metrics': client_metrics,
                'vertex_count': len(vertices)
            })
            
            # Ограничиваем историю последними 100 записями
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            return mesh
            
        except Exception as e:
            print(f"Ошибка при генерации меша: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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
        """Генерация перлин-меша на основе CSG операции (с использованием сервера)"""
        # Удаляем старые модели
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

        # Находим целевую модель
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

        # Подготавливаем целевую сетку для булевой операции (масштабирование и т.д.)
        target_model_trimesh = self._prepare_target_model_for_boolean(target_model)
        if target_model_trimesh is None:
            print("Не удалось подготовить целевую сетку")
            return False
        self.last_target_model_trimesh = target_model_trimesh

        # Сброс позиции и масштаба
        target_model.setScale(1.0, 1.0, 1.0)
        target_model.setPos(0.0, 0.0, 0.0)

        # Загружаем целевую сетку на сервер и получаем session_id
        print("Загрузка целевой сетки на сервер...")
        session_id = self.client.upload_target_mesh(
            target_model_trimesh.vertices,
            target_model_trimesh.faces
        )
        if session_id is None:
            print("Не удалось загрузить целевую сетку на сервер")
            return False

        # Генерируем детальный перлин-меш (grid_size=256)
        perlin_detailed_np = self.generate_perlin_mesh(grid_size=256)
        # Предварительный поиск оптимальной Z позиции (используем сессию)
        best_z = self.find_best_z_position(
            perlin_detailed_np,   # сетка для позиционирования
            target_model_trimesh, # не используется напрямую, передаём для справки
            self.panda_app.Target_Volume,
            session_id=session_id,  # передаём сессию
            initial_z=perlin_detailed_np.getZ()
        )
        self.last_best_z = best_z

        # Устанавливаем финальную позицию
        perlin_detailed_np.setPos(0, 0, best_z)

        # Получаем тримеш перлина
        perlin_trimesh = self.panda_app.panda_to_trimesh(perlin_detailed_np)

        # Выполняем булеву разность на сервере
        print("Выполнение булевой разности на сервере...")
        result_verts, result_faces, metrics = self.client.boolean_difference_with_session(
            session_id,
            perlin_trimesh.vertices,
            perlin_trimesh.faces
        )

        if result_verts is None or result_faces is None:
            print("Булева разность вернула пустой результат")
            perlin_detailed_np.removeNode()
            return False

        # Преобразуем результат в Panda3D модель
        result_trimesh = trimesh.Trimesh(vertices=result_verts, faces=result_faces)
        self.panda_app.final_model = self.panda_app.trimesh_to_panda(result_trimesh)

        # Применяем текстуры и материал
        self._apply_textures_and_material(self.panda_app.final_model)

        # Скрываем промежуточные модели
        target_model.hide()
        perlin_detailed_np.hide()

        print("Финальная модель создана успешно")
        return True
    
    def find_best_z_position(self, mesh_np, target_model_trimesh, target_volume,
                             session_id=None, initial_z=0):
        """
        Поиск оптимальной Z-позиции меша для достижения целевого объема.
        Использует сессию на сервере для ускорения.
        """
        tolerance = 0.2
        min_z = -2
        max_z = 2
        max_iterations = 50

        best_z = initial_z
        best_volume = None
        best_error = float('inf')

        # Убедимся, что у нас есть session_id
        if session_id is None:
            # Если сессия не передана, создаём временную
            session_id = self.client.upload_target_mesh(
                target_model_trimesh.vertices,
                target_model_trimesh.faces
            )
            if session_id is None:
                print("Не удалось создать сессию для поиска Z")
                return initial_z
            temp_session = True
        else:
            temp_session = False

        # Начальные точки поиска
        search_points = [min_z + (max_z - min_z) * i / 10 for i in range(11)]
        search_volumes = []

        print("=== Поиск оптимальной Z позиции ===")
        print(f"Целевой объем: {target_volume}")
        print(f"Диапазон поиска: [{min_z}, {max_z}]")

        for z in search_points:
            mesh_np.setPos(0, 0, z)
            perlin_trimesh = self.panda_app.panda_to_trimesh(mesh_np)

            # Отправляем запрос на сервер
            _, _, metrics = self.client.boolean_difference_with_session(
                session_id,
                perlin_trimesh.vertices,
                perlin_trimesh.faces
            )
            volume = metrics.get('volume', 0.0)
            error = abs(volume - target_volume)

            print(f"  z={z:.4f}: объем={volume:.6f}, ошибка={error:.6f}")
            search_volumes.append((z, volume, error))

            if error < best_error:
                best_error = error
                best_z = z
                best_volume = volume
                print(f"    -> НОВЫЙ ЛУЧШИЙ! best_z={best_z:.4f}, error={best_error:.6f}")

        if best_error <= tolerance:
            mesh_np.setPos(0, 0, best_z)
            print(f"Достигнута требуемая точность! best_z={best_z:.4f}")
            if temp_session:
                # Сессия больше не нужна, но она сама удалится по таймауту
                pass
            return best_z

        # Уточнение методом золотого сечения
        search_volumes.sort(key=lambda x: x[2])
        best_points = search_volumes[:3]

        if len(best_points) >= 2:
            z_values = [p[0] for p in best_points]
            min_search_z = max(min_z, min(z_values) - 0.1)
            max_search_z = min(max_z, max(z_values) + 0.1)

            phi = 0.618
            a = min_search_z
            b = max_search_z
            x1 = b - phi * (b - a)
            x2 = a + phi * (b - a)

            iteration = 0
            while (b - a) > 0.01 and iteration < max_iterations:
                iteration += 1

                # Оценка в точках x1 и x2
                mesh_np.setPos(0, 0, x1)
                perlin_trimesh1 = self.panda_app.panda_to_trimesh(mesh_np)
                _, _, metrics1 = self.client.boolean_difference_with_session(
                    session_id,
                    perlin_trimesh1.vertices,
                    perlin_trimesh1.faces
                )
                vol1 = metrics1.get('volume', 0.0)
                err1 = abs(vol1 - target_volume)

                mesh_np.setPos(0, 0, x2)
                perlin_trimesh2 = self.panda_app.panda_to_trimesh(mesh_np)
                _, _, metrics2 = self.client.boolean_difference_with_session(
                    session_id,
                    perlin_trimesh2.vertices,
                    perlin_trimesh2.faces
                )
                vol2 = metrics2.get('volume', 0.0)
                err2 = abs(vol2 - target_volume)

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
                else:
                    a = x1
                    x1 = x2
                    err1 = err2
                    x2 = a + phi * (b - a)

        mesh_np.setPos(0, 0, best_z)
        print(f"Поиск завершён. best_z={best_z:.4f}, объем={best_volume:.6f}, ошибка={best_error:.6f}")

        if temp_session:
            # Сессия временная, можно не удалять явно
            pass

        return best_z
    
    def _load_height_array(self, height_texture_path):
        """Загружает и обрабатывает текстуру высот"""
        height_image = Image.open(height_texture_path).convert('L')
        height_array = np.array(height_image, dtype=np.float32)
        tex_height, tex_width = height_array.shape
        
        # Нормализация
        height_min = np.min(height_array)
        height_max = np.max(height_array)
        if height_max > height_min:
            height_array = (height_array - height_min) / (height_max - height_min)
        
        # Гамма-коррекция
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
    
    def _create_mesh_from_data(self, vertices, normals, texcoords, grid_size, name="perlin_mesh"):
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
            texcoord_writer.addData2f(texcoords[i][0], texcoords[i][1])
        
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
        # base_material.setDiffuse((0.4, 0.4, 0.4, 1.0))
        # base_material.setAmbient((0.7, 0.7, 0.7, 1.0))
        # base_material.setSpecular((0.1, 0.1, 0.1, 1.0))
        # base_material.setShininess(5.0)
        # base_material.setRoughness(0.85)
        # base_material.setMetallic(0.0)
        # base_material.setRefractiveIndex(1.5)
        model_np.setMaterial(base_material, 1)
        
        model_np.setShaderAuto()
        model_np.setTwoSided(True)
        model_np.setBin("fixed", 0)
        model_np.setDepthOffset(1)