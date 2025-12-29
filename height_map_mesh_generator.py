import numpy as np
from PIL import Image
from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, GeomTriangles, GeomVertexReader,
    Material, NodePath
)
import noise  
import random
from scipy.interpolate import Rbf, griddata
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter  
import sys
import traceback

def format_size(bytes_size):
    """Форматирование размера в читаемом виде"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

class HeightMapMeshGenerator:
    def __init__(self, app, image_path=None):
        self.app = app
        self.image_path = image_path or "height_example/No-visible-Depth.png"
        self.height_map = None
        self.mask = None
        self.height_scale = 0.0  
        self.alpha_threshold = 0.1
        self.mesh_node = None
        self.use_alpha_channel = True
        self.mesh_rotation = 0 

        self.extrapolation_enabled = True

        self.source_offset_x = 0.0
        self.source_offset_y = 0.0
        self.source_offset_z = 0.0

        self.source_rotation_x = 0.0
        self.source_rotation_y = 0.0
        self.source_rotation_z = 0.0

        self.target_width = 2.5  
        self.target_height = 5.0  
        self.grid_resolution = 100
        self.base_height = 0.0  
        
        self.noise_scale = 4.0  
        self.noise_strength = 0.42  
        self.noise_octaves = 12  
        self.noise_persistence = 0.01
        self.noise_lacunarity = 1.0  
        self.noise_seed = random.randint(0, 10000)

        self.displacement_texture_path = "textures/stones_8k/rocks_ground_01_disp_4k.jpg"
        self.displacement_strength = 0.14
        self.texture_repeatX = 1.35
        self.texture_repeatY = 3.2
        self.use_displacement = True
        
        self.interpolation_method = 'rbf'  
        self.rbf_smooth = 0.1 
        self.use_smoothing = True
        self.smoothing_iterations = 2 
        
        self.adaptive_lift_enabled = True 
        self.base_distance = 0.5 
        self.min_distance = 0.1  
        self.max_distance = 3.0  
        self.lift_intensity = 1.0
        
        self.lift_smoothing_enabled = True 
        self.lift_smoothing_sigma = 2.0
        self.lift_blur_enabled = True  
        self.lift_blur_radius = 3 
        
        self.source_mesh_smoothing_enabled = True
        self.source_mesh_smoothing_iterations = 1
        self.source_mesh_smoothing_sigma = 0.5  
        self.source_mesh_edge_preserving = True  
        
        print(f"[DEBUG] Инициализирован HeightMapMeshGenerator с изображением: {self.image_path}")
        print(f"[DEBUG] Планируемое разрешение сетки: {self.grid_resolution}x{self.grid_resolution}")
        
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
                print(f"[DEBUG] Альфа-канал создан: {alpha.shape}, размер: {format_size(alpha.nbytes)}")
                
                if img.mode != 'L':
                    img_gray = img.convert('L')
                    height_data = np.array(img_gray, dtype=np.float32) / 255.0
                else:
                    height_data = np.array(img, dtype=np.float32) / 255.0
                
                mask = alpha > self.alpha_threshold
                
                self.height_map = height_data * mask
                self.mask = mask  
                print(f"[DEBUG] Height map создан: {self.height_map.shape}, размер: {format_size(self.height_map.nbytes)}")
                print(f"[DEBUG] Mask создан: {self.mask.shape}, размер: {format_size(self.mask.nbytes)}")
                print(f"[DEBUG] Пикселей в маске: {np.sum(mask)} из {mask.size}")
            else:
                print(f"[DEBUG] Изображение без альфа-канала")
                img_gray = img.convert('L')
                self.height_map = np.array(img_gray, dtype=np.float32) / 255.0
                
                mask = self.height_map > 0.05
                self.mask = mask
                print(f"[DEBUG] Height map создан (без альфа): {self.height_map.shape}, размер: {format_size(self.height_map.nbytes)}")
                print(f"[DEBUG] Mask создан: {self.mask.shape}, размер: {format_size(self.mask.nbytes)}")
                print(f"[DEBUG] Пикселей в маске: {np.sum(mask)} из {mask.size}")
            
            # Закрываем изображение для освобождения памяти
            img.close()
            print(f"[DEBUG] Загрузка height map завершена успешно")
            return True
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки height map: {e}")
            traceback.print_exc()
            return False
        
    def _apply_rotation_xyz(self, x, y, z, rot_x, rot_y, rot_z):
        """Применить вращение по трем осям в порядке: Z, Y, X"""
        # Вращение вокруг оси Z
        angle_z_rad = np.radians(rot_z)
        cos_z = np.cos(angle_z_rad)
        sin_z = np.sin(angle_z_rad)
        x1 = x * cos_z - y * sin_z
        y1 = x * sin_z + y * cos_z
        z1 = z
        
        # Вращение вокруг оси Y
        angle_y_rad = np.radians(rot_y)
        cos_y = np.cos(angle_y_rad)
        sin_y = np.sin(angle_y_rad)
        x2 = x1 * cos_y + z1 * sin_y
        y2 = y1
        z2 = -x1 * sin_y + z1 * cos_y
        
        # Вращение вокруг оси X
        angle_x_rad = np.radians(rot_x)
        cos_x = np.cos(angle_x_rad)
        sin_x = np.sin(angle_x_rad)
        x3 = x2
        y3 = y2 * cos_x - z2 * sin_x
        z3 = y2 * sin_x + z2 * cos_x
        
        return x3, y3, z3
    
    def _apply_rotation(self, x, y, z):
        angle_rad = np.radians(self.mesh_rotation)
        
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        new_y = y * cos_angle - z * sin_angle
        new_z = y * sin_angle + z * cos_angle
        
        return x, new_y, new_z
    
    def _generate_perlin_noise(self, x, y, use_octaves=True):
        if use_octaves:
            value = 0.0
            max_value = 0.0
            frequency = 1.0
            amplitude = 1.0
            
            for i in range(self.noise_octaves):
                nx = x * frequency + self.noise_seed
                ny = y * frequency + self.noise_seed
                
                octave_value = noise.pnoise2(nx, ny, 
                                            octaves=1,  
                                            persistence=self.noise_persistence,
                                            lacunarity=self.noise_lacunarity,
                                            repeatx=1024, repeaty=1024, 
                                            base=0)
                
                value += octave_value * amplitude
                max_value += amplitude
                
                frequency *= self.noise_lacunarity
                amplitude *= self.noise_persistence
            
            if max_value > 0:
                value /= max_value
                
        else:
            nx = x * self.noise_scale + self.noise_seed
            ny = y * self.noise_scale + self.noise_seed
            value = noise.pnoise2(nx, ny, octaves=1, base=0)
        
        return value
    
    def _smooth_heightfield(self, heightfield, iterations=1):
        print(f"[DEBUG] Начало сглаживания heightfield: {heightfield.shape}, итераций: {iterations}")
        smoothed = heightfield.copy()
        h, w = smoothed.shape
        
        for iteration in range(iterations):
            print(f"[DEBUG] Итерация сглаживания {iteration + 1}/{iterations}")
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
            
            from scipy.ndimage import binary_dilation
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
        print(f"[DEBUG] Начало адаптивного подъема вершин")
        if len(boundary_points) == 0:
            print(f"[DEBUG] Нет граничных точек, подъем не требуется")
            return height_grid, np.zeros_like(source_mask, dtype=bool)
        
        self._current_x_grid = x_grid
        self._current_y_grid = y_grid
        
        boundary_points_array = np.array(boundary_points)
        boundary_heights = boundary_points_array[:, 2]
        
        base_distance, min_distance, max_distance = self._calculate_adaptive_distance_parameters(boundary_heights)
        
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
            use_kdtree = True
            print(f"[DEBUG] KDTree построен для адаптивного подъема")
        except Exception as e:
            use_kdtree = False
            print(f"[DEBUG] Ошибка построения KDTree: {e}, используем линейный поиск")
        
        min_boundary_height = np.min(boundary_heights)
        max_boundary_height = np.max(boundary_heights)
        height_range = max_boundary_height - min_boundary_height
        
        corrected_height = height_grid.copy()
        
        lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        lifted_vertices_count = 0
        total_lift_amount = 0.0
        min_lift = float('inf')
        max_lift = float('-inf')
        
        total_vertices = np.sum(~source_mask)
        print(f"[DEBUG] Всего вершин для проверки подъема: {total_vertices}")
        
        processed_vertices = 0
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                if source_mask[i, j]:
                    continue
                
                x = x_grid[j]
                y = y_grid[i]
                current_z = height_grid[i, j]
                
                if use_kdtree:
                    distances, indices = boundary_tree.query([x, y], k=1)
                    nearest_idx = indices
                    nearest_dist = distances
                    nearest_point = boundary_points_array[nearest_idx]
                else:
                    distances = np.sqrt((boundary_points_array[:, 0] - x)**2 + 
                                      (boundary_points_array[:, 1] - y)**2)
                    nearest_idx = np.argmin(distances)
                    nearest_dist = distances[nearest_idx]
                    nearest_point = boundary_points_array[nearest_idx]
                
                height_diff = nearest_point[2] - current_z
                
                if height_diff > 0 and nearest_dist <= max_distance:
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
                        
                        t = normalized_distance
                        smooth_factor = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
                        
                        lift_amount = height_diff * smooth_factor * self.lift_intensity
                        corrected_height[i, j] += lift_amount
                        
                        lifted_mask[i, j] = True
                        
                        lifted_vertices_count += 1
                        total_lift_amount += lift_amount
                        
                        if lift_amount < min_lift:
                            min_lift = lift_amount
                        if lift_amount > max_lift:
                            max_lift = lift_amount
                
                processed_vertices += 1
                if processed_vertices % 10000 == 0:
                    print(f"[DEBUG] Обработано {processed_vertices}/{total_vertices} вершин ({processed_vertices/total_vertices*100:.1f}%)")
        
        if lifted_vertices_count > 0:
            avg_lift = total_lift_amount / lifted_vertices_count
            print(f"[DEBUG] Адаптивный подъем завершен:")
            print(f"[DEBUG]   Поднято вершин: {lifted_vertices_count} из {total_vertices}")
            print(f"[DEBUG]   Средний подъем: {avg_lift:.4f}")
            print(f"[DEBUG]   Минимальный подъем: {min_lift:.4f}")
            print(f"[DEBUG]   Максимальный подъем: {max_lift:.4f}")
        else:
            print(f"[DEBUG] Адаптивный подъем: ни одна вершина не была поднята")
        
        return corrected_height, lifted_mask
    
    def create_unified_perlin_mesh_with_lift(self, world_position=(0, 0, 2), source_scale_x=None,
                                        source_scale_y=None, source_scale_z=None,
                                        source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0, 
                                        source_rotation_x=None, source_rotation_y=None, source_rotation_z=None):
        print(f"[DEBUG] ====== НАЧАЛО СОЗДАНИЯ МЕША ======")
        print(f"[DEBUG] Позиция в мире: {world_position}")
        print(f"[DEBUG] Разрешение сетки: {self.grid_resolution}")
        
        if self.height_map is None:
            print(f"[DEBUG] Height map не загружен, начинаем загрузку...")
            if not self.load_height_map():
                print(f"[ERROR] Не удалось загрузить height map")
                return None
        
        h, w = self.height_map.shape
        print(f"[DEBUG] Размер height map: {h}x{w}, общее количество пикселей: {h*w}")
        
        if not hasattr(self, 'mask'):
            print(f"[DEBUG] Маска не создана, создаем из height map")
            self.mask = self.height_map > self.alpha_threshold
        
        y_indices, x_indices = np.where(self.mask)
        
        print(f"[DEBUG] Пикселей в маске: {len(y_indices)} из {h*w} ({len(y_indices)/(h*w)*100:.1f}%)")
        
        if len(y_indices) == 0:
            print(f"[ERROR] Нет пикселей в маске")
            return None
        
        if source_scale_x is None:
            source_scale_x = -2.0
        if source_scale_y is None:
            source_scale_y = 2.0
        if source_scale_z is None:
            source_scale_z = self.height_scale
        
        rot_x = source_rotation_x if source_rotation_x is not None else self.source_rotation_x
        rot_y = source_rotation_y if source_rotation_y is not None else self.source_rotation_y
        rot_z = source_rotation_z if source_rotation_z is not None else self.source_rotation_z
        
        print(f"[DEBUG] Масштаб источника: ({source_scale_x}, {source_scale_y}, {source_scale_z})")
        print(f"[DEBUG] Вращение источника: ({rot_x}, {rot_y}, {rot_z})")
        
        # Создание source_points
        print(f"[DEBUG] Создание source_points из {len(y_indices)} пикселей...")
        source_points = []
        for idx, (y, x) in enumerate(zip(y_indices, x_indices)):
            height = self.height_map[y, x]
            
            # Применяем масштабирование
            world_x = (x / w - 0.5) * source_scale_x
            world_y = (y / h - 0.5) * source_scale_y
            world_z = height * source_scale_z

            # world_x = -world_x
            world_y = -world_y
            
            # Применяем вращение
            world_x, world_y, world_z = self._apply_rotation_xyz(world_x, world_y, world_z, rot_x, rot_y, rot_z)
            
            # Применяем смещение
            world_x += source_offset_x
            world_y += source_offset_y
            world_z += source_offset_z
            
            # Добавляем глобальную позицию
            world_x += world_position[0]
            world_y += world_position[1]
            world_z += world_position[2]
            
            source_points.append((world_x, world_y, world_z))
            
            if idx % 10000 == 0 and idx > 0:
                print(f"[DEBUG] Обработано {idx}/{len(y_indices)} пикселей ({idx/len(y_indices)*100:.1f}%)")
        
        source_points = np.array(source_points)
        print(f"[DEBUG] Source_points создан: {source_points.shape}, размер: {format_size(source_points.nbytes)}")
        
        # Создание сетки
        print(f"[DEBUG] Создание сетки...")
        if not self.extrapolation_enabled:
            min_x = world_position[0] - source_scale_x/2 if source_scale_x else world_position[0] - 1.0
            max_x = world_position[0] + source_scale_x/2 if source_scale_x else world_position[0] + 1.0
            min_y = world_position[1] - source_scale_y/2 if source_scale_y else world_position[1] - 1.0
            max_y = world_position[1] + source_scale_y/2 if source_scale_y else world_position[1] + 1.0
        else:
            min_x = world_position[0] - self.target_width / 2
            max_x = world_position[0] + self.target_width / 2
            min_y = world_position[1] - self.target_height / 2
            max_y = world_position[1] + self.target_height / 2
        
        x_grid = np.linspace(min_x, max_x, self.grid_resolution)
        y_grid = np.linspace(min_y, max_y, self.grid_resolution)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        print(f"[DEBUG] Сетка создана: {grid_x.shape}, размер: {format_size(grid_x.nbytes + grid_y.nbytes)}")
        print(f"[DEBUG] x_grid: {len(x_grid)} точек, y_grid: {len(y_grid)} точек")
        
        source_height_grid = np.full((len(y_grid), len(x_grid)), np.nan, dtype=np.float32)
        source_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
        
        print(f"[DEBUG] Заполнение source_height_grid и source_mask...")
        for wx, wy, wz in source_points:
            grid_x_idx = np.argmin(np.abs(x_grid - wx))
            grid_y_idx = np.argmin(np.abs(y_grid - wy))
            
            if (abs(x_grid[grid_x_idx] - wx) < (x_grid[1] - x_grid[0]) * 0.5 and
                abs(y_grid[grid_y_idx] - wy) < (y_grid[1] - y_grid[0]) * 0.5):
                
                source_height_grid[grid_y_idx, grid_x_idx] = wz
                source_mask[grid_y_idx, grid_x_idx] = True
        
        print(f"[DEBUG] Пикселей в source_mask: {np.sum(source_mask)} из {source_mask.size}")
        
        # Поиск граничных точек
        print(f"[DEBUG] Поиск граничных точек...")
        boundary_points = []
        boundary_count = 0
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                if source_mask[i, j]:
                    is_boundary = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < len(y_grid) and 0 <= nj < len(x_grid):
                                if not source_mask[ni, nj]:
                                    is_boundary = True
                                    break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        boundary_points.append((x_grid[j], y_grid[i], source_height_grid[i, j]))
                        boundary_count += 1
        
        print(f"[DEBUG] Найдено граничных точек: {boundary_count}")
        
        # Генерация шума Перлина
        print(f"[DEBUG] Генерация шума Перлина...")
        perlin_height = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)

        # Генерируем перлин-шум только если включена экстраполяция
        if self.extrapolation_enabled:
            for i in range(len(y_grid)):
                for j in range(len(x_grid)):
                    x = x_grid[j]
                    y = y_grid[i]
                    
                    noise_value = self._generate_perlin_noise(x, y)
                    noise_height = (noise_value + 1.0) / 2.0 * self.noise_strength
                    perlin_height[i, j] = noise_height
        else:
            # Без экстраполяции - плоская поверхность
            perlin_height.fill(0.0)
        
        base_height_grid = perlin_height + self.base_height
        
        if len(source_points) > 0:
            key_x = source_points[:, 0]
            key_y = source_points[:, 1]
            key_z = source_points[:, 2]
            
            try:
                # Без экстраполяции - только исходные данные
                corrected_height = base_height_grid.copy()
                
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j] and not np.isnan(source_height_grid[i, j]):
                            corrected_height[i, j] = source_height_grid[i, j]
                
            except Exception as e:
                corrected_height = base_height_grid.copy()
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j] and not np.isnan(source_height_grid[i, j]):
                            corrected_height[i, j] = source_height_grid[i, j]
        else:
            corrected_height = base_height_grid
        
        print(f"[DEBUG] Высота после интерполяции: {corrected_height.shape}, размер: {format_size(corrected_height.nbytes)}")
        
        if self.source_mesh_smoothing_enabled and np.any(source_mask):
            print(f"[DEBUG] Применение сглаживания исходного меша...")
            corrected_height = self._smooth_source_mesh(
                corrected_height, source_mask,
                sigma=self.source_mesh_smoothing_sigma,
                iterations=self.source_mesh_smoothing_iterations,
                preserve_edges=self.source_mesh_edge_preserving
            )
            
            if self.source_mesh_smoothing_sigma > 1.0:  
                print(f"[DEBUG] Постобработка граничной зоны...")
                corrected_height = self._postprocess_boundary_zone(
                    corrected_height, source_mask, 
                    boundary_width=int(self.source_mesh_smoothing_sigma * 2)
                )
        
        if self.extrapolation_enabled and self.adaptive_lift_enabled and len(boundary_points) > 0:
            corrected_height, lifted_mask = self._apply_adaptive_vertex_lift(
                corrected_height, source_mask, x_grid, y_grid, boundary_points
            )
            
            if self.lift_smoothing_enabled and np.any(lifted_mask):
                corrected_height = self._smooth_lifted_area(
                    corrected_height, lifted_mask, sigma=self.lift_smoothing_sigma
                )
            
            if self.lift_blur_enabled and self.lift_blur_radius > 0:
                corrected_height = self._blur_lift_boundary(
                    corrected_height, source_mask, boundary_points, blur_radius=self.lift_blur_radius
                )
        else:
            lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        if self.use_smoothing and self.smoothing_iterations > 0:
            print(f"[DEBUG] Финальное сглаживание...")
            corrected_height = self._smooth_heightfield(corrected_height, self.smoothing_iterations)
        
        # ДОБАВЛЕНО: Применение displacement к высотам
        if hasattr(self, 'use_displacement') and self.use_displacement:
            print(f"[DEBUG] Применение displacement к высотам...")
            corrected_height = self._apply_displacement_to_grid(
                corrected_height, x_grid, y_grid,
                self.displacement_strength,
                self.texture_repeatX,
                self.texture_repeatY
            )
        
        # Создание геометрии
        print(f"[DEBUG] Создание геометрии меша...")
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        vertices = []
        vertex_indices = {}
        
        print(f"[DEBUG] Создание вершин...")
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                z = corrected_height[i, j]
                
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1) 
                
                # UV-координаты с учетом повторения текстуры
                normalized_u = j / (len(x_grid) - 1) if len(x_grid) > 1 else 0.0
                normalized_v = i / (len(y_grid) - 1) if len(y_grid) > 1 else 0.0
                u = normalized_u * self.texture_repeatX
                v = normalized_v * self.texture_repeatY
                texcoord.addData2f(u, v)
                
                vertex_indices[(i, j)] = len(vertices)
                vertices.append((x, y, z))
            
            if i % 10 == 0 and i > 0:
                print(f"[DEBUG] Создано вершин для {i}/{len(y_grid)} строк ({i/len(y_grid)*100:.1f}%)")
        
        print(f"[DEBUG] Всего создано вершин: {len(vertices)}")
        
        triangles = GeomTriangles(Geom.UHStatic)
        
        print(f"[DEBUG] Создание треугольников...")
        total_triangles = (len(y_grid) - 1) * (len(x_grid) - 1) * 2
        created_triangles = 0
        
        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                v1 = vertex_indices[(i, j)]
                v2 = vertex_indices[(i, j + 1)]
                v3 = vertex_indices[(i + 1, j)]
                v4 = vertex_indices[(i + 1, j + 1)]
                
                triangles.addVertices(v1, v2, v3)
                triangles.addVertices(v2, v4, v3)
                
                created_triangles += 2
                
                if created_triangles % 5000 == 0:
                    print(f"[DEBUG] Создано {created_triangles}/{total_triangles} треугольников ({created_triangles/total_triangles*100:.1f}%)")
        
        triangles.closePrimitive()
        print(f"[DEBUG] Создано всего треугольников: {created_triangles}")
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        
        print(f"[DEBUG] Расчет нормалей...")
        self._calculate_normals(vdata, geom)
        
        print(f"[DEBUG] ====== СОЗДАНИЕ МЕША ЗАВЕРШЕНО ======")
        return node
    
    def _load_displacement_texture(self):
        """Загружает и обрабатывает текстуру высот для displacement (аналогично perlin_mesh_generator.py)"""
        if not hasattr(self, 'displacement_texture_path') or self.displacement_texture_path is None:
            print(f"[WARNING] Displacement texture path не указан")
            return None, 0, 0
        
        try:
            height_image = Image.open(self.displacement_texture_path).convert('L')
            height_array = np.array(height_image, dtype=np.float32)
            tex_height, tex_width = height_array.shape
            
            height_min = np.min(height_array)
            height_max = np.max(height_array)
            if height_max - height_min > 0:
                height_array = (height_array - height_min) / (height_max - height_min)
            else:
                height_array = np.zeros_like(height_array)
            
            height_array = np.power(height_array, 0.7)
            
            print(f"[DEBUG] Displacement texture загружена: {tex_width}x{tex_height}")
            return height_array, tex_width, tex_height
            
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки displacement texture: {e}")
            return None, 0, 0

    def _apply_displacement_to_grid(self, height_grid, x_grid, y_grid, strength, repeatX, repeatY):
        """Применяет displacement к сетке высот (аналогично perlin_mesh_generator.py)"""
        height_array, tex_width, tex_height = self._load_displacement_texture()
        if height_array is None:
            print(f"[WARNING] Не удалось загрузить displacement texture, пропускаем displacement")
            return height_grid
        
        h, w = height_grid.shape
        displaced_grid = height_grid.copy()
        
        print(f"[DEBUG] Применение displacement: сетка {w}x{h}, текстура {tex_width}x{tex_height}, сила={strength}")
        
        for i in range(h):
            for j in range(w):
                # Рассчитываем UV-координаты
                normalized_u = j / (w - 1) if w > 1 else 0.0
                normalized_v = i / (h - 1) if h > 1 else 0.0
                u = normalized_u * repeatX
                v = normalized_v * repeatY
                
                # Повторяем текстуру
                u_repeated = u % 1.0
                v_repeated = v % 1.0
                
                # Преобразуем в координаты текстуры
                tex_x = u_repeated * (tex_width - 1)
                tex_y = v_repeated * (tex_height - 1)
                
                # Билинейная интерполяция
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
                
                # Вычисляем смещение
                displacement = (height_value - 0.5) * strength
                displaced_grid[i, j] += displacement
        
        print(f"[DEBUG] Displacement применен успешно")
        return displaced_grid

    # Дополнительные методы для настройки displacement
    def set_displacement_parameters(self, texture_path, strength=0.14, repeatX=1.35, repeatY=3.2, enabled=True):
        """Установка параметров displacement"""
        self.displacement_texture_path = texture_path
        self.displacement_strength = strength
        self.texture_repeatX = repeatX
        self.texture_repeatY = repeatY
        self.use_displacement = enabled
        print(f"[DEBUG] Displacement параметры: texture={texture_path}, strength={strength}, repeat=({repeatX}, {repeatY}), enabled={enabled}")

    def set_displacement_enabled(self, enabled):
        """Включение/выключение displacement"""
        self.use_displacement = enabled
        print(f"[DEBUG] Displacement enabled: {enabled}")
    
    def _postprocess_boundary_zone(self, height_grid, source_mask, boundary_width=5):
        from scipy.ndimage import distance_transform_edt
        
        print(f"[DEBUG] Постобработка граничной зоны шириной {boundary_width}")
        distance_in = distance_transform_edt(source_mask)
        
        corrected_height = height_grid.copy()
        
        for i in range(height_grid.shape[0]):
            for j in range(height_grid.shape[1]):
                if source_mask[i, j]:
                    dist = distance_in[i, j]
                    
                    if dist < boundary_width:
                        min_boundary_height = height_grid[i, j]
                        
                        for di in range(-boundary_width, boundary_width + 1):
                            for dj in range(-boundary_width, boundary_width + 1):
                                ni, nj = i + di, j + dj
                                if (0 <= ni < height_grid.shape[0] and 
                                    0 <= nj < height_grid.shape[1]):
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
        print(f"[DEBUG] Начало расчета нормалей")
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        
        vertices = []
        while not vertex_reader.isAtEnd():
            pos = vertex_reader.getData3f()
            vertices.append((pos.x, pos.y, pos.z))
        
        print(f"[DEBUG] Вершин для расчета нормалей: {len(vertices)}")
        
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
        
        print(f"[DEBUG] Нормализация нормалей...")
        for i in range(len(normals)):
            nx, ny, nz = normals[i]
            norm = (nx*nx + ny*ny + nz*nz) ** 0.5
            if norm > 0:
                nx /= norm
                ny /= norm
                nz /= norm
            
            nx, ny, nz = self._apply_rotation(nx, ny, nz)
            normal_writer.setData3f(nx, ny, nz)
        
        print(f"[DEBUG] Расчет нормалей завершен")
    
    def set_source_scale(self, scale_x, scale_y, scale_z):
        self.source_scale_x = scale_x
        self.source_scale_y = scale_y
        self.source_scale_z = scale_z

    def set_source_offset(self, offset_x, offset_y, offset_z):
        self.source_offset_x = offset_x
        self.source_offset_y = offset_y
        self.source_offset_z = offset_z

    def add_extended_mesh_to_scene(self, position=(0, 0, 2), source_scale_x=None,
                          source_scale_y=None, source_scale_z=None,
                          source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0,
                          source_rotation_x=None, source_rotation_y=None, source_rotation_z=None):
        print(f"[DEBUG] Добавление расширенного меша на сцену")
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        node = self.create_unified_perlin_mesh_with_lift(position, source_scale_x, source_scale_y, 
                                        source_scale_z, source_offset_x, source_offset_y, source_offset_z,
                                        source_rotation_x, source_rotation_y, source_rotation_z)
        if node:
            self.mesh_node = self.app.render.attachNewNode(node)
            
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            self.mesh_node.setShaderAuto()
            
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "extended_height_map_mesh"
            
            return self.mesh_node
        
        return None
    
    def remove_from_scene(self):
        """Удаление меша со сцены"""
        print(f"[DEBUG] Удаление меша со сцены")
        if self.mesh_node:
            if hasattr(self.app, 'loaded_models') and self.mesh_node in self.app.loaded_models:
                self.app.loaded_models.remove(self.mesh_node)
            
            if hasattr(self.app, 'model_paths'):
                model_id = id(self.mesh_node)
                if model_id in self.app.model_paths:
                    del self.app.model_paths[model_id]
            
            self.mesh_node.removeNode()
            self.mesh_node = None
            print(f"[DEBUG] Меш успешно удален со сцены")
        else:
            print(f"[DEBUG] Меш не найден на сцене")
    
    def set_threshold(self, threshold):
        """Установка порога для определения фона"""
        self.alpha_threshold = threshold
    
    def set_rotation(self, rotation_angle):
        """Установка угла поворота меша (в градусах)"""
        self.mesh_rotation = rotation_angle
    
    def set_extended_area(self, width, height):
        """Установка размера целевой области для экстраполяции"""
        self.target_width = width
        self.target_height = height
    
    def set_base_height(self, height):
        """Установка базовой высоты для экстраполируемой области"""
        self.base_height = height
    
    def set_grid_resolution(self, resolution):
        """Установка разрешения сетки для экстраполяции"""
        print(f"[DEBUG] Установка разрешения сетки: {resolution}")
        self.grid_resolution = resolution
    
    def set_noise_scale(self, scale):
        """Установка масштаба шума Перлина"""
        self.noise_scale = scale
    
    def set_noise_strength(self, strength):
        """Установка силы/амплитуды шума Перлина"""
        self.noise_strength = strength
    
    def set_noise_octaves(self, octaves):
        """Установка количества октав шума Перлина"""
        self.noise_octaves = octaves
    
    def set_noise_persistence(self, persistence):
        """Установка персистентности шума Перлина"""
        self.noise_persistence = persistence
    
    def set_noise_lacunarity(self, lacunarity):
        """Установка лакунарности шума Перлина"""
        self.noise_lacunarity = lacunarity
    
    def set_noise_seed(self, seed):
        """Установка seed для генератора шума Перлина"""
        self.noise_seed = seed
    
    def set_interpolation_method(self, method):
        """Установка метода интерполяции"""
        valid_methods = ['rbf', 'linear', 'cubic']
        if method in valid_methods:
            self.interpolation_method = method

    def set_rbf_smooth(self, smooth):
        """Установка параметра сглаживания для RBF"""
        self.rbf_smooth = max(0.0, smooth)

    def set_adaptive_lift_enabled(self, enabled):
        """Включение/выключение адаптивного подъема вершин"""
        self.adaptive_lift_enabled = enabled

    def set_lift_parameters(self, base_distance=None, min_distance=None, max_distance=None, intensity=None):
        """Установка параметров адаптивного подъема"""
        if base_distance is not None:
            self.base_distance = max(0.1, base_distance)
        if min_distance is not None:
            self.min_distance = max(0.01, min_distance)
        if max_distance is not None:
            self.max_distance = max(0.5, max_distance)
        if intensity is not None:
            self.lift_intensity = max(0.0, intensity)
        
    def set_use_smoothing(self, use_smoothing):
        """Включение/выключение сглаживания"""
        self.use_smoothing = use_smoothing

    def set_smoothing_iterations(self, iterations):
        """Установка количества итераций сглаживания"""
        self.smoothing_iterations = max(0, int(iterations))
    
    def set_lift_smoothing_enabled(self, enabled):
        """Включение/выключение сглаживания поднятой области"""
        self.lift_smoothing_enabled = enabled

    def set_lift_smoothing_sigma(self, sigma):
        """Установка параметра сигма для гауссового сглаживания поднятой области"""
        self.lift_smoothing_sigma = max(0.1, sigma)

    def set_lift_blur_enabled(self, enabled):
        """Включение/выключение размытия границ поднятой области"""
        self.lift_blur_enabled = enabled

    def set_lift_blur_radius(self, radius):
        """Установка радиуса размытия границ поднятой области"""
        self.lift_blur_radius = max(0, int(radius))
    
    def set_source_mesh_smoothing_enabled(self, enabled):
        """Включение/выключение сглаживания исходного меша"""
        self.source_mesh_smoothing_enabled = enabled

    def set_source_mesh_smoothing_iterations(self, iterations):
        """Установка количества итераций сглаживания исходного меша"""
        self.source_mesh_smoothing_iterations = max(0, int(iterations))

    def set_source_mesh_smoothing_sigma(self, sigma):
        """Установка параметра сигма для гауссового сглаживания исходного меша"""
        self.source_mesh_smoothing_sigma = max(0.1, sigma)

    def set_source_mesh_edge_preserving(self, preserve):
        """Включение/выключение сохранения границ исходного меша при сглаживании"""
        self.source_mesh_edge_preserving = preserve

    def set_source_rotation(self, rot_x, rot_y, rot_z):
        """Установка вращения исходной модели"""
        self.source_rotation_x = rot_x
        self.source_rotation_y = rot_y
        self.source_rotation_z = rot_z

    def set_source_rotation_x(self, rot_x):
        """Установка вращения вокруг оси X"""
        self.source_rotation_x = rot_x

    def set_source_rotation_y(self, rot_y):
        """Установка вращения вокруг оси Y"""
        self.source_rotation_y = rot_y

    def set_source_rotation_z(self, rot_z):
        """Установка вращения вокруг оси Z"""
        self.source_rotation_z = rot_z