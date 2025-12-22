import numpy as np
from PIL import Image
from panda3d.core import (
    Geom, GeomNode, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, GeomTriangles, GeomVertexReader,
    Material, NodePath
)
import noise  # Для генерации шума Перлина
import random
from scipy.interpolate import Rbf, griddata  # Для интерполяции
from scipy.spatial import KDTree  # Для быстрого поиска ближайших точек
from scipy.ndimage import gaussian_filter  # Для гауссового сглаживания

class HeightMapMeshGenerator:
    """Генератор меша из карты высот с интеграцией шума Перлина"""
    
    def __init__(self, app, image_path=None):
        self.app = app
        self.image_path = image_path or "height_example/No-visible-Depth.png"
        self.height_map = None
        self.mask = None
        self.height_scale = 2.0  # Масштаб высот (можно регулировать)
        self.alpha_threshold = 0.1  # Порог для определения прозрачности фона
        self.mesh_node = None
        self.use_alpha_channel = True  # Использовать альфа-канал для определения фона
        self.mesh_rotation = -90  # Поворот меша по оси X (в градусах)

        self.source_offset_x = 0.0
        self.source_offset_y = 0.0
        self.source_offset_z = 0.0

        # Параметры для экстраполяции
        self.target_width = 2.5  # Ширина целевой области
        self.target_height = 5.0  # Высота целевой области
        self.grid_resolution = 100  # Разрешение сетки для экстраполяции
        self.base_height = 0.0  # Базовая высота для экстраполируемой области
        
        # Параметры шума Перлина
        self.noise_scale = 4.0  # Масштаб шума (аналог noise_scale = 4 из примера)
        self.noise_strength = 0.42  # Сила/амплитуда шума (влияет на высоту рельефа)
        self.noise_octaves = 12  # Количество октав (аналог octaves = 12)
        self.noise_persistence = 0.01  # Персистентность (аналог persistence = 0.01)
        self.noise_lacunarity = 1.0  # Лакунарность (аналог lacunarity = 1.0)
        self.noise_seed = random.randint(0, 10000)  # Seed для воспроизводимости
        
        # Параметры для унифицированного метода с адаптивным подъемом
        self.interpolation_method = 'rbf'  # Метод интерполяции: 'rbf', 'linear', 'cubic'
        self.rbf_smooth = 0.1  # Параметр сглаживания для RBF интерполяции
        self.use_smoothing = True  # Применять ли дополнительное сглаживание
        self.smoothing_iterations = 2  # Количество итераций сглаживания
        
        # Параметры для адаптивного подъема вершин (из create_extended_mesh)
        self.adaptive_lift_enabled = True  # Включить адаптивный подъем вершин
        self.base_distance = 0.5  # Базовое расстояние влияния
        self.min_distance = 0.1  # Минимальное расстояние влияния
        self.max_distance = 3.0  # Максимальное расстояние влияния
        self.lift_intensity = 1.0  # Интенсивность подъема (коэффициент)
        
        # Параметры для устранения волнообразности
        self.lift_smoothing_enabled = True  # Включить сглаживание поднятого рельефа
        self.lift_smoothing_sigma = 2.0  # Сигма для гауссового сглаживания поднятого рельефа
        self.lift_blur_enabled = True  # Включить размытие границ поднятой области
        self.lift_blur_radius = 3  # Радиус размытия границ
        
        # Параметры для сглаживания исходного меша
        self.source_mesh_smoothing_enabled = True  # Включить сглаживание исходного меша
        self.source_mesh_smoothing_iterations = 1  # Количество итераций сглаживания исходного меша
        self.source_mesh_smoothing_sigma = 0.5  # Сигма для гауссового сглаживания исходного меша
        self.source_mesh_edge_preserving = True  # Сохранять границы исходного меша при сглаживании
        
    def load_height_map(self):
        """Загрузка карты высот с учетом прозрачности"""
        try:
            # Загружаем изображение с альфа-каналом
            img = Image.open(self.image_path)
            
            # Если есть альфа-канал, используем его для определения фона
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img_rgba = img.convert('RGBA')
                r, g, b, a = img_rgba.split()
                
                # Нормализуем альфа-канал для использования как маска
                alpha = np.array(a, dtype=np.float32) / 255.0
                
                # Конвертируем в градации серого для высот
                if img.mode != 'L':
                    img_gray = img.convert('L')
                    height_data = np.array(img_gray, dtype=np.float32) / 255.0
                else:
                    height_data = np.array(img, dtype=np.float32) / 255.0
                
                # Создаем маску: 1 = видимые пиксели, 0 = фон
                mask = alpha > self.alpha_threshold
                
                # Применяем маску к высотным данным
                self.height_map = height_data * mask
                self.mask = mask  # Сохраняем маску для создания меша
                
                print(f"Height map loaded with alpha mask: {self.height_map.shape}")
                print(f"Visible pixels: {np.sum(mask)} out of {mask.size}")
                
            else:
                # Если нет альфа-канала, используем только яркость
                img_gray = img.convert('L')
                self.height_map = np.array(img_gray, dtype=np.float32) / 255.0
                
                # Определяем фон как очень темные пиксели
                mask = self.height_map > 0.05
                self.mask = mask
                
                print(f"Height map loaded without alpha: {self.height_map.shape}")
                print(f"Bright pixels: {np.sum(mask)} out of {mask.size}")
            
            return True
            
        except Exception as e:
            print(f"Error loading height map: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _apply_rotation(self, x, y, z):
        """Применяет поворот к координатам вершины"""
        # Преобразуем угол в радианы
        angle_rad = np.radians(self.mesh_rotation)
        
        # Поворачиваем вокруг оси X
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        new_y = y * cos_angle - z * sin_angle
        new_z = y * sin_angle + z * cos_angle
        
        return x, new_y, new_z
    
    def _generate_perlin_noise(self, x, y, use_octaves=True):
        """
        Генерация значения шума Перлина для точки (x, y)
        
        Args:
            x, y: Координаты точки
            use_octaves: Использовать ли несколько октав шума
            
        Returns:
            Значение шума в диапазоне [-1, 1]
        """
        if use_octaves:
            # Реализация с несколькими октавами (аналогично примеру generate_perlin_mesh)
            value = 0.0
            max_value = 0.0
            frequency = 1.0
            amplitude = 1.0
            
            for i in range(self.noise_octaves):
                # Добавляем seed для каждой октавы, чтобы получить разные паттерны
                nx = x * frequency + self.noise_seed
                ny = y * frequency + self.noise_seed
                
                # Генерируем шум для текущей октавы
                octave_value = noise.pnoise2(nx, ny, 
                                            octaves=1,  # Одна октава за раз
                                            persistence=self.noise_persistence,
                                            lacunarity=self.noise_lacunarity,
                                            repeatx=1024, repeaty=1024, 
                                            base=0)
                
                value += octave_value * amplitude
                max_value += amplitude
                
                # Обновляем параметры для следующей октавы
                frequency *= self.noise_lacunarity
                amplitude *= self.noise_persistence
            
            # Нормализуем значение
            if max_value > 0:
                value /= max_value
                
        else:
            # Простая реализация с одной октавой
            nx = x * self.noise_scale + self.noise_seed
            ny = y * self.noise_scale + self.noise_seed
            value = noise.pnoise2(nx, ny, octaves=1, base=0)
        
        return value
    
    def _smooth_heightfield(self, heightfield, iterations=1):
        """Применяет сглаживание к полю высот"""
        smoothed = heightfield.copy()
        h, w = smoothed.shape
        
        for _ in range(iterations):
            new_heights = smoothed.copy()
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Простой фильтр сглаживания (усреднение 3x3)
                    neighbors = [
                        smoothed[i-1, j-1], smoothed[i-1, j], smoothed[i-1, j+1],
                        smoothed[i, j-1], smoothed[i, j], smoothed[i, j+1],
                        smoothed[i+1, j-1], smoothed[i+1, j], smoothed[i+1, j+1]
                    ]
                    new_heights[i, j] = np.mean(neighbors)
            smoothed = new_heights
        
        return smoothed
    
    def _smooth_source_mesh(self, height_grid, source_mask, sigma=0.5, iterations=1, preserve_edges=True):
        """
        Применяет сглаживание к исходному мешу с защитой границ от проседания
        
        Args:
            height_grid: Поле высот
            source_mask: Маска исходного меша
            sigma: Параметр сглаживания для гауссового фильтра
            iterations: Количество итераций сглаживания
            preserve_edges: Сохранять ли границы исходного меша при сглаживании
            
        Returns:
            Сглаженное поле высот
        """
        if not np.any(source_mask):
            return height_grid
        
        # Создаем копию высот
        smoothed_height = height_grid.copy()
        
        # Создаем расширенную маску для защиты граничной зоны
        # Эта маска будет защищать не только граничные точки, но и область вокруг них
        protected_mask = source_mask.copy()
        
        if preserve_edges:
            # Находим граничные точки исходного меша
            edge_mask = self._create_edge_mask(source_mask)
            
            # Расширяем защищенную область на основе sigma
            # Чем больше sigma, тем шире защищаемая зона
            protection_radius = max(1, int(np.ceil(sigma)))
            
            # Создаем расширенную маску защиты
            from scipy.ndimage import binary_dilation
            protected_mask = binary_dilation(edge_mask, 
                                            structure=np.ones((2*protection_radius+1, 2*protection_radius+1)))
            
            # Ограничиваем защищенную область только исходным мешем
            protected_mask = protected_mask & source_mask
            
            print(f"Created protected zone with radius {protection_radius} (sigma={sigma})")
            print(f"Protected points: {np.sum(protected_mask)}/{np.sum(source_mask)}")
        
        try:
            # ПОДХОД 1: Используем маскированное сглаживание
            # Временное решение: заполняем область вне исходного меша значениями NaN
            # и используем функцию, которая игнорирует NaN при сглаживании
            
            # Создаем копию с NaN для внешней области
            masked_height = height_grid.copy()
            masked_height[~source_mask] = np.nan
            
            for _ in range(iterations):
                # Применяем гауссово сглаживание с обработкой NaN
                gaussian_smoothed = self._gaussian_filter_with_nan(masked_height, sigma)
                
                # Заменяем значения только для незащищенных точек исходного меша
                replace_mask = source_mask & ~protected_mask
                smoothed_height[replace_mask] = gaussian_smoothed[replace_mask]
                
                # Обновляем маскированные высоты для следующей итерации
                masked_height[replace_mask] = smoothed_height[replace_mask]
            
            print(f"Applied masked Gaussian smoothing (sigma={sigma}, iterations={iterations})")
            
        except Exception as e:
            print(f"Masked Gaussian smoothing failed: {e}. Using improved simple smoothing.")
            # Резервный метод: улучшенное сглаживание с защитой границ
            smoothed_height = self._improved_simple_smoothing(
                height_grid, source_mask, protected_mask, 
                sigma, iterations, preserve_edges
            )
        
        return smoothed_height
    
    def _gaussian_filter_with_nan(self, data, sigma):
        """
        Применяет гауссово сглаживание, игнорируя значения NaN
        
        Args:
            data: Входные данные с возможными значениями NaN
            sigma: Параметр сглаживания
            
        Returns:
            Сглаженные данные
        """
        from scipy.ndimage import gaussian_filter
        
        # Создаем маску валидных данных
        valid_mask = ~np.isnan(data)
        
        if not np.any(valid_mask):
            return data
        
        # Заменяем NaN на 0 для фильтрации
        data_filled = np.where(valid_mask, data, 0)
        
        # Применяем гауссов фильтр к данным и маске
        smoothed_data = gaussian_filter(data_filled, sigma=sigma)
        smoothed_weights = gaussian_filter(valid_mask.astype(float), sigma=sigma)
        
        # Защита от деления на ноль
        smoothed_weights = np.where(smoothed_weights > 0, smoothed_weights, 1)
        
        # Восстанавливаем значения
        result = smoothed_data / smoothed_weights
        
        # Сохраняем оригинальные значения там, где веса слишком малы
        result[smoothed_weights < 0.01] = data[smoothed_weights < 0.01]
        
        return result

    def _improved_simple_smoothing(self, height_grid, source_mask, protected_mask, 
                                sigma=0.5, iterations=1, preserve_edges=True):
        """
        Улучшенный метод простого сглаживания с защитой границ
        
        Args:
            height_grid: Поле высот
            source_mask: Маска исходного меша
            protected_mask: Маска защищенных точек
            sigma: Параметр сглаживания
            iterations: Количество итераций
            preserve_edges: Сохранять ли границы
            
        Returns:
            Сглаженное поле высот
        """
        smoothed_height = height_grid.copy()
        
        # Определяем радиус сглаживания на основе sigma
        radius = max(1, int(np.ceil(2 * sigma)))
        
        # Предвычисляем гауссовы веса
        weights = np.zeros((2*radius + 1, 2*radius + 1))
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                distance = np.sqrt(i*i + j*j)
                weights[i+radius, j+radius] = np.exp(-distance*distance / (2 * sigma * sigma))
        
        for iteration in range(iterations):
            new_heights = smoothed_height.copy()
            
            # Находим индексы точек для сглаживания (только исходный меш, не защищенные)
            smooth_indices = np.where(source_mask & ~protected_mask)
            
            for idx in range(len(smooth_indices[0])):
                i, j = smooth_indices[0][idx], smooth_indices[1][idx]
                
                # Собираем соседние точки только из исходного меша
                total_weight = 0.0
                weighted_sum = 0.0
                
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        
                        # Проверяем границы и принадлежность к исходному мешу
                        if (0 <= ni < smoothed_height.shape[0] and 
                            0 <= nj < smoothed_height.shape[1] and 
                            source_mask[ni, nj]):
                            
                            weight = weights[di+radius, dj+radius]
                            weighted_sum += weight * smoothed_height[ni, nj]
                            total_weight += weight
                
                if total_weight > 0:
                    new_heights[i, j] = weighted_sum / total_weight
            
            smoothed_height = new_heights
        
        return smoothed_height

    def _create_edge_mask(self, source_mask):
        """
        Создает маску граничных точек исходного меша с расширенной зоной
        
        Args:
            source_mask: Маска исходного меша
            
        Returns:
            Маска граничных точек
        """
        from scipy.ndimage import binary_dilation, binary_erosion
        
        # Расширяем и сужаем маску для получения границы
        dilated = binary_dilation(source_mask, structure=np.ones((3, 3)))
        eroded = binary_erosion(source_mask, structure=np.ones((3, 3)))
        
        # Граница - это разница между расширенной и суженной масками
        edge_mask = dilated & ~eroded
        
        return edge_mask
    
    def _smooth_lifted_area(self, height_grid, lifted_mask, sigma=2.0):
        """
        Применяет гауссово сглаживание к поднятой области для устранения волнообразности
        
        Args:
            height_grid: Поле высот
            lifted_mask: Маска поднятых вершин
            sigma: Параметр сглаживания (чем больше, тем сильнее сглаживание)
            
        Returns:
            Сглаженное поле высот
        """
        if not np.any(lifted_mask):
            return height_grid
        
        # Создаем копию высот
        smoothed_height = height_grid.copy()
        
        try:
            # Применяем гауссово сглаживание ко всему полю
            gaussian_smoothed = gaussian_filter(height_grid, sigma=sigma)
            
            # Заменяем только поднятые области сглаженными значениями
            smoothed_height[lifted_mask] = gaussian_smoothed[lifted_mask]
            
            print(f"Applied Gaussian smoothing to lifted area (sigma={sigma})")
            
        except Exception as e:
            print(f"Gaussian smoothing failed: {e}. Using simple smoothing.")
            # Резервный метод: простое сглаживание
            for _ in range(3):
                new_heights = smoothed_height.copy()
                for i in range(1, smoothed_height.shape[0]-1):
                    for j in range(1, smoothed_height.shape[1]-1):
                        if lifted_mask[i, j]:
                            # Усредняем 5x5 окрестность для более сильного сглаживания
                            neighbors = []
                            for di in range(-2, 3):
                                for dj in range(-2, 3):
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < smoothed_height.shape[0] and 0 <= nj < smoothed_height.shape[1]:
                                        neighbors.append(smoothed_height[ni, nj])
                            new_heights[i, j] = np.mean(neighbors)
                smoothed_height = new_heights
        
        return smoothed_height
    
    def _blur_lift_boundary(self, height_grid, source_mask, boundary_points, blur_radius=3):
        """
        Размывает границу поднятой области для плавного перехода
        
        Args:
            height_grid: Поле высот
            source_mask: Маска исходного меша
            boundary_points: Граничные точки исходного меша
            blur_radius: Радиус размытия
            
        Returns:
            Поле высот с размытой границей
        """
        if len(boundary_points) == 0 or blur_radius <= 0:
            return height_grid
        
        boundary_points_array = np.array(boundary_points)
        
        # Создаем копию высот
        blurred_height = height_grid.copy()
        
        # Создаем KD-дерево для быстрого поиска граничных точек
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
        except Exception as e:
            print(f"KDTree creation for blur failed: {e}")
            return height_grid
        
        # Применяем размытие к области вблизи границы
        h, w = height_grid.shape
        boundary_influence_mask = np.zeros((h, w), dtype=bool)
        
        # Определяем область влияния границы
        for i in range(h):
            for j in range(w):
                # Пропускаем точки исходного меша
                if source_mask[i, j]:
                    continue
                
                # Предполагаем, что x_grid и y_grid доступны через self
                x = self._current_x_grid[j] if hasattr(self, '_current_x_grid') else j
                y = self._current_y_grid[i] if hasattr(self, '_current_y_grid') else i
                
                # Находим расстояние до ближайшей граничной точки
                distances, _ = boundary_tree.query([x, y], k=1)
                
                # Если точка находится в радиусе размытия
                if distances <= blur_radius:
                    boundary_influence_mask[i, j] = True
        
        # Применяем размытие к области влияния границы
        if np.any(boundary_influence_mask):
            for i in range(h):
                for j in range(w):
                    if boundary_influence_mask[i, j]:
                        # Собираем значения из окрестности
                        neighbors = []
                        weights = []
                        
                        for di in range(-blur_radius, blur_radius + 1):
                            for dj in range(-blur_radius, blur_radius + 1):
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    # Вес зависит от расстояния (гауссово ядро)
                                    dist = np.sqrt(di*di + dj*dj)
                                    weight = np.exp(-dist*dist / (2 * (blur_radius/2)**2))
                                    neighbors.append(height_grid[ni, nj] * weight)
                                    weights.append(weight)
                        
                        if weights:
                            blurred_height[i, j] = np.sum(neighbors) / np.sum(weights)
            
            print(f"Applied boundary blur with radius {blur_radius}")
        
        return blurred_height
    
    def _calculate_adaptive_distance_parameters(self, boundary_heights):
        """Вычисляет адаптивные параметры расстояния на основе высот граничных точек"""
        if len(boundary_heights) == 0:
            return self.base_distance, self.min_distance, self.max_distance
        
        min_boundary_height = np.min(boundary_heights)
        max_boundary_height = np.max(boundary_heights)
        height_range = max_boundary_height - min_boundary_height
        
        # Адаптивные параметры на основе статистики
        # Базовое расстояние зависит от диапазона высот
        base_distance = self.base_distance + min(height_range * 0.5, 0.7)  # 0.3-1.2
        min_distance = max(self.min_distance, base_distance * 0.5)        # 0.1-0.6
        max_distance = self.max_distance + min(height_range * 2.0, 4.0)   # 3.0-7.0
        
        return base_distance, min_distance, max_distance
    
    def _apply_adaptive_vertex_lift(self, height_grid, source_mask, x_grid, y_grid, boundary_points):
        """
        Применяет адаптивный подъем вершин к полю высот
        
        Args:
            height_grid: Исходное поле высот
            source_mask: Маска исходного меша
            x_grid, y_grid: Координаты сетки
            boundary_points: Граничные точки исходного меша
            
        Returns:
            Скорректированное поле высот и маска поднятых вершин
        """
        if len(boundary_points) == 0:
            return height_grid, np.zeros_like(source_mask, dtype=bool)
        
        # Сохраняем координаты сетки для использования в других методах
        self._current_x_grid = x_grid
        self._current_y_grid = y_grid
        
        boundary_points_array = np.array(boundary_points)
        boundary_heights = boundary_points_array[:, 2]
        
        # Вычисляем адаптивные параметры расстояния
        base_distance, min_distance, max_distance = self._calculate_adaptive_distance_parameters(boundary_heights)
        
        print(f"Adaptive distance parameters: BASE={base_distance:.3f}, MIN={min_distance:.3f}, MAX={max_distance:.3f}")
        
        # Создаем KD-дерево для быстрого поиска ближайших граничных точек
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
            use_kdtree = True
        except Exception as e:
            print(f"KDTree creation failed: {e}. Using simple distance calculation")
            use_kdtree = False
        
        # Вычисляем статистику высот граничных точек
        min_boundary_height = np.min(boundary_heights)
        max_boundary_height = np.max(boundary_heights)
        height_range = max_boundary_height - min_boundary_height
        
        # Создаем копию высот для коррекции
        corrected_height = height_grid.copy()
        
        # Маска для поднятых вершин
        lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        # Статистика подъема вершин
        lifted_vertices_count = 0
        total_lift_amount = 0.0
        min_lift = float('inf')
        max_lift = float('-inf')
        
        # Применяем подъем вершин для точек вне исходного меша
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                # Пропускаем точки исходного меша
                if source_mask[i, j]:
                    continue
                
                x = x_grid[j]
                y = y_grid[i]
                current_z = height_grid[i, j]
                
                # Находим ближайшую граничную точку
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
                
                # Вычисляем разницу высот
                height_diff = nearest_point[2] - current_z
                
                # Если разница высот положительная и точка находится в зоне влияния
                if height_diff > 0 and nearest_dist <= max_distance:
                    # Вычисляем адаптивное расстояние влияния
                    if height_range > 0:
                        normalized_height_diff = min(height_diff / height_range, 1.0)
                    else:
                        normalized_height_diff = 0.5
                    
                    # Динамическое расстояние на основе разницы высот
                    if normalized_height_diff < 0.1:
                        dynamic_distance = min_distance * 1.2
                    elif normalized_height_diff < 0.3:
                        dynamic_distance = base_distance * (1.0 + normalized_height_diff)
                    else:
                        dynamic_distance = base_distance * (1.0 + normalized_height_diff * 3.0)
                    
                    dynamic_distance = max(min_distance, min(dynamic_distance, max_distance))
                    
                    # Если точка находится в зоне влияния
                    if nearest_dist <= dynamic_distance:
                        # Вычисляем нормализованное расстояние
                        normalized_distance = nearest_dist / dynamic_distance
                        
                        # Smooth Falloff функция - кубическая интерполяция для плавности
                        t = normalized_distance
                        # Более плавная функция затухания
                        smooth_factor = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
                        
                        # Применяем плавное затухание с учетом интенсивности
                        lift_amount = height_diff * smooth_factor * self.lift_intensity
                        corrected_height[i, j] += lift_amount
                        
                        # Отмечаем вершину как поднятую
                        lifted_mask[i, j] = True
                        
                        # Обновляем статистику
                        lifted_vertices_count += 1
                        total_lift_amount += lift_amount
                        
                        if lift_amount < min_lift:
                            min_lift = lift_amount
                        if lift_amount > max_lift:
                            max_lift = lift_amount
        
        # Выводим статистику
        total_vertices = np.sum(~source_mask)
        if lifted_vertices_count > 0:
            avg_lift = total_lift_amount / lifted_vertices_count
            print(f"Adaptive vertex lift applied: {lifted_vertices_count}/{total_vertices} vertices adjusted")
            print(f"Lift statistics: min={min_lift:.3f}, avg={avg_lift:.3f}, max={max_lift:.3f}")
        else:
            print("No vertices were lifted (adaptive lift)")
        
        return corrected_height, lifted_mask
    
    def create_extended_mesh(self, world_position=(0, 0, 2), source_scale_x=None,
                    source_scale_y=None, source_scale_z=None, 
                    source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0):
        """
        Создание экстраполированного меша с использованием нового гибридного алгоритма
        
        Объединяет преимущества:
        1. Единая перлин-поверхность, проходящая через ключевые точки
        2. Адаптивный подъем вершин от границы исходного меша
        """
        return self.create_unified_perlin_mesh_with_lift(world_position, source_scale_x,
                                                        source_scale_y, source_scale_z,
                                                        source_offset_x, source_offset_y, source_offset_z)
    
    def create_unified_perlin_mesh_with_lift(self, world_position=(0, 0, 2), source_scale_x=None,
                                            source_scale_y=None, source_scale_z=None,
                                            source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0):
        """
        Создание единого перлин-меша с адаптивным подъемом вершин
        
        Объединяет преимущества:
        1. Единая перлин-поверхность, проходящая через ключевые точки
        2. Адаптивный подъем вершин от границы исходного меша
        3. Сглаживание поднятой области для устранения волнообразности
        4. Сглаживание исходного меша для улучшения качества
        """
        if self.height_map is None:
            if not self.load_height_map():
                return None
        
        h, w = self.height_map.shape
        
        # Определяем границы исходного меша
        if not hasattr(self, 'mask'):
            self.mask = self.height_map > self.alpha_threshold
        
        y_indices, x_indices = np.where(self.mask)
        
        if len(y_indices) == 0:
            print("No visible pixels in height map")
            return None
        
        # Используем переданные значения масштаба или значения по умолчанию
        if source_scale_x is None:
            source_scale_x = -2.0
        if source_scale_y is None:
            source_scale_y = 2.0
        if source_scale_z is None:
            source_scale_z = self.height_scale
        
        # ------------------------------------------------------------
        # ШАГ 1: Извлечение ключевых точек исходного меша
        # ------------------------------------------------------------
        print("Step 1: Extracting key points from source mesh...")
        
        # Извлекаем все точки исходного меша для создания source_mask на сетке
        source_points = []
        for y, x in zip(y_indices, x_indices):
            height = self.height_map[y, x]
            
            # Преобразуем в мировые координаты
            world_x = (x / w - 0.5) * source_scale_x
            world_y = (y / h - 0.5) * source_scale_y
            world_z = height * source_scale_z
            
            world_x += source_offset_x
            world_y += source_offset_y
            world_z += source_offset_z
            
            world_x, world_y, world_z = self._apply_rotation(world_x, world_y, world_z)
            
            world_x += world_position[0]
            world_y += world_position[1]
            world_z += world_position[2]
            
            source_points.append((world_x, world_y, world_z))
        
        source_points = np.array(source_points)
        print(f"Extracted {len(source_points)} source points from height map")
        
        # ------------------------------------------------------------
        # ШАГ 2: Создание регулярной сетки на целевой области
        # ------------------------------------------------------------
        print("Step 2: Creating regular grid...")
        
        # Определяем границы целевой области
        min_x = world_position[0] - self.target_width / 2
        max_x = world_position[0] + self.target_width / 2
        min_y = world_position[1] - self.target_height / 2
        max_y = world_position[1] + self.target_height / 2
        
        # Создаем регулярную сетку
        x_grid = np.linspace(min_x, max_x, self.grid_resolution)
        y_grid = np.linspace(min_y, max_y, self.grid_resolution)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        # ------------------------------------------------------------
        # ШАГ 3: Создание маски исходного меша на сетке
        # ------------------------------------------------------------
        print("Step 3: Creating source mask on grid...")
        
        # Проецируем высоты исходного меша на сетку
        source_height_grid = np.full((len(y_grid), len(x_grid)), np.nan, dtype=np.float32)
        source_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
        
        for wx, wy, wz in source_points:
            # Находим ближайшую точку сетки
            grid_x_idx = np.argmin(np.abs(x_grid - wx))
            grid_y_idx = np.argmin(np.abs(y_grid - wy))
            
            # Если точка достаточно близка к узлу сетки
            if (abs(x_grid[grid_x_idx] - wx) < (x_grid[1] - x_grid[0]) * 0.5 and
                abs(y_grid[grid_y_idx] - wy) < (y_grid[1] - y_grid[0]) * 0.5):
                
                source_height_grid[grid_y_idx, grid_x_idx] = wz
                source_mask[grid_y_idx, grid_x_idx] = True
        
        # Находим граничные точки исходного меша
        boundary_points = []
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                if source_mask[i, j]:
                    # Проверяем, является ли точка граничной
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
        
        print(f"Created source mask: {np.sum(source_mask)}/{source_mask.size} grid points")
        print(f"Boundary points: {len(boundary_points)}")
        
        # ------------------------------------------------------------
        # ШАГ 4: Генерация перлин-шума на всей сетке
        # ------------------------------------------------------------
        print("Step 4: Generating Perlin noise...")
        
        perlin_height = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)
        
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x = x_grid[j]
                y = y_grid[i]
                
                # Генерируем шум Перлина
                noise_value = self._generate_perlin_noise(x, y)
                # Преобразуем из диапазона [-1, 1] в [0, 1] и умножаем на силу
                noise_height = (noise_value + 1.0) / 2.0 * self.noise_strength
                perlin_height[i, j] = noise_height
        
        # Добавляем базовую высоту
        base_height_grid = perlin_height + self.base_height
        
        # ------------------------------------------------------------
        # ШАГ 5: Коррекция высот для прохождения через ключевые точки
        # ------------------------------------------------------------
        print("Step 5: Correcting heights to pass through key points...")
        
        if len(source_points) > 0:
            # Подготавливаем данные ключевых точек
            key_x = source_points[:, 0]
            key_y = source_points[:, 1]
            key_z = source_points[:, 2]
            
            # Используем RBF интерполяцию для создания гладкой поверхности через ключевые точки
            try:
                if self.interpolation_method == 'rbf':
                    # RBF интерполяция
                    rbf = Rbf(key_x, key_y, key_z, 
                              function='multiquadric', 
                              smooth=self.rbf_smooth)
                    
                    # Интерполируем на сетке
                    key_surface = rbf(grid_x, grid_y)
                    
                elif self.interpolation_method in ['linear', 'cubic']:
                    # Линейная или кубическая интерполяция
                    key_surface = griddata((key_x, key_y), key_z, 
                                          (grid_x, grid_y), 
                                          method=self.interpolation_method,
                                          fill_value=self.base_height)
                else:
                    raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
                
                # Смешиваем перлин-поверхность и поверхность через ключевые точки
                # Используем маску исходного меша для плавного перехода
                blend_weights = np.zeros_like(grid_x, dtype=np.float32)
                
                # Для точек вблизи исходного меша создаем плавный переход
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j]:
                            # Точки внутри исходного меша - полный вес ключевой поверхности
                            blend_weights[i, j] = 1.0
                        else:
                            # Для точек вне исходного меша находим расстояние до ближайшей точки
                            distances = np.sqrt((key_x - grid_x[i, j])**2 + (key_y - grid_y[i, j])**2)
                            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                            
                            # Плавный переход на расстоянии до 1.0 единиц
                            influence_distance = 1.0
                            if min_distance < influence_distance:
                                weight = 1.0 - (min_distance / influence_distance)
                                # Квадратичное затухание для более плавного перехода
                                weight = weight * weight
                                blend_weights[i, j] = weight
                
                # Применяем смешивание
                corrected_height = base_height_grid * (1.0 - blend_weights) + key_surface * blend_weights
                
                # Восстанавливаем точные высоты исходного меша
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j] and not np.isnan(source_height_grid[i, j]):
                            corrected_height[i, j] = source_height_grid[i, j]
                
                print(f"Applied interpolation correction using {self.interpolation_method} method")
                
            except Exception as e:
                print(f"Interpolation failed: {e}. Using pure Perlin noise with source heights.")
                corrected_height = base_height_grid.copy()
                # Восстанавливаем точные высоты исходного меша
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j] and not np.isnan(source_height_grid[i, j]):
                            corrected_height[i, j] = source_height_grid[i, j]
        else:
            corrected_height = base_height_grid
        
        # ------------------------------------------------------------
        # ШАГ 5a: Сглаживание исходного меша (если включено)
        # ------------------------------------------------------------
        if self.source_mesh_smoothing_enabled and np.any(source_mask):
            print("Step 5a: Smoothing source mesh...")
            corrected_height = self._smooth_source_mesh(
                corrected_height, source_mask,
                sigma=self.source_mesh_smoothing_sigma,
                iterations=self.source_mesh_smoothing_iterations,
                preserve_edges=self.source_mesh_edge_preserving
            )
            
            # Дополнительная постобработка граничной зоны
            if self.source_mesh_smoothing_sigma > 1.0:  # Только при больших sigma
                print("Step 5b: Post-processing boundary zone...")
                corrected_height = self._postprocess_boundary_zone(
                    corrected_height, source_mask, 
                    boundary_width=int(self.source_mesh_smoothing_sigma * 2)
                )
        
        # ------------------------------------------------------------
        # ШАГ 6: Адаптивный подъем вершин (если включен)
        # ------------------------------------------------------------
        if self.adaptive_lift_enabled and len(boundary_points) > 0:
            print("Step 6: Applying adaptive vertex lift...")
            corrected_height, lifted_mask = self._apply_adaptive_vertex_lift(
                corrected_height, source_mask, x_grid, y_grid, boundary_points
            )
            
            # --------------------------------------------------------
            # ШАГ 6a: Сглаживание поднятой области для устранения волнообразности
            # --------------------------------------------------------
            if self.lift_smoothing_enabled and np.any(lifted_mask):
                print("Step 6a: Smoothing lifted area to eliminate waviness...")
                corrected_height = self._smooth_lifted_area(
                    corrected_height, lifted_mask, sigma=self.lift_smoothing_sigma
                )
            
            # --------------------------------------------------------
            # ШАГ 6b: Размытие границ поднятой области для плавного перехода
            # --------------------------------------------------------
            if self.lift_blur_enabled and self.lift_blur_radius > 0:
                print("Step 6b: Blurring lift boundary for smooth transition...")
                corrected_height = self._blur_lift_boundary(
                    corrected_height, source_mask, boundary_points, blur_radius=self.lift_blur_radius
                )
        else:
            lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        # ------------------------------------------------------------
        # ШАГ 7: Дополнительное сглаживание всей поверхности
        # ------------------------------------------------------------
        if self.use_smoothing and self.smoothing_iterations > 0:
            print(f"Step 7: Applying general smoothing ({self.smoothing_iterations} iterations)...")
            corrected_height = self._smooth_heightfield(corrected_height, self.smoothing_iterations)
        
        # ------------------------------------------------------------
        # ШАГ 8: Создание меша
        # ------------------------------------------------------------
        print("Step 8: Creating unified mesh with lift...")
        
        # Создаем форматы вершин
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        # Создаем вершины
        vertices = []
        vertex_indices = {}
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                z = corrected_height[i, j]
                
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1)  # Временная нормаль
                texcoord.addData2f(j / (len(x_grid)-1), i / (len(y_grid)-1))
                
                vertex_indices[(i, j)] = len(vertices)
                vertices.append((x, y, z))
        
        # Создаем треугольники
        triangles = GeomTriangles(Geom.UHStatic)
        
        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                v1 = vertex_indices[(i, j)]
                v2 = vertex_indices[(i, j + 1)]
                v3 = vertex_indices[(i + 1, j)]
                v4 = vertex_indices[(i + 1, j + 1)]
                
                triangles.addVertices(v1, v2, v3)
                triangles.addVertices(v2, v4, v3)
        
        triangles.closePrimitive()
        
        # Создаем геометрию
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        
        # Вычисляем нормали
        self._calculate_normals(vdata, geom)
        
        # Статистика
        print(f"Unified Perlin mesh with lift created with {len(vertices)} vertices and {triangles.getNumPrimitives()} triangles")
        print(f"Target area: {self.target_width}x{self.target_height}")
        print(f"Grid resolution: {self.grid_resolution}x{self.grid_resolution}")
        print(f"Source points: {len(source_points)}")
        print(f"Boundary points: {len(boundary_points)}")
        print(f"Interpolation method: {self.interpolation_method}")
        print(f"Source mesh smoothing: {'enabled' if self.source_mesh_smoothing_enabled else 'disabled'} " +
              f"(sigma={self.source_mesh_smoothing_sigma}, iterations={self.source_mesh_smoothing_iterations})")
        print(f"Adaptive lift: {'enabled' if self.adaptive_lift_enabled else 'disabled'}")
        print(f"Lift smoothing: {'enabled' if self.lift_smoothing_enabled else 'disabled'} (sigma={self.lift_smoothing_sigma})")
        print(f"Lift boundary blur: {'enabled' if self.lift_blur_enabled else 'disabled'} (radius={self.lift_blur_radius})")
        print(f"General smoothing: {'enabled' if self.use_smoothing else 'disabled'} ({self.smoothing_iterations} iterations)")
        print(f"Base height: {self.base_height}")
        print(f"Noise strength: {self.noise_strength}")
        print(f"Noise octaves: {self.noise_octaves}")
        
        return node
    
    def _postprocess_boundary_zone(self, height_grid, source_mask, boundary_width=5):
        """
        Постобработка граничной зоны для плавного перехода
        
        Args:
            height_grid: Поле высот
            source_mask: Маска исходного меша
            boundary_width: Ширина граничной зоны для обработки
            
        Returns:
            Скорректированное поле высот
        """
        from scipy.ndimage import distance_transform_edt
        
        # Вычисляем расстояние до границы для каждой точки внутри исходного меша
        # Расстояние положительное внутри меша, отрицательное снаружи
        distance_in = distance_transform_edt(source_mask)
        
        # Создаем градиентный переход на границе
        corrected_height = height_grid.copy()
        
        for i in range(height_grid.shape[0]):
            for j in range(height_grid.shape[1]):
                if source_mask[i, j]:
                    dist = distance_in[i, j]
                    
                    # Если точка близко к границе (в зоне влияния)
                    if dist < boundary_width:
                        # Находим ближайшую граничную точку
                        min_boundary_height = height_grid[i, j]
                        
                        # Ищем минимальную высоту в окрестности границы
                        for di in range(-boundary_width, boundary_width + 1):
                            for dj in range(-boundary_width, boundary_width + 1):
                                ni, nj = i + di, j + dj
                                if (0 <= ni < height_grid.shape[0] and 
                                    0 <= nj < height_grid.shape[1]):
                                    if not source_mask[ni, nj]:
                                        # Это граничная точка с внешней стороны
                                        # Учитываем ее влияние
                                        boundary_dist = np.sqrt(di*di + dj*dj)
                                        if boundary_dist < boundary_width:
                                            weight = 1.0 - (boundary_dist / boundary_width)
                                            # Плавно поднимаем граничные точки к внутренним
                                            corrected_height[i, j] = max(
                                                corrected_height[i, j],
                                                height_grid[ni, nj] * (1.0 - weight) + 
                                                corrected_height[i, j] * weight
                                            )
        
        return corrected_height
    
    def _create_simple_extended_mesh(self, x_grid, y_grid, source_mask, avg_source_height, 
                                     world_position, vdata, vertex, normal, texcoord):
        """Создание простого меша без экстраполяции (резервный метод)"""
        # Создаем вершины
        vertices = []
        vertex_indices = {}
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                if source_mask[i, j]:
                    z = avg_source_height[i, j]
                else:
                    z = self.base_height
                
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1)
                texcoord.addData2f(j / (len(x_grid)-1), i / (len(y_grid)-1))
                
                vertex_indices[(i, j)] = len(vertices)
                vertices.append((x, y, z))
        
        # Создаем треугольники
        triangles = GeomTriangles(Geom.UHStatic)
        
        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                v1 = vertex_indices[(i, j)]
                v2 = vertex_indices[(i, j + 1)]
                v3 = vertex_indices[(i + 1, j)]
                v4 = vertex_indices[(i + 1, j + 1)]
                
                triangles.addVertices(v1, v2, v3)
                triangles.addVertices(v2, v4, v3)
        
        triangles.closePrimitive()
        
        # Создаем геометрию
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("simple_extended_mesh")
        node.addGeom(geom)
        
        self._calculate_normals(vdata, geom)
        
        print(f"Simple extended mesh created with {len(vertices)} vertices")
        return node
    
    def _calculate_normals(self, vdata, geom):
        """Вычисление нормалей для меша с учетом поворота"""
        # Создаем массив для нормалей
        vertex_reader = GeomVertexReader(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        
        # Собираем все вершины
        vertices = []
        while not vertex_reader.isAtEnd():
            pos = vertex_reader.getData3f()
            vertices.append((pos.x, pos.y, pos.z))
        
        # Инициализируем нормали нулями
        normals = [(0.0, 0.0, 0.0) for _ in range(len(vertices))]
        
        # Проходим по всем треугольникам
        for i in range(geom.getNumPrimitives()):
            prim = geom.getPrimitive(i)
            if prim.getNumPrimitives() > 0:
                for j in range(prim.getNumPrimitives()):
                    start = prim.getPrimitiveStart(j)
                    end = prim.getPrimitiveEnd(j)
                    
                    # Берем 3 вершины треугольника
                    if end - start == 3:
                        vi0 = prim.getVertex(start)
                        vi1 = prim.getVertex(start + 1)
                        vi2 = prim.getVertex(start + 2)
                        
                        v0 = np.array(vertices[vi0])
                        v1 = np.array(vertices[vi1])
                        v2 = np.array(vertices[vi2])
                        
                        # Вычисляем нормаль треугольника
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        norm = np.linalg.norm(normal)
                        
                        if norm > 0:
                            normal = normal / norm
                            
                            # Добавляем нормаль к каждой вершине
                            for vi in [vi0, vi1, vi2]:
                                curr = normals[vi]
                                normals[vi] = (
                                    curr[0] + normal[0],
                                    curr[1] + normal[1],
                                    curr[2] + normal[2]
                                )
        
        # Нормализуем и записываем нормали
        for i in range(len(normals)):
            nx, ny, nz = normals[i]
            norm = (nx*nx + ny*ny + nz*nz) ** 0.5
            if norm > 0:
                nx /= norm
                ny /= norm
                nz /= norm
            
            # Применяем тот же поворот к нормалям
            nx, ny, nz = self._apply_rotation(nx, ny, nz)
            normal_writer.setData3f(nx, ny, nz)
    
    def add_to_scene(self, position=(0, 0, 2)):
        """Добавление меша на сцену уже в мировых координатах"""
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        # Создаём меш сразу в мировых координатах
        node = self.create_mesh_in_world_coordinates(position)
        if node:
            # Меш уже содержит мировые координаты, поэтому просто прикрепляем его к сцене
            self.mesh_node = self.app.render.attachNewNode(node)
            # НЕ устанавливаем позицию через setPos, так как вершины уже в мировых координатах
            
            # Настройка материала
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            # Включение шейдеров
            self.mesh_node.setShaderAuto()

            # Добавляем в список моделей
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "height_map_mesh_world"
            
            print(f"Height map mesh added at world position {position}")
            return self.mesh_node
        
        return None
    
    def set_source_scale(self, scale_x, scale_y, scale_z):
        """Установка масштаба для исходного меша по всем осям"""
        self.source_scale_x = scale_x
        self.source_scale_y = scale_y
        self.source_scale_z = scale_z
        print(f"Source mesh scale set to {scale_x}x{scale_y}x{scale_z}")

    def set_source_offset(self, offset_x, offset_y, offset_z):
        """Установка смещения для исходного меша"""
        self.source_offset_x = offset_x
        self.source_offset_y = offset_y
        self.source_offset_z = offset_z
        print(f"Source mesh offset set to ({offset_x}, {offset_y}, {offset_z})")

    def add_extended_mesh_to_scene(self, position=(0, 0, 2), source_scale_x=None,
                              source_scale_y=None, source_scale_z=None,
                              source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0):
        """Добавление экстраполированного меша на сцену с возможностью настройки размеров и позиции исходного меша"""
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        # Создаём экстраполированный меш с указанными параметрами
        node = self.create_extended_mesh(position, source_scale_x, source_scale_y, 
                                        source_scale_z, source_offset_x, source_offset_y, source_offset_z)
        if node:
            # Прикрепляем меш к сцене
            self.mesh_node = self.app.render.attachNewNode(node)
            
            # Настройка материала
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            # Включение шейдеров
            self.mesh_node.setShaderAuto()
            self.mesh_node.setPos(0, 0, 2)
            
            # Добавляем в список моделей
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "extended_height_map_mesh"
            
            print(f"Extended height map mesh added at world position {position}")
            return self.mesh_node
        
        return None
    
    def add_unified_mesh_with_lift_to_scene(self, position=(0, 0, 2), source_scale_x=None,
                                           source_scale_y=None, source_scale_z=None,
                                           source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0):
        """Добавление унифицированного перлин-меша с адаптивным подъемом на сцену"""
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        # Создаём унифицированный перлин-меш с адаптивным подъемом
        node = self.create_unified_perlin_mesh_with_lift(position, source_scale_x, source_scale_y,
                                                        source_scale_z, source_offset_x, source_offset_y, source_offset_z)
        if node:
            # Прикрепляем меш к сцене
            self.mesh_node = self.app.render.attachNewNode(node)
            
            # Настройка материала
            material = Material()
            material.setDiffuse((0.8, 0.8, 0.8, 1.0))
            material.setAmbient((0.3, 0.3, 0.3, 1.0))
            material.setSpecular((0.5, 0.5, 0.5, 1.0))
            material.setShininess(50.0)
            self.mesh_node.setTwoSided(True)
            self.mesh_node.setMaterial(material, 1)
            
            # Включение шейдеров
            self.mesh_node.setShaderAuto()
            self.mesh_node.setPos(0, 0, 2)
            
            # Добавляем в список моделей
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "unified_perlin_mesh_with_lift"
            
            print(f"Unified Perlin mesh with adaptive lift added at world position {position}")
            return self.mesh_node
        
        return None
    
    def remove_from_scene(self):
        """Удаление меша со сцены"""
        if self.mesh_node:
            if self.mesh_node in self.app.loaded_models:
                self.app.loaded_models.remove(self.mesh_node)
            
            model_id = id(self.mesh_node)
            if model_id in self.app.model_paths:
                del self.app.model_paths[model_id]
            
            self.mesh_node.removeNode()
            self.mesh_node = None
            print("Height map mesh removed")
    
    def set_threshold(self, threshold):
        """Установка порога для определения фона"""
        self.alpha_threshold = threshold
        print(f"Alpha threshold set to {threshold}")
    
    def set_rotation(self, rotation_angle):
        """Установка угла поворота меша (в градусах)"""
        self.mesh_rotation = rotation_angle
        print(f"Mesh rotation set to {rotation_angle} degrees")
    
    def set_extended_area(self, width, height):
        """Установка размера целевой области для экстраполяции"""
        self.target_width = width
        self.target_height = height
        print(f"Extended area set to {width}x{height}")
    
    def set_base_height(self, height):
        """Установка базовой высоты для экстраполируемой области"""
        self.base_height = height
        print(f"Base height set to {height}")
    
    def set_grid_resolution(self, resolution):
        """Установка разрешения сетки для экстраполяции"""
        self.grid_resolution = resolution
        print(f"Grid resolution set to {resolution}")
    
    # Методы для настройки шума Перлина
    
    def set_noise_scale(self, scale):
        """Установка масштаба шума Перлина"""
        self.noise_scale = scale
        print(f"Noise scale set to {scale}")
    
    def set_noise_strength(self, strength):
        """Установка силы/амплитуды шума Перлина"""
        self.noise_strength = strength
        print(f"Noise strength set to {strength}")
    
    def set_noise_octaves(self, octaves):
        """Установка количества октав шума Перлина"""
        self.noise_octaves = octaves
        print(f"Noise octaves set to {octaves}")
    
    def set_noise_persistence(self, persistence):
        """Установка персистентности шума Перлина"""
        self.noise_persistence = persistence
        print(f"Noise persistence set to {persistence}")
    
    def set_noise_lacunarity(self, lacunarity):
        """Установка лакунарности шума Перлина"""
        self.noise_lacunarity = lacunarity
        print(f"Noise lacunarity set to {lacunarity}")
    
    def set_noise_seed(self, seed):
        """Установка seed для генератора шума Перлина"""
        self.noise_seed = seed
        print(f"Noise seed set to {seed}")
    
    # Новые методы для настройки унифицированного подхода
    
    def set_interpolation_method(self, method):
        """Установка метода интерполяции"""
        valid_methods = ['rbf', 'linear', 'cubic']
        if method in valid_methods:
            self.interpolation_method = method
            print(f"Interpolation method set to {method}")
        else:
            print(f"Invalid method. Must be one of: {valid_methods}")

    def set_rbf_smooth(self, smooth):
        """Установка параметра сглаживания для RBF"""
        self.rbf_smooth = max(0.0, smooth)
        print(f"RBF smooth parameter set to {smooth}")

    def set_adaptive_lift_enabled(self, enabled):
        """Включение/выключение адаптивного подъема вершин"""
        self.adaptive_lift_enabled = enabled
        print(f"Adaptive vertex lift {'enabled' if enabled else 'disabled'}")

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
        
        print(f"Lift parameters: base_distance={self.base_distance}, min_distance={self.min_distance}, max_distance={self.max_distance}, intensity={self.lift_intensity}")

    def set_use_smoothing(self, use_smoothing):
        """Включение/выключение сглаживания"""
        self.use_smoothing = use_smoothing
        print(f"Smoothing {'enabled' if use_smoothing else 'disabled'}")

    def set_smoothing_iterations(self, iterations):
        """Установка количества итераций сглаживания"""
        self.smoothing_iterations = max(0, int(iterations))
        print(f"Smoothing iterations set to {iterations}")
    
    # Новые методы для устранения волнообразности
    
    def set_lift_smoothing_enabled(self, enabled):
        """Включение/выключение сглаживания поднятой области"""
        self.lift_smoothing_enabled = enabled
        print(f"Lift area smoothing {'enabled' if enabled else 'disabled'}")

    def set_lift_smoothing_sigma(self, sigma):
        """Установка параметра сигма для гауссового сглаживания поднятой области"""
        self.lift_smoothing_sigma = max(0.1, sigma)
        print(f"Lift smoothing sigma set to {sigma}")

    def set_lift_blur_enabled(self, enabled):
        """Включение/выключение размытия границ поднятой области"""
        self.lift_blur_enabled = enabled
        print(f"Lift boundary blur {'enabled' if enabled else 'disabled'}")

    def set_lift_blur_radius(self, radius):
        """Установка радиуса размытия границ поднятой области"""
        self.lift_blur_radius = max(0, int(radius))
        print(f"Lift blur radius set to {radius}")
    
    # Новые методы для сглаживания исходного меша
    
    def set_source_mesh_smoothing_enabled(self, enabled):
        """Включение/выключение сглаживания исходного меша"""
        self.source_mesh_smoothing_enabled = enabled
        print(f"Source mesh smoothing {'enabled' if enabled else 'disabled'}")

    def set_source_mesh_smoothing_iterations(self, iterations):
        """Установка количества итераций сглаживания исходного меша"""
        self.source_mesh_smoothing_iterations = max(0, int(iterations))
        print(f"Source mesh smoothing iterations set to {iterations}")

    def set_source_mesh_smoothing_sigma(self, sigma):
        """Установка параметра сигма для гауссового сглаживания исходного меша"""
        self.source_mesh_smoothing_sigma = max(0.1, sigma)
        print(f"Source mesh smoothing sigma set to {sigma}")

    def set_source_mesh_edge_preserving(self, preserve):
        """Включение/выключение сохранения границ исходного меша при сглаживании"""
        self.source_mesh_edge_preserving = preserve
        print(f"Source mesh edge preserving {'enabled' if preserve else 'disabled'}")