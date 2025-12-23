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

class HeightMapMeshGenerator:
    def __init__(self, app, image_path=None):
        self.app = app
        self.image_path = image_path or "height_example/No-visible-Depth.png"
        self.height_map = None
        self.mask = None
        self.height_scale = 2.0  
        self.alpha_threshold = 0.1
        self.mesh_node = None
        self.use_alpha_channel = True
        self.mesh_rotation = -90  

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
        
    def load_height_map(self):
        try:
            img = Image.open(self.image_path)
            
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
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
                img_gray = img.convert('L')
                self.height_map = np.array(img_gray, dtype=np.float32) / 255.0
                
                mask = self.height_map > 0.05
                self.mask = mask
            return True
            
        except Exception as e:
            print(f"Error loading height map: {e}")
            import traceback
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
        smoothed = heightfield.copy()
        h, w = smoothed.shape
        
        for _ in range(iterations):
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
        
        return smoothed
    
    def _smooth_source_mesh(self, height_grid, source_mask, sigma=0.5, iterations=1, preserve_edges=True):
        if not np.any(source_mask):
            return height_grid
        
        smoothed_height = height_grid.copy()
        
        protected_mask = source_mask.copy()
        
        if preserve_edges:
            edge_mask = self._create_edge_mask(source_mask)
            
            protection_radius = max(1, int(np.ceil(sigma)))
            
            from scipy.ndimage import binary_dilation
            protected_mask = binary_dilation(edge_mask, 
                                            structure=np.ones((2*protection_radius+1, 2*protection_radius+1)))
            
            protected_mask = protected_mask & source_mask
            
        try:
            masked_height = height_grid.copy()
            masked_height[~source_mask] = np.nan
            
            for _ in range(iterations):
                gaussian_smoothed = self._gaussian_filter_with_nan(masked_height, sigma)
                
                replace_mask = source_mask & ~protected_mask
                smoothed_height[replace_mask] = gaussian_smoothed[replace_mask]
                
                masked_height[replace_mask] = smoothed_height[replace_mask]
            
        except Exception as e:
            smoothed_height = self._improved_simple_smoothing(
                height_grid, source_mask, protected_mask, 
                sigma, iterations, preserve_edges
            )
        
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

    def _improved_simple_smoothing(self, height_grid, source_mask, protected_mask, 
                                sigma=0.5, iterations=1, preserve_edges=True):
        smoothed_height = height_grid.copy()
        
        radius = max(1, int(np.ceil(2 * sigma)))
        
        weights = np.zeros((2*radius + 1, 2*radius + 1))
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                distance = np.sqrt(i*i + j*j)
                weights[i+radius, j+radius] = np.exp(-distance*distance / (2 * sigma * sigma))
        
        for iteration in range(iterations):
            new_heights = smoothed_height.copy()
            
            smooth_indices = np.where(source_mask & ~protected_mask)
            
            for idx in range(len(smooth_indices[0])):
                i, j = smooth_indices[0][idx], smooth_indices[1][idx]
                
                total_weight = 0.0
                weighted_sum = 0.0
                
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        ni, nj = i + di, j + dj
                        
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
        from scipy.ndimage import binary_dilation, binary_erosion
        
        dilated = binary_dilation(source_mask, structure=np.ones((3, 3)))
        eroded = binary_erosion(source_mask, structure=np.ones((3, 3)))
        
        edge_mask = dilated & ~eroded
        
        return edge_mask
    
    def _smooth_lifted_area(self, height_grid, lifted_mask, sigma=2.0):
        if not np.any(lifted_mask):
            return height_grid
        
        smoothed_height = height_grid.copy()
        
        try:
            gaussian_smoothed = gaussian_filter(height_grid, sigma=sigma)
            
            smoothed_height[lifted_mask] = gaussian_smoothed[lifted_mask]
            
        except Exception as e:
            for _ in range(3):
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
    
    def _blur_lift_boundary(self, height_grid, source_mask, boundary_points, blur_radius=3):
        if len(boundary_points) == 0 or blur_radius <= 0:
            return height_grid
        
        boundary_points_array = np.array(boundary_points)
        
        blurred_height = height_grid.copy()
        
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
        except Exception as e:
            return height_grid
        
        h, w = height_grid.shape
        boundary_influence_mask = np.zeros((h, w), dtype=bool)
        
        for i in range(h):
            for j in range(w):
                if source_mask[i, j]:
                    continue
                
                x = self._current_x_grid[j] if hasattr(self, '_current_x_grid') else j
                y = self._current_y_grid[i] if hasattr(self, '_current_y_grid') else i
                
                distances, _ = boundary_tree.query([x, y], k=1)
                
                if distances <= blur_radius:
                    boundary_influence_mask[i, j] = True
        
        if np.any(boundary_influence_mask):
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
        
        return base_distance, min_distance, max_distance
    
    def _apply_adaptive_vertex_lift(self, height_grid, source_mask, x_grid, y_grid, boundary_points):
        if len(boundary_points) == 0:
            return height_grid, np.zeros_like(source_mask, dtype=bool)
        
        self._current_x_grid = x_grid
        self._current_y_grid = y_grid
        
        boundary_points_array = np.array(boundary_points)
        boundary_heights = boundary_points_array[:, 2]
        
        base_distance, min_distance, max_distance = self._calculate_adaptive_distance_parameters(boundary_heights)
        
        try:
            boundary_tree = KDTree(boundary_points_array[:, :2])
            use_kdtree = True
        except Exception as e:
            use_kdtree = False
        
        min_boundary_height = np.min(boundary_heights)
        max_boundary_height = np.max(boundary_heights)
        height_range = max_boundary_height - min_boundary_height
        
        corrected_height = height_grid.copy()
        
        lifted_mask = np.zeros_like(source_mask, dtype=bool)
        
        lifted_vertices_count = 0
        total_lift_amount = 0.0
        min_lift = float('inf')
        max_lift = float('-inf')
        
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
        
        total_vertices = np.sum(~source_mask)
        if lifted_vertices_count > 0:
            avg_lift = total_lift_amount / lifted_vertices_count
        
        return corrected_height, lifted_mask
    
    def create_extended_mesh(self, world_position=(0, 0, 2), source_scale_x=None,
                source_scale_y=None, source_scale_z=None, 
                source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0,
                source_rotation_x=None, source_rotation_y=None, source_rotation_z=None):
        return self.create_unified_perlin_mesh_with_lift(world_position, source_scale_x,
                                                        source_scale_y, source_scale_z,
                                                        source_offset_x, source_offset_y, source_offset_z,
                                                        source_rotation_x, source_rotation_y, source_rotation_z)
    
    def create_unified_perlin_mesh_with_lift(self, world_position=(0, 0, 2), source_scale_x=None,
                                            source_scale_y=None, source_scale_z=None,
                                            source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0, source_rotation_x=None, source_rotation_y=None, source_rotation_z=None):
        if self.height_map is None:
            if not self.load_height_map():
                return None
        
        h, w = self.height_map.shape
        
        if not hasattr(self, 'mask'):
            self.mask = self.height_map > self.alpha_threshold
        
        y_indices, x_indices = np.where(self.mask)
        
        if len(y_indices) == 0:
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
        
        source_points = []
        for y, x in zip(y_indices, x_indices):
            height = self.height_map[y, x]
            
            # Применяем масштабирование
            world_x = (x / w - 0.5) * source_scale_x
            world_y = (y / h - 0.5) * source_scale_y
            world_z = height * source_scale_z
            
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
        
        source_points = np.array(source_points)
        
        min_x = world_position[0] - self.target_width / 2
        max_x = world_position[0] + self.target_width / 2
        min_y = world_position[1] - self.target_height / 2
        max_y = world_position[1] + self.target_height / 2
        
        x_grid = np.linspace(min_x, max_x, self.grid_resolution)
        y_grid = np.linspace(min_y, max_y, self.grid_resolution)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        
        source_height_grid = np.full((len(y_grid), len(x_grid)), np.nan, dtype=np.float32)
        source_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)
        
        for wx, wy, wz in source_points:
            grid_x_idx = np.argmin(np.abs(x_grid - wx))
            grid_y_idx = np.argmin(np.abs(y_grid - wy))
            
            if (abs(x_grid[grid_x_idx] - wx) < (x_grid[1] - x_grid[0]) * 0.5 and
                abs(y_grid[grid_y_idx] - wy) < (y_grid[1] - y_grid[0]) * 0.5):
                
                source_height_grid[grid_y_idx, grid_x_idx] = wz
                source_mask[grid_y_idx, grid_x_idx] = True
        
        boundary_points = []
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
        
        perlin_height = np.zeros((len(y_grid), len(x_grid)), dtype=np.float32)
        
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x = x_grid[j]
                y = y_grid[i]
                
                noise_value = self._generate_perlin_noise(x, y)
                noise_height = (noise_value + 1.0) / 2.0 * self.noise_strength
                perlin_height[i, j] = noise_height
        
        base_height_grid = perlin_height + self.base_height
        
        if len(source_points) > 0:
            key_x = source_points[:, 0]
            key_y = source_points[:, 1]
            key_z = source_points[:, 2]
            
            try:
                if self.interpolation_method == 'rbf':
                    rbf = Rbf(key_x, key_y, key_z, 
                              function='multiquadric', 
                              smooth=self.rbf_smooth)
                    
                    key_surface = rbf(grid_x, grid_y)
                    
                elif self.interpolation_method in ['linear', 'cubic']:
                    key_surface = griddata((key_x, key_y), key_z, 
                                          (grid_x, grid_y), 
                                          method=self.interpolation_method,
                                          fill_value=self.base_height)
                else:
                    raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
                
                blend_weights = np.zeros_like(grid_x, dtype=np.float32)
                
                for i in range(len(y_grid)):
                    for j in range(len(x_grid)):
                        if source_mask[i, j]:
                            blend_weights[i, j] = 1.0
                        else:
                            distances = np.sqrt((key_x - grid_x[i, j])**2 + (key_y - grid_y[i, j])**2)
                            min_distance = np.min(distances) if len(distances) > 0 else float('inf')
                            
                            influence_distance = 1.0
                            if min_distance < influence_distance:
                                weight = 1.0 - (min_distance / influence_distance)
                                weight = weight * weight
                                blend_weights[i, j] = weight
                
                corrected_height = base_height_grid * (1.0 - blend_weights) + key_surface * blend_weights
                
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
        
        if self.source_mesh_smoothing_enabled and np.any(source_mask):
            corrected_height = self._smooth_source_mesh(
                corrected_height, source_mask,
                sigma=self.source_mesh_smoothing_sigma,
                iterations=self.source_mesh_smoothing_iterations,
                preserve_edges=self.source_mesh_edge_preserving
            )
            
            if self.source_mesh_smoothing_sigma > 1.0:  
                corrected_height = self._postprocess_boundary_zone(
                    corrected_height, source_mask, 
                    boundary_width=int(self.source_mesh_smoothing_sigma * 2)
                )
        
        if self.adaptive_lift_enabled and len(boundary_points) > 0:
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
            corrected_height = self._smooth_heightfield(corrected_height, self.smoothing_iterations)
        
        format = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("unified_perlin_mesh_with_lift", format, Geom.UHStatic)
        
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        vertices = []
        vertex_indices = {}
        
        for i, y in enumerate(y_grid):
            for j, x in enumerate(x_grid):
                z = corrected_height[i, j]
                
                vertex.addData3f(x, y, z)
                normal.addData3f(0, 0, 1) 
                texcoord.addData2f(j / (len(x_grid)-1), i / (len(y_grid)-1))
                
                vertex_indices[(i, j)] = len(vertices)
                vertices.append((x, y, z))
        
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
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("unified_perlin_mesh_with_lift")
        node.addGeom(geom)
        
        self._calculate_normals(vdata, geom)
        
        return node
    
    def _postprocess_boundary_zone(self, height_grid, source_mask, boundary_width=5):
        from scipy.ndimage import distance_transform_edt
        
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
    
    def _create_simple_extended_mesh(self, x_grid, y_grid, source_mask, avg_source_height, 
                                     world_position, vdata, vertex, normal, texcoord):
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
        
        geom = Geom(vdata)
        geom.addPrimitive(triangles)
        
        node = GeomNode("simple_extended_mesh")
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
            
            nx, ny, nz = self._apply_rotation(nx, ny, nz)
            normal_writer.setData3f(nx, ny, nz)
    
    def add_to_scene(self, position=(0, 0, 2)):
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        node = self.create_mesh_in_world_coordinates(position)
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
                self.app.model_paths[id(self.mesh_node)] = "height_map_mesh_world"
            
            return self.mesh_node
        
        return None
    
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
        if self.mesh_node:
            self.mesh_node.removeNode()
        
        node = self.create_extended_mesh(position, source_scale_x, source_scale_y, 
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
            self.mesh_node.setPos(0, 0, 2)
            
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "extended_height_map_mesh"
            
            return self.mesh_node
        
        return None
    
    def add_unified_mesh_with_lift_to_scene(self, position=(0, 0, 2), source_scale_x=None,
                                       source_scale_y=None, source_scale_z=None,
                                       source_offset_x=0.0, source_offset_y=0.0, source_offset_z=0.0,
                                       source_rotation_x=None, source_rotation_y=None, source_rotation_z=None):
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
            self.mesh_node.setPos(0, 0, 2)
            
            if not hasattr(self.app, 'loaded_models'):
                self.app.loaded_models = []
            if not hasattr(self.app, 'model_paths'):
                self.app.model_paths = {}
            
            if self.mesh_node not in self.app.loaded_models:
                self.app.loaded_models.append(self.mesh_node)
                self.app.model_paths[id(self.mesh_node)] = "unified_perlin_mesh_with_lift"
            
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