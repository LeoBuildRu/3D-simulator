import os
import time
import math
import re
import glob
import json
import datetime
import random
from panda3d.core import *

class RendererUtils:
    def __init__(self, panda_app):
        self.panda_app = panda_app

    def barrel_distortion(self, img, k1=0.15, k2=0.35):
        tex = Texture()
        tex.load(img)
        
        distortion_map = PNMImage(img.getXSize(), img.getYSize())
        
        width = img.getXSize()
        height = img.getYSize()
        center_x = width / 2.0
        center_y = height / 2.0
        
        max_dist = min(center_x, center_y)
        
        for y in range(height):
            for x in range(width):
                norm_x = (x - center_x) / max_dist
                norm_y = (y - center_y) / max_dist
                
                r = math.sqrt(norm_x * norm_x + norm_y * norm_y)
                
                distortion = 1.0 + k1 * r * r + k2 * r * r * r * r
                
                new_x = norm_x * distortion
                new_y = norm_y * distortion
                
                src_x = int(center_x + new_x * max_dist)
                src_y = int(center_y + new_y * max_dist)
                
                if 0 <= src_x < width and 0 <= src_y < height:
                    color = img.getXel(src_x, src_y)
                    distortion_map.setXel(x, y, color)
                else:
                    distortion_map.setXel(x, y, 0, 0, 0)
        
        return distortion_map
    
    def crop_image(self, img, left=270, top=155, right=1850, bottom=925):
        width = img.getXSize()
        height = img.getYSize()
        
        if left < 0 or top < 0 or right > width or bottom > height:
            print(f"Предупреждение: координаты обрезки выходят за пределы изображения")
            print(f"Размер изображения: {width}x{height}")
            print(f"Запрошенные координаты: left={left}, top={top}, right={right}, bottom={bottom}")
            
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
        
        if left >= right or top >= bottom:
            print(f"Ошибка: некорректные координаты обрезки")
            print(f"left={left}, top={top}, right={right}, bottom={bottom}")
            return PNMImage(img)
        
        crop_width = right - left
        crop_height = bottom - top
        
        cropped_img = PNMImage(crop_width, crop_height, img.getNumChannels(), img.getMaxval())
        
        for y in range(crop_height):
            src_y = top + y
            for x in range(crop_width):
                src_x = left + x
                color = img.getXel(src_x, src_y)
                cropped_img.setXel(x, y, color)
        
        return cropped_img
    
    def stretch_to_1920x1080(self, img):
        target_width = 1920
        target_height = 1080
        
        current_width = img.getXSize()
        current_height = img.getYSize()
        
        if current_width == target_width and current_height == target_height:
            return PNMImage(img)  
        
        stretched_img = PNMImage(target_width, target_height, img.getNumChannels(), img.getMaxval())
        
        scale_x = target_width / current_width
        scale_y = target_height / current_height
        
        for y in range(target_height):
            for x in range(target_width):
                src_x = x / scale_x
                src_y = y / scale_y
                
                x0 = int(math.floor(src_x))
                x1 = min(x0 + 1, current_width - 1)
                y0 = int(math.floor(src_y))
                y1 = min(y0 + 1, current_height - 1)
                
                dx = src_x - x0
                dy = src_y - y0
                
                c00 = img.getXel(x0, y0)
                c10 = img.getXel(x1, y0)
                c01 = img.getXel(x0, y1)
                c11 = img.getXel(x1, y1)
                
                c0 = [c00[i] * (1 - dx) + c10[i] * dx for i in range(3)]
                c1 = [c01[i] * (1 - dx) + c11[i] * dx for i in range(3)]
                
                color = [c0[i] * (1 - dy) + c1[i] * dy for i in range(3)]
                
                stretched_img.setXel(x, y, color[0], color[1], color[2])
        
        return stretched_img
    
    def _process_render_image(self, img, camera_fov_x=None, camera_fov_y=None, output_dir="renders", 
                              filename_prefix="render", metadata=None):
        orig_width = img.getXSize()
        orig_height = img.getYSize()
        
        fx = fy = cx = cy = None
        lens = self.panda_app.cam.node().getLens() if hasattr(self.panda_app, 'cam') else None
        
        if camera_fov_x is not None and camera_fov_y is not None:
            fx = (orig_width / 2.0) / math.tan(math.radians(camera_fov_x / 2.0))
            fy = (orig_height / 2.0) / math.tan(math.radians(camera_fov_y / 2.0))
            cx = orig_width / 2.0
            cy = orig_height / 2.0
        
        # Определяем 3D точки для преобразования
        top_points_3d = [
            (-1.03, -2.22, 2.4),   # bottom_left_top
            (-1.03, 2.4, 2.4),     # top_left_top
            (1.045, 2.4, 2.4),     # top_right_top
            (1.045, -2.22, 2.4)    # bottom_right_top
        ]
        
        top_points_2d = []
        distances_to_camera = []  # Для хранения расстояний до камеры
        
        # Преобразуем 3D точки в 2D пиксельные координаты (до всех преобразований)
        if lens:
            for i, point_3d in enumerate(top_points_3d):
                # Создаем точку в координатах Panda3D
                point = LPoint3f(point_3d[0], point_3d[1], point_3d[2])
                
                # Преобразуем мировые координаты в координаты камеры
                point_in_camera_space = self.panda_app.camera.getRelativePoint(
                    self.panda_app.render, 
                    point
                )
                
                # Вычисляем расстояние от камеры до точки (в мировых координатах)
                # Получаем позицию камеры в мировых координатах
                camera_pos = self.panda_app.camera.getPos(self.panda_app.render)
                
                # Вычисляем расстояние между камерой и точкой
                distance = math.sqrt(
                    (point_3d[0] - camera_pos.x) ** 2 +
                    (point_3d[1] - camera_pos.y) ** 2 +
                    (point_3d[2] - camera_pos.z) ** 2
                )
                
                distances_to_camera.append(float(distance))
                
                # Создаем точки для результата
                result_point = LPoint3f()
                
                # Пытаемся спроецировать точку
                success = lens.project(point_in_camera_space, result_point)
                
                if success:
                    # Преобразуем NDC в пиксельные координаты
                    pixel_x = (result_point.x * 0.5 + 0.5) * orig_width
                    pixel_y = (0.5 - result_point.y * 0.5) * orig_height  # Инвертируем Y
                    
                    # Проверяем, находится ли точка в пределах экрана
                    if (0 <= pixel_x < orig_width and 0 <= pixel_y < orig_height and 
                        point_in_camera_space.y > 0 and 0 <= result_point.z <= 1):
                        top_points_2d.append({
                            "x": float(pixel_x),
                            "y": float(pixel_y)
                        })
                    else:
                        top_points_2d.append(None)
                else:
                    top_points_2d.append(None)
        else:
            top_points_2d = [None] * len(top_points_3d)
            distances_to_camera = [None] * len(top_points_3d)
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)

        # Параметры преобразований
        k1 = 0.30
        k2 = 0.35
        crop_left = 423
        crop_top = 238
        crop_right = 1498
        crop_bottom = 840
        final_width = 1920
        final_height = 1080
        
        # Создаем копии изображения для каждого этапа преобразования
        img_distorted = self.barrel_distortion(img, k1=k1, k2=k2)
        img_cropped = self.crop_image(img_distorted, left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom)
        img_final = self.stretch_to_1920x1080(img_cropped)
        
        # Преобразуем 2D точки с учетом всех примененных трансформаций
        transformed_points_2d = []
        
        if top_points_2d:
            for i, point_2d in enumerate(top_points_2d):
                if point_2d is None:
                    transformed_points_2d.append(None)
                    continue
                    
                x = point_2d["x"]
                y = point_2d["y"]
                
                # Итерационный метод для нахождения правильных координат после barrel distortion
                x_dist = x
                y_dist = y
                
                center_x = orig_width / 2.0
                center_y = orig_height / 2.0
                max_dist = min(center_x, center_y)
                
                # Итерационный метод для решения обратной задачи
                for iteration in range(10):
                    norm_x_dist = (x_dist - center_x) / max_dist
                    norm_y_dist = (y_dist - center_y) / max_dist
                    
                    r = math.sqrt(norm_x_dist * norm_x_dist + norm_y_dist * norm_y_dist)
                    distortion = 1.0 + k1 * r * r + k2 * r * r * r * r
                    
                    # Прямое преобразование (точка в искаженном изображении)
                    norm_x_distorted = norm_x_dist * distortion
                    norm_y_distorted = norm_y_dist * distortion
                    
                    x_calc = center_x + norm_x_distorted * max_dist
                    y_calc = center_y + norm_y_distorted * max_dist
                    
                    # Вычисляем ошибку
                    error_x = x_calc - x
                    error_y = y_calc - y
                    
                    # Корректируем предположение
                    x_dist -= error_x * 0.5
                    y_dist -= error_y * 0.5
                    
                    if abs(error_x) < 0.1 and abs(error_y) < 0.1:
                        break
                
                # 2. Применяем crop (вычитаем смещение)
                x_cropped = x_dist - crop_left
                y_cropped = y_dist - crop_top
                
                # Проверяем, попадает ли точка в область crop
                crop_width = crop_right - crop_left
                crop_height = crop_bottom - crop_top
                
                if (0 <= x_cropped < crop_width and 0 <= y_cropped < crop_height):
                    
                    # 3. Применяем stretch (масштабирование до 1920x1080)
                    scale_x = final_width / crop_width
                    scale_y = final_height / crop_height
                    
                    x_final = x_cropped * scale_x
                    y_final = y_cropped * scale_y
                    
                    transformed_points_2d.append({
                        "x": float(x_final),
                        "y": float(y_final)
                    })
                else:
                    transformed_points_2d.append(None)
        
        # === ДОБАВЛЕНИЕ ЦВЕТНЫХ КРУГОВ ===
        try:
            from PIL import Image, ImageDraw
            import io
            from panda3d.core import StringStream
            
            # Конвертируем PNMImage в PIL Image
            stream = StringStream()
            img_final.write(stream, "png")
            pil_img = Image.open(io.BytesIO(stream.getData()))
            draw = ImageDraw.Draw(pil_img)
            
            colors = [
                (255, 0, 0, 200),    # красный для bottom_left_top
                (0, 255, 0, 200),    # зеленый для top_left_top
                (0, 0, 255, 200),    # синий для top_right_top
                (255, 255, 0, 200),  # желтый для bottom_right_top
            ]
            
            # Рисуем круги для каждой точки, которая не None
            for i, point_2d in enumerate(transformed_points_2d):
                if point_2d is not None:
                    x = int(point_2d["x"])
                    y = int(point_2d["y"])
                    color = colors[i % len(colors)]
                    
                    # Рисуем круг с радиусом 10 пикселей
                    draw.ellipse([(x-10, y-10), (x+10, y+10)], fill=color, outline=(255, 255, 255, 255))
            
            # Конвертируем обратно в PNMImage
            output = io.BytesIO()
            pil_img.save(output, format="PNG")
            output.seek(0)
            new_img = PNMImage()
            new_img.read(StringStream(output.read()), "png")
            
            # Заменяем исходное изображение на новое
            img_final = new_img
            
        except ImportError:
            print("Warning: Pillow not installed. Skipping circle drawing.")
        except Exception as e:
            print(f"Warning: Error while drawing circles: {e}")
        
        # Сохраняем финальное изображение
        img_final.write(Filename.from_os_specific(output_path))
        
        # Формируем render_metadata только с необходимыми данными
        render_metadata = {}
        
        # Параметры barrel distortion
        render_metadata["barrel_distortion"] = {
            "k1": k1,
            "k2": k2
        }
        
        # Параметры камеры
        if camera_fov_x is not None and camera_fov_y is not None:
            render_metadata["camera_params"] = {
                "fov_x": camera_fov_x,
                "fov_y": camera_fov_y,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
        
        # 3D точки
        render_metadata["points_3d"] = top_points_3d
        
        # 2D точки после преобразований
        render_metadata["points_2d"] = []
        for i, point_2d in enumerate(transformed_points_2d):
            if point_2d is not None:
                render_metadata["points_2d"].append({
                    "x": point_2d["x"],
                    "y": point_2d["y"]
                })
            else:
                render_metadata["points_2d"].append(None)
        
        # Расстояния от камеры до 3D точек
        render_metadata["distances_to_camera"] = distances_to_camera
        
        # Добавляем Target_Volume
        if hasattr(self.panda_app, 'Target_Volume'):
            render_metadata["target_volume"] = self.panda_app.Target_Volume
        else:
            render_metadata["target_volume"] = None
        
        # Добавляем current_texture_set['diffuse']
        if (hasattr(self.panda_app, 'current_texture_set') and 
            self.panda_app.current_texture_set and 
            'diffuse' in self.panda_app.current_texture_set):
            render_metadata["texture_diffuse"] = self.panda_app.current_texture_set['diffuse']
        else:
            render_metadata["texture_diffuse"] = None
        
        json_path = output_path.replace(".png", ".json")
        with open(json_path, 'w') as f:
            json.dump(render_metadata, f, indent=2)
        
        return output_path
    
    def create_video_from_frames(self, output_dir="renders/datasets_metric_/", video_name="camera_rotation.mp4", fps=20):
        search_pattern = os.path.join(output_dir, "render_*_frame_*.png")
        frame_files = glob.glob(search_pattern)
        
        frame_files.sort(key=lambda x: int(re.search(r'frame_(\d{3})', x).group(1)))
        
        if not frame_files:
            return False
        
        list_file = os.path.join(os.getcwd(), "frames_list.txt")
        with open(list_file, "w") as f:
            for frame in frame_files:
                abs_path = os.path.abspath(frame).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-r', str(fps),
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '22',
                '-pix_fmt', 'yuv420p',
                '-y',
                video_name
            ]
            
            return True
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)
    
    def save_single_render(self):
        lens = self.panda_app.cam.node().getLens()
        if isinstance(lens, PerspectiveLens):
            fov = lens.getFov()
            camera_fov_x = fov[0]
            camera_fov_y = fov[1]
        else:
            camera_fov_x = camera_fov_y = None
        
        img = PNMImage()
        if not self.panda_app.win.getScreenshot(img):
            return False
        
        output_path = self._process_render_image(
            img, 
            camera_fov_x=camera_fov_x,
            camera_fov_y=camera_fov_y,
            output_dir="renders/single",
            filename_prefix="single_render",
            metadata={
                "render_type": "single",
                "camera_position": {
                    "x": float(self.panda_app.camera.getX()),
                    "y": float(self.panda_app.camera.getY()),
                    "z": float(self.panda_app.camera.getZ())
                },
                "camera_rotation": {
                    "h": float(self.panda_app.camera.getH()),
                    "p": float(self.panda_app.camera.getP()),
                    "r": float(self.panda_app.camera.getR())
                },
                "model_set": self.panda_app.current_model_set if hasattr(self.panda_app, 'current_model_set') else None,
                "target_volume": self.panda_app.Target_Volume
            }
        )
        
        return True
    
    def save_dataset_render(self):
        original_pos = self.panda_app.camera.getPos()
        original_hpr = self.panda_app.camera.getHpr()
        original_view = self.panda_app.current_view
        original_target_volume = self.panda_app.Target_Volume
        
        if not self.panda_app.current_model_set:
            return False
        
        if not all([hasattr(self.panda_app, 'current_other_path'), 
                   hasattr(self.panda_app, 'current_cuzov_path'), 
                   hasattr(self.panda_app, 'current_napolnitel_path')]):
            return False
        
        fixed_pos = (8.599995136260986, 6.0011109376791865e-05, 21.70002269744873)
        fixed_hpr = (89.99999237060547, -66.110355377197266, 0.0)
        fixed_fov_x = 48.0
        fixed_fov_y = 26.14091682434082
        
        volumes = [0.5 + i * 0.5 for i in range(99)] 
        passes_per_volume = 4
        
        total_renders = len(volumes) * passes_per_volume
        current_render = 0
        
        for i, volume in enumerate(volumes):
            for pass_num in range(passes_per_volume):
                current_render += 1
                
                self.panda_app.Target_Volume = volume
                
                self.panda_app.clear_scene()
                self.panda_app.load_gltf_model(self.panda_app.current_other_path)
                self.panda_app.load_gltf_model(self.panda_app.current_cuzov_path)
                self.panda_app.load_gltf_model(self.panda_app.current_napolnitel_path)
                
                self.panda_app.create_ground_plane()
                if hasattr(self.panda_app, 'current_ground_plane_z'):
                    self.panda_app.ground_plane.setPos(0, 0, self.panda_app.current_ground_plane_z)
                
                success_aabb = self.panda_app.perform_AABB_plane()
                if not success_aabb:
                    continue
                
                if not hasattr(self.panda_app, 'Perlin_Seed'):
                    self.panda_app.Perlin_Seed = random.randint(0, 10000000)
                else:
                    self.panda_app.Perlin_Seed = random.randint(0, 10000000) + pass_num * 1000000 + i * 100000000
                
                success_perlin = self.panda_app.perlin_generator.generate_perlin_mesh_from_csg()
                if not success_perlin:
                    continue
                
                time.sleep(1.0)
                
                self.panda_app.camera.setPos(*fixed_pos)
                self.panda_app.camera.setHpr(*fixed_hpr)
                
                lens = self.panda_app.cam.node().getLens()
                if isinstance(lens, PerspectiveLens):
                    lens.setFov(fixed_fov_x, fixed_fov_y)
                
                for _ in range(120):  
                    self.panda_app.taskMgr.step()
                    time.sleep(0.01)
                
                current_pos = self.panda_app.camera.getPos()
                current_hpr = self.panda_app.camera.getHpr()
                
                img = PNMImage()
                if not self.panda_app.win.getScreenshot(img):
                    continue
                
                output_path = self._process_render_image(
                    img,
                    camera_fov_x=fixed_fov_x,
                    camera_fov_y=fixed_fov_y,
                    output_dir="renders/datasets_metric_",
                    filename_prefix=f"render_volume_{volume:.1f}_pass_{pass_num:02d}",
                    metadata={
                        "render_type": "dataset",
                        "target_volume": volume,
                        "pass_number": pass_num,
                        "volume_index": i,
                        "perlin_seed": self.panda_app.Perlin_Seed,
                        "model_set": self.panda_app.current_model_set,
                        "camera_position": {
                            "x": float(current_pos.x),
                            "y": float(current_pos.y),
                            "z": float(current_pos.z)
                        },
                        "camera_rotation": {
                            "h": float(current_hpr.x),
                            "p": float(current_hpr.y),
                            "r": float(current_hpr.z)
                        }
                    }
                )
        
        self.panda_app.Target_Volume = original_target_volume
        self.panda_app.camera.setPos(original_pos)
        self.panda_app.camera.setHpr(original_hpr)
        self.panda_app.current_view = original_view
        
        return True