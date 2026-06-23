import os
import time
import math
import re
import glob
import json
import datetime
import random
from panda3d.core import *

# Каталог случайных фонов для замены фона на обычном рендере.
# renderer_utils.py лежит в <root>/src/rendering/, поэтому корень — три уровня вверх.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKGROUNDS_DIR = os.path.join(_PROJECT_ROOT, "assets", "backgrounds")
_BACKGROUND_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class RendererUtils:
    def __init__(self, panda_app):
        self.panda_app = panda_app

    def _pick_random_background(self):
        """Случайный путь к фоновой картинке из assets/backgrounds или None."""
        try:
            files = [
                os.path.join(_BACKGROUNDS_DIR, f)
                for f in os.listdir(_BACKGROUNDS_DIR)
                if f.lower().endswith(_BACKGROUND_EXTS)
            ]
        except OSError:
            return None
        return random.choice(files) if files else None

    def _composite_random_background(self, img_final, mask_final, bg_path):
        """Заменить фон на цветном кадре случайной картинкой.

        Передний план (кузов + груз) определяется по маске сегментации
        mask_final (уже с теми же дисторсией/кропом/растяжением, что и
        img_final). Все пиксели, кроме cargo/cuzov, заполняются картинкой
        bg_path. Возвращает новый PNMImage или None при ошибке.
        """
        try:
            import io
            import numpy as np
            from PIL import Image
            from src.rendering.segmentation_renderer import SEG_COLORS
        except Exception as exc:
            print(f"[Render] замена фона недоступна (PIL/numpy): {exc}")
            return None

        def _pnm_to_rgb_array(pnm):
            stream = StringStream()
            pnm.write(stream, "png")
            pil = Image.open(io.BytesIO(stream.getData())).convert("RGB")
            return np.asarray(pil), pil.size

        try:
            img_arr, size = _pnm_to_rgb_array(img_final)
            mask_arr, _ = _pnm_to_rgb_array(mask_final)
            bg_pil = Image.open(bg_path).convert("RGB").resize(
                size, Image.LANCZOS)
            bg_arr = np.asarray(bg_pil)

            mask_i = mask_arr.astype(np.int16)

            def _close(color, tol=40):
                d = np.abs(mask_i - np.array(color, dtype=np.int16))
                return (d[..., 0] <= tol) & (d[..., 1] <= tol) & (d[..., 2] <= tol)

            # Передний план = груз (cargo) + кузов (cuzov). Остальное — фон.
            cuzov_mask = _close(SEG_COLORS["cuzov"])
            keep = _close(SEG_COLORS["cargo"]) | cuzov_mask

            # 1) Цветовую температуру переднего плана подгоняем под фон, чтобы
            #    вставленный кузов+груз не выглядели «холоднее/теплее» картинки.
            fg_arr = self._match_color_temperature(img_arr, keep, bg_arr)

            # 2) Яркость фоновой картинки подгоняем под яркость рендера кузова
            #    (эталон — пиксели cuzov; если их мало, берём весь передний
            #    план). Так фон тускнеет ночью и светлеет днём вместе со сценой.
            ref_mask = cuzov_mask if int(cuzov_mask.sum()) >= 64 else keep
            bg_arr = self._match_brightness(bg_arr, img_arr, ref_mask)

            out = np.where(keep[..., None], fg_arr, bg_arr).astype(np.uint8)

            out_buf = io.BytesIO()
            Image.fromarray(out).save(out_buf, format="PNG")
            out_buf.seek(0)
            new_pnm = PNMImage()
            new_pnm.read(StringStream(out_buf.read()), "png")
            return new_pnm
        except Exception as exc:
            print(f"[Render] ошибка замены фона: {exc}")
            return None

    def _match_color_temperature(self, img_arr, keep, bg_arr,
                                 strength=0.85, gain_min=0.6, gain_max=1.7):
        """Подогнать цветовую температуру переднего плана под фон.

        Оцениваем «точку белого» каждого изображения как среднюю линейную
        яркость по каналам (gray-world): для фона — по всей картинке, для
        переднего плана — только по сохраняемым пикселям (keep). Затем —
        яркостно-нейтральная адаптация фон Криза: масштабируем каналы R и B
        переднего плана так, чтобы баланс R/G и B/G совпал с фоном (G не
        трогаем, чтобы не менять общую яркость). Тёплый фон → передний план
        теплеет, холодный → холоднеет.

        img_arr, bg_arr — HxWx3 uint8 (RGB); keep — HxW bool.
        Возвращает HxWx3 uint8 (только передний план реально используется).
        strength — сила коррекции (0..1); gain_min/max — клип усиления.
        """
        import numpy as np

        eps = 1e-4

        def to_linear(a):
            # sRGB(8-бит) -> линейный свет (приближённая гамма 2.2).
            return np.power(a.astype(np.float32) / 255.0, 2.2)

        fg_lin = to_linear(img_arr)
        fg_pixels = fg_lin[keep]
        # Слишком мало переднего плана — не из чего оценивать, не трогаем.
        if fg_pixels.shape[0] < 64:
            return img_arr

        fg_mean = fg_pixels.reshape(-1, 3).mean(axis=0) + eps
        bg_mean = to_linear(bg_arr).reshape(-1, 3).mean(axis=0) + eps

        # Баланс относительно зелёного (нормировка по яркости).
        fg_wb = fg_mean / fg_mean[1]
        bg_wb = bg_mean / bg_mean[1]
        gains = bg_wb / fg_wb                       # (g_r, 1.0, g_b)

        # Умеренная сила + клип, чтобы не уехать в крайности.
        gains = 1.0 + (gains - 1.0) * float(strength)
        gains = np.clip(gains, gain_min, gain_max)
        gains[1] = 1.0                              # зелёный не трогаем

        corrected = np.clip(fg_lin * gains, 0.0, 1.0)
        corrected = np.power(corrected, 1.0 / 2.2) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _match_brightness(self, bg_arr, fg_arr, ref_mask,
                          scale_min=0.02, scale_max=8.0):
        """Подогнать яркость фоновой картинки под яркость рендера.

        Считаем среднюю линейную яркость (luma Rec.709) эталонной области
        переднего плана (ref_mask — обычно пиксели кузова) и всей фоновой
        картинки, затем масштабируем фон так, чтобы его средняя яркость
        совпала с эталоном. Тёмный (ночной) рендер -> фон темнеет, светлый
        (дневной) -> фон светлеет.

        bg_arr, fg_arr — HxWx3 uint8 (RGB); ref_mask — HxW bool.
        Возвращает HxWx3 uint8.
        """
        import numpy as np

        eps = 1e-4
        coef = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        def to_linear(a):
            return np.power(a.astype(np.float32) / 255.0, 2.2)

        ref_pixels = to_linear(fg_arr)[ref_mask]
        # Слишком мало эталона — не из чего оценивать, фон не трогаем.
        if ref_pixels.shape[0] < 64:
            return bg_arr

        fg_lum = float((ref_pixels.reshape(-1, 3) @ coef).mean()) + eps
        bg_lin = to_linear(bg_arr)
        bg_lum = float((bg_lin.reshape(-1, 3) @ coef).mean()) + eps

        scale = float(np.clip(fg_lum / bg_lum, scale_min, scale_max))

        out = np.clip(bg_lin * scale, 0.0, 1.0)
        out = np.power(out, 1.0 / 2.2) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

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
    
    def fix_alpha_to_opaque(self, img):
        # If image has no alpha channel, add one
        if not img.hasAlpha():
            img.addAlpha()

        width = img.getXSize()
        height = img.getYSize()

        for y in range(height):
            for x in range(width):
                img.setAlpha(x, y, 1.0)

        return img

    def stretch_to_1920x1080(self, img, nearest=False):
        # nearest=True — ближайший сосед (без интерполяции). Нужен для масок
        # сегментации: билинейное смешение размывало бы границы классов и
        # порождало промежуточные цвета, которых нет в палитре.
        target_width = 1920
        target_height = 1080

        current_width = img.getXSize()
        current_height = img.getYSize()

        if current_width == target_width and current_height == target_height:
            return PNMImage(img)

        stretched_img = PNMImage(target_width, target_height, img.getNumChannels(), img.getMaxval())

        scale_x = target_width / current_width
        scale_y = target_height / current_height

        if nearest:
            for y in range(target_height):
                src_y = min(int(y / scale_y), current_height - 1)
                for x in range(target_width):
                    src_x = min(int(x / scale_x), current_width - 1)
                    stretched_img.setXel(x, y, img.getXel(src_x, src_y))
            return stretched_img

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
    
    def _process_render_image(self, img, depthImg=None, camera_fov_x=None, camera_fov_y=None, output_dir="renders",
                         filename_prefix="render", metadata=None, dataset_type="depth",
                         seg_mask=None, bg_path=None):
        # dataset_type: "depth" — depthImg это карта глубины (суффикс _depth);
        #               "segmentation" — depthImg это маска сегментации
        #               (суффикс _seg, масштабирование ближайшим соседом,
        #                чтобы не размывать границы классов).
        # seg_mask + bg_path: заменить фон на цветном кадре (только на нём!)
        #               случайной картинкой bg_path. seg_mask (1920x1080)
        #               проходит ту же дисторсию и используется как вырез
        #               переднего плана (кузов + груз).
        is_segmentation = (dataset_type == "segmentation")
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
        second_suffix = "_seg" if is_segmentation else "_depth"
        filenameDepth = f"{filename_prefix}_{timestamp}{second_suffix}.png"
        output_path_depth = os.path.join(output_dir, filenameDepth)

        # Параметры преобразований
        k1 = 0.04
        k2 = k1

        crop_div = 1.5

        crop_left = round(423/crop_div)
        crop_top = round(238/crop_div)
        final_width = 1920
        final_height = 1080
        crop_right = final_width - crop_left
        crop_bottom = final_height - crop_top
        
        # Создаем копии изображения для каждого этапа преобразования
        img_distorted = self.barrel_distortion(img, k1=k1, k2=k2)
        img_cropped = self.crop_image(img_distorted, left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom)
        img_final = self.stretch_to_1920x1080(img_cropped)

        # Замена фона случайной картинкой — ТОЛЬКО на цветном кадре и уже
        # после дисторсии. Маску гоним через те же дисторсию/кроп/растяжение,
        # затем оставляем кузов+груз, остальное заливаем картинкой.
        background_name = None
        if bg_path is not None and seg_mask is not None:
            mask_distorted = self.barrel_distortion(seg_mask, k1=k1, k2=k2)
            mask_cropped = self.crop_image(
                mask_distorted, left=crop_left, top=crop_top,
                right=crop_right, bottom=crop_bottom,
            )
            mask_final = self.stretch_to_1920x1080(mask_cropped, nearest=True)
            composited = self._composite_random_background(
                img_final, mask_final, bg_path)
            if composited is not None:
                img_final = composited
                background_name = os.path.basename(bg_path)

        # Те же самые искажения применяем ко второму кадру (карта глубины
        # ИЛИ маска сегментации), чтобы он попиксельно совпадал с цветным.
        # Для сегментации финальный stretch — ближайшим соседом, иначе
        # билинейная интерполяция размыла бы границы классов.
        depth_final = None
        if depthImg is not None:
            depth_distorted = self.barrel_distortion(depthImg, k1=k1, k2=k2)
            depth_cropped = self.crop_image(
                depth_distorted, left=crop_left, top=crop_top,
                right=crop_right, bottom=crop_bottom,
            )
            depth_final = self.stretch_to_1920x1080(
                depth_cropped, nearest=is_segmentation)
            depth_final = self.fix_alpha_to_opaque(depth_final)
        
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
        
        if(False):
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
        if depth_final is not None:
            depth_final.write(Filename.from_os_specific(output_path_depth))
        
        # Формируем render_metadata только с необходимыми данными
        render_metadata = {}

        # Тип датасета (depth / segmentation) + легенда цветов для масок.
        render_metadata["dataset_type"] = dataset_type
        render_metadata["random_background"] = background_name
        render_metadata["second_image"] = (
            os.path.basename(output_path_depth) if depth_final is not None else None
        )
        if is_segmentation:
            try:
                from src.rendering.segmentation_renderer import (
                    SEG_COLORS, SEG_BACKGROUND,
                )
                render_metadata["segmentation_palette"] = {
                    "background": list(SEG_BACKGROUND),
                    **{k: list(v) for k, v in SEG_COLORS.items()},
                }
            except Exception:
                pass

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
        
        # 2D точки до всех преобразований (оригинальные)
        render_metadata["points_2d_original"] = []
        for point_2d in top_points_2d:
            if point_2d is not None:
                render_metadata["points_2d_original"].append({
                    "x": point_2d["x"],
                    "y": point_2d["y"]
                })
            else:
                render_metadata["points_2d_original"].append(None)
        
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
        
        # Сливаем пользовательские метаданные (variant, camera state и т.п.),
        # которые раньше передавались, но игнорировались.
        if metadata:
            for k, v in metadata.items():
                if k not in render_metadata:
                    render_metadata[k] = v
                else:
                    render_metadata[f"extra_{k}"] = v

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

    def wait_panda_render(self, ticks=12):
        # Больше ручных тиков RenderPipeline -> постэффекты (TAA, motion blur,
        # bloom и т.п.) успевают сойтись к стационарной картинке, иначе
        # на скриншотах остаётся "смаз" после движения камеры/смены света.
        for _ in range(max(1, int(ticks))):
            self.panda_app.graphicsEngine.renderFrame()

    def save_single_render(self, output_dir="renders/single",
                           filename_prefix="single_render",
                           extra_metadata=None,
                           dataset_type="depth",
                           random_background=False):
        # dataset_type: "depth" (снимок + карта глубины, как раньше) или
        # "segmentation" (снимок + маска сегментации). Цветной кадр снимается
        # одинаково; меняется только второй кадр.
        #
        # random_background: на ОБЫЧНОМ цветном рендере (после дисторсии) фон
        # сцены/неба заменяется случайной картинкой из assets/backgrounds;
        # передний план (кузов + груз) остаётся. Карта глубины и маска
        # сегментации при этом НЕ меняются — маска используется только чтобы
        # вырезать передний план.
        is_segmentation = (dataset_type == "segmentation")

        lens = self.panda_app.cam.node().getLens()
        if isinstance(lens, PerspectiveLens):
            fov = lens.getFov()
            camera_fov_x = fov[0]
            camera_fov_y = fov[1]
        else:
            camera_fov_x = camera_fov_y = None

        # Скрываем depth overlay, ждём пока кадр устаканится, делаем
        # цветной скриншот. Увеличенный wait_panda_render + sleep здесь
        # нужны, чтобы убрать моушн-блюр от только что выполненных
        # движений камеры / смены освещения.
        self.panda_app.depth_renderer.set_overlay_visibility(False)
        self.wait_panda_render(ticks=14)
        time.sleep(1.0)
        self.wait_panda_render(ticks=6)

        img = PNMImage()
        if not self.panda_app.win.getScreenshot(img):
            self.panda_app.depth_renderer.set_overlay_visibility(False)
            return False

        # Маска сегментации для вырезания переднего плана (только при замене
        # фона). Снимается всегда отдельно — это один дешёвый GPU-кадр.
        seg_mask_raw = None
        if random_background:
            seg_mask_raw = self.panda_app.segmentation_renderer.capture()
            if seg_mask_raw is None:
                print("[Render] seg mask for background replace failed; "
                      "сохраняю без замены фона.")

        if is_segmentation:
            # Маска сегментации рендерится в отдельный offscreen-буфер
            # (плоские цвета, без постобработки). Overlay глубины не нужен.
            if seg_mask_raw is not None:
                depthImg = PNMImage(seg_mask_raw)   # переиспользуем захват
            else:
                depthImg = self.panda_app.segmentation_renderer.capture()
            if depthImg is None:
                print("[Render] segmentation capture failed.")
                return False
        else:
            self.panda_app.depth_renderer.set_overlay_visibility(True)
            self.wait_panda_render(ticks=10)
            time.sleep(1.0)
            self.wait_panda_render(ticks=6)

            depthImg = PNMImage()
            if not self.panda_app.win.getScreenshot(depthImg):
                self.panda_app.depth_renderer.set_overlay_visibility(False)
                return False

            self.panda_app.depth_renderer.set_overlay_visibility(False)

        img = self.stretch_to_1920x1080(img)
        depthImg = self.stretch_to_1920x1080(depthImg, nearest=is_segmentation)
        depthImg = self.fix_alpha_to_opaque(depthImg)

        # Маску под вырезание приводим к 1920x1080 тем же ближайшим соседом
        # (как img/depth), чтобы геометрия совпала.
        seg_mask_1080 = None
        bg_path = None
        if seg_mask_raw is not None:
            seg_mask_1080 = self.stretch_to_1920x1080(seg_mask_raw, nearest=True)
            bg_path = self._pick_random_background()
            if bg_path is None:
                print("[Render] assets/backgrounds пуста — замена фона пропущена.")
                seg_mask_1080 = None

        # Клиентский пересчёт объёма реально сгенерированного меша
        # наполнения (в старой версии писался как actual_volume).
        # Берём final_model — туда perlin_mesh_generator / mesh_reconstruction
        # кладут текущую горку наполнителя.
        actual_volume = None
        calc = getattr(self.panda_app, 'calculate_mesh_volume', None)
        final_model = getattr(self.panda_app, 'final_model', None)
        if callable(calc) and final_model is not None:
            try:
                actual_volume = float(calc(final_model))
            except Exception as exc:
                print(f"[Render] actual_volume calc failed: {exc}")

        metadata = {
            "render_type": "single",
            "camera_position": {
                "x": float(self.panda_app.camera.getX()),
                "y": float(self.panda_app.camera.getY()),
                "z": float(self.panda_app.camera.getZ()),
            },
            "camera_rotation": {
                "h": float(self.panda_app.camera.getH()),
                "p": float(self.panda_app.camera.getP()),
                "r": float(self.panda_app.camera.getR()),
            },
            "model_set": (
                self.panda_app.current_model_set
                if hasattr(self.panda_app, 'current_model_set') else None
            ),
            "target_volume": getattr(self.panda_app, 'Target_Volume', None),
            "actual_volume": actual_volume,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        output_path = self._process_render_image(
            img,
            depthImg,
            camera_fov_x=camera_fov_x,
            camera_fov_y=camera_fov_y,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            metadata=metadata,
            dataset_type=dataset_type,
            seg_mask=seg_mask_1080,
            bg_path=bg_path,
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