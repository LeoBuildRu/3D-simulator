# renderer_utils.py
import os
import time
import math
import re
import glob
import json
import tempfile
import datetime
import random
from PIL import Image
from panda3d.core import PNMImage, Filename, PerspectiveLens

class RendererUtils:
    def __init__(self, panda_app):
        self.panda_app = panda_app
    
    def apply_barrel_distortion(self, img, k1, k2, crop_to_content=False):
        if k1 == 0.0 and k2 == 0.0:
            if crop_to_content:
                width = img.get_x_size()
                height = img.get_y_size()
                num_channels = img.get_num_channels()
                
                if num_channels == 1:
                    mode = 'L'
                    pil_img = Image.new(mode, (width, height))
                    pil_pixels = pil_img.load()
                    for y in range(height):
                        for x in range(width):
                            val = int(img.get_gray(x, y) * 255)
                            pil_pixels[x, y] = val
                elif num_channels >= 3:
                    mode = 'RGB' if num_channels == 3 else 'RGBA'
                    pil_img = Image.new(mode, (width, height))
                    pil_pixels = pil_img.load()
                    for y in range(height):
                        for x in range(width):
                            r = int(img.get_red(x, y) * 255)
                            g = int(img.get_green(x, y) * 255)
                            b = int(img.get_blue(x, y) * 255)
                            if num_channels == 3:
                                pil_pixels[x, y] = (r, g, b)
                            else: 
                                a = int(img.get_alpha(x, y) * 255) if img.has_alpha() else 255
                                pil_pixels[x, y] = (r, g, b, a)
                
                left, top, right, bottom = 270, 155, 1850, 925
                bbox = (left, top, right, bottom)
                cropped_pil_img = pil_img.crop(bbox)
                
                target_width = 1920 
                target_height = 1080 
                aspect_ratio = cropped_pil_img.width / cropped_pil_img.height
                target_ratio = target_width / target_height

                if aspect_ratio > target_ratio:
                    new_height = int(cropped_pil_img.width / target_ratio)
                    resized = cropped_pil_img.resize((target_width, new_height), Image.LANCZOS)
                    padding = (new_height - target_height) // 2
                    final_img = resized.crop((0, padding, target_width, padding + target_height))
                else:
                    new_width = int(cropped_pil_img.height * target_ratio)
                    resized = cropped_pil_img.resize((new_width, target_height), Image.LANCZOS)
                    padding = (new_width - target_width) // 2
                    final_img = resized.crop((padding, 0, padding + target_width, target_height))
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                    tmp_filename = tmpfile.name
                try:
                    final_img.save(tmp_filename, 'PNG')
                    temp_pnm = PNMImage()
                    if temp_pnm.read(Filename.from_os_specific(tmp_filename)):
                        os.unlink(tmp_filename)
                        return temp_pnm
                finally:
                    if os.path.exists(tmp_filename):
                        os.unlink(tmp_filename)       
            
            return img

        width = img.get_x_size()
        height = img.get_y_size()
        num_channels = img.get_num_channels()
        distorted_img = PNMImage(width, height, num_channels)
        cx = width / 2.0
        cy = height / 2.0
        max_r = math.sqrt((width/2)**2 + (height/2)**2)

        for y in range(height):
            for x in range(width):
                dx = x - cx
                dy = y - cy
                r = math.sqrt(dx*dx + dy*dy)
                if r > 0:
                    r_norm = r / max_r
                    r_distorted_norm = r_norm * (1 + k1 * r_norm**2 + k2 * r_norm**4)
                    r_distorted = r_distorted_norm * max_r
                    scale_factor = r_distorted / r
                else:
                    scale_factor = 1.0
                src_x = cx + dx * scale_factor
                src_y = cy + dy * scale_factor
                if 0 <= src_x < width-1 and 0 <= src_y < height-1:
                    x1 = int(src_x)
                    y1 = int(src_y)
                    x2 = x1 + 1
                    y2 = y1 + 1
                    dx_interp = src_x - x1
                    dy_interp = src_y - y1
                    c11 = [img.get_channel(x1, y1, c) for c in range(num_channels)]
                    c12 = [img.get_channel(x1, y2, c) for c in range(num_channels)]
                    c21 = [img.get_channel(x2, y1, c) for c in range(num_channels)]
                    c22 = [img.get_channel(x2, y2, c) for c in range(num_channels)]
                    c_top = [c11[c] * (1 - dx_interp) + c21[c] * dx_interp for c in range(num_channels)]
                    c_bottom = [c12[c] * (1 - dx_interp) + c22[c] * dx_interp for c in range(num_channels)]
                    c_final = [c_top[c] * (1 - dy_interp) + c_bottom[c] * dy_interp for c in range(num_channels)]
                    for c in range(num_channels):
                        distorted_img.set_channel(x, y, c, c_final[c])
                else:
                    src_x_clamped = max(0, min(width-1, src_x))
                    src_y_clamped = max(0, min(height-1, src_y))
                    x1 = int(src_x_clamped)
                    y1 = int(src_y_clamped)
                    x2 = min(width-1, x1 + 1)
                    y2 = min(height-1, y1 + 1)
                    dx_interp = src_x_clamped - x1
                    dy_interp = src_y_clamped - y1
                    c11 = [img.get_channel(x1, y1, c) for c in range(num_channels)]
                    c12 = [img.get_channel(x1, y2, c) for c in range(num_channels)]
                    c21 = [img.get_channel(x2, y1, c) for c in range(num_channels)]
                    c22 = [img.get_channel(x2, y2, c) for c in range(num_channels)]
                    c_top = [c11[c] * (1 - dx_interp) + c21[c] * dx_interp for c in range(num_channels)]
                    c_bottom = [c12[c] * (1 - dx_interp) + c22[c] * dx_interp for c in range(num_channels)]
                    c_final = [c_top[c] * (1 - dy_interp) + c_bottom[c] * dy_interp for c in range(num_channels)]
                    outside_dist = max(0, 
                                    abs(src_x - cx) - width/2 + 1, 
                                    abs(src_y - cy) - height/2 + 1)
                    fade_factor = max(0, min(1, 1 - (outside_dist / 15.0)**2))
                    for c in range(num_channels):
                        distorted_img.set_channel(x, y, c, c_final[c] * fade_factor)

        if crop_to_content:
            if num_channels == 1:
                mode = 'L'
                pil_img = Image.new(mode, (width, height))
                pil_pixels = pil_img.load()
                for y in range(height):
                    for x in range(width):
                        val = int(distorted_img.get_gray(x, y) * 255)
                        pil_pixels[x, y] = val
            elif num_channels >= 3:
                mode = 'RGB' if num_channels == 3 else 'RGBA'
                pil_img = Image.new(mode, (width, height))
                pil_pixels = pil_img.load()
                for y in range(height):
                    for x in range(width):
                        r = int(distorted_img.get_red(x, y) * 255)
                        g = int(distorted_img.get_green(x, y) * 255)
                        b = int(distorted_img.get_blue(x, y) * 255)
                        if num_channels == 3:
                            pil_pixels[x, y] = (r, g, b)
                        else: 
                            a = int(distorted_img.get_alpha(x, y) * 255) if distorted_img.has_alpha() else 255
                            pil_pixels[x, y] = (r, g, b, a)
            
            left, top, right, bottom = 270, 155, 1650, 925
            bbox = (left, top, right, bottom)
            cropped_pil_img = pil_img.crop(bbox)
            
            target_width = 1920 
            target_height = 1080
            resized_pil_img = cropped_pil_img.resize((target_width, target_height), Image.LANCZOS)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                tmp_filename = tmpfile.name
            try:
                resized_pil_img.save(tmp_filename, 'PNG')
                temp_pnm = PNMImage()
                if temp_pnm.read(Filename.from_os_specific(tmp_filename)):
                    os.unlink(tmp_filename)
                    return temp_pnm
            finally:
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                        
        return distorted_img
    
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

    def _process_render_image(self, img, camera_fov_x=None, camera_fov_y=None, output_dir="renders", 
                              filename_prefix="render", metadata=None):
        k1 = 0.15
        k2 = 0.35
        
        left = 270
        top = 155
        right = 1650
        bottom = 925
        crop_width = right - left
        crop_height = bottom - top
        
        target_width = 1920
        target_height = 1080
        
        distorted_img = self.apply_barrel_distortion(
            img, 
            k1=k1, 
            k2=k2, 
            crop_to_content=False
        )
        
        cropped_img = PNMImage(crop_width, crop_height, 3)
        for y in range(crop_height):
            for x in range(crop_width):
                src_x = left + x
                src_y = top + y
                if (0 <= src_x < distorted_img.getXSize() and 
                    0 <= src_y < distorted_img.getYSize()):
                    r, g, b = distorted_img.getXel(src_x, src_y)
                    cropped_img.setXel(x, y, r, g, b)
        
        pil_img = Image.new("RGB", (crop_width, crop_height))
        pixels = pil_img.load()
        for y in range(crop_height):
            for x in range(crop_width):
                r, g, b = cropped_img.getXel(x, y)
                pixels[x, y] = (int(r * 255), int(g * 255), int(b * 255))
        
        resized_pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
        
        final_img = PNMImage(target_width, target_height, 3)
        for y in range(target_height):
            for x in range(target_width):
                r, g, b = resized_pil_img.getpixel((x, y))
                final_img.setXel(x, y, r/255.0, g/255.0, b/255.0)
        
        orig_width = img.getXSize()
        orig_height = img.getYSize()
        scale_x = target_width / crop_width
        scale_y = target_height / crop_height
        
        fx = fy = cx = cy = None
        if camera_fov_x is not None and camera_fov_y is not None:
            fx = (target_width / 2.0) / math.tan(math.radians(camera_fov_x / 2.0)) * scale_x
            fy = (target_height / 2.0) / math.tan(math.radians(camera_fov_y / 2.0)) * scale_y
            cx = target_width / 2.0
            cy = target_height / 2.0
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        final_img.write(Filename.from_os_specific(output_path))
        
        if metadata is None:
            metadata = {}
        
        render_metadata = {
            "image_path": output_path,
            "distortion_params": {
                "k1": k1,
                "k2": k2
            },
            "crop_params": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "crop_width": crop_width,
                "crop_height": crop_height
            },
            "scale_params": {
                "target_width": target_width,
                "target_height": target_height,
                "scale_x": scale_x,
                "scale_y": scale_y
            },
            **metadata
        }
        
        if camera_fov_x is not None and camera_fov_y is not None:
            render_metadata["camera_params"] = {
                "fov_x": camera_fov_x,
                "fov_y": camera_fov_y,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
        
        json_path = output_path.replace(".png", ".json")
        with open(json_path, 'w') as f:
            json.dump(render_metadata, f, indent=2)
        
        return output_path
    
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