import sys
import os
import traceback

# Если приложение скомпилировано, базовый путь — каталог с exe,
# иначе — текущая директория (для разработки)
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

# Вставляем базовый путь в начало sys.path, чтобы локальные копии модулей имели приоритет
sys.path.insert(0, base_path)

# --- Добавлено: настройка путей поиска моделей для Panda3D ---
from panda3d.core import get_model_path, Filename

# Преобразуем пути в формат Panda3D (Unix-стиль)
base_path_p3d = Filename.fromOsSpecific(base_path).getFullpath()
get_model_path().prepend_directory(base_path_p3d)

models_path_p3d = Filename.fromOsSpecific(os.path.join(base_path, "assets", "models")).getFullpath()
get_model_path().prepend_directory(models_path_p3d)
# -------------------------------------------------------------

def load_tls_config(base_path):
    """Загружает активный TLS-сервер из tls_config.yaml"""
    config_path = os.path.join(base_path, "config", "tls_config.yaml")
    default_host = "78.25.191.12"
    default_port = 9998

    if not os.path.exists(config_path):
        print(f"TLS config file not found: {config_path}. Using default.")
        return default_host, default_port

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        servers = config.get("servers", [])
        for server in servers:
            if server.get("active", False):
                host = server.get("host", default_host)
                port = server.get("port", default_port)
                print(f"Using active TLS server: {host}:{port} ({server.get('name', 'unknown')})")
                return host, port
        print("No active server found in config. Using default.")
        return default_host, default_port
    except Exception as e:
        print(f"Error reading TLS config: {e}. Using default.")
        return default_host, default_port

# Теперь можно импортировать остальные модули
# (gui import removed - new MainWindow lives in main_window.py)
from src.ui.panda_widget import Panda3DWidget
from src.rendering.depth_map_renderer import DepthMapRenderer
from src.rendering.perlin_mesh_generator import PerlinMeshGenerator
from src.rendering.renderer_utils import RendererUtils
from src.rendering.mesh_reconstruction import MeshReconstruction
from src.rendering.mesh_distribution import MeshDistributor
from src.core.crash_reporter import TelegramCrashReporter
from src.core.TLS_client import TLS_client
from src.core import graphics_settings

# Particles depend on NVIDIA Warp (CUDA). On machines without it (e.g. the
# Intel-iGPU target of the "performance" preset) the import must not abort
# startup, so it is optional — particle creation is also gated on the preset.
try:
    from src.particles.falling_particles import WarpFallingParticles
except Exception as _particles_exc:  # noqa: BLE001
    WarpFallingParticles = None
    print(f"[main] particles unavailable (warp/CUDA missing?): {_particles_exc}")

BOT_TOKEN = "8773064116:AAEiJdyHYysLpSnAx-gbDHG0DMbvV92IpsA"
CHAT_ID = "-5295757150"

import sys
import os
import math
import time
import json
import random
import datetime
import tempfile
import glob
import re
import yaml
import subprocess
from pathlib import Path

import win32gui
import win32con

import trimesh
import numpy as np
from scipy.spatial import cKDTree

from PIL import Image, ImageDraw, ImageFilter

import tkinter as tk
from tkinter import filedialog
# (PyQt5 imports removed - we use PyQt6 in main())
# (PyQt5 imports removed - we use PyQt6 in main())
# (PyQt5 imports removed - we use PyQt6 in main())

from panda3d.core import *

from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import *
from direct.task import Task
from direct.showbase.DirectObject import DirectObject

from noise import pnoise2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RP_PATH = os.path.join(PROJECT_ROOT, "render_pipeline")
if RP_PATH not in sys.path:
    sys.path.insert(0, RP_PATH)

from rpcore import PointLight
from rpcore import RenderPipeline
from rpcore.util.movement_controller import MovementController

NOISE_AVAILABLE = True
USE_SCIPY = True

class MyApp(ShowBase):
    def __init__(self, parent_hwnd: int,
                 init_size: tuple = (1920, 1080),
                 tls_host: str = "78.25.191.12",
                 tls_port: int = 9998,
                 graphics_preset: str = "ultra"):
        # ------------------------------------------------------------------
        # Graphics preset decides the rendering engine. This MUST happen
        # before ShowBase exists, because RenderPipeline has to be built
        # before the window. See src/core/graphics_settings.py.
        #   ultra / medium -> tobspr RenderPipeline (deferred, heavy)
        #   performance    -> panda3d-simplepbr (forward PBR + sun shadows)
        # ------------------------------------------------------------------
        self.graphics_preset = graphics_preset
        preset = graphics_settings.get_preset(graphics_preset)
        self.use_render_pipeline = preset["engine"] == "render_pipeline"
        self.pbr_pipeline = None
        print(f"[MyApp] graphics preset = '{graphics_preset}' "
              f"(engine={preset['engine']})")

        w, h = init_size

        if self.use_render_pipeline:
            self.render_pipeline = RenderPipeline()
            # MEDIUM points RP at a trimmed config dir (fewer plugins /
            # smaller shadows) — must be set before the VFS is mounted.
            rp_config_dir = preset.get("rp_config_dir")
            if rp_config_dir and os.path.isdir(rp_config_dir):
                self.render_pipeline.mount_mgr.config_dir = rp_config_dir
                print(f"[MyApp] RenderPipeline config dir -> {rp_config_dir}")
            self.render_pipeline.pre_showbase_init()

            loadPrcFileData("", f"win-size {w} {h}")
            loadPrcFileData("", "window-type onscreen")
            loadPrcFileData("", f"parent-window-handle {int(parent_hwnd)}")
            loadPrcFileData("", "fullscreen false")
            loadPrcFileData("", "undecorated true")

            ShowBase.__init__(self)

            self.render_pipeline.create(self)
        else:
            # PERFORMANCE: plain ShowBase with our own window flags (we do
            # NOT load RenderPipeline's panda3d-config.prc, which forces GL
            # 4.3 + compute shaders), then a lightweight simplepbr renderer.
            self.render_pipeline = None
            msaa = int(preset.get("msaa", 0))
            loadPrcFileData("", f"win-size {w} {h}")
            loadPrcFileData("", "window-type onscreen")
            loadPrcFileData("", f"parent-window-handle {int(parent_hwnd)}")
            loadPrcFileData("", "fullscreen false")
            loadPrcFileData("", "undecorated true")
            loadPrcFileData("", "sync-video #f")
            loadPrcFileData("", "show-frame-rate-meter #f")
            loadPrcFileData("", "textures-power-2 none")
            loadPrcFileData("", f"multisamples {msaa}")
            if msaa > 0:
                loadPrcFileData("", "framebuffer-multisample #t")

            ShowBase.__init__(self)

            self._init_lightweight_renderer(preset)

        self.tls_client = TLS_client(host=tls_host, port=tls_port, timeout=300.0)

        self.current_texture_set = {
            'diffuse': "assets/textures/stones_8k/rocks_ground_01_diff_8k.jpg",
            'displacement': "assets/textures/stones_8k/rocks_ground_01_disp_8k.jpg",
            'normal': "assets/textures/stones_8k/rocks_ground_01_nor_dx_8k.jpg",
            'roughness': "assets/textures/stones_8k/rocks_ground_01_rough_8k.jpg",
            'textureRepeatX': 1.35,
            'textureRepeatY': 3.2,
            'strength': 0.14,
            'textureRepeatU': 160.0,
            'textureRepeatV': 160.0
        }

        self.gui = None
        
        self.last_target_model_trimesh = None
        self.last_best_z = None
        self.test_perlin_mesh = None  
        self.last_grid_size = 48  

        self.loaded_models = []
        self.model_paths = {}

        self.setup_scene()
        # ---- Static base scene model -------------------------------------
        # Loads base_without_ground.bam from <project>/models on every
        # launch as the static environment of the scene.
        try:
            base_path = os.path.join(PROJECT_ROOT, "assets", "models",
                                     "base_without_ground.bam")
            if os.path.exists(base_path):
                p3d_path = Filename.fromOsSpecific(base_path).getFullpath()
                base_np = self.loader.load_model(p3d_path, noCache=True)
                if base_np is not None:
                    base_np.reparent_to(self.render)
                    base_np.set_pos(0, 0, 0)
                    self._apply_auto_shader(base_np)
                    # PERFORMANCE: this .bam ships RP-style M_modulate stages
                    # and no tangents — remap stages + build tangents so its
                    # normal/roughness maps render under simplepbr.
                    if not self.use_render_pipeline:
                        self._make_pbr_compatible(base_np)
                    self.base_static_model = base_np
                    print(f"[Scene] base_without_ground.bam loaded from "
                          f"{base_path}")
                else:
                    print(f"[Scene] loader returned None for {base_path}")
            else:
                print(f"[Scene] base_without_ground.bam not found at "
                      f"{base_path}")
        except Exception as exc:
            print(f"[Scene] base_without_ground.bam load failed: {exc}")
            self.base_static_model = None

        # create_top_overlay() removed - the 'Время проезда' DirectFrame
        # is no longer part of the new UI. self.top_overlay stays None.
        self.top_overlay = None

        self.next_model_x = 0

        self.current_model_set = None

        # временный флаг:
        self.particle_flag = False
        self.canDistributeMeshes = False

        self.Target_Cuzov = "Scania-Cuzov.gltf"
        self.Target_Y_offset = 0
        self.Target_Volume = 20
        self.Target_Napolnitel = "Scania-Napolnitel.gltf"
        self.Target_height_val = 66

        self.current_view = "perspective"

        self.mouse_rotation_enabled = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.camera_rotation_speed = 0.5

        self.drag_drop_mode = False
        self.selected_model = None
        self.drag_start_pos = None
        self.drag_start_mouse_pos = None
        self.drag_sensitivity = 2.0

        self.disable_mouse()

        self.key_map = {
            "forward": False, 
            "backward": False,
            "left": False,    
            "right": False,   
            "up": False,      
            "down": False     
        }
        
        self.accept("w", self.set_key, ["forward", True])
        self.accept("w-up", self.set_key, ["forward", False])
        self.accept("s", self.set_key, ["backward", True])
        self.accept("s-up", self.set_key, ["backward", False])
        self.accept("a", self.set_key, ["left", True])
        self.accept("a-up", self.set_key, ["left", False])
        self.accept("d", self.set_key, ["right", True])
        self.accept("d-up", self.set_key, ["right", False])
        self.accept("space", self.set_key, ["up", True])
        self.accept("space-up", self.set_key, ["up", False])
        self.accept("shift", self.set_key, ["down", True])
        self.accept("shift-up", self.set_key, ["down", False])

        self.base_perlin_model = None
        self.dynamic_perlin_model = None

        self.height_values = []

        self.accept("mouse1", self.handle_mouse_left)
        self.accept("mouse1-up", self.handle_mouse_left_up)
        self.accept("mouse3", self.handle_mouse_right)

        self.accept("wheel_up", self.zoom_camera, [1.1])
        self.accept("wheel_down", self.zoom_camera, [0.9])

        self.zoom_sensitivity = 1.1

        self.current_z = 0

        self.final_model = None
        self.final_mesh_node = None

        self.ground_plane = None
        self.plane_size_x = 100.0
        self.plane_size_y = 100.0
        
        self.taskMgr.add(self.move_camera_task, "move_camera_task")
        self.taskMgr.add(self.mouse_rotation_task, "mouse_rotation_task")
        self.taskMgr.add(self.drag_drop_task, "drag_drop_task")
        
        self.depth_renderer = None
        self.init_depth_renderer()

        self.mesh_distributions = []

        self.tls_client = TLS_client(host=tls_host, port=tls_port, timeout=300.0)

        self.perlin_generator = PerlinMeshGenerator(self, tls_client=self.tls_client)
        self.renderer_utils = RendererUtils(self)

        self.mesh_reconstruction = MeshReconstruction(self, tls_client=self.tls_client)

        # ------------------------------------------------------------------
        # Camera bindings: override the legacy SPACE/SHIFT for up/down and
        # mouse1-as-rotation behaviour with the editor-style FlyCamera
        # (WASD, Q/E up/down, Shift sprint, RMB-look).
        # ------------------------------------------------------------------
        try:
            for ev in ("space", "space-up", "shift", "shift-up",
                       "mouse1", "mouse1-up"):
                self.ignore(ev)
            from src.core.camera_controller import FlyCamera
            self.fly_cam = FlyCamera(self)
        except Exception as exc:
            print(f"[MyApp] FlyCamera init failed: {exc}")

        # try:
        #     print("send request to server")
        #     # Получаем текущие настройки текстурных повторов (если есть)
        #     tex_rep_x = 1.35
        #     tex_rep_y = 3.2
        #     if hasattr(self, 'panda_app') and hasattr(self.panda_app, 'current_texture_set'):
        #         tex_set = self.panda_app.current_texture_set
        #         tex_rep_x = tex_set.get('textureRepeatX', tex_rep_x)
        #         tex_rep_y = tex_set.get('textureRepeatY', tex_rep_y)
        #     vertices, triangles, normals, uvs = self.tls_client.generate_landscape(
        #         # Основные геометрические параметры
        #         size=20.0,
        #         subdivisions=128,
        #         height=0.77,
        #         seed=120,
        #         noise_scale=1.36,
        #         name="Landscape",
        #         subdivisions_x=128,
        #         subdivisions_y=128,
        #         mesh_size_x=3.06,
        #         mesh_size_y=6.46,
        #         noise_type="rocks_noise",
        #         noise_basis="BLENDER",
        #         offset_x=-1.05,
        #         offset_y=0.00,
        #         size_x=1.45,
        #         size_y=2.23,
        #         depth=8,
        #         distortion=1.39,
        #         hard_noise="0",
        #         height_offset=0.00,
        #         maximum=10000.0,
        #         minimum=-10000.0,
        #         edge_falloff="3",
        #         edge_level=-0.12,
        #         falloff_x=3.70,
        #         falloff_y=4.00,
        #         strata_type="0",
        #         output_format="ply",
        #         texture_repeat_x=tex_rep_x,
        #         texture_repeat_y=tex_rep_y
        #     )
        #     print("create mesh from data")
        #     mesh = self.create_mesh_from_data(vertices, triangles, normals, uvs)
        #     print("reparent to render")
        #     mesh.reparentTo(self.render)
        # except Exception as e:
        #     import traceback
        #     print("ERROR during landscape generation:")
        #     traceback.print_exc()

        # Particles use the RenderPipeline instancing effect (+ NVIDIA Warp),
        # so they only run under the RP-based presets and when Warp imported.
        if self.use_render_pipeline and WarpFallingParticles is not None:
            try:
                self.particles = WarpFallingParticles(
                    showbase=self,
                    render_pipeline=self.render_pipeline,
                    texture="assets/textures/leaf.png",
                    particle_count=1000,
                    spawn_min=(-30.0, -30.0, 14.0),
                    spawn_max=(30.0, 30.0, 15.0),
                    respawn_threshold=-0.1,
                    rotation_mode=WarpFallingParticles.RANDOM_ROTATION,
                    size_range=(0.05, 0.20),
                    speed_range=(1.5, 3.0),
                    parent=self.render,
                    alpha_blend=False,
                    auto_start=True
                )
            except Exception as e:
                traceback.print_exc()

    # ==================================================================
    # Lightweight renderer (PERFORMANCE preset) — no RenderPipeline
    # ==================================================================
    def _init_lightweight_renderer(self, preset):
        """
        Set up the "performance" preset rendering: panda3d-simplepbr
        (forward PBR with per-pixel lighting, normal maps and a
        shadow-casting directional sun) instead of the heavy deferred
        RenderPipeline. Targets Intel iGPUs (20-30 fps @ 1080p).

        If simplepbr cannot be imported/initialised we fall back to
        Panda's built-in auto-shader so the scene is still lit.
        """
        from panda3d.core import (AmbientLight, DirectionalLight, Fog,
                                  Vec4)

        # Clip planes sized for the large (2000u) ground; natural FOV.
        self.camLens.set_near_far(0.1, 10000.0)
        self.camLens.set_fov(70)

        shadow_res = int(preset.get("shadow_resolution", 2048))
        try:
            import simplepbr

            # Image-based lighting (IBL). This is what actually makes the
            # scene look "PBR": without an environment map simplepbr has no
            # ambient specular and metals render black. We build it from the
            # sky cubemap bundled with RenderPipeline (6 faces). It provides
            # diffuse (SH) + specular (prefiltered) lighting to ALL surfaces
            # — trucks, ground, everything.
            #
            # `ibl_specular_scale` dims the cubemap before prefiltering, which
            # controls the brightness of the REFLECTIONS (the bundled sky is
            # very bright and otherwise blows out metals). The diffuse term is
            # set separately below via the SH coefficients.
            self.ibl_specular_scale = 0.35
            self.ibl_diffuse_scale = 0.25   # absolute, vs the original sky
            env_map = None
            try:
                from panda3d.core import TexturePool
                cube_pattern = Filename.from_os_specific(
                    os.path.join(PROJECT_ROOT, "render_pipeline", "data",
                                 "default_cubemap", "source", "#.png")
                ).get_fullpath()
                cube = TexturePool.load_cube_map(cube_pattern)
                ram = cube.get_ram_image()
                if ram is not None:
                    a = np.frombuffer(bytes(ram), dtype=np.uint8).astype(np.float32)
                    a = np.clip(a * self.ibl_specular_scale, 0, 255).astype(np.uint8)
                    cube.set_ram_image(a.tobytes())
                env_map = simplepbr.EnvMap(cube, blocking_prepare=True)
                print(f"[MyApp] IBL environment map loaded "
                      f"(reflections x{self.ibl_specular_scale})")
            except Exception as exc:
                env_map = None
                print(f"[MyApp] IBL env map load failed ({exc}); "
                      f"metals/reflections will look flat")

            self.pbr_pipeline = simplepbr.init(
                use_normal_maps=True,
                enable_shadows=True,
                enable_fog=bool(preset.get("enable_fog", True)),
                max_lights=int(preset.get("max_lights", 4)),
                msaa_samples=int(preset.get("msaa", 0)),
                use_emission_maps=False,
                # Occlusion OFF: our ground roughness map is a plain grayscale
                # map bound as the metal-roughness (ORM) texture; its R channel
                # is NOT occlusion, so we must not let simplepbr read it as AO.
                use_occlusion_maps=False,
                env_map=env_map,
                # Bias is subtracted in normalized [0,1] shadow-map depth, so
                # its world-space size = bias * (far - near) of the sun lens.
                # With our tight sun frustum (~120u range) this is ~0.16u —
                # enough to avoid acne without the huge Peter-panning offset
                # the default 0.005 caused with a wide frustum.
                shadow_bias=0.0013,
            )
            print("[MyApp] simplepbr renderer initialised")

            # The bundled cubemap integrates to a very bright ambient
            # (SH[0] ~5 → irradiance > 1), which the shader adds to EVERY
            # fragment regardless of shadow — flooding shadows and drowning
            # the sun so normal maps stop reading. Set the DIFFUSE IBL to an
            # absolute, modest level via the SH coefficients. (The cube was
            # already dimmed by ibl_specular_scale, so compensate for that to
            # keep diffuse independent of the reflection brightness.)
            if env_map is not None:
                from panda3d.core import PTA_LVecBase3f, LVecBase3f
                factor = self.ibl_diffuse_scale / max(self.ibl_specular_scale,
                                                      1e-3)
                _scaled = PTA_LVecBase3f()
                for _v in env_map.sh_coefficients:
                    _scaled.push_back(LVecBase3f(
                        _v[0] * factor, _v[1] * factor, _v[2] * factor,
                    ))
                self.render.set_shader_input("sh_coeffs", _scaled)
                print(f"[MyApp] IBL diffuse x{self.ibl_diffuse_scale}, "
                      f"reflections x{self.ibl_specular_scale}")
        except Exception as exc:
            self.pbr_pipeline = None
            print(f"[MyApp] simplepbr init failed ({exc}); "
                  f"falling back to auto-shader")
            self.render.set_shader_auto()

        # Sky colour + very light distance fog (subtle aerial depth only —
        # the ground is ~2000u across, so density must stay low or the
        # scene washes out). set_time_of_day keeps the fog colour in sync
        # with the sky.
        self._perf_fog = None
        self.set_background_color(0.52, 0.62, 0.74, 1.0)
        if preset.get("enable_fog", True):
            fog = Fog("perf_distance_fog")
            fog.set_color(0.60, 0.68, 0.78)
            fog.set_exp_density(0.00035)
            self.render.set_fog(fog)
            self._perf_fog = fog

        # Sun: directional, shadow-casting. DirectionalLight's default lens
        # is orthographic, so set_film_size is in world units — sized to
        # frame the area where models sit (around the origin).
        # Tight orthographic shadow frustum centred on the scene origin (where
        # models/cargo load). Film covers ±40u; the depth range is kept small
        # (sun at a fixed distance, range ~120u) so the normalized-depth bias
        # stays tiny in world units — otherwise shadows Peter-pan badly.
        self._sun_distance = 120.0
        sun = DirectionalLight("perf_sun")
        sun.set_color(Vec4(1.0, 0.96, 0.88, 1.0))
        sun.set_shadow_caster(True, shadow_res, shadow_res)
        sun_lens = sun.get_lens()
        sun_lens.set_film_size(80, 80)
        sun_lens.set_near_far(self._sun_distance - 60.0,
                              self._sun_distance + 60.0)
        self.sun = sun
        self.sun_np = self.render.attach_new_node(sun)
        self.render.set_light(self.sun_np)

        # Low cool ambient fill. The IBL env map already supplies most of the
        # ambient/sky light, so this stays small to keep shadows contrasty.
        # set_time_of_day adjusts it with the sun.
        amb = AmbientLight("perf_ambient")
        amb.set_color(Vec4(0.06, 0.07, 0.09, 1.0))
        self.ambient_light = amb
        self.ambient_np = self.render.attach_new_node(amb)
        self.render.set_light(self.ambient_np)

        # Position the sun to match the daytime slider's default (15:00 — a
        # ~45° afternoon sun: bright, but raking enough to reveal normal-map
        # bumps and cast longer, clearly visible shadows).
        try:
            self.set_time_of_day(15 * 60)
        except Exception as exc:
            print(f"[MyApp] initial set_time_of_day failed: {exc}")

    def _apply_auto_shader(self, np):
        """
        Apply the right shader for the active engine to a scene node.

        * RenderPipeline presets: keep the legacy behaviour (set_shader_auto;
          RP overrides at the render root anyway).
        * simplepbr preset: do nothing — the node inherits simplepbr's
          shader from `render`. Only if simplepbr failed to initialise do
          we fall back to Panda's auto-shader.
        """
        if self.use_render_pipeline or not getattr(self, "pbr_pipeline", None):
            np.set_shader_auto()

    def set_time_of_day(self, minutes: int):
        """
        Set the time of day from minutes-since-midnight (0..1439).

        * RenderPipeline presets: drive RP's DayTime manager.
        * performance preset: rotate/recolour the directional sun and tweak
          ambient so the scene goes warm at dawn/dusk and dim at night.
        """
        minutes = max(0, min(1439, int(minutes)))
        hh, mm = minutes // 60, minutes % 60

        if self.use_render_pipeline:
            rp = getattr(self, "render_pipeline", None)
            dt_mgr = getattr(rp, "daytime_mgr", None) if rp else None
            if dt_mgr is not None:
                dt_mgr.time = f"{hh:02d}:{mm:02d}"
            return

        sun_np = getattr(self, "sun_np", None)
        if sun_np is None:
            return
        from panda3d.core import Vec4

        # Sun elevation: below horizon before 06:00 / after 18:00.
        day_frac = minutes / 1440.0
        elevation = math.sin((minutes - 360) / 720.0 * math.pi)  # -1..1
        # Azimuth sweeps east->west across the day.
        azimuth = -120.0 + 240.0 * day_frac

        # Place the sun at a FIXED distance along its direction so it stays
        # centred in the tight orthographic shadow frustum (set in
        # _init_lightweight_renderer). Only the direction changes with time.
        from panda3d.core import Vec3
        d = getattr(self, "_sun_distance", 120.0)
        direction = Vec3(80.0 * math.sin(math.radians(azimuth)),
                         -80.0 * math.cos(math.radians(azimuth)),
                         max(20.0, 120.0 * max(0.1, elevation)))
        direction.normalize()
        sun_np.set_pos(direction * d)
        sun_np.look_at(0, 0, 0)

        # Colour: warm and dim near the horizon, bright white at noon.
        warmth = max(0.0, min(1.0, elevation))
        if elevation <= 0.0:                       # night
            self.sun.set_color(Vec4(0.05, 0.06, 0.10, 1.0))
            self.ambient_light.set_color(Vec4(0.10, 0.12, 0.18, 1.0))
            bg = (0.06, 0.07, 0.11)
        else:
            g = 0.80 + 0.18 * warmth
            b = 0.62 + 0.30 * warmth
            # Strong directional sun + low ambient floor → high light/shadow
            # contrast. In simplepbr a shadowed fragment only keeps the
            # ambient term, so a low ambient is what makes shadows read dark.
            intensity = 1.8 + 1.0 * warmth
            self.sun.set_color(Vec4(intensity, g * intensity,
                                    b * intensity, 1.0))
            # Small explicit ambient on top of the IBL env-map ambient.
            self.ambient_light.set_color(Vec4(0.04 + 0.03 * warmth,
                                              0.05 + 0.03 * warmth,
                                              0.07 + 0.035 * warmth, 1.0))
            bg = (0.34 + 0.20 * warmth,
                  0.45 + 0.20 * warmth,
                  0.58 + 0.18 * warmth)

        self.set_background_color(bg[0], bg[1], bg[2], 1.0)
        # Keep the distance fog the same colour as the sky so it reads as
        # aerial haze, not a grey veil (and stays dark at night).
        if getattr(self, "_perf_fog", None) is not None:
            self._perf_fog.set_color(bg[0], bg[1], bg[2])

    def _flat_normal_texture(self):
        """A cached 1x1 flat (0,0,1) tangent-space normal map. Bound on meshes
        that have tangents but no real normal map, so simplepbr always samples
        a well-defined normal texture (Panda's default differs on Intel)."""
        from panda3d.core import Texture
        tex = getattr(self, "_flat_normal_tex", None)
        if tex is None:
            tex = Texture("flat_normal")
            tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
            # Panda RAM image is BGR: (B=255, G=128, R=128) -> RGB(128,128,255).
            tex.set_ram_image(bytes([255, 128, 128]))
            tex.set_minfilter(Texture.FTNearest)
            tex.set_magfilter(Texture.FTNearest)
            self._flat_normal_tex = tex
        return tex

    def _apply_pbr_surface(self, model_np, diffuse_path, normal_path=None,
                           roughness_path=None, roughness=1.0,
                           srgb_diffuse=True, has_tangents=False,
                           base_color=(1, 1, 1, 1)):
        """
        simplepbr-friendly PBR material for hand-built meshes (PERFORMANCE).

        Texture-stage modes follow Panda's metalness convention that simplepbr
        reads:
          * base colour  -> default (modulate) stage  -> p3d_TextureBaseColor
          * roughness    -> M_selector stage (ORM)     -> p3d_TextureMetalRoughness
                            (G = roughness; B = metallic, forced 0 via the
                            Material; occlusion is disabled in simplepbr.init so
                            the R channel of a grayscale roughness map is ignored)
          * normal       -> M_normal stage             -> p3d_TextureNormal
                            (only bound when the geometry has tangents — flat
                            meshes get a constant tangent in their vertex data)

        We deliberately avoid RenderPipeline's `emission=(0,1,0,0)` convention
        (simplepbr would render it as a green glow). glTF models are untouched
        here and keep their full PBR maps via simplepbr.
        """
        from panda3d.core import Texture, TextureStage, Material

        mat = Material()
        mat.set_base_color(base_color)
        mat.set_roughness(float(roughness))
        mat.set_metallic(0.0)
        model_np.set_material(mat, 1)

        def _pf(p):
            return Filename.fromOsSpecific(str(p))

        def _setup(tex, srgb=False):
            if srgb:
                tex.set_format(Texture.F_srgb)
            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)

        # Base colour (sRGB) on the default modulate stage.
        if diffuse_path and os.path.exists(diffuse_path):
            tex = self.loader.loadTexture(_pf(diffuse_path))
            if tex:
                _setup(tex, srgb=srgb_diffuse)
                model_np.set_texture(tex, 1)

        # Roughness via the metal-roughness (ORM) selector stage. Linear data.
        if roughness_path and os.path.exists(roughness_path):
            rtex = self.loader.loadTexture(_pf(roughness_path))
            if rtex:
                _setup(rtex)
                ts_mr = TextureStage("mr")
                ts_mr.set_mode(TextureStage.M_selector)
                model_np.set_texture(ts_mr, rtex, 1)

        # Normal map. simplepbr's shader ALWAYS samples p3d_TextureNormal
        # (use_normal_maps is global). A mesh with tangents but no bound normal
        # map falls back to Panda's DEFAULT normal texture, which Intel drivers
        # mishandle → a bad normal → N·L<=0 → the surface gets ambient only
        # (no sun / no shadows), while NVIDIA renders it fine. So whenever the
        # mesh has tangents we bind an EXPLICIT normal map — the real one if
        # available, otherwise a flat (0,0,1) so the geometric normal is used.
        if has_tangents:
            ts_n = TextureStage("nrm")
            ts_n.set_mode(TextureStage.M_normal)
            ntex = None
            if normal_path and os.path.exists(normal_path):
                ntex = self.loader.loadTexture(_pf(normal_path))
                if ntex:
                    _setup(ntex)
            if ntex is None:
                ntex = self._flat_normal_texture()
            model_np.set_texture(ts_n, ntex, 1)

    def _make_pbr_compatible(self, model_np):
        """
        PERFORMANCE preset: make a loaded .bam model render as PBR under
        simplepbr.

        The project's .bam models were authored for RenderPipeline, which
        puts ALL maps (colour / normal / roughness / specular) in M_modulate
        texture stages distinguished only by name/sort. simplepbr instead
        selects PBR maps by stage MODE, so without this fix it only ever sees
        the base colour — no normals, no roughness. We:
          1. remap stages by name (Normal -> M_normal, Roughness -> M_selector,
             Specular/IOR/Metal -> M_gloss which simplepbr ignores);
          2. generate per-vertex tangents (the .bam meshes have none, and
             normal mapping is impossible without them);
          3. force metallic = 0 (a grayscale roughness map reused as the ORM
             selector would otherwise drive metallic via its B channel).
        """
        if self.use_render_pipeline:
            return
        from panda3d.core import TextureStage

        try:
            for ts in model_np.find_all_texture_stages():
                nm = ts.get_name().lower()
                if "normal" in nm:
                    ts.set_mode(TextureStage.M_normal)
                elif "rough" in nm:
                    ts.set_mode(TextureStage.M_selector)   # ORM: G = roughness
                elif "specular" in nm or "_ior" in nm or "metal" in nm:
                    # simplepbr is metal-roughness, not specular/IOR — park this
                    # map on a stage simplepbr ignores so it can't tint base.
                    ts.set_mode(TextureStage.M_gloss)
        except Exception as exc:
            print(f"[MyApp] stage remap failed: {exc}")

        try:
            for mat in model_np.find_all_materials():
                mat.set_metallic(0.0)
                if not mat.has_roughness():
                    mat.set_roughness(0.6)
        except Exception as exc:
            print(f"[MyApp] material fix failed: {exc}")

        # Generate tangents so the normal map can shade (visible up close;
        # at distance it mip-flattens — simplepbr has no parallax/displacement).
        try:
            geomnodes = model_np.find_all_matches("**/+GeomNode")
            for i in range(geomnodes.get_num_paths()):
                gnode = geomnodes.get_path(i).node()
                for g in range(gnode.get_num_geoms()):
                    self._generate_tangents_for_geom(gnode.modify_geom(g))
        except Exception as exc:
            print(f"[MyApp] tangent generation failed: {exc}")

    def _generate_tangents_for_geom(self, geom):
        """
        Add a per-vertex tangent column to `geom` (in place), computed from
        positions/normals/UVs. Adapted from panda3d-gltf's calculate_tangents
        (https://www.marti.works/calculating-tangents-for-your-mesh/). No-op
        if the geom has no texcoords.
        """
        from panda3d.core import (GeomVertexReader, GeomVertexWriter,
                                  InternalName, GeomEnums, LVector2, LVector3)

        gvd = geom.get_vertex_data()
        if not gvd.has_column(InternalName.get_texcoord()):
            return
        if geom.get_num_primitives() == 0:
            return

        prim = geom.get_primitive(0).decompose()
        gvd = gvd.replace_column(
            InternalName.get_tangent(), 4, GeomEnums.NT_float32,
            GeomEnums.C_other)

        def _read3(name):
            r = GeomVertexReader(gvd, name)
            out = []
            while not r.is_at_end():
                d = r.get_data3f()
                out.append(LVector3(d[0], d[1], d[2]))
            return out

        posdata = _read3(InternalName.get_vertex())
        normaldata = _read3(InternalName.get_normal())
        uvr = GeomVertexReader(gvd, InternalName.get_texcoord())
        uvdata = []
        while not uvr.is_at_end():
            d = uvr.get_data2f()
            uvdata.append(LVector2(d[0], d[1]))

        n = len(posdata)
        if n == 0 or len(uvdata) < n or len(normaldata) < n:
            return
        tana = [LVector3(0, 0, 0) for _ in range(n)]
        tanb = [LVector3(0, 0, 0) for _ in range(n)]

        verts = prim.get_vertex_list()
        for t in range(0, len(verts) - 2, 3):
            i0, i1, i2 = verts[t], verts[t + 1], verts[t + 2]
            e1 = posdata[i1] - posdata[i0]
            e2 = posdata[i2] - posdata[i0]
            d1 = uvdata[i1] - uvdata[i0]
            d2 = uvdata[i2] - uvdata[i0]
            denom = d1.x * d2.y - d2.x * d1.y
            if denom != 0.0:
                f = 1.0 / denom
                tan = (e1 * d2.y - e2 * d1.y) * f
                bit = (e2 * d1.x - e1 * d2.x) * f
            else:
                tan = LVector3(0, 0, 0)
                bit = LVector3(0, 0, 0)
            for idx in (i0, i1, i2):
                tana[idx] += tan
                tanb[idx] += bit

        tw = GeomVertexWriter(gvd, InternalName.get_tangent())
        for normal, t0, t1 in zip(normaldata, tana, tanb):
            tangent = t0 - normal * normal.dot(t0)
            if tangent.length_squared() > 1e-12:
                tangent.normalize()
            else:
                tangent = LVector3(1, 0, 0)
            w = -1.0 if normal.cross(t0).dot(t1) < 0 else 1.0
            tw.set_data4f(tangent.x, tangent.y, tangent.z, w)

        geom.set_vertex_data(gvd)

    def relight_generated_mesh(self, model_np, diffuse_path=None,
                               roughness_path=None, base_color=(1, 1, 1, 1)):
        """
        PERFORMANCE preset: make a runtime-generated fill mesh (depth-map,
        CSG, procedural, .ply reconstruction) render as lit PBR under simplepbr.

        Two things are needed:
          1. Consistent normals — these meshes have inverted/inconsistent ones
             (RP tolerates it; simplepbr would shade them black). We recompute
             smooth, centroid-outward vertex normals.
          2. TANGENTS — simplepbr's shader always does
             `normalize(NormalMatrix * p3d_Tangent.xyz)`. With no tangent
             attribute that's `normalize(0)`, which is NaN on Intel GPUs
             (NVIDIA tolerates it) → NaN normal → BLACK. This is exactly why
             the fill meshes render fine on RTX but black on the Intel iGPU,
             while the ground/trucks (which carry tangents) are fine. We
             generate tangents so the TBN is valid everywhere.
        Then a diffuse PBR material is applied. No-op under RenderPipeline.
        """
        if self.use_render_pipeline:
            return
        from panda3d.core import GeomNode
        try:
            geom_nodes = []
            if isinstance(model_np.node(), GeomNode):
                geom_nodes.append(model_np.node())
            matches = model_np.find_all_matches("**/+GeomNode")
            for i in range(matches.get_num_paths()):
                geom_nodes.append(matches.get_path(i).node())
            for gnode in geom_nodes:
                for g in range(gnode.get_num_geoms()):
                    geom = gnode.modify_geom(g)
                    self._recompute_normals_up(geom)
                    # Tangents AFTER normals (the w-sign uses the new normals).
                    self._generate_tangents_for_geom(geom)
        except Exception as exc:
            print(f"[MyApp] fill-mesh normal/tangent fix failed: {exc}")

        # has_tangents=True: we generated tangents above, so bind a normal map
        # (flat if none) — required for correct directional lighting on Intel.
        self._apply_pbr_surface(model_np, diffuse_path or "",
                                roughness_path=roughness_path,
                                base_color=base_color, has_tangents=True)
        model_np.set_two_sided(True)

    def _recompute_normals_up(self, geom):
        """
        Recompute area-weighted smooth vertex normals for `geom` in place and
        orient them upward (negate if the mesh's mean normal points down), so
        simplepbr's directional lighting produces real light/shadow contrast
        instead of a black/flat surface. No-op if the geom has no normals.
        """
        from panda3d.core import (GeomVertexReader, GeomVertexWriter,
                                  InternalName, LVector3)
        gvd = geom.get_vertex_data()
        if not gvd.has_column(InternalName.get_normal()):
            return
        if geom.get_num_primitives() == 0:
            return
        prim = geom.get_primitive(0).decompose()

        preader = GeomVertexReader(gvd, InternalName.get_vertex())
        pos = []
        while not preader.is_at_end():
            d = preader.get_data3f()
            pos.append(LVector3(d[0], d[1], d[2]))
        n = len(pos)
        if n == 0:
            return

        acc = [LVector3(0, 0, 0) for _ in range(n)]
        verts = prim.get_vertex_list()
        for t in range(0, len(verts) - 2, 3):
            i0, i1, i2 = verts[t], verts[t + 1], verts[t + 2]
            if i0 >= n or i1 >= n or i2 >= n:
                continue
            # Area-weighted face normal (un-normalized cross product).
            face_n = (pos[i1] - pos[i0]).cross(pos[i2] - pos[i0])
            acc[i0] += face_n
            acc[i1] += face_n
            acc[i2] += face_n

        # Orient every normal OUTWARD from the mesh centroid. This is robust to
        # the node's later transform (some fill meshes are rotated P=90 after
        # materialing, so a local "+Z up" orientation would end up sideways and
        # dark). "Outward" is a geometric property preserved under rotation,
        # and the meshes are placed roof-up, so outward = world-up on the
        # visible top surface → lit by the sun + sky instead of black. Bumps
        # still tilt the normals, preserving light/shadow contrast.
        centroid = LVector3(0, 0, 0)
        for p in pos:
            centroid += p
        centroid *= (1.0 / n)

        gvd_mod = geom.modify_vertex_data()
        nwriter = GeomVertexWriter(gvd_mod, InternalName.get_normal())
        for idx, v in enumerate(acc):
            nrm = LVector3(v)
            outward = pos[idx] - centroid
            if nrm.dot(outward) < 0.0:
                nrm = LVector3(-nrm.x, -nrm.y, -nrm.z)
            if nrm.length_squared() > 1e-12:
                nrm.normalize()
            else:
                nrm = LVector3(outward) if outward.length_squared() > 1e-9 \
                    else LVector3(0, 0, 1)
                nrm.normalize()
            nwriter.set_data3f(nrm.x, nrm.y, nrm.z)

    def create_mesh_from_data(self, vertices: np.ndarray, triangles: np.ndarray,
                            normals: np.ndarray, uvs: np.ndarray):
        from panda3d.core import (GeomVertexFormat, GeomVertexData, Geom,
                                GeomNode, GeomTriangles, GeomVertexWriter)

        # 1. Поднимаем меш по Z (опционально)
        vertices = vertices.copy()
        vertices[:, 2] += 1.0

        # 2. Создаём геометрию с форматом V3N3T2
        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("landscape", fmt, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")
        normals = -normals

        for i in range(len(vertices)):
            v = vertices[i]
            n = normals[i]
            uv = uvs[i]
            vertex_writer.addData3f(v[0], v[1], v[2])
            normal_writer.addData3f(n[0], n[1], n[2])
            texcoord_writer.addData2f(uv[0], uv[1])

        # 3. Индексы треугольников
        prim = GeomTriangles(Geom.UHStatic)
        for tri in triangles:
            prim.addVertices(int(tri[0]), int(tri[1]), int(tri[2]))
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("landscape")
        node.addGeom(geom)

        np_node = self.render.attachNewNode(node)
        self._apply_auto_shader(np_node)
        np_node.setTwoSided(False)

        # 4. Применение PBR-текстур
        self._apply_textures_and_material(np_node)

        return np_node
    
    def _apply_textures_and_material(self, model_np):
        import os
        from panda3d.core import Texture, TextureStage, Material

        texset = self.current_texture_set

        # Пути к текстурам
        diffuse_path = texset.get("diffuse") or texset.get("albedo") or "assets/textures/concrete_8k/concrete_debris_diff_8k.jpg"
        normal_path = texset.get("normal", "assets/textures/concrete_8k/concrete_debris_nor_dx_8k.jpg")
        roughness_path = texset.get("roughness")
        metallic_path = texset.get("metallic")

        # Проверка существования файлов (с резервными)
        if not os.path.exists(diffuse_path):
            diffuse_path = "assets/textures/concrete_8k/concrete_debris_diff_8k.jpg"
        if not os.path.exists(normal_path):
            normal_path = "assets/textures/concrete_8k/concrete_debris_nor_dx_8k.jpg"

        # PERFORMANCE preset (simplepbr): PBR-friendly path instead of
        # RenderPipeline's green-emission / 4-stage convention. Roughness map
        # is bound (no tangents on these meshes, so the normal map is skipped).
        if not self.use_render_pipeline:
            self._apply_pbr_surface(model_np, diffuse_path,
                                    roughness_path=roughness_path)
            model_np.set_two_sided(True)
            return

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

        # Конвертация ОС-пути в Panda3D-Filename: без этого
        # loader.loadTexture("C:\\...") падает с "Could not load texture",
        # потому что строковый конструктор Filename ждёт Unix-слэши.
        def _pf(path):
            return Filename.fromOsSpecific(str(path))

        # Albedo
        diffuse_tex = self.loader.loadTexture(_pf(diffuse_path))
        setup_tex(diffuse_tex, srgb=True)
        model_np.set_texture(ts_color, diffuse_tex)

        # Normal map
        normal_tex = self.loader.loadTexture(_pf(normal_path))
        setup_tex(normal_tex)
        model_np.set_texture(ts_normal, normal_tex)

        # Metallic (всегда заполняем)
        if metallic_path and os.path.exists(metallic_path):
            metal_tex = self.loader.loadTexture(_pf(metallic_path))
        else:
            metal_tex = Texture("dummy_metal")
            metal_tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
            metal_tex.set_ram_image(b"\x00")  # чёрный = 0 металличности
        setup_tex(metal_tex)
        model_np.set_texture(ts_metal, metal_tex)

        # Roughness (всегда заполняем заглушкой)
        if roughness_path and os.path.exists(roughness_path):
            rough_tex = self.loader.loadTexture(_pf(roughness_path))
        else:
            rough_tex = Texture("dummy_rough")
            rough_tex.setup_2d_texture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
            rough_tex.set_ram_image(b"\x80")  # 0x80 = 0.5 в линейном (средняя шероховатость)
        setup_tex(rough_tex)
        model_np.set_texture(ts_rough, rough_tex)

        # Флаги RP
        # model_np.set_shader_auto()  # Для RP обычно не требуется, но можно оставить
        model_np.set_two_sided(True)

    def create_top_overlay(self):
        from direct.gui.DirectFrame import DirectFrame
        from direct.gui.DirectLabel import DirectLabel
        from direct.gui import DirectGuiGlobals as DGG
        from panda3d.core import TextNode, SamplerState

        # Загрузка шрифта (один раз)
        if not hasattr(self, "ui_font"):
            font_path = Filename.fromOsSpecific("assets/fonts/JOST/static/Jost-Regular.ttf").getFullpath()
            self.ui_font = loader.loadFont(font_path)
            self.ui_font.setPixelsPerUnit(60)
            self.ui_font.setMinfilter(SamplerState.FT_linear)
            self.ui_font.setMagfilter(SamplerState.FT_linear)

        # (по желанию) применить шрифт глобально
        # DGG.setDefaultFont(self.ui_font)

        # Цвета в формате (r, g, b, a) от 0 до 1
        bg_color = (0.102, 0.102, 0.129, 1.0)        # #1a1a21
        accent_color = (0.29, 0.50, 0.75, 1.0)       # #4a7fbe
        text_color = (1.0, 1.0, 1.0, 1.0)            # белый

        panel_width = 500
        panel_height = 220                            # увеличено для заголовка
        margin_x = 20
        margin_z = -20

        # Основной фон панели
        self.top_overlay = DirectFrame(
            parent=pixel2d,
            frameColor=bg_color,
            frameSize=(0, panel_width, -panel_height, 0),
            pos=(margin_x, 0, margin_z),
            sortOrder=100,
            suppressMouse=True
        )

        # Заголовок панели
        title_options = {
            "parent": self.top_overlay,
            "text": "Data Information",
            "text_scale": 18,
            "text_fg": accent_color,
            "text_align": TextNode.ALeft,
            "text_font": self.ui_font,
            "frameColor": (0, 0, 0, 0),
            "relief": None
        }
        DirectLabel(pos=(20, 0, -25), **title_options)

        # Общие настройки для информационных строк
        label_options = {
            "parent": self.top_overlay,
            "text_scale": 16,
            "text_fg": text_color,
            "text_align": TextNode.ALeft,
            "text_font": self.ui_font,
            "frameColor": (0, 0, 0, 0),
            "relief": None
        }

        text_margin = 20
        start_y = -55                                 # начальная позиция после заголовка
        row_height = 28                               # увеличенный шаг между строками

        self.model_label = DirectLabel(
            text="Модель: —",
            pos=(text_margin, 0, start_y),
            **label_options
        )
        self.texture_label = DirectLabel(
            text="Наполнитель: —",
            pos=(text_margin, 0, start_y - row_height * 1),
            **label_options
        )
        self.volume_label = DirectLabel(
            text="Объём: —",
            pos=(text_margin, 0, start_y - row_height * 2),
            **label_options
        )
        self.initial_volume_label = DirectLabel(
            text="Исходный объём: —",
            pos=(text_margin, 0, start_y - row_height * 3),
            **label_options
        )
        self.car_number_label = DirectLabel(
            text="Номер машины: —",
            pos=(text_margin, 0, start_y - row_height * 4),
            **label_options
        )
        self.time_label = DirectLabel(
            text="Время проезда: —",
            pos=(text_margin, 0, start_y - row_height * 5),
            **label_options
        )

    def update_overlay_info(self, model=None, texture=None, volume=None, car_number=None, initial_volume=None, time=None):
        # The legacy create_top_overlay() built model_label / texture_label /
        # volume_label / etc. on the in-scene DirectFrame.  We removed that
        # overlay - so all those attributes may be missing.  Each branch
        # is now guarded with hasattr() so the function is a no-op when
        # the labels don't exist (the new HUD shows the same info via Qt
        # overlays / right-panel Details).
        if model is not None and hasattr(self, "model_label") and self.model_label:
            self.model_label['text'] = f"Модель: {model}"
        if texture is not None and hasattr(self, "texture_label") and self.texture_label:
            import os
            tex_name = os.path.basename(texture) if texture else "—"
            self.texture_label['text'] = f"Наполнитель: {tex_name}"
        if volume is not None and hasattr(self, "volume_label") and self.volume_label:
            self.volume_label['text'] = f"Объём: {volume:.2f}"
        if initial_volume is not None and hasattr(self, "initial_volume_label") and self.initial_volume_label:
            self.initial_volume_label['text'] = f"Исходный объём: {initial_volume:.2f}"
        if car_number is not None and hasattr(self, "car_number_label") and self.car_number_label:
            self.car_number_label['text'] = f"Номер машины: {car_number}"
        if time is not None and hasattr(self, "time_label") and self.time_label:
            self.time_label['text'] = f"Время проезда: {time}"


    def setup_window_for_parenting(self, parent_hwnd):
        if hasattr(self, 'win') and self.win:
            try:
                hwnd = self.win.getWindowHandle()
                
                win32gui.SetParent(hwnd, parent_hwnd)
                
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                style = style & ~(win32con.WS_CAPTION | win32con.WS_THICKFRAME | 
                                 win32con.WS_MINIMIZEBOX | win32con.WS_MAXIMIZEBOX | 
                                 win32con.WS_SYSMENU | win32con.WS_BORDER | 
                                 win32con.WS_DLGFRAME)
                style = style | win32con.WS_CHILD
                win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
                
                return hwnd
            except Exception as e:
                return None
        return None

    def set_texture_set(self, texture_config):
        new_texture_set = texture_config.copy()
        
        defaults = {
            'textureRepeatX': 1.35,
            'textureRepeatY': 3.2,
            'strength': 0.14,
            'textureRepeatU': 160.0,
            'textureRepeatV': 160.0
        }
        
        for key, default_value in defaults.items():
            if key not in new_texture_set:
                new_texture_set[key] = default_value
        
        if 'albedo' in new_texture_set and 'diffuse' not in new_texture_set:
            new_texture_set['diffuse'] = new_texture_set['albedo']
        if 'height' in new_texture_set and 'displacement' not in new_texture_set:
            new_texture_set['displacement'] = new_texture_set['height']
        
        self.current_texture_set = new_texture_set
        # tex_path = new_texture_set.get('diffuse') or new_texture_set.get('albedo')
        # self.update_overlay_info(texture=tex_path)
        
        if('mesh_distributions' in new_texture_set):
            self.mesh_distributions_data = new_texture_set["mesh_distributions"]
        else:
            self.mesh_distributions_data = []
        
        if hasattr(self, 'final_model') and self.final_model is not None:
            self.perlin_generator.create_mesh_from_perlin_data()

        return new_texture_set

    def init_depth_renderer(self):
        self.taskMgr.do_method_later(0.5, self._delayed_depth_init, "delayed_depth_init")

    def _delayed_depth_init(self, task):
        self.depth_renderer = DepthMapRenderer(self)
        self.taskMgr.add(self.update_depth_overlay_task, "update_depth_overlay_task")
            
        return task.done

    def update_depth_overlay_task(self, task):
        if self.depth_renderer and hasattr(self.depth_renderer, 'overlay_node'):
            if not self.depth_renderer.overlay_node.isHidden():
                success = self.depth_renderer.update_depth_texture()
                if not success:
                    print("Failed to update depth texture")
        return task.cont

    def toggle_depth_overlay(self):
        if hasattr(self, 'depth_renderer') and self.depth_renderer:
            is_enabled = self.depth_renderer.toggle_overlay()
            return is_enabled
        else:
            return False
    
    def log_camera_parameters(self):
        lens = self.cam.node().get_lens()
        camera_zoom_data = {}
        focal_length_pixels = None
        perspective_angle_x = None
        perspective_angle_y = None
        
        if isinstance(lens, OrthographicLens):
            film_size = lens.get_film_size()
            camera_zoom_data = {
                "type": "orthographic",
                "film_size": {
                    "x": float(film_size.x),
                    "y": float(film_size.y)
                }
            }
        elif hasattr(lens, 'get_fov'):
            fov = lens.get_fov()
            camera_zoom_data = {
                "type": "perspective",
                "fov": {
                    "x": float(fov.x),
                    "y": float(fov.y)
                }
            }
            
            # Расчет фокусного расстояния в пикселях
            if self.win:
                # Получаем размер окна в пикселях
                width = self.win.getXSize()
                height = self.win.getYSize()
                
                # Конвертируем FOV из градусов в радианы
                fov_x_rad = math.radians(fov.x)
                fov_y_rad = math.radians(fov.y)
                
                # Вычисляем фокусное расстояние по горизонтали и вертикали
                focal_length_x = (width / 2.0) / math.tan(fov_x_rad / 2.0)
                focal_length_y = (height / 2.0) / math.tan(fov_y_rad / 2.0)
                
                # Берем среднее значение для общего фокусного расстояния
                focal_length_pixels = (focal_length_x + focal_length_y) / 2.0
            else:
                # Если окно недоступно, используем стандартные размеры
                width, height = 1920, 1080
                
                fov_x_rad = math.radians(fov.x)
                fov_y_rad = math.radians(fov.y)
                
                focal_length_x = (width / 2.0) / math.tan(fov_x_rad / 2.0)
                focal_length_y = (height / 2.0) / math.tan(fov_y_rad / 2.0)
                
                focal_length_pixels = (focal_length_x + focal_length_y) / 2.0
            
            # Углы наклона камеры по X и Y (в градусах)
            # В Panda3D:
            # - pitch (P) - наклон вверх/вниз (вращение вокруг оси X)
            # - heading (H) - поворот влево/вправо (вращение вокруг оси Z)
            
            # Получаем текущую ориентацию камеры
            camera_hpr = self.camera.getHpr()
            perspective_angle_x = float(camera_hpr.x)  # heading (вращение вокруг Z)
            perspective_angle_y = float(camera_hpr.y)  # pitch (вращение вокруг X)
        
        camera_data = {
            "position": {
                "x": float(self.camera.get_x()),
                "y": float(self.camera.get_y()),
                "z": float(self.camera.get_z())
            },
            "rotation": {
                "h": float(self.camera.get_h()),
                "p": float(self.camera.get_p()),
                "r": float(self.camera.get_r())
            },
            "view": self.current_view,
            "zoom": camera_zoom_data,
            "perspective_focal_length": float(focal_length_pixels) if focal_length_pixels is not None else None,
            "perspective_angle_x": perspective_angle_x,
            "perspective_angle_y": perspective_angle_y
        }
        
        camera_json = json.dumps(camera_data, indent=4)
        
        # Вместо вывода в консоль, отображаем диалог с информацией
        # Для этого нужно иметь доступ к GUI
        if hasattr(self, 'control_panel') and self.control_panel:
            self.control_panel.show_camera_info_dialog(camera_json)
        else:
            # Fallback: выводим в консоль
            print(camera_json)
        
        return camera_json

    def calculate_mesh_volume(self, model):
        if not model:
            return 0.0
            
        node = model.node()
        if isinstance(node, GeomNode):
            geom_node = node
        else:
            geom_node_path = model.find("**/+GeomNode")
            if geom_node_path.isEmpty():
                return 0.0
            geom_node = geom_node_path.node()
        
        if geom_node.getNumGeoms() == 0:
            return 0.0
        
        geom = geom_node.getGeom(0)
        
        if geom.getNumPrimitives() == 0:
            return 0.0
        
        primitive = geom.getPrimitive(0)
        
        if not isinstance(primitive, GeomTriangles):
            return 0.0
        
        transform = model.getNetTransform().getMat()
        
        vertex_data = geom.getVertexData()
        vertex_reader = GeomVertexReader(vertex_data, "vertex")
        
        volume = 0.0
        num_tris = primitive.getNumPrimitives()
        
        for i in range(num_tris):
            vi0 = primitive.getVertex((i * 3) + 0)
            vi1 = primitive.getVertex((i * 3) + 1)
            vi2 = primitive.getVertex((i * 3) + 2)
            
            vertex_reader.setRow(vi0)
            v0 = vertex_reader.getData3f()
            
            vertex_reader.setRow(vi1)
            v1 = vertex_reader.getData3f()
            
            vertex_reader.setRow(vi2)
            v2 = vertex_reader.getData3f()
            
            v0 = transform.xformPoint(v0)
            v1 = transform.xformPoint(v1)
            v2 = transform.xformPoint(v2)
            
            volume += v0.dot(v1.cross(v2))
        
        volume = abs(volume) / 6.0
        return volume
    
    def perform_AABB_plane(self):
        if self.gui:
            self.gui.log_message("Начало perform_AABB_plane...")

        # Поиск модели кузова
        target_model = None
        target_model_path = None
        for model in self.loaded_models:
            model_id = id(model)
            if model_id in self.model_paths:
                model_filename = os.path.basename(self.model_paths[model_id])
                if model_filename == self.Target_Cuzov:
                    target_model = model
                    target_model_path = self.model_paths[model_id]
                    break

        if target_model is None:
            if self.gui:
                self.gui.log_message("❌ Модель кузова не найдена")
            return False

        # Сохраняем и сбрасываем трансформации кузова для корректного AABB
        old_scale = target_model.getScale()
        old_pos = target_model.getPos()
        old_hpr = target_model.getHpr()
        target_model.setScale(1.0, 1.0, 1.0)
        target_model.setPos(0, 0, 0)
        target_model.setHpr(0, 0, 0)

        if self.gui:
            self.gui.log_message("📦 Вычисление AABB модели...")
        min_point, max_point = target_model.getTightBounds()
        aabb_center = (min_point + max_point) / 2.0
        aabb_size = max_point - min_point

        # Восстанавливаем трансформации
        target_model.setScale(old_scale)
        target_model.setPos(old_pos)
        target_model.setHpr(old_hpr)

        ground_pos = self.ground_plane.getPos()
        plane_thickness = 0.05

        # Создаём плоскость и AABB как trimesh объекты
        full_plane_mesh = trimesh.creation.box(
            extents=[self.plane_size_x, self.plane_size_y, plane_thickness]
        )
        aabb_mesh = trimesh.creation.box(
            extents=[aabb_size.x, aabb_size.y, aabb_size.z]
        )

        # Перемещаем AABB в мировые координаты кузова
        aabb_transform = trimesh.transformations.translation_matrix([
            aabb_center.x, aabb_center.y, aabb_center.z
        ])
        aabb_mesh.apply_transform(aabb_transform)

        # Перемещаем плоскость в позицию ground_plane
        full_plane_mesh.apply_translation([ground_pos.x, ground_pos.y, ground_pos.z])

        if self.gui:
            self.gui.log_message(f"Plane size: {self.plane_size_x} x {self.plane_size_y}, AABB size: {aabb_size.x:.2f} x {aabb_size.y:.2f} x {aabb_size.z:.2f}")

        # Проверка наличия TLS клиента
        if not hasattr(self, 'tls_client') or self.tls_client is None:
            print("[ERROR] TLS client not available. Cannot perform intersection.")
            return False

        if self.gui:
            self.gui.log_message("✂️ Выполнение boolean пересечения через TLS-сервер...")
        try:
            result_verts, result_tris = self.tls_client.send_boolean_intersection(
                full_plane_mesh.vertices,
                full_plane_mesh.faces,
                aabb_mesh.vertices,
                aabb_mesh.faces,
                return_volume_only=False
            )
            result_mesh = trimesh.Trimesh(vertices=result_verts, faces=result_tris)
            if result_mesh.is_empty:
                if self.gui:
                    self.gui.log_message("⚠️ Результат пересечения пуст")
                return False
        except Exception as e:
            if self.gui:
                self.gui.log_message(f"❌ Ошибка булевой операции: {e}")
            return False

        if self.gui:
            self.gui.log_message("✅ Boolean операция завершена, создание меша...")
        csg_result_panda = self.trimesh_to_panda(result_mesh)

        # Применяем материал к результату
        material = Material()
        material.setDiffuse((0, 0.7, 0, 1))
        material.setAmbient((0, 0.3, 0, 1))
        material.setSpecular((0.5, 0.5, 0.5, 1))
        material.setShininess(50)
        csg_result_panda.setMaterial(material)
        self._apply_auto_shader(csg_result_panda)

        # Заменяем старую ground_plane на результат CSG
        old_pos_plane = self.ground_plane.getPos()
        old_hpr_plane = self.ground_plane.getHpr()
        old_scale_plane = self.ground_plane.getScale()

        self.ground_plane.removeNode()

        csg_result_panda.reparentTo(self.render)
        csg_result_panda.setPos(old_pos_plane)
        csg_result_panda.setHpr(old_hpr_plane)
        csg_result_panda.setScale(old_scale_plane)

        self.ground_plane = csg_result_panda

        if not hasattr(self, 'csg_results'):
            self.csg_results = []

        self.csg_results.append({
            "target_model_path": target_model_path,
            "result_node": csg_result_panda,
            "original_model": target_model
        })

        self.ground_plane.hide()

        if self.gui:
            self.gui.log_message("✅ AABB plane успешно выполнено")
        return True

    def panda_to_trimesh(self, node_path):
        geom_node = node_path.node()
        if not isinstance(geom_node, GeomNode):
            geom_node_path = node_path.find("**/+GeomNode")
            geom_node = geom_node_path.node()
        
        transform = node_path.getNetTransform().getMat()
        
        vertices = []
        faces = []
        
        for i in range(geom_node.getNumGeoms()):
            geom = geom_node.getGeom(i)
            vdata = geom.getVertexData()
            
            vertex_reader = GeomVertexReader(vdata, "vertex")
            while not vertex_reader.isAtEnd():
                pos = vertex_reader.getData3f()
                pos = transform.xformPoint(pos)
                vertices.append([pos.x, pos.y, pos.z])
            
            for j in range(geom.getNumPrimitives()):
                prim = geom.getPrimitive(j)
                if isinstance(prim, GeomTriangles):
                    for k in range(prim.getNumPrimitives()):
                        start = prim.getPrimitiveStart(k)
                        end = prim.getPrimitiveEnd(k)
                        face = []
                        for idx in range(start, end):
                            vi = prim.getVertex(idx)
                            face.append(vi)
                        if len(face) == 3:
                            faces.append(face)
        
        if not vertices or not faces:
            return None
            
        return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    def trimesh_to_panda(self, trimesh_mesh):
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces

        if not hasattr(trimesh_mesh, 'vertex_normals') or len(trimesh_mesh.vertex_normals) != len(vertices):
            trimesh_mesh.compute_vertex_normals()

        normals = trimesh_mesh.vertex_normals

        # Planar UV-маппинг по XY-bbox меша. Без этого все texcoord = (0, 0)
        # и текстура схлопывается в один тексел (видимо как однотонный цвет).
        # Тот же подход в babylon-viewer.js (см. _loadObjMesh: UV считаются
        # после загрузки OBJ от bbox по (X, Z) Babylon, что соответствует
        # (X, Y) в наших OBJ-файлах Panda-конвенции).
        verts_np = np.asarray(vertices, dtype=np.float64)
        vmin_x, vmax_x = float(verts_np[:, 0].min()), float(verts_np[:, 0].max())
        vmin_y, vmax_y = float(verts_np[:, 1].min()), float(verts_np[:, 1].max())
        range_x = max(vmax_x - vmin_x, 1e-6)
        range_y = max(vmax_y - vmin_y, 1e-6)

        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("trimesh_result", format, Geom.UHStatic)

        vertex_writer = GeomVertexWriter(vdata, "vertex")
        normal_writer = GeomVertexWriter(vdata, "normal")
        texcoord_writer = GeomVertexWriter(vdata, "texcoord")

        for i, vertex in enumerate(vertices):
            vertex_writer.addData3f(vertex[0], vertex[1], vertex[2])

            if i < len(normals):
                normal = normals[i]
                if np.any(np.isnan(normal)) or np.linalg.norm(normal) < 0.1:
                    normal = [0, 0, 1]
                normal_writer.addData3f(normal[0], normal[1], normal[2])
            else:
                normal_writer.addData3f(0, 0, 1)

            u = (float(vertex[0]) - vmin_x) / range_x
            v = (float(vertex[1]) - vmin_y) / range_y
            texcoord_writer.addData2f(u, v)
        
        prim = GeomTriangles(Geom.UHStatic)
        for face in faces:
            prim.addVertices(face[0], face[1], face[2])
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode("trimesh_result")
        node.addGeom(geom)
        print(self.particle_flag)
        if self.particle_flag == True:
            if(self.canDistributeMeshes):
                self.distribute_meshes(node)
        
        result_np = self.render.attachNewNode(node)
        
        return result_np
    
    def extract_mesh_from_geom_node(self, geom_node):
        """
        Extract vertices and triangle indices from a GeomNode.
        Assumes the first Geom uses triangle primitives.
        Returns:
            vertices: np.ndarray of shape (V, 3)
            indices:  np.ndarray of shape (T*3,) with dtype int32
        """
        if geom_node.getNumGeoms() == 0:
            raise ValueError("GeomNode has no geoms")

        geom = geom_node.getGeom(0)
        vdata = geom.getVertexData()
        primitive = geom.getPrimitive(0)

        # Decompose to triangles if needed (e.g., from tristrips or quads)
        primitive = primitive.decompose()

        # --- Extract vertices ---
        vertex_reader = GeomVertexReader(vdata, "vertex")
        num_vertices = vdata.getNumRows()
        vertices = []
        for i in range(num_vertices):
            v = vertex_reader.getData3()
            vertices.append([v[0], v[1], v[2]])
        vertices_np = np.array(vertices, dtype=np.float32)

        # --- Extract indices ---
        indices = []
        for p in range(primitive.getNumPrimitives()):
            start = primitive.getPrimitiveStart(p)
            end = primitive.getPrimitiveEnd(p)
            for i in range(start, end):
                idx = primitive.getVertex(i)
                indices.append(idx)
        indices_np = np.array(indices, dtype=np.int32)

        return vertices_np, indices_np

    def distribute_meshes(self, geom_node):
        if geom_node.getNumGeoms() > 0:
            vertices_np, indices_np = self.extract_mesh_from_geom_node(geom_node)

            # Остановить предыдущие распределения
            for distrib in self.mesh_distributions:
                distrib.stop_rendering()
            self.mesh_distributions.clear()

            # Создать новые распределения на основе данных
            for data in self.mesh_distributions_data:
                distrib = MeshDistributor(self)  # передаём сам MyApp
                model1 = self.loader.load_model(data['mesh'], noCache=True)
                distrib.distribute(vertices_np, indices_np, data['count'], seed=data['seed'])
                distrib.start_rendering(model1, data['size'], data['size_var'])
                self.mesh_distributions.append(distrib)

    def create_ground_plane(self):
        if self.ground_plane:
            self.ground_plane.removeNode()
        
        format = GeomVertexFormat.getV3n3t2()
        format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("ground_plane", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        
        size_x = self.plane_size_x
        size_y = self.plane_size_y
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        
        vertices = [
            (-half_size_x, -half_size_y, 0),  
            (half_size_x, -half_size_y, 0),   
            (half_size_x, half_size_y, 0),    
            (-half_size_x, half_size_y, 0)    
        ]
        
        plane_normal = (0, 0, 1)
        
        for v in vertices:
            vertex.addData3f(v[0], v[1], v[2])
            normal.addData3f(plane_normal[0], plane_normal[1], plane_normal[2])
            u = (v[0] + half_size_x) / size_x
            v_coord = (v[1] + half_size_y) / size_y
            texcoord.addData2f(u, v_coord)
        
        prim = GeomTriangles(Geom.UHStatic)
        prim.addVertices(0, 1, 2)
        prim.addVertices(0, 2, 3)
        prim.closePrimitive()
        
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        
        node = GeomNode("ground_plane")
        node.addGeom(geom)
        
        self.ground_plane = self.render.attachNewNode(node)
        material = Material()
        material.setDiffuse((0, 0.7, 0, 1))
        material.setAmbient((0, 0.3, 0, 1))
        material.setSpecular((0.5, 0.5, 0.5, 1))
        material.setShininess(50)
        # simplepbr reads base_color/roughness/metallic rather than the
        # legacy diffuse/specular, so set those too for the PERFORMANCE preset.
        if not self.use_render_pipeline:
            material.set_base_color((0, 0.7, 0, 1))
            material.set_roughness(0.8)
            material.set_metallic(0.0)
        self.ground_plane.setMaterial(material)
        
        self.ground_plane.setPos(0, 0, 0)

        self._apply_auto_shader(self.ground_plane)

        self.ground_plane.setTwoSided(True)

    def set_plane_size_x(self, size_x):
        self.plane_size_x = size_x

    def set_plane_size_y(self, size_y):
        self.plane_size_y = size_y

    def set_plane_position(self, x, y, z):
        self.ground_plane.setPos(x, y, z)

    def zoom_camera(self, factor):
        cam_pos = self.camera.get_pos()
        cam_hpr = self.camera.get_hpr()

        if self.current_view == "perspective":
            distance = cam_pos.length()
            if distance > 0.1:
                new_distance = distance * (1.0 / factor)
                scale_factor = new_distance / distance
                self.camera.set_pos(cam_pos * scale_factor)

        elif self.current_view in ["front", "back", "left", "right", "top", "bottom"]:
            lens = self.cam.node().get_lens()
            if hasattr(lens, 'get_fov') and hasattr(lens, 'set_fov'):
                if isinstance(lens, OrthographicLens):
                     old_film_size = lens.get_film_size()
                     new_film_size = old_film_size * (1.0 / factor)
                     lens.set_film_size(new_film_size)
                else:
                     old_fov = lens.get_fov()
                     new_fov_x = max(1.0, min(179.0, old_fov.x * (1.0 / factor)))
                     new_fov_y = max(1.0, min(179.0, old_fov.y * (1.0 / factor)))
                     lens.set_fov(LVecBase2f(new_fov_x, new_fov_y))

    def fix_shadow_camera_aspect(self):
        if hasattr(self.render_pipeline, 'light_manager'):
            light_mgr = self.render_pipeline.light_manager

    def create_perlin_noise_mesh(self):
        import os
        from panda3d.core import Texture, TextureStage, Material, GeomVertexFormat, GeomVertexData, \
            GeomVertexWriter, GeomTriangles, Geom, GeomNode, LPoint3f

        size_x = 2000.0
        size_y = 2000.0
        size_z = 2.0
        position = LPoint3f(0.0, 0.0, -1.0)

        # Коэффициенты повторения текстур по-прежнему могут браться из current_texture_set
        texture_repeat_u = self.current_texture_set.get('textureRepeatU', 160.0)
        texture_repeat_v = self.current_texture_set.get('textureRepeatV', 160.0)

        # PERFORMANCE preset adds a tangent column so simplepbr can use the
        # ground normal map. The grid is flat (normal +Z) and the UV layout
        # has U tracking world +Y, so a constant tangent (0,1,0,1) is correct.
        # RenderPipeline keeps the original V3n3t2 format unchanged.
        add_tangent = not self.use_render_pipeline
        if add_tangent:
            from panda3d.core import (GeomVertexArrayFormat, InternalName,
                                      GeomEnums)
            _arr = GeomVertexArrayFormat()
            _arr.add_column(InternalName.get_vertex(), 3,
                            GeomEnums.NT_float32, GeomEnums.C_point)
            _arr.add_column(InternalName.get_normal(), 3,
                            GeomEnums.NT_float32, GeomEnums.C_normal)
            _arr.add_column(InternalName.get_texcoord(), 2,
                            GeomEnums.NT_float32, GeomEnums.C_texcoord)
            _arr.add_column(InternalName.get_tangent(), 4,
                            GeomEnums.NT_float32, GeomEnums.C_vector)
            format = GeomVertexFormat()
            format.add_array(_arr)
            format = GeomVertexFormat.registerFormat(format)
        else:
            format = GeomVertexFormat.getV3n3t2()
            format = GeomVertexFormat.registerFormat(format)
        vdata = GeomVertexData("simplified_perlin_data", format, Geom.UHStatic)
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        tangent = (GeomVertexWriter(vdata, InternalName.get_tangent())
                   if add_tangent else None)

        grid_size = 64
        step_x = size_x / (grid_size - 1) if grid_size > 1 else 0
        step_y = size_y / (grid_size - 1) if grid_size > 1 else 0
        half_size_x = size_x / 2.0
        half_size_y = size_y / 2.0
        pos_z = position.getZ() + (size_z / 2.0)

        for y in range(grid_size):
            for x in range(grid_size):
                world_x = x * step_x - half_size_x
                world_y = y * step_y - half_size_y
                world_z = pos_z
                vertex.addData3f(world_x, world_y, world_z)
                normal.addData3f(0, 0, 1)
                normalized_u = x / (grid_size - 1) if grid_size > 1 else 0.0
                normalized_v = y / (grid_size - 1) if grid_size > 1 else 0.0
                u = normalized_v * texture_repeat_u
                v = -normalized_u * texture_repeat_v
                texcoord.addData2f(u, v)
                if add_tangent:
                    # w = -1 flips the bitangent, converting the DirectX-style
                    # normal maps (_nor_dx, green = -Y) the project ships to the
                    # OpenGL convention simplepbr expects. (If bumps ever look
                    # inverted, flip this back to +1.)
                    tangent.addData4f(0, 1, 0, -1)

        prim = GeomTriangles(Geom.UHStatic)
        for y in range(grid_size - 1):
            for x in range(grid_size - 1):
                i1 = y * grid_size + x
                i2 = y * grid_size + (x + 1)
                i3 = (y + 1) * grid_size + x
                i4 = (y + 1) * grid_size + (x + 1)
                prim.addVertices(i1, i2, i3)
                prim.addVertices(i2, i4, i3)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode("simplified_perlin_noise_mesh")
        node.addGeom(geom)

        self.perlin_model = self.render.attachNewNode(node)
        self.perlin_model.setPos(0, 0, 0)

        if not hasattr(self, 'loaded_models'):
            self.loaded_models = []
        self.loaded_models.append(self.perlin_model)

        if not hasattr(self, 'model_paths'):
            self.model_paths = {}
        self.model_paths[id(self.perlin_model)] = "perlin_noise_mesh"

        # ===== НОВЫЙ КОД ПРИМЕНЕНИЯ ТЕКСТУР (ФИКСИРОВАННЫЕ ПУТИ) =====
        # ------------------------------------------------------------------
        # Прямое указание путей к текстурам в папке groundPerlin_8k
        # ------------------------------------------------------------------
        diffuse_path = "assets/textures/groundPerlin_8k/aerial_beach_03_diff_8k.jpg"
        normal_path = "assets/textures/groundPerlin_8k/aerial_beach_03_nor_dx_8k.jpg"
        roughness_path = "assets/textures/groundPerlin_8k/aerial_beach_03_rough_8k.jpg"
        metallic_path = None  # металличность не используется, будет заглушка

        # PERFORMANCE preset (simplepbr): full PBR-friendly material (base
        # colour + roughness + normal map — the mesh carries a constant
        # tangent). Skip RP's green-emission, 4-stage setup and the "fixed"
        # depth bin so the ground depth-tests normally under forward render.
        if not self.use_render_pipeline:
            self._apply_pbr_surface(self.perlin_model, diffuse_path,
                                    normal_path=normal_path,
                                    roughness_path=roughness_path,
                                    roughness=1.0, has_tangents=True)
            self.perlin_model.set_two_sided(True)
            return

        # ------------------------------------------------------------------
        # Материал, совместимый с RP
        # ------------------------------------------------------------------
        mat = Material()
        mat.set_base_color((1, 1, 1, 1))          # обязательно белый для PBR
        mat.set_emission((0, 1, 0, 0))             # активирует карту нормалей (RP)
        self.perlin_model.set_material(mat)

        # ------------------------------------------------------------------
        # Вспомогательная функция настройки текстур
        # ------------------------------------------------------------------
        def setup_tex(tex, srgb=False):
            if srgb:
                tex.set_format(Texture.F_srgb)
            tex.set_minfilter(Texture.FTLinearMipmapLinear)
            tex.set_magfilter(Texture.FTLinear)
            tex.set_wrap_u(Texture.WMRepeat)
            tex.set_wrap_v(Texture.WMRepeat)

        # ------------------------------------------------------------------
        # Текстурные стадии (строгий порядок для RP)
        # ------------------------------------------------------------------
        ts_color = TextureStage("0-color")
        ts_color.set_sort(0)
        ts_color.set_priority(0)

        ts_normal = TextureStage("1-normal")
        ts_normal.set_sort(1)
        ts_normal.set_priority(1)

        ts_metal = TextureStage("2-metallic")
        ts_metal.set_sort(2)
        ts_metal.set_priority(2)

        ts_rough = TextureStage("3-roughness")
        ts_rough.set_sort(3)
        ts_rough.set_priority(3)

        # ------------------------------------------------------------------
        # Загрузка и назначение текстур
        # ------------------------------------------------------------------

        # Диффузная (albedo)
        diffuse_tex = self.loader.loadTexture(diffuse_path)
        if diffuse_tex:
            setup_tex(diffuse_tex, srgb=True)
            self.perlin_model.set_texture(ts_color, diffuse_tex)

        # Нормалей
        normal_tex = self.loader.loadTexture(normal_path)
        if normal_tex:
            setup_tex(normal_tex)
            self.perlin_model.set_texture(ts_normal, normal_tex)

        # Металличность (заглушка, т.к. metallic_path = None)
        metal_tex = Texture("dummy_metal")
        metal_tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
        metal_tex.setRamImage(b"\x00")
        setup_tex(metal_tex)
        self.perlin_model.set_texture(ts_metal, metal_tex)

        # Шероховатость
        rough_tex = self.loader.loadTexture(roughness_path)
        if rough_tex:
            setup_tex(rough_tex)
            self.perlin_model.set_texture(ts_rough, rough_tex)

        # ------------------------------------------------------------------
        # Обязательные флаги для RP
        # ------------------------------------------------------------------
        self._apply_auto_shader(self.perlin_model)
        self.perlin_model.set_two_sided(True)

        # Дополнительные настройки из оригинального метода
        self.perlin_model.setBin("fixed", 0)
        self.perlin_model.setDepthOffset(1)

    def _set_initial_time(self, task):
        """Установка начального времени суток"""
        if hasattr(self, 'render_pipeline') and hasattr(self.render_pipeline, 'daytime_mgr'):
            self.render_pipeline.daytime_mgr.time = "15:00"
        return task.done

    def setup_scene(self):
        self.quarry_model = None
        
        self.create_perlin_noise_mesh()
        self.taskMgr.do_method_later(0.5, self._set_initial_time, "set_initial_time")
        
        # # Очищаем существующие источники света
        # self._night_lights = []
        # 
        ## Создаем точечный свет (Point Light)
        #main_light = PointLight()
        #
        ## Устанавливаем позицию из Transform
        #main_light.pos = (4.0762, 1.0055, 5.9039)
        #
        ## Устанавливаем цвет из температуры (6500K)
        #main_light.set_color_from_temperature(6500)
        #
        ## Устанавливаем мощность (Power/Exposure 1000.000)
        #main_light.energy = 100.0
        #
        ## Устанавливаем радиус влияния (Custom Distance 40m)
        #main_light.radius = 20.0
        #
        ## Настройки теней
        #main_light.casts_shadows = True
        #main_light.shadow_map_resolution = 1024
        #
        ## В RenderPipeline влияние на диффузные/глянцевые материалы обычно 
        ## настраивается через материалы, а не через свет
        #
        ## Добавляем свет в сцену
        #self.render_pipeline.add_light(main_light)
        #self._night_lights.append(main_light)
        #
        #print(f"Main light added at position {main_light.pos}")
        #print(f"Light parameters: temperature=6500K, energy=1000.0, radius=40.0m")

        self.camera.set_pos(0, -20, 5)
        self.camera.look_at(0, 0, 0)
        self.disable_mouse()

    def get_camera_orientation(self):
        """Получить ориентацию камеры в удобном формате"""
        pos = self.camera.getPos()
        quat = self.camera.getQuat()
        hpr = quat.getHpr()
        
        return {
            'position': (pos.x, pos.y, pos.z),
            'quaternion': (quat.x, quat.y, quat.z, quat.w),
            'hpr': (hpr.x, hpr.y, hpr.z),
            'forward_vector': self.camera.getQuat().getForward()
        }

    def set_camera_look_at(self, target_point):
        """Направить камеру на целевую точку"""
        self.camera.lookAt(target_point)

    def animate_street_lights(self, task):
        frame_time = self.taskMgr.globalClock.get_frame_time()
        
        if hasattr(self, '_night_lights') and len(self._night_lights) >= 3:
            for i, light in enumerate(self._night_lights):
                if hasattr(light, 'energy'):
                    flicker = 1.0 + (random.random() - 0.5) * 0.1
                    light.energy = 100 * flicker
        
        return task.cont

    def set_drag_sensitivity(self, sensitivity):
        self.drag_sensitivity = sensitivity

    def set_quarry_scale(self, scale):
        self.quarry_model.set_scale(scale)

    def set_quarry_position(self, x, y, z):
        self.quarry_model.set_pos(x, y, z)

    def set_cube_size_x(self, size_x):
        self.cube_size_x = size_x
        self.cube_model.setScale(size_x, self.cube_size_y, self.cube_size_z)

    def set_cube_size_y(self, size_y):
        self.cube_size_y = size_y
        self.cube_model.setScale(self.cube_size_x, size_y, self.cube_size_z)

    def set_cube_size_z(self, size_z):
        self.cube_size_z = size_z
        self.cube_model.setScale(self.cube_size_x, self.cube_size_y, size_z)

    def set_cube_position(self, x, y, z):
        self.cube_model.setPos(x, y, z)

    def toggle_drag_drop_mode(self, enabled):
        self.drag_drop_mode = enabled
        if not enabled:
            self.selected_model = None
            self.drag_start_pos = None
            self.drag_start_mouse_pos = None

    def handle_mouse_left(self):
        if self.drag_drop_mode:
            if self.selected_model:
                self.start_drag_drop()
            else:
                self.select_model_under_mouse()
                if self.selected_model and self.selected_model != self.quarry_model:
                    self.start_drag_drop()
        else:
            if self.current_view == "perspective":
                self.mouse_rotation_enabled = True
                if self.mouseWatcherNode.hasMouse():
                    self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                    self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def handle_mouse_left_up(self):
        if self.drag_drop_mode:
            self.stop_drag_drop()
        else:
            self.mouse_rotation_enabled = False

    def handle_mouse_right(self):
        if self.drag_drop_mode:
            self.select_model_under_mouse()

    def start_drag_drop(self):
        if self.selected_model and self.selected_model != self.quarry_model:
            if self.mouseWatcherNode.hasMouse():
                self.drag_start_pos = self.selected_model.get_pos()
                self.drag_start_mouse_pos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())

    def stop_drag_drop(self):
        self.drag_start_pos = None
        self.drag_start_mouse_pos = None

    def select_model_under_mouse(self):
        if self.mouseWatcherNode.hasMouse():
            if self.ground_plane and not self.ground_plane.isHidden():
                self.selected_model = self.ground_plane
                return
                
            for model in reversed(self.loaded_models):
                if model != self.quarry_model and not model.isHidden():
                    self.selected_model = model
                    return
            
            self.selected_model = None

    def drag_drop_task(self, task):
        if self.drag_drop_mode and self.selected_model and self.drag_start_pos and self.drag_start_mouse_pos:
            if self.mouseWatcherNode.hasMouse():
                current_mouse_x, current_mouse_y = self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY()
                
                dx_mouse = current_mouse_x - self.drag_start_mouse_pos[0]
                dy_mouse = current_mouse_y - self.drag_start_mouse_pos[1]
                
                drag_speed = self.drag_sensitivity
                
                new_x, new_y, new_z = self.drag_start_pos.getX(), self.drag_start_pos.getY(), self.drag_start_pos.getZ()
                
                if self.current_view == "top":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                elif self.current_view == "bottom":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                elif self.current_view == "front":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dy_mouse * drag_speed
                elif self.current_view == "back":
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dy_mouse * drag_speed
                elif self.current_view == "left":
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dx_mouse * drag_speed
                elif self.current_view == "right":
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ() - dx_mouse * drag_speed
                else: 
                    new_x = self.drag_start_pos.getX() + dx_mouse * drag_speed
                    new_y = self.drag_start_pos.getY() - dy_mouse * drag_speed
                    new_z = self.drag_start_pos.getZ()
                
                self.selected_model.set_pos(new_x, new_y, new_z)
                
        return task.cont

    def start_mouse_rotation(self):
        if self.drag_drop_mode:
            return
            
        if self.current_view == "perspective":
            self.mouse_rotation_enabled = True
            if self.mouseWatcherNode.hasMouse():
                self.last_mouse_x = self.mouseWatcherNode.getMouseX()
                self.last_mouse_y = self.mouseWatcherNode.getMouseY()

    def stop_mouse_rotation(self):
        self.mouse_rotation_enabled = False

    def mouse_rotation_task(self, task):
        if self.mouse_rotation_enabled and self.current_view == "perspective":
            if self.mouseWatcherNode.hasMouse():
                mouse_x = self.mouseWatcherNode.getMouseX()
                mouse_y = self.mouseWatcherNode.getMouseY()
                
                dx = mouse_x - self.last_mouse_x
                dy = mouse_y - self.last_mouse_y
                
                if abs(dx) > 0.001 or abs(dy) > 0.001:
                    h = self.camera.get_h() - dx * self.camera_rotation_speed * 100
                    p = self.camera.get_p() + dy * self.camera_rotation_speed * 100
                    
                    p = max(-89, min(89, p))
                    
                    self.camera.set_hpr(h, p, 0)
                
                self.last_mouse_x = mouse_x
                self.last_mouse_y = mouse_y
                
        return task.cont

    def set_top_view(self):
        self.current_view = "top"
        self.camera.set_pos(0, 0, 20)
        self.camera.set_hpr(0, -90, 0)

    def set_bottom_view(self):
        self.current_view = "bottom"
        self.camera.set_pos(0, 0, -20)
        self.camera.set_hpr(0, 90, 0)

    def set_front_view(self):
        self.current_view = "front"
        self.camera.set_pos(0, -20, 0)
        self.camera.look_at(0, 0, 0)

    def set_back_view(self):
        self.current_view = "back"
        self.camera.set_pos(0, 20, 0)
        self.camera.set_hpr(180, 0, 0)

    def set_left_view(self):
        self.current_view = "left"
        self.camera.set_pos(20, 0, 0)
        self.camera.set_hpr(90, 0, 0)

    def set_right_view(self):
        self.current_view = "right"
        self.camera.set_pos(-20, 0, 0)
        self.camera.set_hpr(-90, 0, 0)

    def set_perspective_view(self):
        self.current_view = "perspective"
        self.camera.set_pos(0, -20, 5)
        self.camera.look_at(0, 0, 0)

    def set_key(self, key, value):
        self.key_map[key] = value

    def move_camera_task(self, task):
        speed = 0.1
        
        if self.current_view == "top" or self.current_view == "bottom":
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
                
        elif self.current_view == "front" or self.current_view == "back":
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
                
        elif self.current_view == "left" or self.current_view == "right":
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
                
        else: 
            if self.key_map["forward"]:
                self.camera.set_y(self.camera.get_y() + speed)
            if self.key_map["backward"]:
                self.camera.set_y(self.camera.get_y() - speed)
            if self.key_map["left"]:
                self.camera.set_x(self.camera.get_x() - speed)
            if self.key_map["right"]:
                self.camera.set_x(self.camera.get_x() + speed)
            if self.key_map["up"]:
                self.camera.set_z(self.camera.get_z() + speed)
            if self.key_map["down"]:
                self.camera.set_z(self.camera.get_z() - speed)
        
        return task.cont

    def load_gltf_model(self, file_path):
        model_filename = Filename.from_os_specific(file_path)
        
        model_np = self.loader.load_model(model_filename, noCache=True) 
        
        model_np.reparent_to(self.render)
        # RenderPipeline converts Panda lights/materials to RP equivalents.
        # In the simplepbr (performance) preset the glTF PBR materials are
        # rendered directly, so we skip it and let the node inherit the
        # simplepbr shader from render.
        if self.use_render_pipeline and self.render_pipeline is not None:
            self.render_pipeline.prepare_scene(model_np)
        else:
            self._apply_auto_shader(model_np)
            # Remap RP-style texture stages + build tangents so the model's
            # normal/roughness maps render as PBR under simplepbr.
            self._make_pbr_compatible(model_np)
        model_np.set_pos(0, 0, 0)
        model_np.set_hpr(0, 0, 0)
        model_np.set_scale(1)
        
        self.loaded_models.append(model_np)
        self.model_paths[id(model_np)] = file_path
        
        return model_np 

    def load_model_set(self, config, model_set_name):
        self.clear_scene()
        
        if not hasattr(self, 'perlin_model') or self.perlin_model is None:
            self.create_perlin_noise_mesh()
        
        models_loaded = []
        
        # Используем PROJECT_ROOT для построения абсолютных путей
        def get_absolute_path(relative_path):
            if os.path.isabs(relative_path):
                return relative_path
            return os.path.join(PROJECT_ROOT, relative_path)
        
        if 'other' in config and config['other']:
            other_path = get_absolute_path(config['other'])
            if os.path.exists(other_path):
                other_model = self.load_gltf_model(other_path)
                if other_model:
                    models_loaded.append('other')
                    self.current_other_path = other_path
        
        if 'cuzov' in config and config['cuzov']:
            cuzov_path = get_absolute_path(config['cuzov'])
            if os.path.exists(cuzov_path):
                cuzov_model = self.load_gltf_model(cuzov_path)
                if cuzov_model:
                    models_loaded.append('cuzov')
                    self.Target_Cuzov = os.path.basename(cuzov_path)
                    self.current_cuzov_path = cuzov_path
        
        if 'napolnitel' in config and config['napolnitel']:
            napolnitel_path = get_absolute_path(config['napolnitel'])
            if os.path.exists(napolnitel_path):
                napolnitel_model = self.load_gltf_model(napolnitel_path)
                if napolnitel_model:
                    napolnitel_model.hide()
                    # napolnitel_model.set_p(90)
                    models_loaded.append('napolnitel')
                    self.Target_Napolnitel = os.path.basename(napolnitel_path)
                    self.current_napolnitel_path = napolnitel_path
        
        if 'max_volume' in config:
            self.Target_Volume = config['max_volume']
        
        if 'ground_plane' in config:
            self.current_ground_plane_z = config['ground_plane']
        
        self.current_model_set = model_set_name
        self.update_overlay_info(model=model_set_name)
        
        if hasattr(self, 'perlin_model') and self.perlin_model:
            if self.perlin_model.isHidden():
                self.perlin_model.show()
        
        return True

    def clear_scene(self):
        if hasattr(self, 'test_perlin_mesh') and self.test_perlin_mesh:
            if self.test_perlin_mesh in self.loaded_models:
                self.loaded_models.remove(self.test_perlin_mesh)
            self.test_perlin_mesh.removeNode()
            self.test_perlin_mesh = None
        
        self.last_target_model_trimesh = None
        self.last_best_z = None

        if hasattr(self, 'dynamic_perlin_model') and self.dynamic_perlin_model:
            if self.dynamic_perlin_model in self.loaded_models:
                self.loaded_models.remove(self.dynamic_perlin_model)
            self.dynamic_perlin_model.removeNode()
            self.dynamic_perlin_model = None

        if hasattr(self, 'mesh_distributions'):
            for distrib in self.mesh_distributions:
                distrib.stop_rendering()
            self.mesh_distributions.clear()

        if hasattr(self, 'final_mesh_node') and self.final_mesh_node:
            self.final_mesh_node.removeNode()
            self.final_mesh_node = None
        
        if hasattr(self, 'final_model') and self.final_model:
            if self.final_model in self.loaded_models:
                self.loaded_models.remove(self.final_model)
            self.final_model.removeNode()
            self.final_model = None
            
        if hasattr(self, 'csg_results'):
            for csg_info in self.csg_results:
                if "result_node" in csg_info and csg_info["result_node"]:
                    if csg_info["result_node"] in self.loaded_models:
                        self.loaded_models.remove(csg_info["result_node"])
                    csg_info["result_node"].removeNode()
            self.csg_results = []

        models_to_keep = []
        
        if hasattr(self, 'perlin_model') and self.perlin_model:
            models_to_keep.append(self.perlin_model)
        
        if hasattr(self, 'ground_plane') and self.ground_plane:
            models_to_keep.append(self.ground_plane)
        
        if hasattr(self, 'base_perlin_model') and self.base_perlin_model:
            models_to_keep.append(self.base_perlin_model)
        
        models_to_remove = []
        for model in self.loaded_models:
            if model not in models_to_keep:
                models_to_remove.append(model)
        
        for model in models_to_remove:
            model.removeNode()
            model_id = id(model)
            if model_id in self.model_paths:
                del self.model_paths[model_id]
        
        self.loaded_models = models_to_keep
        

    # ==================================================================
    # Model-set caching (ported from legacy gui.py)
    # ==================================================================
    def get_cache_dir(self):
        """Return the global cache dir for downloaded model sets."""
        cache_dir = os.path.join(tempfile.gettempdir(), "vizutil_models_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def download_and_cache_model_set(self, set_name: str, set_config: dict) -> dict:
        """
        Download textures + cuzov/napolnitel/other .bam files for the given
        model set into %TEMP%/vizutil_models_cache and return a dict with
        absolute local paths and (max_volume, ground_plane) metadata.
        Mirrors the legacy gui.py implementation.
        """
        if not isinstance(set_config, dict):
            raise ValueError(f"set_config for '{set_name}' is not a dict")

        cache_dir = self.get_cache_dir()
        downloaded_paths = {}

        # --- 1. Textures ------------------------------------------------
        textures_dir_rel = set_config.get("textures_dir")
        if textures_dir_rel:
            textures_basename = os.path.basename(textures_dir_rel.rstrip("/\\"))
            textures_cache_dir = os.path.join(cache_dir, textures_basename)
            os.makedirs(textures_cache_dir, exist_ok=True)
            try:
                texture_files = self.tls_client.get_texture_list(textures_dir_rel)
            except Exception as exc:
                print(f"[cache] get_texture_list({textures_dir_rel}) failed: {exc}")
                texture_files = []
            for filename in texture_files:
                local_tex_path = os.path.join(textures_cache_dir, filename)
                if not os.path.exists(local_tex_path):
                    try:
                        self.tls_client.download_texture_file(
                            textures_dir_rel, filename, local_tex_path
                        )
                    except Exception as exc:
                        print(f"[cache] download_texture_file {filename} failed: {exc}")

        # --- 2. Models (.bam) ------------------------------------------
        for ftype in ("cuzov", "napolnitel", "other"):
            if ftype not in set_config or not set_config[ftype]:
                continue
            remote_rel_path = set_config[ftype]
            filename = os.path.basename(remote_rel_path)
            local_file = os.path.join(cache_dir, filename)
            if not os.path.exists(local_file):
                try:
                    self.tls_client.download_model_file(set_name, ftype, local_file)
                except Exception as exc:
                    print(f"[cache] download_model_file {ftype} for "
                          f"'{set_name}' failed: {exc}")
                    continue
            downloaded_paths[ftype] = local_file

        downloaded_paths["max_volume"] = set_config.get("max_volume")
        downloaded_paths["ground_plane"] = set_config.get("ground_plane")
        return downloaded_paths

    def cache_and_load_model_set(self, set_name: str, set_config: dict) -> bool:
        """
        One-shot helper: download (if needed) + load the selected model set
        into the active Panda3D scene. Replaces gui.py.load_selected_model_set.
        """
        try:
            cached = self.download_and_cache_model_set(set_name, set_config)
        except Exception as exc:
            print(f"[cache] failed to cache '{set_name}': {exc}")
            return False

        model_config = {
            "cuzov":        cached.get("cuzov"),
            "napolnitel":   cached.get("napolnitel"),
            "other":        cached.get("other"),
            "max_volume":   cached.get("max_volume"),
            "ground_plane": cached.get("ground_plane"),
        }
        try:
            return bool(self.load_model_set(model_config, set_name))
        except Exception as exc:
            print(f"[cache] load_model_set('{set_name}') failed: {exc}")
            return False


def main():
    """
    Entry point: build the new MainWindow first (so its `panda_container`
    has a real HWND), then start `MyApp(ShowBase)` parented to that HWND.
    """
    # PyQt6 application + new top-level window
    from PyQt6.QtWidgets import QApplication as _QApplication
    from src.ui.main_window import MainWindow
    import src.ui.panel_data as _panel_data

    qt_app = _QApplication(sys.argv)

    win = MainWindow()
    win.show()

    # Force Qt to actually allocate the native HWND before we ask Panda
    # to embed into it.
    qt_app.processEvents()
    parent_hwnd = win.panda_container_hwnd()

    # Resolve the active TLS server from tls_config.yaml (active=True),
    # falling back to the legacy default.
    tls_host, tls_port = load_tls_config(base_path)

    # Convert Qt's LOGICAL widget size to PHYSICAL pixels so Panda's initial
    # win-size matches the native HWND at display scaling != 100% (otherwise
    # the embedded render window starts out smaller than its container).
    _dpr = win.panda_container.devicePixelRatioF()
    init_w = max(1, round(win.panda_container.width() * _dpr))
    init_h = max(1, round(win.panda_container.height() * _dpr))

    # Resolve the graphics preset (saved choice, or auto-detected on first
    # run — "performance" on integrated Intel graphics, "ultra" otherwise).
    graphics_preset = graphics_settings.resolve_startup_preset()

    panda_app = MyApp(
        parent_hwnd=parent_hwnd,
        init_size=(init_w, init_h),
        tls_host=tls_host,
        tls_port=tls_port,
        graphics_preset=graphics_preset,
    )

    # ------------------------------------------------------------------
    # Текстурные наборы теперь живут на сервере, в
    # /home/leonid/operation-3d-service/config/textures_napolnitel_config.json
    # Подтягиваем их один раз при старте и кладём в memory-кэш panel_data,
    # из которого read-only считают RightPanel и _on_texture_set_changed.
    # Сами файлы (диффуз / нормали / displacement / roughness) докачаются
    # лениво при первом выборе соответствующего набора —
    # см. ensure_texture_cached.
    # ------------------------------------------------------------------
    panda_app.texture_sets = {}
    try:
        tex_cfg = panda_app.tls_client.get_textures_config()
        if isinstance(tex_cfg, dict) and tex_cfg:
            panda_app.texture_sets = tex_cfg
            _panel_data.set_texture_sets_cache(tex_cfg)
            print(f"[main] textures config loaded from server "
                  f"({len(tex_cfg)} keys)")
        else:
            print(f"[main] empty textures config from server")
    except Exception as exc:
        print(f"[main] не удалось получить textures config с сервера: {exc}")

    # Wire telemetry, controls hint, fly-cam, right panel into the Qt window.
    win.attach_panda(panda_app)

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()
