# -*- coding: utf-8 -*-
"""
Абстрактный мир поверх RenderPipeline.

Своя сцена (`root`) рисуется своей камерой — дочерней к камере пользователя и
с ТОЙ ЖЕ линзой, поэтому оба мира всегда совпадают по ракурсу, — в HDR-буфер
с глубиной. Дальше свой bloom (dual Kawase: 5 шагов вниз и вверх) и сведение:
полноэкранный квад в окне поверх вывода RP (регион RP — sort 5, наш — 20).

Сведение делает всё остальное:
* `abstract` 1 — вместо PBR-мира пустота (фон u_bg), 0 — виден кадр RP;
* волна перехода: круг от точки экрана открывает PBR-мир с горящей кромкой;
* голограмма, стоящая за объектом PBR-мира, прячется (глубина сцены RP);
* плёночная обработка: ACES, хроматическая аберрация, зерно, виньетка,
  «глитч» строк, общее затемнение.

Параметры сведения — атрибуты (`abstract`, `bloom`, `exposure`, ...),
применяются каждый кадр. Кадр RP берётся как текстура (цель FinalStage), а не
через смешивание, поэтому итог можно одинаково вывести и в окно, и в
offscreen-буфер для записи (`capture`).
"""

from __future__ import annotations

from typing import List, Optional

from panda3d.core import (
    Camera, CardMaker, ColorBlendAttrib, FrameBufferProperties, GraphicsOutput,
    GraphicsPipe, LColor, NodePath, OrthographicLens, SamplerState, Texture,
    TransparencyAttrib, Vec2, Vec3, Vec4, WindowProperties,
)

from . import materials

#: Порядок отрисовки буферов (до окна; у окна sort 0).
_SORT_SCENE = -60
_SORT_BLOOM = -50
BLOOM_LEVELS = 5


def _fb(color_bits=16, alpha=True, depth=0):
    fb = FrameBufferProperties()
    fb.set_rgba_bits(color_bits, color_bits, color_bits, color_bits if alpha else 0)
    fb.set_float_color(color_bits >= 16)
    fb.set_depth_bits(depth)
    fb.set_float_depth(False)
    return fb


def _clamped(tex: Texture) -> Texture:
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


class _QuadPass:
    """Полноэкранный квад с шейдером, рисуемый в свой буфер."""

    def __init__(self, base, name, size, sort, shader, fb=None):
        w, h = size
        self.buffer = base.graphicsEngine.make_output(
            base.pipe, name, sort, fb or _fb(), WindowProperties.size(w, h),
            GraphicsPipe.BF_refuse_window, base.win.get_gsg(), base.win)
        if self.buffer is None:
            raise RuntimeError(f"буфер {name} не создан")
        self.tex = _clamped(Texture(name))
        self.buffer.add_render_texture(self.tex, GraphicsOutput.RTM_bind_or_copy,
                                       GraphicsOutput.RTP_color)
        self.buffer.set_clear_color_active(False)
        self.root = NodePath(name + "_root")
        self.root.set_depth_test(False)
        self.root.set_depth_write(False)
        cm = CardMaker(name + "_quad")
        cm.set_frame_fullscreen_quad()
        self.quad = self.root.attach_new_node(cm.generate())
        self.quad.set_shader(shader)
        lens = OrthographicLens()
        lens.set_film_size(2, 2)
        lens.set_near_far(-10, 10)
        cam = self.root.attach_new_node(Camera(name + "_cam", lens))
        cam.set_pos(0, -1, 0)
        dr = self.buffer.make_display_region(0, 1, 0, 1)
        dr.set_camera(cam)

    def destroy(self, base):
        if self.buffer is not None:
            self.buffer.clear_render_textures()
            base.graphicsEngine.remove_window(self.buffer)
            self.buffer = None
        self.root.remove_node()


class Compositor:
    """Абстрактный мир + его постобработка + сведение с кадром RP."""

    def __init__(self, base, render_pipeline=None):
        self.base = base
        self.rp = render_pipeline
        self.root = NodePath("vfx_root")
        self.root.set_two_sided(True)
        self.root.set_attrib(ColorBlendAttrib.make(
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_one,
            ColorBlendAttrib.O_one_minus_incoming_alpha))
        self.root.set_transparency(TransparencyAttrib.M_none)
        self.root.set_shader_input("u_time", 0.0)
        self.root.set_shader_input("u_view", Vec2(1, 1))
        self.root.set_shader_input("u_dwave", Vec3(0, 1, 0))
        self.root.set_shader_input("u_camFwd", Vec3(0, 1, 0))

        # параметры сведения
        self.abstract = 1.0
        self.wave = Vec3(0.5, 0.5, 0.0)       # центр (uv), радиус
        self.wave_width = 0.12
        self.bloom = 0.6
        self.threshold = 1.1
        self.exposure = 1.0
        self.aberration = 0.012
        self.grain = 0.03
        self.vignette = 0.35
        self.tonemap = 1.0
        #: фронт PBR-мира по глубине: (радиус м, ширина м, 1 — включён)
        self.depth_wave = Vec3(0, 1, 0)
        # слой «плоского» снимка (вступление): текстура, непрозрачность,
        # доля выпрямления линзы и параметры объектива (см. set_intro_lens)
        self.intro_tex: Optional[Texture] = None
        self.intro_mix = 0.0
        self.undistort = 0.0
        self.intro_lens = Vec4(1000, 960, 540, 1000)
        self.intro_lens_k = Vec4(0, 0, 1920, 1080)
        self.glitch = 0.0
        self.fade = 0.0
        self.bg = Vec4(0.0011, 0.0022, 0.0045, 1.5)
        self.use_world_depth = True

        self._time = 0.0
        self._size = None
        self.camera = None
        self._passes: List[_QuadPass] = []
        self._scene_buf = None
        self._composite_dr = None
        self._composite_root = None
        self._capture = None
        self.enabled = False
        # после рига кинокамеры (sort 45) и до отрисовки (igLoop, sort 50)
        self._task = base.taskMgr.add(self._update, "vfx_compositor", sort=48)

    # ------------------------------------------------------------------ #
    def _win_size(self):
        win = self.base.win
        return max(16, win.get_x_size()), max(16, win.get_y_size())

    def _build(self, size):
        self._teardown_buffers()
        base = self.base
        w, h = size
        # --- сцена абстрактного мира -----------------------------------
        buf = base.graphicsEngine.make_output(
            base.pipe, "vfx_scene", _SORT_SCENE, _fb(16, True, 24),
            WindowProperties.size(w, h), GraphicsPipe.BF_refuse_window,
            base.win.get_gsg(), base.win)
        if buf is None:
            raise RuntimeError("буфер абстрактного мира не создан")
        self.scene_tex = _clamped(Texture("vfx_scene_color"))
        self.depth_tex = Texture("vfx_scene_depth")
        self.depth_tex.set_minfilter(SamplerState.FT_nearest)
        self.depth_tex.set_magfilter(SamplerState.FT_nearest)
        buf.add_render_texture(self.scene_tex, GraphicsOutput.RTM_bind_or_copy,
                               GraphicsOutput.RTP_color)
        buf.add_render_texture(self.depth_tex, GraphicsOutput.RTM_bind_or_copy,
                               GraphicsOutput.RTP_depth)
        buf.set_clear_color_active(True)
        buf.set_clear_color(LColor(0, 0, 0, 0))
        buf.set_clear_depth_active(True)
        self.camera = self.root.attach_new_node(Camera("vfx_cam", base.camLens))
        dr = buf.make_display_region(0, 1, 0, 1)
        dr.set_camera(self.camera)
        self._scene_buf = buf

        # --- bloom: вниз ------------------------------------------------
        downs = []
        src = self.scene_tex
        for i in range(BLOOM_LEVELS):
            lw, lh = max(1, w >> (i + 1)), max(1, h >> (i + 1))
            p = _QuadPass(base, f"vfx_bloom_d{i}", (lw, lh), _SORT_BLOOM + i,
                          materials.shader("bloom_down"), _fb(16, False))
            p.quad.set_shader_input("u_src", src)
            p.quad.set_shader_input("u_threshold", self.threshold if i == 0 else 0.0)
            downs.append(p)
            src = p.tex
        # --- bloom: вверх -----------------------------------------------
        ups = []
        for i in range(BLOOM_LEVELS - 2, -1, -1):
            lw, lh = max(1, w >> (i + 1)), max(1, h >> (i + 1))
            p = _QuadPass(base, f"vfx_bloom_u{i}", (lw, lh), _SORT_BLOOM + 10 + (BLOOM_LEVELS - i),
                          materials.shader("bloom_up"), _fb(16, False))
            p.quad.set_shader_input("u_src", src)
            p.quad.set_shader_input("u_base", downs[i].tex)
            ups.append(p)
            src = p.tex
        self._passes = downs + ups
        self.bloom_tex = src

        # --- сведение: регион в окне ------------------------------------
        root = NodePath("vfx_composite_root")
        root.set_depth_test(False)
        root.set_depth_write(False)
        cm = CardMaker("vfx_composite_quad")
        cm.set_frame_fullscreen_quad()
        quad = root.attach_new_node(cm.generate())
        quad.set_shader(materials.shader("composite"))
        lens = OrthographicLens()
        lens.set_film_size(2, 2)
        lens.set_near_far(-10, 10)
        cam = root.attach_new_node(Camera("vfx_composite_cam", lens))
        cam.set_pos(0, -1, 0)
        dr = base.win.make_display_region(0, 1, 0, 1)
        dr.set_sort(20)
        dr.set_clear_color_active(False)
        dr.set_clear_depth_active(False)
        dr.set_camera(cam)
        self._composite_dr = dr
        self._composite_root = root
        self._composite_quad = quad
        self._size = size
        self.root.set_shader_input("u_view", Vec2(w, h))
        self._bind_composite_inputs(quad)
        self._scene_buf.set_active(self.enabled)
        for p in self._passes:
            p.buffer.set_active(self.enabled)
        dr.set_active(self.enabled)

    def _rp_textures(self):
        """(кадр RP, глубина RP) или (None, None) без RenderPipeline."""
        rp = self.rp
        if rp is None:
            return None, None
        try:
            final = None
            for st in rp.stage_mgr.stages:
                if type(st).__name__ == "FinalStage":
                    final = st.target.color_tex
            depth = rp.stage_mgr.pipes.get("SceneDepth")
            return final, depth
        except Exception:
            return None, None

    def _bind_composite_inputs(self, quad):
        final, depth = self._rp_textures()
        blank = getattr(self, "_blank", None)
        if blank is None:
            blank = self._blank = Texture("vfx_blank")
            blank.setup_2d_texture(1, 1, Texture.T_float, Texture.F_r32)
            blank.set_ram_image(b"\x00\x00\x80\x3f")          # 1.0
        quad.set_shader_input("u_scene", self.scene_tex)
        quad.set_shader_input("u_depth", self.depth_tex)
        quad.set_shader_input("u_bloom", self.bloom_tex)
        quad.set_shader_input("u_world", final if final is not None else blank)
        quad.set_shader_input("u_worldDepth", depth if depth is not None else blank)
        self._world_bound = (final, depth)

    def _teardown_buffers(self):
        base = self.base
        for p in self._passes:
            p.destroy(base)
        self._passes = []
        if self._scene_buf is not None:
            self._scene_buf.clear_render_textures()
            base.graphicsEngine.remove_window(self._scene_buf)
            self._scene_buf = None
        if getattr(self, "camera", None) is not None:
            self.camera.remove_node()
            self.camera = None
        if self._composite_dr is not None:
            base.win.remove_display_region(self._composite_dr)
            self._composite_dr = None
        if self._composite_root is not None:
            self._composite_root.remove_node()
            self._composite_root = None

    # ------------------------------------------------------------------ #
    def set_enabled(self, on: bool) -> None:
        """Включить абстрактный мир (буферы рисуются только пока включён)."""
        self.enabled = bool(on)
        if on and self._size is None:
            self._build(self._win_size())
        if self._scene_buf is not None:
            self._scene_buf.set_active(self.enabled)
            for p in self._passes:
                p.buffer.set_active(self.enabled)
        if self._composite_dr is not None:
            self._composite_dr.set_active(self.enabled)

    def _update(self, task):
        from direct.task.Task import Task
        if not self.enabled:
            return Task.cont
        size = self._win_size()
        if size != self._size:
            self._build(size)
        # камера абстрактного мира повторяет камеру пользователя: корень
        # абстрактного мира совпадает с render
        self.camera.set_mat(self.base.camera.get_mat(self.base.render))
        final, depth = self._rp_textures()
        if (final, depth) != getattr(self, "_world_bound", (None, None)):
            self._bind_composite_inputs(self._composite_quad)
        q = self._composite_quad
        q.set_shader_input("u_useWorldDepth",
                           1.0 if (self.use_world_depth and depth is not None) else 0.0)
        q.set_shader_input("u_abstract", float(self.abstract))
        q.set_shader_input("u_wave", self.wave)
        q.set_shader_input("u_waveWidth", float(self.wave_width))
        q.set_shader_input("u_bloomK", float(self.bloom))
        q.set_shader_input("u_exposure", float(self.exposure))
        q.set_shader_input("u_aberration", float(self.aberration))
        q.set_shader_input("u_grain", float(self.grain))
        q.set_shader_input("u_vignette", float(self.vignette))
        q.set_shader_input("u_glitch", float(self.glitch))
        q.set_shader_input("u_fade", float(self.fade))
        q.set_shader_input("u_time", float(self._time))
        q.set_shader_input("u_bg", self.bg)
        q.set_shader_input("u_tonemap", float(self.tonemap))
        q.set_shader_input("u_dwave", self.depth_wave)
        # тот же фронт — материалам сцены (голограмма кузова срезается им)
        self.root.set_shader_input("u_dwave", self.depth_wave)
        self.root.set_shader_input("u_camFwd", self.base.render.get_relative_vector(
            self.base.camera, Vec3(0, 1, 0)))
        lens = self.base.camLens
        q.set_shader_input("u_nearFar", Vec2(lens.get_near(), lens.get_far()))
        q.set_shader_input("u_intro", self.intro_tex if self.intro_tex is not None else self._blank)
        q.set_shader_input("u_introMix", float(self.intro_mix if self.intro_tex is not None else 0.0))
        q.set_shader_input("u_undistort", float(self.undistort))
        q.set_shader_input("u_lens", self.intro_lens)
        q.set_shader_input("u_lensK", self.intro_lens_k)
        q.set_shader_input("u_view", Vec2(*self._size))
        self._passes[0].quad.set_shader_input("u_threshold", float(self.threshold))
        if self._capture is not None:
            self._capture.quad.set_state(q.get_state())
            self._poll_capture()
        return Task.cont

    def set_intro_lens(self, f, cx, cy, f_view, k1, k2, width, height) -> None:
        """Объектив снимка (пиксели) и фокус 3D-камеры для морфа дисторсии."""
        self.intro_lens = Vec4(f, cx, cy, f_view)
        self.intro_lens_k = Vec4(k1, k2, width, height)

    def warmup(self, frames: int = 3) -> None:
        """
        Скомпилировать шейдеры и выделить буферы заранее, незаметно.

        Первое включение иначе стоит ~1 с (компиляция GLSL при первой
        отрисовке, генерация глифов шрифта). Здесь сцена включается на пару
        кадров с полностью прозрачными пустышками всех материалов, а сведение
        при abstract = 0 и без плёночных эффектов отдаёт кадр RP как есть.
        Кадры не крутятся здесь — пустышки убирает задача через `frames` кадров.
        """
        if self.enabled or getattr(self, "_warming", False):
            return
        import numpy as np
        from . import primitives as P
        self._warming = True
        saved = (self.abstract, self.bloom, self.vignette, self.grain, self.aberration)
        self.abstract = self.bloom = self.vignette = self.grain = self.aberration = 0.0
        self.set_enabled(True)
        holder = self.camera.attach_new_node("warmup")
        holder.set_pos(0, 3, 0)
        tri_v = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0, 0.01]], float)
        tri_f = np.array([[0, 1, 2]])
        hf = P.heightfield((2, 2), (0, 0), 0.01)
        ht = P.height_texture(np.zeros((2, 2)))
        hf.set_shader_input("u_hA", ht)
        hf.set_shader_input("u_hB", ht)
        hf.set_shader_input("u_color", Vec4(0, 0, 0, 0))
        cloud = P.point_cloud(tri_v, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        cloud.set_shader_input("u_alphas", Vec4(0, 0, 0, 0))
        nodes = [P.holo_mesh(tri_v, tri_f, color=(0, 0, 0, 0)), cloud, hf,
                 P.lines([tri_v], [(0, 0, 0, 0)]),
                 P.sparks(tri_v, tri_v, np.full(3, 99.0), np.ones(3), np.zeros(3)),
                 P.grid_floor((0, 0), size=0.01),
                 P.photo_surface(tri_v, tri_v[:, :2], tri_f, self._blank)]
        for n in nodes:
            n.reparent_to(holder)
            n.set_shader_input("u_alpha", 0.0)
        # все глифы, что встречаются в подписях
        P.HoloText(holder, "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
                   "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                   "0123456789.,:;·×%³()-–—/…", scale=0.001, color=(0, 0, 0, 0))
        left = [frames]

        def done(task):
            left[0] -= 1
            if left[0] > 0:
                return task.cont
            holder.remove_node()
            (self.abstract, self.bloom, self.vignette, self.grain,
             self.aberration) = saved
            self._warming = False
            if not getattr(self, "_claimed", False):
                self.set_enabled(False)
            return task.done
        self.base.taskMgr.add(done, "vfx_warmup", sort=49)

    @classmethod
    def shared(cls, base, render_pipeline=None) -> "Compositor":
        """Один компоновщик на приложение: буферы и шейдеры переживают сцены."""
        comp = getattr(base, "_vfx_compositor", None)
        if comp is None:
            comp = cls(base, render_pipeline)
            base._vfx_compositor = comp
        return comp

    def reset(self) -> None:
        """Убрать содержимое сцены и вернуть параметры сведения."""
        for child in self.root.get_children():
            if self.camera is not None and child == self.camera:
                continue
            child.remove_node()
        for ch in (self.camera.get_children() if self.camera is not None else []):
            ch.remove_node()
        self.abstract, self.wave, self.bloom = 1.0, Vec3(0.5, 0.5, 0.0), 0.6
        self.exposure, self.tonemap, self.glitch, self.fade = 1.0, 1.0, 0.0, 0.0
        self.aberration, self.grain, self.vignette = 0.012, 0.03, 0.35
        self.intro_tex, self.intro_mix, self.undistort = None, 0.0, 0.0
        self.depth_wave = Vec3(0, 1, 0)

    def set_time(self, t: float) -> None:
        self._time = t
        self.root.set_shader_input("u_time", float(t))

    # ------------------------------------------------------------------ #
    def project(self, world_point) -> Optional[Vec2]:
        """Точка мира абстрактной сцены -> uv экрана (0..1) или None."""
        from panda3d.core import Point2, Point3
        cam = self.base.cam
        p = cam.get_relative_point(self.root, Point3(*world_point))
        out = Point2()
        if self.base.camLens.project(p, out):
            return Vec2(out.x * 0.5 + 0.5, out.y * 0.5 + 0.5)
        return None

    def capture(self, path: str) -> None:
        """
        Записать итоговый кадр (тот же шейдер сведения) в файл. Копия
        заказывается сейчас, файл пишется, когда она придёт (1–2 кадра).
        Синхронная копия ждёт GPU — только для отладки и проверок.
        """
        base = self.base
        if self._size is None:
            return
        if self._capture is None or self._capture_size != self._size:
            if self._capture is not None:
                self._capture.destroy(base)
            p = _QuadPass(base, "vfx_capture", self._size, 100,
                          materials.shader("composite"), _fb(8, True))
            self._capture = p
            self._capture_size = self._size
            self._capture_tex = Texture("vfx_capture_ram")
            p.buffer.add_render_texture(self._capture_tex,
                                        GraphicsOutput.RTM_triggered_copy_ram,
                                        GraphicsOutput.RTP_color)
        p = self._capture
        p.quad.set_state(self._composite_quad.get_state())
        self._capture_tex.clear_ram_image()
        p.buffer.trigger_copy()
        self._capture_path = path

    def _poll_capture(self) -> None:
        path = getattr(self, "_capture_path", None)
        if not path or not self._capture_tex.has_ram_image():
            return
        tex = self._capture_tex
        w, h = tex.get_x_size(), tex.get_y_size()
        raw = bytes(tex.get_ram_image_as("RGB"))
        self._capture_path = None

        # PNG кодируется секунды долю — не в кадре, а в фоне
        def write():
            try:
                import numpy as np
                from PIL import Image
                img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)[::-1]
                Image.fromarray(img).save(path)
            except Exception as exc:
                print(f"[vfx] кадр не записан: {path}: {exc}")
        import threading
        threading.Thread(target=write, daemon=True).start()

    @property
    def capture_pending(self) -> bool:
        return bool(getattr(self, "_capture_path", None))

    def destroy(self) -> None:
        if self._task is not None:
            self.base.taskMgr.remove(self._task)
            self._task = None
        if self._capture is not None:
            self._capture.destroy(self.base)
            self._capture = None
        self._teardown_buffers()
        self.root.remove_node()
