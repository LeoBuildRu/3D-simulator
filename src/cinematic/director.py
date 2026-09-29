# -*- coding: utf-8 -*-
"""
Сценарий кинематографичной реконструкции наполнения по проезду.

Этапы (каждый — корутина; нет данных — этап пропускается, а не падает):

  0  вступление: пустота, заголовок проезда, снимок станции на весь экран,
     пока идёт анализ — сканирующая полоса и ход работы;
  1  снимок -> 3D: линза «выпрямляется», кадр подменяется мешем фото-глубины,
     камера отъезжает и показывает его сбоку;
  2  лидар: сенсор на мачте «кидает» точки развёрткой по реальному облаку;
  3  фото-меш растворяется, остаётся облако;
  4  поиск кузова: след, стенки, выборки кромки (годные/выбросы), кромки и
     торцы, углы (volume_calculator/body_geometry, реальные шаги);
  5  фон сдувается, остаются точки машины;
  6  четыре опорные точки одна за другой, размеры и заполнение текстом;
  7  подбор кузова: два случайных, потом нужный — он совмещается опорными
     точками с облаком;
  8  кузов прячется;
  9  карты высот этапов серверного алгоритма: сырая, фильтр выбросов,
     заполнение пропусков, сглаживание, срез бортов, экстраполяция;
  10 булева разность с наполнителем (каркас, срез с искрами), итоговый меш;
  11 кузов возвращается;
  12 волна: абстрактный мир уступает PBR-миру;
  13 наполнитель по фронту меняется с голограммы на PBR, летят искры;
  14 финальный облёт, камера возвращается пользователю.

Мировая система — render (= система модели кузова). Облако и снимок живут
в системе лидара; сервер переводит её в систему модели зеркальной матрицей T
(det = -1). Снимок в зеркальной системе выглядел бы зеркально, поэтому первые
этапы идут в W = Fx·T (отражение поперёк оси кузова — сам кузов симметричен и
от этого не меняется), а на этапе 5, пока фон сдувается, точки машины
перетекают из W в T.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np
from panda3d.core import (
    Filename, LMatrix4f, NodePath, Point3, Vec2, Vec3, Vec4,
)

from src.vfx import easing as E
from src.vfx import primitives as P
from src.vfx.camera import Shot
from src.vfx.sequencer import call, tween, until, wait

CYAN = (0.35, 0.85, 1.0, 1.0)
ORANGE = (1.0, 0.55, 0.18, 1.0)
GREEN = (0.35, 1.0, 0.6, 1.0)
RED = (1.0, 0.25, 0.25, 1.0)
MAGENTA = (0.9, 0.35, 1.0, 1.0)


def mat4(a: np.ndarray) -> LMatrix4f:
    """4x4 (столбцовая конвенция, p' = A p) -> матрица Panda (строковая)."""
    return LMatrix4f(*np.asarray(a, float).T.ravel())


def circle(center, radius, n=48, axis="z"):
    t = np.linspace(0, 2 * np.pi, n + 1)
    c = np.asarray(center, float)
    return np.stack([c[0] + radius * np.cos(t), c[1] + radius * np.sin(t),
                     np.full_like(t, c[2])], 1)


class Director:
    def __init__(self, session):
        self.s = session
        self.app = session.app
        self.comp = session.comp
        self.seq = session.seq
        self.rig = session.rig
        self.d = session.data
        self.root = self.comp.root
        self.nodes: dict = {}
        self.hud_items: list = []
        self.hud = None
        self.center = Vec3(0, 0, 3)
        self.floor_z = 0.0
        #: имя текущего этапа — для журнала и диагностики просадок
        self.stage = "init"

    # ================================================================== #
    # Вспомогательное
    # ================================================================== #
    def burst(self, pos, n=140, speed=1.6, life=1.1, hue=None, up=1.0, spread=1.0,
              gravity=-2.5, size=0.02, parent=None):
        """Искры из точки(ек); сами исчезают."""
        pos = np.asarray(pos, float).reshape(-1, 3)
        rng = np.random.default_rng()
        k = max(1, n // len(pos))
        origins = np.repeat(pos, k, axis=0)
        m = len(origins)
        d = rng.normal(size=(m, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d[:, 2] = np.abs(d[:, 2]) * up + (1 - up) * d[:, 2]
        vel = d * speed * spread * rng.uniform(0.3, 1.0, (m, 1))
        hues = rng.random(m) if hue is None else np.full(m, hue) + rng.normal(0, 0.04, m)
        node = P.sparks(origins, vel, rng.uniform(0, 0.12, m), rng.uniform(0.5, 1.0, m) * life,
                        hues)
        node.set_shader_input("u_gravity", Vec3(0, 0, gravity))
        node.set_shader_input("u_size", size)
        node.reparent_to(parent or self.root)
        node.set_bin("fixed", 40)

        def life_track():
            yield tween(life * 1.2 + 0.15, lambda t: node.set_shader_input("u_time", t),
                        0.0, life * 1.2 + 0.15, E.linear)
            node.remove_node()
        self.seq.spawn(life_track())
        return node

    def label(self, text, pos, scale=0.28, color=CYAN, parent=None, align="center"):
        t = P.HoloText(parent or self.root, text, scale=scale, color=color, align=align)
        t.root.set_pos(Vec3(*pos))
        t.set_reveal(0.0)
        return t

    def type_in(self, text_obj, duration=0.7):
        return tween(duration, text_obj.set_reveal, 0.0, 1.0, E.out_cubic)

    def fade_node(self, node, name, a, b, duration, curve=E.in_out_cubic, scale=1.0):
        """Твин униформы «альфы» узла: u_alpha — float, u_color — Vec4.a."""
        def setter(v):
            if name == "u_color":
                c = Vec4(node.get_shader_input("u_color").get_vector())
                c.w = v
                node.set_shader_input("u_color", c)
            else:
                node.set_shader_input(name, v * scale)
        return tween(duration, setter, a, b, curve)

    # --- HUD: подписи, приклеенные к экрану ------------------------------ #
    def make_hud(self):
        self.hud = self.comp.camera.attach_new_node("hud")
        self.hud.set_depth_test(False)
        self.hud.set_depth_write(False)
        self.hud.set_bin("fixed", 90)
        self.seq.every_frame(self._update_hud)

    def _update_hud(self, t, dt):
        if self.hud is None:
            return
        fov = self.app.camLens.get_fov()
        kx = math.tan(math.radians(fov[0]) / 2)
        ky = math.tan(math.radians(fov[1]) / 2)
        for item, (u, v, s) in self.hud_items:
            if item.root.is_empty():
                continue
            item.root.set_pos(u * kx, 1.0, v * ky)
            item.np.set_scale(s * ky)

    def hud_text(self, text, u, v, scale=0.05, color=CYAN, align="center"):
        """Текст в точке экрана (u, v от -1 до 1), размер — доля высоты кадра."""
        t = P.HoloText(self.hud, text, scale=scale, color=color, align=align)
        t.root.clear_billboard()
        self.hud_items.append((t, (u, v, scale)))
        t.set_reveal(0.0)
        return t

    # ================================================================== #
    # Системы координат
    # ================================================================== #
    def setup_frames(self):
        sc = self.d.scene
        T = sc.get("T") if sc is not None else None
        if T is None:
            T = np.eye(4)
        self.T = np.asarray(T, float)
        Fx = np.diag([-1.0, 1, 1, 1])
        self.A = Fx @ self.T
        self.sensorW = self.root.attach_new_node("sensorW")
        self.sensorW.set_mat(mat4(self.A))
        self.sensorM = self.root.attach_new_node("sensorM")
        self.sensorM.set_mat(mat4(self.T))
        p3 = sc.get("points_3d") if sc is not None else None
        if p3 is not None:
            p3 = np.asarray(p3, float)
            c = p3.mean(0)
            self.center = Vec3(c[0], c[1], c[2])
            self.rim_z = float(c[2])
        else:
            self.rim_z = 3.0
        self.body_len = float((self.d.json.get("body_dimensions") or {}).get("length") or 8.0)
        self.body_w = float((self.d.json.get("body_dimensions") or {}).get("width") or 2.3)
        origin = self.A @ np.array([0, 0, 0, 1.0])
        self.sensor_origin = Vec3(*origin[:3])

    def to_root(self, p_sensor, frame="W"):
        A = self.A if frame == "W" else self.T
        p = np.asarray(p_sensor, float)
        return p @ A[:3, :3].T + A[:3, 3]

    def side_shot(self, angle=-58.0, radius=None, height=None, fov=42.0, look=None):
        r = radius or max(11.0, self.body_len * 1.55)
        h = height if height is not None else r * 0.55
        a = math.radians(angle)
        c = look or self.center
        pos = Vec3(c.x + r * math.cos(a), c.y + r * math.sin(a), c.z + h)
        return Shot(pos, c, (0, 0, 1), fov)

    # ================================================================== #
    # Построение узлов из данных
    # ================================================================== #
    def build_scene_nodes(self):
        sc = self.d.scene
        n = self.nodes
        if sc is None:
            return
        if sc.has("points"):
            pts = sc.get("points")
            lab = sc.get("labels").astype(np.float32)
            hag = sc.get("height_above_ground")
            cl = P.point_cloud(pts, sc.get("intensity"), sc.get("sweep"), lab, hag, "cloudW")
            cl.reparent_to(self.sensorW)
            cl.set_shader_input("u_origin", Vec3(0, 0, 0))
            cl.set_shader_input("u_throw", 0.0)
            cl.set_shader_input("u_hrange", Vec2(0.0, max(0.5, self.rim_z + 0.6)))
            cl.set_bin("fixed", 20)
            n["cloudW"] = cl
            truck = lab > 0.5
            cm = P.point_cloud(pts[truck], sc.get("intensity")[truck], sc.get("sweep")[truck],
                               lab[truck], hag[truck], "cloudM")
            cm.reparent_to(self.sensorM)
            cm.set_shader_input("u_throw", 2.0)
            cm.set_shader_input("u_alphas", Vec4(1, 1, 0, 0))
            cm.set_shader_input("u_hrange", Vec2(0.0, max(0.5, self.rim_z + 0.6)))
            cm.set_bin("fixed", 20)
            n["cloudM"] = cm
        if sc.has("photo_vertices") and self.d.photo_path:
            tex = self.app.loader.loadTexture(Filename.from_os_specific(self.d.photo_path))
            n["photo_tex"] = tex
            ph = P.photo_surface(sc.get("photo_vertices"), sc.get("photo_uv"),
                                 sc.get("photo_faces"), tex, "photo")
            ph.reparent_to(self.sensorW)
            ph.set_shader_input("u_hrange", Vec2(0.0, 4.0))
            ph.set_bin("fixed", 10)
            ph.hide()
            n["photo"] = ph
            cf = sc.get("photo_curtain_faces")
            if cf is not None and len(cf):
                cu = P.photo_surface(sc.get("photo_vertices"), sc.get("photo_uv"), cf, tex,
                                     "curtain")
                cu.reparent_to(self.sensorW)
                cu.set_bin("fixed", 9)
                cu.hide()
                n["curtain"] = cu

    def build_detection(self):
        sc = self.d.scene
        if sc is None or not sc.has("det_corners"):
            return None
        g = NodePath("detection")
        g.reparent_to(self.sensorW)
        items = {}
        if sc.has("det_footprint"):
            f = sc.get("det_footprint").astype(float)
            items["footprint"] = P.lines([np.r_[f, f[:1]]], [(0.3, 0.6, 1.0, 1.0)], 2.0, "fp")
            items["footprint"].set_shader_input("u_dash", 0.35)
        if sc.has("det_wall_points"):
            wp = sc.get("det_wall_points")
            wid = sc.get("det_wall_ids").astype(float)
            cols = np.array([[1.0, 0.3, 1.0], [0.7, 0.35, 1.0], [1.0, 0.45, 0.8],
                             [0.8, 0.3, 1.0], [0.9, 0.5, 1.0], [0.75, 0.25, 0.95]])
            w = P.point_cloud(wp, np.ones(len(wp)), np.zeros(len(wp)), np.ones(len(wp)),
                              np.zeros(len(wp)), "walls", colors=cols[wid.astype(int) % 6] * 3.0)
            w.set_shader_input("u_throw", 2.0)
            w.set_shader_input("u_size", 0.06)
            w.set_shader_input("u_maxPx", 10.0)
            items["walls"] = w
        if sc.has("det_rim_samples"):
            rs = sc.get("det_rim_samples").astype(float)
            ok = sc.get("det_rim_inliers").astype(bool)
            cols = np.where(ok[:, None], np.array([[0.4, 1.4, 1.0]]), np.array([[1.6, 0.25, 0.2]]))
            r = P.point_cloud(rs, np.ones(len(rs)), np.linspace(0, 1, len(rs)), np.ones(len(rs)),
                              np.zeros(len(rs)), "rim", colors=cols)
            r.set_shader_input("u_size", 0.1)
            r.set_shader_input("u_maxPx", 16.0)
            r.set_shader_input("u_fly", 0.3)
            r.set_shader_input("u_throw", 0.0)
            items["rim"] = r
        c = sc.get("det_corners").astype(float)
        items["rims"] = P.lines([c[[0, 1]], c[[3, 2]]], [GREEN, GREEN], 4.0, "rims")
        items["ends"] = P.lines([c[[1, 2]], c[[0, 3]]], [ORANGE, ORANGE], 4.0, "ends")
        for k, v in items.items():
            v.reparent_to(g)
            v.set_bin("fixed", 30)
            v.hide()
        self.nodes["det"] = items
        self.nodes["det_root"] = g
        return items

    def build_marker(self, pos_root, color=ORANGE, parent=None):
        """Опорная точка: кольца, штырь и яркое ядро."""
        m = (parent or self.root).attach_new_node("marker")
        m.set_pos(Vec3(*pos_root))
        rings = P.lines([circle((0, 0, 0), 0.22, 40), circle((0, 0, 0), 0.34, 48)],
                        [color, color], 3.0, "rings")
        rings.reparent_to(m)
        stem = P.lines([np.array([[0, 0, 0], [0, 0, 0.9]])], [color], 2.5, "stem")
        stem.reparent_to(m)
        core = P.point_cloud(np.zeros((1, 3)), np.ones(1), np.zeros(1), np.ones(1),
                             np.zeros(1), "core", colors=np.array([color[:3]]) * 4.0)
        core.set_shader_input("u_throw", 2.0)
        core.set_shader_input("u_size", 0.09)
        core.set_shader_input("u_maxPx", 18.0)
        core.reparent_to(m)
        m.set_bin("fixed", 50)
        m.set_scale(0.001)
        return m

    def build_heightfield(self):
        fa = self.d.fill
        if fa is None or not fa.has("h_raw", "grid_origin"):
            return None
        stages = [("h_raw", "СЫРАЯ КАРТА ВЫСОТ"), ("h_filtered", "ФИЛЬТР ВЫБРОСОВ И БОРТОВ"),
                  ("h_filled", "ЗАПОЛНЕНИЕ ПРОПУСКОВ"), ("h_smooth", "СГЛАЖИВАНИЕ"),
                  ("h_walls", "СРЕЗ СТЕНОК"), ("h_final", "ЭКСТРАПОЛЯЦИЯ")]
        texes = [(P.height_texture(fa.get(k)), name) for k, name in stages if fa.has(k)]
        if len(texes) < 2:
            return None
        hf = P.heightfield(fa.get("grid_shape"), fa.get("grid_origin"), float(fa.get("grid_res")))
        vals = np.concatenate([fa.get(k)[np.isfinite(fa.get(k))].ravel()
                               for k, _ in stages if fa.has(k)])
        lo, hi = np.percentile(vals, [2, 98])
        hf.set_shader_input("u_hrange", Vec2(lo - 0.05, hi + 0.05))
        hf.set_shader_input("u_hA", texes[0][0])
        hf.set_shader_input("u_hB", texes[0][0])
        hf.set_shader_input("u_floor", float(np.nanmin(vals)) - 0.4)
        nv = fa.get("napolnitel_vertices")
        if nv is not None:
            hf.set_shader_input("u_box", Vec4(nv[:, 0].min(), nv[:, 0].max(),
                                              nv[:, 1].min(), nv[:, 1].max()))
        holder = self.root.attach_new_node("fill_space")
        if self.d.direction_yaw:
            holder.set_h(180)          # как сервер развернул результат (Entry)
        hf.reparent_to(holder)
        hf.set_bin("fixed", 25)
        hf.hide()
        self.nodes["fill_space"] = holder
        self.nodes["hf"] = hf
        self.nodes["hf_stages"] = texes
        self.hrange = (lo, hi)
        return hf

    def build_result_holo(self):
        f = self.d.fetched or {}
        mesh = f.get("mesh")
        if mesh is None:
            return None
        from src.rendering import mesh_io
        node = NodePath(mesh_io.geom_node_from_arrays("result_holo", mesh.interleaved,
                                                      mesh.faces))
        from src.vfx import materials
        node.set_shader(materials.shader("holo"))
        P.holo_defaults(node, (0.35, 0.85, 1.0, 0.0))
        lo, hi = getattr(self, "hrange", (float(mesh.vertices[:, 2].min()),
                                          float(mesh.vertices[:, 2].max())))
        node.set_shader_input("u_hrange", Vec2(lo - 0.05, hi + 0.05))
        node.set_shader_input("u_rainbow", 1.0)
        node.set_shader_input("u_style", Vec4(0.8, 0.5, 0.45, 0.05))
        node.reparent_to(self.root)
        node.set_bin("fixed", 26)
        node.hide()
        self.nodes["result_holo"] = node
        return node

    # ================================================================== #
    # Сцена целиком
    # ================================================================== #
    def main(self):
        d = self.d
        comp = self.comp
        comp.abstract = 0.0
        comp.bloom = 0.0
        comp.tonemap = 0.0
        comp.vignette = 0.0
        self.make_hud()
        # из PBR-мира — в пустоту
        yield tween(0.8, lambda v: setattr(comp, "abstract", v), 0.0, 1.0, E.in_out_cubic)
        car = d.rec.car_number if getattr(d.rec, "car_number", "") else ""
        title = self.hud_text(f"ПРОЕЗД  {car}".strip(), 0.0, 0.14, 0.13)
        sub = self.hud_text(f"{d.rec.model}   ·   {d.rec.time}", 0.0, -0.04, 0.065,
                            (0.6, 0.8, 1.0, 0.8))
        yield self.type_in(title, 0.9)
        yield self.type_in(sub, 0.6)
        status = self.hud_text("", 0.0, -0.88, 0.055, (0.5, 0.8, 1.0, 0.8))
        status_track = self.seq.spawn(self._status_loop(status))

        yield until(lambda: d.photo_ready or bool(d.error), timeout=30)
        yield tween(0.5, lambda v: (title.set_alpha(1 - v), sub.set_alpha(1 - v)), 0, 1)
        title.destroy()
        sub.destroy()

        have_photo = bool(d.photo_path)
        if have_photo:
            tex = self.app.loader.loadTexture(Filename.from_os_specific(d.photo_path))
            comp.intro_tex = tex
            comp.undistort = 0.0
            yield tween(0.8, lambda v: setattr(comp, "intro_mix", v), 0.0, 1.0, E.out_cubic)
        # анализ идёт — ждём (на экране снимок и ход работы)
        yield until(lambda: d.scene_ready or bool(d.error), timeout=180)
        self.setup_frames()
        self.build_scene_nodes()
        status_track.cancel()
        yield tween(0.4, lambda v: status.set_alpha(1 - v), 0, 1)
        status.destroy()
        for name, stage in (("photo", self.stage_photo_to_3d), ("lidar", self.stage_lidar),
                            ("detection", self.stage_detection),
                            ("background", self.stage_background),
                            ("keypoints", self.stage_keypoints), ("bodies", self.stage_bodies),
                            ("fill", self.stage_fill), ("boolean", self.stage_boolean),
                            ("pbr", self.stage_pbr), ("finale", self.stage_finale)):
            self.stage = name
            if name == "lidar":
                self.rig.shake = 0.035       # лёгкое «дыхание» ручной камеры
            yield from stage()

    def _status_loop(self, label):
        last = None
        while True:
            txt = self.d.status or ""
            if txt != last:
                last = txt
                label.set_text(txt.upper())
                label.set_reveal(1.0)
            yield wait(0.1)

    # ------------------------------------------------------------------ #
    def stage_photo_to_3d(self):
        comp, rig, n = self.comp, self.rig, self.nodes
        sc = self.d.scene
        if "photo" not in n or sc is None or not sc.has("photo_camera"):
            # нет снимка: сразу в пустоту к облаку
            yield tween(0.6, lambda v: setattr(comp, "intro_mix", v), comp.intro_mix, 0.0)
            yield from self._abstract_look(1.2)
            rig.cut(self.side_shot())
            return
        import cv2
        cam = sc.get("photo_camera")
        size = sc.get("photo_size")
        R, _ = cv2.Rodrigues(cam[:3])
        centre_s = -R.T @ cam[3:6]
        L = self.A[:3, :3]
        centre = L @ centre_s + self.A[:3, 3]
        R_wc = L @ R.T
        f_view = float(cam[6]) * 0.78
        rig.photo_lens(float(cam[6]), float(cam[9]), float(cam[10]), int(size[0]), int(size[1]))
        comp.set_intro_lens(cam[6], cam[9], cam[10], f_view, cam[7], cam[8],
                            int(size[0]), int(size[1]))
        station = rig.photo_shot(R_wc, centre, f_view=f_view)
        rig.cut(station)
        # линза «выпрямляется»: снимок как снят -> как видит обскура
        yield tween(1.6, lambda v: setattr(comp, "undistort", v), 0.0, 1.0, E.in_out_cubic)
        n["photo"].show()
        if "curtain" in n:
            n["curtain"].show()
        yield wait(0.1)
        yield tween(0.35, lambda v: setattr(comp, "intro_mix", v), 1.0, 0.0, E.linear)
        comp.intro_tex = None
        # отъезд: тонмаппинг, свечение и пол проявляются по ходу
        floor = P.grid_floor((self.center.x, self.center.y), z=self.floor_z)
        floor.reparent_to(self.root)
        floor.set_bin("fixed", 0)
        floor.set_shader_input("u_alpha", 0.0)
        n["floor"] = floor
        rise = Shot(station.pos + Vec3(0, 0, 3.0), self.center + Vec3(0, 0, -0.5), (0, 0, 1), 55)
        side = self.side_shot(-62.0)
        look = self.seq.spawn(self._grade_to_holo(3.5))
        yield rig.move([rise, side], 5.0, E.in_out_quint)
        yield look

    def _grade_to_holo(self, dur):
        comp, n = self.comp, self.nodes

        def grade(v):
            comp.tonemap = v
            comp.bloom = v
            comp.vignette = 0.35 * v
            if "floor" in n:
                n["floor"].set_shader_input("u_alpha", v)
            if "curtain" in n:
                n["curtain"].set_shader_input("u_alpha", max(0.0, 1.0 - v * 4))
            n["photo"].set_shader_input("u_holo", 0.18 * v)
        yield tween(dur, grade, 0.0, 1.0, E.in_out_cubic)
        if "curtain" in n:
            n["curtain"].hide()

    def _abstract_look(self, dur):
        comp = self.comp

        def grade(v):
            comp.tonemap = v
            comp.bloom = v
            comp.vignette = 0.35 * v
        yield tween(dur, grade, 0.0, 1.0)
        floor = P.grid_floor((self.center.x, self.center.y), z=self.floor_z)
        floor.reparent_to(self.root)
        floor.set_bin("fixed", 0)
        self.nodes["floor"] = floor

    # ------------------------------------------------------------------ #
    def stage_lidar(self):
        n, rig = self.nodes, self.rig
        cl = n.get("cloudW")
        if cl is None:
            return
        # сенсор на мачте
        glyph = self.root.attach_new_node("lidar")
        glyph.set_pos(self.sensor_origin)
        rings = P.lines([circle((0, 0, 0), 0.25, 40), circle((0, 0, 0), 0.45, 48),
                         np.array([[-0.6, 0, 0], [0.6, 0, 0]])],
                        [CYAN, CYAN, (0.6, 0.9, 1.0, 1.0)], 2.5, "lidar_rings")
        rings.reparent_to(glyph)
        glyph.set_scale(0.001)
        glyph.set_bin("fixed", 45)
        n["lidar"] = glyph

        def spin():
            while not glyph.is_empty():
                glyph.set_h(glyph.get_h() + 7.0)
                yield None
        self.seq.spawn(spin())
        yield tween(0.6, lambda v: glyph.set_scale(max(v, 0.001)), 0.0, 1.0, E.out_back)
        self.burst([tuple(self.sensor_origin)], 80, 1.2, 0.9, hue=0.55)
        orbit = self.seq.spawn(rig.orbit(self.center, max(11.0, self.body_len * 1.5),
                                         7.5, -62, -20, 7.5, E.in_out_sine))
        cl.set_shader_input("u_alphas", Vec4(1, 1, 1, 1))
        yield tween(6.0, lambda v: cl.set_shader_input("u_throw", v), 0.0, 1.02, E.linear)
        yield orbit
        # фото-меш растворяется
        if "photo" in n:
            yield tween(1.6, lambda v: n["photo"].set_shader_input("u_dissolve", v),
                        0.0, 1.0, E.in_cubic)
            n["photo"].hide()
        yield tween(0.5, lambda v: glyph.set_scale(max(1 - v, 0.001)), 0.0, 1.0, E.in_cubic)
        glyph.remove_node()

    # ------------------------------------------------------------------ #
    def stage_detection(self):
        items = self.build_detection()
        if not items:
            return
        rig = self.rig
        c = self.to_root(self.d.scene.get("det_corners")).mean(0)
        look = Vec3(*c)
        yield rig.move(self.side_shot(-38.0, radius=self.body_len * 0.95, height=self.body_len * 0.8,
                                      look=look, fov=46), 2.2, E.in_out_cubic)
        cap = self.hud_text("ПОИСК КУЗОВА", -0.92, 0.82, 0.085, CYAN, "left")
        cl = self.nodes.get("cloudW")
        dim = (lambda v: cl.set_shader_input("u_alphas", Vec4(1 - 0.6 * v, 1 - 0.6 * v, 1, 1)))             if cl is not None else (lambda v: None)
        yield [self.type_in(cap, 0.5), tween(0.8, dim, 0.0, 1.0)]

        def show_lines(key, dur, width=None):
            node = items.get(key)
            if node is None:
                return wait(0)
            node.show()
            node.set_shader_input("u_draw", 0.0)
            return tween(dur, lambda v: node.set_shader_input("u_draw", v), 0.0, 1.0, E.out_cubic)

        step = self.hud_text("след машины", -0.92, 0.7, 0.058, (0.6, 0.85, 1.0, 0.9), "left")
        yield call(lambda: step.set_reveal(1.0))
        yield show_lines("footprint", 1.0)
        if "walls" in items:
            step.set_text("стенки: плоскости по нормалям")
            step.set_reveal(1.0)
            w = items["walls"]
            w.show()
            yield tween(0.7, lambda v: w.set_shader_input("u_alphas", Vec4(1, 1, v, 0)), 0, 1)
            self.seq.spawn(self._pulse(w, 2.4))
            yield wait(0.9)
        if "rim" in items:
            step.set_text("кромка: выборки по длине, выбросы — красным")
            step.set_reveal(1.0)
            r = items["rim"]
            r.show()
            yield tween(1.6, lambda v: r.set_shader_input("u_throw", v), 0.0, 1.3, E.linear)
            yield wait(0.6)
        step.set_text("кромки бортов и торцы")
        step.set_reveal(1.0)
        yield [show_lines("rims", 1.0), show_lines("ends", 1.0)]
        corners = self.to_root(self.d.scene.get("det_corners"))
        for p in corners:
            self.burst([tuple(p)], 70, 1.0, 0.8, hue=0.08)
        yield wait(0.9)
        yield [tween(0.5, lambda v: (cap.set_alpha(1 - v), step.set_alpha(1 - v)), 0, 1),
               tween(0.6, dim, 1.0, 0.0)]
        cap.destroy()
        step.destroy()

    def _pulse(self, node, dur):
        t = 0.0
        while t < dur and not node.is_empty():
            t += 1 / 60
            node.set_shader_input("u_size", 0.06 * (1.0 + 0.4 * math.sin(t * 9)))
            yield None

    # ------------------------------------------------------------------ #
    def stage_background(self):
        n = self.nodes
        cl, cm = n.get("cloudW"), n.get("cloudM")
        if cl is None:
            return
        det = n.get("det_root")

        def fade_det(v):
            if det is None or det.is_empty():
                return
            for node in n["det"].values():
                if node.is_empty():
                    continue
                node.set_shader_input("u_alpha", 1 - v)
                node.set_shader_input("u_alphas", Vec4(1, 1, 1 - v, 0))
        yield tween(0.6, fade_det, 0.0, 1.0)
        if det is not None:
            det.remove_node()
        cap = self.hud_text("ОТДЕЛЕНИЕ ФОНА", -0.92, 0.82, 0.085, CYAN, "left")
        yield self.type_in(cap, 0.4)
        cl.set_shader_input("u_tint", Vec4(1.0, 0.62, 0.25, 0.0))

        def blow(v):
            cl.set_shader_input("u_collapse", v)
            cl.set_shader_input("u_tint", Vec4(1.0, 0.62, 0.25, min(1.0, v * 1.5)))
            cl.set_shader_input("u_alphas", Vec4(max(0.0, 1 - v * 1.3), 1 - v, 1, 0))
            if cm is not None:
                cm.set_shader_input("u_alphas", Vec4(0, 1, v, 0))
                cm.set_shader_input("u_tint", Vec4(1.0, 0.62, 0.25, min(1.0, v * 1.5)))
        yield tween(2.0, blow, 0.0, 1.0, E.in_out_cubic)
        cl.remove_node()
        yield tween(0.4, lambda v: cap.set_alpha(1 - v), 0, 1)
        cap.destroy()

    # ------------------------------------------------------------------ #
    def stage_keypoints(self):
        sc = self.d.scene
        if sc is None or not sc.has("keypoints"):
            return
        kp = self.to_root(sc.get("keypoints"), "M")
        markers = []
        for p in kp:
            m = self.build_marker(p, ORANGE)
            markers.append(m)
            yield tween(0.35, lambda v, m=m: m.set_scale(max(v, 0.001)), 0.0, 1.0, E.out_back)
            self.burst([tuple(p)], 90, 1.4, 0.9, hue=0.07)
            yield wait(0.12)
        self.nodes["kp_markers"] = markers
        self.kp_root = kp
        rect = P.lines([np.r_[kp, kp[:1]]], [ORANGE], 3.5, "kp_rect")
        rect.reparent_to(self.root)
        rect.set_bin("fixed", 48)
        self.nodes["kp_rect"] = rect
        yield tween(0.8, lambda v: rect.set_shader_input("u_draw", v), 0.0, 1.0, E.out_cubic)
        # размеры и заполнение
        dims = self.d.json.get("body_dimensions") or {}
        w, l = dims.get("width"), dims.get("length")
        above = Vec3(*kp.mean(0)) + Vec3(0, 0, 2.3)
        lines = []
        if w and l:
            lines.append(f"ШИРИНА {w:.2f} м    ДЛИНА {l:.2f} м")
        ratio = self.d.fill_ratio
        vol = self.d.json.get("target_volume")
        if ratio is not None:
            lines.append(f"ЗАПОЛНЕНИЕ {ratio * 100:.0f}%   ·   {vol:.1f} м³")
        elif vol:
            lines.append(f"ОБЪЁМ {float(vol):.1f} м³")
        texts = []
        for i, t in enumerate(lines):
            lab = self.label(t, above + Vec3(0, 0, -0.45 * i), 0.36 if i == 0 else 0.3,
                             CYAN if i == 0 else (1.0, 0.75, 0.35, 1.0))
            texts.append(lab)
            yield self.type_in(lab, 0.7)
        # Кадр неподвижен, пока держится текст: здесь реконструкция встаёт в
        # PBR-сцену (под пустотой её не видно) — её короткая пауза тут
        # незаметнее всего.
        yield wait(0.3)
        if self.d.fetch_ready:
            self.s.apply_pbr_hidden()
        yield wait(0.8)
        yield tween(0.6, lambda v: [t.set_alpha(1 - v) for t in texts], 0, 1)
        for t in texts:
            t.destroy()

    # ------------------------------------------------------------------ #
    def stage_bodies(self):
        d = self.d
        yield until(lambda: d.bodies_ready or bool(d.error), timeout=25)
        bodies = [b for b in d.bodies if b.model is not None]
        kp = getattr(self, "kp_root", None)
        if not bodies or kp is None:
            return
        rig = self.rig
        below = -5.6
        look = Vec3(self.center.x, self.center.y, (self.rim_z + below) / 2 + 0.6)
        yield rig.move(self.side_shot(-80.0, radius=self.body_len * 2.15, height=3.2, look=look,
                                      fov=46), 2.4, E.in_out_cubic)
        cap = self.hud_text("ПОДБОР КУЗОВА", -0.92, 0.82, 0.085, CYAN, "left")
        yield self.type_in(cap, 0.4)
        chosen = None
        for b in bodies:
            holo = P.holo_model(b.model, (0.35, 0.8, 1.0, 0.0))
            holo.reparent_to(self.root)
            holo.set_z(below)
            holo.set_shader_input("u_dissolve", 1.0)
            holo.set_bin("fixed", 24)
            top = below + 3.4
            try:
                lo, hi = b.model.get_tight_bounds()
                top = below + float(hi.z)
            except Exception:
                pass
            name = self.label(f"{b.name}", Vec3(self.center.x, self.center.y, top + 1.1), 0.34)
            dims = self.label(f"{b.width:.2f} × {b.length:.2f} м",
                              Vec3(self.center.x, self.center.y, top + 0.55), 0.3,
                              (1.0, 0.8, 0.4, 1.0))
            yield [tween(0.8, lambda v, h=holo: h.set_shader_input("u_dissolve", 1 - v), 0, 1),
                   self.fade_node(holo, "u_color", 0.0, 0.55, 0.8),
                   self.type_in(name, 0.6), self.type_in(dims, 0.6)]
            if not b.correct:
                yield wait(0.7)
                holo.set_shader_input("u_color", Vec4(1.0, 0.3, 0.25, 0.6))
                dims.set_text(f"{b.width:.2f} × {b.length:.2f} м  ·  НЕ ПОДХОДИТ")
                dims.set_reveal(1.0)
                dims.color = Vec4(*RED)
                dims.set_alpha(1)
                yield wait(0.5)
                yield [tween(0.7, lambda v, h=holo: h.set_shader_input("u_dissolve", v), 0, 1),
                       tween(0.5, lambda v, a=name, b2=dims: (a.set_alpha(1 - v), b2.set_alpha(1 - v)), 0, 1)]
                holo.remove_node()
                name.destroy()
                dims.destroy()
            else:
                chosen = (b, holo, name, dims)
                break
        if chosen is None:
            yield tween(0.4, lambda v: cap.set_alpha(1 - v), 0, 1)
            cap.destroy()
            return
        b, holo, name, dims = chosen
        holo.set_shader_input("u_color", Vec4(0.35, 1.0, 0.6, 0.6))
        dims.color = Vec4(*GREEN)
        dims.set_text(f"{b.width:.2f} × {b.length:.2f} м  ·  ПОДХОДИТ")
        dims.set_reveal(1.0)
        # опорные точки кузова (points_3d) загораются
        bm = [self.build_marker(p, GREEN, parent=holo) for p in b.points_3d]
        for m in bm:
            yield tween(0.25, lambda v, m=m: m.set_scale(max(v, 0.001)), 0.0, 1.0, E.out_back)
        yield wait(0.3)
        yield [tween(0.4, lambda v: (name.set_alpha(1 - v), dims.set_alpha(1 - v)), 0, 1),
               rig.move(self.side_shot(-70.0, radius=self.body_len * 1.45,
                                       height=self.body_len * 0.6, fov=44), 2.4, E.in_out_cubic)]
        name.destroy()
        dims.destroy()
        # кузов едет к облаку, точки — к опорным точкам облака
        yield tween(2.2, lambda v: holo.set_z(below * (1 - v)), 0.0, 1.0, E.in_out_cubic)
        for m, p in zip(bm, kp):
            self.burst([tuple(p)], 110, 1.6, 1.0, hue=0.33)
        for m in self.nodes.get("kp_markers", []):
            self.seq.spawn(self._marker_hit(m))
        hit = self.label("СОВМЕЩЕНО", Vec3(*kp.mean(0)) + Vec3(0, 0, 1.4), 0.34, GREEN)
        yield self.type_in(hit, 0.5)
        yield wait(0.9)
        self.nodes["body_holo"] = holo
        self.nodes["body_markers"] = bm
        self.nodes["correct_body"] = b
        yield [tween(0.5, lambda v: (hit.set_alpha(1 - v), cap.set_alpha(1 - v)), 0, 1),
               tween(0.9, lambda v: holo.set_shader_input("u_dissolve", v), 0, 1, E.in_cubic)]
        hit.destroy()
        cap.destroy()
        holo.hide()
        holo.set_shader_input("u_dissolve", 0.0)
        for m in bm:
            m.remove_node()

    def _marker_hit(self, m):
        yield tween(0.25, lambda v: m.set_scale(1 + 0.6 * v), 0, 1, E.out_cubic)
        yield tween(0.5, lambda v: m.set_scale(1.6 - 0.6 * v), 0, 1, E.out_back)

    # ------------------------------------------------------------------ #
    def stage_fill(self):
        d = self.d
        yield until(lambda: d.fill_ready or bool(d.error), timeout=60)
        # PBR-мир готовим сейчас, пока он полностью закрыт пустотой
        self.s.apply_pbr_hidden()
        hf = self.build_heightfield()
        n = self.nodes
        cm = n.get("cloudM")
        for key in ("kp_rect",):
            if key in n:
                yield tween(0.4, lambda v, k=key: n[k].set_shader_input("u_alpha", 1 - v), 0, 1)
                n[key].remove_node()
        for m in n.get("kp_markers", []):
            m.remove_node()
        if hf is None:
            if cm is not None:
                yield tween(0.8, lambda v: cm.set_shader_input("u_alphas", Vec4(0, 1, 1 - v, 0)), 0, 1)
            return
        rig = self.rig
        c = self.center
        yield rig.move(self.side_shot(-55.0, radius=self.body_len * 1.05,
                                      height=self.body_len * 0.55, fov=46), 2.0, E.in_out_cubic)
        cap = self.hud_text("РЕКОНСТРУКЦИЯ НАПОЛНЕНИЯ", -0.92, 0.82, 0.085, CYAN, "left")
        step = self.hud_text("", -0.92, 0.7, 0.058, (0.6, 0.85, 1.0, 0.9), "left")
        yield self.type_in(cap, 0.5)
        stages = n["hf_stages"]
        hf.show()
        hf.set_shader_input("u_rise", 0.0)
        step.set_text(stages[0][1])
        yield [self.type_in(step, 0.5),
               tween(1.4, lambda v: hf.set_shader_input("u_rise", v), 0, 1, E.out_cubic),
               tween(1.4, lambda v: cm.set_shader_input("u_alphas", Vec4(0, 1, 1 - v, 0))
                     if cm is not None else None, 0, 1)]
        if cm is not None:
            cm.remove_node()
        orbit = self.seq.spawn(rig.orbit(c, self.body_len * 1.05, self.body_len * 0.55,
                                         -55, -5, 2.2 * (len(stages) - 1) + 1.0, E.in_out_sine,
                                         fov=46))
        for i in range(1, len(stages)):
            hf.set_shader_input("u_hA", stages[i - 1][0])
            hf.set_shader_input("u_hB", stages[i][0])
            hf.set_shader_input("u_front", -0.1)
            step.set_text(stages[i][1])
            step.set_reveal(0.0)
            yield [self.type_in(step, 0.45),
                   tween(1.7, lambda v: hf.set_shader_input("u_front", v), -0.1, 1.12, E.in_out_sine)]
            hf.set_shader_input("u_hA", stages[i][0])
            yield wait(0.35)
        yield orbit
        self.nodes["fill_caps"] = (cap, step)

    # ------------------------------------------------------------------ #
    def stage_boolean(self):
        n = self.nodes
        hf = n.get("hf")
        fa = self.d.fill
        res = self.build_result_holo()
        cap, step = n.get("fill_caps", (None, None))
        if hf is not None and fa is not None and fa.has("napolnitel_vertices"):
            nv = fa.get("napolnitel_vertices").astype(float)
            try:
                from src.rendering import mesh_io
                path = os.path.join(os.path.dirname(self.d.json_path),
                                    self.d.stem + "_napolnitel.obj")
                v, f = mesh_io.load_obj_arrays(path)
                wire = P.wireframe(v, f, (1.0, 0.6, 0.25, 1.0), 2.2, 25.0, "nap_wire")
            except Exception:
                lo, hi = nv.min(0), nv.max(0)
                box = np.array([[lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                                [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
                wire = P.lines([np.r_[box, box[:1]]], [ORANGE], 2.2, "nap_wire")
            wire.reparent_to(n["fill_space"])
            wire.set_bin("fixed", 47)
            if step is not None:
                step.set_text("БУЛЕВА РАЗНОСТЬ С НАПОЛНИТЕЛЕМ")
                step.set_reveal(0.0)
            yield [self.type_in(step, 0.5) if step else wait(0),
                   tween(1.4, lambda v: wire.set_shader_input("u_draw", v), 0, 1, E.out_cubic)]
            # искры по линии среза
            lo, hi = nv.min(0), nv.max(0)
            z = float(np.nanmedian(fa.get("h_final"))) if fa.has("h_final") else self.rim_z

            def edge_sparks():
                for k in range(10):
                    t = k / 9.0
                    pts = [(lo[0], lo[1] + (hi[1] - lo[1]) * t, z), (hi[0], lo[1] + (hi[1] - lo[1]) * t, z),
                           (lo[0] + (hi[0] - lo[0]) * t, lo[1], z), (lo[0] + (hi[0] - lo[0]) * t, hi[1], z)]
                    self.burst(pts, 60, 1.0, 0.7, hue=0.07, parent=n["fill_space"])
                    yield wait(0.16)
            self.seq.spawn(edge_sparks())
            yield tween(1.8, lambda v: hf.set_shader_input("u_cut", v), 0.0, 1.0, E.in_out_cubic)
            if res is not None:
                res.show()
                yield [tween(0.9, lambda v: hf.set_shader_input("u_color", Vec4(0.3, 0.8, 1.0, 0.7 * (1 - v))), 0, 1),
                       self.fade_node(res, "u_color", 0.0, 0.75, 0.9),
                       tween(0.9, lambda v: wire.set_shader_input("u_alpha", 1 - v), 0, 1)]
            else:
                yield tween(0.9, lambda v: wire.set_shader_input("u_alpha", 1 - v), 0, 1)
            hf.hide()
            wire.remove_node()
        elif res is not None:
            res.show()
            yield self.fade_node(res, "u_color", 0.0, 0.75, 1.0)
        if cap is not None:
            yield tween(0.4, lambda v: (cap.set_alpha(1 - v), step.set_alpha(1 - v)), 0, 1)
            cap.destroy()
            step.destroy()
        # кузов возвращается
        holo = n.get("body_holo")
        if holo is not None:
            holo.show()
            holo.set_shader_input("u_color", Vec4(0.35, 0.8, 1.0, 0.0))
            yield self.fade_node(holo, "u_color", 0.0, 0.4, 1.0)

    # ------------------------------------------------------------------ #
    def stage_pbr(self):
        comp, n = self.comp, self.nodes
        yield until(lambda: self.s.pbr_applied or bool(self.d.error), timeout=30)
        self.s.apply_pbr_hidden()
        uv = comp.project(self.center) or Vec2(0.5, 0.5)
        comp.wave = Vec3(uv.x, uv.y, 0.0)
        comp.wave_width = 0.18
        floor = n.get("floor")
        cap = self.hud_text("ПЕРЕХОД В ФИЗИЧЕСКИЙ МИР", 0.0, 0.82, 0.075, CYAN)
        yield self.type_in(cap, 0.5)

        def wave(v):
            comp.wave = Vec3(uv.x, uv.y, v * 2.3)
            if floor is not None:
                floor.set_shader_input("u_alpha", 1 - v)
        yield tween(2.6, wave, 0.0, 1.0, E.in_out_cubic)
        comp.abstract = 0.0
        comp.wave = Vec3(0.5, 0.5, 0.0)
        if floor is not None:
            floor.remove_node()
        holo = n.get("body_holo")
        if holo is not None:
            yield self.fade_node(holo, "u_color", 0.4, 0.0, 0.7)
            holo.remove_node()
        yield tween(0.4, lambda v: cap.set_alpha(1 - v), 0, 1)
        cap.destroy()
        yield from self.stage_texture_sweep()

    def stage_texture_sweep(self):
        res = self.nodes.get("result_holo")
        fill = self.s.pbr_fill_node()
        f = self.d.fetched or {}
        mesh = f.get("mesh")
        if fill is None or mesh is None:
            if res is not None:
                yield self.fade_node(res, "u_color", 0.75, 0.0, 0.8)
            if fill is not None:
                fill.show()
            return
        v = mesh.vertices
        y0, y1 = float(v[:, 1].min()) - 0.05, float(v[:, 1].max()) + 0.05
        # PBR видно там, где y < фронта; голограмма — где y > фронта
        self.s.clip_pbr_fill(Vec4(0, 1, 0, y0))
        fill.show()
        if res is not None:
            res.set_shader_input("u_reveal", Vec4(0, -1, 0, -y0))
        order = np.argsort(v[:, 1])
        ys = v[order, 1]

        def sweep(t):
            y = y0 + (y1 - y0) * t
            self.s.clip_pbr_fill(Vec4(0, 1, 0, y))
            if res is not None:
                res.set_shader_input("u_reveal", Vec4(0, -1, 0, -y))

        def sparkle():
            k = 0
            while True:
                t = self._sweep_t
                if t >= 1.0:
                    return
                y = y0 + (y1 - y0) * t
                i0, i1 = np.searchsorted(ys, [y - 0.04, y + 0.04])
                if i1 > i0:
                    idx = order[np.random.default_rng(k).integers(i0, i1, 12)]
                    self.burst(v[idx], 90, 1.3, 1.0, hue=None, up=1.0, gravity=-1.6, size=0.018)
                k += 1
                yield wait(0.07)
        self._sweep_t = 0.0

        def track(t):
            self._sweep_t = t
            sweep(t)
        sp = self.seq.spawn(sparkle())
        yield tween(3.2, track, 0.0, 1.0, E.in_out_sine)
        self._sweep_t = 1.0
        yield sp
        self.s.clip_pbr_fill(None)
        if res is not None:
            res.remove_node()

    # ------------------------------------------------------------------ #
    def stage_finale(self):
        rig = self.rig
        yield rig.orbit(self.center, self.body_len * 1.15, self.body_len * 0.95, -40, -18, 3.0,
                        E.out_cubic, fov=48)
        self.rig.shake = 0.0
        comp = self.comp
        yield tween(0.8, lambda v: (setattr(comp, "vignette", 0.35 * (1 - v)),
                                    setattr(comp, "grain", 0.03 * (1 - v))), 0, 1)
