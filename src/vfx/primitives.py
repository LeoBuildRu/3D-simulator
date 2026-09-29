# -*- coding: utf-8 -*-
"""
Узлы абстрактного мира из numpy-массивов.

Геометрия пишется в GeomVertexData одним копированием буфера (без поштучных
вызовов), а всё, что меняется во времени, — униформы шейдеров (см.
materials). Каждый конструктор возвращает NodePath с уже назначенным
шейдером и разумными значениями по умолчанию; менять их — `set_shader_input`.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
from panda3d.core import (
    Geom, GeomEnums, GeomNode, GeomPoints, GeomTriangles,
    GeomVertexArrayFormat, GeomVertexData, GeomVertexFormat, InternalName,
    NodePath, OmniBoundingVolume, RenderState, SamplerState, TextNode, Texture,
    Vec2, Vec3, Vec4,
)

from . import materials

_F32 = Geom.NT_float32


def _point_shader(node: NodePath, name: str) -> None:
    """
    Шейдер точек с размером из gl_PointSize. Без флага F_shader_point_size
    Panda не включает GL_PROGRAM_POINT_SIZE, и в core-профиле все точки
    рисуются в 1 пиксель, что бы шейдер ни писал.
    """
    from panda3d.core import ShaderAttrib
    node.set_shader(materials.shader(name))
    attr = node.get_attrib(ShaderAttrib)
    node.set_attrib(attr.set_flag(ShaderAttrib.F_shader_point_size, True))


def _format(columns) -> GeomVertexFormat:
    """Один чередующийся массив float32: [(имя, число компонент, смысл)]."""
    arr = GeomVertexArrayFormat()
    for name, n, contents in columns:
        arr.add_column(InternalName.make(name), n, _F32, contents)
    return GeomVertexFormat.register_format(GeomVertexFormat(arr))


_FMT_POINTS = None
_FMT_LINES = None
_FMT_SPARKS = None
_FMT_GRIDIDX = None


def _vdata(name, fmt, rows: np.ndarray) -> GeomVertexData:
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    vd = GeomVertexData(name, fmt, Geom.UH_static)
    vd.unclean_set_num_rows(len(rows))
    view = memoryview(vd.modify_array(0)).cast("B")
    view[:] = rows.tobytes()
    return vd


def _triangles(faces: np.ndarray) -> GeomTriangles:
    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(GeomEnums.NT_uint32)
    idx = prim.modify_vertices()
    idx.unclean_set_num_rows(int(faces.size))
    memoryview(idx).cast("B")[:] = np.ascontiguousarray(faces, np.uint32).tobytes()
    return prim


def _node(name, vdata, prim) -> NodePath:
    geom = Geom(vdata)
    geom.add_primitive(prim)
    gn = GeomNode(name)
    gn.add_geom(geom)
    return NodePath(gn)


def apply_inputs(np_: NodePath, **inputs) -> NodePath:
    for k, v in inputs.items():
        np_.set_shader_input(k, v)
    return np_


# --------------------------------------------------------------------------- #

def point_cloud(points: np.ndarray, intensity: np.ndarray, sweep: np.ndarray,
                label: np.ndarray, height: np.ndarray, name="cloud",
                colors: Optional[np.ndarray] = None) -> NodePath:
    """
    Облако лидара. `sweep` 0..1 — когда точку «выстрелит» развёртка,
    `label` 1 — машина, `height` — для радуги по высоте. `colors` (N,3) —
    собственные цвета точек (включаются униформой u_useColor).
    """
    global _FMT_POINTS
    if _FMT_POINTS is None:
        _FMT_POINTS = _format([("vertex", 3, Geom.C_point), ("color", 4, Geom.C_color),
                               ("i_data", 4, Geom.C_other)])
    n = len(points)
    rng = np.random.default_rng(7)
    rows = np.empty((n, 11), np.float32)
    rows[:, 0:3] = points
    if colors is not None:
        rows[:, 3:6] = colors
        rows[:, 6] = 1.0
    else:
        rows[:, 3] = intensity
        rows[:, 4:7] = 1.0
    rows[:, 7] = sweep
    rows[:, 8] = label
    rows[:, 9] = rng.random(n)
    rows[:, 10] = height
    prim = GeomPoints(Geom.UH_static)
    prim.add_consecutive_vertices(0, n)
    node = _node(name, _vdata(name, _FMT_POINTS, rows), prim)
    _point_shader(node, "points")
    # светящиеся точки друг друга не заслоняют: иначе выборки, лежащие в тех
    # же местах, что и облако, проигрывают ему тест глубины и не видны
    node.set_depth_write(False)
    return apply_inputs(node, u_origin=Vec3(0, 0, 0), u_throw=2.0, u_fly=0.025,
                        u_size=0.014, u_maxPx=7.0, u_alphas=Vec4(1, 1, 1, 0), u_rainbow=0.0,
                        u_hrange=Vec2(0, 4), u_tint=Vec4(1, 0.6, 0.2, 0),
                        u_collapse=0.0, u_useColor=1.0 if colors is not None else 0.0)


def holo_mesh(vertices: np.ndarray, faces: np.ndarray, name="holo",
              color=(0.3, 0.8, 1.0, 0.6)) -> NodePath:
    """Голографическая поверхность по произвольному мешу."""
    from src.rendering import mesh_io
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    n = mesh_io.vertex_normals(v, f)
    rows = mesh_io.interleave_v3n3t2(v, n, np.zeros((len(v), 2)))
    node = NodePath(mesh_io.geom_node_from_arrays(name, rows, f))
    node.set_shader(materials.shader("holo"))
    return holo_defaults(node, color)


def holo_defaults(node: NodePath, color=(0.3, 0.8, 1.0, 0.6)) -> NodePath:
    return apply_inputs(node, u_color=Vec4(*color), u_style=Vec4(1.0, 0.6, 0.35, 0.15),
                        u_hrange=Vec2(0, 4), u_rainbow=0.0, u_dissolve=0.0,
                        u_reveal=Vec4(0, 0, 1, 1e6), u_highlight=Vec4(0, 0, 0, 0))


def holo_model(model: NodePath, color=(0.3, 0.8, 1.0, 0.5)) -> NodePath:
    """Загруженная модель (кузов) в голограмму: без материалов RP и текстур."""
    m = model.copy_to(NodePath("holo_model"))
    for sub in [m] + list(m.find_all_matches("**")):
        sub.clear_texture()
        sub.clear_material()
        sub.clear_shader()
        sub.clear_color()
        node = sub.node()
        if isinstance(node, GeomNode):
            for i in range(node.get_num_geoms()):
                node.set_geom_state(i, RenderState.make_empty())
    # приоритет обычный: свои шейдеры дочерних узлов (метки на кузове) важнее
    m.set_shader(materials.shader("holo"))
    holo_defaults(m, color)
    return m


# --------------------------------------------------------------------------- #

def height_texture(h: np.ndarray, fill: Optional[float] = None) -> Texture:
    """
    Карта высот (ny, nx) с NaN -> текстура RGBA32F: R — высота, G — есть ли
    данные. Пустые ячейки берут высоту ближайших (чтобы соседние
    треугольники не тянулись в пол), но помечаются G = 0.
    """
    from scipy import ndimage
    h = np.asarray(h, np.float64)
    valid = np.isfinite(h)
    filled = h.copy()
    if (~valid).any() and valid.any():
        _, idx = ndimage.distance_transform_edt(~valid, return_indices=True)
        filled = h[idx[0], idx[1]]
    if fill is not None:
        filled[~np.isfinite(filled)] = fill
    ny, nx = h.shape
    data = np.zeros((ny, nx, 4), np.float32)          # BGRA в памяти Panda
    data[..., 2] = np.nan_to_num(filled)
    data[..., 1] = valid
    data[..., 3] = 1.0
    tex = Texture("height")
    tex.setup_2d_texture(nx, ny, Texture.T_float, Texture.F_rgba32)
    tex.set_ram_image(data.tobytes())
    tex.set_minfilter(SamplerState.FT_nearest)
    tex.set_magfilter(SamplerState.FT_nearest)
    tex.set_wrap_u(SamplerState.WM_clamp)
    tex.set_wrap_v(SamplerState.WM_clamp)
    return tex


def heightfield(shape, origin, res, name="heightfield") -> NodePath:
    """
    Сетка ячеек (ny, nx) для материала heightfield: вершина хранит только
    индекс ячейки, высоту берёт шейдер из текстур этапов u_hA/u_hB.
    """
    global _FMT_GRIDIDX
    if _FMT_GRIDIDX is None:
        _FMT_GRIDIDX = _format([("vertex", 3, Geom.C_point)])
    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.mgrid[0:ny, 0:nx]
    rows = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)]).astype(np.float32)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    node = _node(name, _vdata(name, _FMT_GRIDIDX, rows), _triangles(faces))
    # границы узла — по реальной геометрии, а не по индексам ячеек
    from panda3d.core import BoundingBox, Point3
    node.node().set_bounds(BoundingBox(Point3(origin[0] - 1, origin[1] - 1, -2),
                                       Point3(origin[0] + nx * res + 1,
                                              origin[1] + ny * res + 1, 8)))
    node.node().set_final(True)
    node.set_shader(materials.shader("heightfield"))
    return apply_inputs(node, u_grid=Vec3(origin[0], origin[1], res), u_front=1.2,
                        u_frontW=0.08, u_axis=1.0, u_rise=1.0, u_floor=0.0,
                        u_color=Vec4(0.3, 0.8, 1.0, 0.7), u_style=Vec4(1.0, 0.5, 0.3, 0.1),
                        u_hrange=Vec2(0, 4), u_rainbow=1.0, u_dissolve=0.0,
                        u_reveal=Vec4(0, 0, 1, 1e6), u_contours=0.4,
                        u_box=Vec4(-1e6, 1e6, -1e6, 1e6), u_cut=0.0)


# --------------------------------------------------------------------------- #

def lines(polylines: Sequence[np.ndarray], colors=None, width=3.0,
          name="lines") -> NodePath:
    """
    Толстые линии: список ломаных (K,3). Длина копится вдоль каждой ломаной —
    `u_draw` 0..1 «прочерчивает» их одновременно, каждую от начала.
    """
    global _FMT_LINES
    if _FMT_LINES is None:
        _FMT_LINES = _format([("vertex", 3, Geom.C_point), ("i_other", 3, Geom.C_other),
                              ("i_line", 4, Geom.C_other), ("color", 4, Geom.C_color)])
    rows, faces = [], []
    base = 0
    for k, pl in enumerate(polylines):
        pl = np.asarray(pl, np.float64)
        if len(pl) < 2:
            continue
        seg = np.linalg.norm(np.diff(pl, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        total = max(cum[-1], 1e-6)
        col = np.asarray(colors[k] if colors is not None else (0.4, 0.9, 1.0, 1.0), np.float32)
        a, b = pl[:-1], pl[1:]
        la, lb = cum[:-1] / total, cum[1:] / total
        m = len(a)
        r = np.zeros((m, 4, 14), np.float32)
        for j, (p, o, side, end, ln) in enumerate(((a, b, -1, 0, la), (a, b, 1, 0, la),
                                                    (b, a, -1, 1, lb), (b, a, 1, 1, lb))):
            r[:, j, 0:3] = p
            r[:, j, 3:6] = o
            r[:, j, 6] = side
            r[:, j, 7] = end
            r[:, j, 8] = ln
            r[:, j, 9] = 1.0
            r[:, j, 10:14] = col
        rows.append(r.reshape(-1, 14))
        q = base + np.arange(m)[:, None] * 4
        faces.append(np.concatenate([q + [0, 1, 2], q + [1, 3, 2]]))
        base += m * 4
    if not rows:
        return NodePath(name)
    node = _node(name, _vdata(name, _FMT_LINES, np.vstack(rows)),
                 _triangles(np.vstack(faces)))
    node.set_shader(materials.shader("lines"))
    node.set_depth_write(False)
    return apply_inputs(node, u_width=float(width), u_draw=1.0, u_alpha=1.0,
                        u_glow=1.5, u_dash=0.0)


def mesh_edges(vertices: np.ndarray, faces: np.ndarray, feature_deg: float = 20.0):
    """
    Рёбра меша для каркаса: только «рисующие» (граница и изломы больше
    feature_deg), иначе плотная сетка превращается в сплошную заливку.
    Возвращает список отрезков (K, 2, 3).
    """
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    owner = np.tile(np.arange(len(f)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = np.r_[(key[1:] == key[:-1]).all(1), False]
    keep = []
    i = 0
    cos_lim = np.cos(np.radians(feature_deg))
    n = len(key)
    while i < n:
        if same[i]:
            if fn[owner[i]] @ fn[owner[i + 1]] < cos_lim:
                keep.append(key[i])
            i += 2
            while i < n and (key[i] == key[i - 1]).all():
                i += 1
        else:
            keep.append(key[i])
            i += 1
    if not keep:
        return np.zeros((0, 2, 3))
    k = np.array(keep)
    return np.stack([v[k[:, 0]], v[k[:, 1]]], axis=1)


def wireframe(vertices, faces, color=(0.4, 0.9, 1.0, 1.0), width=1.6,
              feature_deg=20.0, name="wire") -> NodePath:
    segs = mesh_edges(vertices, faces, feature_deg)
    node = lines([s for s in segs], [color] * len(segs), width, name)
    return node


# --------------------------------------------------------------------------- #

def sparks(origins: np.ndarray, velocities: np.ndarray, delays: np.ndarray,
           lifetimes: np.ndarray, hues: np.ndarray, name="sparks") -> NodePath:
    """Искры: вся траектория считается шейдером по u_time облака."""
    global _FMT_SPARKS
    if _FMT_SPARKS is None:
        _FMT_SPARKS = _format([("vertex", 3, Geom.C_point), ("i_vel", 3, Geom.C_other),
                               ("i_life", 4, Geom.C_other)])
    n = len(origins)
    rng = np.random.default_rng(len(origins))
    rows = np.empty((n, 10), np.float32)
    rows[:, 0:3] = origins
    rows[:, 3:6] = velocities
    rows[:, 6] = delays
    rows[:, 7] = lifetimes
    rows[:, 8] = rng.random(n)
    rows[:, 9] = hues
    prim = GeomPoints(Geom.UH_static)
    prim.add_consecutive_vertices(0, n)
    node = _node(name, _vdata(name, _FMT_SPARKS, rows), prim)
    node.node().set_bounds(OmniBoundingVolume())
    node.node().set_final(True)
    _point_shader(node, "sparks")
    node.set_depth_write(False)
    return apply_inputs(node, u_time=0.0, u_gravity=Vec3(0, 0, -2.5), u_drag=1.2,
                        u_size=0.02, u_alpha=1.0)


def grid_floor(center, size=160.0, z=0.0, name="floor") -> NodePath:
    from panda3d.core import CardMaker
    cm = CardMaker(name)
    cm.set_frame(center[0] - size / 2, center[0] + size / 2,
                 center[1] - size / 2, center[1] + size / 2)
    node = NodePath(cm.generate())
    node.set_p(-90)       # CardMaker строит в XZ — кладём в XY
    node.set_pos(0, 0, z)
    node.set_shader(materials.shader("grid"))
    node.set_depth_write(False)
    return apply_inputs(node, u_alpha=1.0, u_pulse=Vec4(0, 0, 0, 0),
                        u_center=Vec3(center[0], center[1], z))


def photo_surface(vertices, uv, faces, photo: Texture, name="photo") -> NodePath:
    """Снимок на меше глубины (без освещения)."""
    fmt = GeomVertexFormat.get_v3t2()
    rows = np.column_stack([vertices, uv]).astype(np.float32)
    node = _node(name, _vdata(name, fmt, rows), _triangles(np.asarray(faces)))
    node.set_shader(materials.shader("photo"))
    return apply_inputs(node, u_photo=photo, u_alpha=1.0, u_holo=0.0, u_dissolve=0.0,
                        u_scan=Vec4(0, 0, 0, 0), u_hrange=Vec2(0, 4))


# --------------------------------------------------------------------------- #

_FONT = None
FONT_PATH = os.path.join("assets", "fonts", "JOST", "static", "Jost-Medium.ttf")


def font():
    """Шрифт с полем расстояний: чёткий на любом размере и в свечении."""
    global _FONT
    if _FONT is None:
        from panda3d.core import DynamicTextFont, Filename, TextFont
        f = DynamicTextFont(Filename.from_os_specific(os.path.abspath(FONT_PATH)))
        f.set_pixels_per_unit(48)
        f.set_page_size(1024, 1024)
        try:
            f.set_render_mode(TextFont.RM_distance_field)
        except Exception:
            pass
        f.set_minfilter(SamplerState.FT_linear)
        f.set_magfilter(SamplerState.FT_linear)
        _FONT = f
    return _FONT


class HoloText:
    """Текст в мире: поворачивается к камере, проявляется слева направо."""

    def __init__(self, parent: NodePath, text: str, scale=0.3,
                 color=(0.5, 0.95, 1.0, 1.0), align="center"):
        self.tn = TextNode("holo_text")
        self.tn.set_font(font())
        self.tn.set_align({"center": TextNode.A_center, "left": TextNode.A_left,
                           "right": TextNode.A_right}[align])
        self.tn.set_text(text)
        self.root = parent.attach_new_node("holo_text_root")
        self.np = self.root.attach_new_node(self.tn.generate())
        self.np.set_scale(scale)
        self.root.set_billboard_point_eye()
        self.np.set_shader(materials.shader("text"), 20)
        self.np.set_depth_write(False)
        self.np.set_bin("fixed", 60)
        self.color = Vec4(*color)
        apply_inputs(self.np, u_color=self.color, u_reveal=1e6, u_glow=1.2)
        self.text = text

    @property
    def width(self) -> float:
        return self.tn.get_width()

    def set_text(self, text: str) -> None:
        self.text = text
        self.tn.set_text(text)
        self.np.node().remove_all_children()
        new = self.tn.generate()
        old = self.np
        self.np = self.root.attach_new_node(new)
        self.np.set_scale(old.get_scale())
        self.np.set_state(old.get_state())
        old.remove_node()

    def set_reveal(self, k: float) -> None:
        """k 0..1 — доля ширины текста, уже показанная."""
        left = {TextNode.A_center: -self.width / 2, TextNode.A_left: 0.0,
                TextNode.A_right: -self.width}[self.tn.get_align()]
        self.np.set_shader_input("u_reveal", left + k * (self.width + 0.3) - 0.15)

    def set_alpha(self, a: float) -> None:
        c = Vec4(self.color)
        c.w = self.color.w * a
        self.np.set_shader_input("u_color", c)

    def destroy(self):
        self.root.remove_node()
