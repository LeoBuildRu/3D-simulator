# cloth_simulator.py
#
# Симуляция ткани (тент/полог), свисающей с борта самосвала, для датасетов
# сегментации.
#
# Panda3D не имеет солвера ткани, поэтому здесь свой. Физику считает
# XPBD-солвер на GPU (см. cloth_warp.py, NVIDIA Warp): подшаги, двугранный
# изгиб, точная коллизия по BVH и самопересечение. Этот модуль отвечает за
# ПОСТАНОВКУ — как тент разложен на кузове, где закреплён, каким получается меш.
#
# Ниже — запасной солвер на numpy (position-based dynamics), он включается на
# машинах без CUDA. Он заметно грубее (жёсткость зависит от расписания шагов,
# коллизия приближена облаком точек, самопересечения нет), но датасет собирает:
#   * частицы разложены регулярной сеткой (ny x nx);
#   * связи трёх типов — structural (соседи по сетке), shear (диагонали),
#     bend (через одну) — держат форму и не дают ткани «складываться в нить»;
#   * силы: гравитация + аэродинамика (ветер действует через нормаль полотна,
#     что и даёт полоскание/хлопки, а не равномерный снос);
#   * коллизии: РЕАЛЬНАЯ треугольная геометрия кузова и груза (MeshCollider)
#     плюс земля. AABB здесь принципиально не годится: кузов — открытый ящик
#     с тонкими стенками, и параллелепипед, считая его сплошным, выталкивает
#     наружу ткань, которая должна свисать ВНУТРЬ.
#
# Связи решаются ПАКЕТАМИ (batches) с раскраской графа: внутри одного пакета
# ни одна вершина не встречается дважды, поэтому проекция связей — обычная
# векторная операция numpy без np.add.at (тот на порядок медленнее). Это и
# быстрее, и сходится лучше «якобиевого» усреднения.
#
# Итог симуляции — статичная поза (см. ClothSolver.settle): для датасета
# нужен один кадр, поэтому солвер прогоняется offscreen, а сцена получает уже
# готовый меш. «Полоскание на ветру» — это остановка симуляции в момент, когда
# полотно ещё не улеглось, а не анимация.
import math
import os
import random

import numpy as np

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomEnums,
    GeomVertexData,
    GeomVertexFormat,
)

# Основной солвер — XPBD на GPU (NVIDIA Warp). Numpy-солвер ниже остаётся
# запасным: датасет должен собираться и на машине без CUDA.
from src.rendering.cloth_warp import (
    WarpClothSolver,
    WarpMeshCollider,
    warp_available,
)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CLOTH_DIR = os.path.join(_PROJECT_ROOT, "assets", "cloth")

# PBR-набор из assets/cloth (Specular_IOR пайплайном не используется).
CLOTH_TEXTURES = {
    "diffuse": os.path.join(_CLOTH_DIR, "JustPlane_DefaultMaterial_Color.png"),
    "normal": os.path.join(_CLOTH_DIR, "JustPlane_DefaultMaterial_Normal.png"),
    "roughness": os.path.join(
        _CLOTH_DIR, "JustPlane_DefaultMaterial_Roughness.png"),
}

_EPS = 1e-9


def grid_normals(p):
    """Нормали вершин регулярной сетки (ny, nx, 3) через центральные разности."""
    du = np.empty_like(p)
    dv = np.empty_like(p)

    du[:, 1:-1] = p[:, 2:] - p[:, :-2]
    du[:, 0] = p[:, 1] - p[:, 0]
    du[:, -1] = p[:, -1] - p[:, -2]

    dv[1:-1, :] = p[2:] - p[:-2]
    dv[0] = p[1] - p[0]
    dv[-1] = p[-1] - p[-2]

    n = np.cross(du, dv)
    ln = np.linalg.norm(n, axis=2, keepdims=True)
    return n / np.maximum(ln, _EPS)


def subdivide_grid(p, factor):
    """Сгустить сетку (ny, nx, 3) бикубическим сплайном.

    Ткань между узлами гладкая, поэтому сплайн даёт тот же силуэт, что и
    симуляция во столько же раз более плотной сетки, но practически даром.
    """
    if factor <= 1:
        return p
    from scipy.ndimage import map_coordinates

    ny, nx = p.shape[:2]
    if ny < 4 or nx < 4:                 # сплайну не хватит опор
        return p
    NY = (ny - 1) * factor + 1
    NX = (nx - 1) * factor + 1
    rr, cc = np.meshgrid(np.linspace(0, ny - 1, NY),
                         np.linspace(0, nx - 1, NX), indexing="ij")
    out = np.empty((NY, NX, 3), dtype=np.float64)
    for c in range(3):
        out[..., c] = map_coordinates(p[..., c], [rr, cc], order=3,
                                      mode="nearest")
    return out


# ---------------------------------------------------------------------------
# Коллизия с реальной геометрией
# ---------------------------------------------------------------------------
def collect_world_triangles(root):
    """Все треугольники поддерева в МИРОВЫХ координатах -> (V, F) | None.

    MyApp.panda_to_trimesh здесь не подходит: он берёт только первый
    GeomNode, не вызывает decompose() (тристрипы теряются) и складывает
    вершины нескольких Geom'ов без сдвига индексов. Кузов в .bam — это
    несколько GeomNode со своими трансформами, поэтому нужен полный обход.
    """
    from panda3d.core import GeomVertexReader, GeomNode

    if root is None or root.is_empty():
        return None

    nodes = list(root.find_all_matches("**/+GeomNode"))
    if isinstance(root.node(), GeomNode):
        nodes.append(root)

    verts = []
    faces = []
    for gnp in nodes:
        gnode = gnp.node()
        mat = gnp.get_net_transform().get_mat()
        for gi in range(gnode.get_num_geoms()):
            geom = gnode.get_geom(gi)
            vdata = geom.get_vertex_data()
            if not vdata.has_column("vertex"):
                continue

            reader = GeomVertexReader(vdata, "vertex")
            local = []
            while not reader.is_at_end():
                p = mat.xform_point(reader.get_data3f())
                local.append((p[0], p[1], p[2]))
            if not local:
                continue

            base = len(verts)          # сдвиг индексов ЭТОГО geom'а
            got_face = False
            for pi in range(geom.get_num_primitives()):
                prim = geom.get_primitive(pi).decompose()
                idx = list(prim.get_vertex_list())
                for t in range(0, len(idx) - 2, 3):
                    faces.append((idx[t] + base, idx[t + 1] + base,
                                  idx[t + 2] + base))
                    got_face = True
            if got_face:
                verts.extend(local)

    if not verts or not faces:
        return None
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


class MeshCollider:
    """Коллизия ткани с ПРОИЗВОЛЬНОЙ статической геометрией.

    Почему не AABB и не signed distance field:
      * кузов — ОТКРЫТЫЙ ящик с тонкими стенками. AABB считает его сплошным,
        поэтому ткань, свисающая ВНУТРЬ кузова, выталкивается наружу — ровно
        тот артефакт «полотно внутри/сквозь борта»;
      * SDF требует замкнутой поверхности, а у открытой оболочки понятие
        «внутри» не определено.

    Поэтому используется НЕЗНАКОВОЕ расстояние до поверхности: частица просто
    держится в `thickness` от ближайшего треугольника, с какой стороны она
    подошла — с той и остаётся. Тонкая стенка работает как двусторонняя
    преграда, и ткань свободно висит как снаружи, так и внутри кузова.

    Поверхность представлена облаком точек с нормалями (KD-дерево): запрос
    ближайшей точки векторизован и стоит миллисекунды на шаг, в отличие от
    точных запросов к треугольникам.
    """

    # Сколько ближайших точек поверхности проверять на частицу. Одной мало:
    # у тонкой стенки ближайшей может оказаться точка с ДРУГОЙ её грани.
    K_NEAREST = 4

    def __init__(self, spacing, max_samples=120_000):
        self.spacing = float(spacing)
        self.max_samples = int(max_samples)
        self._pts = []
        self._nrm = []
        self.tree = None
        self.points = None
        self.normals = None

    def add_mesh(self, verts, faces, region=None):
        """Засеять поверхность точками с шагом ~spacing.

        region=(min_xyz, max_xyz) — рабочая зона ткани. Треугольники вне её
        пропускаются: тент занимает малую часть кузова, и сеять весь грузовик
        на такой плотности незачем.
        """
        if verts is None or faces is None or len(faces) == 0:
            return

        tri = verts[faces]                      # (T, 3, 3)

        def clip(t):
            if region is None:
                return t
            rmin, rmax = region
            near = np.all((t.max(axis=1) >= rmin) & (t.min(axis=1) <= rmax),
                          axis=1)
            return t[near]

        tri = clip(tri)
        if len(tri) == 0:
            return

        # Крупные треугольники (борт кузова — это одна плита на 8 единиц)
        # дробим до размера шага и на каждом шаге отсекаем ушедшее из рабочей
        # зоны. Без этого решётка сеется по ВСЕЙ плите, упирается в лимит
        # точек, и прореживание оставляет щели шире барьера — ткань уходила
        # в торец борта именно так.
        target = self.spacing * 2.0
        for _ in range(12):
            edge = np.maximum.reduce([
                np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
            ])
            big = edge > target
            if not big.any():
                break
            t = tri[big]
            a, b, c = t[:, 0], t[:, 1], t[:, 2]
            ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
            split = np.concatenate([
                np.stack([a, ab, ca], axis=1),
                np.stack([ab, b, bc], axis=1),
                np.stack([ca, bc, c], axis=1),
                np.stack([ab, bc, ca], axis=1),
            ])
            tri = np.concatenate([tri[~big], clip(split)])
            if len(tri) == 0:
                return

        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        cross = np.cross(e1, e2)
        area2 = np.linalg.norm(cross, axis=1)
        keep = area2 > _EPS                     # вырожденные треугольники прочь
        if not keep.any():
            return
        tri, e1, e2, cross, area2 = (tri[keep], e1[keep], e2[keep],
                                     cross[keep], area2[keep])
        normals = cross / area2[:, None]

        # Число делений ребра. Точки кладутся РЕГУЛЯРНОЙ барицентрической
        # решёткой, а не случайно: у случайной выборки остаются просветы
        # в 2-3 шага, и ткань утекает сквозь них.
        edge = np.maximum.reduce([
            np.linalg.norm(e1, axis=1),
            np.linalg.norm(e2, axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        ])
        n_div = np.clip(np.ceil(edge / max(self.spacing, _EPS)), 1, 512
                        ).astype(np.int64)

        # Бюджет точек: (n+1)(n+2)/2 на треугольник. При переполнении режем
        # разрешение решётки, а не выбрасываем точки — просветы недопустимы.
        while True:
            total = int(np.sum((n_div + 1) * (n_div + 2) // 2))
            if total <= self.max_samples or np.all(n_div <= 1):
                break
            n_div = np.maximum(1, (n_div * 0.7).astype(np.int64))

        # Решётки разного разрешения группируем — внутри группы всё векторно.
        for n in np.unique(n_div):
            sel = np.nonzero(n_div == n)[0]
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1),
                               indexing="ij")
            m = (i + j) <= n
            bi = (i[m] / n)[None, :, None]
            bj = (j[m] / n)[None, :, None]
            bk = 1.0 - bi - bj
            t = tri[sel]
            pts = (bk * t[:, None, 0, :] + bi * t[:, None, 1, :]
                   + bj * t[:, None, 2, :])
            self._pts.append(pts.reshape(-1, 3))
            self._nrm.append(np.repeat(normals[sel], pts.shape[1], axis=0))

    def build(self):
        """Собрать KD-дерево. False, если сеять было нечего."""
        if not self._pts:
            return False
        from scipy.spatial import cKDTree

        pts = np.concatenate(self._pts, axis=0)
        nrm = np.concatenate(self._nrm, axis=0)
        self._pts = self._nrm = None

        # Дробление даёт общие вершины у соседних треугольников — каждая
        # точка повторяется 5-6 раз. На объёме дерева это десятикратная
        # разница, а запрос k ближайших идёт по КАЖДОМУ подшагу симуляции,
        # так что дубликаты стоили больше, чем сама физика.
        # Ключ включает нормаль: у тонкой стенки совпадающие точки передней и
        # задней граней — разные преграды, схлопывать их нельзя.
        step = max(self.spacing * 0.4, _EPS)
        key = np.concatenate([np.round(pts / step),
                              np.round(nrm * 4.0)], axis=1).astype(np.int64)
        _, uniq = np.unique(key, axis=0, return_index=True)
        self.points = np.ascontiguousarray(pts[uniq])
        self.normals = np.ascontiguousarray(nrm[uniq])

        self.tree = cKDTree(self.points)
        return True

    def resolve(self, pos, prev, thickness, movable=None):
        """Вытолкнуть частицы, подошедшие ближе thickness к поверхности.

        Сторона барьера определяется по ПРЕДЫДУЩЕМУ положению, а не по
        текущему. Это принципиально: борт тоньше двух барьеров, поэтому
        быстрая частица успевает проскочить внутрь стенки за шаг. Если брать
        сторону из текущего положения, барьер «согласится» с тем, что частица
        уже внутри, и будет удерживать её там — ровно артефакт «ткань в
        толще борта». Опираясь на prev, барьер всегда возвращает частицу на ту
        сторону, с которой она пришла.
        """
        if self.tree is None:
            return

        # Берём НЕСКОЛЬКО ближайших точек, а не одну. У тонкой стенки
        # ближайшая точка может лежать на ПРОТИВОПОЛОЖНОЙ грани, и барьер по
        # ней вытолкнет частицу вглубь стенки вместо возврата назад.
        k = min(self.K_NEAREST, len(self.points))
        dist, idx = self.tree.query(pos, k=k,
                                    distance_upper_bound=thickness * 4.0)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]

        valid = np.isfinite(dist) & (idx < len(self.points))
        if movable is not None:
            valid &= movable[:, None]
        if not valid.any():
            return

        safe = np.where(valid, idx, 0)
        near = self.points[safe]                      # (N, k, 3)
        n = self.normals[safe]                        # (N, k, 3)

        s = np.einsum("ijk,ijk->ij", pos[:, None, :] - near, n)
        s_prev = np.einsum("ijk,ijk->ij", prev[:, None, :] - near, n)

        # Каждая грань — ОДНОСТОРОННЯЯ преграда со стороны, с которой частица
        # пришла. Грани, за которыми она была изначально, её не касаются.
        front = s_prev > 1e-9
        need = valid & front & (s < thickness)
        if not need.any():
            return

        corr = np.where(need, thickness - s, 0.0)     # (N, k)
        best = np.argmax(corr, axis=1)
        rows = np.arange(len(pos))
        amount = corr[rows, best]
        act = amount > 0.0
        if not act.any():
            return
        pos[act] += n[act, best[act]] * amount[act][:, None]


# ---------------------------------------------------------------------------
# Солвер
# ---------------------------------------------------------------------------
class ClothSolver:
    """PBD-ткань на регулярной сетке.

    Работает в мировых координатах Panda3D (Z вверх). Единицы сцены
    произвольны: сила ветра задаётся В ДОЛЯХ ОТ ГРАВИТАЦИИ, а форма провиса
    определяется геометрией связей, поэтому солвер не зависит от масштаба
    модели грузовика.
    """

    def __init__(self, positions, pinned, *,
                 rest_positions=None,
                 gravity=(0.0, 0.0, -9.81),
                 stiff_structural=1.0,
                 stiff_shear=0.7,
                 stiff_bend=0.25,
                 damping=0.02,
                 iterations=6):
        # positions:      (ny, nx, 3) — стартовая раскладка (как ткань висит).
        # rest_positions: (ny, nx, 3) — НЕДЕФОРМИРОВАННОЕ полотно; из него
        #                 берутся длины связей. Разделение принципиально:
        #                 если длины считать по стартовой раскладке, ткань
        #                 «запоминает» её как ненапряжённую и остаётся плоской.
        #                 Реальный тент собран в сборку — материала больше,
        #                 чем пролёт крепления, и избыток уходит в складки.
        # pinned:         (ny, nx) bool — закреплённые (неподвижные) частицы.
        self.ny, self.nx = positions.shape[:2]
        self.pos = np.array(positions, dtype=np.float64).reshape(-1, 3)
        self.vel = np.zeros_like(self.pos)
        self._rest = (self.pos if rest_positions is None
                      else np.array(rest_positions,
                                    dtype=np.float64).reshape(-1, 3))

        # Обратная масса: 0 у закреплённых частиц — они не двигаются вообще.
        self.inv_mass = np.where(pinned.reshape(-1), 0.0, 1.0)

        self.gravity = np.array(gravity, dtype=np.float64)
        self.damping = float(damping)
        self.iterations = int(iterations)

        self.batches = self._build_batches(
            stiff_structural, stiff_shear, stiff_bend)

        # Препятствия: сетка (MeshCollider), запасные AABB и уровень земли.
        self.collider = None
        self.thickness = 0.0
        self.boxes = []
        self.ground_z = None
        self.friction = 0.35

        # Параметры ветра (см. _wind_at).
        self.wind_dir = np.array([0.0, 1.0, 0.0])
        self.wind_speed = 0.0
        self.wind_gust = 0.0
        self.wind_turbulence = 0.0
        self.drag = 1.4
        self._time = 0.0

    # -- топология связей ---------------------------------------------------
    def _idx(self, i, j):
        return i * self.nx + j

    def _build_batches(self, k_struct, k_shear, k_bend):
        """Пакеты связей с непересекающимися вершинами.

        Раскраска тривиальна благодаря регулярности сетки:
          * горизонтальные рёбра — по чётности столбца;
          * вертикальные — по чётности строки;
          * диагонали — по чётности (строка, столбец), 4 пакета;
          * bend (через одну) — по остатку индекса от 4.
        """
        ny, nx = self.ny, self.nx
        ii, jj = np.mgrid[0:ny, 0:nx]
        batches = []

        def add(sel_a, sel_b, stiffness, key):
            """sel_* — (i, j) массивы концов рёбер; key — маркер раскраски."""
            ia, ja = sel_a
            ib, jb = sel_b
            for color in np.unique(key):
                m = key == color
                if not m.any():
                    continue
                a = self._idx(ia[m], ja[m])
                b = self._idx(ib[m], jb[m])
                rest = np.linalg.norm(self._rest[b] - self._rest[a], axis=1)
                batches.append((a, b, rest, float(stiffness)))

        # structural — горизонталь / вертикаль
        if nx > 1:
            i0, j0 = ii[:, :-1], jj[:, :-1]
            add((i0, j0), (i0, j0 + 1), k_struct, j0 % 2)
        if ny > 1:
            i0, j0 = ii[:-1, :], jj[:-1, :]
            add((i0, j0), (i0 + 1, j0), k_struct, i0 % 2)

        # shear — обе диагонали ячейки
        if nx > 1 and ny > 1:
            i0, j0 = ii[:-1, :-1], jj[:-1, :-1]
            color = (i0 % 2) * 2 + (j0 % 2)
            add((i0, j0), (i0 + 1, j0 + 1), k_shear, color)
            add((i0, j0 + 1), (i0 + 1, j0), k_shear, color)

        # bend — через одну частицу; держит ткань от острых изломов
        if nx > 2:
            i0, j0 = ii[:, :-2], jj[:, :-2]
            add((i0, j0), (i0, j0 + 2), k_bend, j0 % 4)
        if ny > 2:
            i0, j0 = ii[:-2, :], jj[:-2, :]
            add((i0, j0), (i0 + 2, j0), k_bend, i0 % 4)

        return batches

    # -- препятствия --------------------------------------------------------
    def add_box(self, bmin, bmax, margin=0.0):
        bmin = np.array(bmin, dtype=np.float64) - margin
        bmax = np.array(bmax, dtype=np.float64) + margin
        if np.all(bmax > bmin):
            self.boxes.append((bmin, bmax))

    def set_mesh_collider(self, collider, thickness):
        self.collider = collider
        self.thickness = float(thickness)

    def set_ground(self, z):
        self.ground_z = float(z)

    # -- ветер --------------------------------------------------------------
    def set_wind(self, direction, speed, gust=0.0, turbulence=0.0, drag=1.4):
        d = np.array(direction, dtype=np.float64)
        n = np.linalg.norm(d)
        self.wind_dir = d / n if n > _EPS else np.array([0.0, 1.0, 0.0])
        self.wind_speed = float(speed)
        self.wind_gust = float(gust)
        self.wind_turbulence = float(turbulence)
        self.drag = float(drag)

    def _wind_at(self, pos, t):
        """Поле скорости ветра: порыв (общий во времени) + турбулентность
        (бегущие волны по пространству). Волны дают неоднородность фронта —
        без неё полотно движется как жёсткая доска."""
        if self.wind_speed <= 0.0:
            return np.zeros_like(pos)

        gust = 1.0 + self.wind_gust * math.sin(1.7 * t + 0.6)
        w = self.wind_dir * (self.wind_speed * gust)
        wind = np.broadcast_to(w, pos.shape).copy()

        if self.wind_turbulence > 0.0:
            amp = self.wind_speed * self.wind_turbulence
            phase = 2.4 * pos[:, 0] + 1.9 * pos[:, 1] + 3.1 * pos[:, 2]
            wind[:, 0] += amp * np.sin(phase + 2.7 * t)
            wind[:, 1] += amp * np.sin(0.8 * phase - 3.3 * t + 1.2)
            wind[:, 2] += 0.6 * amp * np.sin(1.3 * phase + 2.1 * t + 2.5)
        return wind

    # -- нормали ------------------------------------------------------------
    def normals(self):
        return grid_normals(self.pos.reshape(self.ny, self.nx, 3))

    # -- шаг симуляции ------------------------------------------------------
    def step(self, dt):
        moving = self.inv_mass > 0.0

        # 1) Внешние силы. Аэродинамика считается через нормаль: полотно
        #    «ловит» только ту часть встречного потока, что перпендикулярна
        #    поверхности. Отсюда и хлопки, и планирование краёв.
        acc = np.broadcast_to(self.gravity, self.pos.shape).copy()
        if self.wind_speed > 0.0:
            n = self.normals().reshape(-1, 3)
            v_rel = self.vel - self._wind_at(self.pos, self._time)
            vn = np.einsum("ij,ij->i", v_rel, n)[:, None]
            acc -= self.drag * vn * n

        self.vel[moving] += acc[moving] * dt
        self.vel *= max(0.0, 1.0 - self.damping * dt * 60.0)
        self.vel[~moving] = 0.0

        prev = self.pos.copy()
        self.pos[moving] += self.vel[moving] * dt

        # 2) Проекция связей (Гаусс-Зейдель по непересекающимся пакетам).
        for _ in range(self.iterations):
            for a, b, rest, stiff in self.batches:
                pa = self.pos[a]
                pb = self.pos[b]
                d = pb - pa
                length = np.linalg.norm(d, axis=1)
                safe = np.maximum(length, _EPS)
                wa = self.inv_mass[a]
                wb = self.inv_mass[b]
                wsum = wa + wb
                scale = np.where(wsum > _EPS,
                                 stiff * (length - rest) / (safe * np.maximum(wsum, _EPS)),
                                 0.0)
                corr = d * scale[:, None]
                self.pos[a] = pa + corr * wa[:, None]
                self.pos[b] = pb - corr * wb[:, None]

        # 3) Коллизии — после связей, иначе связи вернут ткань внутрь бортов.
        self._resolve_collisions(prev)

        # 4) Скорость восстанавливаем из фактического смещения: так трение и
        #    выталкивание из коллизий гасят движение физично.
        self.vel = (self.pos - prev) / dt
        self.vel[~moving] = 0.0
        self._time += dt

    def _resolve_collisions(self, prev):
        # Сетка — основной способ. Трение по касательной берётся из смещения
        # за шаг: ткань, легшая на борт, перестаёт сползать.
        if self.collider is not None and self.thickness > 0.0:
            before = self.pos.copy()
            # Закреплённые точки тоже проверяем: якорь, оказавшийся в толще
            # борта, хуже, чем якорь, сдвинутый на поверхность. Коррекция
            # срабатывает только при нарушении, поэтому дрейфа нет.
            self.collider.resolve(self.pos, prev, self.thickness)
            if self.friction > 0.0:
                moved = np.any(np.abs(self.pos - before) > 1e-9, axis=1)
                if moved.any():
                    self.pos[moved] -= self.friction * (
                        before[moved] - prev[moved])

        for bmin, bmax in self.boxes:
            inside = np.all((self.pos > bmin) & (self.pos < bmax), axis=1)
            if not inside.any():
                continue
            p = self.pos[inside]
            # Выталкиваем по оси с минимальным проникновением.
            to_min = p - bmin           # расстояние до «нижних» граней
            to_max = bmax - p           # до «верхних»
            depth = np.minimum(to_min, to_max)
            axis = np.argmin(depth, axis=1)
            rows = np.arange(p.shape[0])
            use_min = to_min[rows, axis] < to_max[rows, axis]
            p[rows, axis] = np.where(use_min,
                                     bmin[axis], bmax[axis])
            self.pos[inside] = p
            # Трение: гасим касательное скольжение по борту.
            if self.friction > 0.0:
                slide = self.pos[inside] - prev[inside]
                slide[rows, axis] = 0.0
                self.pos[inside] -= self.friction * slide

        if self.ground_z is not None:
            below = self.pos[:, 2] < self.ground_z
            if below.any():
                self.pos[below, 2] = self.ground_z
                if self.friction > 0.0:
                    slide = self.pos[below, :2] - prev[below, :2]
                    self.pos[below, :2] -= self.friction * slide

    def settle(self, steps, dt=1.0 / 120.0, substeps=2):
        sub_dt = dt / max(1, substeps)
        for _ in range(int(steps)):
            for _ in range(substeps):
                self.step(sub_dt)
        return self.pos.reshape(self.ny, self.nx, 3)


# ---------------------------------------------------------------------------
# Постановка сцены: размещение, симуляция, меш
# ---------------------------------------------------------------------------
class ClothSimulator:
    """Создаёт случайное полотно, привязанное к кузову (или к кадру камеры).

    Использование в пайплайне датасета:

        app.cloth_simulator.spawn_random()   # перед съёмкой кадра
        ...
        app.cloth_simulator.clear()          # после
    """

    # Потолок числа частиц и размера стороны сетки.
    #
    # На CPU стоимость линейна по частицам, и выше ~3k считать было незачем:
    # шаг сетки всё равно грубее складки, силуэт добирался сплайном.
    # На GPU частицы фактически бесплатны (одно ядро на всю сетку), поэтому
    # сетка берётся такой, чтобы складка разрешалась САМОЙ СИМУЛЯЦИЕЙ, а не
    # достраивалась сглаживанием: мелкие складки у кромки и на свободном
    # крае — то, чего сплайн из грубой сетки не даёт в принципе.
    MAX_PARTICLES = 3200
    MAX_PARTICLES_GPU = 26_000
    MAX_SIDE = 64
    MAX_SIDE_GPU = 170

    # Потолок вычислительной работы одного полотна:
    # частицы x итерации x шаги x подшаги. Стоимость симуляции ему прямо
    # пропорциональна (~0.8 млн/с), поэтому бюджет — это фактически лимит
    # секунд на сэмпл. Упирается только «тяжёлая» комбинация (плотная сетка +
    # долгое укладывание); при исчерпании сокращается число шагов, т.е.
    # полотно не успевает полностью улечься — визуально это чуть более живая
    # поза, а не артефакт.
    WORK_BUDGET = 1_800_000

    # То же для GPU. Ядра запускаются на всю сетку разом, поэтому бюджет
    # ограничивает уже не частицы, а число ЗАПУСКОВ; порядок подобран так,
    # чтобы тяжёлый сэмпл укладывался примерно в те же доли секунды.
    WORK_BUDGET_GPU = 400_000_000

    # Во сколько раз меш для РЕНДЕРА мельче сетки симуляции. Складки ткани
    # гладкие, поэтому подразбиение бикубическим сплайном даёт силуэт, для
    # которого иначе пришлось бы считать сетку в SUBDIVISION^2 раз дороже.
    # Важно и для маски сегментации: гранёный край выдаёт синтетику.
    #
    # Сплайн — костыль под грубую сетку: он сглаживает силуэт, но НЕ добавляет
    # складок, которых в симуляции не было. Поэтому подразбиение считается от
    # фактического размера сетки (см. _subdivision_for): на GPU сетка сама
    # достаточно плотная, и множитель падает до 1-2.
    SUBDIVISION = 3

    # Сторона меша рендера, к которой стремится подразбиение.
    RENDER_SIDE = 190

    # Барьер коллизии в долях шага сетки. У облака точек он не может быть
    # тоньше шага посева, иначе ткань утекает между точками. У BVH расстояние
    # точное, но барьер всё равно не должен быть СЛИШКОМ тонким: сетка на GPU
    # втрое плотнее, и барьер в тех же долях ячейки оказался бы втрое тоньше
    # борта, а такой ткань пробивает численным шумом.
    THICKNESS_CELLS = 0.75
    THICKNESS_CELLS_GPU = 1.2

    # Режимы размещения и их веса при случайном выборе. Основной случай —
    # опорный (референс): небольшой тент на ближней к кабине стенке.
    # «Полотно во весь кадр» оставлено редким: как штатный вид оно слишком
    # закрывает сцену.
    PLACEMENTS = {
        "near_wall": 0.62,   # через ближнюю (к кабине/камере) стенку
        "any_rail": 0.28,    # через произвольный борт
        "full_frame": 0.10,  # у самой камеры, закрывает весь кадр
    }

    def __init__(self, panda_app):
        self.app = panda_app
        self.node = None
        self.last_params = None
        self._tri_cache = {}

    # -- геометрия кузова ---------------------------------------------------
    def _cuzov_bounds(self):
        """AABB кузова в мировых координатах или None."""
        node = self._cuzov_node()
        if node is None:
            return None
        try:
            bounds = node.getTightBounds()
        except Exception:
            return None
        if not bounds:
            return None
        lo, hi = bounds
        return (np.array([lo[0], lo[1], lo[2]], dtype=np.float64),
                np.array([hi[0], hi[1], hi[2]], dtype=np.float64))

    def _cuzov_node(self):
        app = self.app
        cuzov_path = getattr(app, "current_cuzov_path", None)
        model_paths = getattr(app, "model_paths", {}) or {}
        for node in (getattr(app, "loaded_models", []) or []):
            if node is None or node.is_empty():
                continue
            if cuzov_path and model_paths.get(id(node)) == cuzov_path:
                return node
        return None

    # -- раскладка полотна --------------------------------------------------
    def _rails(self, bmin, bmax):
        """Четыре верхних кромки борта: (середина, вдоль, наружу, длина).

        `наружу` — горизонтальная нормаль от центра кузова, поэтому полотно
        всегда знает, где «снаружи», а где «внутрь кузова».
        """
        top_z = bmax[2]
        out = []
        for axis in (0, 1):
            other = 1 - axis
            along = np.zeros(3); along[other] = 1.0
            length = bmax[other] - bmin[other]
            for sign in (1.0, -1.0):
                outward = np.zeros(3); outward[axis] = sign
                mid = np.zeros(3)
                mid[axis] = bmax[axis] if sign > 0 else bmin[axis]
                mid[other] = 0.5 * (bmin[other] + bmax[other])
                mid[2] = top_z
                out.append((mid, along, outward, length))
        return out

    def _camera_pos(self):
        try:
            p = self.app.cam.getPos(self.app.render)
            return np.array([p[0], p[1], p[2]], dtype=np.float64)
        except Exception:
            return None

    def _layout(self, placement, rng, cuzov_verts=None):
        """Стартовая сетка + маска закреплённых точек + препятствия.

        Возвращает dict с полями grid, rest, pinned, ground_z, wind_dir,
        span (характерный размер — от него берутся сила ветра и шаг сетки).
        """
        bounds = self._cuzov_bounds()
        if bounds is None:
            return None

        if placement == "full_frame":
            return self._layout_full_frame(rng, bounds)

        bmin, bmax = bounds
        rails = self._rails(bmin, bmax)

        if placement == "near_wall":
            # Опорный случай (см. референс): тент лежит на ближней к кабине
            # стенке. «Ближняя к кабине» = ближняя к камере — камера в этом
            # проекте и смотрит из кабины в кузов. Так тент гарантированно
            # попадает в кадр, а не прячется за дальним бортом.
            cam = self._camera_pos()
            if cam is not None:
                rail = min(rails, key=lambda r: np.linalg.norm(r[0] - cam))
            else:
                rail = rails[int(rng.integers(0, len(rails)))]
        else:
            rail = rails[int(rng.integers(0, len(rails)))]

        return self._layout_rail(rail, bounds, rng, placement, cuzov_verts)

    def _layout_rail(self, rail, bounds, rng, placement, cuzov_verts=None):
        """Тент, переброшенный через кромку борта.

        Раскладка идёт снизу-снаружи -> вверх по наружной стороне -> через
        кромку -> вниз ВНУТРЬ кузова. Внутренняя часть — основная (как на
        референсе), снаружи свисает лишь край.

        Ткань нигде не закрепляется жёстко: её держит перегиб через кромку и
        трение о борт — именно поэтому нужна коллизия с реальной геометрией,
        а не с AABB.
        """
        mid, along, outward, rail_len = rail
        bmin, bmax = bounds
        size = bmax - bmin
        height = max(size[2], _EPS)

        # Реальная толщина борта — тент ПЕРЕКРЫВАЕТ её, а не протыкает.
        wall = self._wall_thickness(cuzov_verts, rail, bounds)

        # Тент вписывается в ВНУТРЕННИЙ проём кузова, а не в габарит по AABB:
        # свисающая часть висит между боковыми стенками, и полотно шириной во
        # весь габарит въезжает краями в них. Проём = габарит минус толщина
        # двух перпендикулярных стенок (каждая мерится по мешу отдельно —
        # борта у моделей несимметричны).
        axis_along = int(np.argmax(np.abs(along)))
        side_walls = [r for r in self._rails(bmin, bmax)
                      if int(np.argmax(np.abs(r[2]))) == axis_along]
        inset_lo = inset_hi = 0.0
        for r in side_walls:
            w = self._wall_thickness(cuzov_verts, r, bounds)
            if r[2][axis_along] > 0:
                inset_hi = w
            else:
                inset_lo = w
        interior_lo = bmin[axis_along] + inset_lo
        interior_hi = bmax[axis_along] - inset_hi
        usable = max(interior_hi - interior_lo, _EPS)

        # Размеры с референса: тент заметно УЖЕ борта и не достаёт до земли.
        width = usable * float(rng.uniform(0.28, 0.62))
        drop_in = height * float(rng.uniform(0.45, 1.1))    # внутрь кузова
        drop_out = height * float(rng.uniform(0.12, 0.45))  # наружу

        gather = float(rng.uniform(1.10, 1.55))
        rest_width = width * gather

        # Сдвиг вдоль кромки — тент не обязан быть по центру борта, но целиком
        # остаётся в проёме. Место резервируется по REST_WIDTH, а не по
        # стартовой ширине: материала в тенте на gather больше, чем пролёт, и
        # избыток уходит не только в складки — под ветром полотно РАСПРАВЛЯЕТСЯ
        # и достаёт до перпендикулярных стенок, вдавливаясь в них краем.
        # Незнаковая коллизия из толщи стенки уже не вытаскивает.
        span_needed = min(rest_width, usable)
        centre = float(rng.uniform(interior_lo + span_needed * 0.5,
                                   interior_hi - span_needed * 0.5))
        offset = centre - mid[axis_along]

        # Ширина перехода через торец завязана на шаг сетки, а шаг — на полную
        # длину полотна, которая включает переход. Цикл разрываем оценкой: шаг
        # считаем по длине с голой толщиной борта, затем фиксируем переход и
        # пересчитываем сетку под итоговую длину.
        # Плотность заметно выше прежней: складки тента должны быть гладкими,
        # а не гранёными. Ниже ещё идёт сглаживающее подразбиение под рендер.
        density = (rng.uniform(95.0, 145.0) if warp_available()
                   else rng.uniform(44.0, 62.0))
        est = drop_in + drop_out + max(wall, _EPS)
        cell_est = max(rest_width, est) / density

        # Переход не может быть уже пары ячеек: борт бывает тоньше шага сетки,
        # тогда «полка» вырождается в один ряд, сглаживание ниже её срезает, и
        # ЗАКРЕПЛЁННЫЙ ряд перегиба проваливается в толщу борта (коллизия
        # якоря не спасала). Ткань и в жизни ложится на кромку с напуском.
        cross_w = max(wall, cell_est * 2.5)
        total = drop_in + drop_out + cross_w
        nx, ny, cell = self._grid_dims(rest_width, total, cell_est)

        u = np.linspace(-width * 0.5, width * 0.5, nx) + offset
        v = np.linspace(0.0, total, ny)
        uu, vv = np.meshgrid(u, v)

        # Профиль поперёк кромки: сначала наружная стенка, потом перегиб,
        # потом внутренняя. Небольшой зазор от борта — старт не в геометрии.
        # Профиль поперёк борта из трёх участков (как настоящий тент):
        #   v < drop_out          — свисает СНАРУЖИ вдоль наружной грани;
        #   drop_out .. +wall     — лежит на торце борта, пересекая толщину;
        #   дальше                — свисает ВНУТРЬ вдоль внутренней грани.
        # lateral отсчитывается от НАРУЖНОЙ грани: + наружу, − внутрь кузова.
        # Зазор до борта должен быть заметно больше барьера коллизии
        # (thickness = cell*0.75), иначе ткань стартует уже в барьере.
        # Зазор НЕ может зависеть только от шага сетки. Незнаковая коллизия не
        # вытаскивает частицу из толщи борта (ей неоткуда узнать, где «снаружи»
        # у открытой оболочки), поэтому раскладка обязана стартовать заведомо
        # снаружи. Сетка на GPU втрое плотнее, и зазор в долях ячейки съёжился
        # бы втрое — ткань начинала бы внутри стенки и там и оставалась.
        # Поэтому есть второй, независимый от плотности предел: доля борта.
        clear = max(cell * 1.6, wall * 0.4)
        cross0, cross1 = drop_out, drop_out + cross_w
        lateral = np.where(
            vv < cross0,
            clear,
            np.where(vv > cross1,
                     -(wall + clear),
                     clear - (wall + 2.0 * clear)
                     * (vv - cross0) / max(cross_w, _EPS)))
        z = np.where(
            vv < cross0, bmax[2] - (cross0 - vv),
            np.where(vv > cross1, bmax[2] - (vv - cross1), bmax[2] + clear))

        # Острые углы на кромке сглаживаем вдоль v: настоящая ткань ложится на
        # торец борта скруглённо, а не ломается под 90°.
        try:
            from scipy.ndimage import gaussian_filter1d
            sigma = max(1.0, 1.5 * (ny - 1) / max(total / cell, 1.0))
            lateral = gaussian_filter1d(lateral, sigma, axis=0, mode="nearest")
            z = gaussian_filter1d(z, sigma, axis=0, mode="nearest")
        except Exception:
            pass

        grid = (mid[None, None, :]
                + along[None, None, :] * uu[..., None]
                + outward[None, None, :] * lateral[..., None])
        grid[..., 2] = z

        # Затравка складок — поперёк борта. Амплитуда СТРОГО меньше зазора до
        # борта: иначе затравка сама вгоняет ткань в стенку, а незнаковая
        # коллизия из толщи стенки уже не вытаскивает.
        # Складка растёт от перегиба к обоим свободным краям, поэтому глубина
        # отсчитывается от кромки, а не от края полотна.
        free = np.abs(vv - (drop_out + cross_w * 0.5))
        depth = free / max(free.max(), _EPS)
        grid += outward[None, None, :] * self._buckle_seed(
            nx, ny, clear * 0.35, depth, rng)[..., None]

        rest = self._rest_sheet(nx, ny, rest_width, total)

        # Закрепляем только узкую полосу на самом перегибе: физически тент
        # держится именно кромкой. Всё остальное свободно и ложится складками.
        pinned = np.zeros((ny, nx), dtype=bool)
        crease = int(np.argmin(np.abs(v - (drop_out + cross_w * 0.5))))
        pinned[crease, :] = True
        self._punch_pins(pinned, rng, row=crease)

        wind_dir = outward * float(rng.uniform(-0.6, 1.0)) \
            + along * float(rng.uniform(-0.5, 0.5))
        wind_dir[2] = float(rng.uniform(-0.1, 0.3))

        return {
            "grid": grid,
            "rest": rest,
            "pinned": pinned,
            "ground_z": self._ground_z(bmin),
            "wind_dir": wind_dir,
            "span": max(width, total),
            "cell": cell,
            "gather": gather,
        }

    def _layout_full_frame(self, rng, bounds):
        """Полотно перед камерой — закрывает весь кадр или его часть.

        Крепление — горизонтальная линия выше верхней кромки кадра, полотно
        падает поперёк обзора. Ширина/длина берутся из FOV, поэтому режим
        работает при любой позе камеры.
        """
        app = self.app
        cam = app.cam
        render = app.render
        pos = cam.getPos(render)
        cam_pos = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)

        quat = cam.getQuat(render)
        fwd = np.array(list(quat.getForward()), dtype=np.float64)
        right = np.array(list(quat.getRight()), dtype=np.float64)
        up = np.array(list(quat.getUp()), dtype=np.float64)

        lens = cam.node().getLens()
        try:
            fov_x, fov_y = lens.getFov()[0], lens.getFov()[1]
        except Exception:
            fov_x, fov_y = 45.0, 30.0

        bmin, bmax = bounds
        # Дистанция до полотна: между камерой и кузовом.
        to_truck = np.linalg.norm((bmin + bmax) * 0.5 - cam_pos)
        dist = to_truck * rng.uniform(0.25, 0.6)

        half_w = dist * math.tan(math.radians(fov_x * 0.5))
        half_h = dist * math.tan(math.radians(fov_y * 0.5))

        # Запас, чтобы края полотна уходили за кадр (иначе видно кромку).
        width = 2.0 * half_w * rng.uniform(1.15, 1.9)
        drop = 2.0 * half_h * rng.uniform(1.3, 2.4)

        gather = float(rng.uniform(1.06, 1.7))
        rest_width = width * gather

        cell = max(rest_width, drop) / (rng.uniform(70.0, 110.0)
                                        if warp_available()
                                        else rng.uniform(24.0, 34.0))
        nx, ny, cell = self._grid_dims(rest_width, drop, cell)

        centre = cam_pos + fwd * dist + up * (half_h * rng.uniform(0.9, 1.25))
        u = np.linspace(-width * 0.5, width * 0.5, nx)
        v = np.linspace(0.0, drop, ny)
        uu, vv = np.meshgrid(u, v)

        grid = (centre[None, None, :]
                + right[None, None, :] * uu[..., None]
                + up[None, None, :] * (-vv[..., None]))
        # Небольшой наклон к камере — полотно «надувается» в объектив.
        grid -= fwd[None, None, :] * (0.15 * vv[..., None])
        # Складки выпучиваются вдоль оси обзора (см. _buckle_seed). Преград
        # рядом нет, поэтому амплитуду ограничивает только вид.
        grid -= fwd[None, None, :] * self._buckle_seed(
            nx, ny, cell * 0.8, vv / max(drop, _EPS), rng)[..., None]

        pinned = np.zeros((ny, nx), dtype=bool)
        pinned[0, :] = True
        self._punch_pins(pinned, rng)

        # Ветер преимущественно на камеру.
        wind_dir = -fwd + right * rng.uniform(-0.6, 0.6) + up * rng.uniform(-0.2, 0.4)

        return {
            "grid": grid,
            "rest": self._rest_sheet(nx, ny, rest_width, drop),
            "pinned": pinned,
            "boxes": [],          # у камеры кузов не мешает
            "ground_z": None,
            "wind_dir": wind_dir,
            "span": max(width, drop),
            "cell": cell,
            "gather": gather,
        }

    @staticmethod
    def _rest_sheet(nx, ny, width, drop):
        """Недеформированное полотно width x drop в СВОЁЙ плоскости.

        Реальная поза тут ни при чём: из этой сетки берутся только ДЛИНЫ
        связей, поэтому система координат может быть любой. Зато длины
        получаются строго равномерными — ткань не наследует растяжений и
        перекосов стартовой раскладки.
        """
        rest = np.zeros((ny, nx, 3), dtype=np.float64)
        ru, rv = np.meshgrid(np.linspace(0.0, width, nx),
                             np.linspace(0.0, drop, ny))
        rest[..., 0] = ru
        rest[..., 2] = -rv
        return rest

    @staticmethod
    def _buckle_seed(nx, ny, amplitude, depth, rng):
        """Гладкая поперечная затравка складок.

        `depth` (0..1) — насколько точка свободна: 0 у крепления, 1 у дальнего
        свободного края. Складки растут от крепления, поэтому амплитуда на нём
        нулевая. Пара мод со случайными фазами даёт неровный шаг складок
        вместо гофры.

        Сумма мод НОРМИРУЕТСЯ, иначе амплитуда зависит от их числа: именно
        так затравка когда-то превышала зазор до борта и загоняла ткань внутрь
        стенки ещё до первого шага симуляции. Амплитуда — жёсткий контракт,
        вызывающая сторона обязана держать её меньше зазора.
        """
        j = np.linspace(0.0, 1.0, nx)[None, :]
        seed = np.zeros((ny, nx), dtype=np.float64)
        weight = 0.0
        for _ in range(int(rng.integers(2, 5))):
            modes = float(rng.integers(2, 7))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            w = float(rng.uniform(0.4, 1.0))
            seed += w * np.sin(modes * math.pi * j + phase)
            weight += w
        seed /= max(weight, _EPS)            # |seed| <= 1
        return amplitude * seed * depth

    def _grid_dims(self, width, drop, cell):
        """Размеры сетки под ячейку `cell`, с укрупнением ячейки при выходе
        за лимит частиц (пропорции полотна сохраняются)."""
        gpu = warp_available()
        limit = self.MAX_PARTICLES_GPU if gpu else self.MAX_PARTICLES
        side = self.MAX_SIDE_GPU if gpu else self.MAX_SIDE

        cell = max(float(cell), _EPS)
        nx = int(np.clip(round(width / cell) + 1, 8, side))
        ny = int(np.clip(round(drop / cell) + 1, 8, side))
        if nx * ny > limit:
            k = math.sqrt(limit / float(nx * ny))
            nx = max(8, int(nx * k))
            ny = max(8, int(ny * k))
        cell = max(width / max(nx - 1, 1), drop / max(ny - 1, 1))
        return nx, ny, cell

    @staticmethod
    def _punch_pins(pinned, rng, row=0):
        """Разрежаем линию крепления: сплошной ряд даёт скучную «штору».

        Варианты повторяют то, как тент реально держится: сплошная кромка,
        редкие люверсы, пара углов, оторванный край. `row` — индекс ряда с
        креплением (у тента через борт это перегиб, а не верхняя кромка).
        """
        nx = pinned.shape[1]
        pinned_row = pinned[row]
        mode = rng.choice(
            ["full", "grommets", "corners", "torn"],
            p=[0.30, 0.34, 0.20, 0.16])

        if mode == "full":
            return
        if mode == "grommets":
            step = int(rng.integers(3, 7))
            keep = np.zeros(nx, dtype=bool)
            keep[::step] = True
            keep[0] = keep[-1] = True
            pinned_row[~keep] = False
        elif mode == "corners":
            keep = np.zeros(nx, dtype=bool)
            edge = max(1, nx // 10)
            keep[:edge] = True
            keep[-edge:] = True
            pinned_row[~keep] = False
        elif mode == "torn":
            # Крепление обрывается — свободный угол уходит по ветру.
            cut = int(rng.integers(nx // 3, max(nx // 3 + 1, (2 * nx) // 3)))
            if rng.random() < 0.5:
                pinned_row[cut:] = False
            else:
                pinned_row[:cut] = False

        if not pinned_row.any():         # страховка: хоть одна точка держит
            pinned_row[0] = True

    def _ground_z(self, bmin):
        plane = getattr(self.app, "ground_plane", None)
        if plane is not None and not plane.is_empty():
            try:
                return float(plane.getZ(self.app.render))
            except Exception:
                pass
        return float(bmin[2])

    # -- сбор геометрии для коллизий ----------------------------------------
    def _triangles(self, node, cache_key=None):
        """Треугольники узла в мире, с кэшем по ключу.

        Кузов перезагружается на каждый сэмпл (clear_scene + load_gltf_model),
        поэтому кэшировать по id узла бесполезно — ключом служит путь модели
        плюс её трансформ. Груз генерируется заново каждый раз и не кэшируется.
        """
        if node is None or node.is_empty():
            return None

        if cache_key is not None:
            hit = self._tri_cache.get(cache_key)
            if hit is not None:
                return hit

        try:
            data = collect_world_triangles(node)
        except Exception as exc:
            print(f"[Cloth] не удалось прочитать геометрию: {exc}")
            return None

        if cache_key is not None and data is not None:
            self._tri_cache.clear()      # держим только текущий кузов
            self._tri_cache[cache_key] = data
        return data

    def _collect_meshes(self):
        """(меш кузова | None, [все меши для коллизий]).

        Меш кузова нужен отдельно: по нему измеряется РЕАЛЬНАЯ толщина борта
        (см. _wall_thickness), без которой раскладка тента попадает внутрь
        стенки.
        """
        meshes = []

        cuzov_mesh = None
        cuzov = self._cuzov_node()
        if cuzov is not None:
            key = None
            try:
                path = getattr(self.app, "current_cuzov_path", None)
                if path:
                    mat = cuzov.get_net_transform().get_mat()
                    key = (path, tuple(round(mat.get_cell(r, c), 4)
                                       for r in range(4) for c in range(4)))
            except Exception:
                key = None
            cuzov_mesh = self._triangles(cuzov, key)
            if cuzov_mesh is not None:
                meshes.append(cuzov_mesh)

        # Груз: тент на референсе частично лежит на насыпи, поэтому её тоже
        # нужно учитывать. Меш каждый раз новый — кэш неприменим.
        cargo = self._triangles(getattr(self.app, "final_model", None), None)
        if cargo is not None:
            meshes.append(cargo)

        return cuzov_mesh, meshes

    @staticmethod
    def _make_collider(meshes, thickness, region, gpu=False):
        """Коллайдер вокруг рабочей зоны ткани.

        На GPU это BVH по настоящим треугольникам — ни бюджета точек, ни
        просветов. Запасной путь сеет поверхность точками, и там шаг посева
        ОБЯЗАН быть меньше thickness: иначе между точками остаётся щель шире
        барьера, и полотно просачивается сквозь борт.
        """
        if not meshes:
            return None
        collider = (WarpMeshCollider() if gpu
                    else MeshCollider(spacing=max(thickness * 0.6, _EPS)))
        for verts, faces in meshes:
            collider.add_mesh(verts, faces, region=region)
        return collider if collider.build() else None

    @staticmethod
    def _wall_thickness(verts, rail, bounds):
        """Толщина борта у выбранной кромки, измеренная по мешу.

        AABB знает только НАРУЖНУЮ грань борта. Если отложить «внутрь» от неё
        малый зазор, ткань стартует внутри стенки — а незнаковая коллизия из
        такого положения уже не вытолкнет (ей всё равно, с какой стороны).
        Поэтому реальная внутренняя грань ищется по вершинам меша у этой
        кромки; квантиль вместо минимума отсекает случайный мусор геометрии.
        """
        bmin, bmax = bounds
        size = bmax - bmin
        mid, along, outward, _ = rail
        axis = int(np.argmax(np.abs(outward)))
        sign = float(np.sign(outward[axis]))
        outer = mid[axis]

        fallback = float(size[axis]) * 0.04
        if verts is None or len(verts) == 0:
            return fallback

        near_wall = np.abs(verts[:, axis] - outer) < 0.3 * size[axis]
        upper = verts[:, 2] > bmax[2] - 0.6 * size[2]
        sel = near_wall & upper
        if sel.sum() < 8:
            return fallback

        # depth <= 0 — насколько вершина уходит ВНУТРЬ от наружной грани.
        depth = (verts[sel, axis] - outer) * sign
        thickness = float(-np.quantile(depth, 0.02))
        return float(np.clip(thickness, fallback * 0.25, 0.25 * size[axis]))

    # -- публичный API ------------------------------------------------------
    def spawn_random(self, seed=None, placement=None):
        """Собрать случайное полотно и повесить его в сцену.

        Возвращает NodePath или None (если кузова нет / симуляция не удалась).
        Предыдущее полотно снимается автоматически.
        """
        self.clear()

        rng = np.random.default_rng(
            seed if seed is not None else random.randrange(2 ** 31))

        if placement is None:
            names = list(self.PLACEMENTS.keys())
            weights = np.array([self.PLACEMENTS[n] for n in names])
            placement = str(rng.choice(names, p=weights / weights.sum()))

        # Геометрия читается ОДИН раз: раскладке она нужна для замера толщины
        # борта, коллайдеру — как препятствие.
        cuzov_mesh, meshes = self._collect_meshes()
        cuzov_verts = cuzov_mesh[0] if cuzov_mesh is not None else None

        layout = self._layout(placement, rng, cuzov_verts)
        if layout is None:
            print("[Cloth] кузов не найден — полотно не создано")
            return None

        gpu = warp_available()

        # Жёсткость: от мягкого брезента до почти негнущегося полога.
        stiff_bend = float(rng.uniform(0.05, 0.45))
        common = dict(
            rest_positions=layout["rest"],
            stiff_structural=1.0,
            stiff_shear=float(rng.uniform(0.5, 0.9)),
            stiff_bend=stiff_bend,
            damping=float(rng.uniform(0.02, 0.09)),
        )
        cell = layout["cell"]
        if gpu:
            # У XPBD итерации внутри подшага почти не нужны: точность даёт
            # дробление шага (см. WarpClothSolver.SUBSTEPS).
            solver = WarpClothSolver(layout["grid"], layout["pinned"],
                                     iterations=3, cell=cell, **common)
        else:
            solver = ClothSolver(layout["grid"], layout["pinned"],
                                 iterations=int(rng.integers(4, 7)), **common)

        # Коллизия с НАСТОЯЩЕЙ геометрией кузова и груза. У BVH барьер может
        # быть тонким (расстояние точное), у облака точек — не тоньше шага
        # посева, иначе ткань просачивается между точками поверхности.
        thickness = cell * (self.THICKNESS_CELLS_GPU if gpu
                            else self.THICKNESS_CELLS)

        # Рабочая зона: стартовая раскладка плюс запас на провис и раскачку.
        # Полотно закреплено, поэтому дальше своей длины от раскладки уйти не
        # может — по ней и берётся запас. Зона входит в стоимость квадратично
        # (площадь засеваемой геометрии), так что лишний запас дорог.
        grid = layout["grid"]
        margin = layout["span"] * 0.35
        region = (grid.reshape(-1, 3).min(axis=0) - margin,
                  grid.reshape(-1, 3).max(axis=0) + margin)

        collider = self._make_collider(meshes, thickness, region, gpu=gpu)
        if collider is not None:
            solver.set_mesh_collider(collider, thickness=thickness)
        elif not meshes:
            # Геометрию вообще не удалось прочитать — грубый запасной вариант.
            print("[Cloth] сетка кузова недоступна, откат на AABB")
            for bmin, bmax in layout.get("boxes", []):
                solver.add_box(bmin, bmax, margin=cell * 0.35)
        # Иначе рядом с полотном просто нет геометрии (например, режим
        # full_frame у камеры) — это норма, преград не нужно.

        if layout["ground_z"] is not None:
            solver.set_ground(layout["ground_z"] + cell * 0.2)
        solver.friction = float(rng.uniform(0.25, 0.6))

        # Ветер в долях g — не зависит от масштаба сцены. Штиль (тяжёлый
        # провис) и шквал (полотно почти горизонтально) — обе крайности нужны.
        wind_state = str(rng.choice(
            ["calm", "breeze", "gusty", "strong"], p=[0.28, 0.30, 0.24, 0.18]))
        wind_scale = {
            "calm": 0.0, "breeze": 0.35, "gusty": 0.7, "strong": 1.25,
        }[wind_state]
        g = 9.81
        solver.set_wind(
            layout["wind_dir"],
            speed=math.sqrt(max(layout["span"], _EPS) * g) * wind_scale,
            gust=float(rng.uniform(0.15, 0.6)) if wind_scale else 0.0,
            turbulence=float(rng.uniform(0.2, 0.7)) if wind_scale else 0.0,
            drag=float(rng.uniform(0.9, 2.2)),
        )

        # Сколько шагов «отпустить» ткань. Мало шагов = пойманное на лету
        # полоскание, много = улёгшийся тяжёлый провис.
        if wind_scale > 0.0 and rng.random() < 0.5:
            steps = int(rng.integers(45, 110))     # ловим движение
        else:
            steps = int(rng.integers(160, 340))    # даём улечься

        substeps = WarpClothSolver.SUBSTEPS if gpu else 2
        particles = solver.nx * solver.ny
        budget = self.WORK_BUDGET_GPU if gpu else self.WORK_BUDGET
        max_steps = budget // max(
            particles * solver.iterations * substeps, 1)
        steps = int(np.clip(steps, 30, max(30, max_steps)))

        # Масштабируем шаг по времени под размер сцены: dt подобран для
        # «человеческих» метров, а сцена может быть в других единицах.
        dt = (1.0 / 120.0) * math.sqrt(max(layout["span"], _EPS) / 3.0)
        final = solver.settle(steps, dt=dt, substeps=substeps)

        if not np.all(np.isfinite(final)):
            print("[Cloth] симуляция разошлась — полотно пропущено")
            return None

        node = self._build_node(final, solver.normals(), rng, layout)
        self.node = node
        self.last_params = {
            "solver": "warp" if gpu else "numpy",
            "placement": placement,
            "wind": wind_state,
            "steps": steps,
            "resolution": [int(solver.nx), int(solver.ny)],
            "pinned_points": int(layout["pinned"].sum()),
            "bend_stiffness": round(stiff_bend, 3),
            "gather": round(float(layout["gather"]), 3),
        }
        return node

    def clear(self):
        if self.node is not None and not self.node.is_empty():
            self.node.remove_node()
        self.node = None

    # -- меш ----------------------------------------------------------------
    @classmethod
    def _subdivision_for(cls, ny, nx):
        """Множитель подразбиения под сетку (ny, nx) -> 1..SUBDIVISION."""
        side = max(ny, nx)
        return int(np.clip(cls.RENDER_SIDE // max(side, 1), 1, cls.SUBDIVISION))

    def _build_node(self, grid, normals, rng, layout):
        # Меш для рендера плотнее сетки симуляции: складки становятся
        # гладкими, а силуэт в маске — без «лесенки» по краю.
        grid = subdivide_grid(grid, self._subdivision_for(*grid.shape[:2]))
        normals = grid_normals(grid)
        ny, nx = grid.shape[:2]

        # UV: тайлинг привязан к физическому размеру, чтобы плотность нити
        # не зависела от размера полотна. Поворот/сдвиг — против повторов.
        tile = max(layout["span"], _EPS) / rng.uniform(0.6, 2.2)
        u = np.linspace(0.0, 1.0, nx) * tile
        v = np.linspace(0.0, float(ny) / max(nx, 1) * tile, ny)
        uu, vv = np.meshgrid(u, v)
        uu = uu + rng.uniform(0.0, 10.0)
        vv = vv + rng.uniform(0.0, 10.0)

        # Буферы заполняются ОДНИМ блоком памяти, а не построчными writer'ами:
        # сетка тента — это десятки тысяч вершин, и цикл на Python по ним
        # стоил бы больше, чем вся симуляция на GPU. Формат v3n3t2 —
        # чередование 8 float32 на вершину, ровно как в массиве ниже.
        fmt = GeomVertexFormat.getV3n3t2()
        vdata = GeomVertexData("cloth", fmt, Geom.UHStatic)
        vdata.set_num_rows(nx * ny)

        interleaved = np.empty((nx * ny, 8), dtype=np.float32)
        interleaved[:, 0:3] = grid.reshape(-1, 3)
        interleaved[:, 3:6] = normals.reshape(-1, 3)
        interleaved[:, 6] = uu.reshape(-1)
        interleaved[:, 7] = vv.reshape(-1)
        vdata.modify_array(0).modify_handle().set_data(
            interleaved.tobytes())

        idx = np.arange(nx * ny).reshape(ny, nx)
        a = idx[:-1, :-1].reshape(-1)
        b = idx[1:, :-1].reshape(-1)
        c = idx[:-1, 1:].reshape(-1)
        d = idx[1:, 1:].reshape(-1)
        tris = np.empty((2 * a.size, 3), dtype=np.uint32)
        tris[0::2] = np.stack([a, b, c], axis=1)
        tris[1::2] = np.stack([b, d, c], axis=1)

        prim = GeomTriangles(Geom.UHStatic)
        # 32-битные индексы: при плотной сетке вершин заметно больше 65536,
        # а по умолчанию примитив берёт uint16 и индексы заворачиваются.
        prim.set_index_type(GeomEnums.NT_uint32)
        varray = prim.modify_vertices()
        varray.unclean_set_num_rows(tris.size)
        varray.modify_handle().set_data(tris.tobytes())
        prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        # Тангенты нужны нормал-мапе (см. MyApp._apply_pbr_surface).
        has_tangents = False
        gen = getattr(self.app, "_generate_tangents_for_geom", None)
        if gen is not None:
            try:
                gen(geom)
                has_tangents = True
            except Exception as exc:
                print(f"[Cloth] тангенты не построены: {exc}")

        gnode = GeomNode("cloth")
        gnode.addGeom(geom)
        node = self.app.render.attachNewNode(gnode)

        # Ткань видна с обеих сторон — она тонкая и часто повёрнута изнанкой.
        node.setTwoSided(True)

        apply_shader = getattr(self.app, "_apply_auto_shader", None)
        if apply_shader is not None:
            try:
                apply_shader(node)
            except Exception as exc:
                print(f"[Cloth] шейдер не применён: {exc}")

        apply_pbr = getattr(self.app, "_apply_pbr_surface", None)
        if apply_pbr is not None:
            try:
                apply_pbr(
                    node,
                    CLOTH_TEXTURES["diffuse"],
                    normal_path=CLOTH_TEXTURES["normal"],
                    roughness_path=CLOTH_TEXTURES["roughness"],
                    roughness=float(rng.uniform(0.6, 0.95)),
                    has_tangents=has_tangents,
                    # Лёгкий разброс тона — выцветание/загрязнение брезента.
                    base_color=(
                        float(rng.uniform(0.75, 1.0)),
                        float(rng.uniform(0.75, 1.0)),
                        float(rng.uniform(0.75, 1.0)),
                        1.0),
                )
            except Exception as exc:
                print(f"[Cloth] PBR-материал не применён: {exc}")

        return node
