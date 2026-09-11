# -*- coding: utf-8 -*-
"""
Четыре опорные точки верхнего проёма кузова, посчитанные по самому мешу.

Зачем это нужно
---------------
`points_3d` в конфиге модели — прямоугольник ВНУТРЕННИХ углов верхней кромки
кузова: по нему 2D→3D-реконструкция сажает облако точек в сцену (DOCS §8.1), и
он же служит масштабной линейкой прикладным потребителям конфига. На сервере
эти точки правит оператор руками, а при загрузке нового комплекта их надо чем-то
заполнить — вот этим.

Раньше заготовка бралась из габаритов наполнителя (`world_bounds.napolnitel` в
`<stem>.set.json`), и это было неверно сразу по трём осям:

* по Z — наполнитель нарочно продолжен ВВЕРХ на «горку» (метр по умолчанию,
  см. `body_builder/napolnitel.py`), так что верх его рамки на метр выше борта;
* по X — рамка берёт максимум полуширины по всей высоте полости, а у кузова с
  завалом внутрь (`side_flare_deg < 0`) на кромке полость уже, чем внизу;
* по Y — торцы завалены (у KAMAZ-6520 передний борт на 25°), и рамка по всей
  высоте длиннее проёма на кромке.

Что делает модуль
-----------------
Ищет проём геометрически, по мешу кузова, ничего не зная о генераторе:

1. **Высота кромки.** Меш режется горизонтальными плоскостями сверху вниз.
   Кромка — самая высокая плоскость, на которой сечение ещё ЗАМКНУТО, то есть
   в нём есть большая пустота, окружённая материалом со всех сторон. Выше
   кромки стенок уже нет: козырёк, задние стойки и петли пустоту не замыкают.
   Заодно это ровно то правило, по которому точки стоят на сервере, — если
   один борт выше другого (наращенная задняя створка, `features.rear_raise`),
   высота берётся по НИЗКОМУ борту.
2. **Грани проёма.** Из центра пустоты пускаются лучи по ±X и ±Y; первое
   пересечение — внутренняя грань борта.
3. **Экстраполяция на кромку.** Мерить прямо под кромкой нельзя: над проёмом
   нависают козырьки (у `ford_cargo_3536d_dc` козырёк съедает 20 см передней
   части проёма) и обвязка. Поэтому грань меряется на нескольких глубинах, по
   отсчётам проводится прямая (борт плоский, пусть и с завалом) и продолжается
   до кромки. Прямая выбирается по большинству отсчётов, а из нескольких
   подходящих — самая НАРУЖНАЯ: помеха может сделать проём только у́же, шире —
   никогда.

Проверено на всех 20 кузовах сервера (их меши качаются через
`TLS_client.download_model_file`): расхождение с точками, которые расставил
оператор, — медиана 9 см, максимум 14 см, при том что сами серверные точки
разбросаны на те же 5–8 см (у Shackman-X3000-6x4-standard Z четырёх точек
гуляет на 5 см, у MAZ-6x4 — на 8). На шести комплектах генератора результат
совпадает с прямым расчётом по `<stem>.spec.json` до четвёртого знака.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Шаг сканирования по высоте в поисках кромки, м. Мельче не надо, и дело не
#: только в скорости: ниже кромки полость замкнута непрерывно, до самого дна,
#: так что шагом её не проскочить, а точную границу всё равно ищет деление
#: пополам. Зато у цельной модели грузовика над кузовом ещё метр кабины, и
#: мелкий шаг вылился бы в сотню сечений вхолостую.
SCAN_STEP = 0.10

#: Сколько раз делим пополам. 0.10 / 2**12 — сотые доли миллиметра.
BISECT_STEPS = 12

#: Пустота считается проёмом кузова, если занимает столько от габарита меша по
#: XY. Отсекает мелкие замкнутые карманы над кромкой — коробку козырька,
#: полость задней стойки.
MIN_VOID_FRACTION = 0.25

#: Клетка растра сечения, м. По ней ищется замкнутая пустота; сами грани
#: меряются лучами и от растра не зависят.
CELL = 0.01

#: На каких глубинах под кромкой меряется грань борта, м. Верхние отсчёты
#: попадают под козырёк и обвязку, нижние — в скругление дна; прямая строится
#: по большинству, так что и те и другие отбрасываются.
PROBE_DEPTHS = (0.03, 0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.50, 0.65)

#: Допуск на попадание отсчёта в прямую борта, м.
FIT_TOLERANCE = 0.005

#: Стороны проёма в порядке (ось, направление): x_lo, x_hi, y_lo, y_hi.
_SIDES: Tuple[Tuple[int, int], ...] = ((0, -1), (0, +1), (1, -1), (1, +1))


@dataclass
class Opening:
    """Найденный проём: высота кромки и прямоугольник внутренних граней."""

    z: float
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    #: Сколько сечений пригодилось для экстраполяции граней.
    probes: int = 0
    #: Площадь замкнутой пустоты на кромке, м² — для строчки в журнале диалога.
    area: float = 0.0

    @property
    def width(self) -> float:
        return self.x_hi - self.x_lo

    @property
    def length(self) -> float:
        return self.y_hi - self.y_lo

    def points(self) -> List[List[float]]:
        """
        Четыре точки в том порядке, в каком их держит сервер: по часовой
        стрелке от дальнего правого угла, все на одной высоте.
        """
        z = round(float(self.z), 4)
        x0, x1 = round(float(self.x_lo), 4), round(float(self.x_hi), 4)
        y0, y1 = round(float(self.y_lo), 4), round(float(self.y_hi), 4)
        return [[x1, y1, z], [x0, y1, z], [x0, y0, z], [x1, y0, z]]


# ---------------------------------------------------------------------------
# Чтение меша
# ---------------------------------------------------------------------------
def read_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вершины (N, 3) и треугольники (M, 3) из `.bam` или `.obj`.

    Координаты мировые: у .bam трансформации узлов сворачиваются в вершины
    (у комплектов генератора части сдвинуты — см. `offsets` в `.set.json`),
    у .obj они и так мировые.
    """
    if path.lower().endswith(".obj"):
        return _read_obj(path)
    return _read_bam(path)


def _read_bam(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Меш из .bam. Вершины читаются сырым буфером, а не `GeomVertexReader`.

    Разница не косметическая: в цельных моделях грузовиков из
    `assets/models/trucks` по полтора миллиона треугольников, и пословный
    обход на Python занимает секунды, тогда как `np.frombuffer` по тому же
    массиву — миллисекунды. Если формат вершин окажется незнакомым, читаем
    по-старому: правильность важнее скорости.
    """
    from panda3d.core import (Filename, GeomVertexReader, InternalName,
                              Loader, LoaderOptions, NodePath)

    node = Loader.get_global_ptr().load_sync(
        Filename.from_os_specific(os.path.abspath(path)), LoaderOptions())
    if node is None:
        raise RuntimeError(f"Panda3D не смог прочитать {path}")
    root = NodePath(node)

    chunks: List[np.ndarray] = []
    tris: List[np.ndarray] = []
    total = 0
    for geom_np in root.find_all_matches("**/+GeomNode"):
        geom_node = geom_np.node()
        # Трансформация узла относительно корня: у комплектов генератора части
        # сдвинуты (см. `offsets` в .set.json), и без неё кузов уедет.
        mat = geom_np.get_transform(root).get_mat()
        xform = np.array([[mat.get_cell(r, c) for c in range(4)]
                          for r in range(4)], dtype=np.float64)
        for i in range(geom_node.get_num_geoms()):
            geom = geom_node.get_geom(i).decompose()
            vdata = geom.get_vertex_data()
            local = _vertex_array(vdata, InternalName.get_vertex())
            if local is None:
                reader = GeomVertexReader(vdata, "vertex")
                rows: List[Tuple[float, float, float]] = []
                while not reader.is_at_end():
                    point = reader.get_data3()
                    rows.append((point.x, point.y, point.z))
                local = np.array(rows, dtype=np.float64).reshape(-1, 3)
            if not len(local):
                continue
            world = local @ xform[:3, :3] + xform[3, :3]

            base = total
            total += len(world)
            chunks.append(world)
            for p in range(geom.get_num_primitives()):
                prim = geom.get_primitive(p)
                if prim.get_num_vertices_per_primitive() != 3:
                    continue
                idx = _index_array(prim)
                if idx is None or len(idx) < 3:
                    continue
                tris.append(idx[: len(idx) // 3 * 3].reshape(-1, 3) + base)

    verts = (np.concatenate(chunks) if chunks
             else np.zeros((0, 3), dtype=np.float64))
    faces = (np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64))
    return verts, faces.astype(np.int64, copy=False)


#: Числовой тип колонки Panda3D -> тип numpy. Собирается по именам констант, а
#: не по их значениям: значения — деталь реализации движка, а имена стабильны.
_NUMERIC_DTYPES: Dict[int, Any] = {}


def _numeric_dtype(value: int) -> Optional[Any]:
    if not _NUMERIC_DTYPES:
        from panda3d.core import GeomEnums
        for name, dtype in (("NT_uint8", np.uint8), ("NT_uint16", np.uint16),
                            ("NT_uint32", np.uint32), ("NT_int8", np.int8),
                            ("NT_int16", np.int16), ("NT_int32", np.int32),
                            ("NT_float32", np.float32),
                            ("NT_float64", np.float64)):
            code = getattr(GeomEnums, name, None)
            if code is not None:
                _NUMERIC_DTYPES[int(code)] = dtype
    return _NUMERIC_DTYPES.get(int(value))


def _vertex_array(vdata, name) -> Optional[np.ndarray]:
    """Колонка вершин как (N, 3) float64 — прямо из буфера. None, если не вышло."""
    try:
        fmt = vdata.get_format()
        array_index = fmt.get_array_with(name)
        if array_index < 0:
            return None
        column = fmt.get_array(array_index).get_column(name)
        dtype = _numeric_dtype(column.get_numeric_type())
        if dtype is None or column.get_num_components() < 3:
            return None
        stride = fmt.get_array(array_index).get_stride()
        raw = bytes(vdata.get_array(array_index).get_handle().get_data())
        rows = np.frombuffer(raw, dtype=np.uint8)[
            : len(raw) // stride * stride].reshape(-1, stride)
        start = column.get_start()
        width = np.dtype(dtype).itemsize * 3
        return rows[:, start:start + width].copy().view(dtype).astype(
            np.float64)
    except Exception:                                     # noqa: BLE001
        return None


def _index_array(prim) -> Optional[np.ndarray]:
    """Индексы примитива как плоский массив. None — читаем по-старому."""
    try:
        vertices = prim.get_vertices()
        if vertices is None:                # индексы неявные: 0, 1, 2, …
            first = prim.get_first_vertex()
            return np.arange(first, first + prim.get_num_vertices(),
                             dtype=np.int64)
        dtype = _numeric_dtype(prim.get_index_type())
        if dtype is None:
            return np.array(prim.get_vertex_list(), dtype=np.int64)
        raw = bytes(vertices.get_handle().get_data())
        # Буфер может быть длиннее нужного (движок его переиспользует) —
        # берём ровно столько индексов, сколько примитив объявил.
        return np.frombuffer(raw, dtype=dtype)[
            : prim.get_num_vertices()].astype(np.int64)
    except Exception:                                     # noqa: BLE001
        try:
            return np.array(prim.get_vertex_list(), dtype=np.int64)
        except Exception:                                 # noqa: BLE001
            return None


def _read_obj(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """OBJ без материалов и UV: нужны только `v` и `f` (полигоны — веером)."""
    verts: List[Tuple[float, float, float]] = []
    tris: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]),
                              float(parts[3])))
            elif line.startswith("f "):
                idx: List[int] = []
                for token in line.split()[1:]:
                    num = int(token.split("/")[0])
                    idx.append(num - 1 if num > 0 else len(verts) + num)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    return (np.array(verts, dtype=np.float64).reshape(-1, 3),
            np.array(tris, dtype=np.int32).reshape(-1, 3))


# ---------------------------------------------------------------------------
# Сечения
# ---------------------------------------------------------------------------
def _slice_segments(tri: np.ndarray, z: float) -> np.ndarray:
    """
    Отрезки сечения плоскостью z, массив (N, 2, 2) в координатах XY.

    На вход идут уже разложенные треугольники (M, 3, 3): сечений считаются
    десятки, и собирать `verts[tris]` заново на каждое — это десятки мегабайт
    впустую на крупном кузове.
    """
    dz = tri[:, :, 2] - z
    above = dz > 0
    n_above = above.sum(1)
    keep = (n_above == 1) | (n_above == 2)
    tri, dz, above = tri[keep], dz[keep], above[keep]
    if len(tri) == 0:
        return np.zeros((0, 2, 2))


    # У каждого отобранного треугольника плоскость пересекают ровно два ребра
    # из трёх; третье помечается NaN и уходит в конец при сортировке.
    ends: List[np.ndarray] = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        crosses = above[:, a] != above[:, b]
        t = np.zeros(len(tri))
        t[crosses] = dz[crosses, a] / (dz[crosses, a] - dz[crosses, b])
        point = tri[:, a, :2] + t[:, None] * (tri[:, b, :2] - tri[:, a, :2])
        ends.append(np.where(crosses[:, None], point, np.nan))
    stack = np.stack(ends, axis=1)                      # (N, 3, 2)
    order = np.argsort(np.isnan(stack[:, :, 0]), axis=1, kind="stable")[:, :2]
    rows = np.arange(len(stack))[:, None]
    return stack[rows, order]


def _ray_hits(segs: np.ndarray, origin: Sequence[float], axis: int,
              sign: int) -> np.ndarray:
    """Расстояния от origin до пересечений луча вдоль оси axis, по возрастанию."""
    other = 1 - axis
    a, b = segs[:, 0], segs[:, 1]
    da = a[:, other] - origin[other]
    db = b[:, other] - origin[other]
    crosses = (da > 0) != (db > 0)
    if not crosses.any():
        return np.zeros(0)
    a, b, da, db = a[crosses], b[crosses], da[crosses], db[crosses]
    t = da / (da - db)
    hit = a[:, axis] + t * (b[:, axis] - a[:, axis])
    dist = (hit - origin[axis]) * sign
    return np.sort(dist[dist > 1e-6])


def _void_center(segs: np.ndarray, min_area: float
                 ) -> Optional[Tuple[np.ndarray, float]]:
    """
    Центр и площадь самой большой ЗАМКНУТОЙ пустоты сечения.

    Сечение растеризуется, заливка от края помечает наружное пространство, из
    оставшихся кусков берётся самый большой. Замкнутость — единственный
    признак, по которому «внутри кузова» отличается от «снаружи» на срезе, а
    нужен он именно там, где никакого наполнителя ещё нет.
    """
    if len(segs) == 0:
        return None
    try:
        from scipy import ndimage
    except ImportError:
        return None

    lo = segs.reshape(-1, 2).min(0) - 4 * CELL
    hi = segs.reshape(-1, 2).max(0) + 4 * CELL
    nx = int(np.ceil((hi[0] - lo[0]) / CELL)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / CELL)) + 1
    if nx * ny > 8_000_000:                     # заведомо не кузов
        return None

    # Отрезки кладутся в растр «пунктиром» с шагом в полклетки — так линия
    # выходит связной при любом наклоне, и заливка сквозь неё не протекает.
    p0 = (segs[:, 0] - lo) / CELL
    p1 = (segs[:, 1] - lo) / CELL
    steps = np.maximum(np.abs(p1 - p0).max(1).astype(np.int64), 1) * 2 + 2
    seg = np.repeat(np.arange(len(segs)), steps)
    starts = np.concatenate(([0], np.cumsum(steps)[:-1]))
    t = ((np.arange(int(steps.sum())) - starts[seg])
         / (steps[seg] - 1).astype(np.float64))
    pts = p0[seg] + t[:, None] * (p1[seg] - p0[seg])
    ix = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, ny - 1)
    wall = np.zeros((nx, ny), dtype=bool)
    wall[ix, iy] = True

    labels, count = ndimage.label(~wall)
    if count == 0:
        return None
    cells = np.bincount(labels.ravel(), minlength=count + 1)
    cells[0] = 0                                  # сам материал
    cells[np.unique(np.concatenate(
        [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = 0   # наружу
    best = int(cells.argmax())
    area = int(cells[best]) * CELL * CELL
    if best == 0 or area < max(min_area, 0.05):
        return None
    ix, iy = np.nonzero(labels == best)
    center = np.array([lo[0] + ix.mean() * CELL, lo[1] + iy.mean() * CELL])
    return center, area


def _faces_at(segs: np.ndarray, center: np.ndarray) -> Optional[List[float]]:
    """[x_lo, x_hi, y_lo, y_hi] — ближайшие к центру грани. None, если открыто."""
    out: List[float] = []
    for axis, sign in _SIDES:
        hits = _ray_hits(segs, center, axis, sign)
        if len(hits) == 0:
            return None
        out.append(float(center[axis] + sign * hits[0]))
    return out


def _fit_side(depths: Sequence[float], values: Sequence[float],
              sign: int) -> float:
    """
    Положение грани на кромке по отсчётам с разных глубин.

    Борт плоский, поэтому грань линейна по глубине: достаточно провести прямую
    и продолжить её до кромки (`depth = 0`). Прямая берётся по паре отсчётов,
    вокруг которой собирается большинство остальных, а из нескольких таких —
    та, что даёт САМУЮ НАРУЖНУЮ грань: и козырёк над проёмом, и скругление дна
    могут сделать замер только у́же настоящего борта.
    """
    if len(values) < 2:
        return float(values[0])
    d = np.asarray(depths, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    need = max(3, int(round(0.35 * len(v))))
    best: Optional[Tuple[float, float]] = None
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if abs(d[j] - d[i]) < 1e-6:
                continue
            slope = (v[j] - v[i]) / (d[j] - d[i])
            top = float(v[i] - slope * d[i])
            if int((np.abs(slope * d + top - v) <= FIT_TOLERANCE).sum()) < need:
                continue
            if best is None or top * sign > best[0]:
                best = (top * sign, top)
    if best is None:
        return float(v.max() if sign > 0 else v.min())
    return best[1]


# ---------------------------------------------------------------------------
# Проём
# ---------------------------------------------------------------------------
def find_opening(verts: np.ndarray, tris: np.ndarray) -> Optional[Opening]:
    """Проём кузова по мешу. None — если замкнутой полости не нашлось."""
    if len(verts) == 0 or len(tris) == 0:
        return None
    tri = verts[tris]
    z_top, z_bottom = float(verts[:, 2].max()), float(verts[:, 2].min())
    footprint = ((verts[:, 0].max() - verts[:, 0].min())
                 * (verts[:, 1].max() - verts[:, 1].min()))
    min_area = MIN_VOID_FRACTION * float(footprint)

    # 1. Сверху вниз до первой замкнутой пустоты — это и есть верх низкого борта.
    z, z_open = z_top, z_top + SCAN_STEP
    found: Optional[Tuple[float, np.ndarray, float]] = None
    while z > z_bottom:
        void = _void_center(_slice_segments(tri, z), min_area)
        if void is not None:
            found = (z, void[0], void[1])
            break
        z_open = z
        z -= SCAN_STEP
    if found is None:
        return None

    z_closed = found[0]
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (z_closed + z_open)
        void = _void_center(_slice_segments(tri, mid), min_area)
        if void is not None:
            z_closed, found = mid, (mid, void[0], void[1])
        else:
            z_open = mid
    z_rim = 0.5 * (z_closed + z_open)
    center, area = found[1], found[2]

    # 2. Грань борта на нескольких глубинах: под самой кромкой замер портят
    #    козырёк и обвязка, глубоко внизу — скругление дна.
    base = _faces_at(_slice_segments(tri, z_closed), center)
    if base is None:
        return None
    ref_area = (base[1] - base[0]) * (base[3] - base[2])

    depths: List[float] = [0.0]
    rows: List[List[float]] = [base]
    for depth in PROBE_DEPTHS:
        z_probe = z_rim - depth
        if z_probe <= z_bottom:
            break
        faces = _faces_at(_slice_segments(tri, z_probe), center)
        if faces is None:
            continue
        if (faces[1] - faces[0]) * (faces[3] - faces[2]) < 0.5 * ref_area:
            continue                        # ушли в скругление дна или под пол
        if not (faces[0] < center[0] < faces[1]
                and faces[2] < center[1] < faces[3]):
            continue
        depths.append(depth)
        rows.append(faces)

    sides = [_fit_side(depths, [r[k] for r in rows], sign)
             for k, (_axis, sign) in enumerate(_SIDES)]
    return Opening(z=z_rim, x_lo=sides[0], x_hi=sides[1], y_lo=sides[2],
                   y_hi=sides[3], probes=len(depths), area=float(area))


def points_from_mesh(path: str) -> Tuple[Optional[List[List[float]]], str]:
    """
    Четыре точки верхней кромки по мешу кузова и строчка «как получилось».

    Ничего не выбрасывает: не вышло — вернётся (None, причина).
    """
    if not path or not os.path.isfile(path):
        return None, f"нет файла кузова: {path or '—'}"
    try:
        verts, tris = read_mesh(path)
    except Exception as exc:                              # noqa: BLE001
        return None, f"{os.path.basename(path)} не прочитан: {exc}"
    if len(tris) == 0:
        return None, f"в {os.path.basename(path)} нет треугольников"
    try:
        opening = find_opening(verts, tris)
    except Exception as exc:                              # noqa: BLE001
        return None, f"проём в {os.path.basename(path)} не найден: {exc}"
    if opening is None:
        return None, (f"в {os.path.basename(path)} не нашлось замкнутой "
                      "полости — проём кузова определить нечем")
    return opening.points(), (
        f"по мешу {os.path.basename(path)}: проём "
        f"{opening.width:.3f} x {opening.length:.3f} м на высоте "
        f"{opening.z:.3f} м, замер по {opening.probes} сечениям")
