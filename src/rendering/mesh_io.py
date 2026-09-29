# -*- coding: utf-8 -*-
"""
Быстрая загрузка серверного меша наполнителя и сборка Geom без Python-циклов.

Раньше `_result.obj` (сотни тысяч вершин) разбирался через `trimesh.load`, а
`MyApp.trimesh_to_panda` переписывал вершины в Panda поштучно — миллион вызовов
`addData3f`/`addVertices`. Вместе ~6–13 с в UI-потоке, всё это время рендер
стоял. Здесь то же самое делается векторно:

  * разбор OBJ — C-парсером pandas (~0.2 с на 350k вершин), результат кэшируется
    рядом с OBJ в `.npz`, повторная загрузка — чтение двух массивов;
  * нормали — как у trimesh (`vertex_normals`): сумма нормалей граней с весом
    по углу при вершине; точные дубликаты вершин склеиваются, как в его
    `merge_vertices`, иначе на шве появилась бы видимая граница освещения;
  * вершины/нормали/UV пишутся в `GeomVertexData` одним копированием буфера.

Разбор и расчёт нормалей не трогают сцену и Panda-объекты, поэтому их можно
звать из фонового потока. numpy и C-парсер pandas на больших массивах отпускают
GIL, так что поток не подвешивает кадры в главном. `geom_node_from_arrays` —
только в главном потоке.
"""

from __future__ import annotations

import io
import os
from typing import Optional, Tuple

import numpy as np

#: Версия формата кэша. Поднять, если меняется то, что в него пишется.
_CACHE_VERSION = 1


# --------------------------------------------------------------------------- #
# Разбор OBJ
# --------------------------------------------------------------------------- #

def _cache_path(obj_path: str) -> str:
    st = os.stat(obj_path)
    return f"{obj_path}.v{_CACHE_VERSION}_{st.st_size}_{int(st.st_mtime)}.npz"


def _parse_plain_obj(data: bytes) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Разобрать OBJ вида «все `v x y z`, затем все `f a b c`» — ровно так пишет
    сервер. Для всего остального (нормали, UV, `f a/b/c`, перемешанные строки,
    полигоны) возвращает None — тогда разбирает trimesh.
    """
    first_f = data.find(b"\nf ")
    if first_f < 0 or not data.startswith(b"v "):
        return None
    head, tail = data[:first_f + 1], data[first_f + 1:]
    # Всё, кроме v/f, парсер ниже не переварит — лучше честно отказаться.
    if b"/" in tail or b"\nv" in tail or b"\nf" in head:
        return None
    import pandas as pd
    try:
        verts = pd.read_csv(io.BytesIO(head), sep=" ", header=None,
                            usecols=[1, 2, 3], dtype=np.float64,
                            engine="c").to_numpy()
        faces = pd.read_csv(io.BytesIO(tail), sep=" ", header=None,
                            usecols=[1, 2, 3], dtype=np.int64,
                            engine="c").to_numpy()
    except Exception:
        return None
    if faces.size == 0 or verts.size == 0:
        return None
    faces -= 1                      # OBJ индексирует с единицы
    if faces.min() < 0 or faces.max() >= len(verts):
        return None
    return verts, faces


def _merge_duplicates(verts: np.ndarray, faces: np.ndarray):
    """Склеить вершины с совпадающими координатами и убрать висячие."""
    used = np.zeros(len(verts), dtype=bool)
    used[faces.ravel()] = True
    ref = verts[used]
    # То же, что np.unique(ref, axis=0, return_inverse=True), но без него:
    # unique по строкам сортирует структурный тип и держит GIL ~0.3 с на
    # полумиллионе вершин — кадр в главном потоке на это время вставал.
    # lexsort по числовым столбцам GIL отпускает.
    order = np.lexsort((ref[:, 2], ref[:, 1], ref[:, 0]))
    srt = ref[order]
    first = np.ones(len(srt), dtype=bool)
    first[1:] = (srt[1:] != srt[:-1]).any(axis=1)
    inverse = np.empty(len(srt), dtype=np.int64)
    inverse[order] = np.cumsum(first) - 1
    uniq = srt[first]
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[np.flatnonzero(used)] = inverse
    return uniq, remap[faces]


def load_obj_arrays(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вершины (N,3 float64) и треугольники (M,3 int64) из OBJ с кэшем `.npz`.

    Кэш привязан к размеру и времени изменения OBJ: перезаписанный файл
    разбирается заново.
    """
    cache = _cache_path(obj_path)
    if os.path.isfile(cache):
        try:
            with np.load(cache) as z:
                return z["vertices"], z["faces"]
        except Exception:
            pass

    with open(obj_path, "rb") as fh:
        data = fh.read()
    parsed = _parse_plain_obj(data)
    if parsed is None:
        import trimesh
        mesh = trimesh.load(obj_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
    else:
        verts, faces = _merge_duplicates(*parsed)

    try:
        np.savez(cache, vertices=verts, faces=faces)
    except OSError:
        pass
    return verts, faces


# --------------------------------------------------------------------------- #
# Нормали и UV
# --------------------------------------------------------------------------- #

def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Нормали вершин как у `trimesh.Trimesh.vertex_normals`: нормали граней с
    весом по углу грани при вершине. Вырожденные (NaN или длина < 0.1)
    заменяются на (0, 0, 1) — то же правило, что было в `trimesh_to_panda`.
    """
    tri = verts[faces]                                   # (M,3,3)
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    fn = np.cross(e01, -e20)
    fn_len = np.linalg.norm(fn, axis=1)
    ok = fn_len > 0
    fn[ok] /= fn_len[ok, None]
    fn[~ok] = 0.0

    def _angle(a, b):
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        den = la * lb
        cos = np.einsum("ij,ij->i", a, b)
        cos = np.divide(cos, den, out=np.zeros_like(cos), where=den > 0)
        return np.arccos(np.clip(cos, -1.0, 1.0))

    ang = np.stack([_angle(e01, -e20), _angle(e12, -e01), _angle(e20, -e12)],
                   axis=1)                               # (M,3)

    n = len(verts)
    out = np.zeros((n, 3), dtype=np.float64)
    idx = faces.ravel()
    w = ang.ravel()
    for c in range(3):
        contrib = np.repeat(fn[:, c], 3) * w
        out[:, c] = np.bincount(idx, weights=contrib, minlength=n)

    ln = np.linalg.norm(out, axis=1)
    good = ln > 0
    out[good] /= ln[good, None]
    bad = ~np.isfinite(out).all(axis=1) | (np.linalg.norm(out, axis=1) < 0.1)
    out[bad] = (0.0, 0.0, 1.0)
    return out


def planar_uv(verts: np.ndarray, u_scale: float = 1.0,
              v_scale: float = 1.0) -> np.ndarray:
    """
    Планарная развёртка по XY-габариту (как было в `trimesh_to_panda`), сразу
    с масштабом тайлинга: RenderPipeline не уважает матрицу TextureStage, так
    что повторы зашиваются прямо в texcoord.
    """
    lo = verts[:, :2].min(axis=0)
    rng = np.maximum(verts[:, :2].max(axis=0) - lo, 1e-6)
    uv = (verts[:, :2] - lo) / rng
    uv[:, 0] *= float(u_scale)
    uv[:, 1] *= float(v_scale)
    return uv


def interleave_v3n3t2(verts: np.ndarray, normals: np.ndarray,
                      uv: np.ndarray) -> np.ndarray:
    """Упаковать в (N,8) float32 — раскладка `GeomVertexFormat.get_v3n3t2()`."""
    out = np.empty((len(verts), 8), dtype=np.float32)
    out[:, 0:3] = verts
    out[:, 3:6] = normals
    out[:, 6:8] = uv
    return out


# --------------------------------------------------------------------------- #
# Panda — только главный поток
# --------------------------------------------------------------------------- #

def geom_node_from_arrays(name: str, interleaved: np.ndarray,
                          faces: np.ndarray):
    """
    GeomNode из упакованных вершин (N,8 float32: pos, normal, uv) и
    треугольников (M,3). Копирование буферов, без поштучной записи.
    """
    from panda3d.core import (Geom, GeomEnums, GeomNode, GeomTriangles,
                              GeomVertexData, GeomVertexFormat)

    fmt = GeomVertexFormat.get_v3n3t2()
    n = int(len(interleaved))
    vdata = GeomVertexData(name, fmt, Geom.UH_static)
    vdata.unclean_set_num_rows(n)
    arr = vdata.modify_array(0)
    if arr.get_array_format().get_stride() != 32:
        raise RuntimeError("неожиданная раскладка v3n3t2")
    view = memoryview(arr).cast("B")
    view[:] = np.ascontiguousarray(interleaved, dtype=np.float32).tobytes()

    prim = GeomTriangles(Geom.UH_static)
    prim.set_index_type(GeomEnums.NT_uint32)
    idx = prim.modify_vertices()
    idx.unclean_set_num_rows(int(faces.size))
    memoryview(idx).cast("B")[:] = np.ascontiguousarray(
        faces, dtype=np.uint32).tobytes()

    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode(name)
    node.add_geom(geom)
    return node
