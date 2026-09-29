# -*- coding: utf-8 -*-
"""
Анализ проезда для кинематографичной реконструкции — отдельный процесс.

Запускается НЕ интерпретатором утилиты, а тем, где стоят direct_mesh и
volume_calculator (Python 3.11: torch + Depth Anything 3, open3d). Утилите torch
не нужен, а тяжёлый расчёт в своём процессе не делит GIL с рендером. Обмен —
через один .npz (см. `run()`), которую утилита разбирает в
src/cinematic/scan_analysis.py.

    python scan_worker.py --ply X.ply --photo X_topdown.jpg --json X.json \
        --out X.cine.npz [--napolnitel X_napolnitel.obj] \
        [--mesh-before X_mesh_before.obj] [--wall-mask X_wall_mask.obj]

Что считается (всё — для показа; измерения берутся из серверного JSON):

* `T` — облако лидара -> система модели кузова, ровно как на сервере
  (operation-3d-service/src/mesh_reconstruction.cpp, solveTransform:
  перестановка опорных точек {3,4,1,2}, обмен осей y<->z, -0.1 м, жёсткое
  преобразование по трём точкам). Матрица зеркальная (det = -1).
* снимок: полнокадровый меш «фото-глубины» direct_mesh (Depth Anything,
  приведённая лидаром к метрам) во ВСЁМ кадре, а не только в контуре кузова;
  камера станции и её внутренние параметры.
* облако: точки, метки «машина/фон», порядок развёртки для анимации лидара.
* поиск кузова volume_calculator/body_geometry.detect_body — с перехватом
  промежуточных шагов (стенки, след, выборки кромки, плоскость кромки, углы).
* карты высот этапов серверной реконструкции наполнителя на общей сетке в
  системе модели: сырая, после фильтров, заполненная, сглаженная, с
  прижатыми стенками и итоговая дорисованная (по `_mesh_before.obj`).

Каждая часть независима: если что-то не посчиталось, в `notes` пишется
причина, остальное всё равно сохраняется.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

#: Где искать соседний репозиторий, по убыванию приоритета.
REPO_CANDIDATES = [
    os.environ.get("IQOKO_ALEXEY_REPO", ""),
    os.path.join(HERE, "..", "..", "..", "AlexeyPlys", "repo"),
    r"G:\IQoko\AlexeyPlys\repo",
]

NOTES: list = []


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"[cine-worker] {msg}", flush=True)


def stage(msg: str) -> None:
    """Строка хода работы: утилита показывает её в интерфейсе."""
    print(f"@@STAGE {msg}", flush=True)


def find_repo() -> str:
    for raw in REPO_CANDIDATES:
        if raw and os.path.isdir(os.path.join(raw, "direct_mesh")):
            return os.path.abspath(raw)
    raise SystemExit("repo with direct_mesh/volume_calculator not found; "
                     "set IQOKO_ALEXEY_REPO")


# --------------------------------------------------------------------------- #
# Геометрия
# --------------------------------------------------------------------------- #

def server_transform(keypoints, points_3d) -> np.ndarray:
    """
    4x4: точка облака лидара -> система модели кузова, как в
    mesh_reconstruction.cpp::solveTransform (масштаб 1).
    """
    kp = np.asarray(keypoints, float)
    sc = np.asarray(points_3d, float)

    def swap(p):
        return np.asarray(p, float)[..., [0, 2, 1]]

    moved = [None] * 4
    for i, target in enumerate((3, 4, 1, 2)):
        q = swap(kp[i]).copy()
        q[1] += -0.1
        moved[target - 1] = q

    def local_to_world(a, b, c):
        x = (b - a) / np.linalg.norm(b - a)
        ty = c - a
        y = ty - x * (ty @ x)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, a
        return m

    def right_angle(t):
        d01 = np.sum((t[0] - t[1]) ** 2)
        d02 = np.sum((t[0] - t[2]) ** 2)
        d12 = np.sum((t[1] - t[2]) ** 2)
        if d12 >= d02 and d12 >= d01:
            return 0
        if d02 >= d12 and d02 >= d01:
            return 1
        return 2

    t1 = [sc[0], sc[1], sc[2]]
    t2 = [moved[0], moved[1], moved[2]]
    i = right_angle(t1)
    if i:
        t1[0], t1[i] = t1[i], t1[0]
        t2[0], t2[i] = t2[i], t2[0]
    m = local_to_world(*t1) @ np.linalg.inv(local_to_world(*t2))
    s = np.eye(4)
    s[1, 1] = s[2, 2] = 0.0
    s[1, 2] = s[2, 1] = 1.0
    return m @ s


def apply(T, p):
    p = np.asarray(p, float)
    return p @ T[:3, :3].T + T[:3, 3]


def read_obj(path):
    """Вершины и треугольники простого OBJ (v / f, индексы без слешей)."""
    with open(path, "rb") as fh:
        data = fh.read()
    v, f = [], []
    lines = data.split(b"\n")
    vl = [ln[2:] for ln in lines if ln.startswith(b"v ")]
    fl = [ln[2:] for ln in lines if ln.startswith(b"f ")]
    v = np.array(b" ".join(vl).split(), dtype=np.float64).reshape(-1, 3)
    if fl and b"/" in fl[0]:
        fl = [b" ".join(t.split(b"/")[0] for t in ln.split()) for ln in fl]
    f = np.array(b" ".join(fl).split(), dtype=np.int64).reshape(-1, 3) - 1
    return v, f


# --------------------------------------------------------------------------- #
# Снимок: полнокадровый меш фото-глубины
# --------------------------------------------------------------------------- #

def photo_mesh(dm, rec, xyz, photo_path, out, work=3, max_jump=0.05,
               max_ray=0.9, max_range=16.0):
    """
    Метрическая глубина всего кадра по direct_mesh и меш «каждый пиксель на
    своём луче». Треугольники поперёк перепадов глубины (занавес между
    кромкой и землёй) выбрасываются: с исходного ракурса их не видно, а при
    облёте они торчали бы полотнищами.
    """
    import cv2
    ap, pd = dm["appearance"], dm["photodepth"]
    img = cv2.imread(photo_path)
    if img is None:
        note(f"photo unreadable: {photo_path}")
        return
    rigs = ap.load_rigs(os.path.join(dm["path"], "rigs.json"))
    rig = min(rigs, key=lambda r: ap.ground_signature_distance(rec, r))
    sig = ap.ground_signature_distance(rec, rig)
    if sig > 3.0:
        note(f"unknown station (ground signature {sig:.1f}) — no photo mesh")
        return
    cam = rig["camera"]
    size = (img.shape[1], img.shape[0])
    stage("глубина по снимку (Depth Anything)")
    depth = pd.relative_depth(img)
    stage("совмещение снимка с облаком")
    m = np.zeros(4)
    try:
        m, reg = pd.register(rec, xyz, depth, cam, size)
        out["photo_misfit"] = float(reg.get("misfit", -1))
    except Exception as exc:
        note(f"registration failed ({exc}); photo assumed simultaneous")
    stage("метрическая глубина кадра")
    truck, static = pd.lidar_split(rec, xyz)
    Z, U, V, dE, fit, stats = pd.fuse_truck(depth, cam, pd._moved(rec, truck, m),
                                            static, size, work=work)
    xr, yr = pd.pixel_rays(cam, U, V)
    # У края применимости модели линзы (лучи почти на изгибе полинома) и
    # вдали от лидара аффинное поле глубины экстраполируется в десятки
    # метров — такие пиксели не берём.
    ok = (np.isfinite(xr) & np.isfinite(Z) & (Z > 0.5)
          & (np.hypot(xr, yr) < max_ray)
          & (Z * np.sqrt(1 + xr * xr + yr * yr) < max_range))
    H, W = Z.shape
    Pc = np.stack([xr * Z, yr * Z, Z], axis=-1).reshape(-1, 3)
    Ps = pd.from_camera(Pc, cam)                        # система лидара
    idx = np.arange(H * W).reshape(H, W)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    okf = ok.ravel()
    zf = Z.ravel()
    good = okf[tris].all(1)
    zt = zf[tris]
    rel = (zt.max(1) - zt.min(1)) / np.maximum(zt.min(1), 1e-3)
    curtain = good & (rel >= max_jump)
    tris_c = tris[curtain]
    tris = tris[good & (rel < max_jump)]
    used = np.zeros(H * W, bool)
    used[tris.ravel()] = True
    used[tris_c.ravel()] = True
    remap = np.full(H * W, -1, np.int64)
    remap[used] = np.arange(used.sum())
    out["photo_vertices"] = Ps[used].astype(np.float32)
    uv = np.stack([(U.ravel() + 0.5) / size[0], (V.ravel() + 0.5) / size[1]], 1)
    out["photo_uv"] = uv[used].astype(np.float32)
    out["photo_faces"] = remap[tris].astype(np.int32)
    # «Занавесы» поперёк перепадов глубины — отдельно: из ракурса станции
    # они видны ребром и закрывают щель вдоль кромки, при облёте их гасят.
    out["photo_curtain_faces"] = remap[tris_c].astype(np.int32)
    out["photo_camera"] = np.asarray(cam, np.float64)
    out["photo_size"] = np.asarray(size, np.int32)
    out["photo_offset"] = np.asarray(m, np.float64)
    out["photo_rig"] = np.asarray(str(rig.get("id", "?")))
    note(f"photo mesh: {used.sum()} verts, {len(tris)} tris, rig {rig.get('id')}, "
         f"offset {np.round(m[:2], 2).tolist()} m")


# --------------------------------------------------------------------------- #
# Облако и метки
# --------------------------------------------------------------------------- #

def cloud(rec, xyz, inten, out):
    """Точки, метки (1 — машина, 0 — фон) и порядок развёртки лидара."""
    hf = rec.hf
    n, d = rec.ground
    h = xyz @ n + d
    P = np.column_stack([xyz, np.ones(len(xyz))]) @ np.linalg.inv(rec.to_sensor).T
    ix = np.round((P[:, 0] - hf.x0) / hf.res).astype(int)
    iy = np.round((P[:, 1] - hf.y0) / hf.res).astype(int)
    okp = (ix >= 0) & (iy >= 0) & (ix < hf.top.shape[1]) & (iy < hf.top.shape[0])
    from scipy import ndimage
    grown = ndimage.binary_dilation(hf.inside, iterations=max(1, int(0.12 / hf.res)))
    inside = np.zeros(len(xyz), bool)
    inside[okp] = grown[iy[okp], ix[okp]]
    truck = inside & (h > 0.12) & (h < 4.6)
    out["points"] = xyz.astype(np.float32)
    out["intensity"] = np.clip(inten, 0, 1).astype(np.float32)
    out["labels"] = truck.astype(np.int8)
    out["height_above_ground"] = h.astype(np.float32)
    # Лидар на мачте крутится вокруг вертикали: порядок «выстрела» — угол.
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    out["sweep"] = ((az + np.pi) / (2 * np.pi)).astype(np.float32)
    out["ground_plane"] = np.r_[n, d].astype(np.float64)
    note(f"cloud: {len(xyz)} points, truck {int(truck.sum())}")


# --------------------------------------------------------------------------- #
# Поиск кузова (volume_calculator/body_geometry) с перехватом шагов
# --------------------------------------------------------------------------- #

def detection(bg, xyz, expected, out):
    trace: dict = {}
    orig = {k: getattr(bg, k) for k in ("_footprint", "_wall_planes", "_rim_plane")}

    def footprint(points, ground):
        r = orig["_footprint"](points, ground)
        if r is not None and "footprint" not in trace:
            trace["footprint"] = (r, ground)
        return r

    def wall_planes(points):
        r = orig["_wall_planes"](points)
        if "walls" not in trace:
            trace["walls"] = r
        return r

    def rim_plane(samples):
        r = orig["_rim_plane"](samples)
        trace["rim"] = (np.array(samples), r)
        return r

    bg._footprint, bg._wall_planes, bg._rim_plane = footprint, wall_planes, rim_plane
    try:
        found = bg.detect_body(xyz, expected=expected)
    finally:
        for k, v in orig.items():
            setattr(bg, k, v)
    if found is None:
        note("body_geometry: opening not found")
        return
    corners = np.asarray(found.corners, float)
    out["det_corners"] = corners.astype(np.float32)
    ey = corners[1, :2] - corners[0, :2]
    ey /= np.linalg.norm(ey)
    ex = corners[3, :2] - corners[0, :2]
    ex /= np.linalg.norm(ex)

    if "footprint" in trace:
        (fx, fy, flo, fhi), ground = trace["footprint"]
        rect = np.array([[flo[0], flo[1]], [flo[0], fhi[1]], [fhi[0], fhi[1]], [fhi[0], flo[1]]])
        xy = rect[:, :1] * fx + rect[:, 1:] * fy
        out["det_footprint"] = np.column_stack([xy, np.full(4, ground)]).astype(np.float32)
    walls = trace.get("walls") or []
    wp, wid = [], []
    for k, w in enumerate(sorted(walls, key=lambda w: -w["score"])[:6]):
        pts = np.asarray(w["points"])
        if len(pts) > 2500:
            pts = pts[np.random.default_rng(k).choice(len(pts), 2500, replace=False)]
        wp.append(pts)
        wid.append(np.full(len(pts), k, np.int16))
    if wp:
        out["det_wall_points"] = np.vstack(wp).astype(np.float32)
        out["det_wall_ids"] = np.concatenate(wid)
    if "rim" in trace:
        samples, (coef, keep) = trace["rim"]
        world = samples[:, :1] * ex + samples[:, 1:2] * ey
        out["det_rim_samples"] = np.column_stack([world, samples[:, 2]]).astype(np.float32)
        out["det_rim_inliers"] = np.asarray(keep, bool)
        out["det_rim_side"] = samples[:, 3].astype(np.int8)
    out["det_diag"] = np.asarray(json.dumps(found.diagnostics, default=float))
    note(f"body_geometry: {len(walls)} walls, corners found")


# --------------------------------------------------------------------------- #
# Этапы наполнителя: карты высот на общей сетке (система модели)
# --------------------------------------------------------------------------- #

def fill_stages(T, xyz, napolnitel, mesh_before, wall_mask, out, res=0.03):
    from scipy import ndimage
    from scipy.spatial import cKDTree

    nv, _ = read_obj(napolnitel)
    lo = nv[:, :2].min(0) - 0.05
    hi = nv[:, :2].max(0) + 0.05
    nx = int(np.ceil((hi[0] - lo[0]) / res)) + 1
    ny = int(np.ceil((hi[1] - lo[1]) / res)) + 1
    out["grid_origin"] = lo.astype(np.float64)
    out["grid_res"] = np.float64(res)
    out["grid_shape"] = np.array([ny, nx], np.int32)

    P = apply(T, xyz)
    nlo, nhi = nv[:, :2].min(0), nv[:, :2].max(0)
    ztop = nv[:, 2].max()
    body = ((P[:, 0] > nlo[0] - 0.15) & (P[:, 0] < nhi[0] + 0.15)
            & (P[:, 1] > nlo[1] - 0.15) & (P[:, 1] < nhi[1] + 0.15)
            & (P[:, 2] > nv[:, 2].min() - 0.3) & (P[:, 2] < ztop + 0.6))

    def raster(pts, how="mean"):
        ix = np.clip(((pts[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
        flat = iy * nx + ix
        if how == "min":
            g = np.full(nx * ny, np.inf)
            np.minimum.at(g, flat, pts[:, 2])
            g[~np.isfinite(g)] = np.nan
        else:
            s = np.bincount(flat, pts[:, 2], nx * ny)
            c = np.bincount(flat, None, nx * ny)
            g = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        return g.reshape(ny, nx)

    # 1) сырая: всё, что лидар видел над областью кузова (борта, кромки)
    out["h_raw"] = raster(P[body]).astype(np.float32)

    # 2) фильтры сервера: внутри наполнителя + отсев «улетающих вверх»
    inside = (body & (P[:, 0] > nlo[0]) & (P[:, 0] < nhi[0])
              & (P[:, 1] > nlo[1]) & (P[:, 1] < nhi[1]) & (P[:, 2] < ztop))
    Q = P[inside]
    keep = np.ones(len(Q), bool)
    if len(Q) > 64:
        tree = cKDTree(Q[:, :2])
        _, nb = tree.query(Q[:, :2], k=32)
        med = np.median(Q[nb, 2], axis=1)
        keep = Q[:, 2] <= med + 0.08
    out["h_filtered"] = raster(Q[keep]).astype(np.float32)
    out["fill_points_kept"] = Q[keep].astype(np.float32)

    # 3) заполнение пропусков (взвешенное размытие по известным ячейкам)
    h = out["h_filtered"].astype(np.float64)
    known = np.isfinite(h)
    foot = np.zeros_like(known)
    ix0 = int((nlo[0] - lo[0]) / res)
    ix1 = int((nhi[0] - lo[0]) / res)
    iy0 = int((nlo[1] - lo[1]) / res)
    iy1 = int((nhi[1] - lo[1]) / res)
    foot[iy0:iy1 + 1, ix0:ix1 + 1] = True

    def masked_blur(val, msk, sigma):
        num = ndimage.gaussian_filter(np.where(msk, val, 0.0), sigma)
        den = ndimage.gaussian_filter(msk.astype(float), sigma)
        return num / np.maximum(den, 1e-9), den

    filled = h.copy()
    for sigma in (1.0, 2.0, 4.0, 8.0):
        est, den = masked_blur(np.nan_to_num(h), known, sigma)
        hole = ~np.isfinite(filled) & foot & (den > 1e-3)
        filled[hole] = est[hole]
    filled[~foot] = np.nan
    out["h_filled"] = filled.astype(np.float32)

    # 4) сглаживание
    sm, _ = masked_blur(np.nan_to_num(filled), np.isfinite(filled), 1.6)
    smooth = np.where(np.isfinite(filled), sm, np.nan)
    out["h_smooth"] = smooth.astype(np.float32)

    # 5) стенки: маска сервера (_wall_mask.obj) -> уровень соседей
    walls = np.zeros_like(known)
    if wall_mask and os.path.isfile(wall_mask):
        wv, _ = read_obj(wall_mask)
        wix = np.clip(((wv[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
        wiy = np.clip(((wv[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
        walls[wiy, wix] = True
        walls = ndimage.binary_closing(walls, iterations=1) & foot
    flat = smooth.copy()
    if walls.any():
        base = np.isfinite(smooth) & ~walls
        est, _ = masked_blur(np.nan_to_num(smooth), base, 5.0)
        flat[walls] = est[walls]
    out["wall_cells"] = walls
    out["h_walls"] = flat.astype(np.float32)

    # 6) итог сервера: дорисованная поверхность из _mesh_before.obj. Это
    #    замкнутое тело «рельеф + крышка над ним» (из него потом вычитается
    #    наполнитель), рельеф — его НИЗ: сервер тоже берёт min-Z.
    if mesh_before and os.path.isfile(mesh_before):
        mv, mf = read_obj(mesh_before)
        top = raster(mv, how="min")
        miss = ~np.isfinite(top)
        if miss.any() and (~miss).any():
            _, near = ndimage.distance_transform_edt(miss, return_indices=True)
            top = top[near[0], near[1]]
        out["h_final"] = top.astype(np.float32)
    out["napolnitel_vertices"] = nv.astype(np.float32)
    note(f"fill stages on {ny}x{nx} grid, {int(keep.sum())} load points")


# --------------------------------------------------------------------------- #

def _add_paths() -> str:
    repo = find_repo()
    for sub in ("volume_calculator", "direct_mesh"):
        path = os.path.join(repo, sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo


def run(args) -> int:
    t0 = time.time()
    NOTES.clear()
    repo = _add_paths()
    import scan2mesh as sm
    import appearance as ap
    out: dict = {}

    with open(args.json, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    kp, p3 = meta.get("keypoints_3d"), meta.get("points_3d")

    parts = set((args.parts or "scene,fill").split(","))
    stage("чтение облака")
    xyz, inten = sm.read_ply(args.ply)
    rigs = ap.load_rigs(os.path.join(repo, "direct_mesh", "rigs.json"))
    priors = [(np.r_[r["ground_signature"]["normal_xy"],
                     -np.sqrt(max(0., 1 - np.sum(np.square(r["ground_signature"]["normal_xy"]))))],
               r["ground_signature"]["height"]) for r in rigs if r.get("ground_signature")]
    if "scene" in parts:
        stage("земля и след машины")
        rec = sm.reconstruct(xyz, inten, ground_priors=priors)
        cloud(rec, xyz, inten, out)

    T = None
    if kp and p3 and len(kp) == 4 and len(p3) == 4:
        T = server_transform(kp, p3)
        out["T"] = T
        out["keypoints"] = np.asarray(kp, np.float32)
        out["points_3d"] = np.asarray(p3, np.float32)
    else:
        note("JSON has no keypoints_3d/points_3d — no model frame")

    if "scene" in parts and args.photo and os.path.isfile(args.photo):
        try:
            import photodepth as pdm
            photo_mesh(dict(path=os.path.join(repo, "direct_mesh"), appearance=ap,
                            photodepth=pdm), rec, xyz, args.photo, out)
        except Exception as exc:
            traceback.print_exc()
            note(f"photo mesh failed: {exc}")

    if "scene" in parts:
        detect_stage(meta, xyz, out)

    if "fill" in parts and T is not None and args.napolnitel and os.path.isfile(args.napolnitel):
        stage("этапы наполнителя")
        try:
            fill_stages(T, xyz, args.napolnitel, args.mesh_before, args.wall_mask, out)
        except Exception as exc:
            traceback.print_exc()
            note(f"fill stages failed: {exc}")

    out["notes"] = np.asarray(json.dumps(NOTES, ensure_ascii=False))
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, args.out)
    stage(f"готово за {time.time() - t0:.1f} с")
    return 0


def detect_stage(meta, xyz, out):
    stage("поиск кузова")
    try:
        import body_geometry as bg
        dims = meta.get("body_dimensions") or {}
        expected = None
        if dims.get("width") and dims.get("length"):
            expected = (float(dims["width"]), float(dims["length"]))
        detection(bg, xyz, expected, out)
    except Exception as exc:
        traceback.print_exc()
        note(f"detection failed: {exc}")


def serve() -> int:
    """
    Постоянный режим: всё тяжёлое (torch, Depth Anything на GPU, open3d,
    детектор кузова) грузится и прогревается ОДИН раз, дальше запросы идут
    строками JSON через stdin: {"id": ..., "argv": [...]}. Ответ —
    «@@DONE id» / «@@FAIL id текст». Конец stdin (утилита закрылась) —
    выход, так что процесс не переживает родителя.
    """
    t0 = time.time()
    _add_paths()
    import scan2mesh  # noqa: F401
    import appearance  # noqa: F401
    import body_geometry  # noqa: F401
    import photodepth as pdm
    import cv2
    pdm.load_model()
    # первый проход инициализирует CUDA-ядра — делаем его сейчас, не в сцене
    pdm.relative_depth(np.zeros((270, 480, 3), np.uint8))
    print(f"@@READY {time.time() - t0:.1f}", flush=True)
    parser = _parser()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id", "")
        try:
            run(parser.parse_args(req.get("argv", [])))
            print(f"@@DONE {rid}", flush=True)
        except BaseException as exc:          # argparse бросает SystemExit
            traceback.print_exc()
            print(f"@@FAIL {rid} {type(exc).__name__}: {exc}", flush=True)
    return 0


def _parser() -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    a.add_argument("--ply", required=True)
    a.add_argument("--photo")
    a.add_argument("--json", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--napolnitel")
    a.add_argument("--mesh-before")
    a.add_argument("--wall-mask")
    a.add_argument("--parts", default="scene,fill",
                   help="что считать: scene (снимок, облако, поиск кузова), fill "
                        "(этапы наполнителя) — через запятую")
    return a


def main() -> int:
    if "--serve" in sys.argv:
        return serve()
    return run(_parser().parse_args())


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
