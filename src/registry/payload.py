# -*- coding: utf-8 -*-
"""
Что именно уедет на сервер: файлы комплекта и поля `meta`.

Модуль отвечает на один вопрос — «из чего собрать запрос для набора, который
пользователь выбрал в списке кузовов». Он ничего не отправляет и ничего не
спрашивает: диалог берёт отсюда заготовку, показывает её человеку и правит по
его усмотрению.

Отдельная история — `bam_to_obj`. Реестр обязательно требует `-Cuzov.obj`, а
генератор кузовов пишет кузов только в `.bam`: OBJ он экспортирует лишь для
наполнителя. Гонять человека в Blender ради одного файла глупо, поэтому кузов
конвертируется на месте средствами Panda3D — того же движка, который этот .bam
и написал, так что координаты остаются мировыми, как в .bam.

ОСИ У ДВУХ OBJ РАЗНЫЕ, и это не оплошность, а требование сервера
(`IQOKO-AVCS/src/main.cpp`):

* `-Cuzov.obj` грузится с ЕДИНИЧНОЙ матрицей, и по нему трассируются лучи
  камеры прямо в мировых координатах -> кузов обязан быть Z-up, как .bam;
* `-Napolnitel.obj` грузится как `target_model`, и в отсутствие
  `target_model_transform` в meta к нему применяется зашитое
  (x, y, z) -> (-x, -z, y) -> наполнитель обязан быть Y-up.

Так же устроены и ручные комплекты (`SCANIA-P8X400-…`: Cuzov.obj Z-up,
Napolnitel.obj Y-up). Поворот делает сам генератор
(`body_builder/export.py::export_obj`), здесь пересчитывать нечего.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.registry.anchor_points import points_from_mesh
from src.registry.client import FILE_ROLES, REQUIRED_ROLES, suggest_key

#: Суффикс имени файла -> роль в мультипарте. Порядок важен: `-Napolnitel.obj`
#: должен проверяться раньше, чем `.obj`.
_SUFFIX_ROLES: Tuple[Tuple[str, str], ...] = (
    ("-napolnitel.obj", "napolnitel_obj"),
    ("-napolnitel.mtl", "napolnitel_mtl"),
    ("-napolnitel.bam", "napolnitel_bam"),
    ("-cuzov.obj",      "cuzov_obj"),
    ("-cuzov.bam",      "cuzov_bam"),
    ("-other.obj",      "other_obj"),
    ("-other.bam",      "other_bam"),
    ("-body.obj",       "body_obj"),
    ("-full.obj",       "full_obj"),
)

#: По этим суффиксам вычисляется общее имя комплекта (stem) из пути к кузову.
_STEM_SUFFIXES = ("-Cuzov.bam", "-Cuzov.obj", "-Napolnitel.obj",
                  "-Napolnitel.bam", "-Other.bam", ".bam", ".gltf", ".glb",
                  ".obj", ".egg")

_TEXTURE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".dds", ".bmp")

#: Файлы веб-вьюера, которые генератор кладёт рядом с комплектом: сам glTF,
#: его буфер и пожатые в JPEG карты в папке `<stem>/`. Разделение важное:
#: `.png` из той же папки — это текстуры для .bam (роль `texture`,
#: `data/textures/…`), а в `WEB/` они не нужны и только утяжеляют загрузку.
_WEB_TEXTURE_MARK = "_gltf"

#: Сервер принимает в путях WEB только `[A-Za-z0-9._- ]`. Комплекты с
#: кириллицей в имени (а такие у нас есть) в WEB отправить нельзя: glTF
#: ссылается на текстуры по имени папки, переименовать её на лету — значит
#: сломать ссылки.
_WEB_PATH_ALLOWED = ("abcdefghijklmnopqrstuvwxyz"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")

#: Куда складывать OBJ, сконвертированные из .bam. Рядом с комплектом класть
#: нельзя: `plan_model_set_removal` посчитает такой файл частью набора, а
#: `_KIT_SUFFIXES` генератора о нём не знает — вышел бы файл-сирота, который
#: переживает удаление комплекта. Своя папка в стороне от обоих сканеров
#: (`assets/models/trucks` и `.../generated`) этой проблемы не создаёт.
CONVERT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "models", "_registry_cache")


@dataclass
class UploadPlan:
    """Заготовка запроса: ключ, файлы по ролям, текстуры и `meta`."""

    key: str = ""
    display_name: str = ""
    roles: Dict[str, str] = field(default_factory=dict)
    textures: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    #: Поля meta, которые мы посчитали сами, а не взяли из конфига набора.
    guessed: List[str] = field(default_factory=list)
    #: Роли, которые можно получить конвертацией из .bam: роль -> путь к .bam.
    convertible: Dict[str, str] = field(default_factory=dict)
    #: Комплект веб-вьюера: [(имя внутри WEB/, путь на диске)].
    web_files: List[Tuple[str, str]] = field(default_factory=list)
    #: Почему WEB-комплект отправить нельзя (пусто — можно).
    web_blocked: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def missing_required(self) -> List[str]:
        return [r for r in REQUIRED_ROLES if not self.roles.get(r)]


def _stem_of(path: str) -> str:
    name = os.path.basename(path)
    low = name.lower()
    for suffix in _STEM_SUFFIXES:
        if low.endswith(suffix.lower()):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _role_of(filename: str, stem: str) -> str:
    low = filename.lower()
    rest = low[len(stem):] if low.startswith(stem.lower()) else low
    for suffix, role in _SUFFIX_ROLES:
        if rest == suffix:
            return role
    return ""


def detect_role_files(model_path: str,
                      config: Optional[Dict[str, Any]] = None
                      ) -> Tuple[Dict[str, str], List[str]]:
    """
    Найти файлы комплекта рядом с кузовом.

    Возвращает ({роль: путь}, [файлы текстур]). Опознаются файлы, чьё имя —
    это `<stem>` плюс известный суффикс; чужие соседи по папке (другие
    комплекты генератора лежат там же) не подхватываются.
    """
    roles: Dict[str, str] = {}
    textures: List[str] = []
    if not model_path:
        return roles, textures

    base_dir = os.path.dirname(os.path.abspath(model_path))
    stem = _stem_of(model_path)
    if not os.path.isdir(base_dir):
        return roles, textures

    try:
        entries = sorted(os.listdir(base_dir))
    except OSError:
        entries = []

    for entry in entries:
        full = os.path.join(base_dir, entry)
        if not os.path.isfile(full):
            continue
        role = _role_of(entry, stem)
        if role and role not in roles:
            roles[role] = full

    # Конфиг набора точнее, чем угадывание по именам: у генератора в
    # `target_model` лежит именно тот OBJ, по которому считается объём.
    cfg = config or {}
    for cfg_key, role in (("target_model", "napolnitel_obj"),
                          ("cuzov", "cuzov_bam"),
                          ("napolnitel", "napolnitel_bam"),
                          ("other", "other_bam")):
        path = cfg.get(cfg_key)
        if not path or not isinstance(path, str) or not os.path.isfile(path):
            continue
        low = path.lower()
        if role.endswith("_obj") and not low.endswith(".obj"):
            continue
        if role.endswith("_bam") and not low.endswith(".bam"):
            continue
        roles.setdefault(role, os.path.abspath(path))

    # Текстуры комплекта лежат в папке рядом, названной именем комплекта.
    # JPEG с пометкой `_gltf` — это ужатые копии для веб-вьюера, им место в
    # WEB/, а не в data/textures/ (см. `detect_web_files`).
    tex_dir = os.path.join(base_dir, stem)
    if os.path.isdir(tex_dir):
        for entry in sorted(os.listdir(tex_dir)):
            low = entry.lower()
            if low.endswith(_TEXTURE_EXTS) and _WEB_TEXTURE_MARK not in low:
                textures.append(os.path.join(tex_dir, entry))
    return roles, textures


def detect_web_files(model_path: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Комплект веб-вьюера рядом с кузовом: `[(имя внутри WEB/, путь)]`.

    Возвращает ещё и причину, по которой комплект отправлять нельзя (или "").
    Имена сохраняются относительными (`<stem>.bin`, `<stem>/tex.jpg`) — glTF
    ссылается на них именно так, и «сплющивание» в basename ломает вьюер.
    """
    out: List[Tuple[str, str]] = []
    if not model_path:
        return out, ""

    base_dir = os.path.dirname(os.path.abspath(model_path))
    stem = _stem_of(model_path)
    gltf = os.path.join(base_dir, f"{stem}.gltf")
    if not os.path.isfile(gltf):
        return out, ""

    out.append((f"{stem}.gltf", gltf))
    buf = os.path.join(base_dir, f"{stem}.bin")
    if os.path.isfile(buf):
        out.append((f"{stem}.bin", buf))

    tex_dir = os.path.join(base_dir, stem)
    if os.path.isdir(tex_dir):
        for entry in sorted(os.listdir(tex_dir)):
            low = entry.lower()
            if low.endswith(_TEXTURE_EXTS) and _WEB_TEXTURE_MARK in low:
                out.append((f"{stem}/{entry}", os.path.join(tex_dir, entry)))

    bad = sorted({ch for name, _ in out for ch in name
                  if ch not in _WEB_PATH_ALLOWED and ch != "/"})
    if bad:
        return [], ("сервер принимает в путях WEB только латиницу, цифры, "
                    "точку, дефис, подчёркивание и пробел, а в именах файлов "
                    "комплекта есть " + " ".join(repr(ch) for ch in bad)
                    + ". Переименуйте комплект — glTF ссылается на текстуры "
                      "по имени папки, и переименовать её на лету нельзя")
    return out, ""


def _rect_from_set_json(data: Dict[str, Any]) -> Optional[List[List[float]]]:
    """
    Грубая заготовка `points_3d` из `<stem>.set.json`, когда меш недоступен.

    Габариты наполнителя — плохая мерка для верхней кромки: наполнитель нарочно
    продолжен вверх на «горку» (метр по умолчанию), а рамка вокруг завалённых
    торцов длиннее проёма. По XY погрешность в пределах пары сантиметров и
    терпима, а вот верх рамки надо опустить с потолка «горки» на борт — высота
    берётся из объёма полости: `max_volume` в `.set.json` считается ровно до
    верха борта, так что `объём / площадь дна` даёт высоту борта с точностью до
    скруглений (на комплектах генератора это 5–8 см).

    Порядок точек — как на сервере: по часовой стрелке от дальнего правого
    угла, все на одной высоте.
    """
    bounds = (data.get("world_bounds") or {}).get("napolnitel") \
        or (data.get("world_bounds") or {}).get("body")
    try:
        (x0, y0, z0), (x1, y1, z1) = bounds
        x0, y0, z0 = float(x0), float(y0), float(z0)
        x1, y1, z1 = float(x1), float(y1), float(z1)
    except (TypeError, ValueError):
        return None

    x_lo, x_hi = min(x0, x1), max(x0, x1)
    y_lo, y_hi = min(y0, y1), max(y0, y1)
    top = max(z0, z1)
    area = (x_hi - x_lo) * (y_hi - y_lo)
    try:
        volume = float(data.get("max_volume") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    if volume > 0.0 and area > 1e-6:
        top = min(top, min(z0, z1) + volume / area)

    top = round(top, 4)
    return [[x_hi, y_hi, top], [x_lo, y_hi, top],
            [x_lo, y_lo, top], [x_hi, y_lo, top]]


def _read_set_json(model_path: str) -> Dict[str, Any]:
    import json
    stem = _stem_of(model_path)
    path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                        f"{stem}.set.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def _guess_points(model_path: str, plan: UploadPlan
                  ) -> Tuple[Optional[List[List[float]]], str, bool]:
    """
    Заготовка `points_3d` для комплекта: (точки, строчка в журнал, точно ли).

    Три источника по убыванию доверия:

    1. `points_3d` в `<stem>.set.json` — их пишет сам генератор из спека
       кузова, точнее уже некуда;
    2. обмер меша кузова (`src/registry/anchor_points`) — работает и с
       комплектами, которые генератор не собирал, и со старыми комплектами
       генератора, у которых в `.set.json` точек ещё нет;
    3. габариты комплекта из `.set.json` — грубо, но лучше пустого поля.

    Обмер меша занимает секунду на крупном кузове (300 тыс. треугольников), и
    делается он один раз при открытии диалога — асинхронности ради этого не
    городим.
    """
    data = _read_set_json(model_path)
    ready = data.get("points_3d")
    if isinstance(ready, list) and len(ready) == 4:
        return ([[float(v) for v in point] for point in ready],
                "points_3d взяты из .set.json — их посчитал генератор кузова "
                "по обмерам, править не нужно", True)

    mesh = plan.roles.get("cuzov_bam") or plan.roles.get("cuzov_obj") \
        or model_path
    rect, note = points_from_mesh(mesh)
    if rect:
        return rect, ("points_3d обмерены " + note
                      + " — сверьте с верхней кромкой перед загрузкой"), False

    rough = _rect_from_set_json(data)
    if rough:
        return rough, ("points_3d посчитаны по габаритам комплекта: " + note
                       + ". Это грубая оценка, проверьте верхнюю кромку "
                         "кузова перед загрузкой"), False
    return None, ("points_3d посчитать не по чему: " + note
                  + "; задайте четыре точки верхней кромки вручную"), False


def build_upload_plan(info: Any) -> UploadPlan:
    """
    Заготовка загрузки для набора из списка кузовов (`ModelSetInfo`).

    Ничего не проверяет на сервере — это работа диалога. Всё, чего в наборе
    нет, остаётся пустым: пользователь дозаполнит руками.
    """
    cfg = dict(getattr(info, "config", {}) or {})
    model_path = str(getattr(info, "path", "") or "") or str(cfg.get("cuzov")
                                                             or "")
    name = str(getattr(info, "name", "") or getattr(info, "key", ""))

    plan = UploadPlan(key=suggest_key(name) or suggest_key(
        str(getattr(info, "key", ""))), display_name=name)

    plan.roles, plan.textures = detect_role_files(model_path, cfg)
    plan.web_files, plan.web_blocked = detect_web_files(model_path)
    if plan.web_blocked:
        plan.notes.append("веб-вьюер: " + plan.web_blocked)

    # Кузов в OBJ генератор не пишет — но его можно получить из .bam.
    if not plan.roles.get("cuzov_obj") and plan.roles.get("cuzov_bam"):
        plan.convertible["cuzov_obj"] = plan.roles["cuzov_bam"]
    if not plan.roles.get("napolnitel_obj") and plan.roles.get(
            "napolnitel_bam"):
        plan.convertible["napolnitel_obj"] = plan.roles["napolnitel_bam"]

    meta: Dict[str, Any] = {"display_name": name}

    volume = getattr(info, "volume", None)
    if volume is not None:
        meta["max_volume"] = round(float(volume), 4)
    ground = getattr(info, "ground_plane", None)
    if ground is not None:
        meta["ground_plane"] = round(float(ground), 4)

    points = cfg.get("points_3d")
    if isinstance(points, list) and len(points) == 4:
        meta["points_3d"] = points
    else:
        rect, note, exact = _guess_points(model_path, plan)
        if rect:
            meta["points_3d"] = rect
            if not exact:
                plan.guessed.append("points_3d")
        if note:
            plan.notes.append(note)

    cam_pos = [cfg.get("cam_pos_x"), cfg.get("cam_pos_y"), cfg.get("cam_pos_z")]
    cam_rot = [cfg.get("cam_rot_h"), cfg.get("cam_rot_p"), cfg.get("cam_rot_r")]
    if all(v is not None for v in cam_pos) and all(v is not None
                                                   for v in cam_rot):
        meta["camera"] = {
            "pos": [float(v) for v in cam_pos],
            "hpr": [float(v) for v in cam_rot],
            "fov": float(cfg.get("fov") or 87.0),
        }

    if getattr(info, "volume", None) is None:
        plan.notes.append("в наборе не указан max_volume — без него сервер "
                          "не примет модель в режиме «полная замена»")
    plan.meta = meta
    return plan


def meta_from_form(display_name: str, max_volume: Optional[float],
                   ground_plane: Optional[float],
                   points_3d: Optional[List[List[float]]],
                   camera: Optional[Dict[str, Any]],
                   textures_dir: str = "",
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Собрать `meta` из полей диалога, выбрасывая незаполненное."""
    meta: Dict[str, Any] = {}
    if display_name:
        meta["display_name"] = display_name
    if max_volume is not None:
        meta["max_volume"] = max_volume
    if ground_plane is not None:
        meta["ground_plane"] = ground_plane
    if points_3d:
        meta["points_3d"] = points_3d
    if camera:
        meta["camera"] = camera
        # Легаси-поля вьюера: их читает наше же приложение, когда получает
        # конфиг с сервера, — без них модель встанет с камерой по умолчанию.
        pos, hpr = camera.get("pos") or [], camera.get("hpr") or []
        if len(pos) == 3 and len(hpr) == 3:
            meta.update({
                "cam_pos_x": pos[0], "cam_pos_y": pos[1], "cam_pos_z": pos[2],
                "cam_rot_h": hpr[0], "cam_rot_p": hpr[1], "cam_rot_r": hpr[2],
            })
    if textures_dir:
        meta["textures_dir"] = textures_dir
    if extra:
        meta["extra"] = extra
    return meta


# ---------------------------------------------------------------------------
# .bam -> .obj
# ---------------------------------------------------------------------------
def bam_to_obj(bam_path: str, out_path: str = "") -> str:
    """
    Выгрузить геометрию .bam в OBJ и вернуть путь к нему.

    Только вершины и треугольники: реестр и пайплайн ставят по OBJ анкеры,
    ни материалы, ни UV им не нужны. Координаты — мировые, как в .bam, то есть
    Z вверх; поворачивать их в Y-up НЕЛЬЗЯ — сервер трассирует по кузову лучи
    камеры с единичной матрицей (подробности в шапке модуля).

    Panda3D грузит .bam без окна (`Loader.get_global_ptr()`), так что функция
    работает и в headless-режиме.
    """
    from panda3d.core import (Filename, GeomVertexReader, Loader,
                              LoaderOptions, NodePath)

    bam_path = os.path.abspath(bam_path)
    if not os.path.isfile(bam_path):
        raise FileNotFoundError(bam_path)

    if not out_path:
        os.makedirs(CONVERT_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(bam_path))[0]
        out_path = os.path.join(CONVERT_DIR, f"{stem}.obj")

    node = Loader.get_global_ptr().load_sync(
        Filename.from_os_specific(bam_path), LoaderOptions())
    if node is None:
        raise RuntimeError(f"Panda3D не смог прочитать {bam_path}")
    root = NodePath(node)

    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    for geom_np in root.find_all_matches("**/+GeomNode"):
        geom_node = geom_np.node()
        # Трансформация узла относительно корня: у комплектов генератора
        # части сдвинуты (см. `offsets` в .set.json), и без неё кузов уедет.
        xform = geom_np.get_transform(root).get_mat()
        for i in range(geom_node.get_num_geoms()):
            geom = geom_node.get_geom(i).decompose()
            vdata = geom.get_vertex_data()
            base = len(verts)
            reader = GeomVertexReader(vdata, "vertex")
            while not reader.is_at_end():
                point = xform.xform_point(reader.get_data3())
                verts.append((point.x, point.y, point.z))
            for p in range(geom.get_num_primitives()):
                prim = geom.get_primitive(p)
                if prim.get_num_vertices_per_primitive() != 3:
                    continue
                indices = prim.get_vertex_list()
                for k in range(0, len(indices) - 2, 3):
                    faces.append((base + indices[k] + 1,
                                  base + indices[k + 1] + 1,
                                  base + indices[k + 2] + 1))

    if not faces:
        raise RuntimeError(f"в {os.path.basename(bam_path)} нет треугольников")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# сконвертировано утилитой из {os.path.basename(bam_path)}\n")
        fh.write(f"o {os.path.splitext(os.path.basename(out_path))[0]}\n")
        for x, y, z in verts:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a} {b} {c}\n")
    return out_path
