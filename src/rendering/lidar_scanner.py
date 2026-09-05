# lidar_scanner.py
#
# Виртуальный 3D-ЛИДАР: облако точек в .ply как ещё один выход датасета.
#
# Зачем отдельный модуль, а не «глубина в 3D». Карту глубины даёт GPU, но она
# по своей природе РАСТРОВАЯ: точки лежат регулярной сеткой пикселей, каждая
# ровно на луче своего пикселя, без промахов, без потерь и с шумом, которого
# у настоящего дальномера не бывает. Сеть, обученная на таком облаке, к
# реальному лидару не переносится: у него точки идут ПО ТРАЕКТОРИИ развёртки,
# плотность резко неоднородна, часть возвратов теряется, а дальность известна
# с точностью в единицы миллиметров.
#
# Поэтому здесь честная трассировка лучей по треугольникам сцены:
#
#   * развёртка — розетка Ризли (два встречно вращающихся клина), та самая
#     «непериодическая» схема, которую используют Livox и Unitree: узор
#     рисуется окружностями, самые мелкие из которых ложатся ровно на ось
#     сенсора, поэтому плотность максимальна там, куда смотрит камера.
#     Есть и вторая развёртка, «spin», — паспортная механика L2 (см.
#     _beam_directions);
#   * поле зрения у сенсора СВОЁ и с полем зрения камеры не связано: по
#     умолчанию 360° x 90°, как у Unitree 4D LiDAR L2. Общая с камерой
#     только ПОЗА — сенсор стоит В КАМЕРЕ и наклоняется вместе с ней, так
#     что случайная поза кадра из датасета становится позой лидара, а
#     облако при этом захватывает и то, что позади камеры;
#   * луч бьётся о ВСЮ ВИДИМУЮ геометрию сцены: груз, кузов, насыпь, ткань,
#     подложку, строения — всё, обо что он мог бы удариться, а не только о
#     то, что попало в кадр. Промах (небо) возврата не даёт — как и в
#     жизни;
#   * дальность зашумляется, часть возвратов теряется (сильнее на скользящих
#     углах), каждая точка получает интенсивность и МЕТКУ КЛАССА из той же
#     палитры, что и маска сегментации.
#
# Бэкенды трассировки (выбираются автоматически, см. _pick_backend):
#
#   warp-cuda  — NVIDIA Warp на GPU. Миллион лучей — единицы миллисекунд;
#   embree     — Intel Embree через trimesh (многопоточный CPU), ~0.8 с/млн;
#   warp-cpu   — тот же код Warp на CPU, ~15 с/млн. Последний рубеж.
#
# Warp стоит первым не «потому что модно»: на машине с CUDA он на порядок
# быстрее Embree и не требует копий геометрии в float64.
from __future__ import annotations

import math
import time
import weakref

import numpy as np

# ---------------------------------------------------------------------------
# Классы точек. Порядок задаёт числовой id в .ply; "background" (0) остаётся
# за геометрией, которую не удалось отнести ни к одному классу сегментации.
# ---------------------------------------------------------------------------
CLASS_ORDER = ("background", "cargo", "cuzov", "other", "ground", "cloth")
CLASS_ID = {name: i for i, name in enumerate(CLASS_ORDER)}

# Дальность, на которой ослабление возврата принимается за единицу: ближе
# этого расстояния интенсивность определяется только углом падения.
INTENSITY_REF_M = 10.0

# Цвет точек непомеченной геометрии, когда палитра отдаёт под «фон» чёрный.
SCENE_FALLBACK_RGB = (110, 110, 110)

# Лучей в одном пакете трассировки. Пакет нужен, чтобы набирать ЗАДАННОЕ
# число ВОЗВРАТОВ: доля попаданий заранее неизвестна (зависит от позы камеры
# и от того, сколько кадра занимает небо), поэтому лучи досылаются порциями,
# пока облако не наберётся.
BATCH = 262144

# Потолок числа выпущенных лучей относительно заказанного числа точек. Если
# камера смотрит в небо, попаданий не будет никогда — без потолка цикл
# досылки крутился бы вечно.
MAX_BEAM_FACTOR = 6.0


# ---------------------------------------------------------------------------
# Бэкенды трассировки
# ---------------------------------------------------------------------------
class _Backend:
    """Общий интерфейс: собрать BVH и выстрелить пачкой лучей.

    cast() возвращает (t, face): t — расстояние до попадания (<0 = промах),
    face — индекс треугольника. Нормаль здесь НЕ запрашивается: она
    считается из вершин треугольника одинаково для всех бэкендов, иначе
    интенсивность зависела бы от того, какой из них выбрался.
    """

    name = "none"

    def build(self, verts, faces):
        raise NotImplementedError

    def cast(self, origin, dirs, t_max):
        raise NotImplementedError


class _WarpBackend(_Backend):
    def __init__(self, wp, device, name):
        self.wp = wp
        self.device = device
        self.name = name
        self.mesh = None
        self._kernel = None
        self._buffers = None

    def _ensure_kernel(self):
        if self._kernel is not None:
            return
        wp = self.wp

        @wp.kernel
        def k_cast(mesh: wp.uint64,
                   origin: wp.vec3,
                   dirs: wp.array(dtype=wp.vec3),
                   t_max: float,
                   out_t: wp.array(dtype=float),
                   out_face: wp.array(dtype=wp.int32)):
            i = wp.tid()
            q = wp.mesh_query_ray(mesh, origin, dirs[i], t_max)
            if q.result:
                out_t[i] = q.t
                out_face[i] = q.face
            else:
                out_t[i] = -1.0
                out_face[i] = -1

        self._kernel = k_cast

    def build(self, verts, faces):
        wp = self.wp
        self.mesh = wp.Mesh(
            points=wp.array(np.ascontiguousarray(verts, dtype=np.float32),
                            dtype=wp.vec3, device=self.device),
            indices=wp.array(np.ascontiguousarray(faces.reshape(-1),
                                                  dtype=np.int32),
                             dtype=wp.int32, device=self.device),
        )
        self._buffers = None

    def cast(self, origin, dirs, t_max):
        wp = self.wp
        self._ensure_kernel()
        n = len(dirs)
        # Буферы переиспользуются между пакетами: пакеты одного размера, а
        # аллокация миллиона vec3 на каждый — заметная часть времени на CPU.
        if self._buffers is None or self._buffers[0] < n:
            self._buffers = (
                n,
                wp.zeros(n, dtype=float, device=self.device),
                wp.zeros(n, dtype=wp.int32, device=self.device),
            )
        _, t_buf, f_buf = self._buffers
        d_buf = wp.array(np.ascontiguousarray(dirs, dtype=np.float32),
                         dtype=wp.vec3, device=self.device)
        wp.launch(self._kernel, dim=n,
                  inputs=[self.mesh.id,
                          wp.vec3(float(origin[0]), float(origin[1]),
                                  float(origin[2])),
                          d_buf, float(t_max), t_buf, f_buf],
                  device=self.device)
        wp.synchronize()
        return t_buf.numpy()[:n].copy(), f_buf.numpy()[:n].copy()


class _EmbreeBackend(_Backend):
    """Embree через trimesh.

    intersects_first отдаёт только индекс треугольника — расстояние считаем
    сами пересечением луча с плоскостью грани. Это и быстрее (одна выдача
    вместо трёх массивов), и точнее: Embree внутри работает в float32, а
    плоскость мы решаем в float64.
    """

    name = "embree"

    def __init__(self, intersector_cls):
        self._cls = intersector_cls
        self._inter = None
        self._verts = None
        self._faces = None

    def build(self, verts, faces):
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False, validate=False,
        )
        self._inter = self._cls(mesh)
        self._verts = np.asarray(verts, dtype=np.float64)
        self._faces = np.asarray(faces, dtype=np.int64)

    def cast(self, origin, dirs, t_max):
        dirs64 = np.asarray(dirs, dtype=np.float64)
        origins = np.broadcast_to(np.asarray(origin, dtype=np.float64),
                                  dirs64.shape)
        face = np.asarray(self._inter.intersects_first(origins, dirs64),
                          dtype=np.int64)
        t = np.full(len(dirs64), -1.0)
        hit = face >= 0
        if np.any(hit):
            tri = self._verts[self._faces[face[hit]]]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            d = dirs64[hit]
            denom = np.einsum("ij,ij->i", n, d)
            # Луч, идущий вдоль грани, Embree бы и не вернул; страхуемся от
            # деления на ноль на вырожденных треугольниках.
            safe = np.abs(denom) > 1e-12
            num = np.einsum("ij,ij->i", n, tri[:, 0] - origins[hit])
            th = np.where(safe, num / np.where(safe, denom, 1.0), -1.0)
            th[~np.isfinite(th)] = -1.0
            th[th > t_max] = -1.0
            t[hit] = th
        face[t < 0] = -1
        return t, face.astype(np.int32)


_BACKEND_CACHE = None


def _try_warp():
    """(модуль, устройство, имя) либо None. CPU-устройство тоже годится."""
    try:
        import warp as wp
        wp.init()
    except Exception as exc:                      # noqa: BLE001
        print(f"[Lidar] Warp недоступен: {exc}")
        return None
    try:
        if wp.get_cuda_device_count():
            return wp, wp.get_cuda_device(), "warp-cuda"
    except Exception:
        pass
    return wp, "cpu", "warp-cpu"


def _try_embree():
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        return RayMeshIntersector
    except Exception:
        return None


def _pick_backend():
    """Лучший доступный бэкенд (кэшируется на процесс).

    Порядок неслучаен: Warp на CUDA быстрее Embree на порядок, но Warp на CPU
    медленнее Embree примерно в двадцать раз (замер: 1 млн лучей — 14.4 с
    против 0.75 с). Поэтому CPU-Warp стоит ПОСЛЕ Embree, а не сразу за CUDA.
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE or None

    warp = _try_warp()
    if warp is not None and warp[2] == "warp-cuda":
        _BACKEND_CACHE = _WarpBackend(*warp)
        print("[Lidar] трассировка: NVIDIA Warp (CUDA)")
        return _BACKEND_CACHE

    embree = _try_embree()
    if embree is not None:
        _BACKEND_CACHE = _EmbreeBackend(embree)
        print("[Lidar] трассировка: Embree (CPU); CUDA-устройства нет")
        return _BACKEND_CACHE

    if warp is not None:
        _BACKEND_CACHE = _WarpBackend(*warp)
        print("[Lidar] трассировка: Warp на CPU — медленно, но работает")
        return _BACKEND_CACHE

    print("[Lidar] нет ни Warp, ни Embree — облако точек не снимается")
    _BACKEND_CACHE = False
    return None


def backend_name():
    """Имя бэкенда для UI/метаданных; None, если трассировать нечем."""
    backend = _pick_backend()
    return backend.name if backend is not None else None


# ---------------------------------------------------------------------------
# Сбор геометрии сцены с классами
# ---------------------------------------------------------------------------
def _geom_triangles(gnp):
    """Треугольники ОДНОГО GeomNode в МИРОВЫХ координатах -> (V, F) | None.

    Гранулярность — именно геом-узел, а не поддерево: .bam-модель это
    несколько GeomNode со своими трансформациями, и кэш по поддереву протухал
    бы целиком из-за правки одного из них. decompose() обязателен — иначе
    тристрипы теряются.
    """
    from panda3d.core import GeomVertexReader

    gnode = gnp.node()
    mat = gnp.get_net_transform().get_mat()

    verts = []
    faces = []
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
        base = len(verts)
        got = False
        for pi in range(geom.get_num_primitives()):
            prim = geom.get_primitive(pi).decompose()
            idx = list(prim.get_vertex_list())
            for t in range(0, len(idx) - 2, 3):
                faces.append((idx[t] + base, idx[t + 1] + base,
                              idx[t + 2] + base))
                got = True
        if got:
            verts.extend(local)

    if not verts or not faces:
        return None
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int32))


class _GeometryCache:
    """Кэш треугольников по узлам сцены.

    Насыпь и кузов между кадрами не меняются, а разбор их GeomNode'ов — это
    десятки тысяч вызовов GeomVertexReader на кадр, то есть дороже самой
    трассировки. Кэш снимает эту цену со всех кадров, кроме первого.

    Ключ узла — это ЧТО (счётчик правок вершинных буферов) плюс ГДЕ (мировая
    трансформация). Счётчик правок (GeomVertexData.get_modified) в Panda
    глобален и только растёт, поэтому пересобранный меш никогда не совпадёт
    ключом со старым — даже если его подвесили в тот же узел, с тем же числом
    вершин и в ту же позу. Именно так каждое наполнение и попадает в кэш
    как новая геометрия.

    Адрес узла как идентичность не годится: NodePath.node() отдаёт временную
    обёртку, id() которой меняется от вызова к вызову, а слабую ссылку
    PandaNode не поддерживает вовсе. Стабилен NodePath.get_key() — он один и
    тот же у всех NodePath, указывающих на один узел.
    """

    # Узлов в сцене единицы; потолок нужен только чтобы кэш не рос вечно на
    # длинном прогоне, где каждое наполнение приносит новый меш груза.
    MAX_ENTRIES = 32

    def __init__(self):
        self._entries = {}

    @staticmethod
    def _key(gnp):
        mat = gnp.get_net_transform().get_mat()
        rows = tuple(round(float(mat.get_cell(r, c)), 6)
                     for r in range(4) for c in range(4))

        shape = []
        gnode = gnp.node()
        for gi in range(gnode.get_num_geoms()):
            geom = gnode.get_geom(gi)
            vdata = geom.get_vertex_data()
            # Читаются только СЧЁТЧИКИ, не вершины — микросекунды.
            shape.append((vdata.get_num_rows(),
                          vdata.get_modified().get_seq(),
                          geom.get_modified().get_seq()))
        return (rows, tuple(shape))

    def get(self, node_path):
        """(ключ, (V, F) | None). Ключ уходит в подпись сцены — по ней
        решается, пересобирать ли BVH."""
        ident = node_path.get_key()
        key = self._key(node_path)
        cached = self._entries.get(ident)
        if cached is not None and cached[0] == key:
            return key, cached[1]
        data = _geom_triangles(node_path)
        if len(self._entries) >= self.MAX_ENTRIES:
            # FIFO: словарь помнит порядок вставки, старейшая запись — первая.
            self._entries.pop(next(iter(self._entries)), None)
        self._entries[ident] = (key, data)
        return key, data

    def clear(self):
        self._entries.clear()


def _class_map(app):
    """{NodePath.get_key(): класс} — те же узлы и классы, что у маски.

    Источник правды один: SegmentationRenderer._resolve_seg_nodes. Иначе
    метки в облаке точек и цвета в маске разъехались бы при любой правке
    состава сцены.
    """
    items = []
    seg = getattr(app, "segmentation_renderer", None)
    if seg is not None and hasattr(seg, "_resolve_seg_nodes"):
        try:
            items = list(seg._resolve_seg_nodes())
        except Exception as exc:              # noqa: BLE001
            print(f"[Lidar] состав сцены у сегментации не получен: {exc}")

    if not items:
        for attr, cls in (("final_model", "cargo"),
                          ("ground_plane", "ground")):
            node = getattr(app, attr, None)
            if node is not None and not node.is_empty():
                items.append((node, cls))
        for node in (getattr(app, "loaded_models", None) or []):
            if node is not None and not node.is_empty():
                items.append((node, "other"))

    mapping = {}
    for node, class_name in items:
        if node is not None and not node.is_empty():
            mapping[node.get_key()] = class_name
    return mapping


def _excluded_roots(app):
    """Узлы, которых в облаке быть не должно (частицы и т.п.)."""
    keys = set()
    particles = getattr(app, "particles", None)
    node = getattr(particles, "node", None) if particles is not None else None
    if node is not None and not node.is_empty():
        keys.add(node.get_key())
    return keys


def _scene_geom_nodes(app):
    """Все ВИДИМЫЕ GeomNode сцены: [(NodePath, класс)].

    Раньше сюда попадали только классифицированные узлы (груз, кузов, насыпь,
    ткань) — и в облаке не было ни земли, ни окружения, потому что реальная
    подложка сцены к этим узлам не относится, а вспомогательная ground_plane
    после CSG вообще скрыта. Лидар — не камера сегментации: ему нужна ВСЯ
    геометрия, обо что луч может удариться, иначе половина возвратов
    пропадает и облако «висит в пустоте».

    Поэтому обход идёт по всему графу от render, а классы навешиваются по
    БЛИЖАЙШЕМУ классифицированному предку; всё остальное — "background".
    is_hidden() учитывает предков, так что скрытые ветки отсекаются целиком —
    облако видит ровно то же, что и камера.
    """
    render = getattr(app, "render", None)
    if render is None or render.is_empty():
        return []

    classes = _class_map(app)
    excluded = _excluded_roots(app)

    out = []
    for gnp in render.find_all_matches("**/+GeomNode"):
        if gnp.is_hidden():
            continue
        class_name = None
        skip = False
        node = gnp
        while not node.is_empty():
            key = node.get_key()
            if key in excluded:
                skip = True
                break
            if class_name is None and key in classes:
                class_name = classes[key]
            node = node.get_parent()
        if skip:
            continue
        out.append((gnp, class_name or "background"))
    return out


# ---------------------------------------------------------------------------
# Развёртка
# ---------------------------------------------------------------------------
def _beam_directions(index0, count, cfg, rng):
    """Направления лучей в системе сенсора (x — вправо, y — вперёд, z — вверх).

    Поле зрения у лидара СВОЁ и с полем зрения камеры никак не связано. Общая
    с камерой только ПОЗА: сенсор стоит в камере, его ось вращения — это ось
    «вверх» камеры, а нулевой азимут — её направление взгляда. Наклонили
    камеру — вместе с ней наклонилась и вся полоса обзора.

    ПОЧЕМУ ФАЗЫ СЛУЧАЙНЫЕ, А НЕ ПО ПОРЯДКУ. Развёртка — это ОДНА кривая на
    сфере, и если идти по ней подряд, точки ложатся ниткой: замер на
    паспортных частотах L2 даёт шаг 0.44° вдоль траектории при 0.20° между
    соседними её проходами. То есть точки вдоль нитки РЕЖЕ, чем сами нитки
    друг к другу, — глаз читает это как пересекающиеся линии сканирования,
    которых на реальных снимках нет. У живого сенсора миллион точек
    копится секундами: за это время платформа дрожит, фаза вращения уходит,
    и от идеальной кривой остаётся только ПЛОТНОСТЬ — куда механика чаще
    заводит луч, там точек больше.

    Поэтому фазы развёртки берутся случайными, а формулы механики остаются
    ровно те же: распределение углов совпадает с траекторным до последнего
    знака, а нитки исчезают. Кому нужна именно кривая (посмотреть на узор) —
    настройка «trajectory» возвращает последовательный обход.

    «spin» — механика Unitree L2. Голова крутится вокруг оси «вверх» с
    частотой spin_hz (паспортные 5.55 Гц), быстрый элемент гоняет луч по
    КОНУСУ вокруг оси головы с частотой vertical_hz (216 Гц). Половина угла
    конуса — это и есть половина вертикального поля зрения: 45° дают
    паспортные 360°x90°. В траекторном режиме за оборот головы укладывается
    216/5.55 = 38.9 окружности через 9.25° по азимуту; отношение нецелое,
    поэтому следующий оборот кладёт их в промежутки предыдущего.

    Вертикальный обзор шире полусферы (>180°) означает, что луч переваливает
    через полюс: угол места физически не бывает больше 90°, поэтому «96° вверх
    и 96° вниз» — это полный круговой охват с перехлёстом в 6° за полюсами.
    У розетки то же самое выражается углом раствора конуса: 360° — полная
    сфера.

    «rosette» — розетка Ризли (два встречно вращающихся клина), схема Livox:
    луч отклоняется от оси КАМЕРЫ на угол

        theta = half_v * |cos((a1 - a2) / 2)|,  азимут вокруг оси = (a1+a2)/2,

    то есть узор рисуется окружностями, стягивающимися к оси взгляда:
    плотность точек на единицу телесного угла максимальна ровно там, куда
    смотрит камера. Это КОНУСНЫЙ сенсор — его ширина задаётся вертикальным
    полем зрения, а обзор по азимуту к нему не применяется (360° по кругу и
    сгущение в одну точку — вещи несовместимые).

    index0 — глобальный номер первого луча: в траекторном режиме фаза
    продолжается между пакетами, иначе каждый пакет рисовал бы одну и ту же
    розетку поверх предыдущей.
    """
    half_h = math.radians(float(cfg["fov_h_deg"])) * 0.5
    half_v = math.radians(float(cfg["fov_v_deg"])) * 0.5
    jitter = math.radians(float(cfg["jitter_deg"]))
    walk = bool(cfg["trajectory"])
    i = np.arange(index0, index0 + count, dtype=np.float64)

    if cfg["pattern"] == "spin":
        if walk:
            # Время отсчёта: развёртка задана В ГЕРЦАХ, как в паспорте.
            t = i / max(1.0, float(cfg["point_rate"]))
            alpha = 2.0 * math.pi * float(cfg["spin_hz"]) * t
            psi = 2.0 * math.pi * float(cfg["vertical_hz"]) * t
        else:
            # Голова крутится равномерно -> азимут равновероятен; фаза
            # вертикальной развёртки тоже равномерна. Плотность по углу
            # места при этом получается ТА ЖЕ, что и у обхода.
            alpha = rng.uniform(-math.pi, math.pi, size=count)
            psi = rng.uniform(0.0, 2.0 * math.pi, size=count)

        sa = np.sin(alpha)
        ca = np.cos(alpha)

        if half_v <= 0.5 * math.pi + 1e-9:
            # Конус вокруг оси головы h = (sin a, cos a, 0). Ортонормированная
            # пара поперёк неё: u1 = (0,0,1) и u2 = h x u1. Луч =
            # h*cos(beta) + (u1*cos(psi) + u2*sin(psi))*sin(beta), угол места
            # пробегает +-beta — то есть вертикальный обзор равен 2*beta.
            beta = np.full(len(alpha), half_v)
            if jitter > 0.0:
                # Дрожание конуса и привода: ломает идеально гладкую
                # траекторию, по которой сеть иначе выучивает развёртку.
                beta = beta + rng.normal(0.0, jitter, size=len(alpha))
                psi = psi + rng.normal(0.0, jitter, size=len(alpha))
            sb = np.sin(beta)
            cb = np.cos(beta)
            dx = sa * cb + ca * np.sin(psi) * sb
            dy = ca * cb - sa * np.sin(psi) * sb
            dz = np.cos(psi) * sb
        else:
            # Обзор ШИРЕ ПОЛУСФЕРЫ. Конусом его не выразить: при половине
            # угла больше 90° конус раскрывается назад и полоса начинает
            # СУЖАТЬСЯ (у beta = 96° остаётся +-84°, да ещё с дырой у
            # полюсов). Механика тут другая — не наклонённый конус, а
            # зеркало, ведущее луч по БОЛЬШОМУ КРУГУ в вертикальной
            # плоскости: угол phi отсчитывается от оси головы, и за 90° луч
            # переваливает через полюс, выходя с противоположного азимута.
            # На 180° обе ветки совпадают тождественно (конус с beta = 90°
            # и есть большой круг), поэтому шва между ними нет.
            if walk:
                # Зеркало вращается равномерно и гасится вне сектора —
                # качать его на 192° туда-сюда нечем.
                frac = (psi / (2.0 * math.pi)) % 1.0
                phi = -half_v + 2.0 * half_v * frac
            else:
                phi = rng.uniform(-half_v, half_v, size=len(alpha))
            if jitter > 0.0:
                phi = phi + rng.normal(0.0, jitter, size=len(alpha))
                alpha = alpha + rng.normal(0.0, jitter, size=len(alpha))
                sa = np.sin(alpha)
                ca = np.cos(alpha)
            cp = np.cos(phi)
            dx = sa * cp
            dy = ca * cp
            dz = np.sin(phi)

        if half_h < math.pi - 1e-6:
            # Секторный сенсор: лучи вне сектора просто не излучаются.
            # Азимут берём с ГОТОВОГО луча — за полюсом он развёрнут на 180°.
            keep = np.abs(np.arctan2(dx, dy)) <= half_h
            dx, dy, dz = dx[keep], dy[keep], dz[keep]
    else:
        ratio = float(cfg["circle_ratio"])
        if walk:
            per_circle = max(8.0, float(cfg["beams_per_circle"]))
            w1 = 2.0 * math.pi / per_circle
            a1 = i * w1
            a2 = -i * w1 * ratio
        else:
            # Клинья крутятся с несоизмеримыми скоростями, поэтому их фазы
            # равномерны и НЕЗАВИСИМЫ — сумма двух равномерных фаз даёт то же
            # распределение радиуса |cos((a1-a2)/2)|, что и обход.
            a1 = rng.uniform(0.0, 2.0 * math.pi, size=count)
            a2 = rng.uniform(0.0, 2.0 * math.pi, size=count)

        u = np.cos(a1) + np.cos(a2)
        v = np.sin(a1) + np.sin(a2)
        # Нормируем на 2 (максимум суммы двух единичных векторов) — радиус
        # приходит в долях половины поля зрения.
        r = np.hypot(u, v) * 0.5
        psi = np.arctan2(v, u)

        # Сгущение к центру: r' = r^bias. Показатель >1 подтягивает окружности
        # к оси, <1 растаскивает их к краю. Физическая розетка — это bias = 1.
        bias = float(cfg["center_bias"])
        if abs(bias - 1.0) > 1e-3:
            r = np.power(np.clip(r, 0.0, 1.0), max(0.05, bias))

        # theta до 180° — конус, раскрытый в полную сферу; формула
        # (sin t cos p, cos t, sin t sin p) верна на всём диапазоне.
        theta = r * half_v
        if jitter > 0.0:
            theta = theta + rng.normal(0.0, jitter, size=theta.shape)
            # Поперёк луча дрожание тоже равно jitter: sin(theta)*dpsi.
            psi = psi + rng.normal(0.0, jitter, size=psi.shape) / np.maximum(
                np.sin(np.abs(theta)), 1e-3)
        theta = np.clip(theta, 0.0, half_v)

        st = np.sin(theta)
        dx = st * np.cos(psi)
        dy = np.cos(theta)
        dz = st * np.sin(psi)

    dirs = np.empty((len(dx), 3), dtype=np.float32)
    dirs[:, 0] = dx        # вправо
    dirs[:, 1] = dy        # вперёд (ось взгляда камеры)
    dirs[:, 2] = dz        # вверх
    # Дрожание конуса чуть сбивает длину — нормируем, трассировщику нужен
    # единичный вектор.
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    return dirs


# ---------------------------------------------------------------------------
# Результат съёмки
# ---------------------------------------------------------------------------
class LidarScan:
    """Облако точек одного кадра плюс всё, что о нём нужно знать датасету."""

    def __init__(self, points, distances, intensity, labels, colors,
                 scan_time, meta):
        self.points = points          # (N,3) float32
        self.distances = distances    # (N,)  float32, метры
        self.intensity = intensity    # (N,)  float32, 0..1
        self.labels = labels          # (N,)  uint8, см. CLASS_ORDER
        self.colors = colors          # (N,3) uint8 | None
        self.scan_time = scan_time    # (N,)  float32, 0..1 — фаза развёртки
        self.meta = meta

    def __len__(self):
        return len(self.points)

    # ------------------------------------------------------------------
    def write_ply(self, path, binary=True, with_color=True):
        """Записать облако в .ply.

        Свойства сверх xyz: intensity (0..1), метка класса, фаза развёртки и
        (опционально) цвет класса из палитры сегментации. Цвет дублирует
        метку, но без него облако не открыть глазами ни в одном вьюере.
        """
        n = len(self.points)
        colors = self.colors if (with_color and self.colors is not None) else None

        header = [
            "ply",
            "format %s 1.0" % ("binary_little_endian" if binary else "ascii"),
            "comment IQoko 3D-simulator virtual lidar",
        ]
        for key in ("backend", "frame", "pattern", "fov_h_deg", "fov_v_deg",
                    "accuracy_mm", "max_range_m", "sensor_position",
                    "sensor_forward"):
            value = self.meta.get(key)
            if value is not None:
                header.append(f"comment {key} {value}")
        header += [
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float intensity",
            "property float range",
            "property float scan_t",
            "property uchar label",
        ]
        if colors is not None:
            header += ["property uchar red",
                       "property uchar green",
                       "property uchar blue"]
        header.append("end_header")
        blob = ("\n".join(header) + "\n").encode("ascii")

        fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                  ("intensity", "<f4"), ("range", "<f4"), ("scan_t", "<f4"),
                  ("label", "u1")]
        if colors is not None:
            fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        rows = np.empty(n, dtype=np.dtype(fields))
        rows["x"] = self.points[:, 0]
        rows["y"] = self.points[:, 1]
        rows["z"] = self.points[:, 2]
        rows["intensity"] = self.intensity
        rows["range"] = self.distances
        rows["scan_t"] = self.scan_time
        rows["label"] = self.labels
        if colors is not None:
            rows["red"] = colors[:, 0]
            rows["green"] = colors[:, 1]
            rows["blue"] = colors[:, 2]

        with open(path, "wb") as fh:
            fh.write(blob)
            if binary:
                rows.tofile(fh)
            else:
                # ascii пишем сами: np.savetxt на миллионе точек и с
                # разнотипными колонками работает минутами.
                fmt = "%.4f %.4f %.4f %.4f %.4f %.5f %d"
                if colors is not None:
                    fmt += " %d %d %d"
                cols = [self.points[:, 0], self.points[:, 1],
                        self.points[:, 2], self.intensity, self.distances,
                        self.scan_time, self.labels]
                if colors is not None:
                    cols += [colors[:, 0], colors[:, 1], colors[:, 2]]
                stacked = np.column_stack(cols)
                fh.write("\n".join(fmt % tuple(row)
                                   for row in stacked).encode("ascii"))
                fh.write(b"\n")
        return path


# ---------------------------------------------------------------------------
# Сканер
# ---------------------------------------------------------------------------
class LidarScanner:
    """Съёмка облака точек с позы текущей камеры."""

    def __init__(self, panda_app):
        self.app = panda_app
        self._cache = _GeometryCache()
        self._built_signature = None
        self._face_class = None
        self._verts = None
        self._faces = None

    # ------------------------------------------------------------------
    def available(self):
        return _pick_backend() is not None

    # ------------------------------------------------------------------
    def _collect_geometry(self, max_range):
        """Треугольный суп ВСЕЙ видимой сцены + класс на каждый треугольник.

        Возвращает (verts, faces, face_class, signature) либо None. signature
        меняется, только если геометрия действительно другая — по нему
        решается, пересобирать ли BVH.
        """
        chunks_v = []
        chunks_f = []
        chunks_c = []
        signature = []
        offset = 0
        # Задник (небосвод RenderPipeline масштабирован в 40000) — это не
        # поверхность, обо что бьётся луч, а ОБОЛОЧКА вокруг всей сцены:
        # она велика по ВСЕМ ТРЁМ осям сразу. Отсекаем по размеру, а не по
        # имени: у скайбокса имя узла "Sphere" в группе "SceneRoot", по нему
        # его не отличить. Заодно страховка от великанского max_range: без
        # отсечки небо превратилось бы в сплошную стену возвратов.
        #
        # Меряем по МЕНЬШЕМУ габариту, а не по большему. Базовая плоскость
        # мира (create_perlin_noise_mesh: 2000 x 2000 под всеми строениями и
        # грузовиком) по площади крупнее любого горизонта сенсора, но по
        # высоте у неё нуль — оболочку это не напоминает. Пока сравнивался
        # больший габарит, земля улетала в отсев вместе с небом, и в облаке
        # не было ни одной её точки: луч уходил в пустоту там, где в кадре
        # лежит асфальт.
        backdrop_limit = max(1000.0, 20.0 * float(max_range))
        backdrops = 0

        for gnp, class_name in _scene_geom_nodes(self.app):
            key, data = self._cache.get(gnp)
            if data is None:
                continue
            verts, faces = data
            if float(np.min(verts.max(axis=0) - verts.min(axis=0))) > \
                    backdrop_limit:
                backdrops += 1
                continue
            chunks_v.append(verts)
            chunks_f.append(faces + offset)
            chunks_c.append(np.full(len(faces),
                                    CLASS_ID.get(class_name, 0),
                                    dtype=np.uint8))
            # Подпись сцены = ключи её узлов. Тот же ключ, что у кэша,
            # поэтому изменившийся меш одновременно и промахивается мимо
            # кэша, и заставляет пересобрать BVH.
            signature.append((gnp.get_key(), class_name, key))
            offset += len(verts)

        if not chunks_v:
            return None
        if backdrops:
            signature.append(("backdrops", backdrops, backdrop_limit))
        return (np.concatenate(chunks_v),
                np.concatenate(chunks_f),
                np.concatenate(chunks_c),
                tuple(signature))

    def _ensure_bvh(self, backend, max_range):
        geom = self._collect_geometry(max_range)
        if geom is None:
            print("[Lidar] в сцене нет геометрии — облако не снято")
            return False
        verts, faces, face_class, signature = geom
        self._face_class = face_class
        self._verts = verts
        self._faces = faces
        if signature != self._built_signature:
            t0 = time.perf_counter()
            backend.build(verts, faces)
            self._built_signature = signature
            print(f"[Lidar] BVH: {len(faces)} треугольников за "
                  f"{time.perf_counter() - t0:.2f} с ({backend.name})")
        return True

    # ------------------------------------------------------------------
    def _sensor_frame(self):
        """(origin, right, forward, up) камеры в мировых координатах.

        Лидар стоит В КАМЕРЕ: ось сенсора = направление взгляда, поэтому
        случайная поза кадра из датасета автоматически становится позой
        сенсора, и мельчайшие окружности розетки ложатся туда, куда смотрит
        камера.

        Поза берётся с base.cam (узел с линзой), а не с base.camera: у
        родителя может быть своя трансформация.
        """
        app = self.app
        cam = getattr(app, "cam", None) or getattr(app, "camera", None)
        render = app.render
        pos = cam.get_pos(render)
        quat = cam.get_quat(render)
        origin = np.array([pos[0], pos[1], pos[2]], dtype=np.float64)

        def _v(vec):
            return np.array([vec[0], vec[1], vec[2]], dtype=np.float64)

        return (origin, _v(quat.get_right()), _v(quat.get_forward()),
                _v(quat.get_up()))

    # ------------------------------------------------------------------
    def scan(self, settings, seed=None):
        """Снять облако точек. Возвращает LidarScan либо None."""
        backend = _pick_backend()
        if backend is None:
            return None

        cfg = normalize_settings(settings)
        rng = np.random.default_rng(seed)

        if not self._ensure_bvh(backend, cfg["max_range_m"]):
            return None

        origin, right, forward, up = self._sensor_frame()
        basis = np.stack([right, forward, up])       # (3,3): строки — оси

        target = int(rng.integers(int(cfg["points_min"]),
                                  int(cfg["points_max"]) + 1))
        max_beams = int(target * MAX_BEAM_FACTOR)
        t_max = float(cfg["max_range_m"])
        t_min = float(cfg["min_range_m"])
        sigma = float(cfg["accuracy_mm"]) / 1000.0 / 3.0
        dropout = float(cfg["dropout_pct"]) / 100.0

        parts = []
        got = 0
        emitted = 0
        t_start = time.perf_counter()

        while got < target and emitted < max_beams:
            count = int(min(BATCH, max_beams - emitted))
            local = _beam_directions(emitted, count, cfg, rng)
            # Секторный «spin» часть лучей не излучает вовсе — тогда номер
            # луча уже не индекс в пакете, но для фазы развёртки хватает
            # линейной шкалы по излучённым.
            # Номер луча в развёртке; в долю от полной развёртки переведём
            # в конце — сколько лучей уйдёт всего, здесь ещё не известно.
            phase = np.linspace(emitted, emitted + count, num=len(local),
                                endpoint=False, dtype=np.float64)
            emitted += count
            if len(local) == 0:
                continue

            world = local.astype(np.float64) @ basis     # (N,3) в мире
            world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True),
                                1e-12)

            dist, face = backend.cast(origin, world.astype(np.float32), t_max)

            hit = (dist >= t_min) & (dist <= t_max) & (face >= 0)
            if not np.any(hit):
                continue

            dist = dist[hit].astype(np.float64)
            face = face[hit].astype(np.int64)
            local_hit = local[hit]
            world_hit = world[hit]
            phase = phase[hit]

            # Угол падения по ГЕОМЕТРИЧЕСКОЙ нормали грани: она одинакова у
            # всех бэкендов, в отличие от нормали из их собственных запросов.
            tri = self._verts[self._faces[face]].astype(np.float64)
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nlen = np.linalg.norm(nrm, axis=1, keepdims=True)
            nrm = nrm / np.maximum(nlen, 1e-12)
            cos_inc = np.abs(np.einsum("ij,ij->i", nrm, world_hit))

            # Потери возврата. Скользящий луч рассеивает энергию мимо
            # приёмника, поэтому у граней «в профиль» промахов кратно больше,
            # чем номинальные проценты: это и рисует в облаке характерные
            # прорехи на бортах и на дальнем скате насыпи.
            if dropout > 0.0:
                p_drop = np.clip(dropout * (1.5 - cos_inc), 0.0, 1.0)
                keep = rng.random(len(dist)) >= p_drop
                if not np.any(keep):
                    continue
                dist, face, cos_inc = dist[keep], face[keep], cos_inc[keep]
                local_hit = local_hit[keep]
                world_hit = world_hit[keep]
                phase = phase[keep]

            # Шум дальномера. sigma = точность/3: паспортное «±3 мм» — это
            # практически весь разброс, то есть три сигмы, а не одна.
            if sigma > 0.0:
                dist = dist + rng.normal(0.0, sigma, size=dist.shape)
                dist = np.maximum(dist, t_min)

            falloff = np.minimum(1.0,
                                 (INTENSITY_REF_M / np.maximum(dist, 1e-3)) ** 2)
            intensity = np.clip(cos_inc * falloff, 0.0, 1.0)

            if cfg["frame"] == "world":
                pts = origin[None, :] + world_hit * dist[:, None]
            else:
                pts = local_hit.astype(np.float64) * dist[:, None]

            parts.append((
                pts.astype(np.float32),
                dist.astype(np.float32),
                intensity.astype(np.float32),
                self._face_class[face],
                phase,
            ))
            got += len(dist)

        if not parts:
            print("[Lidar] ни одного возврата: камера смотрит мимо геометрии")
            return None

        points = np.concatenate([p[0] for p in parts])[:target]
        dist = np.concatenate([p[1] for p in parts])[:target]
        intensity = np.concatenate([p[2] for p in parts])[:target]
        labels = np.concatenate([p[3] for p in parts])[:target]
        phase = np.concatenate([p[4] for p in parts])[:target]
        # Фаза развёртки 0..1: у настоящего сенсора это метка времени точки,
        # по ней восстанавливается порядок обхода внутри кадра.
        phase = (phase / max(1.0, float(emitted))).astype(np.float32)

        colors = None
        palette = self._palette()
        if palette is not None:
            lut = np.array([palette.get(name, (255, 255, 255))
                            for name in CLASS_ORDER], dtype=np.uint8)
            # В маске сегментации «фон» — это небо, и он честно чёрный. В
            # облаке под тем же классом идёт РЕАЛЬНАЯ геометрия окружения
            # (подложка, строения, всё непомеченное), и чёрные точки на
            # чёрном фоне вьюера просто не видно. Метка класса при этом
            # остаётся прежней — подменяется только цвет.
            if not lut[CLASS_ID["background"]].any():
                lut[CLASS_ID["background"]] = SCENE_FALLBACK_RGB
            colors = lut[labels]

        elapsed = time.perf_counter() - t_start
        meta = {
            "backend": backend.name,
            "frame": cfg["frame"],
            "axes": ("x=right, y=forward, z=up (Panda3D camera)"
                     if cfg["frame"] == "sensor" else "world (Panda3D render)"),
            "points": int(len(points)),
            "points_requested": int(target),
            "beams_emitted": int(emitted),
            "hit_rate": (float(got) / emitted) if emitted else 0.0,
            "seconds": round(elapsed, 3),
            "pattern": cfg["pattern"],
            "fov_h_deg": float(cfg["fov_h_deg"]),
            "fov_v_deg": float(cfg["fov_v_deg"]),
            "accuracy_mm": float(cfg["accuracy_mm"]),
            "range_sigma_m": sigma,
            "min_range_m": t_min,
            "max_range_m": t_max,
            "dropout_pct": float(cfg["dropout_pct"]),
            "jitter_deg": float(cfg["jitter_deg"]),
            "center_bias": float(cfg["center_bias"]),
            "beams_per_circle": float(cfg["beams_per_circle"]),
            "circle_ratio": float(cfg["circle_ratio"]),
            "trajectory": bool(cfg["trajectory"]),
            "spin_hz": float(cfg["spin_hz"]),
            "vertical_hz": float(cfg["vertical_hz"]),
            "point_rate": float(cfg["point_rate"]),
            "sensor_position": [round(float(v), 6) for v in origin],
            "sensor_forward": [round(float(v), 6) for v in forward],
            "sensor_up": [round(float(v), 6) for v in up],
            "sensor_right": [round(float(v), 6) for v in right],
            "classes": {name: CLASS_ID[name] for name in CLASS_ORDER},
            "class_counts": {
                name: int(np.count_nonzero(labels == CLASS_ID[name]))
                for name in CLASS_ORDER
            },
        }
        if len(points) < target:
            meta["truncated"] = True
            print(f"[Lidar] набрано {len(points)} из {target} точек за "
                  f"{emitted} лучей — сцена не закрывает поле зрения")

        print(f"[Lidar] {len(points)} точек за {elapsed:.2f} с "
              f"({backend.name}, попаданий "
              f"{meta['hit_rate'] * 100:.0f}%)")
        return LidarScan(points, dist, intensity, labels, colors, phase, meta)

    # ------------------------------------------------------------------
    def _palette(self):
        seg = getattr(self.app, "segmentation_renderer", None)
        if seg is not None and hasattr(seg, "get_palette"):
            try:
                return seg.get_palette()
            except Exception:                 # noqa: BLE001
                pass
        try:
            from src.rendering.segmentation_renderer import (
                SEG_COLORS, SEG_BACKGROUND,
            )
            palette = {"background": tuple(SEG_BACKGROUND)}
            palette.update({k: tuple(v) for k, v in SEG_COLORS.items()})
            return palette
        except Exception:                     # noqa: BLE001
            return None


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
# Умолчания — паспорт Unitree 4D LiDAR L2: поле зрения 360x90 (96 в режиме
# отрицательных углов), дальность 0.05-30 м, 128 000 отсчётов в секунду,
# горизонтальная развёртка 5.55 Гц, вертикальная 216 Гц.
DEFAULT_SETTINGS = {
    "points_min": 500000,
    "points_max": 1000000,
    "accuracy_mm": 3.0,
    "pattern": "spin",
    "fov_h_deg": 360.0,
    "fov_v_deg": 90.0,
    "min_range_m": 0.05,
    "max_range_m": 30.0,
    "center_bias": 1.0,
    "jitter_deg": 0.05,
    "dropout_pct": 1.5,
    "beams_per_circle": 4000.0,
    "circle_ratio": 0.618,
    "spin_hz": 5.55,
    "vertical_hz": 216.0,
    "point_rate": 128000.0,
    "trajectory": False,
    "frame": "sensor",
    "binary": True,
    "color": True,
}

FRAMES = ("sensor", "world")
PATTERNS = ("rosette", "spin")


def default_settings():
    return dict(DEFAULT_SETTINGS)


def _clamp(value, lo, hi, fallback):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return fallback


def normalize_settings(settings):
    """Дополнить умолчаниями и подрезать до рабочих границ."""
    cfg = dict(DEFAULT_SETTINGS)
    settings = settings or {}
    cfg.update({k: v for k, v in settings.items() if k in cfg})

    # Совместимость с конфигами, снятыми до появления собственного поля
    # зрения: тогда угол был один и задавал конус вокруг оси камеры.
    if "fov_deg" in settings and "fov_h_deg" not in settings:
        cfg["fov_h_deg"] = cfg["fov_v_deg"] = settings["fov_deg"]

    for key in ("points_min", "points_max"):
        try:
            cfg[key] = max(1000, min(20000000, int(cfg[key])))
        except (TypeError, ValueError):
            cfg[key] = DEFAULT_SETTINGS[key]
    if cfg["points_max"] < cfg["points_min"]:
        cfg["points_max"] = cfg["points_min"]

    cfg["accuracy_mm"] = _clamp(cfg["accuracy_mm"], 0.0, 500.0, 3.0)
    if cfg["pattern"] not in PATTERNS:
        cfg["pattern"] = DEFAULT_SETTINGS["pattern"]
    cfg["fov_h_deg"] = _clamp(cfg["fov_h_deg"], 1.0, 360.0, 360.0)
    # Потолок 360°, а не 180°: у сенсора шире полусферы луч уходит за полюс
    # (см. _beam_directions), и запрещать это незачем — паспортные 90/96° тут
    # только умолчание, а не предел.
    cfg["fov_v_deg"] = _clamp(cfg["fov_v_deg"], 1.0, 360.0, 90.0)
    cfg["spin_hz"] = _clamp(cfg["spin_hz"], 0.01, 1000.0, 5.55)
    cfg["vertical_hz"] = _clamp(cfg["vertical_hz"], 0.01, 100000.0, 216.0)
    cfg["point_rate"] = _clamp(cfg["point_rate"], 100.0, 1e8, 128000.0)
    cfg["min_range_m"] = _clamp(cfg["min_range_m"], 0.0, 1000.0, 0.05)
    cfg["max_range_m"] = _clamp(cfg["max_range_m"], 0.1, 10000.0, 30.0)
    if cfg["max_range_m"] <= cfg["min_range_m"]:
        cfg["max_range_m"] = cfg["min_range_m"] + 1.0
    cfg["center_bias"] = _clamp(cfg["center_bias"], 0.05, 8.0, 1.0)
    cfg["jitter_deg"] = _clamp(cfg["jitter_deg"], 0.0, 5.0, 0.05)
    cfg["dropout_pct"] = _clamp(cfg["dropout_pct"], 0.0, 90.0, 1.5)
    cfg["beams_per_circle"] = _clamp(cfg["beams_per_circle"], 8.0, 100000.0,
                                     4000.0)
    # Рациональное отношение замыкает траекторию: развёртка ложится в те же
    # борозды, и вместо облака получается решётка. Отодвигаем от простых
    # дробей, не спрашивая пользователя.
    ratio = _clamp(cfg["circle_ratio"], 0.01, 4.0, 0.618)
    if abs(ratio - round(ratio)) < 1e-3:
        ratio += 0.017
    cfg["circle_ratio"] = ratio

    ratio_spin = cfg["vertical_hz"] / max(1e-6, cfg["spin_hz"])
    if abs(ratio_spin - round(ratio_spin)) < 1e-3:
        # Целое число взмахов на оборот — развёртка замыкается в решётку из
        # вертикальных полос вместо облака.
        cfg["vertical_hz"] = cfg["vertical_hz"] + 0.11 * cfg["spin_hz"]

    if cfg["frame"] not in FRAMES:
        cfg["frame"] = "sensor"
    cfg["trajectory"] = bool(cfg["trajectory"])
    cfg["binary"] = bool(cfg["binary"])
    cfg["color"] = bool(cfg["color"])
    return cfg
