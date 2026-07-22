# cloth_warp.py
#
# GPU-солвер ткани на NVIDIA Warp — замена numpy-PBD из cloth_simulator.py.
#
# Что здесь принципиально лучше numpy-версии:
#
#   * XPBD вместо PBD. У PBD жёсткость зависит от числа итераций и от dt:
#     один и тот же «стиффнес 0.7» даёт разную ткань при разном бюджете шагов.
#     В XPBD жёсткость задаётся ПОДАТЛИВОСТЬЮ (compliance, м/Н), и результат
#     от расписания шагов не зависит — датасет становится воспроизводимым.
#
#   * Small-steps: много подшагов по одной итерации вместо одного шага с
#     шестью итерациями. При равной стоимости это сходится на порядок лучше
#     (Macklin et al. 2019) — исчезает «резиновость» и ткань перестаёт
#     растягиваться под собственным весом.
#
#   * Настоящий изгиб (dihedral bend по паре смежных треугольников) вместо
#     связи «через одну вершину». Связь через одну — это растяжение по
#     диагонали, она сопротивляется и ИЗГИБУ, и сдвигу, поэтому складки выходят
#     угловатыми и с преимущественным направлением вдоль сетки. Двугранный угол
#     штрафует ровно изгиб, поэтому складки получаются гладкими и изотропными.
#
#   * Точная коллизия по треугольникам (wp.Mesh BVH) вместо облака точек в
#     KD-дереве. Облако точек — это приближение поверхности с просветами:
#     шаг посева приходилось держать меньше барьера, упираться в лимит точек и
#     всё равно ловить протекание. BVH даёт ТОЧНОЕ расстояние до поверхности,
#     без бюджета точек и без просветов вообще.
#
#   * Самопересечение (hash grid). Numpy-версия его не считала — свисающие
#     складки проходили сквозь себя, и на маске сегментации это читалось как
#     ткань с «дырами» в силуэте.
#
#   * Аэродинамика по ТРЕУГОЛЬНИКАМ (площадь x нормаль), а не по вершинам с
#     усреднённой нормалью: полотно ловит поток пропорционально реальной
#     проекции площади, что и даёт хлопки вместо равномерного сноса.
#
# Симуляция целиком живёт на GPU: за шаг нет ни одной пересылки, поэтому
# бюджет шагов на сэмпл вырос примерно на два порядка при том же времени.
import math

import numpy as np

_EPS = 1e-9

_wp = None
_device = None


def warp_available():
    """Инициализировать Warp (один раз) и вернуть, доступен ли он."""
    global _wp, _device
    if _device is not None:
        return _device is not False
    try:
        import warp as wp
        wp.init()
        # CPU-Warp медленнее numpy-версии (нет ни SIMD-пакетов, ни графа
        # запусков), поэтому смысл имеет только CUDA.
        dev = wp.get_cuda_device() if wp.get_cuda_device_count() else None
        if dev is None:
            _device = False
            return False
        _wp = wp
        _device = dev
        return True
    except Exception as exc:              # noqa: BLE001 - откат обязан работать
        print(f"[Cloth] Warp недоступен ({exc}) — откат на numpy-солвер")
        _device = False
        return False


def _wp_mod():
    if not warp_available():
        raise RuntimeError("Warp недоступен")
    return _wp, _device


# ---------------------------------------------------------------------------
# Раскраска связей
# ---------------------------------------------------------------------------
def color_grid_constraints(idx, nx, ny):
    """Раскраска связей регулярной сетки без перебора -> (порядок, пакеты).

    Все связи полотна локальны: вершины одной связи лежат в окне m x m узлов
    (m = 2 для structural/shear, 3 для двугранного изгиба). Цвет — это тройка
    (r0 mod m, c0 mod m, форма связи внутри окна), где r0, c0 — левый верхний
    угол окна.

    Почему в ключ обязана входить ФОРМА: у горизонтального и вертикального
    ребра из одного узла угол окна общий, и по одному только остатку они
    попадали бы в один пакет, деля вершину. Одинаковая форма плюс углы,
    разнесённые кратно m, дают непересекающиеся окна — значит и общих вершин
    нет по построению.

    Это O(M) без единого прохода-перебора. Общая жадная раскраска
    (color_constraints) на сетке 170x150 занимала секунды НА КАЖДЫЙ сэмпл —
    больше, чем сама симуляция.

    Возвращает None, если окно шире ожидаемого (тогда нужен общий алгоритм).
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), []

    rows, cols = idx // nx, idx % nx
    r0, c0 = rows.min(axis=1), cols.min(axis=1)
    m = int(max((rows.max(axis=1) - r0).max(), (cols.max(axis=1) - c0).max())) + 1
    if m > 4 or m < 1:
        return None

    shape = (rows - r0[:, None]) * m + (cols - c0[:, None])   # (M, K)
    _, shape_id = np.unique(shape, axis=0, return_inverse=True)
    key = (shape_id * m + (r0 % m)) * m + (c0 % m)
    order = np.argsort(key, kind="stable")
    counts = np.bincount(key, minlength=int(key.max()) + 1)

    ranges = []
    start = 0
    for cnt in counts:
        if cnt:
            ranges.append((start, int(cnt)))
            start += int(cnt)
    return order, ranges


def color_constraints(idx, num_particles):
    """Разбить связи на пакеты без общих вершин -> [(start, count)], порядок.

    Общий (не привязанный к сетке) запасной вариант для color_grid_constraints.

    Внутри пакета связи решаются параллельно без гонок: каждая вершина
    встречается не более одного раза, поэтому запись в позицию не требует
    атомарных операций (а они здесь стоили бы дороже самой физики).

    Жадная раскраска векторизована. Один проход отбирает связи, которые
    владеют всеми своими вершинами («владелец» — претендент с наименьшим
    индексом), но такое множество независимо, а НЕ максимально: у регулярной
    сетки соседние связи идут подряд и глушат друг друга, поэтому за проход
    проходит лишь малая доля. Поэтому пакет добирается ВНУТРЕННИМ циклом, пока
    в него ещё что-то влезает, и только потом начинается следующий.

    Разница принципиальна для скорости: без добора сетка 120x120 давала ~1400
    пакетов вместо ~20, а стоимость шага — это число ЗАПУСКОВ ядер, то есть
    ровно число пакетов.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), []

    k = idx.shape[1]
    remaining = np.arange(len(idx), dtype=np.int64)
    order = []
    ranges = []
    start = 0

    while remaining.size:
        used = np.zeros(num_particles, dtype=bool)
        pool = remaining
        taken = []
        while pool.size:
            rows = idx[pool]
            free = ~np.any(used[rows], axis=1)      # не конфликтует с пакетом
            pool = pool[free]
            if pool.size == 0:
                break
            rows = idx[pool]
            # Владелец вершины — претендент с наименьшим индексом.
            # Инициализация ОБЯЗАНА быть больше любого индекса связи: с -1
            # минимум навсегда остаётся -1, ни одна связь не становится
            # владельцем, и раскраска вырождается в одну связь на пакет.
            first = np.full(num_particles, len(pool), dtype=np.int64)
            owner = np.arange(rows.size, dtype=np.int64) // k
            np.minimum.at(first, rows.reshape(-1), owner)
            keep = np.all(first[rows] == np.arange(len(pool))[:, None], axis=1)
            if not keep.any():                      # страховка от зацикливания
                keep = np.zeros(len(pool), dtype=bool)
                keep[0] = True
            sel = pool[keep]
            taken.append(sel)
            used[idx[sel].reshape(-1)] = True
            pool = pool[~keep]

        sel = np.concatenate(taken)
        order.append(sel)
        ranges.append((start, int(sel.size)))
        start += int(sel.size)
        mask = np.ones(len(remaining), dtype=bool)
        mask[np.searchsorted(remaining, sel)] = False
        remaining = remaining[mask]

    perm = np.concatenate(order)
    return perm, ranges


def dihedral_pairs(faces):
    """Пары смежных треугольников -> (i0, i1, i2, i3).

    i0, i1 — общее ребро; i2, i3 — противолежащие вершины двух треугольников.
    Именно на этой четвёрке считается двугранный угол.
    """
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # Три ребра каждого треугольника + противолежащая вершина.
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    opp = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    key = np.sort(e, axis=1)

    lex = np.lexsort((key[:, 1], key[:, 0]))
    key, opp = key[lex], opp[lex]
    same = np.all(key[1:] == key[:-1], axis=1)
    pos = np.nonzero(same)[0]
    if pos.size == 0:
        return np.zeros((0, 4), dtype=np.int64)

    # Ребро, разделяемое ровно двумя треугольниками (у сетки-полотна других и
    # нет; при не-многообразной геометрии лишние пары просто отбрасываются).
    return np.stack([key[pos, 0], key[pos, 1], opp[pos], opp[pos + 1]], axis=1)


def grid_faces(ny, nx):
    """Треугольники регулярной сетки (та же нарезка, что и в меше рендера)."""
    idx = np.arange(nx * ny).reshape(ny, nx)
    a = idx[:-1, :-1].reshape(-1)
    b = idx[1:, :-1].reshape(-1)
    c = idx[:-1, 1:].reshape(-1)
    d = idx[1:, 1:].reshape(-1)
    return np.concatenate([np.stack([a, b, c], axis=1),
                           np.stack([b, d, c], axis=1)])


# ---------------------------------------------------------------------------
# Ядра Warp
# ---------------------------------------------------------------------------
_KERNELS = {}

# Топология связей (и её раскраска) зависит ТОЛЬКО от размеров сетки, а размеры
# в датасете повторяются. Раскраска стоит секунды на плотной сетке, поэтому
# кэш снимает её со всех сэмплов, кроме первого с такими размерами.
_TOPOLOGY_CACHE = {}


def _cached(key, build):
    hit = _TOPOLOGY_CACHE.get(key)
    if hit is None:
        hit = build()
        if len(_TOPOLOGY_CACHE) > 64:
            _TOPOLOGY_CACHE.clear()
        _TOPOLOGY_CACHE[key] = hit
    return hit


def _build_kernels():
    """Собрать ядра один раз (компиляция кэшируется Warp'ом на диске)."""
    if _KERNELS:
        return _KERNELS
    wp, _ = _wp_mod()

    vec3 = wp.vec3

    @wp.func
    def wind_at(p: vec3, dir: vec3, speed: float, gust: float,
                turb: float, t: float) -> vec3:
        # То же поле, что и в numpy-версии: общий во времени порыв плюс
        # бегущие волны по пространству. Без волн фронт однороден и полотно
        # движется как жёсткая доска.
        g = 1.0 + gust * wp.sin(1.7 * t + 0.6)
        w = dir * (speed * g)
        if turb > 0.0:
            amp = speed * turb
            phase = 2.4 * p[0] + 1.9 * p[1] + 3.1 * p[2]
            w = w + vec3(amp * wp.sin(phase + 2.7 * t),
                         amp * wp.sin(0.8 * phase - 3.3 * t + 1.2),
                         0.6 * amp * wp.sin(1.3 * phase + 2.1 * t + 2.5))
        return w

    @wp.kernel
    def k_clear(f: wp.array(dtype=vec3)):
        f[wp.tid()] = vec3(0.0, 0.0, 0.0)

    @wp.kernel
    def k_clear_f(f: wp.array(dtype=float)):
        f[wp.tid()] = 0.0

    @wp.kernel
    def k_aero(x: wp.array(dtype=vec3),
               v: wp.array(dtype=vec3),
               faces: wp.array2d(dtype=wp.int32),
               force: wp.array(dtype=vec3),
               wind_dir: vec3, speed: float, gust: float, turb: float,
               drag: float, lift: float, time: wp.array(dtype=float)):
        # Сила на ТРЕУГОЛЬНИК: нормальная составляющая относительного потока
        # даёт сопротивление (F ~ A cos^2), касательная — небольшую подъёмную
        # силу, из-за которой свободный край планирует, а не просто сносится.
        # Время живёт в массиве на устройстве, а не в аргументе ядра: подшаг
        # захватывается в CUDA-граф один раз и переигрывается тысячи раз, а у
        # графа аргументы запусков зафиксированы навсегда.
        tid = wp.tid()
        t = time[0]
        i0 = faces[tid, 0]
        i1 = faces[tid, 1]
        i2 = faces[tid, 2]
        p0 = x[i0]
        p1 = x[i1]
        p2 = x[i2]
        cr = wp.cross(p1 - p0, p2 - p0)
        area2 = wp.length(cr)
        if area2 < 1.0e-12:
            return
        n = cr / area2
        area = 0.5 * area2

        vel = (v[i0] + v[i1] + v[i2]) / 3.0
        c = (p0 + p1 + p2) / 3.0
        rel = vel - wind_at(c, wind_dir, speed, gust, turb, t)
        vn = wp.dot(rel, n)
        vt = rel - n * vn

        f = n * (-drag * area * vn * wp.abs(vn)) - vt * (lift * area * wp.length(rel))
        f = f / 3.0
        wp.atomic_add(force, i0, f)
        wp.atomic_add(force, i1, f)
        wp.atomic_add(force, i2, f)

    @wp.kernel
    def k_integrate(x: wp.array(dtype=vec3),
                    v: wp.array(dtype=vec3),
                    x_prev: wp.array(dtype=vec3),
                    inv_mass: wp.array(dtype=float),
                    force: wp.array(dtype=vec3),
                    gravity: vec3, mass: float, damping: float, dt: float):
        tid = wp.tid()
        x_prev[tid] = x[tid]
        w = inv_mass[tid]
        if w == 0.0:
            v[tid] = vec3(0.0, 0.0, 0.0)
            return
        vel = v[tid] + (gravity + force[tid] * (w / wp.max(mass, 1.0e-9))) * dt
        vel = vel * wp.max(0.0, 1.0 - damping * dt)
        v[tid] = vel
        x[tid] = x[tid] + vel * dt

    @wp.kernel
    def k_stretch(x: wp.array(dtype=vec3),
                  inv_mass: wp.array(dtype=float),
                  ia: wp.array(dtype=wp.int32),
                  ib: wp.array(dtype=wp.int32),
                  rest: wp.array(dtype=float),
                  alpha: wp.array(dtype=float),
                  lam: wp.array(dtype=float),
                  offset: int, dt: float):
        c = wp.tid() + offset
        a = ia[c]
        b = ib[c]
        wa = inv_mass[a]
        wb = inv_mass[b]
        wsum = wa + wb
        if wsum <= 0.0:
            return
        d = x[b] - x[a]
        length = wp.length(d)
        if length < 1.0e-12:
            return
        n = d / length
        # XPBD: alpha~ = alpha/dt^2 — податливость, не зависящая от расписания
        # подшагов, поэтому «мягкость» ткани одна и та же при любом бюджете.
        at = alpha[c] / (dt * dt)
        dl = -(length - rest[c] + at * lam[c]) / (wsum + at)
        lam[c] = lam[c] + dl
        corr = n * dl
        x[a] = x[a] - corr * wa
        x[b] = x[b] + corr * wb

    @wp.kernel
    def k_bend(x: wp.array(dtype=vec3),
               inv_mass: wp.array(dtype=float),
               i0: wp.array(dtype=wp.int32),
               i1: wp.array(dtype=wp.int32),
               i2: wp.array(dtype=wp.int32),
               i3: wp.array(dtype=wp.int32),
               phi0: wp.array(dtype=float),
               alpha: wp.array(dtype=float),
               lam: wp.array(dtype=float),
               offset: int, dt: float):
        # Двугранный угол по Bridson/Müller: C = acos(n1.n2) - phi0.
        # Штрафуется ровно изгиб — растяжение и сдвиг конструкция не трогает,
        # поэтому мягкая на изгиб ткань остаётся нерастяжимой.
        c = wp.tid() + offset
        a0 = i0[c]
        a1 = i1[c]
        a2 = i2[c]
        a3 = i3[c]
        w0 = inv_mass[a0]
        w1 = inv_mass[a1]
        w2 = inv_mass[a2]
        w3 = inv_mass[a3]
        if w0 + w1 + w2 + w3 <= 0.0:
            return

        p0 = x[a0]
        p1 = x[a1] - p0
        p2 = x[a2] - p0
        p3 = x[a3] - p0

        c1 = wp.cross(p1, p2)
        c2 = wp.cross(p1, p3)
        l1 = wp.length(c1)
        l2 = wp.length(c2)
        if l1 < 1.0e-10 or l2 < 1.0e-10:
            return
        n1 = c1 / l1
        n2 = c2 / l2

        d = wp.clamp(wp.dot(n1, n2), -1.0, 1.0)
        s = wp.sqrt(1.0 - d * d)
        if s < 1.0e-6:                       # плоско — градиент вырожден
            return

        q2 = (wp.cross(p1, n2) + wp.cross(n1, p1) * d) / l1
        q3 = (wp.cross(p1, n1) + wp.cross(n2, p1) * d) / l2
        q1 = -(wp.cross(p2, n2) + wp.cross(n1, p2) * d) / l1 \
             - (wp.cross(p3, n1) + wp.cross(n2, p3) * d) / l2
        q0 = -q1 - q2 - q3

        denom = (w0 * wp.dot(q0, q0) + w1 * wp.dot(q1, q1)
                 + w2 * wp.dot(q2, q2) + w3 * wp.dot(q3, q3))
        if denom < 1.0e-12:
            return

        # Градиент C по вершине равен -q_i/s, поэтому в знаменателе стоит
        # denom/s^2. Домножение числителя и знаменателя на s^2 убирает деление
        # на s: иначе у почти плоской ткани (s -> 0) поправка улетает в
        # бесконечность — это давало рваные растянутые треугольники у сгиба.
        cval = wp.acos(d) - phi0[c]
        at = alpha[c] / (dt * dt)
        denom_x = denom + at * s * s
        if denom_x < 1.0e-12:
            return
        k = -(cval + at * lam[c]) * s / denom_x
        lam[c] = lam[c] + k * s

        x[a0] = p0 + q0 * (k * w0)
        x[a1] = x[a1] + q1 * (k * w1)
        x[a2] = x[a2] + q2 * (k * w2)
        x[a3] = x[a3] + q3 * (k * w3)

    @wp.kernel
    def k_mesh_collide(x: wp.array(dtype=vec3),
                       x_prev: wp.array(dtype=vec3),
                       inv_mass: wp.array(dtype=float),
                       mesh: wp.uint64,
                       thickness: float, friction: float):
        # Расстояние НЕЗНАКОВОЕ, сторона барьера берётся из ПРЕДЫДУЩЕГО
        # положения. Кузов — открытый ящик с тонкими стенками: знаковое поле
        # там не определено, а по текущему положению барьер «согласился» бы,
        # что частица уже внутри стенки, и удерживал бы её там.
        tid = wp.tid()
        if inv_mass[tid] == 0.0:
            return
        p = x[tid]
        start = x_prev[tid]

        # 1) Непрерывная проверка: не пересёк ли ОТРЕЗОК за подшаг стенку.
        # Борт тоньше, чем путь частицы за подшаг, поэтому запрос ближайшей
        # точки принципиально ненадёжен: у пролетевшей насквозь частицы
        # ближайшей оказывается ДАЛЬНЯЯ грань, и барьер по ней доталкивает её
        # наружу с другой стороны вместо возврата. Луч ловит это точно.
        move = p - start
        dist = wp.length(move)
        if dist > 1.0e-9:
            d = move / dist
            ray = wp.mesh_query_ray(mesh, start, d, dist + thickness)
            if ray.result:
                rn = ray.normal
                if wp.dot(rn, d) > 0.0:      # нормаль навстречу движению
                    rn = -rn
                p = start + d * ray.t + rn * thickness

        # 2) Барьер по ближайшей точке — он держит лежащую на борту ткань.
        query = wp.mesh_query_point_no_sign(mesh, p, thickness * 4.0)
        if not query.result:
            x[tid] = p
            return
        cp = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        n = wp.mesh_eval_face_normal(mesh, query.face)

        # С какой стороны грани частица была в начале подшага — с той и
        # остаётся. Тонкая стенка так работает как двусторонняя преграда.
        if wp.dot(start - cp, n) < 0.0:
            n = -n
        s = wp.dot(p - cp, n)
        if s >= thickness:
            x[tid] = p
            return

        p = p + n * (thickness - s)
        # Трение: гасим касательное смещение за подшаг. Без него ткань,
        # легшая на борт, бесконечно сползает.
        slide = p - start
        slide = slide - n * wp.dot(slide, n)
        x[tid] = p - slide * friction

    @wp.kernel
    def k_ground(x: wp.array(dtype=vec3),
                 x_prev: wp.array(dtype=vec3),
                 inv_mass: wp.array(dtype=float),
                 ground_z: float, friction: float):
        tid = wp.tid()
        if inv_mass[tid] == 0.0:
            return
        p = x[tid]
        if p[2] >= ground_z:
            return
        pr = x_prev[tid]
        x[tid] = vec3(p[0] - (p[0] - pr[0]) * friction,
                      p[1] - (p[1] - pr[1]) * friction,
                      ground_z)

    @wp.kernel
    def k_self_collide(x: wp.array(dtype=vec3),
                       inv_mass: wp.array(dtype=float),
                       grid: wp.uint64,
                       delta: wp.array(dtype=vec3),
                       count: wp.array(dtype=float),
                       nx: int, radius: float):
        # Разведение частиц, сошедшихся ближе radius. Соседи по сетке
        # пропускаются: их расстояние держат связи, и барьер только дрался бы
        # с ними, раздувая полотно.
        # Схема якобиева (накопление в delta с последующим усреднением):
        # у пар нет раскраски, и запись напрямую была бы гонкой.
        tid = wp.tid()
        if inv_mass[tid] == 0.0:
            return
        p = x[tid]
        row = tid / nx
        col = tid - row * nx

        neighbors = wp.hash_grid_query(grid, p, radius)
        j = int(0)
        while wp.hash_grid_query_next(neighbors, j):
            if j == tid:
                continue
            jr = j / nx
            jc = j - jr * nx
            dr = jr - row
            dc = jc - col
            if dr >= -1 and dr <= 1 and dc >= -1 and dc <= 1:
                continue
            d = p - x[j]
            dist = wp.length(d)
            if dist >= radius or dist < 1.0e-9:
                continue
            n = d / dist
            wsum = inv_mass[tid] + inv_mass[j]
            if wsum <= 0.0:
                continue
            corr = n * ((radius - dist) * inv_mass[tid] / wsum)
            wp.atomic_add(delta, tid, corr)
            wp.atomic_add(count, tid, 1.0)

    @wp.kernel
    def k_apply_delta(x: wp.array(dtype=vec3),
                      delta: wp.array(dtype=vec3),
                      count: wp.array(dtype=float)):
        tid = wp.tid()
        c = count[tid]
        if c > 0.0:
            x[tid] = x[tid] + delta[tid] / c

    @wp.kernel
    def k_update_velocity(x: wp.array(dtype=vec3),
                          x_prev: wp.array(dtype=vec3),
                          v: wp.array(dtype=vec3),
                          inv_mass: wp.array(dtype=float),
                          inv_dt: float, max_speed: float):
        # Скорость из ФАКТИЧЕСКОГО смещения: так трение и выталкивание из
        # коллизий гасят движение сами, без отдельной модели импульсов.
        tid = wp.tid()
        if inv_mass[tid] == 0.0:
            v[tid] = vec3(0.0, 0.0, 0.0)
            return
        vel = (x[tid] - x_prev[tid]) * inv_dt
        sp = wp.length(vel)
        if sp > max_speed:
            vel = vel * (max_speed / sp)
        v[tid] = vel

    @wp.kernel
    def k_normals(x: wp.array(dtype=vec3),
                  faces: wp.array2d(dtype=wp.int32),
                  out: wp.array(dtype=vec3)):
        tid = wp.tid()
        i0 = faces[tid, 0]
        i1 = faces[tid, 1]
        i2 = faces[tid, 2]
        # Нескормированное векторное произведение = нормаль, взвешенная по
        # площади: даёт гладкие вершинные нормали без отдельных весов.
        n = wp.cross(x[i1] - x[i0], x[i2] - x[i0])
        wp.atomic_add(out, i0, n)
        wp.atomic_add(out, i1, n)
        wp.atomic_add(out, i2, n)

    @wp.kernel
    def k_advance_time(time: wp.array(dtype=float), dt: float):
        time[0] = time[0] + dt

    @wp.kernel
    def k_normalize(n: wp.array(dtype=vec3)):
        tid = wp.tid()
        l = wp.length(n[tid])
        if l > 1.0e-12:
            n[tid] = n[tid] / l
        else:
            n[tid] = vec3(0.0, 0.0, 1.0)

    _KERNELS.update(
        clear=k_clear, clear_f=k_clear_f, aero=k_aero, integrate=k_integrate,
        stretch=k_stretch, bend=k_bend, mesh_collide=k_mesh_collide,
        ground=k_ground, self_collide=k_self_collide,
        apply_delta=k_apply_delta, update_velocity=k_update_velocity,
        advance_time=k_advance_time, normals=k_normals,
        normalize=k_normalize)
    return _KERNELS


# ---------------------------------------------------------------------------
# Коллайдер: BVH по настоящим треугольникам
# ---------------------------------------------------------------------------
class WarpMeshCollider:
    """Статическая геометрия как wp.Mesh (BVH).

    В отличие от посева поверхности точками, здесь нет ни бюджета точек, ни
    шага решётки, ни просветов между ними: запрос возвращает точное расстояние
    до треугольника. Это и точнее, и дешевле — BVH обходится за log(T).
    """

    def __init__(self):
        self._verts = []
        self._faces = []
        self._offset = 0
        self.mesh = None

    def add_mesh(self, verts, faces, region=None):
        if verts is None or faces is None or len(faces) == 0:
            return
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)

        if region is not None:
            rmin, rmax = region
            tri = verts[faces]
            near = np.all((tri.max(axis=1) >= rmin) & (tri.min(axis=1) <= rmax),
                          axis=1)
            faces = faces[near]
            if len(faces) == 0:
                return
            # Оставляем только используемые вершины: BVH строится по всем
            # вершинам массива, а кузов целиком в разы больше рабочей зоны.
            used, faces = np.unique(faces, return_inverse=True)
            faces = faces.reshape(-1, 3)
            verts = verts[used]

        self._verts.append(verts)
        self._faces.append(faces + self._offset)
        self._offset += len(verts)

    def build(self):
        """Собрать BVH. False, если добавлять было нечего.

        Списки не сбрасываются: add_mesh после build (запасные ящики) должен
        досыпать геометрию к уже накопленной, а не потерять её.
        """
        if not self._verts:
            return self.mesh is not None
        wp, dev = _wp_mod()
        verts = np.concatenate(self._verts).astype(np.float32)
        faces = np.concatenate(self._faces).astype(np.int32).reshape(-1)
        self.mesh = wp.Mesh(
            points=wp.array(verts, dtype=wp.vec3, device=dev),
            indices=wp.array(faces, dtype=wp.int32, device=dev))
        return True


# ---------------------------------------------------------------------------
# Солвер
# ---------------------------------------------------------------------------
class WarpClothSolver:
    """XPBD-ткань на GPU. API совместим с numpy-солвером ClothSolver."""

    # Подшагов на кадр. XPBD «small steps»: качество растёт от дробления шага
    # заметно быстрее, чем от итераций внутри шага, при той же цене.
    SUBSTEPS = 8

    # Потолок скорости в ячейках сетки за подшаг. Частица, пролетевшая за
    # подшаг больше ячейки, проскакивает сквозь барьер коллизии; ограничение
    # скорости надёжнее и дешевле, чем непрерывная коллизия.
    MAX_CELLS_PER_STEP = 0.5

    def __init__(self, positions, pinned, *,
                 rest_positions=None,
                 gravity=(0.0, 0.0, -9.81),
                 stiff_structural=1.0,
                 stiff_shear=0.7,
                 stiff_bend=0.25,
                 damping=0.02,
                 iterations=2,
                 cell=None):
        wp, dev = _wp_mod()
        self.wp = wp
        self.device = dev
        self.k = _build_kernels()

        self.ny, self.nx = positions.shape[:2]
        n = self.nx * self.ny
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        rest = (pos if rest_positions is None
                else np.asarray(rest_positions, dtype=np.float64).reshape(-1, 3))

        self.iterations = max(1, int(iterations))
        self.damping = float(damping) * 60.0     # прежние единицы (за кадр)
        self.gravity = np.asarray(gravity, dtype=np.float64)

        inv_mass = np.where(np.asarray(pinned).reshape(-1), 0.0, 1.0)

        faces = grid_faces(self.ny, self.nx)
        self.faces_np = faces

        # Характерный размер ячейки — единица длины для податливости и для
        # барьеров. Без него все константы зависели бы от масштаба сцены.
        if cell is None:
            cell = float(np.median(np.linalg.norm(
                rest[faces[:, 1]] - rest[faces[:, 0]], axis=1)))
        self.cell = max(float(cell), _EPS)

        self._build_stretch(rest, stiff_structural, stiff_shear, n)
        self._build_bend(rest, faces, stiff_bend, n)

        f = lambda a, d: wp.array(np.ascontiguousarray(a), dtype=d, device=dev)
        self.x = f(pos.astype(np.float32), wp.vec3)
        self.x_prev = f(pos.astype(np.float32), wp.vec3)
        self.v = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.force = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.delta = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.count = wp.zeros(n, dtype=float, device=dev)
        self.inv_mass = f(inv_mass.astype(np.float32), float)
        self.faces = f(faces.astype(np.int32), wp.int32)
        self.normal_buf = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.n_particles = n

        # Препятствия и среда.
        self.collider = None
        self.thickness = 0.0
        self.ground_z = None
        self.friction = 0.35

        # Самопересечение. Радиус чуть меньше половины ячейки: больше — и
        # барьер начинает драться со связями, раздувая полотно.
        self.self_collision = True
        self.self_radius = self.cell * 0.45
        self._grid = wp.HashGrid(32, 32, 32, device=dev)

        self.wind_dir = np.array([0.0, 1.0, 0.0])
        self.wind_speed = 0.0
        self.wind_gust = 0.0
        self.wind_turbulence = 0.0
        self.drag = 1.4
        self.lift = 0.2
        self.time = wp.zeros(1, dtype=float, device=dev)

        self._graph = None
        self._graph_key = None

    # -- связи --------------------------------------------------------------
    def _color(self, idx, n, kind):
        """Раскраска связей: аналитическая по сетке, с откатом на общую."""
        def build():
            got = color_grid_constraints(idx, self.nx, self.ny)
            return got if got is not None else color_constraints(idx, n)

        return _cached((kind, self.ny, self.nx), build)

    def _compliance(self, stiffness):
        """stiffness (0..1] -> податливость XPBD в единицах сцены.

        Жёсткость 1.0 — нерастяжимая связь (alpha = 0). Дальше alpha растёт
        как (1/k - 1): вдвое меньшая жёсткость даёт вдвое большее удлинение
        под той же нагрузкой. Масштаб cell^2 делает поведение независимым от
        единиц сцены и от плотности сетки.
        """
        k = float(np.clip(stiffness, 1e-3, 1.0))
        return (1.0 / k - 1.0) * (self.cell * self.cell) * 1e-3

    def _build_stretch(self, rest, k_struct, k_shear, n):
        """structural (соседи по сетке) + shear (диагонали ячейки).

        Связи «через одну» здесь НЕ нужны: изгиб держит отдельная двугранная
        конструкция, а растяжка через одну только дублировала бы structural.
        """
        ny, nx = self.ny, self.nx
        ii, jj = np.mgrid[0:ny, 0:nx]
        idx = lambda i, j: (i * nx + j).reshape(-1)

        pairs = []
        stiff = []
        if nx > 1:
            i0, j0 = ii[:, :-1], jj[:, :-1]
            pairs.append(np.stack([idx(i0, j0), idx(i0, j0 + 1)], axis=1))
            stiff.append(np.full(pairs[-1].shape[0], k_struct))
        if ny > 1:
            i0, j0 = ii[:-1, :], jj[:-1, :]
            pairs.append(np.stack([idx(i0, j0), idx(i0 + 1, j0)], axis=1))
            stiff.append(np.full(pairs[-1].shape[0], k_struct))
        if nx > 1 and ny > 1:
            i0, j0 = ii[:-1, :-1], jj[:-1, :-1]
            pairs.append(np.stack([idx(i0, j0), idx(i0 + 1, j0 + 1)], axis=1))
            stiff.append(np.full(pairs[-1].shape[0], k_shear))
            pairs.append(np.stack([idx(i0, j0 + 1), idx(i0 + 1, j0)], axis=1))
            stiff.append(np.full(pairs[-1].shape[0], k_shear))

        if not pairs:
            self.stretch_ranges = []
            return

        pairs = np.concatenate(pairs)
        stiff = np.concatenate(stiff)
        perm, ranges = self._color(pairs, n, "stretch")
        pairs, stiff = pairs[perm], stiff[perm]

        wp, dev = self.wp, self.device
        rest_len = np.linalg.norm(rest[pairs[:, 1]] - rest[pairs[:, 0]], axis=1)
        alpha = np.array([self._compliance(s) for s in stiff])

        self.s_a = wp.array(pairs[:, 0].astype(np.int32), dtype=wp.int32, device=dev)
        self.s_b = wp.array(pairs[:, 1].astype(np.int32), dtype=wp.int32, device=dev)
        self.s_rest = wp.array(rest_len.astype(np.float32), dtype=float, device=dev)
        self.s_alpha = wp.array(alpha.astype(np.float32), dtype=float, device=dev)
        self.s_lambda = wp.zeros(len(pairs), dtype=float, device=dev)
        self.stretch_ranges = ranges

    def _build_bend(self, rest, faces, k_bend, n):
        """Двугранный изгиб по парам смежных треугольников.

        Угол покоя берётся из НЕДЕФОРМИРОВАННОГО полотна (оно плоское, значит
        phi0 = pi): ткань стремится распрямиться, а не запомнить стартовую
        раскладку с уже заложенными складками.
        """
        quads = _cached(("quads", self.ny, self.nx),
                        lambda: dihedral_pairs(faces))
        if len(quads) == 0:
            self.bend_ranges = []
            return

        perm, ranges = self._color(quads, n, "bend")
        quads = quads[perm]

        p0 = rest[quads[:, 0]]
        p1 = rest[quads[:, 1]] - p0
        p2 = rest[quads[:, 2]] - p0
        p3 = rest[quads[:, 3]] - p0
        c1 = np.cross(p1, p2)
        c2 = np.cross(p1, p3)
        l1 = np.linalg.norm(c1, axis=1)[:, None]
        l2 = np.linalg.norm(c2, axis=1)[:, None]
        d = np.sum((c1 / np.maximum(l1, _EPS)) * (c2 / np.maximum(l2, _EPS)),
                   axis=1)
        phi0 = np.arccos(np.clip(d, -1.0, 1.0))

        wp, dev = self.wp, self.device
        arr = lambda a: wp.array(a.astype(np.int32), dtype=wp.int32, device=dev)
        self.b_i0, self.b_i1 = arr(quads[:, 0]), arr(quads[:, 1])
        self.b_i2, self.b_i3 = arr(quads[:, 2]), arr(quads[:, 3])
        self.b_phi0 = wp.array(phi0.astype(np.float32), dtype=float, device=dev)
        # Изгиб мягче растяжения на порядки — это и есть ткань, а не резина.
        self.b_alpha = wp.array(
            np.full(len(quads), self._compliance(k_bend) * 1e3 + 1e-6,
                    dtype=np.float32), dtype=float, device=dev)
        self.b_lambda = wp.zeros(len(quads), dtype=float, device=dev)
        self.bend_ranges = ranges

    # -- окружение ----------------------------------------------------------
    def set_mesh_collider(self, collider, thickness):
        self.collider = collider
        self.thickness = float(thickness)

    def set_ground(self, z):
        self.ground_z = float(z)

    def add_box(self, bmin, bmax, margin=0.0):
        """Запасная преграда-параллелепипед (когда меш прочитать не удалось).

        Отдельного пути для ящиков нет: он превращается в те же 12
        треугольников и уходит в общий BVH.
        """
        bmin = np.asarray(bmin, dtype=np.float64) - margin
        bmax = np.asarray(bmax, dtype=np.float64) + margin
        if not np.all(bmax > bmin):
            return
        corners = np.array([[bmin[0], bmin[1], bmin[2]],
                            [bmax[0], bmin[1], bmin[2]],
                            [bmax[0], bmax[1], bmin[2]],
                            [bmin[0], bmax[1], bmin[2]],
                            [bmin[0], bmin[1], bmax[2]],
                            [bmax[0], bmin[1], bmax[2]],
                            [bmax[0], bmax[1], bmax[2]],
                            [bmin[0], bmax[1], bmax[2]]])
        quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
                 (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
        faces = np.array([t for a, b, c, d in quads
                          for t in ((a, b, c), (a, c, d))], dtype=np.int64)

        if self.collider is None:
            self.collider = WarpMeshCollider()
            self.thickness = self.cell * 0.5
        self.collider.add_mesh(corners, faces)
        self.collider.build()

    def set_wind(self, direction, speed, gust=0.0, turbulence=0.0, drag=1.4,
                 lift=0.2):
        d = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(d)
        self.wind_dir = d / norm if norm > _EPS else np.array([0.0, 1.0, 0.0])
        self.wind_speed = float(speed)
        self.wind_gust = float(gust)
        self.wind_turbulence = float(turbulence)
        self.drag = float(drag)
        self.lift = float(lift)

    # -- прогон -------------------------------------------------------------
    def _substep(self, sub_dt, max_speed):
        """Один подшаг XPBD. Ровно эта последовательность идёт в CUDA-граф,
        поэтому здесь не должно быть ни ветвлений по данным, ни обращений к
        GPU за результатом."""
        wp = self.wp
        k = self.k
        n = self.n_particles
        gravity = wp.vec3(*[float(g) for g in self.gravity])
        wind_dir = wp.vec3(*[float(d) for d in self.wind_dir])

        wp.launch(k["clear"], dim=n, inputs=[self.force])
        if self.wind_speed > 0.0:
            wp.launch(k["aero"], dim=len(self.faces_np),
                      inputs=[self.x, self.v, self.faces, self.force,
                              wind_dir, self.wind_speed, self.wind_gust,
                              self.wind_turbulence, self.drag, self.lift,
                              self.time])
        wp.launch(k["integrate"], dim=n,
                  inputs=[self.x, self.v, self.x_prev, self.inv_mass,
                          self.force, gravity, 1.0, self.damping, sub_dt])

        # XPBD: множители обнуляются на КАЖДОМ подшаге — они накапливают
        # импульс связи в пределах одного шага, не дольше.
        if self.stretch_ranges:
            wp.launch(k["clear_f"], dim=len(self.s_lambda),
                      inputs=[self.s_lambda])
        if self.bend_ranges:
            wp.launch(k["clear_f"], dim=len(self.b_lambda),
                      inputs=[self.b_lambda])

        # Изгиб решается ПЕРЕД растяжением: последнее слово должно оставаться
        # за нерастяжимостью. Иначе изгиб сдвигает вершины уже после того, как
        # длины рёбер выправлены, и остаточное растяжение копится у жёстких
        # границ — у ряда крепления оно доходило до 12% вместо доли процента.
        for _ in range(self.iterations):
            for start, cnt in self.bend_ranges:
                wp.launch(k["bend"], dim=cnt,
                          inputs=[self.x, self.inv_mass, self.b_i0, self.b_i1,
                                  self.b_i2, self.b_i3, self.b_phi0,
                                  self.b_alpha, self.b_lambda, start, sub_dt])
            for start, cnt in self.stretch_ranges:
                wp.launch(k["stretch"], dim=cnt,
                          inputs=[self.x, self.inv_mass, self.s_a, self.s_b,
                                  self.s_rest, self.s_alpha, self.s_lambda,
                                  start, sub_dt])

        # Коллизии — ПОСЛЕ связей: иначе связи затянут ткань обратно в борта.
        if self.collider is not None and self.collider.mesh is not None \
                and self.thickness > 0.0:
            wp.launch(k["mesh_collide"], dim=n,
                      inputs=[self.x, self.x_prev, self.inv_mass,
                              self.collider.mesh.id, self.thickness,
                              self.friction])
        if self.self_collision and self.self_radius > 0.0:
            self._grid.build(self.x, self.self_radius)
            wp.launch(k["clear"], dim=n, inputs=[self.delta])
            wp.launch(k["clear_f"], dim=n, inputs=[self.count])
            wp.launch(k["self_collide"], dim=n,
                      inputs=[self.x, self.inv_mass, self._grid.id, self.delta,
                              self.count, self.nx, self.self_radius])
            wp.launch(k["apply_delta"], dim=n,
                      inputs=[self.x, self.delta, self.count])
        if self.ground_z is not None:
            wp.launch(k["ground"], dim=n,
                      inputs=[self.x, self.x_prev, self.inv_mass,
                              self.ground_z, self.friction])

        wp.launch(k["update_velocity"], dim=n,
                  inputs=[self.x, self.x_prev, self.v, self.inv_mass,
                          1.0 / sub_dt, max_speed])
        wp.launch(k["advance_time"], dim=1, inputs=[self.time, sub_dt])

    def settle(self, steps, dt=1.0 / 120.0, substeps=None):
        wp = self.wp
        substeps = int(substeps or self.SUBSTEPS)
        sub_dt = float(dt) / max(1, substeps)
        max_speed = self.MAX_CELLS_PER_STEP * self.cell / max(sub_dt, _EPS)
        total = int(steps) * substeps

        with wp.ScopedDevice(self.device):
            # Подшаг — это под сотню запусков ядер, а подшагов за сэмпл
            # тысячи. Накладные расходы на запуск (десятки микросекунд каждый)
            # стоили бы больше самой физики, поэтому подшаг захватывается в
            # CUDA-граф один раз и дальше только переигрывается.
            graph = self._graph_for(sub_dt, max_speed)
            if graph is not None:
                for _ in range(total):
                    wp.capture_launch(graph)
            else:
                for _ in range(total):
                    self._substep(sub_dt, max_speed)
            wp.synchronize()

        return self.pos.reshape(self.ny, self.nx, 3)

    def _graph_for(self, sub_dt, max_speed):
        """CUDA-граф подшага (или None, если захват не удался).

        Захват может не пройти на старом драйвере или из-за временных
        аллокаций внутри hash grid — тогда просто идут обычные запуски, и
        результат тот же, только медленнее.
        """
        key = (round(sub_dt, 12), round(max_speed, 6))
        if self._graph_key == key:
            return self._graph
        self._graph_key = key
        self._graph = None
        try:
            wp = self.wp
            # Прогреваем: первый запуск ядра компилирует модуль и выделяет
            # временную память, а во время захвата ни того ни другого нельзя.
            self._substep(sub_dt, max_speed)
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                self._substep(sub_dt, max_speed)
            self._graph = capture.graph
        except Exception as exc:              # noqa: BLE001
            print(f"[Cloth] CUDA-граф недоступен ({exc}) — обычные запуски")
            self._graph = None
        return self._graph

    # -- результат ----------------------------------------------------------
    @property
    def pos(self):
        return self.x.numpy().astype(np.float64).reshape(-1, 3)

    def normals(self):
        wp = self.wp
        with wp.ScopedDevice(self.device):
            wp.launch(self.k["clear"], dim=self.n_particles,
                      inputs=[self.normal_buf])
            wp.launch(self.k["normals"], dim=len(self.faces_np),
                      inputs=[self.x, self.faces, self.normal_buf])
            wp.launch(self.k["normalize"], dim=self.n_particles,
                      inputs=[self.normal_buf])
        return self.normal_buf.numpy().astype(np.float64).reshape(
            self.ny, self.nx, 3)
