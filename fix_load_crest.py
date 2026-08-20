#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_load_crest.py
=================
Дотягивает уровень груза в AI-кадре до линии груза из seg-карты.

Зачем
-----
fix_ai_offset.py выравнивает кадр по КУЗОВУ (синий класс) — он геометрически
жёсткий и потому надёжный якорь. Но генеративная модель может при этом насыпать
груза меньше или больше, чем в рендере: силуэт кучи уезжает по вертикали на
50-100 px, и полоса вдоль гребня получает метку `cargo` там, где на картинке уже
фон (или наоборот). Кузов при этом стоит идеально — fix_ai_offset такую ошибку
не видит и видеть не должен.

Этот модуль правит именно её: находит, насколько силуэт груза в AI-кадре
разошёлся с гребнем из маски, и локально сдвигает материал по вертикали, чтобы
гребень сел на линию маски.

Как оценивается промах
----------------------
Классификатора «материал/фон» у нас нет, но seg-карта даёт готовую обучающую
выборку прямо в этом же кадре:
  * «материал» — пиксели глубоко внутри cargo (дальше DEEP_IN px от границы,
    чтобы промах гребня заведомо не попал в обучение);
  * «фон» — пиксели далеко (FAR_BG px) от всего, что не фон.
По ним строится линейный дискриминант (общая ковариация) в 5-мерном признаковом
пространстве: L, a, b и локальное СКО яркости на двух масштабах. Получается
карта M = log-odds «здесь материал».

Дальше для каждой колонки гребня (там, где НАД грузом по маске именно фон)
берётся вертикальная полоса и ищется сдвиг d, при котором линия маски, сдвинутая
на d, лучше всего делит полосу на «материал снизу / фон сверху»:
    s_x(d) = Σ M(y > t+d) − Σ M(y < t+d)
Скоры суммируются по скользящему окну соседних колонок (WINDOW), поэтому d(x)
получается гладким без постобработки argmax'ов.

Как применяется
---------------
Вертикальное поле смещений, локализованное у гребня: ровно d(x) на линии маски,
линейный спад до нуля на UP px вверх и DOWN px вниз, горизонтальный конус на
концах гребня и обнуление рядом с кузовом. Один вызов cv2.remap.

Защита (как и в fix_ai_offset — результат не может стать хуже)
-------------------------------------------------------------
  * |d| зажат в MAX_SHIFT;
  * нужен прирост целевой функции над d=0 не меньше MIN_GAIN (на чистых рендерах
    оценка даёт ровно d=0 и прирост 0 — коррекция не срабатывает);
  * нужно не меньше MIN_COLUMNS колонок гребня;
  * после варпа целевая функция пересчитывается: не улучшилась — кадр остаётся
    как был.

Запуск:
  python fix_load_crest.py <папка | кадр_ai_fix.png> [...] [--jobs N] [--dry-run]
Для каждого <кадр> рядом должен лежать <кадр без суффикса>_seg.png.
"""

import glob
import os
import sys

import cv2
import numpy as np

# ─── палитра seg-карт (из json датасета) ───
CARGO_RGB = (253, 2, 2)
BODY_RGB = (40, 85, 243)
BG_RGB = (0, 0, 0)

# ─── оценка ───
DEEP_IN = 150       # px от границы cargo — глубина «заведомо материала»
FAR_BG = 90         # px от не-фона — «заведомо фон»
SEARCH = (-140, 220)  # диапазон поиска d, px (минус — материал выше маски)
BAND_UP = 150       # полоса анализа вверх от линии маски
BAND_DOWN = 260     # и вниз
WINDOW = 81         # окно сглаживания скоров по колонкам
MIN_EVID = 40       # минимум строк по обе стороны границы, чтобы колонка голосовала

# ─── применение ───
UP = 40             # спад смещения вверх от гребня, px
DOWN = 300          # спад смещения вниз в тело кучи, px
TAPER = 60          # горизонтальный конус на концах гребня, px
BODY_KEEPOUT = 12   # не трогать пиксели ближе этого к кузову, px

# ─── гейты ───
MAX_SHIFT = 130     # максимально допустимое |d|, px
MIN_GAIN = 1500.0   # минимальный прирост целевой функции над d=0
MIN_COLUMNS = 60    # минимум колонок гребня

SUFFIX = "_crest"


def _mask(seg, rgb):
    return np.all(seg[:, :, ::-1] == rgb, axis=-1)


def _features(bgr):
    """Lab + локальное СКО яркости на двух масштабах."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    g = lab[:, :, 0]
    f = [lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]]
    for k in (5, 15):
        mu = cv2.blur(g, (k, k))
        f.append(np.sqrt(np.maximum(cv2.blur(g * g, (k, k)) - mu * mu, 0)))
    return np.stack(f, -1)


def _logodds(F, pos, neg):
    """Линейный дискриминант с общей ковариацией -> карта log-odds «материал»."""
    A = F[pos].reshape(-1, F.shape[-1])
    B = F[neg].reshape(-1, F.shape[-1])
    if len(A) < 500 or len(B) < 500:
        return None
    ma, mb = A.mean(0), B.mean(0)
    n = len(ma)
    S = (np.cov(A, rowvar=False) * len(A) + np.cov(B, rowvar=False) * len(B)) / (len(A) + len(B))
    S = S + np.eye(n, dtype=np.float64) * (1e-3 * np.trace(S) / n)
    w = np.linalg.solve(S, ma - mb)
    return (F @ w - 0.5 * (ma + mb) @ w).astype(np.float32)


def _crest(seg):
    """{x: y} — верхняя строка cargo в колонках, где над грузом именно фон."""
    cargo, bg = _mask(seg, CARGO_RGB), _mask(seg, BG_RGB)
    first = np.argmax(cargo, axis=0)
    has = cargo.any(axis=0)
    tops = {}
    for x in np.flatnonzero(has):
        t = int(first[x])
        if t > 0 and bg[t - 1, x]:
            tops[x] = t
    return tops, cargo, bg


def _column_gains(M, tops, lo, hi):
    """g[x][d] = насколько граница на (t+d) делит полосу лучше, чем на самой t.

    Полоса обрезается краями кадра, поэтому у каждой колонки свой измеримый
    диапазон d: чтобы судить о сдвиге, нужно не меньше MIN_EVID строк по обе
    стороны от предполагаемой границы. Вне этого диапазона прирост обнуляется —
    «нет данных» никогда не выигрывает у «не двигать».
    """
    h = M.shape[0]
    ds = np.arange(lo, hi + 1)
    z = -lo  # индекс d = 0
    xs, out = [], []
    for x, t in tops.items():
        y0, y1 = max(0, t - BAND_UP), min(h, t + BAND_DOWN)
        above, below = t - y0, y1 - t
        if below < MIN_EVID * 2:
            continue
        col = M[y0:y1, x].astype(np.float64)
        c = np.concatenate(([0.0], np.cumsum(col)))
        idx = np.clip(ds + above, 0, len(col))
        s = (c[-1] - c[idx]) - c[idx]
        g = s - s[z]
        g[(ds < MIN_EVID - above) | (ds > below - MIN_EVID)] = 0.0
        xs.append(x)
        out.append(g)
    return np.array(xs), (np.array(out) if out else np.zeros((0, len(ds))))


def _argmax_nearest_zero(curve, ds):
    """argmax с разрешением ничьих в пользу наименьшего |d|."""
    best = curve.max()
    cand = np.flatnonzero(curve >= best - 1e-9)
    return int(cand[np.argmin(np.abs(ds[cand]))])


def _objective(M, tops):
    """Насколько хорошо линия маски делит полосу на «материал снизу / фон сверху».

    Считается как суммарный log-odds: Σ M ниже линии − Σ M выше, по тем же
    колонкам гребня. Растёт, когда под линией материал, а над ней фон.
    """
    h = M.shape[0]
    total = 0.0
    for x, t in tops.items():
        y0, y1 = max(0, t - BAND_UP), min(h, t + BAND_DOWN)
        if y1 - t < MIN_EVID * 2:
            continue
        col = M[y0:y1, x].astype(np.float64)
        total += float(col[t - y0:].sum() - col[: t - y0].sum())
    return total


def estimate(ai, seg, ref=None):
    """-> ((d(x), tops, M), инфо) либо (None, причина).

    `ref` — исходный рендер того же кадра. Он выровнен с маской по построению,
    поэтому оценка на нём — чистая систематическая ошибка дискриминатора для
    данной геометрии гребня и фона (например, тёмные фуры прямо над кучей, к
    которым линейный дискриминант неравнодушен). Её вычитаем из оценки на
    AI-кадре. Без калибровки на некоторых сценах оценщик уверенно «правит»
    заведомо правильный кадр.
    """
    tops, cargo, bg = _crest(seg)
    if len(tops) < MIN_COLUMNS:
        return None, f"колонок гребня {len(tops)} < {MIN_COLUMNS}"

    F = _features(ai)
    pos = cv2.distanceTransform(cargo.astype(np.uint8), cv2.DIST_L2, 5) > DEEP_IN
    neg = cv2.distanceTransform(bg.astype(np.uint8), cv2.DIST_L2, 5) > FAR_BG
    M = _logodds(F, pos, neg)
    if M is None:
        return None, "мало обучающих пикселей для дискриминанта"

    lo, hi = SEARCH
    xs, G = _column_gains(M, tops, lo, hi)
    if len(xs) < MIN_COLUMNS:
        return None, f"пригодных колонок {len(xs)} < {MIN_COLUMNS}"

    # Приросты суммируются по окну соседних колонок -> d(x) выходит гладким.
    k = np.ones(min(WINDOW, len(xs)), np.float64)
    Gw = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, G)
    ds = np.arange(lo, hi + 1)
    d_col = np.array([ds[_argmax_nearest_zero(row, ds)] for row in Gw], np.float64)

    # Прирост целевой функции, нормированный на окно (каждая колонка учтена ~WINDOW раз).
    gain = float(Gw.max(axis=1).sum()) / max(1, min(WINDOW, len(xs)))

    if ref is not None:
        sub = estimate(ref, seg)[0]
        if sub is not None:
            bias = sub[0]
            before = float(np.abs(d_col).mean())
            d_col = np.array([d - bias.get(int(x), 0.0) for d, x in zip(d_col, xs)])
            gain *= min(1.0, float(np.abs(d_col).mean()) / max(before, 1e-6))

    d_col = np.clip(d_col, -MAX_SHIFT, MAX_SHIFT)
    if gain < MIN_GAIN:
        return None, f"прирост {gain:.0f} < {MIN_GAIN:.0f} (кадр и так на месте)"

    # Горизонтальный конус на концах гребня: у края смещение гаснет.
    n = len(xs)
    ramp = np.ones(n)
    m = min(TAPER, n // 2)
    if m > 0:
        ramp[:m] = np.linspace(0, 1, m)
        ramp[-m:] = np.linspace(1, 0, m)
    d_col *= ramp

    info = dict(gain=gain, ncol=n, d_med=float(np.median(d_col)),
                d_min=float(d_col.min()), d_max=float(d_col.max()))
    return (dict(zip(xs.tolist(), d_col.tolist())), tops, M), info


def _field(shape, dmap, tops, seg):
    """Вертикальное поле смещений, локализованное у гребня."""
    h, w = shape
    fy = np.zeros((h, w), np.float32)
    yy = np.arange(h, dtype=np.float32)
    for x, d in dmap.items():
        if d == 0.0:
            continue
        t = tops[x]
        u = yy - t
        f = np.where(u < 0, 1.0 + u / UP, 1.0 - u / DOWN)
        fy[:, x] = np.clip(f, 0.0, 1.0) * d

    # Не тянем пиксели кузова: гасим поле рядом с синим классом.
    body = _mask(seg, BODY_RGB).astype(np.uint8)
    if body.any():
        near = cv2.dilate(body, np.ones((3, 3), np.uint8), iterations=BODY_KEEPOUT)
        keep = cv2.GaussianBlur((1 - near).astype(np.float32), (0, 0), 9)
        fy *= keep
    return cv2.GaussianBlur(fy, (0, 0), 5)


def process(path, save=True, verbose=True, dry=False):
    base = path[:-len(".png")]
    stem = base
    for suf in ("_ai_fix", "_ai"):
        if base.endswith(suf):
            stem = base[: -len(suf)]
            break
    seg_path = stem + "_seg.png"
    ref_path = stem + ".png"

    if not os.path.exists(seg_path):
        if verbose:
            print(f"  SKIP (нет seg): {os.path.basename(path)}")
        return None
    ai = cv2.imread(path, cv2.IMREAD_COLOR)
    seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)
    if ai is None or seg is None:
        print(f"  ERROR чтения {os.path.basename(path)}")
        return None
    h, w = seg.shape[:2]
    if ai.shape[:2] != (h, w):
        ai = cv2.resize(ai, (w, h), interpolation=cv2.INTER_AREA)

    ref = None
    if os.path.abspath(ref_path) != os.path.abspath(path) and os.path.exists(ref_path):
        ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref is not None and ref.shape[:2] != (h, w):
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)

    est, info = estimate(ai, seg, ref=ref)
    if est is None:
        if verbose:
            print(f"  no-op {os.path.basename(path)}: {info}")
        return None
    dmap, tops, M = est

    fy = _field((h, w), dmap, tops, seg)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = cv2.remap(ai, xx, yy + fy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

    # Проверка: целевая функция на линии маски должна вырасти.
    before = _objective(M, tops)
    M2 = _logodds(_features(out),
                  cv2.distanceTransform(_mask(seg, CARGO_RGB).astype(np.uint8),
                                        cv2.DIST_L2, 5) > DEEP_IN,
                  cv2.distanceTransform(_mask(seg, BG_RGB).astype(np.uint8),
                                        cv2.DIST_L2, 5) > FAR_BG)
    after = _objective(M2, tops) if M2 is not None else before
    if after <= before:
        if verbose:
            print(f"  no-op {os.path.basename(path)}: варп не улучшил "
                  f"({before:.0f} -> {after:.0f})")
        return None

    if verbose:
        print(f"  OK d[{info['d_min']:+.0f}..{info['d_max']:+.0f}] med={info['d_med']:+.0f} "
              f"gain={info['gain']:.0f} obj {before:.0f} -> {after:.0f}: "
              f"{os.path.basename(base + SUFFIX + '.png')}")
    if dry or not save:
        return None
    out_path = base + SUFFIX + ".png"
    cv2.imwrite(out_path, out)
    return out_path


def main():
    args = [a for a in sys.argv[1:]]
    dry = "--dry-run" in args
    jobs = 1
    rest = []
    it = iter([a for a in args if a != "--dry-run"])
    for a in it:
        if a == "--jobs":
            jobs = int(next(it))
        elif a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    if not rest:
        print("usage: fix_load_crest.py [--jobs N] [--dry-run] <folder | image.png> [...]")
        return
    targets = []
    for a in rest:
        if os.path.isdir(a):
            targets += sorted(glob.glob(os.path.join(a, "*_ai_fix.png")))
        else:
            targets.append(a)
    print(f"Обработка {len(targets)} кадров (jobs={jobs})...")

    ok = 0
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial
        fn = partial(process, verbose=False, dry=dry)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(fn, p): p for p in targets}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    if fut.result() is not None:
                        ok += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR {os.path.basename(futs[fut])}: {e}")
                if i % 25 == 0 or i == len(targets):
                    print(f"  {i}/{len(targets)} ({ok} исправлено)")
    else:
        for p in targets:
            if process(p, dry=dry) is not None:
                ok += 1
    print(f"Готово. Исправлено {ok}/{len(targets)}.")


if __name__ == "__main__":
    main()
