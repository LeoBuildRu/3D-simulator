# -*- coding: utf-8 -*-
"""
Ужатие карт веб-вьюера перед отправкой в реестр.

Генератор кладёт в `<stem>/` JPEG-копии карт 4096 px с качеством 92-95: по
2-4 МБ на карту, ~20 МБ на кузов. Агрегатор качает их с сервера при каждом
открытии модели, и на его канале это 6-9 секунд на одни текстуры. Здесь каждая
карта тяжелее `BUDGET` пережимается так, чтобы уложиться в бюджет с наименьшей
потерей: перебираются размеры (исходный, 1/2, 1/4) и для каждого ищется
наибольшее качество JPEG, которое влезает; из вариантов берётся ближайший к
оригиналу по PSNR.

Что важно не сломать:

* имя и формат файла не меняются — glTF ссылается на карты по имени, а
  `mimeType` в нём уже записан;
* файлы комплекта на диске не трогаются: ужатые копии лежат в кэше
  (`CACHE_DIR`) под хэшем содержимого оригинала, так что повторная отправка
  того же комплекта ничего не пересчитывает, а пересобранный — пересчитывает;
* сюда попадают только карты WEB. PNG из той же папки — текстуры для .bam
  (роль `texture`), их десктопный клиент читает как есть.

Правило то же, каким 2026-10-07 были разово ужаты карты уже лежавших на
сервере моделей (`operation-3d-service/backups/models_web_textures_orig-*`).
"""

from __future__ import annotations

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Optional, Tuple

#: Потолок на одну карту, байт.
BUDGET = 500_000

#: Ниже этого качества JPEG на плоских заливках кузова видны блоки: лучше
#: взять вдвое меньший размер с высоким качеством.
QUALITY_MIN = 55
QUALITY_MAX = 92

#: Меньше этой стороны не уменьшаем, даже если в бюджет не влезло.
MIN_EDGE = 1024

#: Меняется вместе с правилом — чтобы кэш от старого правила не подхватился.
_RULE = "v1"

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "assets", "models", "_registry_cache", "web_textures")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

Say = Callable[[str], None]


def _fmt(size: int) -> str:
    return f"{size / 1e6:.2f} МБ" if size >= 1e6 else f"{size // 1000} КБ"


def _digest(path: str) -> str:
    h = hashlib.sha1(f"{_RULE}:{BUDGET}:".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sizes(width: int, height: int) -> List[Tuple[int, int]]:
    """Исходный размер и его половины, пока большая сторона не меньше MIN_EDGE."""
    out = [(width, height)]
    k = 2
    while max(width, height) // k >= MIN_EDGE:
        out.append((max(1, width // k), max(1, height // k)))
        k *= 2
    return out


def _psnr(a, b) -> float:
    import numpy as np
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    mse = float(np.mean(diff * diff))
    return 99.0 if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def _renormalize(img):
    """Вернуть нормалям единичную длину после усреднения при уменьшении."""
    import numpy as np
    from PIL import Image
    v = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    n = np.linalg.norm(v, axis=2, keepdims=True)
    v = np.where(n > 0.5, v / np.maximum(n, 1e-6), v)
    return Image.fromarray(
        np.clip((v + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8), "RGB")


def _encode_jpeg(img, quality: int, subsampling: int) -> bytes:
    buf = io.BytesIO()
    kw = {"quality": quality, "optimize": True}
    if img.mode != "L":
        kw["subsampling"] = subsampling
    img.save(buf, "JPEG", **kw)
    return buf.getvalue()


def _best_jpeg(img, subsampling: int) -> Optional[Tuple[int, bytes]]:
    """Наибольшее качество, при котором карта влезает в бюджет."""
    lo, hi, found = 20, QUALITY_MAX, None
    while lo <= hi:
        quality = (lo + hi) // 2
        data = _encode_jpeg(img, quality, subsampling)
        if len(data) <= BUDGET:
            found = (quality, data)
            lo = quality + 1
        else:
            hi = quality - 1
    return found


def _shrink_jpeg(img, is_normal: bool) -> Optional[bytes]:
    from PIL import Image
    if img.mode not in ("RGB", "L"):
        return None
    sizes = _sizes(*img.size)
    # В карте нормалей каналы — независимые данные, а не цвет: прореживание
    # цветности 4:2:0 портит их заметнее, чем более низкое качество.
    modes = (2, 0) if (is_normal and img.mode == "RGB") else (2,)
    variants = []
    for size in sizes:
        scaled = img if size == img.size else img.resize(size, Image.LANCZOS)
        if is_normal and img.mode == "RGB" and size != img.size:
            scaled = _renormalize(scaled)
        for subsampling in modes:
            hit = _best_jpeg(scaled, subsampling)
            if not hit:
                continue
            quality, data = hit
            back = Image.open(io.BytesIO(data))
            if back.size != img.size:
                back = back.resize(img.size, Image.BICUBIC)
            variants.append((_psnr(img, back), quality, size, data))
    good = [v for v in variants if v[1] >= QUALITY_MIN] \
        or [v for v in variants if v[2] == sizes[-1]]
    if not good:
        return None
    return max(good, key=lambda v: v[0])[3]


def _shrink_png(img) -> Optional[bytes]:
    """PNG остаётся PNG в том же режиме: только уменьшение размера."""
    from PIL import Image
    if img.mode not in ("P", "RGB", "RGBA", "L", "LA"):
        return None
    src = img.convert("RGBA" if "transparency" in img.info else "RGB") \
        if img.mode == "P" else img
    data = None
    for size in _sizes(*img.size)[1:]:
        scaled = src.resize(size, Image.LANCZOS)
        if img.mode == "P":
            scaled = scaled.quantize(256, method=Image.Quantize.FASTOCTREE
                                     if scaled.mode == "RGBA"
                                     else Image.Quantize.MEDIANCUT,
                                     dither=Image.Dither.NONE)
        buf = io.BytesIO()
        scaled.save(buf, "PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= BUDGET:
            break
    return data


def shrink_texture(path: str) -> Tuple[str, str]:
    """
    Путь к ужатой копии карты (или к ней самой) и пояснение для журнала.

    Никогда не бросает: карта, которую не удалось пережать, отправляется как
    есть — тяжёлая текстура лучше несостоявшейся загрузки.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return path, ""
    ext = os.path.splitext(path)[1].lower()
    if size <= BUDGET or ext not in _IMAGE_EXTS:
        return path, ""

    name = os.path.basename(path)
    try:
        cached = os.path.join(CACHE_DIR, _digest(path) + ext)
        if os.path.isfile(cached) and os.path.getsize(cached) > 0:
            return cached, f"{name}: {_fmt(size)} -> " \
                           f"{_fmt(os.path.getsize(cached))} (из кэша)"

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as opened:
            opened.load()
            fmt, img = opened.format, opened.copy()
        if fmt == "JPEG":
            data = _shrink_jpeg(img, "normal" in name.lower())
        elif fmt == "PNG":
            data = _shrink_png(img)
        else:
            data = None
        if not data or len(data) >= size:
            return path, f"{name}: {_fmt(size)}, ужать не удалось — " \
                         f"отправляется как есть"

        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cached + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, cached)
        with Image.open(cached) as out:
            side = max(out.size)
        note = f"{name}: {_fmt(size)} -> {_fmt(len(data))}, {side} px"
        if len(data) > BUDGET:
            note += f" (в {_fmt(BUDGET)} не уложилась)"
        return cached, note
    except Exception as exc:                              # noqa: BLE001
        return path, f"{name}: не ужата ({exc}) — отправляется как есть"


def shrink_web_files(web_files: Iterable[Any], say: Optional[Say] = None
                     ) -> List[Any]:
    """
    Тот же список WEB-файлов, где тяжёлые карты заменены ужатыми копиями.

    Элементы — как у `ModelRegistry.upsert_model`: путь либо пара
    `(имя на сервере, путь)`. Имя на сервере не меняется никогда.
    """
    items = list(web_files)
    paths = [item[1] if isinstance(item, (tuple, list)) else item
             for item in items]
    heavy = sorted({p for p in paths
                    if os.path.splitext(p)[1].lower() in _IMAGE_EXTS
                    and os.path.isfile(p) and os.path.getsize(p) > BUDGET})
    if not heavy:
        return items
    if say:
        say(f"ужимаем карты веб-вьюера тяжелее {_fmt(BUDGET)}: "
            f"{len(heavy)} шт.…")

    # Pillow отпускает GIL на кодировании и ресайзе, потоков достаточно.
    with ThreadPoolExecutor(max_workers=min(4, len(heavy))) as pool:
        done = dict(zip(heavy, pool.map(shrink_texture, heavy)))

    out: List[Any] = []
    for item, path in zip(items, paths):
        new_path, _note = done.get(path, (path, ""))
        if isinstance(item, (tuple, list)):
            out.append((item[0], new_path))
        else:
            # Без пары имя на сервере взялось бы из basename, а у копии в кэше
            # он — хэш.
            out.append((os.path.basename(path), new_path)
                       if new_path != path else item)
    if say:
        for path in heavy:
            if done[path][1]:
                say("  " + done[path][1])
        before = sum(os.path.getsize(p) for p in heavy)
        after = sum(os.path.getsize(done[p][0]) for p in heavy)
        say(f"карты веб-вьюера: {_fmt(before)} -> {_fmt(after)}")
    return out
