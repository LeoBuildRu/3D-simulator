# depth_preview.py
# ---------------------------------------------------------------------------
# Превью карты глубины на стороне CPU.
#
# DepthMapRenderer держит НЕЛИНЕЙНЫЙ z-буфер в depth_texture и раскрашивает
# его шейдером оверлея. Для превью в UI тот же самый расчёт повторяется на
# numpy: так картинку можно показать с ЛЮБЫМИ параметрами (диапазон, ч/б или
# радуга), не трогая живой оверлей в сцене. Это важно для диалога настроек
# датасета — там пользователь смотрит, что именно уедет в файл, а сцена в
# окне при этом остаётся в своём обычном виде.
#
# Формула линеаризации и нарезка градиента ОДИН-В-ОДИН повторяют
# depth_map_renderer.setup_depth_overlay(): расхождение здесь означало бы,
# что превью врёт относительно сохранённого файла.
# ---------------------------------------------------------------------------

from __future__ import annotations

# Ключевые точки радужного градиента из шейдера оверлея:
# (t_низ, цвет_низ, t_верх, цвет_верх), t = 1 - нормализованная глубина,
# то есть 0 — дальний план, 1 — ближний.
_RAINBOW_STOPS = [
    (0.00, (0.0, 0.0, 0.3), 0.10, (0.0, 0.0, 1.0)),   # тёмно-синий -> синий
    (0.10, (0.0, 0.0, 1.0), 0.30, (0.1, 0.7, 0.4)),   # синий -> изумруд
    (0.30, (0.1, 0.7, 0.4), 0.50, (1.0, 1.0, 0.0)),   # изумруд -> жёлтый
    (0.50, (1.0, 1.0, 0.0), 0.70, (1.0, 0.5, 0.0)),   # жёлтый -> оранжевый
    (0.70, (1.0, 0.5, 0.0), 0.90, (1.0, 0.0, 0.0)),   # оранжевый -> красный
    (0.90, (1.0, 0.0, 0.0), 1.01, (0.5, 0.0, 0.0)),   # красный -> тёмно-красный
]

_LUT_CACHE: dict = {}


def rainbow_lut():
    """256-элементная RGBA-таблица радужного градиента (как в шейдере)."""
    lut = _LUT_CACHE.get("rainbow")
    if lut is not None:
        return lut

    import numpy as np

    out = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for t_lo, c_lo, t_hi, c_hi in _RAINBOW_STOPS:
            if t_lo <= t < t_hi:
                a = (t - t_lo) / (t_hi - t_lo)
                for ch in range(3):
                    v = c_lo[ch] + (c_hi[ch] - c_lo[ch]) * a
                    out[i, ch] = int(np.clip(v * 255.0, 0, 255))
                out[i, 3] = 255
                break
    _LUT_CACHE["rainbow"] = out
    return out


def grayscale_lut():
    """256-элементная RGBA-таблица ч/б карты: ближе — светлее."""
    lut = _LUT_CACHE.get("gray")
    if lut is not None:
        return lut

    import numpy as np

    out = np.zeros((256, 4), dtype=np.uint8)
    ramp = np.arange(256, dtype=np.uint8)
    out[:, 0] = out[:, 1] = out[:, 2] = ramp
    out[:, 3] = 255
    _LUT_CACHE["gray"] = out
    return out


def lut_for(grayscale: bool):
    return grayscale_lut() if grayscale else rainbow_lut()


def gradient_strip_qimage(width: int, height: int, grayscale: bool):
    """Горизонтальная полоска-легенда: слева дальний край, справа ближний."""
    try:
        import numpy as np
        from PyQt6.QtGui import QImage
    except Exception:
        return None

    width = max(1, int(width))
    height = max(1, int(height))
    lut = lut_for(bool(grayscale))
    idx = (np.linspace(0.0, 1.0, width) * 255.0).astype(np.uint8)
    row = lut[idx]                                  # (width, 4)
    rgba = np.ascontiguousarray(np.repeat(row[None, :, :], height, axis=0))
    img = QImage(rgba.tobytes(), width, height, width * 4,
                 QImage.Format.Format_RGBA8888)
    return img.copy()


def depth_to_qimage(depth_renderer, out_w: int, out_h: int, *,
                    near: float | None = None,
                    far: float | None = None,
                    grad_start: float | None = None,
                    grad_end: float | None = None,
                    grayscale: bool = False):
    """Снять текущую карту глубины и раскрасить её в QImage.

    Параметры, оставленные None, берутся из самого `depth_renderer`, поэтому
    вызов без аргументов даёт ровно то, что видно в живом оверлее.
    Возвращает None, если текстура ещё не заполнена (первые кадры после
    старта) — вызывающий просто пропускает тик.
    """
    if depth_renderer is None:
        return None
    tex = getattr(depth_renderer, "depth_texture", None)
    if tex is None:
        return None

    try:
        import numpy as np
        from PyQt6.QtGui import QImage

        if not tex.has_ram_image():
            return None
        ram = tex.get_ram_image_as("D")
        if ram is None:
            return None
        buf = memoryview(ram).tobytes()
        if not buf:
            return None
        tw = tex.get_x_size()
        th = tex.get_y_size()
        if tw * th * 4 != len(buf):
            return None

        depth = np.frombuffer(buf, dtype=np.float32).reshape(th, tw)

        near = float(getattr(depth_renderer, "min_depth", 0.1)
                     if near is None else near)
        far = float(getattr(depth_renderer, "max_depth", 100.0)
                    if far is None else far)
        gs = float(getattr(depth_renderer, "gradient_start", 0.2)
                   if grad_start is None else grad_start)
        ge = float(getattr(depth_renderer, "gradient_end", 0.4)
                   if grad_end is None else grad_end)

        # Линеаризация нелинейного z-буфера — та же формула, что в шейдере.
        den = (far + near) - depth * (far - near)
        den = np.where(np.abs(den) < 1e-6, 1e-6, den)
        linear = (2.0 * near) / den

        if abs(ge - gs) < 1e-6:
            ge = gs + 1.0
        n = np.clip((linear - gs) / (ge - gs), 0.0, 1.0)
        t = 1.0 - n                     # ближе — «горячее» / светлее

        # Прореживание до размера превью (дёшево и достаточно чётко).
        out_w = max(1, int(out_w))
        out_h = max(1, int(out_h))
        sx = max(1, tw // out_w)
        sy = max(1, th // out_h)
        t_small = t[::sy, ::sx][:out_h, :out_w]
        sh, sw = t_small.shape
        if sh == 0 or sw == 0:
            return None

        lut = lut_for(bool(grayscale))
        idx = (np.clip(t_small, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgba = np.ascontiguousarray(lut[idx])

        img = QImage(rgba.tobytes(), sw, sh, sw * 4,
                     QImage.Format.Format_RGBA8888)
        # Panda пишет текстуры снизу вверх.
        return img.mirrored(False, True).copy()
    except Exception as exc:
        print(f"[DepthPreview] кадр не построен: {exc}")
        return None
