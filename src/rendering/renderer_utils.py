import os
import time
import math
import re
import glob
import json
import datetime
import random
from panda3d.core import *

# Каталог случайных фонов для замены фона на обычном рендере.
# renderer_utils.py лежит в <root>/src/rendering/, поэтому корень — три уровня вверх.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BACKGROUNDS_DIR = os.path.join(_PROJECT_ROOT, "assets", "backgrounds")
_BACKGROUND_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class RendererUtils:
    def __init__(self, panda_app):
        self.panda_app = panda_app

    def _pick_random_background(self):
        """Случайный путь к фоновой картинке из assets/backgrounds или None."""
        try:
            files = [
                os.path.join(_BACKGROUNDS_DIR, f)
                for f in os.listdir(_BACKGROUNDS_DIR)
                if f.lower().endswith(_BACKGROUND_EXTS)
            ]
        except OSError:
            return None
        return random.choice(files) if files else None

    def _composite_random_background(self, img_final, mask_final, bg_path):
        """Заменить фон на цветном кадре случайной картинкой.

        Передний план (кузов + груз + ткань + other/насыпь) определяется по
        маске сегментации mask_final (уже с теми же дисторсией/кропом/
        растяжением, что и img_final). Все остальные пиксели заполняются
        картинкой bg_path. Возвращает новый PNMImage или None при ошибке.
        """
        try:
            import io
            import numpy as np
            from PIL import Image
            from src.rendering.segmentation_renderer import SEG_COLORS
        except Exception as exc:
            print(f"[Render] замена фона недоступна (PIL/numpy): {exc}")
            return None

        def _pnm_to_rgb_array(pnm):
            stream = StringStream()
            pnm.write(stream, "png")
            pil = Image.open(io.BytesIO(stream.getData())).convert("RGB")
            return np.asarray(pil), pil.size

        try:
            img_arr, size = _pnm_to_rgb_array(img_final)
            mask_arr, _ = _pnm_to_rgb_array(mask_final)
            bg_pil = Image.open(bg_path).convert("RGB").resize(
                size, Image.LANCZOS)
            bg_arr = np.asarray(bg_pil)

            mask_i = mask_arr.astype(np.int16)

            def _close(color, tol=40):
                d = np.abs(mask_i - np.array(color, dtype=np.int16))
                return (d[..., 0] <= tol) & (d[..., 1] <= tol) & (d[..., 2] <= tol)

            # Передний план = груз (cargo) + кузов (cuzov) + ткань (cloth).
            # Остальное — фон.
            cuzov_mask = _close(SEG_COLORS["cuzov"])
            keep = _close(SEG_COLORS["cargo"]) | cuzov_mask
            if "cloth" in SEG_COLORS:
                keep |= _close(SEG_COLORS["cloth"])
            if "other" in SEG_COLORS:
                keep |= _close(SEG_COLORS["other"])

            # 1) Цветовую температуру переднего плана подгоняем под фон, чтобы
            #    вставленный кузов+груз не выглядели «холоднее/теплее» картинки.
            fg_arr = self._match_color_temperature(img_arr, keep, bg_arr)

            # 2) Яркость фоновой картинки подгоняем под яркость рендера кузова
            #    (эталон — пиксели cuzov; если их мало, берём весь передний
            #    план). Так фон тускнеет ночью и светлеет днём вместе со сценой.
            ref_mask = cuzov_mask if int(cuzov_mask.sum()) >= 64 else keep
            bg_arr = self._match_brightness(bg_arr, img_arr, ref_mask)

            out = np.where(keep[..., None], fg_arr, bg_arr).astype(np.uint8)

            out_buf = io.BytesIO()
            Image.fromarray(out).save(out_buf, format="PNG")
            out_buf.seek(0)
            new_pnm = PNMImage()
            new_pnm.read(StringStream(out_buf.read()), "png")
            return new_pnm
        except Exception as exc:
            print(f"[Render] ошибка замены фона: {exc}")
            return None

    def _pnm_to_pil(self, pnm):
        """PNMImage -> PIL.Image (RGB)."""
        import io
        from PIL import Image
        stream = StringStream()
        pnm.write(stream, "png")
        return Image.open(io.BytesIO(stream.getData())).convert("RGB")

    def _pil_to_pnm(self, pil):
        """PIL.Image -> PNMImage."""
        import io
        out_buf = io.BytesIO()
        pil.convert("RGB").save(out_buf, format="PNG")
        out_buf.seek(0)
        new_pnm = PNMImage()
        new_pnm.read(StringStream(out_buf.read()), "png")
        return new_pnm

    def _keep_mask_from_array(self, mask_arr, tol=40):
        """bool-маска переднего плана (cargo+cuzov+cloth+other) из RGB-маски."""
        import numpy as np
        from src.rendering.segmentation_renderer import SEG_COLORS
        mask_i = mask_arr.astype(np.int16)

        def _close(color):
            d = np.abs(mask_i - np.array(color, dtype=np.int16))
            return (d[..., 0] <= tol) & (d[..., 1] <= tol) & (d[..., 2] <= tol)

        cuzov = _close(SEG_COLORS["cuzov"])
        keep = _close(SEG_COLORS["cargo"]) | cuzov
        if "cloth" in SEG_COLORS:
            keep |= _close(SEG_COLORS["cloth"])
        if "other" in SEG_COLORS:
            keep |= _close(SEG_COLORS["other"])
        return keep, cuzov

    def _apply_openai(self, img_final, processor, shadow_band=False):
        """OpenAI GPT Image: правка всего кадра одним запросом.

        Чтобы обработанный кадр совпадал с сегментацией, устраняем искажение
        соотношения сторон: OpenAI отдаёт только форматы вроде 1536x1024
        (3:2), а кадр 16:9. Поэтому кадр вписывается в холст нужного формата
        с сохранением пропорций (letterbox, серые поля), а из результата
        вырезается та же область и равномерно масштабируется обратно к
        1920x1080 — без горизонтального растяжения. PIL здесь используется
        ТОЛЬКО для геометрии (масштаб/паддинг/кроп); сам контент целиком
        генерирует OpenAI. shadow_band=True — тень «пополам» через промпт.

        Возвращает (PNMImage | None, meta | None).
        """
        try:
            import io as _io
            from PIL import Image
        except Exception as exc:
            print(f"[OpenAI] PIL недоступен: {exc}")
            return None, None
        try:
            color_pil = self._pnm_to_pil(img_final)
        except Exception as exc:
            print(f"[OpenAI] кодирование кадра не удалось: {exc}")
            return None, None

        W, H = color_pil.size
        cw, ch = 1536, 1024
        try:
            cw, ch = (int(v) for v in str(
                processor.config.get("openai_size", "1536x1024")).lower().split("x"))
        except Exception:
            pass

        # Вписать кадр в холст cw×ch с сохранением пропорций.
        s = min(cw / W, ch / H)
        nw, nh = max(1, round(W * s)), max(1, round(H * s))
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        canvas = Image.new("RGB", (cw, ch), (110, 110, 110))
        canvas.paste(color_pil.resize((nw, nh), Image.LANCZOS), (ox, oy))

        buf = _io.BytesIO()
        canvas.save(buf, format="PNG")
        out_bytes = processor.edit_whole(buf.getvalue(), shadow=shadow_band)
        meta = processor.last_prompts()
        if out_bytes is None:
            return None, meta
        try:
            out_pil = Image.open(_io.BytesIO(out_bytes)).convert("RGB")
            if out_pil.size != (cw, ch):
                out_pil = out_pil.resize((cw, ch), Image.LANCZOS)
            # Вырезаем контентную область (без полей) и возвращаем к кадру.
            crop = out_pil.crop((ox, oy, ox + nw, oy + nh))
            final = crop.resize((W, H), Image.LANCZOS)
            return self._pil_to_pnm(final), meta
        except Exception as exc:
            print(f"[OpenAI] разбор результата не удался: {exc}")
            return None, meta

    def _apply_gemini(self, img_final, mask_final, processor):
        """Gemini-постобработка цветного кадра.

        1) генерируем новый фон;
        2) выветриваем передний план (ржавчина/цвет/вмятины/кабели/фракции);
        3) собираем итог жёстким матированием по маске: силуэт переднего плана
           = mask_final (⇒ depth/seg GT остаётся точным), внутри — выветренный
           передний план, снаружи — новый фон.

        Возвращает (PNMImage | None, meta_dict | None).
        """
        try:
            import numpy as np
            from PIL import Image
        except Exception as exc:
            print(f"[Gemini] PIL/numpy недоступны: {exc}")
            return None, None

        try:
            color_pil = self._pnm_to_pil(img_final)
            mask_pil = self._pnm_to_pil(mask_final)
            size = color_pil.size
            img_arr = np.asarray(color_pil)
            mask_arr = np.asarray(mask_pil)
            keep, cuzov_mask = self._keep_mask_from_array(mask_arr)

            fg_mode = str(processor.config.get("foreground_mode", "procedural"))

            # Режим single_call: фон + AI-выветривание за один запрос (только
            # для fg_mode="ai"; силуэт держится на промпте, без матирования).
            if fg_mode == "ai" and processor.config.get("single_call"):
                full = processor.weather_full_scene(color_pil, mask_pil)
                meta = processor.last_prompts()
                meta["single_call"] = True
                meta["scene_generated"] = full is not None
                if full is None:
                    return None, meta
                return self._pil_to_pnm(full), meta

            # 1) Новый фон (через провайдера, если доступен).
            bg_pil = processor.generate_background(size[0], size[1])

            # 2) Передний план: ai (провайдер img2img) / procedural (оффлайн) /
            #    off (не трогать).
            fg_pil = None
            if fg_mode == "ai":
                fg_pil = processor.weather_foreground(color_pil, mask_pil)

            meta = processor.last_prompts()
            meta["foreground_mode"] = fg_mode
            meta["background_generated"] = bg_pil is not None
            meta["foreground_weathered"] = (
                fg_pil is not None or fg_mode == "procedural")

            fg_arr = (np.asarray(fg_pil.convert("RGB"))
                      if fg_pil is not None else img_arr)
            if fg_mode == "procedural":
                fg_arr = self._procedural_weathering(fg_arr, mask_arr)

            if bg_pil is None and fg_mode == "off":
                # Ни фона, ни выветривания — откат на исходный кадр.
                return None, meta

            if bg_pil is None:
                # Фон не сменился — оставляем исходный задний план.
                bg_arr = img_arr
            else:
                # Провайдер мог вернуть иной размер (FLUX округляет к кратному
                # 16) — приводим к размеру кадра, иначе матирование упадёт.
                if bg_pil.size != size:
                    bg_pil = bg_pil.resize(size, Image.LANCZOS)
                bg_arr = np.asarray(bg_pil.convert("RGB"))
                # Гармонизация (как в _composite_random_background):
                # температуру переднего плана тянем к фону, яркость фона — к
                # рендеру кузова.
                fg_arr = self._match_color_temperature(fg_arr, keep, bg_arr)
                ref_mask = (cuzov_mask if int(cuzov_mask.sum()) >= 64
                            else keep)
                bg_arr = self._match_brightness(bg_arr, img_arr, ref_mask)

            out = np.where(keep[..., None], fg_arr, bg_arr).astype(np.uint8)
            return self._pil_to_pnm(Image.fromarray(out)), meta
        except Exception as exc:
            print(f"[Gemini] ошибка постобработки: {exc}")
            return None, None

    def _procedural_weathering(self, img_arr, mask_arr):
        """Бесплатное оффлайн-выветривание переднего плана (numpy/PIL).

        Внутри масок кузова/груза: сдвиг цвета/выцветание кузова, потёки
        ржавчины, грязь/пыль, разнофракционные цветные вкрапления на грузе.
        Уровень (clean/light/heavy) случайный — часть кадров простые.
        img_arr, mask_arr — HxWx3 uint8 (RGB). Возвращает HxWx3 uint8.
        """
        import numpy as np
        from PIL import Image, ImageDraw
        from src.rendering.segmentation_renderer import SEG_COLORS

        h, w = img_arr.shape[:2]
        out = img_arr.astype(np.float32)
        mi = mask_arr.astype(np.int16)

        def close(color, tol=40):
            d = np.abs(mi - np.array(color, dtype=np.int16))
            return (d[..., 0] <= tol) & (d[..., 1] <= tol) & (d[..., 2] <= tol)

        cuzov = close(SEG_COLORS["cuzov"])
        cargo = close(SEG_COLORS["cargo"])
        tier = random.choices(
            ["clean", "light", "heavy"], weights=[0.2, 0.4, 0.4])[0]

        # 1) Цветовой сдвиг / выцветание кузова (разные цвета).
        if cuzov.any() and (tier != "clean" or random.random() < 0.5):
            gains = np.array([random.uniform(0.75, 1.25),
                              random.uniform(0.80, 1.15),
                              random.uniform(0.75, 1.25)], np.float32)
            gains *= random.uniform(0.8, 1.1)      # общая яркость
            out[cuzov] = np.clip(out[cuzov] * gains, 0, 255)

        # 2) Потёки ржавчины на кузове (вертикальные, сверху вниз).
        if cuzov.any() and tier in ("light", "heavy"):
            ys, xs = np.where(cuzov)
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            n = random.randint(3, 10) if tier == "heavy" else random.randint(1, 4)
            cols = np.unique(xs)
            for _ in range(n):
                x = int(random.choice(cols))
                colys = ys[xs == x]
                if colys.size == 0:
                    continue
                y0 = int(colys.min())
                length = int(random.uniform(0.2, 0.7) * h)
                wdt = random.randint(1, 4)
                base_c = (random.randint(110, 160),
                          random.randint(55, 90),
                          random.randint(25, 50))
                steps = 24
                for s in range(steps):
                    yy = y0 + int(length * s / steps)
                    a = int(150 * (1 - s / steps) * random.uniform(0.6, 1.0))
                    d.line([(x, yy),
                            (x + random.randint(-1, 1), yy + length // steps + 1)],
                           fill=base_c + (a,), width=wdt)
            rust = np.asarray(ov).astype(np.float32)
            a = (rust[..., 3] / 255.0) * cuzov
            for c in range(3):
                out[..., c] = out[..., c] * (1 - a) + rust[..., c] * a

        # 3) Грязь/пыль: низкочастотный шум затемнения на кузове+грузе.
        if tier in ("light", "heavy"):
            rs = np.random.RandomState(random.randint(0, 1 << 30))
            small = rs.rand(max(1, h // 24), max(1, w // 24)).astype(np.float32)
            dirt = np.asarray(
                Image.fromarray((small * 255).astype(np.uint8)).resize(
                    (w, h), Image.BILINEAR)).astype(np.float32) / 255.0
            strength = (random.uniform(0.10, 0.30) if tier == "heavy"
                        else random.uniform(0.05, 0.15))
            fac = 1.0 - strength * dirt
            m = cuzov | cargo
            out[m] = out[m] * fac[m][..., None]

        # 4) Разнофракционный груз: цветные вкрапления (бетон/металл/цветные).
        if cargo.any():
            ys, xs = np.where(cargo)
            ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            n = random.randint(20, 80) if tier == "heavy" else random.randint(8, 30)
            palette = [(150, 150, 145), (120, 120, 125), (80, 80, 85),
                       (60, 60, 60),                       # бетон/металл
                       (170, 120, 80), (200, 80, 60),
                       (90, 140, 90), (70, 90, 150)]        # цветные обломки
            for _ in range(n):
                i = random.randrange(xs.size)
                cx, cy = int(xs[i]), int(ys[i])
                r = random.randint(2, 7)
                col = random.choice(palette)
                a = random.randint(90, 200)
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (a,))
            sp = np.asarray(ov).astype(np.float32)
            a = (sp[..., 3] / 255.0) * cargo
            for c in range(3):
                out[..., c] = out[..., c] * (1 - a) + sp[..., c] * a

        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_shadow_band(self, img_final, mask_final):
        """Затемнить диагональную полосу, рассекающую передний план ~пополам.

        Только цветной кадр (linear-light умножение), GT не трогается. Полоса
        центрируется по bbox переднего плана; угол/ширина/мягкость/сила —
        случайные, чтобы кадры были разными.
        """
        try:
            import numpy as np
            from PIL import Image
        except Exception:
            return None
        try:
            img_arr = np.asarray(self._pnm_to_pil(img_final)).astype(np.float32)
            mask_arr = np.asarray(self._pnm_to_pil(mask_final))
            keep, _ = self._keep_mask_from_array(mask_arr)
            ys, xs = np.where(keep)
            if xs.size < 64:
                return None  # переднего плана почти нет — тень некуда класть

            cx = float(xs.mean())
            cy = float(ys.mean())
            h, w = img_arr.shape[:2]

            # Направление нормали к полосе (угол полосы — случайный).
            ang = random.uniform(0.0, math.pi)
            nx, ny = math.cos(ang), math.sin(ang)

            # Расстояние каждого пикселя от линии через (cx, cy).
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            dist = (xx - cx) * nx + (yy - cy) * ny

            # Мягкий переход тень/свет. half — половина ширины перехода.
            fg_extent = max(xs.ptp(), ys.ptp(), 1)
            half = fg_extent * random.uniform(0.06, 0.18)
            darkness = random.uniform(0.35, 0.6)   # во сколько раз темнее
            # Полоса тени с одной стороны линии; плавный край через tanh.
            t = np.tanh(dist / max(half, 1.0))      # -1..1
            shade = 1.0 - (1.0 - darkness) * (0.5 * (1.0 + t))  # затемн. сторона

            lin = np.power(img_arr / 255.0, 2.2)
            lin *= shade[..., None]
            out = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2) * 255.0
            return self._pil_to_pnm(
                Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)))
        except Exception as exc:
            print(f"[Render] ошибка теневой полосы: {exc}")
            return None

    def _match_color_temperature(self, img_arr, keep, bg_arr,
                                 strength=0.85, gain_min=0.6, gain_max=1.7):
        """Подогнать цветовую температуру переднего плана под фон.

        Оцениваем «точку белого» каждого изображения как среднюю линейную
        яркость по каналам (gray-world): для фона — по всей картинке, для
        переднего плана — только по сохраняемым пикселям (keep). Затем —
        яркостно-нейтральная адаптация фон Криза: масштабируем каналы R и B
        переднего плана так, чтобы баланс R/G и B/G совпал с фоном (G не
        трогаем, чтобы не менять общую яркость). Тёплый фон → передний план
        теплеет, холодный → холоднеет.

        img_arr, bg_arr — HxWx3 uint8 (RGB); keep — HxW bool.
        Возвращает HxWx3 uint8 (только передний план реально используется).
        strength — сила коррекции (0..1); gain_min/max — клип усиления.
        """
        import numpy as np

        eps = 1e-4

        def to_linear(a):
            # sRGB(8-бит) -> линейный свет (приближённая гамма 2.2).
            return np.power(a.astype(np.float32) / 255.0, 2.2)

        fg_lin = to_linear(img_arr)
        fg_pixels = fg_lin[keep]
        # Слишком мало переднего плана — не из чего оценивать, не трогаем.
        if fg_pixels.shape[0] < 64:
            return img_arr

        fg_mean = fg_pixels.reshape(-1, 3).mean(axis=0) + eps
        bg_mean = to_linear(bg_arr).reshape(-1, 3).mean(axis=0) + eps

        # Баланс относительно зелёного (нормировка по яркости).
        fg_wb = fg_mean / fg_mean[1]
        bg_wb = bg_mean / bg_mean[1]
        gains = bg_wb / fg_wb                       # (g_r, 1.0, g_b)

        # Умеренная сила + клип, чтобы не уехать в крайности.
        gains = 1.0 + (gains - 1.0) * float(strength)
        gains = np.clip(gains, gain_min, gain_max)
        gains[1] = 1.0                              # зелёный не трогаем

        corrected = np.clip(fg_lin * gains, 0.0, 1.0)
        corrected = np.power(corrected, 1.0 / 2.2) * 255.0
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _match_brightness(self, bg_arr, fg_arr, ref_mask,
                          scale_min=0.02, scale_max=8.0):
        """Подогнать яркость фоновой картинки под яркость рендера.

        Считаем среднюю линейную яркость (luma Rec.709) эталонной области
        переднего плана (ref_mask — обычно пиксели кузова) и всей фоновой
        картинки, затем масштабируем фон так, чтобы его средняя яркость
        совпала с эталоном. Тёмный (ночной) рендер -> фон темнеет, светлый
        (дневной) -> фон светлеет.

        bg_arr, fg_arr — HxWx3 uint8 (RGB); ref_mask — HxW bool.
        Возвращает HxWx3 uint8.

        В эталон берём ТОЛЬКО пиксели ref_mask (передний план) и при этом
        отбрасываем почти-чёрные: barrel distortion заливает обрезанные
        края чёрным, а из-за того что маска тянется ближайшим соседом, а
        цветной кадр — билинейно, на стыке кузова и чёрной каймы появляется
        тонкое кольцо тёмных пикселей, помеченных как кузов. Если их учесть,
        яркость переднего плана занижается и фон выходит слишком тёмным.
        """
        import numpy as np

        eps = 1e-4
        coef = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

        def to_linear(a):
            return np.power(a.astype(np.float32) / 255.0, 2.2)

        # Только маскированный передний план (НЕ весь кадр).
        ref_rgb = fg_arr[ref_mask]
        # Выкидываем обрезанный фон / чёрную кайму (почти-чёрные пиксели).
        ref_rgb = ref_rgb[ref_rgb.max(axis=1) > 12]
        # Слишком мало эталона — не из чего оценивать, фон не трогаем.
        if ref_rgb.shape[0] < 64:
            return bg_arr

        fg_lum = float((to_linear(ref_rgb) @ coef).mean()) + eps
        bg_lin = to_linear(bg_arr)
        bg_lum = float((bg_lin.reshape(-1, 3) @ coef).mean()) + eps

        scale = float(np.clip(fg_lum / bg_lum, scale_min, scale_max))

        out = np.clip(bg_lin * scale, 0.0, 1.0)
        out = np.power(out, 1.0 / 2.2) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def barrel_distortion(self, img, k1=0.15, k2=0.35):
        tex = Texture()
        tex.load(img)
        
        distortion_map = PNMImage(img.getXSize(), img.getYSize())
        
        width = img.getXSize()
        height = img.getYSize()
        center_x = width / 2.0
        center_y = height / 2.0
        
        max_dist = min(center_x, center_y)
        
        for y in range(height):
            for x in range(width):
                norm_x = (x - center_x) / max_dist
                norm_y = (y - center_y) / max_dist
                
                r = math.sqrt(norm_x * norm_x + norm_y * norm_y)
                
                distortion = 1.0 + k1 * r * r + k2 * r * r * r * r
                
                new_x = norm_x * distortion
                new_y = norm_y * distortion
                
                src_x = int(center_x + new_x * max_dist)
                src_y = int(center_y + new_y * max_dist)
                
                if 0 <= src_x < width and 0 <= src_y < height:
                    color = img.getXel(src_x, src_y)
                    distortion_map.setXel(x, y, color)
                else:
                    distortion_map.setXel(x, y, 0, 0, 0)
        
        return distortion_map
    
    def crop_image(self, img, left=270, top=155, right=1850, bottom=925):
        width = img.getXSize()
        height = img.getYSize()
        
        if left < 0 or top < 0 or right > width or bottom > height:
            print(f"Предупреждение: координаты обрезки выходят за пределы изображения")
            print(f"Размер изображения: {width}x{height}")
            print(f"Запрошенные координаты: left={left}, top={top}, right={right}, bottom={bottom}")
            
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
        
        if left >= right or top >= bottom:
            print(f"Ошибка: некорректные координаты обрезки")
            print(f"left={left}, top={top}, right={right}, bottom={bottom}")
            return PNMImage(img)
        
        crop_width = right - left
        crop_height = bottom - top
        
        cropped_img = PNMImage(crop_width, crop_height, img.getNumChannels(), img.getMaxval())
        
        for y in range(crop_height):
            src_y = top + y
            for x in range(crop_width):
                src_x = left + x
                color = img.getXel(src_x, src_y)
                cropped_img.setXel(x, y, color)
        
        return cropped_img
    
    def fix_alpha_to_opaque(self, img):
        # If image has no alpha channel, add one
        if not img.hasAlpha():
            img.addAlpha()

        width = img.getXSize()
        height = img.getYSize()

        for y in range(height):
            for x in range(width):
                img.setAlpha(x, y, 1.0)

        return img

    def stretch_to_1920x1080(self, img, nearest=False):
        # nearest=True — ближайший сосед (без интерполяции). Нужен для масок
        # сегментации: билинейное смешение размывало бы границы классов и
        # порождало промежуточные цвета, которых нет в палитре.
        target_width = 1920
        target_height = 1080

        current_width = img.getXSize()
        current_height = img.getYSize()

        if current_width == target_width and current_height == target_height:
            return PNMImage(img)

        stretched_img = PNMImage(target_width, target_height, img.getNumChannels(), img.getMaxval())

        scale_x = target_width / current_width
        scale_y = target_height / current_height

        if nearest:
            for y in range(target_height):
                src_y = min(int(y / scale_y), current_height - 1)
                for x in range(target_width):
                    src_x = min(int(x / scale_x), current_width - 1)
                    stretched_img.setXel(x, y, img.getXel(src_x, src_y))
            return stretched_img

        for y in range(target_height):
            for x in range(target_width):
                src_x = x / scale_x
                src_y = y / scale_y

                x0 = int(math.floor(src_x))
                x1 = min(x0 + 1, current_width - 1)
                y0 = int(math.floor(src_y))
                y1 = min(y0 + 1, current_height - 1)

                dx = src_x - x0
                dy = src_y - y0

                c00 = img.getXel(x0, y0)
                c10 = img.getXel(x1, y0)
                c01 = img.getXel(x0, y1)
                c11 = img.getXel(x1, y1)

                c0 = [c00[i] * (1 - dx) + c10[i] * dx for i in range(3)]
                c1 = [c01[i] * (1 - dx) + c11[i] * dx for i in range(3)]

                color = [c0[i] * (1 - dy) + c1[i] * dy for i in range(3)]

                stretched_img.setXel(x, y, color[0], color[1], color[2])

        return stretched_img
    
    def _process_render_image(self, img=None, camera_fov_x=None, camera_fov_y=None, output_dir="renders",
                         filename_prefix="render", metadata=None, dataset_type="depth",
                         seg_mask=None, bg_path=None, gemini_processor=None,
                         shadow_band=False, depth_img=None, seg_img=None,
                         outputs=None, lidar_scan=None, lidar_settings=None):
        # img / depth_img / seg_img — три независимых кадра одного сэмпла.
        # Какие из них лягут на диск, решает `outputs` (см. resolve_outputs);
        # None вместо кадра означает «не снимали». Все три проходят ОДНИ И ТЕ
        # ЖЕ дисторсию/кроп/растяжение, поэтому совпадают попиксельно.
        #
        # seg_mask + bg_path: заменить фон на цветном кадре (только на нём!)
        #               случайной картинкой bg_path. seg_mask (1920x1080)
        #               проходит ту же дисторсию и используется как вырез
        #               переднего плана (кузов + груз).
        outputs = self.resolve_outputs(outputs, dataset_type,
                                       depth_img is not None)
        want_color = "color" in outputs and img is not None
        want_depth = "depth" in outputs and depth_img is not None
        want_seg = "segmentation" in outputs and seg_img is not None
        want_lidar = "lidar" in outputs and lidar_scan is not None
        want_json = "json" in outputs

        # Геометрия кадра берётся с любого снятого изображения — все они
        # выходят из одного и того же окна и имеют один размер.
        size_ref = img if img is not None else (depth_img or seg_img)
        if size_ref is None and not want_lidar:
            print("[Render] нечего сохранять: ни одного кадра не снято.")
            return None
        # Облако точек может быть ЕДИНСТВЕННЫМ выходом кадра: оно не растр и
        # размеров не имеет. Геометрию кадра тогда берём с окна — она нужна
        # только для интринсик в json.
        if size_ref is not None:
            orig_width = size_ref.getXSize()
            orig_height = size_ref.getYSize()
        else:
            win = getattr(self.panda_app, "win", None)
            orig_width = int(win.get_x_size()) if win is not None else 1920
            orig_height = int(win.get_y_size()) if win is not None else 1080
        
        fx = fy = cx = cy = None
        lens = self.panda_app.cam.node().getLens() if hasattr(self.panda_app, 'cam') else None
        
        if camera_fov_x is not None and camera_fov_y is not None:
            fx = (orig_width / 2.0) / math.tan(math.radians(camera_fov_x / 2.0))
            fy = (orig_height / 2.0) / math.tan(math.radians(camera_fov_y / 2.0))
            cx = orig_width / 2.0
            cy = orig_height / 2.0
        
        # Определяем 3D точки для преобразования
        top_points_3d = [
            (-1.03, -2.22, 2.4),   # bottom_left_top
            (-1.03, 2.4, 2.4),     # top_left_top
            (1.045, 2.4, 2.4),     # top_right_top
            (1.045, -2.22, 2.4)    # bottom_right_top
        ]
        
        top_points_2d = []
        distances_to_camera = []  # Для хранения расстояний до камеры
        
        # Преобразуем 3D точки в 2D пиксельные координаты (до всех преобразований)
        if lens:
            for i, point_3d in enumerate(top_points_3d):
                # Создаем точку в координатах Panda3D
                point = LPoint3f(point_3d[0], point_3d[1], point_3d[2])
                
                # Преобразуем мировые координаты в координаты камеры
                point_in_camera_space = self.panda_app.camera.getRelativePoint(
                    self.panda_app.render, 
                    point
                )
                
                # Вычисляем расстояние от камеры до точки (в мировых координатах)
                # Получаем позицию камеры в мировых координатах
                camera_pos = self.panda_app.camera.getPos(self.panda_app.render)
                
                # Вычисляем расстояние между камерой и точкой
                distance = math.sqrt(
                    (point_3d[0] - camera_pos.x) ** 2 +
                    (point_3d[1] - camera_pos.y) ** 2 +
                    (point_3d[2] - camera_pos.z) ** 2
                )
                
                distances_to_camera.append(float(distance))
                
                # Создаем точки для результата
                result_point = LPoint3f()
                
                # Пытаемся спроецировать точку
                success = lens.project(point_in_camera_space, result_point)
                
                if success:
                    # Преобразуем NDC в пиксельные координаты
                    pixel_x = (result_point.x * 0.5 + 0.5) * orig_width
                    pixel_y = (0.5 - result_point.y * 0.5) * orig_height  # Инвертируем Y
                    
                    # Проверяем, находится ли точка в пределах экрана
                    if (0 <= pixel_x < orig_width and 0 <= pixel_y < orig_height and 
                        point_in_camera_space.y > 0 and 0 <= result_point.z <= 1):
                        top_points_2d.append({
                            "x": float(pixel_x),
                            "y": float(pixel_y)
                        })
                    else:
                        top_points_2d.append(None)
                else:
                    top_points_2d.append(None)
        else:
            top_points_2d = [None] * len(top_points_3d)
            distances_to_camera = [None] * len(top_points_3d)
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{filename_prefix}_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        # Суффиксы фиксированы и не зависят от режима: _depth — всегда карта
        # глубины, _seg — всегда маска. Раньше «второй файл» назывался то
        # так, то эдак в зависимости от типа датасета.
        output_path_depth = os.path.join(
            output_dir, f"{filename_prefix}_{timestamp}_depth.png")
        output_path_seg = os.path.join(
            output_dir, f"{filename_prefix}_{timestamp}_seg.png")
        output_path_json = os.path.join(
            output_dir, f"{filename_prefix}_{timestamp}.json")
        output_path_lidar = os.path.join(
            output_dir, f"{filename_prefix}_{timestamp}_lidar.ply")

        # Параметры преобразований
        k1 = 0.04
        k2 = k1

        crop_div = 1.5

        crop_left = round(423/crop_div)
        crop_top = round(238/crop_div)
        final_width = 1920
        final_height = 1080
        crop_right = final_width - crop_left
        crop_bottom = final_height - crop_top
        
        # Создаем копии изображения для каждого этапа преобразования
        img_final = None
        if img is not None:
            img_distorted = self.barrel_distortion(img, k1=k1, k2=k2)
            img_cropped = self.crop_image(img_distorted, left=crop_left, top=crop_top, right=crop_right, bottom=crop_bottom)
            img_final = self.stretch_to_1920x1080(img_cropped)

        # Маску сегментации (если передана) прогоняем через те же
        # дисторсию/кроп/растяжение — она нужна для: (а) замены фона случайной
        # картинкой, (б) Gemini-постобработки (матирование силуэта), (в)
        # теневой полосы. mask_final попиксельно совпадает с img_final.
        mask_final = None
        if seg_mask is not None:
            mask_distorted = self.barrel_distortion(seg_mask, k1=k1, k2=k2)
            mask_cropped = self.crop_image(
                mask_distorted, left=crop_left, top=crop_top,
                right=crop_right, bottom=crop_bottom,
            )
            mask_final = self.stretch_to_1920x1080(mask_cropped, nearest=True)

        # Замена фона случайной картинкой — ТОЛЬКО на цветном кадре и уже
        # после дисторсии. Оставляем кузов+груз, остальное заливаем картинкой.
        background_name = None
        gemini_meta = None
        openai_active = (gemini_processor is not None
                         and hasattr(gemini_processor, "edit_whole"))
        if img_final is None:
            # Цветного кадра нет (снимаем только разметку) — подменять в нём
            # нечего.
            openai_active = False
            gemini_processor = None
            bg_path = None
        if openai_active:
            # OpenAI: редактируем ВЕСЬ кадр одним запросом (без маски и без
            # матирования — силуэт может слегка сместиться, GT не строгий).
            # Тень «пополам» (shadow_band) добавляется через промпт.
            edited, gemini_meta = self._apply_openai(
                img_final, gemini_processor, shadow_band=shadow_band)
            if edited is not None:
                img_final = edited
                background_name = "openai"
        elif gemini_processor is not None and mask_final is not None:
            # Gemini-постобработка: новый фон + выветривание переднего плана.
            # Силуэт GT сохраняется жёстким матированием по mask_final.
            composited, gemini_meta = self._apply_gemini(
                img_final, mask_final, gemini_processor)
            if composited is not None:
                img_final = composited
                background_name = "gemini"
        elif bg_path is not None and mask_final is not None:
            composited = self._composite_random_background(
                img_final, mask_final, bg_path)
            if composited is not None:
                img_final = composited
                background_name = os.path.basename(bg_path)

        # Теневая полоса «рассекает пополам» кузов+груз (только цветной кадр,
        # GT не трогается). Для OpenAI тень уже в промпте — локальную не даём.
        if (shadow_band and img_final is not None and mask_final is not None
                and not openai_active):
            shadowed = self._apply_shadow_band(img_final, mask_final)
            if shadowed is not None:
                img_final = shadowed

        # Те же самые искажения применяем ко второму кадру (карта глубины
        # ИЛИ маска сегментации), чтобы он попиксельно совпадал с цветным.
        # Для сегментации финальный stretch — ближайшим соседом, иначе
        # билинейная интерполяция размыла бы границы классов.
        def _warp_like_color(source, nearest):
            """Те же дисторсия/кроп/растяжение, что и у цветного кадра."""
            distorted = self.barrel_distortion(source, k1=k1, k2=k2)
            cropped = self.crop_image(
                distorted, left=crop_left, top=crop_top,
                right=crop_right, bottom=crop_bottom,
            )
            warped = self.stretch_to_1920x1080(cropped, nearest=nearest)
            return self.fix_alpha_to_opaque(warped)

        # Глубина — величина непрерывная, поэтому билинейно. Маска — набор
        # классов, поэтому ближайшим соседом: интерполяция породила бы на
        # границах цвета несуществующих классов.
        depth_final = (_warp_like_color(depth_img, nearest=False)
                       if want_depth else None)
        seg_final = (_warp_like_color(seg_img, nearest=True)
                     if want_seg else None)

        # Преобразуем 2D точки с учетом всех примененных трансформаций
        transformed_points_2d = []
        
        if top_points_2d:
            for i, point_2d in enumerate(top_points_2d):
                if point_2d is None:
                    transformed_points_2d.append(None)
                    continue
                    
                x = point_2d["x"]
                y = point_2d["y"]
                
                # Итерационный метод для нахождения правильных координат после barrel distortion
                x_dist = x
                y_dist = y
                
                center_x = orig_width / 2.0
                center_y = orig_height / 2.0
                max_dist = min(center_x, center_y)
                
                # Итерационный метод для решения обратной задачи
                for iteration in range(10):
                    norm_x_dist = (x_dist - center_x) / max_dist
                    norm_y_dist = (y_dist - center_y) / max_dist
                    
                    r = math.sqrt(norm_x_dist * norm_x_dist + norm_y_dist * norm_y_dist)
                    distortion = 1.0 + k1 * r * r + k2 * r * r * r * r
                    
                    # Прямое преобразование (точка в искаженном изображении)
                    norm_x_distorted = norm_x_dist * distortion
                    norm_y_distorted = norm_y_dist * distortion
                    
                    x_calc = center_x + norm_x_distorted * max_dist
                    y_calc = center_y + norm_y_distorted * max_dist
                    
                    # Вычисляем ошибку
                    error_x = x_calc - x
                    error_y = y_calc - y
                    
                    # Корректируем предположение
                    x_dist -= error_x * 0.5
                    y_dist -= error_y * 0.5
                    
                    if abs(error_x) < 0.1 and abs(error_y) < 0.1:
                        break
                
                # 2. Применяем crop (вычитаем смещение)
                x_cropped = x_dist - crop_left
                y_cropped = y_dist - crop_top
                
                # Проверяем, попадает ли точка в область crop
                crop_width = crop_right - crop_left
                crop_height = crop_bottom - crop_top
                
                if (0 <= x_cropped < crop_width and 0 <= y_cropped < crop_height):
                    
                    # 3. Применяем stretch (масштабирование до 1920x1080)
                    scale_x = final_width / crop_width
                    scale_y = final_height / crop_height
                    
                    x_final = x_cropped * scale_x
                    y_final = y_cropped * scale_y
                    
                    transformed_points_2d.append({
                        "x": float(x_final),
                        "y": float(y_final)
                    })
                else:
                    transformed_points_2d.append(None)
        
        if(False):
            # === ДОБАВЛЕНИЕ ЦВЕТНЫХ КРУГОВ ===
            try:
                from PIL import Image, ImageDraw
                import io
                from panda3d.core import StringStream
                
                # Конвертируем PNMImage в PIL Image
                stream = StringStream()
                img_final.write(stream, "png")
                pil_img = Image.open(io.BytesIO(stream.getData()))
                draw = ImageDraw.Draw(pil_img)
                
                colors = [
                    (255, 0, 0, 200),    # красный для bottom_left_top
                    (0, 255, 0, 200),    # зеленый для top_left_top
                    (0, 0, 255, 200),    # синий для top_right_top
                    (255, 255, 0, 200),  # желтый для bottom_right_top
                ]
                
                # Рисуем круги для каждой точки, которая не None
                for i, point_2d in enumerate(transformed_points_2d):
                    if point_2d is not None:
                        x = int(point_2d["x"])
                        y = int(point_2d["y"])
                        color = colors[i % len(colors)]
                        
                        # Рисуем круг с радиусом 10 пикселей
                        draw.ellipse([(x-10, y-10), (x+10, y+10)], fill=color, outline=(255, 255, 255, 255))
                
                # Конвертируем обратно в PNMImage
                output = io.BytesIO()
                pil_img.save(output, format="PNG")
                output.seek(0)
                new_img = PNMImage()
                new_img.read(StringStream(output.read()), "png")
                
                # Заменяем исходное изображение на новое
                img_final = new_img
                
            except ImportError:
                print("Warning: Pillow not installed. Skipping circle drawing.")
            except Exception as e:
                print(f"Warning: Error while drawing circles: {e}")
        
        # Сохраняем ровно те файлы, которые запрошены.
        saved_color = saved_depth = saved_seg = None
        if want_color and img_final is not None:
            img_final.write(Filename.from_os_specific(output_path))
            saved_color = output_path
        if depth_final is not None:
            depth_final.write(Filename.from_os_specific(output_path_depth))
            saved_depth = output_path_depth
        if seg_final is not None:
            seg_final.write(Filename.from_os_specific(output_path_seg))
            saved_seg = output_path_seg

        saved_lidar = None
        if want_lidar:
            cfg_lidar = lidar_settings or {}
            try:
                lidar_scan.write_ply(
                    output_path_lidar,
                    binary=bool(cfg_lidar.get("binary", True)),
                    with_color=bool(cfg_lidar.get("color", True)),
                )
                saved_lidar = output_path_lidar
            except Exception as exc:
                print(f"[Lidar] .ply не записан: {exc}")
        
        # Формируем render_metadata только с необходимыми данными
        render_metadata = {}

        # dataset_type остался ради уже собранных датасетов и их
        # разборщиков. Он больше не приходит извне как режим — выводим его
        # из того, что реально легло на диск.
        render_metadata["dataset_type"] = (
            "segmentation" if saved_seg
            else "depth" if saved_depth
            else "color" if saved_color
            else "lidar" if saved_lidar
            else "color"
        )
        render_metadata["random_background"] = background_name
        render_metadata["shadow_band"] = bool(shadow_band)
        if gemini_meta:
            render_metadata["gemini"] = gemini_meta
        # Имена файлов кадра. second_image оставлен для обратной
        # совместимости с уже собранными датасетами и разборщиками: там он
        # означал «второй кадр» — маску, если она есть, иначе глубину.
        render_metadata["outputs"] = sorted(outputs)
        render_metadata["color_image"] = (
            os.path.basename(saved_color) if saved_color else None)
        render_metadata["depth_image"] = (
            os.path.basename(saved_depth) if saved_depth else None)
        render_metadata["segmentation_image"] = (
            os.path.basename(saved_seg) if saved_seg else None)
        render_metadata["lidar_cloud"] = (
            os.path.basename(saved_lidar) if saved_lidar else None)
        if saved_lidar:
            # Полный паспорт съёмки облака: поза сенсора, развёртка, шум и
            # раскладка точек по классам. Без него .ply — просто координаты.
            render_metadata["lidar"] = lidar_scan.meta
        render_metadata["second_image"] = (
            render_metadata["segmentation_image"]
            or render_metadata["depth_image"]
        )
        if saved_seg:
            try:
                from src.rendering.segmentation_renderer import (
                    SEG_COLORS, SEG_BACKGROUND,
                )
                render_metadata["segmentation_palette"] = {
                    "background": list(SEG_BACKGROUND),
                    **{k: list(v) for k, v in SEG_COLORS.items()},
                }
            except Exception:
                pass

        # Параметры barrel distortion
        render_metadata["barrel_distortion"] = {
            "k1": k1,
            "k2": k2
        }
        
        # Параметры камеры
        if camera_fov_x is not None and camera_fov_y is not None:
            render_metadata["camera_params"] = {
                "fov_x": camera_fov_x,
                "fov_y": camera_fov_y,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            }
        
        # 3D точки
        render_metadata["points_3d"] = top_points_3d
        
        # 2D точки до всех преобразований (оригинальные)
        render_metadata["points_2d_original"] = []
        for point_2d in top_points_2d:
            if point_2d is not None:
                render_metadata["points_2d_original"].append({
                    "x": point_2d["x"],
                    "y": point_2d["y"]
                })
            else:
                render_metadata["points_2d_original"].append(None)
        
        # 2D точки после преобразований
        render_metadata["points_2d"] = []
        for i, point_2d in enumerate(transformed_points_2d):
            if point_2d is not None:
                render_metadata["points_2d"].append({
                    "x": point_2d["x"],
                    "y": point_2d["y"]
                })
            else:
                render_metadata["points_2d"].append(None)
        
        # Расстояния от камеры до 3D точек
        render_metadata["distances_to_camera"] = distances_to_camera
        
        # Добавляем Target_Volume
        if hasattr(self.panda_app, 'Target_Volume'):
            render_metadata["target_volume"] = self.panda_app.Target_Volume
        else:
            render_metadata["target_volume"] = None
        
        # Добавляем current_texture_set['diffuse']
        if (hasattr(self.panda_app, 'current_texture_set') and 
            self.panda_app.current_texture_set and 
            'diffuse' in self.panda_app.current_texture_set):
            render_metadata["texture_diffuse"] = self.panda_app.current_texture_set['diffuse']
        else:
            render_metadata["texture_diffuse"] = None
        
        # Сливаем пользовательские метаданные (variant, camera state и т.п.),
        # которые раньше передавались, но игнорировались.
        if metadata:
            for k, v in metadata.items():
                if k not in render_metadata:
                    render_metadata[k] = v
                else:
                    render_metadata[f"extra_{k}"] = v

        if want_json:
            with open(output_path_json, 'w', encoding='utf-8') as f:
                json.dump(render_metadata, f, indent=2, ensure_ascii=False)

        # Возвращаем «главный» файл кадра — цветной, а если его не снимали,
        # то любой сохранённый: вызывающий логирует именно его.
        return saved_color or saved_seg or saved_depth or saved_lidar or (
            output_path_json if want_json else None)
    
    def create_video_from_frames(self, output_dir="renders/datasets_metric_/", video_name="camera_rotation.mp4", fps=20):
        search_pattern = os.path.join(output_dir, "render_*_frame_*.png")
        frame_files = glob.glob(search_pattern)
        
        frame_files.sort(key=lambda x: int(re.search(r'frame_(\d{3})', x).group(1)))
        
        if not frame_files:
            return False
        
        list_file = os.path.join(os.getcwd(), "frames_list.txt")
        with open(list_file, "w") as f:
            for frame in frame_files:
                abs_path = os.path.abspath(frame).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-r', str(fps),
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '22',
                '-pix_fmt', 'yuv420p',
                '-y',
                video_name
            ]
            
            return True
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def wait_panda_render(self, ticks=12):
        # Больше ручных тиков RenderPipeline -> постэффекты (TAA, motion blur,
        # bloom и т.п.) успевают сойтись к стационарной картинке, иначе
        # на скриншотах остаётся "смаз" после движения камеры/смены света.
        for _ in range(max(1, int(ticks))):
            self.panda_app.graphicsEngine.renderFrame()

    def settle_render(self, frames=30):
        """Прокрутить РЕАЛЬНЫЕ кадры пайплайна через taskMgr.step().

        В отличие от wait_panda_render (голый graphicsEngine.renderFrame,
        который НЕ выполняет per-frame апдейты плагинов RenderPipeline),
        taskMgr.step() гоняет on_pre_render_update — в т.ч. переобновление
        камерного рига PSSM под новое положение камеры/солнца. Без этого
        после сдвига камеры каскадные тени не успевают перестроиться и на
        кадр вылезает огромная «ложная» тень на полкадра. Тот же приём уже
        используется в save_dataset_render. Кадры реальные, поэтому обычный
        time.sleep тут не нужен (он лишь блокирует QTimer, который и так
        делает taskMgr.step, т.е. паузы «между кадрами» ничего не сглаживали).
        """
        tm = getattr(self.panda_app, "taskMgr", None)
        if tm is None:
            self.wait_panda_render(ticks=frames)
            return
        for _ in range(max(1, int(frames))):
            tm.step()

    # ==================================================================
    # ЗАХВАТ КАДРА — ТОЛЬКО ИЗ OFFSCREEN-БУФЕРОВ, ОКНО НЕ ЧИТАЕТСЯ НИКОГДА.
    #
    # КОРЕНЬ БАГА («оригинал = чужой кадр» / «в датасет попадают GitHub,
    # VSCode, проводник»).
    #
    # Окно Panda встроено ДОЧЕРНИМ HWND в Qt-виджет (main_window: winId() +
    # SetWindowPos по _panda_hwnd). У такого окна нет собственного
    # композитора, и win.get_screenshot() читает НЕ GL-поверхность, а
    # ПИКСЕЛИ РАБОЧЕГО СТОЛА под областью окна. Отсюда посторонние окна в
    # датасете. Если же окно вдобавок не рисуется (свёрнуто/перекрыто), в
    # буфере остаётся кадр ПРЕДЫДУЩЕГО сэмпла — та самая «маска текущая,
    # цвет предыдущий».
    #
    # Измерено (свёрнутое окно, 8 итераций):
    #     win.get_screenshot()   -> устаревший кадр, 6/8 неверных
    #     triggered copy окна    -> копии НЕТ вообще, 8/8
    #     offscreen-буфер        -> ВЕРНО 8/8
    #
    # И отдельно проверено на RenderPipeline (видимое / свёрнутое /
    # восстановленное окно): чтение цели FinalStage даёт АКТУАЛЬНЫЙ кадр во
    # всех трёх состояниях.
    #
    # ПОЭТОМУ все три выхода датасета берутся из offscreen-FBO:
    #     цвет   <- RenderPipeline FinalStage (полная постобработка)
    #     маска  <- SegmentationRenderer (свой буфер, уже так работало)
    #     глубина<- DepthMapRenderer + свой offscreen-колорайзер
    # Состояние окна не влияет ни на один из них, и все три согласованы по
    # кадру между собой.
    # ==================================================================
    def _rp_final_target(self):
        """RenderTarget стадии FinalStage (цель с полной постобработкой).

        Кэшируем только сам ОБЪЕКТ СТАДИИ (он живёт всё время), но НЕ его
        target: RenderPipeline слушает 'window-event' и при изменении размера
        окна ПЕРЕСОЗДАЁТ все свои render target'ы. Закэшированный target (и
        тем более его буфер) после этого висячий, и trigger_copy по нему роняет
        процесс — ровно это и происходило при разворачивании окна. Поэтому
        target берём заново на каждый захват (обход списка стадий дёшев).
        """
        rp = getattr(self.panda_app, "render_pipeline", None)
        if rp is None:
            return None
        stage = getattr(self, "_final_stage", None)
        if stage is None:
            try:
                for s in rp.stage_mgr.stages:
                    if type(s).__name__ == "FinalStage":
                        stage = s
                        self._final_stage = s
                        break
            except Exception as exc:
                print(f"[Render] поиск FinalStage не удался: {exc}")
                return None
        if stage is None:
            return None
        try:
            return stage.target
        except Exception:
            return None

    def _ensure_triggered_tex(self, buffer, attr):
        """Повесить на буфер triggered-copy-текстуру (с кэшем ПО БУФЕРУ).

        Кэш привязан к конкретному буферу: если RenderPipeline пересоздал
        цель (изменился размер окна), буфер будет ДРУГИМ — тогда вешаем
        текстуру заново, а не используем висячую.

        RTM_triggered_copy_ram копирует содержимое внутри end_frame() — после
        отрисовки и до flip'а — и только по явному запросу, поэтому в обычных
        кадрах не стоит ничего.
        """
        if buffer is None:
            return None
        cache = getattr(self, "_trig_cache", None)
        if cache is None:
            cache = self._trig_cache = {}
        key = (attr, id(buffer))
        entry = cache.get(key)
        if entry is not None:
            return entry
        try:
            tex = Texture(attr)
            ok = buffer.add_render_texture(
                tex,
                GraphicsOutput.RTM_triggered_copy_ram,
                GraphicsOutput.RTP_color,
            )
            # На этой сборке Panda add_render_texture возвращает None при
            # успехе — отказом считаем только явный False.
            if ok is False:
                raise RuntimeError("add_render_texture отклонён")
        except Exception as exc:
            print(f"[Render] triggered-copy недоступен ({attr}): {exc}")
            return None
        # Держим ссылку на буфер: пока запись жива, буфер не будет собран GC,
        # т.е. id() не может быть переиспользован другим объектом.
        cache[key] = tex
        cache[("buf",) + key] = buffer
        return tex

    def _read_triggered(self, buffer, tex, img, *, settle=0, tries=20):
        """Снять кадр буфера через triggered-copy. False, если копия не пришла.

        Отсутствие копии означает, что буфер не рисовался, т.е. изображение
        было бы устаревшим — тогда честный отказ, а не тихая порча датасета.
        """
        if settle:
            self._step_frames(settle)
        tex.clear_ram_image()          # иначе примем копию прошлого захвата
        buffer.trigger_copy()
        self._step_frames(1)
        for _ in range(tries):
            if tex.has_ram_image():
                break
            self._step_frames(1)
        if not tex.has_ram_image():
            return False
        return bool(tex.store(img))

    def _step_frames(self, count):
        """`count` РЕАЛЬНЫХ кадров пайплайна.

        RenderPipeline делает всю пофреймовую работу в тасках
        (RP_UpdateManagers sort=10, RP_Plugin_BeforeRender sort=12,
        RP_UpdateInputsAndStages sort=18) — именно там обновляются матрицы
        камеры и продвигаются ping-pong индексы темпоральных стадий (TAA).
        igLoop (draw+flip) идёт ПОСЛЕ них, sort=50. Голый render_frame()
        выполняет только draw+flip и эти таски НЕ гоняет.

        Шаг идёт через frame_pump приложения, если он есть: это единственная
        точка, где крутится taskMgr, и она защищена от повторного входа.
        Иначе Qt-таймер и наш settle пересекаются, Panda ругается «Ignoring
        recursive poll() within another task» и ПРОПУСКАЕТ кадр.
        """
        pump = getattr(self.panda_app, "frame_pump", None)
        if pump is not None:
            pump.step(count)
            return
        tm = getattr(self.panda_app, "taskMgr", None)
        for _ in range(max(1, int(count))):
            if tm is not None:
                tm.step()
            else:
                self.panda_app.graphicsEngine.render_frame()

    def capture_scene_color(self, img, *, steps=2):
        """Цветной кадр сцены со ВСЕЙ постобработкой, БЕЗ чтения окна.

        Основной путь — цель FinalStage RenderPipeline: обычный offscreen-FBO,
        на который состояние окна не влияет (проверено при свёрнутом окне).
        Для пресета 'performance' (RenderPipeline выключен) FinalStage нет —
        тогда падаем на triggered-copy самого окна: оно всё ещё лучше
        get_screenshot (копия делается до flip'а и её отсутствие детектирует
        неактуальность), но требует видимого окна.
        """
        target = self._rp_final_target()
        if target is not None:
            try:
                buf = target.internal_buffer
                tex = self._ensure_triggered_tex(buf, "_final_tex")
                if tex is not None:
                    if self._read_triggered(buf, tex, img, settle=steps):
                        return True
                    print("[Render] FinalStage не отдал копию кадра")
            except Exception as exc:
                print(f"[Render] чтение FinalStage не удалось: {exc}")

        # --- запасной путь: окно (только без RenderPipeline) ---------------
        win = getattr(self.panda_app, "win", None)
        if win is None:
            return False
        tex = self._ensure_triggered_tex(win, "_win_tex")
        if tex is None:
            print("[Render] ВНИМАНИЕ: triggered-copy недоступен — читаю окно "
                  "напрямую, кадр может быть неактуален.")
            self._step_frames(steps)
            return win.getScreenshot(img)
        for attempt in range(3):
            if self._read_triggered(win, tex, img, settle=steps if not attempt else 1):
                return True
            print(f"[Render] окно не рисуется (копия не пришла), "
                  f"попытка {attempt + 1}/3")
        print("[Render] ОТКАЗ: актуальный кадр получить не удалось. Сэмпл "
              "пропущен намеренно — запись устаревшего кадра испортила бы "
              "датасет.")
        return False

    # Совместимость: старое имя. Весь код датасета должен звать
    # capture_scene_color — окно больше не является источником кадра.
    def _grab_window_screenshot(self, img, *, steps=2):
        return self.capture_scene_color(img, steps=steps)

    # ------------------------------------------------------------------
    # Offscreen-колорайзер карты глубины.
    #
    # DepthMapRenderer раскрашивает глубину картой-оверлеем на render2d, а
    # render2d рисуется ОКНОМ (мимо стадий RenderPipeline) — то есть снять её
    # можно было только чтением окна, со всеми его проблемами. Здесь тот же
    # самый узел-оверлей временно переносится в собственный offscreen-буфер с
    # ортокамерой и снимается оттуда. Переиспользуем ИМЕННО узел оверлея, а не
    # копию шейдера, — тогда все входы (near/far, gradientStart/End,
    # grayscale) гарантированно те же, что выставил DepthMapRenderer.
    # ------------------------------------------------------------------
    # Размер ФИКСИРОВАН и равен размеру depth-текстуры (DepthMapRenderer
    # создаёт её как 1920x1080). Привязывать буфер к размеру окна нельзя:
    # у свёрнутого окна размер бессмысленный, а при разворачивании он меняется
    # — и буфер пересоздавался бы (remove_window + make_output) прямо посреди
    # съёмки, что роняло процесс. Колорайзер лишь рисует полноэкранную карту,
    # сэмплящую depth-текстуру, поэтому размер окна ему не нужен вовсе.
    DEPTH_COLORIZER_SIZE = (1920, 1080)

    def _ensure_depth_colorizer(self):
        state = getattr(self, "_depth_colorizer", None)
        if state is not None:
            return state
        if getattr(self, "_depth_colorizer_failed", False):
            return None

        w, h = self.DEPTH_COLORIZER_SIZE
        app = self.panda_app
        try:
            fb = FrameBufferProperties()
            fb.set_rgba_bits(8, 8, 8, 8)
            fb.set_srgb_color(False)
            fb.set_depth_bits(0)
            fb.set_multisamples(0)

            buf = app.graphicsEngine.make_output(
                app.pipe, "depth_colorize_buffer", -20, fb,
                WindowProperties.size(w, h),
                GraphicsPipe.BF_refuse_window,
                app.win.get_gsg(), app.win)
            if buf is None:
                raise RuntimeError("make_output вернул None")

            buf.set_clear_color_active(True)
            buf.set_clear_color(LColor(0, 0, 0, 1))

            root = NodePath("depth_colorize_root")
            lens = OrthographicLens()
            lens.set_film_size(2, 2)
            lens.set_near_far(-10, 10)
            cam = Camera("depth_colorize_cam", lens)
            cam_np = root.attach_new_node(cam)
            cam_np.set_pos(0, -1, 0)
            cam_np.look_at(0, 0, 0)

            dr = buf.make_display_region(0, 1, 0, 1)
            dr.set_camera(cam_np)
            dr.set_clear_color_active(True)
            dr.set_clear_color(LColor(0, 0, 0, 1))

            buf.set_active(False)
            state = {"buffer": buf, "root": root, "cam": cam_np,
                     "size": (w, h)}
            self._depth_colorizer = state
            return state
        except Exception as exc:
            print(f"[Render] offscreen-колорайзер глубины недоступен: {exc}")
            self._depth_colorizer_failed = True
            return None

    def capture_depth_color(self, img):
        """Снять РАСКРАШЕННУЮ карту глубины из offscreen-буфера.

        Возвращает False, если снять не удалось — вызывающий решает, что
        делать (для датасета это отказ от сэмпла).
        """
        app = self.panda_app
        dr = getattr(app, "depth_renderer", None)
        if dr is None:
            return False
        overlay = getattr(dr, "overlay_node", None)
        if overlay is None or overlay.is_empty():
            return False

        # Карта глубины под ТЕКУЩУЮ позу камеры.
        try:
            dr.update_depth_texture()
        except Exception as exc:
            print(f"[Render] update_depth_texture не удался: {exc}")

        state = self._ensure_depth_colorizer()
        if state is None:
            return False

        buf = state["buffer"]
        tex = self._ensure_triggered_tex(buf, "_depth_color_tex")
        if tex is None:
            return False

        parent = overlay.get_parent()
        was_hidden = overlay.is_hidden()
        try:
            overlay.reparent_to(state["root"])
            overlay.show()
            overlay.set_pos(0, 0, 0)
            buf.set_active(True)
            ok = self._read_triggered(buf, tex, img, settle=1)
        finally:
            buf.set_active(False)
            overlay.reparent_to(parent)
            overlay.set_pos(0, 0, 0)
            if was_hidden:
                overlay.hide()
        return ok
    def _get_gemini_processor(self):
        """Ленивое создание процессора постобработки (провайдер из config).
        None, если недоступен (нет ключа/токена, отключён и т.п.)."""
        proc = getattr(self, "_gemini_processor", None)
        if proc is None:
            try:
                from src.rendering.gemini_postprocess import get_image_processor
                proc = get_image_processor()
                self._gemini_processor = proc
            except Exception as exc:
                print(f"[Postprocess] инициализация не удалась: {exc}")
                self._gemini_processor = False   # помечаем, чтобы не пытаться
                return None
        if proc is False:
            return None
        return proc if proc.available() else None

    def _get_lidar_scanner(self):
        """Сканер лидара; None, если трассировать нечем.

        Создаётся лениво и живёт на panda_app: BVH и разбор геометрии сцены
        кэшируются между кадрами, поэтому пересоздавать сканер на каждый
        сэмпл — значит выбрасывать этот кэш.
        """
        scanner = getattr(self.panda_app, "lidar_scanner", None)
        if scanner is False:
            return None
        if scanner is None:
            try:
                from src.rendering.lidar_scanner import LidarScanner
                scanner = LidarScanner(self.panda_app)
                if not scanner.available():
                    raise RuntimeError("нет ни Warp, ни Embree")
            except Exception as exc:
                print(f"[Lidar] сканер недоступен: {exc}")
                self.panda_app.lidar_scanner = False
                return None
            self.panda_app.lidar_scanner = scanner
        return scanner

    # Что именно кадр оставляет на диске. Раньше это было зашито в
    # dataset_type ("depth" -> цвет+глубина, "segmentation" -> цвет+маска),
    # и снять, скажем, одну маску без цветного кадра было нельзя. Теперь
    # набор файлов приходит отдельным параметром, а dataset_type остался
    # только как способ задать его по-старому (им пользуется cli.py).
    OUTPUT_KEYS = ("color", "depth", "segmentation", "lidar", "json")

    @staticmethod
    def resolve_outputs(outputs=None, dataset_type="depth", also_depth=False):
        """Нормализовать набор выходов; None => старое поведение по типу."""
        if outputs is None:
            resolved = {"color", "json"}
            if dataset_type == "segmentation":
                resolved.add("segmentation")
                if also_depth:
                    resolved.add("depth")
            else:
                resolved.add("depth")
            return resolved
        if isinstance(outputs, dict):
            outputs = [k for k, v in outputs.items() if v]
        return {str(k) for k in outputs
                if str(k) in RendererUtils.OUTPUT_KEYS}

    def save_single_render(self, output_dir="renders/single",
                           filename_prefix="single_render",
                           extra_metadata=None,
                           dataset_type="depth",
                           random_background=False,
                           gemini=False,
                           shadow_band=False,
                           also_depth=False,
                           cloth=False,
                           cloth_probability=0.8,
                           cloth_seed=None,
                           cloth_placement=None,
                           outputs=None,
                           depth_settings=None,
                           lidar_settings=None):
        """Обёртка: ткань живёт ровно один кадр.

        Полотно симулируется под ТЕКУЩУЮ сцену (кузов уже загружен, груз уже
        сгенерирован) и снимается после съёмки — иначе следующий сэмпл
        унаследует чужие складки. finally обязателен: в _save_single_render
        много ранних `return False`.

        cloth_probability < 1 оставляет часть кадров без ткани — датасету
        нужны и негативные примеры.
        """
        cloth_meta = None
        if cloth:
            sim = getattr(self.panda_app, "cloth_simulator", None)
            if sim is None:
                print("[Cloth] cloth_simulator недоступен — кадр без ткани")
            elif random.random() <= cloth_probability:
                try:
                    if sim.spawn_random(seed=cloth_seed,
                                        placement=cloth_placement) is not None:
                        cloth_meta = sim.last_params
                except Exception as exc:
                    print(f"[Cloth] генерация не удалась: {exc}")
                    sim.clear()

        if cloth_meta is not None:
            extra_metadata = dict(extra_metadata or {})
            extra_metadata["cloth"] = cloth_meta

        try:
            return self._save_single_render(
                output_dir=output_dir,
                filename_prefix=filename_prefix,
                extra_metadata=extra_metadata,
                dataset_type=dataset_type,
                random_background=random_background,
                gemini=gemini,
                shadow_band=shadow_band,
                also_depth=also_depth,
                outputs=outputs,
                depth_settings=depth_settings,
                lidar_settings=lidar_settings,
            )
        finally:
            sim = getattr(self.panda_app, "cloth_simulator", None)
            if sim is not None:
                sim.clear()

    def _save_single_render(self, output_dir="renders/single",
                            filename_prefix="single_render",
                            extra_metadata=None,
                            dataset_type="depth",
                            random_background=False,
                            gemini=False,
                            shadow_band=False,
                            also_depth=False,
                            outputs=None,
                            depth_settings=None,
                            lidar_settings=None):
        # dataset_type / also_depth — СТАРЫЙ способ задать набор файлов; им
        # ещё пользуется cli.py. Новый код передаёт `outputs` напрямую, см.
        # resolve_outputs.
        #
        # depth_settings: параметры карты глубины именно этого прогона
        # (диапазон, ч/б или радуга). Применяются на время съёмки и
        # возвращаются обратно.
        #
        # random_background: на ОБЫЧНОМ цветном рендере (после дисторсии) фон
        # сцены/неба заменяется случайной картинкой из assets/backgrounds;
        # передний план (кузов + груз) остаётся. Карта глубины и маска
        # сегментации при этом НЕ меняются — маска используется только чтобы
        # вырезать передний план.
        outputs = self.resolve_outputs(outputs, dataset_type, also_depth)
        want_color = "color" in outputs
        want_depth = "depth" in outputs
        want_seg = "segmentation" in outputs
        want_lidar = "lidar" in outputs

        # Параметры глубины на время съёмки — свои (диапазон, ч/б или радуга).
        # Снимок прежних значений возвращаем в finally: прогон датасета не
        # должен молча переписать то, что пользователь выкрутил для оверлея.
        depth_prev = None
        dr_settings = self.panda_app.depth_renderer
        if want_depth and depth_settings and hasattr(dr_settings,
                                                     "apply_settings"):
            try:
                depth_prev = dr_settings.capture_settings()
                dr_settings.apply_settings(depth_settings)
            except Exception as exc:
                print(f"[Render] параметры глубины не применены: {exc}")
                depth_prev = None
        try:
            return self._capture_and_write(
                output_dir=output_dir,
                filename_prefix=filename_prefix,
                extra_metadata=extra_metadata,
                dataset_type=dataset_type,
                random_background=random_background,
                gemini=gemini,
                shadow_band=shadow_band,
                outputs=outputs,
                want_color=want_color,
                want_depth=want_depth,
                want_seg=want_seg,
                want_lidar=want_lidar,
                depth_settings=depth_settings,
                lidar_settings=lidar_settings,
            )
        finally:
            if depth_prev is not None:
                try:
                    dr_settings.apply_settings(depth_prev)
                except Exception as exc:
                    print(f"[Render] параметры глубины не возвращены: {exc}")

    def _capture_and_write(self, *, output_dir, filename_prefix,
                           extra_metadata, dataset_type, random_background,
                           gemini, shadow_band, outputs, want_color,
                           want_depth, want_seg, want_lidar=False,
                           depth_settings=None, lidar_settings=None):
        """Снять кадры (цвет / глубина / маска) и отдать их на запись."""
        lens = self.panda_app.cam.node().getLens()
        if isinstance(lens, PerspectiveLens):
            fov = lens.getFov()
            camera_fov_x = fov[0]
            camera_fov_y = fov[1]
        else:
            camera_fov_x = camera_fov_y = None

        # Свежесгенерированный меш наполнения (final_model) и его 8K-текстуры
        # грузятся на GPU ЛЕНИВО — RenderPipeline подгружает их в течение
        # нескольких кадров draw'а. Если цветной снимок снять раньше, чем
        # ресурсы доехали до GPU, пайплайн рисует ПУСТОЙ кузов (без груза).
        # Камера сегментации к этому иммунна: у неё свой заранее собранный
        # плоский шейдер и НЕТ текстур — поэтому маска всегда корректна, а
        # цветной кадр иногда пустой (и потом обрезается по правильной маске).
        # Форсируем подготовку ресурсов геометрии на GSG ДО снятия кадра —
        # prepare_scene ставит в очередь ВСЕ текстуры/шейдеры/вершины разом
        # (без пофреймового троттлинга), так что последующие settle-кадры
        # гарантированно их дозагружают. Идемпотентно и дёшево, если ресурс
        # уже на GPU.
        try:
            win = getattr(self.panda_app, "win", None)
            gsg = win.get_gsg() if win is not None else None
            final_model = getattr(self.panda_app, "final_model", None)
            if (gsg is not None and final_model is not None
                    and not final_model.is_empty()):
                final_model.prepare_scene(gsg)
        except Exception as exc:
            print(f"[Render] prepare_scene(final_model) failed: {exc}")

        # Скрываем depth overlay и даём пайплайну устаканиться РЕАЛЬНЫМИ
        # кадрами (taskMgr.step), чтобы PSSM перестроил каскадные тени под
        # новую позу камеры/солнца, а motion blur / TAA сошлись. Иначе на
        # кадр иногда вылезает огромная ложная тень на полкадра.
        self.settle_render(frames=30)
        self.panda_app.depth_renderer.set_overlay_visibility(False)
        self.settle_render(frames=30)

        self.settle_render(frames=30)

        # Цветной кадр — из offscreen-цели FinalStage, окно НЕ читается.
        # Именно здесь раньше в датасет попадал чужой кадр: чтение встроенного
        # дочернего HWND возвращало пиксели рабочего стола (посторонние окна)
        # либо кадр предыдущего сэмпла, который затем вырезался по АКТУАЛЬНОЙ
        # маске. False => сэмпл пропускается целиком.
        img = None
        self.settle_render(frames=30)
        if want_color:
            # Кадр читается ТОЛЬКО когда цветной файл нужен: без него вся
            # цветная ветка (дисторсия, замена фона, теневая полоса) — это
            # работа впустую.
            img = PNMImage()
            if not self.capture_scene_color(img):
                self.panda_app.depth_renderer.set_overlay_visibility(False)
                return False

        # Gemini доступен? (нужно знать заранее — от этого зависит, снимать ли
        # маску сегментации). Недоступность => тихий откат.
        gemini_processor = self._get_gemini_processor() if gemini else None

        # Маска сегментации для вырезания переднего плана. Нужна при замене
        # фона, Gemini-постобработке и теневой полосе. Снимается отдельно —
        # это один дешёвый GPU-кадр.
        # OpenAI редактирует весь кадр и маску не использует (тень тоже через
        # промпт) — снимаем маску только для замены фона / матирования /
        # локальной теневой полосы у НЕ-OpenAI провайдеров.
        openai_active = (
            gemini_processor is not None
            and hasattr(gemini_processor, "edit_whole"))
        gemini_needs_mask = gemini_processor is not None and not openai_active
        need_mask = random_background or gemini_needs_mask or (
            shadow_band and not openai_active)
        seg_mask_raw = None
        if need_mask or want_seg:
            seg_mask_raw = self.panda_app.segmentation_renderer.capture()
            if seg_mask_raw is None:
                print("[Render] seg mask capture failed; "
                      "сохраняю без замены фона/тени.")

        # Маска сегментации как ВЫХОДНОЙ файл. Рендерится в отдельный
        # offscreen-буфер (плоские цвета, без постобработки), поэтому тот же
        # захват годится и для выреза переднего плана.
        seg_img = None
        if want_seg:
            if seg_mask_raw is None:
                print("[Render] segmentation capture failed.")
                return False
            seg_img = PNMImage(seg_mask_raw)

        # Карта глубины. capture_depth_color рисует узел-оверлей глубины в
        # СВОЙ offscreen-буфер, поэтому окно не читается и его состояние
        # (свёрнуто/перекрыто/встроено в Qt) ни на что не влияет.
        depth_img = None
        if want_depth:
            dr = self.panda_app.depth_renderer
            # Палитра карты: явные настройки датасета уже применены выше;
            # иначе — старое поведение (ч/б только для датасета).
            forced_gray = None
            if depth_settings is None:
                is_dataset = bool(
                    extra_metadata
                    and extra_metadata.get("render_type") == "dataset")
                if is_dataset and hasattr(dr, "set_grayscale"):
                    forced_gray = bool(getattr(dr, "grayscale", False))
                    dr.set_grayscale(True)

            depth_img = PNMImage()
            ok_depth = self.capture_depth_color(depth_img)

            if forced_gray is not None:
                dr.set_grayscale(forced_gray)
            if not ok_depth:
                print("[Render] depth capture failed.")
                if want_color or want_seg:
                    depth_img = None
                else:
                    return False

        # Облако точек лидара. Оно НЕ растровое и через дисторсию/кроп не
        # проходит, поэтому снимается отдельно от кадров — но с той же позы
        # камеры и по той же сцене (ткань уже висит, груз уже сгенерирован),
        # так что кадр и облако описывают ровно одно состояние.
        lidar_scan = None
        if want_lidar:
            scanner = self._get_lidar_scanner()
            if scanner is not None:
                try:
                    lidar_scan = scanner.scan(lidar_settings)
                except Exception as exc:
                    print(f"[Lidar] съёмка не удалась: {exc}")
            if lidar_scan is None and not (want_color or want_depth
                                           or want_seg):
                # Кроме облака ничего не просили — сохранять нечего.
                return False

        if img is not None:
            img = self.stretch_to_1920x1080(img)
        if seg_img is not None:
            seg_img = self.stretch_to_1920x1080(seg_img, nearest=True)
            seg_img = self.fix_alpha_to_opaque(seg_img)
        if depth_img is not None:
            depth_img = self.stretch_to_1920x1080(depth_img, nearest=False)
            depth_img = self.fix_alpha_to_opaque(depth_img)

        # Маску под вырезание приводим к 1920x1080 тем же ближайшим соседом
        # (как img/depth), чтобы геометрия совпала.
        seg_mask_1080 = None
        bg_path = None
        if seg_mask_raw is not None:
            seg_mask_1080 = self.stretch_to_1920x1080(seg_mask_raw, nearest=True)
            # bg_path нужен только для «случайного фона из файлов». При Gemini
            # фон генерируется процессором, файл не нужен. Маска остаётся
            # (её используют Gemini-матирование и теневая полоса).
            if random_background and gemini_processor is None:
                bg_path = self._pick_random_background()
                if bg_path is None:
                    print("[Render] assets/backgrounds пуста — "
                          "замена фона пропущена.")

        # Клиентский пересчёт объёма реально сгенерированного меша
        # наполнения (в старой версии писался как actual_volume).
        # Берём final_model — туда perlin_mesh_generator / mesh_reconstruction
        # кладут текущую горку наполнителя.
        actual_volume = None
        calc = getattr(self.panda_app, 'calculate_mesh_volume', None)
        final_model = getattr(self.panda_app, 'final_model', None)
        if callable(calc) and final_model is not None:
            try:
                actual_volume = float(calc(final_model))
            except Exception as exc:
                print(f"[Render] actual_volume calc failed: {exc}")

        metadata = {
            "render_type": "single",
            "camera_position": {
                "x": float(self.panda_app.camera.getX()),
                "y": float(self.panda_app.camera.getY()),
                "z": float(self.panda_app.camera.getZ()),
            },
            "camera_rotation": {
                "h": float(self.panda_app.camera.getH()),
                "p": float(self.panda_app.camera.getP()),
                "r": float(self.panda_app.camera.getR()),
            },
            "model_set": (
                self.panda_app.current_model_set
                if hasattr(self.panda_app, 'current_model_set') else None
            ),
            "target_volume": getattr(self.panda_app, 'Target_Volume', None),
            "actual_volume": actual_volume,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        output_path = self._process_render_image(
            img,
            camera_fov_x=camera_fov_x,
            camera_fov_y=camera_fov_y,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            metadata=metadata,
            dataset_type=dataset_type,
            seg_mask=seg_mask_1080,
            bg_path=bg_path,
            gemini_processor=gemini_processor,
            shadow_band=shadow_band,
            depth_img=depth_img,
            seg_img=seg_img,
            outputs=outputs,
            lidar_scan=lidar_scan,
            lidar_settings=lidar_settings,
        )

        return True
    
    def save_dataset_render(self):
        original_pos = self.panda_app.camera.getPos()
        original_hpr = self.panda_app.camera.getHpr()
        original_view = self.panda_app.current_view
        original_target_volume = self.panda_app.Target_Volume
        
        if not self.panda_app.current_model_set:
            return False
        
        if not all([hasattr(self.panda_app, 'current_other_path'), 
                   hasattr(self.panda_app, 'current_cuzov_path'), 
                   hasattr(self.panda_app, 'current_napolnitel_path')]):
            return False
        
        fixed_pos = (8.599995136260986, 6.0011109376791865e-05, 21.70002269744873)
        fixed_hpr = (89.99999237060547, -66.110355377197266, 0.0)
        fixed_fov_x = 48.0
        fixed_fov_y = 26.14091682434082
        
        volumes = [0.5 + i * 0.5 for i in range(99)] 
        passes_per_volume = 4
        
        total_renders = len(volumes) * passes_per_volume
        current_render = 0
        
        for i, volume in enumerate(volumes):
            for pass_num in range(passes_per_volume):
                current_render += 1
                
                self.panda_app.Target_Volume = volume
                
                self.panda_app.clear_scene()
                self.panda_app.load_gltf_model(self.panda_app.current_other_path)
                self.panda_app.load_gltf_model(self.panda_app.current_cuzov_path)
                self.panda_app.load_gltf_model(self.panda_app.current_napolnitel_path)
                
                self.panda_app.create_ground_plane()
                if hasattr(self.panda_app, 'current_ground_plane_z'):
                    self.panda_app.ground_plane.setPos(0, 0, self.panda_app.current_ground_plane_z)
                
                success_aabb = self.panda_app.perform_AABB_plane()
                if not success_aabb:
                    continue
                
                if not hasattr(self.panda_app, 'Perlin_Seed'):
                    self.panda_app.Perlin_Seed = random.randint(0, 10000000)
                else:
                    self.panda_app.Perlin_Seed = random.randint(0, 10000000) + pass_num * 1000000 + i * 100000000
                
                success_perlin = self.panda_app.perlin_generator.generate_perlin_mesh_from_csg()
                if not success_perlin:
                    continue
                
                time.sleep(1.0)
                
                self.panda_app.camera.setPos(*fixed_pos)
                self.panda_app.camera.setHpr(*fixed_hpr)
                
                lens = self.panda_app.cam.node().getLens()
                if isinstance(lens, PerspectiveLens):
                    lens.setFov(fixed_fov_x, fixed_fov_y)
                
                for _ in range(120):  
                    self.panda_app.taskMgr.step()
                    time.sleep(0.01)
                
                current_pos = self.panda_app.camera.getPos()
                current_hpr = self.panda_app.camera.getHpr()
                
                img = PNMImage()
                # Тот же проверяемый путь, что и в _save_single_render: голый
                # win.getScreenshot() молча отдавал кадр предыдущего сэмпла,
                # когда окно было свёрнуто/перекрыто.
                if not self._grab_window_screenshot(img, steps=2):
                    print(f"[Dataset] сэмпл volume={volume:.1f} "
                          f"pass={pass_num} пропущен: кадр неактуален.")
                    continue
                
                output_path = self._process_render_image(
                    img,
                    camera_fov_x=fixed_fov_x,
                    camera_fov_y=fixed_fov_y,
                    output_dir="renders/datasets_metric_",
                    filename_prefix=f"render_volume_{volume:.1f}_pass_{pass_num:02d}",
                    metadata={
                        "render_type": "dataset",
                        "target_volume": volume,
                        "pass_number": pass_num,
                        "volume_index": i,
                        "perlin_seed": self.panda_app.Perlin_Seed,
                        "model_set": self.panda_app.current_model_set,
                        "camera_position": {
                            "x": float(current_pos.x),
                            "y": float(current_pos.y),
                            "z": float(current_pos.z)
                        },
                        "camera_rotation": {
                            "h": float(current_hpr.x),
                            "p": float(current_hpr.y),
                            "r": float(current_hpr.z)
                        }
                    }
                )
        
        self.panda_app.Target_Volume = original_target_volume
        self.panda_app.camera.setPos(original_pos)
        self.panda_app.camera.setHpr(original_hpr)
        self.panda_app.current_view = original_view
        
        return True