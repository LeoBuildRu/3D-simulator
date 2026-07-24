#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
higgsfield_seedream_cli.py
==========================
Пакетная img2img-обработка датасета моделью Seedream 5.0 lite через
ОФИЦИАЛЬНЫЙ Higgsfield CLI (@higgsfield/cli). Никакого браузера, Playwright,
реверса и captcha — это санкционированный программный канал.

Проверено вручную (2026-07-08):
  higgsfield generate create seedream_v5_lite --prompt "..." \
      --image-references <локальный.png> --aspect_ratio 16:9 --quality high --wait --json
  -> локальный файл авто-загружается; результат 4096x2304 PNG; status completed;
     JSON: [{ "result_url": "...", "status": "completed", ... }].
  Стоимость: ~1 кредит за кадр.

Логика:
  * находит все НЕ-маски в папке (исключая *_seg/_depth/_mask/_ai);
  * ПРОМПТ ВЫБИРАЕТСЯ ПО SEG-КАРТЕ: считает долю ярко-оранжевых (255,128,0)
    пикселей ткани в <кадр>_seg.png и берёт один из трёх вшитых промптов —
      < 2%  -> "none"    : ткани нет, явный запрет её дорисовывать;
      2..70% -> "partial" : ткань есть частично, границу не двигать;
      > 70% -> "full"    : ткань во весь кадр, запрет дорисовывать кузов/фон.
    Без этого модель дорисовывала ткань там, где её нет (ломая GT), и
    ужимала ткань, чтобы впихнуть в кадр самосвал и фон, когда ткань
    должна занимать весь кадр. Пороги — CLOTH_MIN/FULL_FRACTION.
  * запускает до MAX_CONCURRENCY=8 CLI-процессов параллельно (каждый с --wait);
  * скачивает result_url и сохраняет рядом как <оригинал>_ai.png;
  * резюме: готовые кадры пропускает;
  * ЗАЩИТА ОТ СЛИВА КРЕДИТОВ: за один прогон обрабатывает не больше --max
    кадров (по умолчанию 10). Для большего объёма подними --max осознанно.

Требования: установленный Node.js и @higgsfield/cli, выполненные
`higgsfield auth login` и `higgsfield workspace set <id>` (делается один раз).

ВАЖНО (сеть): загрузка картинки в Higgsfield с этой машины работает ТОЛЬКО
через локальный прокси. Без HTTPS_PROXY CLI падает с кодом 3
"request failed (no response received)" на КАЖДОМ кадре. Перед запуском:
  $env:HTTPS_PROXY='http://127.0.0.1:10809'; $env:HTTP_PROXY=$env:HTTPS_PROXY
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import requests

# ─────────────────────────── КОНФИГ ───────────────────────────
INPUT_DIR = Path(r"D:\IQoko\dataset_segmentation_random")
OUTPUT_SUFFIX = "_ai"

MODEL = "seedream_v5_lite"       # Seedream 5.0 lite. ~1 кредит/кадр (nano_banana был ~2).
ASPECT = "16:9"                  # исходники 1920x1080 == ровно 16:9 -> без искажений
QUALITY = "high"                 # seedream_v5_lite: basic | high (у seedream нет --resolution)
MAX_CONCURRENCY = 6
WAIT_TIMEOUT = "20m"

DEFAULT_MAX_PER_RUN = 10         # защита от случайного слива кредитов

# Пути к Node и CLI (найдены на этой машине). При необходимости поправь.
NODE = r"C:\Program Files\nodejs\node.exe"
CLI_JS = r"C:\Users\xmake\AppData\Roaming\npm\node_modules\@higgsfield\cli\bin\higgsfield.js"

# ─── классификация ткани по seg-карте ───
# Палитра seg-карт плоская, без сглаживания (nearest-neighbour), поэтому
# сравнение по точному RGB безопасно. Ткань = ярко-оранжевый.
CLOTH_RGB = (255, 128, 0)
SEG_SUFFIX = "_seg.png"
CLOTH_MIN_FRACTION = 0.02   # < 2% кадра  -> считаем, что ткани нет
CLOTH_FULL_FRACTION = 0.70  # > 70% кадра -> ткань закрывает весь кадр

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_SUFFIXES = ("_seg", "_depth", "_mask", "_ai", "_ai_fix", "_seedream", "_seedream5lite")
DELAY_MIN_S, DELAY_MAX_S = 0.3, 1.0

# CLI отдаёт код 3 на транспортных сбоях ("request failed (no response received)") —
# запрос не дошёл/ответ не вернулся. Кредит при этом не списывается, ретрай безопасен.
CREATE_RETRIES = 3
RETRY_BACKOFF_S = 5.0


# ─────────────────────────── ПРОМПТЫ ───────────────────────────
# Три варианта по доле ткани в seg-карте. Общая часть (геометрия, стиль,
# фон) продублирована в каждом, чтобы промпт читался моделью как единый
# связный текст, а не как база + патч.

_COMMON_HEAD = (
    "Edit this truck on-board camera photo in place. Keep everything pixel-aligned with the "
    "input image: same camera, same framing, no shift, no zoom, no rotation, no crop. "
)
_COMMON_STYLE = (
    "Style: cheap analog CCTV camera look, muted colors but not black-and-white, low dynamic "
    "range with some overexposed and underexposed areas, direct sunlight at a medium-high angle."
)
_BODY_AND_LOAD = (
    "Truck body: weathered, battered, dented, dirty steel with some shiny worn metal, seam and "
    "reinforcing lines along the full length of the body, a few cables hanging off the side; do "
    "not change its shape, thickness or wall height. Load: re-texture as crushed stone rubble "
    "with pieces of concrete and debris — same heap shape, same volume, same surface relief, "
    "exactly the same place in the body. "
)
_BACKGROUND = (
    "Background: heavy-industry transfer yard — factories, piles of materials, other dump trucks, "
    "rail tracks, railguards, asphalt road with metal guardrails parallel to truck body. "
)
# Один материал ткани выбирается СЛУЧАЙНО в Python на кадр и жёстко фиксируется
# в промпте — иначе модель рисовала в одном кадре сразу несколько цветов/тканей.
CLOTH_MATERIALS = (
    "dirty off-white canvas tarpaulin",
    "coarse brown burlap like a potato sack",
    "dark green polyethylene construction tarp with hemmed edges and grommets",
)


def build_cloth_material() -> str:
    material = random.choice(CLOTH_MATERIALS)
    return (
        "Give it visible weave, dust and dirt streaks, worn frayed edges, stains, slight sag, "
        "tie-down ropes, and shading matching the scene light. The cloth is ONE single uniform "
        f"material of ONE colour across its whole surface: {material}. Do NOT mix fabrics or "
        "colours and do NOT show more than one kind of cloth — every part of the covering is the "
        "same fabric and the same colour, only its folds and dirt vary. "
    )

PROMPT_NONE = (
    _COMMON_HEAD
    + "The dump truck body and the load inside it must keep exactly the same position, outline, "
    "silhouette, shape and volume in the frame. Change ONLY surface textures and the background. "
    + _BODY_AND_LOAD
    + "IMPORTANT: there is NO cloth, tarpaulin, sheet or cover of any kind in this image, and the "
    "load is completely uncovered. Do NOT add a cloth, tarp, sheet, netting or any covering "
    "anywhere in the frame. The rubble must stay fully exposed and visible across its whole "
    "surface — every part of the heap that is visible now must remain visible. Do not drape, "
    "wrap or partially cover the load or the body with anything. "
    + _BACKGROUND
    + _COMMON_STYLE
    + " Shadows cast distinctly across the exposed load."
)

# Порог внутри "partial": ниже — ткань это МАЛОЕ пятно (частая ошибка модели —
# раздувает её); выше — ткань покрывает БОЛЬШУЮ часть (частая ошибка — «вскрывает»
# кузов/груз под тканью, уменьшая её площадь). Направление анти-дрейфа зависит
# от того, в какую сторону модель обычно ошибается при данной площади.
CLOTH_PARTIAL_SPLIT = 0.20


def build_partial_prompt(frac: float | None) -> str:
    """Промпт для частичной ткани, зависящий от её площади.

    Модель не видит seg-карту, поэтому а) сообщаем ей примерную долю кадра под
    тканью как якорь площади и б) усиливаем запрет именно того дрейфа, который
    доминирует при этой площади (малая ткань -> растёт; ~половина -> ужимается
    дорисованным кузовом).
    """
    if frac is None:
        anchor = (
            "A cloth covers PART of the frame and the rest is exposed load and truck body. "
            "Keep the covered area and the uncovered area exactly as they are now. "
        )
        directional = (
            "Do not grow, spread or enlarge the cloth over more of the load, and do not shrink it "
            "or reveal any truck body, load or rubble in the area it currently covers. "
        )
    else:
        pct = max(1, round(frac * 100))
        anchor = (
            f"Approximately {pct}% of the frame is covered by the cloth; the remaining "
            f"{100 - pct}% is exposed load and truck body. Keep this proportion exactly — the "
            "cloth must cover the same area, no larger and no smaller. "
        )
        if frac < CLOTH_PARTIAL_SPLIT:
            directional = (
                f"The cloth is a SMALL patch covering only about {pct}% of the frame; most of the "
                "load is bare, exposed crushed-stone rubble. Do NOT enlarge, spread, extend, drape "
                "or unfold the cloth over any more of the load — it must stay this small and its "
                "edges must not creep outward. Every part of the rubble that is bare now stays "
                "bare. "
            )
        else:
            directional = (
                f"The cloth covers a LARGE part of the frame, about {pct}%. Do NOT reveal, uncover, "
                "open up or draw any truck body, load, rubble or interior in the area the cloth "
                "currently covers — whatever is under the cloth stays completely hidden. Do NOT "
                "shrink the cloth, pull it back, fold it down or expose the load; its edges must "
                "not move inward. "
            )

    return (
        _COMMON_HEAD
        + "The dump truck body, the load inside it, and the cloth present must keep exactly the "
        "same position, outline, silhouette, shape, folds and volume in the frame. Change ONLY "
        "surface textures and the background. "
        + _BODY_AND_LOAD
        + "Cloth: a cloth covers PART of the frame — it is already there, with a definite, sharp "
        "edge where it stops. Re-texture it as a realistic heavy covering with its exact edges, "
        "drape and every fold and wrinkle kept in place — do not smooth, re-drape or move it. "
        + anchor
        + directional
        + "The boundary line between covered and uncovered must stay exactly where it is now, to "
        "the pixel. Whatever the cloth hides stays hidden; whatever is uncovered stays uncovered. "
        + build_cloth_material()
        + "Only the material changes, never the pixels it occupies. "
        + _BACKGROUND
        + _COMMON_STYLE
        + " Distinct shadow lines across the load and the folds of the cloth."
    )

def build_full_prompt() -> str:
    return (
        _COMMON_HEAD
        + "This is an EXTREME CLOSE-UP of a heavy cloth covering that fills essentially the whole "
        "frame, edge to edge. The camera is right up against the cloth. This is correct and "
        "intentional — it is not a mistake, not a crop error, and nothing is missing. "
        "Do NOT add a truck, do not draw a dump truck body or a load, do not add a horizon, sky, "
        "yard, buildings or any background scene, and do NOT shrink, zoom out or push back the "
        "cloth to fit a scene around it. There is no room for a background and none should be "
        "invented: every pixel that is cloth now must still be cloth. "
        "Re-texture the cloth as a realistic heavy covering, keeping every fold, wrinkle, crease "
        "and sag exactly where it is, at exactly the same scale — do not smooth, re-drape, re-fold "
        "or move anything. "
        + build_cloth_material()
        + "At this distance the weave, individual threads, dust, grit and stains are clearly "
        "resolved. If a small strip of truck body or surroundings is visible at the very edge of "
        "the frame, keep it exactly in place at its current size — weathered dirty steel, and "
        "nothing more. "
        + _COMMON_STYLE
        + " Raking light picks out the relief of the folds."
    )


PROMPTS = {
    "none": PROMPT_NONE,
    "partial": None,          # None = строить динамически (build_partial_prompt): площадь + случайный материал
    "full": None,             # None = строить динамически (build_full_prompt): случайный материал
    # переопределяются статическими none/partial/full.txt через --prompt-dir
}


def resolve_prompt(prompts: dict, kind: str, frac: float | None) -> str:
    """Готовый текст промпта. partial/full без файла-переопределения строятся динамически
    (площадь ткани + случайный из 3 материал, фиксируемый на кадр)."""
    text = prompts.get(kind)
    if text is not None:  # статическое переопределение из --prompt-dir
        return text
    if kind == "full":
        return build_full_prompt()
    return build_partial_prompt(frac)

# Человекочитаемые метки для лога
LABELS = {"none": "без ткани", "partial": "ткань частично", "full": "ткань во весь кадр"}


# ─────────────────────────── УТИЛИТЫ ───────────────────────────
def is_source_frame(p: Path) -> bool:
    if p.suffix.lower() not in IMG_EXT:
        return False
    stem = p.stem.lower()
    return not any(stem.endswith(s) for s in SKIP_SUFFIXES)


def output_path(src: Path) -> Path:
    return src.with_name(src.stem + OUTPUT_SUFFIX + ".png")


def seg_path(src: Path) -> Path:
    return src.with_name(src.stem + SEG_SUFFIX)


def cloth_fraction(seg: Path) -> float:
    """Доля пикселей ткани (CLOTH_RGB) в seg-карте, 0..1."""
    import numpy as np
    from PIL import Image

    arr = np.array(Image.open(seg).convert("RGB"))
    return float(np.all(arr == CLOTH_RGB, axis=-1).mean())


def classify_frame(src: Path):
    """-> (ключ промпта, доля ткани). Без seg-карты — (None, None)."""
    seg = seg_path(src)
    if not seg.is_file():
        return None, None
    frac = cloth_fraction(seg)
    if frac < CLOTH_MIN_FRACTION:
        return "none", frac
    if frac > CLOTH_FULL_FRACTION:
        return "full", frac
    return "partial", frac


def discover_frames(folder: Path):
    frames = sorted(p for p in folder.iterdir() if p.is_file() and is_source_frame(p))
    todo = [p for p in frames if not output_path(p).exists()]
    return frames, todo


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Batch Seedream img2img через Higgsfield CLI.")
    ap.add_argument("files", nargs="*", type=Path,
                    help="Явные картинки (для теста). Если заданы — папка не сканируется.")
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    ap.add_argument("--prompt-dir", type=Path, default=None,
                    help="Папка с переопределением промптов: none.txt / partial.txt / full.txt. "
                         "Чего нет — берётся вшитый промпт.")
    ap.add_argument("--max", type=int, default=DEFAULT_MAX_PER_RUN,
                    help=f"МАКС кадров за прогон (защита от слива кредитов, по умолч. {DEFAULT_MAX_PER_RUN}). "
                         f"~1 кредит/кадр. Подними осознанно.")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--quality", default=QUALITY, choices=["basic", "high"])
    ap.add_argument("--aspect", default=ASPECT)
    ap.add_argument("--force", action="store_true", help="Переобрабатывать даже если *_ai.png есть.")
    ap.add_argument("--list", action="store_true", help="Показать план и выйти (без генераций).")
    ap.add_argument("--review", action="store_true",
                    help="После генерации открыть UI приёмки (review_ai.py) по этой папке.")
    ap.add_argument("--node", default=NODE)
    ap.add_argument("--cli", default=CLI_JS)
    return ap.parse_args(argv)


def resolve_jobs(args):
    if args.files:
        frames = [f if f.is_absolute() else (Path.cwd() / f) for f in args.files]
        missing = [f for f in frames if not f.is_file()]
        if missing:
            raise SystemExit("Не найдены файлы: " + ", ".join(str(m) for m in missing))
        base_dir = frames[0].parent
        todo = frames if args.force else [f for f in frames if not output_path(f).exists()]
    else:
        base_dir = args.input_dir
        if not base_dir.is_dir():
            raise SystemExit(f"Папка не найдена: {base_dir}")
        all_frames, auto_todo = discover_frames(base_dir)
        todo = all_frames if args.force else auto_todo
        print(f"Всего исходников: {len(all_frames)} | к обработке: {len(todo)} "
              f"| уже готовы: {len(all_frames) - len(todo)}")

    return todo, load_prompts(args.prompt_dir)


def load_prompts(prompt_dir: Path | None) -> dict:
    """Вшитые промпты, поверх — файлы none/partial/full.txt из --prompt-dir."""
    prompts = dict(PROMPTS)
    if not prompt_dir:
        return prompts
    if not prompt_dir.is_dir():
        raise SystemExit(f"Папка промптов не найдена: {prompt_dir}")
    for key in prompts:
        f = prompt_dir / f"{key}.txt"
        if f.is_file():
            prompts[key] = f.read_text(encoding="utf-8").strip()
            print(f"  промпт '{key}' переопределён из {f.name}")
    return prompts


def download_result(url: str, dest: Path) -> bool:
    import time
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


async def process_one(args, src: Path, prompts: dict, sem: asyncio.Semaphore, stats: dict):
    async with sem:
        await asyncio.sleep(random.uniform(DELAY_MIN_S, DELAY_MAX_S))

        kind, frac = classify_frame(src)
        if kind is None:
            # без seg-карты классифицировать нечем — берём осторожный вариант
            print(f"  ~ {src.name}: нет {SEG_SUFFIX}, беру промпт 'partial'")
            kind, tag = "partial", "seg?"
        else:
            tag = f"{LABELS[kind]} {frac * 100:.0f}%"
        prompt = resolve_prompt(prompts, kind, frac)

        cmd = [
            args.node, args.cli, "generate", "create", MODEL,
            "--prompt", prompt,
            "--image-references", str(src),
            "--aspect_ratio", args.aspect,
            "--quality", args.quality,
            "--wait", "--wait-timeout", WAIT_TIMEOUT,
            "--json",
        ]
        print(f"[создаю] {src.name}  [{tag}]")
        out = None
        for attempt in range(1, CREATE_RETRIES + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await proc.communicate()
            except Exception as e:
                print(f"  ! {src.name}: не смог запустить CLI: {e}")
                stats["failed"] += 1
                return

            if proc.returncode == 0:
                break

            msg = (err or b"").decode("utf-8", "replace").strip()
            # код 3 = транспортный сбой, лечится повтором; остальное — фатально
            if proc.returncode == 3 and attempt < CREATE_RETRIES:
                pause = RETRY_BACKOFF_S * attempt
                print(f"  ~ {src.name}: сбой связи (код 3), попытка {attempt}/{CREATE_RETRIES}, "
                      f"повтор через {pause:.0f}s. {msg}")
                await asyncio.sleep(pause)
                continue

            print(f"  ! {src.name}: CLI вернул код {proc.returncode}. {msg}")
            stats["failed"] += 1
            return

        # разбор JSON (ожидаем список job'ов)
        text = (out or b"").decode("utf-8", "replace").strip()
        url = None
        try:
            data = json.loads(text)
            jobs = data if isinstance(data, list) else [data]
            for j in jobs:
                if isinstance(j, dict) and j.get("result_url"):
                    url = j["result_url"]
                    break
        except Exception:
            m = re.search(r'"result_url"\s*:\s*"([^"]+)"', text)
            url = m.group(1) if m else None

        if not url:
            print(f"  ! {src.name}: не нашёл result_url в ответе CLI")
            stats["failed"] += 1
            return

        dest = output_path(src)
        if download_result(url, dest):
            stats["done"] += 1
            stats["created"].append(dest)      # только эти уйдут в UI приёмки
            print(f"  [готово] {dest.name}  ({stats['done']} ок, {stats['failed']} ошибок)")
        else:
            stats["failed"] += 1
            print(f"  ! {src.name}: не смог скачать результат")


async def main(args):
    todo, prompts = resolve_jobs(args)

    if args.list:
        capped = todo[:args.max]
        print(f"К обработке за этот прогон: {len(capped)} (из {len(todo)}, лимит --max={args.max})")
        counts = {"none": 0, "partial": 0, "full": 0, "seg?": 0}
        for p in capped:
            kind, frac = classify_frame(p)
            counts["seg?" if kind is None else kind] += 1
            tag = "нет seg-карты" if kind is None else f"{LABELS[kind]} {frac * 100:.1f}%"
            print(f"  {p.name}  ->  [{tag}]")
        print("\nРаспределение промптов: " + ", ".join(
            f"{LABELS.get(k, k)}={v}" for k, v in counts.items() if v))
        for k in ("none", "partial", "full"):
            sample = resolve_prompt(prompts, k, 0.10)  # partial показываем на примере 10%
            dyn = " (динамич. по площади)" if k == "partial" and prompts.get(k) is None else ""
            print(f"  [{k}]{dyn} {len(sample)} символов: {sample[:70]}...")
        print(f"\nПорог ткани: нет <{CLOTH_MIN_FRACTION:.0%} | весь кадр >{CLOTH_FULL_FRACTION:.0%} "
              f"| partial-порог мал/крупн {CLOTH_PARTIAL_SPLIT:.0%}")
        print(f"Параллельность: {args.concurrency} | model={MODEL} | quality={args.quality} | aspect={args.aspect}")
        return

    if not Path(args.node).is_file():
        raise SystemExit(f"Node не найден: {args.node} (укажи --node)")
    if not Path(args.cli).is_file():
        raise SystemExit(f"CLI не найден: {args.cli} (укажи --cli; установи @higgsfield/cli)")

    if not todo:
        print("Нечего обрабатывать — все кадры уже имеют *_ai.png (или используй --force)")
        return

    # защита от слива кредитов
    if len(todo) > args.max:
        print(f"⚠ К обработке {len(todo)} кадров, но лимит прогона --max={args.max} "
              f"(≈{args.max} кредитов). Обработаю первые {args.max}. "
              f"Подними --max, когда будешь готов тратить больше.")
    batch = todo[:args.max]
    print(f"Обрабатываю {len(batch)} кадров (≈{len(batch)} кредитов), "
          f"по {args.concurrency} параллельно. Прервать — Ctrl+C.\n")

    sem = asyncio.Semaphore(args.concurrency)
    stats = {"done": 0, "failed": 0, "created": []}
    await asyncio.gather(*(process_one(args, src, prompts, sem, stats) for src in batch))

    print(f"\nГотово. Успешно: {stats['done']}, ошибок: {stats['failed']}. "
          f"Осталось необработанных в папке: "
          f"{len(todo) - stats['done']} (перезапусти для следующей порции).")
    # только кадры, созданные ИМЕННО в этом прогоне, идут в приёмку
    return stats["created"]


def launch_review(args, created):
    """Открыть UI приёмки ТОЛЬКО по кадрам, созданным в этом прогоне."""
    if not created:
        print("\nПриёмка пропущена: в этом прогоне не создано ни одного кадра.")
        return
    try:
        import review_ai
    except Exception as e:  # noqa: BLE001
        print(f"Не удалось открыть приёмку (review_ai): {e}")
        return
    print(f"\nОткрываю UI приёмки по {len(created)} кадрам этого прогона…")
    review_ai.run(args.input_dir, files=created)


if __name__ == "__main__":
    _args = parse_args()
    _created = None
    try:
        _created = asyncio.run(main(_args))
    except KeyboardInterrupt:
        print("\nПрервано. Сохранённые *_ai.png на месте; перезапуск продолжит с оставшихся.")
        sys.exit(130)
    if _args.review and not _args.list:
        launch_review(_args, _created)
