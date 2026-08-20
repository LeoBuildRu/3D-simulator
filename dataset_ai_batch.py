#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_ai_batch.py
===================
Сборка единого сегментационного датасета из D:\\IQoko\\datasets\\13.08
(папки full / empty / random) с AI-перетекстуризацией через Higgsfield CLI.

Требования задачи:
  * 500 кадров: ~45% полные, ~40% пустые, остальное — частичное наполнение;
  * 50/50 два материала: сухой карьерный ПЕСОК и серый бетонный ЛОМ/камень;
    в обоих случаях на фоне — кучи ТОГО ЖЕ материала (hard negatives);
  * день, жёсткие сложные тени от елей и мостового пролёта примерно на половину кадра;
  * половина кадров через seedream_v5_lite, половина — через nano_banana_flash;
  * идеальное соответствие обработанного кадра и его seg-маски.

Геометрия кадра не меняется, поэтому исходная seg-маска остаётся валидной.
Для Seedream дополнительно прогоняется fix_ai_offset.py (он ловит паразитный
сдвиг/зум по синему кузову из seg-карты).

Ключевой приём: seg-карта подаётся ВТОРЫМ image-reference как схема раскладки.
Без неё обе модели уводили силуэт груза на 50-90 px по вертикали (Nano вдобавок
зумил кадр) — полоса вдоль гребня оказывалась размечена как cargo поверх фона.
С гайдом seg-скор fix_ai_offset на тестовых кадрах: -2.00 (Seedream) и -2.05
(Nano) против -5.34 и -5.15 без него.

Этапы (--stage-step):
  plan      только показать план (ничего не тратится)
  stage     скопировать исходники и маски в рабочую папку
  generate  запустить генерации (тратит кредиты; ограничено --max)
  fix       прогнать fix_ai_offset.py (нужен обоим движкам)
  crest     ОПЦИОНАЛЬНО: fix_load_crest.py, доводка уровня груза. В "all" не
            входит намеренно — см. комментарий у do_crest().
  finalize  собрать result/ (кадр + маска + json)
  all       stage -> generate -> fix -> finalize
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests

# ─────────────────────────── КОНФИГ ───────────────────────────
ROOT = Path(r"D:\IQoko\datasets\13.08")
RESULT = ROOT / "result"
STAGE = RESULT / "_stage"
PLAN_FILE = RESULT / "plan.json"

TOTAL = 500
SHARE = {"full": 0.45, "empty": 0.40, "partial": 0.15}   # 225 / 200 / 75

# «random» с объёмом >= этого считаем фактически полным кузовом,
# ниже — частичным наполнением. max_volume кузова = 21 м³.
RANDOM_FULL_VOL = 24.0
RANDOM_PARTIAL_MAX_VOL = 23.0

ENGINES = {
    "seedream": {
        "model": "seedream_v5_lite",
        "extra": ["--quality", "high"],
        "needs_fix": True,      # даёт паразитный сдвиг/зум -> fix_ai_offset.py
    },
    "nano": {
        "model": "nano_banana_flash",
        "extra": ["--resolution", "2k"],
        # Проверено на валидации: Nano Banana тоже уводит кадр (зум + сдвиг),
        # так что коррекция нужна обоим движкам.
        "needs_fix": True,
    },
}
ASPECT = "16:9"                 # исходники 1920x1080 == ровно 16:9
WAIT_TIMEOUT = "20m"
MAX_CONCURRENCY = 6
DEFAULT_MAX_PER_RUN = 10        # защита от слива кредитов

NODE = r"C:\Program Files\nodejs\node.exe"
CLI_JS = r"C:\Users\xmake\AppData\Roaming\npm\node_modules\@higgsfield\cli\bin\higgsfield.js"

# fix_ai_offset.py корректирует только кадры, созданные ПОЗЖЕ этого файла-якоря.
# Кладём пустышку в рабочую папку ДО генераций, чтобы все новые кадры попали в окно.
CUTOFF_REFERENCE = "r0006_vol0007.53_random_20260723_141356_127246_ai.png"

# Upload-эндпоинт Higgsfield периодически отдаёт «Failed to upload media bytes» /
# «no response received» целыми окнами по несколько минут. Кредит при этом не
# списывается, помогает только повтор с ощутимой задержкой.
CREATE_RETRIES = 6
RETRY_BACKOFF_S = 20.0
SEED = 20260814


# ─────────────────────────── ПРОМПТЫ ───────────────────────────
# Общая часть: жёсткая фиксация геометрии. Это единственное, что гарантирует
# совпадение обработанного кадра с seg-маской, поэтому формулировки дублируются
# и в начале, и в конце промпта.
_GEOMETRY = (
    "Keep the output pixel-aligned with the input image: identical camera position, identical "
    "framing and field of view, identical fisheye lens distortion, identical horizon. No shift, "
    "no zoom, no rotation, no crop, no re-framing, no change of aspect ratio. Every edge of the "
    "dump truck body and every edge of the material inside it must stay on exactly the same "
    "pixels as in the input. Change ONLY surface textures, materials, lighting and the "
    "background scene — never shapes, outlines, silhouettes or positions. "
)

_BODY = (
    "Truck body: weathered, battered, dented steel with worn shiny patches, rust streaks, seam "
    "and reinforcing ribs running the full length of the walls, scratches and dried dirt. Do not "
    "change its shape, wall thickness, wall height or position. "
)

_STYLE = (
    "Style: cheap analog on-board CCTV camera — realistic photograph, not CGI and not an "
    "illustration; muted colours, limited dynamic range with some clipped highlights and crushed "
    "shadows, mild sensor noise, slight chromatic fringing at the frame edges. "
)

# Жёсткие сложные тени — главный источник сложности для сегментатора.
_SHADOWS = (
    "Lighting: clear daytime, bright direct sun at a medium-high angle. Hard-edged, complex "
    "shadows are cast into the scene from tall spruce trees and from an overhead concrete bridge "
    "span standing just outside the frame. A sharp shadow band with ragged conifer-branch fringes "
    "and dappled light gaps runs diagonally across roughly half of the frame, leaving the other "
    "half in full direct sunlight, so that the same material appears both brightly lit and deeply "
    "shaded within one image. The shadow falls as one continuous cast pattern across the ground, "
    "the truck body and the load alike, crossing the body walls and the heap without breaking. "
    "Contrast between lit and shaded areas is strong; shaded areas are dark but never pure black "
    "and still show texture. The shadow is light only — it must not move, reshape, hide or add "
    "any object. "
)

MATERIALS = {
    "sand": {
        "load": (
            "Load material: dry quarry sand — fine beige-grey granular sand with a naturally "
            "slumped surface, angle-of-repose slopes, small avalanche runnels, loader-bucket "
            "scoop marks, and subtle colour variation where the sand is damp and darker. "
            "Re-texture the existing heap as this sand: same outline, same crest line, same "
            "volume, same surface relief, exactly the same pixels. "
        ),
        "background": (
            "Background: an open industrial sand depot in daylight — several large heaps of the "
            "SAME beige quarry sand standing on the concrete yard behind and beside the truck, "
            "wheel-loader tracks through spilled sand on the ground, a few parked dump trucks, "
            "hangars and a wheel loader. The background sand heaps stand on the ground clearly "
            "separate from the truck: they must never touch, overlap, lean on or visually "
            "continue into the truck body. "
        ),
        "spill": (
            "sand grains, pale dust films and thin trickle streaks of sand "
        ),
    },
    "stone": {
        "load": (
            "Load material: coarse demolition rubble — large angular grey concrete blocks and "
            "broken slabs with exposed aggregate and snapped rebar ends, bent scrap steel, broken "
            "bricks and crushed concrete fines filling the gaps; dusty pale grey with darker "
            "wet-grey patches and rust stains. Re-texture the existing heap as this rubble: same "
            "outline, same crest line, same volume, same surface relief, exactly the same pixels. "
        ),
        "background": (
            "Background: an open demolition-waste yard in daylight — several large piles of the "
            "SAME grey broken concrete and scrap rubble standing on the concrete yard behind and "
            "beside the truck, concrete dust and stray blocks on the ground, a few parked dump "
            "trucks, hangars and an excavator with a sorting grab. The background rubble piles "
            "stand on the ground clearly separate from the truck: they must never touch, overlap, "
            "lean on or visually continue into the truck body. "
        ),
        "spill": (
            "grey concrete dust, fine grit and thin trickle streaks of crushed stone "
        ),
    },
}

# Наполненность. «Наслаивание на борта» для полного кузова сделано как ТЕКСТУРА
# на стали (пыль, потёки, отдельные крупинки), а не как объёмная куча поверх
# борта: объёмная куча изменила бы класс пикселей борта и разошлась бы с маской.
FILL_TEXT = {
    "full": (
        "The body is loaded to the brim and slightly overfilled: the crest of the heap reaches "
        "and rides over the top edge of the side walls, exactly as in the input image. Make the "
        "overfilling read clearly: {spill} lie on the top rails and run down the outside of the "
        "side walls in dusty smears and drift lines, and the material at the crest sits flush "
        "against the rail with no gap. IMPORTANT: this spill is TEXTURE AND DIRT ON THE STEEL "
        "ONLY. Do not build any three-dimensional mound, ridge, lump or pile of material sitting "
        "on top of, over or outside the rails, do not let material hang off the body, and do not "
        "move the boundary between the load and the body by a single pixel. The silhouette of the "
        "heap against the sky and against the body stays exactly as it is. "
        # Nano при песке доводил «наслаивание» до того, что борт исчезал под насыпью
        # целиком: класс cuzov оставался без картинки (40% брака в этой группе).
        "THE SIDE WALLS MUST STAY VISIBLE. Every part of the body that IMAGE 2 marks BLUE must "
        "read as bare steel wall in your output — dirty and dusted, but unmistakably steel, with "
        "its top rail, its seams and its outer face clearly legible along the whole length of the "
        "body. Do NOT bury, drape, blanket, coat over or hide the walls under the material, and "
        "do NOT let the heap flow down over the outside of the body. The spill is a dusting and a "
        "few thin streaks on steel, nothing more — if a viewer could not tell where the steel "
        "wall is, you have gone too far. "
    ),
    "empty": (
        "The truck body is COMPLETELY EMPTY: bare dirty steel floor and bare steel walls, with "
        "nothing inside it. Do NOT put any sand, rubble, gravel, stones, cargo or debris inside "
        "the body — not a heap, not a layer, not a scattering, not even a few grains. The floor of "
        "the body must remain visible bare steel across its whole area. The heaps of material "
        "exist ONLY on the ground outside the truck, in the background; nothing from the "
        "background may spill, extend or be copied into the body. "
    ),
    "partial": (
        "The body is only partly loaded: the material fills the lower part of the body and leaves "
        "bare steel floor and bare steel walls exposed above and around it. Keep the boundary "
        "line between the material and the bare steel exactly where it is now, to the pixel — do "
        "not raise, lower, spread, level or shrink the load, do not fill the body further, and do "
        "not add any material, dust drift or scattering onto the exposed steel floor or walls. "
        "Every part of the steel that is bare now stays bare; every part covered by material now "
        "stays covered. "
    ),
}

# Модель-специфичные обёртки: Seedream лучше слушается описательной подачи с
# явным «re-render the SAME photograph», Nano Banana (Gemini) — инструкции-приказа.
ENGINE_HEAD = {
    "seedream": (
        "Re-render this exact photograph with new materials and new lighting. It is the same shot "
        "of the same scene from the same on-board camera — only the surfaces and the light change. "
    ),
    # Nano Banana на первой версии промпта перерисовывал сцену заново (зум внутрь,
    # другой кузов, груз опущен ниже бортов). Поэтому здесь идёт жёсткий блок
    # ограничений ДО описания сцены и явный запрет зума/перекадрирования.
    "nano": (
        "TEXTURE REPLACEMENT TASK — NOT an image re-imagination. You are given a photograph. "
        "Return the SAME photograph with only the surface materials, the lighting and the "
        "background replaced. Treat the input as a locked layer: trace it exactly. "
        "HARD CONSTRAINTS, in priority order over everything written below:\n"
        "1) Do not zoom in or out. The field of view is unchanged; the same objects touch the "
        "same frame edges at the same points as in the input.\n"
        "2) Do not move, straighten or re-crop the frame. Every edge, corner and vertex of the "
        "dump truck body lands on exactly the same pixel coordinates as in the input.\n"
        "3) Do not redesign the truck body. Its wall height, wall thickness, top rail, corner "
        "angles and perspective are copied from the input, not invented.\n"
        "4) Do not change how much material is in the body, or how high its surface sits relative "
        "to the walls. The silhouette of the load — its crest line and where it meets the walls — "
        "is copied from the input pixel for pixel.\n"
        "5) Keep the strong fisheye barrel distortion of the input lens exactly as it is.\n"
        "Now the appearance to paint onto that locked geometry:\n"
    ),
}
ENGINE_TAIL = {
    "seedream": (
        "Final check: the truck body and the material inside it occupy exactly the same pixels as "
        "in the input photograph, at the same scale and the same position; nothing is zoomed, "
        "shifted, straightened or re-cropped."
    ),
    "nano": (
        "FINAL CHECK before you output: overlay your result on the input. The truck body outline "
        "and the load outline must coincide exactly — same scale, same position, same field of "
        "view, same load height against the walls. If anything is zoomed in, shifted, enlarged, "
        "re-proportioned or re-filled, it is wrong. Only textures, lighting and background differ."
    ),
}


# ─── seg-карта вторым референсом ───
# Решающий приём. Без него обе модели уводили силуэт груза на 50-90 px по
# вертикали (Nano вдобавок зумил кадр), и полоса вдоль гребня получала метку
# cargo там, где на картинке уже фон. С гайдом seg-скор fix_ai_offset улучшился
# с -5.3/-5.2 до -2.0/-2.1 на тестовых кадрах.
_GUIDE = (
    "You are given TWO images. IMAGE 1 is the photograph to edit. IMAGE 2 is a flat colour "
    "layout map of IMAGE 1, at the same size and perfectly aligned with it: RED marks exactly "
    "where the load material is, BLUE marks exactly where the steel truck body is, BLACK marks "
    "the background. IMAGE 2 is a geometry guide ONLY — never copy its colours, its flatness or "
    "its hard edges into your output. Use it to place the boundaries: in your output, every pixel "
    "that is RED in IMAGE 2 must show load material, every pixel that is BLUE must show truck "
    "body steel, and every pixel that is BLACK must show background. "
)
# Seedream при цветном гайде красил кузов в синий цвет метки. Запрет снял это
# полностью (средний оттенок кузова B-R = +5, т.е. нейтральный).
_NOCOLOR = (
    "COLOUR WARNING about IMAGE 2: its red and blue are labels, not paint. The truck body in your "
    "output is weathered dark steel — grey, brown, rusty — and must NOT be blue, blue-grey or "
    "blue-tinted anywhere. The load in your output is its natural material colour and must NOT be "
    "red or red-tinted. Nothing in the output may take a colour cast from IMAGE 2, and the "
    "photograph must keep its full natural colour and contrast. "
)
# Отдельно про линию гребня: именно она — граница cargo/фон в маске.
_CREST = (
    "CRITICAL — the height of the load: its ridge line is silhouetted directly against the "
    "background of the yard, and the red/black boundary in IMAGE 2 is exactly that silhouette. "
    "Copy it at exactly the same height in the frame: where IMAGE 2 has red against black, your "
    "output must show load material against background at the same pixel row. Never lower the "
    "load, never level it flat, and never let the far side wall or the yard show through above "
    "the ridge. "
)


def build_prompt(engine: str, material: str, fill: str) -> str:
    m = MATERIALS[material]
    fill_text = FILL_TEXT[fill].format(spill=m["spill"]) if fill == "full" else FILL_TEXT[fill]
    load = m["load"] if fill != "empty" else ""
    crest = _CREST if fill != "empty" else ""
    return (
        _GUIDE
        + _NOCOLOR
        + crest
        + ENGINE_HEAD[engine]
        + _GEOMETRY
        + fill_text
        + load
        + _BODY
        + m["background"]
        + _SHADOWS
        + _STYLE
        + ENGINE_TAIL[engine]
    )


# ─────────────────────────── ПЛАН ───────────────────────────
VOL_RE = re.compile(r"_vol(\d+\.\d+)_")


def frame_volume(stem: str) -> float:
    m = VOL_RE.search(stem)
    return float(m.group(1)) if m else 0.0


def source_frames(folder: Path) -> list[Path]:
    """Исходники (без *_seg), у которых есть и маска, и json."""
    out = []
    for p in sorted(folder.glob("*.png")):
        if p.stem.endswith("_seg"):
            continue
        if p.with_name(p.stem + "_seg.png").is_file() and p.with_suffix(".json").is_file():
            out.append(p)
    return out


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    want = {k: round(TOTAL * v) for k, v in SHARE.items()}
    want["partial"] = TOTAL - want["full"] - want["empty"]

    full_src = source_frames(ROOT / "full")
    empty_src = source_frames(ROOT / "empty")
    rnd_src = source_frames(ROOT / "random")

    rnd_full = [p for p in rnd_src if frame_volume(p.stem) >= RANDOM_FULL_VOL]
    rnd_part = [p for p in rnd_src if frame_volume(p.stem) < RANDOM_PARTIAL_MAX_VOL]

    # В папке full только 200 кадров, а нужно 225 — добор берём из random с
    # объёмом >= RANDOM_FULL_VOL (кузов там залит выше краёв, визуально полный).
    picked: list[tuple[Path, str]] = []
    fulls = list(full_src)
    if len(fulls) < want["full"]:
        need = want["full"] - len(fulls)
        extra = sorted(rnd_full, key=lambda p: -frame_volume(p.stem))[:need]
        if len(extra) < need:
            raise SystemExit(f"Не хватает полных кадров: нужно {want['full']}, есть "
                             f"{len(fulls) + len(extra)}")
        fulls += extra
    else:
        rng.shuffle(fulls)
        fulls = fulls[:want["full"]]
    picked += [(p, "full") for p in fulls[:want["full"]]]

    empties = list(empty_src)
    if len(empties) < want["empty"]:
        raise SystemExit(f"Не хватает пустых кадров: нужно {want['empty']}, есть {len(empties)}")
    rng.shuffle(empties)
    picked += [(p, "empty") for p in empties[:want["empty"]]]

    parts = [p for p in rnd_part if p not in set(fulls)]
    if len(parts) < want["partial"]:
        raise SystemExit(f"Не хватает частичных кадров: нужно {want['partial']}, есть {len(parts)}")
    rng.shuffle(parts)
    picked += [(p, "partial") for p in parts[:want["partial"]]]

    # Материал и движок раскладываем 50/50 ВНУТРИ каждого класса наполненности,
    # чтобы ни материал, ни модель не коррелировали с заполненностью кузова.
    plan: list[dict] = []
    for fill in ("full", "empty", "partial"):
        group = [p for p, f in picked if f == fill]
        rng.shuffle(group)
        for i, p in enumerate(group):
            plan.append({
                "stem": p.stem,
                "src": str(p),
                "fill": fill,
                "material": "sand" if i % 2 == 0 else "stone",
                # сдвиг на 1 внутри материала: движки не совпадают с материалом
                "engine": "seedream" if (i // 2 + i) % 2 == 0 else "nano",
            })
    # Точная балансировка 250/250 по движкам на весь датасет.
    _rebalance(plan, "engine", ("seedream", "nano"), rng)
    _rebalance(plan, "material", ("sand", "stone"), rng)
    plan.sort(key=lambda r: r["stem"])
    return plan


def _rebalance(plan: list[dict], key: str, values: tuple[str, str], rng: random.Random) -> None:
    """Довести раскладку по `key` до ровно 50/50 минимальным числом перестановок."""
    a, b = values
    target = len(plan) // 2
    while sum(1 for r in plan if r[key] == a) > target:
        cand = [r for r in plan if r[key] == a]
        rng.choice(cand)[key] = b
    while sum(1 for r in plan if r[key] == b) > len(plan) - target:
        cand = [r for r in plan if r[key] == b]
        rng.choice(cand)[key] = a


# ─────────────────────────── ЭТАПЫ ───────────────────────────
def do_stage(plan: list[dict]) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    # Файл-якорь для date-gate внутри fix_ai_offset.py — создаём ПЕРВЫМ, чтобы
    # все сгенерированные позже кадры считались «после отсечки».
    ref = STAGE / CUTOFF_REFERENCE
    if not ref.exists():
        ref.write_bytes(b"")
    n = 0
    for rec in plan:
        src = Path(rec["src"])
        for suffix in ("", "_seg"):
            s = src.with_name(src.stem + suffix + ".png")
            d = STAGE / s.name
            if not d.exists():
                shutil.copy2(s, d)
                n += 1
        j = src.with_suffix(".json")
        if j.is_file() and not (STAGE / j.name).exists():
            shutil.copy2(j, STAGE / j.name)
    print(f"stage: {STAGE} — скопировано {n} файлов, {len(plan)} кадров")


def ai_path(rec: dict) -> Path:
    return STAGE / f"{rec['stem']}_ai.png"


def final_ai_path(rec: dict) -> Path:
    """Лучший доступный вариант кадра: crest > offset-fix > сырой AI."""
    for suffix in ("_ai_fix_crest", "_ai_fix", "_ai"):
        p = STAGE / f"{rec['stem']}{suffix}.png"
        if p.exists():
            return p
    return ai_path(rec)


async def run_one(rec: dict, sem: asyncio.Semaphore, args) -> bool:
    cfg = ENGINES[rec["engine"]]
    prompt = build_prompt(rec["engine"], rec["material"], rec["fill"])
    src = STAGE / f"{rec['stem']}.png"
    out = ai_path(rec)
    cmd = [args.node, args.cli, "generate", "create", cfg["model"],
           "--prompt", prompt,
           "--image-references", str(src),
           "--image-references", str(STAGE / f"{rec['stem']}_seg.png"),
           "--aspect_ratio", ASPECT,
           *cfg["extra"],
           "--wait", "--wait-timeout", WAIT_TIMEOUT, "--json"]

    async with sem:
        for attempt in range(1, CREATE_RETRIES + 1):
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy())
            so, se = await proc.communicate()
            if proc.returncode == 0:
                url = extract_url(so.decode("utf-8", "replace"))
                if url:
                    try:
                        download(url, out)
                        print(f"  OK   [{rec['engine']:8s}/{rec['material']:5s}/{rec['fill']:7s}] "
                              f"{rec['stem']}")
                        return True
                    except Exception as e:  # noqa: BLE001
                        print(f"  DL-ERR {rec['stem']}: {e}")
                else:
                    print(f"  NO-URL {rec['stem']}: {so.decode('utf-8', 'replace')[:200]}")
            else:
                print(f"  FAIL({proc.returncode}) {rec['stem']} try {attempt}/{CREATE_RETRIES}: "
                      f"{se.decode('utf-8', 'replace')[:200]}")
            if attempt < CREATE_RETRIES:
                await asyncio.sleep(RETRY_BACKOFF_S * attempt)
        return False


def extract_url(stdout: str) -> str | None:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        m = re.search(r"https://\S+\.(?:png|jpg|jpeg|webp)", stdout)
        return m.group(0) if m else None
    items = data if isinstance(data, list) else [data]
    for it in items:
        if isinstance(it, dict) and it.get("result_url"):
            return it["result_url"]
    return None


def download(url: str, dst: Path) -> None:
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dst)


async def do_generate(plan: list[dict], args) -> None:
    todo = [r for r in plan if not ai_path(r).exists()]
    if args.engine:
        todo = [r for r in todo if r["engine"] == args.engine]
    if args.only:
        todo = [r for r in todo if r["stem"] in set(args.only)]
    todo = todo[:args.max]
    if not todo:
        print("generate: нечего делать")
        return
    print(f"generate: {len(todo)} кадров, concurrency={args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)
    res = await asyncio.gather(*(run_one(r, sem, args) for r in todo))
    print(f"generate: готово {sum(res)}/{len(todo)}")


def do_fix(plan: list[dict], args) -> None:
    targets = [str(ai_path(r)) for r in plan
               if ENGINES[r["engine"]]["needs_fix"] and ai_path(r).exists()]
    if args.only:
        targets = [t for t in targets if Path(t).stem[:-3] in set(args.only)]
    if not targets:
        print("fix: нечего корректировать")
        return
    script = Path(__file__).with_name("fix_ai_offset.py")
    print(f"fix: {len(targets)} seedream-кадров через {script.name}")
    subprocess.run([sys.executable, str(script), "--jobs", str(args.jobs), *targets], check=False)


def do_crest(plan: list[dict], args) -> None:
    """Опциональная доводка уровня груза (fix_load_crest.py). ПО УМОЛЧАНИЮ ВЫКЛЮЧЕНА.

    Работает по внешнему виду: отличает материал от фона линейным дискриминантом,
    обученным на самом кадре по seg-карте. В этом датасете фон по ТЗ забит кучами
    ТОГО ЖЕ материала, что и груз, поэтому дискриминатор systematically путается —
    на проверке он «исправлял» и заведомо правильный исходный рендер. Включать
    только осознанно и с последующей визуальной приёмкой.
    """
    # Целимся именно в выход fix_ai_offset, а не в свой собственный результат.
    targets = [str(STAGE / f"{r['stem']}_ai_fix.png") for r in plan
               if (STAGE / f"{r['stem']}_ai_fix.png").exists()]
    if args.only:
        targets = [t for t in targets if Path(t).name.startswith(tuple(args.only))]
    if not targets:
        print("crest: нечего обрабатывать")
        return
    script = Path(__file__).with_name("fix_load_crest.py")
    print(f"crest: {len(targets)} кадров через {script.name}")
    subprocess.run([sys.executable, str(script), "--jobs", str(args.jobs), *targets], check=False)


def do_finalize(plan: list[dict]) -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    done = missing = 0
    for rec in plan:
        img = final_ai_path(rec)
        if not img.exists():
            missing += 1
            continue
        stem = rec["stem"]
        shutil.copy2(img, RESULT / f"{stem}.png")
        shutil.copy2(STAGE / f"{stem}_seg.png", RESULT / f"{stem}_seg.png")
        js = STAGE / f"{stem}.json"
        if js.exists():
            meta = json.loads(js.read_text(encoding="utf-8"))
            meta["ai_engine"] = ENGINES[rec["engine"]]["model"]
            meta["ai_material"] = rec["material"]
            meta["ai_fill_class"] = rec["fill"]
            meta["ai_offset_corrected"] = img.name.endswith("_ai_fix.png")
            (RESULT / f"{stem}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        done += 1
    print(f"finalize: {done} кадров в {RESULT} (нет AI-кадра: {missing})")


def print_plan(plan: list[dict]) -> None:
    def cnt(key):
        d: dict[str, int] = {}
        for r in plan:
            d[r[key]] = d.get(r[key], 0) + 1
        return d
    print(f"План: {len(plan)} кадров")
    print(f"  наполненность: {cnt('fill')}")
    print(f"  материал:      {cnt('material')}")
    print(f"  движок:        {cnt('engine')}")
    cross: dict[str, int] = {}
    for r in plan:
        k = f"{r['engine']}/{r['material']}/{r['fill']}"
        cross[k] = cross.get(k, 0) + 1
    for k in sorted(cross):
        print(f"    {k:28s} {cross[k]}")


# ─────────────────────────── CLI ───────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="AI-обработка и сборка сегментационного датасета.")
    ap.add_argument("--stage-step", dest="step", default="plan",
                    choices=["plan", "stage", "generate", "fix", "crest", "finalize", "all"])
    ap.add_argument("--max", type=int, default=DEFAULT_MAX_PER_RUN,
                    help=f"макс. генераций за прогон (по умолч. {DEFAULT_MAX_PER_RUN})")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--jobs", type=int, default=4, help="процессов для fix_ai_offset")
    ap.add_argument("--engine", choices=list(ENGINES), default=None)
    ap.add_argument("--only", nargs="*", default=None, help="обрабатывать только эти stem'ы")
    ap.add_argument("--show-prompt", nargs=3, metavar=("ENGINE", "MATERIAL", "FILL"), default=None)
    ap.add_argument("--replan", action="store_true", help="пересобрать plan.json")
    ap.add_argument("--node", default=NODE)
    ap.add_argument("--cli", default=CLI_JS)
    args = ap.parse_args(argv)

    if args.show_prompt:
        print(build_prompt(*args.show_prompt))
        return

    RESULT.mkdir(parents=True, exist_ok=True)
    if PLAN_FILE.exists() and not args.replan:
        plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    else:
        plan = build_plan()
        PLAN_FILE.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"план записан: {PLAN_FILE}")

    if args.step == "plan":
        print_plan(plan)
        return
    if args.step in ("stage", "all"):
        do_stage(plan)
    if args.step in ("generate", "all"):
        asyncio.run(do_generate(plan, args))
    if args.step in ("fix", "all"):
        do_fix(plan, args)
    if args.step == "crest":   # намеренно НЕ входит в "all" — см. do_crest()
        do_crest(plan, args)
    if args.step in ("finalize", "all"):
        do_finalize(plan)


if __name__ == "__main__":
    main()
