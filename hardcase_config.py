#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hardcase_config.py
==================
Конфиг второго датасета — «сложные случаи» для сегментации кузова и груза.
Подключается к пайплайну dataset_ai_batch.py флагом --dataset hardcase.

Две трудности, ради которых он и делается (обе взяты с реальных кадров IQoko):

  road  — за кузовом асфальтовая дорога РОВНО ТОГО ЖЕ серо-голубого тона, что
          и борт, плюс оцинкованный отбойник того же семейства серого. Граница
          «борт / фон» перестаёт быть контрастной, и сегментатор не может
          опереться на яркостный перепад.

  pit   — карьер, где грунт, стенки и кучи вокруг сделаны из ТОГО ЖЕ материала,
          что и груз в кузове. Насыпь визуально перетекает в фон, а сами борта
          (дальний и правый) почти не видны из-под груза.

Геометрию «борта не видно» промптом не создать — она берётся из маски. Поэтому
под pit отбираются кадры из 13.08/full с минимальной долей синего в силуэте
(до 0.197 — борта скрыты грузом почти целиком).

Освещение здесь намеренно плоское, рассеянное: жёсткие тени дали бы лишний
контраст по краям и разрушили бы ровно ту маскировку, ради которой датасет.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

import dataset_ai_batch as base

ROOT = Path(r"D:\IQoko\datasets\hardcase")
TOTAL = 250

# Источники. r0499 — 500 кадров, ИИ их ещё не касался, и кузов там другой
# (FAW-J6-8x4 вместо Shacman), что даёт разнообразие относительно первого сета.
SRC_ROAD_LOADED = Path(r"D:\IQoko\datasets\r0499_vol0035.34_random_20260806_174859_645484")
SRC_EMPTY = Path(r"D:\IQoko\datasets\13.08\empty")
SRC_BURIED = Path(r"D:\IQoko\datasets\13.08\full")

# 250 кадров: road 150 (40 пустых + 110 гружёных) + pit 100.
N_ROAD_EMPTY, N_ROAD_LOADED, N_PIT = 40, 110, 100
SEED = 20260828

CARGO_RGB, BODY_RGB = (253, 2, 2), (40, 85, 243)


# ─────────────────────────── СЦЕНЫ ───────────────────────────
SCENES = {
    "road": {
        "background": (
            "Background: a busy two-lane asphalt highway running close behind and alongside the "
            "truck — worn grey asphalt with faded white lane markings, patched tar seams and "
            "damp streaks; a galvanised steel W-beam crash barrier on regularly spaced posts runs "
            "along the road edge just behind the body; other trucks and cars pass on the road, "
            "with roadside grass, birches and spruces, power poles and a distant overpass beyond. "
        ),
        "camouflage": (
            "CRITICAL LOW-CONTRAST REQUIREMENT — this is the whole point of the shot: the outer "
            "steel of the truck body is painted a faded, dusty blue-grey of almost exactly the "
            "same lightness and hue as the asphalt road behind it, and the galvanised guardrail "
            "is in that same grey family. Where the edge of the body meets the road behind it, "
            "the two must be genuinely hard to tell apart: NO bright rim, NO dark outline, NO "
            "halo, NO helpful colour shift along that edge — only a faint change of texture and "
            "a soft contact shadow hints at the boundary. Keep the body believable dirty painted "
            "steel — dusty, scratched, rust bleeding from the seams, panel-to-panel variation — "
            "never a flat uniform fill and never the pure saturated blue of IMAGE 2. "
        ),
    },
    "pit": {
        "background": (
            "Background: an open working quarry pit. The ground under the truck, the cut faces of "
            "the pit behind it and several large heaps standing around it are all made of the "
            "SAME material as the load, in the same colour and the same grain; a tracked excavator "
            "and a wheel loader work among the heaps, tyre ruts and track marks run through the "
            "spilled material, and a haze of dust hangs over the pit. "
        ),
        "camouflage": (
            "CRITICAL LOW-CONTRAST REQUIREMENT — this is the whole point of the shot: the material "
            "in the body and the material of the heaps and ground behind it are the same stuff "
            "under the same light, so the load blends into the background almost seamlessly. "
            "Along the crest where the load meets the background there must be NO helpful "
            "contrast: no bright band of sky, no dark gap, no colour or brightness step, no "
            "outline. The eye should struggle to tell where the load in the body ends and the "
            "heap behind it begins; only a subtle change in the direction of the surface and a "
            "faint contact shadow suggest the boundary. Do not brighten, darken, sharpen or "
            "outline the load to separate it from the background. "
            # Первая проба дала груз на 53 единицы яркости темнее фона — край
            # читался легко, маскировка не работала. Отсюда явное требование
            # совпадения тона, а не только материала.
            "MATCH THE TONE, not just the material: the load in the body must be exactly as pale, "
            "as dry and as dust-coated as the heaps standing behind it. Do not paint it darker, "
            "wetter, fresher or more saturated than the background material — measured as average "
            "brightness, the load and the background heaps must come out the same. The same pale "
            "quarry dust settles on everything in the frame. "
        ),
        # В карьере борт не маскируется под асфальт, поэтому уводим его от синего
        # вовсе — иначе он тянется к цвету самой метки (в пробе вышло B-R +33).
        "body_note": (
            "The truck body here is dusty dark grey-brown steel under a film of the same pale "
            "quarry dust — worn, scratched and rust-streaked. It is NOT blue and NOT painted in "
            "any vivid colour. "
        ),
    },
}

# Плоский свет — намеренно. Жёсткая тень дала бы контрастную кромку и убила бы маскировку.
_LIGHT = (
    "Lighting: flat, diffuse daylight under a thin overcast — soft shadows, low overall contrast, "
    "no direct sun and no strong highlights. This flat light is deliberate: it removes the shading "
    "that would otherwise conveniently separate the objects from their background. Colours are "
    "muted and close to each other in value across the whole frame. "
)

# Цветовая оговорка переписана под этот сет: синий борт здесь НУЖЕН (он и есть
# маскировка под асфальт), запрещён только чистый насыщенный синий самой метки.
_NOCOLOR = (
    "COLOUR WARNING about IMAGE 2: its red and blue are labels, not paint. Never reproduce the "
    "flat saturated red or the pure electric blue of IMAGE 2 anywhere in your output, and never "
    "let the photograph take a colour cast from it. The load keeps the natural colour of its "
    "material and is never red; the body is dirty painted steel in the muted tone described "
    "below, never a clean vivid blue. The photograph keeps full natural colour and detail. "
)

MATERIALS = {
    "sand": {
        "load": (
            "Load material: dry quarry sand — fine granular sand of a warm grey-beige to pale "
            "ochre tone, naturally slumped, with angle-of-repose slopes, small avalanche runnels "
            "and loader-bucket scoop marks, evenly dry and dusty. Re-texture the existing heap "
            "as this sand: same outline, same crest line, same volume, same relief, exactly the "
            "same pixels. "
        ),
        "spill": "sand grains, pale dust films and thin trickle streaks of sand ",
    },
    "stone": {
        "load": (
            "Load material: crushed rock and broken concrete rubble — angular pale grey stones and "
            "slabs of mixed size with exposed aggregate, chipped edges and crushed fines filling "
            "the gaps, uniformly coated in pale rock dust. Re-texture the existing heap as this "
            "rubble: same outline, same crest line, same volume, same relief, exactly the same "
            "pixels. "
        ),
        "spill": "rock dust, fine grit and thin trickle streaks of crushed stone ",
    },
}


# Для карьера пункт «борта читаются сталью по всей длине» из основного сета
# противоречит замыслу: там борта как раз завалены грузом. Требование остаётся,
# но привязано строго к синему в маске, а не к «всей длине кузова».
_FILL_FULL_PIT = (
    "The body is loaded far above the brim and the load spills over the rails: the heap buries "
    "most of the side walls, and the far wall and the right-hand wall are almost entirely hidden "
    "under the material, exactly as IMAGE 2 shows. Do not invent visible wall where IMAGE 2 has "
    "none — if the layout map marks a place as load, it is load, even where a truck body would "
    "normally show its rail. Where IMAGE 2 DOES mark blue, however small that strip is, it must "
    "read unmistakably as the painted steel of the body, not as material: keep its edge, its "
    "surface and its shading legible so the strip does not dissolve into the heap. Do not change "
    "the boundary between the load and the body by a single pixel: {spill} lie on whatever rail "
    "is still visible, as texture on steel only — never as an extra mound built on top of it. "
)


def build_prompt(engine: str, material: str, fill: str, scene: str = "road") -> str:
    """Тот же скелет, что в основном сете (геометрия, гайд-маска, борта), но с
    новым фоном, маскировкой и плоским светом."""
    m = MATERIALS[material]
    sc = SCENES[scene]
    if fill == "full":
        tmpl = _FILL_FULL_PIT if scene == "pit" else base.FILL_TEXT["full"]
        fill_text = tmpl.format(spill=m["spill"])
    else:
        fill_text = base.FILL_TEXT[fill]
    load = m["load"] if fill != "empty" else ""
    crest = base._CREST if fill != "empty" else ""
    return (
        base._GUIDE
        + _NOCOLOR
        + crest
        + base.ENGINE_HEAD[engine]
        + base._GEOMETRY
        + fill_text
        + load
        + base._BODY
        + sc.get("body_note", "")
        + sc["background"]
        + sc["camouflage"]
        + _LIGHT
        + base._STYLE
        + base.ENGINE_TAIL[engine]
    )


# ─────────────────────────── ПЛАН ───────────────────────────
def _body_fraction(seg_path: Path) -> float:
    """Доля кузова в силуэте «кузов+груз». Мало = борта скрыты насыпью."""
    seg = cv2.imread(str(seg_path))
    if seg is None:
        return 1.0
    c = np.all(seg[:, :, ::-1] == CARGO_RGB, axis=-1).sum()
    b = np.all(seg[:, :, ::-1] == BODY_RGB, axis=-1).sum()
    return float(b) / max(float(b + c), 1.0)


def _frames(folder: Path) -> list[Path]:
    out = []
    for p in sorted(folder.glob("*.png")):
        if p.stem.endswith(("_seg", "_depth")) or "_ai" in p.stem:
            continue
        if p.with_name(p.stem + "_seg.png").is_file() and p.with_suffix(".json").is_file():
            out.append(p)
    return out


def _fill_of(frac_cargo: float) -> str:
    """Классы наполненности по доле груза в силуэте."""
    if frac_cargo < 0.04:
        return "empty"
    return "partial" if frac_cargo < 0.30 else "full"


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    picked: list[tuple[Path, str, str]] = []   # (кадр, сцена, наполненность)

    # road / пустые — реально пустой кузов на фоне дороги (как на фото 2)
    empties = _frames(SRC_EMPTY)
    rng.shuffle(empties)
    picked += [(p, "road", "empty") for p in empties[:N_ROAD_EMPTY]]

    # road / гружёные — свежий пул r0499, весь диапазон наполненности
    loaded = _frames(SRC_ROAD_LOADED)
    scored = [(p, 1.0 - _body_fraction(p.with_name(p.stem + "_seg.png"))) for p in loaded]
    scored.sort(key=lambda t: t[1])
    # равномерно прореживаем по наполненности, чтобы не собрать одни полупустые
    step = max(1, len(scored) // N_ROAD_LOADED)
    take = [scored[i] for i in range(0, len(scored), step)][:N_ROAD_LOADED]
    if len(take) < N_ROAD_LOADED:
        seen = {p for p, _ in take}
        take += [s for s in scored if s[0] not in seen][: N_ROAD_LOADED - len(take)]
    picked += [(p, "road", _fill_of(f)) for p, f in take]

    # pit / скрытые борта — самые «закопанные» кадры из 13.08/full
    buried = _frames(SRC_BURIED)
    bscored = sorted(((p, _body_fraction(p.with_name(p.stem + "_seg.png"))) for p in buried),
                     key=lambda t: t[1])[:N_PIT]
    picked += [(p, "pit", "full") for p, _ in bscored]

    if len(picked) < TOTAL:
        raise SystemExit(f"Набралось только {len(picked)} кадров из {TOTAL}")

    # Материал 50/50 внутри каждой (сцена, наполненность), движок задаётся снаружи.
    plan: list[dict] = []
    groups: dict[tuple[str, str], list[Path]] = {}
    for p, scene, fill in picked:
        groups.setdefault((scene, fill), []).append(p)
    for (scene, fill), items in groups.items():
        rng.shuffle(items)
        for i, p in enumerate(items):
            plan.append({
                "stem": p.stem,
                "src": str(p),
                "scene": scene,
                "fill": fill,
                "material": "sand" if i % 2 == 0 else "stone",
                "engine": "seedream",
            })
    base._rebalance(plan, "material", ("sand", "stone"), rng)
    plan.sort(key=lambda r: (r["scene"], r["stem"]))
    return plan
