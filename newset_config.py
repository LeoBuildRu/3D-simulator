#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newset_config.py
================
Третий датасет сложных случаев — на свежем пуле D:\\IQoko\\datasets\\dataset
(1000 кадров от 21.09.2026, модель кузова Shackman-X3000-6x4-tall, ИИ их не
касался). Подключается к пайплайну флагом --dataset newset.

Сцены, промпты, материалы и освещение берутся целиком из hardcase_config —
они уже отлажены и приняты на предыдущем датасете. Здесь переопределяются
только источник, объём и раскладка плана.

Про глубину засыпки бортов
--------------------------
В hardcase сцена pit опиралась на кадры 13.08/full с долей кузова в силуэте от
0.197 — там борта скрыты грузом почти целиком (случай с фото пользователя, где
не видно ни правого, ни дальнего борта). В новом пуле такого нет: минимум
0.348, ниже 0.40 всего 36 кадров. Кузов X3000 выше, и даже при объёме 26-27 м³
против паспортного 21 борта остаются частично видны. Поэтому pit здесь берёт
самые засыпанные кадры пула, но засыпка мягче, чем в предыдущем сете. Старую
геометрию не подмешиваем: пользователь дал новые исходники именно чтобы уйти
от повторов.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

import dataset_ai_batch as base
import hardcase_config as hc

# Сцены/промпты/материалы — как в hardcase, без изменений.
SCENES = hc.SCENES
MATERIALS = hc.MATERIALS
build_prompt = hc.build_prompt

ROOT = Path(r"D:\IQoko\datasets\newset")
TOTAL = 300

SRC = Path(r"D:\IQoko\datasets\dataset")

# 300 кадров: road 180 (60 пустых + 120 гружёных) + pit 120.
N_ROAD_EMPTY, N_ROAD_LOADED, N_PIT = 60, 120, 120
SEED = 20260924

CARGO_RGB, BODY_RGB = hc.CARGO_RGB, hc.BODY_RGB
EMPTY_CARGO_PX = 3000          # меньше этого груза в маске — считаем кузов пустым


def _frames(folder: Path) -> list[Path]:
    out = []
    for p in sorted(folder.glob("*.png")):
        if p.stem.endswith(("_seg", "_depth")) or "_ai" in p.stem:
            continue
        if p.with_name(p.stem + "_seg.png").is_file() and p.with_suffix(".json").is_file():
            out.append(p)
    return out


def _areas(seg_path: Path) -> tuple[int, int]:
    seg = cv2.imread(str(seg_path))
    if seg is None:
        return 0, 0
    c = int(np.all(seg[:, :, ::-1] == CARGO_RGB, axis=-1).sum())
    b = int(np.all(seg[:, :, ::-1] == BODY_RGB, axis=-1).sum())
    return c, b


def _fill_of(cargo_frac: float) -> str:
    if cargo_frac < 0.04:
        return "empty"
    return "partial" if cargo_frac < 0.30 else "full"


def build_plan() -> list[dict]:
    rng = random.Random(SEED)
    frames = _frames(SRC)
    if not frames:
        raise SystemExit(f"нет кадров в {SRC}")

    measured = []
    for p in frames:
        c, b = _areas(p.with_name(p.stem + "_seg.png"))
        total = c + b
        measured.append((p, c, b, b / total if total else 1.0, c / total if total else 0.0))

    empties = [m for m in measured if m[1] < EMPTY_CARGO_PX]
    loaded = [m for m in measured if m[1] >= EMPTY_CARGO_PX]
    if len(empties) < N_ROAD_EMPTY:
        raise SystemExit(f"пустых кадров {len(empties)}, нужно {N_ROAD_EMPTY}")

    # pit — самые засыпанные (минимальная доля кузова в силуэте)
    loaded.sort(key=lambda m: m[3])
    pit = loaded[:N_PIT]
    rest = loaded[N_PIT:]
    if len(rest) < N_ROAD_LOADED:
        raise SystemExit(f"гружёных кадров под road не хватает: {len(rest)}")

    # road / гружёные — равномерно по всему диапазону наполненности, чтобы не
    # собрать одни полупустые
    rest.sort(key=lambda m: m[4])
    step = max(1, len(rest) // N_ROAD_LOADED)
    road_loaded = [rest[i] for i in range(0, len(rest), step)][:N_ROAD_LOADED]
    if len(road_loaded) < N_ROAD_LOADED:
        seen = {m[0] for m in road_loaded}
        road_loaded += [m for m in rest if m[0] not in seen][:N_ROAD_LOADED - len(road_loaded)]

    rng.shuffle(empties)
    picked = ([(m, "road", "empty") for m in empties[:N_ROAD_EMPTY]]
              + [(m, "road", _fill_of(m[4])) for m in road_loaded]
              + [(m, "pit", "full") for m in pit])

    # Материал 50/50 внутри каждой пары (сцена, наполненность)
    groups: dict[tuple[str, str], list] = {}
    for m, scene, fill in picked:
        groups.setdefault((scene, fill), []).append(m)
    plan: list[dict] = []
    for (scene, fill), items in groups.items():
        rng.shuffle(items)
        for i, m in enumerate(items):
            plan.append({
                "stem": m[0].stem,
                "src": str(m[0]),
                "scene": scene,
                "fill": fill,
                "material": "sand" if i % 2 == 0 else "stone",
                "engine": "seedream",
            })
    base._rebalance(plan, "material", ("sand", "stone"), rng)
    plan.sort(key=lambda r: (r["scene"], r["stem"]))
    return plan
