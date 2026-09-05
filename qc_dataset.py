#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc_dataset.py
=============
Приёмка датасета: насколько контуры seg-маски ложатся на рёбра AI-кадра.

Метрика — та же, что использует fix_ai_offset для приёмки собственной коррекции:
средняя чемферная дистанция (px) от контура класса до ближайшего края на
картинке, со знаком минус.

Считать её в АБСОЛЮТЕ бесполезно: она зависит от того, насколько контрастны
края в кадре. На тёмном пустом кузове в тени исходный рендер набирает -36 —
хуже любого AI-кадра. Поэтому меряем РАЗНИЦУ с рендером, который выровнен с
маской по построению: 0 — AI не хуже эталона, минус — хуже.

Отдельно по кузову (синий) и по грузу (красный): у них разные режимы отказа.
Кузов ловит сдвиг/зум кадра и случаи, когда борт завален материалом; груз ловит
уехавший уровень насыпки.

Запуск:
  python qc_dataset.py --dataset hardcase [--worst 15] [--redo-out redo.txt]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fix_ai_offset import IDENTITY, seg_score  # noqa: E402

ROOTS = {
    "main": pathlib.Path(r"D:\IQoko\datasets\13.08\result"),
    "hardcase": pathlib.Path(r"D:\IQoko\datasets\hardcase\result"),
}
CARGO, BODY = (253, 2, 2), (40, 85, 243)


def final_path(stage: pathlib.Path, stem: str):
    for suf in ("_ai_fix_crest", "_ai_fix", "_ai"):
        p = stage / f"{stem}{suf}.png"
        if p.exists():
            return p
    return None


def score_one(stage: pathlib.Path, stem: str):
    ai_p = final_path(stage, stem)
    seg_p, orig_p = stage / f"{stem}_seg.png", stage / f"{stem}.png"
    if ai_p is None or not seg_p.exists() or not orig_p.exists():
        return None
    seg = cv2.imread(str(seg_p), cv2.IMREAD_COLOR)
    orig = cv2.imread(str(orig_p), cv2.IMREAD_GRAYSCALE)
    ai = cv2.imread(str(ai_p), cv2.IMREAD_GRAYSCALE)
    if seg is None or orig is None or ai is None:
        return None
    h, w = orig.shape
    if ai.shape != (h, w):
        ai = cv2.resize(ai, (w, h), interpolation=cv2.INTER_AREA)
    out = {}
    for name, rgb in (("body", BODY), ("cargo", CARGO)):
        m = (np.all(seg[:, :, ::-1] == rgb, axis=-1).astype(np.uint8)) * 255
        if m.sum() <= 255 * 3000:
            out[name] = None
            continue
        out[name] = seg_score(orig, ai, m, IDENTITY) - seg_score(orig, orig, m, IDENTITY)
    return out


def stats(vals):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return "нет данных"
    return (f"n={len(v):3d}  медиана {np.median(v):+6.2f}  среднее {v.mean():+6.2f}  "
            f"5% {np.percentile(v, 5):+6.2f}  мин {v.min():+7.2f}  хуже -5: {(v < -5).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(ROOTS), default="hardcase")
    ap.add_argument("--worst", type=int, default=15)
    ap.add_argument("--redo-out", default=None, help="файл со списком кадров на перегенерацию")
    ap.add_argument("--threshold", type=float, default=-5.0)
    args = ap.parse_args()

    root = ROOTS[args.dataset]
    stage = root / "_stage"
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))

    rows = []
    for i, rec in enumerate(plan, 1):
        s = score_one(stage, rec["stem"])
        if s:
            rows.append((rec, s))
        if i % 50 == 0:
            print(f"  {i}/{len(plan)}", flush=True)
    print(f"\nоценено кадров: {len(rows)}")

    for key in ("body", "cargo"):
        print(f"\n=== {key} ===")
        print("  всего     ", stats([s[key] for _, s in rows]))
        for field in ("scene", "fill", "material"):
            if field not in rows[0][0]:
                continue
            for val in sorted({r[field] for r, _ in rows}):
                sub = [s[key] for r, s in rows if r[field] == val]
                if any(x is not None for x in sub):
                    print(f"  {val:9s} ", stats(sub))

    print(f"\n=== худшие {args.worst} по кузову ===")
    for v, r in sorted(((s["body"], r) for r, s in rows if s["body"] is not None))[:args.worst]:
        sc = r.get("scene", "-")
        print(f"  {v:+7.2f}  {sc:5s} {r['material']:5s} {r['fill']:7s} {r['stem']}")

    bad = sorted({r["stem"] for r, s in rows
                  if s["body"] is not None and s["body"] < args.threshold})
    print(f"\nхуже порога {args.threshold} по кузову: {len(bad)}")
    if args.redo_out:
        pathlib.Path(args.redo_out).write_text("\n".join(bad), encoding="utf-8", newline="\n")
        print(f"список записан: {args.redo_out}")

    (root / "qc_scores.json").write_text(json.dumps(
        [{"stem": r["stem"], **{k: r.get(k) for k in ("scene", "fill", "material", "engine")},
          **s} for r, s in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"подробности: {root / 'qc_scores.json'}")


if __name__ == "__main__":
    main()
