#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recover_jobs.py
===============
Забирает с Higgsfield уже ОПЛАЧЕННЫЕ задания, которые потерял CLI.

Зачем это понадобилось
----------------------
`generate create --wait` создаёт задание (кредит списывается сразу), а затем
ждёт результат. Если ожидание падает по сети, CLI возвращает ошибку — но
задание на сервере живёт и считается оплаченным. Прогон 28.08 из-за этого
списал 37 кредитов, отдав всего 2 кадра: ретраи создавали новые платные
задания вместо того, чтобы забрать уже готовые.

Как восстанавливаем соответствие
--------------------------------
`generate get <id>` отдаёт params.medias — идентификаторы входных файлов
задания. Наш uploads.json хранит обратное: имя файла -> upload id. Разворачиваем
его и по первому входному медиа определяем, какому кадру принадлежит результат.
Дальше остаётся скачать result_url в <stem>_ai.png, если его ещё нет.

Запуск:
  python recover_jobs.py --dataset hardcase [--size 200] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

NODE = r"C:\Program Files\nodejs\node.exe"
CLI = r"C:\Users\xmake\AppData\Roaming\npm\node_modules\@higgsfield\cli\bin\higgsfield.js"

STAGES = {
    "main": Path(r"D:\IQoko\datasets\13.08\result\_stage"),
    "hardcase": Path(r"D:\IQoko\datasets\hardcase\result\_stage"),
}


def cli(*args: str, retries: int = 3) -> str | None:
    """Вызов CLI с повторами: чтение статусов бесплатное, повторять безопасно."""
    for _ in range(retries):
        p = subprocess.run([NODE, CLI, *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout
    return None


def list_job_ids(size: int) -> list[str]:
    out = cli("generate", "list", "--size", str(min(size, 100)))
    if not out:
        return []
    ids = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts and len(parts[0]) == 36 and parts[0].count("-") == 4:
            ids.append(parts[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(STAGES), default="hardcase")
    ap.add_argument("--size", type=int, default=100,
                    help="сколько последних заданий смотреть (сервер разрешает максимум 100)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = STAGES[args.dataset]
    up_file = stage / "uploads.json"
    if not up_file.exists():
        raise SystemExit(f"нет карты загрузок: {up_file}")
    uploads: dict[str, str] = json.loads(up_file.read_text(encoding="utf-8"))
    # upload id -> stem (интересуют только исходники кадра, не маски)
    by_id = {uid: name[:-4] for name, uid in uploads.items()
             if name.endswith(".png") and not name.endswith("_seg.png")}
    print(f"карта загрузок: {len(by_id)} исходников")

    ids = list_job_ids(args.size)
    print(f"заданий в выдаче: {len(ids)}")

    saved = skipped = unmatched = pending = 0
    for jid in ids:
        raw = cli("generate", "get", jid, "--json")
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if d.get("status") != "completed" or not d.get("result_url"):
            pending += 1
            continue
        stem = None
        for m in d.get("params", {}).get("medias", []):
            mid = (m.get("data") or {}).get("id")
            if mid in by_id:
                stem = by_id[mid]
                break
        if stem is None:
            unmatched += 1
            continue
        dst = stage / f"{stem}_ai.png"
        if dst.exists():
            skipped += 1
            continue
        print(f"  восстановлен {stem}  (job {jid[:8]})")
        if not args.dry_run:
            r = requests.get(d["result_url"], timeout=300)
            r.raise_for_status()
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dst)
        saved += 1

    print(f"\nвосстановлено {saved}, уже были {skipped}, "
          f"чужих/несопоставленных {unmatched}, ещё в работе {pending}")


if __name__ == "__main__":
    main()
