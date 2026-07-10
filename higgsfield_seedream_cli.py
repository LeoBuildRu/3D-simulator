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
  * запускает до MAX_CONCURRENCY=8 CLI-процессов параллельно (каждый с --wait);
  * скачивает result_url и сохраняет рядом как <оригинал>_ai.png;
  * резюме: готовые кадры пропускает;
  * ЗАЩИТА ОТ СЛИВА КРЕДИТОВ: за один прогон обрабатывает не больше --max
    кадров (по умолчанию 10). Для большего объёма подними --max осознанно.

Требования: установленный Node.js и @higgsfield/cli, выполненные
`higgsfield auth login` и `higgsfield workspace set <id>` (делается один раз).
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

MODEL = "seedream_v5_lite"
ASPECT = "16:9"
QUALITY = "high"                 # basic | high (high ≈ 4096px по ширине)
MAX_CONCURRENCY = 8
WAIT_TIMEOUT = "20m"

DEFAULT_MAX_PER_RUN = 10         # защита от случайного слива кредитов

# Пути к Node и CLI (найдены на этой машине). При необходимости поправь.
NODE = r"C:\Program Files\nodejs\node.exe"
CLI_JS = r"C:\Users\xmake\AppData\Roaming\npm\node_modules\@higgsfield\cli\bin\higgsfield.js"

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_SUFFIXES = ("_seg", "_depth", "_mask", "_ai", "_seedream", "_seedream5lite")
DELAY_MIN_S, DELAY_MAX_S = 0.3, 1.0


# ─────────────────────────── УТИЛИТЫ ───────────────────────────
def is_source_frame(p: Path) -> bool:
    if p.suffix.lower() not in IMG_EXT:
        return False
    stem = p.stem.lower()
    return not any(stem.endswith(s) for s in SKIP_SUFFIXES)


def output_path(src: Path) -> Path:
    return src.with_name(src.stem + OUTPUT_SUFFIX + ".png")


def discover_frames(folder: Path):
    frames = sorted(p for p in folder.iterdir() if p.is_file() and is_source_frame(p))
    todo = [p for p in frames if not output_path(p).exists()]
    return frames, todo


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Batch Seedream img2img через Higgsfield CLI.")
    ap.add_argument("files", nargs="*", type=Path,
                    help="Явные картинки (для теста). Если заданы — папка не сканируется.")
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="Файл промпта (по умолч. <input-dir>/prompt.txt или рядом с картинками).")
    ap.add_argument("--max", type=int, default=DEFAULT_MAX_PER_RUN,
                    help=f"МАКС кадров за прогон (защита от слива кредитов, по умолч. {DEFAULT_MAX_PER_RUN}). "
                         f"~1 кредит/кадр. Подними осознанно.")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--quality", default=QUALITY, choices=["basic", "high"])
    ap.add_argument("--aspect", default=ASPECT)
    ap.add_argument("--force", action="store_true", help="Переобрабатывать даже если *_ai.png есть.")
    ap.add_argument("--list", action="store_true", help="Показать план и выйти (без генераций).")
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

    prompt_file = args.prompt_file or (base_dir / "prompt.txt")
    if not prompt_file.is_file():
        raise SystemExit(f"Нет файла промпта: {prompt_file} (укажи --prompt-file)")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    return todo, prompt


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


async def process_one(args, src: Path, prompt: str, sem: asyncio.Semaphore, stats: dict):
    async with sem:
        await asyncio.sleep(random.uniform(DELAY_MIN_S, DELAY_MAX_S))
        cmd = [
            args.node, args.cli, "generate", "create", MODEL,
            "--prompt", prompt,
            "--image-references", str(src),
            "--aspect_ratio", args.aspect,
            "--quality", args.quality,
            "--wait", "--wait-timeout", WAIT_TIMEOUT,
            "--json",
        ]
        print(f"[создаю] {src.name}")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await proc.communicate()
        except Exception as e:
            print(f"  ! {src.name}: не смог запустить CLI: {e}")
            stats["failed"] += 1
            return

        if proc.returncode != 0:
            msg = (err or b"").decode("utf-8", "replace").strip()[:200]
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
            print(f"  [готово] {dest.name}  ({stats['done']} ок, {stats['failed']} ошибок)")
        else:
            stats["failed"] += 1
            print(f"  ! {src.name}: не смог скачать результат")


async def main(args):
    todo, prompt = resolve_jobs(args)

    if args.list:
        capped = todo[:args.max]
        print(f"К обработке за этот прогон: {len(capped)} (из {len(todo)}, лимит --max={args.max})")
        for p in capped:
            print(f"  {p}  ->  {output_path(p).name}")
        print(f"\nПромпт ({len(prompt)} символов): {prompt[:80]}...")
        print(f"Параллельность: {args.concurrency} | quality={args.quality} | aspect={args.aspect}")
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
    stats = {"done": 0, "failed": 0}
    await asyncio.gather(*(process_one(args, src, prompt, sem, stats) for src in batch))

    print(f"\nГотово. Успешно: {stats['done']}, ошибок: {stats['failed']}. "
          f"Осталось необработанных в папке: "
          f"{len(todo) - stats['done']} (перезапусти для следующей порции).")


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\nПрервано. Сохранённые *_ai.png на месте; перезапуск продолжит с оставшихся.")
        sys.exit(130)
