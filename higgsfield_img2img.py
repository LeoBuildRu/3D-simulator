#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пакетная обработка снимков датасета через higgsfield.ai (img2img, Seedream 5.0 lite).

Берёт из папки датасета N случайных СНИМКОВ (только фотореалистичные кадры, без
масок сегментации / depth), прогоняет каждый через img2img-нейросеть Higgsfield с
заданным промптом и сохраняет результат РЯДОМ с оригиналом с тем же именем + суффикс
`_ai` (например: r0000_..._317605.png -> r0000_..._317605_ai.png).

Ключи Higgsfield:
    - Возьмите их в дашборде https://cloud.higgsfield.ai/ (раздел API -> Create key).
      Higgsfield выдаёт пару key:secret.
    - Положите в config/higgsfield.json (файл в .gitignore), см. higgsfield.example.json,
      ЛИБО задайте переменные окружения HF_API_KEY и HF_API_SECRET
      (или одну HF_KEY="key:secret").

Установка SDK:
    .venv\\Scripts\\python.exe -m pip install higgsfield-client requests

Примеры запуска (из корня репозитория):
    .venv\\Scripts\\python.exe higgsfield_img2img.py --count 10
    .venv\\Scripts\\python.exe higgsfield_img2img.py --folder renders/dataset_segmentation_random --count 5 --prompt "..."
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

# консоль Windows (cp1251) не умеет печатать → ✓ и кириллицу — переключаем на UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------------------
# Конфигурация / ключи
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config" / "higgsfield.json"

# Суффиксы, по которым файл считается НЕ снимком (маска/карта/уже обработанный).
NON_SHOT_SUFFIXES = ("_seg.png", "_depth.png", "_ai.png", "_mask.png")

DEFAULT_FOLDER = "renders/dataset_segmentation_random"
# img2img-модель Higgsfield (проверено вживую, что принимает image_url и
# реально редактирует кадр). Доступные на аккаунте img2img-модели:
#   reve/edit                     — редактирование картинки по промпту (по умолчанию)
#   higgsfield-ai/soul/reference  — генерация с опорой на референс-изображение
# Seedream через публичный API (platform.higgsfield.ai) НЕ отдаётся — есть только
# в веб-приложении. Точный список моделей вашего аккаунта = cloud.higgsfield.ai.
DEFAULT_ENDPOINT = "reve/edit"
DEFAULT_PROMPT = (
    "photorealistic industrial photo, keep composition and objects, "
    "add realistic weathering, dirt, rust, natural daylight, high detail"
)


def load_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] не удалось прочитать {CONFIG_PATH}: {e}")
    # env перекрывает файл
    key = os.getenv("HF_API_KEY") or cfg.get("api_key") or ""
    secret = os.getenv("HF_API_SECRET") or cfg.get("api_secret") or ""
    combined = os.getenv("HF_KEY")
    if combined and ":" in combined:
        key, secret = combined.split(":", 1)
    cfg["api_key"] = key
    cfg["api_secret"] = secret
    return cfg


def ensure_sdk():
    try:
        import higgsfield_client  # noqa: F401
    except ImportError:
        sys.exit(
            "Не установлен SDK. Выполните:\n"
            "    .venv\\Scripts\\python.exe -m pip install higgsfield-client requests"
        )


# ---------------------------------------------------------------------------
# Выбор снимков
# ---------------------------------------------------------------------------
def is_shot(p: Path) -> bool:
    if p.suffix.lower() != ".png":
        return False
    name = p.name.lower()
    return not any(name.endswith(s) for s in NON_SHOT_SUFFIXES)


def pick_shots(folder: Path, count: int, seed=None) -> list[Path]:
    shots = sorted(p for p in folder.iterdir() if p.is_file() and is_shot(p))
    # пропускаем те, у которых _ai уже есть
    shots = [p for p in shots if not (p.with_name(p.stem + "_ai.png")).exists()]
    if not shots:
        return []
    if seed is not None:
        random.seed(seed)
    if count >= len(shots):
        return shots
    return random.sample(shots, count)


# ---------------------------------------------------------------------------
# Обработка одного снимка
# ---------------------------------------------------------------------------
def process_one(src: Path, cfg: dict) -> Path | None:
    import higgsfield_client

    print(f"  → загрузка {src.name} ...")
    image_url = higgsfield_client.upload_file(str(src))

    # Имя параметра с входной картинкой у img2img-модели зависит от модели
    # (смотрите на странице модели в Models Gallery -> cloud.higgsfield.ai).
    # Обычно это image_url (строка) либо image_urls / input_images (список).
    img_param = cfg.get("image_param", "image_url")
    args = {"prompt": cfg["prompt"]}
    if img_param.endswith("s") or img_param in ("input_images", "reference_images"):
        args[img_param] = [image_url]
    else:
        args[img_param] = image_url
    if cfg.get("resolution"):
        args["resolution"] = cfg["resolution"]
    if cfg.get("aspect_ratio"):
        args["aspect_ratio"] = cfg["aspect_ratio"]

    print(f"  → генерация ({cfg['endpoint']}) ...")
    result = higgsfield_client.subscribe(cfg["endpoint"], arguments=args)

    images = result.get("images") or result.get("output") or []
    if not images:
        print(f"  [err] пустой ответ для {src.name}: {result}")
        return None
    out_url = images[0]["url"] if isinstance(images[0], dict) else images[0]

    dst = src.with_name(src.stem + "_ai.png")
    resp = requests.get(out_url, timeout=cfg.get("timeout", 180))
    resp.raise_for_status()
    dst.write_bytes(resp.content)
    print(f"  ✓ сохранено {dst.name}")
    return dst


def main():
    cfg_file = load_config()

    ap = argparse.ArgumentParser(description="Higgsfield img2img по датасету")
    ap.add_argument("--folder", default=cfg_file.get("folder", DEFAULT_FOLDER),
                    help="папка датасета со снимками")
    ap.add_argument("--count", type=int, required=True,
                    help="сколько снимков обработать (случайная выборка)")
    ap.add_argument("--prompt", default=cfg_file.get("prompt", DEFAULT_PROMPT))
    ap.add_argument("--endpoint", default=cfg_file.get("endpoint", DEFAULT_ENDPOINT),
                    help="id модели Higgsfield (img2img seedream)")
    ap.add_argument("--seed", type=int, default=None, help="seed для выборки снимков")
    args = ap.parse_args()

    if not cfg_file.get("api_key") or not cfg_file.get("api_secret"):
        sys.exit(
            "Нет ключей Higgsfield. Впишите api_key/api_secret в config/higgsfield.json "
            "(см. config/higgsfield.example.json) или задайте HF_API_KEY / HF_API_SECRET."
        )

    if not args.endpoint:
        sys.exit("Не задан id модели (endpoint) в config/higgsfield.json или флаге --endpoint.")

    # SDK читает ключи из окружения
    os.environ.setdefault("HF_API_KEY", cfg_file["api_key"])
    os.environ.setdefault("HF_API_SECRET", cfg_file["api_secret"])
    ensure_sdk()

    folder = (REPO_ROOT / args.folder) if not Path(args.folder).is_absolute() else Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Папка не найдена: {folder}")

    cfg = {
        "endpoint": args.endpoint,
        "prompt": args.prompt,
        "image_param": cfg_file.get("image_param", "image_url"),
        # шлём только если явно заданы в конфиге — не все img2img-модели их принимают
        "resolution": cfg_file.get("resolution"),
        "aspect_ratio": cfg_file.get("aspect_ratio"),
        "timeout": cfg_file.get("timeout", 180),
    }

    shots = pick_shots(folder, args.count, args.seed)
    if not shots:
        sys.exit("Не найдено необработанных снимков (все либо маски, либо уже с _ai).")

    print(f"Папка: {folder}")
    print(f"К обработке: {len(shots)} снимк(ов). Эндпоинт: {cfg['endpoint']}")
    ok = 0
    for i, src in enumerate(shots, 1):
        print(f"[{i}/{len(shots)}] {src.name}")
        try:
            if process_one(src, cfg):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [err] {src.name}: {e}")
        time.sleep(0.5)  # мягкий троттлинг

    print(f"\nГотово: {ok}/{len(shots)} обработано.")


if __name__ == "__main__":
    main()
