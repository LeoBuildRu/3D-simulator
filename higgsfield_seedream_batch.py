#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
higgsfield_seedream_batch.py
============================
Пакетная img2img-обработка датасета через ВЕБ-ПРИЛОЖЕНИЕ higgsfield.ai
моделью Seedream 5.0 lite (4K + Unlimited), управляя ТВОИМ реальным
залогиненным браузером Chrome через Playwright.

Почему так, а не через API:
  - Seedream 5.0 lite есть только в веб-приложении (в публичном
    platform.higgsfield.ai его нет).
  - Эндпоинт СОЗДАНИЯ задачи (POST /jobs/v2/seedream_v5_lite) защищён
    captcha/bot-протекшеном. Поэтому создание идёт ТОЛЬКО через настоящий
    UI (клик по кнопке), где приложение само прикладывает нужный токен —
    никакой обход защиты не выполняется.
  - Чтение статуса и результата (GET /jobs/{id}) captcha не требует —
    делаем напрямую по Clerk Bearer-токену из window.Clerk. Быстро и дёшево.

Логика:
  1. Находит в папке все НЕ-маски (.png/.jpg/.jpeg/.webp, исключая
     *_seg/_depth/_mask/_ai).
  2. Держит до MAX_CONCURRENCY=8 задач одновременно "в полёте"
     (8 — лимит Unlimited-режима).
  3. Для каждого кадра: заменяет картинку в композиторе -> ждёт аплоад ->
     ставит промпт -> включает Unlimited -> жмёт Generate -> ловит job_id.
  4. Опрашивает задачи; готовые -> скачивает результат и сохраняет рядом
     как <оригинальное_имя>_ai.png. Затем дозагружает новые кадры.
  5. Резюме: уже готовые (_ai.png существует) пропускает.
  6. Случайные задержки 300–1000 мс на действиях "для правдоподобности".

Требования:
  pip install playwright
  (canал chrome использует твой установленный Chrome — отдельный
   chromium ставить не нужно.)

ВАЖНО перед запуском:
  * Полностью ЗАКРОЙ Chrome (Playwright откроет твой профиль, а профиль
    нельзя держать открытым в двух местах — иначе ошибка блокировки).
  * Один раз вручную открой higgsfield.ai/ai/image?model=seedream_v5_lite,
    выставь модель = Seedream 5.0 lite, качество = 4K, Unlimited = ON.
    Эти настройки сохраняются в профиле; скрипт их ПРОВЕРЯЕТ и, если что-то
    не так, остановится с понятным сообщением.
"""

import argparse
import asyncio
import random
import re
import sys
from pathlib import Path

import requests
from playwright.async_api import async_playwright, TimeoutError as PWTimeout

# ─────────────────────────── КОНФИГ ───────────────────────────
INPUT_DIR = Path(r"D:\IQoko\dataset_segmentation_random")   # папка с кадрами
PROMPT_FILE = INPUT_DIR / "prompt.txt"                       # промпт лежит там же
OUTPUT_SUFFIX = "_ai"                                        # <stem>_ai.png

MAX_CONCURRENCY = 8          # лимит Unlimited-режима (проверено)
MODEL = "seedream_v5_lite"
COMPOSER_URL = f"https://higgsfield.ai/ai/image?model={MODEL}"

# ВАЖНО: Chrome 136+ ЗАПРЕЩАЕТ remote-debugging на стандартной папке
# ...\Chrome\User Data. Playwright использует debugging, поэтому НЕЛЬЗЯ
# указывать реальный профиль Chrome — нужна ОТДЕЛЬНАЯ папка профиля.
# В неё ты один раз логинишься через `--setup` (см. ниже); дальше скрипт
# её переиспользует (логин и настройки Seedream/4K/Unlimited сохраняются).
CHROME_USER_DATA_DIR = r"D:\IQoko\hf-chrome-profile"
CHROME_PROFILE = "Default"

# Ожидаемое состояние композитора (persist в профиле). Скрипт проверяет.
EXPECT_MODEL_LABEL = "Seedream 5.0 lite"
EXPECT_QUALITY_LABEL = "4K"

# Тайминги
UPLOAD_TIMEOUT_MS = 90_000
CREATE_TIMEOUT_MS = 60_000
JOB_TIMEOUT_S = 900          # макс. ожидание готовности одной задачи
POLL_INTERVAL_S = 5
DELAY_MIN_S, DELAY_MAX_S = 0.3, 1.0   # "человеческие" задержки

CHALLENGE = "__CHALLENGE__"        # сигнал: упёрлись в Cloudflare-проверку
CHALLENGE_MAX_RETRIES = 4          # сколько раз пере-пробовать кадр после ручного прохождения

FNF = "https://fnf.higgsfield.ai"
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_SUFFIXES = ("_seg", "_depth", "_mask", "_ai", "_seedream", "_seedream5lite")


# ─────────────────────────── УТИЛИТЫ ───────────────────────────
async def human_delay(a=DELAY_MIN_S, b=DELAY_MAX_S):
    await asyncio.sleep(random.uniform(a, b))


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
    ap = argparse.ArgumentParser(
        description="Пакетный img2img через веб-UI higgsfield.ai (Seedream 5.0 lite, 4K, Unlimited).")
    ap.add_argument("files", nargs="*", type=Path,
                    help="Явные пути к картинкам (для теста). Если заданы — папка не сканируется.")
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                    help=f"Папка с кадрами (по умолчанию {INPUT_DIR}).")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="Файл промпта (по умолчанию <input-dir>/prompt.txt или рядом с картинками).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Обработать только первые N кадров (для теста).")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY,
                    help=f"Сколько задач держать в полёте (по умолчанию {MAX_CONCURRENCY}).")
    ap.add_argument("--profile", default=CHROME_PROFILE,
                    help=f'Профиль Chrome (по умолчанию "{CHROME_PROFILE}").')
    ap.add_argument("--user-data-dir", default=CHROME_USER_DATA_DIR,
                    help="Папка User Data Chrome.")
    ap.add_argument("--force", action="store_true",
                    help="Переобрабатывать даже если *_ai.png уже существует.")
    ap.add_argument("--list", action="store_true",
                    help="Только показать, что будет обработано, и выйти (без браузера).")
    ap.add_argument("--setup", action="store_true",
                    help="Открыть браузер на отдельном профиле для входа/настройки "
                         "(залогинься в higgsfield, выставь Seedream 5.0 lite + 4K + "
                         "Unlimited, затем нажми Enter в консоли).")
    ap.add_argument("--clone-profile", action="store_true",
                    help="Скопировать вход/настройки из реального профиля Chrome в "
                         "автоматизационный профиль (Chrome должен быть ЗАКРЫТ).")
    ap.add_argument("--source-user-data",
                    default=str(Path.home() / r"AppData\Local\Google\Chrome\User Data"),
                    help="Откуда копировать (реальный User Data Chrome).")
    ap.add_argument("--source-profile", default="Default",
                    help='Какой профиль копировать (по умолчанию "Default").')
    return ap.parse_args(argv)


def clone_profile(args):
    """Копирует cookies/Local Storage/Local State из реального профиля Chrome
    в автоматизационный. Chrome должен быть закрыт (иначе файлы заблокированы)."""
    import shutil

    src_root = Path(args.source_user_data)
    src_prof = src_root / args.source_profile
    dst_root = Path(args.user_data_dir)
    dst_prof = dst_root / args.profile

    if not src_prof.is_dir():
        raise SystemExit(f"Исходный профиль не найден: {src_prof}")

    dst_prof.mkdir(parents=True, exist_ok=True)

    # что копируем (относительно папки профиля); плюс Local State из корня
    prof_items = [
        "Network/Cookies", "Network/Cookies-journal",
        "Local Storage", "Session Storage", "IndexedDB",
        "Preferences", "Secure Preferences",
        "Login Data", "Login Data-journal", "Web Data",
    ]
    copied, skipped = [], []

    def copy_item(src: Path, dst: Path):
        if not src.exists():
            skipped.append(src.name)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(src.name)

    # Local State (ключ шифрования cookies) — ОБЯЗАТЕЛЬНО
    copy_item(src_root / "Local State", dst_root / "Local State")
    for rel in prof_items:
        copy_item(src_prof / rel, dst_prof / rel)

    print(f"Скопировано: {', '.join(copied) or '—'}")
    if skipped:
        print(f"Пропущено (нет в источнике): {', '.join(skipped)}")
    print(f"\nПрофиль подготовлен: {dst_root}\n"
          "Проверь вход тестовым запуском на 1 кадре. Если окажешься разлогинен "
          "(Chrome иногда не переносит cookies из-за app-bound шифрования) — "
          "используй  --setup  и войди вручную один раз.")


def resolve_jobs(args):
    """Возвращает (todo_frames, prompt_text). Учитывает явные файлы/лимит/резюме."""
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
        print(f"Всего кадров-исходников: {len(all_frames)} | к обработке: {len(todo)} "
              f"| уже готовы: {len(all_frames) - len(todo)}")

    if args.limit is not None:
        todo = todo[:args.limit]

    prompt_file = args.prompt_file or (base_dir / "prompt.txt")
    if not prompt_file.is_file():
        raise SystemExit(f"Нет файла промпта: {prompt_file} (укажи --prompt-file)")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    return todo, prompt


# ─────────────────── ВЗАИМОДЕЙСТВИЕ С СТРАНИЦЕЙ ───────────────────
# JS выполняется в контексте страницы. Собран из реально снятой структуры UI.

JS_ENSURE_UNLIMITED = r"""
() => {
  const sw = document.querySelector('button[role=switch]');
  if (!sw) return {ok:false, reason:'switch-not-found'};
  const on = sw.getAttribute('aria-checked') === 'true';
  if (!on) sw.click();
  return {ok:true, wasOn:on};
}
"""

JS_READ_STATE = r"""
() => {
  const tb = document.querySelector('[role=textbox]');
  const sw = document.querySelector('button[role=switch]');
  const text = document.body.innerText || '';
  const composer = tb ? tb.closest('div') : null;
  return {
    hasTextbox: !!tb,
    unlimited: sw ? sw.getAttribute('aria-checked') === 'true' : null,
    // грубая проверка наличия подписей модели/качества на экране композитора
    bodyHasModel: text.includes(arguments[0] ?? ''),
  };
}
"""

# Клик по кнопке отправки: большая кнопка композитора, текст начинается
# с "Unlimited" (когда Unlimited ON) или "Generate". role=switch исключаем.
JS_CLICK_SUBMIT = r"""
() => {
  const tb = document.querySelector('[role=textbox]');
  let root = tb;
  for (let i=0;i<8 && root.parentElement;i++) root = root.parentElement;
  const btns = [...root.querySelectorAll('button')].filter(b =>
      b.getAttribute('role') !== 'switch');
  // ищем самую "большую" кнопку с нужным текстом
  const cand = btns.filter(b => /^(unlimited|generate)/i.test(b.textContent.trim()));
  const btn = cand.sort((a,b) =>
      (b.offsetWidth*b.offsetHeight) - (a.offsetWidth*a.offsetHeight))[0];
  if (!btn) return {ok:false, reason:'submit-not-found'};
  if (btn.disabled) return {ok:false, reason:'submit-disabled'};
  btn.click();
  return {ok:true, label: btn.textContent.trim().slice(0,20)};
}
"""

JS_MAIN_IMG_SRC = r"""
() => {
  const tb = document.querySelector('[role=textbox]');
  let root = tb; for (let i=0;i<8 && root.parentElement;i++) root = root.parentElement;
  const img = root.querySelector('img');
  return img ? (img.src || '').split('?')[0] : null;
}
"""

JS_SLOT_IMG = r"""
() => {
  // картинка ИМЕННО в слоте загрузки (кнопка/лейбл вокруг первого file input),
  // а не случайный <img> со страницы
  const inp = document.querySelector('input[type=file]');
  if (!inp) return null;
  const box = inp.closest('button') || inp.closest('label') || inp.parentElement;
  const img = box ? box.querySelector('img') : null;
  if (img) return (img.src || '').split('?')[0];
  // некоторые слоты рисуют превью через background-image
  const bg = box ? getComputedStyle(box).backgroundImage : '';
  return (bg && bg !== 'none') ? bg.split('?')[0] : null;
}
"""

JS_GET_JOB = r"""
async (id) => {
  const t = await window.Clerk.session.getToken();
  const r = await fetch('https://fnf.higgsfield.ai/jobs/' + id, {
    headers: { authorization: 'Bearer ' + t }
  });
  if (!r.ok) return { __status: r.status };
  return await r.json();
}
"""

JS_READ_PROMPT = r"""
() => {
  const tb = document.querySelector('[role=textbox]');
  return tb ? (tb.innerText || '') : null;
}
"""


async def get_clerk_token(page) -> str:
    return await page.evaluate("() => window.Clerk.session.getToken()")


async def dismiss_cookie_banner(page):
    """Убирает баннер согласия на cookies (перехватывает клики в свежем
    профиле). Приватность по умолчанию: отклоняем необязательные; если кнопки
    нет — просто удаляем оверлей, ничего не принимая."""
    for sel in ("#cookiescript_reject", "#cookiescript_close"):
        try:
            loc = page.locator(sel)
            if await loc.count() and await loc.first.is_visible():
                await loc.first.click(timeout=4000)
                await page.wait_for_timeout(400)
                return
        except Exception:
            pass
    # запасной вариант — удалить оверлей, не давая согласия
    try:
        await page.evaluate("""() => {
            for (const id of ['cookiescript_injected_wrapper','cookiescript_injected','cookiescript_badge']) {
                const el = document.getElementById(id);
                if (el) el.remove();
            }
        }""")
    except Exception:
        pass


def _norm(s: str) -> str:
    """Нормализация для сравнения промптов (пробелы/невидимые символы)."""
    return re.sub(r"\s+", " ", (s or "").replace("​", "")).strip()


async def set_prompt_once(page, prompt: str):
    """Ставит промпт ОДИН раз реальной клавиатурой (Lexical не принимает
    синтетические execCommand/paste надёжно). Промпт для всех кадров один."""
    tb = page.locator("[role=textbox]").first
    await tb.click()
    await human_delay()
    await page.keyboard.press("Control+a")
    await page.keyboard.press("Delete")
    await human_delay(0.15, 0.4)
    await page.keyboard.insert_text(prompt)          # CDP Input.insertText — реальный ввод
    await tb.evaluate("el => el.dispatchEvent(new Event('input', {bubbles:true}))")
    await page.wait_for_timeout(300)
    got = await page.evaluate(JS_READ_PROMPT)
    if _norm(got) != _norm(prompt):
        raise SystemExit(
            f"Не удалось выставить промпт (в поле {len(got or '')} симв., "
            f"ожидалось {len(prompt)}). Проверь окно браузера."
        )
    print(f"Промпт выставлен ({len(prompt)} символов).")


async def extract_job_id(page, create_resp_json) -> str | None:
    """Из ответа создания вытаскиваем job_id: берём все uuid и оставляем тот,
    что реально опрашивается как задача seedream."""
    text = str(create_resp_json)
    seen = []
    for uid in UUID_RE.findall(text):
        if uid in seen:
            continue
        seen.append(uid)
    for uid in seen:
        try:
            data = await page.evaluate(JS_GET_JOB, uid)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("job_set_type"):
            return uid
    return seen[0] if seen else None


async def preflight(page):
    """Проверяем, что композитор в нужном состоянии (persist в профиле)."""
    try:
        await page.wait_for_selector("[role=textbox]", timeout=30_000)
    except PWTimeout:
        raise SystemExit(
            "Не вижу композитор — вероятно, в этом профиле ты не залогинен в "
            "higgsfield.ai.\nЗапусти сначала настройку:  "
            "python higgsfield_seedream_batch.py --setup"
        )
    # проверка авторизации: композитор рисуется даже без логина, поэтому
    # проверяем реальную сессию Clerk
    try:
        signed_in = await page.evaluate(
            "() => !!(window.Clerk && window.Clerk.session && window.Clerk.user)")
    except Exception:
        signed_in = False
    if not signed_in:
        raise SystemExit(
            "Ты НЕ залогинен в higgsfield в этом профиле (клонирование не переносит "
            "сессию из-за шифрования cookies Chrome).\nЗалогинься один раз:  "
            "python higgsfield_seedream_batch.py --setup"
        )

    body = await page.evaluate("() => document.body.innerText || ''")
    problems = []
    if EXPECT_MODEL_LABEL not in body:
        problems.append(f'не вижу модель "{EXPECT_MODEL_LABEL}"')
    if EXPECT_QUALITY_LABEL not in body:
        problems.append(f'не вижу качество "{EXPECT_QUALITY_LABEL}"')
    if problems:
        raise SystemExit(
            "ПРЕДПОЛЁТНАЯ ПРОВЕРКА НЕ ПРОЙДЕНА: " + "; ".join(problems) +
            ".\nОткрой higgsfield.ai/ai/image?model=seedream_v5_lite вручную, "
            "выставь Seedream 5.0 lite + 4K + Unlimited и перезапусти скрипт."
        )


async def solve_challenge_pause(page):
    """Ставит паузу и ждёт, пока ПОЛЬЗОВАТЕЛЬ вручную пройдёт Cloudflare-проверку
    в окне браузера. Скрипт captcha не решает и не обходит."""
    try:
        await page.bring_to_front()
    except Exception:
        pass
    print("\n" + "=" * 60)
    print("⚠  Cloudflare требует проверку (bot-защита на создании задачи).")
    print("   1) Перейди в ОКНО браузера, которое открыл скрипт.")
    print("   2) Пройди проверку Cloudflare. Если её не видно — обнови")
    print("      страницу (F5) или зайди на higgsfield.ai, чтобы она появилась.")
    print("   3) Дождись, что страница композитора снова открыта и рабочая.")
    print("   4) Вернись сюда и нажми Enter — продолжу с этого же кадра.")
    print("=" * 60)
    await asyncio.get_event_loop().run_in_executor(
        None, input, "Нажми Enter после прохождения проверки... ")


async def reinit_composer(page, prompt: str):
    """Пере-инициализирует композитор после ручного прохождения проверки
    (навигация могла сбросить страницу): открыть страницу, убрать баннер,
    поставить промпт."""
    await page.goto(COMPOSER_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(2000)
    await dismiss_cookie_banner(page)
    try:
        await page.wait_for_selector("[role=textbox]", timeout=20_000)
    except PWTimeout:
        return
    cur = await page.evaluate(JS_READ_PROMPT)
    if _norm(cur) != _norm(prompt):
        await set_prompt_once(page, prompt)


async def submit_with_challenge(page, src: Path, prompt: str) -> dict | None:
    """Обёртка над submit_one: при Cloudflare-проверке ставит паузу для ручного
    прохождения и повторяет кадр."""
    for attempt in range(CHALLENGE_MAX_RETRIES + 1):
        res = await submit_one(page, src, prompt)
        if res != CHALLENGE:
            return res
        if attempt == CHALLENGE_MAX_RETRIES:
            print(f"  ! {src.name}: проверка не пройдена после "
                  f"{CHALLENGE_MAX_RETRIES} попыток, пропускаю кадр")
            return None
        await solve_challenge_pause(page)
        await reinit_composer(page, prompt)
    return None


async def submit_one(page, src: Path, prompt: str) -> dict | None:
    """Загружает картинку в композитор и создаёт задачу. Возвращает
    {'job_id':..., 'src':..., 'out':...}, None при неудаче, или строку
    CHALLENGE при Cloudflare-проверке (обрабатывает submit_with_challenge)."""
    before_slot = await page.evaluate(JS_SLOT_IMG)

    # сетевые сигналы регистрируем ЗАРАНЕЕ (до выбора файла)
    uploaded = {"done": False}
    seen = []

    def _on_resp(r):
        try:
            u = r.url
            m = r.request.method
            if "higgsfield" in u and m in ("POST", "PUT"):
                seen.append(f"{m} {u.split('?')[0]}")
            if ("upload.higgsfield.ai" in u or "/media/upload" in u
                    or "/media" in u and m == "POST"
                    or (m == "PUT" and "higgsfield" in u)):
                uploaded["done"] = True
        except Exception:
            pass

    page.on("response", _on_resp)
    try:
        await human_delay()
        # 1) загрузка. Диагностика показала: НЕ каждый file-input реагирует —
        #    нужен тот, чей предок открывает системный диалог (это «добавить
        #    изображение»). Перебираем инпуты и используем первый рабочий.
        method_used = None
        n_inputs = await page.locator('input[type=file]').count()
        for i in range(n_inputs):
            anc = page.locator('input[type=file]').nth(i).locator(
                "xpath=ancestor::*[self::label or self::button][1]")
            try:
                async with page.expect_file_chooser(timeout=3500) as fc_info:
                    await anc.click()
                chooser = await fc_info.value
                await chooser.set_files(str(src))
                method_used = f"chooser#input{i}"
                break
            except Exception:
                continue
        if method_used is None:
            # запасной вариант — прямая вставка в каждый input
            for i in range(n_inputs):
                try:
                    await page.locator('input[type=file]').nth(i).set_input_files(str(src))
                    method_used = f"set#input{i}"
                    break
                except Exception:
                    continue
        if method_used is None:
            print(f"  ! {src.name}: не смог отдать файл (диалог не открылся ни у "
                  f"одного из {n_inputs} инпутов)")
            return None

        # 2) ждём аплод: сеть ИЛИ появление превью в слоте
        loop = asyncio.get_event_loop()
        deadline = loop.time() + UPLOAD_TIMEOUT_MS / 1000
        ok = False
        while loop.time() < deadline:
            if uploaded["done"]:
                ok = True
                break
            slot = await page.evaluate(JS_SLOT_IMG)
            if slot and slot != before_slot:
                ok = True
                break
            await asyncio.sleep(0.5)
        if not ok:
            after_slot = await page.evaluate(JS_SLOT_IMG)
            n_inputs = await page.locator('input[type=file]').count()
            print(f"  ! {src.name}: не дождался загрузки картинки, пропускаю\n"
                  f"    [способ={method_used}, net={uploaded['done']}, "
                  f"slot_before={'есть' if before_slot else 'нет'}, "
                  f"slot_after={'есть' if after_slot else 'нет'}, "
                  f"file_inputs={n_inputs}]\n"
                  f"    [сеть higgsfield POST/PUT за время ожидания: "
                  f"{seen[-12:] or 'ничего'}]")
            return None
    finally:
        page.remove_listener("response", _on_resp)

    await page.wait_for_timeout(800)
    await human_delay()

    # 3) промпт НЕ ставим заново, но подстрахуемся: если поле опустело
    #    после замены картинки — восстановим (реальной клавиатурой).
    cur = await page.evaluate(JS_READ_PROMPT)
    if _norm(cur) != _norm(prompt):
        print("  · промпт слетел — переставляю")
        await set_prompt_once(page, prompt)

    # 4) Unlimited ON (дёшево подтверждаем каждый раз)
    await page.evaluate(JS_ENSURE_UNLIMITED)
    await human_delay()

    # 5) клик Generate + ловим ответ создания
    try:
        async with page.expect_response(
            lambda r: f"/jobs/v2/{MODEL}" in r.url and r.request.method == "POST",
            timeout=CREATE_TIMEOUT_MS,
        ) as resp_info:
            res = await page.evaluate(JS_CLICK_SUBMIT)
            if not res.get("ok"):
                print(f"  ! {src.name}: не смог нажать Generate ({res.get('reason')})")
                return None
        resp = await resp_info.value
    except PWTimeout:
        print(f"  ! {src.name}: не дождался ответа создания задачи")
        return None

    if resp.status != 200:
        body = ""
        try:
            body = (await resp.text())[:400]
        except Exception:
            pass
        low = body.lower()
        is_challenge = (resp.status in (403, 429, 503)
                        or "<html" in low or "captcha" in low or "#cmsg" in body
                        or "cloudflare" in low or "just a moment" in low)
        if is_challenge:
            return CHALLENGE          # обработает обёртка submit_with_challenge
        print(f"  ! {src.name}: создание вернуло {resp.status}. {body[:120]}")
        return None

    try:
        data = await resp.json()
    except Exception:
        data = await resp.text()

    job_id = await extract_job_id(page, data)
    if not job_id:
        print(f"  ! {src.name}: не нашёл job_id в ответе создания")
        return None

    # страховка: задача должна содержать РОВНО одну картинку (замена, не
    # добавление второго референса). Если >1 — предупреждаем громко.
    try:
        detail = await page.evaluate(JS_GET_JOB, job_id)
        n = len((detail.get("params") or {}).get("medias") or [])
        if n != 1:
            print(f"  ⚠ {src.name}: в задаче {n} картинок вместо 1! "
                  f"Похоже, слот не заменяется, а добавляется. ОСТАНОВИСЬ и проверь UI.")
    except Exception:
        pass

    return {"job_id": job_id, "src": src, "out": output_path(src)}


def download_result(url: str, dest: Path) -> bool:
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and r.content:
                dest.write_bytes(r.content)
                return True
        except Exception:
            pass
        # небольшой бэкофф
        import time
        time.sleep(1.5 * (attempt + 1))
    return False


async def check_job(page, job: dict):
    """Возвращает 'pending' | 'done' | 'failed'. При done — качает результат."""
    try:
        data = await page.evaluate(JS_GET_JOB, job["job_id"])
    except Exception:
        return "pending"
    if not isinstance(data, dict):
        return "pending"
    status = data.get("status")
    if status == "completed":
        url = (((data.get("results") or {}).get("raw")) or {}).get("url")
        if not url:
            return "failed"
        ok = download_result(url, job["out"])
        return "done" if ok else "failed"
    if status in ("failed", "nsfw", "rejected", "canceled", "cancelled"):
        return "failed"
    return "pending"


# ─────────────────────────── ГЛАВНЫЙ ЦИКЛ ───────────────────────────
async def run_setup(args):
    """Открывает отдельный профиль-браузер для ручного входа и настройки."""
    Path(args.user_data_dir).mkdir(parents=True, exist_ok=True)
    print(f"Открываю браузер на профиле: {args.user_data_dir}")
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=args.user_data_dir,
            channel="chrome",
            headless=False,
            args=[f"--profile-directory={args.profile}"],
            no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(COMPOSER_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        await dismiss_cookie_banner(page)
        print("\n=== НАСТРОЙКА ===\n"
              "1) Залогинься в higgsfield.ai (если ещё не вошёл).\n"
              "2) Убедись, что выбрано: Seedream 5.0 lite + 4K + Unlimited (ON).\n"
              "3) Вернись сюда и нажми Enter — настройки сохранятся в профиль.\n")
        await asyncio.get_event_loop().run_in_executor(
            None, input, "Нажми Enter, когда всё готово... ")
        await ctx.close()
    print("Профиль сохранён. Теперь можно запускать обработку "
          "(без --setup). Chrome при запуске скрипта должен быть закрыт.")


async def main(args):
    if args.clone_profile:
        clone_profile(args)
        return
    if args.setup:
        await run_setup(args)
        return

    todo, prompt = resolve_jobs(args)
    concurrency = max(1, args.concurrency)

    if args.list:
        print(f"К обработке ({len(todo)}):")
        for p in todo:
            print(f"  {p}  ->  {output_path(p).name}")
        print(f"\nПромпт ({len(prompt)} символов): {prompt[:80]}...")
        print(f"Параллельность: {concurrency} | профиль: {args.profile}")
        return

    if not todo:
        print("Нечего обрабатывать — все кадры уже имеют *_ai.png (или используй --force)")
        return

    queue = list(todo)
    in_flight: list[dict] = []
    done = failed = 0

    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            user_data_dir=args.user_data_dir,
            channel="chrome",
            headless=False,
            args=[f"--profile-directory={args.profile}"],
            no_viewport=True,
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(COMPOSER_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)
        await dismiss_cookie_banner(page)
        await preflight(page)
        await dismiss_cookie_banner(page)
        await set_prompt_once(page, prompt)
        print("Предполётная проверка пройдена. Старт.\n")

        total = len(todo)
        while queue or in_flight:
            # дозаполняем "полёт" до лимита
            while queue and len(in_flight) < concurrency:
                src = queue.pop(0)
                print(f"[создаю] {src.name}  (в полёте будет {len(in_flight)+1})")
                job = await submit_with_challenge(page, src, prompt)
                if job:
                    in_flight.append(job)
                else:
                    failed += 1
                await human_delay()

            if not in_flight:
                continue

            # опрашиваем задачи
            await asyncio.sleep(POLL_INTERVAL_S)
            still = []
            for job in in_flight:
                st = await check_job(page, job)
                if st == "done":
                    done += 1
                    print(f"[готово ] {job['out'].name}  "
                          f"({done}/{len(todo)} готово, {failed} ошибок)")
                elif st == "failed":
                    failed += 1
                    print(f"[ОШИБКА ] {job['src'].name} (job {job['job_id'][:8]})")
                else:
                    still.append(job)
            in_flight = still

        print(f"\nГотово. Успешно: {done}, ошибок: {failed}, "
              f"итого обработано в этой сессии: {done + failed}")
        await ctx.close()


if __name__ == "__main__":
    try:
        asyncio.run(main(parse_args()))
    except KeyboardInterrupt:
        print("\nПрервано пользователем. Уже сохранённые *_ai.png на месте; "
              "перезапуск продолжит с оставшихся кадров.")
        sys.exit(130)
