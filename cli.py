#!/usr/bin/env python
"""CLI для генерации датасетов без Qt-интерфейса.

Пример:
    python cli.py --model FAW-J6-8x4-tall --preset 2 --mode seg-random-bg-cloth -n 5

ПОЧЕМУ ОТДЕЛЬНЫЙ ВХОД, А НЕ КНОПКА В UI
---------------------------------------
Генерация датасета через Qt-окно ломалась структурно, по трём независимым
причинам — и все три здесь отсутствуют по построению:

1) ОКНО PANDA ВСТРОЕНО ДОЧЕРНИМ HWND В QT-ВИДЖЕТ (main_window: winId() +
   SetWindowPos). Чтение такого окна через win.getScreenshot() возвращает не
   GL-поверхность, а ПИКСЕЛИ РАБОЧЕГО СТОЛА под областью окна — отсюда
   «захватывает GitHub, VSCode, проводник — что угодно, кроме сцены».
   Здесь окно создаётся БЕЗ родителя (parent_hwnd=0) — обычное top-level окно
   Panda, чтение которого корректно (проверено тестами).

2) RECURSIVE poll(). В UI кадры гонит QTimer -> taskMgr.step(), а датасетный
   цикл внутри себя звал QApplication.processEvents(), который повторно
   входил в тот же taskMgr.step(). Panda ругалась «Ignoring recursive poll()
   within another task» и ПРОПУСКАЛА кадр — сцена переставала обновляться.
   Здесь Qt нет вообще: кадры гонит только наш цикл, ровно в одном месте,
   рекурсия невозможна.

3) Никаких QTimer'ов, дёргающих сцену параллельно съёмке (depth-overlay,
   телеметрия, color-mirror).

Итог: тот же код рендера (RendererUtils.save_single_render), но в среде, где
захват кадра достоверен.
"""

import argparse
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Консоль Windows по умолчанию cp1251 — русские сообщения превращаются в
# мусор. Переключаем потоки на UTF-8 (Python 3.7+).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Режимы датасета: имя -> флаги, которые уходят в save_single_render.
DATASET_MODES = {
    "seg": dict(dataset_type="segmentation",
                random_background=False, cloth=False),
    "seg-bg": dict(dataset_type="segmentation",
                   random_background=True, cloth=False),
    "seg-cloth": dict(dataset_type="segmentation",
                      random_background=False, cloth=True),
    # то, что просили: случайная сегментация + случайный фон + ткань
    "seg-random-bg-cloth": dict(dataset_type="segmentation",
                                random_background=True, cloth=True),
    "depth": dict(dataset_type="depth",
                  random_background=False, cloth=False),
}

# Рамки случайной вариации позы камеры — те же, что в UI-датасете
# (_run_random_seg_dataset), чтобы выборки были сопоставимы.
OFFSET_M = 0.05
ANG_DEG = 10.0


def log(msg):
    print(f"[cli] {msg}", flush=True)


# ----------------------------------------------------------------------
# Перечисление доступных моделей / пресетов (для --list)
# ----------------------------------------------------------------------
def list_models():
    from src.ui.panel_data import load_model_sets
    return load_model_sets()


def load_camera_presets():
    import json
    path = os.path.join(PROJECT_ROOT, "presets", "camera_presets.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("presets", [])


def do_list():
    print("\nМодели (--model <ключ>):")
    for key, display in list_models():
        print(f"  {key:<45} {display}")

    print("\nКамерные пресеты (--preset <N>, нумерация с 1):")
    for i, p in enumerate(load_camera_presets(), start=1):
        if not p:
            print(f"  {i}: <пусто>")
            continue
        pos = p.get("pos", [])
        fov = p.get("fov")
        pos_s = ", ".join(f"{v:.2f}" for v in pos) if pos else "?"
        print(f"  {i}: pos=({pos_s}) fov={fov} model={p.get('model')}")

    print("\nРежимы датасета (--mode):")
    for name, flags in DATASET_MODES.items():
        print(f"  {name:<22} {flags}")
    print()


# ----------------------------------------------------------------------
# Запуск приложения БЕЗ Qt
# ----------------------------------------------------------------------
def build_app(graphics_preset, size):
    """Создать MyApp с СОБСТВЕННЫМ top-level окном (parent_hwnd=0).

    Именно отсутствие родительского HWND делает захват кадра достоверным —
    см. заголовок модуля, пункт (1).
    """
    import main as main_module

    # Активный сервер берём ИЗ КОНФИГА, как это делает main.main(). Значения по
    # умолчанию в MyApp (порт 9998) указывают на другой сервис — с ними все
    # запросы моделей/текстур возвращают 503, модель не скачивается и
    # perform_AABB_plane падает.
    tls_host, tls_port = main_module.load_tls_config(main_module.base_path)
    log(f"сервер: {tls_host}:{tls_port}")

    w, h = size
    app = main_module.MyApp(
        parent_hwnd=0,                 # 0 => окно верхнего уровня, не встроенное
        init_size=(w, h),
        tls_host=tls_host,
        tls_port=tls_port,
        graphics_preset=graphics_preset,
    )

    # Текстурные наборы: тот же путь, что и в main.main().
    app.texture_sets = {}
    try:
        from src.ui import panel_data as _panel_data
        tex_cfg = app.tls_client.get_textures_config()
        if isinstance(tex_cfg, dict) and tex_cfg:
            app.texture_sets = tex_cfg
            _panel_data.set_texture_sets_cache(tex_cfg)
            log(f"текстурные наборы с сервера: {len(tex_cfg)}")
    except Exception as exc:
        log(f"текстурные наборы недоступны: {exc}")

    return app


def step(app, frames):
    """Прогнать N РЕАЛЬНЫХ кадров. Единственное место в CLI, где крутится
    taskMgr — поэтому рекурсивный poll() невозможен."""
    for _ in range(max(1, int(frames))):
        app.taskMgr.step()


def wait_ready(app, seconds=20.0):
    """Дождаться, пока поднимутся отложенные подсистемы.

    depth_renderer создаётся через taskMgr.do_method_later(0.5) — без реальных
    кадров он не появится никогда, а save_single_render его требует.
    """
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        step(app, 10)
        if getattr(app, "depth_renderer", None) is not None:
            log(f"подсистемы готовы за {time.perf_counter() - start:.1f} c")
            return True
    log("ВНИМАНИЕ: depth_renderer не поднялся за отведённое время")
    return False


# ----------------------------------------------------------------------
# Сборка сцены (порт _on_run_simulation без Qt)
# ----------------------------------------------------------------------
def build_scene(app, model_key, texture_key, target_volume):
    """Загрузить модель, задать объём/текстуру, построить наполнение.

    Возвращает True, если наполнение реально сгенерировано.
    """
    from src.ui.panel_data import (get_model_set_config,
                                   get_texture_set_config)

    cfg = get_model_set_config(model_key)
    if not cfg:
        log(f"ОШИБКА: модель '{model_key}' не найдена в конфиге")
        return False

    if not app.cache_and_load_model_set(model_key, cfg):
        log("ОШИБКА: cache_and_load_model_set вернул False")
        return False

    app.Target_Volume = float(target_volume)

    # Что РЕАЛЬНО применено — уходит в метаданные. Писать запрошенный ключ
    # нельзя: при откате на локальный набор он бы врал о содержимом кадра.
    app._cli_texture_applied = None
    applied_tex = False
    if texture_key:
        tex_cfg = get_texture_set_config(str(texture_key))
        if tex_cfg:
            try:
                materialized = _materialize_textures(app, tex_cfg)
                # Принимаем серверный набор, только если диффуз реально лежит
                # на диске — иначе loadTexture упадёт внутри генератора.
                diffuse = materialized.get("diffuse")
                if diffuse and os.path.exists(diffuse):
                    app.current_texture_set_raw = dict(tex_cfg)
                    app.set_texture_set(materialized)
                    app._cli_texture_applied = f"server:{texture_key}"
                    applied_tex = True
                else:
                    log(f"серверный набор '{texture_key}' неполон "
                        f"(нет диффуза на диске)")
            except Exception as exc:
                log(f"текстурный набор не применён: {exc}")

    if not applied_tex:
        local = find_local_texture_set()
        if local:
            local_dir = local.pop("_local_dir")
            log(f"использую локальный набор текстур: assets/textures/{local_dir}")
            try:
                app.set_texture_set(local)
                app._cli_texture_applied = f"local:{local_dir}"
                applied_tex = True
            except Exception as exc:
                log(f"локальный набор не применён: {exc}")
        else:
            log("ВНИМАНИЕ: подходящий набор текстур не найден")

    try:
        app.create_ground_plane()
        if hasattr(app, "current_ground_plane_z"):
            app.ground_plane.setPos(0, 0, app.current_ground_plane_z)
    except Exception as exc:
        log(f"create_ground_plane: {exc}")

    if not app.perform_AABB_plane():
        log("ОШИБКА: perform_AABB_plane вернул False")
        return False

    app.Perlin_Seed = random.randint(0, 10_000_000)
    gen = getattr(app, "perlin_generator", None)
    if gen is None or not gen.generate_perlin_mesh_from_csg():
        log("ОШИБКА: генерация наполнения не удалась")
        return False

    return True


def _materialize_textures(app, tex_cfg):
    """Подменить относительные пути текстур на локальные (порт
    MainWindow._materialize_texture_set, без Qt)."""
    from src.ui.panel_data import TEXTURE_PATH_KEYS, ensure_texture_cached
    out = dict(tex_cfg)
    tls = getattr(app, "tls_client", None)
    missing = 0
    for key in TEXTURE_PATH_KEYS:
        val = tex_cfg.get(key)
        if not isinstance(val, str) or not val:
            continue
        local = ensure_texture_cached(tls, val)
        if local:
            out[key] = local
        else:
            missing += 1
    if missing:
        log(f"с сервера не скачалось текстур: {missing}")
    return out


# Локальные наборы текстур из assets/textures — страховка на случай, когда
# серверные файлы недоступны (404/503). Без валидного набора генерация
# наполнения падает на loadTexture, т.к. дефолт MyApp ссылается на
# assets/textures/stones_8k, которого в репозитории нет.
_LOCAL_TEX_SUFFIXES = {
    "diffuse":      ("_diff_", "_albedo", "_basecolor"),
    "normal":       ("_nor_dx_", "_nor_gl_", "_normal"),
    "displacement": ("_disp_", "_height"),
    "roughness":    ("_rough_",),
}


def find_local_texture_set():
    """Собрать набор текстур из assets/textures/<dir>. None, если нет полного."""
    tex_root = os.path.join(PROJECT_ROOT, "assets", "textures")
    if not os.path.isdir(tex_root):
        return None
    for entry in sorted(os.listdir(tex_root)):
        d = os.path.join(tex_root, entry)
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except OSError:
            continue
        found = {}
        for slot, suffixes in _LOCAL_TEX_SUFFIXES.items():
            for f in sorted(files):
                low = f.lower()
                if low.endswith((".jpg", ".png", ".jpeg")) and \
                        any(s in low for s in suffixes):
                    found[slot] = os.path.join(d, f)
                    break
        # diffuse — минимально необходимое; остальное желательно.
        if "diffuse" in found and len(found) >= 3:
            cfg = dict(app_defaults_texture_params())
            cfg.update(found)
            cfg["_local_dir"] = entry
            return cfg
    return None


def app_defaults_texture_params():
    """Числовые параметры набора (тайлинг/сила) — как в MyApp по умолчанию."""
    return {
        "textureRepeatX": 1.35,
        "textureRepeatY": 3.2,
        "strength": 0.14,
        "textureRepeatU": 160.0,
        "textureRepeatV": 160.0,
    }


def apply_camera_preset(app, preset):
    """Поставить камеру в позу пресета (pos/hpr/fov)."""
    from panda3d.core import PerspectiveLens
    cam = app.camera
    px, py, pz = preset.get("pos", [0.0, 0.0, 0.0])
    h, p, r = preset.get("hpr", [0.0, 0.0, 0.0])
    cam.set_pos(float(px), float(py), float(pz))
    cam.set_hpr(float(h), float(p), float(r))
    fov = preset.get("fov")
    if fov is not None:
        lens = app.cam.node().get_lens()
        if isinstance(lens, PerspectiveLens):
            lens.set_fov(float(fov))
    return (float(px), float(py), float(pz)), (float(h), float(p), float(r))


# ----------------------------------------------------------------------
# Основной цикл
# ----------------------------------------------------------------------
def run(args):
    presets = load_camera_presets()
    idx = args.preset - 1
    if not (0 <= idx < len(presets)) or not presets[idx]:
        log(f"ОШИБКА: пресет {args.preset} отсутствует "
            f"(доступно 1..{len(presets)})")
        return 2
    preset = presets[idx]

    mode = DATASET_MODES[args.mode]
    log(f"модель={args.model} пресет={args.preset} режим={args.mode} "
        f"кадров={args.count}")

    app = build_app(args.graphics, (args.width, args.height))
    wait_ready(app)

    # Текстурный набор ОБЯЗАТЕЛЕН. По умолчанию MyApp.current_texture_set
    # ссылается на assets/textures/stones_8k/*, которого в репозитории нет —
    # без явного выбора генерация наполнения падает на loadTexture. В UI набор
    # всегда выбран в правой панели; здесь берём первый доступный с сервера.
    if not args.texture:
        try:
            from src.ui.panel_data import load_texture_sets
            sets = load_texture_sets()
            if sets:
                args.texture = sets[0][0]
                log(f"текстурный набор не задан — беру первый: "
                    f"{args.texture!r} ({sets[0][1]})")
            else:
                log("ВНИМАНИЕ: текстурные наборы недоступны")
        except Exception as exc:
            log(f"выбор текстурного набора не удался: {exc}")

    ru = app.renderer_utils
    max_volume = None
    try:
        from src.ui.panel_data import get_model_set_config
        cfg = get_model_set_config(args.model) or {}
        max_volume = cfg.get("max_volume")
    except Exception:
        pass

    ok_count = 0
    skipped = 0
    out_dir = args.out or "renders/dataset_cli"

    for i in range(args.count):
        target = (random.uniform(0.0, 1.25 * float(max_volume))
                  if max_volume is not None else 10.0)
        log(f"--- {i + 1}/{args.count}  target_volume={target:.2f} ---")

        if not build_scene(app, args.model, args.texture, target):
            skipped += 1
            continue

        # Кадры на доезд свежего меша и 8K-текстур до GPU.
        step(app, 60)

        base_pos, base_hpr = apply_camera_preset(app, preset)

        # Случайная вариация позы в тех же рамках, что и UI-датасет.
        dh = random.uniform(-ANG_DEG, ANG_DEG)
        dp = random.uniform(-ANG_DEG, ANG_DEG)
        lat = random.uniform(-OFFSET_M, OFFSET_M)
        vert = random.uniform(-OFFSET_M, OFFSET_M)
        cam = app.camera
        cam.set_hpr(base_hpr[0] + dh, base_hpr[1] + dp, base_hpr[2])
        cam.set_pos(cam, lat, 0.0, vert)

        # Солнце в зенит — как в датасетном режиме UI.
        try:
            app.set_sun_overhead(True)
        except Exception as exc:
            log(f"set_sun_overhead: {exc}")

        step(app, 60)

        extra_meta = {
            "render_type": "dataset",
            "dataset_type": mode["dataset_type"],
            "dataset_mode": "cli_" + args.mode,
            "random_background": mode["random_background"],
            "light_mode": "overhead",
            "sun_overhead": True,
            "iteration": i,
            "iteration_total": args.count,
            "camera_preset": args.preset,
            "variant_params": {"dh": dh, "dp": dp,
                               "lat": lat, "vert": vert},
            "base_camera_position": {"x": base_pos[0], "y": base_pos[1],
                                     "z": base_pos[2]},
            "base_camera_rotation": {"h": base_hpr[0], "p": base_hpr[1],
                                     "r": base_hpr[2]},
            "target_volume": float(target),
            "model_key": args.model,
            "texture_key": args.texture,
            # Фактически применённый набор — может отличаться от запрошенного
            # (откат на локальный, если серверные файлы недоступны).
            "texture_applied": getattr(app, "_cli_texture_applied", None),
            "source": "cli",
        }

        try:
            ok = ru.save_single_render(
                output_dir=out_dir,
                filename_prefix=f"r{i:04d}_vol{target:07.2f}_cli",
                extra_metadata=extra_meta,
                dataset_type=mode["dataset_type"],
                random_background=mode["random_background"],
                gemini=False,
                shadow_band=False,
                also_depth=True,
                cloth=mode["cloth"],
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log(f"save_single_render упал: {exc}")
            ok = False

        if ok:
            ok_count += 1
            log(f"{i + 1}/{args.count} сохранён")
        else:
            skipped += 1
            log(f"{i + 1}/{args.count} ПРОПУЩЕН (кадр не подтверждён)")

    log(f"ИТОГО: сохранено {ok_count}, пропущено {skipped}, "
        f"каталог: {out_dir}")
    return 0 if ok_count else 1


def main():
    ap = argparse.ArgumentParser(
        description="Генерация датасетов 3D-симулятора без UI.")
    ap.add_argument("--list", action="store_true",
                    help="показать модели, пресеты и режимы, затем выйти")
    ap.add_argument("--model", default="FAW-J6-8x4-tall",
                    help="ключ модели (см. --list)")
    ap.add_argument("--preset", type=int, default=2,
                    help="номер камерного пресета, с 1 (по умолчанию 2)")
    ap.add_argument("--mode", default="seg-random-bg-cloth",
                    choices=sorted(DATASET_MODES),
                    help="режим датасета")
    ap.add_argument("-n", "--count", type=int, default=5,
                    help="сколько кадров снять")
    ap.add_argument("--texture", default=None, help="ключ текстурного набора")
    ap.add_argument("--out", default=None, help="каталог вывода")
    ap.add_argument("--graphics", default="ultra",
                    choices=["ultra", "medium", "performance"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.list:
        do_list()
        return 0

    if args.seed is not None:
        random.seed(args.seed)

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
