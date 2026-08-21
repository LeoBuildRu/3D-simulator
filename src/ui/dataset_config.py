# dataset_config.py
# ---------------------------------------------------------------------------
# Конфигурация съёмки датасета — единственный источник правды для UI и
# рендера.
#
# Раньше режим съёмки выбирался одним выпадающим списком («глубина» /
# «сегментация» / «случайная сегментация»), и каждый пункт жёстко зашивал
# в себя целый набор решений: что сохранять, как двигать камеру, как ставить
# свет, куда писать файлы. Добавить «случайную камеру к глубине» или «маску
# без цветного кадра» было нельзя — пришлось бы заводить ещё один тип.
#
# Здесь эти решения разложены на НЕЗАВИСИМЫЕ оси:
#
#   outputs   — какие файлы сохранять (цвет / глубина / маска / json);
#   volume    — как выбирается объём наполнения от кадра к кадру;
#   camera    — как варьируется поза камеры;
#   lighting  — как ставится свет;
#   scene     — необязательные украшения сцены (ткань, случайный фон);
#   depth     — как считается и красится карта глубины;
#   segmentation — палитра классов маски;
#   lidar     — развёртка и шум виртуального лидара (облако точек .ply).
#
# Старые режимы выражаются комбинациями:
#   «глубина»               = outputs{color,depth} + volume ramp + camera variants + light cycle
#   «сегментация»           = outputs{color,seg}   + то же самое
#   «случайная сегментация» = outputs{color,seg,depth} + volume random + camera random + light overhead
# ---------------------------------------------------------------------------

from __future__ import annotations

import copy
import json
import os

# Умолчания лидара берём из самого сканера: дублировать их здесь означало бы
# однажды разойтись с тем, что реально умеет трассировщик.
try:
    from src.rendering.lidar_scanner import (
        default_settings as lidar_defaults,
        normalize_settings as lidar_normalize,
    )
except Exception:                             # noqa: BLE001
    # Конфиг обязан читаться и без panda3d/numpy — им пользуется и CLI, и
    # тесты. Если сканер не импортировался, лидар просто останется выключен.
    def lidar_defaults():
        return {}

    def lidar_normalize(settings):
        return dict(settings or {})

# src/ui/dataset_config.py -> src/ui -> src -> <корень проекта>
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "dataset.json")

# Потолок объёма наполнения: 135% паспортного максимума. Перегруз — валидный
# кейс для обучения, поэтому верхняя граница выше «максимума» из паспорта.
DEFAULT_CEILING_K = 1.35

DEFAULTS: dict = {
    # Сколько РАЗНЫХ наполнений сгенерировать. Итоговое число кадров =
    # count * (число кадров на одно наполнение), см. frames_per_fill().
    "count": 10,
    "output_dir": "renders/dataset",

    "outputs": {
        "color": True,
        "depth": True,
        "segmentation": False,
        "lidar": False,
        "json": True,
    },

    "volume": {
        # ramp   — объём растёт линейно от 0 до максимума за `count` шагов;
        # random — случайный объём с заданными долями полных и пустых кузовов.
        "mode": "ramp",
        "full_pct": 0.0,
        "empty_pct": 0.0,
        "ceiling_k": DEFAULT_CEILING_K,
    },

    "camera": {
        # fixed    — только текущая поза пользователя;
        # variants — сетка отклонений (по одному кадру на каждое);
        # random   — случайная поза в тех же рамках, `samples` кадров.
        "mode": "variants",
        "angle_deg": 10.0,
        "offset_m": 0.05,
        "samples": 1,
        "variants": {
            "originals": True,       # базовая поза (по кадру на тип света)
            "angles": True,          # ±angle по рысканью и тангажу
            "offsets": True,         # ±offset по горизонтали и вертикали
            "random_combined": True,  # один кадр со случайной комбинацией
        },
    },

    "lighting": {
        # cycle    — чередование выбранных типов света;
        # overhead — солнце жёстко в зените (тени минимальны);
        # current  — время суток из ползунка в UI, без вмешательства.
        "mode": "cycle",
        "cycle": {"day": True, "dusk": True, "shadow": True},
    },

    "scene": {
        "cloth": False,
        "cloth_probability": 0.8,
        "random_background": False,
    },

    "depth": {
        # grayscale — ч/б карта вместо радужной.
        # near/far — плоскости отсечения камеры глубины, метры.
        # grad_start/grad_end — окно шкалы В ДОЛЯХ far (шейдер нормирует
        # линеаризованную глубину на дальнюю плоскость). Значения по
        # умолчанию — пресет стационарной камеры: 6.4–16 м при far=64.
        "grayscale": True,
        "near": 0.01,
        "far": 64.0,
        "grad_start": 0.10,
        "grad_end": 0.25,
    },

    "segmentation": {
        # Только ПЕРЕОПРЕДЕЛЕНИЯ поверх палитры segmentation_renderer:
        # {"cargo": [253, 2, 2], "background": [0, 0, 0], ...}
        "palette": {},
    },

    # Виртуальный лидар: облако точек .ply рядом с кадром. Значения
    # по умолчанию и границы живут в rendering/lidar_scanner (там же
    # объяснено, что означает каждое), здесь только копия для конфига.
    "lidar": dict(lidar_defaults()),
}

LIGHT_MODES = ("cycle", "overhead", "current")
CAMERA_MODES = ("fixed", "variants", "random")
VOLUME_MODES = ("ramp", "random")
OUTPUT_KEYS = ("color", "depth", "segmentation", "lidar", "json")


def defaults() -> dict:
    return copy.deepcopy(DEFAULTS)


def _merge(base: dict, patch) -> dict:
    """Рекурсивно наложить `patch` на `base`, не теряя незнакомые ключи."""
    if not isinstance(patch, dict):
        return base
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def normalize(cfg) -> dict:
    """Дополнить конфиг умолчаниями и подрезать значения до допустимых.

    Конфиг приезжает из json на диске и переживает смены версий, поэтому
    доверять ему нельзя: недостающие ключи добавляем, мусорные значения
    заменяем умолчанием, а не падаем.
    """
    out = _merge(defaults(), cfg or {})

    try:
        out["count"] = max(1, min(100000, int(out["count"])))
    except (TypeError, ValueError):
        out["count"] = DEFAULTS["count"]

    out["output_dir"] = str(out.get("output_dir") or
                            DEFAULTS["output_dir"]).strip()
    if not out["output_dir"]:
        out["output_dir"] = DEFAULTS["output_dir"]

    outputs = out["outputs"]
    for key in OUTPUT_KEYS:
        outputs[key] = bool(outputs.get(key, DEFAULTS["outputs"][key]))
    # Кадр, из которого не сохраняется НИ ОДНОЙ картинки, — это просто
    # потраченное время рендера; такой конфиг молча чиним.
    if not (outputs["color"] or outputs["depth"] or outputs["segmentation"]
            or outputs["lidar"]):
        outputs["color"] = True

    vol = out["volume"]
    if vol.get("mode") not in VOLUME_MODES:
        vol["mode"] = DEFAULTS["volume"]["mode"]
    vol["full_pct"] = _clamp(vol.get("full_pct"), 0.0, 100.0, 0.0)
    vol["empty_pct"] = _clamp(vol.get("empty_pct"), 0.0, 100.0, 0.0)
    if vol["full_pct"] + vol["empty_pct"] > 100.0:
        vol["empty_pct"] = max(0.0, 100.0 - vol["full_pct"])
    vol["ceiling_k"] = _clamp(vol.get("ceiling_k"), 0.1, 3.0,
                              DEFAULT_CEILING_K)

    cam = out["camera"]
    if cam.get("mode") not in CAMERA_MODES:
        cam["mode"] = DEFAULTS["camera"]["mode"]
    cam["angle_deg"] = _clamp(cam.get("angle_deg"), 0.0, 90.0, 10.0)
    cam["offset_m"] = _clamp(cam.get("offset_m"), 0.0, 5.0, 0.05)
    try:
        cam["samples"] = max(1, min(64, int(cam.get("samples", 1))))
    except (TypeError, ValueError):
        cam["samples"] = 1
    variants = cam["variants"]
    for key, default_on in DEFAULTS["camera"]["variants"].items():
        variants[key] = bool(variants.get(key, default_on))
    if cam["mode"] == "variants" and not any(variants.values()):
        variants["originals"] = True

    light = out["lighting"]
    if light.get("mode") not in LIGHT_MODES:
        light["mode"] = DEFAULTS["lighting"]["mode"]
    cycle = light["cycle"]
    for key in ("day", "dusk", "shadow"):
        cycle[key] = bool(cycle.get(key, True))
    if light["mode"] == "cycle" and not any(cycle.values()):
        cycle["day"] = True

    scene = out["scene"]
    scene["cloth"] = bool(scene.get("cloth", False))
    scene["random_background"] = bool(scene.get("random_background", False))
    scene["cloth_probability"] = _clamp(scene.get("cloth_probability"),
                                        0.0, 1.0, 0.8)

    depth = out["depth"]
    depth["grayscale"] = bool(depth.get("grayscale", True))
    depth["near"] = _clamp(depth.get("near"), 0.001, 1000.0, 0.01)
    depth["far"] = _clamp(depth.get("far"), 0.01, 10000.0, 64.0)
    if depth["far"] <= depth["near"]:
        depth["far"] = depth["near"] + 1.0
    # grad_* — ДОЛЯ дальней плоскости, а не метры: шейдер оверлея нормирует
    # линеаризованную глубину на far (см. depth_map_renderer).
    depth["grad_start"] = _clamp(depth.get("grad_start"), 0.0, 1.0, 0.10)
    depth["grad_end"] = _clamp(depth.get("grad_end"), 0.0, 1.0, 0.25)
    if depth["grad_end"] <= depth["grad_start"]:
        depth["grad_end"] = min(1.0, depth["grad_start"] + 0.01)

    palette = out["segmentation"].get("palette")
    if not isinstance(palette, dict):
        palette = {}
    clean = {}
    for name, rgb in palette.items():
        try:
            r, g, b = (int(c) for c in rgb)
        except (TypeError, ValueError):
            continue
        clean[str(name)] = [max(0, min(255, r)), max(0, min(255, g)),
                            max(0, min(255, b))]
    out["segmentation"]["palette"] = clean

    out["lidar"] = lidar_normalize(out.get("lidar"))

    return out


def _clamp(value, lo, hi, fallback):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return fallback


def frames_per_fill(cfg) -> int:
    """Сколько кадров снимается с ОДНОГО наполнения кузова."""
    cam = cfg.get("camera", {})
    mode = cam.get("mode", "variants")
    if mode == "fixed":
        return 1
    if mode == "random":
        return max(1, int(cam.get("samples", 1)))

    variants = cam.get("variants", {})
    lights = enabled_lights(cfg)
    total = 0
    if variants.get("originals"):
        # Базовая поза снимается по разу на каждый включённый тип света —
        # чтобы «эталонный» кадр был представлен в каждом освещении.
        total += len(lights)
    if variants.get("angles"):
        total += 4                      # ±рысканье, ±тангаж
    if variants.get("offsets"):
        total += 4                      # ±горизонталь, ±вертикаль
    if variants.get("random_combined"):
        total += 1
    return max(1, total)


def enabled_lights(cfg) -> list:
    """Список типов света, между которыми чередуется съёмка."""
    light = cfg.get("lighting", {})
    mode = light.get("mode", "cycle")
    if mode == "overhead":
        return ["overhead"]
    if mode == "current":
        return ["current"]
    cycle = light.get("cycle", {})
    lights = [name for name in ("day", "dusk", "shadow") if cycle.get(name)]
    return lights or ["day"]


def total_frames(cfg) -> int:
    return int(cfg.get("count", 1)) * frames_per_fill(cfg)


def output_list(cfg) -> list:
    """Включённые выходы в порядке отображения."""
    outputs = cfg.get("outputs", {})
    return [key for key in OUTPUT_KEYS if outputs.get(key)]


# ---------------------------------------------------------------------------
# Хранение на диске
# ---------------------------------------------------------------------------
def load() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return normalize(json.load(fh))
    except FileNotFoundError:
        return defaults()
    except Exception as exc:
        print(f"[DatasetConfig] {CONFIG_PATH} не прочитан ({exc}); "
              f"беру умолчания.")
        return defaults()


def save(cfg) -> bool:
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(normalize(cfg), fh, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        print(f"[DatasetConfig] сохранение не удалось: {exc}")
        return False
