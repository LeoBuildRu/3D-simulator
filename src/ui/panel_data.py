# panel_data.py
# ---------------------------------------------------------------------------
# Data layer for the right-side control panel.
#
# Three responsibilities, kept deliberately separate from the UI in
# right_panel.py:
#
#   1. load_texture_sets()
#        Возвращает список текстурных наборов, который должен отображать
#        правый сайдбар. Источник — in-memory кэш модуля, который
#        наполняется один раз при старте приложения через
#        `TLS_client.get_textures_config()` (см. main.py); сам конфиг
#        живёт на сервере в
#        /home/leonid/operation-3d-service/config/textures_napolnitel_config.json.
#
#        Display-label каждой записи — поле `name` (например, "asphalt",
#        "ground V2", "stones"); при отсутствии — сам ключ. Синтетическая
#        запись `default:` (указатель на другой ключ) пропускается.
#
#   2. load_model_sets()
#        Mirrors what the legacy gui.py did:
#          * First try to fetch `models_geometry_config.json` from the TLS
#            server via `TLS_client.get_models_config()`. The active
#            server is taken from `tls_config.yaml`.
#          * On any failure (server down, no network, malformed reply)
#            fall back to the local `models_config.yaml`.
#
#        Each set's display label is the entry's `model` field
#        (e.g. "Shackman-X3000-8x4-tall"); falls back to the set key
#        when `model` is absent.
#
#   3. load_reconstructions()
#        Mirrors `load_recon_jsons` from the legacy gui.py:
#          * Calls `TLS_client.get_verified_models()` for the most
#            recently-verified results (PLY / HEIGHT entries).
#          * Scans the local `height_examples/` folder for additional
#            JSON files the user generated outside the server.
#          * Normalises both into a single `Reconstruction` dataclass,
#            sorted newest-first.
#          * Falls back to local-only when the server is unreachable.
#
# All loaders preserve the config's / server's iteration order. Loaders
# returning combo lists hand back (key, display_name) tuples so the UI
# can show pretty names while emitting canonical backend keys.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_CONFIG_PATH   = os.path.join(PROJECT_ROOT, "config", "models_config.yaml")
TLS_CONFIG_PATH      = os.path.join(PROJECT_ROOT, "config", "tls_config.yaml")
HEIGHT_EXAMPLES_DIR  = os.path.join(PROJECT_ROOT, "assets", "height_examples")
# Snapshot pairs (colour frame + depth map) for manual camera alignment.
STAND_EXAMPLES_DIR   = os.path.join(HEIGHT_EXAMPLES_DIR, "stand")
PLY_EXAMPLES_DIR     = os.path.join(PROJECT_ROOT, "assets", "PLY_examples")

# Local truck models the user drops into assets/models/trucks. They are
# listed in the model-set combo alongside the server-served sets. Their
# config keys are namespaced with LOCAL_MODEL_PREFIX so they never collide
# with server keys, and they carry "local": True so the loader knows to
# load the .bam/.gltf/... straight from disk (no server round-trip, and no
# reference points / camera presets — those simply aren't provided).
TRUCKS_MODELS_DIR    = os.path.join(PROJECT_ROOT, "assets", "models", "trucks")
LOCAL_MODEL_PREFIX   = "local:"

# Комплекты, собранные генератором кузовов (`src/bodygen`). В отличие от
# одиночных моделей в `trucks/` это ПОЛНЫЕ наборы: кузов, наполнитель и шасси
# в общих координатах, — то есть их можно грузить как серверные, вместе с
# генерацией груза, а не только смотреть.
GENERATED_MODELS_DIR = os.path.join(PROJECT_ROOT, "assets", "models",
                                    "generated")
GENERATED_MODEL_PREFIX = "gen:"
_TRUCK_MODEL_EXTS    = (".bam", ".gltf", ".glb", ".egg.pz", ".egg", ".obj")

# Локальный кэш для скачанных файлов текстур
# (диффуз / нормали / displacement / roughness и т.д.).
# Структура внутри: %TEMP%/vizutil_textures_cache/<подкаталог>/<имя_файла>,
# где <подкаталог>/<имя_файла> — это путь из textures_napolnitel_config.json
# с отрезанным префиксом 'textures/'.
TEXTURES_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), "vizutil_textures_cache"
)

# Ключи в записях textures_napolnitel_config.json, значения которых —
# относительные пути к файлам текстур. Используется при ленивой подмене
# на локальные пути в MainWindow._on_texture_set_changed.
TEXTURE_PATH_KEYS = (
    "diffuse", "albedo", "normal", "displacement",
    "roughness", "metallic", "height",
)


# Cached TLS_client import — kept lazy so that simply importing this
# module doesn't blow up if the `requests` / TLS_client tree isn't
# importable in some environment.
_TLS_CLIENT_CLS = None


def _get_tls_client_cls():
    global _TLS_CLIENT_CLS
    if _TLS_CLIENT_CLS is None:
        try:
            from src.core.TLS_client import TLS_client  # local module
            _TLS_CLIENT_CLS = TLS_client
        except Exception as exc:
            print(f"[panel_data] TLS_client import failed: {exc}")
            _TLS_CLIENT_CLS = False                    # sentinel: don't retry
    return _TLS_CLIENT_CLS or None


# ---------------------------------------------------------------------------
# Texture sets — теперь хранятся в памяти, наполняются с сервера при старте
# приложения (см. main.py → tls_client.get_textures_config()).
# ---------------------------------------------------------------------------
_TEXTURE_SETS_CACHE: Optional[Dict[str, Any]] = None


def set_texture_sets_cache(cfg: Optional[Dict[str, Any]]) -> None:
    """
    Сохранить серверный конфиг текстурных наборов в памяти модуля.
    Вызывается один раз при запуске приложения после успешного
    `tls_client.get_textures_config()`. Передача None очищает кэш.
    """
    global _TEXTURE_SETS_CACHE
    if cfg is None:
        _TEXTURE_SETS_CACHE = None
        return
    if not isinstance(cfg, dict):
        print(f"[panel_data] set_texture_sets_cache: ожидался dict, "
              f"получен {type(cfg).__name__} — кэш не обновлён")
        return
    _TEXTURE_SETS_CACHE = cfg


def get_texture_sets_cache() -> Dict[str, Any]:
    """Текущий снимок кэша. Пустой dict, если ещё не инициализирован."""
    return _TEXTURE_SETS_CACHE or {}


def load_texture_sets() -> List[Tuple[str, str]]:
    """
    Вернуть [(key, display_name)] для каждой записи серверного конфига
    текстурных наборов, пропуская синтетический ключ `default`.

    `display_name` — поле `name` записи; при отсутствии — сам ключ
    (чтобы запись не пропала из выпадающего списка).
    """
    cfg = _TEXTURE_SETS_CACHE
    if not cfg:
        return []

    out: List[Tuple[str, str]] = []
    for key, value in cfg.items():
        if key == "default":
            continue
        if not isinstance(value, dict):
            continue
        display = value.get("name") or key
        out.append((key, str(display)))
    return out


def get_default_texture_set_key() -> Optional[str]:
    """Вернуть ключ, на который ссылается синтетическая запись `default:`."""
    cfg = _TEXTURE_SETS_CACHE
    if not cfg:
        return None
    val = cfg.get("default")
    return val if isinstance(val, str) else None


# ---------------------------------------------------------------------------
# Model sets
# ---------------------------------------------------------------------------
def _read_active_tls_server() -> Optional[Tuple[str, int]]:
    """Pick the first server flagged active=true in tls_config.yaml."""
    if not os.path.exists(TLS_CONFIG_PATH):
        return None
    try:
        with open(TLS_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[panel_data] failed to parse tls_config.yaml: {exc}")
        return None

    for srv in data.get("servers", []) or []:
        if srv.get("active"):
            host = srv.get("host")
            port = int(srv.get("port", 9999))
            if host:
                return host, port
    return None


def _fetch_model_sets_from_server() -> Optional[dict]:
    """
    Try to download `models_geometry_config.json` (i.e. the dict served by
    `get_models_config`) from the active TLS server. Returns the parsed
    dict or None on any failure.
    """
    cls = _get_tls_client_cls()
    if cls is None:
        return None

    server = _read_active_tls_server()
    if server is None:
        print("[panel_data] no active TLS server in tls_config.yaml")
        return None

    host, port = server
    try:
        client = cls(host=host, port=port, timeout=5.0)
        cfg = client.get_models_config()
        if isinstance(cfg, dict) and cfg:
            print(f"[panel_data] models config loaded from {host}:{port} "
                  f"({len(cfg)} sets)")
            return cfg
        print(f"[panel_data] empty models config from {host}:{port}")
    except Exception as exc:
        print(f"[panel_data] TLS get_models_config failed: {exc}")
    return None


def _truck_model_display(stem: str) -> str:
    """Human-readable name for a local truck model file/folder."""
    pretty = stem.replace("_", " ").replace("-", " ").strip()
    return pretty or stem


def _is_truck_model_file(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(ext) for ext in _TRUCK_MODEL_EXTS)


def _strip_truck_model_ext(name: str) -> str:
    low = name.lower()
    for ext in _TRUCK_MODEL_EXTS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _local_truck_entries() -> List[Tuple[str, dict, str, str]]:
    """
    Пройти `assets/models/trucks/` и вернуть
    [(set_key, config, stem, model_path)] для каждой найденной модели.

    Вынесено отдельно, потому что нужно дважды: `_scan_local_truck_models`
    делает из этого словарь конфигов для загрузчика, а
    `load_model_sets_detailed` берёт `stem`, чтобы найти рядом
    `<stem>.spec.json` с размерами кузова.

    Поддерживаются две раскладки:

        trucks/<name>.bam            -> одиночный самодостаточный файл
        trucks/<name>/<model>.gltf   -> модель в своей папке
                                        (берётся первый подходящий файл)
    """
    out: List[Tuple[str, dict, str, str]] = []
    if not os.path.isdir(TRUCKS_MODELS_DIR):
        return out

    try:
        entries = sorted(os.listdir(TRUCKS_MODELS_DIR))
    except OSError as exc:
        print(f"[panel_data] failed to list models/trucks/: {exc}")
        return out

    for entry in entries:
        full = os.path.join(TRUCKS_MODELS_DIR, entry)
        model_path = None
        display_stem = entry

        if os.path.isfile(full) and _is_truck_model_file(entry):
            model_path = full
            display_stem = _strip_truck_model_ext(entry)
        elif os.path.isdir(full):
            try:
                sub_entries = sorted(os.listdir(full))
            except OSError:
                sub_entries = []
            for sub in sub_entries:
                sub_full = os.path.join(full, sub)
                if os.path.isfile(sub_full) and _is_truck_model_file(sub):
                    model_path = sub_full
                    display_stem = entry
                    break

        if not model_path:
            continue

        rel = os.path.relpath(model_path, TRUCKS_MODELS_DIR).replace("\\", "/")
        key = f"{LOCAL_MODEL_PREFIX}{rel}"
        cfg = {
            "model": f"{_truck_model_display(display_stem)} (локальная)",
            "cuzov": model_path,
            "local": True,
        }
        out.append((key, cfg, display_stem, model_path))
    return out


def _scan_local_truck_models() -> Dict[str, dict]:
    """
    Локальные модели из `assets/models/trucks/` в виде {set_key: config}.

    Каждый конфиг минимален — ровно столько, чтобы `load_model_set`
    положил модель в сцену:
        {"model": <display>, "cuzov": <abs path>, "local": True}

    Ни ground_plane / max_volume, ни cam_*: по спецификации локальным
    моделям не нужны ни опорные точки, ни пресеты камеры.
    """
    return {key: cfg for key, cfg, _stem, _path in _local_truck_entries()}


def _generated_kit_entries() -> List[Tuple[str, dict, str, dict]]:
    """
    Комплекты генератора из `assets/models/generated/` в виде
    [(set_key, config, stem, set_json)].

    Опознаются по `<name>.set.json`, который пишет сам генератор: в нём уже
    лежат имена частей и объём кузова, так что гадать по файлам не нужно.
    Комплект регистрируется ОДНОЙ записью со всеми тремя частями — иначе в
    списке появились бы три отдельные модели, ни одна из которых не грузится
    целиком.

    `stem` — имя файла без `.set.json`: по нему рядом ищется
    `<stem>.spec.json` с обмерами кузова (см. `load_model_sets_detailed`).
    """
    out: List[Tuple[str, dict, str, dict]] = []
    if not os.path.isdir(GENERATED_MODELS_DIR):
        return out

    try:
        entries = sorted(os.listdir(GENERATED_MODELS_DIR))
    except OSError as exc:
        print(f"[panel_data] failed to list models/generated/: {exc}")
        return out

    for entry in entries:
        if not entry.endswith(".set.json"):
            continue
        stem = entry[: -len(".set.json")]
        path = os.path.join(GENERATED_MODELS_DIR, entry)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[panel_data] {entry}: {exc}")
            continue

        name = data.get("model") or stem

        def abs_part(key):
            v = data.get(key)
            if not v:
                return None
            p = os.path.join(GENERATED_MODELS_DIR, v)
            return p if os.path.exists(p) else None

        cuzov = abs_part("cuzov")
        if not cuzov:
            continue

        cfg = {
            "model": f"{name} (сгенерирован)",
            "cuzov": cuzov,
            "local": True,
        }
        for src_key, dst_key in (("napolnitel", "napolnitel"),
                                 ("other", "other"),
                                 ("target_model", "target_model")):
            p = abs_part(src_key)
            if p:
                cfg[dst_key] = p
        if data.get("max_volume"):
            cfg["max_volume"] = float(data["max_volume"])
        if data.get("ground_plane") is not None:
            cfg["ground_plane"] = float(data["ground_plane"])

        out.append((f"{GENERATED_MODEL_PREFIX}{name}", cfg, stem, data))
    return out


def _scan_generated_kits() -> Dict[str, dict]:
    """Комплекты генератора в виде {set_key: config} — см.
    `_generated_kit_entries`."""
    return {key: cfg for key, cfg, _stem, _data in _generated_kit_entries()}


def _load_local_models_yaml() -> dict:
    """Fallback: parse `models_config.yaml` from the project root."""
    if not os.path.exists(MODELS_CONFIG_PATH):
        return {}
    try:
        with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[panel_data] failed to parse models_config.yaml: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Model set metadata — то, что показывает диалог выбора кузова
# ---------------------------------------------------------------------------
#: Человекочитаемое имя источника набора.
MODEL_SOURCE_LABELS = {
    "server":    "Сервер",
    "yaml":      "Конфиг",
    "generated": "Генератор",
    "local":     "Локальная",
}

#: Колёсная формула в имени модели: "8x4", "6х4", "8×4".
_AXLE_RE = re.compile(r"(?<![0-9])([2-9])\s*[xх×]\s*([2-9])(?![0-9])",
                      re.IGNORECASE)


@dataclass
class ModelSetInfo:
    """
    Одна строка в списке кузовов: ключ набора плюс всё, что о нём удалось
    выяснить, не открывая сам меш.

    Поля намеренно необязательные: у серверных наборов нет обмеров, у
    локальных .bam без `.spec.json` — ничего, кроме имени файла. UI
    показывает прочерк там, где данных нет, а не прячет строку.
    """
    key: str
    name: str                      # имя без служебных суффиксов
    display: str                   # ярлык как в старом комбо-боксе
    source: str = "server"         # server | yaml | generated | local
    origin: str = ""               # откуда взялась геометрия
    chassis: str = ""              # профиль шасси генератора
    axles: str = ""                # колёсная формула, "8x4"
    volume: Optional[float] = None          # м³
    volume_kind: str = ""          # макс. | изм. | расч.
    length: Optional[float] = None          # внутренние габариты кузова, м
    width: Optional[float] = None
    height: Optional[float] = None
    parts: Tuple[str, ...] = ()    # cuzov / napolnitel / other / target_model
    file_size: int = 0             # суммарный размер файлов набора, байт
    mtime: float = 0.0             # время правки кузова (unix)
    path: str = ""                 # путь к кузову (локальные наборы)
    ground_plane: Optional[float] = None     # опорная плоскость набора, м
    ref_rect: Optional[Tuple[float, float]] = None   # прямоугольник points_3d
    has_camera: bool = False       # есть ли пресет камеры (cam_pos_*)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        return MODEL_SOURCE_LABELS.get(self.source, self.source)

    @property
    def dims(self) -> Optional[Tuple[float, float, float]]:
        """(длина, ширина, высота) или None, если обмеров нет."""
        if self.length is None or self.width is None or self.height is None:
            return None
        return (self.length, self.width, self.height)

    @property
    def kit(self) -> str:
        """
        Насколько набор полон. Важно на практике: без `napolnitel` груз
        не сгенерируется — модель годится только для просмотра (или ей
        нужен «донор», см. MyApp.load_model_set).
        """
        if "napolnitel" in self.parts and "other" in self.parts:
            return "полный"
        if "napolnitel" in self.parts:
            return "кузов + груз"
        return "только кузов"

    @property
    def search_text(self) -> str:
        """Строка, по которой ищет поле поиска в диалоге."""
        return " ".join(str(x) for x in (
            self.name, self.key, self.source_label, self.origin,
            self.chassis, self.axles, self.kit,
        ) if x).lower()


def _as_float(value: Any) -> Optional[float]:
    """float() без исключений; NaN тоже считается отсутствующим значением."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _guess_axles(*texts: Any) -> str:
    """Вытащить колёсную формулу из имени/ключа набора."""
    for text in texts:
        if not text:
            continue
        m = _AXLE_RE.search(str(text))
        if m:
            return f"{m.group(1)}x{m.group(2)}"
    return ""


def _config_parts(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(k for k in ("cuzov", "napolnitel", "other", "target_model")
                 if cfg.get(k))


def _path_size(path: Optional[str]) -> int:
    """Размер файла, а для папки — суммарный размер её содержимого."""
    if not path:
        return 0
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total
    except OSError:
        return 0


def _mtime_of(path: Optional[str]) -> float:
    try:
        return os.path.getmtime(path) if path else 0.0
    except OSError:
        return 0.0


def _spec_dims(spec: Dict[str, Any]):
    return (_as_float(spec.get("inner_length")),
            _as_float(spec.get("inner_width")),
            _as_float(spec.get("inner_height")))


def _spec_origin(spec: Dict[str, Any]) -> str:
    """
    Откуда взят кузов по его `.spec.json`: скан облака точек, запись
    справочника или ручные параметры.
    """
    src = str(spec.get("source") or "").strip()
    model_key = str(spec.get("model_key") or "").strip()
    suffix = f" · {model_key}" if model_key else ""
    if src.lower().endswith(".ply"):
        return f"скан {os.path.basename(src.replace(chr(92), '/'))}{suffix}"
    if src == "catalog":
        return f"справочник{suffix}"
    if src:
        return src if len(src) < 60 else os.path.basename(
            src.replace(chr(92), "/"))
    return model_key


def _rect_of(points: Any) -> Optional[Tuple[float, float]]:
    """
    (длинная сторона, короткая сторона) прямоугольника `points_3d`.
    None, если точек нет или они не разбираются.
    """
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return None
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    a, b = max(ys) - min(ys), max(xs) - min(xs)
    if a <= 0 or b <= 0:
        return None
    return (max(a, b), min(a, b))


def _apply_spec(info: "ModelSetInfo", spec: Optional[Dict[str, Any]]) -> None:
    """Дополнить запись обмерами из `<name>.spec.json`, если он есть."""
    if not spec:
        return
    length, width, height = _spec_dims(spec)
    if info.length is None:
        info.length, info.width, info.height = length, width, height
    if not info.origin:
        info.origin = _spec_origin(spec)
    if not info.axles:
        info.axles = _guess_axles(spec.get("model_key"), spec.get("name"))
    if info.volume is None:
        measured = _as_float(spec.get("measured_volume_m3"))
        if measured:
            info.volume, info.volume_kind = measured, "изм."
        elif None not in (length, width, height):
            info.volume = length * width * height
            info.volume_kind = "расч."


def load_model_sets_detailed() -> List[ModelSetInfo]:
    """
    То же, что `load_model_sets`, но с характеристиками каждого набора:
    источник, колёсная формула, объём, внутренние габариты, состав
    комплекта, размер файлов.

    Порядок тот же, что и раньше: серверные наборы, затем комплекты
    генератора, затем локальные модели из `assets/models/trucks`.

    Данные собираются из трёх мест, и ни одно из них не обязательно:
      * серверный `models_geometry_config.json` (или локальный YAML) —
        `max_volume`, состав комплекта;
      * `<name>.set.json` генератора — шасси, объём, габариты по
        `world_bounds`;
      * `<name>.spec.json` — внутренние обмеры кузова и источник
        (скан .ply / справочник).
    """
    out: List[ModelSetInfo] = []

    server_cfg = _fetch_model_sets_from_server()
    if server_cfg:
        server = _read_active_tls_server()
        source = "server"
        origin = f"{server[0]}:{server[1]}" if server else "TLS-сервер"
        cfg = server_cfg
    else:
        source = "yaml"
        origin = os.path.basename(MODELS_CONFIG_PATH)
        cfg = _load_local_models_yaml()

    for key, value in cfg.items():
        if not isinstance(value, dict):
            continue
        name = str(value.get("model") or key)
        volume = _as_float(value.get("max_volume"))
        # `points_3d` — четыре опорные точки верхнего проёма (по ним
        # выравнивается камера). Габаритами кузова они НЕ являются: у
        # большинства наборов это один и тот же шаблонный прямоугольник,
        # поэтому он идёт только в подробности строки, а не в колонку.
        out.append(ModelSetInfo(
            key=str(key), name=name, display=name,
            source=source, origin=origin,
            axles=_guess_axles(name, key),
            volume=volume, volume_kind="макс." if volume else "",
            parts=_config_parts(value),
            ground_plane=_as_float(value.get("ground_plane")),
            ref_rect=_rect_of(value.get("points_3d")),
            has_camera=value.get("cam_pos_x") is not None,
            config=value,
        ))

    # Сгенерированные комплекты идут сразу после серверных: это полные
    # наборы, а не отдельные модели для просмотра.
    for key, value, stem, data in _generated_kit_entries():
        name = key[len(GENERATED_MODEL_PREFIX):]
        volume = _as_float(data.get("max_volume"))
        info = ModelSetInfo(
            key=key, name=name, display=str(value.get("model") or name),
            source="generated",
            chassis=str(data.get("chassis") or ""),
            axles=_guess_axles(data.get("chassis"), name),
            volume=volume, volume_kind="макс." if volume else "",
            parts=_config_parts(value),
            path=value.get("cuzov", ""),
            ground_plane=_as_float(value.get("ground_plane")),
            config=value,
        )
        # Габариты: сперва обмеры кузова, иначе — рамка комплекта в мире.
        bounds = (data.get("world_bounds") or {}).get("body")
        if isinstance(bounds, list) and len(bounds) == 2:
            try:
                (x0, y0, z0), (x1, y1, z1) = bounds
                info.length = abs(float(y1) - float(y0))
                info.width = abs(float(x1) - float(x0))
                info.height = abs(float(z1) - float(z0))
            except (TypeError, ValueError):
                pass
        spec = _read_json(os.path.join(GENERATED_MODELS_DIR,
                                       f"{stem}.spec.json"))
        if spec:
            # Внутренние обмеры точнее внешней рамки — они и идут в колонку.
            length, width, height = _spec_dims(spec)
            if None not in (length, width, height):
                info.length, info.width, info.height = length, width, height
            _apply_spec(info, spec)
        info.file_size = sum(_path_size(value.get(k))
                             for k in ("cuzov", "napolnitel", "other"))
        info.mtime = _mtime_of(value.get("cuzov"))
        out.append(info)

    # Локальные модели из assets/models/trucks — после серверных, чтобы оба
    # источника жили в одном списке.
    for key, value, stem, model_path in _local_truck_entries():
        folder = os.path.dirname(model_path)
        info = ModelSetInfo(
            key=key, name=_truck_model_display(stem),
            display=str(value.get("model") or key),
            source="local",
            axles=_guess_axles(stem, key),
            parts=_config_parts(value),
            path=model_path,
            # Модель в своей папке весит столько, сколько весит вся папка
            # (текстуры, .bin); одиночный файл — сколько сам файл.
            file_size=_path_size(model_path if folder == TRUCKS_MODELS_DIR
                                 else folder),
            mtime=_mtime_of(model_path),
            config=value,
        )
        _apply_spec(info, _read_json(os.path.join(TRUCKS_MODELS_DIR,
                                                  f"{stem}.spec.json")))
        if not info.origin:
            ext = os.path.splitext(model_path)[1].lstrip(".").upper()
            info.origin = f"{ext} из assets/models/trucks"
        out.append(info)

    return out


def load_model_sets() -> List[Tuple[str, str]]:
    """
    Return [(set_key, display_name)] for every model set the panel should
    offer.

    Resolution order matches the legacy gui.py:
        1. `TLS_client.get_models_config()` — server-served JSON.
        2. Local `models_config.yaml`         — offline fallback.

    `display_name` is each entry's `model` field; falls back to the
    set_key when `model` is missing. Тонкая обёртка над
    `load_model_sets_detailed`: характеристики нужны только диалогу
    выбора, остальным (cli.py, поиск модели по имени) хватает пары.
    """
    return [(info.key, info.display) for info in load_model_sets_detailed()]


# ---------------------------------------------------------------------------
# Удаление наборов с диска
# ---------------------------------------------------------------------------
# Генератор кузовов складывает комплект в `assets/models/generated/` россыпью
# файлов с общим именем (`<stem>.set.json`, `<stem>-Cuzov.bam`, `<stem>.gltf`,
# папка `<stem>/` с текстурами и т.д.), а локальные модели лежат в
# `assets/models/trucks/`. Удалять их вручную из проводника неудобно и легко
# оставить хвосты, поэтому список умеет убирать набор целиком.
#
# Правило владения: файл принадлежит набору, если его имя равно `stem` или
# начинается с `stem` + разделитель (`.`, `-`, `_`). Когда под правило подходят
# сразу несколько наборов (`truck` и `truck-2`), файл достаётся ТОМУ, чей stem
# длиннее, — иначе удаление `truck` унесло бы с собой соседний комплект.
#
# Серверные наборы не удаляются: их файлы лежат не у нас.

#: Символы, отделяющие имя набора от суффикса файла.
_STEM_BOUNDARY = (".", "-", "_")


@dataclass
class ModelSetRemoval:
    """
    Что произойдёт при удалении набора: список файлов и папок, их суммарный
    размер и — если удалить нельзя — причина.

    Возвращается ДО удаления, чтобы интерфейс показал пользователю точный
    список, а не «удалить набор?» вслепую.
    """
    key: str
    name: str = ""
    paths: List[str] = field(default_factory=list)
    total_size: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.paths)

    @property
    def size_label(self) -> str:
        """Суммарный размер по-человечески — для окна подтверждения."""
        return _fmt_bytes(self.total_size)


def _belongs_to_stem(entry: str, stem: str) -> bool:
    """Имя файла/папки относится к набору `stem`?"""
    if entry == stem:
        return True
    return (entry.startswith(stem)
            and entry[len(stem):len(stem) + 1] in _STEM_BOUNDARY)


def _entries_owned_by(directory: str, stem: str,
                      siblings: List[str]) -> List[str]:
    """
    Пути внутри `directory`, принадлежащие набору `stem`.

    `siblings` — имена всех наборов в этой папке (включая сам `stem`): по ним
    разрешается спор между `truck` и `truck-2`, см. комментарий к разделу.
    """
    try:
        entries = sorted(os.listdir(directory))
    except OSError as exc:
        print(f"[panel_data] failed to list {directory}: {exc}")
        return []

    out: List[str] = []
    for entry in entries:
        owner = ""
        for cand in siblings:
            if _belongs_to_stem(entry, cand) and len(cand) > len(owner):
                owner = cand
        if owner == stem:
            out.append(os.path.join(directory, entry))
    return out


def _inside(path: str, directory: str) -> bool:
    """Страховка от выхода за пределы папки набора (симлинки, `..` в имени)."""
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(directory)]
        ) == os.path.abspath(directory) and os.path.abspath(path) != \
            os.path.abspath(directory)
    except ValueError:      # разные диски на Windows
        return False


def can_delete_model_set(key: Any) -> bool:
    """Набор лежит у нас на диске и может быть удалён?"""
    text = str(key or "")
    return text.startswith(GENERATED_MODEL_PREFIX) or \
        text.startswith(LOCAL_MODEL_PREFIX)


def plan_model_set_removal(key: Any) -> ModelSetRemoval:
    """
    Собрать список файлов набора, ничего не удаляя.

    Вызывается перед подтверждением: пользователь видит ровно то, что исчезнет
    с диска, и суммарный размер.
    """
    text = str(key or "")
    plan = ModelSetRemoval(key=text)
    if not text:
        plan.error = "набор не выбран"
        return plan

    if text.startswith(GENERATED_MODEL_PREFIX):
        directory = GENERATED_MODELS_DIR
        entries = _generated_kit_entries()
        stems = [stem for _k, _cfg, stem, _data in entries]
        stem = next((s for k, _cfg, s, _d in entries if k == text), "")
        plan.name = text[len(GENERATED_MODEL_PREFIX):]
    elif text.startswith(LOCAL_MODEL_PREFIX):
        directory = TRUCKS_MODELS_DIR
        entries = _local_truck_entries()
        stems = [stem for _k, _cfg, stem, _path in entries]
        stem = next((s for k, _cfg, s, _p in entries if k == text), "")
        plan.name = _truck_model_display(stem) if stem else text
    else:
        plan.error = ("серверный набор нельзя удалить из утилиты — "
                      "его файлы лежат на сервере")
        return plan

    if not stem:
        plan.error = "набор не найден на диске"
        return plan

    paths = [p for p in _entries_owned_by(directory, stem, stems)
             if _inside(p, directory)]
    if not paths:
        plan.error = "файлы набора не найдены"
        return plan

    plan.paths = paths
    plan.total_size = sum(_path_size(p) for p in paths)
    return plan


def delete_model_set(key: Any) -> Tuple[bool, str]:
    """
    Удалить файлы набора с диска. Возвращает (успех, сообщение).

    Ничего не бросает: интерфейсу нужен текст для строки состояния, а не
    исключение посреди обработчика кнопки. Удаляется только то, что попало в
    `plan_model_set_removal` — то есть ровно то, что видел пользователь.
    """
    import shutil

    plan = plan_model_set_removal(key)
    if not plan.ok:
        return False, plan.error or "нечего удалять"

    removed, failed = 0, []
    for path in plan.paths:
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError as exc:
            failed.append(f"{os.path.basename(path)}: {exc.strerror or exc}")

    if failed:
        return False, ("удалено {} из {}; не удалось: {}".format(
            removed, len(plan.paths), "; ".join(failed[:3])))
    return True, f"«{plan.name}» удалён ({_fmt_bytes(plan.total_size)})"


def _fmt_bytes(size: int) -> str:
    """Человекочитаемый размер — для сообщений об удалении."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit in ("Б", "КБ") \
                else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} Б"


def get_model_set_config(key: str) -> Optional[Dict[str, Any]]:
    """
    Return the full dict for one model set (e.g. {"model": "Kamaz",
    "ground_plane": 2.25, "max_volume": 10, "cuzov": "...", ...}).

    Lookup follows the same precedence as `load_model_sets`: server first
    (cached per call — cheap because the YAML is tiny), local YAML second.
    Returns None if the key is unknown.
    """
    if not key:
        return None
    # Local truck models resolve straight from disk — no server lookup.
    if str(key).startswith(GENERATED_MODEL_PREFIX):
        return _scan_generated_kits().get(str(key))
    if str(key).startswith(LOCAL_MODEL_PREFIX):
        return _scan_local_truck_models().get(str(key))
    cfg = _fetch_model_sets_from_server() or _load_local_models_yaml()
    val = cfg.get(key)
    return val if isinstance(val, dict) else None


def get_texture_set_config(key: str) -> Optional[Dict[str, Any]]:
    """
    Вернуть полный dict одного текстурного набора
    (diffuse / normal / displacement / roughness, textureRepeatX/Y,
    strength, mesh_distributions и т.д.) из in-memory кэша,
    наполненного с сервера. Возвращает None, если ключа нет.
    """
    if not key:
        return None
    cfg = _TEXTURE_SETS_CACHE or {}
    val = cfg.get(key)
    return val if isinstance(val, dict) else None


def _normalize_texture_rel_path(remote_path: str) -> str:
    """
    Привести относительный путь из textures_napolnitel_config.json к
    каноничному виду без префикса 'textures/' и с прямыми слэшами.
    """
    p = (remote_path or "").strip().replace("\\", "/")
    while p.startswith("/"):
        p = p[1:]
    if p.startswith("textures/"):
        p = p[len("textures/"):]
    return p


def ensure_texture_cached(tls_client, remote_path: str) -> Optional[str]:
    """
    Гарантировать наличие файла текстуры в локальном кэше и вернуть
    абсолютный локальный путь к нему. Если файл уже есть и непустой —
    сразу возвращается без обращения к сети.

    `remote_path` — относительный путь, как он лежит в
    textures_napolnitel_config.json (например, "textures/asphalt_8k/diff.jpg"
    или "asphalt_8k/diff.jpg"). Возвращает None, если скачать не удалось
    и в кэше тоже ничего нет.
    """
    if not remote_path or not isinstance(remote_path, str):
        return None
    rel = _normalize_texture_rel_path(remote_path)
    if not rel or ".." in rel.split("/"):
        return None

    local_path = os.path.join(TEXTURES_CACHE_DIR, *rel.split("/"))

    try:
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
    except OSError:
        pass

    if tls_client is None:
        print(f"[panel_data] ensure_texture_cached: TLS-клиент не задан, "
              f"кэш не пополнен ({remote_path})")
        return None

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tls_client.download_texture_by_path(remote_path, local_path)
    except Exception as exc:
        print(f"[panel_data] ensure_texture_cached({remote_path}) failed: {exc}")
        # Удаляем повисший пустой файл, чтобы при повторной попытке
        # не считать его валидным кэшем.
        try:
            if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                os.remove(local_path)
        except Exception:
            pass
        return None

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    return None


# ---------------------------------------------------------------------------
# Reconstructions (2D → 3D verified results)
# ---------------------------------------------------------------------------
@dataclass
class Reconstruction:
    """
    One 2D-to-3D reconstruction record, normalised across the two
    sources (TLS server and local height_examples/*.json).

    Field reference (some are optional, depending on source):
        name            file basename or server "name"
        path            absolute local path (only for local entries)
        data_type       "ply" | "height"
        is_local        True if loaded from height_examples/, False if
                        from the server's verified-models list
        model           e.g. "FAW J6 8x4" / "Kamaz" / "Shackman-X3000"
        car_number      Russian-style plate or "Неизвестно"
        time            display string ("dd.mm.yyyy HH:MM")
        datetime        parsed datetime for sorting (always present —
                        falls back to datetime.min when unparseable)
        img_file        thumbnail / preview filename
        ply_file        for height entries that have a paired PLY
        filler          filler material descriptor
        target_volume   numeric m³ value or None
        raw             the original dict (server payload or parsed JSON)
                        — handy for debugging and future fields
    """
    name: str = ""
    path: str = ""
    data_type: str = "ply"
    is_local: bool = False
    model: str = ""
    car_number: str = "Неизвестно"
    time: str = ""
    datetime: datetime = field(default_factory=lambda: datetime.min)
    img_file: str = ""
    ply_file: Optional[str] = None
    filler: str = ""
    target_volume: Optional[float] = None
    # Stand snapshots (data_type == "stand") carry the absolute paths to
    # their colour frame and matching depth map. `color_path` is what the
    # manual camera-alignment overlay shows; `depth_path` stays available
    # for the depth pipeline.
    color_path: str = ""
    depth_path: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def _parse_dt(value: Any) -> datetime:
    """
    Best-effort datetime parsing. Tries the legacy "dd.mm.yyyy HH:MM"
    format first, then ISO-8601, then falls back to datetime.min so
    sort order stays deterministic.
    """
    if not value:
        return datetime.min
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return datetime.min
    # Legacy human format used by both the server and the local JSONs.
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.min


def _fetch_reconstructions_from_server(limit: int = 25) -> List[dict]:
    """
    Pull `get_verified_models()` from the active TLS server. Sorted
    newest-first, capped at `limit`. Returns [] on any failure.
    """
    cls = _get_tls_client_cls()
    if cls is None:
        return []
    server = _read_active_tls_server()
    if server is None:
        return []
    host, port = server
    try:
        client = cls(host=host, port=port, timeout=5.0)
        files = client.get_verified_models() or []
    except Exception as exc:
        print(f"[panel_data] TLS get_verified_models failed: {exc}")
        return []

    # Sort by parsed datetime, newest first.
    files = list(files)
    files.sort(key=lambda f: _parse_dt(f.get("datetime") or f.get("time")),
               reverse=True)
    return files[:limit]


def _scan_local_height_examples() -> List[Reconstruction]:
    """
    Walk `height_examples/`, parse each *.json, and return the resulting
    Reconstruction records. Errors on individual files are logged but
    don't abort the scan.
    """
    out: List[Reconstruction] = []
    if not os.path.isdir(HEIGHT_EXAMPLES_DIR):
        return out

    for filename in sorted(os.listdir(HEIGHT_EXAMPLES_DIR)):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(HEIGHT_EXAMPLES_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[panel_data] failed to read {filename}: {exc}")
            continue

        time_str = data.get("time", "")
        rec = Reconstruction(
            name=filename,
            path=path,
            data_type="height",
            is_local=True,
            model=str(data.get("model", "") or ""),
            car_number=str(data.get("car_number", "Неизвестно") or "Неизвестно"),
            time=time_str,
            datetime=_parse_dt(time_str),
            img_file=str(
                data.get("img_file") or data.get("image_path") or ""
            ),
            ply_file=data.get("ply_file"),
            filler=str(data.get("filler", "") or ""),
            target_volume=(
                float(data["target_volume"])
                if data.get("target_volume") is not None
                else None
            ),
            raw=data,
        )
        out.append(rec)
    return out


# Suffix marking the depth map inside a stand snapshot pair, e.g.
# "example-1.png" (colour) <-> "example-1_depth_da3.png" (depth).
_STAND_DEPTH_SUFFIX = "_depth_da3"
# Per-snapshot fill mask, e.g. "example-1_depth_da3-mask.png". These are an
# implementation detail of reconstruction and must NOT appear as their own
# entries in the snapshot list.
_STAND_MASK_SUFFIX = "-mask.png"


def _scan_local_stand_examples() -> List[Reconstruction]:
    """
    Scan `height_examples/stand/` for snapshot PAIRS and return one
    Reconstruction per colour frame:

        <name>.png            -> the regular camera frame (color_path)
        <name>_depth_da3.png  -> the matching depth map (depth_path)

    Tagged data_type="stand". A colour frame without a matching depth
    map is still listed (depth_path left empty) so the alignment overlay
    keeps working. Stand entries have no timestamp, so they sort to the
    bottom of the merged reconstruction list — same place the old local
    examples appeared.
    """
    out: List[Reconstruction] = []
    if not os.path.isdir(STAND_EXAMPLES_DIR):
        return out

    try:
        pngs = [f for f in os.listdir(STAND_EXAMPLES_DIR)
                if f.lower().endswith(".png")]
    except OSError as exc:
        print(f"[panel_data] failed to list stand/: {exc}")
        return out

    # Fill masks (<depth>-mask.png) are not standalone snapshots — skip them.
    mask_files = {f for f in pngs if f.lower().endswith(_STAND_MASK_SUFFIX)}
    depth_files = {
        f for f in pngs
        if f not in mask_files
        and os.path.splitext(f)[0].endswith(_STAND_DEPTH_SUFFIX)
    }
    color_files = [f for f in pngs
                   if f not in mask_files and f not in depth_files]

    for color_name in sorted(color_files):
        stem = os.path.splitext(color_name)[0]
        color_path = os.path.join(STAND_EXAMPLES_DIR, color_name)
        depth_path = os.path.join(
            STAND_EXAMPLES_DIR, f"{stem}{_STAND_DEPTH_SUFFIX}.png"
        )
        if not os.path.exists(depth_path):
            depth_path = ""
        rec = Reconstruction(
            name=color_name,
            path=color_path,
            data_type="stand",
            is_local=True,
            model="Опорный кадр",
            car_number=stem,
            img_file=color_name,
            color_path=color_path,
            depth_path=depth_path,
            raw={"color": color_path, "depth": depth_path},
        )
        out.append(rec)
    return out


RECON_PAGE_SIZE = 25


# ---------------------------------------------------------------------------
# Server-side depth records (image+meta upload pipeline)
# ---------------------------------------------------------------------------
SERVER_DEPTH_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), "toner_server_depth"
)


def _fetch_depth_records_from_server(limit: int = 50) -> List[Reconstruction]:
    """Запрашивает у активного TLS-сервера список depth-загрузок и
    превращает их в Reconstruction(data_type='depth')."""
    cls = _get_tls_client_cls()
    if cls is None:
        print("[panel_data] depth-records: TLS_client class unavailable")
        return []
    server = _read_active_tls_server()
    if server is None:
        print("[panel_data] depth-records: no active TLS server in tls_config.yaml")
        return []
    host, port = server
    try:
        client = cls(host=host, port=port, timeout=5.0)
        records = client.list_depth_records(limit=limit) or []
    except Exception as exc:
        print(f"[panel_data] TLS list_depth_records failed: {exc}")
        return []
    print(f"[panel_data] depth-records: получено {len(records)} с {host}:{port}")

    out: List[Reconstruction] = []
    for rec in records:
        try:
            stem = str(rec.get("stem") or rec.get("image_filename") or "depth")
            dt_str = str(rec.get("datetime") or "")
            dt = _parse_dt(dt_str)
            time_disp = dt.strftime("%d.%m.%Y %H:%M") if dt is not datetime.min else dt_str
            out.append(Reconstruction(
                name=stem,
                path=str(rec.get("record_name") or ""),
                data_type="depth",
                is_local=False,
                model=str(rec.get("model") or ""),
                car_number=str(rec.get("car_number") or "Неизвестно"),
                time=time_disp,
                datetime=dt,
                # Preview thumbnail и overlay используют ФИНАЛЬНОЕ
                # обработанное изображение — masked PNG (после de-barrel +
                # polygon-crop). Это и есть «та же картинка, что мы
                # используем для восстановления по depth_map-е».
                img_file=str(rec.get("masked_output_name") or ""),
                # depth_path / color_path заполняются именами серверных
                # файлов; main_window резолвит их в локальные через
                # resolve_depth_record_files() перед запуском реконструкции.
                color_path=str(rec.get("masked_output_name") or ""),
                depth_path=str(rec.get("anydepth_depth_png_name") or ""),
                filler="",
                target_volume=None,
                raw=dict(rec),
            ))
        except Exception as exc:
            print(f"[panel_data] skip malformed depth record: {exc}")
    return out


def resolve_depth_record_files(rec: Reconstruction) -> Dict[str, str]:
    """Для серверной depth-записи скачивает все нужные файлы
    в SERVER_DEPTH_CACHE_DIR и возвращает их локальные пути:
    {'depth': <path>, 'color': <path>, 'uploaded': <path>?}.
    Файлы переиспользуются, если уже в кеше."""
    if rec is None or rec.data_type != "depth":
        return {}
    cls = _get_tls_client_cls()
    server = _read_active_tls_server()
    if cls is None or server is None:
        return {}
    host, port = server
    try:
        client = cls(host=host, port=port, timeout=30.0)
    except Exception as exc:
        print(f"[panel_data] TLS_client init failed: {exc}")
        return {}

    raw = rec.raw or {}
    depth_name = raw.get("anydepth_depth_png_name")
    # The masked-depth companion (<stem>_depth_mask.png) is hand-made in
    # Photoshop and dropped into the same server folder as the depth PNG, so
    # it's reachable via the same kind="anydepth_png" endpoint. The record
    # JSON doesn't carry a dedicated field for it — we derive the name from
    # the depth name. Missing on the server is fine (most records won't have
    # one): the download attempt is best-effort and silent.
    if depth_name:
        stem, ext = os.path.splitext(depth_name)
        mask_name = f"{stem}_mask{ext}"
    else:
        mask_name = None
    plan = [
        ("depth",      "anydepth_png", depth_name,                          True),
        ("depth_mask", "anydepth_png", mask_name,                           False),
        ("color",      "masked",       raw.get("masked_output_name"),       True),
        ("uploaded",   "uploaded",     raw.get("uploaded_image_name"),      True),
    ]
    out: Dict[str, str] = {}
    os.makedirs(SERVER_DEPTH_CACHE_DIR, exist_ok=True)
    for key, kind, name, required in plan:
        if not name:
            continue
        local_path = os.path.join(SERVER_DEPTH_CACHE_DIR, kind, name)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            out[key] = local_path
            continue
        try:
            client.download_depth_file(kind, name, local_path)
            out[key] = local_path
        except Exception as exc:
            if required:
                print(f"[panel_data] download_depth_file({kind},{name}) failed: {exc}")
            if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                try:
                    os.remove(local_path)
                except OSError:
                    pass
    return out


def load_reconstructions(limit: int = 25) -> List[Reconstruction]:
    """
    Return the merged list of reconstructions, newest first.

    Источники:
      * `TLS_client.get_verified_models()` — серверные PLY/height записи
        (data_type: "ply" или "height").
      * `TLS_client.list_depth_records()` — серверные depth-загрузки
        (data_type: "depth"), от нашего pipeline /upload_image.

    Локальные источники (height_examples/*.json и stand-снимки) отключены
    — в списке остаются только серверные записи.
    """
    server_files = _fetch_reconstructions_from_server(limit=limit)

    out: List[Reconstruction] = []

    # Server-side entries first (they're already sorted, but we re-sort
    # the merged list at the end anyway).
    for f in server_files:
        time_str = str(f.get("datetime") or f.get("time") or "")
        rec = Reconstruction(
            name=str(f.get("name", "") or ""),
            path=str(f.get("path", "") or ""),
            data_type=str(f.get("data_type", "ply") or "ply"),
            is_local=False,
            model=str(f.get("model", "") or ""),
            car_number=str(f.get("car_number", "Неизвестно") or "Неизвестно"),
            time=time_str,
            datetime=_parse_dt(time_str),
            img_file=str(f.get("img_file", "") or ""),
            ply_file=f.get("ply_file"),
            filler=str(f.get("filler", "") or ""),
            target_volume=(
                float(f["target_volume"])
                if f.get("target_volume") is not None
                else None
            ),
            raw=dict(f),
        )
        out.append(rec)

    # Локальные источники (height_examples/*.json и stand-снимки) отключены —
    # в списке 2D→3D остаются только серверные записи.
    # out.extend(_scan_local_height_examples())
    # out.extend(_scan_local_stand_examples())

    # Серверные depth-записи (от /upload_image pipeline).
    out.extend(_fetch_depth_records_from_server(limit=limit))

    # Newest first; entries with unparseable dates fall to the bottom.
    out.sort(key=lambda r: r.datetime, reverse=True)
    return out


# ---------------------------------------------------------------------------
# On-demand image fetch from the TLS server
# ---------------------------------------------------------------------------
SERVER_IMAGE_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), "toner_server_images"
)


def download_server_image(filename: str) -> Optional[str]:
    """
    Fetch one preview image from the active TLS server and return the
    local path it was cached to. Returns None on any failure.

    Caching: files land in a stable per-session temp directory
    (`%TEMP%/toner_server_images`). If the file is already there we skip
    the network round-trip entirely — opening the same record twice is
    free after the first hit.

    Mirrors the legacy gui.py `show_fullscreen_image` server branch:
        temp_dir = %TEMP%/vizutil_fullsize_images
        client.download_file(img_file, local_img_path)

    Synchronous on purpose — preview images are tiny (a few hundred KB)
    and the photo overlay opens on a user click, so a brief spinner is
    fine. If we ever need async, wrap this call in a QRunnable.
    """
    name = (filename or "").strip()
    if not name:
        return None

    # Reject anything that looks like an absolute path or a relative
    # traversal — `download_file` builds the URL from `name` directly.
    if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
        return None

    os.makedirs(SERVER_IMAGE_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(SERVER_IMAGE_CACHE_DIR, name)

    # Cache hit?
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    cls = _get_tls_client_cls()
    if cls is None:
        return None
    server = _read_active_tls_server()
    if server is None:
        return None
    host, port = server

    try:
        client = cls(host=host, port=port, timeout=15.0)
        client.download_file(name, local_path)
    except Exception as exc:
        print(f"[panel_data] download_server_image({name}) failed: {exc}")
        # Don't leave a half-written file on disk.
        try:
            if os.path.exists(local_path) and os.path.getsize(local_path) == 0:
                os.remove(local_path)
        except Exception:
            pass
        return None

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    return None
