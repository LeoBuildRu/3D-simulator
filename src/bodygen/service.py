# -*- coding: utf-8 -*-
"""
Расчётная часть генератора кузовов: обёртка над пакетом `body_builder`.

Здесь нет ни Qt, ни Panda3D — модуль обязан работать в headless-режиме на
сервере, где графического стека нет вовсе. Интерфейсная часть живёт отдельно
(`src/ui/bodygen_dialog.py`) и обращается сюда через `generate()`.

Где искать `body_builder`
-------------------------
Пакет лежит в соседнем репозитории и НЕ копируется в утилиту — иначе пришлось бы
поддерживать две копии. Путь берётся, в порядке убывания приоритета: переменная
окружения `IQOKO_BODY_BUILDER`, затем известные места рядом с проектом. Если не
нашёлся — `probe()` возвращает причину, а не исключение: генератор опционален,
и без него утилита должна работать как раньше.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: Где искать пакет генератора, по убыванию приоритета.
SEARCH_PATHS = [
    os.environ.get("IQOKO_BODY_BUILDER", ""),
    os.path.join(PROJECT_ROOT, "..", "AlexeyPlys", "repo", "body_builder"),
    r"G:\IQoko\AlexeyPlys\repo\body_builder",
    os.path.join(PROJECT_ROOT, "body_builder"),
]

#: Куда складывать собранные комплекты по умолчанию.
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "generated")

_resolved_path: Optional[str] = None


def find_body_builder() -> Optional[str]:
    """Каталог, в котором лежит пакет `body_builder`, либо None."""
    global _resolved_path
    if _resolved_path:
        return _resolved_path
    for raw in SEARCH_PATHS:
        if not raw:
            continue
        path = os.path.abspath(raw)
        if os.path.isfile(os.path.join(path, "body_builder", "spec.py")):
            _resolved_path = path
            return path
    return None


def _ensure_import() -> None:
    path = find_body_builder()
    if not path:
        raise ImportError(
            "пакет body_builder не найден; укажите путь в переменной "
            "окружения IQOKO_BODY_BUILDER")
    if path not in sys.path:
        sys.path.insert(0, path)


def probe() -> Dict[str, Any]:
    """
    Что доступно на этой машине. Ничего не бросает — панель по этому словарю
    решает, показывать генератор активным или выключенным.
    """
    info: Dict[str, Any] = {
        "path": find_body_builder(),
        "available": False,
        "reason": "",
        "chassis": [],
        "draco": False,
        "decimate": False,
        "panda": False,
        "volume_calculator": False,
    }
    if not info["path"]:
        info["reason"] = "пакет body_builder не найден"
        return info
    try:
        _ensure_import()
        from body_builder.assembly import load_profiles
        info["chassis"] = sorted(load_profiles().keys())
        info["available"] = True
    except Exception as exc:
        info["reason"] = f"body_builder не импортируется: {exc}"
        return info

    for mod, key in (("DracoPy", "draco"), ("fast_simplification", "decimate"),
                     ("panda3d.core", "panda")):
        try:
            __import__(mod)
            info[key] = True
        except Exception:
            pass
    try:
        from body_builder.extract import VC_DIR
        info["volume_calculator"] = os.path.isfile(
            os.path.join(VC_DIR, "Volume_calculator.py"))
    except Exception:
        pass
    if not info["chassis"]:
        info["reason"] = ("каталог шасси пуст — запустите prepare_chassis.py "
                          "в body_builder")
    return info


def list_models() -> List[str]:
    """Ключи справочника `truck_models.TRUCKS`."""
    try:
        _ensure_import()
        from body_builder.catalog import list_models as _lm
        return _lm()
    except Exception:
        return []


def list_chassis() -> List[str]:
    try:
        _ensure_import()
        from body_builder.assembly import load_profiles
        return sorted(load_profiles().keys())
    except Exception:
        return []


# --------------------------------------------------------------------------- #

@dataclass
class BodyGenParams:
    """Всё, что задаёт пользователь. Ровно то же принимает `make_body.py`."""

    source: str = "catalog"          # "catalog" | "ply" | "spec"
    model_key: str = ""              # для source="catalog"
    ply_path: str = ""               # для source="ply"
    spec_path: str = ""              # для source="spec"
    cloud_model: str = ""            # ключ каталога для разбора облака
    rect_width: float = 0.0
    rect_length: float = 0.0

    name: str = ""
    out_dir: str = DEFAULT_OUT_DIR
    chassis: str = "auto"            # "8x4" | "6x4" | "auto" | "none"
    heap: float = 1.0

    density: float = 400.0
    atlas: int = 4096
    with_ao: bool = True
    gltf_decimate: float = 0.0
    gltf_texture_max: int = 0

    paint: str = ""                  # "#RRGGBB"
    wear: float = 0.45
    dirt: float = 0.62
    seed: int = 0

    parts: bool = True
    write_bam: bool = True


@dataclass
class BodyGenResult:
    ok: bool
    name: str = ""
    out_dir: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    error: str = ""
    seconds: float = 0.0


Progress = Callable[[str], None]


def generate(params: BodyGenParams,
             progress: Optional[Progress] = None) -> BodyGenResult:
    """
    Собрать комплект. Никогда не бросает: ошибку возвращает в результате.

    Вызов длинный (минуты на полном разрешении), поэтому в интерфейсе его
    запускают в отдельном потоке, а сюда передают `progress` для строки
    состояния.
    """
    say = progress or (lambda _msg: None)
    t0 = time.time()
    try:
        _ensure_import()
        from body_builder.pipeline import build_model
        from body_builder.spec import BodySpec

        spec = _load_spec(params, say)
        if spec is None:
            return BodyGenResult(ok=False,
                                 error="не удалось получить описание кузова",
                                 seconds=time.time() - t0)

        if params.name:
            spec.name = params.name
        if params.paint:
            spec.appearance.paint_rgb = _parse_color(params.paint)
        spec.appearance.wear = float(params.wear)
        spec.appearance.dirt = float(params.dirt)
        spec.appearance.seed = int(params.seed)

        out_dir = params.out_dir or DEFAULT_OUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        say("сборка геометрии и текстур…")
        result = build_model(
            spec, out_dir,
            density=float(params.density),
            atlas_max=int(params.atlas),
            with_ao=bool(params.with_ao),
            write_glb=False,
            chassis=(params.chassis or None),
            heap=float(params.heap),
            parts=bool(params.parts),
            gltf_decimate=float(params.gltf_decimate),
            gltf_texture_max=int(params.gltf_texture_max),
            write_bam=bool(params.write_bam),
            verbose=True,
        )
        say("готово")
        return BodyGenResult(ok=True, name=spec.name, out_dir=out_dir,
                             files=dict(result.files),
                             summary=result.summary(),
                             seconds=time.time() - t0)
    except Exception as exc:
        traceback.print_exc()
        return BodyGenResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                             seconds=time.time() - t0)


def _load_spec(params: BodyGenParams, say: Progress):
    from body_builder.spec import BodySpec

    if params.source == "spec" and params.spec_path:
        say("чтение описания…")
        return BodySpec.from_json(params.spec_path)

    if params.source == "ply" and params.ply_path:
        say("разбор облака точек (это самая долгая часть)…")
        from body_builder.extract import spec_from_ply
        return spec_from_ply(
            params.ply_path,
            model=(params.cloud_model or None),
            name=(params.name or None),
            rect_width=(params.rect_width or None),
            rect_length=(params.rect_length or None),
        )

    say("описание из справочника…")
    from body_builder.catalog import spec_from_catalog
    return spec_from_catalog(params.model_key or "Shacman X3000 8x4 (стандарт)")


def _parse_color(text: str):
    """«#RRGGBB» -> линейный RGB (в спеке цвет хранится линейным)."""
    t = text.strip().lstrip("#")
    if len(t) != 6:
        raise ValueError("цвет ожидается в виде #RRGGBB")
    srgb = [int(t[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                 for c in srgb)
