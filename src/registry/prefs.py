# -*- coding: utf-8 -*-
"""
Что диалог загрузки помнит между открытиями.

Большая часть полей диалога выводится из самого комплекта (ключ, объём,
камера, points_3d) — их пересчитывать дешевле, чем хранить. Но есть решения,
которые принимает человек и которые из комплекта не следуют: как называется
модель на сервере, надо ли беречь серверные points_3d, поправленный вручную
объём или пресет камеры. Их приходилось вводить заново при каждом открытии,
и «Сохранить точки с сервера» забывали включить — а полная замена молча
затирала подгонку оператора.

Пишет сюда диалог только те поля, которые отличаются от посчитанных по
комплекту: перегенерировали кузов — новые числа приезжают из набора, а не
из вчерашнего снимка.

Файл лежит рядом с конфигами (`config/registry_ui.json`), ключ записи — ключ
набора в списке кузовов. Ничего секретного тут нет, но и в git ему незачем:
это память конкретной машины.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from src.registry.settings import PROJECT_ROOT

PREFS_PATH = os.path.join(PROJECT_ROOT, "config", "registry_ui.json")

#: Поля, которые помним, — все настраиваемые в диалоге.
FIELDS = ("key", "display_name", "mode", "max_volume", "ground_plane",
          "points_3d", "keep_points", "camera_on", "camera", "textures",
          "web")


def _read_all() -> Dict[str, Any]:
    if not os.path.isfile(PREFS_PATH):
        return {}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:                              # noqa: BLE001
        print(f"[registry] не прочитан {PREFS_PATH}: {exc}")
        return {}


def load_prefs(set_key: str) -> Dict[str, Any]:
    """Что помним про этот набор. Пусто — открываем как в первый раз."""
    if not set_key:
        return {}
    entry = (_read_all().get("models") or {}).get(set_key)
    return dict(entry) if isinstance(entry, dict) else {}


def save_prefs(set_key: str, values: Dict[str, Any]) -> None:
    """Запомнить решения оператора по набору. Ошибки записи не критичны."""
    if not set_key:
        return
    data = _read_all()
    models = data.get("models")
    if not isinstance(models, dict):
        models = {}
    entry = {name: values[name] for name in FIELDS if name in values}
    if models.get(set_key) == entry:
        return
    models[set_key] = entry
    data["models"] = models
    try:
        os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)
        with open(PREFS_PATH, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:                              # noqa: BLE001
        print(f"[registry] не сохранён {PREFS_PATH}: {exc}")
