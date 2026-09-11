# -*- coding: utf-8 -*-
"""
Забрать `points_3d` модели с сервера, чтобы вернуть их туда же нетронутыми.

Зачем это нужно
---------------
`points_3d` — единственное поле записи конфига, которое правят НЕ у нас:
четыре точки верхней кромки кузова подгоняет оператор через утилиту
(`setModelPoints`), и правка живёт только на сервере. А загрузка модели в
реестр в режиме «полная замена» пересобирает запись конфига с нуля из
присланного `meta` — то есть обновление геометрии кузова затирает чужую
подгонку молча. Отсюда опция «сохранить точки с сервера»: перед отправкой
точки читаются с сервера и уезжают обратно как есть.

Откуда читать: два дерева
-------------------------
На сервере `config/` существует в двух копиях (см. MODEL_REGISTRY_API.md §2.1):

* `photo-to-volume/config` — основное дерево, его отдаёт карточка реестра
  (`GET /models/{key}` -> `config.points_3d`);
* `operation-3d-service/config` — зеркало, его отдаёт TLS-9999
  (`get_models_config`), и именно в него пишет `setModelPoints`.

Реестр при загрузке пишет запись в оба дерева, а расходятся они как раз по
`points_3d` — по документации так у 13 моделей. Значит, спрашивать надо
зеркало: там лежит то, что человек реально подогнал. Основное дерево —
запасной вариант, если TLS-сервер недоступен.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.registry.settings import TLS_CONFIG_PATH

#: Сколько ждать TLS-9999: это маленький JSON, а не загрузка комплекта.
TLS_TIMEOUT = 10.0


@dataclass
class RemotePoints:
    """Что нашлось на сервере и где именно."""

    points: Optional[List[List[float]]] = None
    #: Человекочитаемый источник — уходит в журнал диалога.
    source: str = ""
    #: Точки из основного дерева, если они отличаются от зеркальных.
    other: Optional[List[List[float]]] = None
    #: Почему не получилось (или что стоит знать), по строке на попытку.
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.points is not None


def normalize_points(value: Any) -> Optional[List[List[float]]]:
    """Четыре точки `[x, y, z]` из чего угодно — или None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out: List[List[float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        try:
            out.append([float(v) for v in item])
        except (TypeError, ValueError):
            return None
    return out


def _same(a: Optional[List[List[float]]],
          b: Optional[List[List[float]]]) -> bool:
    if a is None or b is None:
        return False
    return all(abs(u - v) < 1e-6 for pa, pb in zip(a, b)
               for u, v in zip(pa, pb))


def _active_tls_server() -> Optional[Tuple[str, int]]:
    """Хост и порт активного TLS-сервера из `config/tls_config.yaml`."""
    if not os.path.exists(TLS_CONFIG_PATH):
        return None
    try:
        with open(TLS_CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return None
    for srv in data.get("servers", []) or []:
        if isinstance(srv, dict) and srv.get("active") and srv.get("host"):
            return str(srv["host"]), int(srv.get("port", 9999))
    return None


def _lookup(config: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """Запись модели в конфиге: точное совпадение ключа, иначе без регистра."""
    if not isinstance(config, dict):
        return None
    record = config.get(key)
    if isinstance(record, dict):
        return record
    low = key.lower()
    for name, value in config.items():
        if isinstance(value, dict) and str(name).lower() == low:
            return value
    return None


def points_from_mirror(key: str) -> Tuple[Optional[List[List[float]]], str]:
    """Точки из дерева `operation-3d-service` (TLS-9999). ("", если нет)."""
    server = _active_tls_server()
    if server is None:
        return None, ("в config/tls_config.yaml нет сервера с active: true — "
                      "не у кого спросить текущие точки")
    host, port = server
    try:
        from src.core.TLS_client import TLS_client
        client = TLS_client(host=host, port=port, timeout=TLS_TIMEOUT)
        config = client.get_models_config()
    except Exception as exc:
        return None, f"TLS {host}:{port} не отдал конфиг моделей: {exc}"

    record = _lookup(config or {}, key)
    if record is None:
        return None, f"на TLS {host}:{port} нет модели «{key}» в конфиге"
    points = normalize_points(record.get("points_3d"))
    if points is None:
        return None, (f"у модели «{key}» на TLS {host}:{port} нет пригодных "
                      "points_3d")
    return points, f"TLS {host}:{port}, get_models_config"


def points_from_registry(registry_client: Any, key: str
                         ) -> Tuple[Optional[List[List[float]]], str]:
    """Точки из карточки реестра (дерево `photo-to-volume`)."""
    try:
        model = registry_client.get_model(key)
    except Exception as exc:
        return None, f"реестр не отдал карточку модели: {exc}"
    if not model:
        return None, f"в реестре нет модели «{key}»"
    points = normalize_points((model.get("config") or {}).get("points_3d"))
    if points is None:
        return None, f"в записи конфига модели «{key}» нет пригодных points_3d"
    return points, "карточка реестра, config.points_3d"


def fetch_remote_points(registry_client: Any, key: str) -> RemotePoints:
    """
    Актуальные `points_3d` модели `key` — сперва зеркало, потом реестр.

    Ничего не выбрасывает: неудачные попытки складываются в `notes`, а
    `ok == False` значит «сохранять нечего, решай сам».
    """
    out = RemotePoints()
    key = str(key or "").strip()
    if not key:
        out.notes.append("не задан ключ модели")
        return out

    mirror, mirror_note = points_from_mirror(key)
    if mirror is None:
        out.notes.append(mirror_note)

    registry, registry_note = points_from_registry(registry_client, key)
    if registry is None:
        out.notes.append(registry_note)

    if mirror is not None:
        out.points, out.source = mirror, mirror_note
        if registry is not None and not _same(mirror, registry):
            out.other = registry
            out.notes.append(
                "в дереве пайплайна точки другие — на сервер уедут те, что "
                "показывает утилита (зеркало)")
    elif registry is not None:
        out.points, out.source = registry, registry_note
    return out
