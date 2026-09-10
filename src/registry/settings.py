# -*- coding: utf-8 -*-
"""
Откуда утилита узнаёт адрес реестра моделей.

Ничего вводить руками не нужно: реестр живёт на том же хосте, что и активный
TLS-сервер из `config/tls_config.yaml`, только на своём порту. То есть выбрал
сервер в конфиге — получил и реестр.

Порядок разрешения (что раньше, то главнее):

1. переменные окружения `MODEL_REGISTRY_URL` / `MODEL_REGISTRY_TOKEN` —
   ими удобно подсунуть тестовый реестр, не трогая конфиг;
2. секция `model_registry:` в `config/tls_config.local.yaml` — файл не в git;
3. секция `model_registry:` в `config/tls_config.yaml`;
4. активный сервер из `tls_config.yaml` + порт по умолчанию (8099).

Почему два файла
----------------
Адрес реестра — не секрет, ему место в общем `tls_config.yaml` рядом с
адресами серверов. А токен на запись — секрет, и в отслеживаемый git-ом файл
его класть нельзя: он останется в истории навсегда, даже если потом стереть.
Поэтому рядом лежит `tls_config.local.yaml` (в `.gitignore`), секция
`model_registry:` из которого перекрывает основную по полям. Один механизм на
оба случая: то, что общее для команды, — в основном файле, машинное и
секретное — в локальном.

Токен нужен только если демон запущен без `--allow-anonymous-write`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

TLS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "tls_config.yaml")

#: Локальные переопределения (токен, свой адрес). В git не попадает.
LOCAL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config",
                                 "tls_config.local.yaml")

ENV_URL = "MODEL_REGISTRY_URL"
ENV_TOKEN = "MODEL_REGISTRY_TOKEN"

#: Порт демона `model-registry` по умолчанию (его же ставит сам демон).
DEFAULT_PORT = 8099


@dataclass
class RegistryEndpoint:
    """Куда стучаться и чем представляться."""

    url: str = ""
    token: str = ""
    verify_tls: bool = True
    #: Человекочитаемое «откуда это взялось» — показывается в диалоге, чтобы
    #: при неверном адресе было понятно, какой файл править.
    source: str = ""
    #: Почему адреса нет (нет активного сервера, не читается конфиг…).
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.url)


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[registry] не прочитан {path}: {exc}")
        return {}


def _read_tls_config() -> Dict[str, Any]:
    return _read_yaml(TLS_CONFIG_PATH)


def _registry_section() -> Tuple[Dict[str, Any], bool]:
    """
    Настройки реестра из обоих конфигов и признак «локальный файл вмешался».

    Локальный перекрывает основной по полям, а не целиком: адрес обычно общий
    (лежит в git), а секретный токен добавляется рядом только на этой машине.
    """
    section = _read_tls_config().get("model_registry") or {}
    if not isinstance(section, dict):
        section = {}
    local = _read_yaml(LOCAL_CONFIG_PATH).get("model_registry") or {}
    if not isinstance(local, dict) or not local:
        return dict(section), False
    merged = dict(section)
    merged.update(local)
    return merged, True


def _active_server(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for srv in data.get("servers", []) or []:
        if isinstance(srv, dict) and srv.get("active") and srv.get("host"):
            return srv
    return None


def resolve_registry() -> RegistryEndpoint:
    """Адрес реестра для текущего окружения. Никаких диалогов и вопросов."""
    data = _read_tls_config()
    section, from_local = _registry_section()
    where = ("config/tls_config.local.yaml" if from_local
             else "config/tls_config.yaml")

    token = str(section.get("token") or "")
    verify = bool(section.get("verify_tls", True))

    env_token = os.environ.get(ENV_TOKEN, "").strip()
    if env_token:
        token = env_token

    env_url = os.environ.get(ENV_URL, "").strip()
    if env_url:
        return RegistryEndpoint(url=env_url.rstrip("/"), token=token,
                                verify_tls=verify,
                                source=f"переменная окружения {ENV_URL}")

    url = str(section.get("url") or "").strip()
    if url:
        return RegistryEndpoint(url=url.rstrip("/"), token=token,
                                verify_tls=verify,
                                source=f"{where}, секция model_registry")

    server = _active_server(data)
    if server is None:
        return RegistryEndpoint(
            error="В config/tls_config.yaml нет сервера с active: true — "
                  "непонятно, у какого хоста спрашивать реестр моделей.")

    host = str(server.get("host"))
    port = int(section.get("port") or DEFAULT_PORT)
    scheme = str(section.get("scheme") or "http")
    prefix = str(section.get("prefix") or "").strip("/")
    name = str(server.get("name") or host)
    url = f"{scheme}://{host}:{port}" + (f"/{prefix}" if prefix else "")
    return RegistryEndpoint(
        url=url, token=token, verify_tls=verify,
        source=f"активный сервер из config/tls_config.yaml ({name})")
