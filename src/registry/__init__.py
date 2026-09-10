# -*- coding: utf-8 -*-
"""
Работа с реестром моделей photo-to-volume (HTTP-демон `model-registry`).

`client`   — сам протокол: чтение, загрузка/замена, удаление.
`payload`  — из чего собрать запрос для набора из списка кузовов.
`settings` — адрес и токен реестра.

Пакет намеренно не зависит ни от Qt, ни от Panda3D (кроме `bam_to_obj`,
который импортирует Panda внутри функции), чтобы его можно было дёргать из
скриптов и с сервера.
"""

from src.registry.client import (COMMON_ROLES, FILE_ROLES, REQUIRED_ROLES,
                                 ROLE_HINTS, ROLE_LABELS, ROLE_TARGET_SUFFIX,
                                 ModelRegistry, ModelRegistryError,
                                 RegistryConnectionError, UploadCancelled,
                                 suggest_key, validate_key)
from src.registry.payload import (UploadPlan, bam_to_obj, build_upload_plan,
                                  detect_role_files, detect_web_files,
                                  meta_from_form)
from src.registry.settings import RegistryEndpoint, resolve_registry

__all__ = [
    "COMMON_ROLES", "FILE_ROLES", "REQUIRED_ROLES", "ROLE_HINTS",
    "ROLE_LABELS", "ROLE_TARGET_SUFFIX", "ModelRegistry", "ModelRegistryError",
    "RegistryConnectionError", "UploadCancelled", "suggest_key",
    "validate_key", "UploadPlan", "bam_to_obj", "build_upload_plan",
    "detect_role_files", "detect_web_files", "meta_from_form",
    "RegistryEndpoint", "resolve_registry",
]
