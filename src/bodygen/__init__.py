# -*- coding: utf-8 -*-
"""
Генератор кузовов — опциональный модуль утилиты.

Модуль СОЗНАТЕЛЬНО не импортирует ни PyQt, ни Panda3D на уровне пакета: он
должен запускаться и на сервере, где ни того, ни другого нет. Всё, что нужно
интерфейсу, лежит в `src.ui.bodygen_dialog`; здесь только расчётная часть.

Если сам пакет `body_builder` не найден, модуль не падает: `probe()` вернёт
причину, а панель просто покажет карточку выключенной.
"""

from .service import (BodyGenParams, BodyGenResult, generate, list_chassis,
                      list_models, probe)

__all__ = ["BodyGenParams", "BodyGenResult", "generate", "list_chassis",
           "list_models", "probe"]
