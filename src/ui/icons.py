# icons.py
# ---------------------------------------------------------------------------
# Линейные иконки в духе SF Symbols.
#
# Иконки описаны SVG-разметкой на сетке 24x24 прямо в коде: никаких бинарных
# ассетов, и любую можно перекрасить под состояние (обычная / выключенная /
# активная). `{c}` в разметке — место для цвета.
#
#   icon("camera")                       -> QIcon (обычная + disabled + on)
#   pixmap("truck", "#FFFFFF", 16)       -> QPixmap
#   svg_file("chevron_down", "#98989F")  -> путь к .svg для QSS `image:`
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
from functools import lru_cache

# Толщина линии по умолчанию: у SF Symbols в regular-начертании около 1.6-1.8
# на 24 px.
_STROKE = 1.7

# name -> (разметка, толщина линии или None)
_ICONS: dict[str, tuple[str, float | None]] = {
    # --- навигация / служебные ---------------------------------------------
    "chevron_down":  ('<path d="M6 9.5l6 6 6-6"/>', 2.0),
    "chevron_up":    ('<path d="M6 14.5l6-6 6 6"/>', 2.0),
    "chevron_right": ('<path d="M9.5 6l6 6-6 6"/>', 2.0),
    "chevron_left":  ('<path d="M14.5 6l-6 6 6 6"/>', 2.0),
    "check":         ('<path d="M5 12.5l4.5 4.5L19 7.5"/>', 2.0),
    "check_bold":    ('<path d="M5 12.5l4.5 4.5L19 7.5"/>', 3.2),
    "close":         ('<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>', 1.9),
    "plus":          ('<path d="M12 5v14M5 12h14"/>', 1.9),
    "ellipsis":      ('<circle cx="6" cy="12" r="1.4" fill="{c}"/>'
                      '<circle cx="12" cy="12" r="1.4" fill="{c}"/>'
                      '<circle cx="18" cy="12" r="1.4" fill="{c}"/>', None),
    "info":          ('<circle cx="12" cy="12" r="8.5"/>'
                      '<path d="M12 11v5.5"/>'
                      '<circle cx="12" cy="7.8" r="1" fill="{c}" stroke="none"/>',
                      None),
    "help":          ('<circle cx="12" cy="12" r="8.5"/>'
                      '<path d="M9.6 9.6a2.5 2.5 0 1 1 3.4 2.3c-.6.3-1 .8-1 1.4v.6"/>'
                      '<circle cx="12" cy="16.8" r="1" fill="{c}" stroke="none"/>',
                      None),
    "keyboard":      ('<rect x="2.5" y="6" width="19" height="12" rx="2.2"/>'
                      '<path d="M8 14.5h8" />'
                      '<g fill="{c}" stroke="none">'
                      '<circle cx="6.3" cy="10" r="1"/><circle cx="9.4" cy="10" r="1"/>'
                      '<circle cx="12.5" cy="10" r="1"/><circle cx="15.6" cy="10" r="1"/>'
                      '<circle cx="18.2" cy="10" r="1"/></g>', None),
    "search":        ('<circle cx="10.5" cy="10.5" r="6.5"/>'
                      '<path d="M15.5 15.5l5 5"/>', None),
    "zoom_in":       ('<circle cx="10.5" cy="10.5" r="6.5"/>'
                      '<path d="M15.5 15.5l5 5M10.5 7.8v5.4M7.8 10.5h5.4"/>', None),
    "expand":        ('<path d="M14 4h6v6M10 20H4v-6M20 4l-6.5 6.5M4 20l6.5-6.5"/>',
                      None),
    "copy":          ('<rect x="8.5" y="8.5" width="12" height="12" rx="2.2"/>'
                      '<path d="M15.5 8.5V5.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"/>',
                      None),
    "trash":         ('<path d="M4 7h16M9.5 7V4.5h5V7M6.2 7l.9 12.3a1.5 1.5 0 0 0 1.5 1.2h6.8'
                      'a1.5 1.5 0 0 0 1.5-1.2L17.8 7M10 11v5.5M14 11v5.5"/>', None),
    "upload":        ('<path d="M7 18.5a4.5 4.5 0 0 1-.7-8.9A6 6 0 0 1 17.9 9a4 4 0 0 1-.4 9.5"/>'
                      '<path d="M12 13v7.5M9 15.8l3-3 3 3"/>', None),
    "download":      ('<path d="M12 3.5v12M7 11l5 5 5-5M4.5 20.5h15"/>', None),
    "arrow_down_circle": ('<circle cx="12" cy="12" r="8.5"/>'
                          '<path d="M12 7.5v9M8.2 12.8l3.8 3.8 3.8-3.8"/>', None),
    "arrow_up_right": ('<path d="M7.5 16.5l9-9M9 7.5h7.5V15"/>', None),
    "rotate_ccw":    ('<path d="M4 12.5A8 8 0 1 0 6.5 6.2L4 8.6"/>'
                      '<path d="M4 4.2v4.4h4.4"/>', None),
    "swap":          ('<path d="M4 8.5h15l-3.5-3.5M20 15.5H5l3.5 3.5"/>', None),
    "sliders":       ('<path d="M4 7h8.5M17.5 7H20M4 17h2.5M11.5 17H20"/>'
                      '<circle cx="15" cy="7" r="2.4"/><circle cx="9" cy="17" r="2.4"/>',
                      None),
    "display":       ('<rect x="2.8" y="4" width="18.4" height="12.5" rx="2"/>'
                      '<path d="M8.5 20.5h7M12 16.5v4"/>', None),
    "folder":        ('<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5'
                      'a2 2 0 0 1-2-2z"/>', None),
    "doc":           ('<path d="M6.5 3.5h7.5l4.5 4.5v11.5a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-15'
                      'a1 1 0 0 1 1-1z"/><path d="M14 3.5V8h4.5"/>', None),
    "clock":         ('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>', None),
    "number":        ('<path d="M9.5 4L7.5 20M16.5 4l-2 16M4.5 9h15.5M4 15h15.5"/>', None),
    "eye":           ('<path d="M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12s-3.5 6.5-9.5 6.5'
                      'S2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>', None),
    "eye_off":       ('<path d="M9.9 5.8A9.9 9.9 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a16.5 16.5 0 0 1-2.4 3.2'
                      'M6.4 7.3C3.9 9 2.5 12 2.5 12s3.5 6.5 9.5 6.5c1.7 0 3.2-.5 4.5-1.2'
                      'M9.9 9.9a3 3 0 0 0 4.2 4.2M4 4l16 16"/>', None),
    "dot":           ('<circle cx="12" cy="12" r="4" fill="{c}" stroke="none"/>', None),

    # --- камера / вид ------------------------------------------------------
    "move":          ('<path d="M12 3v18M3 12h18M9.2 5.8L12 3l2.8 2.8M9.2 18.2L12 21l2.8-2.8'
                      'M5.8 9.2L3 12l2.8 2.8M18.2 9.2L21 12l-2.8 2.8"/>', None),
    "tripod":        ('<rect x="6" y="3.5" width="12" height="7.5" rx="1.8"/>'
                      '<circle cx="12" cy="7.25" r="1.6"/>'
                      '<path d="M12 11v3.2M12 14.2L7 21M12 14.2L17 21M12 14.2V21"/>', None),
    "truck":         ('<path d="M2.5 6.5a1 1 0 0 1 1-1h9.5a1 1 0 0 1 1 1V16H2.5z"/>'
                      '<path d="M14 9h3.8l3.2 3.6V16H14"/>'
                      '<circle cx="7" cy="17.5" r="2" fill="{c}" stroke="none"/>'
                      '<circle cx="17.2" cy="17.5" r="2" fill="{c}" stroke="none"/>', None),
    "camera":        ('<path d="M4 7.5h3l1.6-2.5h6.8L17 7.5h3a1.5 1.5 0 0 1 1.5 1.5v9'
                      'a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 18V9A1.5 1.5 0 0 1 4 7.5z"/>'
                      '<circle cx="12" cy="13" r="3.5"/>', None),
    "bookmark":      ('<path d="M7 3.5h10a1 1 0 0 1 1 1V20.5l-6-4-6 4V4.5a1 1 0 0 1 1-1z"/>', None),
    "bookmark_plus": ('<path d="M7 3.5h10a1 1 0 0 1 1 1V20.5l-6-4-6 4V4.5a1 1 0 0 1 1-1z"/>'
                      '<path d="M12 7.3v5.4M9.3 10h5.4"/>', None),
    "sun":           ('<circle cx="12" cy="12" r="4"/>'
                      '<path d="M12 2.5v2M12 19.5v2M4.6 4.6L6 6M18 18l1.4 1.4M2.5 12h2'
                      'M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4"/>', None),
    "moon":          ('<path d="M19.5 14.5A7.8 7.8 0 1 1 9.5 4.5a6.3 6.3 0 0 0 10 10z"/>', None),
    "pip":           ('<rect x="2.5" y="4.5" width="19" height="15" rx="2.2"/>'
                      '<rect x="11.5" y="11.5" width="7" height="5" rx="1" fill="{c}"/>', None),
    "angle":         ('<path d="M4 19L19 6M4 19h16"/>'
                      '<path d="M12.5 19a8.5 8.5 0 0 0-2.4-5.9"/>', None),
    "horizon":       ('<circle cx="12" cy="12" r="8.5"/><path d="M4.2 14.7l15.6-5.4"/>', None),
    "halfcircle":    ('<circle cx="12" cy="12" r="8.5"/>'
                      '<path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="{c}"/>', None),
    "cursor":        ('<path d="M5.5 3.5l13 6.2-5.6 1.9-2.3 5.6z"/>'
                      '<path d="M16 16.5v5M13.5 19h5"/>', None),
    "target":        ('<circle cx="12" cy="12" r="7.5"/><circle cx="12" cy="12" r="2.5"/>'
                      '<path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>', None),
    "photo":         ('<rect x="3" y="4.5" width="18" height="15" rx="2.2"/>'
                      '<circle cx="8.5" cy="9.5" r="1.7"/>'
                      '<path d="M3.5 17l5-5 4 4 2.7-2.7L20.5 18.5"/>', None),

    # --- разделы -----------------------------------------------------------
    "cube":          ('<path d="M12 2.8l8.5 4.6v9.2L12 21.2l-8.5-4.6V7.4z"/>'
                      '<path d="M3.5 7.4L12 12l8.5-4.6M12 12v9.2"/>', None),
    "list":          ('<path d="M9 6.5h11M9 12h11M9 17.5h11"/>'
                      '<g fill="{c}" stroke="none"><circle cx="4.8" cy="6.5" r="1.3"/>'
                      '<circle cx="4.8" cy="12" r="1.3"/><circle cx="4.8" cy="17.5" r="1.3"/></g>',
                      None),
    "stack":         ('<path d="M12 3.5l8.5 4.5-8.5 4.5L3.5 8z"/>'
                      '<path d="M3.5 12l8.5 4.5 8.5-4.5M3.5 16l8.5 4.5 8.5-4.5"/>', None),

    # --- действия ----------------------------------------------------------
    "play":          ('<path d="M7.5 5.2v13.6a.8.8 0 0 0 1.2.7l10.7-6.8a.8.8 0 0 0 0-1.4'
                      'L8.7 4.5a.8.8 0 0 0-1.2.7z" fill="{c}"/>', 1.2),
    "stop":          ('<rect x="6" y="6" width="12" height="12" rx="2.2" fill="{c}"/>', 1.2),
    "record":        ('<circle cx="12" cy="12" r="8.5"/>'
                      '<circle cx="12" cy="12" r="4.6" fill="{c}" stroke="none"/>', None),
    "sparkles":      ('<path d="M10.5 3.5l1.7 4.8 4.8 1.7-4.8 1.7-1.7 4.8-1.7-4.8L4 10l4.8-1.7z"/>'
                      '<path d="M18 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>', None),
    "wand":          ('<path d="M4 20l10.5-10.5M13 8l3 3"/>'
                      '<path d="M18 3v3.5M16.25 4.75h3.5M20.5 9.5v2.4M19.3 10.7h2.4'
                      'M9 3.5v2.4M7.8 4.7h2.4"/>', None),
    "film":          ('<rect x="3" y="3.5" width="18" height="17" rx="2.2"/>'
                      '<path d="M7.5 3.5v17M16.5 3.5v17M3 8h4.5M3 12h18M3 16h4.5'
                      'M16.5 8H21M16.5 16H21"/>', None),

    # --- данные ------------------------------------------------------------
    "heap":          ('<path d="M2.5 19.5l6-8.5 3.8 4.6 2.6-3.3 6.6 7.2z"/>', None),
    "texture":       ('<rect x="3.5" y="3.5" width="17" height="17" rx="2.2"/>'
                      '<path d="M3.5 12h17M12 3.5v17"/>'
                      '<path d="M5.5 3.5H12V12H3.5V5.5a2 2 0 0 1 2-2z" fill="{c}" fill-opacity=".45" stroke="none"/>'
                      '<path d="M12 12h8.5v6.5a2 2 0 0 1-2 2H12z" fill="{c}" fill-opacity=".45" stroke="none"/>',
                      None),
    "depth":         ('<path d="M4 6h16M4 10h12M4 14h8M4 18h4"/>', 2.0),
    "mask":          ('<rect x="3.5" y="3.5" width="17" height="17" rx="2.2"/>'
                      '<path d="M3.5 18.5V5.5a2 2 0 0 1 2-2h13z" fill="{c}" fill-opacity=".5" stroke="none"/>',
                      None),
    "lidar":         ('<circle cx="5" cy="19" r="1.6" fill="{c}" stroke="none"/>'
                      '<path d="M5 13a6 6 0 0 1 6 6M5 8a11 11 0 0 1 11 11M5 3a16 16 0 0 1 16 16"/>',
                      None),
    "braces":        ('<path d="M8.5 4H7.5a2 2 0 0 0-2 2v3.5L4 12l1.5 2.5V18a2 2 0 0 0 2 2h1'
                      'M15.5 4h1a2 2 0 0 1 2 2v3.5L20 12l-1.5 2.5V18a2 2 0 0 1-2 2h-1"/>', None),
    "bars":          ('<path d="M5.5 20v-5M10 20V9M14.5 20v-7.5M19 20V5"/>', 2.4),
    "points":        ('<g fill="{c}" stroke="none">'
                      '<circle cx="5.5" cy="9" r="1.4"/><circle cx="9.5" cy="5" r="1.4"/>'
                      '<circle cx="12" cy="11.5" r="1.4"/><circle cx="16.5" cy="7" r="1.4"/>'
                      '<circle cx="19" cy="12.5" r="1.4"/><circle cx="7" cy="15" r="1.4"/>'
                      '<circle cx="13" cy="17" r="1.4"/><circle cx="17.5" cy="18" r="1.4"/>'
                      '<circle cx="9.5" cy="20" r="1.4"/></g>', None),
}


def names() -> list[str]:
    return sorted(_ICONS)


def svg_markup(name: str, color: str) -> str:
    body, stroke = _ICONS.get(name, _ICONS["dot"])
    sw = _STROKE if stroke is None else stroke
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{sw}" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f'{body.replace("{c}", color)}</svg>'
    )


_CACHE_DIR = os.path.join(tempfile.gettempdir(), "iqoko_ui_icons")


def svg_file(name: str, color: str, size: int = 16) -> str:
    """Записать иконку в .svg (кэш во временном каталоге) и вернуть путь.
    Нужна там, где QSS хочет `image: url(...)`."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe = color.lstrip("#").replace(",", "_").replace(" ", "")
    path = os.path.join(_CACHE_DIR, f"{name}_{safe}_{size}.svg")
    markup = svg_markup(name, color)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if fh.read() == markup:
                return path
    except OSError:
        pass
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(markup)
    return path


@lru_cache(maxsize=512)
def pixmap(name: str, color: str, size: int = 18, dpr: float = 2.0):
    """Иконка как QPixmap с учётом плотности пикселей."""
    from PyQt6.QtCore import QByteArray, QRectF, Qt
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtSvg import QSvgRenderer

    px = max(1, int(round(size * dpr)))
    pm = QPixmap(px, px)
    pm.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg_markup(name, color).encode("utf-8")))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(p, QRectF(0, 0, px, px))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str | None = None, size: int = 18,
         active_color: str | None = None,
         disabled_color: str | None = None):
    """
    QIcon с тремя состояниями: обычное, выключенное и «включено» (для
    checkable-кнопок — цветом акцента).
    """
    from PyQt6.QtGui import QIcon
    from src.ui.ui_theme import COLOR_ACCENT, COLOR_TEXT, COLOR_TEXT_DIM

    color = color or COLOR_TEXT
    active_color = active_color or COLOR_ACCENT
    disabled_color = disabled_color or COLOR_TEXT_DIM
    ic = QIcon()
    ic.addPixmap(pixmap(name, color, size), QIcon.Mode.Normal, QIcon.State.Off)
    ic.addPixmap(pixmap(name, color, size), QIcon.Mode.Active, QIcon.State.Off)
    ic.addPixmap(pixmap(name, active_color, size), QIcon.Mode.Normal, QIcon.State.On)
    ic.addPixmap(pixmap(name, active_color, size), QIcon.Mode.Active, QIcon.State.On)
    ic.addPixmap(pixmap(name, disabled_color, size), QIcon.Mode.Disabled, QIcon.State.Off)
    ic.addPixmap(pixmap(name, disabled_color, size), QIcon.Mode.Disabled, QIcon.State.On)
    return ic
