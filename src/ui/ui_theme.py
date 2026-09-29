# ui_theme.py
# ---------------------------------------------------------------------------
# Тема интерфейса в духе macOS (dark appearance).
# ---------------------------------------------------------------------------
# Язык оформления:
#   * Поверхности — тёмные «материалы» Apple: окно #1C1C1E, контролы #2C2C2E,
#     приподнятые элементы #3A3A3C; разделители — тонкие и приглушённые.
#   * Текст — три уровня яркости (label / secondaryLabel / tertiaryLabel).
#   * Один системный акцент — systemBlue #0A84FF; зелёный / оранжевый /
#     красный — только для статусов.
#   * Скругления: 14 px у плавающих панелей, 10 px у карточек, 7 px у
#     кнопок и полей.
#   * Шрифт — SF Pro, на Windows — Segoe UI Variable (ближайший по рисунку).
#
# Имена констант сохранены со старой темы: их импортируют диалоги и
# виджеты, и смена значений перекрашивает их без правок.
# ---------------------------------------------------------------------------

from __future__ import annotations

# -- Палитра -----------------------------------------------------------------
COLOR_BG                = "#1C1C1E"   # фон окна / диалога
COLOR_SURFACE           = "#2C2C2E"   # поля, кнопки, карточки
COLOR_SURFACE_ELEVATED  = "#3A3A3C"   # hover, выделенные сегменты
COLOR_HAIRLINE          = "#38383A"   # разделители
COLOR_HAIRLINE_HOVER    = "#4A4A4E"

COLOR_TEXT              = "#F5F5F7"   # label
COLOR_TEXT_MUTED        = "#98989F"   # secondaryLabel
COLOR_TEXT_DIM          = "#636366"   # tertiaryLabel

COLOR_ACCENT            = "#0A84FF"   # systemBlue (dark)
COLOR_ACCENT_SOFT       = "#409CFF"   # hover / светлее
COLOR_ACCENT_GLOW       = "rgba(10, 132, 255, 40)"

COLOR_SUCCESS           = "#30D158"
COLOR_WARN              = "#FF9F0A"
COLOR_DANGER            = "#FF453A"
COLOR_PURPLE            = "#BF5AF2"
COLOR_TEAL              = "#64D2FF"
COLOR_YELLOW            = "#FFD60A"

#: Заливка плавающих панелей поверх 3D-вида (RGBA). Почти непрозрачная:
#: сцена под ней пёстрая, и сквозь сильную прозрачность текст не читается.
MATERIAL_RGBA           = (30, 30, 32, 238)
MATERIAL_BORDER_RGBA    = (255, 255, 255, 22)

RADIUS_PANEL   = 8
RADIUS_CARD    = 8
RADIUS_CONTROL = 7

FONT_STACK = ("'SF Pro Text', 'Segoe UI Variable Text', 'Segoe UI Variable', "
              "'Segoe UI', 'Inter', system-ui, sans-serif")
FONT_DISPLAY = ("'SF Pro Display', 'Segoe UI Variable Display', "
                "'Segoe UI Variable', 'Segoe UI', sans-serif")
FONT_MONO  = "'SF Mono', 'Cascadia Mono', 'JetBrains Mono', Consolas, monospace"


def rgba(hex_color: str, alpha: int) -> str:
    """'#RRGGBB' + альфа 0..255 -> 'rgba(r, g, b, a)' для QSS."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {int(alpha)})"


def _icon_url(name: str, color: str, size: int = 16) -> str:
    """Путь к SVG-иконке для QSS `image:` (лениво, без QApplication)."""
    try:
        from src.ui.icons import svg_file
        return svg_file(name, color, size).replace("\\", "/")
    except Exception:
        return ""


def build_qss(root_bg: str | None = COLOR_BG) -> str:
    """
    Полная таблица стилей. `root_bg=None` — прозрачный корень: так её
    получают плавающие панели, которые рисуют свой материал сами.
    """
    root_fill = root_bg if root_bg else "transparent"
    chevron = _icon_url("chevron_down", COLOR_TEXT_MUTED, 12)
    chevron_up = _icon_url("chevron_up", COLOR_TEXT_MUTED, 10)
    chevron_dn = _icon_url("chevron_down", COLOR_TEXT_MUTED, 10)
    check = _icon_url("check_bold", "#FFFFFF", 12)
    arrow_css = f"image: url('{chevron}');" if chevron else ""
    up_css = f"image: url('{chevron_up}');" if chevron_up else ""
    dn_css = f"image: url('{chevron_dn}');" if chevron_dn else ""
    check_css = f"image: url('{check}');" if check else "image: none;"

    return f"""
/* ===== Корень ===== */
QWidget {{
    background-color: {root_fill};
    color: {COLOR_TEXT};
    font-family: {FONT_STACK};
    font-size: 13px;
    border: none;
    outline: 0;
    selection-background-color: {COLOR_ACCENT};
    selection-color: #FFFFFF;
}}
QDialog, QMainWindow, QMessageBox {{
    background-color: {COLOR_BG};
}}

QToolTip {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE_HOVER};
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
}}

/* ===== Типографика ===== */
QLabel {{ background: transparent; }}
QLabel[role="eyebrow"] {{
    color: {COLOR_TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
}}
QLabel[role="title"] {{
    font-family: {FONT_DISPLAY};
    color: {COLOR_TEXT};
    font-size: 20px;
    font-weight: 700;
}}
QLabel[role="headline"] {{
    color: {COLOR_TEXT};
    font-size: 13px;
    font-weight: 600;
}}
QLabel[role="metric"] {{
    font-family: {FONT_DISPLAY};
    color: {COLOR_TEXT};
    font-size: 28px;
    font-weight: 600;
}}
QLabel[role="metric-unit"] {{
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
}}
QLabel[role="muted"] {{
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
}}
QLabel[role="caption"] {{
    color: {COLOR_TEXT_DIM};
    font-size: 11px;
}}
QLabel[role="mono"] {{
    font-family: {FONT_MONO};
    color: {COLOR_TEXT};
    font-size: 12px;
}}

/* ===== Чипы статуса ===== */
QLabel[role="chip-live"] {{
    color: {COLOR_SUCCESS};
    background-color: {rgba(COLOR_SUCCESS, 36)};
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel[role="chip-idle"] {{
    color: {COLOR_TEXT_MUTED};
    background-color: {rgba("#FFFFFF", 18)};
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}}
QLabel[role="chip-err"] {{
    color: {COLOR_DANGER};
    background-color: {rgba(COLOR_DANGER, 36)};
    border-radius: 9px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}}

/* ===== Карточки (inset grouped) ===== */
QFrame#Surface, QWidget#Surface {{
    background-color: {rgba("#FFFFFF", 12)};
    border-radius: {RADIUS_CARD}px;
}}
QFrame#SurfaceElevated, QWidget#SurfaceElevated {{
    background-color: {COLOR_SURFACE};
    border-radius: {RADIUS_CARD}px;
}}
QFrame[role="hairline"] {{
    background-color: {rgba("#FFFFFF", 20)};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}

/* ===== GroupBox — секция в стиле «Системных настроек» ===== */
QGroupBox {{
    background-color: {rgba("#FFFFFF", 12)};
    border: none;
    border-radius: {RADIUS_CARD}px;
    margin-top: 26px;
    padding: 12px 12px 10px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 4px;
    top: 4px;
    padding: 0;
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}}

/* ===== Кнопки ===== */
QPushButton {{
    background-color: {rgba("#FFFFFF", 26)};
    color: {COLOR_TEXT};
    border: none;
    border-radius: {RADIUS_CONTROL}px;
    padding: 6px 14px;
    min-height: 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{ background-color: {rgba("#FFFFFF", 38)}; }}
QPushButton:pressed {{ background-color: {rgba("#FFFFFF", 18)}; }}
QPushButton:disabled {{
    color: {COLOR_TEXT_DIM};
    background-color: {rgba("#FFFFFF", 12)};
}}
QPushButton:checked {{
    background-color: {rgba(COLOR_ACCENT, 60)};
    color: {COLOR_ACCENT_SOFT};
}}
QPushButton:default {{
    background-color: {COLOR_ACCENT};
    color: #FFFFFF;
}}

QPushButton[variant="primary"] {{
    background-color: {COLOR_ACCENT};
    color: #FFFFFF;
    font-weight: 600;
    padding: 8px 16px;
}}
QPushButton[variant="primary"]:hover {{ background-color: {COLOR_ACCENT_SOFT}; }}
QPushButton[variant="primary"]:pressed {{ background-color: #0060DF; }}
QPushButton[variant="primary"]:disabled {{
    background-color: {rgba("#FFFFFF", 16)};
    color: {COLOR_TEXT_DIM};
}}

QPushButton[variant="danger"] {{
    color: {COLOR_DANGER};
}}
QPushButton[variant="danger"]:hover {{
    background-color: {rgba(COLOR_DANGER, 40)};
}}

QPushButton[variant="ghost"] {{
    background-color: transparent;
    color: {COLOR_TEXT_MUTED};
    padding: 6px 10px;
}}
QPushButton[variant="ghost"]:hover {{
    color: {COLOR_TEXT};
    background-color: {rgba("#FFFFFF", 18)};
}}
QPushButton[variant="link"] {{
    background-color: transparent;
    color: {COLOR_ACCENT_SOFT};
    padding: 4px 6px;
}}
QPushButton[variant="link"]:hover {{ color: #6CB4FF; }}

QPushButton[variant="icon"] {{
    padding: 0;
    min-width: 28px; max-width: 28px;
    min-height: 28px; max-height: 28px;
    border-radius: 7px;
    background-color: transparent;
}}
QPushButton[variant="icon"]:hover {{ background-color: {rgba("#FFFFFF", 26)}; }}

QToolButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: none;
    border-radius: 6px;
    padding: 4px 8px;
}}
QToolButton:hover {{ background-color: {rgba("#FFFFFF", 26)}; }}
QToolButton:checked {{
    background-color: {rgba(COLOR_ACCENT, 60)};
    color: {COLOR_ACCENT_SOFT};
}}
QToolButton::menu-indicator {{ image: none; width: 0; }}

/* ===== Поля ввода ===== */
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit,
QDateTimeEdit, QTimeEdit {{
    background-color: {rgba("#FFFFFF", 18)};
    color: {COLOR_TEXT};
    border: 1px solid {rgba("#FFFFFF", 14)};
    border-radius: {RADIUS_CONTROL}px;
    padding: 5px 9px;
    min-height: 18px;
    selection-background-color: {COLOR_ACCENT};
    selection-color: #FFFFFF;
}}
QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    background-color: {rgba("#FFFFFF", 24)};
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {COLOR_ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled,
QSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
    color: {COLOR_TEXT_DIM};
    background-color: {rgba("#FFFFFF", 8)};
}}

QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    background: transparent;
    border: none;
    width: 16px;
    margin: 2px 3px 0 0;
}}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    background: transparent;
    border: none;
    width: 16px;
    margin: 0 3px 2px 0;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background: {rgba("#FFFFFF", 26)};
    border-radius: 3px;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    {up_css} width: 9px; height: 9px;
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    {dn_css} width: 9px; height: 9px;
}}

QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    border: none;
    width: 24px;
    background: transparent;
}}
QComboBox::down-arrow {{
    {arrow_css}
    width: 11px; height: 11px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE_HOVER};
    border-radius: 8px;
    padding: 4px;
    outline: 0;
    selection-background-color: {COLOR_ACCENT};
    selection-color: #FFFFFF;
}}
QComboBox QAbstractItemView::item {{
    min-height: 24px;
    padding: 2px 8px;
    border-radius: 5px;
}}

/* ===== CheckBox / Radio ===== */
QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {COLOR_TEXT};
    background: transparent;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {COLOR_HAIRLINE_HOVER};
    background-color: {rgba("#FFFFFF", 22)};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {COLOR_TEXT_MUTED};
}}
QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border: 1px solid {COLOR_ACCENT};
    {check_css}
}}
QRadioButton::indicator:checked {{
    width: 6px;
    height: 6px;
    background-color: #FFFFFF;
    border: 5px solid {COLOR_ACCENT};
}}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    border-color: {COLOR_HAIRLINE};
    background-color: {rgba("#FFFFFF", 8)};
}}
QCheckBox::indicator:checked:disabled {{
    background-color: {COLOR_TEXT_DIM};
    border-color: {COLOR_TEXT_DIM};
}}
QRadioButton::indicator:checked:disabled {{
    background-color: {COLOR_TEXT_MUTED};
    border-color: {COLOR_TEXT_DIM};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {COLOR_TEXT_DIM}; }}

/* ===== Слайдеры ===== */
QSlider {{ background: transparent; min-height: 20px; }}
QSlider::groove:horizontal {{
    background-color: {rgba("#FFFFFF", 36)};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background-color: {COLOR_ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: #FFFFFF;
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: none;
}}
QSlider::handle:horizontal:hover {{ background-color: #E8E8ED; }}
QSlider::sub-page:horizontal:disabled {{ background: {COLOR_TEXT_DIM}; }}
QSlider::handle:horizontal:disabled {{ background-color: {COLOR_TEXT_MUTED}; }}

/* ===== ProgressBar ===== */
QProgressBar {{
    background-color: {rgba("#FFFFFF", 26)};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 3px;
}}

/* ===== Скроллбары — тонкие, как оверлейные в macOS ===== */
QScrollArea {{ background: transparent; }}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px 2px 2px 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0 2px 2px 2px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {rgba("#FFFFFF", 50)};
    border-radius: 3px;
    min-height: 28px;
    min-width: 28px;
    margin: 2px;
}}
QScrollBar::handle:hover {{ background-color: {rgba("#FFFFFF", 90)}; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
    height: 0;
    width: 0;
}}

/* ===== Вкладки — как сегментированный контрол ===== */
QTabWidget::pane {{
    background-color: transparent;
    border: none;
    top: 6px;
}}
QTabBar {{
    qproperty-drawBase: 0;
    background: transparent;
}}
QTabBar::tab {{
    background: {rgba("#767680", 60)};
    color: {COLOR_TEXT};
    padding: 5px 14px;
    margin: 0;
    border: none;
    font-size: 12px;
    font-weight: 500;
}}
QTabBar::tab:first {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QTabBar::tab:last  {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
QTabBar::tab:hover {{ background: {rgba("#767680", 90)}; }}
QTabBar::tab:selected {{
    background: #636366;
    color: #FFFFFF;
}}

/* ===== Списки / деревья / таблицы ===== */
QListWidget, QTreeWidget, QListView, QTreeView, QTableView, QTableWidget {{
    background-color: transparent;
    border: none;
    padding: 0;
    outline: 0;
    alternate-background-color: {rgba("#FFFFFF", 6)};
}}
QListWidget::item, QTreeWidget::item, QListView::item, QTreeView::item {{
    padding: 6px 8px;
    border-radius: 6px;
    color: {COLOR_TEXT};
}}
QListWidget::item:hover, QTreeWidget::item:hover,
QListView::item:hover, QTreeView::item:hover {{
    background-color: {rgba("#FFFFFF", 14)};
}}
QListWidget::item:selected, QTreeWidget::item:selected,
QListView::item:selected, QTreeView::item:selected {{
    background-color: {rgba(COLOR_ACCENT, 80)};
    color: #FFFFFF;
}}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{
    background: transparent;
    color: {COLOR_TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {COLOR_HAIRLINE};
    padding: 4px 6px 6px 6px;
    font-size: 11px;
    font-weight: 600;
}}

/* ===== Меню ===== */
QMenu {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE_HOVER};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{
    padding: 5px 22px 5px 10px;
    border-radius: 5px;
    background: transparent;
}}
QMenu::item:selected {{ background-color: {COLOR_ACCENT}; color: #FFFFFF; }}
QMenu::item:disabled {{ color: {COLOR_TEXT_DIM}; }}
QMenu::separator {{
    height: 1px;
    background: {COLOR_HAIRLINE_HOVER};
    margin: 5px 8px;
}}
QMenu::icon {{ padding-left: 6px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 4px; }}
QMenu::indicator:checked {{ {check_css} }}

/* ===== Превью карты глубины ===== */
QFrame#DepthMapFrame {{
    background-color: #000000;
    border: none;
    border-radius: 10px;
}}
QLabel#DepthMapCanvas {{
    background-color: #000000;
    border-radius: 8px;
}}

/* ===== Строка статуса ===== */
QWidget#StatusBar {{
    background-color: transparent;
    border-top: 1px solid {COLOR_HAIRLINE};
}}
QLabel#StatusText {{
    color: {COLOR_TEXT_MUTED};
    font-family: {FONT_MONO};
    font-size: 11px;
    padding: 6px 12px;
}}

QWidget#Overlay, QFrame#Overlay {{
    background-color: rgba({MATERIAL_RGBA[0]}, {MATERIAL_RGBA[1]}, {MATERIAL_RGBA[2]}, {MATERIAL_RGBA[3]});
    border: 1px solid {rgba("#FFFFFF", 22)};
    border-radius: {RADIUS_PANEL}px;
}}
"""


#: Совместимость: старый код брал готовую строку.
QSS = build_qss()


def apply_theme(widget) -> None:
    """Тема для обычных окон и диалогов (непрозрачный фон)."""
    widget.setStyleSheet(build_qss())


def apply_hud_theme(widget) -> None:
    """Тема для плавающих панелей поверх 3D-вида (прозрачный корень)."""
    widget.setStyleSheet(build_qss(root_bg=None))
