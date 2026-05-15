# ui_theme.py
# ---------------------------------------------------------------------------
# Digital Engineering 2026 — UI Theme for PyQt6
# Professional AI / Robotics tooling aesthetic.
# ---------------------------------------------------------------------------
# Visual language:
#   * Anthracite background           #101010
#   * Surfaces / elevated panels      #161616  / #181818
#   * Hairline dividers               #252525  (NO boxed frames, only lines + whitespace)
#   * Muted text                      #7A7A7A
#   * Primary text                    #E8E8E8
#   * Accent (system-wide, single)    #00FF88  (Vivid Mint)
#   * Warning                         #FF7A1A  (Cyber Orange, used VERY sparingly)
#   * Danger                          #FF3355
# Typography: Geist Sans (fallback: Inter, IBM Plex Sans, Segoe UI Variable).
# Radii: 8px for surfaces, 6px for controls, 4px for tags/chips.
# ---------------------------------------------------------------------------

from __future__ import annotations

# -- Palette -----------------------------------------------------------------
COLOR_BG                = "#101010"
COLOR_SURFACE           = "#161616"
COLOR_SURFACE_ELEVATED  = "#1B1B1B"
COLOR_HAIRLINE          = "#252525"
COLOR_HAIRLINE_HOVER    = "#3A3A3A"

COLOR_TEXT              = "#E8E8E8"
COLOR_TEXT_MUTED        = "#7A7A7A"
COLOR_TEXT_DIM          = "#555555"

COLOR_ACCENT            = "#00FF88"
COLOR_ACCENT_SOFT       = "#00CC6A"
COLOR_ACCENT_GLOW       = "rgba(0, 255, 136, 30)"

COLOR_WARN              = "#FF7A1A"
COLOR_DANGER            = "#FF3355"

# Font stack — Geist Sans first, with solid fallbacks so the file is portable.
FONT_STACK = "'Geist', 'Geist Sans', 'Inter', 'IBM Plex Sans', " \
             "'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif"
FONT_MONO  = "'Geist Mono', 'JetBrains Mono', 'IBM Plex Mono', Consolas, monospace"


# -- Master QSS --------------------------------------------------------------
QSS = f"""
/* ===== Root ===== */
QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: {FONT_STACK};
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.1px;
    border: none;
    outline: 0;
    selection-background-color: {COLOR_ACCENT};
    selection-color: {COLOR_BG};
}}

QToolTip {{
    background-color: {COLOR_SURFACE_ELEVATED};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
}}

/* ===== Surfaces ===== */
/* Anchor surfaces via objectName so we can get elevation without boxed look. */
QWidget#RightPanel {{
    background-color: {COLOR_BG};
    border-left: 1px solid {COLOR_HAIRLINE};
}}

QWidget#Surface {{
    background-color: {COLOR_SURFACE};
    border-radius: 8px;
}}

QWidget#SurfaceElevated {{
    background-color: {COLOR_SURFACE_ELEVATED};
    border-radius: 8px;
}}

/* Hairline horizontal rule used instead of GroupBox borders */
QFrame[role="hairline"] {{
    background-color: {COLOR_HAIRLINE};
    max-height: 1px;
    min-height: 1px;
    border: none;
    margin: 4px 0;
}}

/* ===== Typography helpers ===== */
QLabel[role="eyebrow"] {{
    color: {COLOR_TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    padding: 0 0 2px 0;
}}

QLabel[role="title"] {{
    color: {COLOR_TEXT};
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.2px;
}}

QLabel[role="metric"] {{
    color: {COLOR_TEXT};
    font-family: {FONT_MONO};
    font-size: 22px;
    font-weight: 500;
    letter-spacing: -0.5px;
}}

QLabel[role="metric-unit"] {{
    color: {COLOR_TEXT_MUTED};
    font-family: {FONT_MONO};
    font-size: 11px;
    font-weight: 400;
}}

QLabel[role="muted"] {{
    color: {COLOR_TEXT_MUTED};
    font-size: 11px;
}}

/* ===== Status chip (LIVE / IDLE / ERR) ===== */
QLabel[role="chip-live"] {{
    color: {COLOR_ACCENT};
    background-color: rgba(0, 255, 136, 18);
    border: 1px solid rgba(0, 255, 136, 60);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: {FONT_MONO};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel[role="chip-idle"] {{
    color: {COLOR_TEXT_MUTED};
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 4px;
    padding: 2px 8px;
    font-family: {FONT_MONO};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}}
QLabel[role="chip-err"] {{
    color: {COLOR_DANGER};
    background-color: rgba(255, 51, 85, 18);
    border: 1px solid rgba(255, 51, 85, 60);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: {FONT_MONO};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}}

/* ===== GroupBox ===== */
/* No visible frame: just the title as a small caps eyebrow and a hairline below. */
QGroupBox {{
    background: transparent;
    border: none;
    border-top: 1px solid {COLOR_HAIRLINE};
    margin-top: 22px;
    padding: 16px 0 4px 0;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    top: 2px;
    padding: 0;
    color: {COLOR_TEXT_MUTED};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 6px;
    padding: 8px 14px;
    min-height: 18px;
    font-weight: 500;
}}
QPushButton:hover {{
    border-color: {COLOR_HAIRLINE_HOVER};
    background-color: rgba(255, 255, 255, 4);
}}
QPushButton:pressed {{
    background-color: rgba(255, 255, 255, 8);
}}
QPushButton:disabled {{
    color: {COLOR_TEXT_DIM};
    border-color: #1F1F1F;
}}
QPushButton:checked {{
    border-color: {COLOR_ACCENT};
    color: {COLOR_ACCENT};
    background-color: rgba(0, 255, 136, 10);
}}

/* Primary (accent) — used ONCE per screen for the main action */
QPushButton[variant="primary"] {{
    background-color: {COLOR_ACCENT};
    color: {COLOR_BG};
    border: 1px solid {COLOR_ACCENT};
    font-weight: 700;
    padding: 10px 16px;
    letter-spacing: 0.3px;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {COLOR_ACCENT_SOFT};
    border-color: {COLOR_ACCENT_SOFT};
}}
QPushButton[variant="primary"]:pressed {{
    background-color: #00B85E;
}}
QPushButton[variant="primary"]:disabled {{
    background-color: #1F1F1F;
    border-color: {COLOR_HAIRLINE};
    color: {COLOR_TEXT_DIM};
}}

/* Danger */
QPushButton[variant="danger"] {{
    color: {COLOR_DANGER};
    border-color: rgba(255, 51, 85, 60);
}}
QPushButton[variant="danger"]:hover {{
    background-color: rgba(255, 51, 85, 14);
    border-color: {COLOR_DANGER};
}}

/* Ghost (text-only) */
QPushButton[variant="ghost"] {{
    border-color: transparent;
    color: {COLOR_TEXT_MUTED};
    padding: 6px 10px;
}}
QPushButton[variant="ghost"]:hover {{
    color: {COLOR_TEXT};
    background-color: rgba(255, 255, 255, 4);
}}

/* Icon-only square button */
QPushButton[variant="icon"] {{
    padding: 0;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    border-radius: 6px;
    font-size: 13px;
}}

/* ===== Inputs ===== */
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit, QTextEdit {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {COLOR_ACCENT};
    selection-color: {COLOR_BG};
}}
QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {COLOR_HAIRLINE_HOVER};
}}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border-color: {COLOR_ACCENT};
    background-color: {COLOR_SURFACE_ELEVATED};
}}
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
    color: {COLOR_TEXT_DIM};
    background-color: #141414;
}}

/* SpinBox steppers — flat, minimal */
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 18px;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    width: 8px; height: 8px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {COLOR_TEXT_MUTED};
}}
QDoubleSpinBox::up-arrow:hover, QSpinBox::up-arrow:hover {{
    border-bottom-color: {COLOR_ACCENT};
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    width: 8px; height: 8px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLOR_TEXT_MUTED};
}}
QDoubleSpinBox::down-arrow:hover, QSpinBox::down-arrow:hover {{
    border-top-color: {COLOR_ACCENT};
}}

/* ComboBox dropdown */
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    width: 9px; height: 9px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {COLOR_TEXT_MUTED};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLOR_SURFACE_ELEVATED};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 6px;
    padding: 4px;
    outline: 0;
    selection-background-color: rgba(0, 255, 136, 18);
    selection-color: {COLOR_ACCENT};
}}

/* ===== CheckBox / Radio ===== */
QCheckBox, QRadioButton {{
    spacing: 8px;
    color: {COLOR_TEXT};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLOR_HAIRLINE_HOVER};
    background-color: {COLOR_SURFACE};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {COLOR_ACCENT};
}}
QCheckBox::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
    image: none;
}}
QRadioButton::indicator:checked {{
    background-color: {COLOR_BG};
    border: 4px solid {COLOR_ACCENT};
}}

/* ===== Sliders ===== */
QSlider::groove:horizontal {{
    background-color: {COLOR_HAIRLINE};
    height: 2px;
    border-radius: 1px;
}}
QSlider::sub-page:horizontal {{
    background-color: {COLOR_ACCENT};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background-color: {COLOR_TEXT};
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 6px;
    border: 2px solid {COLOR_BG};
}}
QSlider::handle:horizontal:hover {{
    background-color: {COLOR_ACCENT};
}}

/* ===== ProgressBar ===== */
QProgressBar {{
    background-color: {COLOR_SURFACE};
    border: none;
    border-radius: 2px;
    max-height: 4px;
    min-height: 4px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 2px;
}}

/* ===== Scrollbars — slim, overlay-style ===== */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 4px 2px 4px 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0 4px 2px 4px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {COLOR_HAIRLINE_HOVER};
    border-radius: 3px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{
    background-color: {COLOR_TEXT_MUTED};
}}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
    height: 0;
    width: 0;
}}

/* ===== Tabs — underline style, no boxed tabs ===== */
QTabWidget::pane {{
    background-color: transparent;
    border: none;
    border-top: 1px solid {COLOR_HAIRLINE};
    top: -1px;
}}
QTabBar {{
    qproperty-drawBase: 0;
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    color: {COLOR_TEXT_MUTED};
    padding: 10px 4px;
    margin-right: 18px;
    border: none;
    border-bottom: 1px solid transparent;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}}
QTabBar::tab:hover {{
    color: {COLOR_TEXT};
}}
QTabBar::tab:selected {{
    color: {COLOR_TEXT};
    border-bottom: 1px solid {COLOR_ACCENT};
}}

/* ===== List / Tree ===== */
QListWidget, QTreeWidget {{
    background-color: transparent;
    border: none;
    padding: 0;
    outline: 0;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 8px 10px;
    border-radius: 6px;
    color: {COLOR_TEXT};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: rgba(255, 255, 255, 6);
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: rgba(0, 255, 136, 14);
    color: {COLOR_ACCENT};
}}

/* ===== Depth Map Preview frame ===== */
QFrame#DepthMapFrame {{
    background-color: #0A0A0A;
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 8px;
}}
QLabel#DepthMapCanvas {{
    background-color: #000000;
    border-radius: 6px;
}}

/* ===== Status bar ===== */
QWidget#StatusBar {{
    background-color: transparent;
    border-top: 1px solid {COLOR_HAIRLINE};
}}
QLabel#StatusText {{
    color: {COLOR_TEXT_MUTED};
    font-family: {FONT_MONO};
    font-size: 10px;
    letter-spacing: 0.5px;
    padding: 6px 12px;
}}

/* ===== Overlay widgets (floating over 3D scene) ===== */
QWidget#Overlay {{
    background-color: rgba(16, 16, 16, 210);
    border: 1px solid {COLOR_HAIRLINE};
    border-radius: 8px;
}}
"""


def apply_theme(widget) -> None:
    """Apply the Digital Engineering 2026 theme to a widget (and its children)."""
    widget.setStyleSheet(QSS)
