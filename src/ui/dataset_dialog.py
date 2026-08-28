# dataset_dialog.py
# ---------------------------------------------------------------------------
# Центральный модальный диалог настроек съёмки датасета.
#
# Раньше все эти опции жили в нижнем углу карточки телеметрии: выпадающий
# список типов, четыре спинбокса и три чекбокса шириной в 240 пикселей. Там
# не было места ни на подписи, ни на объяснения, ни на превью, а любая новая
# настройка вытесняла старую.
#
# Здесь то же самое разложено по разделам с человеческими описаниями, а к
# глубине и сегментации добавлены их собственные настройки и живое превью.
# Диалог НИЧЕГО не рендерит сам: он только редактирует dict конфигурации
# (см. dataset_config) и возвращает его вызывающему.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QWidget, QFrame, QLabel, QPushButton, QCheckBox, QRadioButton,
    QButtonGroup, QSpinBox, QDoubleSpinBox, QLineEdit, QVBoxLayout,
    QHBoxLayout, QGridLayout, QScrollArea, QSizePolicy, QFileDialog,
    QColorDialog, QGraphicsDropShadowEffect, QStackedWidget,
)

from src.ui.ui_theme import (
    COLOR_ACCENT, COLOR_HAIRLINE, COLOR_TEXT, COLOR_TEXT_MUTED,
    COLOR_TEXT_DIM, COLOR_WARN, FONT_MONO, apply_theme,
)
from src.ui import dataset_config as dscfg
from src.ui.depth_preview import depth_to_qimage, gradient_strip_qimage


# ---------------------------------------------------------------------------
# Мелкие фабрики виджетов в стиле HUD
# ---------------------------------------------------------------------------
_FIELD_CSS = (
    "  background: rgba(255,255,255,4);"
    f"  color: {COLOR_TEXT};"
    f"  border: 1px solid {COLOR_HAIRLINE};"
    "  border-radius: 4px;"
    "  padding: 2px 6px;"
    f"  font-family: {FONT_MONO};"
    "  font-size: 11px;"
)

# Qt считает width/height индикатора по области содержимого, без рамки.
# Поэтому размер прописан отдельно для каждого состояния так, чтобы внешний
# габарит везде был 14x14: иначе выбранный вариант «раздувается» и круг
# радиокнопки превращается в скруглённый квадрат.
_CHECK_CSS = (
    f"QCheckBox {{ color: {COLOR_TEXT}; font-size: 12px;"
    f" background: transparent; }}"
    "QCheckBox::indicator:unchecked {"
    "  width: 12px; height: 12px;"
    f"  border: 1px solid {COLOR_HAIRLINE}; border-radius: 3px;"
    "  background: rgba(255,255,255,4); }"
    "QCheckBox::indicator:checked {"
    "  width: 12px; height: 12px;"
    f"  border: 1px solid {COLOR_ACCENT}; border-radius: 3px;"
    f"  background: {COLOR_ACCENT}; }}"
    f"QCheckBox:disabled {{ color: {COLOR_TEXT_DIM}; }}"
    "QCheckBox::indicator:unchecked:disabled {"
    f"  border-color: {COLOR_HAIRLINE}; background: #101010; }}"
    "QCheckBox::indicator:checked:disabled {"
    f"  border-color: {COLOR_TEXT_DIM}; background: {COLOR_TEXT_DIM}; }}"
)

_RADIO_CSS = (
    f"QRadioButton {{ color: {COLOR_TEXT}; font-size: 12px;"
    f" background: transparent; }}"
    "QRadioButton::indicator:unchecked {"
    "  width: 12px; height: 12px;"
    f"  border: 1px solid {COLOR_HAIRLINE}; border-radius: 7px;"
    "  background: rgba(255,255,255,4); }"
    "QRadioButton::indicator:checked {"
    "  width: 6px; height: 6px;"
    f"  border: 4px solid {COLOR_ACCENT}; border-radius: 7px;"
    "  background: #101010; }"
    f"QRadioButton:disabled {{ color: {COLOR_TEXT_DIM}; }}"
    "QRadioButton::indicator:unchecked:disabled {"
    f"  border-color: {COLOR_HAIRLINE}; background: #101010; }}"
    "QRadioButton::indicator:checked:disabled {"
    f"  border-color: {COLOR_TEXT_DIM}; background: #101010; }}"
)


def _label(text, *, size=12, color=COLOR_TEXT, mono=False, bold=False,
           wrap=False):
    lbl = QLabel(text)
    lbl.setWordWrap(wrap)
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size}px; background: transparent;"
        + (f" font-family: {FONT_MONO};" if mono else "")
        + (" font-weight: 600;" if bold else "")
    )
    return lbl


def _hint(text):
    """Пояснение под опцией — то, ради чего затевался этот диалог."""
    return _label(text, size=11, color=COLOR_TEXT_MUTED, wrap=True)


def _spin(rng, value, step, decimals=2, suffix="", width=88):
    sp = QDoubleSpinBox()
    sp.setRange(*rng)
    sp.setSingleStep(step)
    sp.setDecimals(decimals)
    sp.setValue(float(value))
    if suffix:
        sp.setSuffix(suffix)
    sp.setFixedHeight(24)
    sp.setFixedWidth(width)
    sp.setStyleSheet(
        "QDoubleSpinBox {" + _FIELD_CSS + "}"
        "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0; }"
        f"QDoubleSpinBox:disabled {{ color: {COLOR_TEXT_DIM}; }}"
    )
    return sp


def _int_spin(rng, value, suffix="", width=88):
    sp = QSpinBox()
    sp.setRange(*rng)
    sp.setValue(int(value))
    if suffix:
        sp.setSuffix(suffix)
    sp.setFixedHeight(24)
    sp.setFixedWidth(width)
    sp.setStyleSheet(
        "QSpinBox {" + _FIELD_CSS + "}"
        "QSpinBox::up-button, QSpinBox::down-button { width: 0; }"
        f"QSpinBox:disabled {{ color: {COLOR_TEXT_DIM}; }}"
    )
    return sp


def _check(text, checked=False, tip=""):
    box = QCheckBox(text)
    box.setChecked(bool(checked))
    box.setCursor(Qt.CursorShape.PointingHandCursor)
    box.setStyleSheet(_CHECK_CSS)
    if tip:
        box.setToolTip(tip)
    return box


def _radio(text, checked=False):
    btn = QRadioButton(text)
    btn.setChecked(bool(checked))
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(_RADIO_CSS)
    return btn


def _button(text, *, accent=False, width=None):
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedHeight(28)
    if width:
        btn.setFixedWidth(width)
    if accent:
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(0, 255, 136, 30);"
            f"  color: {COLOR_TEXT};"
            f"  border: 1px solid {COLOR_ACCENT};"
            "  border-radius: 5px; padding: 3px 16px;"
            "  font-size: 12px; font-weight: 600; letter-spacing: 0.4px;"
            "}"
            "QPushButton:hover { background-color: rgba(0, 255, 136, 55); }"
            "QPushButton:pressed { background-color: rgba(0, 255, 136, 90); }"
            "QPushButton:disabled {"
            "  background: rgba(255,255,255,4);"
            f"  color: {COLOR_TEXT_DIM}; border: 1px solid {COLOR_HAIRLINE}; }}"
        )
    else:
        btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(255,255,255,6);"
            f"  color: {COLOR_TEXT};"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 5px; padding: 3px 14px; font-size: 12px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,14); }"
            f"QPushButton:disabled {{ color: {COLOR_TEXT_DIM}; }}"
        )
    return btn


def _hairline():
    line = QFrame()
    line.setFixedHeight(1)
    line.setStyleSheet(f"background-color: {COLOR_HAIRLINE}; border: none;")
    return line


class _Section(QFrame):
    """Блок настроек: заголовок-надглазник, подпись и вертикальный стек."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        self.setStyleSheet(
            "QFrame {"
            "  background: rgba(255,255,255,3);"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 8px;"
            "}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 14)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        dot = _label("●", size=9, color=COLOR_ACCENT)
        head.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addWidget(_label(title.upper(), size=11, color=COLOR_TEXT,
                              bold=True), 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        lay.addLayout(head)

        if subtitle:
            lay.addWidget(_hint(subtitle))
        lay.addWidget(_hairline())

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        lay.addLayout(self.body)

    def add(self, widget_or_layout):
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout)
        else:
            self.body.addLayout(widget_or_layout)

    def add_option(self, control, hint_text):
        """Контрол + пояснение под ним, с отступом под индикатор."""
        wrap = QVBoxLayout()
        wrap.setSpacing(2)
        wrap.addWidget(control)
        hint = _hint(hint_text)
        hint.setContentsMargins(21, 0, 0, 0)
        wrap.addWidget(hint)
        self.body.addLayout(wrap)
        return control

    def add_field(self, title, hint_text, *widgets):
        """Строка «подпись — поля» с пояснением под ней."""
        wrap = QVBoxLayout()
        wrap.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(_label(title, size=12), 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        # Диапазон «от … до» рядом с подписью требует под 450 px минимума и
        # в одиночку решает, сколько колонок влезет в окно. Такие поля уходят
        # на свою строку под подписью.
        if len(widgets) >= 3:
            wrap.addLayout(row)
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addStretch(1)
        for widget in widgets:
            row.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        wrap.addLayout(row)
        if hint_text:
            wrap.addWidget(_hint(hint_text))
        self.body.addLayout(wrap)


class _Columns(QWidget):
    """Раскладка разделов в N колонок, где N зависит от ширины окна.

    Раньше страница была жёстко трёхколоночной с минимумом 310 px на
    колонку: на развёрнутом во весь экран окне половина ширины пустовала,
    а всё содержимое всё равно приходилось прокручивать. Здесь число
    колонок пересчитывается на каждом изменении ширины, а разделы
    раскладываются жадно — очередной уходит в самую короткую колонку,
    поэтому низ страницы получается ровным и прокручивать почти нечего.
    """

    def __init__(self, min_col=340, max_col=560, max_cols=4, spacing=12):
        super().__init__()
        self.setStyleSheet("background: transparent;")
        self._items: list[QWidget] = []
        self._cols = 0
        self._cw = 0
        self._min_col = int(min_col)
        self._min_cache = None
        self._max_col = int(max_col)
        self._max_cols = int(max_cols)
        self._spacing = int(spacing)
        self._row = QHBoxLayout(self)
        # Справа оставлен зазор под полосу прокрутки страницы.
        self._row.setContentsMargins(0, 2, 10, 2)
        self._row.setSpacing(spacing)

    def add(self, widget):
        self._items.append(widget)
        self._min_cache = None
        self._cols = 0            # заставить перестроить на ближайшем показе
        return widget

    def _eff_min(self):
        """Колонка не может быть уже самого широкого поля в разделах."""
        if self._min_cache is None:
            widest = max([w.minimumSizeHint().width() for w in self._items]
                         or [0])
            self._min_cache = max(self._min_col, widest)
        return self._min_cache

    def _usable(self, width):
        margins = self._row.contentsMargins()
        return max(1, int(width) - margins.left() - margins.right())

    def _wanted(self, width):
        if not self._items:
            return 1
        free = self._usable(width) + self._spacing
        fit = free // (self._eff_min() + self._spacing)
        return max(1, min(self._max_cols, len(self._items), int(fit)))

    @staticmethod
    def _item_height(widget, width):
        # У разделов внутри есть переносимые подписи, поэтому высота зависит
        # от ширины колонки — sizeHint по текущей ширине тут врёт.
        if widget.hasHeightForWidth():
            return max(1, widget.heightForWidth(width))
        return max(1, widget.sizeHint().height())

    def col_width(self, width):
        cols = self._wanted(width)
        free = self._usable(width) - (cols - 1) * self._spacing
        return max(self._eff_min(), min(self._max_col, free // cols))

    def ideal_height(self, width):
        """Высота, при которой странице не понадобится прокрутка."""
        cols = self._wanted(width)
        cw = self.col_width(width)
        heights = [0] * cols
        for widget in self._items:
            idx = heights.index(min(heights))
            heights[idx] += self._item_height(widget, cw) + self._spacing
        top, _, bottom = (self._row.contentsMargins().top(), 0,
                          self._row.contentsMargins().bottom())
        return max(heights or [0]) + top + bottom

    def _relayout(self, cols):
        # Сначала вынимаем разделы из старых колонок, иначе deleteLater
        # унесёт их с собой.
        for widget in self._items:
            widget.setParent(None)
        while self._row.count():
            item = self._row.takeAt(0)
            holder = item.widget()
            if holder is not None:
                holder.deleteLater()

        cw = self.col_width(self.width())
        lays, heights = [], []
        for _ in range(cols):
            holder = QWidget()
            holder.setStyleSheet("background: transparent;")
            lay = QVBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(self._spacing)
            # Без потолка одинокий раздел растягивался бы на всю ширину
            # экрана, и строка подписи уезжала от своего поля на метр.
            holder.setFixedWidth(cw)
            lays.append(lay)
            heights.append(0)
            self._row.addWidget(holder, 1)
        self._row.addStretch(1)

        for widget in self._items:
            idx = heights.index(min(heights))
            lays[idx].addWidget(widget)
            widget.setVisible(True)   # setParent(None) выставил скрытие явно
            heights[idx] += self._item_height(widget, cw)
        # Колонку с растягивающимся разделом (превью глубины) распорка снизу
        # прижала бы к верху и оставила бы превью крошечным.
        for lay in lays:
            grows = any(
                lay.itemAt(i).widget() is not None
                and lay.itemAt(i).widget().sizePolicy().verticalPolicy()
                == QSizePolicy.Policy.Expanding
                for i in range(lay.count())
            )
            if not grows:
                lay.addStretch(1)
        self._cols = cols
        self._cw = cw

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cols = self._wanted(self.width())
        if cols != self._cols or self.col_width(self.width()) != self._cw:
            self._relayout(cols)


class _NavButton(QPushButton):
    """Пункт бокового списка страниц: заголовок, подпись и метка состояния."""

    def __init__(self, title, subtitle):
        super().__init__()
        self._title = title
        self._subtitle = subtitle
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "QPushButton {"
            "  text-align: left; padding: 6px 10px;"
            "  background: transparent;"
            "  border: 1px solid transparent; border-radius: 6px;"
            f"  color: {COLOR_TEXT_MUTED}; font-size: 12px;"
            "}"
            "QPushButton:hover { background: rgba(255,255,255,10); }"
            "QPushButton:checked {"
            "  background: rgba(0, 255, 136, 22);"
            f"  border: 1px solid {COLOR_ACCENT};"
            f"  color: {COLOR_TEXT}; font-weight: 600;"
            "}"
        )
        self._render()

    def set_subtitle(self, text):
        if text != self._subtitle:
            self._subtitle = text
            self._render()

    def _render(self):
        self.setText("\n".join([self._title, self._subtitle])
                     if self._subtitle else self._title)
        self.setToolTip(self._subtitle)


class _Swatch(QPushButton):
    """Квадратик цвета класса сегментации; клик открывает палитру."""

    colorPicked = pyqtSignal(tuple)

    def __init__(self, rgb):
        super().__init__()
        self._rgb = tuple(rgb)
        self.setFixedSize(QSize(26, 22))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._refresh()

    def rgb(self):
        return self._rgb

    def set_rgb(self, rgb):
        self._rgb = tuple(int(c) for c in rgb)
        self._refresh()

    def _refresh(self):
        r, g, b = self._rgb
        self.setToolTip(f"RGB {r}, {g}, {b} — нажмите, чтобы изменить")
        self.setStyleSheet(
            "QPushButton {"
            f"  background-color: rgb({r},{g},{b});"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 4px;"
            "}"
            f"QPushButton:hover {{ border: 1px solid {COLOR_ACCENT}; }}"
        )

    def _pick(self):
        r, g, b = self._rgb
        color = QColorDialog.getColor(
            QColor(r, g, b), self, "Цвет класса сегментации")
        if color.isValid():
            self.set_rgb((color.red(), color.green(), color.blue()))
            self.colorPicked.emit(self._rgb)


# ---------------------------------------------------------------------------
# Диалог
# ---------------------------------------------------------------------------
class DatasetSettingsDialog(QDialog):
    """Настройки съёмки датасета.

    После `exec()` смотрите `action`:
        "start"  — пользователь нажал «Начать съёмку»;
        "save"   — сохранил настройки и закрыл;
        None     — отменил (конфиг менять не нужно).
    Актуальный конфиг всегда в `config` (нормализованный).
    """

    # Потолок нужен только чтобы на 4K диалог не растянулся на два
    # метра текста; во всём остальном он занимает почти всё окно.
    CARD_MAX = (2200, 1500)
    CARD_MARGIN = (72, 56)

    def __init__(self, config, parent=None, panda_app=None):
        super().__init__(parent)
        self.config = dscfg.normalize(config)
        self.action = None
        self._panda_app = panda_app
        self._swatches: dict = {}
        # Сигналы контролов прилетают уже во время сборки (QButtonGroup
        # переключает радиокнопки), а обработчики трогают виджеты из ещё не
        # построенных разделов. Пока флаг не поднят — они ничего не делают.
        self._ready = False

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setModal(True)
        apply_theme(self)

        self._backdrop = QFrame(self)
        self._backdrop.setStyleSheet("background-color: rgba(0, 0, 0, 170);")
        self._backdrop.lower()

        self.card = QFrame(self)
        self.card.setObjectName("Overlay")
        self.card.setStyleSheet(
            "QFrame#Overlay {"
            "  background-color: rgba(16, 16, 16, 245);"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 12px;"
            "}"
        )
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(0, 0, 0, 220))
        self.card.setGraphicsEffect(shadow)

        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(18, 16, 18, 16)
        card_lay.setSpacing(12)
        card_lay.addLayout(self._build_header())
        card_lay.addWidget(_hairline())
        card_lay.addLayout(self._build_body(), 1)
        card_lay.addWidget(_hairline())
        card_lay.addLayout(self._build_footer())

        outer = QGridLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._backdrop, 0, 0)
        outer.addWidget(self.card, 0, 0, Qt.AlignmentFlag.AlignCenter)

        self._ready = True
        self._sync_enabled()

        # Живое превью глубины: тянем кадры из того же буфера, что кормит
        # оверлей в углу экрана, но раскрашиваем ПАРАМЕТРАМИ ДАТАСЕТА.
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._tick_preview)
        self._preview_timer.start(200)
        self._tick_preview()

    # ------------------------------------------------------------------
    # Шапка и подвал
    # ------------------------------------------------------------------
    def _build_header(self):
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(_label("●", size=10, color=COLOR_ACCENT), 0,
                      Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(_label("ДАТАСЕТ", size=10, color=COLOR_TEXT_MUTED,
                             bold=True), 0, Qt.AlignmentFlag.AlignVCenter)
        sep = QFrame()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet(f"background-color: {COLOR_HAIRLINE};")
        row.addWidget(sep, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(_label("Настройки съёмки", size=14, bold=True), 0,
                      Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)

        self.lbl_summary = _label("", size=11, color=COLOR_TEXT_MUTED,
                                  mono=True)
        row.addWidget(self.lbl_summary, 0, Qt.AlignmentFlag.AlignVCenter)

        btn_close = _button("✕", width=32)
        btn_close.setToolTip("Закрыть без сохранения (Esc)")
        btn_close.clicked.connect(self.reject)
        row.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def _build_footer(self):
        row = QHBoxLayout()
        row.setSpacing(8)

        btn_reset = _button("Сбросить")
        btn_reset.setToolTip("Вернуть все настройки к значениям по умолчанию")
        btn_reset.clicked.connect(self._on_reset)
        row.addWidget(btn_reset, 0)

        self.lbl_footer_hint = _label("", size=11, color=COLOR_TEXT_MUTED,
                                      wrap=True)
        row.addWidget(self.lbl_footer_hint, 1)

        btn_cancel = _button("Отмена")
        btn_cancel.clicked.connect(self.reject)
        row.addWidget(btn_cancel, 0)

        btn_save = _button("Сохранить")
        btn_save.setToolTip("Запомнить настройки и закрыть, ничего не снимая")
        btn_save.clicked.connect(lambda: self._finish("save"))
        row.addWidget(btn_save, 0)

        self.btn_start = _button("Начать съёмку", accent=True)
        self.btn_start.clicked.connect(lambda: self._finish("start"))
        row.addWidget(self.btn_start, 0)
        return row

    # ------------------------------------------------------------------
    # Содержимое
    # ------------------------------------------------------------------
    def _build_body(self):
        """Боковой список страниц + стек самих страниц.

        Всё содержимое раньше жило в одной прокрутке из трёх колонок: даже
        на весь экран приходилось листать, а лидар с его двумя десятками
        полей выталкивал остальное далеко вниз. Теперь разделы разложены по
        страницам, каждая страница сама раскладывается в столько колонок,
        сколько влезает по ширине, и почти всегда помещается целиком.
        """
        row = QHBoxLayout()
        row.setSpacing(14)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background: transparent;")

        self._nav_buttons = {}
        self.grp_nav = QButtonGroup(self)
        self.grp_nav.setExclusive(True)

        nav_holder = QWidget()
        nav_holder.setStyleSheet("background: transparent;")
        nav_holder.setFixedWidth(206)
        nav = QVBoxLayout(nav_holder)
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setSpacing(4)
        nav.addWidget(_label("РАЗДЕЛЫ", size=10, color=COLOR_TEXT_MUTED,
                             bold=True))
        nav.addSpacing(2)

        pages = [
            ("scope", "Съёмка", "объём, выходы, наполнение",
             [self._section_scope(), self._section_outputs(),
              self._section_volume()]),
            ("capture", "Камера и сцена", "поза, свет, усложнения",
             [self._section_camera(), self._section_lighting(),
              self._section_scene()]),
            ("depth", "Глубина", "превью и диапазон",
             [self._section_depth_preview(), self._section_depth_range()]),
            ("segmentation", "Сегментация", "палитра классов",
             [self._section_segmentation()]),
            ("lidar", "Лидар", "сенсор, развёртка, вывод",
             [self._section_lidar_sensor(), self._section_lidar_pattern(),
              self._section_lidar_output()]),
        ]

        for index, (key, title, subtitle, sections) in enumerate(pages):
            btn = _NavButton(title, subtitle)
            btn.clicked.connect(
                lambda _=False, i=index: self._go_page(i))
            if key == "depth":
                self._depth_page = index
            self.grp_nav.addButton(btn, index)
            self._nav_buttons[key] = btn
            nav.addWidget(btn)
            self.stack.addWidget(self._build_page(sections))

        nav.addStretch(1)
        self._nav_subtitles = {k: b._subtitle
                               for k, b in self._nav_buttons.items()}
        self.grp_nav.button(0).setChecked(True)

        row.addWidget(nav_holder, 0)
        divider = QFrame()
        divider.setFixedWidth(1)
        divider.setStyleSheet(f"background-color: {COLOR_HAIRLINE};")
        row.addWidget(divider, 0)
        row.addWidget(self.stack, 1)
        return row

    def _go_page(self, index):
        index = max(0, min(self.stack.count() - 1, int(index)))
        self.stack.setCurrentIndex(index)
        btn = self.grp_nav.button(index)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        # Превью глубины считается только на своей странице, поэтому при
        # переходе его надо разбудить сразу, не дожидаясь тика таймера.
        if index == getattr(self, "_depth_page", -1):
            QTimer.singleShot(0, self._tick_preview)

    def _build_page(self, sections):
        """Одна страница: прокрутка на случай низкого окна + авто-колонки."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: transparent; width: 8px; }"
            "QScrollBar::handle:vertical {"
            f"  background: {COLOR_HAIRLINE}; border-radius: 4px;"
            "  min-height: 40px; }"
            "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
        )
        columns = _Columns()
        for section in sections:
            columns.add(section)
        scroll.setWidget(columns)
        return scroll

    # -- Объём съёмки ---------------------------------------------------
    def _section_scope(self):
        sec = _Section(
            "Объём съёмки",
            "Сколько раз пересобрать сцену и куда сложить результат.",
        )
        self.spn_count = _int_spin((1, 100000), self.config["count"])
        self.spn_count.valueChanged.connect(self._update_summary)
        sec.add_field(
            "Наполнений",
            "Каждое наполнение — заново сгенерированный груз. С одного "
            "наполнения снимается столько кадров, сколько даёт раздел "
            "«Камера».",
            self.spn_count,
        )

        # wrap=True: строка длинная, а без переноса она задаёт минимальную
        # ширину всей колонки и ломает раскладку на узком окне.
        self.lbl_frames = _label("", size=11, color=COLOR_ACCENT, mono=True,
                                 wrap=True)
        sec.add(self.lbl_frames)

        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        self.edt_out = QLineEdit(self.config["output_dir"])
        self.edt_out.setFixedHeight(24)
        self.edt_out.setStyleSheet("QLineEdit {" + _FIELD_CSS + "}")
        btn_browse = _button("Обзор…", width=78)
        btn_browse.clicked.connect(self._on_browse)
        path_row.addWidget(self.edt_out, 1)
        path_row.addWidget(btn_browse, 0)

        wrap = QVBoxLayout()
        wrap.setSpacing(2)
        wrap.addWidget(_label("Каталог вывода", size=12))
        wrap.addLayout(path_row)
        wrap.addWidget(_hint(
            "Все файлы кадра (цвет, глубина, маска, json) складываются сюда "
            "рядом друг с другом и различаются суффиксом имени. Относительный "
            "путь считается от корня проекта."
        ))
        sec.add(wrap)
        return sec

    # -- Что сохранять --------------------------------------------------
    def _section_outputs(self):
        sec = _Section(
            "Что сохранять",
            "Раньше это решал «тип датасета». Теперь выходы независимы: "
            "можно снять маску без цветного кадра или глубину вместе с "
            "маской за один проход.",
        )
        outputs = self.config["outputs"]
        self.chk_out = {}

        specs = [
            ("color", "Цветной кадр",
             "Обычный рендер сцены с дисторсией и кропом — файл без "
             "суффикса."),
            ("depth", "Карта глубины",
             "Файл с суффиксом _depth. Настройки диапазона и палитры — "
             "в разделе «Глубина»."),
            ("segmentation", "Маска сегментации",
             "Файл с суффиксом _seg: плоские цвета классов без "
             "постобработки. Цвета — в разделе «Сегментация»."),
            ("lidar", "Облако точек (ply)",
             "Файл с суффиксом _lidar: съёмка виртуальным 3D-лидаром с той "
             "же позы, что и кадр. Развёртка и шум — в разделе «Лидар»."),
            ("json", "Метаданные (json)",
             "Поза камеры, объём, интринсики, параметры дисторсии, палитра "
             "классов и все опции этого кадра."),
        ]
        for key, title, hint in specs:
            box = _check(title, outputs.get(key, False))
            box.toggled.connect(self._on_outputs_changed)
            self.chk_out[key] = box
            sec.add_option(box, hint)

        self.lbl_out_warn = _label("", size=11, color=COLOR_WARN, wrap=True)
        sec.add(self.lbl_out_warn)
        return sec

    # -- Наполнение -----------------------------------------------------
    def _section_volume(self):
        sec = _Section(
            "Наполнение кузова",
            "Как выбирается объём груза от кадра к кадру.",
        )
        vol = self.config["volume"]
        self.grp_volume = QButtonGroup(self)
        self.rb_vol_ramp = _radio("Линейный рост",
                                  vol["mode"] == "ramp")
        self.rb_vol_random = _radio("Случайный объём",
                                    vol["mode"] == "random")
        for btn in (self.rb_vol_ramp, self.rb_vol_random):
            self.grp_volume.addButton(btn)
            btn.toggled.connect(self._sync_enabled)

        sec.add_option(self.rb_vol_ramp,
                       "Объём равномерно растёт от нуля до паспортного "
                       "максимума за все наполнения — предсказуемое покрытие "
                       "диапазона.")
        sec.add_option(self.rb_vol_random,
                       "Каждое наполнение — случайный объём от нуля до "
                       "потолка. Доли ниже позволяют подмешать крайние "
                       "случаи.")

        self.spn_full = _int_spin((0, 100), vol["full_pct"], suffix="%")
        self.spn_empty = _int_spin((0, 100), vol["empty_pct"], suffix="%")
        self.spn_full.valueChanged.connect(
            lambda v: self._clamp_pair(self.spn_full, self.spn_empty, v))
        self.spn_empty.valueChanged.connect(
            lambda v: self._clamp_pair(self.spn_empty, self.spn_full, v))
        sec.add_field(
            "Полный кузов",
            "Доля кадров с объёмом 95–100% потолка.",
            self.spn_full,
        )
        sec.add_field(
            "Пустой кузов",
            "Доля кадров совсем без груза — датасету нужны и такие примеры. "
            "Сумма долей не превышает 100%, остальное — равномерно "
            "случайный объём.",
            self.spn_empty,
        )
        self.spn_ceiling = _spin((0.1, 3.0), vol["ceiling_k"], 0.05,
                                 decimals=2, width=78)
        sec.add_field(
            "Потолок",
            "Множитель к паспортному max_volume. 1.35 означает, что "
            "допускается перегруз до 135% — это валидный кейс на практике.",
            self.spn_ceiling,
        )
        return sec

    # -- Камера ---------------------------------------------------------
    def _section_camera(self):
        sec = _Section(
            "Камера",
            "Что делать с позой камеры на каждом наполнении. Базовая поза — "
            "та, что выставлена в окне сейчас.",
        )
        cam = self.config["camera"]
        self.grp_camera = QButtonGroup(self)
        self.rb_cam_fixed = _radio("Только базовая поза",
                                   cam["mode"] == "fixed")
        self.rb_cam_variants = _radio("Набор отклонений",
                                      cam["mode"] == "variants")
        self.rb_cam_random = _radio("Случайная поза",
                                    cam["mode"] == "random")
        for btn in (self.rb_cam_fixed, self.rb_cam_variants,
                    self.rb_cam_random):
            self.grp_camera.addButton(btn)
            btn.toggled.connect(self._sync_enabled)

        sec.add_option(self.rb_cam_fixed,
                       "Один кадр с наполнения, камера не двигается.")
        sec.add_option(self.rb_cam_variants,
                       "Детерминированная сетка отклонений — по кадру на "
                       "каждое включённое ниже.")
        sec.add_option(self.rb_cam_random,
                       "Случайное отклонение в тех же рамках, заданное число "
                       "кадров с наполнения.")

        self.chk_var = {}
        specs = [
            ("originals", "Базовая поза",
             "По кадру на каждый включённый тип освещения."),
            ("angles", "Повороты ±",
             "Четыре кадра: ±угол по рысканью и ±угол по тангажу."),
            ("offsets", "Сдвиги ±",
             "Четыре кадра: ±смещение по горизонтали и по вертикали."),
            ("random_combined", "Случайная комбинация",
             "Один кадр со случайными поворотом и сдвигом сразу."),
        ]
        for key, title, hint in specs:
            box = _check(title, cam["variants"].get(key, True))
            box.toggled.connect(self._update_summary)
            self.chk_var[key] = box
            sec.add_option(box, hint)

        self.spn_samples = _int_spin((1, 64), cam["samples"])
        self.spn_samples.valueChanged.connect(self._update_summary)
        sec.add_field("Случайных кадров",
                      "Сколько случайных поз снять с одного наполнения.",
                      self.spn_samples)

        self.spn_angle = _spin((0.0, 90.0), cam["angle_deg"], 1.0,
                               decimals=1, suffix="°", width=78)
        self.spn_offset = _spin((0.0, 5.0), cam["offset_m"], 0.01,
                                decimals=3, suffix=" м", width=88)
        sec.add_field("Предел поворота",
                      "Максимальное отклонение по рысканью и тангажу.",
                      self.spn_angle)
        sec.add_field("Предел сдвига",
                      "Максимальное смещение камеры вбок и по высоте.",
                      self.spn_offset)
        return sec

    # -- Освещение ------------------------------------------------------
    def _section_lighting(self):
        sec = _Section(
            "Освещение",
            "Свет ставится перед каждым кадром и возвращается к исходному "
            "после съёмки.",
        )
        light = self.config["lighting"]
        self.grp_light = QButtonGroup(self)
        self.rb_light_cycle = _radio("Чередование типов",
                                     light["mode"] == "cycle")
        self.rb_light_overhead = _radio("Солнце в зените",
                                        light["mode"] == "overhead")
        self.rb_light_current = _radio("Как в окне",
                                       light["mode"] == "current")
        for btn in (self.rb_light_cycle, self.rb_light_overhead,
                    self.rb_light_current):
            self.grp_light.addButton(btn)
            btn.toggled.connect(self._sync_enabled)

        sec.add_option(self.rb_light_cycle,
                       "Кадры по очереди получают один из включённых ниже "
                       "типов света.")
        sec.add_option(self.rb_light_overhead,
                       "Солнце жёстко направлено сверху вниз: минимум теней, "
                       "ровная освещённость груза.")
        sec.add_option(self.rb_light_current,
                       "Время суток берётся из ползунка в окне и не меняется.")

        self.chk_light = {}
        specs = [
            ("day", "День",
             "Случайное время в интервале 10:00–16:00."),
            ("dusk", "Сумерки",
             "Случайное утро 05:00–06:15 либо вечер 19:30–21:15 — низкое "
             "солнце и длинные тени."),
            ("shadow", "Тень пополам",
             "Дневной свет плюс теневая полоса, рассекающая кузов: тень "
             "накладывается только на цветной кадр, разметка не страдает."),
        ]
        for key, title, hint in specs:
            box = _check(title, light["cycle"].get(key, True))
            box.toggled.connect(self._update_summary)
            self.chk_light[key] = box
            sec.add_option(box, hint)
        return sec

    # -- Сцена ----------------------------------------------------------
    def _section_scene(self):
        sec = _Section(
            "Сцена",
            "Необязательные усложнения кадра.",
        )
        scene = self.config["scene"]
        self.chk_cloth = _check("Ткань на борту", scene["cloth"])
        sec.add_option(self.chk_cloth,
                       "Тент, свисающий с борта: физическая симуляция "
                       "провиса, складок и полоскания на ветру. Место "
                       "крепления, размер и сила ветра случайны от кадра к "
                       "кадру. В маске сегментации — отдельный класс.")
        self.spn_cloth_p = _spin((0.0, 1.0), scene["cloth_probability"], 0.05,
                                 decimals=2, width=78)
        self.chk_cloth.toggled.connect(self._sync_enabled)
        sec.add_field("Доля кадров с тканью",
                      "1.00 — ткань всегда; меньше единицы оставляет часть "
                      "кадров без неё, чтобы в датасете были и негативные "
                      "примеры.",
                      self.spn_cloth_p)

        self.chk_bg = _check("Случайный фон", scene["random_background"])
        sec.add_option(self.chk_bg,
                       "Фон сцены на цветном кадре заменяется случайной "
                       "картинкой из assets/backgrounds; кузов и груз "
                       "остаются. Глубина и маска не меняются — они и "
                       "задают вырез переднего плана.")
        return sec

    # -- Глубина --------------------------------------------------------
    def _section_depth_preview(self):
        sec = _Section(
            "Глубина · превью",
            "Живой кадр из того же буфера, что кормит оверлей в углу окна, "
            "но раскрашенный ПАРАМЕТРАМИ ДАТАСЕТА. Сам оверлей эти "
            "настройки не трогают.",
        )
        # Превью тянется по высоте: на широком окне карточка выше самой
        # длинной страницы, и пустое место логично отдать картинке.
        sec.setSizePolicy(QSizePolicy.Policy.Preferred,
                          QSizePolicy.Policy.Expanding)
        depth = self.config["depth"]

        self.depth_canvas = QLabel()
        self.depth_canvas.setMinimumHeight(200)
        self.depth_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.depth_canvas.setStyleSheet(
            "background-color: #050505; border-radius: 6px;"
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;"
        )
        self.depth_canvas.setText("Превью недоступно")
        sec.add(self.depth_canvas)

        self.depth_legend = QLabel()
        self.depth_legend.setFixedHeight(10)
        self.depth_legend.setStyleSheet("border-radius: 3px;")
        sec.add(self.depth_legend)

        legend_row = QHBoxLayout()
        legend_row.addWidget(_label("дальше", size=10,
                                    color=COLOR_TEXT_MUTED), 0)
        legend_row.addStretch(1)
        legend_row.addWidget(_label("ближе", size=10,
                                    color=COLOR_TEXT_MUTED), 0)
        sec.add(legend_row)

        self.grp_depth_color = QButtonGroup(self)
        self.rb_depth_gray = _radio("Чёрно-белая", depth["grayscale"])
        self.rb_depth_rainbow = _radio("Цветная", not depth["grayscale"])
        for btn in (self.rb_depth_gray, self.rb_depth_rainbow):
            self.grp_depth_color.addButton(btn)
            btn.toggled.connect(self._tick_preview)
        sec.add_option(self.rb_depth_gray,
                       "Яркость линейно кодирует расстояние — то, что обычно "
                       "и нужно модели.")
        sec.add_option(self.rb_depth_rainbow,
                       "Радужный градиент: читается глазом, но как обучающий "
                       "сигнал хуже.")
        return sec

    def _section_depth_range(self):
        sec = _Section(
            "Глубина · диапазон",
            "Что именно попадает в шкалу сохраняемой карты.",
        )
        depth = self.config["depth"]

        # Границы шкалы заданы ДОЛЕЙ дальней плоскости, а не метрами: шейдер
        # линеаризует z как (2*near) / (far + near - z*(far - near)), то есть
        # получает расстояние, поделённое на far. Метровый эквивалент
        # показываем строкой ниже, чтобы значения читались.
        self.spn_grad_a = _spin((0.0, 1.0), depth["grad_start"], 0.01,
                                decimals=3, width=78)
        self.spn_grad_b = _spin((0.0, 1.0), depth["grad_end"], 0.01,
                                decimals=3, width=78)
        for spin in (self.spn_grad_a, self.spn_grad_b):
            spin.valueChanged.connect(self._tick_preview)
        sec.add_field("Начало шкалы",
                      "Ближний край диапазона, доля дальней плоскости. Всё, "
                      "что ближе, сливается в один цвет.",
                      self.spn_grad_a)
        sec.add_field("Конец шкалы",
                      "Дальний край диапазона. Чем уже диапазон, тем больше "
                      "разрешение по глубине внутри кузова.",
                      self.spn_grad_b)
        self.lbl_grad_meters = _label("", size=11, color=COLOR_ACCENT,
                                      mono=True, wrap=True)
        sec.add(self.lbl_grad_meters)

        self.spn_near = _spin((0.001, 1000.0), depth["near"], 0.01,
                              decimals=3, suffix=" м")
        self.spn_far = _spin((0.01, 10000.0), depth["far"], 1.0,
                             decimals=1, suffix=" м")
        for spin in (self.spn_near, self.spn_far):
            spin.valueChanged.connect(self._tick_preview)
        sec.add_field("Ближняя плоскость",
                      "Ближняя плоскость отсечения камеры глубины.",
                      self.spn_near)
        sec.add_field("Дальняя плоскость",
                      "Дальняя плоскость отсечения. Она же задаёт точность "
                      "z-буфера: слишком большое значение съедает "
                      "разрешение вблизи.",
                      self.spn_far)

        btn_take = _button("Взять из окна")
        btn_take.setToolTip(
            "Скопировать значения из настроек живого оверлея глубины")
        btn_take.clicked.connect(self._on_take_depth_from_live)
        sec.add(btn_take)
        return sec

    # -- Сегментация ----------------------------------------------------
    def _section_segmentation(self):
        sec = _Section(
            "Сегментация",
            "Цвета классов в маске. Они же уходят в json как легенда и "
            "используются для выреза переднего плана при замене фона.",
        )
        try:
            from src.rendering.segmentation_renderer import (
                SEG_COLORS, SEG_BACKGROUND, SEG_LABELS,
            )
        except Exception as exc:
            sec.add(_hint(f"Палитра недоступна: {exc}"))
            return sec

        palette = dict(self.config["segmentation"].get("palette") or {})
        base = {"background": tuple(SEG_BACKGROUND)}
        base.update({k: tuple(v) for k, v in SEG_COLORS.items()})

        labels = dict(SEG_LABELS)
        labels["background"] = ("Фон", "Всё, что не относится к классам выше")

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        order = [k for k in ("cargo", "cuzov", "cloth", "other", "ground")
                 if k in base] + ["background"]
        for row, key in enumerate(order):
            rgb = tuple(palette.get(key, base[key]))
            swatch = _Swatch(rgb)
            self._swatches[key] = swatch
            title, hint = labels.get(key, (key, ""))
            grid.addWidget(swatch, row, 0, Qt.AlignmentFlag.AlignTop)
            text = QVBoxLayout()
            text.setSpacing(1)
            text.addWidget(_label(title, size=12))
            text.addWidget(_hint(hint))
            holder = QWidget()
            holder.setStyleSheet("background: transparent;")
            holder.setLayout(text)
            grid.addWidget(holder, row, 1)
        grid.setColumnStretch(1, 1)
        sec.add(grid)

        sec.add(_hint(
            "Классы «Груз», «Кузов» и «Ткань» разбираются по цвету при "
            "замене фона, поэтому держите их различимыми — цвета, "
            "отличающиеся меньше чем на 40 по каналу, могут слиться."
        ))

        btn_reset = _button("Вернуть стандартные цвета")
        btn_reset.clicked.connect(self._on_reset_palette)
        sec.add(btn_reset)
        return sec

    # -- Лидар ----------------------------------------------------------
    def _section_lidar_sensor(self):
        sec = _Section(
            "Лидар · сенсор",
            "Виртуальный 3D-лидар стоит В КАМЕРЕ и наклоняется вместе с ней, "
            "но поле зрения у него СВОЁ — по умолчанию 360°×90°, как у "
            "Unitree 4D LiDAR L2. Он видит и то, что позади камеры, и бьёт "
            "лучом во всю видимую геометрию сцены: груз, кузов, насыпь, "
            "ткань, подложку, окружение. Промах (небо) возврата не даёт.",
        )
        lid = self.config["lidar"]

        self.lbl_lidar_backend = _label("", size=11, color=COLOR_TEXT_MUTED,
                                        mono=True, wrap=True)
        sec.add(self.lbl_lidar_backend)

        self.spn_lid_min = _int_spin((1000, 20000000), lid["points_min"],
                                     width=94)
        self.spn_lid_max = _int_spin((1000, 20000000), lid["points_max"],
                                     width=94)
        for spin in (self.spn_lid_min, self.spn_lid_max):
            spin.setSingleStep(25000)
            spin.setGroupSeparatorShown(True)
        self.spn_lid_min.valueChanged.connect(
            lambda v: self._clamp_order(self.spn_lid_min, self.spn_lid_max))
        self.spn_lid_max.valueChanged.connect(
            lambda v: self._clamp_order(self.spn_lid_min, self.spn_lid_max))
        sec.add_field(
            "Точек в облаке",
            "Число ВОЗВРАТОВ в кадре выбирается случайно внутри диапазона — "
            "у настоящего сенсора оно тоже плавает от кадра к кадру. Лучей "
            "выпускается больше: те, что ушли в небо, точек не дают.",
            self.spn_lid_min, _label("…", size=12, color=COLOR_TEXT_MUTED),
            self.spn_lid_max,
        )

        self.spn_lid_acc = _spin((0.0, 500.0), lid["accuracy_mm"], 0.5,
                                 decimals=2, suffix=" мм", width=88)
        sec.add_field(
            "Точность дальности",
            "Паспортный разброс дальномера, ±. Считается ТРЕМЯ сигмами "
            "гауссова шума: почти все возвраты укладываются в указанное "
            "значение, единицы — чуть дальше, как в жизни.",
            self.spn_lid_acc,
        )

        self.spn_lid_fov_h = _spin((1.0, 360.0), lid["fov_h_deg"], 10.0,
                                   decimals=1, suffix="°", width=88)
        self.spn_lid_fov_v = _spin((1.0, 360.0), lid["fov_v_deg"], 5.0,
                                   decimals=1, suffix="°", width=88)
        sec.add_field(
            "Обзор по азимуту",
            "360° — сенсор кругового обзора: в облако попадает и то, что "
            "камера не видит. Меньше 360° делает из него секторный лидар, "
            "смотрящий вдоль оси камеры. Только для развёртки «Вращение»: "
            "розетка — конусный сенсор, её ширину задаёт обзор по вертикали.",
            self.spn_lid_fov_h,
        )
        sec.add_field(
            "Обзор по вертикали",
            "Полная высота полосы обзора вокруг направления взгляда камеры; "
            "у розетки — полный раствор её конуса. У паспортного L2 это 90°, "
            "а 96° — это тот же обзор плюс 6° режима отрицательных углов, "
            "не 96° вверх и 96° вниз. Больше 180° тоже можно: луч тогда "
            "переваливает через полюс (угол места физически не бывает "
            "больше 90°), 180° — уже полный круговой охват, 360° у розетки — "
            "полная сфера.",
            self.spn_lid_fov_v,
        )

        self.spn_lid_rmin = _spin((0.0, 1000.0), lid["min_range_m"], 0.05,
                                  decimals=2, suffix=" м", width=88)
        self.spn_lid_rmax = _spin((0.1, 10000.0), lid["max_range_m"], 5.0,
                                  decimals=1, suffix=" м", width=88)
        sec.add_field(
            "Дальность",
            "Ближе минимума и дальше максимума возвратов нет. Максимум "
            "заодно отсекает небосвод и дальнюю бутафорию сцены.",
            self.spn_lid_rmin, _label("…", size=12, color=COLOR_TEXT_MUTED),
            self.spn_lid_rmax,
        )
        return sec

    def _section_lidar_pattern(self):
        sec = _Section(
            "Лидар · развёртка",
            "Как луч обходит поле зрения. Часть полей принадлежит только "
            "одной механике — лишние гаснут сами.",
        )
        lid = self.config["lidar"]

        self.grp_lid_pattern = QButtonGroup(self)
        self.rb_lid_rosette = _radio("Розетка", lid["pattern"] != "spin")
        self.rb_lid_spin = _radio("Вращение (как у L2)",
                                  lid["pattern"] == "spin")
        for btn in (self.rb_lid_rosette, self.rb_lid_spin):
            self.grp_lid_pattern.addButton(btn)
            btn.toggled.connect(self._sync_enabled)
        sec.add_option(self.rb_lid_rosette,
                       "Схема Livox: два встречно вращающихся клина рисуют "
                       "окружности, стягивающиеся к оси взгляда — точки "
                       "гуще всего там, куда смотрит камера. Сенсор "
                       "конусный, кругового обзора не даёт.")
        sec.add_option(self.rb_lid_spin,
                       "Паспортная механика L2: голова крутится вокруг оси "
                       "«вверх» камеры на 5.55 Гц, а быстрый элемент гоняет "
                       "луч по конусу на 216 Гц — 38.9 окружности на "
                       "оборот, через 9.25° по азимуту. Единственная "
                       "развёртка, дающая круговой обзор 360°×90°.")

        self.spn_lid_bias = _spin((0.05, 8.0), lid["center_bias"], 0.05,
                                  decimals=2, width=88)
        sec.add_field(
            "Сгущение к центру",
            "Развёртка — розетка из окружностей (как у Livox и Unitree L2): "
            "самые мелкие ложатся на ось камеры, поэтому в центре кадра "
            "точек и так гуще всего. 1.00 — честная физика розетки, больше "
            "единицы дополнительно стягивает точки к оси, меньше — "
            "растаскивает к краю.",
            self.spn_lid_bias,
        )

        self.spn_lid_circle = _spin((8.0, 100000.0), lid["beams_per_circle"],
                                    100.0, decimals=0, width=88)
        self.spn_lid_ratio = _spin((0.01, 4.0), lid["circle_ratio"], 0.01,
                                   decimals=3, width=88)
        sec.add_field(
            "Лучей на окружность",
            "Шаг развёртки: сколько лучей приходится на один оборот клина. "
            "Меньше — реже точки вдоль самой траектории, крупнее рисунок.",
            self.spn_lid_circle,
        )
        sec.add_field(
            "Отношение клиньев",
            "Скорость второго клина к первому. Иррациональное отношение "
            "(0.618 — золотое) не даёт траектории замкнуться: обороты не "
            "ложатся в те же борозды, развёртка непериодическая. Круглые "
            "дроби диалог сам чуть сдвигает.",
            self.spn_lid_ratio,
        )

        self.spn_lid_spin_hz = _spin((0.01, 1000.0), lid["spin_hz"], 0.05,
                                     decimals=2, suffix=" Гц", width=88)
        self.spn_lid_vert_hz = _spin((0.01, 100000.0), lid["vertical_hz"],
                                     1.0, decimals=1, suffix=" Гц", width=88)
        self.spn_lid_rate = _int_spin((100, 100000000), int(lid["point_rate"]),
                                      width=94)
        self.spn_lid_rate.setSingleStep(1000)
        self.spn_lid_rate.setGroupSeparatorShown(True)
        sec.add_field(
            "Оборотов в секунду",
            "Горизонтальная развёртка. У L2 — 5.55 Гц.",
            self.spn_lid_spin_hz,
        )
        sec.add_field(
            "Вертикальная развёртка",
            "У L2 — 216 Гц, то есть 38.9 взмаха на оборот. Отношение "
            "НЕЦЕЛОЕ, поэтому каждый следующий оборот кладёт взмахи в "
            "промежутки предыдущего и развёртка не замыкается; круглое "
            "отношение диалог сам чуть сдвигает.",
            self.spn_lid_vert_hz,
        )
        sec.add_field(
            "Отсчётов в секунду",
            "Частота дальномера: у L2 128 000 (эффективных 64 000). Вместе "
            "с частотами развёртки задаёт шаг точек по траектории.",
            self.spn_lid_rate,
        )
        return sec

    def _section_lidar_output(self):
        sec = _Section(
            "Лидар · шум и вывод",
            "Что портит идеальное облако и в каком виде оно ложится в файл.",
        )
        lid = self.config["lidar"]

        self.spn_lid_jit = _spin((0.0, 5.0), lid["jitter_deg"], 0.01,
                                 decimals=3, suffix="°", width=88)
        sec.add_field(
            "Дрожание луча",
            "Угловая неровность развёртки: люфт привода и дрожание клиньев. "
            "Без неё траектория идеально гладкая, и сеть выучивает саму "
            "развёртку вместо формы груза.",
            self.spn_lid_jit,
        )

        self.spn_lid_drop = _spin((0.0, 90.0), lid["dropout_pct"], 0.5,
                                  decimals=1, suffix="%", width=88)
        sec.add_field(
            "Потери возвратов",
            "Доля потерянных точек. На скользящих углах теряется кратно "
            "больше — отсюда характерные прорехи на бортах и на дальнем "
            "скате насыпи.",
            self.spn_lid_drop,
        )

        self.grp_lid_frame = QButtonGroup(self)
        self.rb_lid_sensor = _radio("Система сенсора",
                                    lid["frame"] != "world")
        self.rb_lid_world = _radio("Мировая система",
                                   lid["frame"] == "world")
        for btn in (self.rb_lid_sensor, self.rb_lid_world):
            self.grp_lid_frame.addButton(btn)
        sec.add_option(self.rb_lid_sensor,
                       "Координаты относительно сенсора (x — вправо, y — "
                       "вперёд, z — вверх), как отдаёт настоящий лидар. Поза "
                       "камеры пишется в json, поэтому облако всегда можно "
                       "перевести в мир.")
        sec.add_option(self.rb_lid_world,
                       "Координаты сразу в мировой системе сцены: удобно "
                       "сравнивать облако с исходной геометрией и считать "
                       "объём.")

        self.chk_lid_traj = _check("Точки строго по траектории",
                                   lid.get("trajectory", False))
        sec.add_option(self.chk_lid_traj,
                       "Развёртка — это ОДНА кривая на сфере. Если идти по "
                       "ней подряд, точки ложатся ниткой: на паспортных "
                       "частотах L2 шаг вдоль траектории 0.44°, а соседние "
                       "её проходы отстоят на 0.20° — глаз читает это как "
                       "пересекающиеся линии сканирования, которых на живых "
                       "снимках нет (там миллион точек копится секундами, "
                       "платформа дрожит, фаза уходит, и от кривой остаётся "
                       "только плотность). По умолчанию фазы развёртки "
                       "случайны: распределение углов то же самое, ниток "
                       "нет. Включайте, только чтобы посмотреть на сам "
                       "узор.")

        self.chk_lid_color = _check("Цвет классов в точках",
                                    lid.get("color", True))
        sec.add_option(self.chk_lid_color,
                       "Каждая точка получает rgb класса из палитры "
                       "сегментации — облако открывается глазами в любом "
                       "вьюере. Числовая метка класса пишется всегда.")
        self.chk_lid_binary = _check("Двоичный ply",
                                     lid.get("binary", True))
        sec.add_option(self.chk_lid_binary,
                       "Миллион точек в тексте — это ~40 МБ и секунды на "
                       "запись. Снимайте галочку только чтобы заглянуть в "
                       "файл глазами.")
        return sec

    # ------------------------------------------------------------------
    # Реакция на изменения
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_order(low_spin, high_spin):
        """Верхняя граница диапазона не может опуститься ниже нижней."""
        if high_spin.value() < low_spin.value():
            high_spin.blockSignals(True)
            high_spin.setValue(low_spin.value())
            high_spin.blockSignals(False)

    @staticmethod
    def _clamp_pair(changed, other, value):
        """Две доли не могут в сумме превышать 100% — подрезаем соседа."""
        try:
            if int(value) + int(other.value()) > 100:
                other.blockSignals(True)
                other.setValue(max(0, 100 - int(value)))
                other.blockSignals(False)
        except Exception:
            pass

    def _on_outputs_changed(self):
        self._sync_enabled()
        self._update_summary()

    def _sync_enabled(self):
        if not self._ready:
            return
        depth_on = self.chk_out["depth"].isChecked()
        seg_on = self.chk_out["segmentation"].isChecked()
        color_on = self.chk_out["color"].isChecked()
        lidar_on = self.chk_out["lidar"].isChecked()

        for widget in (self.depth_canvas, self.depth_legend,
                       self.lbl_grad_meters, self.rb_depth_gray,
                       self.rb_depth_rainbow, self.spn_grad_a, self.spn_grad_b,
                       self.spn_near, self.spn_far):
            widget.setEnabled(depth_on)
        for swatch in self._swatches.values():
            swatch.setEnabled(seg_on)

        for widget in (self.spn_lid_min, self.spn_lid_max, self.spn_lid_acc,
                       self.spn_lid_fov_v,
                       self.spn_lid_rmin, self.spn_lid_rmax,
                       self.spn_lid_jit, self.spn_lid_drop,
                       self.rb_lid_rosette, self.rb_lid_spin,
                       self.rb_lid_sensor, self.rb_lid_world,
                       self.chk_lid_traj, self.chk_lid_color,
                       self.chk_lid_binary):
            widget.setEnabled(lidar_on)
        # Параметры развёртки принадлежат разным механикам: клинья розетке,
        # частоты вращению. Гасим то, что сейчас ни на что не влияет.
        rosette_on = lidar_on and self.rb_lid_rosette.isChecked()
        for widget in (self.spn_lid_bias, self.spn_lid_circle,
                       self.spn_lid_ratio):
            widget.setEnabled(rosette_on)
        # Обзор по азимуту осмыслен только у вращающейся головы: у розетки
        # это конус вокруг оси камеры, и «360° по кругу» для неё пустой звук.
        spin_on = lidar_on and self.rb_lid_spin.isChecked()
        for widget in (self.spn_lid_spin_hz, self.spn_lid_vert_hz,
                       self.spn_lid_rate, self.spn_lid_fov_h):
            widget.setEnabled(spin_on)
        if lidar_on:
            self._refresh_lidar_backend()
        else:
            self.lbl_lidar_backend.setText("")

        variants_on = self.rb_cam_variants.isChecked()
        for box in self.chk_var.values():
            box.setEnabled(variants_on)
        self.spn_samples.setEnabled(self.rb_cam_random.isChecked())
        moving = not self.rb_cam_fixed.isChecked()
        self.spn_angle.setEnabled(moving)
        self.spn_offset.setEnabled(moving)

        cycle_on = self.rb_light_cycle.isChecked()
        for box in self.chk_light.values():
            box.setEnabled(cycle_on)

        self.spn_cloth_p.setEnabled(self.chk_cloth.isChecked())
        # Замена фона живёт только на цветном кадре: без него опция ничего
        # не делает.
        self.chk_bg.setEnabled(color_on)

        # Потолок и доли крайних случаев работают только в случайном режиме:
        # линейный рост доводит объём ровно до паспортного максимума.
        random_vol = self.rb_vol_random.isChecked()
        for spin in (self.spn_full, self.spn_empty, self.spn_ceiling):
            spin.setEnabled(random_vol)

        self._sync_nav()

        if not (color_on or depth_on or seg_on):
            self.lbl_out_warn.setText(
                "Выберите хотя бы один файл — иначе кадр рендерится впустую.")
        else:
            self.lbl_out_warn.setText("")
        self._update_summary()

    def _sync_nav(self):
        """Подписи страниц показывают, что сейчас реально снимается.

        Раньше про выключенную глубину можно было узнать, только пролистав
        до её полей. Теперь это видно прямо в списке разделов.
        """
        if not getattr(self, "_nav_buttons", None):
            return
        states = {
            "depth": self.chk_out["depth"].isChecked(),
            "segmentation": self.chk_out["segmentation"].isChecked(),
            "lidar": self.chk_out["lidar"].isChecked(),
        }
        for key, on in states.items():
            btn = self._nav_buttons.get(key)
            if btn is None:
                continue
            btn.set_subtitle(self._nav_subtitles[key] if on
                             else "выход выключен")

    def _refresh_lidar_backend(self):
        """Чем будем трассировать. Считается ЛЕНИВО и один раз.

        Определение бэкенда тянет за собой импорт Warp (секунда и печать в
        консоль), поэтому его нельзя делать при сборке диалога: пользователь,
        который лидар не включал, платить за это не должен.
        """
        if getattr(self, "_lidar_backend", None) is not None:
            return
        try:
            from src.rendering.lidar_scanner import backend_name
            name = backend_name()
        except Exception as exc:              # noqa: BLE001
            name = None
            print(f"[Dataset] бэкенд лидара не определён: {exc}")
        self._lidar_backend = name or "нет"
        if name == "warp-cuda":
            text = "трассировка: NVIDIA Warp (CUDA) — миллион лучей за кадр"
        elif name == "embree":
            text = "трассировка: Embree на CPU (~1 с на миллион лучей)"
        elif name == "warp-cpu":
            text = ("трассировка: Warp на CPU — около 15 с на миллион лучей; "
                    "поставьте embreex, если нет CUDA")
        else:
            text = ("нечем трассировать: нет ни warp-lang, ни embreex — "
                    "облако точек сниматься не будет")
        self.lbl_lidar_backend.setText(text)

    def _update_summary(self):
        if not self._ready:
            return
        cfg = self._collect(validate=False)
        per_fill = dscfg.frames_per_fill(cfg)
        total = dscfg.total_frames(cfg)
        files = len(dscfg.output_list(cfg))
        self.lbl_frames.setText(
            f"{per_fill} кадр(ов) с наполнения → {total} кадров всего")
        self.lbl_summary.setText(f"{total} кадров · {files} файла на кадр")
        if getattr(self, "_nav_buttons", None):
            self._nav_buttons["scope"].set_subtitle(
                f"{total} кадров · {files} файла")
        if hasattr(self, "lbl_lidar_backend") and \
                self.chk_out["lidar"].isChecked():
            self._refresh_lidar_backend()
        can_start = files > 0 and total > 0
        if hasattr(self, "btn_start"):
            self.btn_start.setEnabled(can_start)
        if hasattr(self, "lbl_footer_hint"):
            self.lbl_footer_hint.setText(
                "Съёмка блокирует окно до конца прогона; прогресс виден на "
                "кнопке в карточке камеры."
                if can_start else ""
            )

    def _tick_preview(self):
        if not self._ready:
            return
        # Чтение буфера глубины стоит кадра; на других страницах превью всё
        # равно не видно, поэтому там таймер не делает ничего.
        if hasattr(self, "stack") and \
                self.stack.currentIndex() != getattr(self, "_depth_page", -1):
            return
        grayscale = self.rb_depth_gray.isChecked()

        strip = gradient_strip_qimage(
            max(1, self.depth_legend.width()), 10, grayscale)
        if strip is not None:
            self.depth_legend.setPixmap(QPixmap.fromImage(strip))

        far = float(self.spn_far.value())
        self.lbl_grad_meters.setText(
            f"шкала ≈ {self.spn_grad_a.value() * far:.2f} – "
            f"{self.spn_grad_b.value() * far:.2f} м от камеры"
        )

        if not self.chk_out["depth"].isChecked():
            self.depth_canvas.setText("Карта глубины выключена")
            self.depth_canvas.setPixmap(QPixmap())
            return

        dr = getattr(self._panda_app, "depth_renderer", None) \
            if self._panda_app is not None else None
        if dr is None:
            self.depth_canvas.setText("Рендер глубины недоступен")
            return

        width = max(160, self.depth_canvas.width())
        img = depth_to_qimage(
            dr, width, int(width * 9 / 16),
            near=self.spn_near.value(),
            far=self.spn_far.value(),
            grad_start=self.spn_grad_a.value(),
            grad_end=self.spn_grad_b.value(),
            grayscale=grayscale,
        )
        if img is None:
            self.depth_canvas.setText("Ожидание кадра глубины…")
            return
        pix = QPixmap.fromImage(img).scaled(
            self.depth_canvas.width(), self.depth_canvas.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.depth_canvas.setPixmap(pix)

    def _on_take_depth_from_live(self):
        dr = getattr(self._panda_app, "depth_renderer", None) \
            if self._panda_app is not None else None
        if dr is None:
            return
        live = dr.capture_settings() if hasattr(dr, "capture_settings") else {}
        for spin, key in ((self.spn_near, "near"), (self.spn_far, "far"),
                          (self.spn_grad_a, "grad_start"),
                          (self.spn_grad_b, "grad_end")):
            value = live.get(key)
            if value is not None:
                spin.setValue(float(value))
        self._tick_preview()

    def _on_browse(self):
        start = self.edt_out.text().strip() or dscfg.PROJECT_ROOT
        if not os.path.isabs(start):
            start = os.path.join(dscfg.PROJECT_ROOT, start)
        chosen = QFileDialog.getExistingDirectory(
            self, "Каталог для датасета", start)
        if not chosen:
            return
        # Путь внутри проекта показываем относительным — так конфиг остаётся
        # переносимым между машинами.
        try:
            rel = os.path.relpath(chosen, dscfg.PROJECT_ROOT)
            if not rel.startswith(".."):
                chosen = rel.replace("\\", "/")
        except ValueError:
            pass
        self.edt_out.setText(chosen)

    def _on_reset_palette(self):
        try:
            from src.rendering.segmentation_renderer import (
                DEFAULT_SEG_COLORS, DEFAULT_SEG_BACKGROUND,
            )
        except Exception:
            return
        base = {"background": tuple(DEFAULT_SEG_BACKGROUND)}
        base.update({k: tuple(v) for k, v in DEFAULT_SEG_COLORS.items()})
        for key, swatch in self._swatches.items():
            if key in base:
                swatch.set_rgb(base[key])

    def _on_reset(self):
        self.config = dscfg.defaults()
        self.action = "reset"
        self.accept()

    # ------------------------------------------------------------------
    # Сбор результата
    # ------------------------------------------------------------------
    def _collect(self, validate=True) -> dict:
        cam_mode = ("fixed" if self.rb_cam_fixed.isChecked()
                    else "random" if self.rb_cam_random.isChecked()
                    else "variants")
        light_mode = ("overhead" if self.rb_light_overhead.isChecked()
                      else "current" if self.rb_light_current.isChecked()
                      else "cycle")
        cfg = {
            "count": int(self.spn_count.value()),
            "output_dir": self.edt_out.text().strip(),
            "outputs": {k: box.isChecked()
                        for k, box in self.chk_out.items()},
            "volume": {
                "mode": ("random" if self.rb_vol_random.isChecked()
                         else "ramp"),
                "full_pct": float(self.spn_full.value()),
                "empty_pct": float(self.spn_empty.value()),
                "ceiling_k": float(self.spn_ceiling.value()),
            },
            "camera": {
                "mode": cam_mode,
                "angle_deg": float(self.spn_angle.value()),
                "offset_m": float(self.spn_offset.value()),
                "samples": int(self.spn_samples.value()),
                "variants": {k: box.isChecked()
                             for k, box in self.chk_var.items()},
            },
            "lighting": {
                "mode": light_mode,
                "cycle": {k: box.isChecked()
                          for k, box in self.chk_light.items()},
            },
            "scene": {
                "cloth": self.chk_cloth.isChecked(),
                "cloth_probability": float(self.spn_cloth_p.value()),
                "random_background": self.chk_bg.isChecked(),
            },
            "depth": {
                "grayscale": self.rb_depth_gray.isChecked(),
                "near": float(self.spn_near.value()),
                "far": float(self.spn_far.value()),
                "grad_start": float(self.spn_grad_a.value()),
                "grad_end": float(self.spn_grad_b.value()),
            },
            "segmentation": {
                "palette": {k: list(s.rgb())
                            for k, s in self._swatches.items()},
            },
            "lidar": {
                "points_min": int(self.spn_lid_min.value()),
                "points_max": int(self.spn_lid_max.value()),
                "accuracy_mm": float(self.spn_lid_acc.value()),
                "pattern": ("spin" if self.rb_lid_spin.isChecked()
                            else "rosette"),
                "fov_h_deg": float(self.spn_lid_fov_h.value()),
                "fov_v_deg": float(self.spn_lid_fov_v.value()),
                "spin_hz": float(self.spn_lid_spin_hz.value()),
                "vertical_hz": float(self.spn_lid_vert_hz.value()),
                "point_rate": float(self.spn_lid_rate.value()),
                "min_range_m": float(self.spn_lid_rmin.value()),
                "max_range_m": float(self.spn_lid_rmax.value()),
                "center_bias": float(self.spn_lid_bias.value()),
                "jitter_deg": float(self.spn_lid_jit.value()),
                "dropout_pct": float(self.spn_lid_drop.value()),
                "beams_per_circle": float(self.spn_lid_circle.value()),
                "circle_ratio": float(self.spn_lid_ratio.value()),
                "frame": ("world" if self.rb_lid_world.isChecked()
                          else "sensor"),
                "trajectory": self.chk_lid_traj.isChecked(),
                "binary": self.chk_lid_binary.isChecked(),
                "color": self.chk_lid_color.isChecked(),
            },
        }
        return dscfg.normalize(cfg) if validate else cfg

    def _finish(self, action):
        self.config = self._collect()
        self.action = action
        self.accept()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    # Ширина всего, что окружает колонки страницы: боковой список,
    # разделитель, поля карточки и полоса прокрутки. Считается один раз по
    # факту первой раскладки, до неё берётся эта оценка.
    CHROME_W = 283
    CHROME_H = 140

    def _fit_card(self, avail_w, avail_h):
        """Подобрать размер карточки под содержимое.

        Растягивать диалог на весь экран бессмысленно: колонки шире 560 px
        читаются плохо, и при пяти колонках половина карточки оставалась бы
        пустой. Поэтому перебираем варианты «сколько колонок», берём самый
        узкий, при котором ни одна страница не требует прокрутки, и уже под
        него подгоняем высоту.
        """
        pages = [self.stack.widget(i).widget()
                 for i in range(self.stack.count())]
        best = None
        for content_w in (440, 892, 1344, 1796):
            card_w = min(avail_w, content_w + self.CHROME_W)
            # +24 — запас на округления переноса подписей: без него страница
            # промахивается на десяток пикселей и получает полосу прокрутки
            # ради одной строки.
            need_h = max(page.ideal_height(card_w - self.CHROME_W)
                         for page in pages) + self.CHROME_H + 24
            best = (card_w, need_h)
            if need_h <= avail_h or card_w >= avail_w:
                break
        card_w, need_h = best
        self.card.setFixedSize(max(760, card_w),
                               max(520, min(avail_h, need_h)))

    def showEvent(self, event):
        parent = self.parentWidget()
        top = parent.window() if parent is not None else None
        if top is not None:
            geo = top.geometry()
            self.setGeometry(geo)
            max_w, max_h = self.CARD_MAX
            mar_w, mar_h = self.CARD_MARGIN
            self._avail = (min(max_w, max(760, geo.width() - mar_w)),
                           min(max_h, max(520, geo.height() - mar_h)))
            self._fit_card(*self._avail)
        super().showEvent(event)
        QTimer.singleShot(0, self._refit_card)
        QTimer.singleShot(0, self._tick_preview)

    def _refit_card(self):
        """Уточнить высоту по реальной раскладке.

        Оценка полей выше сделана до первого показа; когда страница уже
        разложена, ширину колонок и высоту обвязки можно измерить точно.
        """
        avail = getattr(self, "_avail", None)
        page = self.stack.currentWidget()
        if avail is None or page is None:
            return
        viewport = page.viewport().height()
        if viewport <= 0:
            return
        type(self).CHROME_W = max(120, self.card.width()
                                  - page.viewport().width())
        type(self).CHROME_H = max(60, self.card.height() - viewport)
        self._fit_card(*avail)

    def closeEvent(self, event):
        self._preview_timer.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        # Enter в спинбоксе не должен запускать съёмку — она стоит слишком
        # дорого, чтобы стартовать от случайного нажатия.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            keys = (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4,
                    Qt.Key.Key_5, Qt.Key.Key_6, Qt.Key.Key_7)
            if event.key() in keys:
                self._go_page(keys.index(event.key()))
                return
            if event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_PageDown):
                self._go_page(
                    (self.stack.currentIndex() + 1) % self.stack.count())
                return
            if event.key() in (Qt.Key.Key_Backtab, Qt.Key.Key_PageUp):
                self._go_page(
                    (self.stack.currentIndex() - 1) % self.stack.count())
                return
        super().keyPressEvent(event)
