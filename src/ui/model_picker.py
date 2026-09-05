# model_picker.py
# ---------------------------------------------------------------------------
# Выбор набора моделей (кузова) для правой панели.
#
# Зачем не QComboBox
# ------------------
# Панель шириной 320 px, а имена наборов вроде
# "SCANIA YS2P380 CB8X4EHZ (локальная)" в неё не влезают: список превращался
# в колонку обрезанных строк, по которой невозможно ни найти нужный кузов, ни
# сравнить их между собой. Здесь вместо выпадающего списка — всплывающая
# таблица шириной ~760 px:
#
#     Модель │ Источник │ Шасси │ Объём, м³ │ Габариты Д×Ш×В │ Комплект
#
# плюс строка поиска, фильтр по источнику (сервер / генератор / локальные),
# сортировка по клику на заголовок и подвал с подробностями выделенной
# строки (откуда взята геометрия, опорная плоскость, файл, ключ набора).
#
# API
# ---
# `ModelPickerCombo` намеренно повторяет тот кусок API QComboBox, которым
# пользуется right_panel.py (addItem / clear / count / itemData / itemText /
# currentIndex / setCurrentIndex / currentData / findData / currentText +
# сигнал currentIndexChanged). Благодаря этому подмена виджета не потребовала
# правок ни в MainWindow, ни в cli.py; характеристики строк передаются
# отдельно через `set_details()` и не влияют на выбор ключа.
#
# Сверх комбо-бокса есть один сигнал — `deleteRequested(key)`: пользователь
# просит убрать набор с диска (кнопка «Удалить» в подвале, клавиша Del,
# контекстное меню строки). Сам виджет ничего не удаляет и ни о чём не
# спрашивает: он лишь закрывается и отдаёт ключ наверх, потому что список —
# Qt.Popup, поверх которого модальное подтверждение не показать, а удаляемый
# набор может быть сейчас в сцене. Подтверждение и работу с файлами делают
# `MainWindow._on_model_delete_requested` и `panel_data.delete_model_set`.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QPainter, QPainterPath,
                         QPalette, QPen, QPolygonF)
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QFrame,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMenu, QPushButton, QSizePolicy, QStyle,
                             QStyleOptionViewItem, QStyledItemDelegate,
                             QToolButton, QTreeWidget, QTreeWidgetItem,
                             QVBoxLayout, QWidget)

from src.ui.panel_data import ModelSetInfo, can_delete_model_set
from src.ui.ui_theme import (COLOR_ACCENT, COLOR_ACCENT_SOFT, COLOR_BG,
                             COLOR_DANGER, COLOR_HAIRLINE,
                             COLOR_HAIRLINE_HOVER, COLOR_SURFACE,
                             COLOR_SURFACE_ELEVATED, COLOR_TEXT,
                             COLOR_TEXT_DIM, COLOR_TEXT_MUTED, FONT_MONO)

# Колонки таблицы.
COL_NAME, COL_SOURCE, COL_CHASSIS, COL_VOLUME, COL_DIMS, COL_KIT = range(6)

COLUMNS = (
    ("Модель",          260),
    ("Источник",         96),
    ("Шасси",            64),
    ("Объём, м³",        86),
    ("Габариты Д×Ш×В",  148),
    ("Комплект",        108),
)

#: Роль, в которой строка хранит значение для сортировки (число, а не текст:
#: иначе "9.5" оказывается больше "21").
SORT_ROLE = Qt.ItemDataRole.UserRole + 1

#: Подписи кнопок-фильтров: во множественном числе, в отличие от чипа в
#: колонке «Источник».
FILTER_LABELS = {
    "server":    "Серверные",
    "yaml":      "Из конфига",
    "generated": "Генератор",
    "local":     "Локальные",
}

#: Цвет чипа источника — чтобы происхождение набора читалось не по буквам.
SOURCE_COLORS = {
    # Не полный акцент: серверных наборов в списке большинство, и ярко-мятная
    # колонка на 18 строк перетягивает на себя всё внимание.
    "server":    COLOR_ACCENT_SOFT,
    "yaml":      COLOR_TEXT_MUTED,
    "generated": "#9B8CFF",
    "local":     "#FFB020",
}

_EM_DASH = "—"


def _fmt_size(size: int) -> str:
    if size <= 0:
        return ""
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.1f} ГБ"
    if size >= 1024 ** 2:
        return f"{size / 1024 ** 2:.0f} МБ"
    return f"{size / 1024:.0f} КБ"


def _fmt_volume(info: ModelSetInfo) -> str:
    if info.volume is None:
        return _EM_DASH
    return f"{info.volume:.2f}".rstrip("0").rstrip(".")


def _fmt_dims(info: ModelSetInfo) -> str:
    dims = info.dims
    if not dims:
        return _EM_DASH
    return " × ".join(f"{v:.2f}" for v in dims)


class _SortableItem(QTreeWidgetItem):
    """Строка таблицы, которая умеет сортироваться по числу, а не по тексту."""

    def __lt__(self, other: QTreeWidgetItem) -> bool:  # type: ignore[override]
        col = self.treeWidget().sortColumn() if self.treeWidget() else 0
        a = self.data(col, SORT_ROLE)
        b = other.data(col, SORT_ROLE)
        if a is not None and b is not None:
            return a < b
        return self.text(col).lower() < other.text(col).lower()


class _ColorDelegate(QStyledItemDelegate):
    """
    Раскраска ячеек по ForegroundRole.

    Ни `setForeground`, ни палитра варианта отрисовки сами по себе не
    работают: у таблицы есть свой QSS, а QStyleSheetStyle рисует текст
    невыделенных строк цветом из таблицы стилей. Поэтому фон и рамку
    рисует стиль (как обычно), а текст — мы сами, нужным цветом.
    """

    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        color = None
        if brush is not None:
            color = brush.color() if hasattr(brush, "color") else QColor(brush)

        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        text, opt.text = opt.text, ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt,
                          painter, widget)
        if not text:
            return

        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText,
                                    opt, widget).adjusted(3, 0, -3, 0)
        painter.save()
        painter.setPen(color or opt.palette.color(QPalette.ColorRole.Text))
        painter.setFont(opt.font)
        align = int(opt.displayAlignment) or int(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        painter.drawText(rect, align, QFontMetrics(opt.font).elidedText(
            text, Qt.TextElideMode.ElideRight, rect.width()))
        painter.restore()


class _FieldButton(QPushButton):
    """
    Закрытое состояние выбора: слева — имя набора (с эллипсисом), справа —
    короткая сводка (шасси · объём) и треугольник. Рисуется вручную, потому
    что QPushButton не умеет две строки разной яркости в одной строке.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self._primary = "—"
        self._secondary = ""
        self._accent = COLOR_TEXT_MUTED
        self._open = False

    def set_content(self, primary: str, secondary: str, accent: str) -> None:
        self._primary, self._secondary, self._accent = (primary, secondary,
                                                        accent)
        self.update()

    def set_open(self, is_open: bool) -> None:
        self._open = is_open
        self.update()

    def paintEvent(self, _event) -> None:                # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        enabled = self.isEnabled()
        hot = self._open or self.underMouse()
        border = QColor(COLOR_ACCENT if self._open else
                        (COLOR_HAIRLINE_HOVER if hot else COLOR_HAIRLINE))
        path = QPainterPath()
        path.addRoundedRect(rect, 6, 6)
        p.fillPath(path, QColor(COLOR_SURFACE_ELEVATED if hot
                                else COLOR_SURFACE))
        p.strokePath(path, QPen(border, 1))

        # Левый маркер-полоска цветом источника: видно, серверный это набор
        # или локальный, не читая текста.
        if enabled and self._accent:
            bar = QPainterPath()
            bar.addRoundedRect(QRectF(rect.left() + 4, rect.center().y() - 7,
                                      2.5, 14), 1.2, 1.2)
            p.fillPath(bar, QColor(self._accent))

        # Правый блок: сводка + треугольник.
        p.setFont(QFont(self.font()))
        caret_w = 16
        right = rect.right() - 8
        tri = QPolygonF([QPointF(right - caret_w + 3, rect.center().y() - 2),
                         QPointF(right - 3, rect.center().y() - 2),
                         QPointF(right - caret_w / 2, rect.center().y() + 3.5)])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(COLOR_ACCENT if self._open else COLOR_TEXT_MUTED))
        p.drawPolygon(tri)
        right -= caret_w + 2

        if self._secondary:
            f = QFont(self.font())
            f.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
            p.setFont(f)
            fm = QFontMetrics(f)
            w = min(fm.horizontalAdvance(self._secondary) + 2,
                    rect.width() * 0.6)
            p.setPen(QColor(COLOR_TEXT_DIM))
            p.drawText(QRectF(right - w, rect.top(), w, rect.height()),
                       int(Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter),
                       fm.elidedText(self._secondary,
                                     Qt.TextElideMode.ElideRight, int(w)))
            right -= w + 8

        p.setFont(self.font())
        fm = QFontMetrics(self.font())
        left = rect.left() + 13
        avail = max(10, right - left)
        p.setPen(QColor(COLOR_TEXT if enabled else COLOR_TEXT_DIM))
        p.drawText(QRectF(left, rect.top(), avail, rect.height()),
                   int(Qt.AlignmentFlag.AlignLeft
                       | Qt.AlignmentFlag.AlignVCenter),
                   fm.elidedText(self._primary, Qt.TextElideMode.ElideRight,
                                 int(avail)))
        p.end()


class _PickerPopup(QFrame):
    """
    Всплывающая таблица наборов.

    Qt.Popup: закрывается по клику мимо и по Esc, как обычный выпадающий
    список, и не отбирает активацию у главного окна (важно, потому что окно
    Panda3D встроено в Qt — лишние top-level окна ему противопоказаны).
    """

    picked = pyqtSignal(int)          # индекс выбранной строки в комбо
    removeRequested = pyqtSignal(int)  # индекс строки, которую просят удалить

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("ModelPickerPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._infos: List[Optional[ModelSetInfo]] = []
        self._source_filter = ""
        self._build()

    # -- построение ----------------------------------------------------
    def _build(self) -> None:
        self.setStyleSheet(f"""
            QFrame#ModelPickerPopup {{
                background-color: {COLOR_BG};
                border: 1px solid {COLOR_HAIRLINE_HOVER};
                border-radius: 10px;
            }}
            QLineEdit {{
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_HAIRLINE};
                border-radius: 6px;
                padding: 6px 10px;
                color: {COLOR_TEXT};
            }}
            QLineEdit:focus {{ border-color: {COLOR_ACCENT}; }}
            QToolButton {{
                background: transparent;
                border: 1px solid {COLOR_HAIRLINE};
                border-radius: 4px;
                padding: 3px 9px;
                color: {COLOR_TEXT_MUTED};
                font-size: 11px;
            }}
            QToolButton:hover {{ border-color: {COLOR_HAIRLINE_HOVER};
                                 color: {COLOR_TEXT}; }}
            QToolButton:checked {{
                border-color: {COLOR_ACCENT};
                color: {COLOR_ACCENT};
                background: rgba(0, 255, 136, 18);
            }}
            QToolButton:disabled {{
                color: {COLOR_TEXT_DIM};
                border-color: {COLOR_HAIRLINE};
            }}
            QToolButton#PickerDelete:enabled {{ color: {COLOR_TEXT_MUTED}; }}
            QToolButton#PickerDelete:hover:enabled {{
                border-color: {COLOR_DANGER};
                color: {COLOR_DANGER};
            }}
            QTreeWidget {{
                background: transparent;
                border: none;
                outline: 0;
                font-size: 12px;
            }}
            QTreeWidget::item {{ padding: 5px 4px; border: none; }}
            QTreeWidget::item:hover {{ background: rgba(255, 255, 255, 10); }}
            QTreeWidget::item:selected {{
                background: rgba(0, 255, 136, 22);
            }}
            QHeaderView::section {{
                background: transparent;
                color: {COLOR_TEXT_DIM};
                border: none;
                border-bottom: 1px solid {COLOR_HAIRLINE};
                padding: 4px 4px 6px 4px;
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.6px;
            }}
            QHeaderView::section:hover {{ color: {COLOR_TEXT_MUTED}; }}
            QScrollBar:vertical {{
                background: transparent; width: 8px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {COLOR_HAIRLINE_HOVER};
                border-radius: 4px; min-height: 28px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 10)
        lay.setSpacing(8)

        # ---- Строка поиска + фильтры по источнику --------------------
        top = QHBoxLayout()
        top.setSpacing(8)
        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Поиск: имя, источник, шасси, комплект…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.search.installEventFilter(self)
        top.addWidget(self.search, 1)

        self._filter_buttons: Dict[str, QToolButton] = {}
        self._filter_bar = QHBoxLayout()
        self._filter_bar.setSpacing(4)
        top.addLayout(self._filter_bar)
        lay.addLayout(top)

        # ---- Таблица -------------------------------------------------
        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels([c[0] for c in COLUMNS])
        self.tree.setRootIsDecorated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setAllColumnsShowFocus(True)
        self.tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        # Сортировка включается только когда пользователь сам щёлкнет по
        # заголовку: по умолчанию список идёт в «родном» порядке —
        # серверные наборы, комплекты генератора, локальные модели.
        self.tree.setSortingEnabled(False)
        self.tree.setMouseTracking(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSortIndicatorShown(False)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(self._on_header_clicked)
        for col, (_title, width) in enumerate(COLUMNS):
            if col != COL_NAME:
                self.tree.setColumnWidth(col, width)
        head_item = self.tree.headerItem()
        head_item.setTextAlignment(COL_VOLUME,
                                   int(Qt.AlignmentFlag.AlignRight
                                       | Qt.AlignmentFlag.AlignVCenter))
        for col, tip in (
            (COL_SOURCE, "Откуда набор: TLS-сервер, генератор кузовов "
                         "или файл в assets/models/trucks"),
            (COL_CHASSIS, "Колёсная формула"),
            (COL_VOLUME, "Вместимость кузова: заявленная в конфиге (макс.), "
                         "обмеренная по скану (изм.) или посчитанная по "
                         "габаритам (расч.) — уточнение в строке снизу"),
            (COL_DIMS, "Внутренние габариты кузова: длина × ширина × высота, "
                       "м. Известны там, где рядом лежит .spec.json"),
            (COL_KIT, "Что есть в наборе. Без наполнителя груз не "
                      "сгенерируется — модель только для просмотра"),
        ):
            head_item.setToolTip(col, tip)
        self.tree.setItemDelegate(_ColorDelegate(self.tree))
        self.tree.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        self.tree.installEventFilter(self)
        lay.addWidget(self.tree, 1)

        # ---- Подвал: подробности выделенной строки -------------------
        self.detail = QLabel("")
        self.detail.setWordWrap(False)
        self.detail.setMinimumWidth(120)
        self.detail.setSizePolicy(QSizePolicy.Policy.Ignored,
                                  QSizePolicy.Policy.Preferred)
        self.detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.count = QLabel("")
        self.count.setStyleSheet(
            f"color: {COLOR_TEXT_DIM}; font-size: 10px;"
            f" font-family: {FONT_MONO};")
        # Удалять можно только то, что лежит у нас на диске: комплекты
        # генератора и локальные модели. Для серверных наборов кнопка
        # выключена — файлы не наши, а сообщение об этом висит подсказкой.
        self.btn_delete = QToolButton()
        self.btn_delete.setObjectName("PickerDelete")
        self.btn_delete.setText("Удалить")
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._request_remove_current)

        foot = QHBoxLayout()
        foot.setSpacing(10)
        foot.addWidget(self.detail, 1)
        foot.addWidget(self.btn_delete, 0)
        foot.addWidget(self.count, 0)
        lay.addLayout(foot)

    # -- наполнение ----------------------------------------------------
    def set_rows(self, texts: List[str], keys: List[Any],
                 infos: List[Optional[ModelSetInfo]]) -> None:
        """Перестроить таблицу под текущее содержимое комбо."""
        self._infos = infos
        was_sorting = self.tree.isSortingEnabled()
        self.tree.setSortingEnabled(False)
        self.tree.clear()

        for idx, text in enumerate(texts):
            info = infos[idx] if idx < len(infos) else None
            item = _SortableItem()
            item.setData(0, Qt.ItemDataRole.UserRole, idx)

            if info is None:
                item.setText(COL_NAME, text)
                item.setText(COL_SOURCE, _EM_DASH)
                for col in (COL_CHASSIS, COL_VOLUME, COL_DIMS, COL_KIT):
                    item.setText(col, _EM_DASH)
                self.tree.addTopLevelItem(item)
                continue

            item.setText(COL_NAME, info.name)
            item.setData(COL_NAME, SORT_ROLE, info.name.lower())
            item.setText(COL_SOURCE, info.source_label)
            item.setForeground(COL_SOURCE, QColor(
                SOURCE_COLORS.get(info.source, COLOR_TEXT_MUTED)))
            item.setText(COL_CHASSIS, info.axles or _EM_DASH)
            item.setText(COL_VOLUME, _fmt_volume(info))
            item.setTextAlignment(COL_VOLUME,
                                  int(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter))
            if info.volume is not None:
                item.setData(COL_VOLUME, SORT_ROLE, info.volume)
            item.setText(COL_DIMS, _fmt_dims(info))
            if info.dims:
                item.setData(COL_DIMS, SORT_ROLE, info.length)
            item.setText(COL_KIT, info.kit)
            for col in (COL_CHASSIS, COL_VOLUME, COL_DIMS, COL_KIT):
                item.setForeground(col, QColor(COLOR_TEXT_MUTED))
            if "napolnitel" not in info.parts:
                # Без наполнителя груз не сгенерируется — приглушаем, чтобы
                # это было видно до выбора, а не после запуска.
                item.setForeground(COL_KIT, QColor(COLOR_TEXT_DIM))
            item.setToolTip(COL_NAME, self._tooltip_for(info))
            self.tree.addTopLevelItem(item)

        self.tree.setSortingEnabled(was_sorting)
        self._rebuild_filters(infos)
        self._apply_filter()

    def _on_header_clicked(self, column: int) -> None:
        """Первый клик по заголовку включает сортировку, дальше — Qt сам."""
        if self.tree.isSortingEnabled():
            return
        self.tree.header().setSortIndicatorShown(True)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(column, Qt.SortOrder.AscendingOrder)

    def _rebuild_filters(self,
                         infos: List[Optional[ModelSetInfo]]) -> None:
        """Кнопки-фильтры строятся по фактически найденным источникам."""
        while self._filter_bar.count():
            item = self._filter_bar.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._filter_buttons.clear()

        present: List[str] = []
        for info in infos:
            if info is not None and info.source not in present:
                present.append(info.source)
        if len(present) < 2:
            self._source_filter = ""
            return

        for source in [""] + present:
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setText(FILTER_LABELS.get(source, "Все") if source
                        else "Все")
            btn.setChecked(source == self._source_filter)
            btn.clicked.connect(
                lambda _checked, s=source: self._set_source_filter(s))
            self._filter_bar.addWidget(btn)
            self._filter_buttons[source] = btn

    def _set_source_filter(self, source: str) -> None:
        self._source_filter = source
        for key, btn in self._filter_buttons.items():
            btn.setChecked(key == source)
        self._apply_filter()

    def _tooltip_for(self, info: ModelSetInfo) -> str:
        lines = [info.name, f"ключ: {info.key}"]
        if info.origin:
            lines.append(f"источник: {info.source_label} · {info.origin}")
        if info.volume is not None:
            lines.append(f"объём: {_fmt_volume(info)} м³"
                         f"{' (' + info.volume_kind + ')' if info.volume_kind else ''}")
        if info.dims:
            lines.append(f"внутренние габариты: {_fmt_dims(info)} м")
        lines.append("состав: " + (", ".join(info.parts) or "—"))
        return "\n".join(lines)

    def _detail_for(self, info: Optional[ModelSetInfo]) -> str:
        if info is None:
            return ""
        bits: List[str] = []
        if info.origin:
            bits.append(info.origin)
        if info.chassis:
            bits.append(f"шасси {info.chassis}")
        if info.volume is not None and info.volume_kind:
            bits.append(f"объём {info.volume_kind}")
        if info.ref_rect:
            bits.append("опорный прямоугольник "
                        f"{info.ref_rect[0]:.2f}×{info.ref_rect[1]:.2f} м")
        if info.ground_plane is not None:
            bits.append(f"плоскость {info.ground_plane:g} м")
        if info.has_camera:
            bits.append("пресет камеры")
        if info.file_size:
            size = _fmt_size(info.file_size)
            if info.mtime:
                size += " · " + datetime.fromtimestamp(
                    info.mtime).strftime("%d.%m.%Y")
            bits.append(size)
        if info.path:
            bits.append(os.path.basename(info.path))
        return "  ·  ".join(bits)

    # -- фильтрация / навигация ---------------------------------------
    def _apply_filter(self) -> None:
        query = self.search.text().strip().lower()
        terms = [t for t in query.split() if t]
        shown = 0
        first_visible: Optional[QTreeWidgetItem] = None
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            idx = item.data(0, Qt.ItemDataRole.UserRole)
            info = self._infos[idx] if isinstance(idx, int) and \
                idx < len(self._infos) else None
            haystack = (info.search_text if info is not None
                        else item.text(COL_NAME).lower())
            ok = all(t in haystack for t in terms)
            if ok and self._source_filter:
                ok = info is not None and info.source == self._source_filter
            item.setHidden(not ok)
            if ok:
                shown += 1
                if first_visible is None:
                    first_visible = item
        total = self.tree.topLevelItemCount()
        self.count.setText(f"{shown}/{total}" if shown != total
                           else f"{total}")
        current = self.tree.currentItem()
        if current is None or current.isHidden():
            if first_visible is not None:
                self.tree.setCurrentItem(first_visible)
            else:
                self._set_detail("ничего не найдено")
                self._sync_delete_button(None)

    def _set_detail(self, text: str) -> None:
        """
        Подвал не должен растягивать всплывашку, поэтому текст хранится
        целиком, а на экран идёт обрезанный по фактической ширине ярлыка
        (её знает только layout — то есть уже после resize).
        """
        self._detail_text = text
        self._update_detail()

    def _update_detail(self) -> None:
        text = getattr(self, "_detail_text", "")
        width = max(120, self.detail.width())
        self.detail.setText(QFontMetrics(self.detail.font()).elidedText(
            text, Qt.TextElideMode.ElideMiddle, width))
        self.detail.setToolTip(text)

    def resizeEvent(self, event) -> None:                # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._update_detail()

    def _visible_items(self) -> List[QTreeWidgetItem]:
        return [self.tree.topLevelItem(i)
                for i in range(self.tree.topLevelItemCount())
                if not self.tree.topLevelItem(i).isHidden()]

    def _step(self, delta: int) -> None:
        items = self._visible_items()
        if not items:
            return
        cur = self.tree.currentItem()
        pos = items.index(cur) if cur in items else -1
        pos = max(0, min(len(items) - 1, pos + delta)) if pos >= 0 else 0
        self.tree.setCurrentItem(items[pos])
        self.tree.scrollToItem(items[pos])

    def _on_current_changed(self, current: Optional[QTreeWidgetItem],
                            _prev) -> None:
        if current is None:
            self._sync_delete_button(None)
            return
        info = self._info_of(current)
        self._set_detail(self._detail_for(info))
        self._sync_delete_button(info)

    # -- удаление наборов ----------------------------------------------
    def _info_of(self,
                 item: Optional[QTreeWidgetItem]) -> Optional[ModelSetInfo]:
        """Характеристики строки таблицы (строки хранят индекс в комбо)."""
        if item is None:
            return None
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and idx < len(self._infos):
            return self._infos[idx]
        return None

    def _sync_delete_button(self, info: Optional[ModelSetInfo]) -> None:
        """Кнопка активна только для наборов, файлы которых лежат у нас."""
        deletable = info is not None and can_delete_model_set(info.key)
        self.btn_delete.setEnabled(deletable)
        if info is None:
            self.btn_delete.setToolTip("Выберите набор в списке")
        elif deletable:
            self.btn_delete.setToolTip(
                f"Удалить «{info.name}» с диска (Del).\n"
                "Утилита покажет список файлов и спросит подтверждение.")
        else:
            self.btn_delete.setToolTip(
                f"«{info.name}» — набор с сервера: его файлы лежат не в "
                "проекте, удалить их отсюда нельзя")

    def _request_remove_current(self) -> None:
        self._request_remove(self.tree.currentItem())

    def _request_remove(self, item: Optional[QTreeWidgetItem]) -> None:
        """
        Отдать запрос наверх и закрыться.

        Всплывашка — Qt.Popup: пока она открыта, модальное окно подтверждения
        показать нельзя (первый же клик мимо закроет popup вместе с
        обработчиком). Поэтому список закрывается сразу, а диалог показывает
        уже MainWindow.
        """
        info = self._info_of(item)
        if info is None or not can_delete_model_set(info.key):
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        self.close()
        if isinstance(idx, int):
            self.removeRequested.emit(idx)

    def _on_context_menu(self, pos: QPoint) -> None:
        item = self.tree.itemAt(pos)
        info = self._info_of(item)
        if info is None:
            return
        menu = QMenu(self)
        act_pick = menu.addAction("Выбрать")
        act_del = menu.addAction("Удалить с диска…")
        act_del.setEnabled(can_delete_model_set(info.key))
        chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
        if chosen is act_pick:
            self._commit(item)
        elif chosen is act_del:
            self._request_remove(item)

    def _on_item_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        self._commit(item)

    def _commit(self, item: Optional[QTreeWidgetItem]) -> None:
        if item is None:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if isinstance(idx, int):
            self.picked.emit(idx)
        self.close()

    # -- события -------------------------------------------------------
    def eventFilter(self, obj, event) -> bool:           # noqa: N802 (Qt API)
        """
        Клавиатура живёт в поле поиска: стрелки и Enter перекидываются в
        таблицу, чтобы можно было печатать и выбирать не отрывая рук.
        """
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._step(1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                self._step(10 if key == Qt.Key.Key_PageDown else -10)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._commit(self.tree.currentItem())
                return True
            if key == Qt.Key.Key_Escape:
                self.close()
                return True
            if key == Qt.Key.Key_Delete and obj is self.tree:
                # Только из таблицы: в строке поиска Delete правит текст.
                self._request_remove(self.tree.currentItem())
                return True
            if obj is self.tree and event.text() and event.text().isprintable():
                # Печать в таблице продолжает поиск, а не прыгает по буквам.
                self.search.setFocus(Qt.FocusReason.OtherFocusReason)
                self.search.setText(self.search.text() + event.text())
                return True
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:                  # noqa: N802 (Qt API)
        super().showEvent(event)
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)
        self.search.selectAll()
        cur = self.tree.currentItem()
        if cur is not None:
            self.tree.scrollToItem(cur,
                                   QAbstractItemView.ScrollHint.PositionAtCenter)


class ModelPickerCombo(QWidget):
    """
    Замена QComboBox для выбора набора моделей: кнопка + всплывающая
    таблица. Поддерживает тот набор методов комбо-бокса, которым
    пользуется правая панель, поэтому вызывающий код не меняется.
    """

    currentIndexChanged = pyqtSignal(int)
    #: Пользователь просит удалить набор с диска. Полезная нагрузка — ключ
    #: набора. Сам виджет ничего не удаляет: подтверждение и работу с файлами
    #: берёт на себя MainWindow (см. `_on_model_delete_requested`), потому что
    #: удалённый набор может быть сейчас в сцене.
    deleteRequested = pyqtSignal(object)

    #: Ширина всплывающей таблицы. Панель — 320 px, а сравнивать наборы
    #: удобно только когда все колонки видны разом.
    POPUP_WIDTH = 780
    POPUP_MAX_HEIGHT = 460

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._texts: List[str] = []
        self._keys: List[Any] = []
        self._details: Dict[Any, ModelSetInfo] = {}
        self._index = -1
        self._popup: Optional[_PickerPopup] = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.button = _FieldButton(self)
        self.button.clicked.connect(self._toggle_popup)
        lay.addWidget(self.button)
        self._refresh_button()

    # ------------------------------------------------------------------
    # QComboBox-совместимая часть API
    # ------------------------------------------------------------------
    def addItem(self, text: str, userData: Any = None) -> None:  # noqa: N803
        self._texts.append(str(text))
        self._keys.append(userData)
        if self._index < 0:
            self._index = 0
            self._refresh_button()
        self._sync_popup()

    def clear(self) -> None:
        self._texts.clear()
        self._keys.clear()
        self._index = -1
        self._refresh_button()
        self._sync_popup()

    def count(self) -> int:
        return len(self._texts)

    def itemText(self, index: int) -> str:
        return self._texts[index] if 0 <= index < len(self._texts) else ""

    def itemData(self, index: int) -> Any:
        return self._keys[index] if 0 <= index < len(self._keys) else None

    def currentIndex(self) -> int:
        return self._index

    def currentData(self) -> Any:
        return self.itemData(self._index)

    def currentText(self) -> str:
        return self.itemText(self._index)

    def findData(self, value: Any) -> int:
        for i, key in enumerate(self._keys):
            if key == value:
                return i
        return -1

    def setCurrentIndex(self, index: int) -> None:       # noqa: N802 (Qt API)
        if index == self._index:
            return
        if not (-1 <= index < len(self._texts)):
            return
        self._index = index
        self._refresh_button()
        self.currentIndexChanged.emit(index)

    # ------------------------------------------------------------------
    # Дополнения поверх комбо-бокса
    # ------------------------------------------------------------------
    def set_details(self, infos: List[ModelSetInfo]) -> None:
        """
        Передать характеристики наборов. Ключи, которых нет в списке,
        игнорируются; строки без характеристик показываются как есть —
        это позволяет вызывать set_details до/после addItem.
        """
        self._details = {info.key: info for info in infos}
        self._refresh_button()
        self._sync_popup()

    def info_for(self, key: Any) -> Optional[ModelSetInfo]:
        return self._details.get(key)

    def current_info(self) -> Optional[ModelSetInfo]:
        return self._details.get(self.currentData())

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _row_infos(self) -> List[Optional[ModelSetInfo]]:
        return [self._details.get(key) for key in self._keys]

    def _refresh_button(self) -> None:
        info = self.current_info()
        text = self.currentText() or "— модели не найдены —"
        if info is not None:
            summary_bits = [b for b in (info.axles,
                                        (f"{_fmt_volume(info)} м³"
                                         if info.volume is not None else ""))
                            if b]
            self.button.set_content(
                info.name, " · ".join(summary_bits),
                SOURCE_COLORS.get(info.source, COLOR_TEXT_MUTED))
            self.button.setToolTip(self._tooltip(info))
        else:
            self.button.set_content(text, "", "")
            self.button.setToolTip(text)

    @staticmethod
    def _tooltip(info: ModelSetInfo) -> str:
        lines = [info.name, f"{info.source_label}"
                 + (f" · {info.origin}" if info.origin else ""),
                 f"ключ: {info.key}"]
        if info.volume is not None:
            kind = f" ({info.volume_kind})" if info.volume_kind else ""
            lines.append(f"объём: {_fmt_volume(info)} м³{kind}")
        if info.dims:
            lines.append(f"внутренние габариты: {_fmt_dims(info)} м")
        lines.append(f"комплект: {info.kit}")
        return "\n".join(lines)

    def _sync_popup(self) -> None:
        if self._popup is not None:
            self._popup.set_rows(self._texts, self._keys, self._row_infos())

    def _toggle_popup(self) -> None:
        if self._popup is not None and self._popup.isVisible():
            self._popup.close()
            return
        self._open_popup()

    def _open_popup(self) -> None:
        if not self._texts:
            return
        if self._popup is None:
            self._popup = _PickerPopup(self)
            self._popup.picked.connect(self._on_picked)
            self._popup.removeRequested.connect(self._on_remove_requested)
            self._popup.destroyed.connect(self._on_popup_destroyed)
        popup = self._popup
        popup.set_rows(self._texts, self._keys, self._row_infos())

        # Выделяем текущий набор, чтобы список открывался «на себе».
        for i in range(popup.tree.topLevelItemCount()):
            item = popup.tree.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == self._index:
                popup.tree.setCurrentItem(item)
                break
        popup.search.clear()

        screen = (self.screen() or QApplication.primaryScreen())
        area = screen.availableGeometry() if screen else None
        width = self.POPUP_WIDTH
        rows = max(1, len(self._texts))
        height = min(self.POPUP_MAX_HEIGHT, 150 + rows * 27)
        if area is not None:
            width = min(width, area.width() - 40)
            height = min(height, area.height() - 80)
        popup.resize(width, height)

        # Панель прижата к правому краю окна, поэтому таблица раскрывается
        # влево от поля и вверх, если снизу не хватает места.
        anchor = self.button.mapToGlobal(QPoint(self.button.width(),
                                                self.button.height() + 4))
        x, y = anchor.x() - width, anchor.y()
        if area is not None:
            x = max(area.left() + 8, min(x, area.right() - width - 8))
            if y + height > area.bottom() - 8:
                above = self.button.mapToGlobal(QPoint(0, -4)).y() - height
                y = above if above > area.top() + 8 else max(
                    area.top() + 8, area.bottom() - height - 8)
        popup.move(x, y)
        popup.show()
        self.button.set_open(True)
        popup.installEventFilter(self)

    def _on_popup_destroyed(self, *_args) -> None:
        self._popup = None
        self.button.set_open(False)

    def _on_picked(self, index: int) -> None:
        self.button.set_open(False)
        self.setCurrentIndex(index)

    def _on_remove_requested(self, index: int) -> None:
        self.button.set_open(False)
        key = self.itemData(index)
        if key is not None:
            self.deleteRequested.emit(key)

    def eventFilter(self, obj, event) -> bool:           # noqa: N802 (Qt API)
        if obj is self._popup and event.type() in (QEvent.Type.Close,
                                                   QEvent.Type.Hide):
            self.button.set_open(False)
        return super().eventFilter(obj, event)
