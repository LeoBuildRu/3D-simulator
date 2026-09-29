# hud.py
# ---------------------------------------------------------------------------
# Строительные блоки интерфейса поверх 3D-вида (в стиле macOS).
#
# Почему плавающие окна, а не дочерние виджеты
# --------------------------------------------
# 3D-вид — нативное окно Panda3D, вставленное в QFrame через SetParent.
# Нативное дочернее окно на Windows всегда рисуется поверх того, что Qt
# нарисовал в родителе, поэтому обычный дочерний виджет «над» сценой просто
# не виден. Каждая панель здесь — отдельное безрамочное окно `Qt.Tool`,
# принадлежащее главному окну: без кнопки на панели задач, прячется и
# сворачивается вместе с ним, но не всплывает над чужими приложениями
# (никакого WindowStaysOnTopHint). Позиция пересчитывается по событиям
# якорного виджета (panda_container) и главного окна.
#
# Состав
#   FloatingPanel    — панель-«материал» с тенью, привязанная к краю/центру
#   Popover          — всплывающая карточка под кнопкой (Qt.Popup)
#   IconButton       — квадратная кнопка-иконка (с бейджем-счётчиком)
#   TileButton       — плитка «иконка + подпись», как в Пункте управления
#   SegmentedControl — сегментированный переключатель с анимацией
#   Switch           — тумблер
#   Keycap           — клавиша для справки по управлению
# ---------------------------------------------------------------------------

from __future__ import annotations

import time
from typing import Iterable

from PyQt6.QtCore import (
    QEasingCurve, QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt,
    QVariantAnimation, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath, QPen,
)
from PyQt6.QtWidgets import (
    QAbstractButton, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout,
    QWidget,
)

from src.ui import icons
from src.ui.ui_theme import (
    COLOR_ACCENT, COLOR_ACCENT_SOFT, COLOR_TEXT, COLOR_TEXT_DIM,
    COLOR_TEXT_MUTED, FONT_MONO, MATERIAL_BORDER_RGBA, MATERIAL_RGBA,
    RADIUS_PANEL, apply_hud_theme,
)

#: Поле под тень вокруг карточки внутри окна. Ноль: тень, скругление углов и
#: тонкую кромку рисует сама Windows 11 (DWM) — отключить их у Tool-окна
#: нельзя, а поверх своей тени системная давала грязный светлый прямоугольник.
#: Поэтому окно панели равно карточке, а форма — системная.
SHADOW = 0

#: Радиус скругления, который DWM даёт окнам (DWMWCP_ROUND).
NATIVE_RADIUS = 8


def _qcolor(hex_color: str, alpha: int | None = None) -> QColor:
    c = QColor(hex_color)
    if alpha is not None:
        c.setAlpha(int(alpha))
    return c


def _dwm_set(widget: QWidget, attr: int, value: int) -> None:
    try:
        import ctypes
        v = ctypes.c_uint(value)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(int(widget.winId())), attr, ctypes.byref(v),
            ctypes.sizeof(v))
    except Exception:
        pass


def native_frame(widget: QWidget) -> None:
    """Скруглённые углы Windows 11 для плавающей панели."""
    _dwm_set(widget, 33, 2)                 # CORNER_PREFERENCE = ROUND


def disable_dwm_frame(widget: QWidget) -> None:
    """Без скругления и рамки — для полноэкранной подложки-снимка."""
    _dwm_set(widget, 33, 1)                 # DONOTROUND
    _dwm_set(widget, 34, 0xFFFFFFFE)        # BORDER_COLOR = COLOR_NONE


def paint_material(p: QPainter, card: QRectF, radius: float,
                   shadow: int = SHADOW, fill=MATERIAL_RGBA) -> None:
    """Заливка «материалом». Тень и кромку добавляет DWM."""
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(*fill))
    p.drawRoundedRect(card, radius, radius)


# ===========================================================================
# FloatingPanel
# ===========================================================================
class FloatingPanel(QWidget):
    """
    Плавающая панель над 3D-видом.

    anchor: top-left | top-right | bottom-left | bottom-right |
            top-center | bottom-center | right-stretch
    Для «center» панель центрируется в свободной части вида между
    `inset_left` и `inset_right` (чтобы не залезать под инспектор).
    """

    ANCHORS = {"top-left", "top-right", "bottom-left", "bottom-right",
               "top-center", "bottom-center", "right-stretch"}

    def __init__(self, owner: QWidget, anchor: str = "top-left",
                 margin: int = 16, offset: tuple[int, int] = (0, 0),
                 radius: int = NATIVE_RADIUS,
                 padding: tuple[int, int, int, int] = (12, 12, 12, 12),
                 width: int | None = None, click_through: bool = False,
                 layout: str = "v"):
        assert anchor in self.ANCHORS, f"bad anchor: {anchor}"
        assert owner is not None, "FloatingPanel needs an anchor widget"
        top = owner.window() or owner
        super().__init__(top, Qt.WindowType.Tool
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        # `_owner` читает MainWindow.fade_overlays — по нему он находит все
        # панели сцены.
        self._owner = owner
        self._anchor = anchor
        self._margin = margin
        self._offset = offset
        self._radius = radius
        self._fixed_width = width
        self.inset_left = 0
        self.inset_right = 0
        #: Пользователь сам скрыл панель — не показывать её обратно при
        #: восстановлении главного окна.
        self.user_hidden = False

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        if click_through:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents,
                              True)
        apply_hud_theme(self)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW, SHADOW, SHADOW, SHADOW)
        outer.setSpacing(0)
        self.body = QWidget(self)
        self.body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer.addWidget(self.body)
        self.body_layout = (QVBoxLayout(self.body) if layout == "v"
                            else QHBoxLayout(self.body))
        self.body_layout.setContentsMargins(*padding)
        self.body_layout.setSpacing(8)

        if width is not None:
            self.setFixedWidth(width + 2 * SHADOW)

    # -- API -----------------------------------------------------------------
    def attach(self) -> None:
        owner = self._owner
        owner.installEventFilter(self)
        top = owner.window()
        if top is not None and top is not owner:
            top.installEventFilter(self)
        self._reposition()
        native_frame(self)
        self.show()
        self.raise_()

    def relayout(self) -> None:
        """Пересчитать размер по содержимому и переставить."""
        self.body_layout.invalidate()
        self.body_layout.activate()
        self.adjustSize()
        self._reposition()

    def set_user_visible(self, visible: bool) -> None:
        self.user_hidden = not visible
        if visible:
            self._reposition()
            self.show()
            self.raise_()
        else:
            self.hide()

    def card_rect(self) -> QRect:
        """Геометрия видимой карточки в глобальных координатах."""
        g = self.geometry()
        return g.adjusted(SHADOW, SHADOW, -SHADOW, -SHADOW)

    # -- геометрия -----------------------------------------------------------
    def _card_size(self, pw: int, ph: int) -> QSize:
        hint = self.sizeHint()
        cw = (self._fixed_width if self._fixed_width is not None
              else hint.width() - 2 * SHADOW)
        if self._anchor == "right-stretch":
            ch = max(120, ph - 2 * self._margin)
        else:
            ch = hint.height() - 2 * SHADOW
            ch = min(ch, max(40, ph - 2 * self._margin))
        return QSize(max(10, cw), max(10, ch))

    def _reposition(self) -> None:
        owner = self._owner
        if owner is None:
            return
        pw, ph = owner.width(), owner.height()
        size = self._card_size(pw, ph)
        cw, ch = size.width(), size.height()
        m = self._margin
        ox, oy = self._offset
        a = self._anchor
        if a == "top-left":
            x, y = m + ox, m + oy
        elif a == "top-right":
            x, y = pw - cw - m - ox, m + oy
        elif a == "bottom-left":
            x, y = m + ox, ph - ch - m - oy
        elif a == "bottom-right":
            x, y = pw - cw - m - ox, ph - ch - m - oy
        elif a in ("top-center", "bottom-center"):
            left = self.inset_left
            right = pw - self.inset_right
            if right - left < cw:                  # не влезает — центр вида
                left, right = 0, pw
            x = left + (right - left - cw) // 2 + ox
            y = (m + oy) if a == "top-center" else (ph - ch - m - oy)
        else:  # right-stretch
            x, y = pw - cw - m, m
        gp = owner.mapToGlobal(QPoint(int(x) - SHADOW, int(y) - SHADOW))
        self.setGeometry(gp.x(), gp.y(), cw + 2 * SHADOW, ch + 2 * SHADOW)

    # -- отрисовка -----------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        card = QRectF(self.rect()).adjusted(SHADOW, SHADOW, -SHADOW, -SHADOW)
        paint_material(p, card, self._radius)
        p.end()

    # -- слежение за якорем --------------------------------------------------
    def eventFilter(self, obj, event):
        owner = self._owner
        if owner is None:
            return super().eventFilter(obj, event)
        et = event.type()
        if et in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show,
                  QEvent.Type.WindowStateChange):
            self._reposition()
        if obj is owner.window():
            if et == QEvent.Type.Hide:
                self.hide()
            elif et == QEvent.Type.Show and not self.user_hidden:
                self.show()
                self._reposition()
        return super().eventFilter(obj, event)


# ===========================================================================
# Popover
# ===========================================================================
class Popover(QWidget):
    """
    Всплывающая карточка (Qt.Popup): закрывается кликом мимо и по Esc.
    `toggle_for(button)` открывает её под кнопкой или закрывает.
    """

    closed = pyqtSignal()

    def __init__(self, parent: QWidget, width: int | None = None,
                 padding: tuple[int, int, int, int] = (14, 12, 14, 14),
                 radius: int = NATIVE_RADIUS):
        super().__init__(parent.window() if parent else None,
                         Qt.WindowType.Popup
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        apply_hud_theme(self)
        self._radius = radius
        self._hidden_at = 0.0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW, SHADOW, SHADOW, SHADOW)
        body = QWidget(self)
        body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer.addWidget(body)
        self.body_layout = QVBoxLayout(body)
        self.body_layout.setContentsMargins(*padding)
        self.body_layout.setSpacing(10)
        if width is not None:
            self.setFixedWidth(width + 2 * SHADOW)

    def show_for(self, anchor: QWidget, placement: str = "below") -> None:
        self.adjustSize()
        w, h = self.width(), self.height()
        r = QRect(anchor.mapToGlobal(QPoint(0, 0)), anchor.size())
        gap = 6
        if placement == "above":
            x = r.center().x() - w // 2
            y = r.top() - h + SHADOW - gap
        elif placement == "left":
            x = r.left() - w + SHADOW - gap
            y = r.top() - SHADOW
        elif placement == "right":
            x = r.right() - SHADOW + gap
            y = r.top() - SHADOW
        else:
            x = r.center().x() - w // 2
            y = r.bottom() - SHADOW + gap
        screen = (QGuiApplication.screenAt(r.center())
                  or QGuiApplication.primaryScreen())
        if screen is not None:
            av = screen.availableGeometry()
            x = max(av.left() - SHADOW, min(x, av.right() - w + SHADOW))
            y = max(av.top() - SHADOW, min(y, av.bottom() - h + SHADOW))
        self.move(int(x), int(y))
        native_frame(self)
        self.show()

    def toggle_for(self, anchor: QWidget, placement: str = "below") -> None:
        # Клик по кнопке, пока поповер открыт, сначала закрывает Popup (клик
        # «мимо»), а затем долетает до кнопки — без этой паузы он тут же
        # открылся бы снова.
        if self.isVisible():
            self.hide()
            return
        if time.monotonic() - self._hidden_at < 0.25:
            return
        self.show_for(anchor, placement)

    def hideEvent(self, event) -> None:
        self._hidden_at = time.monotonic()
        super().hideEvent(event)
        self.closed.emit()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        card = QRectF(self.rect()).adjusted(SHADOW, SHADOW, -SHADOW, -SHADOW)
        paint_material(p, card, self._radius, fill=(40, 40, 43, 246))
        p.end()


# ===========================================================================
# Кнопки
# ===========================================================================
class _HoverMixin:
    def enterEvent(self, e):                     # noqa: N802 (Qt API)
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):                     # noqa: N802 (Qt API)
        self._hover = False
        self.update()
        super().leaveEvent(e)


class IconButton(_HoverMixin, QAbstractButton):
    """Кнопка-иконка. `filled=True` — с постоянной подложкой."""

    def __init__(self, icon_name: str, tooltip: str = "", size: int = 30,
                 icon_size: int = 18, checkable: bool = False,
                 color: str | None = None, filled: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._icon = icon_name
        self._icon_size = icon_size
        self._color = color or COLOR_TEXT
        self._filled = filled
        self._hover = False
        self._badge = ""
        self._radius = 8
        self._on_dark = False
        self.setCheckable(checkable)
        self.setToolTip(tooltip)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggled.connect(lambda _c: self.update())

    def set_icon(self, name: str) -> None:
        self._icon = name
        self.update()

    def set_badge(self, text: str) -> None:
        self._badge = str(text or "")
        self.update()

    def set_on_image(self, on: bool = True) -> None:
        """Кнопка лежит поверх картинки — всегда с тёмной подложкой."""
        self._on_dark = on
        self.update()

    def sizeHint(self) -> QSize:
        return self.size()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        enabled = self.isEnabled()
        checked = self.isChecked()
        down = self.isDown()

        bg = None
        if checked:
            bg = _qcolor(COLOR_ACCENT, 70 if not down else 50)
        elif down:
            bg = QColor(255, 255, 255, 20)
        elif self._hover and enabled:
            bg = QColor(255, 255, 255, 34)
        elif self._filled:
            bg = QColor(255, 255, 255, 22)
        if self._on_dark and not checked:
            base = QColor(20, 20, 22, 190 if not self._hover else 225)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(base)
            p.drawRoundedRect(r, self._radius, self._radius)
        if bg is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawRoundedRect(r, self._radius, self._radius)

        color = (COLOR_TEXT_DIM if not enabled
                 else COLOR_ACCENT_SOFT if checked else self._color)
        pm = icons.pixmap(self._icon, color, self._icon_size)
        s = self._icon_size
        p.drawPixmap(QRectF((r.width() - s) / 2, (r.height() - s) / 2, s, s),
                     pm, QRectF(pm.rect()))

        if self._badge:
            f = QFont(self.font())
            f.setPixelSize(10)
            f.setWeight(QFont.Weight.DemiBold)
            fm = QFontMetrics(f)
            bw = max(15, fm.horizontalAdvance(self._badge) + 8)
            br = QRectF(r.width() - bw + 2, -1, bw, 15)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_qcolor(COLOR_ACCENT))
            p.drawRoundedRect(br, 7.5, 7.5)
            p.setPen(QColor("#FFFFFF"))
            p.setFont(f)
            p.drawText(br, int(Qt.AlignmentFlag.AlignCenter), self._badge)
        p.end()


class TileButton(_HoverMixin, QAbstractButton):
    """Плитка «иконка над подписью» — для частых действий без длинных надписей."""

    def __init__(self, icon_name: str, text: str, tooltip: str = "",
                 checkable: bool = False, height: int = 58,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._icon = icon_name
        self._hover = False
        self._badge = ""
        self.setText(text)
        self.setCheckable(checkable)
        self.setToolTip(tooltip)
        self.setFixedHeight(height)
        self.setMinimumWidth(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.toggled.connect(lambda _c: self.update())

    def set_badge(self, text: str) -> None:
        self._badge = str(text or "")
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        enabled = self.isEnabled()
        checked = self.isChecked()
        if checked:
            bg = _qcolor(COLOR_ACCENT, 230 if not self.isDown() else 190)
        elif self.isDown():
            bg = QColor(255, 255, 255, 16)
        elif self._hover and enabled:
            bg = QColor(255, 255, 255, 34)
        else:
            bg = QColor(255, 255, 255, 20 if enabled else 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(r, 10, 10)

        fg = ("#FFFFFF" if checked else
              COLOR_TEXT if enabled else COLOR_TEXT_DIM)
        s = 20
        pm = icons.pixmap(self._icon, fg, s)
        p.drawPixmap(QRectF((r.width() - s) / 2, 9, s, s), pm,
                     QRectF(pm.rect()))
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.Medium)
        p.setFont(f)
        p.setPen(QColor(fg if checked or not enabled else COLOR_TEXT_MUTED))
        fm = QFontMetrics(f)
        text = fm.elidedText(self.text(), Qt.TextElideMode.ElideRight,
                             int(r.width()) - 8)
        p.drawText(QRectF(0, r.height() - 22, r.width(), 16),
                   int(Qt.AlignmentFlag.AlignCenter), text)

        if self._badge:
            f.setPixelSize(10)
            f.setWeight(QFont.Weight.DemiBold)
            fm = QFontMetrics(f)
            bw = max(16, fm.horizontalAdvance(self._badge) + 8)
            br = QRectF(r.width() / 2 + 6, 4, bw, 15)
            p.setBrush(QColor("#FFFFFF") if checked else _qcolor(COLOR_ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(br, 7.5, 7.5)
            p.setPen(_qcolor(COLOR_ACCENT) if checked else QColor("#FFFFFF"))
            p.setFont(f)
            p.drawText(br, int(Qt.AlignmentFlag.AlignCenter), self._badge)
        p.end()


class SegmentedControl(_HoverMixin, QWidget):
    """
    Сегментированный переключатель: items = [(key, icon, tooltip, text), ...].
    Выбранный сегмент подсвечивается «таблеткой», которая плавно едет.
    """

    changed = pyqtSignal(str)

    def __init__(self, items: Iterable[tuple], height: int = 30,
                 segment_width: int | None = None, icon_size: int = 16,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._items = [tuple(it) + (None,) * (4 - len(it)) for it in items]
        self._keys = [it[0] for it in self._items]
        self._index = 0
        self._pill_x = 0.0
        self._hover = False
        self._hover_idx = -1
        self._badges: set[str] = set()
        self._icon_size = icon_size
        self._seg_w = segment_width
        self.setFixedHeight(height)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred if segment_width
                           else QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)

    # -- API -----------------------------------------------------------------
    def current(self) -> str:
        return self._keys[self._index]

    def set_current(self, key: str, animate: bool = True) -> None:
        if key not in self._keys:
            return
        idx = self._keys.index(key)
        if idx == self._index and not self._anim.state():
            self._pill_x = self._seg_left(idx)
            self.update()
            return
        self._index = idx
        target = self._seg_left(idx)
        if animate and self.isVisible():
            self._anim.stop()
            self._anim.setStartValue(float(self._pill_x))
            self._anim.setEndValue(float(target))
            self._anim.start()
        else:
            self._pill_x = target
            self.update()

    def set_badge(self, key: str, on: bool) -> None:
        (self._badges.add if on else self._badges.discard)(key)
        self.update()

    # -- геометрия -----------------------------------------------------------
    def _seg_width(self) -> float:
        n = max(1, len(self._items))
        return (self.width() - 4) / n

    def _seg_left(self, idx: int) -> float:
        return 2 + idx * self._seg_width()

    def sizeHint(self) -> QSize:
        n = len(self._items)
        if self._seg_w:
            return QSize(n * self._seg_w + 4, self.height())
        fm = QFontMetrics(self.font())
        w = 0
        for _k, ic, _tip, text in self._items:
            ww = 16
            if ic:
                ww += self._icon_size
            if text:
                ww += fm.horizontalAdvance(text) + (6 if ic else 0)
            w = max(w, ww)
        return QSize(n * max(w, 36) + 4, self.height())

    def minimumSizeHint(self) -> QSize:
        return QSize(len(self._items) * 30, self.height())

    def resizeEvent(self, e) -> None:
        self._pill_x = self._seg_left(self._index)
        super().resizeEvent(e)

    def _on_anim(self, v) -> None:
        self._pill_x = float(v)
        self.update()

    def _idx_at(self, x: float) -> int:
        i = int((x - 2) // max(1.0, self._seg_width()))
        return max(0, min(len(self._items) - 1, i))

    # -- события -------------------------------------------------------------
    def mouseMoveEvent(self, e) -> None:
        i = self._idx_at(e.position().x())
        if i != self._hover_idx:
            self._hover_idx = i
            tip = self._items[i][2] or ""
            self.setToolTip(tip)
            self.update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e) -> None:
        self._hover_idx = -1
        super().leaveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            i = self._idx_at(e.position().x())
            if i != self._index:
                self.set_current(self._keys[i])
                self.changed.emit(self._keys[i])
        super().mouseReleaseEvent(e)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        rad = min(7.0, r.height() / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(118, 118, 128, 56))
        p.drawRoundedRect(r, rad, rad)

        sw = self._seg_width()
        if 0 <= self._hover_idx != self._index:
            p.setBrush(QColor(255, 255, 255, 16))
            p.drawRoundedRect(QRectF(self._seg_left(self._hover_idx), 2,
                                     sw, r.height() - 4), rad - 2, rad - 2)
        pill = QRectF(self._pill_x, 2, sw, r.height() - 4)
        p.setBrush(QColor(0, 0, 0, 50))
        p.drawRoundedRect(pill.adjusted(0, 1, 0, 1), rad - 2, rad - 2)
        p.setBrush(QColor(99, 99, 102))
        p.drawRoundedRect(pill, rad - 2, rad - 2)

        fm = QFontMetrics(self.font())
        for i, (key, ic, _tip, text) in enumerate(self._items):
            seg = QRectF(self._seg_left(i), 0, sw, r.height())
            selected = i == self._index
            fg = (COLOR_TEXT if selected else
                  COLOR_TEXT_MUTED if self.isEnabled() else COLOR_TEXT_DIM)
            s = self._icon_size
            tw = fm.horizontalAdvance(text) if text else 0
            total = (s if ic else 0) + (tw + (6 if ic else 0) if text else 0)
            x = seg.center().x() - total / 2
            if ic:
                pm = icons.pixmap(ic, fg, s)
                p.drawPixmap(QRectF(x, (r.height() - s) / 2, s, s), pm,
                             QRectF(pm.rect()))
                x += s + 6
            if text:
                f = QFont(self.font())
                f.setWeight(QFont.Weight.DemiBold if selected
                            else QFont.Weight.Medium)
                p.setFont(f)
                p.setPen(QColor(fg))
                p.drawText(QRectF(x, 0, tw + 4, r.height()),
                           int(Qt.AlignmentFlag.AlignVCenter
                               | Qt.AlignmentFlag.AlignLeft), text)
            if key in self._badges:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(_qcolor(COLOR_ACCENT))
                bx = seg.center().x() + (s / 2 if ic else tw / 2) + 3
                p.drawEllipse(QPointF(bx, r.height() / 2 - s / 2 + 1), 3.2, 3.2)
        p.end()


class Switch(QAbstractButton):
    """Тумблер в стиле macOS; цвет «включено» — акцент."""

    def __init__(self, checked: bool = False, tooltip: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setToolTip(tooltip)
        self.setFixedSize(38, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.toggled.connect(self._animate)

    def _on_anim(self, v) -> None:
        self._pos = float(v)
        self.update()

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def sizeHint(self) -> QSize:
        return self.size()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        off = QColor(120, 120, 128, 90)
        on = _qcolor(COLOR_ACCENT)
        t = self._pos
        track = QColor(
            int(off.red() + (on.red() - off.red()) * t),
            int(off.green() + (on.green() - off.green()) * t),
            int(off.blue() + (on.blue() - off.blue()) * t),
            int(off.alpha() + (255 - off.alpha()) * t),
        )
        if not self.isEnabled():
            track.setAlpha(60)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.left() + 2 + (r.width() - d - 4) * t
        p.setBrush(QColor(0, 0, 0, 60))
        p.drawEllipse(QRectF(x, r.top() + 2.6, d, d))
        p.setBrush(QColor("#FFFFFF") if self.isEnabled() else QColor("#A0A0A5"))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))
        p.end()


class Chip(QWidget):
    """Капсула «иконка + текст» (метки, выходы датасета)."""

    def __init__(self, icon_name: str, text: str, tooltip: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._icon = icon_name
        self._text = text
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.Medium)
        self._font = f
        w = QFontMetrics(f).horizontalAdvance(text) + 36
        self.setFixedSize(w, 22)
        if tooltip:
            self.setToolTip(tooltip)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 24))
        p.drawRoundedRect(r, 11, 11)
        pm = icons.pixmap(self._icon, COLOR_TEXT, 13)
        p.drawPixmap(QRectF(9, 4.5, 13, 13), pm, QRectF(pm.rect()))
        p.setFont(self._font)
        p.setPen(QColor(COLOR_TEXT))
        p.drawText(QRectF(26, 0, r.width() - 30, r.height()),
                   int(Qt.AlignmentFlag.AlignVCenter), self._text)
        p.end()


class Keycap(QLabel):
    """Клавиша для справки по управлению."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(24)
        self.setStyleSheet(
            "QLabel {"
            "  background: rgba(255,255,255,26);"
            f" color: {COLOR_TEXT};"
            "  border-radius: 5px;"
            "  border-bottom: 2px solid rgba(0,0,0,90);"
            "  padding: 2px 7px;"
            "  font-size: 11px; font-weight: 600;"
            "}")


# ===========================================================================
# Мелкие помощники вёрстки
# ===========================================================================
def vline(height: int = 20) -> QFrame:
    f = QFrame()
    f.setFixedSize(1, height)
    f.setStyleSheet("background: rgba(255,255,255,30); border: none;")
    return f


def hline() -> QFrame:
    f = QFrame()
    f.setProperty("role", "hairline")
    f.setFrameShape(QFrame.Shape.NoFrame)
    return f


def label(text: str = "", role: str | None = None, mono: bool = False,
          size: int | None = None, color: str | None = None,
          weight: int | None = None) -> QLabel:
    lbl = QLabel(text)
    if role:
        lbl.setProperty("role", role)
    css = []
    if mono:
        css.append(f"font-family: {FONT_MONO};")
    if size:
        css.append(f"font-size: {size}px;")
    if color:
        css.append(f"color: {color};")
    if weight:
        css.append(f"font-weight: {weight};")
    if css:
        lbl.setStyleSheet("QLabel { " + " ".join(css) + " }")
    return lbl


def icon_label(name: str, color: str = COLOR_TEXT_MUTED,
               size: int = 16, tooltip: str = "") -> QLabel:
    lbl = QLabel()
    lbl.setPixmap(icons.pixmap(name, color, size))
    lbl.setFixedSize(size + 2, size + 2)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if tooltip:
        lbl.setToolTip(tooltip)
    return lbl


class Card(QFrame):
    """Сгруппированная карточка (inset grouped) со строками и разделителями."""

    def __init__(self, parent: QWidget | None = None,
                 padding: tuple[int, int, int, int] = (12, 10, 12, 10),
                 spacing: int = 8):
        super().__init__(parent)
        self.setObjectName("Surface")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(*padding)
        self.lay.setSpacing(spacing)

    def add(self, w) -> None:
        if isinstance(w, QWidget):
            self.lay.addWidget(w)
        else:
            self.lay.addLayout(w)
