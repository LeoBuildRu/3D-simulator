# overlay_widgets.py
# ---------------------------------------------------------------------------
# Оверлеи над 3D-видом:
#
#   TelemetryHUD           — тонкая строка внизу слева: углы, FOV, позиция.
#                            Сквозная для мыши: читается, но не мешает WASD.
#   DepthMapOverlay        — «картинка в картинке» вверху слева: живая карта
#                            глубины (или обычный кадр, если их поменяли
#                            местами). Настройки диапазона спрятаны под
#                            кнопку-иконку в поповер.
#   CameraReferenceOverlay — полноэкранный полупрозрачный снимок стенда для
#                            ручного совмещения камеры.
#
# Все они — отдельные окна Qt.Tool, принадлежащие главному окну (почему —
# см. src/ui/hud.py).
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, QEvent, QRect, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from src.ui import icons
from src.ui.hud import (
    FloatingPanel, IconButton, Popover, disable_dwm_frame, label, vline,
)
from src.ui.ui_theme import (
    COLOR_TEXT, COLOR_TEXT_DIM, COLOR_TEXT_MUTED, FONT_MONO,
)


# ===========================================================================
# TelemetryHUD
# ===========================================================================
class TelemetryHUD(FloatingPanel):
    """
    Строка телеметрии камеры. У каждого числа — подпись словами под ним:

        -30.7°   24.7°   0.0°  │  101°   │   4.2    -3.7    5.2
        наклон  поворот  крен  │  обзор  │  X, м   Y, м    Z, м

    API как у прежней карточки: `update_row(key, value)` с ключами
    PITCH / YAW / ROLL / FOV / X / Y / Z.
    """

    # ключ, подпись, подсказка, единица, ширина ячейки
    _GROUPS = (
        (("PITCH", "наклон", "Наклон камеры вверх / вниз (pitch)", "°", 50),
         ("YAW", "поворот", "Поворот камеры влево / вправо (yaw)", "°", 50),
         ("ROLL", "крен", "Крен — наклон горизонта (roll)", "°", 44)),
        (("FOV", "обзор", "Угол обзора камеры (FOV)", "°", 44),),
        (("X", "X, м", "Позиция камеры по X", "", 50),
         ("Y", "Y, м", "Позиция камеры по Y", "", 50),
         ("Z", "Z, м", "Позиция камеры по Z", "", 50)),
    )

    def __init__(self, owner: QWidget, margin: int = 16):
        super().__init__(owner, anchor="bottom-left", margin=margin,
                         padding=(12, 6, 12, 6),
                         click_through=True, layout="h")
        lay = self.body_layout
        lay.setSpacing(4)
        self._values: dict[str, QLabel] = {}
        self._units: dict[str, str] = {}
        for gi, group in enumerate(self._GROUPS):
            if gi:
                lay.addSpacing(6)
                lay.addWidget(vline(24), 0, Qt.AlignmentFlag.AlignVCenter)
                lay.addSpacing(6)
            for key, caption, tip, unit, width in group:
                cell = QWidget()
                cell.setFixedWidth(width)
                cell.setToolTip(tip)
                col = QVBoxLayout(cell)
                col.setContentsMargins(0, 0, 0, 0)
                col.setSpacing(0)
                v = label("0.0", mono=True, size=12, color=COLOR_TEXT)
                v.setAlignment(Qt.AlignmentFlag.AlignCenter)
                c = label(caption, size=10, color=COLOR_TEXT_MUTED)
                c.setAlignment(Qt.AlignmentFlag.AlignCenter)
                col.addWidget(v)
                col.addWidget(c)
                lay.addWidget(cell, 0, Qt.AlignmentFlag.AlignVCenter)
                self._values[key] = v
                self._units[key] = unit

    def update_row(self, key: str, value: str) -> None:
        lbl = self._values.get(key)
        if lbl is None:
            return
        text = str(value).strip()
        if key == "FOV":
            try:
                text = f"{float(text):.0f}"
            except ValueError:
                pass
        text += self._units.get(key, "")
        if lbl.text() != text:
            lbl.setText(text)


# ===========================================================================
# DepthMapOverlay — «картинка в картинке»
# ===========================================================================
class _PiPCanvas(QWidget):
    """Кадр со скруглёнными углами (QLabel не умеет обрезать pixmap)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pm: QPixmap | None = None
        self._chip_icon = "depth"
        self._chip_text = "Глубина"

    def set_pixmap(self, pm: QPixmap | None) -> None:
        self._pm = pm
        self.update()

    def set_chip(self, icon_name: str, text: str) -> None:
        self._chip_icon, self._chip_text = icon_name, text
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 5, 5)
        p.setClipPath(clip)
        p.fillRect(r, QColor(8, 8, 10))
        if self._pm is not None and not self._pm.isNull():
            p.drawPixmap(r, self._pm, QRectF(self._pm.rect()))
        else:
            pm = icons.pixmap("depth", COLOR_TEXT_DIM, 28)
            p.drawPixmap(QRectF(r.center().x() - 14, r.center().y() - 14,
                                28, 28), pm, QRectF(pm.rect()))
        p.setClipping(False)

        # Чип «что показано» в левом нижнем углу.
        f = QFont(self.font())
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(self._chip_text)
        chip = QRectF(8, r.height() - 30, tw + 34, 22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(20, 20, 22, 200))
        p.drawRoundedRect(chip, 11, 11)
        ic = icons.pixmap(self._chip_icon, COLOR_TEXT, 13)
        p.drawPixmap(QRectF(chip.left() + 8, chip.top() + 4.5, 13, 13), ic,
                     QRectF(ic.rect()))
        p.setPen(QColor(COLOR_TEXT))
        p.drawText(QRectF(chip.left() + 25, chip.top(), tw + 4, chip.height()),
                   int(Qt.AlignmentFlag.AlignVCenter), self._chip_text)
        p.end()


class DepthMapOverlay(FloatingPanel):
    """
    Превью 16:9 вверху слева. Кнопки поверх кадра:
      ⇄ — поменять местами главный вид и превью (сцена ↔ глубина);
      ⚙ — поповер с диапазоном глубины (виджеты добавляет attach_extra).
    """

    toggleRequested = pyqtSignal()

    def __init__(self, parent: QWidget, anchor: str = "top-left",
                 margin: int = 16, width: int = 300, right_inset: int = 0):
        super().__init__(parent, anchor=anchor, margin=margin,
                         padding=(4, 4, 4, 4), width=width)
        inner_w = width - 8
        self._inner_h = int(inner_w * 9 / 16)
        self.canvas = _PiPCanvas(self.body)
        self.canvas.setFixedSize(inner_w, self._inner_h)
        self.body_layout.addWidget(self.canvas)

        self.toggle_btn = IconButton("swap", "Поменять местами: сцена ↔ глубина",
                                     size=28, icon_size=15, parent=self.canvas)
        self.toggle_btn.set_on_image(True)
        self.toggle_btn.move(8, 8)
        self.toggle_btn.clicked.connect(self.toggleRequested.emit)

        self.settings_btn = IconButton("sliders", "Диапазон глубины",
                                       size=28, icon_size=15,
                                       parent=self.canvas)
        self.settings_btn.set_on_image(True)
        self.settings_btn.move(inner_w - 36, 8)

        self.settings_popover = Popover(self, width=260)
        head = label("Диапазон глубины", role="headline")
        self.settings_popover.body_layout.addWidget(head)
        self.settings_btn.clicked.connect(
            lambda: self.settings_popover.toggle_for(self.settings_btn,
                                                     "right"))

    # -- API (как у прежней версии) -----------------------------------------
    def attach_extra(self, widget) -> None:
        """Настройки глубины живут в поповере под кнопкой-шестерёнкой."""
        self.settings_popover.body_layout.addWidget(widget)

    def set_image(self, qimage) -> None:
        if qimage is None:
            return
        pm = QPixmap.fromImage(qimage)
        self.canvas.set_pixmap(pm)

    def set_toggle_state(self, depth_in_main: bool) -> None:
        """depth_in_main=True — в превью обычный кадр, в главном виде глубина."""
        if depth_in_main:
            self.canvas.set_chip("camera", "Сцена")
        else:
            self.canvas.set_chip("depth", "Глубина")


# ===========================================================================
# CameraReferenceOverlay
# ===========================================================================
class CameraReferenceOverlay(QWidget):
    """
    Full-viewport, click-through, translucent reference-image layer.

    Used to manually line the live camera up with a captured snapshot:
    the colour frame of a stand snapshot is shown semi-transparently over
    the 3D viewport so the user can fly the camera until the rendered
    scene matches the photo.

    Like the HUD overlays it is a top-level frameless `Qt.Tool` window
    owned by the main window. Crucially it is CLICK-THROUGH
    (WA_TransparentForMouseEvents) so WASD / RMB-look still reach the
    embedded Panda HWND underneath. The interactive controls (opacity /
    show-hide) live on the right panel, which is re-raised above this
    layer by the main window.
    """

    def __init__(self, parent: QWidget, margin: int = 0):
        assert parent is not None, "CameraReferenceOverlay needs an anchor"

        owner_window = parent.window() or parent
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(owner_window, flags)

        self._owner = parent
        self._margin = margin
        self._src = None            # original QPixmap (unscaled)

        # Translucent + click-through so the 3D scene stays interactive.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # The photo is painted directly in paintEvent (no child QLabel /
        # layout) so the window NEVER grows to fit an oversized pixmap and
        # the image is always clipped to the window bounds — it can't spill
        # onto the desktop / other apps.
        self.set_opacity(0.5)

    # -- Public API ---------------------------------------------------
    def attach(self) -> None:
        """Install event filters so the layer tracks the viewport; stays
        hidden until `show_overlay` is called."""
        owner = self._owner
        if owner is None:
            return
        owner.installEventFilter(self)
        top = owner.window()
        if top is not None and top is not owner:
            top.installEventFilter(self)

    def set_image(self, path: str) -> None:
        """Load the colour frame to display (no-op if it can't be read)."""
        from PyQt6.QtGui import QPixmap
        pm = QPixmap(path) if path else QPixmap()
        self._src = None if pm.isNull() else pm
        self.update()

    def set_opacity(self, value: float) -> None:
        """0..1 — how solid the reference image is over the scene
        (0 = fully transparent / invisible)."""
        try:
            v = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        self.setWindowOpacity(v)

    def show_overlay(self) -> None:
        self._reposition()
        self.show()
        # WA_TransparentForMouseEvents alone is NOT enough for a top-level
        # window on Windows — the OS still delivers clicks here (swallowing
        # them and stealing focus from the embedded Panda HWND, which kills
        # WASD/RMB-look). Force real OS-level pass-through via WS_EX_TRANSPARENT.
        self._apply_native_click_through()
        disable_dwm_frame(self)
        self.raise_()
        # Panda3D рендерится в DirectX/OpenGL native child HWND внутри
        # panda_container, и Qt.Tool top-level окна не всегда выходят
        # поверх него — особенно после перерисовки кадра. Принудительно
        # ставим overlay в HWND_TOPMOST: фрейм-окно всегда выше Panda HWND.
        self._force_topmost(True)

    def hide_overlay(self) -> None:
        # Снимаем topmost при скрытии, чтобы окно не висело поверх
        # системных диалогов / messageboxes, когда оно невидимо.
        self._force_topmost(False)
        self.hide()

    # -- Internals ----------------------------------------------------
    def _apply_native_click_through(self) -> None:
        """On Windows, OR WS_EX_TRANSPARENT (+ WS_EX_LAYERED) into the
        native window's extended style so mouse input falls through to the
        3D viewport beneath. No-op on non-Windows / if win32 is missing."""
        try:
            import win32gui
            import win32con
        except Exception:
            return
        try:
            hwnd = int(self.winId())
            ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            new_ex = ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            if new_ex != ex:
                win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, new_ex)
        except Exception as exc:
            print(f"[ReferenceOverlay] click-through setup failed: {exc}")

    def _force_topmost(self, on: bool) -> None:
        """SetWindowPos(HWND_TOPMOST | HWND_NOTOPMOST) — поднимает overlay
        поверх Panda DX/GL native child HWND. SWP_NOMOVE|SWP_NOSIZE —
        чтобы не двигать/ресайзить окно; SWP_NOACTIVATE — чтобы не
        перехватывать фокус у Panda (важно для WASD/RMB-look)."""
        try:
            import win32gui
            import win32con
        except Exception:
            return
        try:
            hwnd = int(self.winId())
            target = win32con.HWND_TOPMOST if on else win32con.HWND_NOTOPMOST
            flags = (win32con.SWP_NOMOVE
                     | win32con.SWP_NOSIZE
                     | win32con.SWP_NOACTIVATE)
            win32gui.SetWindowPos(hwnd, target, 0, 0, 0, 0, flags)
        except Exception as exc:
            print(f"[ReferenceOverlay] force_topmost({on}) failed: {exc}")

    def paintEvent(self, event):
        # Draw the photo scaled to the FULL window WIDTH, centred vertically,
        # clipped to the window. The live camera pins its HORIZONTAL FOV and
        # lets the vertical follow the window aspect (set_fov(single) +
        # set_aspect_ratio on resize), so the rendered scene scales by window
        # width — scaling the photo by width keeps it matched at any window
        # size without the user touching the camera. Overflow top/bottom is
        # clipped exactly like the camera crops vertically.
        if self._src is None:
            return
        from PyQt6.QtGui import QPainter
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return
        sw = self._src.width()
        sh = self._src.height()
        if sw <= 0 or sh <= 0:
            return
        draw_w = w
        draw_h = int(round(w * sh / sw))
        x = 0
        y = (h - draw_h) // 2          # centre vertically (camera principal pt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.drawPixmap(QRect(x, y, draw_w, draw_h), self._src)
        p.end()

    def _reposition(self) -> None:
        owner = self._owner
        if owner is None:
            return
        pw, ph = owner.width(), owner.height()
        m = self._margin
        gp = owner.mapToGlobal(QPoint(m, m))
        x, y = gp.x(), gp.y()
        w = max(1, pw - 2 * m)
        h = max(1, ph - 2 * m)
        # Clamp to the owner's top-level frame so the layer can never spill
        # beyond the app window onto the desktop / other applications.
        top = owner.window()
        if top is not None:
            tg = top.frameGeometry()
            right = min(x + w, tg.x() + tg.width())
            bottom = min(y + h, tg.y() + tg.height())
            x = max(x, tg.x())
            y = max(y, tg.y())
            w = max(1, right - x)
            h = max(1, bottom - y)
        self.setGeometry(x, y, w, h)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def eventFilter(self, obj, event):
        owner = self._owner
        if owner is None:
            return super().eventFilter(obj, event)
        et = event.type()
        top = owner.window()
        if et in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
            QEvent.Type.WindowStateChange,
        ):
            if self.isVisible():
                self._reposition()
        if obj is top:
            if et == QEvent.Type.Hide:
                self.hide()
        return super().eventFilter(obj, event)
