# overlay_widgets.py
# ---------------------------------------------------------------------------
# Floating "Data Information" overlays for the 3D viewport.
#
# Why TOP-LEVEL tool windows?
#   The 3D viewport is a native Panda3D HWND reparented into a QFrame via
#   win32gui.SetParent. On Windows, a native child HWND ALWAYS paints over
#   any Qt-painted content of its parent — so a plain Qt child widget
#   placed "on top" is completely covered by the Panda3D rendering and
#   stays invisible (raise_() doesn't help: Qt z-order doesn't reach into
#   GDI z-order).
#
#   The fix is to make each overlay a tiny top-level frameless `Qt.Tool`
#   window, OWNED by the main window:
#     * Tool window → no taskbar entry, follows the owner's visibility.
#     * Owner = main window → when the main window is minimised / hidden /
#       sent behind another app, the overlay follows it (no "floats above
#       every desktop app" problem).
#     * No WindowStaysOnTopHint — that's exactly what caused the previous
#       overlay to leak above other apps.
#     * Translucent background + click-through, so it reads as a HUD.
#
#   We then track the anchored container's global geometry and reposition
#   the overlay to follow it on resize / move / show / hide events.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint, QEvent, QRect
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGraphicsDropShadowEffect,
)

from src.ui.ui_theme import (
    COLOR_ACCENT, COLOR_TEXT_MUTED, COLOR_TEXT, COLOR_HAIRLINE, apply_theme,
)


class SceneOverlay(QWidget):
    """
    HUD-style overlay that visually sits over the 3D viewport, but is
    actually a top-level frameless tool window owned by the main window.

    Usage:
        ov = SceneOverlay("Camera · Telemetry", anchor="top-left",
                          parent=panda_container)
        ov.set_rows([("Pitch", "-90.0°"), ("Yaw", "0.0°")])
        ov.attach()            # installs event filters + repositions + shows
    """

    ANCHORS = {"top-left", "top-right", "bottom-left", "bottom-right"}

    def __init__(self, heading: str, anchor: str = "top-left",
                 parent: QWidget | None = None, margin: int = 16,
                 width: int = 240):
        assert anchor in self.ANCHORS, f"bad anchor: {anchor}"
        assert parent is not None, "SceneOverlay must have an anchor widget"

        # Top-level frameless tool window, OWNED by the parent's top-level
        # window (NOT by the inner container) — that way z-order and
        # visibility properly follow the main window.
        owner_window = parent.window() or parent
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(owner_window, flags)

        self._anchor = anchor
        self._margin = margin
        self._owner = parent          # widget we anchor against (panda_container)

        # Translucent painting + click-through so the 3D scene stays interactive.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Don't steal focus when shown.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        apply_theme(self)

        # The visual card
        self.card = QFrame(self)
        self.card.setObjectName("Overlay")

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.card.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.card)

        body = QVBoxLayout(self.card)
        body.setContentsMargins(14, 12, 14, 12)
        body.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")
        self.eyebrow = QLabel(heading.upper())
        self.eyebrow.setProperty("role", "eyebrow")
        header.addWidget(self.dot)
        header.addWidget(self.eyebrow)
        header.addStretch(1)
        body.addLayout(header)

        self._rows_holder = QVBoxLayout()
        self._rows_holder.setSpacing(4)
        body.addLayout(self._rows_holder)

        # Keep a handle on the body layout so callers can append extra
        # widgets below the rows (e.g. a daytime slider).
        self._body_layout = body

        self._row_cache: dict[str, QLabel] = {}
        self.setFixedWidth(width)

    # ------------------------------------------------------------------
    def attach_extra(self, widget) -> None:
        """
        Append an arbitrary widget to the bottom of the overlay's card
        body, below the key/value rows. Used to host secondary controls
        like the time-of-day slider or camera-mode buttons.
        """
        if not hasattr(self, "_body_layout") or self._body_layout is None:
            return
        self._body_layout.addWidget(widget)
        # Force the layout system to recompute *now* so sizeHint() picks
        # up the new content for the next _reposition. Without this,
        # extras that are appended after the first show end up clipped
        # below the card's earlier height.
        try:
            self._body_layout.invalidate()
            self._body_layout.activate()
            self.card.adjustSize()
            self.adjustSize()
            self.updateGeometry()
        except Exception:
            pass
        try:
            self._reposition()
        except Exception:
            pass
        # Belt-and-braces: schedule one more reposition for the next
        # event-loop tick so any deferred Qt geometry updates settle
        # before we paint.
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, lambda: (
            self.adjustSize(),
            self._reposition() if hasattr(self, "_reposition") else None,
        ))

    # -- Public API ---------------------------------------------------
    def attach(self) -> None:
        """
        Install event filters on the anchor widget and on its top-level
        window, then reposition + show the overlay.
        """
        owner = self._owner
        if owner is None:
            return

        # Resize / show / hide events fire on the anchor (panda_container).
        owner.installEventFilter(self)

        # Move events typically fire on the top-level window when the user
        # drags the main window across the desktop — those don't reach the
        # inner container, so we listen there too.
        top = owner.window()
        if top is not None and top is not owner:
            top.installEventFilter(self)

        self._reposition()
        self.show()
        self.raise_()

    def set_rows(self, rows: list[tuple[str, str]],
                 columns: int = 1) -> None:
        """
        Fill the card with key/value rows. `columns` > 1 packs several
        pairs onto one physical line — the card then needs far less
        vertical space, which matters for the bottom-left telemetry card
        that also hosts the daytime slider, camera modes, presets and the
        whole dataset/save block.
        """
        self._clear_rows()
        columns = max(1, int(columns))
        if columns == 1:
            for key, value in rows:
                self._rows_holder.addWidget(self._make_row(key, value))
        else:
            for i in range(0, len(rows), columns):
                chunk = rows[i:i + columns]
                line = QWidget()
                lay = QHBoxLayout(line)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(14)
                for key, value in chunk:
                    lay.addWidget(self._make_row(key, value), 1)
                # Pad the last, short line so its cells keep the same
                # width as the full lines above.
                for _ in range(columns - len(chunk)):
                    lay.addStretch(1)
                self._rows_holder.addWidget(line)
        # Re-fit height to contents after the row list changes.
        self.adjustSize()
        self._reposition()

    def update_row(self, key: str, value: str) -> None:
        lbl = self._row_cache.get(key)
        if lbl is not None:
            lbl.setText(value)

    # -- Internals ----------------------------------------------------
    def _clear_rows(self) -> None:
        self._row_cache.clear()
        while self._rows_holder.count():
            item = self._rows_holder.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _make_row(self, key: str, value: str) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)

        k = QLabel(key)
        k.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
            f"letter-spacing: 0.8px; text-transform: uppercase;"
        )

        v = QLabel(value)
        v.setStyleSheet(
            f"color: {COLOR_TEXT}; font-family: "
            f"'Geist Mono','JetBrains Mono',monospace;"
            f"font-size: 12px; font-weight: 500;"
        )
        v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        h.addWidget(k)
        h.addStretch(1)
        h.addWidget(v)

        self._row_cache[key] = v
        return row

    def _reposition(self) -> None:
        owner = self._owner
        if owner is None:
            return

        # Sizes
        pw, ph = owner.width(), owner.height()
        w = self.width()
        h = self.sizeHint().height()
        m = self._margin

        # A bottom-anchored card that grew taller than the viewport would
        # otherwise run off the top edge and lose its first rows. Clamp the
        # height so the card always starts at the top margin and stays
        # fully inside the viewport.
        if self._anchor.startswith("bottom"):
            h = min(h, max(1, ph - 2 * m))

        # Anchor point in `owner` local coordinates.
        if self._anchor == "top-left":
            local = QPoint(m, m)
        elif self._anchor == "top-right":
            local = QPoint(pw - w - m, m)
        elif self._anchor == "bottom-left":
            local = QPoint(m, max(m, ph - h - m))
        else:  # bottom-right
            local = QPoint(pw - w - m, max(m, ph - h - m))

        # Translate to screen-space — we're a top-level window now.
        gp = owner.mapToGlobal(local)
        self.setGeometry(gp.x(), gp.y(), w, h)
        self.raise_()

    def eventFilter(self, obj, event):
        owner = self._owner
        if owner is None:
            return super().eventFilter(obj, event)

        et = event.type()
        top = owner.window()

        # Anything that can shift the anchor's screen position triggers a
        # reposition: container resize, container show, top-level move, or
        # top-level state change (minimise/restore/maximise).
        if et in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
            QEvent.Type.WindowStateChange,
        ):
            self._reposition()

        # When the owning window is hidden / minimised, hide the overlay too.
        if obj is top:
            if et == QEvent.Type.Hide:
                self.hide()
            elif et == QEvent.Type.Show:
                self.show()
                self._reposition()

        return super().eventFilter(obj, event)



# ===========================================================================
# DepthMapOverlay
#   Minimal floating card containing ONLY the live image plus a small
#   toggle button in its top-left corner.  Click cycles main viewport
#   between normal render and depth render, while the widget holds
#   whichever is NOT in the main viewport.
# ===========================================================================

from PyQt6.QtCore import pyqtSignal as _pyqtSignal
from PyQt6.QtWidgets import QPushButton as _QPushButton


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


class DepthMapOverlay(QWidget):
    """Minimal 16:9 card with a single image + a top-left toggle button."""

    ANCHORS = {"top-left", "top-right", "bottom-left", "bottom-right"}

    # Emitted on toggle-button click.
    toggleRequested = _pyqtSignal()

    def __init__(self, parent: QWidget, anchor: str = "top-left",
                 margin: int = 16, width: int = 320,
                 right_inset: int = 0):
        assert parent is not None, "DepthMapOverlay needs an anchor"
        assert anchor in self.ANCHORS, f"bad anchor: {anchor}"

        owner_window = parent.window() or parent
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(owner_window, flags)

        self._owner = parent
        self._margin = margin
        self._anchor = anchor
        self._fixed_w = width
        self._right_inset = right_inset

        self._inner_pad = 8
        inner_w = self._fixed_w - 2 * self._inner_pad
        self._inner_h = int(inner_w * 9 / 16)
        self._card_h = self._inner_h + 2 * self._inner_pad

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # NOT click-through anymore - the toggle button needs to receive
        # mouse events. The card is small (~320x190) so this is fine.
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        apply_theme(self)

        self.card = QFrame(self)
        self.card.setObjectName("Overlay")

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.card.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.card)

        body = QVBoxLayout(self.card)
        body.setContentsMargins(self._inner_pad, self._inner_pad,
                                self._inner_pad, self._inner_pad)
        body.setSpacing(0)

        self.canvas = QLabel(self.card)
        self.canvas.setStyleSheet(
            "background-color: #050505; border-radius: 4px;"
        )
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumSize(inner_w, self._inner_h)
        body.addWidget(self.canvas)

        # Stash so attach_extra can append rows below the canvas.
        self._body_layout = body

        # ---- Toggle button (free-floating child of the card) -------
        # Stays anchored to the top-left corner of the canvas via
        # `move()` in `_position_button`.
        self.toggle_btn = _QPushButton("⇄", self.card)   # ⇄
        self.toggle_btn.setToolTip("Переключить вид: обычный ↔ глубина")
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setFixedSize(26, 26)
        self.toggle_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: rgba(16, 16, 16, 220);"
            f" color: {COLOR_ACCENT};"
            f" border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 4px;"
            "  font-size: 14px;"
            "  font-weight: 600;"
            "  padding: 0;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(40, 40, 40, 230);"
            f" border: 1px solid {COLOR_ACCENT};"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(0, 60, 30, 230);"
            "}"
        )
        self.toggle_btn.clicked.connect(self.toggleRequested.emit)
        self.toggle_btn.raise_()

        self.setFixedWidth(self._fixed_w)
        # Use a minimum height instead of a fixed one so the card can
        # grow when attach_extra() adds a settings strip below.
        self.setMinimumHeight(self._card_h)

        # Initial button placement.
        self._position_button()

    # -- Public API ---------------------------------------------------
    def attach(self) -> None:
        owner = self._owner
        if owner is None:
            return
        owner.installEventFilter(self)
        top = owner.window()
        if top is not None and top is not owner:
            top.installEventFilter(self)
        self._reposition()
        self.show()
        self.raise_()
        self._position_button()

    def attach_extra(self, widget) -> None:
        """Append `widget` below the canvas inside the card body."""
        if not hasattr(self, "_body_layout") or self._body_layout is None:
            return
        self._body_layout.addWidget(widget)
        try:
            self._body_layout.invalidate()
            self._body_layout.activate()
            self.card.adjustSize()
            self.adjustSize()
            self.updateGeometry()
        except Exception:
            pass
        try:
            self._reposition()
        except Exception:
            pass
        from PyQt6.QtCore import QTimer as _QT
        _QT.singleShot(0, lambda: (
            self.adjustSize(),
            self._reposition() if hasattr(self, "_reposition") else None,
        ))

    def set_image(self, qimage) -> None:
        """Push a QImage to the canvas (called from the Qt main thread)."""
        from PyQt6.QtGui import QPixmap
        if qimage is None:
            return
        pm = QPixmap.fromImage(qimage).scaled(
            self.canvas.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(pm)

    def set_toggle_state(self, depth_in_main: bool) -> None:
        """
        Reflect the current toggle state in the button glyph + tooltip.
        depth_in_main=True => widget shows NORMAL render.
        """
        if depth_in_main:
            self.toggle_btn.setText("◉")    # solid disc - "depth in main"
            self.toggle_btn.setToolTip(
                "Главное: карта глубины · виджет: обычный. Клик — поменять."
            )
        else:
            self.toggle_btn.setText("⇄")    # arrows - default
            self.toggle_btn.setToolTip(
                "Главное: обычный · виджет: карта глубины. Клик — поменять."
            )

    # -- Geometry -----------------------------------------------------
    def _position_button(self) -> None:
        """Pin the toggle button at the inner-canvas top-left corner."""
        if not hasattr(self, "toggle_btn"):
            return
        # The canvas sits at (inner_pad, inner_pad) inside the card.
        # Place the button with a 6px inset on top of the canvas.
        x = self._inner_pad + 6
        y = self._inner_pad + 6
        self.toggle_btn.move(x, y)
        self.toggle_btn.raise_()

    def _reposition(self) -> None:
        owner = self._owner
        if owner is None:
            return
        pw, ph = owner.width(), owner.height()
        m = self._margin
        w = self._fixed_w
        # Track real natural height so attach_extra's settings strip
        # is visible (instead of being clipped at _card_h).
        h = max(self._card_h, self.sizeHint().height())
        if self._anchor == "top-left":
            local = QPoint(m, m)
        elif self._anchor == "top-right":
            local = QPoint(pw - w - m - self._right_inset, m)
        elif self._anchor == "bottom-left":
            local = QPoint(m, ph - h - m)
        else:
            local = QPoint(pw - w - m - self._right_inset, ph - h - m)
        gp = owner.mapToGlobal(local)
        self.setGeometry(gp.x(), gp.y(), w, h)
        self.raise_()
        self._position_button()

    def eventFilter(self, obj, event):
        owner = self._owner
        if owner is None:
            return super().eventFilter(obj, event)
        et = event.type()
        top = owner.window()
        if et in (QEvent.Type.Resize, QEvent.Type.Move,
                  QEvent.Type.Show, QEvent.Type.WindowStateChange):
            self._reposition()
        if obj is top:
            if et == QEvent.Type.Hide:
                self.hide()
            elif et == QEvent.Type.Show:
                self.show()
                self._reposition()
        return super().eventFilter(obj, event)
