# right_panel.py
# ---------------------------------------------------------------------------
# Stub right-side control panel for the Toner simulator.
#
# Now built as an OVERLAY WIDGET, exactly like the top-left "Camera ·
# Telemetry" and bottom-left "Controls" HUDs (see overlay_widgets.py):
#
#   * Top-level frameless `Qt.Tool` window OWNED by the main window
#     (so z-order and visibility follow the owner — no taskbar entry,
#     no "floats above every desktop app" leakage).
#   * Translucent background; the visible surface is an inner card with
#     `objectName="Overlay"` (the same QSS rule the small overlays use)
#     so it gets the same dark-translucent fill, hairline border, radius,
#     and drop-shadow as the other HUDs.
#   * Anchored to the RIGHT EDGE of the embedded Panda3D viewport and
#     repositioned on every resize / move / show / hide / state-change
#     event, exactly like SceneOverlay.
#
# Why a top-level tool window (and not a child widget):
#   The 3D viewport is a native Panda3D HWND embedded inside `panda_container`.
#   On Windows, a native child HWND ALWAYS paints over Qt-painted content
#   of its parent — so a plain Qt child widget placed "on top" is
#   completely covered by the Panda3D rendering and stays invisible.
#   This is the same problem the small overlays already solved.
#
# Why NOT WA_TransparentForMouseEvents (unlike the small overlays):
#   This panel is INTERACTIVE — combo boxes, list, buttons all need
#   clicks. We deliberately let mouse events through to it; the small
#   readout HUDs are click-through because they have no controls.
#
# Stub purpose:
#   Same four sections as before — just a UI placeholder, no backend:
#     1. Model set         — combo box picking a curated set of meshes
#     2. Texture set       — combo box picking a PBR material set
#     3. Reconstructions   — list of 2D→3D reconstruction jobs / outputs
#     4. Details           — key/value readout for the currently-selected
#                            reconstruction
#
#   Public signals (silent for now — connect them when a real backend
#   comes online):
#       modelSetChanged(str)
#       textureSetChanged(str)
#       reconstructionSelected(str)
#       applyClicked()
#
# Wiring into MainWindow (mirrors SceneOverlay):
#   from right_panel import RightPanel
#   self.right_panel = RightPanel(parent=self.panda_container)
#   self.right_panel.attach()
# ---------------------------------------------------------------------------

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, QSize, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QBrush, QPixmap, QIcon, QFontMetrics,
    QPainterPath,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QListWidget, QListWidgetItem, QGroupBox, QPushButton, QFrame,
    QScrollArea, QSizePolicy, QGraphicsDropShadowEffect, QDialog,
    QGridLayout, QDoubleSpinBox, QMenu, QApplication, QSlider, QDial,
    QToolButton,
)

from src.ui.ui_theme import (
    apply_theme, COLOR_ACCENT, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_TEXT_DIM,
    COLOR_HAIRLINE, COLOR_HAIRLINE_HOVER, COLOR_WARN, FONT_MONO,
)
from src.ui.panel_data import (
    load_model_sets, load_texture_sets, get_default_texture_set_key,
    load_reconstructions, Reconstruction, PROJECT_ROOT, HEIGHT_EXAMPLES_DIR,
    get_model_set_config, get_texture_set_config, download_server_image,
    SERVER_IMAGE_CACHE_DIR, RECON_PAGE_SIZE,
)
from src.core import graphics_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hline() -> QFrame:
    """A 1px hairline separator using the theme's `[role="hairline"]` style."""
    f = QFrame()
    f.setProperty("role", "hairline")
    f.setFrameShape(QFrame.Shape.NoFrame)
    return f


def _make_chip(text: str, role: str = "chip-live") -> QLabel:
    """Pill-style status chip; `role` is one of chip-live / chip-idle / chip-err."""
    lbl = QLabel(text)
    lbl.setProperty("role", role)
    return lbl


def _format_short_dt(rec: Reconstruction) -> str:
    """
    Compact two-line-friendly timestamp ("23 Apr · 15:32"). Falls back
    to the raw `time` string if `datetime` couldn't be parsed.
    """
    from datetime import datetime as _dt
    if rec.datetime and rec.datetime != _dt.min:
        return rec.datetime.strftime("%d %b · %H:%M")
    return rec.time or "—"


def _resolve_image_path(rec: Reconstruction) -> str | None:
    """
    Resolve `rec.img_file` into an absolute path suitable for QPixmap,
    WITHOUT touching the network.

    LOCAL entries:   look under `height_examples/`, then PROJECT_ROOT.
    SERVER entries:  look in the server-image cache directory; if the
                     image was already downloaded for a previous open,
                     reuse it. Otherwise return None and let the caller
                     trigger `download_server_image()` explicitly.
    """
    # Stand snapshots carry an explicit absolute colour-frame path.
    color_path = (getattr(rec, "color_path", "") or "").strip()
    if color_path and os.path.exists(color_path):
        return color_path

    img = (rec.img_file or "").strip()
    if not img:
        return None

    # Already absolute (or absolute-ish) — trust it.
    if os.path.isabs(img):
        return img if os.path.exists(img) else None

    # Local entries: probe the height_examples directory.
    if rec.is_local:
        candidate = os.path.join(HEIGHT_EXAMPLES_DIR, img)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(PROJECT_ROOT, img)
        if os.path.exists(candidate):
            return candidate
        return None

    # Server entries: cache hit?
    cached = os.path.join(SERVER_IMAGE_CACHE_DIR, img)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return cached
    return None


def _resolve_or_fetch_image_path(rec: Reconstruction) -> str | None:
    """
    Same as `_resolve_image_path`, but for SERVER entries it ALSO
    triggers a synchronous download via TLS_client when no cache hit
    exists. Returns the local path on success, or None on any failure.
    """
    p = _resolve_image_path(rec)
    if p is not None:
        return p
    if rec.is_local:
        return None
    return download_server_image(rec.img_file or "")


# ---------------------------------------------------------------------------
# Iconography (programmatic, single-file, theme-aware)
# ---------------------------------------------------------------------------
# Drawing icons in code keeps the project free of binary asset dependencies
# and lets us re-tint them with the active theme color (accent / muted).
# All icons render onto a 1:1 transparent QPixmap at @2x for sharpness.
# ---------------------------------------------------------------------------

def _make_dtype_icon(data_type: str, size: int = 18) -> QPixmap:
    """
    Render a small data-type icon (left of each list row).

    height → three ascending bars (heightmap profile, accent color)
    ply    → wireframe cube (point cloud surrogate, muted color)
    other  → empty hairline-bordered square (warn color)
    """
    s = size
    scale = 2  # @2x supersampling for crisp edges
    pm = QPixmap(s * scale, s * scale)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(scale, scale)

    if data_type == "height":
        # Three short bars rising left-to-right, like a height profile.
        col = QColor(COLOR_ACCENT)
        p.setBrush(QBrush(col))
        p.setPen(Qt.PenStyle.NoPen)
        # bar geometry in the s×s box, with 2px outer padding.
        pad = 3
        gap = 2
        bar_w = (s - 2 * pad - 2 * gap) / 3
        heights = [s * 0.30, s * 0.55, s * 0.80]
        for i, h in enumerate(heights):
            x = pad + i * (bar_w + gap)
            y = s - pad - h
            p.drawRoundedRect(QRectF(x, y, bar_w, h), 1.2, 1.2)
    elif data_type == "ply":
        # Stylized iso wireframe cube — 6 visible edges.
        col = QColor(COLOR_TEXT_MUTED)
        pen = QPen(col, 1.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Iso projection of a unit cube, projected into the s×s box.
        cx, cy = s / 2, s / 2
        r = s * 0.34  # half-size
        # Six points of the iso silhouette (top, mid-right, bot-right,
        # bottom, mid-left, top-left), then internal Y-junction.
        top   = (cx,         cy - r)
        tr    = (cx + r,     cy - r * 0.5)
        br    = (cx + r,     cy + r * 0.5)
        bot   = (cx,         cy + r)
        bl    = (cx - r,     cy + r * 0.5)
        tl    = (cx - r,     cy - r * 0.5)
        center = (cx, cy)
        path = QPainterPath()
        path.moveTo(*top)
        path.lineTo(*tr)
        path.lineTo(*br)
        path.lineTo(*bot)
        path.lineTo(*bl)
        path.lineTo(*tl)
        path.closeSubpath()
        p.drawPath(path)
        # Y-junction inner edges.
        p.drawLine(int(top[0]),    int(top[1]),    int(center[0]), int(center[1]))
        p.drawLine(int(tr[0]),     int(tr[1]),     int(center[0]), int(center[1]))
        p.drawLine(int(tl[0]),     int(tl[1]),     int(center[0]), int(center[1]))
    elif data_type == "stand":
        # Camera glyph — body + lens — for a captured reference snapshot.
        col = QColor(COLOR_ACCENT)
        pen = QPen(col, 1.3)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pad = 3
        body = QRectF(pad, pad + s * 0.12,
                      s - 2 * pad, s - 2 * pad - s * 0.12)
        p.drawRoundedRect(body, 2, 2)
        # Viewfinder bump on top-left.
        p.drawRoundedRect(
            QRectF(pad + s * 0.12, pad - s * 0.02, s * 0.28, s * 0.16),
            1, 1,
        )
        # Lens.
        p.drawEllipse(QPointF(s / 2, s / 2 + s * 0.06), s * 0.18, s * 0.18)
    else:
        col = QColor(COLOR_WARN)
        pen = QPen(col, 1.2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        pad = 3
        p.drawRoundedRect(QRectF(pad, pad, s - 2 * pad, s - 2 * pad), 2, 2)

    p.end()
    pm.setDevicePixelRatio(scale)
    return pm


def _make_view_icon(size: int = 14, color: str = COLOR_TEXT_MUTED) -> QIcon:
    """
    Magnifying-glass icon (circle + handle + tiny "+" inside) for the
    per-row preview button.  Reads instantly as "look at this".
    """
    s = size
    scale = 2
    pm = QPixmap(s * scale, s * scale)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(scale, scale)

    pen = QPen(QColor(color), 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)

    # Glass body: small circle in upper-left of the icon area.
    cx = s * 0.42
    cy = s * 0.42
    r  = s * 0.30
    p.drawEllipse(QPointF(cx, cy), r, r)

    # Plus inside the glass.
    inner = r * 0.55
    p.drawLine(QPointF(cx - inner, cy), QPointF(cx + inner, cy))
    p.drawLine(QPointF(cx, cy - inner), QPointF(cx, cy + inner))

    # Handle: short diagonal stroke from the SE rim of the circle.
    import math
    angle = math.radians(45)
    sx = cx + r * math.cos(angle)
    sy = cy + r * math.sin(angle)
    ex = cx + (r + s * 0.25) * math.cos(angle)
    ey = cy + (r + s * 0.25) * math.sin(angle)
    p.setPen(QPen(QColor(color), 1.8, Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(sx, sy), QPointF(ex, ey))

    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def _make_copy_icon(size: int = 14, color: str = COLOR_TEXT_MUTED) -> QIcon:
    """Two overlapping rounded rectangles — the classic 'copy' glyph."""
    s = size
    scale = 2
    pm = QPixmap(s * scale, s * scale)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(scale, scale)

    pen = QPen(QColor(color), 1.3)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Back sheet (top-right).
    back = QRectF(s * 0.32, s * 0.16, s * 0.50, s * 0.55)
    p.drawRoundedRect(back, 1.5, 1.5)
    # Front sheet (bottom-left), slightly larger to overlap.
    front = QRectF(s * 0.16, s * 0.30, s * 0.50, s * 0.55)
    # Clear the part of the back sheet that the front sheet covers so the
    # two outlines don't visually merge into a single shape.
    p.save()
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    p.fillRect(front, Qt.GlobalColor.transparent)
    p.restore()
    p.drawRoundedRect(front, 1.5, 1.5)

    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def _make_close_icon(size: int = 14, color: str = COLOR_TEXT_MUTED) -> QIcon:
    """Small × glyph for the photo overlay's close button."""
    s = size
    scale = 2
    pm = QPixmap(s * scale, s * scale)
    pm.fill(Qt.GlobalColor.transparent)

    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.scale(scale, scale)

    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    pad = 3
    p.drawLine(pad, pad, s - pad, s - pad)
    p.drawLine(s - pad, pad, pad, s - pad)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def _elide(text: str, font, max_w: int,
           mode=Qt.TextElideMode.ElideRight) -> str:
    """Return `text` truncated with an ellipsis to fit within `max_w` px."""
    if not text:
        return ""
    fm = QFontMetrics(font)
    return fm.elidedText(text, mode, max_w)


# ---------------------------------------------------------------------------
# Reconstruction list row — Digital Engineering 2026 styling
# ---------------------------------------------------------------------------
# Layout (renders inside the QListWidget item via setItemWidget):
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │ [▙]  А123ВС777                       HEIGHT             [⤢]  │
#   │      FAW J6 8x4 · 23 Apr · 15:32                              │
#   └──────────────────────────────────────────────────────────────┘
#
#   * Left icon  : data-type pictogram (heightmap bars / wire cube).
#   * Top line   : car_number (mono, prominent) + tiny HEIGHT/PLY tag.
#   * Bottom line: "model · timestamp" (muted; both elided if too long).
#   * Right side : "open" view-photo button — emits viewRequested.
#
# Long text is elided with an ellipsis so nothing ever overflows the
# panel's fixed inner width. Background is transparent — the
# QListWidget's item:hover / item:selected rule paints behind us.
# ---------------------------------------------------------------------------
class ReconRowWidget(QWidget):
    """Custom 2-line row widget for one Reconstruction record."""

    # Inner width budget — rows render inside the right-panel scroll area
    # whose effective inner width is roughly PANEL_WIDTH (320) minus card
    # padding (40), list padding (~16), so we cap at ~250 px.
    ROW_FIXED_HEIGHT = 42

    # Emitted when the user clicks the per-row "open" button.
    # Signal lives on the widget; the panel re-fans it as a higher-level
    # `viewRequested(int)` carrying the recon index.
    viewClicked = pyqtSignal()

    def __init__(self, rec: Reconstruction, max_text_width: int = 200,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._rec = rec
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("background: transparent;")
        self.setMinimumHeight(self.ROW_FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(10)
        # All three children (icon, text col, view button) should be
        # vertically centred regardless of their natural height.
        outer.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # ---- Left: data-type icon ----------------------------------
        self.icon = QLabel()
        self.icon.setPixmap(_make_dtype_icon(rec.data_type or "", size=18))
        self.icon.setFixedSize(20, 20)
        self.icon.setStyleSheet("background: transparent;")
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.icon, 0,
                        Qt.AlignmentFlag.AlignVCenter)

        # ---- Middle: two-line text block ----------------------------
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)

        # ----- Top line: car_number (left) + type tag (right).
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        self.car = QLabel()
        self.car.setStyleSheet(
            f"color: {COLOR_TEXT}; font-family: {FONT_MONO};"
            f"font-size: 12px; font-weight: 600;"
            f"background: transparent;"
        )
        self.car.setText(_elide(
            rec.car_number or "—", self.car.font(), max_text_width - 70
        ))

        type_text = (rec.data_type or "—").upper()
        # Color-code the tag like the icon: accent for height, muted for ply.
        tag_color = (
            COLOR_ACCENT if rec.data_type in ("height", "stand")
            else COLOR_TEXT_MUTED if rec.data_type == "ply"
            else COLOR_WARN
        )
        self.tag = QLabel(type_text)
        self.tag.setStyleSheet(
            f"color: {tag_color}; font-family: {FONT_MONO};"
            f"font-size: 9px; font-weight: 600; letter-spacing: 1.2px;"
            f"background: transparent;"
        )
        self.tag.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        top_row.addWidget(self.car, 1)
        top_row.addWidget(self.tag, 0)
        text_col.addLayout(top_row)

        # ----- Bottom line: "model · timestamp" (single label, elided).
        meta_text = " · ".join(filter(None, [
            (rec.model or "").strip(),
            _format_short_dt(rec),
        ])) or "—"

        self.meta = QLabel()
        self.meta.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;"
            f"background: transparent;"
        )
        self.meta.setText(_elide(
            meta_text, self.meta.font(), max_text_width
        ))
        text_col.addWidget(self.meta)

        outer.addLayout(text_col, 1)

        # ---- Right: "view photo" icon button ------------------------
        self.btn_view = QPushButton(self)
        self.btn_view.setProperty("variant", "icon")
        self.btn_view.setIcon(_make_view_icon(14, COLOR_TEXT_MUTED))
        self.btn_view.setIconSize(QSize(14, 14))
        self.btn_view.setFixedSize(24, 24)
        self.btn_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_view.setToolTip("Просмотр фото и подробной информации")
        # Make it visually lighter than the standard icon button so it
        # blends into the row but reveals on hover.
        self.btn_view.setStyleSheet(
            f"QPushButton {{ background: transparent;"
            f"  border: 1px solid transparent; border-radius: 6px;"
            f"  padding: 0; min-width: 24px; max-width: 24px;"
            f"  min-height: 24px; max-height: 24px; }}"
            f"QPushButton:hover {{ background-color: rgba(255,255,255,10);"
            f"  border-color: {COLOR_HAIRLINE}; }}"
            f"QPushButton:pressed {{ background-color: rgba(0,255,136,18);"
            f"  border-color: {COLOR_ACCENT}; }}"
        )
        # `clicked(bool)` → 0-arg `viewClicked()`. Use a lambda to drop the
        # extra arg explicitly (avoids any PyQt arg-count edge-cases).
        self.btn_view.clicked.connect(lambda _checked=False: self.viewClicked.emit())
        outer.addWidget(self.btn_view, 0, Qt.AlignmentFlag.AlignVCenter)


# ---------------------------------------------------------------------------
# Photo viewer overlay — fullscreen modal with image + metadata
# ---------------------------------------------------------------------------
class RecordPhotoOverlay(QDialog):
    """
    Full-screen modal photo viewer.

    Visual contract matches the rest of the simulator HUD:
        * Translucent dark backdrop dimming the main window
        * Single `QFrame#Overlay` card sized to the WHOLE window minus
          a 16-px margin on every side (same as SceneOverlay /
          DepthMapOverlay).
        * Compact header strip: accent dot + small-caps eyebrow + title
          + close button.
        * The image fills the bulk of the card, scaled to fit.
        * A short "info strip" at the bottom carries CAR / MODEL / TYPE
          / TARGET / TIME / FILE inline (no two-column grid).

    Click on the dim backdrop or press Esc to dismiss.
    """

    OUTER_MARGIN = 16   # same as the other overlays

    def __init__(self, rec: Reconstruction, parent: QWidget | None = None):
        super().__init__(parent)
        self._rec = rec

        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setModal(True)
        apply_theme(self)

        # ---- Backdrop fills the whole parent window -----------------
        self._backdrop = QFrame(self)
        self._backdrop.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180);"
        )
        self._backdrop.lower()

        # ---- Card: spans the whole window minus OUTER_MARGIN --------
        self.card = QFrame(self)
        self.card.setObjectName("Overlay")
        # Slightly larger radius than the SceneOverlay default so this
        # full-screen card visually matches the right-panel cards.
        self.card.setStyleSheet(
            "QFrame#Overlay {"
            "  background-color: rgba(16, 16, 16, 230);"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 10px;"
            "}"
        )

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 200))
        self.card.setGraphicsEffect(shadow)

        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(16, 14, 16, 14)
        card_lay.setSpacing(10)

        # Header strip.
        card_lay.addLayout(self._build_header())

        # Image takes everything else (1 weight).
        card_lay.addWidget(self._build_image_preview(), 1)

        # Compact horizontal info strip at the bottom.
        card_lay.addLayout(self._build_info_strip())

        # Outer fills the dialog with the card pinned to window minus
        # 16 px on each side.
        outer = QGridLayout(self)
        outer.setContentsMargins(
            self.OUTER_MARGIN, self.OUTER_MARGIN,
            self.OUTER_MARGIN, self.OUTER_MARGIN,
        )
        outer.addWidget(self._backdrop, 0, 0, 1, 1)
        outer.addWidget(self.card, 0, 0, 1, 1)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        """
        Compact header strip identical in voice to the other overlays:
        accent dot + small-caps eyebrow + monospace title + × button.
        """
        rec = self._rec
        h = QHBoxLayout()
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(10)

        # Accent dot (same glyph used by every other HUD card).
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")

        # Eyebrow: "RECON · PLY"
        eyebrow = QLabel(f"ЗАПИСЬ · {(rec.data_type or '—').upper()}")
        eyebrow.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
            f" font-weight: 600; letter-spacing: 1.2px;"
        )

        # Title: car number or filename, monospace, accent-toned for emphasis.
        title_text = rec.car_number or rec.name or "—"
        title = QLabel(title_text)
        title.setStyleSheet(
            f"color: {COLOR_TEXT}; font-family: {FONT_MONO};"
            f"font-size: 13px; font-weight: 600;"
        )

        # × close button.
        btn_x = QPushButton()
        btn_x.setIcon(_make_close_icon(14, COLOR_TEXT_MUTED))
        btn_x.setIconSize(QSize(14, 14))
        btn_x.setFixedSize(26, 26)
        btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_x.setToolTip("Закрыть")
        btn_x.setStyleSheet(
            "QPushButton {"
            "  background: transparent;"
            "  border: 1px solid transparent; border-radius: 6px;"
            "  padding: 0; min-width: 26px; max-width: 26px;"
            "  min-height: 26px; max-height: 26px;"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(255,255,255,12);"
            f"  border-color: {COLOR_HAIRLINE};"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(255,51,85,22);"
            "}"
        )
        btn_x.clicked.connect(self.close)

        h.addWidget(dot, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(eyebrow, 0, Qt.AlignmentFlag.AlignVCenter)
        # Vertical hairline spacer between eyebrow and title.
        sep = QFrame()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet(f"background-color: {COLOR_HAIRLINE};")
        h.addWidget(sep, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
        h.addStretch(1)
        h.addWidget(btn_x, 0, Qt.AlignmentFlag.AlignVCenter)
        return h

    def _build_image_preview(self) -> QWidget:
        """
        Bare image canvas - no dark frame, no inner border. The photo
        sits directly on the card's surface.
        """
        rec = self._rec
        frame = QFrame()
        frame.setStyleSheet("background: transparent; border: none;")
        frame.setMinimumHeight(280)

        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)

        canvas = QLabel()
        canvas.setStyleSheet("background: transparent; border: none;")
        canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas.setMinimumHeight(260)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)

        path = _resolve_or_fetch_image_path(rec)
        if path:
            pm = QPixmap(path)
            if pm.isNull():
                canvas.setText("Файл изображения повреждён")
                canvas.setStyleSheet(
                    f"background: transparent; border: none;"
                    f"color: {COLOR_WARN}; font-family: {FONT_MONO};"
                    f"font-size: 11px;"
                )
            else:
                self._original_pm = pm
                canvas.setPixmap(self._scale_pixmap(pm, 800, 480))
                # Re-scale on resize.
                self._image_canvas = canvas

                def _on_resize(_e, c=canvas, p=pm):
                    c.setPixmap(self._scale_pixmap(
                        p, c.width() - 8, c.height() - 8
                    ))

                canvas.resizeEvent = _on_resize  # type: ignore[assignment]
        else:
            placeholder = (
                "Не удалось загрузить изображение\n"
                "(сервер недоступен или файл отсутствует)"
                if not rec.is_local
                else "Файл изображения не найден локально"
            )
            canvas.setText(placeholder)
            canvas.setStyleSheet(
                f"background: transparent; border: none;"
                f"color: {COLOR_TEXT_MUTED}; font-family: {FONT_MONO};"
                f"font-size: 11px; padding: 24px;"
            )

        lay.addWidget(canvas)
        return frame

    @staticmethod
    def _scale_pixmap(pm: QPixmap, w: int, h: int) -> QPixmap:
        if w <= 0 or h <= 0:
            return pm
        return pm.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _build_info_strip(self) -> QHBoxLayout:
        """
        Single horizontal strip carrying every metadata pair as a tiny
        "KEY  value" cell separated by hairline dots.  Replaces the
        old 2-column grid that ate vertical space.
        """
        rec = self._rec
        target_str = (
            f"{rec.target_volume:.2f} m³"
            if rec.target_volume is not None
            else "—"
        )
        time_str = _format_short_dt(rec)
        # FILE can be very long; keep just the trailing 24 chars for
        # the strip - the full thing is in the row tooltip.
        file_short = rec.name or "—"
        if len(file_short) > 28:
            file_short = "..." + file_short[-24:]

        cells: list[tuple[str, str, str | None]] = [
            ("А/Н",    rec.car_number or "—",            None),
            ("МОДЕЛЬ", rec.model or "—",                 None),
            ("ТИП",    (rec.data_type or "—").upper(),   COLOR_ACCENT),
            ("НАПОЛНИТЕЛЬ", rec.filler or "—",           None),
            ("ЦЕЛЬ",   target_str,                       None),
            ("ВРЕМЯ",  time_str,                         None),
            ("ФАЙЛ",   file_short,                       None),
        ]

        h = QHBoxLayout()
        h.setContentsMargins(2, 4, 2, 0)
        h.setSpacing(0)

        for i, (label, value, hi) in enumerate(cells):
            cell = QHBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(8)

            k = QLabel(label)
            k.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
                f" font-weight: 600; letter-spacing: 1.2px;"
            )
            v = QLabel(value)
            v.setToolTip(value)
            v.setStyleSheet(
                f"color: {hi or COLOR_TEXT}; font-family: {FONT_MONO};"
                f"font-size: 11.5px; font-weight: 500;"
            )
            cell.addWidget(k)
            cell.addWidget(v)
            wrapper = QWidget()
            wrapper.setLayout(cell)
            h.addWidget(wrapper)

            if i < len(cells) - 1:
                # Hairline · between cells.
                dot = QLabel("·")
                dot.setStyleSheet(
                    f"color: {COLOR_HAIRLINE}; padding: 0 12px;"
                )
                h.addWidget(dot)

        h.addStretch(1)
        return h

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def showEvent(self, event):
        # Re-cover the parent window on every show — handles re-use.
        parent = self.parentWidget()
        if parent is not None:
            top = parent.window()
            if top is not None:
                self.setGeometry(top.geometry())
        super().showEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        # Click on the dim backdrop dismisses the dialog. Clicks on the
        # card itself bubble up here only if their target had no own
        # mousePressEvent — which is the case for QFrame, QLabel.
        # Filter by checking the global pos against card geometry.
        gp = event.globalPosition().toPoint()
        if not self.card.geometry().contains(self.mapFromGlobal(gp)):
            self.close()
            return
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Right-side control panel — overlay edition
# ---------------------------------------------------------------------------
class RightPanel(QWidget):
    """
    HUD-style right-edge control panel built on the same overlay pattern
    as `SceneOverlay`: a top-level frameless `Qt.Tool` window owned by
    the main window, anchored to the right edge of the viewport.
    """

    modelSetChanged          = pyqtSignal(str)
    textureSetChanged        = pyqtSignal(str)
    reconstructionSelected   = pyqtSignal(str)
    # Emitted when the user CLICKS a reconstruction row (not just
    # selects programmatically) - payload is the Reconstruction
    # dataclass. MainWindow consumes this to drive
    # `panda_app.mesh_reconstruction.run_2d_to_3d_reconstruction_from`.
    reconstructionRunRequested = pyqtSignal(object)
    # Emitted when the selected recon is (or stops being) a "stand"
    # snapshot. Payload is the Reconstruction when a stand row is
    # selected, or None when selection moves to a non-stand row. The
    # MainWindow uses it to show/hide the full-screen camera-alignment
    # reference overlay.
    standReferenceSelected     = pyqtSignal(object)
    # Emitted when the FOV slider moves. Payload is the new FOV (degrees).
    fovChanged                 = pyqtSignal(float)
    # Emitted when the roll dial moves. Payload is the new roll (degrees,
    # rotation about the view axis / centre of the screen).
    rollChanged                = pyqtSignal(float)
    # Emitted when the reference-overlay opacity slider moves (0..1).
    referenceOpacityChanged    = pyqtSignal(float)
    # Emitted when the reference-overlay visibility toggle flips.
    referenceVisibleToggled    = pyqtSignal(bool)
    # Emitted when the "pick bed corners" toggle flips (start/stop the
    # 4-point picking mode used for depth-fill reconstruction).
    pointPickingToggled        = pyqtSignal(bool)
    # Emitted when the user asks to clear the picked points / reconstruction.
    pointsResetRequested       = pyqtSignal()
    # Emitted when the anchor-point visualization toggle flips.
    pointVizToggled            = pyqtSignal(bool)
    # Emitted when the user requests the automatic anchor-point search + build.
    autoPointsRequested        = pyqtSignal()
    # Emitted when the user presses "Run Simulation". Payload is a dict:
    #   {
    #     "model_key":     str | None,   # current model set key
    #     "texture_key":   str | None,   # current texture set key
    #     "target_volume": float,        # cubic-metre target
    #   }
    # MainWindow consumes this and orchestrates the equivalent of the
    # legacy `run_full_process` (target volume → texture set → ground
    # plane → AABB plane → Perlin mesh from CSG).
    runRequested             = pyqtSignal(dict)
    # Emitted when the user picks a graphics preset (ultra/medium/performance).
    # MainWindow persists it and prompts for a restart (the rendering engine
    # is chosen before the window exists).
    graphicsPresetChanged    = pyqtSignal(str)

    PANEL_WIDTH = 320

    # ------------------------------------------------------------------
    def __init__(self, parent: QWidget, margin: int = 16):
        assert parent is not None, "RightPanel must have an anchor widget"

        # Top-level frameless tool window OWNED by the parent's top-level
        # window (NOT by the inner container) — that way z-order and
        # visibility properly follow the main window.
        owner_window = parent.window() or parent
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(owner_window, flags)

        self._owner = parent       # widget we anchor against (panda_container)
        self._margin = margin

        # Translucent painting so the inner card's rounded translucent
        # fill shows correctly. Do NOT set WA_TransparentForMouseEvents
        # here — this panel is interactive (combos, list, buttons).
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        apply_theme(self)

        # ---- The visible card (same QSS rule as SceneOverlay) -------
        self.card = QFrame(self)
        self.card.setObjectName("Overlay")

        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.card.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.card)

        # ---- Card body: scroll area wrapping the actual content -----
        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(0, 0, 0, 0)
        card_lay.setSpacing(0)

        scroll = QScrollArea(self.card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        # Make the scroll area pick up the translucent overlay fill from
        # its parent card (otherwise it would paint a solid bg).
        scroll.setStyleSheet("background: transparent;")
        scroll.viewport().setStyleSheet("background: transparent;")
        card_lay.addWidget(scroll)

        body = QWidget()
        body.setObjectName("RightPanelBody")
        body.setStyleSheet("background: transparent;")
        # Constrain body width to the panel — guarantees children clip
        # via word wrap / elision rather than overflowing the card.
        body.setMaximumWidth(self.PANEL_WIDTH)
        scroll.setWidget(body)

        col = QVBoxLayout(body)
        col.setContentsMargins(14, 14, 14, 14)
        col.setSpacing(10)

        # ---- Header (compact: brand dot + small caps) ---------------
        col.addLayout(self._build_header())

        # ---- Load configs --------------------------------------------
        # Model sets come from the TLS server's `models_geometry_config.json`
        # (with a local `models_config.yaml` fallback); texture sets come
        # from the TLS server's `textures_napolnitel_config.json`, which
        # `main.py` pre-loads into panel_data's in-memory cache before
        # this panel is constructed. Both lists carry the canonical
        # backend key alongside the human-readable display name so we
        # can emit the key on selection.
        model_sets   = load_model_sets()
        texture_sets = load_texture_sets()
        default_tex  = get_default_texture_set_key()

        # ---- Section: Model set --------------------------------------
        self.cmb_model = QComboBox()
        for key, display in model_sets:
            self.cmb_model.addItem(display, userData=key)
        if model_sets:
            self.cmb_model.setCurrentIndex(0)
        else:
            self.cmb_model.addItem("— модели не найдены —", userData=None)
            self.cmb_model.setEnabled(False)
        self.cmb_model.currentIndexChanged.connect(self._on_model_index_changed)
        col.addWidget(self._make_card(
            "Набор моделей",
            self._make_row("Набор", self.cmb_model),
            status="v1.4",
        ))

        # ---- Section: Texture set ------------------------------------
        self.cmb_texture = QComboBox()
        default_index = 0
        for i, (key, display) in enumerate(texture_sets):
            self.cmb_texture.addItem(display, userData=key)
            if default_tex and key == default_tex:
                default_index = i
        if texture_sets:
            self.cmb_texture.setCurrentIndex(default_index)
        else:
            self.cmb_texture.addItem("— текстуры не найдены —", userData=None)
            self.cmb_texture.setEnabled(False)
        self.cmb_texture.currentIndexChanged.connect(self._on_texture_index_changed)
        col.addWidget(self._make_card(
            "Текстуры",
            self._make_row("Текстура", self.cmb_texture),
        ))

        # ---- Section: Graphics preset --------------------------------
        # ultra / medium use RenderPipeline; performance uses simplepbr.
        # Switching the engine requires a restart (RP is built before the
        # window), so MainWindow only persists the choice + asks to restart.
        self.cmb_graphics = QComboBox()
        cur_preset = (graphics_settings.load_saved()
                      or graphics_settings.DEFAULT_PRESET)
        graphics_index = 0
        for i, pkey in enumerate(graphics_settings.PRESET_ORDER):
            self.cmb_graphics.addItem(
                graphics_settings.PRESETS[pkey]["name"], userData=pkey
            )
            if pkey == cur_preset:
                graphics_index = i
        self.cmb_graphics.setCurrentIndex(graphics_index)
        self.cmb_graphics.currentIndexChanged.connect(
            self._on_graphics_index_changed
        )
        col.addWidget(self._make_card(
            "Графика",
            self._make_row("Качество", self.cmb_graphics),
            status="Перезапуск",
        ))

        # ---- Section: Fill (target volume) --------------------------
        self.spn_target = QDoubleSpinBox()
        self.spn_target.setDecimals(2)
        self.spn_target.setRange(0.1, 999.0)
        self.spn_target.setSingleStep(0.5)
        self.spn_target.setSuffix(" m³")
        initial_volume = 10.0
        cur_model_key = self.cmb_model.itemData(self.cmb_model.currentIndex())
        if cur_model_key:
            mc = get_model_set_config(str(cur_model_key))
            if mc and mc.get("max_volume") is not None:
                try:
                    initial_volume = float(mc["max_volume"])
                except (TypeError, ValueError):
                    pass
        self.spn_target.setValue(initial_volume)
        col.addWidget(self._make_card(
            "Наполнение",
            self._make_row("Объём", self.spn_target),
            status="Параметр",
        ))

        # ---- Section: 2D · 3D Reconstructions -----------------------
        self._recons: list[Reconstruction] = load_reconstructions()

        self.lst_recon = QListWidget()
        self.lst_recon.setUniformItemSizes(False)
        self.lst_recon.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        # Local padding override - the global QSS rule "padding: 8px 10px"
        # was eating uneven slices of every item-widget's vertical space.
        # Setting it to 0 lets ReconRowWidget's own internal layout (which
        # already balances top/bottom via stretches) actually centre.
        self.lst_recon.setStyleSheet(
            "QListWidget::item { padding: 0px; margin: 0px; }"
        )
        self.lst_recon.setSpacing(4)
        self.lst_recon.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        try:
            from PyQt6.QtWidgets import QListView
            self.lst_recon.setResizeMode(QListView.ResizeMode.Adjust)
        except Exception:
            pass

        # Width budget for elision: panel width minus card padding (28),
        # list padding (~14), icon (20+10), button (24+10).
        row_text_width = self.PANEL_WIDTH - 28 - 14 - 30 - 34

        if self._recons:
            for idx, rec in enumerate(self._recons):
                row_w = ReconRowWidget(rec, max_text_width=row_text_width)
                row_w.viewClicked.connect(
                    lambda i=idx: self._on_view_requested(i)
                )
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, row_w.ROW_FIXED_HEIGHT))
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self.lst_recon.addItem(item)
                self.lst_recon.setItemWidget(item, row_w)
        else:
            placeholder = QListWidgetItem("— записей нет —")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.lst_recon.addItem(placeholder)

        self.lst_recon.currentItemChanged.connect(self._on_recon_changed)
        self.lst_recon.itemClicked.connect(self._on_recon_clicked)
        self.lst_recon.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.lst_recon.customContextMenuRequested.connect(
            self._on_recon_context_menu
        )
        # ---- Reset + Generate row (above the Recon list) -----------
        # Used to live in a footer at the very bottom; moved up so the
        # primary scene-pipeline trigger sits right next to the controls
        # that feed it (model / texture / fill).
        gen_row = QHBoxLayout()
        gen_row.setContentsMargins(0, 0, 0, 0)
        gen_row.setSpacing(8)

        btn_reset = QPushButton("Сброс")
        btn_reset.setProperty("variant", "ghost")
        btn_reset.clicked.connect(self._reset_selections)

        # Inline accent stylesheet (the property-selector path produced
        # invisible text on a green border).
        self.btn_run = QPushButton("Сгенерировать")
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setStyleSheet(self._soft_accent_button_qss(strong=True))
        self.btn_run.clicked.connect(self._emit_run_requested)

        gen_row.addWidget(btn_reset)
        gen_row.addStretch(1)
        gen_row.addWidget(self.btn_run)
        col.addLayout(gen_row)

        # ---- 2D · 3D Reconstruction card (list only) ----------------
        recon_count = len(self._recons) if self._recons else 0
        self._recon_card = self._make_card(
            "2D · 3D Реконструкции",
            self.lst_recon,
            status=str(recon_count),
            stretch=True,
        )
        col.addWidget(self._recon_card, 1)

        # ---- Selected record · Details card -------------------------
        self._details_form = QFormLayout()
        self._details_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._details_form.setHorizontalSpacing(10)
        self._details_form.setVerticalSpacing(4)
        self._details_form.setContentsMargins(0, 0, 0, 0)
        self._details_holder = QWidget()
        self._details_holder.setLayout(self._details_form)

        col.addWidget(self._make_card(
            "Выбранная запись",
            self._details_holder,
            status="Подробно",
        ))

        # ---- Camera controls card (FOV + reference overlay) ---------
        col.addWidget(self._make_card(
            "Камера · Выравнивание",
            self._build_camera_controls(),
        ))

        # ---- Bottom row: Load more + Reconstruct -------------------
        # Both buttons are kept here so they share the same horizontal
        # rhythm as the Reset/Generate row above the recon list.
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 2, 0, 0)
        bottom_row.setSpacing(8)

        self.btn_load_more = QPushButton("Загрузить ещё")
        self.btn_load_more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load_more.setStyleSheet(
            "QPushButton {"
            "  background: rgba(255,255,255,4);"
            f"  color: {COLOR_TEXT_MUTED};"
            f"  border: 1px dashed {COLOR_HAIRLINE};"
            "  border-radius: 6px;"
            "  padding: 6px 12px;"
            "  font-size: 12px;"
            "  font-weight: 500;"
            "  letter-spacing: 0.3px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(255,255,255,10);"
            f"  color: {COLOR_TEXT};"
            f"  border-color: {COLOR_HAIRLINE_HOVER};"
            "}"
            "QPushButton:disabled {"
            f"  color: {COLOR_TEXT_DIM};"
            "  border-style: solid;"
            "}"
        )
        self.btn_load_more.clicked.connect(self._on_load_more)

        self.btn_run_recon = QPushButton("Реконструировать")
        self.btn_run_recon.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_recon.setStyleSheet(self._soft_accent_button_qss())
        self.btn_run_recon.clicked.connect(self._emit_recon_run_requested)

        bottom_row.addWidget(self.btn_load_more)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.btn_run_recon)
        col.addLayout(bottom_row)

        # ---- Initial selection -------------------------------------
        if self._recons:
            self._populate_details(self._recons[0])
            self.lst_recon.setCurrentRow(0)
            self._selected_rec = self._recons[0]
        else:
            self._populate_details(None)
            self._selected_rec = None
            self.btn_run_recon.setEnabled(False)

        # Card width is fixed; height tracks the container in _reposition.
        self.setFixedWidth(self.PANEL_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Expanding)

    # ==================================================================
    # Public API (mirrors SceneOverlay.attach)
    # ==================================================================
    def attach(self) -> None:
        """
        Install event filters on the anchor widget and on its top-level
        window, then reposition + show the panel.
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

    # ==================================================================
    # Section builders
    # ==================================================================
    def _build_header(self) -> QVBoxLayout:
        """Compact pill: brand dot + IQOKO label + LIVE chip."""
        v = QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 4)
        v.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 10px;")

        brand = QLabel("IQOKO · 3D СИМУЛЯТОР")
        brand.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 11px; font-weight: 600;"
            f" letter-spacing: 1.2px;"
        )

        chip = _make_chip("LIVE", "chip-live")

        row.addWidget(dot)
        row.addWidget(brand)
        row.addStretch(1)
        row.addWidget(chip)
        v.addLayout(row)
        return v

    # ------------------------------------------------------------------
    # Camera-alignment controls (FOV + reference overlay)
    # ------------------------------------------------------------------
    _FOV_MIN = 20
    _FOV_MAX = 150
    _FOV_DEFAULT = 100

    # Roll (rotation about the view axis / centre of the screen), degrees.
    _ROLL_MIN = -180
    _ROLL_MAX = 180

    @staticmethod
    def _thin_slider_qss() -> str:
        return (
            "QSlider::groove:horizontal {"
            f"  background: {COLOR_HAIRLINE};"
            "  height: 3px; border-radius: 1px;"
            "}"
            "QSlider::sub-page:horizontal {"
            f"  background: {COLOR_ACCENT}; height: 3px; border-radius: 1px;"
            "}"
            "QSlider::handle:horizontal {"
            f"  background: {COLOR_ACCENT};"
            "  width: 10px; height: 10px;"
            "  margin: -4px 0; border-radius: 5px;"
            "}"
            "QSlider::handle:horizontal:hover { background: #00FFAA; }"
            "QSlider:disabled { }"
            "QSlider::sub-page:horizontal:disabled {"
            f"  background: {COLOR_TEXT_DIM};"
            "}"
            "QSlider::handle:horizontal:disabled {"
            f"  background: {COLOR_TEXT_DIM};"
            "}"
        )

    def _build_camera_controls(self) -> QWidget:
        """
        Build the camera-alignment controls:
          • FOV slider (20..150°) — always active, drives the live lens.
          • Reference-overlay opacity slider + show/hide toggle — active
            only while a `stand` snapshot is selected (the overlay shows
            that snapshot's colour frame over the 3D viewport so the user
            can match the camera by hand).
        """
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        v = QVBoxLayout(holder)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        def _caption(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
                f" letter-spacing: 1.0px; background: transparent;"
            )
            return lbl

        def _value_lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {COLOR_TEXT}; font-family: {FONT_MONO};"
                f" font-size: 11px; background: transparent;"
            )
            lbl.setMinimumWidth(40)
            lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            return lbl

        # Everything below lives inside the collapsible "Дополнительно"
        # container (FOV, roll, overlay + point-picking controls). The main
        # camera card stays minimal — just the disclosure toggle.
        self._adv_holder = QWidget()
        self._adv_holder.setStyleSheet("background: transparent;")
        ah = QVBoxLayout(self._adv_holder)
        ah.setContentsMargins(0, 0, 0, 0)
        ah.setSpacing(8)

        # ----- FOV row ------------------------------------------------
        fov_row = QHBoxLayout()
        fov_row.setContentsMargins(0, 0, 0, 0)
        fov_row.setSpacing(8)

        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(self._FOV_MIN, self._FOV_MAX)
        self.fov_slider.setValue(self._FOV_DEFAULT)
        self.fov_slider.setFixedHeight(18)
        self.fov_slider.setStyleSheet(self._thin_slider_qss())

        self.fov_value_lbl = _value_lbl(f"{self._FOV_DEFAULT}°")

        def _on_fov(val: int):
            self.fov_value_lbl.setText(f"{int(val)}°")
            self.fovChanged.emit(float(val))

        self.fov_slider.valueChanged.connect(_on_fov)

        fov_row.addWidget(_caption("FOV"), 0, Qt.AlignmentFlag.AlignVCenter)
        fov_row.addWidget(self.fov_slider, 1, Qt.AlignmentFlag.AlignVCenter)
        fov_row.addWidget(self.fov_value_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        ah.addLayout(fov_row)

        # ----- Roll dial ("крутилка" about the view axis) -------------
        roll_row = QHBoxLayout()
        roll_row.setContentsMargins(0, 0, 0, 0)
        roll_row.setSpacing(8)

        self.roll_dial = QDial()
        self.roll_dial.setRange(self._ROLL_MIN, self._ROLL_MAX)
        self.roll_dial.setValue(0)
        self.roll_dial.setWrapping(True)       # angle wraps -180 <-> 180
        self.roll_dial.setNotchesVisible(True)
        self.roll_dial.setFixedSize(46, 46)
        self.roll_dial.setCursor(Qt.CursorShape.PointingHandCursor)
        self.roll_dial.setToolTip(
            "Крен камеры (поворот вокруг центра экрана).\n"
            "Кнопка ⟲ справа — сбросить в 0"
        )
        self.roll_dial.setStyleSheet(
            "QDial { background: transparent; }"
        )

        self.roll_value_lbl = _value_lbl("0°")

        def _on_roll(val: int):
            self.roll_value_lbl.setText(f"{int(val)}°")
            self.rollChanged.emit(float(val))

        self.roll_dial.valueChanged.connect(_on_roll)

        # Reset-to-zero affordance (the default presets also zero it).
        self.btn_roll_reset = QToolButton()
        self.btn_roll_reset.setText("⟲")
        self.btn_roll_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_roll_reset.setToolTip("Сбросить крен в 0")
        self.btn_roll_reset.setStyleSheet(
            "QToolButton {"
            "  background: transparent;"
            f"  color: {COLOR_TEXT_MUTED};"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 5px; padding: 2px 6px; font-size: 13px;"
            "}"
            "QToolButton:hover {"
            "  background: rgba(255,255,255,8);"
            f"  color: {COLOR_TEXT};"
            "}"
        )
        # setValue(0) fires valueChanged -> _on_roll -> emits rollChanged,
        # so the camera actually rolls back to level.
        self.btn_roll_reset.clicked.connect(lambda: self.roll_dial.setValue(0))

        roll_row.addWidget(_caption("КРЕН"), 0, Qt.AlignmentFlag.AlignVCenter)
        roll_row.addStretch(1)
        roll_row.addWidget(self.roll_dial, 0, Qt.AlignmentFlag.AlignVCenter)
        roll_row.addWidget(self.roll_value_lbl, 0, Qt.AlignmentFlag.AlignVCenter)
        roll_row.addWidget(self.btn_roll_reset, 0, Qt.AlignmentFlag.AlignVCenter)
        ah.addLayout(roll_row)

        # ----- Reference-overlay controls (stand snapshots only) ------
        self._ref_controls_holder = QWidget()
        self._ref_controls_holder.setStyleSheet("background: transparent;")
        self._ref_controls_holder.setEnabled(False)
        rc = QVBoxLayout(self._ref_controls_holder)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(8)

        # Opacity row.
        op_row = QHBoxLayout()
        op_row.setContentsMargins(0, 0, 0, 0)
        op_row.setSpacing(8)

        self.ref_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ref_opacity_slider.setRange(0, 100)   # percent (0 = invisible)
        self.ref_opacity_slider.setValue(50)
        self.ref_opacity_slider.setFixedHeight(18)
        self.ref_opacity_slider.setStyleSheet(self._thin_slider_qss())

        self.ref_opacity_lbl = _value_lbl("50%")

        def _on_opacity(val: int):
            self.ref_opacity_lbl.setText(f"{int(val)}%")
            self.referenceOpacityChanged.emit(float(val) / 100.0)

        self.ref_opacity_slider.valueChanged.connect(_on_opacity)

        op_row.addWidget(_caption("ПРОЗР"), 0, Qt.AlignmentFlag.AlignVCenter)
        op_row.addWidget(self.ref_opacity_slider, 1,
                         Qt.AlignmentFlag.AlignVCenter)
        op_row.addWidget(self.ref_opacity_lbl, 0,
                         Qt.AlignmentFlag.AlignVCenter)
        rc.addLayout(op_row)

        # Show/hide toggle.
        self.btn_ref_toggle = QPushButton("Скрыть снимок")
        self.btn_ref_toggle.setCheckable(True)
        self.btn_ref_toggle.setChecked(True)
        self.btn_ref_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_ref_toggle.setStyleSheet(self._soft_accent_button_qss())

        def _on_toggle(checked: bool):
            self.btn_ref_toggle.setText(
                "Скрыть снимок" if checked else "Показать снимок"
            )
            self.referenceVisibleToggled.emit(bool(checked))

        self.btn_ref_toggle.toggled.connect(_on_toggle)
        rc.addWidget(self.btn_ref_toggle)

        # ----- Bed-corner picking + reconstruction --------------------
        pick_row = QHBoxLayout()
        pick_row.setContentsMargins(0, 0, 0, 0)
        pick_row.setSpacing(6)

        self.btn_pick_points = QPushButton("Выбрать точки")
        self.btn_pick_points.setCheckable(True)
        self.btn_pick_points.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pick_points.setToolTip(
            "Включите и кликайте опорные точки на кузове (любое число).\n"
            "ПКМ или Esc — завершить выбор и построить наполнение."
        )
        self.btn_pick_points.setStyleSheet(self._soft_accent_button_qss())
        self.btn_pick_points.toggled.connect(
            lambda checked: self.pointPickingToggled.emit(bool(checked))
        )

        self.btn_pick_reset = QToolButton()
        self.btn_pick_reset.setText("⟲")
        self.btn_pick_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pick_reset.setToolTip("Сбросить точки и реконструкцию")
        self.btn_pick_reset.setStyleSheet(
            "QToolButton {"
            "  background: transparent;"
            f"  color: {COLOR_TEXT_MUTED};"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 5px; padding: 4px 8px; font-size: 13px;"
            "}"
            "QToolButton:hover {"
            "  background: rgba(255,255,255,8);"
            f"  color: {COLOR_TEXT};"
            "}"
        )
        self.btn_pick_reset.clicked.connect(
            lambda _=False: self.pointsResetRequested.emit()
        )

        pick_row.addWidget(self.btn_pick_points, 1)
        pick_row.addWidget(self.btn_pick_reset, 0)
        rc.addLayout(pick_row)

        # ----- Auto anchor-points + visualization toggle --------------
        pts_row = QHBoxLayout()
        pts_row.setContentsMargins(0, 0, 0, 0)
        pts_row.setSpacing(6)

        self.btn_auto_points = QPushButton("Авто-точки")
        self.btn_auto_points.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_points.setToolTip(
            "Автоматически найти опорные точки на кузове и построить "
            "наполнение."
        )
        self.btn_auto_points.setStyleSheet(self._soft_accent_button_qss())
        self.btn_auto_points.clicked.connect(
            lambda _=False: self.autoPointsRequested.emit()
        )

        self.btn_point_viz = QPushButton("Точки")
        self.btn_point_viz.setCheckable(True)
        self.btn_point_viz.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_point_viz.setToolTip(
            "Показать использованные опорные точки на экране (зелёные)."
        )
        self.btn_point_viz.setStyleSheet(self._soft_accent_button_qss())
        self.btn_point_viz.toggled.connect(
            lambda checked: self.pointVizToggled.emit(bool(checked))
        )

        pts_row.addWidget(self.btn_auto_points, 1)
        pts_row.addWidget(self.btn_point_viz, 0)
        rc.addLayout(pts_row)

        ah.addWidget(self._ref_controls_holder)

        # ----- Collapsible "Дополнительно" section --------------------
        # FOV, roll, overlay opacity/visibility, manual/auto point picking and
        # the point visualisation all live behind a disclosure toggle so the
        # main camera card stays clean. Collapsed by default.
        self._adv_toggle = QToolButton()
        self._adv_toggle.setText("  Дополнительно")
        self._adv_toggle.setCheckable(True)
        self._adv_toggle.setChecked(False)
        self._adv_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._adv_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._adv_toggle.setToolTip(
            "FOV, крен, наложение снимка, ручной/авто выбор опорных точек, "
            "визуализация точек."
        )
        self._adv_toggle.setStyleSheet(
            "QToolButton {"
            "  background: transparent;"
            f"  color: {COLOR_TEXT_MUTED};"
            "  border: none; padding: 2px 0;"
            "  font-size: 10px; font-weight: 600; letter-spacing: 1.0px;"
            "}"
            f"QToolButton:hover {{ color: {COLOR_TEXT}; }}"
        )

        def _on_adv_toggled(checked: bool):
            self._adv_holder.setVisible(bool(checked))
            self._adv_toggle.setArrowType(
                Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
            )
            if hasattr(self, "_reposition"):
                try:
                    self._reposition()
                except Exception:
                    pass

        self._adv_toggle.toggled.connect(_on_adv_toggled)

        # Collapsed initially: the advanced controls are hidden until expanded.
        self._adv_holder.setVisible(False)

        v.addWidget(self._adv_toggle)
        v.addWidget(self._adv_holder)
        return holder

    def set_point_count(self, n: int) -> None:
        """Update the pick-button label with the current point count."""
        btn = getattr(self, "btn_pick_points", None)
        if btn is not None:
            n = max(0, int(n))
            btn.setText("Выбрать точки" if n == 0
                        else f"Выбрать точки ({n})")

    def set_picking_active(self, active: bool) -> None:
        """Reflect picking state on the toggle without re-emitting."""
        btn = getattr(self, "btn_pick_points", None)
        if btn is None:
            return
        blocked = btn.blockSignals(True)
        btn.setChecked(bool(active))
        btn.blockSignals(blocked)

    def set_fov_value(self, fov: float) -> None:
        """Sync the FOV slider to an externally-applied lens FOV without
        re-emitting `fovChanged` (used when camera modes change FOV)."""
        sl = getattr(self, "fov_slider", None)
        if sl is None:
            return
        try:
            v = int(round(float(fov)))
        except (TypeError, ValueError):
            return
        v = max(self._FOV_MIN, min(self._FOV_MAX, v))
        blocked = sl.blockSignals(True)
        sl.setValue(v)
        sl.blockSignals(blocked)
        if hasattr(self, "fov_value_lbl"):
            self.fov_value_lbl.setText(f"{v}°")

    def set_roll_value(self, roll: float) -> None:
        """Sync the roll dial to an externally-applied camera roll without
        re-emitting `rollChanged` (used when camera modes / presets set
        roll)."""
        d = getattr(self, "roll_dial", None)
        if d is None:
            return
        try:
            v = int(round(float(roll)))
        except (TypeError, ValueError):
            return
        # Normalise into the dial's [-180, 180] wrapping range.
        while v > self._ROLL_MAX:
            v -= 360
        while v < self._ROLL_MIN:
            v += 360
        blocked = d.blockSignals(True)
        d.setValue(v)
        d.blockSignals(blocked)
        if hasattr(self, "roll_value_lbl"):
            self.roll_value_lbl.setText(f"{v}°")

    # ------------------------------------------------------------------
    # IQoko-style card builders (replacing the old QGroupBox approach).
    # ------------------------------------------------------------------
    def _make_card(self, title: str, content: QWidget,
                   status: str = "", stretch: bool = False) -> QFrame:
        """
        Card = small QFrame with translucent fill + hairline border +
        radius. Header is a small-caps eyebrow on the left and an
        optional monospace status string on the right (e.g. "v1.4",
        "Target", record count).
        """
        card = QFrame()
        card.setObjectName("PanelCard")
        card.setStyleSheet(
            "QFrame#PanelCard {"
            "  background-color: rgba(22, 22, 22, 0.55);"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "  border-radius: 10px;"
            "}"
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        t = QLabel(title.upper())
        t.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
            f" font-weight: 600; letter-spacing: 1.2px;"
        )
        head.addWidget(t)
        head.addStretch(1)
        if status:
            s = QLabel(status)
            s.setStyleSheet(
                f"color: {COLOR_TEXT_DIM}; font-size: 10px;"
                f" font-family: 'Geist Mono','JetBrains Mono',monospace;"
            )
            head.addWidget(s)
        v.addLayout(head)

        if stretch:
            v.addWidget(content, 1)
        else:
            v.addWidget(content)
        return card

    def _make_row(self, label: str, widget: QWidget) -> QWidget:
        """88px-label + field row, no hint underneath. Mirrors IQoko's
        `.row { grid-template-columns: 88px 1fr }` block."""
        w = QWidget()
        from PyQt6.QtWidgets import QGridLayout
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(0)

        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;"
        )
        g.addWidget(lbl, 0, 0, Qt.AlignmentFlag.AlignVCenter)
        g.addWidget(widget, 0, 1, Qt.AlignmentFlag.AlignVCenter)
        g.setColumnMinimumWidth(0, 70)
        g.setColumnStretch(1, 1)
        return w

    # ---- Legacy stubs (kept for backwards compatibility) ------------
    def _wrap_group(self, title: str, content: QWidget,
                    content_height: int | None = None) -> QFrame:
        return self._make_card(title, content)

    def _build_combo_section(self, combo: QComboBox, hint: str = "") -> QWidget:
        return self._make_row("", combo)

    def _build_target_section(self, spin: QDoubleSpinBox) -> QWidget:
        return self._make_row("Volume", spin)

    # ==================================================================
    # Reconstruction list helpers
    # ==================================================================
    def _on_recon_changed(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        rec = self._recons[idx]
        self._selected_rec = rec
        self._populate_details(rec)
        if hasattr(self, "btn_run_recon"):
            self.btn_run_recon.setEnabled(True)
        self.reconstructionSelected.emit(str(rec.name))
        self._emit_stand_reference(rec)

    def _on_recon_clicked(self, item: QListWidgetItem) -> None:
        """
        Click on a row now ONLY selects (caches the rec for the Details
        card + Run-reconstruction button) - the actual reconstruction
        pipeline is triggered explicitly by `btn_run_recon`.
        """
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        self._selected_rec = self._recons[idx]
        self._populate_details(self._selected_rec)
        if hasattr(self, "btn_run_recon"):
            self.btn_run_recon.setEnabled(True)
        self._emit_stand_reference(self._selected_rec)

    def _emit_stand_reference(self, rec: Reconstruction | None) -> None:
        """Tell the MainWindow to show the alignment overlay for `stand`
        snapshots, or hide it for any other selection."""
        is_stand = bool(rec is not None and rec.data_type == "stand")
        self.standReferenceSelected.emit(rec if is_stand else None)
        # Keep the reference-controls card enabled only while a stand row
        # is the active selection.
        if hasattr(self, "_ref_controls_holder"):
            self._ref_controls_holder.setEnabled(is_stand)

    def _on_recon_context_menu(self, pos: QPoint) -> None:
        """Right-click menu on a recon row: copy its file name to clipboard."""
        item = self.lst_recon.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        rec = self._recons[idx]
        name = str(rec.name or "").strip()
        if not name:
            return

        menu = QMenu(self.lst_recon)
        act_copy = menu.addAction("Скопировать имя файла")
        chosen = menu.exec(self.lst_recon.viewport().mapToGlobal(pos))
        if chosen is act_copy:
            QApplication.clipboard().setText(name)

    def _on_view_requested(self, idx: int) -> None:
        """
        Per-row "open" button handler — pops the photo viewer modal over
        the main window. Selects the row first so the side panel's
        Details section stays in sync.
        """
        if not (0 <= idx < len(self._recons)):
            return
        self.lst_recon.setCurrentRow(idx)
        rec = self._recons[idx]
        # Anchor the dialog on the main top-level window so its modality
        # blocks the right tool app correctly and `geometry()` covers the
        # full window.
        owner_top = self._owner.window() if self._owner else None
        dlg = RecordPhotoOverlay(rec, parent=owner_top)
        # The dialog is modal-app — input is blocked everywhere else,
        # but the right panel is allowed to stay visible behind it
        # (looks cleaner: the user keeps their context).
        try:
            dlg.exec()
        finally:
            self._reposition()
            self.raise_()

    # ==================================================================
    # Details population
    # ==================================================================
    # ------------------------------------------------------------------
    def _emit_recon_run_requested(self) -> None:
        """Fire the reconstructionRunRequested signal for the selected row."""
        rec = getattr(self, "_selected_rec", None)
        if rec is None:
            return
        self.reconstructionRunRequested.emit(rec)

    # ------------------------------------------------------------------
    def _on_load_more(self) -> None:
        """
        Pull a larger page of reconstructions from the server and rebuild
        the list widget with the merged set, preserving the current
        selection if possible.
        """
        # Track current paging cap on the panel - bumps by RECON_PAGE_SIZE
        # each click.
        new_limit = getattr(self, "_recon_limit",
                            RECON_PAGE_SIZE) + RECON_PAGE_SIZE
        self._recon_limit = new_limit
        try:
            self.btn_load_more.setEnabled(False)
            self.btn_load_more.setText("Загрузка...")
            new_recons = load_reconstructions(limit=new_limit)
        except Exception as exc:
            print(f"[RightPanel] load_reconstructions failed: {exc}")
            self.btn_load_more.setEnabled(True)
            self.btn_load_more.setText("Загрузить ещё")
            return

        # Rebuild the list widget contents.
        sel_name = (self._selected_rec.name
                    if getattr(self, "_selected_rec", None) else None)
        self.lst_recon.clear()
        self._recons = new_recons
        row_text_width = self.PANEL_WIDTH - 28 - 14 - 30 - 34
        if self._recons:
            for idx, rec in enumerate(self._recons):
                row_w = ReconRowWidget(rec, max_text_width=row_text_width)
                row_w.viewClicked.connect(
                    lambda i=idx: self._on_view_requested(i)
                )
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, row_w.ROW_FIXED_HEIGHT))
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self.lst_recon.addItem(item)
                self.lst_recon.setItemWidget(item, row_w)

            # Restore selection by name if possible.
            if sel_name:
                for i, r in enumerate(self._recons):
                    if r.name == sel_name:
                        self.lst_recon.setCurrentRow(i)
                        break
                else:
                    self.lst_recon.setCurrentRow(0)
            else:
                self.lst_recon.setCurrentRow(0)

        # Refresh the recon card header status.
        try:
            self._refresh_recon_card_count()
        except Exception:
            pass

        self.btn_load_more.setEnabled(True)
        self.btn_load_more.setText("Load more")

    def _refresh_recon_card_count(self) -> None:
        """Update the small status label on the Recon card header."""
        if not hasattr(self, "_recon_card"):
            return
        # The card header is `head -> [title, addStretch, status_label]`
        # (see _make_card). We just rewalk it to find the second QLabel.
        try:
            head = self._recon_card.layout().itemAt(0).layout()
            for i in range(head.count()):
                w = head.itemAt(i).widget()
                if isinstance(w, QLabel) and w.styleSheet().find(
                        "Geist Mono") != -1:
                    w.setText(str(len(self._recons)))
                    return
        except Exception:
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def _soft_accent_button_qss(strong: bool = False) -> str:
        """
        IQoko-style soft accent button. `strong=True` for the primary
        Run footer button (more saturated fill so it reads as the main
        action); the default flavour is for secondary actions like
        "Run reconstruction".
        """
        bg_a = "rgba(0, 255, 136, 50)" if strong else "rgba(0, 255, 136, 30)"
        bg_h = "rgba(0, 255, 136, 90)" if strong else "rgba(0, 255, 136, 55)"
        return (
            "QPushButton {"
            f"  background-color: {bg_a};"
            f"  color: {COLOR_TEXT};"
            f"  border: 1px solid {COLOR_ACCENT};"
            "  border-radius: 6px;"
            "  padding: 6px 14px;"
            "  font-weight: 600;"
            "  letter-spacing: 0.3px;"
            "}"
            "QPushButton:hover {"
            f"  background-color: {bg_h};"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(0, 255, 136, 110);"
            "}"
            "QPushButton:disabled {"
            "  background-color: rgba(255, 255, 255, 4);"
            f"  color: {COLOR_TEXT_DIM};"
            f"  border: 1px solid {COLOR_HAIRLINE};"
            "}"
        )

    def _populate_details(self, rec: Reconstruction | None) -> None:
        """Render the Details form for one reconstruction (or empty state)."""
        # Clear existing rows.
        while self._details_form.rowCount():
            self._details_form.removeRow(0)

        if rec is None:
            empty = QLabel("Выберите запись, чтобы увидеть подробности.")
            empty.setProperty("role", "muted")
            empty.setWordWrap(True)
            self._details_form.addRow(empty)
            return

        # Type chip — accent for height/stand, idle for ply, warn otherwise.
        type_chip_role = (
            "chip-live" if rec.data_type in ("height", "stand")
            else "chip-idle" if rec.data_type == "ply"
            else "chip-err"
        )

        target_str = (
            f"{rec.target_volume:.2f} m³"
            if rec.target_volume is not None
            else "—"
        )
        time_str = _format_short_dt(rec)

        rows: list[tuple[str, str, str | None]] = [
            ("А/Н",     rec.car_number or "—",                None),
            ("МОДЕЛЬ",  rec.model or "—",                     None),
            ("ТИП",     (rec.data_type or "—").upper(),       type_chip_role),
            ("НАПОЛНИТЕЛЬ", rec.filler or "—",                None),
            ("ЦЕЛЕВОЙ ОБЪЁМ", target_str,                     None),
            ("ВРЕМЯ",   time_str,                             None),
            ("ФАЙЛ",    rec.name or "—",                      None),
        ]
        # Cap value-cell width so long strings wrap inside the panel
        # instead of pushing the form rightwards.
        value_cap = self.PANEL_WIDTH - 40 - 70  # card margins + label col

        # Map chip-role -> colour so TYPE keeps its visual cue without
        # a bordered chip (the bordered pill clashed with the rest of
        # the borderless rows in the Details card).
        _chip_color = {
            "chip-live": COLOR_ACCENT,
            "chip-idle": COLOR_TEXT_MUTED,
            "chip-err":  COLOR_WARN,
        }

        for label, value, chip_role in rows:
            k = QLabel(label)
            k.setProperty("role", "eyebrow")

            v = QLabel(value)
            v.setWordWrap(True)
            v.setMaximumWidth(value_cap)

            if chip_role is not None and chip_role in _chip_color:
                col = _chip_color[chip_role]
                v.setStyleSheet(
                    f"color: {col}; font-family: {FONT_MONO};"
                    f"font-size: 11px; font-weight: 600;"
                    f"background: transparent; border: none; padding: 0;"
                )
            else:
                v.setProperty("role", "muted")

            if label == "ФАЙЛ" and value and value != "—":
                # Pair the file name with a small copy-to-clipboard button.
                # The button is allowed to be QLabel-sized so it lines up
                # with the first text line even when the file name wraps.
                v.setTextInteractionFlags(
                    Qt.TextInteractionFlag.TextSelectableByMouse
                )
                cell = QWidget()
                cell_lay = QHBoxLayout(cell)
                cell_lay.setContentsMargins(0, 0, 0, 0)
                cell_lay.setSpacing(6)
                cell_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
                cell_lay.addWidget(v, 1)

                btn_copy = QPushButton()
                btn_copy.setIcon(_make_copy_icon(14, COLOR_TEXT_MUTED))
                btn_copy.setIconSize(QSize(14, 14))
                btn_copy.setFixedSize(22, 22)
                btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
                btn_copy.setToolTip("Скопировать имя файла")
                btn_copy.setStyleSheet(
                    "QPushButton {"
                    "  background: transparent;"
                    "  border: 1px solid transparent; border-radius: 5px;"
                    "  padding: 0; min-width: 22px; max-width: 22px;"
                    "  min-height: 22px; max-height: 22px;"
                    "}"
                    "QPushButton:hover {"
                    "  background-color: rgba(255,255,255,12);"
                    f"  border-color: {COLOR_HAIRLINE};"
                    "}"
                    "QPushButton:pressed {"
                    "  background-color: rgba(0,255,136,18);"
                    f"  border-color: {COLOR_ACCENT};"
                    "}"
                )
                btn_copy.clicked.connect(
                    lambda _checked=False, _name=value, _b=btn_copy:
                        self._copy_filename_to_clipboard(_name, _b)
                )
                cell_lay.addWidget(btn_copy, 0, Qt.AlignmentFlag.AlignTop)
                self._details_form.addRow(k, cell)
            else:
                self._details_form.addRow(k, v)

    def _copy_filename_to_clipboard(self, name: str,
                                    btn: QPushButton) -> None:
        """Copy `name` to clipboard and briefly flash the button tooltip."""
        QApplication.clipboard().setText(name)
        btn.setToolTip("Скопировано")
        QTimer.singleShot(
            1200,
            lambda b=btn: b.setToolTip("Скопировать имя файла"),
        )

    # ==================================================================
    # Combo handlers
    # ==================================================================
    def _on_model_index_changed(self, idx: int) -> None:
        """
        Translate combo index to backend key, sync the target-volume
        spinbox to the new model's `max_volume`, and emit modelSetChanged.
        """
        key = self.cmb_model.itemData(idx)
        if key and hasattr(self, "spn_target"):
            mc = get_model_set_config(str(key))
            if mc and mc.get("max_volume") is not None:
                try:
                    self.spn_target.setValue(float(mc["max_volume"]))
                except (TypeError, ValueError):
                    pass
        if key:
            self.modelSetChanged.emit(str(key))

    def _on_texture_index_changed(self, idx: int) -> None:
        key = self.cmb_texture.itemData(idx)
        if key:
            self.textureSetChanged.emit(str(key))

    def _on_graphics_index_changed(self, idx: int) -> None:
        key = self.cmb_graphics.itemData(idx)
        if key:
            self.graphicsPresetChanged.emit(str(key))

    # ==================================================================
    # External hook: пересборка списка текстурных наборов
    # ==================================================================
    def update_texture_sets(self, texture_sets_list, default_key=None) -> None:
        """
        Перезалить выпадающий список текстурных наборов.

        Принимает список пар (key, display_name); если есть `default_key`
        и он встречается среди ключей — именно этот элемент становится
        выбранным. Сигнал textureSetChanged во время перезалива не
        эмитится: подписчики получают только финальное состояние
        (если оно отличается от исходного — через стандартный
        currentIndexChanged).

        Используется из MainWindow.attach_panda после того, как клиент
        получил с сервера актуальный textures_napolnitel_config.json.
        """
        if not hasattr(self, "cmb_texture"):
            return

        items = []
        for entry in (texture_sets_list or []):
            try:
                key, display = entry
            except (TypeError, ValueError):
                continue
            if not key or key == "default":
                continue
            items.append((str(key), str(display) if display else str(key)))

        self.cmb_texture.blockSignals(True)
        try:
            self.cmb_texture.clear()
            if items:
                target_index = 0
                for i, (key, display) in enumerate(items):
                    self.cmb_texture.addItem(display, userData=key)
                    if default_key and key == default_key:
                        target_index = i
                self.cmb_texture.setEnabled(True)
                self.cmb_texture.setCurrentIndex(target_index)
            else:
                self.cmb_texture.addItem("— текстуры не найдены —",
                                         userData=None)
                self.cmb_texture.setEnabled(False)
        finally:
            self.cmb_texture.blockSignals(False)

    # ==================================================================
    # Public accessors
    # ==================================================================
    def current_model_key(self):
        return self.cmb_model.itemData(self.cmb_model.currentIndex())

    def set_current_model_key(self, key) -> bool:
        """
        Програмно выставить выбранный model set в комбо-боксе.
        Используется, когда модель загружается в сцену в обход юзера
        (например, при запуске реконструкции из JSON) — комбо тогда
        отставал, и current_model_key() возвращал устаревшее значение.

        Сигналы блокируются, чтобы не триггерить повторный
        cache_and_load_model_set (модель уже загружена вызывающим кодом).
        Однако спинбокс target-volume и details-форму синхронизируем
        вручную — как это сделал бы _on_model_index_changed.

        Возвращает True, если ключ найден в комбо и индекс выставлен.
        """
        if key is None:
            return False
        for i in range(self.cmb_model.count()):
            if self.cmb_model.itemData(i) == key:
                self.cmb_model.blockSignals(True)
                try:
                    self.cmb_model.setCurrentIndex(i)
                finally:
                    self.cmb_model.blockSignals(False)
                if hasattr(self, "spn_target"):
                    mc = get_model_set_config(str(key))
                    if mc and mc.get("max_volume") is not None:
                        try:
                            self.spn_target.setValue(float(mc["max_volume"]))
                        except (TypeError, ValueError):
                            pass
                return True
        return False

    def current_texture_key(self):
        return self.cmb_texture.itemData(self.cmb_texture.currentIndex())

    def current_target_volume(self) -> float:
        try:
            return float(self.spn_target.value())
        except Exception:
            return 0.0

    def _emit_run_requested(self) -> None:
        payload = {
            "model_key":     self.current_model_key(),
            "texture_key":   self.current_texture_key(),
            "target_volume": self.current_target_volume(),
        }
        self.runRequested.emit(payload)

    def _reset_selections(self) -> None:
        if self.cmb_model.count():
            self.cmb_model.setCurrentIndex(0)
        default_tex = get_default_texture_set_key()
        if default_tex is not None:
            for i in range(self.cmb_texture.count()):
                if self.cmb_texture.itemData(i) == default_tex:
                    self.cmb_texture.setCurrentIndex(i)
                    break
            else:
                self.cmb_texture.setCurrentIndex(0)
        elif self.cmb_texture.count():
            self.cmb_texture.setCurrentIndex(0)
        if self._recons:
            self.lst_recon.setCurrentRow(0)

    # ==================================================================
    # Overlay positioning + event tracking (mirrors SceneOverlay)
    # ==================================================================
    def _reposition(self) -> None:
        owner = self._owner
        if owner is None:
            return
        pw, ph = owner.width(), owner.height()
        w = self.width()
        m = self._margin
        h = max(120, ph - 2 * m)
        local = QPoint(pw - w - m, m)
        gp = owner.mapToGlobal(local)
        self.setGeometry(gp.x(), gp.y(), w, h)
        self.raise_()

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
            self._reposition()
        if obj is top:
            if et == QEvent.Type.Hide:
                self.hide()
            elif et == QEvent.Type.Show:
                self.show()
                self._reposition()
        return super().eventFilter(obj, event)
