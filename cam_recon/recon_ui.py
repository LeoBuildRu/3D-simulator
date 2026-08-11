"""
recon_ui — interactive test bench for `camera_recon`.

Load a still, type the truck body's real width and length, trace three of its
four top-rim edges, and the reconstructed camera is drawn straight back over
the photo as a full four-sided wireframe.  The fourth edge — the one you never
traced — is the actual accuracy check: it is a pure prediction, so how well it
lands on the real rim is a direct read-out of the method's precision.

Usage
-----
    python cam_recon/recon_ui.py [image.png]

Controls
--------
    left drag          trace the active edge (drag an endpoint to adjust it)
    1 2 3 4            pick which edge you are tracing
    right click        clear the edge under the cursor
    wheel              zoom       middle drag / space+drag   pan
    F                  fit to window
    S                  solve now (it also solves automatically)

Tracing order: the two long sides first, then the far short edge.  Strokes may
be partial — only the line they lie on is used.  Optionally trace any visible
fragment of the near edge as a 4th stroke; that is what makes the FOV solvable.

Requires PyQt5 (or PyQt6) and the sibling `camera_recon` module.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        raise SystemExit("recon_ui needs PyQt5 or PyQt6 installed.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_recon as cr  # noqa: E402


Qt = QtCore.Qt
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_IMAGE = os.path.join(_HERE, "test.png")

# The four strokes, in the order the operator is walked through them.
SLOTS = ["side_a", "side_b", "far", "near"]
SLOT_LABEL = {
    "side_a": "1  Side A  (long edge)",
    "side_b": "2  Side B  (long edge)",
    "far":    "3  Far edge  (short, opposite the camera)",
    "near":   "4  Near edge  (optional — unlocks FOV)",
}
SLOT_COLOR = {
    "side_a": QtGui.QColor("#ff5f56"),
    "side_b": QtGui.QColor("#ffbd2e"),
    "far":    QtGui.QColor("#4aa3ff"),
    "near":   QtGui.QColor("#c678dd"),
}
# The reconstruction overlay: the three solved-for edges vs. the predicted one.
FIT_COLOR = QtGui.QColor("#27c93f")
PREDICT_COLOR = QtGui.QColor("#00e5ff")
# The body box hanging below the rim — same green family as the rim it grows
# out of, but lighter, so it never gets mistaken for a fitted edge.
BODY_COLOR = QtGui.QColor("#7ee787")
BODY_RING_COLOR = QtGui.QColor("#3d7d4d")

HANDLE_PX = 7.0        # endpoint grab radius, in screen pixels
_DRAW_LIMIT = 20000.0  # drop overlay points further out than this (Qt gets unhappy)


def _evt_pos(event) -> QtCore.QPointF:
    """Mouse position as QPointF, across the PyQt5 / PyQt6 API change."""
    if hasattr(event, "position"):
        return QtCore.QPointF(event.position())
    return QtCore.QPointF(event.pos())


# ==========================================================================
# canvas
# ==========================================================================

class Canvas(QtWidgets.QWidget):
    """Image view with zoom/pan, edge tracing, and the reconstruction overlay."""

    changed = QtCore.pyqtSignal()          # a stroke was created / moved / cleared
    activeChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMinimumSize(480, 360)

        self.image: QtGui.QImage = QtGui.QImage()
        self.image_path = ""
        self.strokes = {name: None for name in SLOTS}
        self.active = SLOTS[0]
        self.solution = None
        self.show_support_lines = True
        self.body_depth = None         # metres below the rim, None = box hidden
        self.body_shaded = True
        self.body_rings = 1            # intermediate cross-sections to draw

        self._zoom = 1.0
        self._pan = QtCore.QPointF(0.0, 0.0)
        self._drag = None                  # ('new'|'handle'|'pan', ...)
        self._space = False
        self._user_view = False            # True once the operator zooms or pans

    # -- image ------------------------------------------------------------
    def load_image(self, path: str) -> bool:
        img = QtGui.QImage(path)
        if img.isNull():
            return False
        self.image = img
        self.image_path = path
        self.fit()
        return True

    @property
    def image_size(self):
        return (self.image.width(), self.image.height())

    # -- view transform ---------------------------------------------------
    def to_widget(self, p) -> QtCore.QPointF:
        return QtCore.QPointF(p[0] * self._zoom + self._pan.x(),
                              p[1] * self._zoom + self._pan.y())

    def to_image(self, p: QtCore.QPointF) -> np.ndarray:
        return np.array([(p.x() - self._pan.x()) / self._zoom,
                         (p.y() - self._pan.y()) / self._zoom])

    def fit(self):
        if self.image.isNull():
            return
        z = min(self.width() / self.image.width(),
                self.height() / self.image.height())
        self._zoom = max(z, 1e-6)
        self._pan = QtCore.QPointF(
            (self.width() - self.image.width() * self._zoom) * 0.5,
            (self.height() - self.image.height() * self._zoom) * 0.5)
        self._user_view = False
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the still filling the view until the operator takes over the
        # zoom/pan themselves; after that, leave their framing alone.
        if not self._user_view:
            self.fit()

    # -- stroke bookkeeping -----------------------------------------------
    def set_active(self, name: str):
        if name in SLOTS and name != self.active:
            self.active = name
            self.activeChanged.emit(name)
            self.update()

    def clear_stroke(self, name: str):
        if self.strokes.get(name) is not None:
            self.strokes[name] = None
            self.changed.emit()
            self.update()

    def clear_all(self):
        self.strokes = {name: None for name in SLOTS}
        self.solution = None
        self.set_active(SLOTS[0])
        self.changed.emit()
        self.update()

    def _hit_handle(self, pos: QtCore.QPointF):
        """Endpoint under the cursor, as (slot, endpoint index)."""
        for name in SLOTS:
            seg = self.strokes[name]
            if seg is None:
                continue
            for idx, p in enumerate((seg.p0, seg.p1)):
                if (self.to_widget(p) - pos).manhattanLength() <= HANDLE_PX * 2:
                    return name, idx
        return None

    def _hit_stroke(self, pos: QtCore.QPointF, tol_px: float = 8.0):
        """Traced stroke under the cursor (distance measured on screen)."""
        best, best_d = None, tol_px
        for name in SLOTS:
            seg = self.strokes[name]
            if seg is None:
                continue
            a, b = self.to_widget(seg.p0), self.to_widget(seg.p1)
            ab = np.array([b.x() - a.x(), b.y() - a.y()])
            ap = np.array([pos.x() - a.x(), pos.y() - a.y()])
            denom = float(ab @ ab)
            t = 0.0 if denom < 1e-9 else float(np.clip((ap @ ab) / denom, 0.0, 1.0))
            d = float(np.linalg.norm(ap - t * ab))
            if d < best_d:
                best, best_d = name, d
        return best

    # -- input ------------------------------------------------------------
    def mousePressEvent(self, event):
        pos = _evt_pos(event)
        btn = event.button()

        if btn == Qt.MouseButton.MiddleButton or (
                btn == Qt.MouseButton.LeftButton and self._space):
            self._drag = ("pan", pos, QtCore.QPointF(self._pan))
            self._user_view = True
            return

        if btn == Qt.MouseButton.RightButton:
            self.clear_stroke(self._hit_stroke(pos) or self.active)
            return

        if btn == Qt.MouseButton.LeftButton:
            hit = self._hit_handle(pos)
            if hit is not None:
                self.set_active(hit[0])
                self._drag = ("handle", hit[0], hit[1])
                return
            p = self.to_image(pos)
            self.strokes[self.active] = cr.Segment(p, p)
            self._drag = ("new", self.active)
            self.update()

    def mouseMoveEvent(self, event):
        pos = _evt_pos(event)
        if self._drag is None:
            self.setCursor(Qt.CursorShape.SizeAllCursor
                           if self._hit_handle(pos) else Qt.CursorShape.CrossCursor)
            return

        kind = self._drag[0]
        if kind == "pan":
            _, origin, pan0 = self._drag
            self._pan = pan0 + (pos - origin)
        elif kind == "new":
            self.strokes[self._drag[1]].p1 = self.to_image(pos)
        elif kind == "handle":
            _, name, idx = self._drag
            seg = self.strokes[name]
            if idx == 0:
                seg.p0 = self.to_image(pos)
            else:
                seg.p1 = self.to_image(pos)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        kind = self._drag[0]
        self._drag = None
        if kind == "new":
            seg = self.strokes[self.active]
            if seg is not None and seg.length * self._zoom < 4.0:
                self.strokes[self.active] = None      # a click, not a stroke
                self.update()
                return
            # Walk on to the next un-traced slot so the three edges flow.
            order = SLOTS[SLOTS.index(self.active) + 1:] + SLOTS
            for name in order:
                if self.strokes[name] is None:
                    self.set_active(name)
                    break
        if kind in ("new", "handle"):
            self.changed.emit()
        self.update()

    def wheelEvent(self, event):
        if self.image.isNull():
            return
        self._user_view = True
        before = self.to_image(_evt_pos(event))
        step = event.angleDelta().y() / 120.0
        self._zoom = float(np.clip(self._zoom * (1.15 ** step), 0.02, 60.0))
        after = self.to_image(_evt_pos(event))
        self._pan += QtCore.QPointF(float(after[0] - before[0]) * self._zoom,
                                    float(after[1] - before[1]) * self._zoom)
        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._space = True
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_4:
            self.set_active(SLOTS[key - Qt.Key.Key_1])
        elif key == Qt.Key.Key_F:
            self.fit()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.clear_stroke(self.active)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._space = False
        else:
            super().keyReleaseEvent(event)

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        qp.fillRect(self.rect(), QtGui.QColor("#15171c"))

        if self.image.isNull():
            qp.setPen(QtGui.QColor("#8a8f98"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                        "No image loaded — use Load image…")
            return

        target = QtCore.QRectF(self._pan,
                               QtCore.QSizeF(self.image.width() * self._zoom,
                                             self.image.height() * self._zoom))
        qp.drawImage(target, self.image)
        qp.setPen(QtGui.QPen(QtGui.QColor("#3a3f4b"), 1))
        qp.drawRect(target)

        self._paint_strokes(qp)
        if self.solution is not None:
            self._paint_solution(qp)
        self._paint_legend(qp)

    def _paint_strokes(self, qp):
        for name in SLOTS:
            seg = self.strokes[name]
            if seg is None:
                continue
            colour = SLOT_COLOR[name]
            a, b = self.to_widget(seg.p0), self.to_widget(seg.p1)
            is_active = (name == self.active)

            if self.show_support_lines:
                # The stroke is only a sample of an infinite line; drawing that
                # line full-width shows at a glance whether it hugs the real edge
                # well beyond the bit that was traced.
                d = b - a
                n = math.hypot(d.x(), d.y())
                if n > 1e-6:
                    d /= n
                    far = 6000.0
                    pen = QtGui.QPen(QtGui.QColor(colour.red(), colour.green(),
                                                  colour.blue(), 90), 1)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    qp.setPen(pen)
                    qp.drawLine(a - d * far, b + d * far)

            qp.setPen(QtGui.QPen(colour, 3.0 if is_active else 2.0))
            qp.drawLine(a, b)

            qp.setBrush(QtGui.QBrush(colour))
            qp.setPen(QtGui.QPen(QtGui.QColor("#101216"), 1))
            for p in (a, b):
                qp.drawRect(QtCore.QRectF(p.x() - HANDLE_PX * 0.5,
                                          p.y() - HANDLE_PX * 0.5,
                                          HANDLE_PX, HANDLE_PX))
            qp.setBrush(Qt.BrushStyle.NoBrush)

            qp.setPen(colour)
            qp.drawText((a + b) * 0.5 + QtCore.QPointF(6, -6),
                        SLOT_LABEL[name].split("  ")[0] + " " + name)

    def _paint_body(self, qp):
        """The box hanging below the rim, drawn under the rim wireframe."""
        sol = self.solution
        depth = self.body_depth
        if depth is None or depth <= 0.0:
            return

        if self.body_shaded:
            qp.setPen(Qt.PenStyle.NoPen)
            for name, quad in sol.body_faces(depth):
                poly = QtGui.QPolygonF([self.to_widget(p) for p in quad])
                # The floor reads as the "far" surface, so keep it fainter than
                # the walls or it fights with them where the box is edge-on.
                alpha = 18 if name == "floor" else 26
                qp.setBrush(QtGui.QBrush(QtGui.QColor(
                    BODY_COLOR.red(), BODY_COLOR.green(), BODY_COLOR.blue(), alpha)))
                qp.drawPolygon(poly)
            qp.setBrush(Qt.BrushStyle.NoBrush)

        rings = tuple((i + 1) / (self.body_rings + 1.0)
                      for i in range(max(0, self.body_rings)))
        pens = {
            "pillar": QtGui.QPen(BODY_COLOR, 1.8),
            "bottom": QtGui.QPen(BODY_COLOR, 1.8),
            "ring": QtGui.QPen(BODY_RING_COLOR, 1.0),
        }
        pens["bottom"].setStyle(Qt.PenStyle.DashLine)
        pens["ring"].setStyle(Qt.PenStyle.DotLine)
        for kind, poly in sol.body_polylines(depth, rings=rings):
            if poly.shape[0] < 2:
                continue
            qp.setPen(pens[kind])
            for chunk in self._chunks(poly):
                qp.drawPolyline(QtGui.QPolygonF(chunk))

    def _paint_solution(self, qp):
        sol = self.solution
        self._paint_body(qp)
        for name, poly in sol.edge_polylines():
            if poly.shape[0] < 2:
                continue
            predicted = (name == "near" and self.strokes["near"] is None)
            colour = PREDICT_COLOR if predicted else FIT_COLOR
            pen = QtGui.QPen(colour, 2.4 if predicted else 1.8)
            if predicted:
                pen.setStyle(Qt.PenStyle.DashLine)
            qp.setPen(pen)
            for chunk in self._chunks(poly):
                qp.drawPolyline(QtGui.QPolygonF(chunk))

        qp.setBrush(QtGui.QBrush(FIT_COLOR))
        qp.setPen(QtGui.QPen(QtGui.QColor("#101216"), 1))
        for corner in sol.corners_img:
            if not np.all(np.isfinite(corner)):
                continue
            p = self.to_widget(corner)
            qp.drawEllipse(p, 4.0, 4.0)
        qp.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy = sol.principal_point
        c = self.to_widget((cx, cy))
        qp.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        qp.drawLine(c + QtCore.QPointF(-7, 0), c + QtCore.QPointF(7, 0))
        qp.drawLine(c + QtCore.QPointF(0, -7), c + QtCore.QPointF(0, 7))

    def _chunks(self, poly):
        """Split a projected polyline where it goes off to (near) infinity."""
        out, run = [], []
        for pt in poly:
            if not np.all(np.isfinite(pt)):
                if len(run) > 1:
                    out.append(run)
                run = []
                continue
            w = self.to_widget(pt)
            if abs(w.x()) > _DRAW_LIMIT or abs(w.y()) > _DRAW_LIMIT:
                if len(run) > 1:
                    out.append(run)
                run = []
                continue
            run.append(w)
        if len(run) > 1:
            out.append(run)
        return out

    def _paint_legend(self, qp):
        entries = [(SLOT_COLOR[n], SLOT_LABEL[n]) for n in SLOTS
                   if self.strokes[n] is not None]
        if self.solution is not None:
            entries.append((FIT_COLOR, "reconstructed rim (fitted edges)"))
            if self.strokes["near"] is None:
                entries.append((PREDICT_COLOR, "predicted 4th edge — the accuracy check"))
            if self.body_depth:
                entries.append((BODY_COLOR,
                                "body box, %.2f m deep — corners must run vertical"
                                % self.body_depth))
        if not entries:
            return
        pad, row = 8, 16
        box = QtCore.QRectF(10, 10, 380, pad * 2 + row * len(entries))
        qp.setBrush(QtGui.QBrush(QtGui.QColor(16, 18, 22, 205)))
        qp.setPen(QtGui.QPen(QtGui.QColor("#3a3f4b"), 1))
        qp.drawRoundedRect(box, 6, 6)
        qp.setBrush(Qt.BrushStyle.NoBrush)
        font = qp.font()
        font.setPointSizeF(8.5)
        qp.setFont(font)
        for i, (colour, text) in enumerate(entries):
            y = box.top() + pad + row * i + row * 0.5
            qp.setPen(QtGui.QPen(colour, 3))
            qp.drawLine(QtCore.QPointF(box.left() + pad, y),
                        QtCore.QPointF(box.left() + pad + 22, y))
            qp.setPen(QtGui.QColor("#d8dce3"))
            qp.drawText(QtCore.QPointF(box.left() + pad + 30, y + 4), text)


# ==========================================================================
# main window
# ==========================================================================

class ReconWindow(QtWidgets.QMainWindow):

    def __init__(self, image_path: str = ""):
        super().__init__()
        self.setWindowTitle("cam_recon — 3-line camera reconstruction test bench")
        self.resize(1500, 900)

        self.canvas = Canvas()
        self.canvas.changed.connect(self.solve)
        self.canvas.activeChanged.connect(self._sync_slot_list)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self._build_panel())
        self.setCentralWidget(central)

        self._apply_style()
        if image_path or os.path.exists(_DEFAULT_IMAGE):
            self._load_image(image_path or _DEFAULT_IMAGE)
        self._sync_slot_list()
        self._sync_fov_widgets()
        self._push_body()

    # -- panel ------------------------------------------------------------
    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(460)
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        body = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(0, 12, 0, 0)      # `outer` already pads left/right
        v.setSpacing(10)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # ---- image ------------------------------------------------------
        gb = self._group("Image")
        self.lbl_image = QtWidgets.QLabel("—")
        self.lbl_image.setWordWrap(True)
        btn_load = QtWidgets.QPushButton("Load image…")
        btn_load.clicked.connect(self._on_load_image)
        gb.layout().addWidget(self.lbl_image)
        gb.layout().addWidget(btn_load)
        v.addWidget(gb)

        # ---- body dimensions --------------------------------------------
        gb = self._group("Truck body rim (real dimensions)")
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_width = self._spin(0.05, 100.0, 2.5, 3, " m")
        self.sp_length = self._spin(0.05, 100.0, 7.0, 3, " m")
        self.sp_width.valueChanged.connect(self._on_dims_changed)
        self.sp_length.valueChanged.connect(self._on_dims_changed)
        form.addRow("Width (short edge)", self.sp_width)
        form.addRow("Length (long edge)", self.sp_length)
        gb.layout().addLayout(form)
        hint = QtWidgets.QLabel(
            "Width sets the scale of the whole solution. Length only places the "
            "untraced near edge — which is exactly why that edge is a fair test.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        gb.layout().addWidget(hint)
        v.addWidget(gb)

        # ---- body depth / volume -----------------------------------------
        gb = self._group("Body depth (3D check)")
        self.chk_body = QtWidgets.QCheckBox("Draw the body as a 3D box")
        self.chk_body.setChecked(True)
        self.chk_body.toggled.connect(self._on_body_toggle)
        gb.layout().addWidget(self.chk_body)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_volume = self._spin(0.01, 1000.0, 20.0, 3, " m³")
        self.sp_depth = self._spin(0.01, 20.0, 1.143, 3, " m")
        self.sp_volume.valueChanged.connect(self._on_volume_changed)
        self.sp_depth.valueChanged.connect(self._on_depth_changed)
        form.addRow("Volume", self.sp_volume)
        form.addRow("Depth (derived)", self.sp_depth)
        gb.layout().addLayout(form)

        self.chk_body_shade = QtWidgets.QCheckBox("Shade the box faces")
        self.chk_body_shade.setChecked(True)
        self.chk_body_shade.toggled.connect(self._on_body_toggle)
        gb.layout().addWidget(self.chk_body_shade)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Cross-sections"))
        self.sp_rings = QtWidgets.QSpinBox()
        self.sp_rings.setRange(0, 8)
        self.sp_rings.setValue(1)
        self.sp_rings.valueChanged.connect(self._on_body_toggle)
        row.addWidget(self.sp_rings)
        row.addStretch(1)
        gb.layout().addLayout(row)

        hint = QtWidgets.QLabel(
            "Depth is just volume / (width × length) — a flat-floored body, which "
            "is close enough for a sanity check. The four corner drops are the "
            "point: they are predicted, not fitted, so if the pose or the FOV is "
            "off they will lean away from the real vertical edges of the body long "
            "before the flat rim looks wrong.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        gb.layout().addWidget(hint)
        v.addWidget(gb)

        # ---- strokes ----------------------------------------------------
        gb = self._group("Traced edges")
        self.slot_list = QtWidgets.QListWidget()
        self.slot_list.setFixedHeight(96)
        for name in SLOTS:
            item = QtWidgets.QListWidgetItem(SLOT_LABEL[name])
            item.setForeground(QtGui.QBrush(SLOT_COLOR[name]))
            self.slot_list.addItem(item)
        self.slot_list.currentRowChanged.connect(
            lambda r: self.canvas.set_active(SLOTS[r]) if r >= 0 else None)
        gb.layout().addWidget(self.slot_list)

        row = QtWidgets.QHBoxLayout()
        btn_clear_one = QtWidgets.QPushButton("Clear selected")
        btn_clear_one.clicked.connect(lambda: self.canvas.clear_stroke(self.canvas.active))
        btn_clear_all = QtWidgets.QPushButton("Clear all")
        btn_clear_all.clicked.connect(self.canvas.clear_all)
        row.addWidget(btn_clear_one)
        row.addWidget(btn_clear_all)
        gb.layout().addLayout(row)

        self.chk_support = QtWidgets.QCheckBox("Extend traced lines across the frame")
        self.chk_support.setChecked(True)
        self.chk_support.toggled.connect(self._on_support_toggle)
        gb.layout().addWidget(self.chk_support)
        v.addWidget(gb)

        # ---- intrinsics --------------------------------------------------
        gb = self._group("Field of view")
        self.chk_fov_known = QtWidgets.QCheckBox("FOV is known (horizontal, degrees)")
        self.chk_fov_known.toggled.connect(self._on_fov_mode)
        gb.layout().addWidget(self.chk_fov_known)

        self.sp_fov = self._spin(5.0, 175.0, 60.0, 2, " °")
        self.sp_fov.valueChanged.connect(self._on_fov_spin)
        gb.layout().addWidget(self.sp_fov)

        self.sld_fov = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_fov.setRange(500, 17500)          # hundredths of a degree
        self.sld_fov.setValue(6000)
        self.sld_fov.valueChanged.connect(self._on_fov_slider)
        gb.layout().addWidget(self.sld_fov)

        self.chk_aniso = QtWidgets.QCheckBox(
            "Non-square pixels — solve H and V FOV separately")
        self.chk_aniso.toggled.connect(self._on_fov_mode)
        gb.layout().addWidget(self.chk_aniso)

        self.lbl_fov_mode = QtWidgets.QLabel()
        self.lbl_fov_mode.setWordWrap(True)
        self.lbl_fov_mode.setObjectName("hint")
        gb.layout().addWidget(self.lbl_fov_mode)

        self.chk_pp = QtWidgets.QCheckBox("Override principal point")
        self.chk_pp.toggled.connect(self._on_pp_toggle)
        gb.layout().addWidget(self.chk_pp)
        pp_row = QtWidgets.QHBoxLayout()
        self.sp_ppx = self._spin(-10000.0, 10000.0, 0.0, 1, " px")
        self.sp_ppy = self._spin(-10000.0, 10000.0, 0.0, 1, " px")
        for sp in (self.sp_ppx, self.sp_ppy):
            sp.setEnabled(False)
            sp.valueChanged.connect(self.solve)
            pp_row.addWidget(sp)
        gb.layout().addLayout(pp_row)

        self.chk_refine = QtWidgets.QCheckBox("Least-squares refinement")
        self.chk_refine.setChecked(True)
        self.chk_refine.toggled.connect(self.solve)
        gb.layout().addWidget(self.chk_refine)
        v.addWidget(gb)

        # ---- optional ground truth ---------------------------------------
        gb = self._group("Compare against known telemetry (optional)")
        self.chk_gt = QtWidgets.QCheckBox("I know the real camera angles / FOV")
        self.chk_gt.toggled.connect(self._on_gt_toggle)
        gb.layout().addWidget(self.chk_gt)
        gt_form = QtWidgets.QFormLayout()
        gt_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_gt_yaw = self._spin(-360.0, 360.0, 0.0, 2, " °")
        self.sp_gt_pitch = self._spin(-180.0, 180.0, 0.0, 2, " °")
        self.sp_gt_roll = self._spin(-180.0, 180.0, 0.0, 2, " °")
        self.sp_gt_fov = self._spin(5.0, 175.0, 60.0, 2, " °")
        for sp, label in ((self.sp_gt_yaw, "YAW"), (self.sp_gt_pitch, "PITCH"),
                          (self.sp_gt_roll, "ROLL"), (self.sp_gt_fov, "FOV")):
            sp.setEnabled(False)
            sp.valueChanged.connect(self.solve)
            gt_form.addRow(label, sp)
        gb.layout().addLayout(gt_form)
        hint = QtWidgets.QLabel(
            "The body sits on level ground, so PITCH, ROLL and FOV are directly "
            "comparable. YAW is only comparable up to the body's own heading, "
            "which is reported instead.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        gb.layout().addWidget(hint)
        v.addWidget(gb)

        # ---- session ------------------------------------------------------
        row = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("Save session…")
        btn_save.clicked.connect(self._on_save)
        btn_open = QtWidgets.QPushButton("Load session…")
        btn_open.clicked.connect(self._on_open)
        row.addWidget(btn_save)
        row.addWidget(btn_open)
        v.addLayout(row)

        v.addStretch(1)

        # ---- results ------------------------------------------------------
        # Pinned below the scroll area rather than inside it: this is the
        # output the operator is actually reading, so it must never scroll away.
        gb = self._group("Reconstruction")
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        font.setPointSize(8)
        self.out.setFont(font)
        gb.layout().addWidget(self.out)
        gb.setMinimumHeight(330)
        outer.addWidget(gb)
        outer.setStretch(0, 1)
        outer.setStretch(1, 0)
        outer.setContentsMargins(12, 0, 12, 12)
        return panel

    def _group(self, title: str) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(gb)
        lay.setSpacing(6)
        return gb

    def _spin(self, lo, hi, val, decimals, suffix) -> QtWidgets.QDoubleSpinBox:
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.setSingleStep(0.1 if decimals >= 2 else 1.0)
        sp.setKeyboardTracking(False)
        # Force '.' as the decimal separator so what is typed here matches the
        # numbers printed in the report, whatever the system locale is.
        sp.setLocale(QtCore.QLocale.c())
        return sp

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1b1e24; color: #d8dce3;
                                   font-family: 'Segoe UI'; font-size: 12px; }
            QGroupBox { border: 1px solid #2f343d; border-radius: 6px;
                        margin-top: 14px; padding: 10px 8px 8px 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px;
                               padding: 0 4px; color: #8fb7ff; }
            QLabel#hint { color: #8a8f98; font-size: 11px; }
            QPushButton { background: #262b34; border: 1px solid #39404b;
                          border-radius: 4px; padding: 5px 10px; }
            QPushButton:hover { background: #2f3642; }
            QDoubleSpinBox, QPlainTextEdit, QListWidget {
                background: #12151a; border: 1px solid #2f343d;
                border-radius: 4px; padding: 3px; selection-background-color: #2d5aa8; }
            QListWidget::item:selected { background: #2d3a52; }
            QScrollArea { border: none; }
        """)

    # -- panel callbacks ---------------------------------------------------
    def _floor_area(self) -> float:
        return max(self.sp_width.value() * self.sp_length.value(), 1e-6)

    @staticmethod
    def _set_quiet(spin, value):
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _on_dims_changed(self, _value=None):
        # Width and length are the measured truth; the volume the operator typed
        # is kept and the derived depth follows the new footprint.
        self._set_quiet(self.sp_depth, self.sp_volume.value() / self._floor_area())
        self._push_body()
        self.solve()

    def _on_volume_changed(self, value):
        self._set_quiet(self.sp_depth, value / self._floor_area())
        self._push_body()

    def _on_depth_changed(self, value):
        self._set_quiet(self.sp_volume, value * self._floor_area())
        self._push_body()

    def _on_body_toggle(self, _on=None):
        self._push_body()

    def _push_body(self):
        on = self.chk_body.isChecked()
        for w in (self.sp_volume, self.sp_depth, self.chk_body_shade, self.sp_rings):
            w.setEnabled(on)
        self.canvas.body_depth = self.sp_depth.value() if on else None
        self.canvas.body_shaded = self.chk_body_shade.isChecked()
        self.canvas.body_rings = self.sp_rings.value()
        self.canvas.update()

    def _on_support_toggle(self, on):
        self.canvas.show_support_lines = on
        self.canvas.update()

    def _on_pp_toggle(self, on):
        self.sp_ppx.setEnabled(on)
        self.sp_ppy.setEnabled(on)
        if not on and not self.canvas.image.isNull():
            w, h = self.canvas.image_size
            self.sp_ppx.setValue(w * 0.5)
            self.sp_ppy.setValue(h * 0.5)
        self.solve()

    def _on_gt_toggle(self, on):
        for sp in (self.sp_gt_yaw, self.sp_gt_pitch, self.sp_gt_roll, self.sp_gt_fov):
            sp.setEnabled(on)
        self.solve()

    def _on_fov_mode(self, _on):
        self._sync_fov_widgets()
        self.solve()

    def _on_fov_spin(self, value):
        if not self.chk_fov_known.isChecked():
            return
        self.sld_fov.blockSignals(True)
        self.sld_fov.setValue(int(round(value * 100)))
        self.sld_fov.blockSignals(False)
        self.solve()

    def _on_fov_slider(self, value):
        self.sp_fov.blockSignals(True)
        self.sp_fov.setValue(value / 100.0)
        self.sp_fov.blockSignals(False)
        self.solve()

    def _sync_fov_widgets(self):
        known = self.chk_fov_known.isChecked()
        has_near = self.canvas.strokes.get("near") is not None
        self.sp_fov.setEnabled(known)
        # Separating fx from fy needs all four corners, so a homography exists.
        self.chk_aniso.setEnabled(has_near)
        aniso = has_near and self.chk_aniso.isChecked()
        # The slider is only meaningful in the one case where FOV is a free
        # parameter: three strokes and no external knowledge.
        self.sld_fov.setEnabled(not known and not has_near)

        if aniso and known:
            text = ("Horizontal FOV is taken from the box; the vertical one is "
                    "solved from the 4th stroke, so a squeezed frame is handled.")
        elif aniso:
            text = ("Solving fx and fy separately. Four corners give a homography "
                    "of a plane whose real width:length you supplied, which is "
                    "two constraints — enough to separate them. This is much more "
                    "noise-sensitive than the square-pixel solve, so watch the "
                    "sensitivity figure in the report.")
        elif known:
            text = "Using the entered FOV. The solution is then exact and fully determined."
        elif has_near:
            text = ("FOV is being solved from the two vanishing points, thanks to the "
                    "4th stroke. Trace both short edges as long as you can — that is "
                    "what conditions the estimate. Tick the box above if the frame "
                    "may be squeezed rather than merely wide.")
        else:
            text = ("Three strokes alone cannot fix the FOV: every focal length "
                    "reproduces them exactly. Drag the slider until the dashed "
                    "predicted 4th edge sits on the real near edge — or trace that "
                    "edge as stroke 4 and it will be solved for you.")
        self.lbl_fov_mode.setText(text)

    def _sync_slot_list(self, *_):
        for i, name in enumerate(SLOTS):
            traced = self.canvas.strokes[name] is not None
            item = self.slot_list.item(i)
            item.setText(SLOT_LABEL[name] + ("   ✓" if traced else ""))
            font = item.font()
            font.setBold(name == self.canvas.active)
            item.setFont(font)
        self.slot_list.blockSignals(True)
        self.slot_list.setCurrentRow(SLOTS.index(self.canvas.active))
        self.slot_list.blockSignals(False)

    # -- file actions ------------------------------------------------------
    def _load_image(self, path):
        if not self.canvas.load_image(path):
            QtWidgets.QMessageBox.warning(self, "cam_recon", "Could not load %s" % path)
            return
        w, h = self.canvas.image_size
        self.lbl_image.setText("%s\n%d x %d px" % (os.path.basename(path), w, h))
        self.sp_ppx.setValue(w * 0.5)
        self.sp_ppy.setValue(h * 0.5)
        self.solve()

    def _on_load_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load image", _HERE, "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self._load_image(path)

    def _on_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", os.path.join(_HERE, "session.json"), "JSON (*.json)")
        if not path:
            return
        cr.save_session(
            path, self.canvas.strokes, self.sp_width.value(), self.sp_length.value(),
            self.sp_fov.value() if self.chk_fov_known.isChecked() else None,
            self.canvas.image_path,
            volume=self.sp_volume.value(), depth=self.sp_depth.value())

    def _on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", _HERE, "JSON (*.json)")
        if not path:
            return
        data = cr.load_session(path)
        img = data.get("image")
        if img and os.path.exists(img) and img != self.canvas.image_path:
            self._load_image(img)
        self.sp_width.setValue(float(data.get("width", 2.5)))
        self.sp_length.setValue(float(data.get("length", 7.0)))
        # Depth wins over volume when both are stored: it is what was drawn.
        if data.get("depth") is not None:
            self.sp_depth.setValue(float(data["depth"]))
        elif data.get("volume") is not None:
            self.sp_volume.setValue(float(data["volume"]))
        fov = data.get("fov_x_deg")
        self.chk_fov_known.setChecked(fov is not None)
        if fov is not None:
            self.sp_fov.setValue(float(fov))
        for name in SLOTS:
            self.canvas.strokes[name] = data["strokes"].get(name)
        self.canvas.update()
        self._sync_slot_list()
        self.solve()

    # -- the actual run ----------------------------------------------------
    def solve(self):
        self._sync_slot_list()
        self._sync_fov_widgets()
        self.canvas.solution = None

        strokes = self.canvas.strokes
        missing = [SLOT_LABEL[n] for n in SLOTS[:3] if strokes[n] is None]
        if self.canvas.image.isNull():
            self.out.setPlainText("Load an image to start.")
            self.canvas.update()
            return
        if missing:
            self.out.setPlainText(
                "Trace the three edges to solve.\n\nStill missing:\n  "
                + "\n  ".join(missing))
            self.canvas.update()
            return

        pp = ((self.sp_ppx.value(), self.sp_ppy.value())
              if self.chk_pp.isChecked() else None)
        try:
            sol = cr.reconstruct(
                strokes["side_a"], strokes["side_b"], strokes["far"],
                width=self.sp_width.value(),
                length=self.sp_length.value(),
                image_size=self.canvas.image_size,
                fov_x_deg=self.sp_fov.value() if self.chk_fov_known.isChecked() else None,
                near_edge=strokes["near"],
                principal_point=pp,
                refine=self.chk_refine.isChecked(),
                assumed_fov_x_deg=self.sld_fov.value() / 100.0,
                anamorphic=(self.chk_aniso.isChecked()
                            and strokes["near"] is not None),
            )
        except Exception as exc:
            self.canvas.solution = None
            self.out.setPlainText("Reconstruction failed.\n\n%s" % exc)
            self.canvas.update()
            return

        self.canvas.solution = sol
        self.out.setPlainText(self._report(sol))
        self.canvas.update()

    def _report(self, sol) -> str:
        lines = [sol.summary(), ""]

        pos, hpr = sol.to_world()
        lines.append("Panda3D drop-in (body-relative frame: +X width, +Y away")
        lines.append("from the camera, +Z up, origin at the rim centre):")
        lines.append("    camera.set_pos(%.4f, %.4f, %.4f)" % tuple(pos))
        lines.append("    camera.set_hpr(%.3f, %.3f, %.3f)" % tuple(hpr))
        if sol.anamorphic_solved:
            lines.append("    lens.set_fov(%.3f, %.3f)   # h, v (non-square pixels)"
                         % (sol.fov_x_deg, sol.fov_y_deg))
        else:
            lines.append("    lens.set_fov(%.3f)   # horizontal" % sol.fov_x_deg)
        lines.append("")

        if self.chk_body.isChecked():
            depth = self.sp_depth.value()
            lines.append("body box  %.3f m deep  (%.2f m3 over a %.2f x %.2f m floor)"
                         % (depth, self.sp_volume.value(),
                            self.sp_width.value(), self.sp_length.value()))
            lines.append("camera sits %.3f m above the rim, %.3f m above the floor"
                         % (sol.height_above_plane, sol.height_above_plane + depth))
            lines.append("")

        if self.canvas.strokes["near"] is None:
            lines.append("The cyan dashed edge was never traced — it is predicted "
                         "purely from the length you entered. How closely it lands "
                         "on the real near edge is the accuracy of this solve.")
        else:
            lines.append("All four edges were traced, so the overlay is a fit, not a "
                         "prediction. Untrace the near edge to get an independent check.")

        if self.chk_gt.isChecked():
            lines.append("")
            lines.append("--- against known telemetry -------------------------")
            _, pitch, roll = sol.hpr_deg
            yaw = sol.hpr_deg[0]
            d_pitch = pitch - self.sp_gt_pitch.value()
            d_roll = roll - self.sp_gt_roll.value()
            lines.append("PITCH  solved %+7.2f   real %+7.2f   delta %+6.2f deg"
                         % (pitch, self.sp_gt_pitch.value(), d_pitch))
            lines.append("ROLL   solved %+7.2f   real %+7.2f   delta %+6.2f deg"
                         % (roll, self.sp_gt_roll.value(), d_roll))
            if self.chk_fov_known.isChecked():
                lines.append("FOV    entered %6.2f  (an input here, so not a check)"
                             % sol.fov_x_deg)
            elif sol.fov_determined:
                d_fov = sol.fov_x_deg - self.sp_gt_fov.value()
                lines.append("FOV    solved %7.2f   real %7.2f   delta %+6.2f deg"
                             % (sol.fov_x_deg, self.sp_gt_fov.value(), d_fov))
            else:
                lines.append("FOV    assumed %6.2f  (not solved — nothing to compare)"
                             % sol.fov_x_deg)
            heading = (self.sp_gt_yaw.value() - yaw + 180.0) % 360.0 - 180.0
            lines.append("Implied body heading in the world: %+7.2f deg" % heading)
            lines.append("(YAW is body-relative here, so this is the check's residue,")
            lines.append(" not an error — it should match the truck's real heading.)")
        return "\n".join(lines)


def main():
    app = QtWidgets.QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    win = ReconWindow(path)
    win.show()
    run = getattr(app, "exec", None) or app.exec_
    sys.exit(run())


if __name__ == "__main__":
    main()
