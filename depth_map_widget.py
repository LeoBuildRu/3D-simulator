# depth_map_widget.py
# ---------------------------------------------------------------------------
# Real-time Depth Map Preview widget (16:9) fed from a Panda3D Texture.
#
# Pipeline:
#   Panda3D depth pass  ->  Texture (copy-to-ram enabled)
#         |
#         v
#   taskMgr task (run each frame)  ->  Texture.getRamImage()
#         |
#         v
#   bytes + (w, h)  ->  QImage (Format_Grayscale8 / Format_RGBA8888)
#         |
#         v
#   QLabel.setPixmap(QPixmap.fromImage(img).scaled(...))
#
# We use Qt signals (Signal(bytes, int, int)) to marshal data from the
# Panda task thread onto the Qt GUI thread safely.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget


class AspectRatioFrame(QFrame):
    """Container that locks its children to a 16:9 aspect ratio."""

    def __init__(self, ratio_w: int = 16, ratio_h: int = 9, parent=None):
        super().__init__(parent)
        self._rw = ratio_w
        self._rh = ratio_h
        self.setObjectName("DepthMapFrame")

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(w * self._rh / self._rw)

    def sizeHint(self) -> QSize:
        return QSize(320, int(320 * self._rh / self._rw))


class DepthMapPreview(QWidget):
    """
    Embeddable 16:9 preview of the Panda3D depth buffer.
    Thread-safe: the Panda task emits `frame_ready` with raw bytes, and
    the Qt main thread assembles the QImage / QPixmap here.
    """

    frame_ready = pyqtSignal(bytes, int, int, str)   # raw bytes, w, h, format

    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # --- Header row: eyebrow + live chip -----------------------------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        eyebrow = QLabel("DEPTH MAP / REAL-TIME")
        eyebrow.setProperty("role", "eyebrow")

        self.status_chip = QLabel("IDLE")
        self.status_chip.setProperty("role", "chip-idle")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header.addWidget(eyebrow)
        header.addStretch(1)
        header.addWidget(self.status_chip)

        root.addLayout(header)

        # --- 16:9 canvas --------------------------------------------------
        self.frame = AspectRatioFrame(16, 9, self)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(6, 6, 6, 6)
        frame_layout.setSpacing(0)

        self.canvas = QLabel("NO SIGNAL", self.frame)
        self.canvas.setObjectName("DepthMapCanvas")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumHeight(120)
        frame_layout.addWidget(self.canvas)

        root.addWidget(self.frame)

        # --- Footer: resolution + fps counter ----------------------------
        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 0)
        footer.setSpacing(12)

        self.res_label = QLabel("—×—")
        self.res_label.setProperty("role", "muted")

        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setProperty("role", "muted")

        footer.addWidget(self.res_label)
        footer.addStretch(1)
        footer.addWidget(self.fps_label)

        root.addLayout(footer)

        # --- Wire the signal → slot (queued across threads) --------------
        self.frame_ready.connect(self._on_frame_ready, Qt.ConnectionType.QueuedConnection)

        # FPS smoothing state
        self._frame_count = 0
        self._last_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API — called from the Panda3D task on each frame.
    # It only emits a signal, no Qt widget is touched off-thread.
    # ------------------------------------------------------------------
    def push_frame(self, raw: bytes, width: int, height: int, fmt: str = "gray8") -> None:
        """fmt ∈ {'gray8', 'rgba8'}"""
        self.frame_ready.emit(raw, width, height, fmt)

    def set_live(self, live: bool) -> None:
        if live:
            self.status_chip.setText("● LIVE")
            self.status_chip.setProperty("role", "chip-live")
        else:
            self.status_chip.setText("IDLE")
            self.status_chip.setProperty("role", "chip-idle")
        self.status_chip.style().unpolish(self.status_chip)
        self.status_chip.style().polish(self.status_chip)

    # ------------------------------------------------------------------
    # Slot — runs on the Qt main thread
    # ------------------------------------------------------------------
    def _on_frame_ready(self, raw: bytes, w: int, h: int, fmt: str) -> None:
        import time

        if fmt == "gray8":
            img = QImage(raw, w, h, w, QImage.Format.Format_Grayscale8)
        elif fmt == "rgba8":
            img = QImage(raw, w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            return

        # Panda3D textures arrive flipped vertically in the ram image buffer.
        img = img.mirrored(False, True)

        pm = QPixmap.fromImage(img).scaled(
            self.canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(pm)
        self.res_label.setText(f"{w}×{h}")

        # Rolling FPS over 0.5 s windows
        now = time.monotonic()
        self._frame_count += 1
        if self._last_ts == 0.0:
            self._last_ts = now
        elif now - self._last_ts >= 0.5:
            fps = self._frame_count / (now - self._last_ts)
            self.fps_label.setText(f"{fps:5.1f} FPS")
            self._frame_count = 0
            self._last_ts = now

        if self.status_chip.text() != "● LIVE":
            self.set_live(True)
